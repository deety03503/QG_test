"""Task-specific adapter building blocks for frozen transformer backbones."""

from torch import Tensor, nn


class ParallelAdapter(nn.Module):
    """Bottleneck adapter added in parallel with a transformer MLP output."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden_dim and bottleneck_dim must be positive")
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, mlp_input: Tensor) -> Tensor:
        return self.up(self.activation(self.down(mlp_input)))


class RoutedAdapterMLP(nn.Module):
    """Frozen transformer MLP plus task adapters with optional soft routing."""

    def __init__(self, base_mlp: nn.Module, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.base_mlp = base_mlp
        for parameter in self.base_mlp.parameters():
            parameter.requires_grad_(False)
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.adapters = nn.ModuleList()
        self._routing_weights: Tensor | None = None

    def add_task_adapter(self) -> ParallelAdapter:
        adapter = ParallelAdapter(self.hidden_dim, self.bottleneck_dim)
        reference_parameter = next(self.base_mlp.parameters(), None)
        if reference_parameter is not None:
            adapter.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        if self.adapters:
            adapter.load_state_dict(self.adapters[0].state_dict())
        for old_adapter in self.adapters:
            for parameter in old_adapter.parameters():
                parameter.requires_grad_(False)
        adapter.requires_grad_(True)
        self.adapters.append(adapter)
        return adapter

    def set_routing_weights(self, weights: Tensor | None) -> None:
        if weights is not None and (weights.ndim != 2 or weights.shape[1] != len(self.adapters)):
            raise ValueError("routing weights must have shape (batch, number_of_tasks)")
        self._routing_weights = weights

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.base_mlp(inputs)
        if not self.adapters:
            return outputs

        if self._routing_weights is None:
            routing_weights = inputs.new_zeros((inputs.shape[0], len(self.adapters)))
            routing_weights[:, -1] = 1
        else:
            routing_weights = self._routing_weights.to(device=inputs.device, dtype=inputs.dtype)
            if routing_weights.shape[0] != inputs.shape[0]:
                raise ValueError("routing weights must share the input batch dimension")

        adapter_sum = outputs.new_zeros(outputs.shape)
        for task_index, adapter in enumerate(self.adapters):
            task_weight = routing_weights[:, task_index]
            broadcast_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
            adapter_sum = adapter_sum + adapter(inputs) * task_weight.reshape(broadcast_shape)
        return outputs + adapter_sum
