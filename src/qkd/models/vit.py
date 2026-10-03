"""Pretrained Vision Transformer wrapper with task-specific parallel adapters."""

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from qkd.models.adapters import RoutedAdapterMLP


class PretrainedViT(nn.Module):
    """Frozen timm Vision Transformer with adapters inserted in every block."""

    def __init__(self, backbone: nn.Module, bottleneck_dim: int = 64):
        super().__init__()
        if not hasattr(backbone, "blocks") or not hasattr(backbone, "num_features"):
            raise ValueError("backbone must expose timm ViT blocks and num_features")
        self.backbone = backbone
        self.hidden_dim = int(backbone.num_features)
        self.bottleneck_dim = bottleneck_dim

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for block in self.backbone.blocks:
            block.mlp = RoutedAdapterMLP(block.mlp, self.hidden_dim, bottleneck_dim)
        self.add_task_adapter()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        model_name: str = "vit_base_patch16_224.augreg_in21k",
        bottleneck_dim: int = 64,
    ) -> "PretrainedViT":
        """Build a timm ViT and load an explicit checkpoint before adaptation."""
        try:
            import timm
        except ImportError as error:
            raise ImportError("Install the project with `pip install -e .` to use timm") from error

        checkpoint_file = Path(checkpoint_path).expanduser()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"ViT checkpoint not found: {checkpoint_file}")
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        checkpoint: Any = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, Mapping):
            checkpoint = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must contain a state dictionary")
        state_dict = _remove_common_prefixes(checkpoint)
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(("head.", "head_dist."))
        }
        incompatible = backbone.load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if not key.startswith("head.")]
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint does not match the requested ViT architecture: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        return cls(backbone, bottleneck_dim=bottleneck_dim)

    @property
    def num_tasks(self) -> int:
        return len(self._routed_mlps[0].adapters)

    @property
    def _routed_mlps(self) -> list[RoutedAdapterMLP]:
        return [block.mlp for block in self.backbone.blocks]

    def add_task_adapter(self) -> None:
        """Append one adapter to every transformer block, freezing earlier tasks."""
        for routed_mlp in self._routed_mlps:
            routed_mlp.add_task_adapter()

    @contextmanager
    def _route_adapters(self, weights: Tensor | None):
        for routed_mlp in self._routed_mlps:
            routed_mlp.set_routing_weights(weights)
        try:
            yield
        finally:
            for routed_mlp in self._routed_mlps:
                routed_mlp.set_routing_weights(None)

    def forward_features(self, images: Tensor, adapter_weights: Tensor | None = None) -> Tensor:
        """Return pooled image features using the latest or weighted adapters."""
        if adapter_weights is not None and adapter_weights.shape != (images.shape[0], self.num_tasks):
            raise ValueError("adapter_weights must have shape (batch, number_of_tasks)")
        with self._route_adapters(adapter_weights):
            tokens = self.backbone.forward_features(images)
            if hasattr(self.backbone, "forward_head"):
                return self.backbone.forward_head(tokens, pre_logits=True)
            return tokens[:, 0]

    def forward(self, images: Tensor, adapter_weights: Tensor | None = None) -> Tensor:
        return self.forward_features(images, adapter_weights)

    def trainable_parameters(self):
        """Yield only parameters currently eligible for optimization."""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)


def _remove_common_prefixes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = str(key)
        while True:
            matching_prefix = next(
                (prefix for prefix in ("module.", "model.", "backbone.") if normalized_key.startswith(prefix)),
                None,
            )
            if matching_prefix is None:
                break
            normalized_key = normalized_key[len(matching_prefix) :]
        cleaned[normalized_key] = value
    return cleaned
