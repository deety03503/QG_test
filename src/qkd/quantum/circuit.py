"""Differentiable state-vector quantum feature map used by QGTM."""

from math import pi

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QuantumFeatureMap(nn.Module):
    """Encode sample/task vectors and return their quantum-state fidelities.

    Input features are pooled to one angle per qubit and normalized to the
    ``[-pi, pi]`` range. Trainable ``Ry`` angle offsets and a
    nearest-neighbour CNOT chain form the state.
    """

    def __init__(self, input_dim: int, num_qubits: int, num_layers: int = 1):
        super().__init__()
        if input_dim <= 0 or num_qubits <= 0 or num_layers <= 0:
            raise ValueError("input_dim, num_qubits, and num_layers must be positive")
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.rotation_angles = nn.Parameter(torch.zeros(num_layers, num_qubits))

    def forward(self, sample_features: Tensor, task_states: Tensor) -> Tensor:
        """Return state fidelities with shape ``(batch, tasks)``."""
        if sample_features.ndim != 2 or sample_features.shape[1] != self.input_dim:
            raise ValueError("sample_features must have shape (batch, input_dim)")
        if task_states.ndim != 2 or task_states.shape[1] != self.input_dim:
            raise ValueError("task_states must have shape (tasks, input_dim)")
        if sample_features.shape[0] == 0:
            raise ValueError("sample_features must contain at least one sample")
        if not torch.isfinite(sample_features).all() or not torch.isfinite(task_states).all():
            raise ValueError("sample_features and task_states must contain finite values")
        if task_states.shape[0] == 0:
            return sample_features.new_empty((sample_features.shape[0], 0))

        dtype = self.rotation_angles.dtype
        sample_state = self._encode(sample_features.to(dtype=dtype))
        
        # Eq 10: |\phi_i> = \tilde{s}_i
        task_state = task_states.to(device=sample_features.device, dtype=dtype)
        state_dim = 1 << self.num_qubits
        if task_state.shape[1] > state_dim:
            task_state = task_state[:, :state_dim]
        elif task_state.shape[1] < state_dim:
            padding = task_state.new_zeros((task_state.shape[0], state_dim - task_state.shape[1]))
            task_state = torch.cat([task_state, padding], dim=1)
        task_state = F.normalize(task_state, p=2, dim=1)
        
        return (sample_state @ task_state.transpose(0, 1)).square().clamp(0.0, 1.0)

    def _encode(self, vectors: Tensor) -> Tensor:
        normalized = F.normalize(vectors, p=2, dim=1)
        pooled = F.adaptive_avg_pool1d(
            normalized.unsqueeze(1),
            output_size=self.num_qubits,
        ).squeeze(1)
        angles = F.normalize(pooled, p=2, dim=1) * pi

        batch_size = angles.shape[0]
        state = angles.new_zeros((batch_size, 1 << self.num_qubits))
        state[:, 0] = 1.0
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                angle = angles[:, qubit] + self.rotation_angles[layer, qubit]
                state = self._apply_ry(state, angle, qubit)
            for control in range(self.num_qubits - 1):
                state = self._apply_cnot(state, control, control + 1)
        return state

    def _apply_ry(self, state: Tensor, angles: Tensor, qubit: int) -> Tensor:
        tensor_state = state.reshape(state.shape[0], *([2] * self.num_qubits))
        axis = qubit + 1
        moved_state = tensor_state.movedim(axis, 1).reshape(state.shape[0], 2, -1)
        cosine = torch.cos(angles / 2).unsqueeze(1)
        sine = torch.sin(angles / 2).unsqueeze(1)
        zero = cosine * moved_state[:, 0] - sine * moved_state[:, 1]
        one = sine * moved_state[:, 0] + cosine * moved_state[:, 1]
        rotated = torch.stack((zero, one), dim=1).reshape_as(moved_state)
        return (
            rotated.reshape(state.shape[0], *([2] * self.num_qubits))
            .movedim(1, axis)
            .reshape_as(state)
        )

    def _apply_cnot(self, state: Tensor, control: int, target: int) -> Tensor:
        basis = torch.arange(state.shape[1], device=state.device)
        control_mask = 1 << (self.num_qubits - control - 1)
        target_mask = 1 << (self.num_qubits - target - 1)
        permutation = torch.where((basis & control_mask) != 0, basis ^ target_mask, basis)
        return state.index_select(1, permutation)
