"""Parameterized quantum feature-map interface for QGTM."""

from torch import Tensor, nn


class QuantumFeatureMap(nn.Module):
    """Interface for the paper's encoding, variational circuit, and measurement.

    State preparation and measurement are intentionally left for the resolved
    backend implementation; the supplied paper text does not define them
    sufficiently to choose a faithful circuit implementation.
    """

    def __init__(self, input_dim: int, num_qubits: int, num_layers: int = 1):
        super().__init__()
        if input_dim <= 0 or num_qubits <= 0 or num_layers <= 0:
            raise ValueError("input_dim, num_qubits, and num_layers must be positive")
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def forward(self, sample_features: Tensor, task_states: Tensor) -> Tensor:
        """Return one sample-to-task correlation score per input/task pair."""
        raise NotImplementedError("Implement the paper-verified QGTM circuit backend")
