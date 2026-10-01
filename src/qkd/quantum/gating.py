"""Task-relevance scoring utilities for QGTM."""

import torch
from torch import Tensor


def normalize_task_scores(scores: Tensor, temperature: float = 1.0) -> Tensor:
    """Convert per-sample task scores to relevance weights.

    Args:
        scores: Tensor shaped ``(batch, tasks)``.
        temperature: Positive softmax temperature from Eq. 12.
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape (batch, tasks)")
    if scores.shape[1] == 0:
        raise ValueError("at least one historical task score is required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(scores / temperature, dim=1)