"""Optimization objectives used by QKD."""

import torch
from torch import Tensor
from torch.nn import functional as F


def task_interaction_distillation(
    historical_logits: Tensor,
    current_logits: Tensor,
    relevance: Tensor,
) -> Tensor:
    """Compute relevance-weighted KL(old adapter || current adapter).

    ``historical_logits`` has shape ``(batch, tasks, classes)``;
    ``current_logits`` has shape ``(batch, classes)``; and ``relevance``
    has shape ``(batch, tasks)``. The paper defines KL between softmax
    outputs without a distillation temperature.
    """
    if historical_logits.ndim != 3 or current_logits.ndim != 2:
        raise ValueError("logits must have shapes (batch, tasks, classes) and (batch, classes)")
    expected_relevance_shape = historical_logits.shape[:2]
    if relevance.shape != expected_relevance_shape:
        raise ValueError("relevance must have shape (batch, tasks)")
    if historical_logits.shape[0] != current_logits.shape[0]:
        raise ValueError("historical and current logits must share the batch dimension")
    if historical_logits.shape[2] != current_logits.shape[1]:
        raise ValueError("historical and current logits must share the class dimension")

    if historical_logits.dtype in (torch.float16, torch.bfloat16):
        historical_logits = historical_logits.float()
    if current_logits.dtype in (torch.float16, torch.bfloat16):
        current_logits = current_logits.float()
    if relevance.dtype in (torch.float16, torch.bfloat16):
        relevance = relevance.float()
    old_log_probs = F.log_softmax(historical_logits, dim=-1)
    old_probs = old_log_probs.exp()
    new_log_probs = F.log_softmax(current_logits, dim=-1).unsqueeze(1)
    per_task_kl = (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1)
    return (per_task_kl * relevance).sum(dim=1).mean()


def task_gate_sparsity(relevance: Tensor) -> Tensor:
    """Compute mean entropy of the normalized task relevance distribution."""
    if relevance.ndim != 2:
        raise ValueError("relevance must have shape (batch, tasks)")
    if relevance.shape[1] == 0:
        raise ValueError("at least one task relevance score is required")
    if not torch.isfinite(relevance).all():
        raise ValueError("relevance must contain finite values")
    if (relevance < 0).any():
        raise ValueError("relevance must contain non-negative probabilities")

    if relevance.dtype in (torch.float16, torch.bfloat16):
        relevance = relevance.float()
    log_relevance = relevance.clamp_min(torch.finfo(relevance.dtype).tiny).log()
    entropy = -(relevance * log_relevance).sum(dim=1)
    return entropy.mean()
