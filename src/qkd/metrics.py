"""Evaluation metrics for class-incremental learning."""

from collections.abc import Sequence


def average_incremental_accuracy(stage_accuracies: Sequence[float]) -> float:
    """Compute A-bar as the mean accuracy across completed stages."""
    if not stage_accuracies:
        raise ValueError("at least one stage accuracy is required")
    return sum(stage_accuracies) / len(stage_accuracies)


def final_accuracy(stage_accuracies: Sequence[float]) -> float:
    """Return accuracy after the final incremental stage."""
    if not stage_accuracies:
        raise ValueError("at least one stage accuracy is required")
    return stage_accuracies[-1]
