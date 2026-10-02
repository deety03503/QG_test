"""Class-order and task splitting utilities for class-incremental benchmarks."""

from collections.abc import Sequence

import numpy as np


def make_class_order(num_classes: int, seed: int = 1993) -> list[int]:
    """Return a reproducible class permutation using NumPy's legacy RNG."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return np.random.RandomState(seed).permutation(num_classes).tolist()


def make_task_class_splits(
    class_order: Sequence[int],
    initial_classes: int,
    incremental_classes: int,
) -> list[list[int]]:
    """Split an ordered class list into the base task and incremental tasks."""
    if not class_order:
        raise ValueError("class_order must not be empty")
    if initial_classes < 0 or incremental_classes <= 0:
        raise ValueError("initial_classes must be non-negative and increment must be positive")
    if initial_classes >= len(class_order):
        raise ValueError("initial_classes must be smaller than the number of classes")
    if initial_classes and (len(class_order) - initial_classes) % incremental_classes:
        raise ValueError("remaining classes must divide evenly into incremental tasks")
    if not initial_classes and len(class_order) % incremental_classes:
        raise ValueError("class count must divide evenly into incremental tasks when no base task is used")

    splits: list[list[int]] = []
    offset = 0
    if initial_classes:
        splits.append(list(class_order[:initial_classes]))
        offset = initial_classes
    while offset < len(class_order):
        splits.append(list(class_order[offset : offset + incremental_classes]))
        offset += incremental_classes
    return splits
