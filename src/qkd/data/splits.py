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


def make_train_dev_split(
    labels: Sequence[int],
    dev_fraction: float = 0.1,
    seed: int = 1993,
) -> tuple[list[int], list[int]]:
    """Create a reproducible, class-stratified split of training-set indices."""
    if len(labels) == 0:
        raise ValueError("labels must not be empty")
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")

    targets = np.asarray(labels)
    if targets.ndim != 1:
        raise ValueError("labels must be a one-dimensional sequence")
    rng = np.random.RandomState(seed)
    train_indices: list[int] = []
    dev_indices: list[int] = []
    for class_id in np.unique(targets):
        class_indices = np.flatnonzero(targets == class_id)
        if class_indices.size < 2:
            raise ValueError("each class must have at least two samples to split train and dev")
        shuffled_indices = rng.permutation(class_indices)
        dev_count = min(
            max(int(round(class_indices.size * dev_fraction)), 1),
            class_indices.size - 1,
        )
        dev_indices.extend(shuffled_indices[:dev_count].tolist())
        train_indices.extend(shuffled_indices[dev_count:].tolist())

    return sorted(train_indices), sorted(dev_indices)
