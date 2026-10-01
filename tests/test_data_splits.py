import pytest

from qkd.data.splits import make_class_order, make_task_class_splits


def test_cifar100_base0_inc10_has_ten_non_overlapping_tasks():
    order = make_class_order(100, seed=1993)
    tasks = make_task_class_splits(order, initial_classes=0, incremental_classes=10)

    assert len(tasks) == 10
    assert all(len(task) == 10 for task in tasks)
    assert sorted(class_id for task in tasks for class_id in task) == list(range(100))


def test_base_increment_split_rejects_uneven_remainder():
    with pytest.raises(ValueError, match="divide evenly"):
        make_task_class_splits(range(100), initial_classes=50, incremental_classes=12)
