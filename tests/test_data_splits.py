import pytest

from qkd.data.splits import make_class_order, make_task_class_splits, make_train_dev_split


def test_cifar100_base0_inc10_has_ten_non_overlapping_tasks():
    order = make_class_order(100, seed=1993)
    tasks = make_task_class_splits(order, initial_classes=0, incremental_classes=10)

    assert len(tasks) == 10
    assert all(len(task) == 10 for task in tasks)
    assert sorted(class_id for task in tasks for class_id in task) == list(range(100))


def test_base_increment_split_rejects_uneven_remainder():
    with pytest.raises(ValueError, match="divide evenly"):
        make_task_class_splits(range(100), initial_classes=50, incremental_classes=12)


def test_train_dev_split_is_reproducible_and_stratified():
    labels = [class_id for class_id in range(3) for _ in range(10)]

    train_indices, dev_indices = make_train_dev_split(labels, dev_fraction=0.2, seed=7)
    repeated_train_indices, repeated_dev_indices = make_train_dev_split(
        labels,
        dev_fraction=0.2,
        seed=7,
    )

    assert train_indices == repeated_train_indices
    assert dev_indices == repeated_dev_indices
    assert set(train_indices).isdisjoint(dev_indices)
    assert sorted(train_indices + dev_indices) == list(range(len(labels)))
    assert [sum(labels[index] == class_id for index in dev_indices) for class_id in range(3)] == [
        2,
        2,
        2,
    ]


@pytest.mark.parametrize("dev_fraction", [0, 1, -0.1, 1.1])
def test_train_dev_split_rejects_invalid_fraction(dev_fraction):
    with pytest.raises(ValueError, match="dev_fraction"):
        make_train_dev_split([0, 0, 1, 1], dev_fraction=dev_fraction)
