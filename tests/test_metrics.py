import pytest

from qkd.metrics import average_incremental_accuracy, final_accuracy


def test_incremental_accuracy_metrics():
    accuracies = [0.9, 0.7, 0.5]

    assert average_incremental_accuracy(accuracies) == pytest.approx(0.7)
    assert final_accuracy(accuracies) == 0.5
