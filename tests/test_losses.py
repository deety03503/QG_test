import pytest
import torch

from qkd.losses import task_gate_sparsity


def test_task_gate_sparsity_is_lower_for_concentrated_relevance() -> None:
    uniform = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    concentrated = torch.tensor([[0.9, 0.1], [0.95, 0.05]])

    assert task_gate_sparsity(concentrated) < task_gate_sparsity(uniform)


def test_task_gate_sparsity_has_nonzero_gradient_for_softmax_scores() -> None:
    scores = torch.tensor([[0.2, 0.8, 0.5]], requires_grad=True)
    relevance = torch.softmax(scores, dim=1)

    task_gate_sparsity(relevance).backward()

    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad) > 0


def test_task_gate_sparsity_is_zero_for_single_task() -> None:
    relevance = torch.ones((2, 1))

    assert task_gate_sparsity(relevance).item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("relevance", "message"),
    [
        (torch.empty((2, 0)), "at least one task"),
        (torch.tensor([[-0.1, 1.1]]), "non-negative"),
        (torch.tensor([[float("nan"), 1.0]]), "finite"),
    ],
)
def test_task_gate_sparsity_rejects_invalid_relevance(
    relevance: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        task_gate_sparsity(relevance)
