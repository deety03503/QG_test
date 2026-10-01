import pytest
import torch

from qkd.losses import task_interaction_distillation
from qkd.quantum.gating import normalize_task_scores


def test_normalize_task_scores_is_per_sample_distribution():
    weights = normalize_task_scores(torch.tensor([[0.0, 1.0], [2.0, 0.0]]))

    assert weights.shape == (2, 2)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    assert weights[0, 1] > weights[0, 0]
    assert weights[1, 0] > weights[1, 1]


def test_normalize_task_scores_rejects_invalid_temperature():
    with pytest.raises(ValueError, match="temperature"):
        normalize_task_scores(torch.zeros(1, 2), temperature=0)


def test_task_interaction_distillation_is_zero_for_matching_logits():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    weights = torch.tensor([[0.25, 0.75]])

    loss = task_interaction_distillation(logits, logits[:, 0, :], weights)

    assert loss >= 0
    torch.testing.assert_close(loss, torch.tensor(0.75 * (torch.softmax(logits[0, 1], -1) * (
        torch.log_softmax(logits[0, 1], -1) - torch.log_softmax(logits[0, 0], -1)
    )).sum()))


def test_task_interaction_distillation_backpropagates_to_current_logits():
    old_logits = torch.randn(3, 2, 4)
    new_logits = torch.randn(3, 4, requires_grad=True)
    weights = torch.full((3, 2), 0.5)

    task_interaction_distillation(old_logits, new_logits, weights).backward()

    assert new_logits.grad is not None
    assert torch.isfinite(new_logits.grad).all()