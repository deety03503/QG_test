import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qkd.engine.trainer import IncrementalTrainer
from qkd.models.vit import PretrainedViT


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(2, 2)


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 2
        self.blocks = nn.ModuleList([_ToyBlock()])

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.blocks[0].mlp(images)

    def forward_head(self, features: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return features


def _loader(labels: list[int]) -> DataLoader:
    images = torch.tensor(
        [[float(label % 2), float((label + 1) % 2)] for label in labels]
    )
    return DataLoader(
        TensorDataset(images, torch.tensor(labels, dtype=torch.long)),
        batch_size=4,
        shuffle=False,
    )


def test_later_task_preserves_old_classifier_rows_and_adapters(
    monkeypatch,
) -> None:
    torch.manual_seed(7)
    encoder = PretrainedViT(_ToyBackbone(), bottleneck_dim=2)
    trainer = IncrementalTrainer(
        encoder,
        device=torch.device("cpu"),
        num_qubits=2,
        svd_dim=1,
    )

    trainer.fit_task(0, _loader([0, 1, 0, 1]), [0, 1], epochs=1, learning_rate=0.05)
    assert trainer.classifier is not None
    old_classifier_weight = trainer.classifier.weight.detach().clone()
    old_classifier_bias = trainer.classifier.bias.detach().clone()
    old_adapter_state = {
        name: parameter.detach().clone()
        for name, parameter in encoder._routed_mlps[0].adapters[0].state_dict().items()
    }
    expand_classifier = trainer._expand_classifier
    new_classifier_weight: list[torch.Tensor] = []

    def capture_expanded_rows(output_dim: int, hidden_dim: int) -> None:
        expand_classifier(output_dim, hidden_dim)
        if output_dim > 2:
            assert trainer.classifier is not None
            new_classifier_weight.append(trainer.classifier.weight[2:].detach().clone())

    monkeypatch.setattr(trainer, "_expand_classifier", capture_expanded_rows)

    trainer.fit_task(1, _loader([2, 3, 2, 3]), [2, 3], epochs=1, learning_rate=0.05)

    assert trainer.classifier is not None
    assert len(new_classifier_weight) == 1
    torch.testing.assert_close(trainer.classifier.weight[:2], old_classifier_weight)
    torch.testing.assert_close(trainer.classifier.bias[:2], old_classifier_bias)
    assert not torch.equal(trainer.classifier.weight[2:], new_classifier_weight[0])
    for name, parameter in encoder._routed_mlps[0].adapters[0].state_dict().items():
        torch.testing.assert_close(parameter, old_adapter_state[name])
