import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qkd.engine.trainer import IncrementalTrainer
from qkd.models.vit import PretrainedViT


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(4, 4)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.blocks = nn.ModuleList([TinyBlock()])

    def forward_features(self, inputs):
        for block in self.blocks:
            inputs = inputs + block.mlp(inputs)
        return inputs

    def forward_head(self, features, pre_logits=False):
        return features


def test_tiny_vit_uses_adapters_and_freezes_backbone():
    timm = pytest.importorskip("timm")
    backbone = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=32,
    )
    model = PretrainedViT(backbone, bottleneck_dim=8)
    model.eval()

    with torch.no_grad():
        features = model.forward_features(torch.randn(2, 3, 32, 32))

    assert features.shape == (2, backbone.num_features)
    assert model.num_tasks == 1
    assert all(
        not parameter.requires_grad
        for routed_mlp in model._routed_mlps
        for parameter in routed_mlp.base_mlp.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.trainable_parameters())


def test_adding_task_freezes_previous_adapter():
    timm = pytest.importorskip("timm")
    backbone = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=0,
        img_size=32,
    )
    model = PretrainedViT(backbone, bottleneck_dim=8)
    previous_adapter = model._routed_mlps[0].adapters[0]

    model.add_task_adapter()

    assert model.num_tasks == 2
    assert all(not parameter.requires_grad for parameter in previous_adapter.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model._routed_mlps[0].adapters[1].parameters()
    )


def test_incremental_trainer_freezes_completed_tasks():
    model = PretrainedViT(TinyBackbone(), bottleneck_dim=2)
    trainer = IncrementalTrainer(model, device=torch.device("cpu"))
    inputs = torch.randn(8, 4)
    labels = torch.tensor([0, 1] * 4)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)

    trainer.fit_task(0, loader, [0, 1], epochs=1)
    first_adapter = model._routed_mlps[0].adapters[0]
    frozen_weights = [parameter.detach().clone() for parameter in first_adapter.parameters()]
    assert all(not parameter.requires_grad for parameter in first_adapter.parameters())

    second_labels = torch.tensor([2, 3] * 4)
    second_loader = DataLoader(TensorDataset(inputs, second_labels), batch_size=4)
    trainer.fit_task(1, second_loader, [2, 3], epochs=1)

    assert model.num_tasks == 2
    assert all(not parameter.requires_grad for parameter in first_adapter.parameters())
    assert all(
        torch.equal(parameter, frozen)
        for parameter, frozen in zip(first_adapter.parameters(), frozen_weights)
    )
    assert all(
        not parameter.requires_grad
        for routed_mlp in model._routed_mlps
        for parameter in routed_mlp.adapters[1].parameters()
    )
    assert all(
        not parameter.requires_grad
        for head in trainer.task_heads
        for parameter in head.parameters()
    )


def test_incremental_trainer_reports_running_loss_and_accuracy(monkeypatch):
    import qkd.engine.trainer as trainer_module

    recorded_postfixes = []

    class RecordingProgress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **values):
            recorded_postfixes.append(values)

    monkeypatch.setattr(trainer_module, "tqdm", RecordingProgress)
    model = PretrainedViT(TinyBackbone(), bottleneck_dim=2)
    trainer = IncrementalTrainer(model, device=torch.device("cpu"))
    inputs = torch.randn(8, 4)
    labels = torch.tensor([0, 1] * 4)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)

    trainer.fit_task(0, loader, [0, 1], epochs=1)

    assert len(recorded_postfixes) == len(loader)
    assert all(set(metrics) == {"loss", "accuracy"} for metrics in recorded_postfixes)
    assert all(metrics["loss"] for metrics in recorded_postfixes)
    assert all(metrics["accuracy"].endswith("%") for metrics in recorded_postfixes)