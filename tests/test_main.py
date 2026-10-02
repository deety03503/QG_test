import json
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest
import torch

import main


def test_create_encoder_uses_timm_pretrained_weights(monkeypatch):
    calls = []

    class StubPretrainedViT:
        def __init__(self, backbone, bottleneck_dim):
            self.backbone = backbone
            self.bottleneck_dim = bottleneck_dim

    monkeypatch.setattr(main, "PretrainedViT", StubPretrainedViT)
    monkeypatch.setitem(
        sys.modules,
        "timm",
        SimpleNamespace(
            create_model=lambda *args, **kwargs: calls.append((args, kwargs)) or "backbone"
        ),
    )
    args = Namespace(
        checkpoint_path=None,
        pretrained=True,
        model_name="vit_base_patch16_224",
        bottleneck_dim=64,
    )

    encoder = main.create_encoder(args)

    assert encoder.backbone == "backbone"
    assert calls == [
        (
            ("vit_base_patch16_224",),
            {"pretrained": True, "num_classes": 0},
        )
    ]


def test_default_model_is_standard_pretrained_vit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = main.parse_args()

    assert args.model_name == "vit_base_patch16_224"
    assert args.dev_fraction == 0.1


def test_create_encoder_accepts_standard_pretrained_vit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main,
        "PretrainedViT",
        lambda backbone, bottleneck_dim: (backbone, bottleneck_dim),
    )
    monkeypatch.setitem(
        sys.modules,
        "timm",
        SimpleNamespace(
            create_model=lambda *args, **kwargs: calls.append((args, kwargs)) or "backbone"
        ),
    )
    args = Namespace(
        checkpoint_path=None,
        pretrained=True,
        model_name="vit_base_patch16_224",
        bottleneck_dim=64,
    )

    assert main.create_encoder(args) == ("backbone", 64)
    assert calls == [
        (
            ("vit_base_patch16_224",),
            {"pretrained": True, "num_classes": 0},
        )
    ]


def test_create_encoder_rejects_configured_missing_checkpoint(tmp_path):
    args = Namespace(
        checkpoint_path=str(tmp_path / "missing.pth"),
        pretrained=False,
        model_name="vit_base_patch16_224",
        bottleneck_dim=64,
    )

    with pytest.raises(FileNotFoundError, match="ViT checkpoint not found"):
        main.create_encoder(args)


def test_run_records_final_incremental_dev_accuracy(monkeypatch, tmp_path):
    class Dataset:
        targets = [0, 1, 0, 1]

        def __len__(self):
            return len(self.targets)

    class Encoder:
        backbone = object()

        def to(self, device):
            return self

        def eval(self):
            return self

        def state_dict(self):
            return {}

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.classifier = SimpleNamespace(state_dict=lambda: {})
            self.qgtm = SimpleNamespace(state_dict=lambda: {})
            self.task_class_ids = []
            self.task_representations = []
            self.epoch_dev_accuracies = []
            self.evaluation_results = iter([0.4, 0.65, 0.7])

        def fit_task(self, task_id, **kwargs):
            return [1.0]

        def evaluate(self, data_loader):
            return next(self.evaluation_results)

    class Loader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset

    timm = ModuleType("timm")
    timm.__path__ = []
    timm_data = ModuleType("timm.data")
    timm_data.resolve_model_data_config = lambda backbone: {}
    timm_data.create_transform = lambda **kwargs: object()
    timm.data = timm_data
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "timm.data", timm_data)
    monkeypatch.setattr(main, "set_seed", lambda seed: None)
    monkeypatch.setattr(main, "create_encoder", lambda args: Encoder())
    monkeypatch.setattr(main, "load_cifar100", lambda *args: (Dataset(), Dataset()))
    monkeypatch.setattr(main, "make_train_dev_split", lambda *args, **kwargs: ([0, 1], [2, 3]))
    monkeypatch.setattr(main, "make_class_order", lambda **kwargs: [0, 1, 2, 3])
    monkeypatch.setattr(main, "make_task_class_splits", lambda *args, **kwargs: [[0, 1], [2, 3]])
    monkeypatch.setattr(main, "DataLoader", Loader)
    monkeypatch.setattr(main, "IncrementalTrainer", Trainer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    saved_checkpoints = []
    monkeypatch.setattr(torch, "save", lambda payload, path: saved_checkpoints.append(payload))

    args = Namespace(
        batch_size=2,
        num_workers=0,
        dev_fraction=0.25,
        epochs_per_task=1,
        learning_rate=0.1,
        svd_dim=1,
        num_qubits=1,
        circuit_layers=1,
        temperature=1.0,
        lambda_kd=1.0,
        lambda_sparse=0.05,
        seed=42,
        data_root="unused",
        output_dir=str(tmp_path),
        initial_classes=2,
        incremental_classes=2,
    )

    main.run(args)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["task_agnostic_dev_accuracies"] == [0.4, 0.65]
    assert metrics["final_incremental_dev_accuracy"] == 0.65
    assert metrics["final_test_accuracy"] == 0.7
    assert [
        checkpoint["final_incremental_dev_accuracy"]
        for checkpoint in saved_checkpoints
    ] == [0.4, 0.65]
