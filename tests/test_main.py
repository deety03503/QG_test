import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

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
