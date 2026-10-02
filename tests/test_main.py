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
        model_name="vit_base_patch16_224.augreg_in21k",
        bottleneck_dim=64,
    )

    encoder = main.create_encoder(args)

    assert encoder.backbone == "backbone"
    assert calls == [
        (
            ("vit_base_patch16_224.augreg_in21k",),
            {"pretrained": True, "num_classes": 0},
        )
    ]


def test_default_model_is_imagenet21k_pretrained_variant(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = main.parse_args()

    assert args.model_name == "vit_base_patch16_224.augreg_in21k"


def test_create_encoder_rejects_non_in21k_timm_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace())
    args = Namespace(
        checkpoint_path=None,
        pretrained=True,
        model_name="vit_base_patch16_224.augreg_in1k",
        bottleneck_dim=64,
    )

    with pytest.raises(ValueError, match="ImageNet-21K"):
        main.create_encoder(args)


def test_create_encoder_rejects_configured_missing_checkpoint(tmp_path):
    args = Namespace(
        checkpoint_path=str(tmp_path / "missing.pth"),
        pretrained=False,
        model_name="vit_base_patch16_224",
        bottleneck_dim=64,
    )

    with pytest.raises(FileNotFoundError, match="ViT checkpoint not found"):
        main.create_encoder(args)
