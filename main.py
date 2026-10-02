"""Run sequential CIFAR-100 task training for the QKD reproduction scaffold."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qkd.data.splits import make_class_order, make_task_class_splits
from qkd.engine.trainer import IncrementalTrainer
from qkd.models.vit import PretrainedViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("QKD_CIFAR100_ROOT", "/kaggle/input/Cifar_100"),
        help="Directory containing the extracted cifar-100-python folder.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get("QKD_VIT_B16_IN21K_WEIGHTS"),
        help="Optional exact ViT-B/16-IN21K checkpoint; absent means random initialization.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("QKD_OUTPUT_ROOT", "/kaggle/working/qkd_outputs"),
    )
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs-per-task", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--initial-classes", type=int, default=0)
    parser.add_argument("--incremental-classes", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cifar100(data_root: str):
    from torchvision import datasets, transforms

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    candidates: list[Path] = [Path(data_root).expanduser()]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for mount in kaggle_input.iterdir():
            if mount.is_dir():
                candidates.append(mount)
                candidates.extend(
                    extracted.parent
                    for extracted in mount.rglob("cifar-100-python")
                    if extracted.is_dir()
                )

    checked: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        try:
            train_dataset = datasets.CIFAR100(
                root=str(candidate),
                train=True,
                download=False,
                transform=image_transform,
            )
            test_dataset = datasets.CIFAR100(
                root=str(candidate),
                train=False,
                download=False,
                transform=image_transform,
            )
            print(f"Using CIFAR-100 from: {candidate}")
            print(f"Train images: {len(train_dataset):,}; test images: {len(test_dataset):,}")
            return train_dataset, test_dataset
        except (OSError, RuntimeError, FileNotFoundError):
            continue

    raise FileNotFoundError(
        "Could not find an extracted CIFAR-100 dataset. Set --data-root or "
        "QKD_CIFAR100_ROOT to its parent directory."
    )


def create_encoder(args: argparse.Namespace) -> PretrainedViT:
    checkpoint_path = Path(args.checkpoint_path).expanduser() if args.checkpoint_path else None
    if checkpoint_path is not None and checkpoint_path.is_file():
        print(f"Loading pretrained checkpoint: {checkpoint_path}")
        return PretrainedViT.from_pretrained(
            checkpoint_path=checkpoint_path,
            model_name=args.model_name,
            bottleneck_dim=args.bottleneck_dim,
        )

    import timm

    if checkpoint_path is None:
        print("No ViT checkpoint configured; initializing the backbone with random weights.")
    else:
        print(f"Checkpoint not found at {checkpoint_path}; initializing with random weights.")
    print("Starting sequential training at task 1 with this new model.")
    print("The randomly initialized backbone remains frozen; results are not paper-comparable.")
    backbone = timm.create_model(args.model_name, pretrained=False, num_classes=0)
    return PretrainedViT(backbone, bottleneck_dim=args.bottleneck_dim)


def run(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers cannot be negative")
    if args.epochs_per_task <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs-per-task and learning-rate must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"PyTorch: {torch.__version__}; device: {device}; visible GPUs: {gpu_count}")
    train_dataset, _ = load_cifar100(args.data_root)
    class_order = make_class_order(num_classes=100, seed=args.seed)
    task_classes = make_task_class_splits(
        class_order,
        initial_classes=args.initial_classes,
        incremental_classes=args.incremental_classes,
    )
    print(f"Task class counts: {[len(classes) for classes in task_classes]}")

    encoder = create_encoder(args).to(device).eval()
    model = torch.nn.DataParallel(encoder) if gpu_count >= 2 else encoder
    trainer = IncrementalTrainer(model, device=device, use_amp=device.type == "cuda")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    losses_by_task: list[list[float]] = []

    for task_id, classes in enumerate(task_classes):
        class_set = set(classes)
        indices = [
            index
            for index, label in enumerate(train_dataset.targets)
            if label in class_set
        ]
        train_loader = DataLoader(
            Subset(train_dataset, indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        epoch_losses = trainer.fit_task(
            task_id=task_id,
            train_loader=train_loader,
            seen_classes=classes,
            epochs=args.epochs_per_task,
            learning_rate=args.learning_rate,
        )
        losses_by_task.append(epoch_losses)
        checkpoint_file = output_dir / f"qkd_task_{task_id + 1:02d}.pt"
        torch.save(
            {
                "task_id": task_id,
                "encoder": encoder.state_dict(),
                "task_heads": trainer.task_heads.state_dict(),
                "task_class_ids": trainer.task_class_ids,
                "epoch_losses": losses_by_task,
                "config": vars(args),
            },
            checkpoint_file,
        )
        print(
            f"Task {task_id + 1}/{len(task_classes)} complete; "
            f"final loss={epoch_losses[-1]:.4f}; saved {checkpoint_file}"
        )
        del train_loader

    print("All task adapters and task-local classifiers are frozen.")
    print("QGTM, TIKD, and task-agnostic evaluation are not implemented yet.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
