"""Run sequential CIFAR-100 task training for the QKD reproduction scaffold."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qkd.data.splits import make_class_order, make_task_class_splits
from qkd.engine.trainer import IncrementalTrainer
from qkd.metrics import final_accuracy
from qkd.models.vit import PretrainedViT

DEFAULT_MODEL_NAME = "vit_base_patch16_224.augreg_in21k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("QKD_CIFAR100_ROOT", "/kaggle/input/Cifar_100"),
        help="Directory containing the extracted cifar-100-python folder.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get(
            "QKD_VIT_B16_WEIGHTS",
            os.environ.get("QKD_VIT_B16_IN21K_WEIGHTS"),
        ),
        help="Optional local ViT checkpoint. If omitted, use --pretrained or random initialization.",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        action="store_true",
        help="Explicitly load pretrained weights from timm (the default).",
    )
    pretrained_group.add_argument(
        "--random-init",
        dest="pretrained",
        action="store_false",
        help="Skip pretrained weights and use a randomly initialized backbone.",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("QKD_OUTPUT_ROOT", "/kaggle/working/qkd_outputs"),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="timm model/weight tag; defaults to ViT-B/16 pretrained on ImageNet-21k.",
    )
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs-per-task", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--initial-classes", type=int, default=0)
    parser.add_argument("--incremental-classes", type=int, default=10)
    parser.add_argument("--svd-dim", type=int, default=12)
    parser.add_argument("--num-qubits", type=int, default=9)
    parser.add_argument("--circuit-layers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--lambda-sparse", type=float, default=0.05)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cifar100(data_root: str, image_transform: Callable | None = None):
    from torchvision import datasets, transforms

    if image_transform is None:
        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
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
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ViT checkpoint not found: {checkpoint_path}")
        print(f"Loading pretrained checkpoint: {checkpoint_path}")
        return PretrainedViT.from_pretrained(
            checkpoint_path=checkpoint_path,
            model_name=args.model_name,
            bottleneck_dim=args.bottleneck_dim,
        )

    import timm

    if args.pretrained:
        print(f"Loading pretrained ViT weights ({args.model_name}) from timm.")
        backbone = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    else:
        print("Random initialization requested; initializing the backbone randomly.")
        print("The frozen random backbone is not paper-comparable.")
        backbone = timm.create_model(args.model_name, pretrained=False, num_classes=0)
    return PretrainedViT(backbone, bottleneck_dim=args.bottleneck_dim)


def run(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers cannot be negative")
    if args.epochs_per_task <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs-per-task and learning-rate must be positive")
    if args.svd_dim <= 0 or args.num_qubits <= 0 or args.circuit_layers <= 0:
        raise ValueError("svd-dim, num-qubits, and circuit-layers must be positive")
    if args.temperature <= 0 or args.lambda_kd < 0 or args.lambda_sparse < 0:
        raise ValueError("temperature must be positive and loss weights non-negative")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    print(f"PyTorch: {torch.__version__}; device: {device}; visible GPUs: {gpu_count}")
    encoder = create_encoder(args).to(device).eval()
    import timm.data

    data_config = timm.data.resolve_model_data_config(encoder.backbone)
    image_transform = timm.data.create_transform(**data_config, is_training=False)
    train_dataset, test_dataset = load_cifar100(args.data_root, image_transform)
    train_indices = range(len(train_dataset))
    print(f"Training images: {len(train_dataset):,}; test images: {len(test_dataset):,}")
    class_order = make_class_order(num_classes=100, seed=args.seed)
    task_classes = make_task_class_splits(
        class_order,
        initial_classes=args.initial_classes,
        incremental_classes=args.incremental_classes,
    )
    print(f"Task class counts: {[len(classes) for classes in task_classes]}")

    model = torch.nn.DataParallel(encoder) if gpu_count >= 2 else encoder
    trainer = IncrementalTrainer(
        model,
        device=device,
        use_amp=device.type == "cuda",
        num_qubits=args.num_qubits,
        circuit_layers=args.circuit_layers,
        svd_dim=args.svd_dim,
        temperature=args.temperature,
        lambda_kd=args.lambda_kd,
        lambda_sparse=args.lambda_sparse,
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    losses_by_task: list[list[float]] = []
    task_agnostic_test_accuracies: list[float] = []
    per_task_test_accuracies: list[list[float]] = []
    accumulated_classes: set[int] = set()

    for task_id, classes in enumerate(task_classes):
        class_set = set(classes)
        indices = [
            index for index in train_indices if train_dataset.targets[index] in class_set
        ]
        train_loader = DataLoader(
            Subset(train_dataset, indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        seen_classes = accumulated_classes | class_set
        seen_test_indices = [
            index
            for index in range(len(test_dataset))
            if test_dataset.targets[index] in seen_classes
        ]
        seen_test_loader = DataLoader(
            Subset(test_dataset, seen_test_indices),
            batch_size=args.batch_size,
            shuffle=False,
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
        task_agnostic_test_accuracy = trainer.evaluate(seen_test_loader)
        task_agnostic_test_accuracies.append(task_agnostic_test_accuracy)
        current_final_test_accuracy = final_accuracy(task_agnostic_test_accuracies)
        current_task_test_accuracies: list[float] = []
        for learned_task_id, learned_classes in enumerate(task_classes[: task_id + 1]):
            learned_class_set = set(learned_classes)
            learned_test_indices = [
                index
                for index in range(len(test_dataset))
                if test_dataset.targets[index] in learned_class_set
            ]
            learned_test_loader = DataLoader(
                Subset(test_dataset, learned_test_indices),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            learned_task_accuracy = trainer.evaluate(learned_test_loader)
            current_task_test_accuracies.append(learned_task_accuracy)
            print(
                f"Task {learned_task_id + 1} predict test accuracy: "
                f"{100 * learned_task_accuracy:.2f}%"
            )
            del learned_test_loader
        per_task_test_accuracies.append(current_task_test_accuracies)
        accumulated_classes.update(classes)
        checkpoint_file = output_dir / f"qkd_task_{task_id + 1:02d}.pt"
        torch.save(
            {
                "task_id": task_id,
                "encoder": encoder.state_dict(),
                "classifier": trainer.classifier.state_dict(),
                "qgtm": trainer.qgtm.state_dict(),
                "task_class_ids": trainer.task_class_ids,
                "task_representations": trainer.task_representations,
                "epoch_losses": losses_by_task,
                "task_agnostic_test_accuracies": task_agnostic_test_accuracies,
                "per_task_test_accuracies": per_task_test_accuracies,
                "final_incremental_test_accuracy": current_final_test_accuracy,
                "config": vars(args),
            },
            checkpoint_file,
        )
        print(
            f"Task {task_id + 1}/{len(task_classes)} complete; "
            f"final loss={epoch_losses[-1]:.4f}; "
            f"task-agnostic test accuracy={task_agnostic_test_accuracy:.4f}; "
            f"final incremental test accuracy={current_final_test_accuracy:.4f}; "
            f"saved {checkpoint_file}"
        )
        del train_loader
        del seen_test_loader

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    final_test_accuracy = trainer.evaluate(test_loader)
    final_incremental_test_accuracy = final_accuracy(task_agnostic_test_accuracies)

    metrics = {
        "task_agnostic_test_accuracies": task_agnostic_test_accuracies,
        "per_task_test_accuracies": per_task_test_accuracies,
        "final_incremental_test_accuracy": final_incremental_test_accuracy,
        "final_test_accuracy": final_test_accuracy,
    }
    metrics_file = output_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Final incremental test accuracy: {final_incremental_test_accuracy:.4f}")
    print(f"Final test accuracy: {final_test_accuracy:.4f}")
    print(f"Saved evaluation metrics to {metrics_file}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
