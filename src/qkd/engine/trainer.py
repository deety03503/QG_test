"""Task-by-task training orchestration for QKD."""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class IncrementalTrainer:
    """Train one task head and adapter at a time while freezing prior parameters."""

    def __init__(self, encoder: nn.Module, device: torch.device, use_amp: bool = False):
        self.encoder = encoder.to(device)
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.task_heads = nn.ModuleList()
        self.task_class_ids: list[list[int]] = []

    def fit_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        seen_classes: Iterable[int],
        epochs: int = 20,
        learning_rate: float = 0.05,
    ) -> list[float]:
        """Train the next adapter and its task-local classifier, then freeze both."""
        if task_id != len(self.task_heads):
            raise ValueError("tasks must be trained sequentially starting at task 0")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")

        class_ids = [int(class_id) for class_id in seen_classes]
        if not class_ids or len(set(class_ids)) != len(class_ids):
            raise ValueError("seen_classes must contain unique class IDs")

        encoder = self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder
        if not hasattr(encoder, "add_task_adapter") or not hasattr(encoder, "num_tasks"):
            raise TypeError("encoder must expose add_task_adapter() and num_tasks")
        if task_id == 0:
            if encoder.num_tasks != 1:
                raise ValueError("task 0 requires an encoder initialized with exactly one adapter")
        else:
            if encoder.num_tasks != task_id:
                raise ValueError("encoder adapter count does not match the next task ID")
            encoder.add_task_adapter()

        for previous_head in self.task_heads:
            previous_head.requires_grad_(False)
        head = nn.Linear(encoder.hidden_dim, len(class_ids)).to(self.device)
        class_id_tensor = torch.tensor(class_ids, device=self.device)
        trainable_parameters = [parameter for parameter in self.encoder.parameters() if parameter.requires_grad]
        trainable_parameters.extend(head.parameters())
        optimizer = torch.optim.SGD(trainable_parameters, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        epoch_losses: list[float] = []

        self.encoder.eval()
        head.train()
        for epoch in range(epochs):
            total_loss = 0.0
            sample_count = 0
            correct_count = 0
            progress = tqdm(
                train_loader,
                desc=f"Task {task_id + 1} | Epoch {epoch + 1}/{epochs}",
                unit="batch",
            )
            for images, labels in progress:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                matches = labels.unsqueeze(1).eq(class_id_tensor)
                if not matches.any(dim=1).all():
                    raise ValueError("training batch contains labels outside seen_classes")
                targets = matches.to(torch.long).argmax(dim=1)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    logits = head(self.encoder(images))
                    loss = F.cross_entropy(logits, targets)
                batch_size = labels.size(0)
                total_loss += loss.detach().item() * batch_size
                correct_count += (logits.detach().argmax(dim=1) == targets).sum().item()
                sample_count += batch_size
                progress.set_postfix(
                    loss=f"{total_loss / sample_count:.4f}",
                    accuracy=f"{100 * correct_count / sample_count:.2f}%",
                )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if sample_count == 0:
                raise ValueError("train_loader must contain at least one batch")
            epoch_losses.append(total_loss / sample_count)
            scheduler.step()

        for routed_mlp in encoder._routed_mlps:
            routed_mlp.adapters[-1].requires_grad_(False)
        head.requires_grad_(False)
        self.task_heads.append(head)
        self.task_class_ids.append(class_ids)
        return epoch_losses
