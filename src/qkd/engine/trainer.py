"""Task-by-task training and task-agnostic inference for QKD."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from qkd.losses import task_gate_sparsity, task_interaction_distillation
from qkd.quantum.circuit import QuantumFeatureMap
from qkd.quantum.gating import normalize_task_scores


class IncrementalTrainer:
    """Train a global classifier and one new adapter at each CIL stage."""

    def __init__(
        self,
        encoder: nn.Module,
        device: torch.device,
        use_amp: bool = False,
        num_qubits: int = 9,
        circuit_layers: int = 1,
        svd_dim: int = 12,
        temperature: float = 1.0,
        lambda_kd: float = 1.0,
        lambda_sparse: float = 0.05,
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        base_encoder = self._base_encoder
        if not hasattr(base_encoder, "hidden_dim"):
            raise TypeError("encoder must expose hidden_dim")
        if svd_dim <= 0:
            raise ValueError("svd_dim must be positive")
        if temperature <= 0 or lambda_kd < 0 or lambda_sparse < 0:
            raise ValueError("temperature must be positive and loss weights non-negative")
        self.qgtm = QuantumFeatureMap(
            input_dim=base_encoder.hidden_dim,
            num_qubits=num_qubits,
            num_layers=circuit_layers,
        ).to(device)
        self.temperature = temperature
        self.lambda_kd = lambda_kd
        self.lambda_sparse = lambda_sparse
        self.svd_dim = svd_dim
        self.classifier: nn.Linear | None = None
        self.task_class_ids: list[list[int]] = []
        self.task_representations: list[Tensor] = []

    @property
    def _base_encoder(self) -> nn.Module:
        return self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder

    def fit_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        seen_classes: Iterable[int],
        epochs: int = 20,
        learning_rate: float = 0.05,
    ) -> list[float]:
        """Train the next adapter with CE, QKD, and entropy-based gate sparsity."""
        if task_id != len(self.task_class_ids):
            raise ValueError("tasks must be trained sequentially starting at task 0")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")

        class_ids = [int(class_id) for class_id in seen_classes]
        if not class_ids or len(set(class_ids)) != len(class_ids):
            raise ValueError("seen_classes must contain unique class IDs")
        if any(class_id < 0 for class_id in class_ids):
            raise ValueError("class IDs must be non-negative")
        previously_seen = {class_id for task in self.task_class_ids for class_id in task}
        if previously_seen.intersection(class_ids):
            raise ValueError("incremental tasks must contain disjoint class IDs")

        encoder = self._base_encoder
        if not hasattr(encoder, "add_task_adapter") or not hasattr(encoder, "num_tasks"):
            raise TypeError("encoder must expose add_task_adapter() and num_tasks")
        if task_id == 0:
            if encoder.num_tasks != 1:
                raise ValueError("task 0 requires an encoder initialized with exactly one adapter")
        else:
            if encoder.num_tasks != task_id:
                raise ValueError("encoder adapter count does not match the next task ID")
            encoder.add_task_adapter()

        all_seen_classes = [class_id for task in self.task_class_ids for class_id in task]
        all_seen_classes.extend(class_ids)
        self._expand_classifier(max(all_seen_classes) + 1, encoder.hidden_dim)
        assert self.classifier is not None
        self.classifier.requires_grad_(True)

        trainable_parameters = [
            parameter for parameter in self.encoder.parameters() if parameter.requires_grad
        ]
        trainable_parameters.extend(self.classifier.parameters())
        trainable_parameters.extend(self.qgtm.parameters())
        optimizer = torch.optim.SGD(trainable_parameters, lr=learning_rate, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)
        current_class_position = {
            class_id: index for index, class_id in enumerate(class_ids)
        }
        current_column_indices = torch.tensor(class_ids, device=self.device)
        previous_classes = [
            class_id for task in self.task_class_ids for class_id in task
        ]
        previous_column_indices = torch.tensor(
            previous_classes,
            device=self.device,
        )
        target_lookup = torch.full(
            (max(all_seen_classes) + 1,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        target_lookup[torch.tensor(class_ids, device=self.device)] = torch.tensor(
            [current_class_position[class_id] for class_id in class_ids],
            device=self.device,
        )
        epoch_losses: list[float] = []
        self.encoder.eval()
        self.qgtm.train()
        self.classifier.train()
        for epoch in range(epochs):
            total_loss = 0.0
            sample_count = 0
            correct_count = 0
            progress = tqdm(
                train_loader,
                desc=f"Task {task_id + 1} | Epoch {epoch + 1}/{epochs}",
                unit="batch",
            )
            for batch_index, (images, labels) in enumerate(progress):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
                if not torch.isfinite(images).all():
                    raise FloatingPointError(
                        f"non-finite input images in task {task_id + 1}, "
                        f"epoch {epoch + 1}, batch {batch_index + 1}"
                    )
                if labels.numel() and (
                    labels.min().item() < 0 or labels.max().item() >= target_lookup.numel()
                ):
                    raise ValueError("training batch contains class IDs outside seen_classes")
                targets = target_lookup[labels]
                if (targets < 0).any():
                    raise ValueError("training batch contains labels outside the current task")

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    current_features = self.encoder(images)
                    all_current_logits = self.classifier(current_features)
                    current_logits = all_current_logits[:, current_column_indices]
                    if not torch.isfinite(current_features).all():
                        raise FloatingPointError(
                            f"non-finite encoder features in task {task_id + 1}, "
                            f"epoch {epoch + 1}, batch {batch_index + 1}"
                        )
                    if not torch.isfinite(all_current_logits).all():
                        raise FloatingPointError(
                            f"non-finite classifier logits in task {task_id + 1}, "
                            f"epoch {epoch + 1}, batch {batch_index + 1}"
                        )
                    loss_ce = F.cross_entropy(current_logits.float(), targets)

                    if self.task_representations:
                        query_features = self._routed_features(images, task_id=0)
                        task_states = torch.stack(self.task_representations).to(self.device)
                        task_scores = self.qgtm(query_features, task_states)
                        relevance = normalize_task_scores(task_scores, self.temperature)
                        historical_logits = self._historical_logits(images, len(self.task_representations))
                        historical_logits = historical_logits[:, :, previous_column_indices]
                        loss_kd = task_interaction_distillation(
                            historical_logits,
                            all_current_logits[:, previous_column_indices],
                            relevance,
                        )
                        loss_sparse = task_gate_sparsity(relevance)
                    else:
                        loss_kd = current_logits.new_zeros(())
                        loss_sparse = current_logits.new_zeros(())
                    loss = loss_ce + self.lambda_kd * loss_kd + self.lambda_sparse * loss_sparse
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss in task {task_id + 1}, epoch {epoch + 1}, "
                            f"batch {batch_index + 1} "
                            f"(ce={loss_ce.detach().item()}, "
                            f"kd={loss_kd.detach().item()}, "
                            f"sparse={loss_sparse.detach().item()})"
                        )

                batch_size = labels.size(0)
                total_loss += loss.detach().item() * batch_size
                correct_count += (current_logits.detach().argmax(dim=1) == targets).sum().item()
                sample_count += batch_size
                progress.set_postfix(
                    loss=f"{total_loss / sample_count:.4f}",
                    accuracy=f"{100 * correct_count / sample_count:.2f}%",
                )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                self._mask_previous_classifier_gradients(previous_classes)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=1.0,
                    error_if_nonfinite=False,
                )
                if not scaler.is_enabled() and not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"non-finite gradient norm in task {task_id + 1}, "
                        f"epoch {epoch + 1}, batch {batch_index + 1}"
                    )
                scaler.step(optimizer)
                scaler.update()
                if not scaler.is_enabled() and any(
                    not torch.isfinite(parameter).all()
                    for parameter in trainable_parameters
                ):
                    raise FloatingPointError(
                        f"optimizer produced non-finite parameters in task {task_id + 1}, "
                        f"epoch {epoch + 1}, batch {batch_index + 1}"
                    )

            if sample_count == 0:
                raise ValueError("train_loader must contain at least one batch")
            epoch_losses.append(total_loss / sample_count)
            scheduler.step()

        for routed_mlp in encoder._routed_mlps:
            routed_mlp.adapters[-1].requires_grad_(False)
        self.classifier.requires_grad_(False)
        self.task_class_ids.append(class_ids)
        self.task_representations.append(
            self._adapter_task_representation(encoder, task_id).cpu()
        )
        return epoch_losses

    def _mask_previous_classifier_gradients(self, previous_classes: list[int]) -> None:
        """Keep learned class weights fixed while later tasks are trained."""
        if self.classifier is None or not previous_classes:
            return
        previous_indices = torch.tensor(previous_classes, device=self.device)
        for parameter in (self.classifier.weight, self.classifier.bias):
            if parameter.grad is not None:
                parameter.grad.index_fill_(0, previous_indices, 0)

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        """Predict original class IDs using QGTM-weighted adapter fusion."""
        if self.classifier is None or not self.task_representations:
            raise RuntimeError("fit at least one task before inference")
        images = images.to(self.device, non_blocking=True)
        query_features = self._routed_features(images, task_id=0)
        task_states = torch.stack(self.task_representations).to(self.device)
        scores = self.qgtm(query_features, task_states)
        relevance = normalize_task_scores(scores, self.temperature)
        fused_features = self.encoder(images, adapter_weights=relevance)
        class_ids = [class_id for task in self.task_class_ids for class_id in task]
        logits = self.classifier(fused_features)[:, class_ids]
        return torch.tensor(class_ids, device=self.device)[logits.argmax(dim=1)]

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> float:
        """Return task-agnostic top-1 accuracy as a fraction in ``[0, 1]``."""
        if not self.task_class_ids:
            raise RuntimeError("fit at least one task before evaluation")
        self.encoder.eval()
        self.qgtm.eval()
        if self.classifier is not None:
            self.classifier.eval()
        correct = 0
        total = 0
        for images, labels in data_loader:
            predictions = self.predict(images)
            labels = labels.to(self.device, dtype=torch.long, non_blocking=True)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
        if total == 0:
            raise ValueError("evaluation data loader must contain at least one sample")
        return correct / total

    def _expand_classifier(self, output_dim: int, hidden_dim: int) -> None:
        if self.classifier is not None and self.classifier.out_features >= output_dim:
            return
        expanded = nn.Linear(hidden_dim, output_dim).to(self.device)
        if self.classifier is not None:
            with torch.no_grad():
                expanded.weight[: self.classifier.out_features].copy_(self.classifier.weight)
                expanded.bias[: self.classifier.out_features].copy_(self.classifier.bias)
        self.classifier = expanded

    def _routed_features(self, images: Tensor, task_id: int) -> Tensor:
        adapter_count = self._base_encoder.num_tasks
        if task_id < 0 or task_id >= adapter_count:
            raise ValueError("task_id is outside the available adapter range")
        weights = images.new_zeros((images.shape[0], adapter_count))
        weights[:, task_id] = 1.0
        return self.encoder(images, adapter_weights=weights)

    def _historical_logits(self, images: Tensor, task_count: int) -> Tensor:
        if self.classifier is None:
            raise RuntimeError("classifier is not initialized")
        logits = []
        with torch.no_grad():
            for task_id in range(task_count):
                old_features = self._routed_features(images, task_id)
                logits.append(self.classifier(old_features))
        return torch.stack(logits, dim=1)

    def _adapter_task_representation(self, encoder: nn.Module, task_id: int) -> Tensor:
        """Build the normalized rank-r task vector from adapter projection matrices."""
        projection_columns = []
        for routed_mlp in encoder._routed_mlps:
            adapter = routed_mlp.adapters[task_id]
            projection_columns.extend((adapter.down.weight.detach().T, adapter.up.weight.detach()))
        representation_matrix = torch.cat(projection_columns, dim=1).to(
            device="cpu",
            dtype=torch.float32,
        )
        left, singular_values, right = torch.linalg.svd(
            representation_matrix,
            full_matrices=False,
        )
        rank = min(self.svd_dim, singular_values.numel())
        ones = representation_matrix.new_ones((right.shape[1],))
        representation = left[:, :rank] @ (
            singular_values[:rank] * (right[:rank] @ ones)
        )
        norm = torch.linalg.vector_norm(representation)
        if norm <= torch.finfo(representation.dtype).eps:
            raise ValueError("adapter task representation has zero norm")
        return representation / norm
