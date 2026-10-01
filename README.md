# QKD Reproduction

Reproduction scaffold for **Quantum-Gated Task-interaction Knowledge
Distillation for Pre-trained Model-based Class-Incremental Learning**. The
companion *Class-Incremental Learning: A Survey* grounds the CIL protocol,
class-split convention, and memory-accounting choices.

The repository has a frozen ViT wrapper with per-task parallel adapters,
paper-derived CIL split/metric utilities, and sequential training for one
task-local adapter and classifier at a time. The quantum circuit, QGTM,
quantum-guided distillation, task-agnostic inference, and result aggregation
are still pending; this is not yet an end-to-end reproduction.

## Setup

Install a PyTorch build matching the local CPU/CUDA environment, then install
the project:

```powershell
python -m pip install -e ".[dev]"
```

Run the tests with:

```powershell
python -m pytest
```

For paper-faithful runs, set `QKD_VIT_B16_IN21K_WEIGHTS` to the exact local
ViT-B/16-IN21K checkpoint. `main.py` validates checkpoint/model key
compatibility when the file exists. If it is unset or missing, `main.py`
creates a randomly initialized ViT and starts training from task 1; the
backbone remains frozen, so those results are not paper-comparable. Do not
assume a similarly named ImageNet-21K checkpoint is identical to the authors'
checkpoint.

Run the configured CIFAR-100 B0-Inc10 training with:

```powershell
python main.py --data-root <cifar-100-root> --output-dir outputs/cifar100_b0_inc10
```

Set `QKD_VIT_B16_IN21K_WEIGHTS` or pass `--checkpoint-path <file>` to load the
pretrained backbone. Training settings can be changed with CLI options; the
Kaggle notebook only forwards its configuration to `main.py`.

```python
import os

from qkd.models.vit import PretrainedViT

encoder = PretrainedViT.from_pretrained(
  checkpoint_path=os.environ["QKD_VIT_B16_IN21K_WEIGHTS"],
  model_name="vit_base_patch16_224",
  bottleneck_dim=64,
)
features = encoder.forward_features(images)  # Uses the newest task adapter.
encoder.add_task_adapter()  # Call once when a new incremental task arrives.
features = encoder.forward_features(images, adapter_weights=task_weights)
```

## Layout

```text
configs/                 Experiment configurations
docs/                    Reproduction protocol and paper ambiguities
src/qkd/data/             Class-order and incremental split utilities
src/qkd/engine/           Sequential task training
src/qkd/models/           Frozen ViT and task-specific parallel adapters
src/qkd/quantum/          QGTM score/circuit components
src/qkd/losses.py         Relevance-weighted task distillation
src/qkd/metrics.py        Average and final incremental accuracy
tests/                    Unit and integration tests
```

Datasets, pretrained weights, checkpoints, and experiment outputs are excluded
from version control.

## Papers

- Da-Wei Zhou et al., *Class-Incremental Learning: A Survey*, TPAMI 2024,
  arXiv:2302.03648v2, available in the workspace as `2302.03648v2.pdf`.
- Linjie Li et al., *Quantum-Gated Task-interaction Knowledge Distillation
  for Pre-trained Model-based Class-Incremental Learning*, CVPR 2026 paper,
  available in the workspace as the accompanying QKD PDF.
