# QKD Reproduction

Reproduction scaffold for **Quantum-Gated Task-interaction Knowledge
Distillation for Pre-trained Model-based Class-Incremental Learning**. The
companion *Class-Incremental Learning: A Survey* grounds the CIL protocol,
class-split convention, and memory-accounting choices.

The repository implements a frozen ViT wrapper with per-task parallel
adapters, the paper's trainable `Ry`/CNOT state-vector QGTM, fidelity-based
task relevance, relevance-weighted task-interaction distillation, incremental
global classification, and task-agnostic adapter fusion. It evaluates after
each task and reports average incremental and final accuracy. Use the exact
pretrained checkpoint and the paper's dataset protocol for a comparable run.

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
compatibility. To download pretrained weights through `timm`, pass
`--pretrained`; the default model is explicitly
`vit_base_patch16_224.augreg_in21k` (ImageNet-21K, not the ImageNet-1K
fine-tuned variant). The first run requires network access. Preprocessing is
resolved from that model's timm config, including its ImageNet-21K mean/std.
Without a checkpoint or `--pretrained`, the backbone is randomly initialized
and then frozen, which is not paper-comparable. A configured but missing local
checkpoint is an error; it does not silently fall back to random weights.

Run the configured CIFAR-100 B0-Inc10 training and task-agnostic evaluation with:

```powershell
python main.py --data-root <cifar-100-root> --pretrained --output-dir outputs/cifar100_b0_inc10
```

During training, each task/epoch displays running total, CE, KD, and sparsity
losses, plus accuracy over all seen classes and over only the current task's
classes. The run also evaluates task-agnostically on all classes learned so
far after every task. It writes checkpoints and a
`metrics.json` containing stage accuracies, average incremental accuracy, and
final accuracy to the output directory. CUDA training uses BF16 autocast when
supported, otherwise FP16 with gradient scaling, and clips the global gradient
norm to 1. Non-finite inputs, activations, logits, or losses stop training with
the task/epoch/batch location instead of silently propagating NaNs.

The simulator adaptively pools each full feature vector to the qubit count,
normalizes and maps the pooled values to `[-pi, pi]` rotation angles, then
applies trainable data-conditioned `Ry` angle scales and a nearest-neighbour
CNOT chain. Task representations are formed by truncated SVD of the down/up
adapter projection matrices, then normalized. The paper defines its sparsity
term as the L1 norm of softmax relevance weights; this expression is retained
literally, although it equals one and therefore does not itself encourage
sparsity.

Set `QKD_VIT_B16_IN21K_WEIGHTS` or pass `--checkpoint-path <file>` to load a
local pretrained backbone. To use timm's ImageNet-21K weights, run with
`--pretrained`; `--model-name` defaults to
`vit_base_patch16_224.augreg_in21k` and rejects non-IN21K tags when downloading.
Training settings can be changed with CLI options; the Kaggle notebook
supports either local weights or timm's pretrained download.

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
src/qkd/quantum/          Differentiable QGTM circuit and relevance weights
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
