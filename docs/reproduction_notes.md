# Reproduction Notes

## Protocol

Target the QKD paper's exemplar-free pretrained-model CIL setting. Tasks have
disjoint class sets; only the current task's data is available during its
training; evaluation is task-agnostic across all classes seen so far. Record
top-1 accuracy after every stage, average incremental accuracy
`A = mean(A_b)`, and final accuracy `A_B`.

Use the survey's class-order convention (NumPy seed 1993). The initial config
is CIFAR100 `B0-Inc10`. Keep exemplar-free QKD results separate from
exemplar-replay baselines: the survey demonstrates that memory budgets can
change method rankings. For extended comparisons, record model size,
exemplar count, and total memory budget.

## Reported settings

| Component | Setting |
| --- | --- |
| Backbone | Frozen ViT-B/16 pretrained on ImageNet-21K |
| Adapter | Parallel to each block MLP; bottleneck dimension 64 |
| Task representation | Adapter parameter matrix; truncated SVD dimension 12 |
| Quantum module | Trainable `Ry` rotations and CNOT chain; one circuit layer described |
| Training | SGD, learning rate 0.05, cosine decay, batch 32, 20 epochs/task |
| Objective | CE + weighted task KL (`lambda_kd=1.0`) + sparsity term (`lambda_s=0.05`) |
| Relevance | Softmax temperature 1.0 |
| Exemplars | None |
| Benchmarks | CIFAR100, CUB200, ImageNet-A, ImageNet-R, VTAB |

QKD reports CIFAR `B0-Inc10`, CUB `B0-Inc20`, IN-A `B0-Inc10`, IN-R
`B0-Inc20`, and VTAB `B0-Inc10`. The paper also includes large-base and other
session splits; maintain separate config files for each.

The 9-qubit default in the starter config is provisional: it is the middle
setting in the paper's 3/6/9/12/15-qubit sweep, not a clearly identified
default in the supplied text.

## Details to resolve

1. The paper discusses projected quantum kernels/local observables but defines
   the method's task relevance as state fidelity. The exact task-state
   preparation from the SVD vector and mapping of feature dimensions to qubit
   angles are not fully specified. Recover these from the authors' code or
   document the implementation choice and backend/version.
2. The text/figure describes feature-level distillation, while Eq. 13 defines
   KL divergence from old-adapter logits to new-adapter logits. The current
   loss helper follows Eq. 13; confirm intended behavior before final runs.
3. The sparsity term is written as `||alpha||_1` after softmax. Nonnegative
   softmax weights sum to one, so this term is constant and cannot induce
   sparsity. Verify whether the intended penalty is applied before softmax or
   uses a different expression.
4. Inference is described as adapter selection or fusion, without a fully
   specified classifier-head procedure in the supplied text. Implement and
   report the routing/head procedure explicitly, cross-checking the released
   code.
5. Exact checkpoint, image preprocessing, label order, split files, and random
   seeds affect results. Record their versions/hashes and do not substitute an
   unverified checkpoint. Adapter initialization and the exact input
   preprocessing should also be verified against the released implementation.

## Implementation order

1. Verify dataset splits and preprocessing against the cited benchmark code;
   add loaders and per-task datasets.
2. Load the exact frozen ViT-B/16-IN21K checkpoint and validate adapters on
   every block; the `PretrainedViT` wrapper currently covers this portion.
3. Implement and test QGTM state preparation, parameterized circuit,
   measurement, and gradients using the resolved quantum backend.
4. Add task heads, current/previous adapter outputs, confirmed distillation and
   sparsity losses, and per-task optimizer lifecycle.
5. Implement task-agnostic inference routing, per-stage evaluation, and result
   aggregation.
6. Reproduce baselines and ablations; log seeds, parameters, memory, latency,
   and checkpoint provenance.
