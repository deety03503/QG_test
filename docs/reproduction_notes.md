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
| Backbone | Frozen ViT-B/16 pretrained on ImageNet-21K (reported paper setting) |
| Adapter | Parallel to each block MLP; bottleneck dimension 64; task 0 has zero-initialized output projection |
| Task representation | Adapter parameter matrix; truncated SVD dimension 12 |
| Quantum module | Trainable data-conditioned `Ry` angles and CNOT chain; one circuit layer described |
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

## Paper-faithful implementation choices

- The QGTM uses a differentiable PyTorch state-vector simulator. It adaptively
  average-pools each full input vector to `q` coordinates, normalizes the
  pooled vector, and maps it to angles in `[-pi, pi]`. Trainable per-qubit
  scales modulate those data-conditioned `Ry` angles before the paper's
  adjacent-qubit CNOT chain. Fidelity is the squared state-vector inner
  product. This avoids discarding all but the first `q` coordinates and avoids
  a shared final unitary cancelling from the fidelity.
- A task vector is computed from the learned adapter down/up projection
  matrices: concatenate them by feature columns, retain the rank-`r` truncated
  SVD, multiply the reconstruction by an all-ones vector, and normalize.
- Task 0's final adapter projection is zero-initialized, so it starts as a
  no-op residual. Each later task copies task 0's adapter parameters at the
  task boundary, then trains that copy while freezing all previous adapters.
- Training keeps the pretrained backbone and previous adapters frozen; it
  trains only the current adapter, a global classifier, and the QGTM rotations.
  The classification head grows with the set of seen class IDs. The TIKD loss
  uses relevance-weighted `KL(old adapter || current adapter)` over the
  previous tasks' class logits only; newly introduced classes are excluded
  because historical adapters have no trained outputs for them.
- Task-agnostic inference computes QGTM relevances against all learned task
  vectors, uses the resulting softmax weights to fuse adapter features, and
  applies the global classifier over seen classes.
- The paper's Eq. 11 applies the L1 norm to softmax-normalized relevance
  weights, which is constant because the weights sum to one. To make the
  sparsity term effective, this implementation instead minimizes the entropy
  of those relevance weights; this is an intentional deviation from Eq. 11
  that encourages the gate to concentrate on fewer tasks.
- The full official training split is used for training; the official test
  split is used for evaluation, with no held-out development set. During each
  training epoch, the progress bar reports cumulative total loss and training
  accuracy from the first batch through the current batch. After each task,
  task-agnostic test accuracy over seen classes and separate test accuracy for
  each learned task are reported. The CE targets remain indices in the global
  seen-class logits.
- The paper does not fully specify adapter-vector construction, mapping vector
  coordinates when `q` differs from feature dimension, or all initialization
  details. The current default loads `vit_base_patch16_224` pretrained weights
  through `timm`, rather than the paper's reported ImageNet-21K checkpoint.
  The zero-output initialization, adapter-copy initialization,
  adaptive-pooling, and angle-scaling choices above are deterministic and are
  not claimed as author-released implementation details.

For comparable results, provide the exact frozen ViT-B/16-IN21K checkpoint,
dataset preprocessing, class order, and random seed. Checkpoint provenance and
preprocessing should be recorded with experiment outputs. Baseline comparison,
ablations, and extended benchmark datasets remain separate follow-up work.
