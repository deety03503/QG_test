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

## Paper-faithful implementation choices

- The QGTM uses a differentiable PyTorch state-vector simulator. Data angles
  are the first `q` coordinates of normalized vectors; each layer applies
  per-qubit data `Ry` rotations, trainable `Ry` rotations, and the paper's
  adjacent-qubit CNOT chain. Fidelity is the squared state-vector inner
  product.
- A task vector is computed from the learned adapter down/up projection
  matrices: concatenate them by feature columns, retain the rank-`r` truncated
  SVD, multiply the reconstruction by an all-ones vector, and normalize.
- Training keeps the pretrained backbone and previous adapters frozen; it
  trains only the current adapter, a global classifier, and the QGTM rotations.
  The classification head grows with the set of seen class IDs. The TIKD loss
  follows Eq. 13 exactly: relevance-weighted `KL(old adapter || current
  adapter)` over the seen-class logits.
- Task-agnostic inference computes QGTM relevances against all learned task
  vectors, uses the resulting softmax weights to fuse adapter features, and
  applies the global classifier over seen classes.
- The paper's Eq. 11 applies the L1 norm to softmax-normalized relevance
  weights. The implementation preserves this formula exactly; since the
  weights sum to one, the term is constant and supplies no sparsity gradient.
- The paper does not fully specify adapter-vector construction, mapping vector
  coordinates when `q` differs from feature dimension, or all initialization
  details. The choices above make these steps deterministic and are not
  claimed as author-released implementation details.

For comparable results, provide the exact frozen ViT-B/16-IN21K checkpoint,
dataset preprocessing, class order, and random seed. Checkpoint provenance and
preprocessing should be recorded with experiment outputs. Baseline comparison,
ablations, and extended benchmark datasets remain separate follow-up work.
