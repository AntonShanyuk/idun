# idun_tabm: TabM

idun_tabm is TabM (Gorishniy et al., *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*,
ICLR 2025): `members` MLPs that share each block's weight matrix and differ by rank-1 adapters and their own biases,
trained together, predicting their mean. Its defaults are the `tabm` package's (`TabM.make`, 0.0.3): 32 members, 3
blocks of 512, dropout 0.1, AdamW at a learning rate of 2e-3 with weight decay 3e-4. Optional piecewise-linear
embeddings of the numeric inputs and gradient clipping follow pytabkit's TabM.

It trains on Metal and CUDA in float32 with its own kernels (a tiled GEMM, the fused adapters, the loss and AdamW), and
has a float64 NumPy reference backend that the GPU backends are tested against. A fit is deterministic: the same data,
parameters and seed give the same bits run after run on the same GPU; across backends the networks agree within
stated tolerances of the float64 reference, not bit for bit ([Determinism and testing](../testing.md)).

What it does differently from the `tabm` package: no early stopping unless you give a validation set (then the best
epoch's weights are kept, and `patience` stops the fit); inputs are standardized by the estimator's table encoder
(numeric columns) and one-hot encoded (categorical columns); the epoch order and the dropout masks come from idun's own
seeded generators, so a checkpointed fit resumes bit for bit.

## A first fit

```python
from idun.tabm import TabMClassifier, TabMRegressor

clf = TabMClassifier(n_seeds=2, categorical=["sector"], backend="metal").fit(X_train, labels)
proba = clf.predict_proba(X_test)  # (n, n_classes): the seed bag's mean of the members' softmax

reg = TabMRegressor(epochs=30, backend="cuda").fit(X_train, y_train, eval_set=(X_val, y_val))
pred = reg.predict(X_test)  # in the target's units; the best epoch's weights on the validation rows
```

`fit(X, y, sample_weight=None, *, eval_set=None)` takes row weights (each step's loss becomes its batch's weighted
mean) and a validation set (scored after every epoch: the error rate for the classifier, the squared error on the
standardized target for the regressor; `best_epoch_` and `valid_score_` read the result). `save` and `idun.load`
round-trip the model.

The lower-level `train_tabm(params, data, *, rows, backend, ...)` trains one network from a `TabMParams` on a
`MatrixData` or a packed dataset, with callbacks per epoch, checkpoints that resume bit for bit, a loss or a batch plan
of your own, a temporal encoder over per-row sequences and validation; see the
[Python API](../api/python-api.md#iduntabm-idun_tabm-design-notes).

## Backends

| backend | where | notes |
|---|---|---|
| `"metal"` | Apple silicon, macOS 15 or later | the estimators' default |
| `"cuda"` | NVIDIA GPUs, `nvidia-cuda-nvrtc` | the same kernels, compiled at run time |
| `"reference"` | anywhere | float64 NumPy; slow, for checks and small tables |

The GPU backends take at most 8 classes. `storage="bf16"` and `micro_batch` apply to them; the reference backend always
computes the whole batch in float64.

## The network and its training

`TabMClassifier`, `TabMRegressor` and `TabMParams` take these, with the same defaults: the paper's network, AdamW and
batch size ([Performance guide](../performance.md) on what batch size costs).

| parameter | default | what it does |
|---|---|---|
| `members` | `32` | k, the networks of the ensemble |
| `blocks` | `3` | the hidden blocks, each Linear, ReLU and dropout |
| `width` | `512` | each block's units |
| `dropout` | `0.1` | the dropout probability after each block |
| `learning_rate` | `0.002` | AdamW's learning rate |
| `weight_decay` | `0.0003` | AdamW's decoupled weight decay, on every parameter |
| `eps` | `1e-8` | AdamW's epsilon |
| `batch_size` | `256` | rows per step; an epoch is one seeded permutation of the training rows |
| `epochs` | `50` | passes over the training rows (with a validation set, the most) |
| `lr_schedule` | `"constant"` | the learning rate over the steps: `"constant"`, `"cosine"` or `"linear"` decay |
| `lr_warmup_steps` | `0` | a linear warm-up over the first steps |
| `gradient_clipping_norm` | `0.0` | above 0, the gradients are scaled to at most this norm before each step, as `torch.nn.utils.clip_grad_norm_` (pytabkit's) |
| `num_emb_type` | `"none"` | `"pwl"`: each numeric input enters block 0 as a piecewise-linear embedding (rtdl's version B, as pytabkit's TabM) |
| `num_emb_n_bins` | `48` | the embedding's most quantile bins per input |
| `d_embedding` | `16` | the values each embedded input becomes |
| `storage` | `"fp32"` | `"bf16"`: the GPU stores activations and their gradients in bfloat16 (weights, AdamW and arithmetic stay float32) |
| `micro_batch` | `0` | above 0, each batch runs as micro-batches of this many rows with their gradients summed before the one step: the same update up to summation order, for less GPU memory |
| `seed` | `0` | the initial weights, the epoch order and the dropout masks; a seed bag uses `seed`, `seed + 1`, ... |
| `objective` | `"mse"` (`TabMParams`: `"cross_entropy"`) | `"cross_entropy"` (classification), `"mse"` (regression: the members' mean output) or `"hl_gauss"` (regression as classification over bins, Imani and White 2018). The classifier has no such parameter |
| `n_bins` | `64` | `"hl_gauss"`: the bins, a multiple of 8 up to 512 |
| `hl_low` | `-4.0` | `"hl_gauss"`: the lowest bin edge, in standardized target units |
| `hl_high` | `4.0` | `"hl_gauss"`: the highest bin edge |
| `hl_sigma` | `0.09375` | `"hl_gauss"`: the width of each label's Gaussian (0.75 bin widths by default) |

## Estimator parameters

| parameter | default | what it does |
|---|---|---|
| `n_seeds` | `1` | a seed bag: one network per seed, predicting their mean (`per_seed=True` gives each) |
| `backend` | `"metal"` | `"metal"`, `"cuda"` or `"reference"` |
| `categorical` | `()` | the categorical columns, by name or position, one-hot encoded |
| `patience` | `None` | with a validation set, stop after this many epochs without a better score |

## `TabMParams` only

| parameter | default | what it does |
|---|---|---|
| `n_classes` | `2` | the classes of a cross-entropy network (1 for `"mse"` and `"hl_gauss"`); the estimators take it from the labels |
| `beta1` | `0.9` | AdamW's first-moment decay |
| `beta2` | `0.999` | AdamW's second-moment decay |
| `encoder` | `"none"` | `"gru"`: a temporal encoder reads a sequence per row (`idun.tabm.Sequences`) and prepends its last state to the row's inputs |
| `encoder_embed` | `32` | the encoder's per-step embedding units |
| `encoder_hidden` | `64` | the encoder's GRU units |
