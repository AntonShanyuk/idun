# idun_realmlp: RealMLP

idun_realmlp is RealMLP-TD (Holzmüller et al., *Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular
Data*, NeurIPS 2024), as pytabkit 1.7.3's `RealMLP_TD_Classifier` builds it: robust scaling and smooth clipping of the
inputs, periodic embeddings of the numeric inputs, a learned scale on each input, SELU blocks inside parametric
activations, label smoothing, and Adam with pytabkit's schedules (the learning rate on `coslog4`, dropout and weight
decay on `flat_cos`) and its per-parameter learning-rate factors. Categorical columns are one-hot encoded, or embedded
from a learned table (pytabkit's `max_one_hot_cat_size` and `embedding_size`).

It trains on Metal and CUDA in float32 and has a float64 NumPy reference backend; as for idun_tabm, a fit is the same
bits run after run on the same GPU, and the backends agree within stated tolerances of the reference
([Determinism and testing](../testing.md)).

What it does differently from pytabkit: without a validation set it keeps the last epoch's weights, where pytabkit
holds out a fifth of the rows to choose the best epoch (give `eval_set` to do the same); the regressor uses the
classifier's block (parametric SELU at a learning rate of 0.04), where pytabkit's regressor uses Mish at 0.2
(`act="mish"`, `learning_rate=0.2`); and the preprocessing is fitted on `init_rows` rows drawn from the training rows, pytabkit's
subsample size, with idun's own seeded generator.

## A first fit

```python
from idun.realmlp import RealMLPClassifier, RealMLPRegressor

clf = RealMLPClassifier(n_seeds=4, categorical=["sector"], backend="metal").fit(X_train, labels)
proba = clf.predict_proba(X_test)

reg = RealMLPRegressor(act="mish", patience=16, backend="cuda").fit(X_train, y_train, eval_set=(X_val, y_val))
pred = reg.predict(X_test)
```

`fit`, `predict`, `predict_proba`, `save` and `idun.load` work as for idun_tabm's estimators. The lower-level
`train_realmlp(params, data, *, rows, backend, ...)` and `train_realmlp_seeds` train from a `RealMLPParams`, with
snapshots during training, categorical embeddings over level codes and validation; see the
[Python API](../api/python-api.md#idunrealmlp-idun_realmlp-design-notes-8).

## Backends

| backend | where | notes |
|---|---|---|
| `"metal"` | Apple silicon, macOS 15 or later | the estimators' default |
| `"cuda"` | NVIDIA GPUs, `nvidia-cuda-nvrtc` | the same kernels, compiled at run time |
| `"reference"` | anywhere | float64 NumPy; slow, for checks and small tables |

On the GPU backends the hidden widths are multiples of 64, and a network has at most 8 classes and 8 embedding values
per numeric input.

## The network and its training

`RealMLPClassifier`, `RealMLPRegressor` and `RealMLPParams` take these, with the same defaults: pytabkit's
RealMLP-TD classifier.

| parameter | default | what it does |
|---|---|---|
| `blocks` | `3` | the hidden blocks |
| `hidden` | `256` | each block's units |
| `hidden_sizes` | `()` | each hidden block's units instead of `blocks` × `hidden` (pytabkit's); empty keeps those |
| `act` | `"selu"` | the activation inside the parametric activation: `"selu"`, `"mish"` or `"relu"` |
| `frequencies` | `16` | the periodic embedding's frequencies per numeric input |
| `embed` | `3` | the values each numeric input's embedding gives the first block |
| `embed_sigma` | `0.1` | the spread of the embedding's initial frequencies |
| `clip` | `3.0` | the smooth clipping bound of the robust-scaled inputs |
| `dropout` | `0.15` | the dropout probability, scaled over training by `dropout_schedule` |
| `label_smoothing` | `0.1` | the classifier's label smoothing |
| `learning_rate` | `0.04` | Adam's base learning rate, scaled over training by `lr_schedule` |
| `weight_decay` | `0.02` | the decoupled weight decay, scaled by `wd_schedule` |
| `eps` | `1e-8` | Adam's epsilon |
| `scale_lr_factor` | `6.0` | the learning-rate factor of the inputs' learned scale |
| `bias_lr_factor` | `0.1` | the learning-rate factor of the biases (which are not decayed) |
| `act_lr_factor` | `0.1` | the learning-rate factor of the parametric activations |
| `embed_lr_factor` | `0.1` | the learning-rate factor of the periodic embeddings |
| `lr_schedule` | `"coslog4"` | the learning rate over training: `"coslog4"` (pytabkit's), `"cosine"` or `"constant"` |
| `dropout_schedule` | `"flat_cos"` | dropout over training: `"flat_cos"` (pytabkit's) or `"constant"` |
| `wd_schedule` | `"flat_cos"` | weight decay over training: `"flat_cos"` (pytabkit's) or `"constant"` |
| `batch_size` | `256` | rows per step; an epoch's last partial batch is dropped |
| `epochs` | `256` | passes over the training rows |
| `init_rows` | `0` | the training rows the preprocessing and the data-dependent initialisation are fitted on; `0` takes pytabkit's own count for the table, `floor(2^28 / (62F + 1))` rows for F inputs (every row of a table with fewer) |
| `embedding_size` | `8` | the learned values of each embedded categorical column |
| `seed` | `0` | the initialisation rows, the weights, the epoch order and the dropout masks; a seed bag uses `seed`, `seed + 1`, ... |

## Estimator parameters

| parameter | default | what it does |
|---|---|---|
| `n_seeds` | `1` | a seed bag: one network per seed, predicting their mean |
| `backend` | `"metal"` | `"metal"`, `"cuda"` or `"reference"` |
| `categorical` | `()` | the categorical columns, by name or position |
| `max_one_hot_cat_size` | `-1` | a categorical column of at least this many levels (counting a missing level, as pytabkit) is embedded instead of one-hot encoded; `-1` one-hot encodes every column (pytabkit's RealMLP-TD uses 9) |
| `patience` | `None` | with a validation set, stop after this many epochs without a better score |

## `RealMLPParams` only

| parameter | default | what it does |
|---|---|---|
| `n_classes` | `2` | the classes of a cross-entropy network (1 for `"mse"`); the estimators take it from the labels |
| `objective` | `"cross_entropy"` | `"cross_entropy"` or `"mse"` (regression); the estimators set it from the task |
| `beta1` | `0.9` | Adam's first-moment decay |
| `beta2` | `0.95` | Adam's second-moment decay |
