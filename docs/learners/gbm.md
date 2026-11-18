# idun_gbm: gradient boosting

idun_gbm is histogram gradient boosting with LightGBM's model: quantile-binned features, depth-wise or leaf-wise
(best-first) trees, quantized gradients, LightGBM's categorical splits and regularisers, under LightGBM's parameter
names and defaults wherever it has the same parameter. It runs on three backends, the CPU (numba), Apple's Metal and
NVIDIA's CUDA, and **all three build the same model, bit for bit**: the histograms are integer sums, and the split
search and the leaf values are the host's ([Determinism and testing](../testing.md)).

What it does differently from LightGBM: the gradients are always quantized (`grad_quant_bins`, 4 by default, LightGBM's
`use_quantized_grad`), so every histogram is an exact integer sum on any device; depth-wise trees are the default and
are limited to depth 8 (256 leaves); a categorical column has at most 64 levels; and there is no automatic choice of
backend, so a fit runs where you say.

## A first fit

```python
from idun.gbm import GBMClassifier, GBMRegressor

model = GBMRegressor(num_trees=500, learning_rate=0.05, backend="metal").fit(X_train, y_train)
pred = model.predict(X_test)

clf = GBMClassifier(num_trees=300, categorical=["sector"], backend="cuda").fit(X_train, labels)
proba = clf.predict_proba(X_test)  # [1 - p, p]
```

`X` is a NumPy array, a pandas DataFrame or an Arrow table; missing values are allowed (they get a bin of their own).
`model.save("model.idun")` and `idun.load("model.idun")` round-trip any estimator. The estimators follow
scikit-learn's conventions without importing it, and pass its `check_estimator` but for two checks
([Python API](../api/python-api.md#estimators)).

The lower-level interface trains on an `idun.Dataset` (or any `BinnedData`) with a `Params`:

```python
import idun
from idun import gbm

data = idun.Dataset(X, y, categorical=["sector"])
booster = gbm.train(gbm.Params(num_trees=500, learning_rate=0.05), data, rows=slice(0, len(y)), backend="metal")
pred = booster.predict(data.transform(X_test), rows=slice(0, len(X_test)), backend="metal")
```

`gbm.train` adds what the estimators do not expose: custom objectives (`Objective`, `NewtonObjective`), multi-class
boosting (`train_multiclass`), continued fits (`init_model`), validation sets with early stopping and callbacks
(`Validation`), seed bags (`train_seeds`) and many fits on one dataset (`train_many`); TreeSHAP contributions and
leaf indices come from `Booster.predict_contributions` and `Booster.predict_leaves`. See the
[Python API](../api/python-api.md#idungbm-idun_gbm).

## Backends

| backend | where | notes |
|---|---|---|
| `"metal"` | Apple silicon, macOS 15 or later (the arm64 wheel) | the estimators' default |
| `"cuda"` | NVIDIA GPUs with a CUDA 12 or 13 driver and `nvidia-cuda-nvrtc` | one fit per GPU in a seed bag |
| `"cpu"` | anywhere | numba; the reference the GPU backends are tested against |

Every parameter below works on every backend, and the three give the same trees.

## Tree parameters

`GBMRegressor`, `GBMClassifier` and `Params` take these; `JARFRegressor` takes them too, but `second_order`,
`extra_trees` and `feature_fraction_bynode`. Every value is checked when the
object is made, and a bad one raises `ValueError` naming it.

| parameter | default | what it does |
|---|---|---|
| `num_trees` | `100` | boosting iterations, one tree each |
| `learning_rate` | `0.1` | shrinkage applied to each leaf value |
| `growth` | `"depthwise"` | `"depthwise"`: every level split at once, up to `max_depth`. `"leafwise"`: LightGBM's best-first growth, the leaf of highest gain split until `max_leaves` leaves |
| `max_depth` | `6` | depth-wise: the trees' depth, 1 to 8. Leaf-wise: `-1` for no limit, or the depth no leaf may exceed |
| `max_leaves` | `64` | depth-wise: a cap on the leaves (2 to 2^`max_depth`), the highest-gain splits kept when a level would exceed it. Leaf-wise: the leaves per tree, 2 to 256 |
| `min_data_in_leaf` | `20` | the fewest in-bag rows on each side of a split |
| `min_sum_hessian_in_leaf` | `0.001` | the smallest hessian sum on each side of a split; with plain gradients the hessians are row counts, so it binds only above `min_data_in_leaf` |
| `min_gain_to_split` | `0.0` | the gain a split must exceed |
| `lambda_l2` | `0.0` | L2 regularisation of the leaf values |
| `lambda_l1` | `0.0` | L1 regularisation: a leaf's gradient sum is moved towards 0 by it, in the gains and the leaf values |
| `max_delta_step` | `0.0` | above 0, each leaf's output before the learning rate is clipped to plus or minus it; 0 sets no limit |
| `second_order` | `False` | the split search reads the hessians of a Newton objective, of row weights or of a multi-class fit, as LightGBM does; off, the splits are first-order (rows as the hessians) |
| `feature_fraction` | `1.0` | the fraction of the features each tree may split on, drawn per tree |
| `feature_fraction_bynode` | `1.0` | the fraction of the tree's features each node searches, drawn per node |
| `extra_trees` | `False` | each node tries one random threshold per numeric feature instead of every bin (categorical features keep their search) |
| `bagging_fraction` | `1.0` | each row's probability of being in the bag |
| `bagging_freq` | `0` | redraw the bag every k trees; 0 turns bagging off |
| `grad_quant_bins` | `4` | the levels the gradients are quantized to, a power of 2 from 2 to 128 (LightGBM's `num_grad_quant_bins`); more levels, finer gradients, the same speed |
| `categorical_split` / `Params.categorical` | `"sorted"` | how categorical columns split: `"sorted"` (LightGBM's many-vs-many, the levels sorted by their gradient ratio) or `"one_vs_rest"`. `Params` calls it `categorical` |
| `cat_smooth` | `10.0` | LightGBM's smoothing of the levels' gradient ratios |
| `cat_l2` | `10.0` | LightGBM's L2 regularisation of categorical splits |
| `max_cat_threshold` | `32` | the most levels on one side of a sorted categorical split |
| `min_data_per_group` | `100` | the fewest rows of a level for it to be split on its own |
| `seed` | `0` | drives the bagging, the feature sampling, the node draws and the gradients' stochastic rounding; a seed bag uses `seed`, `seed + 1`, ... |

## Estimator parameters

`GBMRegressor` and `GBMClassifier` take these besides the tree parameters.

| parameter | default | what it does |
|---|---|---|
| `n_seeds` | `1` | a seed bag: one model per seed, predicting their mean (`predict(X, per_seed=True)` gives each) |
| `workers` | `None` | how many of a bag's fits run side by side on a GPU backend; `None` picks for the device (one per GPU on CUDA) |
| `backend` | `"metal"` | `"metal"`, `"cuda"` or `"cpu"` |
| `categorical` | `()` | the categorical columns, by name or position; text and integer levels, at most 64 per column |
| `codes` | `()` | columns of small integers 0 to 4 with no missing value, stored as 4-bit codes: half the memory and faster histograms |
| `max_bin` | `255` | the most bins of a numeric column, 2 to 255 |
| `subsample_for_bin` | `200000` | the rows the bin edges are computed from (a seeded sample) |
| `eval_metric` | `None` | the validation metric of `fit(..., eval_set=...)`: `"l2"`, `"rmse"` or `"l1"` for the regressor, `"binary_logloss"` or `"auc"` for the classifier; `None` takes the first of each |
| `early_stopping_rounds` | `None` | stop when the first validation set has not improved for this many trees, and keep the trees up to its best |
| `verbose` | `0` | above 0, print every k-th tree's validation scores |
| `importance_type` | `"split"` | what `feature_importances_` counts: the splits on each feature |

`fit(X, y, sample_weight=None, *, eval_set=(), eval_names=None, callback=None, eval_sample_weight=None)` takes row
weights and validation sets; `predict(X, num_iteration=None, start_iteration=0, pred_contrib=False, pred_leaf=False)`
predicts with a range of trees, or returns TreeSHAP contributions or leaf indices.
