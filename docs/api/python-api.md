# Python API

The public interface of `idun`. A name is public when it is in a module's `__all__`; every public name of `idun`,
`idun.gbm`, `idun.tabm`, `idun.realmlp` and `idun.data` is on this page, as are the lower-level modules' main names.
Change the API and this page in the same commit.

**Stability.** Semantic versioning: within a major version a public name, its parameters' names, order
and defaults, and the meaning of each, stay; a new parameter comes with a default that keeps the old behaviour, bit
for bit; every change to the public surface (each public module's `__all__`, with every function's and class's
parameters and defaults, a dataclass's fields and a class's public members) is named in the changelog. Model files
load in every later release and predict the same bits; an added key or entry
keeps a format's version, a changed meaning raises it, and an unknown version raises `ValueError` when loaded.

`idun.__version__` is the package version.
`idun.load(path)` reads a fitted estimator written by its `save` (the `idun-model` format, v1; `GBMRegressor` below;
v2 is a `JARFRegressor`'s, `idun.jarf`).

## Estimators

`idun.gbm.GBMRegressor` / `GBMClassifier`, `idun.tabm.TabMClassifier` / `TabMRegressor`,
`idun.realmlp.RealMLPClassifier` / `RealMLPRegressor` and `idun.jarf.JARFRegressor` follow scikit-learn's conventions without importing it (`idun.base`):
parameters only in the
constructor, `fit` returns the estimator, `get_params` / `set_params`, `n_features_in_` and `feature_names_in_`
after `fit`, `score` (R² for the regressors, accuracy for the classifiers, with an optional `sample_weight`), and
scikit-learn's tags (`__sklearn_tags__`, which imports scikit-learn only when scikit-learn asks for them). Each one
passes scikit-learn 1.9's `check_estimator` except for two checks, the same for all seven:

- `check_estimators_unfitted`: predicting before `fit` raises `idun.errors.NotFittedError`, a `ValueError` and an
  `AttributeError` as scikit-learn's own class is, but not that class: idun does not depend on scikit-learn.
- `check_supervised_y_2d`: a column-vector `y` of shape `(n, 1)` is refused with a `ValueError` (pass `y.ravel()`).
  scikit-learn's estimators flatten it with a warning; idun fails loudly instead.

Inputs are checked as scikit-learn's `validate_data` checks them, with the fix in the message: sparse matrices
(`TypeError`), 1-D, complex, empty or column-less `X`, a missing, complex or non-finite `y`, and at predict time a
column count other than the fitted one (`ValueError`). Missing values in `X` are allowed: idun_gbm bins them apart, the
networks read them as 0.

## `idun.Dataset`

Tabular data from a 2-D NumPy array, a pandas DataFrame, an Arrow table or a list of `(name, 1-D array)` pairs (columns
the caller built, as `JARFRegressor` appends projections), binned once for idun_gbm (it satisfies `BinnedData`). Every
column is numeric unless named as codes or categorical; nothing is chosen from the data.

```python
import idun

data = idun.Dataset(X, y, categorical=["sector"], codes=["era_bucket"])  # names, or positions for an array
test = data.transform(X_test)  # the same edges and levels; an unseen level raises
part = data.subset(slice(0, 1000))  # rows, as a slice or indices
```

| kind | storage | rule |
|---|---|---|
| numeric | uint8 quantile bins, ≤ `max_bin` (255) plus bin 255 for NaN | edges from `bin_rows` (default: a seeded sample of `subsample_for_bin` rows, 200,000), exactly as the packed cache bins (a test) |
| codes | 4-bit codes, two rows per byte | integers 0..4, no missing value |
| categorical | uint8 code per row over the sorted levels (as text) | at most 64 levels, no missing value |

Attributes: `columns` (the input's order), `feature_names` (numeric, then codes, then categorical),
`named_features`, `obfuscated_features` (the codes), `categorical_features`, `categories`, `binning` (an id of the
edges and levels: a model refuses data binned another way), `n_rows`, `n_features`, and `BinnedData`'s arrays.
`y` (optional, numeric) becomes `target` (float32). `max_bin` (2 to 255) and `subsample_for_bin` are LightGBM's
binning settings: fewer bins coarsen the numeric columns; the histograms keep their cells, so a fit's
cost does not change, and the split search's empty cells repeat a candidate, which its ties give to the lowest
threshold, LightGBM's split. `bin_rows` and `subsample_for_bin` exclude each other.

## `idun.gbm`: idun_gbm

`idun.gbm.__all__`: `Params`, `train`, `train_seeds`, `predict_mean`, `train_many`,
`train_multi` (experimental, not covered by this documentation), `train_multiclass`, `Booster`, `FitReport`, `Objective`, `NewtonObjective`, `MulticlassObjective`, `LogLoss`,
`Softmax`, `BinnedData`, `BinnedRows`, `take_rows`, `GBMRegressor`, `GBMClassifier`, `Validation`, `Metric`,
`METRICS`, `Evaluation`, `log_evaluation`. Backends:
`"cpu"`, `"metal"` (Apple7 GPUs and newer, M1 and later) and `"cuda"` (Linux with an NVIDIA GPU); both GPU backends
are bit-identical to `"cpu"`.

**CUDA.** `"cuda"` needs NVIDIA's driver (`libcuda.so.1`) and NVRTC, its runtime compiler: the `nvidia-cuda-nvrtc`
pip package in the environment, or the CUDA toolkit's library on the loader's path. Nothing is built on the machine;
the kernels are compiled when a backend opens. `IDUN_CUDA_DEVICE` names the GPU (default 0); `IDUN_CUDA_LIB` and
`IDUN_NVRTC_LIB` name a library file. Without a GPU, the driver or the compiler, `BackendUnavailableError` says which
is missing. The panel is uploaded to the GPU's memory when a fit or a prediction starts. One limit: a pass's grid
holds at most 65,535 row groups, so at the widest gradient quantization (64 levels a side, 480 rows a group) a fit
of more than about 31.4 million rows is refused with a `ValueError`, for idun_gbm, idun_tabm,
idun_realmlp and JARF.

**CPU threads.** idun's CPU code runs numba's parallel loops, and numba starts a thread for each CPU the process may
be scheduled on. In a Linux container held to fewer CPUs by a cgroup CPU quota (Docker's `--cpus`, a Kubernetes
limit), `import idun` sets `NUMBA_NUM_THREADS` to the quota's CPUs, rounded up, unless the variable is set already;
child processes inherit it. numba imported before idun has counted its threads already: then a `RuntimeWarning`
names the value to set before numba is imported. The thread count never changes a result.

### Example

```python
from idun import data as hdata, gbm

data = hdata.PackedDataset.open("cache", binning="r0")  # memory-mapped, read-only; the first window's binning
window = data.refits[0]
params = gbm.Params(num_trees=300, seed=0)
model = gbm.train(params, data, rows=slice(0, window.train_hi), backend="metal")
pred = model.predict(data, rows=slice(window.pred_lo, window.pred_hi), backend="metal")
model.save("model.igbm")
same = gbm.Booster.load("model.igbm")
```

### `GBMRegressor`

idun_gbm behind scikit-learn's conventions (scikit-learn itself is not imported):

```python
import idun
from idun.gbm import GBMRegressor

model = GBMRegressor(num_trees=500, learning_rate=0.05, n_seeds=4, categorical=["sector"]).fit(X, y)
pred = model.predict(X_test)  # binned with the training binning; a seed bag predicts the mean
leafwise = GBMRegressor(growth="leafwise", max_depth=-1, max_leaves=63)  # LightGBM's best-first trees
model.save("model.idun")
model = idun.load("model.idun")
```

- Parameters: every `Params` field with its default, except the split method, `categorical_split` (`Params.categorical`),
  since `categorical` names the categorical columns as in `idun.Dataset`; plus `codes`, `n_seeds` (a seed bag through
  `train_seeds`, seeds `seed`, `seed + 1`, ...), `backend` (`"metal"` by default; `"cpu"` or `"cuda"` explicitly) and `workers`;
  and the binning of `fit(X, y)`'s table, `max_bin` (255) and `subsample_for_bin` (200,000), `idun.Dataset`'s (a `Dataset`
  passed to `fit` keeps its own).
- `fit(X, y)` takes a 2-D NumPy array, a pandas DataFrame or an Arrow table, or `fit(dataset)` an `idun.Dataset` with its
  target. `predict(X)` takes raw rows or a `Dataset` of the same binning and returns the seed bag's mean;
  `predict(X, per_seed=True)` returns each model's prediction, `(n, n_seeds)` in seed order (column i is what a one-seed
  fit with seed `seed + i` predicts). The estimator gives the functional layer's bits (`train` on the same `Dataset`),
  and its backends are bit-identical (tests).
- `fit(X, y, sample_weight)`: one non-negative weight per row (a list, a pandas Series or an array), `train`'s
  `weight`; scikit-learn's weight checks pass. `fit(..., eval_sample_weight=[w_0, ...])`: each eval set's weights (or
  None), `Validation`'s `weight`.
- `fit(X, y, eval_set=[(X_val, y_val), ...], eval_names=None, callback=None)` scores each tree on the validation sets
  (binned with the training binning; a `Dataset` with its target is taken as it is) with `eval_metric` (`"l2"` by
  default, `"rmse"`, `"l1"`); with `early_stopping_rounds` the fit stops when the first set has not improved for that
  many trees and keeps the trees up to its best; `verbose` = k > 0 prints every k-th tree's scores (LightGBM's
  lines); `callback` is `train`'s.
- `predict(X, num_iteration=None, start_iteration=0)`: `num_iteration` trees from tree `start_iteration` (from 0), all
  the model keeps by default (an early-stopped model keeps the trees up to its best).
- `predict(X, pred_contrib=True)`: each row's TreeSHAP contributions, `(n, n_features_in_ + 1)`, the columns in
  `feature_names_in_` order and the expected value last; they add up to `predict(X)` (to float32 rounding) and take
  `num_iteration` and `start_iteration` (LightGBM's layout). A seed bag's are its models' mean, with `per_seed=True`
  each model's, `(n, n_seeds, n_features_in_ + 1)`. `predict(X, pred_leaf=True)`: the leaf each row reaches in each
  tree, `(n, n_trees)`; a seed bag's need `per_seed=True`, `(n, n_seeds, n_trees)`. Not both at once. They run on the
  host whatever the backend: [`Booster.predict_contributions`](#booster).
- `eval_metric` None is the estimator's own: `"l2"` for `GBMRegressor`, `"binary_logloss"` for `GBMClassifier`.

### `GBMClassifier`

Binary classification with `GBMRegressor`'s parameters and methods: LightGBM's `binary` objective. The trees fit
`LogLoss` (Newton leaves; second-order splits with `second_order=True`) from the log-odds of the training labels' mean
(`boost_from_average`). `fit(X, y)` takes any two labels (`classes_`, sorted; the second is the positive class), or a
`Dataset` whose target is 0 and 1; more than two classes raise `ValueError` (the functional layer fits them:
[`train_multiclass`](#train_multiclass-softmax)). `predict_proba(X)` is `[1 − p, p]` with `p = 1 / (1 + e^−s)` of the
raw score `s` (a seed bag's mean score), `predict(X)` the label of the larger probability, `decision_function(X)` the
raw score, each with `num_iteration` and `start_iteration`; `score` is the accuracy. `predict(X, pred_contrib=True)` and
`predict(X, pred_leaf=True)` are the regressor's: the contributions add up to the raw score, as LightGBM's. `eval_metric`: `"binary_logloss"`
(default) or `"auc"`. `classes_` is saved with the model. It tells scikit-learn it is binary
(`classifier_tags.multi_class = False`: more classes raise "Only binary classification is supported") and passes its
estimator checks with the two exceptions every idun estimator has (below).
- `get_params()`, `set_params(**params)`, `params()` (the validated `Params`), and after `fit`: `boosters_`,
  `n_features_in_`, `feature_names_in_`, `best_iteration_` (early stopping's best tree, from 0; None without early
  stopping), `evals_result_` (`{set: {metric: [score after each tree]}}`) and `feature_importances_` (splits per
  feature, `importance_type="split"`, summed over a bag's models); in a seed bag `best_iteration_` and `evals_result_`
  are lists, one per model.
- `save(path)` writes one `idun-model` file (an uncompressed `.npz` without object arrays: a JSON `meta` entry with the
  estimator's name and parameters, the binning spec and each booster's parameters; `named_edges`; each booster's forest
  arrays as `b<i>_<field>`). `idun.load(path)` reads any idun estimator back and refuses anything else.

### `Params`

Frozen dataclass. Every field is type- and range-checked at construction; a bad value raises `ValueError` naming the
field. `to_dict()` / `from_dict()` round-trip it (unknown keys raise).

| field | type | default | meaning |
|---|---|---|---|
| `num_trees` | int | 100 | boosting iterations |
| `learning_rate` | float | 0.1 | shrinkage, applied to each leaf value |
| `max_depth` | int | 6 | depth-wise: the trees' depth (1..8; node ids are uint8). Leaf-wise: `-1` (no limit) or the depth no leaf may exceed (≥ 1) |
| `max_leaves` | int | 64 | depth-wise: leaf cap (2..2^max_depth), when a level would exceed it the highest-gain splits are kept. Leaf-wise: leaves per tree (2..256, and ≤ 2^max_depth with a depth limit) |
| `growth` | `"depthwise"` \| `"leafwise"` | `"depthwise"` | `"leafwise"`: LightGBM's best-first growth: split the leaf of highest gain (ties: the lowest leaf id) until `max_leaves` leaves or no leaf has a split. Composes with every other field |
| `min_data_in_leaf` | int | 20 | minimum in-bag rows on each side of a split |
| `lambda_l2` | float | 0.0 | L2 regularization on leaf values |
| `feature_fraction` | float | 1.0 | fraction of features sampled per tree (`max(round(f·n), 1)`) |
| `bagging_fraction` | float | 1.0 | per-row probability of being in the bag |
| `bagging_freq` | int | 0 | re-draw the bag every k iterations (0 = no bagging) |
| `grad_quant_bins` | int | 4 | quantization levels for gradients (LightGBM `num_grad_quant_bins`; 2..128, a power of 2) |
| `categorical` | `"sorted"` \| `"one_vs_rest"` | `"sorted"` | split method for categorical columns ([determinism.md](../architecture/determinism.md) §4) |
| `cat_smooth`, `cat_l2`, `max_cat_threshold`, `min_data_per_group` | float, float, int, int | 10, 10, 32, 100 | LightGBM's categorical parameters, with its defaults |
| `seed` | int | 0 | uint32; drives bagging, feature sampling and stochastic rounding |
| `lambda_l1` | float | 0.0 | L1 regularization (LightGBM's): a leaf's gradient sum moves towards 0 by it (`ThresholdL1`) in the gains and the leaf values |
| `min_gain_to_split` | float | 0.0 | the gain a split must exceed (LightGBM's) |
| `min_sum_hessian_in_leaf` | float | 1e-3 | the hessian sum each side of a split needs (LightGBM's): under L2 and an `Objective` the hessians are the in-bag row counts, so it binds only above `min_data_in_leaf`; a `NewtonObjective`'s splits (and `GBMClassifier`'s) sum its hessians |
| `max_delta_step` | float | 0.0 | > 0: a leaf's output before the learning rate is clipped to ±it, and the gains are the clipped outputs' (LightGBM's); 0: no limit |
| `second_order` | bool | False | the split search reads a `NewtonObjective`'s, a weighted or a multi-class fit's hessians (LightGBM's rule); False keeps first-order splits (rows as the hessian) for every objective |
| `extra_trees` | bool | False | LightGBM's: each node tries one random threshold per numeric feature (both missing directions at it; categorical features keep their search), drawn from the seed, the tree and the node on the host ([determinism.md](../architecture/determinism.md) §1) |
| `feature_fraction_bynode` | float | 1.0 | LightGBM's: each node searches this fraction of the tree's sampled features (drawn per node) |

The four regularisers' defaults turn them off; with them set, the CPU backend agrees with
the brute-force oracle bit for bit, and the GPU backends with the CPU (the split search and the leaf values are the
host's). The defaults are neutral.

### `train`

```python
def train(
    params: Params,
    data: BinnedData,
    *,
    rows: slice,
    backend: str,
    y: NDArray[np.float32] | None = None,
    objective: Objective | NewtonObjective | None = None,
    valid: Sequence[Validation] = (),
    metric: str | Metric = "l2",
    early_stopping_rounds: int | None = None,
    callback: Callable[[Evaluation], bool | None] | None = None,
    init_model: Booster | None = None,
    boost_from_average: bool = False,
    weight: NDArray[np.float32] | None = None,
) -> Booster: ...
```

- `rows` is a contiguous training range; `y` defaults to `data.target[rows]`. For other rows (a subset, a sample,
  repeats), take them first: [`take_rows`](#take_rows-binnedrows).
- `backend` is required. Nothing selects a backend implicitly; an unavailable one raises `BackendUnavailableError`.
- `objective`: None is L2 on `y`; see [`Objective`, `NewtonObjective`](#objective-newtonobjective).
- Raises `OverflowError` if `max|g| × rows ≥ 2^31` (the exact int64 leaf sums would overflow; determinism.md §5).
- The returned `Booster.fit_report` is a `FitReport`.
- `valid`: validation sets, each `Validation(data, rows, y=None, name=None, weight=None)` (rows of a panel binned as
  the training data, `y` = `data.target[rows]` by default, named `valid_<i>` by default), scored with `metric` after
  every tree. With `weight` (one non-negative weight per row) a built-in metric is LightGBM's weighted
  form, `Σ w v / Σ w` per row's loss (AUC: each positive's weight times the negatives' weight scored below it, half of
  those tied); unit weights give the unweighted score, and a custom `Metric` refuses weights (`ValueError`).
  Each new tree's outputs are added on the host, in tree order as `Booster.predict` adds them, so the score after tree
  i is the metric of what the first i + 1 trees predict, the same float on every backend. The scores are
  `Booster.evals`. `metric`: `"l2"`, `"rmse"`, `"l1"`, `"binary_logloss"` or `"auc"` (LightGBM's names; `METRICS`;
  `"multi_logloss"` and `"multi_error"` are `train_multiclass`'s;
  each on raw scores) or a `Metric(name, fn, higher_is_better=False)` with `fn(y, pred) -> float`.
- `early_stopping_rounds`: stop when the first set's score has not improved for that many trees, and keep the trees up
  to its best (`Booster.best_iteration`, from 0): the model is then the model of `best_iteration + 1` trees, bit for
  bit (`params.num_trees` says so). It needs a validation set. LightGBM watches every set and metric; idun the first.
- `callback(evaluation)` runs after every tree with an `Evaluation(iteration, metric, scores)`; returning True stops the
  fit there and keeps every tree trained. `log_evaluation(period)` is a callback that prints LightGBM's lines.
- `train_seeds` takes the same four, for each seed's fit (each stops on its own); `train_many` takes
  none of them.
- `init_model`: continue that model. The fit starts from its predictions on the training rows (which equal its
  running predictions bit for bit), its trees come first in the returned model, and the new trees draw their bags,
  feature samples and rounding as the model's trees `init_model.n_trees` and on (a bag drawn every `bagging_freq`
  trees continues mid-period). So a model of a trees continued for b is the model of a + b trees, bit for bit, on
  every backend. It must have the data's binning and features and the same `growth`, `max_depth` and `max_leaves`;
  the other parameters may change. Validation sets start from its predictions; `best_iteration` counts from the
  model's first tree.
- `boost_from_average`: LightGBM's start, the mean of `y` (an objective's `start(y)`, as `LogLoss.start`, the
  log-odds), set as every row's starting prediction and folded into the first tree's leaves, so `predict` gives the
  fit's running predictions. Off by default (today's models start at 0); not with `init_model`.
- `weight`: one non-negative weight per training row (LightGBM's dataset weights). Each row's gradient and hessian
  are multiplied by it on the host, in float32 (the built-in L2's hessian, and an `Objective`'s, is 1, so its weighted
  hessian is the weight), so a weighted fit's leaves are `−Σ w g / (Σ w h + λ) · learning_rate` and, with
  `second_order`, its splits read the weighted hessians, LightGBM's weighted fit; `min_data_in_leaf` still counts rows, and a row of weight 0 still takes part in the binning, the
  bag and the quantization scale. With `boost_from_average` the start is the weighted mean (an objective's
  `start(y, weight)`: `LogLoss.start` takes the weights; another objective's `start` without a `weight` parameter
  raises). All weights 1 give the unweighted model bit for bit. `ValueError` for a weight that is negative, not
  finite or of the wrong length, and when every weight is zero. `train_seeds` and `train_many` take it too (one
  weight per row for every model).

### `Booster`

```python
class Booster:
    params: Params
    forest: Forest | NodeForest  # depth-wise heaps or leaf-wise node lists (data-layout.md §6)
    n_trees: int
    binning: str
    feature_names: tuple[str, ...]
    categories: tuple[tuple[str, ...], ...]  # each categorical column's levels
    train_rows: tuple[int, int]
    backend: str
    fit_report: FitReport | None  # None after load
    best_iteration: int | None  # early stopping's best tree; None without early stopping
    evals: dict[str, list[float]]  # each validation set's score after every tree trained

    def predict(
        self, data: BinnedData, *, rows: slice, backend: str, num_trees: int | None = None, start_tree: int = 0
    ) -> NDArray[np.float32]: ...
    def predict_contributions(
        self, data: BinnedData, *, rows: slice, num_trees: int | None = None, start_tree: int = 0
    ) -> NDArray[np.float64]: ...
    def predict_leaves(
        self, data: BinnedData, *, rows: slice, num_trees: int | None = None, start_tree: int = 0
    ) -> NDArray[np.int32]: ...
    def feature_importance(self, kind: str = "split") -> NDArray[np.int64]: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> Booster: ...
```

- `predict` raises if `data.binning`, `data.feature_names` or `data.categories` differ from the model's: bins from
  another refit's edges mean wrong splits.
- `num_trees` predicts with the first k trees (for example, to read several training lengths from one model); with `start_tree`, the k
  trees from that one (all from it by default).
- `predict_contributions` returns each row's TreeSHAP values (path-dependent TreeSHAP, Lundberg et al. 2018), `(n,
  n_features + 1)` in the data's feature order with the expected value last (LightGBM's `pred_contrib` layout): each
  tree's expected output over its in-bag training rows plus each feature's Shapley value under them, so a row's
  contributions add up to its raw prediction. Float64, computed on the host, the same for every backend; the
  covers are the trees' leaf counts (`forest.leaf_count`, the in-bag rows per leaf), and a model
  loaded from a file without them raises `ValueError`. `predict_leaves` returns the leaf each row
  reaches in each tree, `(n, n_trees)`, int32 (LightGBM's `pred_leaf`; a depth-wise leaf's index in its heap's leaves).
  Both check the data as `predict` does and take its tree window.
- `feature_importance()` counts each feature's splits, in the data's feature order (LightGBM's `"split"`; the gains
  are not kept with the trees). `best_iteration` and `evals` are saved with the model.
- Predictions on the training rows are bit-identical to the fit's running predictions (one float32 addition per tree,
  in order). A saved and loaded model predicts bit-identically.
- File format: [model-format.md](model-format.md).

### `FitReport`

Frozen dataclass: `seconds` (wall-clock per phase: `upload`, `gradients`, `histograms`, `splits`, `partition`,
`leaves`), `backend`, `train_rows`, and the `total_seconds` property. Reported, never used for decisions.

### `Objective`, `NewtonObjective`

```python
class Objective(Protocol):  # unit hessians
    constant_hessian: Literal[True]

    def gradients(self, pred: NDArray[np.float32], y: NDArray[np.float32], out: NDArray[np.float32]) -> None: ...


class NewtonObjective(Protocol):  # its own hessians
    constant_hessian: Literal[False]

    def gradients(
        self, pred: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None: ...
```

A loss for `train`, `train_seeds` and `train_many`, given as its derivatives. Each iteration `gradients` is called on
the host with the model's predictions and labels of every training row (in-bag or not), in training order, and writes
`g = ∂loss/∂pred` into `out` / `grad` (and `h = ∂²loss/∂pred²` into `hess`), every row, in place. `pred` is for
reading only (on a GPU backend the array is read-only), and none of the arrays may be kept for a later call. The
gradients steer the whole tree:

- **Splits**: the split search reads the quantized gradients and counts rows as the hessian (first-order splits),
  under both protocols. With `Params.second_order=True` (the estimators' `second_order`)
  a `NewtonObjective`'s splits read its quantized hessians, `G² / (H + λ)` with `H` their sum, as LightGBM's
  ([determinism.md](../architecture/determinism.md) §2–4). The hessians are rounded as the gradients are, to
  `[0, grad_quant_bins / 2]` (LightGBM: `[0, grad_quant_bins]`), and their histograms are a second pass of the
  histogram kernels, so such a fit spends about twice the histogram time. `min_data_in_leaf` is in rows either way,
  `min_sum_hessian_in_leaf` in hessians with `second_order` (rows without). Unit hessians give the `Objective`'s fit
  bit for bit either way.
- **Leaves**: `−Σg / (n + λ) · learning_rate` for an `Objective`, the Newton step `−Σg / (Σh + λ) · learning_rate`
  for a `NewtonObjective`, over the leaf's in-bag rows; both sums are exact (determinism.md §5).

`LogLoss()` is the built-in binary log loss on raw scores, a `NewtonObjective` (`g = p − y`, `h = p (1 − p)`, `p` the
score's sigmoid, in float64) with `start(y)`, the log-odds of the labels' mean; `GBMClassifier` trains it.

The arrays are views of the backend's buffers (shared memory on Metal): write them, do not keep them. The rows are
the training rows in order, so an objective may hold per-row data aligned with them (weights, group ids, its own
labels) and ignore `y`. It must be a pure function of its arguments and its own constants: the same call gives the
same floats, or fits stop repeating. `train_seeds` calls one objective from fits side by side; one that keeps
scratch state between calls needs `workers=1`.

Checked every iteration: gradients finite with `max|g| × rows < 2^31`; hessians finite, non-negative, with
`max h × rows < 2^31` (`ValueError`, `OverflowError`). A leaf that holds rows while `Σh + λ = 0` raises `ValueError`:
give positive hessians or set `lambda_l2 > 0`. The estimators take no objective.

Annotate `constant_hessian` as the literal (`constant_hessian: Literal[False] = False`) for a type checker to pick
the protocol. Two examples (idun ships no objective):

```python
class Logistic:
    """Binary cross-entropy on 0/1 labels; the model predicts the logit."""

    constant_hessian: Literal[False] = False

    def gradients(self, pred, y, grad, hess):
        p = (1.0 / (1.0 + np.exp(-pred.astype(np.float64)))).astype(np.float32)
        np.subtract(p, y, out=grad)
        np.multiply(p, np.float32(1.0) - p, out=hess)


class WeightedL2:
    """L2 with a weight per training row: a leaf is the weighted mean residual."""

    constant_hessian: Literal[False] = False

    def __init__(self, weights):  # float32, one per training row, >= 0
        self.weights = weights

    def gradients(self, pred, y, grad, hess):
        np.multiply(self.weights, pred - y, out=grad)
        hess[:] = self.weights


model = gbm.train(params, data, rows=rows, backend="metal", y=labels01, objective=Logistic())
```

### `BinnedData`

The structural interface `train` and `predict` read (`idun.data.PackedDataset` satisfies it): `n_rows`, `binning`,
`feature_names`, `named_features`, `obfuscated_features`, `categories` (each categorical column's levels, at most 64),
`named_n_bins`, `named_edges`, `named_bins` `(n_named, n_rows)`, `obfuscated` `(n_obfuscated, ceil(n_rows/2))`,
`categorical` `(n_categorical, n_rows)` uint8 codes, `target`. Features are named, then obfuscated, then categorical;
any kind may have no columns.

### `take_rows`, `BinnedRows`

```python
def take_rows(data: BinnedData, rows: NDArray[np.integer]) -> BinnedRows: ...
```

Copies the rows named into memory: the binned features and the targets of `rows`, in the order given, as a
`BinnedRows` (a dataclass with `BinnedData`'s members). `train` fits one contiguous range, so a fit on other rows
starts here:

```python
train_part = slice(0, n_train)
subset = gbm.take_rows(data, np.flatnonzero(mask[train_part]))  # a subset by a boolean mask
sample = gbm.take_rows(data, np.sort(rng.choice(n_train, 400_000, replace=False)))
copies = gbm.take_rows(data, np.repeat(np.arange(n_train), counts))  # integer row weights as row copies
model = gbm.train(params, subset, rows=slice(0, subset.n_rows), backend="metal")
pred = model.predict(data, rows=slice(n_train, data.n_rows), backend="metal")
```

- `rows` is a 1-D integer array of indices in `[0, data.n_rows)`: any order, and a row named twice is there twice.
  Anything else raises `ValueError` (a boolean mask goes through `np.flatnonzero`).
- The names, edges, categories and the `binning` id are `data`'s, so the booster predicts `data` and any data binned
  the same way. `target` holds the rows' targets; pass `y=` for other labels, one per taken row.
- The result is a dataset of its own, with rows numbered from 0. A fit's per-row draws go by a row's number in the
  data it trains on, so a fit on `take_rows(data, np.arange(n))` equals `train(..., rows=slice(0, n))` bit for bit,
  and a fit on other rows is as reproducible as the rows given.
- Memory: one byte per row and named or categorical feature, half a byte per row and obfuscated feature.

### `train_seeds`, `predict_mean` (multi-seed mode)

```python
def train_seeds(
    params: Params,
    data: BinnedData,
    *,
    rows: slice,
    backend: str,
    seeds: Sequence[int],
    y: NDArray[np.float32] | None = None,
    workers: int | None = None,  # Metal: SEED_WORKERS = 3, capped by the GPU's working set; CUDA: one fit a GPU
    objective: Objective | NewtonObjective | None = None,  # one for every seed
    weight: NDArray[np.float32] | None = None,  # train's, for every seed
) -> list[Booster]: ...


def predict_mean(models: Sequence[Booster], data: BinnedData, *, rows: slice, backend: str) -> NDArray[np.float32]: ...
```

A seed bag: one model per seed, each **bit for bit** the model `train` fits with `params` and that seed, returned in
`seeds` order. Seeds share no arithmetic (each draws its own bag, feature samples and quantization draws), but one Metal
fit leaves much of the GPU idle, so up to `workers` fits run side by side, each on its own backend and thread: three
at a time gave 2.4× the throughput of one after another on an Apple M4. Every fit references about 4 GB on the GPU
(`fit_bytes`), and fits beyond the GPU's recommended working set thrash (two RealMLP networks ran 16× slower than one
on an 8 GB Apple M1), so `workers=None` takes 3 or as many as fit (`idun.metal.runtime.workers_side_by_side`: 3 on a
24 GB M4, 1 on an 8 GB M1), and a larger explicit `workers` raises
`ValueError`. On CUDA `workers=None` runs one fit on each GPU at a time (on two Tesla T4s a bag of four seeds took
30 s, against 54 s one after another; two fits a GPU gained nothing); the fits are spread over the machine's GPUs in
turn (two fits on two GPUs take one each), or all run on the GPU `IDUN_CUDA_DEVICE` names (one at a time unless
`workers` says otherwise), and each fit releases its GPU memory before the next takes the GPU. `workers` > 1 needs `backend="metal"` or `"cuda"` (the CPU
backend's parallel loops take turns, so its fits gain nothing side by side; `ValueError` otherwise); duplicate or no seeds raise
`ValueError`.
`predict_mean` is the bag's prediction: the mean of the models' raw predictions (float64 sum in model order, then
float32).

**Small tables.** A Metal fit grows its trees in GPU round trips, one per level (depth-wise) or per split (leaf-wise),
and on a small table each carries little work: the GPU idles between them, and one fit alone runs slowly. Fits side
by side fill those gaps: on an 8 GB Apple M1, a table of half a million rows fitted two seeds side by side
(`n_seeds=2` with the default `workers`) in less time than one seed alone. idun_gbm is built for panels of millions
of rows, where each round trip carries enough work.

### `train_many`

```python
def train_many(
    params: Params,
    data: BinnedData,
    *,
    rows: slice,
    backend: str,
    ys: Sequence[NDArray[np.float32]],
    seeds: Sequence[int] | None = None,
    objective: Objective | NewtonObjective | Sequence[Objective | NewtonObjective] | None = None,
    weight: NDArray[np.float32] | None = None,  # train's, the same rows' weights for every model
) -> list[Booster]: ...
```

Fits K = `len(ys) × len(seeds)` models (seed-major order). The targets of one seed share the bag, the feature sample
and the quantization draws, so they run in lockstep; each tree is identical to the one a separate `train` call gives
(tested on CPU and Metal). `objective`: one for every target, or a sequence with one per target of `ys` (another
length raises `ValueError`); each is called with its own model's predictions and labels, once per iteration, in
target order.

On Metal (`MetalMany`) the models share each pass over the data. Measured on an Apple M4: with the obfuscated
features only (a panel of Numerai's data, K = 5–10) lockstep was **not faster** than separate fits (0.93–0.96×;
that pass is bound by the per-(row, model, feature) counter updates, which batching does not reduce); on 400,000
sampled rows of the whole feature set, four lockstep targets took 31–33 s where four separate fits took 60–68 s.

### `train_multiclass`, `Softmax`

```python
def train_multiclass(
    params: Params,
    data: BinnedData,
    *,
    rows: slice,
    backend: str,
    y: NDArray,  # class ids 0 .. n_classes - 1, one per training row
    n_classes: int,
    objective: MulticlassObjective | None = None,  # Softmax(n_classes) by default
    valid: Sequence[Validation] = (),
    metric: str | Metric = "multi_logloss",
    early_stopping_rounds: int | None = None,
    callback: Callable[[Evaluation], bool | None] | None = None,
    boost_from_average: bool = False,
    weight: NDArray[np.float32] | None = None,
) -> list[Booster]: ...  # one per class, in class order
```

LightGBM's `multiclass`: each iteration grows one tree per class. The K trees of an iteration share the bag, the
feature sample and the quantization draws (one lockstep group, as `train_many`'s), and the objective reads every
class's scores: `Softmax(K)` writes `g_k = p_k − [y = k]` and `h_k = K / (K − 1) · p_k (1 − p_k)` (LightGBM's
factor), `p = softmax(s)` in float64, so every class's tree has Newton leaves (and second-order splits with
`Params.second_order`, LightGBM's). A row's
class probabilities are the softmax of the K Boosters' raw predictions. Each class's tree t is, bit for bit, the tree
a one-tree continuation of that class's first t trees fits (`init_model`) with `Softmax`'s gradient for that class and
the other classes' scores held fixed (a test), on every backend. `valid` (each `Validation`'s `y` class ids),
`metric` (`"multi_logloss"`, `"multi_error"`, or a `Metric` whose `fn(y, scores)` takes the `(K, n)` scores),
`early_stopping_rounds` and `callback` are `train`'s over all K classes (they stop together); `boost_from_average`
starts each class at `Softmax.start(y, weight)`, the log of its share of the labels (LightGBM's softmax starts at 0,
the default); `weight` is `train`'s. A `MulticlassObjective` has `n_classes` and `gradients(preds, y, grad, hess)`
on `(K, n)` arrays. `multi_logloss` and `multi_error` need a multi-class fit, and the one-model metrics a one-model
fit (`ValueError`).

### `idun.rng`

Counter-based draws shared by every backend. Contract: [determinism.md](../architecture/determinism.md) §1.

| function | signature |
|---|---|
| `lowbias32` | `(x: int) -> int`: hash a uint32 |
| `iteration_key` | `(seed: int, iteration: int) -> int`: the key for one boosting iteration |
| `uniform_bits` | `(key: int, n_rows: int) -> NDArray[np.uint32]`: 24 random bits per row; multiply by `2**-24` for `[0, 1)` |
| `derive_key` | `(seed: int, *counters: int) -> int`: hash counters into a key; `derive_key(seed, it) == iteration_key(seed, it)` |
| `element_bits` | `(key: int, rows: NDArray[np.int64], n_elements: int) -> NDArray[np.uint32]`: 24 bits per (absolute row, element) |

It lives outside `idun.gbm` so the networks can share it without importing idun_gbm.

## `idun.tabm`: idun_tabm ([design notes](../architecture/networks.md))

TabM (Gorishniy et al.) on Metal, with the float64 NumPy reference: `from idun.tabm import TabMParams, TabMModel,
train_tabm`.

### Example

```python
from idun.data import PackedDataset
from idun.tabm import TabMParams, train_tabm
from idun_numerai.inputs import classes

data = PackedDataset.open("cache", binning="r0")
refit = data.refits[0]
params = TabMParams(n_classes=5)  # Numerai's targets have 5 classes
model = train_tabm(params, data, rows=slice(0, refit.train_hi), backend="metal", classes=classes(data.target))
pred = model.predict(data, rows=slice(refit.pred_lo, refit.pred_hi), backend="metal")
model.save("tabm.npz")
```

### `TabMClassifier`, `TabMRegressor`

```python
from idun.tabm import TabMClassifier

clf = TabMClassifier(n_seeds=2, categorical=["sector"]).fit(X, labels)  # any labels, >= 2 classes
proba = clf.predict_proba(X_test)  # (n, n_classes): the mean of the seed bag's member-averaged softmax
each = clf.predict_proba(X_test, per_seed=True)  # (n, n_seeds, n_classes): each network's, in seed order
pred = clf.predict(X_test)  # the most probable class's label
```

The parameters are `TabMParams`' network and training fields (not `beta1`, `beta2` or the temporal encoder's; the class
count comes from the labels, at most 8 classes on the GPUs) plus `n_seeds`, `backend` (`"metal"` by default; `"cuda"`,
or `"reference"` for the float64 network) and `categorical`. The defaults are
`TabMParams`' (TabM's paper: k = 32, 3 blocks of 512, dropout 0.1, AdamW at 2e-3 with weight decay 3e-4, 256-row
batches). The paper trains until 16 epochs pass without
improvement on a validation set: `fit(X, y, eval_set=(X_val, y_val))` scores the validation rows after every epoch
(the error rate of the most probable class; for `TabMRegressor` the squared error on the standardized target), each
network keeps the weights of its best epoch, and `patience=16` stops it after 16 epochs without a better one;
`best_epoch_` and `valid_score_` read them back (lists in a seed bag; kept with the model file).
`fit(X, y, sample_weight)`: one finite, non-negative weight per row (not all zero), `train_tabm`'s `weight`: each
step's loss is its batch's weighted mean; the validation score and the regressor's standardization stay unweighted.
Without a validation set `epochs` is fixed: 50 by default. Inputs pass a `idun.nn.table.TableEncoder`:
numeric columns standardized with the training rows' mean and spread (NaN → 0), categorical columns one-hot. `classes_`
holds the sorted labels. `save` writes an `idun-model` file with each network's own model file inside; `idun.load` reads
it. The estimator gives `train_tabm`'s bits on the encoded `MatrixData` (a test). Shared with `RealMLPClassifier`
through `idun.nn.estimator.NetClassifier`.

`TabMRegressor` takes the same parameters for regression: the target is standardized with the training rows' mean and
spread (`target_mean_`, `target_scale_`), the networks train on `objective` and `predict` returns their prediction in
the target's units (`predict(X, per_seed=True)`: each network's, `(n, n_seeds)` in seed order; column i is what a
one-seed fit with seed `seed + i` predicts). `objective` is `"mse"` (the default: the members' mean output) or `"hl_gauss"` (HL-Gauss on the
standardized target, with `n_bins`, `hl_low`, `hl_high` and `hl_sigma` as in `TabMParams`; the expected bin centre,
scaled back); `TabMClassifier` has no `objective` parameter. Classifiers and regressors share `idun.nn.estimator.NetEstimator` (the encoder, the seed
bag, the `idun-model` file); `TabMClassifier` and `TabMRegressor` share TabM's settings.

### `train_tabm`

`train_tabm(params, data, *, rows: slice, backend: str, on_epoch=None, classes=None, on_weights=None, targets=None,
sequences=None, checkpoint=None, loss=None, batches=None, valid=None, patience=None, weight=None) -> TabMModel`:
`weight`: one finite, non-negative weight per training row (`(hi − lo,)`, not all zero); each step's
loss is its batch's weighted mean `Σ w l / Σ w` (and the mean over the members) in place of the plain
mean, so the batches, the dropout and the AdamW steps stay as they are and a row of weight 0 only leaves its batch's
loss; a batch whose rows all weigh 0 has loss 0 and gradient 0. The loss kernels scale each row's gradient by
`w / (Σ w · k)` where they scaled by `1 / (B · k)`, so unit weights give the unweighted fit bit for bit on the
GPUs (a test); the reported losses are the weighted means; a validation set's score is not weighted. Not with `loss`
(which weighs its own rows: `ValueError`); a bad weight array raises `ValueError` before any training.
`valid` (`Validation(data, rows)`: contiguous rows of a `MatrixData` with their labels or targets) is scored after
every epoch by the backend itself (the members' mean softmax's error rate for `"cross_entropy"`, the prediction's
squared error for `"mse"` and `"hl_gauss"`); the model keeps the weights of the epoch with the lowest score (the first
on a tie), and `patience` stops training after that many epochs without a lower one. `report.valid_score` and
`report.best_epoch` record them. One target, no sequences, no checkpoint (each raises `ValueError`).
`on_weights(epoch, weights)` gets the float32 weights after each epoch, which are bit for bit the fit with
`epochs = epoch + 1` (the learning rate is constant and an epoch's order depends only on the seed and the epoch; a
test), so one run gives every shorter training. The loop ([networks.md](../architecture/networks.md) §6), on the
contiguous training rows: each epoch a fresh permutation (`PCG64(SeedSequence([seed, epoch, 2]))`),
batches of `batch_size`, the ragged last batch kept, one AdamW step per batch; the next batch (inputs and labels) is
gathered on a worker thread while the backend runs the current step. `backend` is `"metal"`, `"cuda"` (an NVIDIA GPU,
`CudaTabM` below) or `"reference"` (the float64 oracle, small problems only); no automatic choice. `on_epoch(epoch, mean_loss, seconds)` is called after every
epoch. `params.micro_batch` reaches the Metal backend unchanged (training and `TabMModel.predict`); the reference
backend ignores it.

`checkpoint`: a file for the training state, so that a long fit can stop and carry on. After each epoch, before
`on_epoch` and `on_weights` run, the weights, AdamW's moments and step count, the epochs done and the report so far are
written there atomically (a temporary file, synced, then renamed). A call that finds the file resumes after its last
epoch and returns the uninterrupted fit, bit for bit (a test, on both backends): stopping the process at any moment, or
raising from `on_epoch`, loses at most the epoch in progress. The fit's parameters, rows, inputs, targets, sequence
shape and backend are recorded and must match (`ValueError` names what differs); the data itself is not checked, so
resume with the same data. Of a `loss` and of `batches` (below) the checkpoint records that one was given, not which:
pass the same ones. A finished checkpoint returns the model without training. The file stays after the fit: remove it
once the model is saved.

```python
model = train_tabm(
    params, data, rows=slice(0, train_hi), backend="metal", classes=labels, checkpoint="fit.ckpt.npz"
)  # stopped and run again: carries on after the last epoch
model.save("fit.npz")
Path("fit.ckpt.npz").unlink()
```

`classes`: a `PackedDataset`'s labels, an `(n_rows, 1)` uint8 class array (row r's class at `[r, 0]`,
every training row's class below `n_classes`; `inputs.MISSING_CLASS` may mark other rows). Numerai's are
`idun_numerai.inputs.classes(target)`, `rint(4 · y)`. The network predicts `(n,)`. A
`MatrixData` carries its own labels; a packed dataset without `classes`, a `MatrixData` with them, a wrong shape or
dtype, or a training row's class at or above `n_classes` (when its batch is gathered) raises `ValueError`.

`targets`: for `objective="mse"` (`n_classes=1`), a `PackedDataset`'s
regression target instead, an `(n_rows, 1)` float32 target array (row r's at `[r, 0]`;
`inputs.target_array(n_rows, 1)` allocates one, filled with NaN). The target of every training row must be finite,
checked before any training; NaN may mark other rows. The batches are gathered on the host as for `classes`, so a
packed fit equals the fit on `MatrixData(dense_inputs(data, all rows), y)` bit for bit (a test); the model predicts
the members' mean, `(n,)`. A packed dataset takes `classes` for `"cross_entropy"` and `targets` for
`"mse"` or `"hl_gauss"`, never both (`inputs.check_label_source`); any other combination raises `ValueError`.

With `objective="hl_gauss"` the `targets` (one column) are
turned into soft labels batch by batch on the host (`idun.tabm.hl_gauss.labels`, float64; the Metal backend gets them
as float32), the network trains on soft-label cross-entropy and the model predicts the expected bin centre in the
targets' units. The defaults' range, [−4, 4], suits a standardized target.

`sequences`: for a network with a temporal encoder (`params.encoder="gru"`, below), every row's sequence, a
`Sequences` over the same rows as `data` (`ValueError` without them, with them and no encoder, or for another row
count); each batch's sequences are gathered on the host with its inputs. The model keeps the inputs per step
(`TabMModel.sequence_inputs`), and `predict` and `predict_proba` take sequences built the same way.

#### A loss and a batch plan of the caller's: `loss`, `batches`

```python
Loss = Callable[[NDArray[np.floating], NDArray[np.int64]], tuple[float, NDArray[np.floating]]]
BatchPlan = Callable[[int], Sequence[NDArray[np.integer]]]
```

`Loss` and `BatchPlan` are exported from `idun.tabm`.

`loss(outputs, rows) -> (value, d_outputs)` replaces the training loss. It is called once per step, on the host, with
the batch's member outputs `(B, k, c)` (training mode: that step's dropout; float64 on the reference, float32 on
Metal) and the batch's absolute row indices `(B,)`, and returns the loss (a float; the report's epoch loss is the
mean over the epoch's steps) and its gradient with respect to `outputs`, in their shape.

- The loss holds what it needs, indexed by row: labels, weights, anything else. Pass no `classes` or `targets`; a
  `MatrixData`'s own labels are not read.
- `params.objective` still declares the outputs and the prediction: `"mse"` (`n_classes=1`) is one output per
  member, c = 1, and the model predicts the members' mean; `"cross_entropy"` is `n_classes` logits and the model
  predicts the expected class (`predict_proba` the member-averaged softmax). `"hl_gauss"` takes no loss. One target.
- The gradient is used as returned: a mean over the batch carries its `1 / (B · k)`. Every member is trained by its
  own outputs' gradient; a loss of the members' mean trains them as one.
- A gradient of another shape, or a loss or a gradient that is not finite, raises `ValueError`.
- It must be a pure function of its arguments and its own constants, or fits stop repeating.
- A direction of the outputs the loss ignores gets a gradient of rounding size, which AdamW normalises into a drift:
  a loss blind to the outputs' level lets the head's bias wander, and differ between float32 and float64; where the
  level matters, add a term that pins it (a small multiple of the outputs' mean square does).
- On Metal a step is two command buffers with the host between them, and a micro-batched fit runs each
  micro-batch's forward pass twice. With the built-in loss written out (`"mse"`: `2 (o − y) / (B · k)`) the
  reference gives the built-in fit bit for bit, Metal within float32 rounding (tests).

`batches(epoch) -> the epoch's batches` replaces the seeded shuffle: a sequence of 1-D integer arrays, each
1 .. `params.batch_size` absolute row indices inside `rows`. Any number of batches of any such sizes; a row may
repeat or be left out; an epoch needs at least one batch (`ValueError` for each of these checks, raised when the
epoch is planned). Each batch is one AdamW step, and the dropout masks go by the count of steps taken. It must be a
pure function of the epoch: a resumed fit calls it for the epochs left, and may call it for an epoch before the one
before has finished. On Metal the buffers are sized for `params.batch_size` rows. The default plan, written as one,
gives the default fit bit for bit (a test).

```python
class WeightedSquares:
    """The weighted mean of (member output − y)² over the batch's rows and members, one weight per row."""

    def __init__(self, y, w):  # one per row of the data
        self.y, self.w = y.astype(np.float64), w.astype(np.float64)

    def __call__(self, outputs, rows):
        out = outputs[:, :, 0].astype(np.float64)
        error, w = out - self.y[rows][:, None], self.w[rows][:, None]
        scale = 1.0 / (w.sum() * out.shape[1])
        return float(scale * (w * error**2).sum()), (2.0 * scale * w * error)[:, :, None].astype(outputs.dtype)


def repeated_rows(lo, hi, batch_size, repeats, seed):
    """Each epoch: the rows of [lo, hi) with ``repeats[i]`` copies of row lo + i, shuffled, in batches."""
    rows = np.repeat(np.arange(lo, hi), repeats)

    def plan(epoch):
        order = np.random.default_rng([seed, epoch]).permutation(rows)
        return [order[i : i + batch_size] for i in range(0, order.size, batch_size)]

    return plan


params = TabMParams(objective="mse", n_classes=1, batch_size=256)
model = train_tabm(
    params,
    data,
    rows=slice(lo, hi),
    backend="metal",
    loss=WeightedSquares(y, w),
    batches=repeated_rows(lo, hi, 256, repeats, seed=0),
)
```

idun ships neither a loss nor a plan: these are examples. `TabMClassifier`, `TabMRegressor` and idun_realmlp take
neither.

### Temporal encoder: `Sequences` and `TabMParams(encoder="gru")`

A row's history as a sequence of steps, oldest first, each step a row of a channel matrix; the encoder's output joins
the row's own inputs ([networks.md](../architecture/networks.md) §6):

```python
import numpy as np
from idun.data import lag_index
from idun.tabm import Sequences, TabMParams, train_tabm

lags = [3, 2, 1, 0]  # three steps back, then the row itself
index = lag_index(np.asarray(data.era), np.asarray(data.ticker), lags, step=7)  # (n_rows, 4) int32; -1: absent
seq = Sequences(channels, index)  # channels: (n_rows, C) float32, finite, e.g. a few rank-normalised features per row
params = TabMParams(encoder="gru")
model = train_tabm(params, data, rows=slice(0, train_hi), backend="metal", classes=labels, sequences=seq)
pred = model.predict(data, rows=slice(pred_lo, pred_hi), backend="metal", sequences=seq)
```

`Sequences(channels, index)`: frozen; `channels` `(m, C)` float32 (finite; a memory map stays one), `index`
`(n_rows, S)` int32 rows of `channels` (−1: the step is absent), both checked (`ValueError`). The channel matrix may
have a row per panel row, as above, or be a smaller table that rows share: with one row per date and a one-step index
of each row's date, every row of a date reads the same context. `gather(rows)` returns
`(len(rows), S, C + 1)` float32: each step's channels then a present flag, an absent step all zeros. Properties
`n_rows`, `steps` (S), `n_channels` (C), `inputs` (C + 1).

`TabMParams(encoder="gru", encoder_embed=32, encoder_hidden=64)` (each width 1–128; `encoder="none"`, the default, is
the network without an encoder): per step a Linear from the step's C + 1 inputs to
`encoder_embed` units and an exact GELU (x Φ(x)), then a GRU of `encoder_hidden` units over the steps (Torch's gates
r, z, n), from a zero state; its last state is prepended to the row's inputs, so block 0 reads `encoder_hidden +
n_inputs` inputs, and the encoder trains with the network (one AdamW group, no dropout in the encoder). Its weights
start as Torch initialises a Linear and a GRU, drawn from idun's generators (`idun.tabm.encoder`). On Metal the encoder
runs in float32 whatever `storage`, one thread per hidden unit and row; its gradients are summed in a fixed order, so
fits repeat their bits. The estimators do not take an encoder; `idun_numerai.upload` refuses a model with one.

`TabMParams(num_emb_type="pwl", num_emb_n_bins=48, d_embedding=16)` (`num_emb_type="none"`, the
default, feeds the inputs as they are): piecewise-linear numeric embeddings, rtdl's
`PiecewiseLinearEmbeddings` in its version "B" without the activation, as pytabkit's TabM builds them for
`num_emb_type="pwl"` (`idun.tabm.embeddings`). `train_tabm(..., embedded=F)` embeds the first F inputs (all of them by
default): each one's bin edges are the distinct quantiles 0, 1/T, ..., 1 (T = `num_emb_n_bins`) of its values on the
training rows (at most 50,000 of them, drawn with the seed, where pytabkit takes them all; a column with fewer than
two distinct values raises `ValueError`, as rtdl), its encoding the bins' clamped `(x − left) / width` laid out as
rtdl lays them, and its `d_embedding` values `x · a + c + encoding @ W` with `a`, `c` uniform on ±1/√d and `W` zero at
the start; block 0 reads the F · d embedded values, then the other inputs, and its `r` starts N(0, 1), one draw per
member and column shared by the column's d values, where it starts as random signs without embeddings (pytabkit's).
AdamW decays the embeddings as every other array (pytabkit leaves their linear term undecayed), and pytabkit's 2
blocks with embeddings are not imposed: `blocks` stays the caller's. On the GPUs the embeddings run in float32 in
their own kernels, their weight gradients summed over the batch's rows in order, so fits repeat their bits; not with
an encoder or `storage="bf16"` (`ValueError`). `TabMClassifier` and `TabMRegressor` take the three settings and embed
their numeric columns (the encoder's first ones), the one-hot blocks entering as they are.

`TabMParams(gradient_clipping_norm=0.0)` (pytabkit's): above 0, each step's gradients are scaled as
`torch.nn.utils.clip_grad_norm_` scales them before AdamW, by min(1, norm / (‖g‖ + 1e-6)) with ‖g‖ over every
parameter (the members' and the embeddings' or the encoder's too). On the GPUs the step's passes then run, the
gradients are read back and their norm is summed on the host in float64, array by array in a fixed order, and AdamW
takes the factor; a factor of 1 (the default 0 clips nothing; a norm above ‖g‖ neither) leaves every bit as unclipped.
The estimators take it too.

### `TabMModel`

| member | what |
|---|---|
| `params`, `weights`, `n_inputs`, `train_rows`, `report` | the configuration, `reference.Weights` (float32), the inputs it expects, the training rows `(lo, hi)`, a `FitReport` (per-epoch loss and seconds, total seconds) |
| `multi_target`, `targets` | experimental, not covered by this documentation |
| `sequence_inputs` | with a temporal encoder, the inputs per step of its sequences (C + 1); 0 without one |
| `predict(data, *, rows, backend, chunk=4096, sequences=None)` | expected class per row of a contiguous slice, `(n,)`; the members' mean for `"mse"`, the expected bin centre for `"hl_gauss"`; `sequences` for a model with an encoder |
| `predict_proba(data, *, rows, backend, chunk=4096, sequences=None)` | the members' mean softmax `(n, n_classes)` of a single-target classifier |
| `save(path)` / `TabMModel.load(path)` | uncompressed `.npz` without object arrays (format versions 1 and 2; a model with an encoder version 3, its meta carrying `sequence_inputs`; a model with embeddings version 4, its encoding in `bins_scale`, `bins_shift` and `bins_single` beside the weights); a file without an encoder or embeddings leaves their settings out when they are the defaults, as earlier versions wrote it; `load` reads all four and raises `ValueError` for anything else |

### `TabMParams`

Frozen, validated dataclass; the defaults are the `tabm` package's (0.0.3) `TabM.make` defaults and the paper's
training ([networks.md](../architecture/networks.md) §6): `members=32`, `blocks=3`,
`width=512`, `dropout=0.1`, `n_classes=2`, `learning_rate=2e-3`, `weight_decay=3e-4`, `beta1=0.9`, `beta2=0.999`,
`eps=1e-8`, `batch_size=256`, `epochs=50` (the estimators' too), `seed=0`, `storage="fp32"` (or `"bf16"`: the Metal backend stores
activations and their gradients in bfloat16; weights, AdamW and arithmetic stay fp32), `micro_batch=0` (0 ≤
`micro_batch` < `batch_size`, so a micro-batch splits the batch: the whole batch or more raises `ValueError`; > 0: the
Metal backend runs each batch as micro-batches of that many rows, the last one possibly smaller, with activation buffers for one micro-batch and the gradients summed over them in a fixed order before
the one AdamW step; same update up to float summation order, for about a third of the memory at 1,024 of 4,096
rows), `objective="cross_entropy"` (or `"mse"`: regression with `n_classes=1`, one output per member, the
loss the mean of `(o − y)²` over the B × k outputs, the prediction the members' mean; the targets come as float32,
a `MatrixData` with `n_classes` 1; the Metal kernels `mse_loss` and `predict_mean`; or `"hl_gauss"`, below),
`n_bins=64`, `hl_low=-4.0`, `hl_high=4.0`, `hl_sigma=0.09375`, `encoder="none"`, `encoder_embed=32`,
`encoder_hidden=64` (the temporal encoder, above), `lr_schedule="constant"` (or `"cosine"`, `"linear"`) and
`lr_warmup_steps=0`: the learning rate of each AdamW step, computed on the host
(`idun.tabm.params.scheduled_rate`): a linear warm-up over the first `lr_warmup_steps` steps, then the schedule over
the rest of the seeded plan's `epochs × ⌈rows / batch_size⌉` steps, cosine `(1 + cos πt) / 2` or linear `1 − t` of
`learning_rate`; the defaults leave every step at `learning_rate` itself (today's bits), and a schedule refuses
`train_tabm(batches=)`, whose steps it cannot count. The estimators take both. A bad value raises `ValueError`.
`outputs` (property) is the head's
outputs per member: `n_classes`, 1 for `"mse"`, `n_bins` for `"hl_gauss"`; `encoded` (property) the inputs the encoder
prepends, `encoder_hidden` or 0.

`objective="hl_gauss"` (Imani & White 2018; regression, `n_classes=1`): the target, clamped to `[hl_low, hl_high]`, is
labelled with the mass of N(target, `hl_sigma`) in each of `n_bins` uniform bins (a multiple of 8, at most 512),
renormalised over them; every member's `n_bins` outputs train on soft-label cross-entropy (the mean over the B × k
member outputs of −Σ p log softmax, gradient (softmax − p) / (B · k)), and the prediction is the expected bin centre
under the members' mean softmax. The defaults (64 bins on [−4, 4], σ 0.75 bin widths) suit a standardized target. On
Metal the head runs as `n_bins / 8` groups of 8 outputs; the kernels `hl_gauss_ce` and `predict_centre`.

### `idun.tabm.hl_gauss`

`labels(z, n_bins, low, high, sigma) -> NDArray[np.float64]`: the `(len(z), n_bins)` soft labels, each row summing to 1
(`cdf = Φ((edges − z) / σ)`, `p = diff(cdf) / (cdf[-1] − cdf[0])`, z clamped; Φ through numba's `erfc` in float64);
`ValueError` for a target that is not finite or σ ≤ 0. `edges(n_bins, low, high)`, `centres(n_bins, low, high)`: the
bins' `n_bins + 1` edges (`linspace`) and their midpoints.

### `idun.tabm.directions`: the directions a TabM is sensitive to (JARF)

The directions a trained TabM responds to most, for the trees in front of which they go (JARF, Rauniyar 2025;
`idun.jarf`'s feature front and `JARFRegressor`): the top eigenvectors of the expected gradient outer product (EGOP)
`H = E[g gᵀ]`, `g` the gradient of the predicted expected class with respect to the inputs.

```python
from idun.tabm import Directions, fit_directions

rows = np.random.default_rng(0).choice(train_hi, 10_000, replace=False)
found = fit_directions([tabm_a, tabm_b], data, rows=rows, k=16)  # single-target classifiers, no encoder
found.save("directions.npz")
found = Directions.load("directions.npz")
projected = found.project(x)  # (n, 16) float32 of a dense input matrix (n, n_inputs)
```

| name | what |
|---|---|
| `input_gradients(model, x)` | `(B, d)`: d(expected class)/d(inputs) of a `TabMModel` at `x` `(B, d)`, on the float reference in the dtype of the weights and `x`; the members' sum, since every member reads `x` |
| `egop(models, data, rows)` | `(d, d)` float64: the mean of `g gᵀ` over `rows` of `data` (a `PackedDataset` or `MatrixData`) and the models, summed in a fixed order (rows in chunks of `CHUNK` = 512, models in the given order), so it repeats its bits on one machine |
| `top_directions(h, k)` | the top `k` eigenvectors as columns `(d, k)`, each with its largest-magnitude entry positive, and every eigenvalue, largest first (`numpy.linalg.eigh`) |
| `sensitivity(h)` | the EGOP's diagonal: each input's mean squared gradient, a per-input ranking |
| `check_surrogate(model)` | `ValueError` unless the model is a single-target classifier (`cross_entropy`) without a temporal encoder |
| `model_digest(model)` | the sha256 of a model's weights, how `Directions.surrogates` names them |
| `fit_directions(models, data, *, rows, k=16)` | the whole step: `Directions(top_directions(egop(...)))` |
| `Directions(vectors, eigenvalues, surrogates)` | frozen: `vectors` `(d, k)` float32, `eigenvalues` `(d,)` float64 descending, `surrogates` the models' digests; `n_inputs`, `k`, `trace`, `top_share` (the share of the trace the `k` directions hold), `digest` (the sha256 of the vectors, the id a binning carries); `project(x)` `(n, k)` float32 in matmuls of `PROJECT_ROWS` = 50,000 rows (a fixed shape, so the float32 sums repeat); `save(path)` / `Directions.load(path)`: an uncompressed `.npz` without object arrays (`idun-directions` v1; anything else raises `ValueError`) |

`eigh` is LAPACK's and may differ in its last bits between machines, which is why a `Directions` is computed once and
stored with the model that uses it ([determinism.md](../architecture/determinism.md)).

### `idun.tabm.reference`

The float64/float32 NumPy oracle ([networks.md](../architecture/networks.md) §3): `init(params, n_inputs, targets=1,
sequence_inputs=0, bins=None) -> Weights` (`bins`: `idun.tabm.embeddings.fit_bins` of the embedded inputs), `dropout_masks(params, step, rows, dtype)`, `forward(weights, x, masks,
sequences=None) -> (logits, Cache)`,
`loss_and_logit_grad(logits, y)`, `host_loss(loss, outputs, rows)` (a caller's `Loss` called and checked: the value
as a float, the gradient in the outputs' shape and dtype, both finite), `backward(weights, cache, d_logits) -> Weights`,
`predict(logits, targets=None)`,
`label_matrix(y, n_rows, targets, n_classes)`, `adam_state(weights)`, `adamw_step(params, weights, grads, state)`.
With an encoder, `Weights.encoder` holds its arrays
(`idun.tabm.encoder.EncoderWeights`: `embed` (E, C), `embed_bias`, `w_ih` (3H, E), `w_hh` (3H, H), `b_ih`, `b_hh`),
appended to `Weights.arrays()`; `forward` takes the sequences `(B, S, C)` and `backward` carries block 0's input
gradient into the encoder. `idun.tabm.encoder` is the encoder's own reference: `init(params, inputs)`,
`forward(weights, x) -> (last state, EncoderCache)`, `backward(weights, cache, d_out) -> EncoderWeights`, `gelu`,
`gelu_grad`, `normal_cdf`, `sigmoid`.

### `idun.tabm.metal.MetalTabM`

`MetalTabM(params, n_inputs, max_batch, targets=1, sequence_shape=None, *, profile=False)`: one TabM network on the GPU,
fp32 (kernels in `packages/idun/src/idun/nn/kernels/metal/common.metal` and `tabm.metal`). It is
`idun.tabm.device.DeviceTabM` on a Metal device (`net.device`): the network's passes are written once against
`idun.device.Device`; with `profile=True` every pass runs and is timed on its own (`net.device.kernel_seconds`), the
results unchanged. Widths are padded to multiples of 64 and the
batch so that rows × k is one; padded entries stay zero. Matches `idun.tabm.reference` within float32 error (gradients
about 1e-7 relative) and gives identical bits run to run ([networks.md](../architecture/networks.md) §3). With
`params.micro_batch` between 1 and `max_batch - 1` the activation buffers hold one micro-batch (`rows_pad` rows, padded)
and every call splits its rows into micro-batches of `micro_batch` rows run in order in one command buffer: the first
stores each parameter gradient, later ones add to it, the logit gradient keeps the whole batch's scale 1 / (B · k) and
the dropout masks are keyed by the absolute row, so logits and the loss are bit-identical to the unsplit batch and the
gradients differ only by summation order. A call takes up to `capacity` rows (`max_batch` rounded up to whole
micro-batches; `rows_pad` without micro-batches). The GPU keeps the head transposed (`set_weights` and `weights`
convert). `idun.tabm.cuda.CudaTabM(params, n_inputs, max_batch,
targets=1, sequence_shape=None, *, profile=False, device=None)` is the same network on an NVIDIA GPU (`DeviceTabM` on
`idun.cuda.device.CudaDevice`, kernels `nn/kernels/cuda/common.cu` and `tabm.cu` of the Metal kernels' names, buffers
and launch shapes; `device` the GPU's ordinal, else `$IDUN_CUDA_DEVICE` or 0), with the same methods: within float32
error of the reference and identical bits run to run, not Metal's bits. `sequence_shape` `(S, C)`: the sequences of a
network with an encoder; every method below then takes them as `seq` `(B, S, C)` after its other arguments, and the encoder's
kernels (`seq_forward`, `seq_input_grad`, `seq_member_sum`, `seq_backward`, `seq_weight_grad`, `seq_column_partials`)
run in the same command buffer, per micro-batch.

| method | what |
|---|---|
| `set_weights(weights)` / `weights()` | copy `reference.Weights` in (resets AdamW) / out, unpadded |
| `forward(x, rows, step)` | logits `(B, k, c)`; `step=None` is inference, otherwise the dropout masks of that step |
| `backward(x, y, rows, step)` | forward and backward without an update (labels `y` `(B,)`); returns the mean loss; `gradients()` reads them |
| `train_step(x, y, rows, step)` | forward, loss, backward and AdamW in one command buffer; returns the mean loss |
| `loss_step(x, loss, rows, step)` | a training step under a caller's `Loss`: the forward pass, `loss(outputs, rows)` on the host with the outputs `(B, k, c)` in float32, then the backward pass with its gradient and AdamW in a second command buffer (a micro-batched network encodes each micro-batch's forward pass again); returns the loss it gave. One target; not for `"hl_gauss"` |
| `predict(x)` | expected class of the member-averaged softmax, in calls of `capacity` rows: `(B,)` |
| `close()` | release the GPU buffers |

`rows` are absolute panel row indices (they key the dropout masks); `x` is `(B, n_inputs)` from `idun.nn.inputs`.

## `idun.realmlp`: idun_realmlp ([design notes](../architecture/networks.md) §8)

RealMLP-TD (Holzmüller et al.) on Metal, with the float64 NumPy reference: `from idun.realmlp import RealMLPParams,
RealMLPModel, train_realmlp, train_realmlp_seeds, predict_mean`.

### `RealMLPClassifier`, `RealMLPRegressor`

As `TabMClassifier` and `TabMRegressor`, with `RealMLPParams`' fields. The defaults are pytabkit 1.7.3's `RealMLP_TD_Classifier` (as `RealMLPParams`'); without a validation set idun keeps the last epoch's weights, where pytabkit picks the
best epoch on 20 % of the rows held out: `fit(X, y, eval_set=(X_val, y_val))` keeps the best epoch's on those rows
(with `patience`, `best_epoch_` and `valid_score_`, as `TabMClassifier`). The regressor uses the classifier's block (parametric SELU, learning rate 0.04), not
`RealMLP_TD_Regressor`'s Mish at 0.2. With fewer training rows than `batch_size`, a batch holds every row
(`train_realmlp` itself needs at least `batch_size` rows, since it drops an epoch's last partial batch). RealMLP scales
its inputs itself (robust scaling, smooth clipping), so the encoder does not standardize; batches are built on the host
for a `MatrixData` on either backend. `RealMLPParams.objective` `"mse"` (with `n_classes=1`) is the regression loss
(mean of `(o − y)²`, gradient `2 (o − y) / (B · K)`; Metal kernels `rmlp_mse` and `rmlp_predict_value`); a regression
network takes host batches (`MetalRealMLP.bind` builds classification batches).

### `RealMLPParams`, `idun.realmlp.reference`

`RealMLPParams` (in `idun.realmlp.params`): frozen, validated; the defaults are pytabkit 1.7.3's
`RealMLP_TD_Classifier` defaults ([networks.md](../architecture/networks.md) §8):
3 × 256, 16 frequencies → 3 values per input, dropout 0.15, label smoothing 0.1, Adam (0.9, 0.95) lr 0.04 on coslog4,
weight decay 0.02 on flat_cos, lr factors scale 6, biases / activations / embedding 0.1, batch 256, 256 epochs, 2
classes, and `init_rows=0`: pytabkit's own preprocessing subsample for the table,
`idun.realmlp.train.pytabkit_init_rows` (`floor(2^28 / (62F + 1))` rows for F inputs, every training row of a table
with fewer; an explicit count takes that many). `lr_schedule` (`"coslog4"`, or `"cosine"` `(1 + cos πt) / 2` or
`"constant"`), `dropout_schedule` and `wd_schedule` (`"flat_cos"` or `"constant"`) are pytabkit's schedules as
parameters, the estimators' too; `reference.multipliers(params, t)` gives a step's three multipliers on every backend.
`act` is the hidden layers' activation f inside
the parametric activation `v + (f(v) − v) · a`: `"selu"` (default, the classifier's), `"mish"` (pytabkit's regressor's)
or `"relu"`, the estimators' parameter too; Mish in float32 on the GPUs is `v · n / (n + 2)` with `n = e^v (e^v + 2)`
(`v` above 20), within the networks' tolerance of the reference on Metal and CUDA, and SELU is the default. `hidden_sizes` (pytabkit's) gives each hidden layer's width, multiples of 64 on the GPUs, instead of
`blocks` × `hidden` (the default, empty, keeps them; not both). The
estimators take it too. Categorical embeddings: `train_realmlp(..., categories=(m_1,
..., m_G))` reads the last G inputs of a `MatrixData` as G categorical inputs' level codes (0 .. m_g − 1, as floats;
anything else raises `ValueError`, as a packed dataset with them does), bypassing the preprocessing, and embeds each
in `embedding_size` values (`RealMLPParams`, 8, pytabkit's) from a learned table, N(0, 1) at the start as pytabkit's;
block 0 reads them after the numeric inputs' embeddings, under the same front scale, and AdamW steps the table with
factors 1 and 1. On the GPUs the look-up and the table's gradient run in their own kernels, each gradient value a sum
over the batch's rows in order. A model with them is RealMLP's file format version 3 (`categories` in its meta, the
table after the other arrays). `RealMLPClassifier` and `RealMLPRegressor` take pytabkit's `max_one_hot_cat_size` and
`embedding_size`: a categorical column of at least `max_one_hot_cat_size` levels is embedded instead of one-hot
(pytabkit counts a missing level, so its 9 one-hot encodes up to 8 levels); the default, -1, one-hot encodes every
column (pytabkit's RealMLP-TD uses 9). `idun.realmlp.reference` is the float64 oracle: `fit_preprocess`, `preprocess`, `init(params, c,
n_targets=1, categories=())`, `schedules`, `dropout_masks`, `forward`, `loss_and_logit_grad(params, logits, y)`, `backward`,
`predict(logits, n_targets=None)`, `adam_state`, `adam_step`.
`idun.realmlp.metal.MetalRealMLP(params, n_inputs, max_batch, n_targets=None, *, profile=False)` runs it on the GPU
(fp32; `idun.realmlp.device.DeviceRealMLP` on a Metal device, `profile` as for `MetalTabM`;
`idun.realmlp.cuda.CudaRealMLP(params, n_inputs, max_batch, n_targets=None, *, profile=False, device=None)` the same
network on an NVIDIA GPU, as `CudaTabM` is TabM's), with
the same methods as `MetalTabM` plus the training fraction ``t`` of each step: `set_weights`, `weights`, `gradients`,
`forward(c, rows, step, t)`, `backward(c, y, rows, step, t)`, `train_step(c, y, rows, step, t)`, `predict(c)`,
`close`; inputs are the preprocessed rows (`idun.realmlp.reference.preprocess`). `bind(data, preprocess, classes=None, codes=None, *, targets=None)` wraps a
`PackedDataset`'s mapped arrays without a copy and copies its codes row-major on the host, as `shared_codes` does
(`inputs.codes_row_major`, 3.6 GB at v6), and places them; the labels come from `classes`, a compact `(n_rows, 1)` uint8 class array
(row r's class at `[r, 0]`, read on the GPU by kernel `rmlp_class_labels`; wrapped without a copy, so page-aligned
with contiguous rows: `inputs.class_array(n_rows, 1)` allocates one, filled with `inputs.MISSING_CLASS`), or
for a regression network (`objective="mse"`) from `targets`, an `(n_rows, 1)` float32 target array wrapped
the same way (`inputs.target_array`; kernel `rmlp_target_values` copies each value as is); the other kind raises
`ValueError`. On CUDA `bind` uploads these arrays instead (repacked through a transient host copy when their rows are
not 16-byte aligned) and keeps no host copy. A network bound without labels only predicts. Then batches are built on
the GPU from row indices alone: `gathered(rows) -> (c, y)` (for checks),
`train_step_rows(rows, step, t) -> loss`, `train_steps_rows(steps) -> losses` (an iterable of `(rows, step, t)`,
pipelined: the host encodes a step while the GPU runs the one before; the same bits as one call per step),
`predict_rows(lo, hi)`.

`idun.realmlp.train.train_realmlp(params, data, *, rows, backend, on_epoch=None, classes=None, snapshots=(),
on_snapshot=None, codes=None, targets=None, valid=None, patience=None, weight=None, categories=()) -> RealMLPModel`
(`categories`: above; `valid` and
`patience` as `train_tabm`'s, one target, host batches; `weight` as `train_tabm`'s, on host batches and
on the GPU's batches of a packed dataset alike): RealMLP's loop ([networks.md](../architecture/networks.md)
§8): preprocessing and initialisation on `params.init_rows` random training rows, a fresh permutation each epoch,
batches of `batch_size` with the last partial batch dropped, step k of K at training fraction k / K; backends `"metal"`,
`"cuda"` (built the same way, from the uploaded cache) and `"reference"`. `classes`: a `PackedDataset`'s labels, as for `train_tabm`; the Metal backend needs the array page-aligned (above). A wrong shape or dtype, a
missing `classes` for a packed dataset, or a training row's class at or above `n_classes` (255 included) raises
`ValueError` before any training. `targets`: for `objective="mse"`, a packed dataset's regression targets, as for
`train_tabm`; the Metal backend builds the batches on the GPU from the wrapped array (page-aligned, above), so a
regression fit on the cache needs no dense copy of the inputs, and the model predicts the output values. A non-finite
target on a training row raises `ValueError` before any training. `codes`: a `shared_codes` copy, which a GPU backend
places instead of making its own copy; the bits are the same. The GPU backends only (`ValueError` otherwise).

`train_realmlp_seeds(params, data, *, rows, backend, seeds, workers=None, classes=None, codes=None, targets=None) ->
list[RealMLPModel]` (multi-seed mode): one network per seed, each bit for bit `train_realmlp` with that seed, in `seeds`
order; up to `workers` train side by side on their own backends and threads, all reading one `shared_codes` copy. The
gain is modest for RealMLP (four seeds on an Apple M4: 1.16× with two networks in flight, 1.13× with three; its steps
already keep the GPU busy). Each network references about 4.2 GB on the GPU (`network_bytes`), so `workers=None` takes 2
or as many as fit the GPU's working set (1 on an 8 GB Apple M1, where two ran 16× slower than one), and a larger
explicit `workers` raises `ValueError`.
`workers` > 1 needs `backend="metal"`.
`predict_mean(models, data, *, rows, backend, codes=None)`: the bag's prediction, the mean of the networks'
predictions.

`idun.nn.inputs.dense_inputs(..., raw_codes=True)` gives RealMLP's inputs (codes 0..4);
`inputs.codes_row_major(data, out)` writes the obfuscated codes row-major into a zeroed `(n_rows, ld)` uint8 array, two
per byte (feature 2j in the low nibble of byte j).

## `idun.jarf`: idun_gbm behind idun_tabm's directions (JARF)

JARF (Rauniyar 2025): a trained network responds to combinations of inputs, a tree splits one input at a time; give the
trees the directions the network is most sensitive to (`idun.tabm.directions`) as extra features and an axis-aligned
split acts as an oblique one. The package composes the two learners, which stay independent of each other.

### The feature front: `extend`, `keep`, `Front`

A TabM's directions (`idun.tabm.Directions`) appended to a packed dataset as named features, and the obfuscated
features reduced to a subset; both return a `Front`, a `BinnedData` whose `binning` id carries what was done
(`<data.binning>+jarf:<digest>`, `+keep:<id>`), so a booster refuses data extended another way.

```python
from idun import gbm, jarf

rows = np.random.default_rng(0).choice(train_hi, 100_000, replace=False)  # the rows the edges come from
data_j = jarf.extend(data, found, edge_rows=rows)  # named features tabm_direction_0 .. tabm_direction_15 appended
booster = gbm.train(params, data_j, rows=slice(0, train_hi), backend="metal")
fewer = jarf.keep(data_j, chosen_obfuscated_names)  # the kept subset, in the data's order
```

| name | what |
|---|---|
| `extend(data, found, *, edge_rows, prefix="tabm_direction")` | every row's dense inputs (`idun.nn.inputs.dense_inputs`, the networks' view) projected on `found` (`project_rows`: chunks of `idun.tabm.directions.PROJECT_ROWS`, one matmul each), binned at the distinct quantiles `1..254 / 255` of `edge_rows`' projections (`quantile_cuts`; a value falls in the first bin whose edge is at or above it, `bin_projections`), appended after the named features with edges +inf padded to 255; a name clash with the data's features raises `ValueError` |
| `keep(data, obfuscated)` | the obfuscated features reduced to the names given, in the data's order (unknown or repeated names raise); the other columns are the data's own arrays, nothing copied but the kept codes |
| `Front` | the dataclass both return, with every `BinnedData` field |
| `direction_names(k, prefix)` | `("tabm_direction_0", ...)` |

The bins are this rule, not the packed cache's (`idun.data.binning.quantile_edges`).

### `JARFRegressor`

The whole method as one estimator under scikit-learn's conventions (`idun.base`), for any table:

```python
from idun.jarf import JARFRegressor

model = JARFRegressor(num_trees=500, learning_rate=0.05, k=16, n_surrogates=2, categorical=["sector"]).fit(X, y)
pred = model.predict(X_test)
model.save("jarf.npz")  # idun.load reads it back
```

`fit(X, y)`: fits `n_surrogates` `TabMClassifier`s on `y`'s classes (`quantile_classes(y, surrogate_classes)`: `y`'s
distinct values when there are at most `surrogate_classes` of them, else its quantile bins) with `surrogate_params`
(`TabMClassifier`'s keywords; its defaults otherwise) on `surrogate_backend` and seeds `seed`, `seed + 1`, ..., or
takes fitted ones (`surrogates=[...]`; their encoders must agree); computes the EGOP of every surrogate network on
`egop_rows` rows drawn with `seed` (`sensitivity_`, per encoder input) and its top `k` directions (`directions_`, a
`Directions`); appends the rows' projections to the table as numeric columns `tabm_direction_0 .. k-1`, keeping every
column or, with `keep=N`, the `N` non-categorical columns with the largest sensitivity (categorical columns are always
kept; `kept_`); fits a `GBMRegressor` (`gbm_`) with the first group of parameters, which are `GBMRegressor`'s own, on
that table (`categorical` and `codes` name columns of `X`; not among them: the binning settings `max_bin` and
`subsample_for_bin`, as JARF bins its front table itself, and `second_order`, `extra_trees` and
`feature_fraction_bynode`). `predict(X, per_seed=False)` encodes, projects and predicts.
`save` / `idun.load`: an `idun-model` v2 file with the encoder, the directions, the kept columns and the `GBMRegressor`'s
own file inside; `surrogates` is saved as None (the networks are not kept, the directions are what they gave).
`k` is capped at the encoder's inputs (a table of fewer columns than `k` gets as many directions as it has inputs).
Bad values raise `ValueError` (`k` below 1, `keep` outside `1..` the non-categorical columns, a column of `X` named
`tabm_direction_*`, unfitted surrogates). `n_features_in_`, `feature_names_in_` are `X`'s.

### `JARFModel`: the packed data path

A `Booster` trained on a `Front`, with the `Directions` and the kept obfuscated features (`None`: all) it was trained
behind, so new rows of the same packed dataset get the same front:

```python
from idun.jarf import JARFModel

model = JARFModel(booster, found, kept=None)
pred = model.predict(data, rows=slice(pred_lo, pred_hi), backend="metal")  # builds the front, predicts
model.save("jarf_r0_s0.npz")
model = JARFModel.load("jarf_r0_s0.npz")
```

| member | what |
|---|---|
| `booster`, `directions`, `kept`, `prefix` | the pieces (`prefix` the direction columns' name stem, `tabm_direction` by default); a booster without the `k` direction columns raises `ValueError` |
| `n_named` | the named features before the direction columns in the booster's features |
| `front(data)` | `data` extended with its projections binned at the booster's own edges for those columns (`apply`) and reduced to `kept` (`keep`); a dataset with another number of named features raises |
| `predict(data, *, rows, backend)` | `booster.predict(front(data), ...)` |
| `save(path)` / `JARFModel.load(path)` | an uncompressed `.npz` without object arrays (`idun-jarf` v1: the booster's own file inside, the directions, the kept names); anything else raises `ValueError` |

`apply(data, found, edges, n_bins, *, prefix)` is `extend` with the edges given (`(k, 255)`, +inf padded, as a booster
stores them) instead of fitted: `extend` fits them from `edge_rows` and calls it.

## `idun.nn`: what the networks share

`idun.nn.metal`: the kernel directory (`KERNELS`: `common.metal`, `tabm.metal`, `rmlp.metal`), the GEMM tile
(`TILE`, `BK`), the ctypes mirrors of common.metal's `GemmParams` and `AdamParams`, and `round_up`. `idun.nn.training`:
`epoch_order(seed, epoch, n)` (each epoch's permutation of the training rows), `FitReport` (per-epoch
losses and seconds; with a validation set its `valid_score` per epoch and `best_epoch`), `Validation` and `score`.
Neither learner imports the other; both import these.

`idun.nn.inputs.MatrixData(x, y, n_classes)`: the generic interface's input source, a finite float32
`(n_rows, n_inputs)` matrix with int64 classes `y` 0..n_classes − 1; `train_tabm`, `train_realmlp` and their models'
`predict` and `predict_proba` take it where they take the packed cache (`Data = PackedDataset | MatrixData`).
`TabMModel.predict_proba` and `RealMLPModel.predict_proba` return `(n, n_classes)` probabilities (single-target models)
from the backends' `inference_logits`. `idun.nn.training.softmax(logits)` is the float64 softmax over the last axis.
`idun.nn.table.TableEncoder(categorical=..., standardize=..., max_one_hot_levels=None)`: `fit(X)`, `transform(X)`
(float32 matrix: numeric columns in the input's order, then one one-hot block per categorical column, then one column
of level codes per categorical column of more than `max_one_hot_levels` levels), `n_inputs`,
`coded_levels` (the coded columns' level counts), `spec()` / `from_spec(spec)`.
`idun.base.Estimator`: `get_params`, `set_params` and the repr every estimator shares.

### `idun.nn.inputs`

| function | signature |
|---|---|
| `n_inputs` | `(data: PackedDataset) -> int`: 1,086 for the v6 cache |
| `dense_inputs` | `(data, rows) -> NDArray[np.float32]`: named factors (NaN → 0), codes as `(x − 2) / 2`, country one-hot |
| `labels` | `(data: MatrixData, rows) -> NDArray`: the matrix's own labels; `TypeError` for a packed dataset, whose classes are passed |
| `class_array` | `(n_rows, n_targets) -> NDArray[np.uint8]`: a page-aligned class array of that shape, one column per target (one for every use documented here), every entry `MISSING_CLASS` (Metal wraps it without a copy) |
| `class_labels` | `(classes, rows, n_classes) -> NDArray[np.int64]`: the classes of `rows` from a compact `(n_rows, 1)` uint8 class array, `(len(rows), 1)`; `ValueError` for another shape or dtype, or a class of `rows` at or above `n_classes` (`MISSING_CLASS` = 255 marks a row without that label) |
| `target_array` | `(n_rows, n_targets) -> NDArray[np.float32]`: a page-aligned float32 target array of that shape, one column per target (one for every use documented here), every entry NaN (Metal wraps it without a copy) |
| `checked_targets` | `(targets, n_rows, lo, hi) -> NDArray[np.float32]`: `targets` unchanged (no copy); `ValueError` unless it is an `(n_rows, 1)` float32 array whose rows `[lo, hi)` are finite (checked `CHECK_ROWS` = 2²⁰ rows at a time) |
| `check_label_source` | `(data, objective, classes, targets) -> None`: `ValueError` unless a `MatrixData` gets neither array and a packed dataset gets `classes` for `"cross_entropy"` or `targets` for `"mse"` |

### `idun_numerai.inputs`

Numerai's labels, kept out of idun: `classes(targets) -> NDArray[np.uint8]` returns the `(n_rows, 1)` class array
`rint(4 · y)` of a target `(n_rows,)` (NaN, a missing target, is `MISSING_CLASS`; any
other value off the 5 bins raises `ValueError`), page-aligned for Metal; `expected(proba)` is the expected class
`Σ_c c · p_c` of `(n, 5)` probabilities; `N_CLASSES` = 5.
`per_era_z(targets, era) -> NDArray[np.float32]` is a regression label, the per-era z: the `(n_rows, 1)` target array
of a target `(n_rows,)` standardised within each era (`(y − mean) / sd` in float64 over
the era's rows with a target, sd with ddof 0, 0 where the sd is 0; NaN stays NaN), for `train_tabm(..., targets=)` and
`train_realmlp(..., targets=)` with `objective="mse"`. `era` is one sorted value per row (a packed dataset's `era`;
unsorted raises `ValueError`). `era_bounds(era, n_rows)` returns the row bounds `[0, ..., n_rows]` of a sorted `era`'s
runs (`ValueError` for another length or an unsorted era).

### `idun_numerai.score`

Numerai's per-era scores in NumPy and SciPy, without pandas, as `numerai-tools` defines them; the tests check each
against `numerai-tools` per era (within 1e-12). `numerai-tools` is not a dependency.

| name | signature and meaning |
|---|---|
| `ncorr` | `(predictions, targets, neutralizers) -> float`: one era's neutral correlation, `numerai_tools.scoring.neutral_correlation`: the tie-kept ranked, gaussianized prediction without its least-squares fit on the `(n, m)` neutralizers and a constant, correlated with the centred target (through an SVD basis of the centred neutralizers, singular values at most `RCOND` = 1e-6 of the largest dropped) |
| `corr` | `(predictions, targets) -> float`: one era's correlation, `numerai_tools.scoring.numerai_corr`: the gaussianized prediction and the centred target, each to the power 1.5 with its sign, correlated |
| `gaussianized` | `(predictions) -> NDArray[np.float64]`: `Φ⁻¹((rank − ½) / n)`, ties averaged |
| `per_era` | `(score, era, *columns) -> NDArray[np.float64]`: `score` of each era's rows of the row-aligned `columns`, `era` sorted (a packed dataset's `era`) |

Each score raises `ValueError` for mismatched shapes, a value that is not finite, fewer than 2 rows, or a constant
prediction or target (the score is undefined there; `numerai-tools` asserts in `neutral_correlation` and gives NaN in `numerai_corr`).

### `idun_numerai.upload`

Numerai's model uploads run a cloudpickled `predict(live_features, live_benchmark_models)` on Linux with one CPU, in a
fixed environment without idun (numerai-predict's Python 3.12 image: NumPy 2.1, pandas, cloudpickle). The upload
function carries the models as arrays and predicts with NumPy alone.

```python
from idun_numerai import upload

predict = upload.predictor([booster, [tabm_a, tabm_b], realmlp], data)  # data: the PackedDataset; weights optional
upload.save(predict, "model.pkl")  # under Python 3.12, the runtime's
frame = predict(live_features, live_benchmark_models)  # one column, "prediction", in (0, 1]
raw = predict.raw(live_features)  # each entry's predictions before ranking
```

| name | signature and meaning |
|---|---|
| `predictor` | `(entries, data, *, weights=None) -> predict`: each entry a `Booster`, `TabMModel` or `RealMLPModel` trained on `data`, or a bag (a list of one kind, predicting its float64 mean as `predict_mean`); single-target networks only, and no TabM with a temporal encoder. The live features are mapped onto `data`'s features: named values binned with the booster's edges (NaN to the missing bin), codes exactly 0..4, the country one of `data`'s levels; anything else raises `ValueError`. idun_gbm's predictions equal `Booster.predict`'s bit for bit; the networks' are the `"reference"` backend's forward pass in float64 (equal up to float64 rounding, a test bounds them by 1e-6): the expected class, the output (`"mse"`) or an HL-Gauss TabM's expected bin centre. Entries are blended by rank: each ranked within the era (ties averaged, scaled to (0, 1]), the weighted mean ranked again |
| `save` | `(predict, path, *, python=RUNTIME_PYTHON) -> Path`: cloudpickle with `idun_numerai.upload` by value, so the file needs neither idun nor numba; `RuntimeError` unless this interpreter is `python` (3.12), since the pickle carries bytecode |
| `RUNTIME_PYTHON` | `(3, 12)` |

A `JARFModel` (`idun.jarf`), or a bag of them, is an entry like a `Booster`: the live features' dense inputs (as the
networks' are built) are projected on the stored vectors in NumPy (matmuls of 50,000 rows, as idun projects), appended
to the named values, the kept obfuscated features selected by name, and the trees walked as for a `Booster`; the
predictions equal `JARFModel.predict`'s bit for bit. A model whose booster does not read the dataset's features plus
its direction columns and kept set raises `ValueError`.

## `idun.data`: the packed dataset

Format and design: [data-layout.md](../architecture/data-layout.md). A packed dataset is a date-sorted panel on disk,
column-major and memory-mapped: 4-bit codes, raw numeric values and their bins per window, one categorical column, a
target, a time key and an entity id, and its walk-forward windows (`Refit`). Numerai's v6 data is written into it by
the port, `idun_numerai.data.build_cache`, from the parquet files `idun_numerai.download.download` fetches.

### `PackedWriter`

```python
with PackedWriter(
    out, n_rows=n, codes=code_names, numeric=numeric_names, categorical="sector", block_rows=131_072
) as writer:
    for codes, values in blocks:  # (len(codes), k) uint8 in 0..4 and (len(numeric), k) float32, in row order
        writer.add(codes, values)
    writer.end_features()
    writer.write_binnings(windows, seed=0, sample_rows=200_000)  # one binning per Refit, fitted on its training rows
    writer.write_rows(target=y, era=days, ticker=ids, tickers=names, country=codes, countries=levels)
    path = writer.finish({"source_dir": ..., "sources": ...})  # checksums, the manifest, <out>.partial -> <out>
```

- Rows are staged in column-major blocks and written with `pwrite` at each column's offset, so a table larger than
  memory streams through; `block_rows` changes speed and memory, never the bytes written.
- `FileExistsError` if `out` (or `out.partial`) exists; `ValueError` for a wrong row count or array shape. An exception
  inside the `with` block removes the partial directory.
- `Refit(date, train_hi, pred_lo, pred_hi)`: one window, training rows `[0, train_hi)` and predicted rows
  `[pred_lo, pred_hi)`.

### `idun_numerai.download.download`

`download(out_dir, *, files=SOURCE_FILES, version="v6.0", api=None) -> list[Path]` fetches Numerai's dataset files
through `numerapi` (no credentials for the public data) into one directory under their own names, the layout
`build_cache` reads; `files` defaults to its `train.parquet`, `validation.parquet` and `features.json`
(`SCORING_FILES` adds the `validation_neutralizer.parquet` of `score.ncorr`). Each file is written to `<name>.partial` and renamed
when complete. `ValueError` if Numerai lists no such file (nothing is downloaded), `FileExistsError` if `out_dir`
already holds one (a validation file grows every round, so an old copy is never reused silently). `api`: a
`numerapi.NumerAPI`, a new one when None.

### `idun_numerai.data.build_cache`

```python
def build_cache(
    v6_dir: str | Path,
    out_dir: str | Path,
    *,
    seed: int,
    target: str,
    refit_dates: Sequence[str],
    purge: int,
    sample_rows: int = 200_000,
    columns_per_read: int = 32,
    block_rows: int = 131_072,
) -> Path: ...
```

- Streams `train.parquet` then `validation.parquet` in record batches into a `PackedWriter`; never holds the panel in
  memory.
- Keeps every row with a value of `target` (a column of the files; `ValueError` if they have none), in date order,
  and the feature order named, obfuscated, country; the windows are
  `idun_numerai.refits.refit_table(eras, refit_dates, purge)`: each trains on the eras up to `purge` before its date
  and predicts from its date to the next. The target, the dates and the purge are the caller's, without defaults
 .
- `seed` seeds each refit's binning sample (`PCG64([seed, refit])`). `columns_per_read` and `block_rows` change speed and
  memory, never the bytes written.
- Raises `FileExistsError` if `out_dir` (or `out_dir.partial`) exists; `ValueError` when the input breaks an assumption
  (rows out of date order, an obfuscated value outside 0..4, a null key column); `CacheError` if a source file changes
  during the build. A failed build leaves nothing behind.

### `PackedDataset`

```python
class PackedDataset:
    @classmethod
    def open(cls, path: str | Path, *, binning: str, verify: bool = False) -> PackedDataset: ...

    path: Path
    n_rows: int
    binning: str  # the named-feature binning in use: "r0", "r1", ... (one per window, fitted on its training rows)
    seed: int
    named_features: tuple[str, ...]  # 22, sorted
    obfuscated_features: tuple[str, ...]  # 1,017, sorted
    feature_names: tuple[str, ...]  # named, obfuscated, "feature_country"
    countries: tuple[str, ...]  # sorted; `country` indexes into it
    categories: tuple[tuple[str, ...], ...]  # (countries,): BinnedData's one categorical column
    tickers: tuple[str, ...]  # sorted; `ticker` indexes into it (read on first use)
    refits: tuple[Refit, ...]

    target: NDArray[np.float32]  # (n_rows,) the target the dataset was built with
    era: NDArray[np.int32]  # (n_rows,) days since 1970-01-01
    ticker: NDArray[np.int32]  # (n_rows,)
    country: NDArray[np.uint8]  # (n_rows,)
    categorical: NDArray[np.uint8]  # (1, n_rows): `country` as a zero-stride view, nothing copied
    obfuscated: NDArray[np.uint8]  # (1017, ceil(n_rows / 2)): two rows per byte, low nibble first
    named_values: NDArray[np.float32]  # (22, n_rows) raw values, NaN kept
    named_bins: NDArray[np.uint8]  # (22, n_rows) bins under `binning`; 255 = missing
    named_edges: NDArray[np.float32]  # (22, 255) upper edges under `binning`, +inf padded
    named_n_bins: tuple[int, ...]  # real bins per named feature (≤ 255)

    def obfuscated_codes(self, feature: int, lo: int = 0, hi: int | None = None) -> NDArray[np.uint8]: ...
    def verify(self) -> None: ...
```

With `IDUN_CACHE_COPY=1`, `open` checks a copied cache (no source files on that machine) by the checksum of every
array file instead of by its sources ([data-layout.md](../architecture/data-layout.md)).

- Every array is a read-only `np.memmap` view; each column is contiguous. Nothing is loaded into memory up front.
- `open` raises `CacheError` when the cache is missing, of another format version, **stale** (a source file's size or
  modification time changed since the build), truncated, or lacks `binning`. It never rebuilds silently.
- `verify` (or `open(..., verify=True)`) recomputes the sha256 of every array file (a few seconds for a cache of millions of rows).
- `obfuscated_codes` unpacks rows `[lo, hi)` of one obfuscated feature to codes 0..4.

### `Refit`

Frozen dataclass `Refit(date: str, train_hi: int, pred_lo: int, pred_hi: int)`: train on rows `[0, train_hi)` and
predict rows `[pred_lo, pred_hi)` of the date-sorted panel. `PackedDataset.refits` holds the dataset's
windows.

### `lag_index`

`lag_index(time, group, lags, step) -> NDArray[np.int32]`: for a panel whose rows each have a time (an integer, such as
days) and a group (an integer id, such as a ticker), the `(n_rows, len(lags))` index of each row's row of the same
group at `time − lag · step` (lag 0: the row itself), `ABSENT` (−1) where there is none; a positive lag always points
to an earlier time. `ValueError` unless `time` and `group` are 1-D integer arrays of one length (below 2^31 rows) with
one row per (group, time) pair, the group ids fit in 32 bits and the times span less than 2^32, `step` is a positive
int and every lag a non-negative int. On Numerai's cache (`era`, `ticker`, `step=7`) four lags take about a second.

## `idun.metal.runtime`: Metal device runtime (internal-facing)

Used by the Metal backend. Public so that diagnostics and tests can probe the machine.

| name | signature | raises |
|---|---|---|
| `probe` | `() -> DeviceInfo` | `BackendUnavailableError` |
| `compile_check` | `(source: str) -> None` | `KernelCompileError` (with the compiler log), `BackendUnavailableError` |
| `source_prelude` | `() -> str`: text prepended to every kernel library | `BackendUnavailableError` |
| `library_path` | `() -> Path`: `$IDUN_METAL_LIB`, else the package's copy (installed by the macOS arm64 wheel, built by `make native` in a checkout; the pure wheel has none) | — |
| `side_by_side` | `(bytes_each: int) -> int`: how many jobs referencing `bytes_each` GPU bytes fit the recommended working set at once (≥ 1) | `ValueError` (bytes_each < 1), `BackendUnavailableError` |
| `workers_side_by_side` | `(requested: int \| None, default: int, bytes_each: int) -> int`: `requested`, or `default` capped by `side_by_side` when None (the seed bags' `workers`) | `ValueError` (more than fit) |
| `DeviceInfo` | frozen dataclass: name, threadgroup memory, working set, max buffer, threads per threadgroup, page size, unified memory, Apple7, Apple9, Metal 4 | — |
| `Context` | `()`: a device context. `load(source)`, `kernel(name) -> Kernel`, `alloc(nbytes) -> Buffer` (shared, zeroed), `wrap(array, nbytes) -> Buffer` (no copy; page-aligned), `batch(wait=True)` (context manager) or `begin()` / `dispatch(kernel, buffers, params, groups, threads)` / `commit() -> gpu seconds` (waits for every committed batch), `commit_async()` and `wait(keep=0) -> gpu seconds` (v5: the GPU runs a batch while the host encodes the next), `gpu_seconds`, `close()` (waits for committed batches) | `BackendUnavailableError`, `KernelCompileError`, `IdunError` (calls out of order, bad arguments, GPU failure) |
| `Buffer` | `view(dtype, shape) -> NDArray`: the buffer's memory as NumPy (unified memory, valid until the context closes) | `ValueError` |
| `Kernel` | frozen dataclass: name, `max_threads`, `simd_width` | — |

## `idun.errors`

| exception | raised when |
|---|---|
| `IdunError` | base class |
| `BackendUnavailableError` | the requested backend cannot run: wrong platform, missing native library (the pure wheel), a library that cannot load (macOS older than 15), a stale one, no device |
| `KernelCompileError` | a device kernel fails to compile; the message is the compiler log |
| `CacheError` | the packed data cache is missing, stale, truncated, corrupt or of another format version |
| `NotFittedError` | an estimator predicts or saves before `fit` (also a `ValueError` and an `AttributeError`) |

| warning | emitted when |
|---|---|
| `MultiTargetQualityWarning` (a `UserWarning`) | a model is asked to fit several targets at once (experimental, not covered by this documentation) |
