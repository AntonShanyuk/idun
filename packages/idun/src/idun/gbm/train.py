"""The boosting loop: host orchestration of the trees over a backend (docs/architecture/overview.md).

Per iteration: the bag (host, every ``bagging_freq`` iterations), gradients and their quantization (device), then one
tree, grown level by level (``growth="depthwise"``) or best-first (``"leafwise"``): histograms of the smaller child of
each split (device), the sibling by subtraction and the split search (host), the partition (device). Then exact leaf
sums of the gradients (device), leaf values (host), and the prediction update (device). A custom objective
(`idun.gbm.objective`) writes the gradients on the host instead, and its hessians if it has them. Determinism rules:
docs/architecture/determinism.md.
"""

import dataclasses
import inspect
import queue
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from idun import rng
from idun.cuda import runtime as cuda_runtime
from idun.gbm import sampling
from idun.gbm.backends import (
    Backend,
    CudaBackend,
    LeafSplit,
    LevelSplits,
    Lockstep,
    ManyBackend,
    get_backend,
    get_many,
)
from idun.gbm.booster import Booster, FitReport
from idun.gbm.evaluation import Callback, Evaluator, Metric, Validation
from idun.gbm.features import BinnedData, FeatureLayout
from idun.gbm.objective import MulticlassObjective, NewtonObjective, Objective, Softmax
from idun.gbm.params import NO_DEPTH_LIMIT, Params
from idun.gbm.split import NUMERIC, NodeSearch, Split, best_splits
from idun.gbm.tree import Forest, NodeForest, Trees, concatenate, empty_trees
from idun.metal import runtime

FIXED_POINT_BOUND = 2**31  # max|g| · rows must stay below this: Σ |g| · 2^32 < 2^63 (determinism.md §5)

AnyObjective = Objective | NewtonObjective


def train(
    params: Params,
    data: BinnedData,
    *,
    rows: slice,
    backend: str | Backend | ManyBackend,
    y: NDArray[np.float32] | None = None,
    objective: AnyObjective | None = None,
    valid: Sequence[Validation] = (),
    metric: str | Metric = "l2",
    early_stopping_rounds: int | None = None,
    callback: Callback | None = None,
    init_model: Booster | None = None,
    boost_from_average: bool = False,
    weight: NDArray[np.float32] | None = None,
) -> Booster:
    """Fit ``params.num_trees`` trees (``params.growth``) on rows ``rows`` of ``data`` with backend ``backend``.

    Parameters
    ----------
    params
        Training parameters.
    data
        The binned panel (`idun.data.PackedDataset`).
    rows
        A contiguous training range, ``slice(lo, hi)``.
    backend
        ``"cpu"``, ``"metal"`` or ``"cuda"``, or a backend instance (for diagnostics, e.g.
        ``MetalBackend(profile=True)``; ``CudaBackend(device=1)`` for another GPU than the first).
        Required: nothing is selected implicitly, and an unavailable backend raises.
    y
        Targets of the training rows; ``data.target[rows]`` by default.
    objective
        A custom objective (`Objective`, or `NewtonObjective` with its own hessians): it writes every training row's
        gradient each iteration, and both the splits and the leaf values follow it. L2 (``g = pred − y``) by default.
    valid
        Validation sets (`Validation`: rows of a panel binned as ``data``), each scored with ``metric`` after every
        tree; the scores are the model's `Booster.evals`. Scoring runs on the host and gives the same numbers on
        every backend (`idun.gbm.evaluation`).
    metric
        ``"l2"``, ``"rmse"``, ``"l1"`` (LightGBM's names) or a `Metric`.
    early_stopping_rounds
        Stop when the first validation set's score has not improved for this many trees, and keep the trees up to
        the best (`Booster.best_iteration`): the model is then the model of that many trees, bit for bit.
    callback
        Called after every tree with an `Evaluation` (the scores so far); returning True stops the fit there.
    init_model
        Continue this model: the fit starts from its predictions, its trees come first in the result, and the new
        trees draw their bags, feature samples and rounding as trees ``init_model.n_trees`` and on, so a model of
        ``a`` trees continued for ``b`` is the model of ``a + b`` trees, bit for bit. It must have this data's
        binning and features and ``params``' tree shape (growth, ``max_depth``, ``max_leaves``).
    boost_from_average
        LightGBM's: start from the mean of ``y`` and fold it into the first tree's leaves (with an `Objective`, its
        ``start`` if it has one). Not with ``init_model``.
    weight
        One non-negative weight per training row (LightGBM's ``weight``): each row's gradient and hessian are
        multiplied by it (the built-in L2's and an `Objective`'s hessian is 1), so the fit takes second-order splits
        and the leaves are ``−Σ w g / (Σ w h + λ)``; ``min_data_in_leaf`` still counts rows. With
        ``boost_from_average`` the start is the weighted mean (an objective's ``start(y, weight)``). All weights 1
        give the unweighted model, bit for bit.
    """
    evaluator = None
    if valid or callback is not None or early_stopping_rounds is not None:
        evaluator = Evaluator(valid, data.binning, metric, early_stopping_rounds, callback)
    lo, hi = _bounds(rows, data.n_rows)
    targets = np.ascontiguousarray(data.target[lo:hi] if y is None else y, dtype=np.float32)
    if targets.shape != (hi - lo,):
        raise ValueError(f"y must hold one float32 per training row ({hi - lo}), got shape {targets.shape}")
    weights = _weights(weight, hi - lo)
    start = _start(params, data, lo, hi, targets, objective, init_model, boost_from_average, evaluator, weights)
    if isinstance(backend, str):
        device: ManyBackend = Lockstep([get_backend(backend)])
    elif hasattr(backend, "k"):  # already a K-model backend (K = 1 here)
        device = cast("ManyBackend", backend)
    else:
        device = Lockstep([backend])
    objectives = _weighted(_objectives(objective, 1), weights, 1)
    return _fit_group(params, data, lo, hi, targets[None], device, objectives, evaluator, start)[0]


@dataclass(frozen=True)
class _Start:
    """Where a fit starts.

    The predictions it starts from, the index in the model of its first tree, the trees before it (continued
    training) and the constant folded into its first tree (``boost_from_average``).
    """

    pred: NDArray[np.float32] | None = None
    first_tree: int = 0
    before: Booster | None = None
    bias: np.float32 | NDArray[np.float32] | None = None  # one for every model, or each model's


def _start(
    params: Params,
    data: BinnedData,
    lo: int,
    hi: int,
    targets: NDArray[np.float32],
    objective: AnyObjective | None,
    init_model: Booster | None,
    boost_from_average: bool,  # noqa: FBT001
    evaluator: Evaluator | None,
    weights: NDArray[np.float32] | None = None,
) -> _Start:
    """Return the fit's start and set the validation sets' starting predictions to match."""
    if init_model is not None:
        if boost_from_average:
            raise ValueError("boost_from_average starts a new model: not with init_model")
        if (init_model.params.growth, init_model.params.max_depth, init_model.params.max_leaves) != (
            params.growth,
            params.max_depth,
            params.max_leaves,
        ):
            raise ValueError("init_model's trees have another shape (growth, max_depth, max_leaves) than params'")
        pred = init_model.predict(data, rows=slice(lo, hi), backend="cpu")  # the fit's own running predictions
        if evaluator is not None:
            evaluator.start_from(init_model)
        return _Start(pred=pred, first_tree=init_model.n_trees, before=init_model)
    if not boost_from_average:
        return _Start()
    if objective is not None and not hasattr(objective, "start"):
        raise ValueError("boost_from_average with an objective needs its start(y) (LogLoss has one)")
    if weights is None:
        bias = np.float32(np.mean(targets, dtype=np.float64) if objective is None else objective.start(targets))
    elif objective is None:
        bias = np.float32(np.average(targets.astype(np.float64), weights=weights.astype(np.float64)))
    else:
        if "weight" not in inspect.signature(objective.start).parameters:
            raise ValueError("boost_from_average with weights needs the objective's start(y, weight)")
        bias = np.float32(objective.start(targets, weight=weights))
    return _Start(pred=np.full(hi - lo, bias, np.float32), bias=bias)


def train_many(
    params: Params,
    data: BinnedData,
    *,
    rows: slice,
    backend: str,
    ys: Sequence[NDArray[np.float32]],
    seeds: Sequence[int] | None = None,
    objective: AnyObjective | Sequence[AnyObjective] | None = None,
    weight: NDArray[np.float32] | None = None,
) -> list[Booster]:
    """Fit one model per (seed, target), sharing work across the targets of each seed.

    The models of one seed share the bag, the feature sample and the quantization draws, so they run in lockstep: one
    pass over the feature columns serves every target. Each model is identical to the one `train` fits with
    ``params`` (seed replaced) and that target. Returns the models seed-major: ``[(s0, y0), (s0, y1), ..., (s1, y0)]``.
    Every model of a group carries the group's `FitReport`.

    ``objective``: a custom objective for every target, or one per target in ``ys`` order (`train`'s ``objective``);
    each iteration it is called once per target, in order, with that target's predictions and labels. ``weight``:
    `train`'s, the same rows' weights for every target.
    """
    lo, hi = _bounds(rows, data.n_rows)
    targets = np.ascontiguousarray(np.stack([np.asarray(y, np.float32) for y in ys]))
    if targets.ndim != 2 or targets.shape[1] != hi - lo:
        raise ValueError(f"each target must hold one float32 per training row ({hi - lo})")
    k, weights = targets.shape[0], _weights(weight, hi - lo)
    models = []
    for seed in [params.seed] if seeds is None else seeds:
        group = dataclasses.replace(params, seed=seed)
        objectives = _weighted(_objectives(objective, k), weights, k)
        models += _fit_group(group, data, lo, hi, targets, get_many(backend, k), objectives)
    return models


def train_multiclass(
    params: Params,
    data: BinnedData,
    *,
    rows: slice,
    backend: str,
    y: NDArray[Any],
    n_classes: int,
    objective: MulticlassObjective | None = None,
    valid: Sequence[Validation] = (),
    metric: str | Metric = "multi_logloss",
    early_stopping_rounds: int | None = None,
    callback: Callback | None = None,
    boost_from_average: bool = False,
    weight: NDArray[np.float32] | None = None,
) -> list[Booster]:
    """Fit a multi-class model (LightGBM's ``multiclass``): each iteration one tree per class; K Boosters, class order.

    The K trees of an iteration share the bag, the feature sample and the quantization draws (one lockstep group, as
    LightGBM's classes do), and ``objective`` (`Softmax` by default) writes every class's gradients and hessians from
    all K classes' scores: second-order splits and Newton leaves. ``y`` holds class ids ``0 .. n_classes − 1``; a
    row's class probabilities are the softmax of the K Boosters' raw predictions. ``valid`` (each `Validation`'s ``y``
    class ids too), ``metric`` (``"multi_logloss"`` or ``"multi_error"``, or a `Metric` of the ``(K, n)`` scores),
    ``early_stopping_rounds`` and ``callback`` are `train`'s over the K classes' scores (the classes stop together);
    ``boost_from_average`` starts each class at the objective's ``start``, the log of its share of the labels
    (LightGBM's softmax starts at 0, the default); ``weight`` is `train`'s.
    """
    lo, hi = _bounds(rows, data.n_rows)
    labels = np.ascontiguousarray(y, dtype=np.float32)
    if labels.shape != (hi - lo,):
        raise ValueError(f"y must hold one class id per training row ({hi - lo}), got shape {labels.shape}")
    if not (np.isfinite(labels).all() and np.array_equal(labels, np.round(labels))) or not (
        labels.size and labels.min() >= 0 and labels.max() < n_classes
    ):
        raise ValueError(f"y must hold class ids 0 .. {n_classes - 1}")
    objective = Softmax(n_classes) if objective is None else objective
    if objective.n_classes != n_classes:
        raise ValueError(f"the objective has {objective.n_classes} classes, n_classes is {n_classes}")
    weights = _weights(weight, hi - lo)
    evaluator = None
    if valid or callback is not None or early_stopping_rounds is not None:
        evaluator = Evaluator(valid, data.binning, metric, early_stopping_rounds, callback, k=n_classes)
    targets = np.stack([(labels == c).astype(np.float32) for c in range(n_classes)])  # each class's model: one-hot
    start = _Start()
    if boost_from_average:
        begin = getattr(objective, "start", None)
        if begin is None:
            raise ValueError("boost_from_average needs the objective's start(y, weight) (Softmax has one)")
        bias = np.asarray(begin(labels, weight=weights), np.float32)
        start = _Start(pred=np.repeat(bias[:, None], hi - lo, axis=1), bias=bias)
    device = get_many(backend, n_classes)
    return _fit_group(
        params, data, lo, hi, targets, device, None, evaluator, start, _Coupled(objective, labels, weights)
    )


def _objectives(objective: AnyObjective | Sequence[AnyObjective] | None, k: int) -> list[AnyObjective] | None:
    """Return the K models' objectives (one given for all, or one each), or None for the built-in L2."""
    if objective is None:
        return None
    objectives = list(objective) if isinstance(objective, Sequence) else [objective] * k
    if len(objectives) != k:
        raise ValueError(f"{len(objectives)} objectives for {k} targets")
    for one in objectives:
        if getattr(one, "constant_hessian", None) not in (True, False) or not callable(getattr(one, "gradients", None)):
            raise ValueError(
                "an objective needs constant_hessian (True, or False for one that writes hessians) and gradients()"
            )
    return objectives


def _weights(weight: NDArray[np.float32] | None, n: int) -> NDArray[np.float32] | None:
    """Return the training rows' weights as float32, checked: finite, non-negative and not all 0."""
    if weight is None:
        return None
    weights = np.ascontiguousarray(weight, dtype=np.float32)
    if weights.shape != (n,):
        raise ValueError(f"weight must hold one float32 per training row ({n}), got shape {weights.shape}")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    if not (weights > 0).any():
        raise ValueError("every sample weight is zero")
    return weights


class _Weighted:
    """One model's objective under row weights, as a `NewtonObjective`: ``g · w`` and ``h · w`` (LightGBM's).

    The built-in L2 (``base`` None: ``g = pred − y``, as the device computes it) and an `Objective` have ``h = 1``,
    so their weighted hessians are the weights. Products in float32.
    """

    constant_hessian: Literal[False] = False

    def __init__(self, base: AnyObjective | None, weight: NDArray[np.float32]) -> None:
        self.base, self.weight = base, weight

    def gradients(
        self, pred: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        """Write the base objective's gradients and hessians times the weights."""
        if self.base is None:
            np.subtract(pred, y, out=grad)
            hess.fill(1.0)
        elif self.base.constant_hessian is True:
            self.base.gradients(pred, y, grad)
            hess.fill(1.0)
        else:
            self.base.gradients(pred, y, grad, hess)
        np.multiply(grad, self.weight, out=grad)
        np.multiply(hess, self.weight, out=hess)


def _weighted(
    objectives: list[AnyObjective] | None, weights: NDArray[np.float32] | None, k: int
) -> list[AnyObjective] | None:
    """Return the K models' objectives under ``weights`` (unchanged without them)."""
    if weights is None:
        return objectives
    return [_Weighted(None if objectives is None else objectives[model], weights) for model in range(k)]


SEED_WORKERS = 3  # fits in flight in train_seeds: an Apple M4's GPU fills at about three


def train_seeds(
    params: Params,
    data: BinnedData,
    *,
    rows: slice,
    backend: str,
    seeds: Sequence[int],
    y: NDArray[np.float32] | None = None,
    workers: int | None = None,
    objective: AnyObjective | None = None,
    valid: Sequence[Validation] = (),
    metric: str | Metric = "l2",
    early_stopping_rounds: int | None = None,
    callback: Callback | None = None,
    boost_from_average: bool = False,
    weight: NDArray[np.float32] | None = None,
) -> list[Booster]:
    """Fit a seed bag: one model per seed in ``seeds``, each identical to `train` with ``params`` and that seed.

    Seeds share no arithmetic (each draws its own bag and feature samples), but one Metal fit leaves much of the GPU
    idle, so up to ``workers`` fits run side by side, each on its own backend and thread: on an Apple M4 three at a time
    gave about 2.4x the throughput of one after another. The models come back in ``seeds`` order, bit for bit as
    separate fits. ``workers`` None takes `SEED_WORKERS`, or fewer when the GPU's working set cannot hold them
    (`runtime.side_by_side`: one on an 8 GB M1, where fits side by side thrash); a larger explicit ``workers`` raises
    ``ValueError``. ``workers`` > 1 needs a GPU backend (the CPU backend's parallel loops take turns, so its fits gain
    nothing side by side). With ``backend="cuda"``, ``workers`` None runs one fit on each GPU at a time (on two Tesla
    T4s a bag of four seeds took 30 s against 54 s one after another; two fits a GPU gained nothing), and fits side by
    side are spread over the GPUs (`idun.cuda.runtime.device_ordinals`: every GPU the driver lists, or the one
    ``$IDUN_CUDA_DEVICE`` names). A bag predicts the mean of its models' raw predictions (`predict_mean`).

    ``objective``: a custom objective for every seed (`train`'s ``objective``). Fits side by side call it from their
    own threads at the same time, so it must keep no state between calls, or the call must pass ``workers=1``.
    ``valid``, ``metric``, ``early_stopping_rounds``, ``callback``, ``boost_from_average`` and ``weight`` are
    `train`'s, for each seed's fit (each stops on its own); a callback is called from the fits' threads.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct and at least one, got {list(seeds)}")
    if workers is not None and (workers < 1 or (workers > 1 and backend not in ("metal", "cuda"))):
        raise ValueError(f"workers must be 1, or > 1 with backend 'metal' or 'cuda'; got {workers} with {backend!r}")
    if backend == "metal":
        workers = runtime.workers_side_by_side(workers, SEED_WORKERS, fit_bytes(data))
    elif backend == "cuda" and workers is None:
        workers = len(cuda_runtime.device_ordinals())  # one fit a GPU
    workers = 1 if workers is None else workers
    spread = backend == "cuda" and workers > 1
    gpus: queue.SimpleQueue[int] = queue.SimpleQueue()  # CUDA, side by side: a GPU for each fit that may run
    if spread:
        ordinals = cuda_runtime.device_ordinals()
        for worker in range(workers):
            gpus.put(ordinals[worker % len(ordinals)])

    def fit(seed: int) -> Booster:
        one = dataclasses.replace(params, seed=seed)
        if not spread:
            return train(one, data, rows=rows, backend=backend, y=y, objective=objective, valid=valid,
                         metric=metric, early_stopping_rounds=early_stopping_rounds, callback=callback,
                         boost_from_average=boost_from_average, weight=weight)  # fmt: skip
        gpu = gpus.get()
        try:
            device = CudaBackend(device=gpu)
            try:
                return train(one, data, rows=rows, backend=device, y=y, objective=objective, valid=valid,
                             metric=metric, early_stopping_rounds=early_stopping_rounds, callback=callback,
                             boost_from_average=boost_from_average, weight=weight)  # fmt: skip
            finally:
                device.close()  # the GPU's memory is free before the next fit takes the GPU
        finally:
            gpus.put(gpu)

    if workers == 1:
        return [fit(seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=min(workers, len(seeds))) as pool:
        return list(pool.map(fit, seeds))


def fit_bytes(data: BinnedData) -> int:
    """Estimate the bytes one Metal fit references on the GPU: the panel's binned arrays and 32 bytes per row."""
    arrays = (getattr(data, name, None) for name in ("obfuscated", "named_bins", "categorical"))
    return sum(np.asarray(a).nbytes for a in arrays if a is not None) + 32 * data.n_rows


def predict_mean(models: Sequence[Booster], data: BinnedData, *, rows: slice, backend: str) -> NDArray[np.float32]:
    """Return a bag's prediction: the mean of ``models``' raw predictions of ``rows`` (float64 sum in order)."""
    if not models:
        raise ValueError("a bag needs at least one model")
    total = np.zeros(len(range(*rows.indices(data.n_rows))), np.float64)
    for model in models:
        total += model.predict(data, rows=rows, backend=backend)
    return (total / len(models)).astype(np.float32)


def _fit_group(
    params: Params,
    data: BinnedData,
    lo: int,
    hi: int,
    targets: NDArray[np.float32],
    device: ManyBackend,
    objectives: list[AnyObjective] | None,
    evaluator: Evaluator | None = None,
    start: "_Start | None" = None,
    coupled: "_Coupled | None" = None,
) -> list[Booster]:
    """Fit the K models of one seed in lockstep on ``device``; one Booster per row of ``targets``.

    ``evaluator`` (one model, or a multi-class fit's K) scores each tree and may stop the fit early; the models keep
    what it says. ``coupled``: a multi-class fit's objective, which reads every model's predictions.
    """
    if evaluator is not None and evaluator.k != targets.shape[0]:
        raise ValueError("validation sets, early stopping and callbacks fit one model at a time (train, train_seeds)")
    newton = [coupled is not None or (objectives is not None and objectives[model].constant_hessian is False)
              for model in range(len(targets))]  # fmt: skip  # Newton leaves
    second = [on and params.second_order for on in newton]  # second-order splits: opt-in (first-order by default)
    layout = FeatureLayout.of(data)
    timer = _Timer()
    start = _Start() if start is None else start
    with timer("upload"):
        device.start(data, layout, lo, targets)
        if start.pred is not None:
            device.set_predictions(np.broadcast_to(start.pred, targets.shape).astype(np.float32))
    k = targets.shape[0]
    first, freq = start.first_tree, params.bagging_freq
    forests = [
        empty_trees(params.num_trees, leafwise=params.leafwise, max_depth=params.max_depth,
                    max_leaves=params.max_leaves)
        for _ in range(k)
    ]  # fmt: skip
    bag = np.ones(hi - lo, np.uint8)
    with ThreadPoolExecutor(max_workers=1) as pool:
        # The bag is a pure function of (seed, iteration): the next one is drawn on a worker thread while the device
        # grows the current trees (NumPy releases the GIL while filling). Same bags, same trees, less wall-clock.
        end = first + params.num_trees
        next_bag = _draw_bag(pool, params, hi - lo, first // freq * freq if freq else 0, end)
        for iteration in range(params.num_trees):
            tree = first + iteration  # the tree's index in the model: its draws are those of a fit from scratch
            if next_bag is not None and (iteration == 0 or tree % freq == 0):
                bag = next_bag.result()
                next_bag = _draw_bag(pool, params, hi - lo, (tree // freq + 1) * freq, end)
            with timer("gradients"):
                scales, inv_scales, hessian_scales, inv_hessians = _gradients(
                    device, targets, objectives, params.grad_quant_bins, coupled
                )
                key = rng.iteration_key(params.seed, tree)
                device.quantize(bag, key, inv_scales, params.grad_quant_bins // 2)
                if any(second):  # second-order splits: the hessians rounded with the gradients' draws
                    device.quantize_hessians(key, inv_hessians, params.grad_quant_bins // 2)
            features = sampling.feature_sample(layout.n_features, params.feature_fraction, seed=params.seed, tree=tree)
            quantized = _Quantized(scales, [float(h) if on else None for h, on in zip(hessian_scales, second,
                                                                                        strict=True)])  # fmt: skip
            draws = _Draws(params, tree, features, layout.real_bins[features])
            if params.leafwise:
                _grow_leafwise(device, cast("list[NodeForest]", forests), iteration, features, layout, params,
                               quantized, timer, draws)  # fmt: skip
            else:
                _grow(device, cast("list[Forest]", forests), iteration, features, layout, params, quantized, timer,
                      draws)  # fmt: skip
            with timer("leaves"):
                _leaves(device, forests, iteration, params, newton, start.bias if iteration == 0 else None)
            if evaluator is not None:
                with timer("evaluation"):
                    stop = evaluator.after(iteration, forests)
                if stop:
                    trained = iteration + 1
                    break
        else:
            trained = params.num_trees
    return _models(params, data, lo, hi, device, forests, timer, evaluator, start, trained)


def _leaves(
    device: ManyBackend,
    forests: Sequence[Trees],
    iteration: int,
    params: Params,
    newton: list[bool],
    bias: np.float32 | NDArray[np.float32] | None,
) -> None:
    """Set tree ``iteration``'s leaf values from the device's sums, add them to the predictions, fold in ``bias``."""
    n_leaves = forests[0].leaf_value.shape[1]
    sums, counts = device.leaf_sums(n_leaves)
    hessians = device.hessian_sums(n_leaves) if any(newton) else None
    for model, forest in enumerate(forests):
        if hessians is not None and newton[model]:
            forest.leaf_value[iteration] = newton_leaf_values(
                sums[model], hessians[model], counts[model], params.lambda_l2, params.learning_rate,
                params.lambda_l1, params.max_delta_step,
            )  # fmt: skip
        else:
            forest.leaf_value[iteration] = leaf_values(
                sums[model], counts[model], params.lambda_l2, params.learning_rate, params.lambda_l1,
                params.max_delta_step,
            )  # fmt: skip
    device.update_predictions(np.stack([forest.leaf_value[iteration] for forest in forests]))
    for model, forest in enumerate(forests):
        if forest.leaf_count is not None:  # in-bag rows per leaf: the covers of TreeSHAP's contributions
            forest.leaf_count[iteration] = counts[model]
    if bias is not None:  # LightGBM's AddBias: the start the predictions began at is the first tree's
        for forest, own in zip(forests, np.broadcast_to(np.asarray(bias, np.float32), (len(forests),)), strict=True):
            forest.leaf_value[iteration] = (forest.leaf_value[iteration] + own).astype(np.float32)


def _models(
    params: Params,
    data: BinnedData,
    lo: int,
    hi: int,
    device: ManyBackend,
    forests: list[Trees],
    timer: "_Timer",
    evaluator: Evaluator | None,
    start: "_Start",
    trained: int,
) -> list[Booster]:
    """Return one Booster per forest: the trees kept (up to early stopping's best), after continued training's."""
    first = start.first_tree
    kept = params.num_trees if evaluator is None else evaluator.kept(trained)
    if kept < params.num_trees:  # stopped early: the model of the trees kept
        params = dataclasses.replace(params, num_trees=kept)
        forests = [forest.head(kept) for forest in forests]
    if start.before is not None:  # continued: the model's trees, then the new ones
        params = dataclasses.replace(params, num_trees=first + kept)
        forests = [concatenate(start.before.forest, forest) for forest in forests]
    report = FitReport(
        seconds=dict(timer.seconds),
        backend=device.name,
        train_rows=(lo, hi),
        gpu_seconds=getattr(device, "gpu_seconds", None),
        kernel_seconds=dict(getattr(device, "kernel_seconds", {})),
    )
    return [
        Booster(
            params,
            forest,
            binning=data.binning,
            feature_names=data.feature_names,
            categories=data.categories,
            named_edges=np.array(data.named_edges, np.float32),
            named_n_bins=np.array(data.named_n_bins, np.uint16),
            train_rows=(lo, hi),
            backend=device.name,
            fit_report=report,
            best_iteration=None
            if evaluator is None or evaluator.early_stopping_rounds is None
            else first + evaluator.best_iteration,
            evals=None if evaluator is None else evaluator.history,
        )
        for forest in forests
    ]


def leaf_values(
    sums: NDArray[np.int64],
    counts: NDArray[np.int64],
    lambda_l2: float,
    learning_rate: float,
    lambda_l1: float = 0.0,
    max_delta_step: float = 0.0,
) -> NDArray[np.float32]:
    """``float32(−(S · 2^-32) / (n + λ) · lr)`` per leaf, computed in float64; 0 for a leaf without rows.

    With ``lambda_l1`` or ``max_delta_step``, LightGBM's output: ``−ThresholdL1(G, l1) / (n + λ)``, clipped to
    ``±max_delta_step``, times ``lr``.
    """
    empty = counts == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        if lambda_l1 == 0.0 and max_delta_step == 0.0:
            value = np.negative(sums * 2.0**-32) / (counts + lambda_l2) * learning_rate
        else:
            value = _step(sums * 2.0**-32, counts + lambda_l2, lambda_l1, max_delta_step) * learning_rate
    return np.where(empty, 0.0, value).astype(np.float32)


def _step(g: NDArray[Any], curvature: NDArray[Any], l1: float, max_delta: float) -> NDArray[np.float64]:
    """LightGBM's leaf output ``−ThresholdL1(G, l1) / curvature``, clipped to ``±max_delta`` when that is > 0."""
    shrunk = np.sign(g) * np.maximum(np.abs(g) - l1, 0.0)
    step = np.negative(shrunk) / curvature
    return np.asarray(np.clip(step, -max_delta, max_delta) if max_delta > 0.0 else step, np.float64)


def newton_leaf_values(
    sums: NDArray[np.int64], hessians: NDArray[np.int64], counts: NDArray[np.int64], lambda_l2: float,
    learning_rate: float, lambda_l1: float = 0.0, max_delta_step: float = 0.0,
) -> NDArray[np.float32]:  # fmt: skip
    """``float32(−(S_g · 2^-32) / (S_h · 2^-32 + λ) · lr)`` per leaf, in float64; 0 for a leaf without rows.

    ``ValueError`` for a leaf with rows whose ``S_h · 2^-32 + λ`` is 0: its Newton step is undefined.
    """
    empty = counts == 0
    curvature = np.where(empty, 1.0, hessians * 2.0**-32 + lambda_l2)
    if (curvature <= 0.0).any():
        raise ValueError(
            "a leaf's hessians sum to zero and lambda_l2 is 0, so its Newton step is undefined: give the objective "
            "positive hessians or set lambda_l2 > 0"
        )
    if lambda_l1 == 0.0 and max_delta_step == 0.0:
        value = np.negative(sums * 2.0**-32) / curvature * learning_rate
    else:
        value = _step(sums * 2.0**-32, curvature, lambda_l1, max_delta_step) * learning_rate
    return np.where(empty, 0.0, value).astype(np.float32)


def _draw_bag(
    pool: ThreadPoolExecutor, params: Params, n_rows: int, iteration: int, end: int
) -> Future[NDArray[np.uint8]] | None:
    """Start drawing iteration ``iteration``'s bag, or return None when there is no bagging or no iteration left."""
    if not params.bagging or iteration >= end:
        return None
    return pool.submit(sampling.bag_mask, n_rows, params.bagging_fraction, seed=params.seed, iteration=iteration)


def _bounds(rows: slice, n_rows: int) -> tuple[int, int]:
    if rows.step not in (None, 1):
        raise ValueError("rows must be a contiguous slice")
    lo, hi, _ = rows.indices(n_rows)
    if hi <= lo:
        raise ValueError(f"rows {rows} select no row of the panel's {n_rows}")
    return lo, hi


def _gradients(
    device: ManyBackend,
    ys: NDArray[np.float32],
    objectives: list[AnyObjective] | None,
    bins: int,
    coupled: "_Coupled | None" = None,
) -> tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.float64], NDArray[np.float32]]:
    """Compute g for every model (and h, for an objective with hessians); return the quantization scales and inverses.

    The gradients' scales and inverses, then the hessians' (0 for a model without hessians). ``ValueError`` unless the
    hessians are finite and non-negative; ``OverflowError`` if a model's gradients or hessians could overflow the
    fixed-point leaf sums.
    """
    hessians = np.zeros(ys.shape[0], np.float64)
    if coupled is not None:
        largest, hessians = coupled.gradients(device)
    elif objectives is None:
        largest = device.gradients()
    else:
        largest = np.empty(ys.shape[0], np.float64)
        for model, objective in enumerate(objectives):
            pred, g = device.views(model)
            if objective.constant_hessian is True:
                objective.gradients(pred, ys[model], g)
            else:
                h = device.hessians(model)
                objective.gradients(pred, ys[model], g, h)
                hessians[model] = _checked_hessians(h)
            largest[model] = float(np.abs(g).max())
    return (*quantization_scales(largest, ys.shape[1], bins), *quantization_scales(hessians, ys.shape[1], bins))


def _checked_hessians(h: NDArray[np.float32]) -> float:
    """Return ``max h``; ``ValueError`` unless every hessian is finite and >= 0, ``OverflowError`` past the bound."""
    low, high = float(h.min()), float(h.max())
    if not (np.isfinite(high) and low >= 0.0):
        raise ValueError(f"an objective's hessians must be finite and non-negative, got {low} .. {high}")
    if high * h.size >= FIXED_POINT_BOUND:
        raise OverflowError(f"max h = {high} over {h.size} rows overflows the int64 fixed-point leaf sums")
    return high


@dataclass(frozen=True)
class _Coupled:
    """A multi-class fit's objective: every class's gradients from every class's predictions (and the row weights)."""

    objective: MulticlassObjective
    labels: NDArray[np.float32]
    weights: NDArray[np.float32] | None

    def gradients(self, device: ManyBackend) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Write every model's g and h on ``device``; return each one's ``max |g|`` and ``max h``."""
        k = device.k
        preds = np.stack([device.views(model)[0] for model in range(k)])
        grad, hess = np.empty_like(preds), np.empty_like(preds)
        self.objective.gradients(preds, self.labels, grad, hess)
        if self.weights is not None:
            grad *= self.weights
            hess *= self.weights
        largest, highest = np.empty(k, np.float64), np.empty(k, np.float64)
        for model in range(k):
            device.views(model)[1][:] = grad[model]
            device.hessians(model)[:] = hess[model]
            largest[model], highest[model] = float(np.abs(grad[model]).max()), _checked_hessians(hess[model])
        return largest, highest


def quantization_scales(
    largest: NDArray[np.float64], n_rows: int, bins: int
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    """Per model, the quantization scale ``max |g| / (bins / 2)`` and its float32 inverse; check the leaf-sum bound.

    The hessians' take the same half range: ``q_h`` in ``[0, bins / 2]``, within the histogram kernels' bounds.
    """
    if not np.isfinite(largest).all() or (largest * n_rows >= FIXED_POINT_BOUND).any():
        raise OverflowError(f"max |g| = {largest.max()} over {n_rows} rows overflows the int64 fixed-point leaf sums")
    scales = np.where(largest > 0.0, largest / (bins // 2), 0.0)
    inv_scales = np.array([np.float32(1.0 / s) if s > 0.0 else np.float32(0.0) for s in scales], np.float32)
    return scales, inv_scales


@dataclass(frozen=True)
class _Quantized:
    """One tree's quantization scales per model: the gradients', and the hessians' (None: first-order splits)."""

    gradients: NDArray[np.float64]
    hessians: list[float | None]

    @property
    def second(self) -> list[bool]:
        """Per model, whether its splits read the hessians."""
        return [scale is not None for scale in self.hessians]

    def channels(self, model: int) -> int:
        """Return the model's histogram channels: Σq and the count, then Σq_h for second-order splits."""
        return 2 if self.hessians[model] is None else 3


EXTRA_SALT = 11  # derive_key(seed, EXTRA_SALT, tree, a, b): extra_trees' thresholds at node (a, b)
BYNODE_SALT = 2  # SeedSequence([seed, tree, BYNODE_SALT, a, b]): feature_fraction_bynode's features at node (a, b)


@dataclass(frozen=True)
class _Draws:
    """One tree's per-node draws, ``extra_trees``' thresholds and ``feature_fraction_bynode``'s features.

    A node is ``(a, b)``: depth-wise its heap index and 0, leaf-wise the tree's leaf count when the leaf is searched
    and its id. ``tree`` is the tree's index in the model, so a continued fit draws as a fit from scratch.
    """

    params: Params
    tree: int
    features: NDArray[np.int32]
    real_bins: NDArray[np.int32]  # the features' real bins: a drawn threshold is one of 0 .. real − 2

    def search(self, nodes: Sequence[tuple[int, int]]) -> NodeSearch | None:
        """Return the nodes' restrictions, or None when neither setting is on (every feature, every threshold)."""
        p, k = self.params, self.features.size
        if not p.extra_trees and p.feature_fraction_bynode == 1.0:
            return None
        allowed, only = np.ones((len(nodes), k), np.uint8), np.full((len(nodes), k), -1, np.int64)
        real = self.real_bins.astype(np.int64)
        for i, (a, b) in enumerate(nodes):
            if p.feature_fraction_bynode < 1.0:
                m = max(int(k * p.feature_fraction_bynode + 0.5), 1)
                draws = np.random.Generator(np.random.PCG64(np.random.SeedSequence([p.seed, self.tree, BYNODE_SALT, a,
                                                                                    b])))  # fmt: skip
                allowed[i] = 0
                allowed[i, draws.choice(k, size=m, replace=False)] = 1
            if p.extra_trees:
                bits = rng.uniform_bits(rng.derive_key(p.seed, EXTRA_SALT, self.tree, a, b), k).astype(np.int64)
                only[i] = np.where(real >= 2, (bits * (real - 1)) >> rng.UNIFORM_BITS, -2)
        return NodeSearch(allowed, only)


def _histograms(
    device: ManyBackend,
    features: NDArray[np.int32],
    offsets: NDArray[np.int32],
    cells: int,
    slots: NDArray[np.int16],
    quantized: _Quantized,
) -> list[NDArray[np.int32]]:
    """Each model's histograms; a model with second-order splits gets the hessians' sums as a third channel."""
    built = device.histograms(features, offsets, cells, slots)
    second = quantized.second
    if not any(second):
        return built
    mine = np.where(np.array(second)[:, None], slots, np.int16(-1)).astype(np.int16)  # the others build nothing
    sums = device.histograms(features, offsets, cells, mine, hessians=True)
    return [np.concatenate([b, h[..., :1]], axis=-1) if on else b for b, h, on in zip(built, sums, second,
                                                                                     strict=True)]  # fmt: skip


def _grow(
    device: ManyBackend,
    forests: list[Forest],
    tree: int,
    features: NDArray[np.int32],
    layout: FeatureLayout,
    params: Params,
    quantized: _Quantized,
    timer: "_Timer",
    draws: _Draws,
) -> None:
    """Grow tree ``tree`` of every model, level by level in lockstep; leaves each model's ``node`` on its leaves.

    Every level is partitioned, even when no node splits (all rows then go left), so ``node`` always ends as a leaf
    index of the complete tree. Each model's decisions depend only on its own histograms: lockstep changes nothing.
    """
    offsets, cells = layout.offsets(features)
    k = len(forests)
    device.start_tree()
    with timer("histograms"):
        roots = _histograms(device, features, offsets, cells, np.zeros((k, 1), np.int16), quantized)
    hists: list[dict[int, NDArray[np.int32]]] = [{0: root[0]} for root in roots]
    leaves = [1] * k
    for depth in range(params.max_depth):
        levels, chosen = [], []
        with timer("splits"):
            for model in range(k):
                nodes = sorted(hists[model])
                stacked = (np.stack([hists[model][j] for j in nodes]) if nodes
                           else np.zeros((0, cells, quantized.channels(model)), np.int32))  # fmt: skip
                search = draws.search([(2**depth - 1 + j, 0) for j in nodes])
                found = best_splits(stacked, features, layout, params, float(quantized.gradients[model]),
                                    quantized.hessians[model], search)  # fmt: skip
                valid = {j: split for j, split in zip(nodes, found, strict=True) if split is not None}
                best_first = sorted(valid, key=lambda j: (-valid[j].gain, j))  # the leaf cap keeps the highest gains
                picked = {j: valid[j] for j in sorted(best_first[: params.max_leaves - leaves[model]])}
                leaves[model] += len(picked)
                chosen.append(picked)
                levels.append(_level(forests[model], tree, depth, picked))
        with timer("partition"):
            device.partition(levels)
        if depth + 1 == params.max_depth:
            break
        hists = _child_histograms(device, hists, chosen, 2 ** (depth + 1), features, offsets, cells, params,
                                  quantized, timer)  # fmt: skip


def _child_histograms(
    device: ManyBackend,
    hists: list[dict[int, NDArray[np.int32]]],
    chosen: list[dict[int, Split]],
    width: int,
    features: NDArray[np.int32],
    offsets: NDArray[np.int32],
    cells: int,
    params: Params,
    quantized: _Quantized,
    timer: "_Timer",
) -> list[dict[int, NDArray[np.int32]]]:
    """Each model's next-level splittable nodes: build each split's smaller child, subtract for the sibling."""
    slots = np.full((len(chosen), width), -1, np.int16)  # one entry per node of the next level, per model
    counts: list[dict[int, int]] = []
    smaller: list[list[int]] = []
    for model, picked in enumerate(chosen):
        sizes: dict[int, int] = {}
        for j, split in picked.items():
            sizes[2 * j], sizes[2 * j + 1] = split.left_count, split.right_count
        built_here = [2 * j if sizes[2 * j] <= sizes[2 * j + 1] else 2 * j + 1 for j in picked]
        slots[model, built_here] = np.arange(len(built_here), dtype=np.int16)
        counts.append(sizes)
        smaller.append(built_here)
    if not any(smaller):
        return [{} for _ in chosen]
    with timer("histograms"):
        built = _histograms(device, features, offsets, cells, slots, quantized)
    out = []
    for model, picked in enumerate(chosen):
        children = {}
        for j, small in zip(picked, smaller[model], strict=True):
            children[small] = built[model][slots[model, small]]
            children[small ^ 1] = hists[model][j] - built[model][slots[model, small]]  # the sibling: exact
        out.append({j: h for j, h in children.items() if counts[model][j] >= 2 * params.min_data_in_leaf})
    return out


def _grow_leafwise(
    device: ManyBackend,
    forests: list[NodeForest],
    tree: int,
    features: NDArray[np.int32],
    layout: FeatureLayout,
    params: Params,
    quantized: _Quantized,
    timer: "_Timer",
    draws: _Draws,
) -> None:
    """Grow tree ``tree`` of every model best-first; leaves each model's ``node`` on its rows' leaf ids.

    Each step splits, per model, the leaf whose best split has the highest gain (ties: the lowest leaf id): its rows
    that go left keep the leaf's id, those that go right take the next one, and the split becomes the tree's next node
    (`idun.gbm.tree`). The smaller child's histogram is built and the sibling's is the parent's minus it; a child that
    cannot split (fewer than ``2 · min_data_in_leaf`` in-bag rows, or at ``max_depth``) gets none. A model stops at
    ``max_leaves`` leaves or when no leaf has a split. Each model's decisions depend only on its own histograms.
    """
    offsets, cells = layout.offsets(features)
    k = len(forests)
    device.start_tree()
    with timer("histograms"):
        roots = _histograms(device, features, offsets, cells, np.zeros((k, 1), np.int16), quantized)
    growers = [_BestFirst(forest, tree, params, float(quantized.gradients[model]), quantized.hessians[model], draws)
               for model, forest in enumerate(forests)]  # fmt: skip
    with timer("splits"):
        for model, grower in enumerate(growers):
            grower.consider({0: roots[model][0]}, features, layout)
    while True:
        with timer("splits"):
            picks = [grower.pick() for grower in growers]
        if all(pick is None for pick in picks):
            return
        with timer("partition"):
            device.split_leaves([None if pick is None else pick.leaf_split() for pick in picks])
        slots = np.full((k, params.max_leaves), -1, np.int16)
        for model, (grower, pick) in enumerate(zip(growers, picks, strict=True)):
            if pick is not None and (small := grower.child_to_build(pick)) is not None:
                slots[model, small] = 0
        if (slots < 0).all():
            continue
        with timer("histograms"):
            built = _histograms(device, features, offsets, cells, slots, quantized)
        with timer("splits"):
            for model, (grower, pick) in enumerate(zip(growers, picks, strict=True)):
                if pick is not None and len(built[model]):
                    grower.children(pick, built[model][0], features, layout)


@dataclass(frozen=True)
class _Pick:
    """A split chosen by `_BestFirst.pick`: leaf ``leaf`` splits, its right child becomes leaf ``right``."""

    leaf: int
    right: int
    split: Split
    parent_hist: NDArray[np.int32]

    def leaf_split(self) -> LeafSplit:
        """Return the split as the backend's partition takes it."""
        split = self.split
        threshold = split.threshold if split.kind == NUMERIC else 0
        return LeafSplit(self.leaf, self.right, split.kind, split.feature, threshold, int(split.missing_left),
                         split.cat_left)  # fmt: skip


class _BestFirst:
    """One model's tree under best-first growth: its open leaves' best splits and histograms, and its forest."""

    def __init__(
        self, forest: NodeForest, tree: int, params: Params, scale: float, hessian_scale: float | None, draws: _Draws
    ) -> None:
        self.forest, self.tree, self.params = forest, tree, params
        self.scale, self.hessian_scale = scale, hessian_scale  # this tree's quantization (best_splits)
        self.draws = draws
        self.leaves = 1
        self.depth = [0] * params.max_leaves  # per leaf id
        self.parent = [(-1, 0)] * params.max_leaves  # per leaf id: (node, side) that points at it; the root has none
        self.best: dict[int, Split] = {}  # open leaves: those with a split
        self.hists: dict[int, NDArray[np.int32]] = {}

    def consider(self, hists: dict[int, NDArray[np.int32]], features: NDArray[np.int32], layout: FeatureLayout) -> None:
        """Search the best split of each leaf in ``hists``; keep the leaves that have one open."""
        if not hists:
            return
        leaves = sorted(hists)
        search = self.draws.search([(self.leaves, leaf) for leaf in leaves])
        found = best_splits(np.stack([hists[leaf] for leaf in leaves]), features, layout, self.params, self.scale,
                            self.hessian_scale, search)  # fmt: skip
        for leaf, split in zip(leaves, found, strict=True):
            if split is not None:
                self.best[leaf], self.hists[leaf] = split, hists[leaf]

    def pick(self) -> _Pick | None:
        """Split the open leaf of highest gain (ties: the lowest id) as the tree's next node; None when done."""
        if self.leaves == self.params.max_leaves or not self.best:
            return None
        leaf = max(self.best, key=lambda j: (self.best[j].gain, -j))
        split = self.best.pop(leaf)
        node, right = self.leaves - 1, self.leaves
        self.leaves += 1
        forest, tree = self.forest, self.tree
        forest.kind[tree, node] = split.kind
        forest.feature[tree, node] = split.feature
        forest.threshold[tree, node] = split.threshold if split.kind == NUMERIC else 0
        forest.missing_left[tree, node] = split.missing_left
        forest.cat_left[tree, node] = split.cat_left
        forest.left[tree, node], forest.right[tree, node] = ~leaf, ~right
        up, side = self.parent[leaf]
        if up >= 0:
            (forest.right if side else forest.left)[tree, up] = node
        self.parent[leaf], self.parent[right] = (node, 0), (node, 1)
        self.depth[leaf] += 1
        self.depth[right] = self.depth[leaf]
        return _Pick(leaf, right, split, self.hists.pop(leaf))

    def _can_split(self, leaf: int, count: int) -> bool:
        depth_ok = self.params.max_depth == NO_DEPTH_LIMIT or self.depth[leaf] < self.params.max_depth
        return depth_ok and count >= 2 * self.params.min_data_in_leaf

    def child_to_build(self, pick: _Pick) -> int | None:
        """Return the leaf whose histogram to build after ``pick`` (the smaller child); None if no child can split."""
        sizes = {pick.leaf: pick.split.left_count, pick.right: pick.split.right_count}
        if self.leaves == self.params.max_leaves or not any(self._can_split(j, n) for j, n in sizes.items()):
            return None
        return pick.leaf if sizes[pick.leaf] <= sizes[pick.right] else pick.right

    def children(
        self, pick: _Pick, built: NDArray[np.int32], features: NDArray[np.int32], layout: FeatureLayout
    ) -> None:
        """Open the children of ``pick`` that can split, from the smaller one's histogram ``built``."""
        sizes = {pick.leaf: pick.split.left_count, pick.right: pick.split.right_count}
        small, large = (pick.leaf, pick.right) if sizes[pick.leaf] <= sizes[pick.right] else (pick.right, pick.leaf)
        hists = {small: built, large: pick.parent_hist - built}  # the sibling: exact
        self.consider({j: h for j, h in hists.items() if self._can_split(j, sizes[j])}, features, layout)


def _level(forest: Forest, tree: int, depth: int, splits: dict[int, Split]) -> LevelSplits:
    """Write one level's splits into the forest (other nodes stay pass-through) and return them for the partition."""
    first, width = 2**depth - 1, 2**depth
    for j, split in splits.items():
        h = first + j
        forest.kind[tree, h] = split.kind
        forest.feature[tree, h] = split.feature
        forest.threshold[tree, h] = split.threshold if split.kind == NUMERIC else 0
        forest.missing_left[tree, h] = split.missing_left
        forest.cat_left[tree, h] = split.cat_left
    nodes = slice(first, first + width)
    return LevelSplits(
        kind=forest.kind[tree, nodes],
        feature=forest.feature[tree, nodes],
        threshold=forest.threshold[tree, nodes],
        missing_left=forest.missing_left[tree, nodes],
        cat_left=forest.cat_left[tree, nodes],
    )


@dataclass
class _Timer:
    """Wall-clock seconds per phase of the fit (reported in `FitReport`, never used for decisions)."""

    seconds: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))

    def __call__(self, phase: str) -> "_Phase":
        return _Phase(self, phase)


@dataclass
class _Phase:
    timer: _Timer
    phase: str
    start: float = 0.0

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self, *_: object) -> None:
        self.timer.seconds[self.phase] += time.perf_counter() - self.start
