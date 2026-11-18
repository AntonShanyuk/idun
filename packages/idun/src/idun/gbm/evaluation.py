"""Validation rows scored after every tree, early stopping and per-tree callbacks (`train`'s ``valid``).

Each new tree's outputs on a validation set are added on the host (the CPU backend's predictor, which every backend
equals bit for bit) to the set's running predictions, in tree order as `Booster.predict` adds them: so the score after
tree ``i`` is the metric of what the model's first ``i + 1`` trees predict, on any backend. Early stopping watches
the first set: the fit stops when its score has not improved for ``early_stopping_rounds`` trees and keeps the trees
up to its best, so an early-stopped model is the model of that many trees, bit for bit. A multi-class fit
(`train_multiclass`) scores its K classes' trees together: the sets' predictions are ``(K, n)`` and its metrics
(``multi_logloss``, ``multi_error``) read every class's scores.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from idun.gbm.backends.cpu import CPUBackend
from idun.gbm.features import BinnedData, FeatureLayout
from idun.gbm.tree import Trees

if TYPE_CHECKING:
    from idun.gbm.booster import Booster


@dataclass(frozen=True)
class Metric:
    """A score of predictions against targets, ``fn(y, pred)``: lower is better unless ``higher_is_better``."""

    name: str
    fn: Callable[[NDArray[np.float32], NDArray[np.float32]], float]
    higher_is_better: bool = False


Weights = NDArray[np.float64] | None  # a validation set's row weights (None: every row counts once)


def _mean(values: NDArray[Any], weights: Weights) -> float:
    """Return the mean, or with weights LightGBM's weighted mean ``Σ w v / Σ w``."""
    return float(np.mean(values) if weights is None else np.sum(values * weights) / np.sum(weights))


def _l2(y: NDArray[np.float32], pred: NDArray[np.float32], weights: Weights = None) -> float:
    d = pred.astype(np.float64) - y.astype(np.float64)
    return _mean(d * d, weights)


def _rmse(y: NDArray[np.float32], pred: NDArray[np.float32], weights: Weights = None) -> float:
    return math.sqrt(_l2(y, pred, weights))


def _l1(y: NDArray[np.float32], pred: NDArray[np.float32], weights: Weights = None) -> float:
    return _mean(np.abs(pred.astype(np.float64) - y.astype(np.float64)), weights)


def _binary_logloss(y: NDArray[np.float32], pred: NDArray[np.float32], weights: Weights = None) -> float:
    """Log loss of labels 0 and 1 against raw scores (``log(1 + e^s) − y s``, stable)."""
    s = pred.astype(np.float64)
    return _mean(np.logaddexp(0.0, s) - y.astype(np.float64) * s, weights)


def _auc(y: NDArray[np.float32], pred: NDArray[np.float32], weights: Weights = None) -> float:
    """Area under the ROC curve of labels 0 and 1 against scores, ties at their mean rank (Mann-Whitney).

    With weights, LightGBM's weighted AUC: each positive's weight times the negatives' weight scored below it, half
    of those tied, over the product of the two classes' weights.
    """
    positive = y.astype(np.float64) > 0.5
    if weights is not None:
        return _weighted_auc(positive, pred, weights)
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    _, inverse, counts = np.unique(pred.astype(np.float64), return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts  # each distinct score's first 0-based position
    ranks = (starts + (counts + 1) / 2.0)[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _weighted_auc(positive: NDArray[np.bool_], pred: NDArray[np.float32], weights: NDArray[np.float64]) -> float:
    _, inverse = np.unique(pred.astype(np.float64), return_inverse=True)
    groups = int(inverse.max()) + 1
    pos = np.bincount(inverse, weights=np.where(positive, weights, 0.0), minlength=groups)
    neg = np.bincount(inverse, weights=np.where(positive, 0.0, weights), minlength=groups)
    if pos.sum() == 0.0 or neg.sum() == 0.0:
        return math.nan
    below = np.cumsum(neg) - neg  # the negatives' weight at lower scores
    return float(np.sum(pos * (below + neg / 2.0)) / (pos.sum() * neg.sum()))


def _multi_logloss(y: NDArray[np.float32], pred: NDArray[np.float32], weights: Weights = None) -> float:
    """Log loss of class ids against ``(K, n)`` raw scores: ``logsumexp(s) − s_y`` per row, stable."""
    s = pred.astype(np.float64)
    top = s.max(axis=0)
    total = top + np.log(np.exp(s - top).sum(axis=0))
    return _mean(total - s[np.asarray(y, np.int64), np.arange(s.shape[1])], weights)


def _multi_error(y: NDArray[np.float32], pred: NDArray[np.float32], weights: Weights = None) -> float:
    """Return the share of rows whose highest score is not their class's (the first class on a tie)."""
    return _mean(np.argmax(pred, axis=0) != np.asarray(y, np.int64), weights)


METRICS = {  # LightGBM's names; each on raw scores
    "l2": Metric("l2", _l2),
    "rmse": Metric("rmse", _rmse),
    "l1": Metric("l1", _l1),
    "binary_logloss": Metric("binary_logloss", _binary_logloss),
    "auc": Metric("auc", _auc, higher_is_better=True),
    "multi_logloss": Metric("multi_logloss", _multi_logloss),
    "multi_error": Metric("multi_error", _multi_error),
}
MULTICLASS = frozenset({"multi_logloss", "multi_error"})  # they read every class's scores: multi-class fits only


def metric_of(metric: str | Metric) -> Metric:
    """Return the `Metric` a name stands for (`METRICS`), or ``metric`` itself."""
    if isinstance(metric, Metric):
        return metric
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)} or a Metric, got {metric!r}")
    return METRICS[metric]


@dataclass(frozen=True)
class Validation:
    """Rows to score after every tree: ``rows`` of ``data``, binned as the training data, against ``y``.

    ``y`` is ``data.target[rows]`` by default. ``name`` keys the set's scores (``valid_<i>`` by position by default).
    ``weight``: one non-negative weight per row, for the built-in metrics' weighted forms (LightGBM's).
    """

    data: BinnedData
    rows: slice
    y: NDArray[np.float32] | None = None
    name: str | None = None
    weight: NDArray[np.float32] | None = None


@dataclass(frozen=True)
class Evaluation:
    """What a callback sees after each tree: the tree's ``iteration`` (from 0) and each set's ``scores`` so far."""

    iteration: int
    metric: str
    scores: dict[str, float]


Callback = Callable[[Evaluation], bool | None]  # True stops the fit after this tree


class Evaluator:
    """One fit's validation sets: scores each tree, keeps the history and the best tree, says when to stop."""

    def __init__(
        self,
        valid: Sequence[Validation],
        binning: str,
        metric: str | Metric,
        early_stopping_rounds: int | None,
        callback: Callback | None,
        k: int = 1,
    ) -> None:
        """``k``: the forests a tree of the fit adds, one per class of a multi-class fit (1 otherwise)."""
        if early_stopping_rounds is not None and (early_stopping_rounds < 1 or not valid):
            raise ValueError(
                f"early_stopping_rounds needs a validation set and a positive count, got {early_stopping_rounds!r} "
                f"with {len(valid)} sets"
            )
        self.metric = metric_of(metric)
        if (self.metric.name in MULTICLASS) != (k > 1) and not isinstance(metric, Metric):
            raise ValueError(f"metric {self.metric.name!r} is for {'' if k > 1 else 'multi-class '}fits; a "
                             f"{'multi-class' if k > 1 else 'one-model'} fit takes "
                             f"{sorted(MULTICLASS) if k > 1 else sorted(set(METRICS) - MULTICLASS)}")  # fmt: skip
        self.k = k
        self.early_stopping_rounds = early_stopping_rounds
        self.callback = callback
        # (name, data, layout, lo, hi, y, the (k, n) running predictions), and each set's weights
        self.sets: list[tuple[str, BinnedData, FeatureLayout, int, int, NDArray[np.float32], NDArray[np.float32]]] = []
        self.weights: list[NDArray[np.float64] | None] = []
        for i, entry in enumerate(valid):
            name = f"valid_{i}" if entry.name is None else entry.name
            if entry.data.binning != binning:
                raise ValueError(f"validation set {name!r} is binned as {entry.data.binning}, the training data as "
                                 f"{binning}")  # fmt: skip
            lo, hi, _ = entry.rows.indices(entry.data.n_rows)
            if entry.rows.step not in (None, 1) or not 0 <= lo < hi:
                raise ValueError(
                    f"validation set {name!r} needs a contiguous non-empty range of rows, got {entry.rows}"
                )
            y = np.ascontiguousarray(entry.data.target[lo:hi] if entry.y is None else entry.y, dtype=np.float32)
            if y.shape != (hi - lo,):
                raise ValueError(f"validation set {name!r}: y must hold one float32 per row ({hi - lo}), got {y.shape}")
            pred = np.zeros((k, hi - lo), np.float32)
            self.sets.append((name, entry.data, FeatureLayout.of(entry.data), lo, hi, y, pred))
            self.weights.append(None if entry.weight is None else _set_weights(name, entry.weight, hi - lo))
        if any(w is not None for w in self.weights) and METRICS.get(self.metric.name) is not self.metric:
            raise ValueError(f"a weighted validation set needs a built-in metric ({sorted(METRICS)}); "
                             f"{self.metric.name!r} takes no weights")  # fmt: skip
        if len({name for name, *_ in self.sets}) != len(self.sets):
            raise ValueError(f"validation sets need distinct names, got {[name for name, *_ in self.sets]}")
        self.history: dict[str, list[float]] = {name: [] for name, *_ in self.sets}
        self.best_iteration = -1
        self._best = math.nan
        self._cpu = CPUBackend()

    def start_from(self, model: "Booster") -> None:
        """Start every set's predictions at ``model``'s (continued training)."""
        for name, data, _, lo, hi, _, pred in self.sets:
            if data.binning != model.binning:
                raise ValueError(f"validation set {name!r} is binned otherwise than init_model")
            pred[0] = model.predict(data, rows=slice(lo, hi), backend="cpu")

    def after(self, iteration: int, forests: Sequence[Trees]) -> bool:
        """Score tree ``iteration`` of the ``forests`` on every set; return whether the fit stops after it."""
        scores = {}
        for (name, data, layout, lo, hi, y, pred), weights in zip(self.sets, self.weights, strict=True):
            for model, forest in enumerate(forests):
                tree = forest.trees(iteration, iteration + 1)
                np.add(pred[model], self._cpu.predict(data, layout, lo, hi, tree), out=pred[model])
            scored = pred[0] if self.k == 1 else pred
            scores[name] = self.metric.fn(y, scored) if weights is None else self.metric.fn(y, scored, weights)  # type: ignore[call-arg]
            self.history[name].append(scores[name])
        if self.sets:
            first = scores[self.sets[0][0]]
            if self.best_iteration < 0 or self._better(first):
                self._best, self.best_iteration = first, iteration
        stop = self.early_stopping_rounds is not None and iteration - self.best_iteration >= self.early_stopping_rounds
        if self.callback is not None and self.callback(Evaluation(iteration, self.metric.name, scores)) is True:
            stop = True
        return stop

    def _better(self, score: float) -> bool:
        if math.isnan(self._best):
            return not math.isnan(score)
        return score > self._best if self.metric.higher_is_better else score < self._best

    def kept(self, trained: int) -> int:
        """Return how many trees the model keeps: up to the best with early stopping, else every tree trained."""
        return self.best_iteration + 1 if self.early_stopping_rounds is not None else trained


def _set_weights(name: str, weight: NDArray[np.float32], n: int) -> NDArray[np.float64]:
    """Return a validation set's weights as float64, checked: one per row, finite, non-negative, not all zero."""
    weights = np.asarray(weight, np.float64)
    if weights.shape != (n,):
        raise ValueError(f"validation set {name!r}: weight must hold one weight per row ({n}), got {weights.shape}")
    if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
        raise ValueError(f"validation set {name!r}: weights must be finite and non-negative, not all zero")
    return weights


def log_evaluation(period: int = 1) -> Callback:
    """Return a callback that prints every ``period``-th tree's scores, as LightGBM's ``log_evaluation`` does."""
    if period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    def log(evaluation: Evaluation) -> None:
        if (evaluation.iteration + 1) % period == 0:
            scores = "\t".join(f"{name}'s {evaluation.metric}: {value:g}" for name, value in evaluation.scores.items())
            print(f"[{evaluation.iteration + 1}]\t{scores}", flush=True)  # noqa: T201 - asked for

    return log
