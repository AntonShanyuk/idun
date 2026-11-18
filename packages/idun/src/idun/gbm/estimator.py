"""`GBMRegressor` and `GBMClassifier`: idun_gbm behind scikit-learn's estimator conventions.

``fit(X, y)`` takes a 2-D NumPy array, a pandas DataFrame or an Arrow table (binned into an `idun.Dataset` with the
estimator's ``categorical`` and ``codes`` columns), or a `Dataset` (its target is ``y``). ``predict(X)`` bins raw
rows with the training binning. ``n_seeds`` > 1 trains a seed bag (`train_seeds`, each model bit for bit its separate
fit) and predicts the mean. The estimator's parameters are `Params`' fields, except the categorical split method,
``categorical_split`` (`Params.categorical`), since ``categorical`` names the categorical columns as in `Dataset`.
``fit(X, y, eval_set=[(X_val, y_val)])`` scores each tree on validation rows (binned with the training binning) with
``eval_metric``, and with ``early_stopping_rounds`` stops and keeps the trees up to the first set's best
(`train`'s ``valid``); ``verbose`` > 0 prints every ``verbose``-th tree's scores. scikit-learn is not imported:
`get_params` and `set_params` follow its conventions.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

from idun._version import __version__
from idun.base import Classifier, Regressor
from idun.dataset import Column, Dataset
from idun.gbm.booster import COUNTS, Booster, trees_from
from idun.gbm.evaluation import Callback, Evaluation, Validation, log_evaluation
from idun.gbm.objective import LogLoss
from idun.gbm.params import Params
from idun.gbm.train import train, train_seeds
from idun.gbm.tree import NODE_FIELDS, fields_of

MODEL_FORMAT = "idun-model"
MODEL_FORMAT_VERSION = 1


class GBMRegressor(Regressor):
    """Gradient-boosted regression trees (L2) on Metal, CUDA or the CPU; see the module docstring and `Params`."""

    def __init__(
        self,
        *,
        num_trees: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_leaves: int = 64,
        growth: Literal["depthwise", "leafwise"] = "depthwise",
        min_data_in_leaf: int = 20,
        lambda_l2: float = 0.0,
        lambda_l1: float = 0.0,
        min_gain_to_split: float = 0.0,
        min_sum_hessian_in_leaf: float = 1e-3,
        max_delta_step: float = 0.0,
        second_order: bool = False,
        extra_trees: bool = False,
        feature_fraction_bynode: float = 1.0,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
        grad_quant_bins: int = 4,
        categorical_split: Literal["sorted", "one_vs_rest"] = "sorted",
        cat_smooth: float = 10.0,
        cat_l2: float = 10.0,
        max_cat_threshold: int = 32,
        min_data_per_group: int = 100,
        seed: int = 0,
        n_seeds: int = 1,
        backend: str = "metal",
        workers: int | None = None,
        categorical: Sequence[Column] = (),
        codes: Sequence[Column] = (),
        eval_metric: str | None = None,
        early_stopping_rounds: int | None = None,
        verbose: int = 0,
        importance_type: Literal["split"] = "split",
        max_bin: int = 255,
        subsample_for_bin: int = 200_000,
    ) -> None:
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.growth = growth
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.min_gain_to_split = min_gain_to_split
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.max_delta_step = max_delta_step
        self.second_order = second_order
        self.extra_trees = extra_trees
        self.feature_fraction_bynode = feature_fraction_bynode
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.grad_quant_bins = grad_quant_bins
        self.categorical_split = categorical_split
        self.cat_smooth = cat_smooth
        self.cat_l2 = cat_l2
        self.max_cat_threshold = max_cat_threshold
        self.min_data_per_group = min_data_per_group
        self.seed = seed
        self.n_seeds = n_seeds
        self.backend = backend
        self.workers = workers
        self.categorical = categorical
        self.codes = codes
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.importance_type = importance_type
        self.max_bin = max_bin
        self.subsample_for_bin = subsample_for_bin

    def params(self) -> Params:
        """Return the `Params` of these settings (validated: a bad value raises ``ValueError``)."""
        fields = {name: getattr(self, name) for name in Params.__dataclass_fields__ if name != "categorical"}
        return Params(**fields, categorical=self.categorical_split)

    def fit(
        self,
        X: Any,  # noqa: N803 - scikit-learn's name
        y: Any = None,
        sample_weight: Any = None,
        *,
        eval_set: Sequence[tuple[Any, Any] | Dataset] = (),
        eval_names: Sequence[str] | None = None,
        callback: Callback | None = None,
        eval_sample_weight: Sequence[Any] | None = None,
    ) -> Self:
        """Train on ``X`` and ``y`` (or on a `Dataset` and its target); return the estimator.

        ``sample_weight``: one non-negative weight per row (`train`'s ``weight``: second-order splits, weighted
        leaves and start). ``eval_set``: validation sets, each ``(X, y)`` or a `Dataset` with its target, scored after
        every tree; ``eval_names`` names them (``valid_<i>`` by default); ``callback`` is called after every tree
        (`train`'s); ``eval_sample_weight``: each set's row weights (or None), for the metric's weighted form.
        """
        if isinstance(X, Dataset):
            if y is not None:
                raise ValueError("a Dataset carries its target: pass y to Dataset(X, y), not to fit")
            self._check_targets(X.target)
            data = X
        else:
            self._require_y(y, " (or fit a Dataset built with its target)")
            data = Dataset(X, self._encode(y, fit=True), categorical=self.categorical, codes=self.codes,
                           max_bin=self.max_bin, subsample_for_bin=self.subsample_for_bin)  # fmt: skip
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be a positive integer, got {self.n_seeds!r}")
        params, rows = self.params(), slice(0, data.n_rows)
        weight = None if sample_weight is None else _row_weights(sample_weight, data.n_rows)
        template = data.subset(slice(0, 0))
        valid, each = self._validation(template, eval_set, eval_names, eval_sample_weight), self._callback(callback)
        stop, metric, objective, start = self.early_stopping_rounds, self._metric(), self._objective(), self._BOOST
        if self.n_seeds == 1:
            self.boosters_ = [train(params, data, rows=rows, backend=self.backend, objective=objective, valid=valid,
                                    metric=metric, early_stopping_rounds=stop, callback=each,
                                    boost_from_average=start, weight=weight)]  # fmt: skip
        else:
            seeds = [self.seed + i for i in range(self.n_seeds)]
            self.boosters_ = train_seeds(params, data, rows=rows, backend=self.backend, seeds=seeds,
                                         workers=self.workers, objective=objective, valid=valid, metric=metric,
                                         early_stopping_rounds=stop, callback=each,
                                         boost_from_average=start, weight=weight)  # fmt: skip
        self.template_ = template
        self.n_features_in_ = len(data.columns)
        self.feature_names_in_ = np.array(data.columns, dtype=object)
        return self

    def _validation(
        self,
        template: Dataset,
        eval_set: Sequence[tuple[Any, Any] | Dataset],
        names: Sequence[str] | None,
        weights: Sequence[Any] | None = None,
    ) -> list[Validation]:
        """Bin each validation set with the training binning (with its weights, if given)."""
        if names is not None and len(names) != len(eval_set):
            raise ValueError(f"eval_names must name each of the {len(eval_set)} eval sets, got {list(names)}")
        if weights is not None and len(weights) != len(eval_set):
            raise ValueError(
                f"eval_sample_weight must hold one entry per eval set ({len(eval_set)}), got {len(weights)}"
            )
        found = []
        for i, entry in enumerate(eval_set):
            if isinstance(entry, Dataset):
                self._check_targets(entry.target)
                data = entry
            else:
                rows, labels = entry
                data = template.transform(rows, self._encode(labels, fit=False))
            if data.binning != template.binning:
                raise ValueError(f"eval set {i} is a Dataset binned otherwise than the training data")
            weight = None if weights is None or weights[i] is None else _row_weights(weights[i], data.n_rows)
            found.append(
                Validation(data, slice(0, data.n_rows), name=None if names is None else names[i], weight=weight)
            )
        return found

    # The hooks GBMClassifier fills: the targets the trees fit, their loss, their start and the default metric.
    _DEFAULT_METRIC = "l2"
    _BOOST = False

    def _encode(self, y: Any, *, fit: bool) -> Any:  # noqa: ARG002 - the classifier's labels need it
        """Return the targets the trees fit for labels ``y`` (the regressor's: ``y`` itself)."""
        return y

    def _check_targets(self, target: NDArray[np.float32]) -> None:
        """Refuse a `Dataset`'s target that the trees cannot fit (any is fine for the regressor)."""

    def _objective(self) -> LogLoss | None:
        return None

    def _metric(self) -> str:
        return self._DEFAULT_METRIC if self.eval_metric is None else self.eval_metric

    def _callback(self, callback: Callback | None) -> Callback | None:
        """Join ``verbose``'s printing and the caller's callback."""
        if self.verbose < 0:
            raise ValueError(f"verbose must be 0 (silent) or a print period, got {self.verbose!r}")
        printing = log_evaluation(self.verbose) if self.verbose > 0 else None
        if printing is None or callback is None:
            return printing or callback

        def both(evaluation: Evaluation) -> bool | None:
            printing(evaluation)
            return callback(evaluation)

        return both

    @property
    def best_iteration_(self) -> int | list[int | None] | None:
        """Early stopping's best tree (from 0), or each seed's in a bag; None without early stopping."""
        self._check_fitted("boosters_")
        found = [booster.best_iteration for booster in self.boosters_]
        return found[0] if len(found) == 1 else found

    @property
    def evals_result_(self) -> dict[str, dict[str, list[float]]] | list[dict[str, dict[str, list[float]]]]:
        """Each validation set's score after every tree, ``{set: {metric: scores}}`` (each seed's in a bag)."""
        self._check_fitted("boosters_")
        metric = self._metric()
        found = [{name: {metric: scores} for name, scores in booster.evals.items()} for booster in self.boosters_]
        return found[0] if len(found) == 1 else found

    @property
    def feature_importances_(self) -> NDArray[np.int64]:
        """Splits per feature, summed over the bag's models, in ``feature_names_in_`` order (``importance_type``)."""
        self._check_fitted("boosters_")
        total = sum(booster.feature_importance(self.importance_type) for booster in self.boosters_)
        binned = list(self.template_.feature_names)  # the boosters' order: features grouped by kind
        return np.asarray(total, np.int64)[[binned.index(name) for name in self.template_.columns]]

    def _data(self, X: Any) -> Dataset:  # noqa: N803 - scikit-learn's name
        self._check_fitted("boosters_")
        if isinstance(X, Dataset):
            return X
        self._check_n_features(X)
        return self.template_.transform(X)

    def predict(
        self,
        X: Any,  # noqa: N803 - scikit-learn's name
        *,
        per_seed: bool = False,
        num_iteration: int | None = None,
        start_iteration: int = 0,
        pred_contrib: bool = False,
        pred_leaf: bool = False,
    ) -> NDArray[Any]:
        """Return the prediction of each row of ``X`` (the mean of the seed bag's models).

        ``per_seed``: each model's prediction instead, ``(n_rows, n_seeds)`` in seed order; column i is what a one-seed
        fit with seed ``seed + i`` predicts. ``num_iteration`` trees from ``start_iteration`` (from 0) predict, all
        the model keeps by default (LightGBM's names; an early-stopped model keeps the trees up to its best).
        ``pred_contrib``: each row's TreeSHAP contributions instead, ``(n, n_features_in_ + 1)`` in
        ``feature_names_in_`` order, then the expected value (LightGBM's; a bag's the mean of its models', which adds
        up to the bag's mean raw prediction; with ``per_seed``, ``(n, n_seeds, n_features_in_ + 1)``).
        ``pred_leaf``: the leaf each row reaches in each tree, ``(n, n_trees)`` (with ``per_seed``, one per model).
        """
        data = self._data(X)
        rows = slice(0, data.n_rows)
        if pred_contrib and pred_leaf:
            raise ValueError("pred_contrib or pred_leaf, not both")
        if pred_contrib or pred_leaf:
            return self._explain(data, rows, num_iteration, start_iteration, per_seed=per_seed, leaves=pred_leaf)
        each = [b.predict(data, rows=rows, backend=self.backend, num_trees=num_iteration, start_tree=start_iteration)
                for b in self.boosters_]  # fmt: skip
        if per_seed:
            return np.column_stack(each)
        if len(each) == 1:
            return each[0]
        total = np.zeros(data.n_rows, np.float64)
        for prediction in each:  # predict_mean's sum, in order
            total += prediction
        return (total / len(each)).astype(np.float32)

    def _explain(
        self,
        data: Dataset,
        rows: slice,
        num_iteration: int | None,
        start_iteration: int,
        *,
        per_seed: bool,
        leaves: bool,
    ) -> NDArray[Any]:
        """Return ``predict``'s contributions (in the columns' order) or leaves."""
        if leaves:
            if len(self.boosters_) > 1 and not per_seed:
                raise ValueError("pred_leaf reads one model: a seed bag's come with per_seed=True")
            found = [b.predict_leaves(data, rows=rows, num_trees=num_iteration, start_tree=start_iteration)
                     for b in self.boosters_]  # fmt: skip
            return np.stack(found, axis=1) if per_seed else found[0]
        binned = list(self.template_.feature_names)
        order = [*(binned.index(name) for name in self.template_.columns), len(binned)]  # then the expected value
        each = [b.predict_contributions(data, rows=rows, num_trees=num_iteration, start_tree=start_iteration)[:, order]
                for b in self.boosters_]  # fmt: skip
        return np.stack(each, axis=1) if per_seed else np.mean(each, axis=0)

    def save(self, path: str | Path) -> None:
        """Write the fitted estimator to ``path`` (an uncompressed ``.npz``, no object arrays); `idun.load` reads it."""
        self._check_fitted("boosters_")
        meta = {
            "format": MODEL_FORMAT, "format_version": MODEL_FORMAT_VERSION, "idun_version": __version__,
            "estimator": type(self).__name__,
            "params": {k: list(v) if isinstance(v, tuple) else v for k, v in self.get_params().items()},
            "dataset": self.template_.spec(),
            "boosters": [{"params": b.params.to_dict(), "train_rows": list(b.train_rows), "backend": b.backend,
                          "best_iteration": b.best_iteration, "evals": b.evals} for b in self.boosters_],
        }  # fmt: skip
        if hasattr(self, "classes_"):
            meta["classes"] = self.classes_.tolist()
        arrays: dict[str, NDArray[Any]] = {"meta": np.frombuffer(json.dumps(meta).encode(), np.uint8),
                                           "named_edges": self.template_.named_edges}  # fmt: skip
        for i, booster in enumerate(self.boosters_):
            arrays |= {f"b{i}_{name}": getattr(booster.forest, name) for name in fields_of(booster.forest)}
            if booster.forest.leaf_count is not None:
                arrays[f"b{i}_{COUNTS}"] = booster.forest.leaf_count
        with Path(path).open("wb") as handle:
            np.savez(handle, allow_pickle=False, **arrays)

    @classmethod
    def _from_archive(cls, meta: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> Self:
        model = cls(**meta["params"])
        template = Dataset.from_spec(meta["dataset"], arrays["named_edges"])
        model.boosters_ = []
        for i, entry in enumerate(meta["boosters"]):
            params = Params.from_dict(entry["params"])
            trees = {name: arrays[f"b{i}_{name}"] for name in (*NODE_FIELDS, COUNTS) if f"b{i}_{name}" in arrays}
            model.boosters_.append(
                Booster(
                    params,
                    trees_from(params, trees, f"model {i}"),
                    binning=template.binning,
                    feature_names=template.feature_names,
                    categories=template.categories,
                    named_edges=template.named_edges,
                    named_n_bins=np.asarray(template.named_n_bins, np.uint16),
                    train_rows=(int(entry["train_rows"][0]), int(entry["train_rows"][1])),
                    backend=entry["backend"],
                    best_iteration=entry.get("best_iteration"),  # optional in a file, as evals
                    evals=entry.get("evals"),
                )
            )
        model.template_ = template
        if "classes" in meta:
            setattr(model, "classes_", np.array(meta["classes"]))  # noqa: B010 - GBMClassifier's
        model.n_features_in_ = len(template.columns)
        model.feature_names_in_ = np.array(template.columns, dtype=object)
        return model


def _row_weights(sample_weight: Any, n_rows: int) -> NDArray[np.float32]:
    """Return ``sample_weight`` as one float32 per row (a list, a pandas Series or any array-like)."""
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (n_rows,):
        raise ValueError(f"sample_weight must hold one weight per row ({n_rows}), got shape {weights.shape}")
    return weights.astype(np.float32)


class GBMClassifier(Classifier, GBMRegressor):
    """Binary gradient-boosted classification trees: LightGBM's ``binary`` objective, with GBMRegressor's settings.

    The trees fit the log loss of the two labels (`LogLoss`, Newton leaves) from the log-odds of the training labels'
    mean (``boost_from_average``). ``classes_`` are the two labels, sorted; the second is the positive one.
    ``predict_proba`` gives ``[1 − p, p]`` with ``p = 1 / (1 + e^−s)`` of the raw score ``s`` (a seed bag's mean
    score), ``predict`` the label of the larger probability (the first on a tie), ``decision_function`` the raw score.
    ``eval_metric`` is ``"binary_logloss"`` by default (or ``"auc"``). More than two classes raise ``ValueError``.
    """

    _DEFAULT_METRIC = "binary_logloss"
    _BOOST = True

    def __sklearn_tags__(self) -> Any:
        """Return scikit-learn's tags: a binary classifier (its checks then draw two classes)."""
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _objective(self) -> LogLoss:
        return LogLoss()

    def _encode(self, y: Any, *, fit: bool) -> NDArray[np.float32]:
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError(f"y must be one label per row, got shape {labels.shape}")
        if labels.dtype.kind == "f" and not np.array_equal(labels, np.round(labels)):
            raise ValueError("Unknown label type: continuous; a classifier needs class labels (use GBMRegressor)")
        if fit:
            classes = np.unique(labels)
            if classes.size > 2:
                raise ValueError(f"Only binary classification is supported. GBMClassifier fits two classes, y has "
                                 f"{classes.size}: {classes[:10].tolist()}")  # fmt: skip
            if classes.size < 2:
                raise ValueError(f"GBMClassifier fits two classes, y has {classes.size} class: {classes.tolist()}")
            self.classes_ = classes
        unknown = ~np.isin(labels, self.classes_)
        if unknown.any():
            raise ValueError(f"labels outside the training classes {self.classes_.tolist()}: "
                             f"{np.unique(labels[unknown])[:10].tolist()}")  # fmt: skip
        encoded: NDArray[np.float32] = (labels == self.classes_[1]).astype(np.float32)
        return encoded

    def _check_targets(self, target: NDArray[np.float32]) -> None:
        if not np.isin(target, (0.0, 1.0)).all():
            raise ValueError("a Dataset for GBMClassifier needs the targets 0 and 1 (or pass the labels as y)")
        self.classes_ = np.array([0.0, 1.0])

    def decision_function(
        self,
        X: Any,  # noqa: N803 - scikit-learn's name
        *,
        num_iteration: int | None = None,
        start_iteration: int = 0,
    ) -> NDArray[np.float32]:
        """Return each row's raw score (a seed bag's mean), from GBMRegressor's ``predict``."""
        score: NDArray[np.float32] = GBMRegressor.predict(
            self, X, num_iteration=num_iteration, start_iteration=start_iteration
        )
        return score

    def predict_proba(
        self,
        X: Any,  # noqa: N803 - scikit-learn's name
        *,
        num_iteration: int | None = None,
        start_iteration: int = 0,
    ) -> NDArray[np.float64]:
        """Return ``[1 − p, p]`` per row, ``p`` the probability of ``classes_[1]``."""
        score = self.decision_function(X, num_iteration=num_iteration, start_iteration=start_iteration)
        p = 1.0 / (1.0 + np.exp(-score.astype(np.float64)))
        return np.column_stack([1.0 - p, p])

    def predict(  # type: ignore[override]
        self,
        X: Any,  # noqa: N803 - scikit-learn's name
        *,
        num_iteration: int | None = None,
        start_iteration: int = 0,
        pred_contrib: bool = False,
        pred_leaf: bool = False,
    ) -> NDArray[Any]:
        """Return each row's label: the class of the larger probability.

        ``pred_contrib`` and ``pred_leaf`` as GBMRegressor's: the contributions add up to the raw score
        (`decision_function`), as LightGBM's do.
        """
        if pred_contrib or pred_leaf:
            found: NDArray[Any] = GBMRegressor.predict(self, X, num_iteration=num_iteration,
                                                       start_iteration=start_iteration, pred_contrib=pred_contrib,
                                                       pred_leaf=pred_leaf)  # fmt: skip
            return found
        p = self.predict_proba(X, num_iteration=num_iteration, start_iteration=start_iteration)[:, 1]
        return self.classes_[(p > 0.5).astype(np.int64)]
