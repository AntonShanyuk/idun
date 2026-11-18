"""`JARFRegressor`: idun_gbm behind a TabM's gradient directions (JARF, Rauniyar 2025) as a scikit-learn estimator.

``fit(X, y)`` fits ``n_surrogates`` `TabMClassifier`s on ``y``'s quantile classes (or takes fitted ones), computes the
directions of their expected gradient outer product on a seeded sample of the rows (`idun.tabm.directions`), appends
the rows' projections to the table as numeric columns ``tabm_direction_0 .. k-1`` (keeping only the ``keep`` most
sensitive non-categorical columns when asked) and fits a `GBMRegressor` on the result; ``predict(X)`` encodes,
projects and predicts. ``save`` writes one ``idun-model`` v2 file (the encoder, the directions, the kept columns and the
trees inside), which `idun.load` reads back. The trees' parameters are `GBMRegressor`'s.
"""

import inspect
import io
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray

from idun._version import __version__
from idun.base import Regressor
from idun.dataset import Column, Dataset, columns_of, resolve_columns
from idun.gbm.estimator import MODEL_FORMAT, GBMRegressor
from idun.jarf.front import PREFIX, direction_names
from idun.nn.inputs import MatrixData
from idun.nn.table import TableEncoder
from idun.tabm.directions import Directions, egop, model_digest, sensitivity, top_directions
from idun.tabm.estimator import TabMClassifier
from idun.tabm.train import TabMModel

MODEL_FORMAT_VERSION = 2  # an idun-model file with the directions and a GBMRegressor's file inside
# GBMRegressor's parameters this estimator sets itself, or does not take yet (validation sets: not in JARF's fit; the
# binning settings: JARF bins its own front table; second_order: its trees fit L2, which has no hessians; the
# per-node draws not yet: `idun.jarf`'s functions take any Params of idun_gbm)
GBM_NOT_SHARED = ("self", "categorical", "codes", "eval_metric", "early_stopping_rounds", "verbose", "importance_type",
                  "max_bin", "subsample_for_bin", "second_order", "extra_trees", "feature_fraction_bynode")  # fmt: skip


def quantile_classes(y: NDArray[np.floating], n_classes: int) -> NDArray[np.int64]:
    """Return ``y``'s classes for the surrogates.

    Its distinct values when there are at most ``n_classes`` of them, else its ``n_classes`` quantile bins (bin b holds
    the values above the b-th of the ``n_classes − 1`` inner quantiles).
    """
    if n_classes < 2:
        raise ValueError(f"surrogate_classes must be at least 2, got {n_classes}")
    distinct = np.unique(y)
    if distinct.size <= n_classes:
        return np.searchsorted(distinct, y).astype(np.int64)
    edges = np.quantile(y, np.arange(1, n_classes) / n_classes)
    return np.searchsorted(edges, y, side="right").astype(np.int64)


class JARFRegressor(Regressor):
    """Trees on a table extended with the projections on a TabM's gradient directions; see the module docstring.

    The first group of parameters is `GBMRegressor`'s; then ``surrogates`` (fitted `TabMClassifier`s to take the
    directions from, or None to fit ``n_surrogates`` with ``surrogate_params``, `TabMClassifier`'s keywords, on
    ``surrogate_backend`` and seeds ``seed``, ``seed + 1``, ...), ``surrogate_classes`` (the classes the surrogates
    train on: ``y``'s values when there are that many or fewer, else its quantile bins), ``k`` directions from the EGOP
    over ``egop_rows`` rows drawn with ``seed``, and ``keep`` (None: every column; an int: the non-categorical columns
    kept, the most sensitive by the EGOP's diagonal; categorical columns are always kept).
    """

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
        surrogates: Sequence[TabMClassifier] | None = None,
        n_surrogates: int = 1,
        surrogate_params: dict[str, Any] | None = None,
        surrogate_backend: str = "metal",
        surrogate_classes: int = 4,
        k: int = 16,
        egop_rows: int = 10_000,
        keep: int | None = None,
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
        self.surrogates = surrogates
        self.n_surrogates = n_surrogates
        self.surrogate_params = surrogate_params
        self.surrogate_backend = surrogate_backend
        self.surrogate_classes = surrogate_classes
        self.k = k
        self.egop_rows = egop_rows
        self.keep = keep

    # -- fitting ---------------------------------------------------------------------------------------------------

    def _gbm(self, categorical: Sequence[str], codes: Sequence[str]) -> GBMRegressor:
        names = [n for n in inspect.signature(GBMRegressor.__init__).parameters if n not in GBM_NOT_SHARED]
        return GBMRegressor(**{n: getattr(self, n) for n in names}, categorical=categorical, codes=codes)

    def _surrogates(self, X: Any, target: NDArray[np.float64], categorical: Sequence[str]) -> list[TabMClassifier]:  # noqa: N803
        if self.surrogates is not None:
            given = list(self.surrogates)
            if not given or any(not hasattr(s, "models_") for s in given):
                raise ValueError("surrogates must be fitted TabMClassifiers (at least one)")
            return given
        if self.n_surrogates < 1:
            raise ValueError(f"n_surrogates must be a positive integer, got {self.n_surrogates!r}")
        labels = quantile_classes(target, self.surrogate_classes)
        settings = dict(self.surrogate_params or {})
        return [
            TabMClassifier(**settings, seed=self.seed + i, backend=self.surrogate_backend, categorical=categorical).fit(
                X, labels
            )
            for i in range(self.n_surrogates)
        ]

    def _kept(self, encoder: TableEncoder) -> tuple[str, ...]:
        """Return the columns kept, in the table's order: all, or the ``keep`` most sensitive non-categorical ones."""
        if self.keep is None:
            return encoder.columns_
        numeric = encoder.numeric_
        if not 1 <= self.keep <= len(numeric):
            raise ValueError(f"keep must be in 1..{len(numeric)} (the non-categorical columns), got {self.keep!r}")
        order = np.argsort(-self.sensitivity_[: len(numeric)], kind="stable")[: self.keep]
        chosen = {numeric[i] for i in order}
        return tuple(c for c in encoder.columns_ if c in chosen or c in encoder.categorical_)

    def _table(self, X: Any, x: NDArray[np.float32]) -> list[tuple[str, NDArray[Any]]]:  # noqa: N803
        """Return the kept columns of ``X`` and the projections of its encoded rows ``x``, as a column list."""
        by_name = dict(columns_of(X, list(self.encoder_.columns_) if isinstance(X, np.ndarray) else None))
        projections = self.directions_.project(x)
        names = direction_names(self.directions_.k)
        return [*((c, by_name[c]) for c in self.kept_), *((n, projections[:, j]) for j, n in enumerate(names))]

    def fit(self, X: Any, y: Any = None) -> Self:  # noqa: N803 - scikit-learn's name
        """Fit the surrogates, the directions and the trees on ``X`` and ``y``; return the estimator."""
        self._require_y(y)
        if np.asarray(y).dtype.kind == "c":
            raise ValueError("Complex data not supported: y must be real")
        target = np.asarray(y, np.float64)
        if target.ndim != 1 or not np.isfinite(target).all():
            raise ValueError(f"y must be 1-D finite targets, got shape {target.shape}")
        names = [name for name, _ in columns_of(X, None)]
        if any(name.startswith(f"{PREFIX}_") for name in names):
            raise ValueError(f"X must not have columns named {PREFIX}_*: the projections take those names")
        categorical = resolve_columns(self.categorical, names, "categorical")
        codes = resolve_columns(self.codes, names, "codes")
        surrogates = self._surrogates(X, target, categorical)
        encoder = surrogates[0].encoder_
        if any(s.encoder_.spec() != encoder.spec() for s in surrogates[1:]):
            raise ValueError("the surrogates were fitted on different tables (their encoders differ)")
        x = encoder.transform(X)
        if x.shape[0] != target.size:
            raise ValueError(f"X has {x.shape[0]} rows and y {target.size}")
        models: list[TabMModel] = []
        for surrogate in surrogates:
            for model in surrogate.models_:
                if not isinstance(model, TabMModel):
                    raise TypeError(f"a surrogate holds a {type(model).__name__}, not a TabMModel")
                models.append(model)
        n = x.shape[0]
        if self.egop_rows < 1:
            raise ValueError(f"egop_rows must be a positive integer, got {self.egop_rows!r}")
        rows = np.sort(np.random.default_rng(self.seed).choice(n, min(self.egop_rows, n), replace=False))
        h = egop(models, MatrixData(x, np.zeros(n, np.float32), 1), rows)
        if self.k < 1:
            raise ValueError(f"k must be a positive integer, got {self.k!r}")
        vectors, eigenvalues = top_directions(h, min(self.k, x.shape[1]))  # at most the encoder's inputs
        self.directions_ = Directions(vectors.astype(np.float32), eigenvalues, tuple(model_digest(m) for m in models))
        self.sensitivity_ = sensitivity(h)
        self.encoder_ = encoder
        self.kept_ = self._kept(encoder)
        table = self._table(X, x)
        kept = set(self.kept_)
        data = Dataset(table, target, categorical=[c for c in categorical if c in kept],
                       codes=[c for c in codes if c in kept])  # fmt: skip
        self.gbm_ = self._gbm(data.categorical_features, data.obfuscated_features).fit(data)
        self.n_features_in_ = len(encoder.columns_)
        self.feature_names_in_ = np.array(encoder.columns_, dtype=object)
        return self

    # -- predicting ------------------------------------------------------------------------------------------------

    def predict(self, X: Any, *, per_seed: bool = False) -> NDArray[np.float32]:  # noqa: N803 - scikit-learn's name
        """Return the trees' prediction of each row of ``X`` (``per_seed``: each seed's, ``(n_rows, n_seeds)``)."""
        self._check_fitted("gbm_")
        self._check_n_features(X)
        return self.gbm_.predict(self._table(X, self.encoder_.transform(X)), per_seed=per_seed)

    # -- files -----------------------------------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the fitted estimator to ``path`` (an uncompressed ``.npz``, no object arrays); `idun.load` reads it."""
        self._check_fitted("gbm_")
        params = {k: list(v) if isinstance(v, tuple) else v for k, v in self.get_params().items()}
        params["surrogates"] = None  # the directions carry what the surrogates gave; the networks are not kept
        meta = {
            "format": MODEL_FORMAT, "format_version": MODEL_FORMAT_VERSION, "idun_version": __version__,
            "estimator": type(self).__name__, "params": params, "encoder": self.encoder_.spec(),
            "kept": list(self.kept_), "surrogates": list(self.directions_.surrogates),
        }  # fmt: skip
        with tempfile.TemporaryDirectory() as scratch:
            file = Path(scratch) / "gbm.npz"
            self.gbm_.save(file)
            gbm = np.frombuffer(file.read_bytes(), np.uint8)
        with Path(path).open("wb") as handle:
            np.savez(handle, allow_pickle=False, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
                     vectors=self.directions_.vectors, eigenvalues=self.directions_.eigenvalues,
                     sensitivity=self.sensitivity_, gbm=gbm)  # fmt: skip

    @classmethod
    def _from_archive(cls, meta: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> Self:
        out = cls(**meta["params"])
        out.encoder_ = TableEncoder.from_spec(meta["encoder"])
        out.directions_ = Directions(arrays["vectors"], arrays["eigenvalues"], tuple(meta["surrogates"]))
        out.sensitivity_ = arrays["sensitivity"]
        out.kept_ = tuple(meta["kept"])
        with np.load(io.BytesIO(arrays["gbm"].tobytes()), allow_pickle=False) as archive:
            inner = {name: archive[name] for name in archive.files}
        out.gbm_ = GBMRegressor._from_archive(json.loads(inner.pop("meta").tobytes()), inner)
        out.n_features_in_ = len(out.encoder_.columns_)
        out.feature_names_in_ = np.array(out.encoder_.columns_, dtype=object)
        return out
