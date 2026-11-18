"""`Booster`: a trained forest, its prediction, and the ``.igbm`` file format (docs/api/model-format.md)."""

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from idun._version import __version__
from idun.gbm.backends import get_backend
from idun.gbm.features import BinnedData, FeatureLayout
from idun.gbm.params import Params
from idun.gbm.shap import contributions, leaf_indices
from idun.gbm.split import PASS
from idun.gbm.tree import FIELDS, NODE_FIELDS, Forest, NodeForest, Trees, fields_of

FORMAT = "idun-gbm"
FORMAT_VERSION = 3  # 3: params.growth and leaf-wise node lists; 2: every categorical column's levels (1: the country)
_EXTRA = ("named_edges", "named_n_bins", "meta")
COUNTS = "leaf_count"  # each tree's in-bag rows per leaf, which TreeSHAP needs; optional in a file


@dataclass(frozen=True)
class FitReport:
    """How a fit went: wall-clock seconds per phase (upload, gradients, histograms, splits, partition, leaves).

    GPU backends also report ``gpu_seconds`` (all command buffers) and, when profiling, ``kernel_seconds``.
    """

    seconds: dict[str, float]
    backend: str
    train_rows: tuple[int, int]
    gpu_seconds: float | None = None
    kernel_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        """Sum over phases (excludes host work between phases, such as sampling)."""
        return sum(self.seconds.values())


def check_shape(params: Params, forest: Trees) -> None:
    """Raise ``ValueError`` unless ``forest`` is the kind and shape of forest ``params`` grows."""
    if params.leafwise:
        if not isinstance(forest, NodeForest) or forest.max_leaves != params.max_leaves:
            raise ValueError("a leaf-wise model needs node-list trees of params.max_leaves leaves")
        forest.check()
    elif not isinstance(forest, Forest) or forest.depth != params.max_depth:
        raise ValueError("the forest's shape does not match params.max_depth")
    if forest.n_trees != params.num_trees:
        raise ValueError("the forest's shape does not match params.num_trees")


class Booster:
    """A trained forest (depth-wise heaps or leaf-wise node lists), tied to the binning and features of its data."""

    def __init__(
        self,
        params: Params,
        forest: Trees,
        *,
        binning: str,
        feature_names: tuple[str, ...],
        categories: tuple[tuple[str, ...], ...],
        named_edges: NDArray[np.float32],
        named_n_bins: NDArray[np.uint16],
        train_rows: tuple[int, int],
        backend: str,
        fit_report: FitReport | None = None,
        best_iteration: int | None = None,
        evals: dict[str, list[float]] | None = None,
    ) -> None:
        check_shape(params, forest)
        self.params = params
        self.forest = forest
        self.binning = binning
        self.feature_names = tuple(feature_names)
        self.categories = tuple(tuple(levels) for levels in categories)
        self.named_edges = named_edges
        self.named_n_bins = named_n_bins
        self.train_rows = train_rows
        self.backend = backend
        self.fit_report = fit_report
        self.best_iteration = best_iteration  # early stopping's best tree (from 0): the model ends there
        self.evals = {} if evals is None else {name: list(scores) for name, scores in evals.items()}

    @property
    def n_trees(self) -> int:
        """Number of trees."""
        return self.forest.n_trees

    def predict(
        self, data: BinnedData, *, rows: slice, backend: str, num_trees: int | None = None, start_tree: int = 0
    ) -> NDArray[np.float32]:
        """Predict rows ``rows`` of ``data`` with ``num_trees`` trees from ``start_tree`` (all of them by default).

        Raises ``ValueError`` if ``data`` was binned differently or has other features than the training data, or if
        the trees asked for are outside the model.
        """
        lo, hi = self._rows(data, rows)
        forest = self._window(num_trees, start_tree)
        return get_backend(backend).predict(data, FeatureLayout.of(data), lo, hi, forest)

    def _rows(self, data: BinnedData, rows: slice) -> tuple[int, int]:
        """Check ``data`` against the model's binning and features; return ``rows`` as ``(lo, hi)``."""
        if data.binning != self.binning:
            raise ValueError(f"the model was trained on binning {self.binning!r}, the data uses {data.binning!r}")
        if tuple(data.feature_names) != self.feature_names or tuple(map(tuple, data.categories)) != self.categories:
            raise ValueError("the data's features or categories differ from the model's")
        if rows.step not in (None, 1):
            raise ValueError("rows must be a contiguous slice")
        lo, hi, _ = rows.indices(data.n_rows)
        return lo, max(lo, hi)

    def _window(self, num_trees: int | None, start_tree: int) -> Trees:
        """Return ``num_trees`` trees from ``start_tree`` (all of them by default)."""
        if num_trees is not None and num_trees < 0:
            raise ValueError(f"num_trees must be non-negative, got {num_trees}")
        return self.forest.trees(start_tree, self.n_trees if num_trees is None else start_tree + num_trees)

    def predict_contributions(
        self, data: BinnedData, *, rows: slice, num_trees: int | None = None, start_tree: int = 0
    ) -> NDArray[np.float64]:
        """Return each row's TreeSHAP contributions ``(n, n_features + 1)``: each feature's, then the expected value.

        They add up to the raw prediction (`idun.gbm.shap`, LightGBM's ``pred_contrib``); float64, on the host, the
        same for every backend. A model loaded from a file without leaf counts raises ``ValueError``.
        """
        lo, hi = self._rows(data, rows)
        return contributions(self._window(num_trees, start_tree), data, lo, hi, len(self.feature_names))

    def predict_leaves(
        self, data: BinnedData, *, rows: slice, num_trees: int | None = None, start_tree: int = 0
    ) -> NDArray[np.int32]:
        """Return the leaf each row reaches in each tree, ``(n, n_trees)`` (LightGBM's ``pred_leaf``), on the host.

        ``num_trees`` and ``start_tree`` as `predict`'s (as are those of `predict_contributions`).
        """
        lo, hi = self._rows(data, rows)
        return leaf_indices(self._window(num_trees, start_tree), data, lo, hi)

    def feature_importance(self, kind: str = "split") -> NDArray[np.int64]:
        """Return how many splits each feature makes, in the data's feature order (LightGBM's ``"split"``)."""
        if kind != "split":
            raise ValueError(f"kind must be 'split' (the gains are not kept with the trees), got {kind!r}")
        used = self.forest.kind != PASS
        return np.bincount(self.forest.feature[used].astype(np.int64), minlength=len(self.feature_names))

    def save(self, path: str | Path) -> None:
        """Write the model to ``path`` (an uncompressed ``.npz`` without object arrays)."""
        meta = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "idun_version": __version__,
            "params": self.params.to_dict(),
            "feature_names": list(self.feature_names),
            "categories": [list(levels) for levels in self.categories],
            "binning": self.binning,
            "growth": self.params.growth,
            "max_depth": self.params.max_depth,
            "n_trees": self.n_trees,
            "train_rows": list(self.train_rows),
            "backend": self.backend,
            "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "best_iteration": self.best_iteration,
            "evals": self.evals,
        }
        arrays: dict[str, NDArray[Any]] = {name: getattr(self.forest, name) for name in fields_of(self.forest)}
        if self.forest.leaf_count is not None:
            arrays[COUNTS] = self.forest.leaf_count
        arrays["named_edges"] = self.named_edges
        arrays["named_n_bins"] = self.named_n_bins
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        with Path(path).open("wb") as handle:
            np.savez(handle, allow_pickle=False, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a model written by `save`; anything else raises ``ValueError``."""
        with np.load(path, allow_pickle=False) as archive:
            files = sorted(archive.files)
            kept = sorted(name for name in files if name != COUNTS)  # the leaf counts: optional
            if kept not in (sorted((*FIELDS, *_EXTRA)), sorted((*NODE_FIELDS, *_EXTRA))):
                raise ValueError(f"{path}: not a .igbm v{FORMAT_VERSION} file (entries {files})")
            arrays = {name: archive[name] for name in files}
        meta = json.loads(arrays.pop("meta").tobytes())
        if meta.get("format") != FORMAT or meta.get("format_version") not in (1, 2, FORMAT_VERSION):
            raise ValueError(f"{path}: format {meta.get('format')!r} v{meta.get('format_version')!r} is not supported")
        params = Params.from_dict(meta["params"])
        forest = trees_from(params, arrays, path)
        return cls(
            params,
            forest,
            binning=meta["binning"],
            feature_names=tuple(meta["feature_names"]),
            categories=tuple(map(tuple, meta["categories"]))
            if meta["format_version"] > 1
            else (tuple(meta["countries"]),),
            named_edges=arrays["named_edges"],
            named_n_bins=arrays["named_n_bins"],
            train_rows=(int(meta["train_rows"][0]), int(meta["train_rows"][1])),
            backend=meta["backend"],
            best_iteration=meta.get("best_iteration"),  # optional in a file, as evals
            evals=meta.get("evals"),
        )


_DTYPES: dict[str, type[np.generic]] = {
    "kind": np.uint8, "feature": np.int32, "threshold": np.uint8, "missing_left": np.uint8, "cat_left": np.uint64,
    "left": np.int16, "right": np.int16, "leaf_value": np.float32,
}  # fmt: skip


def trees_from(params: Params, arrays: dict[str, NDArray[Any]], source: str | Path) -> Trees:
    """Rebuild the forest of a model trained with ``params`` from its arrays; raise ``ValueError`` on a mismatch."""
    names = NODE_FIELDS if params.leafwise else FIELDS
    if not set(names) <= set(arrays) or any(arrays[name].dtype != _DTYPES[name] for name in names):
        raise ValueError(f"{source}: the tree arrays do not match a {params.growth} model (names or dtypes)")
    counts = arrays.get(COUNTS)
    if counts is not None and (counts.dtype != np.int32 or counts.shape != arrays["leaf_value"].shape):
        raise ValueError(f"{source}: the leaf counts do not match the leaf values (dtype or shape)")
    forest = (NodeForest if params.leafwise else Forest)(**{name: arrays[name] for name in names}, leaf_count=counts)
    try:
        check_shape(params, forest)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from None
    return forest
