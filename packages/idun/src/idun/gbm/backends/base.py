"""The backend protocol: the small kernel set every device implements (docs/architecture/backend-interface.md §1).

The host (`idun.gbm.train`) owns sampling, split finding and leaf values; a backend owns the per-row state of one fit
and the passes over rows. Every backend must produce bit-identical outputs to the CPU backend.

Per-fit state, indexed by training row ``i`` (absolute panel row ``lo + i``): ``pred`` and ``g`` (float32), ``q``
(int8), ``node`` (uint8: depth-wise, the node within the current level; leaf-wise, the row's leaf id).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from idun.gbm.features import BinnedData, FeatureLayout
from idun.gbm.tree import Trees


@dataclass(frozen=True)
class LevelSplits:
    """The splits of one tree level, one entry per node of the level (pass-through nodes have kind 0)."""

    kind: NDArray[np.uint8]
    feature: NDArray[np.int32]
    threshold: NDArray[np.uint8]
    missing_left: NDArray[np.uint8]
    cat_left: NDArray[np.uint64]


@dataclass(frozen=True)
class LeafSplit:
    """One leaf-wise split: the rows of leaf ``leaf`` that go right under the split move to leaf ``right``."""

    leaf: int
    right: int
    kind: int
    feature: int
    threshold: int
    missing_left: int
    cat_left: int


class Backend(Protocol):
    """One fit's device state and passes. Construction raises `BackendUnavailableError` when it cannot run."""

    name: str

    @property
    def pred(self) -> NDArray[np.float32]:
        """Running predictions per training row (read-only on a GPU backend); ask again after a pass."""

    @property
    def g(self) -> NDArray[np.float32]:
        """Gradients per training row, to read or write before the next pass (a custom objective writes them)."""

    @property
    def h(self) -> NDArray[np.float32]:
        """Hessians per training row to write before the next pass (allocated when first asked for)."""

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, y: NDArray[np.float32]) -> None:
        """Bind the training rows ``[lo, lo + len(y))`` of ``data``; ``pred = 0``."""

    def set_predictions(self, pred: NDArray[np.float32]) -> None:
        """Start ``pred`` at these values instead of 0 (continued training), after `start`."""

    def gradients(self) -> float:
        """Set ``g = pred − y`` on every training row and return ``max |g|``."""

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scale: np.float32, half_bins: int) -> None:
        """Set ``q`` by stochastic rounding of ``g · inv_scale`` (determinism.md §2); 0 for rows out of the bag."""

    def quantize_hessians(self, key: int, inv_scale: np.float32, half_bins: int) -> None:
        """Set the hessians' ``q`` by `quantize`'s rounding of ``h · inv_scale``, after it (its bag and key)."""

    def start_tree(self) -> None:
        """Put every row in the root: ``node = 0``."""

    def histograms(
        self,
        features: NDArray[np.int32],
        offsets: NDArray[np.int32],
        cells: int,
        slots: NDArray[np.int16],
        *,
        hessians: bool = False,
    ) -> NDArray[np.int32]:
        """Integer ``(Σq, count)`` over in-bag rows, ``(n_slots, cells, 2)``; ``slots[node]`` is −1 to skip a node.

        With ``hessians``, the sums are of the hessians' ``q`` (`quantize_hessians`).
        """

    def partition(self, splits: LevelSplits) -> None:
        """Move every training row (in and out of the bag) to its child: ``node = 2·node + go_right``."""

    def split_leaf(self, split: LeafSplit) -> None:
        """Leaf-wise: move the rows (in and out of the bag) of leaf ``split.leaf`` that go right to ``split.right``."""

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Per leaf, over in-bag rows: ``Σ rint(g · 2^32)`` and the row count, exact."""

    def hessian_sums(self, n_leaves: int) -> NDArray[np.int64]:
        """Per leaf, over in-bag rows: ``Σ rint(h · 2^32)``, exact (an objective with hessians wrote ``h``)."""

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """``pred += leaf_values[node]`` on every training row, one float32 addition."""

    def predict(self, data: BinnedData, layout: FeatureLayout, lo: int, hi: int, forest: Trees) -> NDArray[np.float32]:
        """Sum of the forest's outputs for rows ``[lo, hi)``, trees added in order from 0.0."""


class ManyBackend(Protocol):
    """K models of one seed in lockstep: one bag and one feature sample, K targets.

    The same passes as `Backend` with a model axis. A backend that implements it natively can share every read of the
    feature columns across the K models; `Lockstep` runs K single-model backends instead.
    """

    name: str

    @property
    def k(self) -> int:
        """Number of models."""

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, ys: NDArray[np.float32]) -> None:
        """Bind training rows ``[lo, lo + n)`` for targets ``ys`` of shape ``(K, n)``; every ``pred = 0``."""

    def set_predictions(self, preds: NDArray[np.float32]) -> None:
        """Start every model's ``pred`` at the ``(K, n)`` values instead of 0, after `start`."""

    def views(self, model: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """``pred`` to read and ``g`` to write of one model, until the next pass (a custom objective writes ``g``)."""

    def hessians(self, model: int) -> NDArray[np.float32]:
        """``h`` of one model to write before the next pass, allocated when first asked for."""

    def gradients(self) -> NDArray[np.float64]:
        """``g = pred − y`` for every model; ``max |g|`` per model."""

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scales: NDArray[np.float32], half_bins: int) -> None:
        """Stochastic rounding of each model's ``g · inv_scales[k]`` (one bag and one key for all models)."""

    def quantize_hessians(self, key: int, inv_scales: NDArray[np.float32], half_bins: int) -> None:
        """Round each model's ``h · inv_scales[k]`` as `quantize` rounds ``g``, into its hessians' ``q``.

        A model that wrote no hessians quantizes zeros.
        """

    def start_tree(self) -> None:
        """Every row of every model to the root."""

    def histograms(
        self,
        features: NDArray[np.int32],
        offsets: NDArray[np.int32],
        cells: int,
        slots: NDArray[np.int16],
        *,
        hessians: bool = False,
    ) -> list[NDArray[np.int32]]:
        """Per model, its ``(n_slots_k, cells, 2)`` histograms; ``slots`` is ``(K, nodes)`` with −1 to skip a node.

        With ``hessians``, the sums are of the hessians' ``q``.
        """

    def partition(self, splits: list[LevelSplits]) -> None:
        """Every row of each model to its child under that model's level splits."""

    def split_leaves(self, splits: list[LeafSplit | None]) -> None:
        """Leaf-wise: each model's split of one of its leaves (None: that model does not split this step)."""

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """``(K, n_leaves)`` exact fixed-point gradient sums and counts."""

    def hessian_sums(self, n_leaves: int) -> NDArray[np.int64]:
        """``(K, n_leaves)`` exact fixed-point hessian sums."""

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """``pred[k] += leaf_values[k, node[k]]`` for every model."""


class Lockstep:
    """K single-model backends driven as one `ManyBackend`: no shared reads, the reference for batched fits."""

    def __init__(self, backends: list[Backend]) -> None:
        if not backends:
            raise ValueError("need at least one backend")
        self.backends = backends
        self.name = backends[0].name

    @property
    def k(self) -> int:
        """Number of models."""
        return len(self.backends)

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, ys: NDArray[np.float32]) -> None:
        """Start every model."""
        if ys.shape[0] != self.k:
            raise ValueError(f"{ys.shape[0]} targets for {self.k} models")
        for backend, y in zip(self.backends, ys, strict=True):
            backend.start(data, layout, lo, y)

    def set_predictions(self, preds: NDArray[np.float32]) -> None:
        """Start every model's ``pred`` at its row of ``preds``."""
        if preds.shape[0] != self.k:
            raise ValueError(f"{preds.shape[0]} prediction rows for {self.k} models")
        for backend, pred in zip(self.backends, preds, strict=True):
            backend.set_predictions(pred)

    def views(self, model: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """``pred`` and ``g`` of one model."""
        return self.backends[model].pred, self.backends[model].g

    def hessians(self, model: int) -> NDArray[np.float32]:
        """``h`` of one model."""
        return self.backends[model].h

    def gradients(self) -> NDArray[np.float64]:
        """Every model's gradients; their maxima."""
        return np.array([backend.gradients() for backend in self.backends], np.float64)

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scales: NDArray[np.float32], half_bins: int) -> None:
        """Quantize every model."""
        for backend, inv_scale in zip(self.backends, inv_scales, strict=True):
            backend.quantize(bag, key, np.float32(inv_scale), half_bins)

    def quantize_hessians(self, key: int, inv_scales: NDArray[np.float32], half_bins: int) -> None:
        """Quantize every model's hessians."""
        for backend, inv_scale in zip(self.backends, inv_scales, strict=True):
            backend.quantize_hessians(key, np.float32(inv_scale), half_bins)

    def start_tree(self) -> None:
        """Every model to the root."""
        for backend in self.backends:
            backend.start_tree()

    def histograms(
        self,
        features: NDArray[np.int32],
        offsets: NDArray[np.int32],
        cells: int,
        slots: NDArray[np.int16],
        *,
        hessians: bool = False,
    ) -> list[NDArray[np.int32]]:
        """One histogram pass per model (``hessians`` passed on only when set: a backend from before it still fits)."""
        extra = {"hessians": True} if hessians else {}
        return [b.histograms(features, offsets, cells, s, **extra) for b, s in zip(self.backends, slots, strict=True)]

    def partition(self, splits: list[LevelSplits]) -> None:
        """One partition per model."""
        for backend, level in zip(self.backends, splits, strict=True):
            backend.partition(level)

    def split_leaves(self, splits: list[LeafSplit | None]) -> None:
        """One leaf split per model that splits."""
        for backend, split in zip(self.backends, splits, strict=True):
            if split is not None:
                backend.split_leaf(split)

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Stacked leaf sums."""
        sums, counts = zip(*(backend.leaf_sums(n_leaves) for backend in self.backends), strict=True)
        return np.stack(sums), np.stack(counts)

    def hessian_sums(self, n_leaves: int) -> NDArray[np.int64]:
        """Stacked hessian sums."""
        return np.stack([backend.hessian_sums(n_leaves) for backend in self.backends])

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """Update every model."""
        for backend, values in zip(self.backends, leaf_values, strict=True):
            backend.update_predictions(values)

    @property
    def gpu_seconds(self) -> float | None:
        """Total GPU seconds of the models' backends, if they report any."""
        seconds = [getattr(backend, "gpu_seconds", None) for backend in self.backends]
        return None if any(s is None for s in seconds) else float(sum(s for s in seconds if s is not None))

    @property
    def kernel_seconds(self) -> dict[str, float]:
        """Per-kernel GPU seconds summed over the models' backends."""
        total: dict[str, float] = {}
        for backend in self.backends:
            for name, seconds in getattr(backend, "kernel_seconds", {}).items():
                total[name] = total.get(name, 0.0) + seconds
        return total
