"""The CPU backend: numba kernels, the correctness reference every other backend is compared with.

Written for exactness and clarity before speed. Every pass is a loop over rows that the Metal kernels mirror line by
line. Integer accumulations are exact in any order, so ``prange`` changes nothing in the results. Float arithmetic is
float32 throughout, with fastmath off (`idun._jit`).
"""

from collections.abc import Sequence

import numpy as np
from numba import prange
from numpy.typing import NDArray

from idun._jit import njit
from idun.gbm.backends.base import LeafSplit, LevelSplits
from idun.gbm.features import CATEGORICAL, NAMED, BinnedData, FeatureLayout
from idun.gbm.tree import NodeForest, Trees
from idun.rng import UNIFORM_BITS, row_bits

_UNIT = np.float32(2.0**-UNIFORM_BITS)
_FIXED = np.float32(2.0**32)
_ZERO = np.float32(0.0)
_MISSING = 255
_ONE = np.uint64(1)


@njit()
def _bin(
    kind: np.int8,
    column: np.int32,
    row: np.int64,
    named: NDArray[np.uint8],
    packed: NDArray[np.uint8],
    categorical: NDArray[np.uint8],
) -> np.int64:
    """Bin of panel row ``row`` in one feature."""
    if kind == NAMED:
        return np.int64(named[column, row])
    if kind == CATEGORICAL:
        return np.int64(categorical[column, row])
    return np.int64((packed[column, row >> 1] >> ((row & 1) << 2)) & 15)


@njit()
def _goes_right(
    kind: np.uint8, bin_: np.int64, threshold: np.uint8, missing_left: np.uint8, cat_left: np.uint64
) -> np.int64:
    """Child taken at an internal node: 0 = left, 1 = right. Pass-through nodes (kind 0) send every row left."""
    if kind == 0:
        return np.int64(0)
    if kind == 2:  # categorical: listed categories go left
        return np.int64(0) if (cat_left >> np.uint64(bin_)) & _ONE else np.int64(1)
    if bin_ == _MISSING:
        return np.int64(0) if missing_left else np.int64(1)
    return np.int64(1) if bin_ > threshold else np.int64(0)


@njit(parallel=True)
def _gradients(pred: NDArray[np.float32], y: NDArray[np.float32], g: NDArray[np.float32]) -> None:
    for i in prange(g.size):
        g[i] = pred[i] - y[i]


@njit(parallel=True)
def _quantize(
    g: NDArray[np.float32],
    bag: NDArray[np.uint8],
    key: np.uint64,
    lo: np.int64,
    inv_scale: np.float32,
    half: np.int64,
    q: NDArray[np.int8],
) -> None:
    for i in prange(g.size):
        if bag[i] == 0:
            q[i] = 0
            continue
        u = np.float32(row_bits(key, np.uint64(lo + i))) * _UNIT
        x = g[i] * inv_scale
        v = x + u if g[i] >= _ZERO else x - u
        t = int(v)  # truncation toward zero, as LightGBM's int8 cast
        if t > half:
            t = int(half)
        elif t < -half:
            t = -int(half)
        q[i] = t


@njit(parallel=True)
def _histograms(
    kinds: NDArray[np.int8],
    columns: NDArray[np.int32],
    offsets: NDArray[np.int32],
    named: NDArray[np.uint8],
    packed: NDArray[np.uint8],
    categorical: NDArray[np.uint8],
    lo: np.int64,
    bag: NDArray[np.uint8],
    node: NDArray[np.uint8],
    q: NDArray[np.int8],
    slots: NDArray[np.int16],
    hist: NDArray[np.int32],
) -> None:
    for j in prange(kinds.size):  # features own disjoint cells: no two threads write the same cell
        for i in range(node.size):
            slot = slots[node[i]]
            if bag[i] == 0 or slot < 0:
                continue
            cell = offsets[j] + _bin(kinds[j], columns[j], lo + i, named, packed, categorical)
            hist[slot, cell, 0] += q[i]
            hist[slot, cell, 1] += 1


@njit(parallel=True)
def _partition(
    kinds: NDArray[np.int8],
    columns: NDArray[np.int32],
    split_kind: NDArray[np.uint8],
    split_feature: NDArray[np.int32],
    threshold: NDArray[np.uint8],
    missing_left: NDArray[np.uint8],
    cat_left: NDArray[np.uint64],
    named: NDArray[np.uint8],
    packed: NDArray[np.uint8],
    categorical: NDArray[np.uint8],
    lo: np.int64,
    node: NDArray[np.uint8],
) -> None:
    for i in prange(node.size):
        at = node[i]
        f = split_feature[at]
        b = _bin(kinds[f], columns[f], lo + i, named, packed, categorical)
        node[i] = 2 * at + _goes_right(split_kind[at], b, threshold[at], missing_left[at], cat_left[at])


@njit(parallel=True)
def _split_leaf(
    kinds: NDArray[np.int8],
    columns: NDArray[np.int32],
    leaf: np.int64,
    right: np.int64,
    kind: np.uint8,
    feature: np.int64,
    threshold: np.uint8,
    missing_left: np.uint8,
    cat_left: np.uint64,
    named: NDArray[np.uint8],
    packed: NDArray[np.uint8],
    categorical: NDArray[np.uint8],
    lo: np.int64,
    node: NDArray[np.uint8],
) -> None:
    for i in prange(node.size):
        if node[i] == leaf:
            b = _bin(kinds[feature], columns[feature], lo + i, named, packed, categorical)
            if _goes_right(kind, b, threshold, missing_left, cat_left):
                node[i] = right


@njit()
def _leaf_sums(
    values: NDArray[np.float32],
    bag: NDArray[np.uint8],
    node: NDArray[np.uint8],
    sums: NDArray[np.int64],
    counts: NDArray[np.int64],
) -> None:
    for i in range(node.size):
        if bag[i]:
            sums[node[i]] += np.int64(np.rint(values[i] * _FIXED))
            counts[node[i]] += 1


@njit(parallel=True)
def _update(pred: NDArray[np.float32], node: NDArray[np.uint8], leaf_values: NDArray[np.float32]) -> None:
    for i in prange(node.size):
        pred[i] = pred[i] + leaf_values[node[i]]


@njit(parallel=True)
def _predict(
    kinds: NDArray[np.int8],
    columns: NDArray[np.int32],
    kind: NDArray[np.uint8],
    feature: NDArray[np.int32],
    threshold: NDArray[np.uint8],
    missing_left: NDArray[np.uint8],
    cat_left: NDArray[np.uint64],
    leaf_value: NDArray[np.float32],
    named: NDArray[np.uint8],
    packed: NDArray[np.uint8],
    categorical: NDArray[np.uint8],
    lo: np.int64,
    depth: np.int64,
    out: NDArray[np.float32],
) -> None:
    internal = leaf_value.shape[1] - 1
    for i in prange(out.size):
        acc = _ZERO
        for t in range(leaf_value.shape[0]):
            h = np.int64(0)
            for _ in range(depth):
                f = feature[t, h]
                b = _bin(kinds[f], columns[f], lo + i, named, packed, categorical)
                h = 2 * h + 1 + _goes_right(kind[t, h], b, threshold[t, h], missing_left[t, h], cat_left[t, h])
            acc = acc + leaf_value[t, h - internal]
        out[i] = acc


@njit(parallel=True)
def _predict_nodes(
    kinds: NDArray[np.int8],
    columns: NDArray[np.int32],
    kind: NDArray[np.uint8],
    feature: NDArray[np.int32],
    threshold: NDArray[np.uint8],
    missing_left: NDArray[np.uint8],
    cat_left: NDArray[np.uint64],
    left: NDArray[np.int16],
    right: NDArray[np.int16],
    leaf_value: NDArray[np.float32],
    named: NDArray[np.uint8],
    packed: NDArray[np.uint8],
    categorical: NDArray[np.uint8],
    lo: np.int64,
    out: NDArray[np.float32],
) -> None:
    internal = leaf_value.shape[1] - 1
    for i in prange(out.size):
        acc = _ZERO
        for t in range(leaf_value.shape[0]):
            h = np.int64(0)
            child = np.int64(-1)
            for _ in range(internal):  # every child node comes after its parent: a leaf within `internal` steps
                f = feature[t, h]
                b = _bin(kinds[f], columns[f], lo + i, named, packed, categorical)
                go = _goes_right(kind[t, h], b, threshold[t, h], missing_left[t, h], cat_left[t, h])
                child = np.int64(right[t, h]) if go else np.int64(left[t, h])
                if child < 0:
                    break
                h = child
            acc = acc + leaf_value[t, ~child]
        out[i] = acc


class CPUBackend:
    """The numba CPU backend. Always available."""

    name = "cpu"

    def __init__(self) -> None:
        self._started = False

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, y: NDArray[np.float32]) -> None:
        """Bind training rows ``[lo, lo + len(y))``."""
        n = int(y.size)
        if not 0 <= lo <= lo + n <= data.n_rows:
            raise ValueError(f"rows [{lo}, {lo + n}) are outside the panel's {data.n_rows} rows")
        self._data, self._layout, self._lo = data, layout, lo
        self.y = np.ascontiguousarray(y, dtype=np.float32)
        self.pred = np.zeros(n, np.float32)
        self.g = np.zeros(n, np.float32)
        self._h: NDArray[np.float32] | None = None
        self.q = np.zeros(n, np.int8)
        self.q_h: NDArray[np.int8] | None = None  # the hessians' q, for second-order splits
        self.node = np.zeros(n, np.uint8)
        self.bag = np.ones(n, np.uint8)
        self._started = True

    @property
    def h(self) -> NDArray[np.float32]:
        """Hessians, one per training row (allocated when first asked for)."""
        self._arrays()
        if self._h is None:
            self._h = np.zeros(self.g.size, np.float32)
        return self._h

    def _arrays(self) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
        if not self._started:
            raise RuntimeError("call start() first")
        return self._data.named_bins, self._data.obfuscated, self._data.categorical

    def set_predictions(self, pred: NDArray[np.float32]) -> None:
        """Start ``pred`` at these values instead of 0."""
        if pred.shape != self.pred.shape:
            raise ValueError(f"pred must be {self.pred.shape}, got {pred.shape}")
        self.pred[:] = pred

    def gradients(self) -> float:
        """``g = pred − y``; returns ``max |g|``."""
        self._arrays()
        _gradients(self.pred, self.y, self.g)
        return float(np.abs(self.g).max()) if self.g.size else 0.0

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scale: np.float32, half_bins: int) -> None:
        """Stochastic rounding of ``g · inv_scale`` for in-bag rows."""
        self._arrays()
        if bag.shape != self.g.shape or bag.dtype != np.uint8:
            raise ValueError("bag must be uint8 with one entry per training row")
        self.bag = bag
        _quantize(self.g, bag, np.uint64(key), np.int64(self._lo), np.float32(inv_scale), np.int64(half_bins), self.q)

    def quantize_hessians(self, key: int, inv_scale: np.float32, half_bins: int) -> None:
        """Stochastic rounding of ``h · inv_scale`` for the in-bag rows of `quantize`'s bag."""
        if self.q_h is None:
            self.q_h = np.zeros(self.q.size, np.int8)
        _quantize(self.h, self.bag, np.uint64(key), np.int64(self._lo), np.float32(inv_scale), np.int64(half_bins),
                  self.q_h)  # fmt: skip

    def start_tree(self) -> None:
        """Every row to the root."""
        self.node[:] = 0

    def histograms(
        self,
        features: NDArray[np.int32],
        offsets: NDArray[np.int32],
        cells: int,
        slots: NDArray[np.int16],
        *,
        hessians: bool = False,
    ) -> NDArray[np.int32]:
        """Integer histograms of the nodes whose slot is not −1 (of the hessians' q, with ``hessians``)."""
        named, packed, categorical = self._arrays()
        if hessians and self.q_h is None:
            raise RuntimeError("call quantize_hessians() first")
        q = self.q_h if hessians and self.q_h is not None else self.q
        n_slots = int(slots.max()) + 1 if slots.size else 0
        hist = np.zeros((n_slots, cells, 2), np.int32)
        kinds, columns = self._layout.kind[features], self._layout.column[features]
        _histograms(
            kinds,
            columns,
            offsets,
            named,
            packed,
            categorical,
            np.int64(self._lo),
            self.bag,
            self.node,
            q,
            slots,
            hist,
        )
        return hist

    def partition(self, splits: LevelSplits) -> None:
        """Every training row to its child."""
        named, packed, categorical = self._arrays()
        _partition(
            self._layout.kind,
            self._layout.column,
            splits.kind,
            splits.feature,
            splits.threshold,
            splits.missing_left,
            splits.cat_left,
            named,
            packed,
            categorical,
            np.int64(self._lo),
            self.node,
        )

    def split_leaf(self, split: LeafSplit) -> None:
        """Rows of leaf ``split.leaf`` that go right to leaf ``split.right``."""
        named, packed, categorical = self._arrays()
        _split_leaf(
            self._layout.kind,
            self._layout.column,
            np.int64(split.leaf),
            np.int64(split.right),
            np.uint8(split.kind),
            np.int64(split.feature),
            np.uint8(split.threshold),
            np.uint8(split.missing_left),
            np.uint64(split.cat_left),
            named,
            packed,
            categorical,
            np.int64(self._lo),
            self.node,
        )

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Exact fixed-point gradient sums and counts per leaf (in-bag rows)."""
        self._arrays()
        sums, counts = np.zeros(n_leaves, np.int64), np.zeros(n_leaves, np.int64)
        _leaf_sums(self.g, self.bag, self.node, sums, counts)
        return sums, counts

    def hessian_sums(self, n_leaves: int) -> NDArray[np.int64]:
        """Exact fixed-point hessian sums per leaf (in-bag rows)."""
        sums, counts = np.zeros(n_leaves, np.int64), np.zeros(n_leaves, np.int64)
        _leaf_sums(self.h, self.bag, self.node, sums, counts)
        return sums

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """``pred += leaf_values[node]``."""
        self._arrays()
        _update(self.pred, self.node, leaf_values)

    def predict(self, data: BinnedData, layout: FeatureLayout, lo: int, hi: int, forest: Trees) -> NDArray[np.float32]:
        """Forest output for rows ``[lo, hi)`` of ``data``."""
        if not 0 <= lo <= hi <= data.n_rows:
            raise ValueError(f"rows [{lo}, {hi}) are outside the panel's {data.n_rows} rows")
        out = np.zeros(hi - lo, np.float32)
        if isinstance(forest, NodeForest):
            _predict_nodes(
                layout.kind, layout.column, forest.kind, forest.feature, forest.threshold, forest.missing_left,
                forest.cat_left, forest.left, forest.right, forest.leaf_value, data.named_bins, data.obfuscated,
                data.categorical, np.int64(lo), out,
            )  # fmt: skip
            return out
        _predict(
            layout.kind,
            layout.column,
            forest.kind,
            forest.feature,
            forest.threshold,
            forest.missing_left,
            forest.cat_left,
            forest.leaf_value,
            data.named_bins,
            data.obfuscated,
            data.categorical,
            np.int64(lo),
            np.int64(forest.depth),
            out,
        )
        return out


class CPUMultiTarget:
    """The multi-target model on the CPU (experimental, not covered by the documentation).

    The reference for `MetalMultiTarget`.
    """

    name = "cpu"

    def __init__(self) -> None:
        self._started = False

    @property
    def k(self) -> int:
        """Number of targets."""
        return int(self.y.shape[0])

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, ys: NDArray[np.float32]) -> None:
        """Bind training rows ``[lo, lo + n)`` for the ``(K, n)`` targets ``ys``."""
        k, n = ys.shape
        if not 0 <= lo <= lo + n <= data.n_rows:
            raise ValueError(f"rows [{lo}, {lo + n}) are outside the panel's {data.n_rows} rows")
        self._data, self._layout, self._lo = data, layout, lo
        self.y = np.ascontiguousarray(ys, dtype=np.float32)
        self.pred, self.g = np.zeros((k, n), np.float32), np.zeros((k, n), np.float32)
        self.q = np.zeros((k, n), np.int8)
        self.node, self.bag = np.zeros(n, np.uint8), np.ones(n, np.uint8)
        self._single = CPUBackend()
        self._single.start(data, layout, lo, self.y[0])  # its node array is the shared one
        self.node = self._single.node
        self.node_own: NDArray[np.uint8] | None = None  # (K, n) after partition_own, until start_tree
        self._started = True

    def predictions(self, model: int) -> NDArray[np.float32]:
        """``pred`` of one target."""
        row: NDArray[np.float32] = self.pred[model]
        return row

    def gradients(self) -> NDArray[np.float64]:
        """``g = pred − y`` per target; ``max |g|`` per target."""
        for target in range(self.k):
            _gradients(self.pred[target], self.y[target], self.g[target])
        return np.abs(self.g).max(axis=1).astype(np.float64)

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scales: NDArray[np.float32], half_bins: int) -> None:
        """Each target's stochastic rounding (one bag and one key)."""
        if bag.shape != (self.y.shape[1],) or bag.dtype != np.uint8:
            raise ValueError("bag must be uint8 with one entry per training row")
        self.bag = bag
        self._single.bag = bag
        for target in range(self.k):
            _quantize(
                self.g[target], bag, np.uint64(key), np.int64(self._lo), np.float32(inv_scales[target]),
                np.int64(half_bins), self.q[target],
            )  # fmt: skip

    def start_tree(self) -> None:
        """Every row to the root (one shared node array again)."""
        self.node[:] = 0
        self.node_own = None

    def histograms(
        self, features: NDArray[np.int32], offsets: NDArray[np.int32], cells: int, slots: NDArray[np.int16]
    ) -> NDArray[np.int32]:
        """``(n_slots, cells, K + 1)``: per target the sums of q, then the shared count."""
        n_slots = int(slots.max()) + 1 if slots.size else 0
        out = np.zeros((n_slots, cells, self.k + 1), np.int32)
        for target in range(self.k):
            self._single.q = self.q[target]
            hist = self._single.histograms(features, offsets, cells, slots)
            out[..., target] = hist[..., 0]
            out[..., self.k] = hist[..., 1]
        return out

    def partition(self, splits: LevelSplits) -> None:
        """Every row to its child (one node array for every target)."""
        self._single.partition(splits)

    def partition_own(self, levels: Sequence[LevelSplits]) -> None:
        """Partition the last level with each target's own splits: per-target node arrays from the shared one."""
        shared = self.node.copy()
        own = np.empty((self.k, shared.size), np.uint8)
        for target, level in enumerate(levels):
            self.node[:] = shared
            self._single.partition(level)
            own[target] = self.node
        self.node[:] = shared
        self.node_own = own

    def _nodes(self, target: int) -> NDArray[np.uint8]:
        return self.node if self.node_own is None else self.node_own[target]

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """``(K, n_leaves)`` exact fixed-point sums and ``(K, n_leaves)`` counts."""
        sums = np.zeros((self.k, n_leaves), np.int64)
        counts = np.zeros((self.k, n_leaves), np.int64)
        for target in range(self.k):
            _leaf_sums(self.g[target], self.bag, self._nodes(target), sums[target], counts[target])
        return sums, counts

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """``pred[k] += leaf_values[k, node_k]``."""
        for target in range(self.k):
            _update(self.pred[target], self._nodes(target), leaf_values[target])

    def predict(self, data: BinnedData, layout: FeatureLayout, lo: int, hi: int, forest: Trees) -> NDArray[np.float32]:
        """Forest output for rows ``[lo, hi)``."""
        return CPUBackend().predict(data, layout, lo, hi, forest)
