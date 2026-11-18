"""idun_gbm's GPU passes, written once against `idun.device.Device`: which kernels run, over which buffers.

`DeviceMany` fits K models of one seed in lockstep: every per-row array has a model axis, the bag and the feature
columns are shared, and the obfuscated-feature histogram pass reads each block of codes once for all the (model, slot)
pairs of a chunk. `DeviceBackend` is the single-model face of it (K = 1); `DeviceMultiTarget` shares one tree among
K targets. Every pass equals the CPU backend's bit for bit, on every device.

- **Kernels:** `TABLE` names each kernel's buffers, the ones it writes and its parameter structure; a device's kernel
  source implements these names with these buffer orders.
- **Host access:** only through the device's ``read``, ``write`` and ``fill``, so that a device with memory of its
  own knows what to copy and when.
- **Batches:** passes that return nothing to the host are encoded lazily; a pass whose result the host needs waits for
  the encoded ones. A tree therefore costs about ``max_depth + 2`` round trips, whatever K.
"""

import ctypes
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from idun.device import Buffer, Device, Signature
from idun.gbm.backends.base import LeafSplit, LevelSplits
from idun.gbm.features import OBFUSCATED, OBFUSCATED_BINS, BinnedData, FeatureLayout
from idun.gbm.tree import NodeForest, Trees, fields_of

THREADS = 256
LEAF_ROWS_PER_GROUP = 4096  # bounds the 32-bit partials of leaf_partials (see the kernel)
LANES = 32
POPCOUNT_MAX_SLOTS = 2  # the bitmask path when every model has at most this many slots (4 measured slower)
POPCOUNT_PAIRS = 8  # (model, slot) pairs per chunk on the bitmask path: 10 KiB of counters per SIMD group
ROW_PAIRS = 16  # pairs per chunk on the per-row path: 10 KiB of counters per SIMD group
SINGLE_SLOTS_PER_CHUNK = 16  # histograms_obfuscated_one's per-row path: 10 KiB of threadgroup memory per SIMD group
OBFUSCATED_ROWS_PER_GROUP = 16384  # at most; also <= 32767 / half_bins so the packed 16-bit sums cannot overflow
MAX_LEVEL_NODES = 256
SHARED_ROWS_PER_GROUP = 65536  # rows a threadgroup of build_histograms_shared counts
SHARED_HISTOGRAM_BYTES = 32768  # its threadgroup memory at most: a chunk of slots of one feature's cells
NO_LEAF = 256  # split_leaf: a leaf id no row holds (node ids are uint8)


def _u32(*names: str) -> tuple[tuple[str, type[ctypes.c_uint32]], ...]:
    return tuple((name, ctypes.c_uint32) for name in names)


class CountParams(ctypes.Structure):
    """Mirror of the kernels' ``CountParams``."""

    _fields_ = _u32("n")


class RngParams(ctypes.Structure):
    """Mirror of ``RngParams``."""

    _fields_ = _u32("key", "first_row", "n")


class ModelParams(ctypes.Structure):
    """Mirror of ``ModelParams``."""

    _fields_ = _u32("n", "k")


class QuantizeParams(ctypes.Structure):
    """Mirror of ``QuantizeParams``."""

    _fields_ = (*_u32("n", "first_row", "key"), ("half_bins", ctypes.c_int32))


class HistogramParams(ctypes.Structure):
    """Mirror of ``HistogramParams``."""

    _fields_ = _u32("n", "first_row", "n_features", "cells", "named_stride", "packed_stride", "width", "cat_stride")


class SharedHistogramParams(ctypes.Structure):
    """Mirror of ``SharedHistogramParams``."""

    _fields_ = _u32(
        "n", "first_row", "n_features", "cells", "named_stride", "packed_stride", "width", "cat_stride",
        "rows_per_group", "slots_per_chunk", "chunks",
    )  # fmt: skip


class ObfuscatedParams(ctypes.Structure):
    """Mirror of ``ObfuscatedParams``."""

    _fields_ = _u32(
        "n", "first_row", "n_features", "cells", "packed_stride", "rows_per_group", "width", "q_bits", "popcount",
        "max_pairs",
    )  # fmt: skip


class SingleObfuscatedParams(ctypes.Structure):
    """Mirror of ``SingleObfuscatedParams`` (``histograms_obfuscated_one``, the K = 1 kernel)."""

    _fields_ = _u32(
        "n", "first_row", "n_features", "cells", "packed_stride", "rows_per_group", "slots_per_chunk", "n_slots",
        "q_bits", "popcount",
    )  # fmt: skip


class Entry(ctypes.Structure):
    """Mirror of ``Entry``: model ``model``'s slots ``[first_slot, end_slot)`` from local pair ``pair_base``."""

    _fields_ = _u32("model", "first_slot", "end_slot", "pair_base")


class PartitionParams(ctypes.Structure):
    """Mirror of ``PartitionParams``."""

    _fields_ = _u32("n", "first_row", "named_stride", "packed_stride", "width", "cat_stride")


class LeafSplitParams(ctypes.Structure):
    """Mirror of ``LeafSplitParams``."""

    _fields_ = _u32("n", "first_row", "named_stride", "packed_stride", "cat_stride")


class LeafParams(ctypes.Structure):
    """Mirror of ``LeafParams``."""

    _fields_ = _u32("n", "n_leaves", "rows_per_group", "groups", "node_stride", "residual")


class UpdateParams(ctypes.Structure):
    """Mirror of ``UpdateParams``."""

    _fields_ = _u32("n", "n_leaves", "node_stride")


class MultiParams(ctypes.Structure):
    """Mirror of ``MultiParams``."""

    _fields_ = _u32(
        "n", "first_row", "n_features", "cells", "named_stride", "packed_stride", "rows_per_group", "k", "half_bins",
        "slots_per_chunk", "n_slots", "targets_per_chunk", "target_chunks", "cat_stride",
    )  # fmt: skip


class PredictParams(ctypes.Structure):
    """Mirror of ``PredictParams``."""

    _fields_ = _u32("n", "first_row", "n_trees", "depth", "named_stride", "packed_stride", "cat_stride")


class NodePredictParams(ctypes.Structure):
    """Mirror of ``NodePredictParams``."""

    _fields_ = _u32("n", "first_row", "n_trees", "max_leaves", "named_stride", "packed_stride", "cat_stride")


def chunks(
    n_slots: NDArray[np.int64], cap: int, *, share: bool = True
) -> tuple[list[tuple[int, int, int, int]], list[int]]:
    """Pack every model's slots into chunks of at most ``cap`` (model, slot) pairs.

    Returns the entries ``(model, first_slot, end_slot, pair_base)`` and the index of each chunk's first entry (plus
    the end). A model whose slots do not fit the current chunk continues in the next one. Without ``share``, every
    model starts a new chunk (its chunks then have one entry: the kernel's register path).
    """
    entries: list[tuple[int, int, int, int]] = []
    starts = [0]
    used = 0
    for model, count in enumerate(int(c) for c in n_slots):
        if not share and used:
            starts.append(len(entries))
            used = 0
        slot = 0
        while slot < count:
            if used == cap:
                starts.append(len(entries))
                used = 0
            take = min(count - slot, cap - used)
            entries.append((model, slot, slot + take, used))
            used += take
            slot += take
    starts.append(len(entries))
    return entries, starts


def _groups(n: int, width: int = THREADS) -> tuple[int, int, int]:
    return (max(-(-n // width), 1), 1, 1)


TABLE: dict[str, Signature] = {
    "rng_uniform_bits": Signature(1, (0,), RngParams),
    "fma_probe": Signature(4, (3,), CountParams),
    "compute_gradients": Signature(4, (2, 3), ModelParams),
    "quantize_gradients": Signature(4, (2,), QuantizeParams),
    "max_residuals": Signature(3, (2,), ModelParams),
    "quantize_residuals": Signature(5, (3,), QuantizeParams),
    "build_histograms": Signature(12, (11,), HistogramParams),
    "build_histograms_shared": Signature(13, (12,), SharedHistogramParams),
    "histograms_obfuscated": Signature(11, (10,), ObfuscatedParams),
    "histograms_obfuscated_one": Signature(8, (7,), SingleObfuscatedParams),
    "partition_rows": Signature(11, (10,), PartitionParams),
    "partition_own": Signature(12, (11,), PartitionParams),
    "split_leaf": Signature(12, (11,), LeafSplitParams),
    "leaf_partials": Signature(6, (4, 5), LeafParams),
    "update_predictions": Signature(3, (0,), UpdateParams),
    "predict_trees": Signature(12, (11,), PredictParams),
    "predict_nodes": Signature(14, (13,), NodePredictParams),
    "build_histograms_multi": Signature(11, (10,), MultiParams),
    "row_slots": Signature(4, (3,), CountParams),
    "histograms_multi": Signature(6, (5,), MultiParams),
}
"""Every kernel of idun_gbm: its buffers in binding order, the indices of the ones it writes, its parameters."""

BLOCK = (THREADS, 1, 1)  # threads per group of the per-row kernels
LANE_BLOCK = (LANES, 1, 1)  # of the lane kernels (one SIMD group or warp per group)


def _leaf_total(parts: NDArray[np.uint32]) -> NDArray[np.int64]:
    """Add leaf_partials' (model, group, leaf, [hi, mid, low]) partials exactly over the groups (axis 1)."""
    hi = parts[..., 0].view(np.int32).astype(np.int64)
    total = hi * (1 << 32) + parts[..., 1].astype(np.int64) * (1 << 16) + parts[..., 2].astype(np.int64)
    return total.sum(axis=1, dtype=np.int64)


class DeviceMany:
    """K models of one seed in lockstep on ``device`` (`ManyBackend`): shared bag and columns, shared code reads.

    The backend owns ``device`` and closes it. A subclass names the backend (``name``). A device's own scheduling
    changes the time only, never a sum: ``rows_per_group`` bounds the rows a threadgroup of the single-model obfuscated
    pass reads (K = 1; `OBFUSCATED_ROWS_PER_GROUP` by default), and ``shared_named`` counts the named and categorical
    features in threadgroup memory first (``build_histograms_shared``) instead of row by row in the histogram.
    """

    name: str

    def __init__(
        self, device: Device, k: int, *, rows_per_group: int | None = None, shared_named: bool = False
    ) -> None:
        self._device = device
        if k < 1:
            raise ValueError(f"need at least one model, got {k}")
        if rows_per_group is not None and rows_per_group < 32:
            raise ValueError(f"rows_per_group must be at least 32, got {rows_per_group}")
        self._rows_per_group = rows_per_group
        self._shared_named = shared_named
        self._k = k
        self._n = 0
        self._hist: Buffer | None = None
        self._entries: Buffer | None = None
        self._partials: tuple[Buffer, Buffer] | None = None
        self._h: Buffer | None = None  # hessians: only for an objective that has them
        self._q_h: Buffer | None = None  # their q, for second-order splits
        self._inv_h: Buffer | None = None
        self._half_bins = 64  # the widest quantization until quantize() says otherwise

    @property
    def k(self) -> int:
        """Number of models."""
        return self._k

    @property
    def placements(self) -> dict[str, str]:
        """How each data array reached the device.

        ``"wrapped"`` (Metal: no copy), ``"copied"`` (Metal: an unaligned array copied first), ``"uploaded"`` (CUDA:
        as it lies), ``"repacked"`` (CUDA: its rows realigned first) or ``"empty"``.
        """
        return dict(self._device.placements)

    @property
    def gpu_seconds(self) -> float:
        """GPU time of every committed batch so far."""
        return self._device.gpu_seconds

    @property
    def kernel_seconds(self) -> dict[str, float]:
        """GPU seconds per kernel; filled only by a profiling backend (every pass runs on its own)."""
        return dict(self._device.kernel_seconds)

    def _data(self, data: BinnedData, layout: FeatureLayout) -> None:
        device = self._device
        self._named, self._named_stride = device.place("named_bins", data.named_bins)
        self._packed, self._packed_stride = device.place("obfuscated", data.obfuscated)
        self._country, self._cat_stride = device.place("categorical", data.categorical)
        self._kinds = device.copy_in(layout.kind.astype(np.int32))
        self._columns = device.copy_in(layout.column)

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, ys: NDArray[np.float32]) -> None:
        """Bind training rows ``[lo, lo + n)`` for the ``(K, n)`` targets ``ys``; every ``pred = 0``."""
        if ys.ndim != 2 or ys.shape[0] != self._k:
            raise ValueError(f"ys must be ({self._k}, n), got {ys.shape}")
        k, n = ys.shape
        if not 0 <= lo <= lo + n <= data.n_rows:
            raise ValueError(f"rows [{lo}, {lo + n}) are outside the panel's {data.n_rows} rows")
        self._layout, self._lo, self._n = layout, lo, n
        self._data(data, layout)
        device = self._device
        self._y = device.copy_in(np.asarray(ys, np.float32))
        self._pred, self._g = device.alloc(4 * k * n), device.alloc(4 * k * n)
        self._h = self._q_h = self._inv_h = None
        self._q, self._node = device.alloc(k * n), device.alloc(k * n)
        self._bag = device.alloc(n)
        device.fill(self._bag, 1, n)
        self._max, self._inv = device.alloc(4 * k), device.alloc(4 * k)
        n_features = layout.n_features
        self._f_kinds, self._f_columns, self._f_offsets = (device.alloc(4 * n_features) for _ in range(3))
        self._f_widths = device.alloc(4 * n_features)
        self._o_columns, self._o_offsets = (device.alloc(4 * n_features) for _ in range(2))
        nodes = k * MAX_LEVEL_NODES
        self._slots, self._slot_base = device.alloc(4 * nodes), device.alloc(4 * k)
        self._chunk_start = device.alloc(4 * (nodes + 1))
        self._split_kind, self._split_threshold, self._split_missing = (device.alloc(nodes) for _ in range(3))
        self._split_feature, self._split_cat = device.alloc(4 * nodes), device.alloc(8 * nodes)
        self._split_leaves = device.alloc(8 * k)  # split_leaf: (leaf, right) per model
        self._leaf_values = device.alloc(4 * nodes)

    def views(self, model: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """``pred`` (read-only) and ``g`` (to write) of one model; valid until the next pass (ask again after it)."""
        return self.predictions(model), self.gradient_values(model)

    def predictions(self, model: int) -> NDArray[np.float32]:
        """``pred`` of one model, read-only (encoded passes finish first)."""
        self._device.flush()
        pred: NDArray[np.float32] = self._device.read(self._pred, np.float32, (self._k, self._n))[model]
        return pred

    def set_predictions(self, preds: NDArray[np.float32]) -> None:
        """Start every model's ``pred`` at the ``(K, n)`` values instead of 0 (encoded passes finish first)."""
        if preds.shape != (self._k, self._n):
            raise ValueError(f"preds must be {(self._k, self._n)}, got {preds.shape}")
        self._device.flush()
        self._device.write(self._pred, np.float32, (self._k, self._n))[:] = preds

    def gradient_values(self, model: int) -> NDArray[np.float32]:
        """``g`` of one model, for the host to read or write before the next pass (encoded passes finish first)."""
        self._device.flush()
        g: NDArray[np.float32] = self._device.write(self._g, np.float32, (self._k, self._n))[model]
        return g

    def hessians(self, model: int) -> NDArray[np.float32]:
        """``h`` of one model to write (allocated when first asked for; encoded passes finish first)."""
        self._device.flush()
        if self._h is None:
            self._h = self._device.alloc(4 * self._k * self._n)
        h: NDArray[np.float32] = self._device.write(self._h, np.float32, (self._k, self._n))[model]
        return h

    def _grid(self, width: int = THREADS) -> tuple[int, int, int]:
        return (max(-(-self._n // width), 1), self._k, 1)

    def gradients(self) -> NDArray[np.float64]:
        """``g = pred − y`` for every model; ``max |g|`` per model."""
        device = self._device
        device.fill(self._max, 0, 4 * self._k)
        buffers = [self._pred, self._y, self._g, self._max]
        device.encode("compute_gradients", buffers, ModelParams(self._n, self._k), self._grid(), BLOCK)
        device.flush()
        return device.read(self._max, np.uint32, (self._k,)).view(np.float32).astype(np.float64)

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scales: NDArray[np.float32], half_bins: int) -> None:
        """Stochastic rounding of each model's ``g · inv_scales[k]`` for in-bag rows."""
        if bag.shape != (self._n,) or bag.dtype != np.uint8:
            raise ValueError("bag must be uint8 with one entry per training row")
        device = self._device
        device.write(self._bag, np.uint8, (self._n,))[:] = bag
        device.write(self._inv, np.float32, (self._k,))[:] = inv_scales
        self._half_bins = half_bins
        params = QuantizeParams(self._n, self._lo, key, half_bins)
        device.encode("quantize_gradients", [self._g, self._bag, self._q, self._inv], params, self._grid(), BLOCK)

    def quantize_hessians(self, key: int, inv_scales: NDArray[np.float32], half_bins: int) -> None:
        """Round each model's ``h · inv_scales[k]`` into the hessians' q as `quantize` rounds ``g`` (its bag).

        The gradients' kernel on the hessians' buffers; a model that wrote no hessians quantizes zeros.
        """
        if half_bins != self._half_bins:
            raise ValueError(f"the hessians take the gradients' {self._half_bins} half bins, got {half_bins}")
        device = self._device
        if self._h is None:
            device.flush()
            self._h = device.alloc(4 * self._k * self._n)
            device.fill(self._h, 0, 4 * self._k * self._n)
        if self._q_h is None or self._inv_h is None:
            self._q_h, self._inv_h = device.alloc(self._k * self._n), device.alloc(4 * self._k)
        device.write(self._inv_h, np.float32, (self._k,))[:] = inv_scales  # own buffer: quantize's pass reads _inv
        params = QuantizeParams(self._n, self._lo, key, half_bins)
        device.encode("quantize_gradients", [self._h, self._bag, self._q_h, self._inv_h], params, self._grid(), BLOCK)

    def start_tree(self) -> None:
        """Every row of every model to the root."""
        self._device.fill(self._node, 0, self._k * self._n)

    def histograms(
        self,
        features: NDArray[np.int32],
        offsets: NDArray[np.int32],
        cells: int,
        slots: NDArray[np.int16],
        *,
        hessians: bool = False,
    ) -> list[NDArray[np.int32]]:
        """Per model, the integer histograms of the nodes whose slot is not −1 (``slots`` is ``(K, nodes)``).

        With ``hessians``, of the hessians' q (`quantize_hessians`): the same kernels on the other buffer.
        """
        if hessians and self._q_h is None:
            raise RuntimeError("call quantize_hessians() first")
        q = self._q_h if hessians and self._q_h is not None else self._q
        device = self._device
        k, width = slots.shape
        n_slots = np.where(slots.max(axis=1) >= 0, slots.max(axis=1).astype(np.int64) + 1, 0)
        base = np.concatenate([[0], np.cumsum(n_slots)]).astype(np.int64)
        total = int(base[-1])
        size = max(total * cells * 2 * 4, 4)
        if self._hist is None or self._hist.nbytes < size:
            device.flush()
            self._hist = device.alloc(size)
        popcount = bool(n_slots.max(initial=0) <= POPCOUNT_MAX_SLOTS)
        # Sharing a chunk across models pays on the bitmask path (the code masks are computed once for all of them);
        # on the per-row path it only adds work (measured), so there each model runs its own one-entry chunks.
        entries, starts = chunks(n_slots, POPCOUNT_PAIRS if popcount else ROW_PAIRS, share=popcount)
        if entries and (self._entries is None or self._entries.nbytes < 16 * len(entries)):
            device.flush()
            self._entries = device.alloc(16 * max(len(entries), 64))
        obfuscated = self._layout.kind[features] == OBFUSCATED
        others, packed_features = features[~obfuscated], features[obfuscated]
        device.write(self._f_kinds, np.int32, (others.size,))[:] = self._layout.kind[others]
        device.write(self._f_columns, np.int32, (others.size,))[:] = self._layout.column[others]
        device.write(self._f_offsets, np.int32, (others.size,))[:] = offsets[~obfuscated]
        device.write(self._o_columns, np.int32, (packed_features.size,))[:] = self._layout.column[packed_features]
        device.write(self._o_offsets, np.int32, (packed_features.size,))[:] = offsets[obfuscated]
        device.write(self._slots, np.int32, (k, width))[:] = slots
        device.write(self._slot_base, np.int32, (k,))[:] = base[:-1]
        device.fill(self._hist, 0, total * cells * 2 * 4)
        if others.size and total and self._shared_named:
            self._shared_histograms(others, cells, width, int(n_slots.max()), self._hist, q)
        elif others.size and total:
            params = HistogramParams(
                self._n, self._lo, others.size, cells, self._named_stride, self._packed_stride, width, self._cat_stride
            )
            buffers = [
                self._f_kinds, self._f_columns, self._f_offsets, self._named, self._packed, self._country,
                self._bag, self._node, q, self._slots, self._slot_base, self._hist,
            ]  # fmt: skip
            device.encode("build_histograms", buffers, params, self._grid(), BLOCK)
        if packed_features.size and total and k == 1:
            self._single_obfuscated(packed_features.size, cells, int(n_slots[0]), self._hist, q, popcount=popcount)
        elif packed_features.size and entries and self._entries is not None:
            device.write(self._entries, np.uint32, (len(entries), 4))[:] = entries  # one `Entry` per row
            device.write(self._chunk_start, np.uint32, (len(starts),))[:] = starts
            pairs = max(entries[stop - 1][3] + entries[stop - 1][2] - entries[stop - 1][1] for stop in starts[1:])
            per_pair = OBFUSCATED_BINS * (2 if popcount else 1)
            rows = min(OBFUSCATED_ROWS_PER_GROUP, 32767 // max(self._half_bins, 1)) // 32 * 32
            lane_params = ObfuscatedParams(
                self._n, self._lo, packed_features.size, cells, self._packed_stride, rows, width,
                max(self._half_bins, 1).bit_length(), int(popcount), pairs,
            )  # fmt: skip
            groups = (-(-packed_features.size // LANES), max(-(-self._n // rows), 1), len(starts) - 1)
            buffers = [
                self._o_columns, self._o_offsets, self._packed, self._bag, self._node, q, self._slots,
                self._slot_base, self._entries, self._chunk_start, self._hist,
            ]  # fmt: skip
            local_bytes = -(-(pairs * per_pair + pairs) * LANES * 4 // 16) * 16
            device.encode("histograms_obfuscated", buffers, lane_params, groups, LANE_BLOCK, local_bytes)
        device.flush()
        hist = device.read(self._hist, np.int32, (total, cells, 2))
        return [hist[base[model] : base[model + 1]].copy() for model in range(k)]

    def _shared_histograms(
        self, others: NDArray[np.int32], cells: int, width: int, n_slots: int, hist: Buffer, q: Buffer
    ) -> None:
        """Encode the named and categorical features' histograms with ``build_histograms_shared``.

        A threadgroup counts one feature's cells for a chunk of slots (as many as `SHARED_HISTOGRAM_BYTES` holds of the
        widest feature) over `SHARED_ROWS_PER_GROUP` rows; the grid's z axis runs over the models and their chunks.
        """
        widths = self._layout.hist_bins[others]
        widest = int(widths.max())
        per_chunk = max(min(n_slots, SHARED_HISTOGRAM_BYTES // (widest * 8)), 1)
        n_chunks = -(-n_slots // per_chunk)
        self._device.write(self._f_widths, np.int32, (others.size,))[:] = widths
        params = SharedHistogramParams(
            self._n, self._lo, others.size, cells, self._named_stride, self._packed_stride, width, self._cat_stride,
            SHARED_ROWS_PER_GROUP, per_chunk, n_chunks,
        )  # fmt: skip
        buffers = [
            self._f_kinds, self._f_columns, self._f_offsets, self._f_widths, self._named, self._packed, self._country,
            self._bag, self._node, q, self._slots, self._slot_base, hist,
        ]  # fmt: skip
        groups = (max(-(-self._n // SHARED_ROWS_PER_GROUP), 1), others.size, self._k * n_chunks)
        local_bytes = -(-per_chunk * widest * 8 // 16) * 16  # whole 16-byte lines, as Metal requires
        self._device.encode("build_histograms_shared", buffers, params, groups, BLOCK, local_bytes)

    def _single_obfuscated(
        self, n_features: int, cells: int, n_slots: int, hist: Buffer, q: Buffer, *, popcount: bool
    ) -> None:
        """One model's obfuscated histograms with ``histograms_obfuscated_one`` (the K-model kernel's sums, faster).

        The K-model kernel's chunk entries and per-model words cost a single fit about 20 % of its histogram time, so
        K = 1 keeps the single-model kernel. The buffers are the K-model ones: with one model the slots, node ids,
        gradients and histogram rows are laid out as that kernel expects.
        """
        per_chunk = n_slots if popcount else min(SINGLE_SLOTS_PER_CHUNK, n_slots)
        bound = OBFUSCATED_ROWS_PER_GROUP if self._rows_per_group is None else self._rows_per_group
        rows = min(bound, 32767 // max(self._half_bins, 1)) // 32 * 32
        params = SingleObfuscatedParams(
            self._n, self._lo, n_features, cells, self._packed_stride, rows, per_chunk, n_slots,
            max(self._half_bins, 1).bit_length(), int(popcount),
        )  # fmt: skip
        groups = (-(-n_features // LANES), max(-(-self._n // rows), 1), -(-n_slots // per_chunk))
        buffers = [self._o_columns, self._o_offsets, self._packed, self._bag, self._node, q, self._slots,
                   hist]  # fmt: skip
        local_bytes = per_chunk * OBFUSCATED_BINS * (2 if popcount else 1) * LANES * 4
        self._device.encode("histograms_obfuscated_one", buffers, params, groups, LANE_BLOCK, local_bytes)

    def partition(self, splits: list[LevelSplits]) -> None:
        """Every row of every model to its child."""
        device = self._device
        buffers = [self._split_kind, self._split_feature, self._split_threshold, self._split_missing, self._split_cat]
        width = splits[0].kind.size
        for buffer, name in zip(buffers, ("kind", "feature", "threshold", "missing_left", "cat_left"), strict=True):
            values = np.stack([getattr(level, name) for level in splits])
            device.write(buffer, values.dtype.type, values.shape)[...] = values
        params = PartitionParams(self._n, self._lo, self._named_stride, self._packed_stride, width, self._cat_stride)
        all_buffers = [self._kinds, self._columns, *buffers, self._named, self._packed, self._country, self._node]
        device.encode("partition_rows", all_buffers, params, self._grid(), BLOCK)

    def split_leaves(self, splits: list[LeafSplit | None]) -> None:
        """Leaf-wise: per model, the rows of its split leaf that go right to the split's new leaf (None: no split)."""
        if len(splits) != self._k:
            raise ValueError(f"{len(splits)} splits for {self._k} models")
        device = self._device
        buffers = [self._split_kind, self._split_feature, self._split_threshold, self._split_missing, self._split_cat]
        names = ("kind", "feature", "threshold", "missing_left", "cat_left")
        dtypes: tuple[type[np.generic], ...] = (np.uint8, np.int32, np.uint8, np.uint8, np.uint64)
        for buffer, name, dtype in zip(buffers, names, dtypes, strict=True):
            values = [0 if split is None else getattr(split, name) for split in splits]
            device.write(buffer, dtype, (self._k,))[:] = values
        leaves = device.write(self._split_leaves, np.uint32, (self._k, 2))
        leaves[:] = [(NO_LEAF, NO_LEAF) if split is None else (split.leaf, split.right) for split in splits]
        params = LeafSplitParams(self._n, self._lo, self._named_stride, self._packed_stride, self._cat_stride)
        all_buffers = [self._kinds, self._columns, *buffers, self._split_leaves, self._named, self._packed,
                       self._country, self._node]  # fmt: skip
        device.encode("split_leaf", all_buffers, params, self._grid(), BLOCK)

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """``(K, n_leaves)`` exact fixed-point gradient sums and counts (in-bag rows)."""
        return self._leaf_sums(self._g, n_leaves)

    def hessian_sums(self, n_leaves: int) -> NDArray[np.int64]:
        """``(K, n_leaves)`` exact fixed-point hessian sums (in-bag rows)."""
        if self._h is None:
            raise RuntimeError("no hessians were written (call hessians() first)")
        return self._leaf_sums(self._h, n_leaves)[0]

    def _leaf_sums(self, values: Buffer, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Per model and leaf, the exact sums of ``rint(values · 2^32)`` over in-bag rows, and the row counts."""
        device = self._device
        groups = max(-(-self._n // LEAF_ROWS_PER_GROUP), 1)
        cells = self._k * groups * n_leaves
        if self._partials is None or self._partials[1].nbytes < 4 * cells:
            device.flush()
            self._partials = (device.alloc(12 * cells), device.alloc(4 * cells))  # hi, mid, low per leaf
        sums, counts = self._partials
        buffers = [values, values, self._bag, self._node, sums, counts]
        params = LeafParams(self._n, n_leaves, LEAF_ROWS_PER_GROUP, groups, self._n, 0)
        device.encode("leaf_partials", buffers, params, (groups, self._k, 1), BLOCK)
        device.flush()
        shape = (self._k, groups, n_leaves)
        return (
            _leaf_total(device.read(sums, np.uint32, (*shape, 3))),
            device.read(counts, np.uint32, shape).sum(axis=1, dtype=np.int64),
        )

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """``pred[k] += leaf_values[k, node[k]]``."""
        device = self._device
        device.write(self._leaf_values, np.float32, leaf_values.shape)[...] = leaf_values
        params = UpdateParams(self._n, leaf_values.shape[1], self._n)
        device.encode("update_predictions", [self._pred, self._node, self._leaf_values], params, self._grid(), BLOCK)

    def predict(self, data: BinnedData, layout: FeatureLayout, lo: int, hi: int, forest: Trees) -> NDArray[np.float32]:
        """Forest output for rows ``[lo, hi)`` of ``data``."""
        if not 0 <= lo <= hi <= data.n_rows:
            raise ValueError(f"rows [{lo}, {hi}) are outside the panel's {data.n_rows} rows")
        device = self._device
        device.flush()
        self._data(data, layout)
        n = hi - lo
        out = device.alloc(4 * max(n, 1))
        trees = [device.copy_in(getattr(forest, name)) for name in fields_of(forest)]
        if isinstance(forest, NodeForest):
            kernel = "predict_nodes"
            params: ctypes.Structure = NodePredictParams(
                n, lo, forest.n_trees, forest.max_leaves, self._named_stride, self._packed_stride, self._cat_stride
            )
        else:
            kernel = "predict_trees"
            params = PredictParams(
                n, lo, forest.n_trees, forest.depth, self._named_stride, self._packed_stride, self._cat_stride
            )
        if n and forest.n_trees:
            buffers = [self._kinds, self._columns, *trees, self._named, self._packed, self._country, out]
            device.encode(kernel, buffers, params, _groups(n), BLOCK)
            device.flush()
        return device.read(out, np.float32, (max(n, 1),))[:n].copy()

    def close(self) -> None:
        """Release the device (also done when the backend is garbage-collected)."""
        self._device.close()

    def __del__(self) -> None:
        device = getattr(self, "_device", None)
        if device is not None:
            device.close()


class DeviceBackend:
    """One model on ``device`` (`Backend`): `DeviceMany` with K = 1. A subclass names the backend (``name``)."""

    name: str

    def __init__(self, device: Device, *, rows_per_group: int | None = None, shared_named: bool = False) -> None:
        self._many = DeviceMany(device, 1, rows_per_group=rows_per_group, shared_named=shared_named)
        self._device = device

    @property
    def placements(self) -> dict[str, str]:
        """How each data array reached the device."""
        return self._many.placements

    @property
    def gpu_seconds(self) -> float:
        """GPU time of every committed batch so far."""
        return self._many.gpu_seconds

    @property
    def kernel_seconds(self) -> dict[str, float]:
        """GPU seconds per kernel (profiling backends)."""
        return self._many.kernel_seconds

    @property
    def _node(self) -> Buffer:
        return self._many._node

    @property
    def _q(self) -> Buffer:
        return self._many._q

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, y: NDArray[np.float32]) -> None:
        """Bind training rows ``[lo, lo + len(y))``; ``pred = 0``."""
        self._many.start(data, layout, lo, np.asarray(y, np.float32)[None])

    def set_predictions(self, pred: NDArray[np.float32]) -> None:
        """Start ``pred`` at these values instead of 0."""
        self._many.set_predictions(np.asarray(pred, np.float32)[None])

    @property
    def pred(self) -> NDArray[np.float32]:
        """Running predictions, read-only (encoded passes finish first; ask again after the next pass)."""
        return self._many.predictions(0)

    @property
    def g(self) -> NDArray[np.float32]:
        """Gradients, for the host to read or write before the next pass (encoded passes finish first)."""
        return self._many.gradient_values(0)

    @property
    def h(self) -> NDArray[np.float32]:
        """Hessians to write (allocated when first asked for; encoded passes finish first)."""
        return self._many.hessians(0)

    def gradients(self) -> float:
        """``g = pred − y``; returns ``max |g|``."""
        return float(self._many.gradients()[0])

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scale: np.float32, half_bins: int) -> None:
        """Stochastic rounding of ``g · inv_scale`` for in-bag rows."""
        self._many.quantize(bag, key, np.array([inv_scale], np.float32), half_bins)

    def quantize_hessians(self, key: int, inv_scale: np.float32, half_bins: int) -> None:
        """Stochastic rounding of ``h · inv_scale`` for `quantize`'s in-bag rows."""
        self._many.quantize_hessians(key, np.array([inv_scale], np.float32), half_bins)

    def start_tree(self) -> None:
        """Every row to the root."""
        self._many.start_tree()

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
        return self._many.histograms(features, offsets, cells, slots[None], hessians=hessians)[0]

    def partition(self, splits: LevelSplits) -> None:
        """Every training row to its child."""
        self._many.partition([splits])

    def split_leaf(self, split: LeafSplit) -> None:
        """Rows of leaf ``split.leaf`` that go right to leaf ``split.right``."""
        self._many.split_leaves([split])

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Exact fixed-point gradient sums and counts per leaf (in-bag rows)."""
        sums, counts = self._many.leaf_sums(n_leaves)
        return sums[0], counts[0]

    def hessian_sums(self, n_leaves: int) -> NDArray[np.int64]:
        """Exact fixed-point hessian sums per leaf (in-bag rows)."""
        sums: NDArray[np.int64] = self._many.hessian_sums(n_leaves)[0]
        return sums

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """``pred += leaf_values[node]``."""
        self._many.update_predictions(np.asarray(leaf_values, np.float32)[None])

    def predict(self, data: BinnedData, layout: FeatureLayout, lo: int, hi: int, forest: Trees) -> NDArray[np.float32]:
        """Forest output for rows ``[lo, hi)`` of ``data``."""
        return self._many.predict(data, layout, lo, hi, forest)

    def close(self) -> None:
        """Release the device."""
        self._many.close()


MULTI_MAX_TARGETS = 128  # a sanity bound: device memory is the real limit (about 13 bytes per row and target)
MULTI_CHUNK_TARGETS = 15  # per threadgroup: 15 sums and the count fill four 64-bit words of 16-bit fields
MULTI_LOCAL_BYTES = 5 * 1024  # threadgroup counters per SIMD group in histograms_multi: small for occupancy


class DeviceMultiTarget:
    """The multi-target model on ``device`` (experimental, not covered by the documentation).

    Bit-identical to `CPUMultiTarget`.

    The backend owns ``device`` and closes it; a subclass names the backend (``name``).
    """

    name: str

    def __init__(self, device: Device) -> None:
        self._device = device
        self._k = 0
        self._n = 0
        self._hist: Buffer | None = None
        self._partials: tuple[Buffer, Buffer] | None = None
        self._half_bins = 64

    @property
    def k(self) -> int:
        """Number of targets."""
        return self._k

    @property
    def gpu_seconds(self) -> float:
        """GPU time of every committed batch so far."""
        return self._device.gpu_seconds

    @property
    def kernel_seconds(self) -> dict[str, float]:
        """GPU seconds per kernel (profiling backends)."""
        return dict(self._device.kernel_seconds)

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, ys: NDArray[np.float32]) -> None:
        """Bind rows ``[lo, lo + n)`` for ``(K, n)`` targets."""
        k, n = ys.shape
        if not 1 <= k <= MULTI_MAX_TARGETS:
            raise ValueError(f"1..{MULTI_MAX_TARGETS} targets, got {k}")
        if not 0 <= lo <= lo + n <= data.n_rows:
            raise ValueError(f"rows [{lo}, {lo + n}) are outside the panel's {data.n_rows} rows")
        device = self._device
        self._k, self._n, self._lo, self._layout = k, n, lo, layout
        self._named, self._named_stride = device.place("named_bins", data.named_bins)
        self._packed, self._packed_stride = device.place("obfuscated", data.obfuscated)
        self._country, self._cat_stride = device.place("categorical", data.categorical)
        self._kinds = device.copy_in(layout.kind.astype(np.int32))
        self._columns = device.copy_in(layout.column)
        self._y = device.copy_in(np.asarray(ys, np.float32))
        self._pred = device.alloc(4 * k * n)  # no g buffer: 9 bytes per row and target (y, pred, q)
        self._q, self._node, self._bag = device.alloc(k * n), device.alloc(n), device.alloc(n)
        self._row_slot = device.alloc(n)
        device.fill(self._bag, 1, n)
        self._max, self._inv = device.alloc(4 * k), device.alloc(4 * k)
        n_features = layout.n_features
        self._f_kinds, self._f_columns, self._f_offsets = (device.alloc(4 * n_features) for _ in range(3))
        self._o_columns, self._o_offsets = (device.alloc(4 * n_features) for _ in range(2))
        self._slots = device.alloc(4 * MAX_LEVEL_NODES)
        self._split_kind, self._split_threshold, self._split_missing = (device.alloc(MAX_LEVEL_NODES) for _ in range(3))
        self._split_feature, self._split_cat = (
            device.alloc(4 * MAX_LEVEL_NODES),
            device.alloc(8 * MAX_LEVEL_NODES),
        )
        self._leaf_values = device.alloc(4 * k * MAX_LEVEL_NODES)
        self._node_own: Buffer | None = None  # (K, n) node arrays of an own last level, allocated on first use
        self._own_splits: list[Buffer] = []
        self._own = False

    def predictions(self, model: int) -> NDArray[np.float32]:
        """``pred`` of one target, read-only (encoded passes finish first)."""
        self._device.flush()
        row: NDArray[np.float32] = self._device.read(self._pred, np.float32, (self._k, self._n))[model]
        return row

    def _grid(self, models: int) -> tuple[int, int, int]:
        return (max(-(-self._n // THREADS), 1), models, 1)

    def gradients(self) -> NDArray[np.float64]:
        """``max |pred − y|`` per target (``quantize`` recomputes ``g = pred − y``)."""
        device = self._device
        device.fill(self._max, 0, 4 * self._k)
        buffers = [self._pred, self._y, self._max]
        device.encode("max_residuals", buffers, ModelParams(self._n, self._k), self._grid(self._k), BLOCK)
        device.flush()
        return device.read(self._max, np.uint32, (self._k,)).view(np.float32).astype(np.float64)

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scales: NDArray[np.float32], half_bins: int) -> None:
        """Each target's stochastic rounding (one bag, one key)."""
        if bag.shape != (self._n,) or bag.dtype != np.uint8:
            raise ValueError("bag must be uint8 with one entry per training row")
        device = self._device
        device.write(self._bag, np.uint8, (self._n,))[:] = bag
        device.write(self._inv, np.float32, (self._k,))[:] = inv_scales
        self._half_bins = half_bins
        params = QuantizeParams(self._n, self._lo, key, half_bins)
        buffers = [self._pred, self._y, self._bag, self._q, self._inv]
        device.encode("quantize_residuals", buffers, params, self._grid(self._k), BLOCK)

    def start_tree(self) -> None:
        """Every row to the root (one shared node array again)."""
        self._device.fill(self._node, 0, self._n)
        self._own = False

    def histograms(
        self, features: NDArray[np.int32], offsets: NDArray[np.int32], cells: int, slots: NDArray[np.int16]
    ) -> NDArray[np.int32]:
        """``(n_slots, cells, K + 1)``: the K targets' sums of q, then the shared count."""
        device = self._device
        k = self._k
        n_slots = int(slots.max()) + 1 if slots.size else 0
        size = max(n_slots * cells * (k + 1) * 4, 4)
        if self._hist is None or self._hist.nbytes < size:
            device.flush()
            self._hist = device.alloc(size)
        obfuscated = self._layout.kind[features] == OBFUSCATED
        others, packed_features = features[~obfuscated], features[obfuscated]
        device.write(self._f_kinds, np.int32, (others.size,))[:] = self._layout.kind[others]
        device.write(self._f_columns, np.int32, (others.size,))[:] = self._layout.column[others]
        device.write(self._f_offsets, np.int32, (others.size,))[:] = offsets[~obfuscated]
        device.write(self._o_columns, np.int32, (packed_features.size,))[:] = self._layout.column[packed_features]
        device.write(self._o_offsets, np.int32, (packed_features.size,))[:] = offsets[obfuscated]
        device.write(self._slots, np.int32, (slots.size,))[:] = slots
        device.fill(self._hist, 0, n_slots * cells * (k + 1) * 4)
        target_chunks = -(-k // MULTI_CHUNK_TARGETS)
        per_target_chunk = -(-k // target_chunks)  # balanced chunks
        per_slot = OBFUSCATED_BINS * -(-(per_target_chunk + 1) // 4) * LANES * 8
        per_chunk = max(1, min(n_slots, MULTI_LOCAL_BYTES // per_slot))
        rows = min(16384, 65535 // (2 * max(self._half_bins, 1))) // 32 * 32
        params = MultiParams(
            self._n, self._lo, 0, cells, self._named_stride, self._packed_stride, rows, k, self._half_bins,
            per_chunk, n_slots, per_target_chunk, target_chunks, self._cat_stride,
        )  # fmt: skip
        if others.size and n_slots:
            params.n_features = others.size
            buffers = [
                self._f_kinds, self._f_columns, self._f_offsets, self._named, self._packed, self._country,
                self._bag, self._node, self._q, self._slots, self._hist,
            ]  # fmt: skip
            device.encode("build_histograms_multi", buffers, params, self._grid(1), BLOCK)
        if packed_features.size and n_slots:
            if n_slots > 127:
                raise ValueError(f"histograms_multi takes at most 127 slots, got {n_slots}")
            buffers = [self._bag, self._node, self._slots, self._row_slot]
            device.encode("row_slots", buffers, CountParams(self._n), self._grid(1), BLOCK)
            lane_params = MultiParams.from_buffer_copy(params)
            lane_params.n_features = packed_features.size
            chunks = -(-n_slots // per_chunk) * target_chunks
            groups = (chunks, -(-packed_features.size // LANES), max(-(-self._n // rows), 1))
            buffers = [self._o_columns, self._o_offsets, self._packed, self._row_slot, self._q, self._hist]
            device.encode("histograms_multi", buffers, lane_params, groups, LANE_BLOCK, per_chunk * per_slot)
        device.flush()
        return device.read(self._hist, np.int32, (n_slots, cells, k + 1)).copy()

    def partition(self, splits: LevelSplits) -> None:
        """Every row to its child (one node array)."""
        device = self._device
        buffers = [self._split_kind, self._split_feature, self._split_threshold, self._split_missing, self._split_cat]
        width = splits.kind.size
        for buffer, name in zip(buffers, ("kind", "feature", "threshold", "missing_left", "cat_left"), strict=True):
            values = getattr(splits, name)
            device.write(buffer, values.dtype.type, values.shape)[...] = values
        params = PartitionParams(self._n, self._lo, self._named_stride, self._packed_stride, width, self._cat_stride)
        all_buffers = [self._kinds, self._columns, *buffers, self._named, self._packed, self._country, self._node]
        device.encode("partition_rows", all_buffers, params, self._grid(1), BLOCK)

    def partition_own(self, levels: Sequence[LevelSplits]) -> None:
        """Partition the last level with each target's own splits: ``node_own[k] = 2 · node + go_right_k``."""
        device = self._device
        k, width = self._k, levels[0].kind.size
        names = ("kind", "feature", "threshold", "missing_left", "cat_left")
        if self._node_own is None:
            device.flush()
            self._node_own = device.alloc(k * self._n)
            self._own_splits = [
                device.alloc(getattr(levels[0], name).dtype.itemsize * k * MAX_LEVEL_NODES) for name in names
            ]
        for buffer, name in zip(self._own_splits, names, strict=True):
            values = np.stack([getattr(level, name) for level in levels])
            device.write(buffer, values.dtype.type, values.shape)[...] = values
        params = PartitionParams(self._n, self._lo, self._named_stride, self._packed_stride, width, self._cat_stride)
        buffers = [
            self._kinds, self._columns, *self._own_splits, self._named, self._packed, self._country, self._node,
            self._node_own,
        ]  # fmt: skip
        device.encode("partition_own", buffers, params, self._grid(k), BLOCK)
        self._own = True

    def _nodes(self) -> tuple[Buffer, int]:
        """Return the node arrays and their per-target stride: the shared array (0) or the own last level's (n)."""
        if self._own and self._node_own is not None:
            return self._node_own, self._n
        return self._node, 0

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """``(K, n_leaves)`` exact sums and ``(K, n_leaves)`` counts."""
        device = self._device
        groups = max(-(-self._n // LEAF_ROWS_PER_GROUP), 1)
        cells = self._k * groups * n_leaves
        if self._partials is None or self._partials[1].nbytes < 4 * cells:
            device.flush()
            self._partials = (device.alloc(12 * cells), device.alloc(4 * cells))  # hi, mid, low per leaf
        sums, counts = self._partials
        node, stride = self._nodes()
        params = LeafParams(self._n, n_leaves, LEAF_ROWS_PER_GROUP, groups, stride, 1)
        buffers = [self._pred, self._y, self._bag, node, sums, counts]
        device.encode("leaf_partials", buffers, params, (groups, self._k, 1), BLOCK)
        device.flush()
        shape = (self._k, groups, n_leaves)
        return (
            _leaf_total(device.read(sums, np.uint32, (*shape, 3))),
            device.read(counts, np.uint32, shape).sum(axis=1, dtype=np.int64),
        )

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """``pred[k] += leaf_values[k, node_k]``."""
        device = self._device
        device.write(self._leaf_values, np.float32, leaf_values.shape)[...] = leaf_values
        node, stride = self._nodes()
        params = UpdateParams(self._n, leaf_values.shape[1], stride)
        device.encode("update_predictions", [self._pred, node, self._leaf_values], params, self._grid(self._k), BLOCK)

    def close(self) -> None:
        """Release the device."""
        self._device.close()

    def __del__(self) -> None:
        device = getattr(self, "_device", None)
        if device is not None:
            device.close()
