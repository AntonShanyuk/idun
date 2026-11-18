"""Multi-target boosting: one tree structure for several targets (experimental, not covered by the documentation)."""

import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from idun import rng
from idun.errors import MultiTargetQualityWarning
from idun.gbm import sampling
from idun.gbm.backends.base import LevelSplits
from idun.gbm.backends.cpu import CPUMultiTarget
from idun.gbm.backends.cuda import CudaMultiTarget
from idun.gbm.backends.metal import MetalMultiTarget
from idun.gbm.booster import Booster, FitReport
from idun.gbm.features import BinnedData, FeatureLayout
from idun.gbm.params import Params
from idun.gbm.split import Split, best_splits, best_splits_multi
from idun.gbm.train import _bounds, _draw_bag, _level, _Timer, leaf_values, quantization_scales
from idun.gbm.tree import Forest

# not in its search: the regularisers, and the per-node draws
REGULARISERS = ("lambda_l1", "min_gain_to_split", "min_sum_hessian_in_leaf", "max_delta_step", "extra_trees",
                "feature_fraction_bynode")  # fmt: skip


class MultiTargetBackend(Protocol):
    """K targets sharing one tree per iteration: shared node array, per-target gradients and sums."""

    name: str

    @property
    def k(self) -> int:
        """Number of targets."""

    def start(self, data: BinnedData, layout: FeatureLayout, lo: int, ys: NDArray[np.float32]) -> None:
        """Bind rows ``[lo, lo + n)`` for ``(K, n)`` targets."""

    def predictions(self, model: int) -> NDArray[np.float32]:
        """``pred`` of one target."""

    def gradients(self) -> NDArray[np.float64]:
        """``max |g|`` per target, ``g = pred − y`` (stored or recomputed by ``quantize``)."""

    def quantize(self, bag: NDArray[np.uint8], key: int, inv_scales: NDArray[np.float32], half_bins: int) -> None:
        """Each target's stochastic rounding."""

    def start_tree(self) -> None:
        """Every row to the root."""

    def histograms(
        self, features: NDArray[np.int32], offsets: NDArray[np.int32], cells: int, slots: NDArray[np.int16]
    ) -> NDArray[np.int32]:
        """``(n_slots, cells, K + 1)``: K sums of q, then the shared count."""

    def partition(self, splits: LevelSplits) -> None:
        """Every row to its child."""

    def partition_own(self, levels: Sequence[LevelSplits]) -> None:
        """Partition the last level with one set of splits per target: per-target node arrays until ``start_tree``."""

    def leaf_sums(self, n_leaves: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """``(K, n_leaves)`` sums and ``(K, n_leaves)`` counts."""

    def update_predictions(self, leaf_values: NDArray[np.float32]) -> None:
        """``pred[k] += leaf_values[k, node_k]``."""


def get_multi(name: str) -> MultiTargetBackend:
    """Construct a multi-target backend: ``"cpu"`` (reference), ``"metal"`` or ``"cuda"``."""
    if name == "cpu":
        return CPUMultiTarget()
    if name == "metal":
        return MetalMultiTarget()
    if name == "cuda":
        return CudaMultiTarget()
    raise ValueError(f"unknown multi-target backend {name!r}")


def train_multi(
    params: Params,
    data: BinnedData,
    *,
    rows: slice,
    backend: str | MultiTargetBackend,
    ys: Sequence[NDArray[np.float32]] | NDArray[np.float32],
    own_last_level: bool = False,
) -> list[Booster]:
    """Fit several targets with one shared tree structure (experimental, not covered by the documentation)."""
    if params.leafwise:
        raise ValueError("train_multi grows depth-wise trees only: use growth='depthwise' (or train per target)")
    defaults = Params()
    if set_ := [name for name in REGULARISERS if getattr(params, name) != getattr(defaults, name)]:
        raise ValueError(f"train_multi does not take {', '.join(set_)} yet: use train per target (or the defaults)")
    lo, hi = _bounds(rows, data.n_rows)
    targets = np.ascontiguousarray(ys if isinstance(ys, np.ndarray) else np.stack(ys), dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != hi - lo:
        raise ValueError(f"each target must hold one float32 per training row ({hi - lo})")
    if targets.shape[0] > 1:
        warnings.warn(
            f"train_multi fits {targets.shape[0]} targets with one shared tree structure (experimental): less "
            "accurate than separate fits; use train where accuracy matters",
            MultiTargetQualityWarning,
            stacklevel=2,
        )
    device = get_multi(backend) if isinstance(backend, str) else backend
    layout = FeatureLayout.of(data)
    timer = _Timer()
    with timer("upload"):
        device.start(data, layout, lo, targets)
    k = targets.shape[0]
    del targets  # the backend holds its own copy (K targets x rows: gigabytes at K = 100)
    structure = Forest.empty(params.num_trees, params.max_depth)
    own = [Forest.empty(params.num_trees, params.max_depth) for _ in range(k)] if own_last_level else None
    leaves = np.zeros((params.num_trees, k, 2**params.max_depth), np.float32)
    leaf_counts = np.zeros((params.num_trees, k, 2**params.max_depth), np.int32)
    bag = np.ones(hi - lo, np.uint8)
    with ThreadPoolExecutor(max_workers=1) as pool:
        next_bag = _draw_bag(pool, params, hi - lo, 0, params.num_trees)
        for iteration in range(params.num_trees):
            if next_bag is not None and iteration % params.bagging_freq == 0:
                bag = next_bag.result()
                next_bag = _draw_bag(pool, params, hi - lo, iteration + params.bagging_freq, params.num_trees)
            with timer("gradients"):
                scales, inv_scales = quantization_scales(device.gradients(), hi - lo, params.grad_quant_bins)
                key = rng.iteration_key(params.seed, iteration)
                device.quantize(bag, key, inv_scales, params.grad_quant_bins // 2)
            features = sampling.feature_sample(
                layout.n_features, params.feature_fraction, seed=params.seed, tree=iteration
            )
            _grow_multi(device, structure, own, iteration, features, layout, params, scales, timer)
            with timer("leaves"):
                sums, counts = device.leaf_sums(2**params.max_depth)
                for target in range(k):
                    leaves[iteration, target] = leaf_values(
                        sums[target], counts[target], params.lambda_l2, params.learning_rate
                    )
                leaf_counts[iteration] = counts
                device.update_predictions(leaves[iteration])
    report = FitReport(
        seconds=dict(timer.seconds),
        backend=device.name,
        train_rows=(lo, hi),
        gpu_seconds=getattr(device, "gpu_seconds", None),
        kernel_seconds=dict(getattr(device, "kernel_seconds", {})),
    )
    forests = [_target_forest(structure, None if own is None else own[t], leaves[:, t], leaf_counts[:, t])
               for t in range(k)]  # fmt: skip
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
        )
        for forest in forests
    ]


def _target_forest(
    structure: Forest, own: Forest | None, leaf_value: NDArray[np.float32], leaf_count: NDArray[np.int32]
) -> Forest:
    """One target's forest: the shared structure arrays (shared, not copied), with its own deepest level if any."""
    names = ("kind", "feature", "threshold", "missing_left", "cat_left")
    arrays = {name: getattr(structure, name) for name in names}
    if own is not None:
        depth = int(np.log2(structure.kind.shape[1] + 1))
        last = slice(2 ** (depth - 1) - 1, 2**depth - 1)
        for name in names:
            arrays[name] = arrays[name].copy()
            arrays[name][:, last] = getattr(own, name)[:, last]
    return Forest(**arrays, leaf_value=np.ascontiguousarray(leaf_value), leaf_count=np.ascontiguousarray(leaf_count))


def _grow_multi(
    device: MultiTargetBackend,
    structure: Forest,
    own: list[Forest] | None,
    tree: int,
    features: NDArray[np.int32],
    layout: FeatureLayout,
    params: Params,
    scales: NDArray[np.float64],
    timer: _Timer,
) -> None:
    """Grow one shared tree level by level (as `idun.gbm.train._grow`, with summed gains).

    ``own``: per-target forests that receive each target's own deepest level.
    """
    offsets, cells = layout.offsets(features)
    device.start_tree()
    with timer("histograms"):
        hists = {0: device.histograms(features, offsets, cells, np.zeros(1, np.int16))[0]}
    leaves = 1
    for depth in range(params.max_depth):
        if own is not None and depth + 1 == params.max_depth:
            _own_last_level(device, own, hists, tree, depth, leaves, features, layout, params, scales, timer)
            return
        with timer("splits"):
            nodes = sorted(hists)
            k1 = scales.size + 1
            stacked = np.stack([hists[j] for j in nodes]) if nodes else np.zeros((0, cells, k1), np.int32)
            found = best_splits_multi(stacked, features, layout, params, scales)
            valid = {j: split for j, split in zip(nodes, found, strict=True) if split is not None}
            best_first = sorted(valid, key=lambda j: (-valid[j].gain, j))
            chosen: dict[int, Split] = {j: valid[j] for j in sorted(best_first[: params.max_leaves - leaves])}
            leaves += len(chosen)
            level = _level(structure, tree, depth, chosen)
        with timer("partition"):
            device.partition(level)
        if depth + 1 == params.max_depth:
            break
        if not chosen:
            hists = {}
            continue
        counts: dict[int, int] = {}
        for j, split in chosen.items():
            counts[2 * j], counts[2 * j + 1] = split.left_count, split.right_count
        smaller = [2 * j if counts[2 * j] <= counts[2 * j + 1] else 2 * j + 1 for j in chosen]
        slots = np.full(2 ** (depth + 1), -1, np.int16)
        slots[smaller] = np.arange(len(smaller), dtype=np.int16)
        with timer("histograms"):
            built = device.histograms(features, offsets, cells, slots)
        children = {}
        for j, small in zip(chosen, smaller, strict=True):
            children[small] = built[slots[small]]
            children[small ^ 1] = hists[j] - built[slots[small]]
        hists = {j: h for j, h in children.items() if counts[j] >= 2 * params.min_data_in_leaf}


def _own_last_level(
    device: MultiTargetBackend,
    own: list[Forest],
    hists: dict[int, NDArray[np.int32]],
    tree: int,
    depth: int,
    leaves: int,
    features: NDArray[np.int32],
    layout: FeatureLayout,
    params: Params,
    scales: NDArray[np.float64],
    timer: _Timer,
) -> None:
    """Each target's best splits of the deepest level from its own histogram column, then the per-target partition."""
    nodes = sorted(hists)
    k = scales.size
    levels = []
    with timer("splits"):
        for target in range(k):
            columns = [np.ascontiguousarray(hists[j][:, [target, k]]) for j in nodes]
            stacked = np.stack(columns) if nodes else np.zeros((0, 1, 2), np.int32)
            found = best_splits(stacked, features, layout, params, float(scales[target]))
            valid = {j: split for j, split in zip(nodes, found, strict=True) if split is not None}
            best_first = sorted(valid, key=lambda j: (-valid[j].gain, j))
            chosen = {j: valid[j] for j in sorted(best_first[: params.max_leaves - leaves])}
            levels.append(_level(own[target], tree, depth, chosen))
    with timer("partition"):
        device.partition_own(levels)
