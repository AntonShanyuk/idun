"""Split finding: one host-side implementation shared by every backend (docs/architecture/determinism.md §4).

The input is one node's integer histogram over the tree's sampled features; the output is the node's best split, or
None. Everything is computed in float64 from exact integer sums, so identical histograms give identical splits on
every backend. Ties go to the lowest feature, then the lowest threshold, then missing-left.

Numeric features are searched per kind in one compiled loop over nodes, features and thresholds (sampled features are
sorted, so named, obfuscated and categorical features occupy consecutive histogram blocks); it does numpy's float64
operations in numpy's order, so its gains and picks are those of the same search vectorized in numpy. Each categorical
feature follows LightGBM's integer-histogram search (``FindBestThresholdCategoricalIntInner``) step by step, compiled
too.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from idun._jit import njit
from idun.gbm.features import (
    CATEGORICAL,
    MISSING_BIN,
    NAMED,
    NAMED_HIST_BINS,
    OBFUSCATED,
    OBFUSCATED_BINS,
    FeatureLayout,
)
from idun.gbm.params import Params

PASS = 0  # tree node kinds (model format): the node did not split, every row goes left
NUMERIC = 1
CATEGORY = 2


@dataclass(frozen=True)
class NodeSearch:
    """Per node, what its search may use, ``(nodes, features)`` each in the tree's sampled order.

    ``allowed``: the node's features (``feature_fraction_bynode``); ``only``: per numeric feature the one threshold it
    tries (``extra_trees``; -1 every threshold, -2 none).
    """

    allowed: NDArray[np.uint8]
    only: NDArray[np.int64]


@dataclass(frozen=True)
class Split:
    """A node's chosen split. ``cat_left`` is the bitmask of categories sent left (categorical splits only)."""

    gain: float
    feature: int
    kind: int
    threshold: int
    missing_left: bool
    cat_left: int
    left_count: int
    right_count: int


def leaf_gain(
    sum_q: int | NDArray[np.int64],
    hessian: float | NDArray[np.int64] | NDArray[np.float64],
    scale: float | NDArray[np.float64],
    lam: float,
    l1: float = 0.0,
    max_delta: float = 0.0,
) -> NDArray[np.float64]:
    """``G² / (H + λ)`` with ``G = Σq · scale``, in float64; with L1 or a step limit, LightGBM's.

    ``H`` is the row count (first-order splits) or the hessians' sum (second-order).

    ``l1`` > 0 shrinks ``G`` towards 0 by ``l1`` (``ThresholdL1``); ``max_delta`` > 0 clips the leaf's output
    ``−G' / (H + λ)`` to ``±max_delta`` and gives the gain of the clipped output, ``−(2 G' w + (H + λ) w²)``.
    """
    g = np.multiply(sum_q, scale, dtype=np.float64)
    h = np.add(hessian, lam, dtype=np.float64)
    h = np.where(h > 0.0, h, 1.0)  # H + λ = 0 (hessians of 0): the node has no split, its gain is never read
    if l1 == 0.0 and max_delta == 0.0:
        return np.asarray(np.square(g) / h)
    shrunk = np.sign(g) * np.maximum(np.abs(g) - l1, 0.0)
    if max_delta == 0.0:
        return np.asarray(np.square(shrunk) / h)
    w = np.clip(-shrunk / h, -max_delta, max_delta)
    return np.asarray(-(2.0 * shrunk * w + h * w * w))


@njit()
def _regularised(g: float, hessian: float, lam: float, l1: float, max_delta: float) -> float:
    """`leaf_gain` with ``l1`` or ``max_delta`` set, for one leaf, in its operations' order."""
    h = hessian + lam
    shrunk = math.copysign(max(abs(g) - l1, 0.0), g) if g != 0.0 else 0.0
    if max_delta == 0.0:
        return shrunk * shrunk / h
    w = min(max(-shrunk / h, -max_delta), max_delta)
    return -(2.0 * shrunk * w + h * w * w)


@njit()
def _enough(left: float, right: float, min_hessian: float, lam: float) -> bool:
    """Both sides have the hessian sum they need and a positive ``H + λ`` (always, for row counts)."""
    return left >= min_hessian and right >= min_hessian and left + lam > 0.0 and right + lam > 0.0


def _numeric_block(
    cells: NDArray[np.int32],
    total_q: NDArray[np.int64],
    total_c: NDArray[np.int64],
    total_h: NDArray[np.int64],
    parent: NDArray[np.float64],
    scale: float,
    hessian: tuple[int, float],
    params: Params,
    *,
    missing: bool,
    search: NodeSearch | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Best (gain, flat candidate index, left count) per (node, feature) of a ``(nodes, features, bins, 2 or 3)`` block.

    The candidates of a feature are its thresholds in bin order; with ``missing``, each threshold twice (missing left,
    then right) over the real bins, the last bin being the missing one. The best is the first of the largest gains.
    ``hessian`` is the hessian's channel and scale: the count's and 1.0, or the quantized hessians' and theirs.
    ``search``: the block's features' per-node restrictions (`NodeSearch`); None, every feature and threshold.
    """
    n, n_features = cells.shape[0], cells.shape[1]
    allowed = np.ones((n, n_features), np.uint8) if search is None else search.allowed
    only = np.full((n, n_features), -1, np.int64) if search is None else search.only
    return _numeric_search(
        cells, total_q, total_c, total_h, parent, float(scale), hessian[0], float(hessian[1]),
        float(params.lambda_l2), int(params.min_data_in_leaf), missing, float(params.lambda_l1),
        float(params.max_delta_step), float(params.min_sum_hessian_in_leaf), allowed, only,
    )  # fmt: skip


@njit()
def _numeric_search(
    cells: NDArray[np.int32],
    total_q: NDArray[np.int64],
    total_c: NDArray[np.int64],
    total_h: NDArray[np.int64],
    parent: NDArray[np.float64],
    scale: float,
    channel: int,
    h_scale: float,
    lam: float,
    min_data: int,
    missing: bool,  # noqa: FBT001 - numba compiles positional arguments
    l1: float,
    max_delta: float,
    min_hessian: float,
    allowed: NDArray[np.uint8],
    only: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    n, n_features, bins = cells.shape[0], cells.shape[1], cells.shape[2]
    plain = l1 == 0.0 and max_delta == 0.0
    real = MISSING_BIN if missing else bins
    sides = 2 if missing else 1
    gains = np.empty((n, n_features), np.float64)
    flat = np.empty((n, n_features), np.int64)
    left = np.empty((n, n_features), np.int64)
    for node in range(n):
        tq, tc, th, base = int(total_q[node]), int(total_c[node]), int(total_h[node]), float(parent[node])
        for feature in range(n_features):
            q, c, h = cells[node, feature, :, 0], cells[node, feature, :, 1], cells[node, feature, :, channel]
            miss_q = int(q[MISSING_BIN]) if missing else 0
            miss_c = int(c[MISSING_BIN]) if missing else 0
            miss_h = int(h[MISSING_BIN]) if missing else 0
            left_q, left_c, left_h = 0, 0, 0
            best, best_at, best_left, seen_nan = -math.inf, 0, 0, False
            # -1: every threshold; else the one extra_trees drew; -2: none (also a feature not in the node's sample)
            pick = only[node, feature] if allowed[node, feature] != 0 else -2
            for b in range(real):
                left_q += int(q[b])
                left_c += int(c[b])
                left_h += int(h[b])
                for side in range(sides):
                    cq = left_q + miss_q if side == 0 else left_q  # missing left first
                    cc = left_c + miss_c if side == 0 else left_c
                    ch = left_h + miss_h if side == 0 else left_h
                    at = b * sides + side
                    if at == 0:
                        best_left = cc
                    rq, rc = tq - cq, tc - cc
                    hl, hr = float(ch) * h_scale, float(th - ch) * h_scale  # the counts, exactly, at scale 1.0
                    gain = -math.inf
                    # a candidate: every threshold (pick -1), or only the one extra_trees drew
                    if pick in (-1, b) and cc >= min_data and rc >= min_data and _enough(hl, hr, min_hessian, lam):
                        gl, gr = float(cq) * scale, float(rq) * scale  # leaf_gain's operations, in order
                        if plain:
                            gain = (gl * gl / (hl + lam) + gr * gr / (hr + lam)) - base
                        else:
                            gain = _regularised(gl, hl, lam, l1, max_delta) + _regularised(gr, hr, lam, l1, max_delta)
                            gain -= base
                    if seen_nan:
                        continue
                    if math.isnan(gain) or gain > best:  # np.argmax: the first NaN, else the first largest
                        best, best_at, best_left, seen_nan = gain, at, cc, math.isnan(gain)
            gains[node, feature], flat[node, feature], left[node, feature] = best, best_at, best_left
    return gains, flat, left


def _categorical(
    cells: NDArray[np.int32],
    total: tuple[int, int, int],
    parent: float,
    scale: float,
    hessian: tuple[int, float],
    params: Params,
) -> tuple[float, int, int] | None:
    """Best (gain, left bitmask, left count) for the categorical feature, or None; ``total`` is (Σq, count, Σh)."""
    found, gain, mask, left_c = _categorical_search(
        cells[:, 0].astype(np.int64),
        cells[:, 1].astype(np.int64),
        cells[:, hessian[0]].astype(np.int64),
        int(total[0]),
        int(total[1]),
        int(total[2]),
        float(parent),
        float(scale),
        float(hessian[1]),
        params.categorical == "one_vs_rest",
        float(params.lambda_l2),
        float(params.cat_l2),
        float(params.cat_smooth),
        int(params.min_data_in_leaf),
        int(params.min_data_per_group),
        int(params.max_cat_threshold),
        float(params.lambda_l1),
        float(params.max_delta_step),
        float(params.min_sum_hessian_in_leaf),
        float(params.min_gain_to_split),
    )
    return (float(gain), int(mask), int(left_c)) if found else None


@njit()
def _categorical_search(
    q: NDArray[np.int64],
    c: NDArray[np.int64],
    h: NDArray[np.int64],
    total_q: int,
    total_c: int,
    total_h: int,
    parent: float,
    scale: float,
    h_scale: float,
    one_vs_rest: bool,  # noqa: FBT001 - numba compiles positional arguments
    lambda_l2: float,
    cat_l2: float,
    cat_smooth: float,
    min_data: int,
    min_data_per_group: int,
    max_cat_threshold: int,
    l1: float,
    max_delta: float,
    min_hessian: float,
    min_gain: float,
) -> tuple[bool, float, np.uint64, int]:
    if one_vs_rest:
        return _one_vs_rest(q, c, h, total_q, total_c, total_h, parent, scale, h_scale, lambda_l2, min_data, l1,
                            max_delta, min_hessian, min_gain)  # fmt: skip
    found, best, best_mask, best_left = False, 0.0, np.uint64(0), 0
    plain = l1 == 0.0 and max_delta == 0.0
    lam = lambda_l2 + cat_l2
    # never an empty category; with hessians, never one whose key G / (H + cat_smooth) is G / 0
    used = np.array([cat for cat in range(q.size) if c[cat] >= max(cat_smooth, 1.0)
                     and float(h[cat]) * h_scale + cat_smooth > 0.0], np.int64)  # fmt: skip
    ctr = np.empty(used.size, np.float64)
    for i in range(used.size):
        ctr[i] = float(q[used[i]]) * scale / (float(h[used[i]]) * h_scale + cat_smooth)
    ordered = used[np.argsort(ctr, kind="mergesort")]  # stable, as LightGBM's std::stable_sort
    max_cats = min(max_cat_threshold, (used.size + 1) // 2)
    for direction in range(2):
        left_q, left_c, left_h, group, mask = 0, 0, 0, 0, np.uint64(0)
        for step in range(min(max_cats, ordered.size)):
            cat = ordered[step] if direction == 0 else ordered[ordered.size - 1 - step]
            left_q, left_c, group = left_q + int(q[cat]), left_c + int(c[cat]), group + int(c[cat])
            left_h += int(h[cat])
            mask |= np.uint64(1) << np.uint64(cat)
            hl, hr = float(left_h) * h_scale, float(total_h - left_h) * h_scale
            if left_c < min_data or hl < min_hessian or hl + lam <= 0.0:
                continue
            right_c = total_c - left_c
            if right_c < min_data or right_c < min_data_per_group or hr < min_hessian or hr + lam <= 0.0:
                break
            if group < min_data_per_group:
                continue
            group = 0
            gl, gr = float(left_q) * scale, float(total_q - left_q) * scale
            if plain:
                gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent
            else:
                gain = _regularised(gl, hl, lam, l1, max_delta) + _regularised(gr, hr, lam, l1, max_delta)
                gain -= parent
            if gain > min_gain and (not found or gain > best):
                found, best, best_mask, best_left = True, gain, mask, left_c
    return found, best, best_mask, best_left


@njit()
def _one_vs_rest(
    q: NDArray[np.int64],
    c: NDArray[np.int64],
    h: NDArray[np.int64],
    total_q: int,
    total_c: int,
    total_h: int,
    parent: float,
    scale: float,
    h_scale: float,
    lambda_l2: float,
    min_data: int,
    l1: float,
    max_delta: float,
    min_hessian: float,
    min_gain: float,
) -> tuple[bool, float, np.uint64, int]:
    """`_categorical_search`'s one-category-left search."""
    found, best, best_mask, best_left = False, 0.0, np.uint64(0), 0
    plain = l1 == 0.0 and max_delta == 0.0
    for cat in range(q.size):
        left_c, right_c = int(c[cat]), total_c - int(c[cat])
        hl, hr = float(h[cat]) * h_scale, float(total_h - int(h[cat])) * h_scale
        if left_c < min_data or right_c < min_data or not _enough(hl, hr, min_hessian, lambda_l2):
            continue
        gl, gr = float(q[cat]) * scale, float(total_q - int(q[cat])) * scale
        if plain:
            gain = gl * gl / (hl + lambda_l2) + gr * gr / (hr + lambda_l2) - parent
        else:
            gain = _regularised(gl, hl, lambda_l2, l1, max_delta) + _regularised(gr, hr, lambda_l2, l1, max_delta)
            gain -= parent  # (left + right) - parent, as everywhere
        if gain > min_gain and (not found or gain > best):
            found, best, best_mask, best_left = True, gain, np.uint64(1) << np.uint64(cat), left_c
    return found, best, best_mask, best_left


def best_split(
    hist: NDArray[np.int32],
    features: NDArray[np.int32],
    layout: FeatureLayout,
    params: Params,
    scale: float,
    hessian_scale: float | None = None,
    search: NodeSearch | None = None,
) -> Split | None:
    """Best split of one node from its histogram ``(cells, 2 or 3)``; see `best_splits`."""
    return best_splits(hist[None], features, layout, params, scale, hessian_scale, search)[0]


def best_splits(
    hists: NDArray[np.int32],
    features: NDArray[np.int32],
    layout: FeatureLayout,
    params: Params,
    scale: float,
    hessian_scale: float | None = None,
    search: NodeSearch | None = None,
) -> list[Split | None]:
    """Best split of each node from histograms ``(nodes, cells, 2)`` over the sorted sampled ``features``.

    ``scale`` turns quantized gradient sums into gradient sums (determinism.md §2). A split needs at least
    ``min_data_in_leaf`` in-bag rows on each side and a positive gain; a node without one gets None. The numeric search
    runs over all nodes and features at once; every gain is the same float64 expression as for a single node.
    First-order splits take the row count as the hessian. With ``hessian_scale`` the histograms are ``(nodes, cells,
    3)``, the third channel the quantized hessians' sums, and the splits are second-order: ``H = Σq_h ·
    hessian_scale`` in the gains and in ``min_sum_hessian_in_leaf``, the count in ``min_data_in_leaf``. ``search``
    restricts each node to a feature sample and each numeric feature to one threshold (`NodeSearch`).
    """
    n = hists.shape[0]
    if n == 0:
        return []
    channels = 2 if hessian_scale is None else 3
    if hists.shape[-1] != channels:
        raise ValueError(f"histograms with {hists.shape[-1]} channels for {channels}-channel splits")
    hessian = (1, 1.0) if hessian_scale is None else (2, float(hessian_scale))
    kinds = layout.kind[features]
    first = int(layout.hist_bins[features[0]])
    total_q = hists[:, :first, 0].sum(axis=1, dtype=np.int64)
    total_c = hists[:, :first, 1].sum(axis=1, dtype=np.int64)
    total_h = hists[:, :first, hessian[0]].sum(axis=1, dtype=np.int64)
    total_hessian = np.multiply(total_h, hessian[1], dtype=np.float64)  # the counts, exactly, at scale 1.0
    parent = leaf_gain(total_q, total_hessian, scale, params.lambda_l2, params.lambda_l1, params.max_delta_step)
    splittable = (total_c >= 2 * params.min_data_in_leaf) & (total_hessian >= 2 * params.min_sum_hessian_in_leaf)
    best: list[Split | None] = [None] * n

    def offer(node: int, split: Split) -> None:  # in feature order: a later candidate must be strictly better
        current = best[node]
        if current is None or split.gain > current.gain:
            best[node] = split

    at = 0
    for kind, width, missing in ((NAMED, NAMED_HIST_BINS, True), (OBFUSCATED, OBFUSCATED_BINS, False)):
        block = features[kinds == kind]
        if not block.size:
            continue
        cells = hists[:, at : at + width * block.size].reshape(n, block.size, width, channels)
        part = None if search is None else NodeSearch(search.allowed[:, kinds == kind], search.only[:, kinds == kind])
        gains, flat, left = _numeric_block(cells, total_q, total_c, total_h, parent, scale, hessian, params,
                                           missing=missing, search=part)  # fmt: skip
        choice = np.argmax(gains, axis=1)
        for node in np.flatnonzero(splittable):
            i = int(choice[node])
            if gains[node, i] > params.min_gain_to_split:
                candidate = int(flat[node, i])
                threshold, missing_left = (candidate // 2, candidate % 2 == 0) if missing else (candidate, True)
                offer(
                    int(node),
                    Split(
                        gain=float(gains[node, i]),
                        feature=int(block[i]),
                        kind=NUMERIC,
                        threshold=threshold,
                        missing_left=missing_left,
                        cat_left=0,
                        left_count=int(left[node, i]),
                        right_count=int(total_c[node]) - int(left[node, i]),
                    ),
                )
        at += width * block.size
    for position in np.flatnonzero(kinds == CATEGORICAL):  # in feature order, each its own block of cells
        feature = features[position]
        width = int(layout.hist_bins[feature])
        for node in np.flatnonzero(splittable):
            if search is not None and search.allowed[node, position] == 0:
                continue
            total = (int(total_q[node]), int(total_c[node]), int(total_h[node]))
            found = _categorical(hists[node, at : at + width], total, float(parent[node]), scale, hessian, params)
            if found is not None:
                gain, mask, left_c = found
                offer(
                    int(node),
                    Split(
                        gain=gain,
                        feature=int(feature),
                        kind=CATEGORY,
                        threshold=0,
                        missing_left=False,
                        cat_left=mask,
                        left_count=left_c,
                        right_count=total[1] - left_c,
                    ),
                )
        at += width
    return best


def _numeric_block_multi(
    cells: NDArray[np.int32],
    total_q: NDArray[np.int64],
    total_c: NDArray[np.int64],
    parent: NDArray[np.float64],
    scales: NDArray[np.float64],
    params: Params,
    *,
    missing: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """`_numeric_block` for K targets sharing the tree: ``cells`` is ``(nodes, features, bins, K + 1)`` (sums, count).

    The gain of a candidate is the sum over targets of each target's gain; for K = 1 it is the single-target value.
    """
    k = scales.size
    real = cells[..., :MISSING_BIN, :] if missing else cells
    left_q = np.cumsum(real[..., :k], axis=-2, dtype=np.int64)  # (n, f, T, K)
    left_c = np.cumsum(real[..., k], axis=-1, dtype=np.int64)  # (n, f, T)
    if missing:  # candidates per threshold: missing left, then missing right
        miss_q = cells[..., MISSING_BIN, :k].astype(np.int64)[..., None, :]
        miss_c = cells[..., MISSING_BIN, k].astype(np.int64)[..., None]
        left_q = np.stack([left_q + miss_q, left_q], axis=-2).reshape(*cells.shape[:2], -1, k)
        left_c = np.stack([left_c + miss_c, left_c], axis=-1).reshape(*cells.shape[:2], -1)
    right_q = total_q[:, None, None, :] - left_q
    right_c = total_c[:, None, None] - left_c
    valid = (left_c >= params.min_data_in_leaf) & (right_c >= params.min_data_in_leaf)
    with np.errstate(divide="ignore", invalid="ignore"):  # empty sides with λ = 0: masked out just below
        per_target = leaf_gain(left_q, left_c[..., None], scales, params.lambda_l2) + leaf_gain(
            right_q, right_c[..., None], scales, params.lambda_l2
        )
    gain = np.where(valid, per_target.sum(axis=-1) - parent[:, None, None], -np.inf)
    best = np.argmax(gain, axis=-1)[..., None]
    return (
        np.take_along_axis(gain, best, axis=-1)[..., 0],
        best[..., 0],
        np.take_along_axis(left_c, best, axis=-1)[..., 0],
    )


def _categorical_multi(
    cells: NDArray[np.int32],
    total_q: NDArray[np.int64],
    total_c: int,
    parent: float,
    scales: NDArray[np.float64],
    params: Params,
) -> tuple[float, int, int] | None:
    """`_categorical` for K targets: categories ordered by ``Σ_k G_c,k / (H_c + cat_smooth)``; gains summed."""
    k = scales.size
    q = cells[:, :k].astype(np.int64)
    c = cells[:, k].astype(np.int64)

    def gain_of(left_q: NDArray[np.int64], left_c: int, lam: float) -> float:
        total = 0.0
        for target in range(k):
            total += float(
                leaf_gain(int(left_q[target]), left_c, float(scales[target]), lam)
                + leaf_gain(int(total_q[target] - left_q[target]), total_c - left_c, float(scales[target]), lam)
            )
        return total - parent

    best: tuple[float, int, int] | None = None
    if params.categorical == "one_vs_rest":
        for cat in range(len(cells)):
            left_c = int(c[cat])
            if left_c < params.min_data_in_leaf or total_c - left_c < params.min_data_in_leaf:
                continue
            gain = gain_of(q[cat], left_c, params.lambda_l2)
            if gain > 0.0 and (best is None or gain > best[0]):
                best = (gain, 1 << cat, left_c)
        return best
    lam = params.lambda_l2 + params.cat_l2
    used = [cat for cat in range(len(cells)) if c[cat] >= max(params.cat_smooth, 1.0)]
    ctr = {
        cat: sum(float(q[cat, t] * scales[t]) for t in range(k)) / (float(c[cat]) + params.cat_smooth) for cat in used
    }
    ordered = sorted(used, key=lambda cat: ctr[cat])
    max_cats = min(params.max_cat_threshold, (len(used) + 1) // 2)
    for order in (ordered, ordered[::-1]):
        left_q = np.zeros(k, np.int64)
        left_c = group = mask = 0
        for cat in order[:max_cats]:
            left_q = left_q + q[cat]
            left_c, group, mask = left_c + int(c[cat]), group + int(c[cat]), mask | 1 << cat
            if left_c < params.min_data_in_leaf:
                continue
            right_c = total_c - left_c
            if right_c < params.min_data_in_leaf or right_c < params.min_data_per_group:
                break
            if group < params.min_data_per_group:
                continue
            group = 0
            gain = gain_of(left_q, left_c, lam)
            if gain > 0.0 and (best is None or gain > best[0]):
                best = (gain, mask, left_c)
    return best


def best_splits_multi(
    hists: NDArray[np.int32],
    features: NDArray[np.int32],
    layout: FeatureLayout,
    params: Params,
    scales: NDArray[np.float64],
) -> list[Split | None]:
    """Return the multi-target model's splits (experimental, not covered by the documentation).

    With one target, `best_splits` exactly.
    """
    n = hists.shape[0]
    if n == 0:
        return []
    k = scales.size
    kinds = layout.kind[features]
    first = int(layout.hist_bins[features[0]])
    total_q = hists[:, :first, :k].sum(axis=1, dtype=np.int64)  # (n, K)
    total_c = hists[:, :first, k].sum(axis=1, dtype=np.int64)
    parent = leaf_gain(total_q, total_c[:, None], scales, params.lambda_l2).sum(axis=-1)
    splittable = total_c >= 2 * params.min_data_in_leaf
    best: list[Split | None] = [None] * n

    def offer(node: int, split: Split) -> None:
        current = best[node]
        if current is None or split.gain > current.gain:
            best[node] = split

    at = 0
    for kind, width, missing in ((NAMED, NAMED_HIST_BINS, True), (OBFUSCATED, OBFUSCATED_BINS, False)):
        block = features[kinds == kind]
        if not block.size:
            continue
        cells = hists[:, at : at + width * block.size].reshape(n, block.size, width, k + 1)
        gains, flat, left = _numeric_block_multi(cells, total_q, total_c, parent, scales, params, missing=missing)
        choice = np.argmax(gains, axis=1)
        for node in np.flatnonzero(splittable):
            i = int(choice[node])
            if gains[node, i] > 0.0:
                candidate = int(flat[node, i])
                threshold, missing_left = (candidate // 2, candidate % 2 == 0) if missing else (candidate, True)
                offer(
                    int(node),
                    Split(
                        gain=float(gains[node, i]),
                        feature=int(block[i]),
                        kind=NUMERIC,
                        threshold=threshold,
                        missing_left=missing_left,
                        cat_left=0,
                        left_count=int(left[node, i]),
                        right_count=int(total_c[node]) - int(left[node, i]),
                    ),
                )
        at += width * block.size
    for feature in features[kinds == CATEGORICAL]:  # in feature order, each its own block of cells
        width = int(layout.hist_bins[feature])
        for node in np.flatnonzero(splittable):
            found = _categorical_multi(
                hists[node, at : at + width], total_q[node], int(total_c[node]), float(parent[node]), scales, params
            )
            if found is not None:
                gain, mask, left_c = found
                offer(
                    int(node),
                    Split(
                        gain=gain,
                        feature=int(feature),
                        kind=CATEGORY,
                        threshold=0,
                        missing_left=False,
                        cat_left=mask,
                        left_count=left_c,
                        right_count=int(total_c[node]) - left_c,
                    ),
                )
        at += width
    return best
