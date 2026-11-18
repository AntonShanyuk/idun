"""A brute-force NumPy oracle of a whole fit, independent of idun's histogram, split and kernel code.

It follows the written rules of docs/architecture/determinism.md directly: per node, every threshold of every sampled
feature is evaluated from row masks (no histograms, no subtraction), the categorical search is LightGBM's procedure in
plain Python, leaf sums are Python integers, and prediction walks each tree per row. The only shared pieces are the
specified primitives: `idun.rng.uniform_bits` (pinned by known answers from C) and NumPy's PCG64 draws.
Float64 expressions keep the operation order of the specification, so results are comparable with ``==``.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun import rng
from idun.gbm import Params
from tests.unit.gbm.tiny import TinyData

PASS, NUMERIC, CATEGORY = 0, 1, 2


@dataclass
class OracleFit:
    kind: NDArray[np.uint8]
    feature: NDArray[np.int32]
    threshold: NDArray[np.uint8]
    missing_left: NDArray[np.uint8]
    cat_left: NDArray[np.uint64]
    leaf_value: NDArray[np.float32]
    pred: NDArray[np.float32]  # training rows, after the last tree
    left: NDArray[np.int16] | None = None  # leaf-wise: child pointers (>= 0 a node, else leaf -child - 1)
    right: NDArray[np.int16] | None = None


def feature_bins(data: TinyData) -> tuple[NDArray[np.int64], list[str], list[int]]:
    """All features as one ``(features, rows)`` bin matrix, with their kinds and real bin counts."""
    bins = np.concatenate([data.named_bins, data.codes, data.categorical]).astype(np.int64)
    kinds = (
        ["named"] * len(data.named_features) + ["obf"] * len(data.obfuscated_features) + ["cat"] * len(data.categories)
    )
    real = [*data.named_n_bins, *[5] * len(data.obfuscated_features), *(len(levels) for levels in data.categories)]
    return bins, kinds, real


def gain_of(sum_q: int, hessian: float, scale: float, lam: float, p: Params | None = None) -> float:
    """``G² / (H + λ)``; with ``p``'s L1 or step limit, LightGBM's ``GetLeafGain``."""
    g = sum_q * scale
    if p is None or (p.lambda_l1 == 0.0 and p.max_delta_step == 0.0):
        return g * g / (hessian + lam)
    shrunk = math.copysign(max(abs(g) - p.lambda_l1, 0.0), g) if g != 0.0 else 0.0
    if p.max_delta_step == 0.0:
        return shrunk * shrunk / (hessian + lam)
    w = min(max(-shrunk / (hessian + lam), -p.max_delta_step), p.max_delta_step)
    return -(2.0 * shrunk * w + (hessian + lam) * w * w)


def enough(count: int, hessian: float, p: Params, lam: float) -> bool:
    """A side of a split has the rows and the hessian sum it needs, and a positive ``H + λ``."""
    return count >= p.min_data_in_leaf and hessian >= p.min_sum_hessian_in_leaf and hessian + lam > 0.0


def best_split(
    rows: NDArray[np.bool_],
    q: NDArray[np.int64],
    bins: NDArray[np.int64],
    kinds: list[str],
    real: list[int],
    features: list[int],
    p: Params,
    scale: float,
    q_h: NDArray[np.int64] | None = None,
    h_scale: float = 1.0,
    only: dict[int, int] | None = None,
) -> tuple[float, int, int, int, bool, int, int] | None:
    """(gain, feature, kind, threshold, missing_left, cat_mask, left_count) by exhaustive search, or None.

    First-order splits take the row count as the hessian; with ``q_h`` (second-order) it is ``Σq_h · h_scale``.
    ``only``: per numeric feature the one threshold to try (extra_trees; -2 none).
    """
    unit = np.ones(q.size, np.int64) if q_h is None else q_h  # the count: one per row, at scale 1.0
    total_q, total_c, total_h = int(q[rows].sum()), int(rows.sum()), int(unit[rows].sum())
    if total_c < 2 * p.min_data_in_leaf or total_h * h_scale < 2 * p.min_sum_hessian_in_leaf:
        return None
    parent = gain_of(total_q, total_h * h_scale, scale, p.lambda_l2, p)
    best: tuple[float, int, int, int, bool, int, int] | None = None

    def consider(gain: float, candidate: tuple[int, int, int, bool, int, int]) -> None:
        nonlocal best
        if gain > p.min_gain_to_split and (best is None or gain > best[0]):
            best = (gain, *candidate)

    for f in features:
        b = bins[f]
        if kinds[f] in ("named", "obf"):
            directions = (True, False) if kinds[f] == "named" else (True,)
            for t in range(real[f]):
                if only is not None and t != only[f]:
                    continue
                for missing_left in directions:
                    go_left = rows & (((b <= t) & (b != 255)) | ((b == 255) & missing_left))
                    lq, lc, lh = int(q[go_left].sum()), int(go_left.sum()), int(unit[go_left].sum())
                    rq, rc, hl, hr = total_q - lq, total_c - lc, lh * h_scale, (total_h - lh) * h_scale
                    if not (enough(lc, hl, p, p.lambda_l2) and enough(rc, hr, p, p.lambda_l2)):
                        continue
                    gain = gain_of(lq, hl, scale, p.lambda_l2, p) + gain_of(rq, hr, scale, p.lambda_l2, p) - parent
                    consider(gain, (f, NUMERIC, t, missing_left, 0, lc))
            continue
        cat_q = {c: int(q[rows & (b == c)].sum()) for c in range(real[f])}
        cat_c = {c: int((rows & (b == c)).sum()) for c in range(real[f])}
        cat_h = {c: int(unit[rows & (b == c)].sum()) for c in range(real[f])}
        if p.categorical == "one_vs_rest":
            for c in range(real[f]):
                lc, rc = cat_c[c], total_c - cat_c[c]
                hl, hr = cat_h[c] * h_scale, (total_h - cat_h[c]) * h_scale
                if enough(lc, hl, p, p.lambda_l2) and enough(rc, hr, p, p.lambda_l2):
                    gain = gain_of(cat_q[c], hl, scale, p.lambda_l2, p) + gain_of(
                        total_q - cat_q[c], hr, scale, p.lambda_l2, p
                    )
                    consider(gain - parent, (f, CATEGORY, 0, False, 1 << c, lc))
            continue
        used = [
            c for c in range(real[f]) if cat_c[c] >= max(p.cat_smooth, 1.0) and cat_h[c] * h_scale + p.cat_smooth > 0
        ]
        used.sort(key=lambda c: (cat_q[c] * scale) / (cat_h[c] * h_scale + p.cat_smooth))
        limit = min(p.max_cat_threshold, (len(used) + 1) // 2)
        lam = p.lambda_l2 + p.cat_l2
        for order in (used, used[::-1]):
            lq = lc = lh = group = mask = 0
            for c in order[:limit]:
                lq, lc, group, mask = lq + cat_q[c], lc + cat_c[c], group + cat_c[c], mask | (1 << c)
                lh += cat_h[c]
                hl, hr = lh * h_scale, (total_h - lh) * h_scale
                if not enough(lc, hl, p, lam):
                    continue
                rc = total_c - lc
                if not enough(rc, hr, p, lam) or rc < p.min_data_per_group:
                    break
                if group < p.min_data_per_group:
                    continue
                group = 0
                gain = gain_of(lq, hl, scale, lam, p) + gain_of(total_q - lq, hr, scale, lam, p) - parent
                consider(gain, (f, CATEGORY, 0, False, mask, lc))
    return best


def leaf_value(g: float, curvature: float, p: Params) -> np.float32:
    """``float32(−G / (H + λ) · lr)``; with L1 or a step limit, LightGBM's ``CalculateSplittedLeafOutput``."""
    if p.lambda_l1 == 0.0 and p.max_delta_step == 0.0:
        return np.float32(-g / (curvature + p.lambda_l2) * p.learning_rate)
    shrunk = math.copysign(max(abs(g) - p.lambda_l1, 0.0), g) if g != 0.0 else 0.0
    step = -shrunk / (curvature + p.lambda_l2)
    if p.max_delta_step > 0.0:
        step = min(max(step, -p.max_delta_step), p.max_delta_step)
    return np.float32(step * p.learning_rate)


def goes_right(kind: int, b: int, threshold: int, missing_left: int, cat_left: int) -> int:
    if kind == PASS:
        return 0
    if kind == CATEGORY:
        return 0 if (cat_left >> b) & 1 else 1
    if b == 255:
        return 0 if missing_left else 1
    return 1 if b > threshold else 0


def fit(
    p: Params, data: TinyData, lo: int, hi: int, objective: Any = None, y: NDArray[np.float32] | None = None
) -> OracleFit:
    """Train ``p.num_trees`` trees on rows ``[lo, hi)`` the slow, obvious way (level by level, or best-first).

    ``objective``: a custom objective (`idun.gbm.objective`); its gradients steer the splits and the leaves, and its
    hessians, if it has them, the leaves' denominators, and with ``p.second_order`` the splits' too. ``y``: the labels
    in place of the panel's target.
    """
    bins_all, kinds, real = feature_bins(data)
    bins = bins_all[:, lo:hi]
    n, n_features = hi - lo, bins.shape[0]
    y = (data.target[lo:hi] if y is None else y).astype(np.float32)
    pred = np.zeros(n, np.float32)
    n_internal, n_leaves = (p.max_leaves - 1, p.max_leaves) if p.leafwise else (2**p.max_depth - 1, 2**p.max_depth)
    out = OracleFit(
        kind=np.zeros((p.num_trees, n_internal), np.uint8),
        feature=np.zeros((p.num_trees, n_internal), np.int32),
        threshold=np.zeros((p.num_trees, n_internal), np.uint8),
        missing_left=np.zeros((p.num_trees, n_internal), np.uint8),
        cat_left=np.zeros((p.num_trees, n_internal), np.uint64),
        leaf_value=np.zeros((p.num_trees, n_leaves), np.float32),
        pred=pred,
        left=np.full((p.num_trees, n_internal), -1, np.int16) if p.leafwise else None,
        right=np.full((p.num_trees, n_internal), -1, np.int16) if p.leafwise else None,
    )
    in_bag = np.ones(n, bool)
    half = p.grad_quant_bins // 2
    for t in range(p.num_trees):
        if p.bagging_fraction < 1.0 and p.bagging_freq > 0 and t % p.bagging_freq == 0:
            draws = np.random.Generator(np.random.PCG64(np.random.SeedSequence([p.seed, t, 0]))).random(
                n, dtype=np.float32
            )
            in_bag = draws < np.float32(p.bagging_fraction)
        g, h = (pred - y).astype(np.float32), None
        if objective is not None and objective.constant_hessian:
            objective.gradients(pred, y, g)
        elif objective is not None:
            h = np.zeros(n, np.float32)
            objective.gradients(pred, y, g, h)
        largest = float(np.abs(g).max())
        scale = largest / half if largest > 0 else 0.0
        inv = np.float32(1.0 / scale) if largest > 0 else np.float32(0.0)
        u = rng.uniform_bits(rng.iteration_key(p.seed, t), hi)[lo:].astype(np.float32) * np.float32(2.0**-24)
        x = g * inv
        v = np.where(g >= 0, x + u, x - u).astype(np.float32)
        q = np.clip(np.trunc(v), -half, half).astype(np.int64)
        q[~in_bag] = 0
        q_h, h_scale = None, 1.0
        if h is not None and p.second_order:  # second-order splits: the hessians rounded as the gradients, same draws
            high = float(h.max())
            h_scale = high / half if high > 0 else 0.0
            inv_h = np.float32(1.0 / h_scale) if high > 0 else np.float32(0.0)
            q_h = np.clip(np.trunc((h * inv_h + u).astype(np.float32)), -half, half).astype(np.int64)
            q_h[~in_bag] = 0
        k = max(int(n_features * p.feature_fraction + 0.5), 1)
        if k == n_features:
            features = list(range(n_features))
        else:
            generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([p.seed, t, 1])))
            features = sorted(int(f) for f in generator.choice(n_features, size=k, replace=False))
        search = Search(in_bag, q, bins, kinds, real, features, p, scale, q_h, h_scale, t)
        node = _grow_best_first(out, t, search) if p.leafwise else _grow_levels(out, t, search)
        for leaf in range(n_leaves):
            members = in_bag & (node == leaf)
            total = sum(int(np.rint(np.float32(gi) * np.float32(2.0**32))) for gi in g[members])
            count = int(members.sum())
            curvature = float(count)
            if h is not None:
                curvature = sum(int(np.rint(np.float32(hv) * np.float32(2.0**32))) for hv in h[members]) * 2.0**-32
            out.leaf_value[t, leaf] = 0.0 if count == 0 else leaf_value(total * 2.0**-32, curvature, p)
        pred[:] = (pred + out.leaf_value[t, node]).astype(np.float32)
    return out


@dataclass
class Search:
    """One tree's split search: the bag, quantized gradients, bins and sampled features."""

    in_bag: NDArray[np.bool_]
    q: NDArray[np.int64]
    bins: NDArray[np.int64]
    kinds: list[str]
    real: list[int]
    features: list[int]
    p: Params
    scale: float
    q_h: NDArray[np.int64] | None = None
    h_scale: float = 1.0
    tree: int = 0

    def best(
        self, rows: NDArray[np.bool_], node: tuple[int, int]
    ) -> tuple[float, int, int, int, bool, int, int] | None:
        """The best split of ``rows``, node ``node`` = (heap index, 0) depth-wise, (leaves so far, leaf) leaf-wise."""
        features, only, p, k = self.features, None, self.p, len(self.features)
        if p.feature_fraction_bynode < 1.0:  # LightGBM's per-node sample of the tree's features
            m = max(int(k * p.feature_fraction_bynode + 0.5), 1)
            draws = np.random.Generator(np.random.PCG64(np.random.SeedSequence([p.seed, self.tree, 2, *node])))
            features = [self.features[i] for i in sorted(draws.choice(k, size=m, replace=False))]
        if p.extra_trees:  # one threshold per numeric feature: 24 bits of idun.rng, t = u · (real − 1) / 2^24
            u = rng.uniform_bits(rng.derive_key(p.seed, 11, self.tree, *node), k)
            only = {f: int(u[i]) * (self.real[f] - 1) >> 24 if self.real[f] >= 2 else -2
                    for i, f in enumerate(self.features)}  # fmt: skip
        return best_split(self.in_bag & rows, self.q, self.bins, self.kinds, self.real, features, self.p,
                          self.scale, self.q_h, self.h_scale, only)  # fmt: skip

    def right(self, i: int, kind: int, f: int, threshold: int, missing_left: int, mask: int) -> int:
        return goes_right(kind, int(self.bins[f, i]), threshold, missing_left, mask)


def _grow_levels(out: OracleFit, t: int, s: Search) -> NDArray[np.int64]:
    """Depth-wise: every level's open nodes, the leaf cap keeping the highest gains. Returns each row's leaf."""
    n = s.bins.shape[1]
    node = np.zeros(n, np.int64)
    open_nodes = {0}
    leaves = 1
    for d in range(s.p.max_depth):
        found = {}
        for j in sorted(open_nodes):
            split = s.best(node == j, (2**d - 1 + j, 0))
            if split is not None:
                found[j] = split
        keep = sorted(sorted(found, key=lambda j: (-found[j][0], j))[: s.p.max_leaves - leaves])
        leaves += len(keep)
        for j in keep:
            h = 2**d - 1 + j
            _, f, kind, threshold, missing_left, mask, _ = found[j]
            out.kind[t, h], out.feature[t, h], out.threshold[t, h] = kind, f, threshold
            out.missing_left[t, h], out.cat_left[t, h] = missing_left, mask
        right = np.zeros(n, np.int64)
        for i in range(n):
            h = 2**d - 1 + int(node[i])
            right[i] = s.right(i, int(out.kind[t, h]), int(out.feature[t, h]), int(out.threshold[t, h]),
                               int(out.missing_left[t, h]), int(out.cat_left[t, h]))  # fmt: skip
        node = 2 * node + right
        open_nodes = {c for j in keep for c in (2 * j, 2 * j + 1)}
    return node


def _grow_best_first(out: OracleFit, t: int, s: Search) -> NDArray[np.int64]:
    """Leaf-wise: split the leaf of highest gain (ties: lowest id) until max_leaves. Returns each row's leaf id.

    Splitting leaf ``l`` as the tree's ``m``-th split makes node ``m`` with children leaf ``l`` (left) and leaf
    ``m + 1`` (right), and points the node above at ``m``.
    """
    assert out.left is not None
    assert out.right is not None
    n, p = s.bins.shape[1], s.p
    node = np.zeros(n, np.int64)
    depth, above = {0: 0}, {0: (-1, 0)}
    found = {}

    leaves = 1

    def consider(leaf: int) -> None:
        if p.max_depth == -1 or depth[leaf] < p.max_depth:
            split = s.best(node == leaf, (leaves, leaf))
            if split is not None:
                found[leaf] = split

    consider(0)
    while leaves < p.max_leaves and found:
        leaf = max(found, key=lambda j: (found[j][0], -j))
        _, f, kind, threshold, missing_left, mask, _ = found.pop(leaf)
        m, new = leaves - 1, leaves
        leaves += 1
        out.kind[t, m], out.feature[t, m], out.threshold[t, m] = kind, f, threshold
        out.missing_left[t, m], out.cat_left[t, m] = missing_left, mask
        out.left[t, m], out.right[t, m] = -leaf - 1, -new - 1
        up, side = above[leaf]
        if up >= 0:
            (out.right if side else out.left)[t, up] = m
        above[leaf], above[new] = (m, 0), (m, 1)
        for i in np.flatnonzero(node == leaf):
            if s.right(int(i), kind, f, threshold, int(missing_left), mask):
                node[i] = new
        depth[leaf] += 1
        depth[new] = depth[leaf]
        consider(leaf)
        consider(new)
    return node


def predict(fit_: OracleFit, data: TinyData, lo: int, hi: int) -> NDArray[np.float32]:
    """Walk every tree for every row of ``[lo, hi)`` and add the leaves in tree order, in float32."""
    bins, _, _ = feature_bins(data)
    depth = fit_.leaf_value.shape[1].bit_length() - 1
    out = np.zeros(hi - lo, np.float32)
    for i, r in enumerate(range(lo, hi)):
        acc = np.float32(0.0)
        for t in range(fit_.leaf_value.shape[0]):

            def right(h: int, t: int = t, r: int = r) -> int:
                return goes_right(int(fit_.kind[t, h]), int(bins[fit_.feature[t, h], r]), int(fit_.threshold[t, h]),
                                  int(fit_.missing_left[t, h]), int(fit_.cat_left[t, h]))  # fmt: skip

            if fit_.left is not None and fit_.right is not None:
                child = 0
                while child >= 0:
                    child = int(fit_.right[t, child] if right(child) else fit_.left[t, child])
                leaf = -child - 1
            else:
                h = 0
                for _ in range(depth):
                    h = 2 * h + 1 + right(h)
                leaf = h - (2**depth - 1)
            acc = np.float32(acc + fit_.leaf_value[t, leaf])
        out[i] = acc
    return out
