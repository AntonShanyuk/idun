"""The split search as numpy and Python wrote it before it was compiled: the reference the compiled one must equal.

`idun.gbm.split`'s ``_numeric_block`` and ``_categorical`` give these functions' results bit for bit
(tests/unit/gbm/test_split.py).
"""

import numpy as np
from numpy.typing import NDArray

from idun.gbm.features import MISSING_BIN
from idun.gbm.params import Params
from idun.gbm.split import NodeSearch, leaf_gain


def numeric_block(
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

    ``hessian``: the hessian's channel and scale (the count's and 1.0 for first-order splits); ``search``: each node's
    allowed features and one threshold per feature (`idun.gbm.split.NodeSearch`).
    """
    channel, h_scale = hessian
    real = cells[..., :MISSING_BIN, :] if missing else cells
    left_q = np.cumsum(real[..., 0], axis=-1, dtype=np.int64)
    left_c = np.cumsum(real[..., 1], axis=-1, dtype=np.int64)
    left_h = np.cumsum(real[..., channel], axis=-1, dtype=np.int64)
    if missing:  # candidates per threshold: missing left, then missing right
        miss_q = cells[..., MISSING_BIN, 0].astype(np.int64)[..., None]
        miss_c = cells[..., MISSING_BIN, 1].astype(np.int64)[..., None]
        miss_h = cells[..., MISSING_BIN, channel].astype(np.int64)[..., None]
        left_q = np.stack([left_q + miss_q, left_q], axis=-1).reshape(*cells.shape[:2], -1)
        left_c = np.stack([left_c + miss_c, left_c], axis=-1).reshape(*cells.shape[:2], -1)
        left_h = np.stack([left_h + miss_h, left_h], axis=-1).reshape(*cells.shape[:2], -1)
    right_q, right_c = total_q[:, None, None] - left_q, total_c[:, None, None] - left_c
    h_left = np.multiply(left_h, h_scale, dtype=np.float64)
    h_right = np.multiply(total_h[:, None, None] - left_h, h_scale, dtype=np.float64)
    lam, least = params.lambda_l2, params.min_sum_hessian_in_leaf
    valid = (left_c >= params.min_data_in_leaf) & (right_c >= params.min_data_in_leaf)
    valid &= (h_left >= least) & (h_right >= least) & (h_left + lam > 0.0) & (h_right + lam > 0.0)
    if search is not None:  # the node's features, and per feature every threshold (-1) or the one drawn
        threshold = np.arange(valid.shape[-1]) // (2 if missing else 1)
        only = search.only[..., None]
        valid &= (search.allowed[..., None] == 1) & ((only == -1) | (threshold == only))
    with np.errstate(divide="ignore", invalid="ignore"):  # empty sides with λ = 0: masked out just below
        gain = leaf_gain(left_q, h_left, scale, lam) + leaf_gain(right_q, h_right, scale, lam)
    gain = np.where(valid, gain - parent[:, None, None], -np.inf)
    best = np.argmax(gain, axis=-1)[..., None]
    return (
        np.take_along_axis(gain, best, axis=-1)[..., 0],
        best[..., 0],
        np.take_along_axis(left_c, best, axis=-1)[..., 0],
    )


def categorical(
    cells: NDArray[np.int32],
    total: tuple[int, int, int],
    parent: float,
    scale: float,
    hessian: tuple[int, float],
    params: Params,
) -> tuple[float, int, int] | None:
    """Best (gain, left bitmask, left count) for the categorical feature, or None; ``total`` is (Σq, count, Σh)."""
    q = cells[:, 0].astype(np.int64)
    c = cells[:, 1].astype(np.int64)
    h = cells[:, hessian[0]].astype(np.int64)
    h_scale, least = hessian[1], params.min_sum_hessian_in_leaf

    def fits(left: float, right: float, lam: float) -> bool:
        return left >= least and right >= least and left + lam > 0.0 and right + lam > 0.0

    best: tuple[float, int, int] | None = None
    if params.categorical == "one_vs_rest":
        for cat in range(len(cells)):
            left_c, right_c = int(c[cat]), total[1] - int(c[cat])
            h_left, h_right = int(h[cat]) * h_scale, (total[2] - int(h[cat])) * h_scale
            if left_c < params.min_data_in_leaf or right_c < params.min_data_in_leaf:
                continue
            if not fits(h_left, h_right, params.lambda_l2):
                continue
            gain = float(
                leaf_gain(int(q[cat]), h_left, scale, params.lambda_l2)
                + leaf_gain(total[0] - int(q[cat]), h_right, scale, params.lambda_l2)
                - parent
            )
            if gain > 0.0 and (best is None or gain > best[0]):
                best = (gain, 1 << cat, left_c)
        return best
    lam = params.lambda_l2 + params.cat_l2
    used = [
        cat
        for cat in range(len(cells))
        if c[cat] >= max(params.cat_smooth, 1.0)  # never an empty category,
        and float(h[cat]) * h_scale + params.cat_smooth > 0.0
    ]  # nor one whose hessians give G / 0
    ctr = {cat: float(q[cat] * scale) / (float(h[cat]) * h_scale + params.cat_smooth) for cat in used}
    ordered = sorted(used, key=lambda cat: ctr[cat])  # stable, as LightGBM's std::stable_sort
    max_cats = min(params.max_cat_threshold, (len(used) + 1) // 2)
    for order in (ordered, ordered[::-1]):
        left_q = left_c = left_h = group = mask = 0
        for cat in order[:max_cats]:
            left_q, left_c, left_h, group, mask = (
                left_q + int(q[cat]),
                left_c + int(c[cat]),
                left_h + int(h[cat]),
                group + int(c[cat]),
                mask | 1 << cat,
            )
            h_left, h_right = left_h * h_scale, (total[2] - left_h) * h_scale
            if left_c < params.min_data_in_leaf or h_left < least or h_left + lam <= 0.0:
                continue
            right_c = total[1] - left_c
            if right_c < params.min_data_in_leaf or right_c < params.min_data_per_group:
                break
            if h_right < least or h_right + lam <= 0.0:
                break
            if group < params.min_data_per_group:
                continue
            group = 0
            gain = float(
                leaf_gain(left_q, h_left, scale, lam) + leaf_gain(total[0] - left_q, h_right, scale, lam) - parent
            )
            if gain > 0.0 and (best is None or gain > best[0]):
                best = (gain, mask, left_c)
    return best
