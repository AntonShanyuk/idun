"""TreeSHAP contributions: each feature's share of a raw prediction (Lundberg, Erion and Lee 2018, path-dependent).

For one tree the expected output over its training rows, each leaf's value weighted by its in-bag rows (the tree's
covers, `Forest.leaf_count`), plus each feature's Shapley value under those covers, is the tree's output on the row;
summed over the trees, a row's contributions and the expected value add up to the model's raw prediction. The
layout is LightGBM's ``predict(pred_contrib=True)``: ``(n, n_features + 1)``, the last column the expected value.
Computed on the host from the binned rows, in float64, so every backend's model gives the same numbers. The algorithm
is the polynomial one (the shap package's path bookkeeping, `_extend`, `_unwind` and `_unwound_sum`; its recursion
as a stack, `_row`: the recursive form crashed when it ran from numba's cache); a depth-wise
node that did not split sends every row left and passes through to its left child.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun._jit import njit
from idun.gbm.backends.cpu import _bin, _goes_right
from idun.gbm.features import BinnedData, FeatureLayout
from idun.gbm.split import PASS
from idun.gbm.tree import NodeForest, Trees

Nodes = tuple[NDArray[Any], ...]  # one tree as node arrays: feature, kind, threshold, missing_left, cat_left,
# left, right (a leaf's are -1), value, cover; node 0 is the root


def tree_nodes(forest: Trees, tree: int) -> Nodes:
    """Return tree ``tree`` as node arrays, leaves included, with each node's cover (the in-bag rows below it)."""
    if forest.leaf_count is None:
        raise ValueError("the model has no leaf counts (its file has no leaf_count): refit it for its contributions")
    rows: list[list[Any]] = []  # feature, kind, threshold, missing_left, cat_left, left, right, value, cover

    def leaf(index: int) -> int:
        cover = float(forest.leaf_count[tree, index])  # type: ignore[index]
        rows.append([0, PASS, 0, 0, 0, -1, -1, float(forest.leaf_value[tree, index]), cover])
        return len(rows) - 1

    def internal(at: int) -> tuple[Any, ...]:
        return (int(forest.feature[tree, at]), int(forest.kind[tree, at]), int(forest.threshold[tree, at]),
                int(forest.missing_left[tree, at]), int(forest.cat_left[tree, at]))  # fmt: skip

    def node(child: int) -> int:  # the node-list's pointer (leaf-wise) or the heap index (depth-wise)
        if isinstance(forest, NodeForest):
            if child < 0:
                return leaf(~child)
            if int(forest.kind[tree, child]) == PASS:  # a tree that never split: its root, to leaf 0
                return leaf(~int(forest.left[tree, child]))
            left_child, right_child = int(forest.left[tree, child]), int(forest.right[tree, child])
        else:
            internal_nodes = forest.kind.shape[1]
            if child >= internal_nodes:
                return leaf(child - internal_nodes)
            if int(forest.kind[tree, child]) == PASS:  # every row goes left: the node passes through
                return node(2 * child + 1)
            left_child, right_child = 2 * child + 1, 2 * child + 2
        rows.append([*internal(child), 0, 0, 0.0, 0.0])
        mine = len(rows) - 1
        left, right = node(left_child), node(right_child)
        rows[mine][5], rows[mine][6], rows[mine][8] = left, right, rows[left][8] + rows[right][8]
        return mine

    node(0)
    columns = list(zip(*rows, strict=True))
    dtypes = (np.int32, np.uint8, np.uint8, np.uint8, np.uint64, np.int64, np.int64, np.float64, np.float64)
    return tuple(np.asarray(column, dtype) for column, dtype in zip(columns, dtypes, strict=True))


@njit()
def _extend(fi: Any, zf: Any, of: Any, pw: Any, base: int, depth: int, zero: float, one: float, feature: int) -> None:
    fi[base + depth], zf[base + depth], of[base + depth] = feature, zero, one
    pw[base + depth] = 1.0 if depth == 0 else 0.0
    for i in range(depth - 1, -1, -1):
        pw[base + i + 1] += one * pw[base + i] * (i + 1) / (depth + 1)
        pw[base + i] = zero * pw[base + i] * (depth - i) / (depth + 1)


@njit()
def _unwind(fi: Any, zf: Any, of: Any, pw: Any, base: int, depth: int, index: int) -> None:
    one, zero = of[base + index], zf[base + index]
    next_one = pw[base + depth]
    for i in range(depth - 1, -1, -1):
        if one != 0.0:
            kept = pw[base + i]
            pw[base + i] = next_one * (depth + 1) / ((i + 1) * one)
            next_one = kept - pw[base + i] * zero * (depth - i) / (depth + 1)
        else:
            pw[base + i] = pw[base + i] * (depth + 1) / (zero * (depth - i))
    for i in range(index, depth):
        fi[base + i], zf[base + i], of[base + i] = fi[base + i + 1], zf[base + i + 1], of[base + i + 1]


@njit()
def _unwound_sum(zf: Any, of: Any, pw: Any, base: int, depth: int, index: int) -> float:
    one, zero = of[base + index], zf[base + index]
    next_one, total = pw[base + depth], 0.0
    for i in range(depth - 1, -1, -1):
        if one != 0.0:
            part = next_one * (depth + 1) / ((i + 1) * one)
            total += part
            next_one = pw[base + i] - part * zero * ((depth - i) / (depth + 1))
        else:
            total += (pw[base + i] / zero) / ((depth - i) / (depth + 1))
    return total


@njit()
def _row(
    split: Any, left: Any, right: Any, value: Any, cover: Any, go_right: Any, phi: Any, fi: Any, zf: Any, of: Any,
    pw: Any, frames: Any, fractions: Any,
) -> None:  # fmt: skip
    """Add one row's contributions to ``phi``: the recursion over the tree as a stack, in the same order.

    Each frame is a node to visit with its depth, the buffer offset of its parent's path, the fractions and the
    feature it extends the path with; a level's path keeps its own slice of the buffer, and the children write above
    it, so a cold child visited after its hot sibling's subtree still reads its parent's path.
    """
    top = 0
    frames[0, 0], frames[0, 1], frames[0, 2], frames[0, 3] = 0, 0, 0, -1
    fractions[0, 0], fractions[0, 1] = 1.0, 1.0
    while top >= 0:
        node, depth, parent, feature = frames[top, 0], frames[top, 1], frames[top, 2], frames[top, 3]
        zero, one = fractions[top, 0], fractions[top, 1]
        top -= 1
        base = parent + depth + 1
        for i in range(depth + 1):
            fi[base + i], zf[base + i], of[base + i], pw[base + i] = (
                fi[parent + i],
                zf[parent + i],
                of[parent + i],
                pw[parent + i],
            )
        _extend(fi, zf, of, pw, base, depth, zero, one, feature)
        if left[node] < 0:
            for i in range(1, depth + 1):
                weight = _unwound_sum(zf, of, pw, base, depth, i)
                phi[fi[base + i]] += weight * (of[base + i] - zf[base + i]) * value[node]
            continue
        hot, cold = (right[node], left[node]) if go_right[node] else (left[node], right[node])
        incoming_zero, incoming_one, index = 1.0, 1.0, depth + 1
        for k in range(depth + 1):
            if fi[base + k] == split[node]:
                index = k
                break
        if index != depth + 1:  # split on this feature above: undo it, to redo it here
            incoming_zero, incoming_one = zf[base + index], of[base + index]
            _unwind(fi, zf, of, pw, base, depth, index)
            depth -= 1
        for child, one_fraction in ((cold, 0.0), (hot, incoming_one)):  # the hot child on top: visited first
            top += 1
            frames[top, 0], frames[top, 1], frames[top, 2], frames[top, 3] = child, depth + 1, base, split[node]
            fractions[top, 0], fractions[top, 1] = cover[child] / cover[node] * incoming_zero, one_fraction


@njit()
def _tree_contributions(
    kinds: Any, columns: Any, split: Any, kind: Any, threshold: Any, missing_left: Any, cat_left: Any, left: Any,
    right: Any, value: Any, cover: Any, named: Any, packed: Any, categorical: Any, lo: np.int64, depth: int, out: Any,
) -> None:  # fmt: skip
    size = (depth + 2) * (depth + 3) // 2 + depth + 3
    fi, zf = np.zeros(size, np.int64), np.zeros(size, np.float64)
    of, pw = np.zeros(size, np.float64), np.zeros(size, np.float64)
    go_right = np.zeros(left.size, np.bool_)
    frames, fractions = np.zeros((2 * depth + 4, 4), np.int64), np.zeros((2 * depth + 4, 2), np.float64)
    for i in range(out.shape[0]):
        for m in range(left.size):
            if left[m] >= 0:
                f = split[m]
                b = _bin(kinds[f], columns[f], lo + i, named, packed, categorical)
                go_right[m] = _goes_right(kind[m], b, threshold[m], missing_left[m], cat_left[m])
        _row(split, left, right, value, cover, go_right, out[i], fi, zf, of, pw, frames, fractions)


def contributions(forest: Trees, data: BinnedData, lo: int, hi: int, n_features: int) -> NDArray[np.float64]:
    """Return rows ``[lo, hi)``'s contributions ``(n, n_features + 1)``: each feature's, then the expected value."""
    layout = FeatureLayout.of(data)
    out = np.zeros((hi - lo, n_features + 1), np.float64)
    for tree in range(forest.n_trees):
        split, kind, threshold, missing_left, cat_left, left, right, value, cover = tree_nodes(forest, tree)
        depth = _depth(left, right)
        _tree_contributions(layout.kind, layout.column, split, kind, threshold, missing_left, cat_left, left, right,
                            value, cover, data.named_bins, data.obfuscated, data.categorical, np.int64(lo), depth,
                            out)  # fmt: skip
        leaves = left < 0
        out[:, n_features] += float(np.sum(value[leaves] * cover[leaves]) / cover[0]) if cover[0] > 0 else value[0]
    return out


def _depth(left: NDArray[np.int64], right: NDArray[np.int64]) -> int:
    """Return the tree's depth: the longest walk from the root to a leaf."""
    depth = np.zeros(left.size, np.int64)
    for m in range(left.size):  # children come after their parent
        if left[m] >= 0:
            depth[left[m]] = depth[right[m]] = depth[m] + 1
    return int(depth.max())


@njit()
def _leaves_heap(
    kinds: Any, columns: Any, kind: Any, feature: Any, threshold: Any, missing_left: Any, cat_left: Any, named: Any,
    packed: Any, categorical: Any, lo: np.int64, depth: np.int64, out: Any,
) -> None:  # fmt: skip
    internal = (np.int64(1) << depth) - 1
    for i in range(out.shape[0]):
        for t in range(out.shape[1]):
            h = np.int64(0)
            for _ in range(depth):
                f = feature[t, h]
                b = _bin(kinds[f], columns[f], lo + i, named, packed, categorical)
                h = 2 * h + 1 + _goes_right(kind[t, h], b, threshold[t, h], missing_left[t, h], cat_left[t, h])
            out[i, t] = h - internal


@njit()
def _leaves_nodes(
    kinds: Any, columns: Any, kind: Any, feature: Any, threshold: Any, missing_left: Any, cat_left: Any, left: Any,
    right: Any, named: Any, packed: Any, categorical: Any, lo: np.int64, out: Any,
) -> None:  # fmt: skip
    internal = left.shape[1]
    for i in range(out.shape[0]):
        for t in range(out.shape[1]):
            h, child = 0, -1
            for _ in range(internal):
                f = feature[t, h]
                b = _bin(kinds[f], columns[f], lo + i, named, packed, categorical)
                go = _goes_right(kind[t, h], b, threshold[t, h], missing_left[t, h], cat_left[t, h])
                child = right[t, h] if go else left[t, h]
                if child < 0:
                    break
                h = child
            out[i, t] = ~child


def leaf_indices(forest: Trees, data: BinnedData, lo: int, hi: int) -> NDArray[np.int32]:
    """Return the leaf each of rows ``[lo, hi)`` reaches in each tree, ``(n, n_trees)`` (LightGBM's ``pred_leaf``)."""
    layout = FeatureLayout.of(data)
    out = np.zeros((hi - lo, forest.n_trees), np.int32)
    arrays = (forest.kind, forest.feature, forest.threshold, forest.missing_left, forest.cat_left)
    if isinstance(forest, NodeForest):
        _leaves_nodes(layout.kind, layout.column, *arrays, forest.left, forest.right, data.named_bins, data.obfuscated,
                      data.categorical, np.int64(lo), out)  # fmt: skip
    else:
        _leaves_heap(layout.kind, layout.column, *arrays, data.named_bins, data.obfuscated, data.categorical,
                     np.int64(lo), np.int64(forest.depth), out)  # fmt: skip
    return out
