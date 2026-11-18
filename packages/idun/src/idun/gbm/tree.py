"""Trees: complete binary heaps (depth-wise) and node lists (leaf-wise); docs/architecture/data-layout.md §6.

Depth-wise: a forest of ``T`` depth-``D`` trees is a set of ``(T, 2^D − 1)`` arrays for the internal nodes and a
``(T, 2^D)`` array of leaf values. Traversal: ``h = 0``, then ``D`` times ``h = 2h + 1 + go_right``, then ``leaf = h −
(2^D − 1)``. A node on level ``d`` with index ``j`` within its level sits at ``h = 2^d − 1 + j``; its children are
``2j`` and ``2j + 1`` on level ``d + 1``.

Leaf-wise (`NodeForest`, LightGBM's layout): ``(T, L − 1)`` internal nodes and ``(T, L)`` leaf values for at most
``L`` leaves per tree. Node ``m`` sends a row to ``left[m]`` or ``right[m]``: an internal node when the child is
``>= 0``, else leaf ``~child``. The root is node 0. Splitting leaf ``l`` as a tree's ``m``-th split makes node ``m``,
whose left child is leaf ``l`` and right child leaf ``m + 1``, and points the parent's child at ``m``; so every
child node comes after its parent and a walk ends within ``L − 1`` steps. A tree that never split keeps a pass-through
root whose children are both leaf 0.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Forest:
    """Internal-node fields and leaf values of ``n_trees`` complete trees of depth ``depth``."""

    kind: NDArray[np.uint8]  # PASS, NUMERIC or CATEGORY (idun.gbm.split)
    feature: NDArray[np.int32]  # loader-order feature index
    threshold: NDArray[np.uint8]  # numeric: bins <= threshold go left
    missing_left: NDArray[np.uint8]  # numeric: where bin 255 goes
    cat_left: NDArray[np.uint64]  # categorical: bitmask of codes that go left
    leaf_value: NDArray[np.float32]  # (n_trees, 2^depth), learning rate included
    leaf_count: NDArray[np.int32] | None = None  # (n_trees, 2^depth) in-bag rows per leaf: the covers TreeSHAP reads

    @classmethod
    def empty(cls, n_trees: int, depth: int) -> Self:
        """Return a forest of pass-through trees with zero leaves."""
        internal = (n_trees, 2**depth - 1)
        return cls(
            kind=np.zeros(internal, np.uint8),
            feature=np.zeros(internal, np.int32),
            threshold=np.zeros(internal, np.uint8),
            missing_left=np.zeros(internal, np.uint8),
            cat_left=np.zeros(internal, np.uint64),
            leaf_value=np.zeros((n_trees, 2**depth), np.float32),
            leaf_count=np.zeros((n_trees, 2**depth), np.int32),
        )

    @property
    def n_trees(self) -> int:
        """Number of trees."""
        return int(self.leaf_value.shape[0])

    @property
    def depth(self) -> int:
        """Depth of every tree."""
        return int(self.leaf_value.shape[1]).bit_length() - 1

    def head(self, n_trees: int) -> Self:
        """Return the first ``n_trees`` trees (views)."""
        if not 0 <= n_trees <= self.n_trees:
            raise ValueError(f"n_trees must be in 0..{self.n_trees}, got {n_trees}")
        counts = None if self.leaf_count is None else self.leaf_count[:n_trees]
        return type(self)(**{name: getattr(self, name)[:n_trees] for name in FIELDS}, leaf_count=counts)

    def trees(self, start: int, stop: int) -> Self:
        """Return trees ``start`` to ``stop - 1`` (views)."""
        if not 0 <= start <= stop <= self.n_trees:
            raise ValueError(f"trees [{start}, {stop}) are outside the forest's {self.n_trees}")
        counts = None if self.leaf_count is None else self.leaf_count[start:stop]
        return type(self)(**{name: getattr(self, name)[start:stop] for name in FIELDS}, leaf_count=counts)


FIELDS = ("kind", "feature", "threshold", "missing_left", "cat_left", "leaf_value")


@dataclass(frozen=True)
class NodeForest:
    """Internal nodes as lists with child pointers, and leaf values, of ``n_trees`` trees of at most ``max_leaves``."""

    kind: NDArray[np.uint8]  # as Forest, (n_trees, max_leaves − 1)
    feature: NDArray[np.int32]
    threshold: NDArray[np.uint8]
    missing_left: NDArray[np.uint8]
    cat_left: NDArray[np.uint64]
    left: NDArray[np.int16]  # child: internal node when >= 0, else leaf ~left
    right: NDArray[np.int16]
    leaf_value: NDArray[np.float32]  # (n_trees, max_leaves), indexed by leaf id
    leaf_count: NDArray[np.int32] | None = None  # (n_trees, max_leaves) in-bag rows per leaf (TreeSHAP's covers)

    @classmethod
    def empty(cls, n_trees: int, max_leaves: int) -> Self:
        """Return a forest of pass-through trees (every row to leaf 0) with zero leaves."""
        internal = (n_trees, max_leaves - 1)
        return cls(
            kind=np.zeros(internal, np.uint8),
            feature=np.zeros(internal, np.int32),
            threshold=np.zeros(internal, np.uint8),
            missing_left=np.zeros(internal, np.uint8),
            cat_left=np.zeros(internal, np.uint64),
            left=np.full(internal, -1, np.int16),
            right=np.full(internal, -1, np.int16),
            leaf_value=np.zeros((n_trees, max_leaves), np.float32),
            leaf_count=np.zeros((n_trees, max_leaves), np.int32),
        )

    @property
    def n_trees(self) -> int:
        """Number of trees."""
        return int(self.leaf_value.shape[0])

    @property
    def max_leaves(self) -> int:
        """Leaves per tree at most."""
        return int(self.leaf_value.shape[1])

    def head(self, n_trees: int) -> Self:
        """Return the first ``n_trees`` trees (views)."""
        if not 0 <= n_trees <= self.n_trees:
            raise ValueError(f"n_trees must be in 0..{self.n_trees}, got {n_trees}")
        counts = None if self.leaf_count is None else self.leaf_count[:n_trees]
        return type(self)(**{name: getattr(self, name)[:n_trees] for name in NODE_FIELDS}, leaf_count=counts)

    def trees(self, start: int, stop: int) -> Self:
        """Return trees ``start`` to ``stop - 1`` (views)."""
        if not 0 <= start <= stop <= self.n_trees:
            raise ValueError(f"trees [{start}, {stop}) are outside the forest's {self.n_trees}")
        counts = None if self.leaf_count is None else self.leaf_count[start:stop]
        return type(self)(**{name: getattr(self, name)[start:stop] for name in NODE_FIELDS}, leaf_count=counts)

    def check(self) -> None:
        """Raise ``ValueError`` unless every walk ends at a leaf: each child node after its parent, leaves in range."""
        n_internal = self.max_leaves - 1
        if self.leaf_value.ndim != 2 or n_internal < 1:
            raise ValueError("a leaf-wise forest needs (n_trees, max_leaves >= 2) leaf values")
        if any(getattr(self, name).shape != (self.n_trees, n_internal) for name in NODE_FIELDS[:-1]):
            raise ValueError("the internal-node arrays must be (n_trees, max_leaves - 1)")
        node = np.arange(n_internal, dtype=np.int64)
        for children in (self.left.astype(np.int64), self.right.astype(np.int64)):
            inner = (children > node) & (children < n_internal)
            leaf = (children < 0) & (~children < self.max_leaves)
            if not (inner | leaf).all():
                raise ValueError("a child pointer is out of range or points back up the tree")


NODE_FIELDS = ("kind", "feature", "threshold", "missing_left", "cat_left", "left", "right", "leaf_value")

Trees = Forest | NodeForest


def concatenate(first: Trees, second: Trees) -> Trees:
    """Return ``first``'s trees, then ``second``'s: forests of one kind and shape."""
    if type(first) is not type(second) or first.leaf_value.shape[1:] != second.leaf_value.shape[1:]:
        raise ValueError("only forests of one kind and shape concatenate")
    names = NODE_FIELDS if isinstance(first, NodeForest) else FIELDS
    counts = None
    if first.leaf_count is not None and second.leaf_count is not None:
        counts = np.concatenate([first.leaf_count, second.leaf_count])
    joined = {name: np.concatenate([getattr(first, name), getattr(second, name)]) for name in names}
    return type(first)(**joined, leaf_count=counts)


def fields_of(forest: Trees) -> tuple[str, ...]:
    """Return the array fields of ``forest``, in constructor order."""
    return NODE_FIELDS if isinstance(forest, NodeForest) else FIELDS


def empty_trees(n_trees: int, *, leafwise: bool, max_depth: int, max_leaves: int) -> Trees:
    """Return the empty forest of a fit: node lists of ``max_leaves`` leaves, or heaps of depth ``max_depth``."""
    return NodeForest.empty(n_trees, max_leaves) if leafwise else Forest.empty(n_trees, max_depth)
