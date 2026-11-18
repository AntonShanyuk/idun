import numpy as np
import pytest

from idun.gbm.tree import FIELDS, NODE_FIELDS, Forest, NodeForest, concatenate, empty_trees, fields_of


def test_an_empty_forest_has_heap_shapes() -> None:
    forest = Forest.empty(3, 4)
    assert (forest.n_trees, forest.depth) == (3, 4)
    assert forest.kind.shape == forest.cat_left.shape == (3, 15)
    assert forest.leaf_value.shape == (3, 16)
    assert forest.cat_left.dtype == np.uint64
    assert not forest.kind.any()


def test_head_takes_the_first_trees() -> None:
    forest = Forest.empty(3, 2)
    forest.leaf_value[:] = np.arange(3, dtype=np.float32)[:, None]
    head = forest.head(2)
    assert head.n_trees == 2
    assert head.leaf_value[:, 0].tolist() == [0, 1]
    assert all(getattr(head, name).base is not None for name in FIELDS)  # views, not copies
    with pytest.raises(ValueError, match="n_trees"):
        forest.head(4)


def test_an_empty_node_forest_sends_every_row_to_leaf_0() -> None:
    forest = NodeForest.empty(3, 5)
    assert (forest.n_trees, forest.max_leaves) == (3, 5)
    assert forest.kind.shape == forest.left.shape == forest.right.shape == (3, 4)
    assert forest.leaf_value.shape == (3, 5)
    assert forest.left.dtype == forest.right.dtype == np.int16
    assert (forest.left == -1).all()
    assert (forest.right == -1).all()
    forest.check()
    assert all(getattr(forest.head(2), name).base is not None for name in NODE_FIELDS)
    with pytest.raises(ValueError, match="n_trees"):
        forest.head(4)


def test_fields_and_empty_trees_follow_the_growth() -> None:
    assert fields_of(Forest.empty(1, 2)) == FIELDS
    assert fields_of(NodeForest.empty(1, 2)) == NODE_FIELDS
    assert isinstance(empty_trees(2, leafwise=True, max_depth=-1, max_leaves=63), NodeForest)
    heap = empty_trees(2, leafwise=False, max_depth=3, max_leaves=8)
    assert isinstance(heap, Forest)
    assert heap.depth == 3


@pytest.mark.parametrize(("side", "value"), [("left", 0), ("right", 4), ("left", -6), ("right", 1)])
def test_check_refuses_pointers_out_of_range_or_back_up(side: str, value: int) -> None:
    forest = NodeForest.empty(1, 5)
    forest.left[0, :2], forest.right[0, :2] = [1, -1], [-2, -3]  # a valid start: node 0 -> node 1
    forest.check()
    getattr(forest, side)[0, 1] = value  # 0: node 1 to itself; 4: no node 4; -6: no leaf 5; 1: itself again
    with pytest.raises(ValueError, match="out of range or points back up"):
        forest.check()


def test_forests_concatenate_and_take_windows() -> None:
    a, b = Forest.empty(2, 3), Forest.empty(3, 3)
    a.leaf_value[:] = 1.0
    b.leaf_value[:] = 2.0
    both = concatenate(a, b)
    assert both.n_trees == 5
    np.testing.assert_array_equal(both.trees(1, 3).leaf_value[:, 0], [1.0, 2.0])
    with pytest.raises(ValueError, match="one kind and shape"):
        concatenate(a, Forest.empty(1, 4))
    with pytest.raises(ValueError, match="outside the forest"):
        both.trees(4, 6)
