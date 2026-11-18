import numpy as np
import pytest

from idun import rng
from idun.gbm.backends import CPUBackend, LeafSplit, LevelSplits
from idun.gbm.features import FeatureLayout
from idun.gbm.tree import Forest, NodeForest
from tests import oracle
from tests.unit.gbm.tiny import make_tiny


def started(n_rows: int = 120, lo: int = 5) -> tuple[CPUBackend, FeatureLayout]:
    data = make_tiny(n_rows + lo + 3, n_countries=6, seed=7)
    layout = FeatureLayout.of(data)
    backend = CPUBackend()
    backend.start(data, layout, lo, data.target[lo : lo + n_rows])
    return backend, layout


def test_gradients_are_pred_minus_y() -> None:
    backend, _ = started()
    backend.pred[:] = 0.25
    largest = backend.gradients()
    np.testing.assert_array_equal(backend.g, np.float32(0.25) - backend.y)
    assert largest == float(np.abs(backend.g).max())


def test_quantization_is_stochastic_rounding_toward_zero() -> None:
    backend, _ = started(lo=5)
    backend.g[:] = np.linspace(-1, 1, backend.g.size, dtype=np.float32)
    bag = np.ones(backend.g.size, np.uint8)
    bag[::7] = 0
    key, inv = rng.iteration_key(9, 2), np.float32(2.0)
    backend.quantize(bag, key, inv, 2)
    u = rng.uniform_bits(key, 5 + backend.g.size)[5:].astype(np.float32) * np.float32(2.0**-24)
    x = backend.g * inv
    expected = np.clip(np.trunc(np.where(backend.g >= 0, x + u, x - u)), -2, 2).astype(np.int8)
    expected[bag == 0] = 0
    np.testing.assert_array_equal(backend.q, expected)
    assert backend.q.min() == -2
    assert backend.q.max() == 2


def test_rounding_is_clamped_to_half_the_bins() -> None:
    backend, _ = started(n_rows=1)  # panel row 5
    key = next(k for k in range(1_000) if rng.uniform_bits(k, 6)[5] >= 1 << 23)  # u >= 0.5
    for g, expected in ((1.25, 2), (-1.25, -2)):  # |x| = 2.5, so |x| + u truncates to 3 before the clamp
        backend.g[:] = np.float32(g)
        backend.quantize(np.ones(1, np.uint8), key, np.float32(2.0), 2)
        assert backend.q.tolist() == [expected]


def test_histograms_count_in_bag_rows_of_selected_nodes() -> None:
    backend, layout = started()
    backend.q[:] = np.arange(backend.q.size) % 5 - 2
    backend.bag = (np.arange(backend.q.size) % 3 != 0).astype(np.uint8)
    backend.node[:] = np.arange(backend.q.size) % 4
    features = np.array([0, 3, 6], np.int32)
    offsets, cells = layout.offsets(features)
    slots = np.array([-1, 0, -1, 1], np.int16)
    hist = backend.histograms(features, offsets, cells, slots)
    bins, _, _ = oracle.feature_bins(backend._data)  # type: ignore[arg-type]
    expected = np.zeros((2, cells, 2), np.int64)
    for i in range(backend.q.size):
        slot = slots[backend.node[i]]
        if backend.bag[i] and slot >= 0:
            for f, offset in zip(features, offsets, strict=True):
                expected[slot, offset + bins[f, 5 + i]] += (backend.q[i], 1)
    np.testing.assert_array_equal(hist, expected)


def test_partition_moves_every_row_to_its_child() -> None:
    backend, _ = started()
    backend.node[:] = np.arange(backend.node.size) % 2
    splits = LevelSplits(
        kind=np.array([1, 2], np.uint8),
        feature=np.array([0, 6], np.int32),  # named feature 0; the country
        threshold=np.array([3, 0], np.uint8),
        missing_left=np.array([0, 0], np.uint8),
        cat_left=np.array([0, 0b10110], np.uint64),
    )
    before = backend.node.copy()
    backend.partition(splits)
    bins, _, _ = oracle.feature_bins(backend._data)  # type: ignore[arg-type]
    for i, at in enumerate(before):
        b = int(bins[0 if at == 0 else 6, 5 + i])
        right = oracle.goes_right(int(splits.kind[at]), b, int(splits.threshold[at]), 0, int(splits.cat_left[at]))
        assert backend.node[i] == 2 * at + right


def test_leaf_sums_are_exact_fixed_point() -> None:
    backend, _ = started()
    backend.pred[:] = np.float32(0.3)
    backend.node[:] = np.arange(backend.node.size) % 3
    backend.bag = (np.arange(backend.node.size) % 4 != 0).astype(np.uint8)
    backend.gradients()
    sums, counts = backend.leaf_sums(4)
    g = backend.pred - backend.y
    for leaf in range(4):
        members = (backend.node == leaf) & (backend.bag == 1)
        assert sums[leaf] == sum(int(np.rint(v * np.float32(2.0**32))) for v in g[members])
        assert counts[leaf] == members.sum()


def test_update_adds_the_leaf_of_every_row() -> None:
    backend, _ = started()
    backend.node[:] = np.arange(backend.node.size) % 2
    backend.update_predictions(np.array([0.5, -0.25], np.float32))
    np.testing.assert_array_equal(backend.pred, np.where(backend.node == 0, 0.5, -0.25).astype(np.float32))


def test_predict_walks_the_heap() -> None:
    data = make_tiny(50, seed=8)
    layout = FeatureLayout.of(data)
    forest = Forest.empty(2, 2)
    forest.kind[0, 0], forest.feature[0, 0], forest.threshold[0, 0] = 1, 2, 1  # obfuscated feature 0 at threshold 1
    forest.leaf_value[0] = [1, 2, 3, 4]
    forest.leaf_value[1] = [10, 20, 30, 40]  # tree 1 passes every row to its leftmost leaf
    out = CPUBackend().predict(data, layout, 3, 50, forest)
    expected = np.where(data.codes[0, 3:] <= 1, 1, 3) + 10
    np.testing.assert_array_equal(out, expected.astype(np.float32))


def test_misuse_raises() -> None:
    backend = CPUBackend()
    with pytest.raises(RuntimeError, match="start"):
        backend.gradients()
    data = make_tiny(40)
    layout = FeatureLayout.of(data)
    with pytest.raises(ValueError, match="outside"):
        backend.start(data, layout, 30, data.target[:20])
    with pytest.raises(ValueError, match="outside"):
        backend.predict(data, layout, 0, 41, Forest.empty(1, 1))
    backend.start(data, layout, 0, data.target[:20])
    with pytest.raises(ValueError, match="bag"):
        backend.quantize(np.ones(19, np.uint8), 0, np.float32(1.0), 2)


def test_split_leaf_moves_only_the_leafs_rows_that_go_right() -> None:
    backend, _ = started()
    backend.node[:] = np.arange(backend.node.size) % 3
    before = backend.node.copy()
    backend.split_leaf(LeafSplit(leaf=1, right=3, kind=2, feature=6, threshold=0, missing_left=0, cat_left=0b10110))
    bins, _, _ = oracle.feature_bins(backend._data)  # type: ignore[arg-type]
    for i, at in enumerate(before):
        right = at == 1 and oracle.goes_right(2, int(bins[6, 5 + i]), 0, 0, 0b10110)
        assert backend.node[i] == (3 if right else at)
    assert (backend.node == 3).any()
    assert ((backend.node == 1) == ((before == 1) & (backend.node != 3))).all()


def test_predict_walks_the_node_lists() -> None:
    data = make_tiny(50, seed=8)
    layout = FeatureLayout.of(data)
    forest = NodeForest.empty(2, 3)
    # tree 0: node 0 splits obfuscated feature 0 at 1; its right child, node 1, splits the country {0, 2} left
    forest.kind[0], forest.feature[0], forest.threshold[0] = [1, 2], [2, 6], [1, 0]
    forest.cat_left[0, 1] = 0b101
    forest.left[0], forest.right[0] = [-1, -2], [1, -3]  # node 0: leaf 0 | node 1; node 1: leaf 1 | leaf 2
    forest.leaf_value[0] = [1, 2, 3]
    forest.leaf_value[1] = [10, 20, 30]  # tree 1 never split: every row to leaf 0
    out = CPUBackend().predict(data, layout, 3, 50, forest)
    country = data.categorical[0, 3:]
    expected = np.where(data.codes[0, 3:] <= 1, 1, np.where((country == 0) | (country == 2), 2, 3)) + 10
    np.testing.assert_array_equal(out, expected.astype(np.float32))
