"""The GPU backends: every kernel against the CPU backend, bit for bit, on the same state.

Each test runs on Metal, on CUDA, and on CUDA's host layer over the fake driver with Metal running the kernels
(the ``gpu`` fixture, tests/unit/gbm/conftest.py).
"""

import gc

import numpy as np
import pytest
from numpy.typing import NDArray

from idun import gbm, rng
from idun.gbm.backends import CPUBackend, LeafSplit, LevelSplits, get_backend, get_many
from idun.gbm.backends import device as passes
from idun.gbm.backends.cpu import CPUMultiTarget
from idun.gbm.backends.device import DeviceBackend, DeviceMultiTarget
from idun.gbm.features import FeatureLayout
from idun.gbm.multi import get_multi
from idun.gbm.tree import FIELDS, NODE_FIELDS, Forest, NodeForest
from tests.unit.gbm.tiny import TinyData, make_tiny

LO = 7


def backend_of(gpu: str) -> DeviceBackend:
    backend = get_backend(gpu)
    assert isinstance(backend, DeviceBackend)
    assert backend.name == gpu
    return backend


def multi_of(gpu: str) -> DeviceMultiTarget:
    backend = get_multi(gpu)
    assert isinstance(backend, DeviceMultiTarget)
    assert backend.name == gpu
    return backend


def pair(gpu: str, n_rows: int = 5000, seed: int = 3) -> tuple[TinyData, FeatureLayout, CPUBackend, DeviceBackend]:
    data = make_tiny(n_rows + LO + 13, n_countries=11, seed=seed)
    layout = FeatureLayout.of(data)
    cpu, fast = CPUBackend(), backend_of(gpu)
    for backend in (cpu, fast):
        backend.start(data, layout, LO, data.target[LO : LO + n_rows])
    return data, layout, cpu, fast


def set_pred(backend: CPUBackend | DeviceBackend, values: NDArray[np.float32] | float) -> None:
    """Set a backend's running predictions (a test's starting state; fits never write them from the host)."""
    if isinstance(backend, CPUBackend):
        backend.pred[:] = values
    else:
        backend._device.write(backend._many._pred, np.float32, (backend.pred.size,))[:] = values


def set_node(fast: DeviceBackend | DeviceMultiTarget, node: NDArray[np.uint8]) -> None:
    fast._device.write(fast._node, np.uint8, (node.size,))[:] = node


def node_of(fast: DeviceBackend, n_rows: int) -> NDArray[np.uint8]:
    nodes: NDArray[np.uint8] = fast._device.read(fast._node, np.uint8, (n_rows,))
    return nodes


def test_rng_uniform_bits_matches_idun_rng(gpu: str) -> None:
    fast = backend_of(gpu)
    device = fast._device
    out = device.alloc(4 * 3000)
    key = rng.iteration_key(5, 77)
    device.encode("rng_uniform_bits", [out], passes.RngParams(key, 100, 3000), passes._groups(3000), passes.BLOCK)
    device.flush()
    np.testing.assert_array_equal(device.read(out, np.uint32, (3000,)), rng.uniform_bits(key, 3100)[100:])


def test_fma_probe_rounds_multiply_and_add_separately(gpu: str) -> None:
    fast = backend_of(gpu)
    device = fast._device
    generator = np.random.default_rng(0)
    a, b, c = (generator.standard_normal(1 << 16).astype(np.float32) for _ in range(3))
    buffers = [device.copy_in(v) for v in (a, b, c)] + [device.alloc(4 << 16)]
    device.encode("fma_probe", buffers, passes.CountParams(1 << 16), passes._groups(1 << 16), passes.BLOCK)
    device.flush()
    out = device.read(buffers[3], np.float32, (1 << 16,))
    np.testing.assert_array_equal(out, a * b + c)  # NumPy: two float32 roundings
    fused = (a.astype(np.float64) * b + c).astype(np.float32)  # one rounding (what an FMA gives)
    assert (fused != a * b + c).any()  # the test can tell the two apart


def test_compute_gradients_and_quantize_gradients_match(gpu: str) -> None:
    """Kernels: compute_gradients, quantize_gradients."""
    _, _, cpu, fast = pair(gpu)
    for backend in (cpu, fast):
        set_pred(backend, np.linspace(-0.5, 0.7, backend.pred.size, dtype=np.float32))
    assert fast.gradients() == cpu.gradients()
    np.testing.assert_array_equal(fast.g, cpu.g)
    bag = (np.arange(cpu.g.size) % 5 != 0).astype(np.uint8)
    for backend in (cpu, fast):
        backend.quantize(bag, rng.iteration_key(1, 2), np.float32(1.7), 2)
    fast._device.flush()
    np.testing.assert_array_equal(fast._device.read(fast._q, np.int8, (cpu.q.size,)), cpu.q)


def test_build_histograms_partition_rows_and_leaf_partials_match(gpu: str) -> None:
    """Kernels: build_histograms, partition_rows, leaf_partials, update_predictions."""
    data, layout, cpu, fast = pair(gpu)
    bag = (np.arange(cpu.g.size) % 3 != 0).astype(np.uint8)
    for backend in (cpu, fast):
        set_pred(backend, 0.2)
        backend.gradients()
        backend.quantize(bag, rng.iteration_key(4, 0), np.float32(3.0), 2)
        backend.start_tree()
    features = np.array([0, 2, 4, layout.n_features - 1], np.int32)
    offsets, cells = layout.offsets(features)
    np.testing.assert_array_equal(
        fast.histograms(features, offsets, cells, np.zeros(1, np.int16)),
        cpu.histograms(features, offsets, cells, np.zeros(1, np.int16)),
    )
    level = LevelSplits(
        kind=np.array([1], np.uint8), feature=np.array([0], np.int32), threshold=np.array([3], np.uint8),
        missing_left=np.array([0], np.uint8), cat_left=np.array([0], np.uint64),
    )  # fmt: skip
    for backend in (cpu, fast):
        backend.partition(level)
    level = LevelSplits(
        kind=np.array([2, 1], np.uint8), feature=np.array([layout.n_features - 1, 3], np.int32),
        threshold=np.array([0, 1], np.uint8), missing_left=np.array([0, 1], np.uint8),
        cat_left=np.array([0b10010110101, 0], np.uint64),
    )  # fmt: skip
    for backend in (cpu, fast):
        backend.partition(level)
    fast._device.flush()
    np.testing.assert_array_equal(node_of(fast, cpu.node.size), cpu.node)
    slots = np.array([1, -1, 0, 2], np.int16)
    np.testing.assert_array_equal(
        fast.histograms(features, offsets, cells, slots), cpu.histograms(features, offsets, cells, slots)
    )
    for left, right in zip(fast.leaf_sums(4), cpu.leaf_sums(4), strict=True):
        np.testing.assert_array_equal(left, right)
    values = np.array([0.25, -0.5, 1e-3, 7.0], np.float32)
    for backend in (cpu, fast):
        backend.update_predictions(values)
    np.testing.assert_array_equal(fast.pred, cpu.pred)
    assert data.n_rows > 0


def test_leaf_partials_span_many_threadgroups(gpu: str) -> None:
    _, _, cpu, fast = pair(gpu, n_rows=3 * passes.LEAF_ROWS_PER_GROUP + 17)
    for backend in (cpu, fast):
        set_pred(backend, np.linspace(-3, 3, backend.pred.size, dtype=np.float32))
    cpu.node[:] = np.arange(cpu.node.size) % 7
    set_node(fast, cpu.node)
    for left, right in zip(fast.leaf_sums(8), cpu.leaf_sums(8), strict=True):
        np.testing.assert_array_equal(left, right)


def test_predict_trees_matches(gpu: str) -> None:
    """Kernel: predict_trees."""
    data = make_tiny(4000, seed=9)
    params = gbm.Params(num_trees=6, max_depth=3, max_leaves=8, min_data_in_leaf=20, cat_l2=0.0, min_data_per_group=5)
    booster = gbm.train(params, data, rows=slice(0, 3000), backend="cpu")
    for rows in (slice(0, 4000), slice(3000, 3001), slice(5, 5)):
        np.testing.assert_array_equal(
            booster.predict(data, rows=rows, backend=gpu), booster.predict(data, rows=rows, backend="cpu")
        )
    fast = backend_of(gpu)
    assert fast.predict(data, FeatureLayout.of(data), 0, 10, Forest.empty(0, 2)).tolist() == [0.0] * 10


def test_a_short_fit_is_identical(gpu: str) -> None:
    data = make_tiny(3000, n_countries=9, seed=4)
    params = gbm.Params(
        num_trees=5,
        max_depth=4,
        max_leaves=10,
        min_data_in_leaf=10,
        lambda_l2=1.0,
        cat_l2=0.0,
        cat_smooth=1.0,
        min_data_per_group=5,
        feature_fraction=0.8,
        bagging_fraction=0.7,
        bagging_freq=1,
    )
    cpu = gbm.train(params, data, rows=slice(0, 2900), backend="cpu")
    fast = gbm.train(params, data, rows=slice(0, 2900), backend=gpu)
    for name in ("kind", "feature", "threshold", "missing_left", "cat_left", "leaf_value", "leaf_count"):
        np.testing.assert_array_equal(getattr(fast.forest, name), getattr(cpu.forest, name), err_msg=name)


def test_misuse_raises_and_close_releases(gpu: str) -> None:
    data = make_tiny(40)
    layout = FeatureLayout.of(data)
    fast = backend_of(gpu)
    with pytest.raises(ValueError, match="outside"):
        fast.start(data, layout, 30, data.target[:20])
    with pytest.raises(ValueError, match="outside"):
        fast.predict(data, layout, 0, 41, Forest.empty(1, 1))
    fast.start(data, layout, 0, data.target[:20])
    with pytest.raises(ValueError, match="bag"):
        fast.quantize(np.ones(19, np.uint8), 0, np.float32(1.0), 2)
    fast.close()
    del fast
    gc.collect()


@pytest.mark.parametrize(("n_slots", "half_bins"), [(1, 2), (2, 2), (4, 2), (5, 2), (20, 2), (3, 64), (17, 8)])
def test_histograms_obfuscated_matches(gpu: str, n_slots: int, half_bins: int) -> None:
    """Kernel histograms_obfuscated_one (a single model, K = 1): the bitmask path, the per-row path, wide q."""
    n_rows = 2 * passes.OBFUSCATED_ROWS_PER_GROUP + 77  # several row groups; LO = 7 is not 32-row aligned
    data, layout, cpu, fast = pair(gpu, n_rows=n_rows, seed=n_slots)
    generator = np.random.default_rng(n_slots)
    bag = (generator.random(n_rows) < 0.8).astype(np.uint8)
    width = 32 if n_slots > 2 else 4
    node = generator.integers(0, width, n_rows).astype(np.uint8)
    slots = np.full(width, -1, np.int16)
    slots[generator.permutation(width)[:n_slots]] = np.arange(n_slots, dtype=np.int16)
    for backend in (cpu, fast):
        set_pred(backend, generator.standard_normal(n_rows).astype(np.float32) if backend is cpu else cpu.pred)
        largest = backend.gradients()
        backend.quantize(bag, rng.iteration_key(n_slots, 1), np.float32(half_bins / largest), half_bins)
    cpu.node[:] = node
    set_node(fast, node)
    features = np.arange(layout.n_features, dtype=np.int32)  # 4 obfuscated (a partial lane group), named and country
    offsets, cells = layout.offsets(features)
    np.testing.assert_array_equal(
        fast.histograms(features, offsets, cells, slots), cpu.histograms(features, offsets, cells, slots)
    )
    assert data.n_rows > n_rows


@pytest.mark.parametrize(
    ("k", "n_slots", "half_bins"),
    [
        (1, 1, 2),
        (3, 2, 2),
        (4, 5, 2),
        (5, 20, 2),
        (11, 7, 8),
        (15, 3, 64),
        (12, 32, 2),
        (8, 6, 2),
        (16, 3, 2),
        (31, 5, 8),
        (47, 2, 2),
    ],
)
def test_histograms_multi_and_build_histograms_multi_match(gpu: str, k: int, n_slots: int, half_bins: int) -> None:
    """Kernels max_residuals, quantize_residuals, row_slots, histograms_multi (1..4 words, one and several slot and
    target chunks, wide q), build_histograms_multi, leaf_partials and update_predictions (one shared node array)
    against the CPU reference on K targets."""
    n_rows = 3 * 16384 // 2 + 77  # several row groups at every half_bins; LO = 7 is not 32-row aligned
    data = make_tiny(n_rows + LO + 13, n_countries=11, seed=k)
    layout = FeatureLayout.of(data)
    generator = np.random.default_rng(k)
    ys = np.stack([generator.random(n_rows) for _ in range(k)]).astype(np.float32)
    cpu, fast = CPUMultiTarget(), multi_of(gpu)
    bag = (generator.random(n_rows) < 0.8).astype(np.uint8)
    width = 32 if n_slots > 2 else 4
    node = generator.integers(0, width, n_rows).astype(np.uint8)
    slots = np.full(width, -1, np.int16)
    slots[generator.permutation(width)[:n_slots]] = np.arange(n_slots, dtype=np.int16)
    preds = generator.standard_normal((k, n_rows)).astype(np.float32)
    largest = {}
    for backend in (cpu, fast):
        backend.start(data, layout, LO, ys)
        if isinstance(backend, CPUMultiTarget):
            backend.pred[:] = preds
        else:
            backend._device.write(backend._pred, np.float32, preds.shape)[:] = preds
        largest[backend.name] = backend.gradients()
        backend.quantize(
            bag, rng.iteration_key(k, 1), (half_bins / largest[backend.name]).astype(np.float32), half_bins
        )
    np.testing.assert_array_equal(largest[gpu], largest["cpu"])  # max_residuals; quantize_residuals via the sums
    cpu.node[:] = node
    set_node(fast, node)
    features = np.arange(layout.n_features, dtype=np.int32)  # 4 obfuscated (a partial lane group), named and country
    offsets, cells = layout.offsets(features)
    expected = cpu.histograms(features, offsets, cells, slots)
    assert expected.shape == (n_slots, cells, k + 1)
    np.testing.assert_array_equal(fast.histograms(features, offsets, cells, slots), expected)
    sums, counts = cpu.leaf_sums(width)
    fast_sums, fast_counts = fast.leaf_sums(width)
    np.testing.assert_array_equal(fast_sums, sums)
    np.testing.assert_array_equal(fast_counts, counts)
    values = generator.standard_normal((k, width)).astype(np.float32)
    cpu.update_predictions(values)
    fast.update_predictions(values)
    for target in range(k):
        np.testing.assert_array_equal(fast.predictions(target), cpu.pred[target])
    fast.close()


@pytest.mark.parametrize("own_last_level", [False, True])
def test_a_short_multi_target_fit_is_identical(gpu: str, own_last_level: bool) -> None:
    """A train_multi fit on the CPU and a GPU; with own_last_level, kernel partition_own and per-target node arrays."""
    data = make_tiny(6000, n_countries=9, seed=8)
    generator = np.random.default_rng(8)
    y = np.asarray(data.target[:5000])
    ys = [np.clip(y + generator.normal(0, 0.2, y.size), 0, 1).astype(np.float32) for _ in range(6)]
    params = gbm.Params(
        num_trees=6, max_depth=4, max_leaves=11, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.7,
        bagging_freq=1, seed=3,
    )  # fmt: skip
    cpu = gbm.train_multi(params, data, rows=slice(0, 5000), backend="cpu", ys=ys, own_last_level=own_last_level)
    fast = gbm.train_multi(params, data, rows=slice(0, 5000), backend=gpu, ys=ys, own_last_level=own_last_level)
    assert fast[0].backend == gpu
    assert (cpu[0].forest.kind != 0).sum() >= 20
    for a, b in zip(cpu, fast, strict=True):
        for name in ("kind", "feature", "threshold", "missing_left", "cat_left", "leaf_value", "leaf_count"):
            np.testing.assert_array_equal(getattr(b.forest, name), getattr(a.forest, name), err_msg=name)


def test_multi_target_rejects_too_many_targets(gpu: str) -> None:
    data = make_tiny(100, seed=1)
    fast = multi_of(gpu)
    with pytest.raises(ValueError, match=r"1\.\.128 targets"):
        fast.start(data, FeatureLayout.of(data), 0, np.zeros((129, 50), np.float32))
    fast.close()


@pytest.mark.parametrize(
    "shape",
    [
        {"more_categorical": (3, 17)},  # three categorical columns
        {"n_countries": 0},  # none
        {"n_obfuscated": 0, "more_categorical": (4,)},  # no codes
        {"n_named": 0},  # no named bins
    ],
)
def test_any_number_of_columns_per_kind_fits_the_same_on_cpu_and_the_gpu(gpu: str, shape: dict[str, object]) -> None:
    data = make_tiny(1500, seed=9, **shape)  # type: ignore[arg-type]
    params = gbm.Params(num_trees=4, max_depth=3, max_leaves=8, min_data_in_leaf=10, seed=2)
    cpu, fast = (gbm.train(params, data, rows=slice(0, 1200), backend=b) for b in ("cpu", gpu))
    for name in (*FIELDS, "leaf_count"):
        np.testing.assert_array_equal(getattr(cpu.forest, name), getattr(fast.forest, name), err_msg=name)
    np.testing.assert_array_equal(
        cpu.predict(data, rows=slice(1200, 1500), backend="cpu"),
        fast.predict(data, rows=slice(1200, 1500), backend=gpu),
    )


def test_split_leaf_matches(gpu: str) -> None:
    """Kernel: split_leaf (leaf-wise growth), with build_histograms and leaf_partials over leaf ids."""
    _, layout, cpu, fast = pair(gpu)
    bag = (np.arange(cpu.g.size) % 5 != 0).astype(np.uint8)
    for backend in (cpu, fast):
        set_pred(backend, 0.3)
        backend.gradients()
        backend.quantize(bag, rng.iteration_key(2, 1), np.float32(2.0), 2)
        backend.start_tree()
    splits = [
        LeafSplit(leaf=0, right=1, kind=1, feature=0, threshold=3, missing_left=1, cat_left=0),
        LeafSplit(leaf=0, right=2, kind=1, feature=4, threshold=2, missing_left=1, cat_left=0),
        LeafSplit(leaf=1, right=3, kind=2, feature=layout.n_features - 1, threshold=0, missing_left=0,
                  cat_left=0b1001101),
        LeafSplit(leaf=3, right=200, kind=1, feature=1, threshold=4, missing_left=0, cat_left=0),
    ]  # fmt: skip
    for split in splits:
        for backend in (cpu, fast):
            backend.split_leaf(split)
    fast._device.flush()
    np.testing.assert_array_equal(node_of(fast, cpu.node.size), cpu.node)
    assert set(np.unique(cpu.node).tolist()) == {0, 1, 2, 3, 200}
    features = np.array([0, 2, 4, layout.n_features - 1], np.int32)
    offsets, cells = layout.offsets(features)
    slots = np.full(256, -1, np.int16)
    slots[[2, 200]] = [1, 0]
    np.testing.assert_array_equal(
        fast.histograms(features, offsets, cells, slots), cpu.histograms(features, offsets, cells, slots)
    )
    for left, right in zip(fast.leaf_sums(256), cpu.leaf_sums(256), strict=True):
        np.testing.assert_array_equal(left, right)


def test_predict_nodes_matches(gpu: str) -> None:
    """Kernel: predict_nodes."""
    data = make_tiny(4000, seed=9)
    params = gbm.Params(num_trees=6, growth="leafwise", max_depth=-1, max_leaves=19, min_data_in_leaf=20, cat_l2=0.0,
                        min_data_per_group=5)  # fmt: skip
    booster = gbm.train(params, data, rows=slice(0, 3000), backend="cpu")
    for rows in (slice(0, 4000), slice(3000, 3001), slice(5, 5)):
        np.testing.assert_array_equal(
            booster.predict(data, rows=rows, backend=gpu), booster.predict(data, rows=rows, backend="cpu")
        )
    fast = backend_of(gpu)
    assert fast.predict(data, FeatureLayout.of(data), 0, 10, NodeForest.empty(0, 2)).tolist() == [0.0] * 10


@pytest.mark.parametrize("depth", [-1, 5])
def test_a_short_leafwise_fit_is_identical(gpu: str, depth: int) -> None:
    data = make_tiny(6000, n_countries=9, more_categorical=(4,), seed=4)
    params = gbm.Params(
        num_trees=5, growth="leafwise", max_depth=depth, max_leaves=31, min_data_in_leaf=10, lambda_l2=1.0, cat_l2=0.0,
        cat_smooth=1.0, min_data_per_group=5, feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1,
    )  # fmt: skip
    cpu = gbm.train(params, data, rows=slice(0, 5900), backend="cpu")
    fast = gbm.train(params, data, rows=slice(0, 5900), backend=gpu)
    assert (cpu.forest.kind != 0).sum() > 100
    for name in (*NODE_FIELDS, "leaf_count"):
        np.testing.assert_array_equal(getattr(fast.forest, name), getattr(cpu.forest, name), err_msg=name)


def test_a_fit_may_start_from_given_predictions(gpu: str) -> None:
    data = make_tiny(500, seed=41)
    layout = FeatureLayout.of(data)
    start = np.linspace(-1.0, 1.0, 400).astype(np.float32)
    found = []
    for backend in (backend_of(gpu), CPUBackend()):
        backend.start(data, layout, 0, data.target[:400])
        backend.set_predictions(start)
        found.append((np.array(backend.pred), backend.gradients()))
    np.testing.assert_array_equal(found[0][0], start)
    assert found[0][1] == found[1][1]  # max |pred - y| from the given start, as the CPU's
    many = get_many(gpu, 2)
    many.start(data, layout, 0, np.stack([data.target[:400], -data.target[:400]]))
    many.set_predictions(np.stack([start, -start]))
    np.testing.assert_array_equal(many.views(1)[0], -start)
    with pytest.raises(ValueError, match="preds must be"):
        many.set_predictions(start[None])


@pytest.mark.parametrize("k", [1, 2])
def test_quantized_hessians_and_their_histograms_match(gpu: str, k: int) -> None:
    """Kernels: quantize_gradients and the histogram passes (one model's and K models') on the hessians' buffers."""
    n = 5000
    data = make_tiny(n + LO + 13, n_countries=11, seed=5)
    layout = FeatureLayout.of(data)
    ys = np.stack([np.asarray(data.target[LO : LO + n], np.float32)] * k)
    cpu, fast = get_many("cpu", k), get_many(gpu, k)
    bag = (np.arange(n) % 4 != 0).astype(np.uint8)
    key = rng.iteration_key(6, 1)
    for backend in (cpu, fast):
        backend.start(data, layout, LO, ys)
        backend.gradients()
        for model in range(k):
            backend.hessians(model)[:] = np.linspace(0.01 * (model + 1), 0.25, n, dtype=np.float32)
        backend.quantize(bag, key, np.full(k, 3.0, np.float32), 2)
        backend.quantize_hessians(key, np.full(k, 8.0, np.float32), 2)
        backend.start_tree()
    features = np.arange(layout.n_features, dtype=np.int32)  # named, obfuscated and categorical
    offsets, cells = layout.offsets(features)
    slots = np.zeros((k, 1), np.int16)
    found = {on: fast.histograms(features, offsets, cells, slots, hessians=on) for on in (False, True)}
    for on in (False, True):
        for mine, theirs in zip(found[on], cpu.histograms(features, offsets, cells, slots, hessians=on), strict=True):
            np.testing.assert_array_equal(mine, theirs)
    hess, grads = found[True][0], found[False][0]
    np.testing.assert_array_equal(hess[..., 1], grads[..., 1])  # the same rows counted
    assert (hess[..., 0] >= 0).all()
    assert not np.array_equal(hess[..., 0], grads[..., 0])  # the hessians' q, not the gradients'
