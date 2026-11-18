"""MetalRealMLP and CudaRealMLP against the float64 reference (within tolerance; identical bits run to run).

Every network test runs once per GPU backend (the ``network`` fixture), marked ``metal`` or ``cuda`` for it; the
CUDA ones run where an NVIDIA GPU is. Metal wraps the cache's arrays (page-aligned, no copy); CUDA uploads them.
"""

import ctypes
import dataclasses
import mmap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from idun.data import PackedDataset
from idun.nn import inputs
from idun.realmlp import cuda as realmlp_cuda
from idun.realmlp import metal as realmlp_metal
from idun.realmlp import reference as ref
from idun.realmlp import train as realmlp_train
from idun.realmlp.device import DeviceRealMLP as Device
from idun.realmlp.params import RealMLPParams
from idun_numerai.inputs import classes as numerai_classes

Network = Callable[..., Device]

SMALL = RealMLPParams(
    blocks=3,
    hidden=64,
    frequencies=4,
    embed=3,
    dropout=0.2,
    seed=6,
    learning_rate=0.01,
    n_classes=5,
    epochs=5,
    init_rows=3986,
)
OWN = dataclasses.replace(SMALL, own_last_block=True)
N_INPUTS, BATCH = 9, 20
KERNELS = (
    "rmlp_embed",
    "rmlp_forward_scaled",
    "rmlp_forward",
    "rmlp_out_forward",
    "rmlp_loss",
    "rmlp_out_backward",
    "rmlp_out_block_backward",
    "rmlp_out_block_backward_own",
    "rmlp_input_backward",
    "rmlp_embed_input_grad",
    "rmlp_weight_grad",
    "rmlp_block_sums",
    "rmlp_scale_grad",
    "rmlp_embed_backward",
    "rmlp_predict",
    "rmlp_mse",
    "rmlp_predict_value",
    "adamw",
    "rmlp_gather",
    "rmlp_class_labels",
    "rmlp_target_values",
    "rmlp_cat_embed",
    "rmlp_cat_backward",
)  # fmt: skip


def problem(n_targets: int | None = None, params: RealMLPParams = SMALL
            ) -> tuple[ref.Weights, np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Weights off the initial pattern, a batch, its labels ((BATCH,), or (BATCH, K) for K targets) and its rows."""
    generator = np.random.default_rng(5)
    x = generator.normal(size=(200, N_INPUTS))
    c = ref.preprocess(params, ref.fit_preprocess(x), x)
    weights = ref.init(params, c, n_targets or 1)
    for array in weights.arrays():  # off the initial pattern
        array[...] = array + generator.normal(0, 0.1, array.shape)
    y = generator.integers(0, 5, BATCH if n_targets is None else (BATCH, n_targets))
    return weights.astype(np.float32), c[:BATCH].astype(np.float32), y, np.arange(900, 900 + BATCH, dtype=np.int64)


def target_file(data: PackedDataset, path: Path) -> np.memmap:
    """Three targets of the cache's rows, (3, n_rows) float32, memory-mapped (page-aligned): y, 1 - y, y of row - 1."""
    y = np.asarray(data.target, np.float32)
    np.stack([y, 1 - y, np.roll(y, 1)]).tofile(path)
    return np.memmap(path, np.float32, "r", shape=(3, data.n_rows))


def class_array_of(targets: np.ndarray) -> np.ndarray:
    """The classes rint(4 · y) of a (K, n_rows) target array as a compact (n_rows, K) uint8 array Metal can wrap."""
    return numerai_classes(targets)


def relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30))


@pytest.fixture(
    params=[
        pytest.param(realmlp_metal.MetalRealMLP, marks=pytest.mark.metal, id="metal"),
        pytest.param(realmlp_cuda.CudaRealMLP, marks=pytest.mark.cuda, id="cuda"),
    ]
)
def network(request: pytest.FixtureRequest) -> Network:
    """The network class a test builds: once per GPU backend."""
    built: Network = request.param
    return built


@pytest.fixture
def net(network: Network) -> Iterator[Device]:
    model = network(SMALL, N_INPUTS, max_batch=BATCH)
    yield model
    model.close()


def test_structs_and_kernels(net: Device) -> None:
    assert ctypes.sizeof(realmlp_metal.RmlpParams) == 72
    assert ctypes.sizeof(realmlp_metal.RmlpFused) == 108
    assert ctypes.sizeof(realmlp_metal.GatherParams) == 40
    assert ctypes.sizeof(realmlp_metal.CatParams) == 28
    assert [realmlp_metal.codes_stride(n) for n in (1, 32, 33, 1017)] == [16, 16, 32, 512]
    assert set(KERNELS) == set(realmlp_metal.TABLE)
    assert (net.ldz, net.rows_pad) == (64, 64)
    weights, *_ = problem()
    net.set_weights(weights)
    assert all(np.array_equal(a, b) for a, b in zip(net.weights().arrays(), weights.arrays(), strict=True))


def test_forward_matches_the_reference(net: Device) -> None:
    weights, c, _, rows = problem()
    net.set_weights(weights)
    w64, c64 = weights.astype(np.float64), c.astype(np.float64)
    assert relative(net.forward(c, rows, step=None), ref.forward(w64, c64, None)[0]) < 1e-5
    masks = ref.dropout_masks(SMALL, step=4, t=0.3, rows=rows, dtype=np.float64)
    assert relative(net.forward(c, rows, step=4, t=0.3), ref.forward(w64, c64, masks)[0]) < 1e-5


def test_gradients_match_the_reference(net: Device) -> None:
    weights, c, y, rows = problem()
    net.set_weights(weights)
    loss = net.backward(c, y, rows, step=4, t=0.3)
    w64 = weights.astype(np.float64)
    logits, cache = ref.forward(w64, c.astype(np.float64), ref.dropout_masks(SMALL, 4, 0.3, rows, np.float64))
    want_loss, d_logits = ref.loss_and_logit_grad(SMALL, logits, y)
    assert loss == pytest.approx(want_loss, rel=1e-6)
    for got, expected in zip(net.gradients().arrays(), ref.backward(w64, cache, d_logits).arrays(), strict=True):
        assert got.shape == expected.shape
        assert relative(got, expected) < 1e-5


def test_weighted_gradients_match_the_reference_and_unit_weights_are_unweighted(net: Device) -> None:
    weights, c, y, rows = problem()
    w = np.random.default_rng(4).choice(np.array([0.0, 0.5, 1.0, 2.0], np.float32), len(rows))
    net.set_weights(weights)
    loss = net.backward(c, y, rows, step=4, t=0.3, weights=w)
    w64 = weights.astype(np.float64)
    logits, cache = ref.forward(w64, c.astype(np.float64), ref.dropout_masks(SMALL, 4, 0.3, rows, np.float64))
    want_loss, d_logits = ref.loss_and_logit_grad(SMALL, logits, y, w)
    assert loss == pytest.approx(want_loss, rel=1e-6)
    for got, expected in zip(net.gradients().arrays(), ref.backward(w64, cache, d_logits).arrays(), strict=True):
        assert relative(got, expected) < 1e-5
    net.backward(c, y, rows, step=4, t=0.3)
    plain = net.gradients().arrays()
    net.backward(c, y, rows, step=4, t=0.3, weights=np.ones(len(rows), np.float32))
    for one, other in zip(net.gradients().arrays(), plain, strict=True):
        np.testing.assert_array_equal(one, other)  # unit weights: the unweighted step, bit for bit
    assert net.backward(c, y, rows, step=4, t=0.3, weights=np.zeros(len(rows), np.float32)) == 0.0
    assert not any(a.any() for a in net.gradients().arrays())  # a batch of weight 0: no gradient
    with pytest.raises(ValueError, match="one weight per row"):
        net.backward(c, y, rows, step=4, t=0.3, weights=w[:-1])


@pytest.mark.parametrize(("act", "sizes"), [("mish", ()), ("relu", ()), ("selu", (128, 64, 192)), ("mish", (64,))])
def test_each_activation_matches_the_reference(network: Network, act: str, sizes: tuple[int, ...]) -> None:
    params = (
        dataclasses.replace(SMALL, act=act, blocks=3, hidden=256, hidden_sizes=sizes)
        if sizes
        else dataclasses.replace(SMALL, act=act)
    )
    model = network(params, N_INPUTS, max_batch=BATCH)
    try:
        weights, c, y, rows = problem(params=params)
        model.set_weights(weights)
        w64, c64 = weights.astype(np.float64), c.astype(np.float64)
        assert relative(model.forward(c, rows, step=None), ref.forward(w64, c64, None, act)[0]) < 1e-5
        loss = model.backward(c, y, rows, step=4, t=0.3)
        logits, cache = ref.forward(w64, c64, ref.dropout_masks(params, 4, 0.3, rows, np.float64), act)
        want_loss, d_logits = ref.loss_and_logit_grad(params, logits, y)
        assert loss == pytest.approx(want_loss, rel=1e-6)
        for got, expected in zip(model.gradients().arrays(), ref.backward(w64, cache, d_logits, act).arrays(),
                                 strict=True):  # fmt: skip
            assert relative(got, expected) < 1e-5
    finally:
        model.close()


def test_unequal_widths_and_an_own_last_block_follow_the_reference(network: Network) -> None:
    params = dataclasses.replace(OWN, blocks=3, hidden=256, hidden_sizes=(128, 64))
    model = network(params, N_INPUTS, max_batch=BATCH, n_targets=3)
    try:
        weights, c, y, rows = problem(3, params)
        model.set_weights(weights)
        w64, c64 = weights.astype(np.float64), c.astype(np.float64)
        logits, cache = ref.forward(w64, c64, ref.dropout_masks(params, 4, 0.3, rows, np.float64))
        loss = model.backward(c, y, rows, step=4, t=0.3)
        want_loss, d_logits = ref.loss_and_logit_grad(params, logits, y)
        assert loss == pytest.approx(want_loss, rel=1e-6)
        for got, expected in zip(model.gradients().arrays(), ref.backward(w64, cache, d_logits).arrays(), strict=True):
            assert relative(got, expected) < 1e-5
    finally:
        model.close()


def test_training_steps_are_reference_adam(net: Device) -> None:
    weights, c, y, rows = problem()
    net.set_weights(weights)
    w64 = weights.astype(np.float64)
    state = ref.adam_state(w64)
    for step, t in enumerate((0.1, 0.4, 0.8)):  # lr > 0 at each, dropout and decay on and off the flat part
        net.train_step(c, y, rows, step, t)
        logits, cache = ref.forward(w64, c.astype(np.float64), ref.dropout_masks(SMALL, step, t, rows, np.float64))
        ref.adam_step(SMALL, w64, ref.backward(w64, cache, ref.loss_and_logit_grad(SMALL, logits, y)[1]), state, t)
    for got, expected in zip(net.weights().arrays(), w64.arrays(), strict=True):
        assert relative(got, expected) < 1e-4
    want = ref.predict(ref.forward(w64, c.astype(np.float64), None)[0])
    assert relative(net.predict(np.concatenate([c] * 4)), np.concatenate([want] * 4)) < 1e-5


# K heads of 5 classes (K = 100 is 500 outputs), on a shared trunk or each with its own last block (K · 64 units)
@pytest.mark.parametrize(("n_targets", "params"), [(3, SMALL), (100, SMALL), (3, OWN), (100, OWN)])
def test_k_targets_match_the_reference(network: Network, n_targets: int, params: RealMLPParams) -> None:
    weights, c, y, rows = problem(n_targets, params)
    net = network(params, N_INPUTS, max_batch=BATCH, n_targets=n_targets)
    net.set_weights(weights)
    w64, c64 = weights.astype(np.float64), c.astype(np.float64)
    logits, cache = ref.forward(w64, c64, ref.dropout_masks(SMALL, 4, 0.3, rows, np.float64))
    assert relative(net.forward(c, rows, step=4, t=0.3), logits) < 1e-5
    loss = net.backward(c, y, rows, step=4, t=0.3)
    want_loss, d_logits = ref.loss_and_logit_grad(SMALL, logits, y)
    assert loss == pytest.approx(want_loss, rel=1e-6)
    for got, expected in zip(net.gradients().arrays(), ref.backward(w64, cache, d_logits).arrays(), strict=True):
        assert got.shape == expected.shape
        assert relative(got, expected) < 1e-5
    state = ref.adam_state(w64)
    for step, t in enumerate((0.1, 0.4, 0.8)):
        net.train_step(c, y, rows, step, t)
        logits, cache = ref.forward(w64, c64, ref.dropout_masks(SMALL, step, t, rows, np.float64))
        ref.adam_step(SMALL, w64, ref.backward(w64, cache, ref.loss_and_logit_grad(SMALL, logits, y)[1]), state, t)
    for got, expected in zip(net.weights().arrays(), w64.arrays(), strict=True):
        assert relative(got, expected) < 1e-4
    pred = net.predict(np.concatenate([c] * 4))
    assert pred.shape == (4 * BATCH, n_targets)
    assert relative(pred, np.concatenate([ref.predict(ref.forward(w64, c64, None)[0], n_targets)] * 4)) < 1e-5
    net.close()


def test_one_target_as_an_array_is_the_single_target_network(network: Network) -> None:
    weights, c, y, rows = problem()
    runs, preds = [], []
    # one target as an array, also with its own last block (the own-block kernels at K = 1)
    for n_targets, labels, params in ((None, y, SMALL), (1, y[:, None], SMALL), (1, y[:, None], OWN)):
        net = network(params, N_INPUTS, max_batch=BATCH, n_targets=n_targets)
        net.set_weights(weights)
        logits = net.forward(c, rows, step=4, t=0.3)
        loss = net.backward(c, labels, rows, step=4, t=0.3)
        grads = [g.tobytes() for g in net.gradients().arrays()]
        losses = [net.train_step(c, labels, rows, step, t) for step, t in enumerate((0.1, 0.4, 0.8))]
        runs.append((logits.tobytes(), loss, grads, losses, [a.tobytes() for a in net.weights().arrays()]))
        preds.append(net.predict(c))
        net.close()
    assert runs[0] == runs[1] == runs[2]
    assert (preds[0].shape, preds[1].shape) == ((BATCH,), (BATCH, 1))
    np.testing.assert_array_equal(preds[0], preds[1][:, 0])
    np.testing.assert_array_equal(preds[1], preds[2])


@pytest.mark.parametrize(("n_targets", "params"), [(None, SMALL), (3, SMALL), (3, OWN)])
def test_two_runs_give_identical_bits(network: Network, n_targets: int | None, params: RealMLPParams) -> None:
    weights, c, y, rows = problem(n_targets, params)
    runs = []
    for _ in range(2):
        model = network(params, N_INPUTS, max_batch=BATCH, n_targets=n_targets)
        model.set_weights(weights)
        losses = [model.train_step(c, y, rows, step, step / 3) for step in range(3)]
        runs.append((losses, [a.tobytes() for a in model.weights().arrays()]))
        model.close()
    assert runs[0] == runs[1]


def test_bad_arguments_raise(network: Network, net: Device) -> None:
    _, c, _, rows = problem()
    with pytest.raises(ValueError, match="rows of 9 inputs"):
        net.forward(c[:, :3], rows, step=None)
    with pytest.raises(ValueError, match="multiple of 64"):
        network(RealMLPParams(hidden=100, n_classes=5, epochs=5, init_rows=3986), N_INPUTS, 1)
    with pytest.raises(ValueError, match="at most 8 classes"):
        network(RealMLPParams(n_classes=9, epochs=5, init_rows=3986), N_INPUTS, 1)
    with pytest.raises(ValueError, match="max_batch"):
        network(SMALL, N_INPUTS, 0)
    with pytest.raises(ValueError, match="n_targets"):
        network(SMALL, N_INPUTS, 1, n_targets=0)
    weights, c, y, rows = problem(3)
    with pytest.raises(ValueError, match=r"labels of shape \(20,\)"):
        net.backward(c, y, rows, step=4, t=0.3)  # the single-target network takes (B,) labels
    with pytest.raises(ValueError, match="this network's are"):
        net.set_weights(weights)  # three heads into one
    three = network(SMALL, N_INPUTS, max_batch=BATCH, n_targets=3)
    with pytest.raises(ValueError, match=r"labels of shape \(20, 3\)"):
        three.train_step(c, y[:, 0], rows, step=4, t=0.3)
    three.close()
    with pytest.raises(ValueError, match="own_last_block needs a network of K targets"):
        network(OWN, N_INPUTS, 1)
    own = network(OWN, N_INPUTS, max_batch=BATCH, n_targets=3)
    with pytest.raises(ValueError, match="this network's are"):
        own.set_weights(weights)  # a shared last block into an own one
    own.close()


def test_batches_built_on_the_gpu_equal_the_host_path(network: Network, nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = RealMLPParams(blocks=1, hidden=64, frequencies=4, embed=3, init_rows=80, n_classes=5, epochs=5)
    pre = ref.fit_preprocess(realmlp_train.raw_inputs(data, np.arange(80)))
    model = network(params, inputs.n_inputs(data), max_batch=64)
    with pytest.raises(RuntimeError, match="bind"):
        model.gathered(np.arange(3))
    model.bind(data, pre, numerai_classes(data.target))
    rows = np.array([5, 0, 3, data.n_rows - 1, 2, 2])  # any order, odd and even rows, repeats
    c, y = model.gathered(rows)
    np.testing.assert_allclose(c, ref.preprocess(params, pre, realmlp_train.raw_inputs(data, rows)), atol=1e-6)
    np.testing.assert_array_equal(y, np.rint(4 * np.asarray(data.target)[rows]))
    with pytest.raises(ValueError, match="row indices of the bound data"):
        model.gathered(np.array([data.n_rows]))
    every = [model.gathered(np.arange(lo, min(lo + 64, data.n_rows)))[0] for lo in range(0, data.n_rows, 64)]
    np.testing.assert_allclose(np.concatenate(every), ref.preprocess(params, pre, realmlp_train.raw_inputs(
        data, np.arange(data.n_rows))), atol=1e-6)  # fmt: skip
    with pytest.raises(ValueError, match="this network takes"):
        network(params, inputs.n_inputs(data) + 1, 8).bind(data, pre, numerai_classes(data.target))
    model.close()


def test_shared_codes_give_the_bound_copys_batches(network: Network, nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    codes = realmlp_metal.shared_codes(data)
    want = np.zeros_like(codes)
    inputs.codes_row_major(data, want)
    np.testing.assert_array_equal(codes, want)
    assert codes.__array_interface__["data"][0] % mmap.PAGESIZE == 0  # the OS page: 16 KiB on Apple silicon
    params = RealMLPParams(blocks=1, hidden=64, frequencies=4, embed=3, init_rows=80, n_classes=5, epochs=5)
    pre = ref.fit_preprocess(realmlp_train.raw_inputs(data, np.arange(80)))
    rows = np.array([5, 0, 3, data.n_rows - 1, 2, 2])
    nets = [network(params, inputs.n_inputs(data), max_batch=8) for _ in range(3)]
    labels = numerai_classes(data.target)
    nets[0].bind(data, pre, labels)
    nets[1].bind(data, pre, labels, codes=codes)
    nets[2].bind(data, pre, labels, codes=codes)  # two networks wrap the one copy
    batches = [net.gathered(rows)[0] for net in nets]
    np.testing.assert_array_equal(batches[1], batches[0])
    np.testing.assert_array_equal(batches[2], batches[0])
    with pytest.raises(ValueError, match="shared_codes"):
        nets[0].bind(data, pre, labels, codes=codes[:-1])
    for net in nets:
        net.close()


def test_a_network_bound_without_classes_only_predicts(network: Network, nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = RealMLPParams(blocks=1, hidden=64, frequencies=4, embed=3, init_rows=80, n_classes=5, epochs=5)
    pre = ref.fit_preprocess(realmlp_train.raw_inputs(data, np.arange(80)))
    unlabelled = network(params, inputs.n_inputs(data), max_batch=64, n_targets=3)
    unlabelled.bind(data, pre)  # three targets bound without their classes: it predicts, but cannot train
    assert unlabelled.predict_rows(0, 70).shape == (70, 3)
    with pytest.raises(RuntimeError, match="targets"):
        unlabelled.train_step_rows(np.arange(4), 0, 0.1)
    assert unlabelled.predict_rows(0, 5).shape == (5, 3)  # raised before any command buffer: still usable
    unlabelled.close()


@pytest.mark.parametrize(("n_targets", "own"), [(None, False), (3, False), (3, True)])
def test_pipelined_steps_equal_one_step_at_a_time(network: Network, nn_cache: Path, tmp_path: Path,
                                                  n_targets: int | None, own: bool) -> None:  # fmt: skip
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = None if n_targets is None else target_file(data, tmp_path / "t.f32")
    params = RealMLPParams(
        blocks=2,
        hidden=64,
        frequencies=4,
        embed=3,
        dropout=0.2,
        init_rows=80,
        own_last_block=own,
        n_classes=5,
        epochs=5,
    )
    x = realmlp_train.raw_inputs(data, np.arange(80))
    pre = ref.fit_preprocess(x)
    weights = ref.init(params, ref.preprocess(params, pre, x), n_targets or 1).astype(np.float32)
    generator = np.random.default_rng(3)
    plan = [(generator.choice(data.n_rows, size, replace=False), k, k / 5) for k, size in enumerate((16, 9, 16, 16, 4))]
    runs = []
    for pipelined in (False, True):
        model = network(params, inputs.n_inputs(data), max_batch=16, n_targets=n_targets)
        model.set_weights(weights)
        model.bind(data, pre, numerai_classes(data.target) if targets is None else class_array_of(targets))
        if pipelined:
            losses = model.train_steps_rows(iter(plan))
        else:
            losses = [model.train_step_rows(rows, k, t) for rows, k, t in plan]
        runs.append((losses, [a.tobytes() for a in model.weights().arrays()]))
        assert model.train_steps_rows([]) == []
        model.close()
    assert runs[0] == runs[1]
    assert len(runs[0][0]) == len(plan)


def test_weighted_gpu_batches_pipeline_and_unit_weights_are_unweighted(network: Network, nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = RealMLPParams(
        blocks=2, hidden=64, frequencies=4, embed=3, dropout=0.2, init_rows=80, n_classes=5, epochs=5
    )
    x = realmlp_train.raw_inputs(data, np.arange(80))
    pre = ref.fit_preprocess(x)
    weights = ref.init(params, ref.preprocess(params, pre, x), 1).astype(np.float32)
    generator = np.random.default_rng(3)
    plan = [(generator.choice(data.n_rows, size, replace=False), k, k / 5) for k, size in enumerate((16, 9, 16, 4))]
    w = [generator.choice(np.array([0.0, 0.5, 1.0, 2.0], np.float32), rows.size) for rows, _, _ in plan]

    def run(steps: list[tuple[Any, ...]], pipelined: bool = True) -> tuple[list[float], list[bytes]]:
        model = network(params, inputs.n_inputs(data), max_batch=16)
        model.set_weights(weights)
        model.bind(data, pre, numerai_classes(data.target))
        losses = model.train_steps_rows(iter(steps)) if pipelined else [model.train_step_rows(*s) for s in steps]
        found = (losses, [a.tobytes() for a in model.weights().arrays()])
        model.close()
        return found

    weighted = [(*step, each) for step, each in zip(plan, w, strict=True)]
    assert run(weighted) == run(weighted, pipelined=False)
    assert run([(*step, np.ones(step[0].size, np.float32)) for step in plan]) == run(plan)  # unit weights
    assert run(weighted) != run(plan)


def test_k_targets_gathered_on_the_gpu_from_classes(network: Network, nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = target_file(data, tmp_path / "t.f32")
    wide = inputs.class_array(data.n_rows, 5)  # 5 columns, the network reads the first 3: a row stride of 5
    wide[...] = 4
    wide[:, :3] = class_array_of(targets)
    params = RealMLPParams(blocks=1, hidden=64, frequencies=4, embed=3, init_rows=80, n_classes=5, epochs=5)
    pre = ref.fit_preprocess(realmlp_train.raw_inputs(data, np.arange(80)))
    rows = np.array([5, 0, 3, data.n_rows - 1, 2, 2])  # any order, odd and even rows, repeats
    for classes in (class_array_of(targets), wide[:, :3]):
        model = network(params, inputs.n_inputs(data), max_batch=64, n_targets=3)
        model.bind(data, pre, classes=classes)
        c, y = model.gathered(rows)
        np.testing.assert_array_equal(y, np.rint(4 * np.asarray(targets)[:, rows]).T)
        np.testing.assert_allclose(c, ref.preprocess(params, pre, realmlp_train.raw_inputs(data, rows)), atol=1e-6)
        every = np.concatenate([model.gathered(np.arange(lo, min(lo + 64, data.n_rows)))[1]
                                for lo in range(0, data.n_rows, 64)])  # fmt: skip
        np.testing.assert_array_equal(every, inputs.class_labels(classes, np.arange(data.n_rows), 5))
        model.close()
    model = network(params, inputs.n_inputs(data), max_batch=64, n_targets=3)
    classes = class_array_of(targets)
    with pytest.raises(ValueError, match="classes has 2 columns"):
        model.bind(data, pre, classes=classes[:, :2])
    bads: list[np.ndarray] = [classes[:-1], classes.astype(np.int16), np.ascontiguousarray(classes.T), classes[:, 0]]
    for bad in bads:
        with pytest.raises(ValueError, match=r"classes must be an \(277, K\) uint8"):
            model.bind(data, pre, classes=bad)
    unaligned = np.zeros(3 * data.n_rows + 1, np.uint8)[1:].reshape(-1, 3)
    if network is realmlp_metal.MetalRealMLP:
        with pytest.raises(ValueError, match="page-aligned"):
            model.bind(data, pre, classes=unaligned)
    else:  # uploaded through a transient host copy
        model.bind(data, pre, classes=unaligned)
        assert model.device.placements["target"] == "repacked"
    model.close()


def test_regression_gradients_steps_and_predictions_follow_the_reference(network: Network) -> None:
    params = dataclasses.replace(SMALL, n_classes=1, objective="mse")
    generator = np.random.default_rng(8)
    x = generator.normal(size=(200, N_INPUTS))
    c = ref.preprocess(params, ref.fit_preprocess(x), x)
    weights = ref.init(params, c, 1)
    for array in weights.arrays():
        array[...] = array + generator.normal(0, 0.1, array.shape)
    weights = weights.astype(np.float32)
    c, y = c[:BATCH].astype(np.float32), generator.normal(size=BATCH).astype(np.float32)
    rows = np.arange(300, 300 + BATCH, dtype=np.int64)
    net = network(params, N_INPUTS, BATCH)
    try:
        net.set_weights(weights)
        w64 = weights.astype(np.float64)
        loss = net.backward(c, y, rows, step=2, t=0.3)
        logits, cache = ref.forward(w64, c.astype(np.float64), ref.dropout_masks(params, 2, 0.3, rows, np.float64))
        want_loss, d_logits = ref.loss_and_logit_grad(params, logits, y)
        assert loss == pytest.approx(want_loss, rel=1e-5)
        for got, expected in zip(net.gradients().arrays(), ref.backward(w64, cache, d_logits).arrays(), strict=True):
            assert relative(got, expected) < 1e-4
        net.set_weights(weights)
        state = ref.adam_state(w64)
        for step, t in enumerate((0.1, 0.4, 0.8)):
            net.train_step(c, y, rows, step, t)
            logits, cache = ref.forward(w64, c.astype(np.float64), ref.dropout_masks(params, step, t, rows, np.float64))
            ref.adam_step(
                params, w64, ref.backward(w64, cache, ref.loss_and_logit_grad(params, logits, y)[1]), state, t
            )
        for got, expected in zip(net.weights().arrays(), w64.arrays(), strict=True):
            assert relative(got, expected) < 1e-2  # Adam divides by sqrt(v): a looser bound for the weights
        want = ref.predict(ref.forward(w64, c.astype(np.float64), None)[0], None, "mse")
        np.testing.assert_allclose(net.predict(c), want, atol=1e-4)
        with pytest.raises(ValueError, match="takes its labels as targets"):
            net.bind(None, ref.fit_preprocess(x), np.zeros((4, 1), np.uint8))  # type: ignore[arg-type]
    finally:
        net.close()


def test_targets_gathered_on_the_gpu_are_the_arrays_own(network: Network, nn_cache: Path) -> None:
    """A regression network's GPU batches read a wrapped float32 target array: exact values, no copy."""
    data = PackedDataset.open(nn_cache, binning="r1")
    params = RealMLPParams(
        blocks=1, hidden=64, frequencies=4, embed=3, init_rows=80, n_classes=1, objective="mse", epochs=5
    )
    pre = ref.fit_preprocess(realmlp_train.raw_inputs(data, np.arange(80)))
    generator = np.random.default_rng(4)
    wide = inputs.target_array(data.n_rows, 5)  # 5 columns, the network reads the first 3: a row stride of 5
    wide[...] = generator.normal(size=wide.shape).astype(np.float32)
    rows = np.array([5, 0, 3, data.n_rows - 1, 2, 2])  # any order, odd and even rows, repeats
    compact = inputs.target_array(data.n_rows, 3)
    compact[...] = wide[:, :3]
    for array in (compact, wide[:, :3]):
        model = network(params, inputs.n_inputs(data), max_batch=64, n_targets=3)
        model.bind(data, pre, targets=array)
        c, y = model.gathered(rows)
        assert y.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(y, array[rows])
        np.testing.assert_allclose(c, ref.preprocess(params, pre, realmlp_train.raw_inputs(data, rows)), atol=1e-6)
        array[5, 1] = 123.5  # on Metal wrapped, not copied: the GPU reads the change; CUDA's upload does not
        seen = np.float32(123.5) if network is realmlp_metal.MetalRealMLP else y[0, 1]
        assert model.gathered(np.array([5]))[1][0, 1] == seen
        model.close()
    single = network(params, inputs.n_inputs(data), max_batch=64)
    single.bind(data, pre, targets=wide[:, :1])
    np.testing.assert_array_equal(single.gathered(rows)[1], wide[rows, 0])  # the single-target network: (n,)
    single.close()
    model = network(params, inputs.n_inputs(data), max_batch=64, n_targets=3)
    with pytest.raises(ValueError, match="targets has 2 columns"):
        model.bind(data, pre, targets=wide[:, :2])
    bads: list[np.ndarray] = [wide[:-1, :3], wide[:, :3].astype(np.float64), wide[:, 0]]
    for bad in bads:
        with pytest.raises(ValueError, match=r"targets must be an \(277, K\) float32"):
            model.bind(data, pre, targets=bad)
    with pytest.raises(ValueError, match="takes its labels as targets"):
        model.bind(data, pre, numerai_classes(np.stack([data.target] * 3)))
    unaligned = np.zeros(3 * data.n_rows + 1, np.float32)[1:].reshape(-1, 3)
    if network is realmlp_metal.MetalRealMLP:
        with pytest.raises(ValueError, match="page-aligned"):
            model.bind(data, pre, targets=unaligned)
    else:  # uploaded through a transient host copy
        model.bind(data, pre, targets=unaligned)
        assert model.device.placements["target"] == "repacked"
    model.close()
    classifier = network(dataclasses.replace(params, n_classes=5, objective="cross_entropy"),
                                            inputs.n_inputs(data), max_batch=64)  # fmt: skip
    with pytest.raises(ValueError, match="takes its labels as classes"):
        classifier.bind(data, pre, targets=wide[:, :1])
    classifier.close()


def test_categorical_embeddings_follow_the_reference(network: Network) -> None:
    generator = np.random.default_rng(8)
    params, categories = dataclasses.replace(SMALL, embedding_size=5), (6, 3)
    numeric = ref.preprocess(SMALL, ref.fit_preprocess(generator.normal(size=(200, N_INPUTS))),
                             generator.normal(size=(BATCH, N_INPUTS)))  # fmt: skip
    codes = np.column_stack([generator.integers(0, m, BATCH) for m in categories])
    c = np.concatenate([numeric, codes], axis=1).astype(np.float32)
    weights = ref.init(params, c.astype(np.float64), categories=categories)
    for array in weights.arrays():
        array[...] = array + generator.normal(0, 0.1, array.shape)
    weights = weights.astype(np.float32)
    y, rows = generator.integers(0, 5, BATCH), np.arange(300, 300 + BATCH, dtype=np.int64)
    net = network(params, N_INPUTS + 2, max_batch=BATCH, categories=categories)
    try:
        net.set_weights(weights)
        assert all(np.array_equal(a, b) for a, b in zip(net.weights().arrays(), weights.arrays(), strict=True))
        loss = net.backward(c, y, rows, step=4, t=0.3)
        w64 = weights.astype(np.float64)
        logits, cache = ref.forward(w64, c.astype(np.float64), ref.dropout_masks(params, 4, 0.3, rows, np.float64))
        want_loss, d_logits = ref.loss_and_logit_grad(params, logits, y)
        assert loss == pytest.approx(want_loss, rel=1e-6)
        for got, expected in zip(net.gradients().arrays(), ref.backward(w64, cache, d_logits).arrays(), strict=True):
            assert got.shape == expected.shape
            assert relative(got, expected) < 1e-5
        want_pred = ref.predict(ref.forward(w64, c.astype(np.float64), None)[0])
        assert relative(net.predict(c), want_pred) < 1e-5
        bad = c.copy()
        bad[0, -1] = 3.0  # input 1 has 3 levels
        with pytest.raises(ValueError, match="level codes"):
            net.backward(bad, y, rows, step=4, t=0.3)
        with pytest.raises(ValueError, match="categories"):
            net.set_weights(ref.init(SMALL, c.astype(np.float64)[:, :-2]).astype(np.float32))
    finally:
        net.close()
