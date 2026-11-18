"""MetalTabM and CudaTabM against the float64 reference (within tolerance; identical bits run to run).

Every network test runs once per GPU backend (the ``network`` fixture), marked ``metal`` or ``cuda`` for it;
the CUDA ones run where an NVIDIA GPU is.
"""

import ctypes
import dataclasses
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from idun.tabm import TabMParams, cuda, hl_gauss, metal, reference
from idun.tabm import embeddings as emb
from idun.tabm.device import DeviceTabM as Device

Network = Callable[..., Device]

TINY = TabMParams(
    members=3,
    blocks=3,
    width=40,
    dropout=0.25,
    seed=11,
    learning_rate=1e-2,
    weight_decay=0.1,
    n_classes=5,
    batch_size=4096,
    epochs=10,
)
N_INPUTS, BATCH = 37, 10
KERNELS = (
    "gemm_forward",
    "gemm_backward",
    "gemm_tn_members",
    "member_weight_grad",
    "member_adapter_grad",
    "head_forward",
    "softmax_ce",
    "head_backward",
    "head_block_backward",
    "member_sums",
    "sum_partials",
    "predict_class",
    "mse_loss",
    "predict_mean",
    "hl_gauss_ce",
    "predict_centre",
    "adamw",
)


def problem(seed: int = 3) -> tuple[reference.Weights, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    weights = reference.init(TINY, N_INPUTS)
    for array in weights.arrays():  # off the initial pattern, so every gradient path is exercised
        array[...] = array + generator.normal(0, 0.2, array.shape).astype(np.float32)
    x = generator.normal(size=(BATCH, N_INPUTS)).astype(np.float32)
    y = generator.integers(0, 5, BATCH)
    rows = np.arange(500, 500 + BATCH, dtype=np.int64)
    return weights, x, y, rows


def relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30))


@pytest.fixture(
    params=[
        pytest.param(metal.MetalTabM, marks=pytest.mark.metal, id="metal"),
        pytest.param(cuda.CudaTabM, marks=pytest.mark.cuda, id="cuda"),
    ]
)
def network(request: pytest.FixtureRequest) -> Network:
    """The network class a test builds: once per GPU backend."""
    built: Network = request.param
    return built


@pytest.fixture
def net(network: Network) -> Iterator[Device]:
    model = network(TINY, N_INPUTS, max_batch=BATCH)
    yield model
    model.close()


def test_struct_sizes_match_the_kernels() -> None:
    assert ctypes.sizeof(metal.NetParams) == 68
    assert ctypes.sizeof(metal.FusedParams) == 104
    assert ctypes.sizeof(metal.MemberGemmParams) == 48
    assert ctypes.sizeof(metal.EmbParams) == 40
    assert set(metal.EMBEDDING_KERNELS) == {"emb_forward", "emb_member_sum", "emb_weight_grad"}
    assert set(KERNELS) == set(metal.TABLE) - set(metal.ENCODER_KERNELS) - set(metal.EMBEDDING_KERNELS)


def test_padding_keeps_tiles_whole(net: Device) -> None:
    assert (net.ldw, net.ldd, net.rows_pad, net.m) == (64, 64, 64, 192)
    weights, *_ = problem()
    net.set_weights(weights)
    assert all(np.array_equal(a, b) for a, b in zip(net.weights().arrays(), weights.arrays(), strict=True))


def test_forward_matches_the_reference_with_and_without_dropout(net: Device) -> None:
    weights, x, _, rows = problem()
    net.set_weights(weights)
    w64, x64 = weights.astype(np.float64), x.astype(np.float64)
    masks = reference.dropout_masks(TINY, step=4, rows=rows, dtype=np.float64)
    assert relative(net.forward(x, rows, step=4), reference.forward(w64, x64, masks)[0]) < 1e-5
    assert relative(net.forward(x, rows, step=None), reference.forward(w64, x64, None)[0]) < 1e-5


def test_gradients_match_the_reference(net: Device) -> None:
    weights, x, y, rows = problem()
    net.set_weights(weights)
    loss = net.backward(x, y, rows, step=7)
    w64 = weights.astype(np.float64)
    logits, cache = reference.forward(w64, x.astype(np.float64), reference.dropout_masks(TINY, 7, rows, np.float64))
    want_loss, d_logits = reference.loss_and_logit_grad(logits, y)
    assert loss == pytest.approx(want_loss, rel=1e-6)
    want = reference.backward(w64, cache, d_logits)
    for got, expected in zip(net.gradients().arrays(), want.arrays(), strict=True):
        assert got.shape == expected.shape
        assert relative(got, expected) < 1e-5  # observed: 1e-7 to 4e-7


def test_weighted_gradients_match_the_reference_and_unit_weights_are_unweighted(net: Device) -> None:
    weights, x, y, rows = problem()
    w = np.random.default_rng(4).choice(np.array([0.0, 0.5, 1.0, 2.0], np.float32), len(rows))
    net.set_weights(weights)
    loss = net.backward(x, y, rows, step=7, weights=w)
    w64 = weights.astype(np.float64)
    logits, cache = reference.forward(w64, x.astype(np.float64), reference.dropout_masks(TINY, 7, rows, np.float64))
    want_loss, d_logits = reference.loss_and_logit_grad(logits, y, weights=w)
    assert loss == pytest.approx(want_loss, rel=1e-6)
    for got, expected in zip(net.gradients().arrays(), reference.backward(w64, cache, d_logits).arrays(), strict=True):
        assert relative(got, expected) < 1e-5
    net.backward(x, y, rows, step=7)
    plain = net.gradients().arrays()
    net.backward(x, y, rows, step=7, weights=np.ones(len(rows), np.float32))
    for one, other in zip(net.gradients().arrays(), plain, strict=True):
        np.testing.assert_array_equal(one, other)  # unit weights: the unweighted step, bit for bit
    assert net.backward(x, y, rows, step=7, weights=np.zeros(len(rows), np.float32)) == 0.0
    assert not any(a.any() for a in net.gradients().arrays())  # a batch of weight 0: no gradient
    with pytest.raises(ValueError, match="one weight per row"):
        net.backward(x, y, rows, step=7, weights=w[:-1])


def test_a_training_step_is_reference_adamw(net: Device) -> None:
    weights, x, y, rows = problem()
    net.set_weights(weights)
    state = reference.adam_state(weights.astype(np.float64))
    w64 = weights.astype(np.float64)
    for step in range(3):
        net.train_step(x, y, rows, step)
        logits, cache = reference.forward(w64, x.astype(np.float64),
                                          reference.dropout_masks(TINY, step, rows, np.float64))  # fmt: skip
        grads = reference.backward(w64, cache, reference.loss_and_logit_grad(logits, y)[1])
        reference.adamw_step(TINY, w64, grads, state)
    for got, expected in zip(net.weights().arrays(), w64.arrays(), strict=True):
        assert relative(got, expected) < 1e-4


def test_two_runs_give_identical_bits(network: Network) -> None:
    weights, x, y, rows = problem()
    results = []
    for _ in range(2):
        model = network(TINY, N_INPUTS, max_batch=BATCH)
        model.set_weights(weights)
        losses = [model.train_step(x, y, rows, step) for step in range(3)]
        results.append((losses, [a.tobytes() for a in model.weights().arrays()], model.predict(x).tobytes()))
        model.close()
    assert results[0] == results[1]


def test_predict_is_the_reference_expected_class(net: Device) -> None:
    weights, x, _, _ = problem()
    net.set_weights(weights)
    many = np.concatenate([x] * 7)  # 70 rows: two chunks of at most 64
    want = reference.predict(reference.forward(weights.astype(np.float64), many.astype(np.float64), None)[0])
    assert relative(net.predict(many), want) < 1e-5


def test_bad_batches_raise(network: Network, net: Device) -> None:
    _, x, _, rows = problem()
    with pytest.raises(ValueError, match="rows of 37 inputs"):
        net.forward(x[:, :5], rows, step=None)
    with pytest.raises(ValueError, match="rows of 37 inputs"):
        net.forward(np.zeros((65, N_INPUTS), np.float32), np.arange(65), step=None)
    with pytest.raises(ValueError, match="at most 8 classes"):
        network(TabMParams(n_classes=9, batch_size=4096, epochs=10), N_INPUTS, 1)
    with pytest.raises(ValueError, match="max_batch"):
        network(TINY, N_INPUTS, 0)
    with pytest.raises(ValueError, match="positive int"):
        network(TINY, N_INPUTS, 1, targets=0)


def test_bf16_rounding_is_to_nearest_even() -> None:
    values = np.array([1.0, 1.00390625, 1.01171875, -2.5, 0.0], np.float32)  # ties: 1 + 2^-8 down, 1 + 3·2^-8 up
    np.testing.assert_array_equal(metal.to_bf16(values), [0x3F80, 0x3F80, 0x3F82, 0xC020, 0x0000])
    np.testing.assert_array_equal(metal.from_bf16(metal.to_bf16(values)), [1.0, 1.0, 1.015625, -2.5, 0.0])
    with pytest.raises(ValueError, match="unknown storage"):
        metal.source("fp16")
    with pytest.raises(ValueError, match="storage='fp16'"):
        TabMParams(storage="fp16", n_classes=5, batch_size=4096, epochs=10)


def test_bf16_storage_follows_the_reference_and_is_deterministic(network: Network) -> None:
    params = TabMParams(
        members=3, blocks=3, width=40, dropout=0.25, seed=11, storage="bf16", n_classes=5, batch_size=4096, epochs=10
    )
    weights, x, y, rows = problem()
    w64, x64 = weights.astype(np.float64), x.astype(np.float64)
    runs = []
    for _ in range(2):
        model = network(params, N_INPUTS, max_batch=BATCH)
        model.set_weights(weights)
        assert relative(model.forward(x, rows, step=None), reference.forward(w64, x64, None)[0]) < 1e-2
        model.backward(x, y, rows, step=7)
        masks = reference.dropout_masks(params, 7, rows, np.float64)
        logits, cache = reference.forward(w64, x64, masks)
        want = reference.backward(w64, cache, reference.loss_and_logit_grad(logits, y)[1])
        for got, expected in zip(model.gradients().arrays(), want.arrays(), strict=True):
            cosine = (got * expected).sum() / np.sqrt((got * got).sum() * (expected * expected).sum())
            assert cosine > 0.995  # bf16 keeps the gradient's direction
        runs.append([a.tobytes() for a in model.gradients().arrays()])
        model.close()
    assert runs[0] == runs[1]


@pytest.mark.parametrize(("micro", "sizes"), [(1, [1] * 10), (3, [3, 3, 3, 1]), (4, [4, 4, 2]), (7, [7, 3])])
def test_micro_batches_give_the_whole_batch_step(network: Network, micro: int, sizes: list[int]) -> None:
    """Each row's forward and loss are the whole batch's bits; the summed gradients agree within float32 error."""
    weights, x, y, rows = problem()
    w64, x64 = weights.astype(np.float64), x.astype(np.float64)
    whole = network(TINY, N_INPUTS, max_batch=BATCH)
    split = network(dataclasses.replace(TINY, micro_batch=micro), N_INPUTS, max_batch=BATCH)
    assert (split.slice_rows, split.capacity, split.rows_pad) == (micro, -(-BATCH // micro) * micro, 64)
    for net in (whole, split):
        net.set_weights(weights)
    np.testing.assert_array_equal(split.forward(x, rows, step=4), whole.forward(x, rows, step=4))
    assert split.sizes == sizes
    assert split.backward(x, y, rows, step=7) == whole.backward(x, y, rows, step=7)
    logits, cache = reference.forward(w64, x64, reference.dropout_masks(TINY, 7, rows, np.float64))
    want = reference.backward(w64, cache, reference.loss_and_logit_grad(logits, y)[1])
    for got, unsplit, expected in zip(split.gradients().arrays(), whole.gradients().arrays(), want.arrays(),
                                      strict=True):  # fmt: skip
        assert relative(got, expected) < 1e-5  # observed: up to 4.2e-7, as the whole batch
        assert relative(got, unsplit) < 1e-5  # observed: up to 2.3e-7 (summation order only)
    state = reference.adam_state(w64)
    for step in range(3):
        split.train_step(x, y, rows, step)
        logits, cache = reference.forward(w64, x64, reference.dropout_masks(TINY, step, rows, np.float64))
        reference.adamw_step(TINY, w64, reference.backward(w64, cache, reference.loss_and_logit_grad(logits, y)[1]),
                             state)  # fmt: skip
    for got, expected in zip(split.weights().arrays(), w64.arrays(), strict=True):
        assert relative(got, expected) < 1e-4
    many = np.concatenate([x] * 7)  # 70 rows: several calls of `capacity` rows, each in micro-batches
    want_pred = reference.predict(reference.forward(w64, many.astype(np.float64), None)[0])
    assert relative(split.predict(many), want_pred) < 1e-5
    whole.close()
    split.close()


def test_a_short_batch_uses_the_micro_batches_it_needs(network: Network) -> None:
    weights, x, y, rows = problem()
    net = network(dataclasses.replace(TINY, micro_batch=3), N_INPUTS, max_batch=BATCH)
    net.set_weights(weights)
    loss = net.backward(x[:5], y[:5], rows[:5], step=2)
    assert net.sizes == [3, 2]
    w64 = weights.astype(np.float64)
    masks = reference.dropout_masks(TINY, 2, rows[:5], np.float64)
    logits, cache = reference.forward(w64, x[:5].astype(np.float64), masks)
    want_loss, d_logits = reference.loss_and_logit_grad(logits, y[:5])
    assert loss == pytest.approx(want_loss, rel=1e-6)
    for got, expected in zip(net.gradients().arrays(), reference.backward(w64, cache, d_logits).arrays(), strict=True):
        assert relative(got, expected) < 1e-5
    with pytest.raises(ValueError, match=r"need 1\.\.12 rows"):
        net.forward(np.zeros((13, N_INPUTS), np.float32), np.arange(13), step=None)
    net.close()


@pytest.mark.parametrize("micro", [0, 10, 64])
def test_a_micro_batch_of_the_whole_batch_or_more_is_the_unsplit_network(network: Network, micro: int) -> None:
    net = network(dataclasses.replace(TINY, micro_batch=micro), N_INPUTS, max_batch=BATCH)
    assert (net.rows_pad, net.slice_rows, net.capacity, len(net.slices)) == (64, 64, 64, 1)
    net.close()


@pytest.mark.parametrize("storage", ["fp32", "bf16"])
def test_two_micro_batched_runs_give_identical_bits(network: Network, storage: str) -> None:
    params = dataclasses.replace(TINY, micro_batch=4, storage=storage)
    weights, x, y, rows = problem()
    results = []
    for _ in range(2):
        model = network(params, N_INPUTS, max_batch=BATCH)
        model.set_weights(weights)
        losses = [model.train_step(x, y, rows, step) for step in range(3)]
        results.append((losses, [a.tobytes() for a in model.weights().arrays()], model.predict(x).tobytes()))
        model.close()
    assert results[0] == results[1]


# -- multi-target heads -----------------------------------------------------------------------------------------------


def multi_problem(targets: int, seed: int = 3) -> tuple[reference.Weights, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    weights = reference.init(TINY, N_INPUTS, targets)
    for array in weights.arrays():
        array[...] = array + generator.normal(0, 0.2, array.shape).astype(np.float32)
    x = generator.normal(size=(BATCH, N_INPUTS)).astype(np.float32)
    y = generator.integers(0, 5, (BATCH, targets))
    return weights, x, y, np.arange(500, 500 + BATCH, dtype=np.int64)


@pytest.mark.parametrize("targets", [3, 20])  # 20: two threadgroups of targets (TARGETS_PER_GROUP = 16)
@pytest.mark.parametrize("micro", [0, 3])
def test_multi_target_heads_match_the_reference(network: Network, targets: int, micro: int) -> None:
    params = dataclasses.replace(TINY, micro_batch=micro)
    weights, x, y, rows = multi_problem(targets)
    w64, x64 = weights.astype(np.float64), x.astype(np.float64)
    net = network(params, N_INPUTS, max_batch=BATCH, targets=targets)
    net.set_weights(weights)
    assert all(np.array_equal(a, b) for a, b in zip(net.weights().arrays(), weights.arrays(), strict=True))
    masks = reference.dropout_masks(TINY, 7, rows, np.float64)
    logits, cache = reference.forward(w64, x64, masks)
    assert logits.shape == (BATCH, TINY.members, 5 * targets)
    assert relative(net.forward(x, rows, step=7), logits) < 1e-5  # observed: 2e-7 to 3e-7
    want_loss, d_logits = reference.loss_and_logit_grad(logits, y)
    assert net.backward(x, y, rows, step=7) == pytest.approx(want_loss, rel=1e-6)
    for got, expected in zip(net.gradients().arrays(), reference.backward(w64, cache, d_logits).arrays(),
                             strict=True):  # fmt: skip
        assert got.shape == expected.shape
        assert relative(got, expected) < 1e-5  # observed: 3e-7 to 5e-7
    pred = net.predict(np.concatenate([x] * 7))  # 70 rows: several calls
    assert pred.shape == (70, targets)
    want = reference.predict(reference.forward(w64, np.concatenate([x64] * 7), None)[0], targets)
    assert relative(pred, want) < 1e-5
    state = reference.adam_state(w64)
    net.set_weights(weights)
    for step in range(3):
        net.train_step(x, y, rows, step)
        logits, cache = reference.forward(w64, x64, reference.dropout_masks(TINY, step, rows, np.float64))
        reference.adamw_step(TINY, w64, reference.backward(w64, cache, reference.loss_and_logit_grad(logits, y)[1]),
                             state)  # fmt: skip
    for got, expected in zip(net.weights().arrays(), w64.arrays(), strict=True):
        assert relative(got, expected) < 1e-4
    net.close()


@pytest.mark.parametrize("storage", ["fp32", "bf16"])
@pytest.mark.parametrize("micro", [0, 4])
def test_one_target_as_a_matrix_gives_the_single_target_bits(network: Network, storage: str, micro: int) -> None:
    params = dataclasses.replace(TINY, storage=storage, micro_batch=micro)
    weights, x, y, rows = problem()
    runs = []
    for labels, targets in ((y, ()), (y[:, None], (1,))):  # the single-target network, and K = 1 given explicitly
        net = network(params, N_INPUTS, BATCH, *targets)
        net.set_weights(weights)
        loss = net.backward(x, labels, rows, step=7)
        grads = [a.tobytes() for a in net.gradients().arrays()]
        losses = [net.train_step(x, labels, rows, step) for step in range(3)]
        pred = net.predict(x)
        assert pred.shape == (BATCH,)
        runs.append((loss, grads, losses, [a.tobytes() for a in net.weights().arrays()], pred.tobytes()))
        net.close()
    assert runs[0] == runs[1]


def test_micro_batched_multi_target_heads_are_the_whole_batch_step(network: Network) -> None:
    weights, x, y, rows = multi_problem(4)
    whole = network(TINY, N_INPUTS, BATCH, targets=4)
    split = network(dataclasses.replace(TINY, micro_batch=3), N_INPUTS, BATCH, targets=4)
    for net in (whole, split):
        net.set_weights(weights)
    np.testing.assert_array_equal(split.forward(x, rows, step=4), whole.forward(x, rows, step=4))
    assert split.backward(x, y, rows, step=7) == whole.backward(x, y, rows, step=7)
    for got, unsplit in zip(split.gradients().arrays(), whole.gradients().arrays(), strict=True):
        assert relative(got, unsplit) < 1e-5  # summation order only
    whole.close()
    split.close()


@pytest.mark.parametrize("storage", ["fp32", "bf16"])
def test_two_multi_target_runs_give_identical_bits(network: Network, storage: str) -> None:
    params = dataclasses.replace(TINY, micro_batch=4, storage=storage)
    weights, x, y, rows = multi_problem(5)
    results = []
    for _ in range(2):
        model = network(params, N_INPUTS, max_batch=BATCH, targets=5)
        model.set_weights(weights)
        losses = [model.train_step(x, y, rows, step) for step in range(3)]
        results.append((losses, [a.tobytes() for a in model.weights().arrays()], model.predict(x).tobytes()))
        model.close()
    assert results[0] == results[1]


def test_bad_multi_target_labels_raise(network: Network) -> None:
    weights, x, y, rows = multi_problem(3)
    net = network(TINY, N_INPUTS, BATCH, targets=3)
    net.set_weights(weights)
    with pytest.raises(ValueError, match=r"shape \(10, 3\), got \(10,\)"):
        net.backward(x, y[:, 0], rows, step=0)
    with pytest.raises(ValueError, match=r"got \(10, 2\)"):
        net.train_step(x, y[:, :2], rows, step=0)
    with pytest.raises(ValueError, match=r"integer classes 0 \.\. 4"):
        net.train_step(x, np.where(y == 0, 5, y), rows, step=0)
    net.close()


def test_regression_steps_and_predictions_follow_the_reference(network: Network) -> None:
    params = dataclasses.replace(TINY, n_classes=1, objective="mse")
    generator = np.random.default_rng(11)
    weights = reference.init(params, N_INPUTS)
    for array in weights.arrays():
        array[...] = array + generator.normal(0, 0.2, array.shape).astype(np.float32)
    x = generator.normal(size=(BATCH, N_INPUTS)).astype(np.float32)
    y = generator.normal(size=BATCH).astype(np.float32)
    rows = np.arange(100, 100 + BATCH, dtype=np.int64)
    net = network(params, N_INPUTS, max_batch=BATCH)
    try:
        net.set_weights(weights)
        w64, state = weights.astype(np.float64), reference.adam_state(weights.astype(np.float64))
        net.backward(x, y, rows, 0)  # the gradients themselves: float32 error only
        logits, cache = reference.forward(
            w64, x.astype(np.float64), reference.dropout_masks(params, 0, rows, np.float64)
        )
        grads = reference.backward(w64, cache, reference.loss_and_logit_grad(logits, y, "mse")[1])
        for got, expected in zip(net.gradients().arrays(), grads.arrays(), strict=True):
            assert relative(got, expected) < 1e-4
        for step in range(3):
            loss = net.train_step(x, y, rows, step)
            logits, cache = reference.forward(w64, x.astype(np.float64),
                                              reference.dropout_masks(params, step, rows, np.float64))  # fmt: skip
            expected_loss, d_logits = reference.loss_and_logit_grad(logits, y, "mse")
            assert abs(loss - expected_loss) < 1e-4 * max(1.0, expected_loss)
            reference.adamw_step(params, w64, reference.backward(w64, cache, d_logits), state)
        for got, expected in zip(net.weights().arrays(), w64.arrays(), strict=True):
            assert relative(got, expected) < 1e-2  # AdamW divides by sqrt(v): a looser bound for the weights
        want = reference.predict(reference.forward(w64, x.astype(np.float64), None)[0], None, "mse")
        np.testing.assert_allclose(net.predict(x), want, atol=1e-4)
    finally:
        net.close()


HL = dataclasses.replace(TINY, n_classes=1, objective="hl_gauss", n_bins=24, hl_low=-3.0, hl_high=3.0, hl_sigma=0.2)


def hl_problem() -> tuple[reference.Weights, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(12)
    weights = reference.init(HL, N_INPUTS)
    for array in weights.arrays():
        array[...] = array + generator.normal(0, 0.2, array.shape).astype(np.float32)
    x = generator.normal(size=(BATCH, N_INPUTS)).astype(np.float32)
    z = (1.5 * generator.normal(size=BATCH)).astype(np.float32)
    z[0], z[1] = 5.0, -7.0  # beyond the support: clamped
    return weights, x, z, np.arange(300, 300 + BATCH, dtype=np.int64)


def test_hl_gauss_ce_and_predict_centre_follow_the_reference(network: Network) -> None:
    """The 24-bin head runs as 3 groups of 8 (head_forward, head_backward), the soft-label loss in hl_gauss_ce and
    the expected bin centre in predict_centre."""
    weights, x, z, rows = hl_problem()
    soft = hl_gauss.labels(z, HL.n_bins, HL.hl_low, HL.hl_high, HL.hl_sigma)
    net = network(HL, N_INPUTS, max_batch=BATCH)
    try:
        net.set_weights(weights)
        w64, state = weights.astype(np.float64), reference.adam_state(weights.astype(np.float64))
        assert net.forward(x, rows, step=None).shape == (BATCH, HL.members, 24)
        net.backward(x, z, rows, 0)
        logits, cache = reference.forward(w64, x.astype(np.float64), reference.dropout_masks(HL, 0, rows, np.float64))
        grads = reference.backward(w64, cache, reference.loss_and_logit_grad(logits, soft, "hl_gauss")[1])
        for got, expected in zip(net.gradients().arrays(), grads.arrays(), strict=True):
            assert relative(got, expected) < 1e-4
        for step in range(3):
            loss = net.train_step(x, z, rows, step)
            logits, cache = reference.forward(w64, x.astype(np.float64),
                                              reference.dropout_masks(HL, step, rows, np.float64))  # fmt: skip
            expected_loss, d_logits = reference.loss_and_logit_grad(logits, soft, "hl_gauss")
            assert abs(loss - expected_loss) < 1e-4 * max(1.0, expected_loss)
            reference.adamw_step(HL, w64, reference.backward(w64, cache, d_logits), state)
        for got, expected in zip(net.weights().arrays(), w64.arrays(), strict=True):
            assert relative(got, expected) < 1e-2
        centres = hl_gauss.centres(HL.n_bins, HL.hl_low, HL.hl_high)
        want = reference.predict(reference.forward(w64, x.astype(np.float64), None)[0], None, "hl_gauss", centres)
        np.testing.assert_allclose(net.predict(x), want, atol=1e-5)
    finally:
        net.close()


def test_micro_batched_hl_gauss_is_the_whole_batch_step_and_repeats_its_bits(network: Network) -> None:
    weights, x, z, rows = hl_problem()
    whole = network(HL, N_INPUTS, max_batch=BATCH)
    split = network(dataclasses.replace(HL, micro_batch=4), N_INPUTS, max_batch=BATCH)
    again = network(dataclasses.replace(HL, micro_batch=4), N_INPUTS, max_batch=BATCH)
    try:
        for net in (whole, split, again):
            net.set_weights(weights)
        assert split.backward(x, z, rows, step=7) == whole.backward(x, z, rows, step=7)
        for got, unsplit in zip(split.gradients().arrays(), whole.gradients().arrays(), strict=True):
            assert relative(got, unsplit) < 1e-5  # summation order only
        for step in range(2):
            split.train_step(x, z, rows, step)
            again.train_step(x, z, rows, step)
        assert all(np.array_equal(a, b) for a, b in zip(split.weights().arrays(), again.weights().arrays(),
                                                        strict=True))  # fmt: skip
        np.testing.assert_array_equal(split.predict(x), again.predict(x))
    finally:
        for net in (whole, split, again):
            net.close()


def test_hl_gauss_has_one_target(network: Network) -> None:
    with pytest.raises(ValueError, match="one target"):
        network(HL, N_INPUTS, max_batch=BATCH, targets=2)


# -- steps under a loss of the caller's -------------------------------------------------------------------------------


def kernel_mse(y: np.ndarray) -> reference.Loss:
    """Return the loss whose gradient is the ``mse_loss`` kernel's, one float32 operation for another."""

    def loss(outputs: np.ndarray, rows: np.ndarray) -> tuple[float, np.ndarray]:
        diff = outputs - y[: len(rows), None, None]
        return float((diff.astype(np.float64) ** 2).mean()), np.float32(2.0) * diff * np.float32(1.0 / diff.size)

    return loss


@pytest.mark.parametrize(("storage", "micro"), [("fp32", 0), ("fp32", 3), ("fp32", 4), ("bf16", 0), ("bf16", 7)])
def test_a_loss_step_with_the_kernels_gradient_is_the_training_step(network: Network, storage: str, micro: int) -> None:
    params = dataclasses.replace(TINY, n_classes=1, objective="mse", storage=storage, micro_batch=micro)
    generator = np.random.default_rng(13)
    weights = reference.init(params, N_INPUTS)
    x = generator.normal(size=(BATCH, N_INPUTS)).astype(np.float32)
    y = generator.normal(size=BATCH).astype(np.float32)
    rows = np.arange(100, 100 + BATCH, dtype=np.int64)
    built_in, hooked = (network(params, N_INPUTS, max_batch=BATCH) for _ in range(2))
    try:
        for net in (built_in, hooked):
            net.set_weights(weights)
        for step, n in enumerate([BATCH, BATCH, 4, BATCH]):  # a short batch between whole ones
            want = built_in.train_step(x[:n], y[:n], rows[:n], step)
            got = hooked.loss_step(x[:n], kernel_mse(y), rows[:n], step)
            assert got == pytest.approx(want, rel=1e-6)
            for a, b in zip(hooked.weights().arrays(), built_in.weights().arrays(), strict=True):
                np.testing.assert_array_equal(a, b)
        assert hooked.adam_steps == built_in.adam_steps == 4
    finally:
        built_in.close()
        hooked.close()


@pytest.mark.parametrize("micro", [0, 3])
def test_a_loss_step_follows_the_reference(network: Network, micro: int) -> None:
    weights, x, y, rows = problem()
    seen: list[tuple[tuple[int, ...], np.dtype, bytes]] = []

    def cross_entropy(outputs: np.ndarray, index: np.ndarray) -> tuple[float, np.ndarray]:
        seen.append((outputs.shape, outputs.dtype, index.tobytes()))
        return reference.loss_and_logit_grad(outputs, y)

    net = network(dataclasses.replace(TINY, micro_batch=micro), N_INPUTS, max_batch=BATCH)
    try:
        net.set_weights(weights)
        w64, state = weights.astype(np.float64), reference.adam_state(weights.astype(np.float64))
        for step in range(3):
            loss = net.loss_step(x, cross_entropy, rows, step)
            logits, cache = reference.forward(w64, x.astype(np.float64),
                                              reference.dropout_masks(TINY, step, rows, np.float64))  # fmt: skip
            expected, d_logits = reference.loss_and_logit_grad(logits, y)
            assert loss == pytest.approx(expected, rel=1e-5)
            reference.adamw_step(TINY, w64, reference.backward(w64, cache, d_logits), state)
        for got, want in zip(net.weights().arrays(), w64.arrays(), strict=True):
            assert relative(got, want) < 1e-4
        assert seen == [((BATCH, 3, 5), np.dtype(np.float32), rows.tobytes())] * 3
    finally:
        net.close()


def test_a_loss_step_refuses_hl_gauss_and_a_bad_gradient(network: Network, net: Device) -> None:
    weights, x, _, rows = problem()
    net.set_weights(weights)
    with pytest.raises(ValueError, match="gradient of shape"):
        net.loss_step(x, lambda outputs, index: (0.0, outputs[:, 0]), rows, 0)
    with pytest.raises(ValueError, match="not finite"):
        net.loss_step(x, lambda outputs, index: (0.0, outputs * np.float32(np.inf)), rows, 0)
    assert net.adam_steps == 0  # no update from a refused loss
    for got, want in zip(net.weights().arrays(), weights.arrays(), strict=True):
        np.testing.assert_array_equal(got, want)
    hl = network(HL, N_INPUTS, max_batch=BATCH)
    try:
        with pytest.raises(ValueError, match="HL-Gauss network takes no loss"):
            hl.loss_step(x, lambda outputs, index: (0.0, outputs), rows, 0)
    finally:
        hl.close()


# -- piecewise-linear embeddings --------------------------------------------------------------------------------------

PWL = dataclasses.replace(TINY, num_emb_type="pwl", num_emb_n_bins=6, d_embedding=5)
EMBEDDED = 20  # of the N_INPUTS: the others enter block 0 as they are


def embedded_problem(seed: int = 3) -> tuple[reference.Weights, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    bins = emb.fit_bins(generator.normal(size=(200, EMBEDDED)), PWL.num_emb_n_bins)
    weights = reference.init(PWL, N_INPUTS, bins=bins)
    for array in weights.arrays():  # off the initial pattern (the PLE weights start at zero)
        array[...] = array + generator.normal(0, 0.2, array.shape).astype(np.float32)
    x = generator.normal(size=(BATCH, N_INPUTS)).astype(np.float32)
    return weights, x, generator.integers(0, 5, BATCH), np.arange(500, 500 + BATCH, dtype=np.int64)


@pytest.mark.parametrize("micro", [0, 4])
def test_embedded_networks_follow_the_reference(network: Network, micro: int) -> None:
    weights, x, y, rows = embedded_problem()
    assert weights.bins is not None
    params = dataclasses.replace(PWL, micro_batch=micro)
    net = network(params, N_INPUTS, max_batch=BATCH, bins_shape=weights.bins.scale.shape)
    try:
        assert net.ldd == 128  # 20 · 5 embedded values and 17 inputs
        net.set_weights(weights)
        assert all(np.array_equal(a, b) for a, b in zip(net.weights().arrays(), weights.arrays(), strict=True))
        w64 = weights.astype(np.float64)
        logits, cache = reference.forward(
            w64, x.astype(np.float64), reference.dropout_masks(params, 7, rows, np.float64)
        )
        assert relative(net.forward(x, rows, step=7), logits) < 1e-5
        loss = net.backward(x, y, rows, step=7)
        want_loss, d_logits = reference.loss_and_logit_grad(logits, y)
        assert loss == pytest.approx(want_loss, rel=1e-6)
        want = reference.backward(w64, cache, d_logits)
        for got, expected in zip(net.gradients().arrays(), want.arrays(), strict=True):
            assert got.shape == expected.shape
            assert relative(got, expected) < 1e-5
        state = reference.adam_state(w64)
        for step in range(3):  # AdamW moves every array, the embeddings' too
            net.train_step(x, y, rows, step)
            logits, cache = reference.forward(w64, x.astype(np.float64),
                                              reference.dropout_masks(params, step, rows, np.float64))  # fmt: skip
            grads = reference.backward(w64, cache, reference.loss_and_logit_grad(logits, y)[1])
            reference.adamw_step(params, w64, grads, state)
        for got, expected in zip(net.weights().arrays(), w64.arrays(), strict=True):
            assert relative(got, expected) < 1e-4
        many = np.concatenate([x] * 3)
        want_pred = reference.predict(reference.forward(w64, many.astype(np.float64), None)[0])
        assert relative(net.predict(many), want_pred) < 1e-5
        plain, *_ = problem()
        with pytest.raises(ValueError, match="embeddings"):
            net.set_weights(plain)
    finally:
        net.close()
    with pytest.raises(ValueError, match="bins"):
        network(TINY, N_INPUTS, max_batch=BATCH, bins_shape=(EMBEDDED, 7))
    with pytest.raises(ValueError, match="bins"):
        network(PWL, N_INPUTS, max_batch=BATCH)


def test_clipped_steps_follow_the_reference_and_a_wide_norm_clips_nothing(network: Network) -> None:
    weights, x, y, rows = problem()
    w64, x64 = weights.astype(np.float64), x.astype(np.float64)
    logits, cache = reference.forward(w64, x64, reference.dropout_masks(TINY, 0, rows, np.float64))
    grads = reference.backward(w64, cache, reference.loss_and_logit_grad(logits, y)[1])
    norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.arrays())))
    clipped = dataclasses.replace(TINY, gradient_clipping_norm=norm / 3)
    net = network(clipped, N_INPUTS, max_batch=BATCH)
    plain, wide = (
        network(TINY, N_INPUTS, max_batch=BATCH),
        network(dataclasses.replace(TINY, gradient_clipping_norm=1e9), N_INPUTS, max_batch=BATCH),
    )
    try:
        for model in (net, plain, wide):
            model.set_weights(weights)
        state = reference.adam_state(w64)
        for step in range(3):
            for model in (net, plain, wide):
                model.train_step(x, y, rows, step)
            logits, cache = reference.forward(w64, x64, reference.dropout_masks(TINY, step, rows, np.float64))
            grads = reference.backward(w64, cache, reference.loss_and_logit_grad(logits, y)[1])
            reference.adamw_step(clipped, w64, grads, state)
        for got, expected in zip(net.weights().arrays(), w64.arrays(), strict=True):
            assert relative(got, expected) < 1e-4
        for one, other in zip(wide.weights().arrays(), plain.weights().arrays(), strict=True):
            np.testing.assert_array_equal(one, other)  # a factor of 1: the unclipped bits
        assert not np.array_equal(net.weights().arrays()[0], plain.weights().arrays()[0])
    finally:
        for model in (net, plain, wide):
            model.close()
