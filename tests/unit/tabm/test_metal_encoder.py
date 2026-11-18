"""MetalTabM and CudaTabM with a temporal encoder against the float64 reference (within tolerance; identical bits run
to run).

Every network test runs once per GPU backend (the ``network`` fixture), marked ``metal`` or ``cuda`` for it.
"""

import ctypes
import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from idun.tabm import TabMParams, cuda, metal, reference
from idun.tabm.device import DeviceTabM as Device

Network = Callable[..., Device]

TINY = TabMParams(members=3, blocks=2, width=40, dropout=0.25, seed=11, learning_rate=1e-2, weight_decay=0.1,
                  encoder="gru", encoder_embed=48, encoder_hidden=40, n_classes=5, batch_size=4096,
                  epochs=10)  # fmt: skip
N_INPUTS, BATCH, STEPS, SEQ_INPUTS = 37, 10, 5, 7


def problem(
    params: TabMParams = TINY, seed: int = 3
) -> tuple[reference.Weights, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    weights = reference.init(params, N_INPUTS, sequence_inputs=SEQ_INPUTS)
    for array in weights.arrays():  # off the initial pattern, so every gradient path is exercised
        array[...] = array + generator.normal(0, 0.2, array.shape).astype(np.float32)
    x = generator.normal(size=(BATCH, N_INPUTS)).astype(np.float32)
    seq = generator.normal(size=(BATCH, STEPS, SEQ_INPUTS)).astype(np.float32)
    seq[2, :3] = 0.0  # absent steps: channels and flag 0
    y = generator.integers(0, 5, BATCH)
    rows = np.arange(500, 500 + BATCH, dtype=np.int64)
    return weights, x, seq, y, rows


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


def encoder_net(network: Network, params: TabMParams = TINY, max_batch: int = BATCH) -> Device:
    return network(params, N_INPUTS, max_batch, sequence_shape=(STEPS, SEQ_INPUTS))


def reference_step(params: TabMParams, weights: reference.Weights, x: np.ndarray, seq: np.ndarray, y: np.ndarray,
                   rows: np.ndarray, step: int) -> tuple[float, reference.Weights]:  # fmt: skip
    w64 = weights.astype(np.float64)
    masks = reference.dropout_masks(params, step, rows, np.float64)
    logits, cache = reference.forward(w64, x.astype(np.float64), masks, seq.astype(np.float64))
    loss, d_logits = reference.loss_and_logit_grad(logits, y)
    return loss, reference.backward(w64, cache, d_logits)


def test_the_struct_size_matches_the_kernel() -> None:
    assert ctypes.sizeof(metal.SeqParams) == 48


def test_an_encoder_network_loads_the_encoder_kernels(network: Network) -> None:
    names = ("seq_forward", "seq_input_grad", "seq_member_sum", "seq_backward", "seq_weight_grad",
             "seq_column_partials")  # fmt: skip
    assert names == metal.ENCODER_KERNELS
    weights, x, seq, y, rows = problem()
    net = network(TINY, N_INPUTS, BATCH, sequence_shape=(STEPS, SEQ_INPUTS), profile=True)
    net.set_weights(weights)
    net.train_step(x, y, rows, 0, seq)
    assert set(names) <= set(net.device.kernel_seconds)  # the kernels the step ran
    net.close()
    plain = network(dataclasses.replace(TINY, encoder="none"), N_INPUTS, BATCH, profile=True)
    plain.set_weights(reference.init(dataclasses.replace(TINY, encoder="none"), N_INPUTS))
    plain.train_step(x, y, rows, 0)
    assert set(plain.device.kernel_seconds)
    assert not set(names) & set(plain.device.kernel_seconds)
    plain.close()


@pytest.mark.parametrize(("embed", "hidden"), [(48, 40), (32, 64), (128, 128), (3, 1)])
def test_forward_and_gradients_follow_the_reference(network: Network, embed: int, hidden: int) -> None:
    params = dataclasses.replace(TINY, encoder_embed=embed, encoder_hidden=hidden)
    weights, x, seq, y, rows = problem(params)
    net = encoder_net(network, params)
    net.set_weights(weights)
    masks = reference.dropout_masks(params, 4, rows, np.float64)
    want = reference.forward(weights.astype(np.float64), x.astype(np.float64), masks, seq.astype(np.float64))[0]
    assert relative(net.forward(x, rows, 4, seq), want) < 1e-5  # observed: 1.9e-7
    loss = net.backward(x, y, rows, 7, seq)
    want_loss, want_grads = reference_step(params, weights, x, seq, y, rows, 7)
    assert loss == pytest.approx(want_loss, rel=1e-6)
    got = net.gradients()
    assert got.encoder is not None
    assert [a.shape for a in got.arrays()] == [a.shape for a in want_grads.arrays()]
    for name, a, b in zip(range(len(got.arrays())), got.arrays(), want_grads.arrays(), strict=True):
        assert relative(a, b) < 1e-5, name  # observed: up to 7.5e-7
    net.close()


def test_training_steps_are_reference_adamw(network: Network) -> None:
    weights, x, seq, y, rows = problem()
    net = encoder_net(network)
    net.set_weights(weights)
    w64 = weights.astype(np.float64)
    state = reference.adam_state(w64)
    for step in range(3):
        net.train_step(x, y, rows, step, seq)
        reference.adamw_step(TINY, w64, reference_step(TINY, w64, x, seq, y, rows, step)[1], state)
    for got, want in zip(net.weights().arrays(), w64.arrays(), strict=True):
        assert relative(got, want) < 1e-4
    net.close()


@pytest.mark.parametrize("micro", [1, 3, 7])
def test_micro_batches_give_the_whole_batch_step(network: Network, micro: int) -> None:
    weights, x, seq, y, rows = problem()
    whole, split = encoder_net(network), encoder_net(network, dataclasses.replace(TINY, micro_batch=micro))
    for net in (whole, split):
        net.set_weights(weights)
    np.testing.assert_array_equal(split.forward(x, rows, 4, seq), whole.forward(x, rows, 4, seq))
    assert split.backward(x, y, rows, 7, seq) == whole.backward(x, y, rows, 7, seq)
    want = reference_step(TINY, weights, x, seq, y, rows, 7)[1]
    for got, unsplit, expected in zip(split.gradients().arrays(), whole.gradients().arrays(), want.arrays(),
                                      strict=True):  # fmt: skip
        assert relative(got, expected) < 1e-5
        assert relative(got, unsplit) < 1e-5  # summation order only
    whole.close()
    split.close()


@pytest.mark.parametrize(("storage", "micro"), [("fp32", 0), ("fp32", 4), ("bf16", 0)])
def test_two_runs_give_identical_bits(network: Network, storage: str, micro: int) -> None:
    params = dataclasses.replace(TINY, storage=storage, micro_batch=micro)
    weights, x, seq, y, rows = problem()
    runs = []
    for _ in range(2):
        net = encoder_net(network, params)
        net.set_weights(weights)
        losses = [net.train_step(x, y, rows, step, seq) for step in range(3)]
        runs.append((losses, [a.tobytes() for a in net.weights().arrays()], net.predict(x, seq).tobytes()))
        net.close()
    assert runs[0] == runs[1]


def test_bf16_storage_keeps_the_gradients_direction(network: Network) -> None:
    weights, x, seq, y, rows = problem()
    net = encoder_net(network, dataclasses.replace(TINY, storage="bf16"))
    net.set_weights(weights)
    net.backward(x, y, rows, 7, seq)
    want = reference_step(TINY, weights, x, seq, y, rows, 7)[1]
    for got, expected in zip(net.gradients().arrays(), want.arrays(), strict=True):
        a, b = got.ravel().astype(np.float64), expected.ravel()
        assert a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.99
    net.close()


def test_predict_is_the_reference_expected_class_over_several_calls(network: Network) -> None:
    weights, x, seq, _, _ = problem()
    net = encoder_net(network, dataclasses.replace(TINY, micro_batch=4))
    net.set_weights(weights)
    many, many_seq = np.concatenate([x] * 5), np.concatenate([seq] * 5)  # 50 rows: calls of `capacity` rows
    logits = reference.forward(weights.astype(np.float64), many.astype(np.float64), None, many_seq.astype(np.float64))
    assert relative(net.predict(many, many_seq), reference.predict(logits[0])) < 1e-5
    assert relative(net.inference_logits(x, seq), logits[0][:BATCH]) < 1e-5
    net.close()


def test_sequences_must_fit_the_network(network: Network) -> None:
    weights, x, seq, _, rows = problem()
    net = encoder_net(network)
    net.set_weights(weights)
    with pytest.raises(ValueError, match="takes sequences"):
        net.forward(x, rows, None)
    with pytest.raises(ValueError, match="sequences of shape"):
        net.forward(x, rows, None, seq[:, 1:])
    plain_weights = reference.init(dataclasses.replace(TINY, encoder="none"), N_INPUTS)
    with pytest.raises(ValueError, match="encoder does not match"):
        net.set_weights(plain_weights)
    assert net.steps_inputs() == (STEPS, SEQ_INPUTS)
    net.close()
    plain = network(dataclasses.replace(TINY, encoder="none"), N_INPUTS, BATCH)
    with pytest.raises(ValueError, match="takes none"):
        plain.predict(x, seq)
    with pytest.raises(ValueError, match="no encoder"):
        plain.steps_inputs()
    plain.close()
    with pytest.raises(ValueError, match="sequences of shape"):
        network(TINY, N_INPUTS, BATCH)
    with pytest.raises(ValueError, match="sequences of shape"):
        network(dataclasses.replace(TINY, encoder="none"), N_INPUTS, BATCH, sequence_shape=(2, 3))
    with pytest.raises(ValueError, match="at least one step"):
        network(TINY, N_INPUTS, BATCH, sequence_shape=(0, 3))


@pytest.mark.parametrize("micro", [0, 3])
def test_a_loss_step_with_the_kernels_gradient_is_the_training_step(network: Network, micro: int) -> None:
    params = dataclasses.replace(TINY, n_classes=1, objective="mse", micro_batch=micro)
    weights, x, seq, _, rows = problem(params)
    y = np.random.default_rng(17).normal(size=BATCH).astype(np.float32)

    def loss(outputs: np.ndarray, index: np.ndarray) -> tuple[float, np.ndarray]:  # the mse_loss kernel's arithmetic
        diff = outputs - y[: len(index), None, None]
        return float((diff.astype(np.float64) ** 2).mean()), np.float32(2.0) * diff * np.float32(1.0 / diff.size)

    built_in, hooked = encoder_net(network, params), encoder_net(network, params)
    for net in (built_in, hooked):
        net.set_weights(weights)
    for step in range(3):
        want = built_in.train_step(x, y, rows, step, seq)
        assert hooked.loss_step(x, loss, rows, step, seq) == pytest.approx(want, rel=1e-6)
    for a, b in zip(hooked.weights().arrays(), built_in.weights().arrays(), strict=True):
        np.testing.assert_array_equal(a, b)
    built_in.close()
    hooked.close()
