import math

import numpy as np
import pytest

from idun.tabm import TabMParams, encoder, reference

TINY = TabMParams(members=3, blocks=2, width=5, dropout=0.25, n_classes=4, seed=11, encoder="gru", encoder_embed=3,
                  encoder_hidden=4, batch_size=4096, epochs=10)  # fmt: skip


def tiny_problem() -> tuple[reference.Weights, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    generator = np.random.default_rng(5)
    weights = reference.init(TINY, n_inputs=6, sequence_inputs=4).astype(np.float64)
    for array in weights.arrays():  # off the initial pattern (adapters 1, biases equal)
        array[...] = array + generator.normal(0, 0.3, array.shape)
    x = generator.normal(size=(5, 6))
    seq = generator.normal(size=(5, 3, 4))
    seq[1, 0] = 0.0  # an absent step (channels and flag 0)
    y = generator.integers(0, 4, 5)
    masks = reference.dropout_masks(TINY, step=2, rows=np.arange(40, 45), dtype=np.float64)
    return weights, x, seq, y, masks


def test_gradients_match_finite_differences() -> None:
    weights, x, seq, y, masks = tiny_problem()

    def loss() -> float:
        logits, _ = reference.forward(weights, x, masks, seq)
        return reference.loss_and_logit_grad(logits, y)[0]

    logits, cache = reference.forward(weights, x, masks, seq)
    grads = reference.backward(weights, cache, reference.loss_and_logit_grad(logits, y)[1])
    assert grads.encoder is not None
    assert len(grads.arrays()) == len(weights.arrays()) == 4 * TINY.blocks + 2 + 6
    h = 1e-6
    for array, grad in zip(weights.arrays(), grads.arrays(), strict=True):
        assert grad.shape == array.shape
        numeric = np.empty_like(array)
        for index in np.ndindex(array.shape):
            keep = array[index]
            array[index] = keep + h
            up = loss()
            array[index] = keep - h
            down = loss()
            array[index] = keep
            numeric[index] = (up - down) / (2 * h)
        scale = max(np.abs(numeric).max(), 1e-12)
        assert np.abs(grad - numeric).max() / scale < 1e-6


def test_one_step_is_the_gru_cell_from_a_zero_state() -> None:
    w = encoder.init(TINY, inputs=4).astype(np.float64)
    x = np.random.default_rng(1).normal(size=(2, 1, 4))
    out, _ = encoder.forward(w, x)
    e = encoder.gelu(x[:, 0] @ w.embed.T + w.embed_bias)
    gi, gh = e @ w.w_ih.T + w.b_ih, w.b_hh  # h = 0: W_hh h vanishes
    r = 1 / (1 + np.exp(-(gi[:, 0:4] + gh[0:4])))
    z = 1 / (1 + np.exp(-(gi[:, 4:8] + gh[4:8])))
    n = np.tanh(gi[:, 8:12] + r * gh[8:12])
    np.testing.assert_allclose(out, (1 - z) * n, rtol=1e-12)


def test_steps_run_oldest_first() -> None:
    w = encoder.init(TINY, inputs=4).astype(np.float64)
    x = np.random.default_rng(2).normal(size=(3, 4, 4))
    out, cache = encoder.forward(w, x)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(cache.h[:, 0], 0.0)
    step_by_step = encoder.forward(w, x[:, 3:])[0]  # only the last step, from a zero state
    assert not np.allclose(out, step_by_step)
    np.testing.assert_allclose(encoder.forward(w, x[:, :3])[0], cache.h[:, 3], rtol=1e-12)


def test_gelu_is_exact() -> None:
    x = np.array([-3.0, -1.0, 0.0, 0.5, 1.0, 4.0])
    want = [v * 0.5 * (1 + math.erf(v / math.sqrt(2))) for v in x]
    np.testing.assert_allclose(encoder.gelu(x), want, rtol=1e-13)
    assert encoder.gelu(np.array([1.0], np.float32)).dtype == np.float32
    h = 1e-6
    np.testing.assert_allclose(encoder.gelu_grad(x), (encoder.gelu(x + h) - encoder.gelu(x - h)) / (2 * h), rtol=1e-8)


def test_init_is_torch_s_bounds_from_its_own_streams() -> None:
    w = encoder.init(TINY, inputs=4)
    assert [a.shape for a in w.arrays()] == [(3, 4), (3,), (12, 3), (12, 4), (12,), (12,)]
    assert all(a.dtype == np.float32 for a in w.arrays())
    assert np.abs(w.embed).max() <= 1 / 2
    assert np.abs(w.embed_bias).max() <= 1 / 2
    assert all(np.abs(a).max() <= 1 / 2 for a in w.arrays()[2:])
    assert len({a.tobytes()[:8] for a in w.arrays()}) == 6  # one draw per array
    again = encoder.init(TINY, inputs=4)
    assert all(np.array_equal(a, b) for a, b in zip(w.arrays(), again.arrays(), strict=True))
    with pytest.raises(ValueError, match="inputs"):
        encoder.init(TINY, inputs=0)


def test_the_encoder_leaves_tabm_s_other_draws_alone() -> None:
    plain = reference.init(
        TabMParams(members=3, blocks=2, width=5, n_classes=4, seed=11, batch_size=4096, epochs=10), n_inputs=6
    )
    temporal = reference.init(TINY, n_inputs=6, sequence_inputs=4)
    assert temporal.weight[0].shape == (5, 4 + 6)
    assert temporal.r[0].shape == (3, 10)
    np.testing.assert_array_equal(plain.weight[1], temporal.weight[1])
    np.testing.assert_array_equal(plain.head, temporal.head)
    np.testing.assert_array_equal(plain.head_bias, temporal.head_bias)
    assert plain.encoder is None
    assert temporal.encoder is not None


def test_encoder_and_sequences_go_together() -> None:
    weights, x, seq, _, _ = tiny_problem()
    with pytest.raises(ValueError, match="sequences"):
        reference.forward(weights, x, None)
    plain = reference.init(
        TabMParams(members=3, blocks=2, width=5, n_classes=4, batch_size=4096, epochs=10), n_inputs=6
    )
    with pytest.raises(ValueError, match="sequences"):
        reference.forward(plain.astype(np.float64), x, None, seq)
    with pytest.raises(ValueError, match="sequence inputs"):
        reference.init(TINY, n_inputs=6)
    with pytest.raises(ValueError, match="sequence inputs"):
        reference.init(TabMParams(n_classes=5, batch_size=4096, epochs=10), n_inputs=6, sequence_inputs=4)


def test_astype_carries_the_encoder() -> None:
    weights = reference.init(TINY, n_inputs=6, sequence_inputs=4)
    wide = weights.astype(np.float64)
    assert wide.encoder is not None
    assert all(a.dtype == np.float64 for a in wide.arrays())
