import dataclasses

import numpy as np
import pytest

from idun import rng
from idun.tabm import TabMParams, hl_gauss, reference
from idun.tabm import embeddings as emb

TINY = TabMParams(members=3, blocks=3, width=5, dropout=0.25, n_classes=5, seed=11, batch_size=4096, epochs=10)


def tiny_problem(
    dtype: type[np.floating] = np.float64,
) -> tuple[reference.Weights, np.ndarray, np.ndarray, list[np.ndarray]]:
    generator = np.random.default_rng(3)
    weights = reference.init(TINY, n_inputs=7).astype(dtype)
    for array in weights.arrays():  # move every parameter off its initial pattern (adapters 1, biases equal)
        array[...] = array + generator.normal(0, 0.3, array.shape).astype(dtype)
    x = generator.normal(size=(6, 7)).astype(dtype)
    y = generator.integers(0, 5, 6)
    masks = reference.dropout_masks(TINY, step=4, rows=np.arange(100, 106), dtype=dtype)
    return weights, x, y, masks


def loss(weights: reference.Weights, x: np.ndarray, y: np.ndarray, masks: list[np.ndarray]) -> float:
    logits, _ = reference.forward(weights, x, masks)
    return reference.loss_and_logit_grad(logits, y)[0]


def test_gradients_match_finite_differences() -> None:
    weights, x, y, masks = tiny_problem()
    logits, cache = reference.forward(weights, x, masks)
    grads = reference.backward(weights, cache, reference.loss_and_logit_grad(logits, y)[1])
    h = 1e-6
    for array, grad in zip(weights.arrays(), grads.arrays(), strict=True):
        assert grad.shape == array.shape
        numeric = np.empty_like(array)
        for index in np.ndindex(array.shape):
            keep = array[index]
            array[index] = keep + h
            up = loss(weights, x, y, masks)
            array[index] = keep - h
            down = loss(weights, x, y, masks)
            array[index] = keep
            numeric[index] = (up - down) / (2 * h)
        scale = max(np.abs(numeric).max(), 1e-12)
        assert np.abs(grad - numeric).max() / scale < 1e-6


def test_dropout_masks_are_scaled_draws_keyed_by_the_absolute_row() -> None:
    p = TabMParams(n_classes=5, members=4, blocks=2, width=64, dropout=0.1, seed=5, batch_size=4096, epochs=10)
    rows = np.arange(1000, 1600)
    masks = reference.dropout_masks(p, step=7, rows=rows, dtype=np.float32)
    assert [m.shape for m in masks] == [(600, 4, 64)] * 2
    assert set(np.unique(masks[0])) == {np.float32(0.0), np.float32(1 / 0.9)}
    assert abs((masks[0] == 0).mean() - 0.1) < 0.005
    shuffled = rows[::-1].copy()
    np.testing.assert_array_equal(reference.dropout_masks(p, 7, shuffled, np.float32)[1], masks[1][::-1])
    assert not np.array_equal(masks[0], masks[1])
    assert not np.array_equal(reference.dropout_masks(p, 8, rows, np.float32)[0], masks[0])
    key = rng.derive_key(5, reference.DROPOUT_SALT, 7, 1)
    kept = rng.element_bits_one(np.uint64(key), np.uint64(1003), np.uint64(2 * 64 + 9)) >= round(0.1 * 2**24)
    assert (masks[1][3, 2, 9] != 0) == kept


def test_init_matches_tabm_defaults() -> None:
    w = reference.init(TabMParams(n_classes=5, members=4, width=16, seed=1, batch_size=4096, epochs=10), n_inputs=10)
    assert [a.shape for a in w.weight] == [(16, 10), (16, 16), (16, 16)]
    assert all(a.dtype == np.float32 for a in w.arrays())
    assert set(np.unique(w.r[0])) == {-1.0, 1.0}
    assert all((a == 1).all() for a in (*w.r[1:], *w.s))
    assert all((b == b[0]).all() for b in w.bias)  # one bias draw shared by every member
    assert np.abs(w.weight[0]).max() <= 1 / np.sqrt(10)
    assert np.abs(w.weight[1]).max() <= 1 / np.sqrt(16)
    assert w.head.shape == (4, 16, 5)
    assert w.head_bias.shape == (4, 5)
    assert np.abs(w.head).max() <= 1 / 4
    again = reference.init(
        TabMParams(n_classes=5, members=4, width=16, seed=1, batch_size=4096, epochs=10), n_inputs=10
    )
    other = reference.init(
        TabMParams(n_classes=5, members=4, width=16, seed=2, batch_size=4096, epochs=10), n_inputs=10
    )
    assert all(np.array_equal(a, b) for a, b in zip(w.arrays(), again.arrays(), strict=True))
    assert not np.array_equal(w.weight[0], other.weight[0])


def test_inference_is_the_unmasked_forward_and_predicts_the_expected_class() -> None:
    weights, x, _, _ = tiny_problem()
    ones: list[np.ndarray] = [np.ones((6, 3, 5))] * 3
    np.testing.assert_array_equal(reference.forward(weights, x, None)[0], reference.forward(weights, x, ones)[0])
    logits = np.zeros((2, 3, 5))
    logits[0, :, 4] = 50.0  # every member certain of class 4
    np.testing.assert_allclose(reference.predict(logits), [4.0, 2.0])


def test_adamw_step_is_torch_adamw() -> None:
    p = TabMParams(
        n_classes=5, learning_rate=0.1, weight_decay=0.5, beta1=0.8, beta2=0.9, eps=1e-3, batch_size=4096, epochs=10
    )
    weights, _, _, _ = tiny_problem()
    grads = weights.astype(np.float64)
    for g in grads.arrays():
        g[...] = 0.5
    state = reference.adam_state(weights)
    before = weights.weight[0][0, 0]
    reference.adamw_step(p, weights, grads, state)
    reference.adamw_step(p, weights, grads, state)
    x = before * (1 - 0.1 * 0.5)  # step 1: decay, then m̂ / (√v̂ + eps) with m̂ = 0.5, v̂ = 0.25
    x -= 0.1 * 0.5 / (0.5 + 1e-3)
    m, v = 0.8 * 0.1 + 0.2 * 0.5, 0.9 * 0.025 + 0.1 * 0.25  # step 2
    x = x * (1 - 0.05) - (0.1 / (1 - 0.8**2)) * m / (np.sqrt(v) / np.sqrt(1 - 0.9**2) + 1e-3)
    assert weights.weight[0][0, 0] == pytest.approx(x, rel=1e-12)
    assert state.step == 2


def multi_target_problem(targets: int = 3) -> tuple[reference.Weights, np.ndarray, np.ndarray, list[np.ndarray]]:
    generator = np.random.default_rng(5)
    weights = reference.init(TINY, n_inputs=7, targets=targets).astype(np.float64)
    for array in weights.arrays():
        array[...] = array + generator.normal(0, 0.3, array.shape)
    x = generator.normal(size=(6, 7))
    y = generator.integers(0, 5, (6, targets))
    masks = reference.dropout_masks(TINY, step=4, rows=np.arange(100, 106), dtype=np.float64)
    return weights, x, y, masks


def test_multi_target_gradients_match_finite_differences() -> None:
    weights, x, y, masks = multi_target_problem()
    logits, cache = reference.forward(weights, x, masks)
    assert logits.shape == (6, 3, 15)
    grads = reference.backward(weights, cache, reference.loss_and_logit_grad(logits, y)[1])
    h = 1e-6
    for array, grad in zip(weights.arrays(), grads.arrays(), strict=True):
        numeric = np.empty_like(array)
        for index in np.ndindex(array.shape):
            keep = array[index]
            array[index] = keep + h
            up = loss(weights, x, y, masks)
            array[index] = keep - h
            down = loss(weights, x, y, masks)
            array[index] = keep
            numeric[index] = (up - down) / (2 * h)
        assert np.abs(grad - numeric).max() / max(np.abs(numeric).max(), 1e-12) < 1e-6


def test_the_multi_target_loss_is_the_mean_of_the_targets_losses() -> None:
    weights, x, y, masks = multi_target_problem(targets=4)
    logits, _ = reference.forward(weights, x, masks)
    total, grad = reference.loss_and_logit_grad(logits, y)
    parts = [reference.loss_and_logit_grad(logits[..., 5 * t : 5 * t + 5], y[:, t]) for t in range(4)]
    assert total == pytest.approx(np.mean([p[0] for p in parts]), rel=1e-14)
    np.testing.assert_allclose(grad, np.concatenate([p[1] for p in parts], axis=-1) / 4, rtol=1e-14)
    pred = reference.predict(logits, targets=4)
    assert pred.shape == (6, 4)
    np.testing.assert_allclose(pred, np.stack([reference.predict(logits[..., 5 * t : 5 * t + 5]) for t in range(4)],
                                              axis=1), rtol=1e-14)  # fmt: skip


def test_one_target_as_a_matrix_is_the_single_target_bits() -> None:
    weights, x, y, masks = tiny_problem()
    logits, _ = reference.forward(weights, x, masks)
    single, matrix = reference.loss_and_logit_grad(logits, y), reference.loss_and_logit_grad(logits, y[:, None])
    assert single[0] == matrix[0]
    np.testing.assert_array_equal(single[1], matrix[1])
    np.testing.assert_array_equal(reference.predict(logits), reference.predict(logits, targets=1)[:, 0])
    np.testing.assert_array_equal(reference.label_matrix(y, 6, 1, 5), y[:, None])


def test_init_grows_only_the_head_with_the_targets() -> None:
    one, three = reference.init(TINY, 7), reference.init(TINY, 7, targets=3)
    assert (three.head.shape, three.head_bias.shape) == ((3, 5, 15), (3, 15))
    for a, b in zip(one.arrays()[:-2], three.arrays()[:-2], strict=True):
        np.testing.assert_array_equal(a, b)
    assert all(np.array_equal(a, b) for a, b in zip(one.arrays(), reference.init(TINY, 7, 1).arrays(), strict=True))
    assert np.abs(three.head).max() <= 1 / np.sqrt(5)


def test_bad_targets_and_labels_raise() -> None:
    with pytest.raises(ValueError, match="positive int"):
        reference.init(TINY, 7, targets=0)
    y = np.zeros((6, 2), np.int64)
    with pytest.raises(ValueError, match=r"shape \(6, 3\), got \(6, 2\)"):
        reference.label_matrix(y, 6, 3, 5)
    with pytest.raises(ValueError, match=r"or \(6,\)"):
        reference.label_matrix(np.zeros(5, np.int64), 6, 1, 5)
    with pytest.raises(ValueError, match="integer classes"):
        reference.label_matrix(np.full((6, 2), 5), 6, 2, 5)
    with pytest.raises(ValueError, match="integer classes"):
        reference.label_matrix(np.zeros((6, 2)), 6, 2, 5)
    with pytest.raises(ValueError, match="do not fit"):
        reference.loss_and_logit_grad(np.zeros((6, 3, 10)), np.zeros((6, 3), np.int64))
    with pytest.raises(ValueError, match="do not fit"):
        reference.loss_and_logit_grad(np.zeros((6, 3, 10)), np.zeros((5, 2), np.int64))
    with pytest.raises(ValueError, match="not 3 targets"):
        reference.predict(np.zeros((6, 3, 10)), targets=3)


def test_the_mse_logit_gradient_matches_finite_differences() -> None:
    generator = np.random.default_rng(5)
    logits = generator.normal(size=(6, 3, 2))
    y = generator.normal(size=(6, 2))
    loss, grad = reference.loss_and_logit_grad(logits, y, "mse")
    np.testing.assert_allclose(loss, ((logits - y[:, None, :]) ** 2).mean())
    h, numeric = 1e-6, np.zeros_like(logits)
    for index in np.ndindex(logits.shape):
        up, down = logits.copy(), logits.copy()
        up[index] += h
        down[index] -= h
        numeric[index] = (
            reference.loss_and_logit_grad(up, y, "mse")[0] - reference.loss_and_logit_grad(down, y, "mse")[0]
        ) / (2 * h)
    np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(reference.predict(logits[:, :, :1], None, "mse"), logits[:, :, 0].mean(axis=1))
    np.testing.assert_allclose(reference.predict(logits, 2, "mse"), logits.mean(axis=1))
    with pytest.raises(ValueError, match="do not fit"):
        reference.loss_and_logit_grad(logits, y[:, :1], "mse")


def test_the_hl_gauss_loss_and_gradient_match_finite_differences() -> None:
    generator = np.random.default_rng(6)
    logits = generator.normal(size=(5, 3, 16))
    soft = hl_gauss.labels(generator.normal(size=5), 16, -2.0, 2.0, 0.2)
    loss, grad = reference.loss_and_logit_grad(logits, soft, "hl_gauss")
    log_p = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(loss, -(soft[:, None, :] * log_p).sum(axis=-1).mean())
    h, numeric = 1e-6, np.zeros_like(logits)
    for index in np.ndindex(logits.shape):
        up, down = logits.copy(), logits.copy()
        up[index] += h
        down[index] -= h
        numeric[index] = (
            reference.loss_and_logit_grad(up, soft, "hl_gauss")[0]
            - reference.loss_and_logit_grad(down, soft, "hl_gauss")[0]
        ) / (2 * h)
    np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-10)
    with pytest.raises(ValueError, match="do not fit"):
        reference.loss_and_logit_grad(logits, soft[:, :8], "hl_gauss")


def test_the_hl_gauss_prediction_is_the_expected_bin_centre() -> None:
    generator = np.random.default_rng(7)
    logits = generator.normal(size=(4, 3, 8))
    centres = hl_gauss.centres(8, -4.0, 4.0)
    p = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(reference.predict(logits, None, "hl_gauss", centres), p.mean(axis=1) @ centres)
    sharp = np.full((1, 2, 8), -50.0)
    sharp[0, :, 6] = 50.0
    assert reference.predict(sharp, None, "hl_gauss", centres)[0] == pytest.approx(centres[6])
    with pytest.raises(ValueError, match="one target"):
        reference.predict(logits, None, "hl_gauss", None)
    with pytest.raises(ValueError, match="one target"):
        reference.predict(logits, 2, "hl_gauss", centres)


def test_an_hl_gauss_head_has_n_bins_outputs() -> None:
    params = TabMParams(
        members=3, blocks=1, width=4, n_classes=1, objective="hl_gauss", n_bins=16, batch_size=4096, epochs=10
    )
    weights = reference.init(params, n_inputs=5)
    assert (weights.head.shape, weights.head_bias.shape) == ((3, 4, 16), (3, 16))


def test_a_host_loss_is_checked() -> None:
    outputs = np.random.default_rng(8).normal(size=(4, 3, 2)).astype(np.float32)
    rows = np.arange(10, 14)
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def half_square(out: np.ndarray, index: np.ndarray) -> tuple[float, np.ndarray]:
        seen.append((out, index))
        return np.float64(0.5 * (out.astype(np.float64) ** 2).sum()), out.astype(np.float64)

    value, gradient = reference.host_loss(half_square, outputs, rows)
    assert seen[0][0] is outputs
    assert seen[0][1] is rows
    assert type(value) is float
    assert value == pytest.approx(0.5 * (outputs.astype(np.float64) ** 2).sum())
    assert gradient.dtype == np.float32  # the outputs' dtype
    np.testing.assert_array_equal(gradient, outputs)
    with pytest.raises(ValueError, match=r"gradient of shape \(4, 3\) for outputs of shape \(4, 3, 2\)"):
        reference.host_loss(lambda out, index: (0.0, out[:, :, 0]), outputs, rows)
    with pytest.raises(ValueError, match="not finite"):
        reference.host_loss(lambda out, index: (float("nan"), out), outputs, rows)
    with pytest.raises(ValueError, match="not finite"):
        reference.host_loss(lambda out, index: (0.0, np.full_like(out, np.inf)), np.ones((2, 1, 1)), rows[:2])


@pytest.mark.parametrize("objective", ["cross_entropy", "mse", "hl_gauss"])
def test_weighted_losses_and_their_gradients(objective: str) -> None:
    generator = np.random.default_rng(9)
    b, k = 6, 3
    if objective == "cross_entropy":
        logits, y = generator.normal(size=(b, k, 2 * 4)), generator.integers(0, 4, (b, 2))
    elif objective == "mse":
        logits, y = generator.normal(size=(b, k, 2)), generator.normal(size=(b, 2))
    else:
        logits = generator.normal(size=(b, k, 8))
        y = generator.dirichlet(np.ones(8), b)
    w = np.array([0.0, 1.0, 2.5, 0.5, 1.0, 3.0])

    def loss(at: np.ndarray) -> float:
        return reference.loss_and_logit_grad(at, y, objective, w)[0]

    value, grad = reference.loss_and_logit_grad(logits, y, objective, w)
    numeric, h = np.zeros_like(logits), 1e-6
    for index in np.ndindex(logits.shape):
        up, down = logits.copy(), logits.copy()
        up[index] += h
        down[index] -= h
        numeric[index] = (loss(up) - loss(down)) / (2 * h)
    np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-9)
    plain = reference.loss_and_logit_grad(logits, y, objective)
    ones = reference.loss_and_logit_grad(logits, y, objective, np.ones(b))
    np.testing.assert_allclose(ones[0], plain[0], rtol=1e-12)
    np.testing.assert_allclose(ones[1], plain[1], rtol=1e-12, atol=1e-15)
    kept = w > 0  # a row of weight 0 is a row left out: the others' weighted mean
    without = reference.loss_and_logit_grad(logits[kept], y[kept], objective, w[kept])
    np.testing.assert_allclose(value, without[0], rtol=1e-12)
    np.testing.assert_allclose(grad[kept], without[1], rtol=1e-12, atol=1e-15)
    assert not grad[~kept].any()
    nothing = reference.loss_and_logit_grad(logits, y, objective, np.zeros(b))  # a batch of weight 0
    assert nothing[0] == 0.0
    assert not nothing[1].any()


PWL = TabMParams(members=3, blocks=2, width=5, dropout=0.25, n_classes=5, seed=11, num_emb_type="pwl",
                 num_emb_n_bins=4, d_embedding=3, batch_size=4096, epochs=10)  # fmt: skip


def embedded_problem() -> tuple[reference.Weights, np.ndarray, np.ndarray, list[np.ndarray]]:
    generator = np.random.default_rng(3)
    x = generator.normal(size=(12, 7))
    bins = emb.fit_bins(x[:, :5], PWL.num_emb_n_bins)  # five embedded columns, two passed as they are
    weights = reference.init(PWL, n_inputs=7, bins=bins).astype(np.float64)
    for array in weights.arrays():
        array[...] = array + generator.normal(0, 0.3, array.shape)
    masks = reference.dropout_masks(PWL, step=4, rows=np.arange(100, 112), dtype=np.float64)
    return weights, x, generator.integers(0, 5, 12), masks


def test_embedded_inputs_and_their_gradients() -> None:
    weights, x, y, masks = embedded_problem()
    assert weights.weight[0].shape == (5, 5 * 3 + 2)
    assert weights.embedding is not None
    assert weights.bins is not None
    logits, cache = reference.forward(weights, x, masks)
    assert cache.h[0].shape == (12, 3, 17)
    embedded = emb.forward(weights.embedding, weights.bins, x[:, :5])[0]
    np.testing.assert_array_equal(cache.h[0][:, 0], np.concatenate([embedded, x[:, 5:]], axis=1))
    grads = reference.backward(weights, cache, reference.loss_and_logit_grad(logits, y)[1])
    assert len(grads.arrays()) == len(weights.arrays())
    h = 1e-6
    for array, grad in zip(weights.arrays(), grads.arrays(), strict=True):
        numeric = np.empty_like(array)
        for index in np.ndindex(array.shape):
            keep = array[index]
            array[index] = keep + h
            up = loss(weights, x, y, masks)
            array[index] = keep - h
            down = loss(weights, x, y, masks)
            array[index] = keep
            numeric[index] = (up - down) / (2 * h)
        scale = max(np.abs(numeric).max(), 1e-12)
        assert np.abs(grad - numeric).max() / scale < 1e-6


def test_embeddings_init_as_pytabkits_tabm() -> None:
    bins = emb.fit_bins(np.random.default_rng(1).normal(size=(50, 4)), 8)
    weights = reference.init(PWL, n_inputs=6, bins=bins)
    first = weights.r[0]
    assert first.shape == (3, 4 * 3 + 2)
    for f in range(4):  # one N(0, 1) draw per member and column, shared by the column's embedded values
        assert (first[:, 3 * f : 3 * f + 3] == first[:, 3 * f : 3 * f + 1]).all()
    assert len(np.unique(first)) == 3 * 6
    assert not np.isin(first, [-1.0, 1.0]).all()  # no random signs
    assert weights.embedding is not None
    assert not weights.embedding.ple.any()
    with pytest.raises(ValueError, match="bins"):
        reference.init(PWL, n_inputs=6)
    with pytest.raises(ValueError, match="bins"):
        reference.init(TINY, n_inputs=6, bins=bins)
    with pytest.raises(ValueError, match="columns of"):
        reference.init(PWL, n_inputs=3, bins=bins)
    for bad in ({"encoder": "gru"}, {"storage": "bf16"}):
        with pytest.raises(ValueError, match="num_emb_type"):
            TabMParams(num_emb_type="pwl", **bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_emb_n_bins"):
        TabMParams(n_classes=5, num_emb_type="pwl", num_emb_n_bins=1, batch_size=4096, epochs=10)


def test_gradient_clipping_is_torchs_clip_grad_norm() -> None:
    assert reference.clip_scale(1.0, [9.0, 16.0]) == pytest.approx(1.0 / (5.0 + 1e-6))  # ‖g‖ = 5
    assert reference.clip_scale(10.0, [9.0, 16.0]) == 1.0  # under the norm: unscaled
    weights, x, y, masks = tiny_problem()
    logits, cache = reference.forward(weights, x, masks)
    grads = reference.backward(weights, cache, reference.loss_and_logit_grad(logits, y)[1])
    norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.arrays())))
    params = dataclasses.replace(TINY, gradient_clipping_norm=norm / 4)
    got, want = weights.astype(np.float64), weights.astype(np.float64)
    reference.adamw_step(params, got, grads, reference.adam_state(got))
    scale = (norm / 4) / (norm + 1e-6)
    weight, r, s, bias = ([g * scale for g in group] for group in (grads.weight, grads.r, grads.s, grads.bias))
    scaled = reference.Weights(weight, r, s, bias, grads.head * scale, grads.head_bias * scale)
    reference.adamw_step(TINY, want, scaled, reference.adam_state(want))
    for a, b in zip(got.arrays(), want.arrays(), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12)
    with pytest.raises(ValueError, match="gradient_clipping_norm"):
        dataclasses.replace(TINY, gradient_clipping_norm=-1.0)
