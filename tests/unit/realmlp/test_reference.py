import dataclasses
import math
from typing import Any

import numpy as np
import pytest

from idun import rng
from idun.realmlp import reference as ref
from idun.realmlp.params import RealMLPParams

TINY = RealMLPParams(
    n_classes=5, blocks=2, hidden=5, frequencies=3, embed=2, dropout=0.2, seed=4, epochs=5, init_rows=3986
)
OWN = dataclasses.replace(TINY, own_last_block=True)


def tiny_problem(n_targets: int | None = None, params: RealMLPParams = TINY
                 ) -> tuple[ref.Weights, np.ndarray, np.ndarray, list[np.ndarray]]:  # fmt: skip
    """A tiny network off its initial pattern, with one target's labels (40,) or K targets' (40, K)."""
    generator = np.random.default_rng(1)
    x = generator.normal(size=(40, 4))
    pre = ref.fit_preprocess(x)
    c = ref.preprocess(params, pre, x)
    weights = ref.init(params, c, n_targets or 1)
    for array in weights.arrays():  # off the initial pattern (scale and act weights 1)
        array[...] = array + generator.normal(0, 0.2, array.shape)
    y = generator.integers(0, 5, 40 if n_targets is None else (40, n_targets))
    masks = ref.dropout_masks(params, step=3, t=0.6, rows=np.arange(40), dtype=np.float64)
    return weights, c, y, masks


# one target, the mean loss of three heads, three heads with their own last blocks; Mish and ReLU in place of SELU
@pytest.mark.parametrize(("n_targets", "params"), [(None, TINY), (3, TINY), (3, OWN),
                                                   (None, dataclasses.replace(TINY, act="mish")),
                                                   (3, dataclasses.replace(OWN, act="relu")),
                                                   (None, dataclasses.replace(TINY, blocks=3, hidden=256,
                                                                              hidden_sizes=(6, 3, 4))),
                                                   (3, dataclasses.replace(OWN, blocks=3, hidden=256,
                                                                           hidden_sizes=(4, 7)))])  # fmt: skip
def test_gradients_match_finite_differences(n_targets: int | None, params: RealMLPParams) -> None:
    weights, c, y, masks = tiny_problem(n_targets, params)

    def loss() -> float:
        return ref.loss_and_logit_grad(TINY, ref.forward(weights, c, masks, params.act)[0], y)[0]

    logits, cache = ref.forward(weights, c, masks, params.act)
    grads = ref.backward(weights, cache, ref.loss_and_logit_grad(TINY, logits, y)[1], params.act)
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
        assert np.abs(grad - numeric).max() / max(np.abs(numeric).max(), 1e-12) < 1e-6


def test_preprocessing_is_pytabkits() -> None:
    x = np.array([[0.0, 5.0, 1.0, 7.0], [1.0, 5.0, 0.0, 7.0], [2.0, 5.0, 0.0, 7.0], [3.0, 9.0, 0.0, 7.0],
                  [4.0, 5.0, 0.0, 7.0]])  # fmt: skip
    pre = ref.fit_preprocess(x)
    np.testing.assert_array_equal(pre.median, [2.0, 5.0, 0.0, 7.0])
    # column 0: IQR 2; column 1: IQR 0, range 4 -> 1 / 2; column 2: IQR 0, range 1 -> 2; column 3 constant -> 0
    np.testing.assert_allclose(pre.factor, [0.5, 0.5, 2.0, 0.0])
    c = ref.preprocess(TINY, pre, x)
    v = (x - pre.median) * pre.factor
    np.testing.assert_allclose(c, v / np.sqrt(1 + v * v / 9))
    assert ref.preprocess(TINY, pre, np.array([[1e9, 0, 0, 0]]))[0, 0] == pytest.approx(3.0)  # the clip's limit


def test_schedules_are_coslog4_and_flat_cos() -> None:
    for t in (0.0, 1 / 15, 3 / 15, 7 / 15):
        assert ref.schedules(t)[0] == pytest.approx(0.0, abs=1e-12)  # coslog4's zeros: the first step has lr 0
    assert ref.schedules(1.0)[0] == pytest.approx(0.0, abs=1e-12)
    assert ref.schedules(2 / 15)[0] == pytest.approx(0.5 * (1 - math.cos(2 * math.pi * math.log2(3))))
    assert ref.schedules(0.3)[1] == 1.0
    assert ref.schedules(0.75)[1] == pytest.approx(0.5)
    assert ref.schedules(1.0)[1] == pytest.approx(0.0, abs=1e-12)


def test_init_normalises_each_layer_on_the_init_rows() -> None:
    generator = np.random.default_rng(2)
    c = ref.preprocess(TINY, ref.fit_preprocess(x := generator.normal(size=(200, 4))), x)
    w = ref.init(TINY, c)
    assert w.w1.shape == w.b1.shape == (4, 3)
    assert w.w2.shape == (4, 3, 2)
    assert w.b2.shape == (4, 2)
    assert w.scale.shape == (12,)
    assert [a.shape for a in w.weight] == [(12, 5), (5, 5)]
    assert w.out.shape == (5, 5)
    assert np.abs(w.b1).max() <= math.pi
    assert np.abs(w.w2).max() <= 1 / math.sqrt(3)
    assert np.abs(w.b2).max() <= 0.25
    h = ref.embed(w, c)[0]
    for i, weight in enumerate(w.weight):
        pre = h @ weight / math.sqrt(weight.shape[0])
        np.testing.assert_allclose(pre.std(axis=0), 1.0)  # unit population std per column
        assert ((-w.bias[i] >= pre.min(axis=0)) & (-w.bias[i] <= pre.max(axis=0))).all()  # minus a convex combination
        h = ref.selu(pre + w.bias[i])
    again = ref.init(TINY, c)
    assert all(np.array_equal(a, b) for a, b in zip(w.arrays(), again.arrays(), strict=True))
    heads = ref.init(TINY, c, n_targets=3)  # the same trunk; the output layer 3 · 5 columns of unit std
    assert all(np.array_equal(a, b) for a, b in zip(w.arrays()[:-2], heads.arrays()[:-2], strict=True))
    assert heads.out.shape == (5, 15)
    assert heads.out_bias.shape == (15,)
    np.testing.assert_allclose((h @ heads.out / math.sqrt(5)).std(axis=0), 1.0)
    with pytest.raises(ValueError, match="n_targets"):
        ref.init(TINY, c, n_targets=0)


def test_an_own_last_block_starts_as_k_copies_of_the_single_target_networks() -> None:
    generator = np.random.default_rng(2)
    c = ref.preprocess(TINY, ref.fit_preprocess(x := generator.normal(size=(200, 4))), x)
    single, own = ref.init(TINY, c), ref.init(OWN, c, n_targets=3)
    assert (ref.own_targets(single), ref.own_targets(own)) == (1, 3)
    assert [a.shape for a in own.weight] == [(12, 5), (5, 15)]
    assert (own.bias[-1].shape, own.act[-1].shape, own.out.shape, own.out_bias.shape) == ((15,), (15,), (5, 15), (15,))

    def trunk(w: ref.Weights) -> list[np.ndarray]:
        return [*w.arrays()[:5], *w.weight[:-1], *w.bias[:-1], *w.act[:-1]]

    assert all(np.array_equal(a, b) for a, b in zip(trunk(single), trunk(own), strict=True))
    np.testing.assert_array_equal(own.weight[-1], np.tile(single.weight[-1], (1, 3)))  # as 3 networks of one seed
    np.testing.assert_array_equal(own.bias[-1], np.tile(single.bias[-1], 3))
    np.testing.assert_array_equal(own.out, np.tile(single.out, (1, 3)))
    np.testing.assert_array_equal(own.out_bias, np.tile(single.out_bias, 3))
    one = ref.init(OWN, c, n_targets=1)  # K = 1 is the single-target network
    assert all(np.array_equal(a, b) for a, b in zip(one.arrays(), single.arrays(), strict=True))


def test_an_own_last_block_gives_each_target_a_single_target_networks_top() -> None:
    weights, c, y, masks = tiny_problem(3, OWN)  # off the initial pattern: the three blocks differ
    logits, cache = ref.forward(weights, c, masks)
    assert logits.shape == (40, 15)
    h = TINY.hidden
    for k in range(3):  # target k: the shared blocks, then its own block and output layer
        mine = slice(k * h, (k + 1) * h)
        single = ref.Weights(weights.w1, weights.b1, weights.w2, weights.b2, weights.scale,
                             [weights.weight[0], weights.weight[1][:, mine]],
                             [weights.bias[0], weights.bias[1][mine]], [weights.act[0], weights.act[1][mine]],
                             weights.out[:, 5 * k : 5 * k + 5], weights.out_bias[5 * k : 5 * k + 5])  # fmt: skip
        np.testing.assert_allclose(logits[:, 5 * k : 5 * k + 5], ref.forward(single, c, masks)[0], rtol=1e-13)
    d_logits = ref.loss_and_logit_grad(OWN, logits, y)[1]
    d_logits[:, 5:] = 0.0  # target 0's loss alone moves no other target's own block
    grads = ref.backward(weights, cache, d_logits)
    assert not grads.weight[1][:, h:].any()
    assert not grads.act[1][h:].any()
    assert not grads.out[:, 5:].any()
    assert grads.weight[1][:, :h].any()
    assert grads.weight[0].any()


def test_dropout_follows_the_schedule_and_the_absolute_row() -> None:
    params = RealMLPParams(n_classes=5, hidden=64, dropout=0.15, seed=1, epochs=5, init_rows=3986)
    rows = np.arange(3000, 4000)
    early = ref.dropout_masks(params, step=5, t=0.2, rows=rows, dtype=np.float32)
    assert abs((early[0] == 0).mean() - 0.15) < 0.01
    assert set(np.unique(early[0])) == {np.float32(0.0), np.float32(1 / 0.85)}
    late = ref.dropout_masks(params, step=5, t=1.0, rows=rows, dtype=np.float32)
    assert (late[0] == 1.0).all()  # flat_cos(1) = 0: no dropout at the end
    np.testing.assert_array_equal(ref.dropout_masks(params, 5, 0.2, rows[::-1].copy(), np.float32)[1], early[1][::-1])
    bits = rng.element_bits(rng.derive_key(1, ref.DROPOUT_SALT, 5, 1), rows[:1], 64)
    np.testing.assert_array_equal(early[1][0] != 0, bits[0] >= round(0.15 * 2**24))


def test_label_smoothing_and_prediction() -> None:
    logits = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0, 0.0]])
    loss, grad = ref.loss_and_logit_grad(TINY, logits, np.array([2, 0]))
    target = np.full((2, 5), 0.02)
    target[0, 2] = target[1, 0] = 0.92
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert loss == pytest.approx(-(np.log(p) * target).sum(axis=1).mean())
    np.testing.assert_allclose(grad, (p - target) / 2)
    np.testing.assert_allclose(ref.predict(logits), [2.0, (p[1] * np.arange(5)).sum()])


def test_k_targets_are_the_mean_of_k_single_target_heads() -> None:
    generator = np.random.default_rng(3)
    logits = generator.normal(size=(6, 3 * 5))
    y = generator.integers(0, 5, (6, 3))
    loss, grad = ref.loss_and_logit_grad(TINY, logits, y)
    singles = [ref.loss_and_logit_grad(TINY, logits[:, 5 * k : 5 * k + 5], y[:, k]) for k in range(3)]
    assert loss == pytest.approx(np.mean([s[0] for s in singles]), rel=1e-14)
    np.testing.assert_allclose(grad, np.concatenate([s[1] / 3 for s in singles], axis=1), rtol=1e-14)  # 1 / (B · K)
    pred = ref.predict(logits, 3)
    assert pred.shape == (6, 3)
    np.testing.assert_allclose(pred, np.stack([ref.predict(logits[:, 5 * k : 5 * k + 5]) for k in range(3)], axis=1))
    # one target given as (B, 1) is the single-target path bit for bit
    one, single = ref.loss_and_logit_grad(TINY, logits[:, :5], y[:, :1]), ref.loss_and_logit_grad(TINY, logits[:, :5],
                                                                                                   y[:, 0])  # fmt: skip
    assert one[0] == single[0]
    np.testing.assert_array_equal(one[1], single[1])
    np.testing.assert_array_equal(ref.predict(logits[:, :5], 1)[:, 0], ref.predict(logits[:, :5]))
    with pytest.raises(ValueError, match="multiple of 3"):
        ref.loss_and_logit_grad(TINY, logits[:, :14], y)
    with pytest.raises(ValueError, match="multiple of 4"):
        ref.predict(logits, 4)


def test_adam_step_decays_with_squared_factors_then_steps() -> None:
    params = RealMLPParams(n_classes=5, blocks=1, hidden=2, frequencies=1, embed=1, learning_rate=0.1, weight_decay=0.5,
                           beta1=0.8, beta2=0.9, eps=1e-3, epochs=5, init_rows=3986)  # fmt: skip
    weights = ref.Weights(np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1, 1)), np.ones((1, 1)), np.ones(2),
                          [np.ones((2, 2))], [np.ones(2)], [np.ones(2)], np.ones((2, 5)), np.ones(5))  # fmt: skip
    grads = weights.astype(np.float64)
    for g in grads.arrays():
        g[...] = 0.5
    state = ref.adam_state(weights)
    t = 2 / 15 + 0.01
    lr_mult, wd_mult = ref.schedules(t)
    ref.adam_step(params, weights, grads, state, t)
    lr, wd = 0.1 * lr_mult, 0.5 * wd_mult
    step = 0.5 / (0.5 + 1e-3)  # m̂ / (√v̂ + eps) on the first step
    assert weights.weight[0][0, 0] == pytest.approx((1 - wd * lr) - lr * step)
    assert weights.scale[0] == pytest.approx((1 - wd * lr * 36) - lr * 6 * step)  # front scale: f_lr 6, decayed 36x
    assert weights.w1[0, 0] == pytest.approx((1 - wd * lr * 0.01) - lr * 0.1 * step)  # embedding: f_lr 0.1
    assert weights.bias[0][0] == pytest.approx(1 - lr * 0.1 * step)  # biases: not decayed
    assert weights.act[0][0] == pytest.approx((1 - wd * lr * 0.01) - lr * 0.1 * step)
    assert len(ref.factors(params, 1)) == len(weights.arrays())


def test_the_mse_logit_gradient_matches_finite_differences() -> None:
    params = RealMLPParams(n_classes=1, objective="mse", epochs=5, init_rows=3986)
    generator = np.random.default_rng(4)
    logits, y = generator.normal(size=(7, 2)), generator.normal(size=(7, 2))
    loss, grad = ref.loss_and_logit_grad(params, logits, y)
    np.testing.assert_allclose(loss, ((logits - y) ** 2).mean())
    h, numeric = 1e-6, np.zeros_like(logits)
    for index in np.ndindex(logits.shape):
        up, down = logits.copy(), logits.copy()
        up[index] += h
        down[index] -= h
        numeric[index] = (ref.loss_and_logit_grad(params, up, y)[0] - ref.loss_and_logit_grad(params, down, y)[0]) / (
            2 * h
        )
    np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(ref.predict(logits[:, :1], None, "mse"), logits[:, 0])
    with pytest.raises(ValueError, match="do not fit"):
        ref.loss_and_logit_grad(params, logits, y[:, :1])


def test_the_schedules_as_parameters() -> None:
    plain = RealMLPParams(n_classes=5, epochs=5, init_rows=3986)
    for t in (0.0, 0.3, 0.7, 1.0):
        coslog4, flat_cos = ref.schedules(t)
        assert ref.multipliers(plain, t) == (coslog4, flat_cos, flat_cos)  # the defaults: pytabkit's
    other = dataclasses.replace(plain, lr_schedule="cosine", dropout_schedule="constant", wd_schedule="constant")
    lr, drop, wd = ref.multipliers(other, 0.5)
    assert lr == pytest.approx(0.5)
    assert (drop, wd) == (1.0, 1.0)
    assert ref.multipliers(dataclasses.replace(plain, lr_schedule="constant"), 0.9)[0] == 1.0
    for name in ("lr_schedule", "dropout_schedule", "wd_schedule"):
        bad: dict[str, Any] = {name: "step"}
        with pytest.raises(ValueError, match=f"out of range: {name}"):
            dataclasses.replace(plain, **bad)


def test_the_activations_are_torchs() -> None:
    v = np.array([-50.0, -3.0, -0.5, 0.0, 0.7, 4.0, 30.0])
    softplus = np.log1p(np.exp(np.minimum(v, 30.0))) + np.maximum(v - 30.0, 0.0)
    np.testing.assert_allclose(ref.activation(v, "mish"), v * np.tanh(softplus), rtol=1e-12, atol=1e-300)
    np.testing.assert_array_equal(ref.activation(v, "relu"), np.maximum(v, 0.0))
    np.testing.assert_array_equal(ref.activation(v, "selu"), ref.selu(v))
    h = 1e-6
    for act in ("mish", "selu"):
        numeric = (ref.activation(v + h, act) - ref.activation(v - h, act)) / (2 * h)
        np.testing.assert_allclose(ref.activation_grad(v, act)[v != 0], numeric[v != 0], rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(ref.activation_grad(v, "relu"), (v > 0).astype(float))


@pytest.mark.parametrize("objective", ["cross_entropy", "mse"])
def test_weighted_losses_and_their_gradients(objective: str) -> None:
    generator = np.random.default_rng(9)
    b = 6
    y: np.ndarray
    if objective == "mse":
        params = RealMLPParams(n_classes=1, objective="mse", epochs=5, init_rows=3986)
        logits, y = generator.normal(size=(b, 2)), generator.normal(size=(b, 2))
    else:
        params = TINY
        logits, y = generator.normal(size=(b, 2 * TINY.n_classes)), generator.integers(0, TINY.n_classes, (b, 2))
    w = np.array([0.0, 1.0, 2.5, 0.5, 1.0, 3.0])

    def loss(at: np.ndarray) -> float:
        return ref.loss_and_logit_grad(params, at, y, w)[0]

    value, grad = ref.loss_and_logit_grad(params, logits, y, w)
    numeric, h = np.zeros_like(logits), 1e-6
    for index in np.ndindex(logits.shape):
        up, down = logits.copy(), logits.copy()
        up[index] += h
        down[index] -= h
        numeric[index] = (loss(up) - loss(down)) / (2 * h)
    np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-9)
    plain = ref.loss_and_logit_grad(params, logits, y)
    ones = ref.loss_and_logit_grad(params, logits, y, np.ones(b))
    np.testing.assert_allclose(ones[0], plain[0], rtol=1e-12)
    np.testing.assert_allclose(ones[1], plain[1], rtol=1e-12, atol=1e-15)
    kept = w > 0  # a row of weight 0 is a row left out: the others' weighted mean
    without = ref.loss_and_logit_grad(params, logits[kept], y[kept], w[kept])
    np.testing.assert_allclose(value, without[0], rtol=1e-12)
    np.testing.assert_allclose(grad[kept], without[1], rtol=1e-12, atol=1e-15)
    assert not grad[~kept].any()
    nothing = ref.loss_and_logit_grad(params, logits, y, np.zeros(b))  # a batch of weight 0
    assert nothing[0] == 0.0
    assert not nothing[1].any()
    with pytest.raises(ValueError, match="do not fit"):
        ref.loss_and_logit_grad(params, logits, y, w[:-1])


def test_categorical_embeddings_and_their_gradients() -> None:
    generator = np.random.default_rng(7)
    params = dataclasses.replace(TINY, embedding_size=3)
    numeric = generator.normal(size=(20, 4))
    codes = np.column_stack([generator.integers(0, 5, 20), generator.integers(0, 2, 20)]).astype(np.float64)
    c = np.concatenate([numeric, codes], axis=1)  # two categorical inputs of 5 and 2 levels, after four numeric
    weights = ref.init(params, c, categories=(5, 2))
    assert weights.table is not None
    assert weights.table.shape == (7, 3)
    assert weights.scale.shape == ((TINY.embed + 1) * 4 + 3 * 2,)
    z = ref.embed(weights, c)[0]
    np.testing.assert_array_equal(z[:, -6:-3], weights.table[codes[:, 0].astype(int)])
    np.testing.assert_array_equal(z[:, -3:], weights.table[5 + codes[:, 1].astype(int)])  # input 1 from row 5 on
    for array in weights.arrays():
        array[...] = array + generator.normal(0, 0.2, array.shape)
    masks = ref.dropout_masks(params, 3, 0.4, np.arange(20), np.float64)
    y = generator.integers(0, params.n_classes, 20)

    def loss() -> float:
        return ref.loss_and_logit_grad(params, ref.forward(weights, c, masks)[0], y)[0]

    logits, cache = ref.forward(weights, c, masks)
    grads = ref.backward(weights, cache, ref.loss_and_logit_grad(params, logits, y)[1])
    h = 1e-6
    for array, grad in zip(weights.arrays(), grads.arrays(), strict=True):
        numeric_grad = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            keep = array[index]
            array[index] = keep + h
            up = loss()
            array[index] = keep - h
            numeric_grad[index] = (up - loss()) / (2 * h)
            array[index] = keep
        np.testing.assert_allclose(grad, numeric_grad, rtol=1e-5, atol=1e-8)
    for bad in (np.c_[numeric, codes[:, :1], np.full(20, 2.0)], np.c_[numeric, codes[:, :1] + 0.5, codes[:, 1:]]):
        with pytest.raises(ValueError, match="level codes"):
            ref.forward(weights, bad, None)
    with pytest.raises(ValueError, match="categories"):
        ref.init(params, c, categories=(5, 0))
