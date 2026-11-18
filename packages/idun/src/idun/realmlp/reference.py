"""The NumPy reference of RealMLP-TD with pytabkit 1.7.3's defaults (docs/architecture/networks.md §8).

Preprocessing, initialisation, forward, backward, the schedules and Adam with decoupled decay. Shapes: ``B`` rows,
``F`` inputs, ``Q`` frequencies, ``E`` embedding values per input, ``H`` hidden units, ``C`` classes. The model, per
row:

    c = clip((x − median) ⊙ factor)                     clip(v) = v / sqrt(1 + v² / clip²)
    e[f] = cos(2π c[f] w1[f] + b1[f]) · W2[f] + b2[f]   (E values per input)
    z = [e (feature-major, E·F), c (F)] ⊙ scale         (the front scale)
    h ← act(h · W / sqrt(fan_in) + b) ⊙ m               per hidden block; act(v) = v + (f(v) − v) ⊙ a, f the activation
    logits = h · W_out / sqrt(H) + b_out                (C values)

The loss is the mean cross-entropy against labels smoothed by ``label_smoothing``, over the rows; the prediction is
the expected class. Several targets and ``RealMLPParams.own_last_block``: experimental, not covered by the
documentation.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun import rng
from idun.realmlp.params import RealMLPParams

Array = NDArray[np.floating]
DROPOUT_SALT = 7  # derive_key(seed, DROPOUT_SALT, step, block): the dropout masks' key
INIT_SALT = 9  # PCG64(SeedSequence([seed, INIT_SALT, part])): embedding, weights, he+5 biases
CATEGORY_SALT = 15  # PCG64(SeedSequence([seed, CATEGORY_SALT])): the categorical embeddings' tables
SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946


@dataclass
class Preprocess:
    """Per-input statistics fitted on the init rows: subtract ``median``, multiply by ``factor``."""

    median: Array
    factor: Array


def fit_preprocess(x: Array) -> Preprocess:
    """Fit the median and the robust scale on ``x`` (the init rows), as pytabkit's median_center and robust_scale.

    The scale is 1 / (q75 − q25 + 1e-30) of the centred data; an IQR of 0 falls back to (max − min) / 2, and a column
    whose range is 0 too gets the factor 0 (it is then always 0).
    """
    x = np.asarray(x, np.float64)
    median = np.quantile(x, 0.5, axis=0)
    centred = x - median
    spread = np.quantile(centred, 0.75, axis=0) - np.quantile(centred, 0.25, axis=0)
    spread = np.where(spread == 0.0, 0.5 * (centred.max(axis=0) - centred.min(axis=0)), spread)
    factor = np.where(spread == 0.0, 0.0, 1.0 / (spread + 1e-30))
    return Preprocess(median, factor)


def preprocess(params: RealMLPParams, pre: Preprocess, x: Array) -> Array:
    """Return the centred, scaled and smoothly clipped inputs."""
    v = (np.asarray(x, np.float64) - pre.median) * pre.factor
    return v / np.sqrt(1.0 + v * v / (params.clip * params.clip))


@dataclass
class Weights:
    """RealMLP's parameters.

    The embedding (``w1``, ``b1``: (F, Q); ``w2``: (F, Q, E); ``b2``: (F, E)), the front ``scale`` ((E + 1)·F + D·G),
    per hidden block ``weight`` (fan_in, H), ``bias`` (H) and ``act`` (H), and the output layer ``out`` (H, K · C),
    ``out_bias`` (K · C). With G categorical inputs of ``categories`` levels each (the last G of a row's inputs,
    their levels' codes), ``table`` (Σ categories, D) holds their embeddings, input g's levels from row
    ``Σ categories[:g]`` on; both None without (`levels`: the counts, () without).
    """

    w1: Array
    b1: Array
    w2: Array
    b2: Array
    scale: Array
    weight: list[Array]
    bias: list[Array]
    act: list[Array]
    out: Array
    out_bias: Array
    table: Array | None = None
    categories: tuple[int, ...] | None = None

    @property
    def levels(self) -> tuple[int, ...]:
        """The categorical inputs' level counts, () without categorical inputs."""
        return self.categories or ()

    def arrays(self) -> list[Array]:
        """Return every parameter array in a fixed order (the order of `factors`): the table last, if any."""
        tail = [] if self.table is None else [self.table]
        return [self.w1, self.b1, self.w2, self.b2, self.scale, *self.weight, *self.bias, *self.act, self.out,
                self.out_bias, *tail]  # fmt: skip

    def astype(self, dtype: type[np.floating]) -> "Weights":
        """Return a copy in ``dtype``."""

        def cast(arrays: Sequence[Array]) -> list[Array]:
            return [a.astype(dtype) for a in arrays]

        w1, b1, w2, b2, scale = cast([self.w1, self.b1, self.w2, self.b2, self.scale])
        table = None if self.table is None else self.table.astype(dtype)
        return Weights(w1, b1, w2, b2, scale, cast(self.weight), cast(self.bias), cast(self.act),
                       self.out.astype(dtype), self.out_bias.astype(dtype), table, self.categories)  # fmt: skip


def own_targets(weights: Weights) -> int:
    """Return K for weights whose last hidden block is each of K targets' own, else 1 (that block is shared)."""
    return int(weights.act[-1].size // weights.out.shape[0])


def factors(params: RealMLPParams, blocks: int, *, table: bool = False) -> list[tuple[float, float]]:
    """Return (learning-rate factor, weight-decay factor) of each array of `Weights.arrays`.

    ``table``: the weights have the categorical embeddings' table (pytabkit's factors 1 and 1).
    """
    embed = (params.embed_lr_factor, 1.0)
    return ([embed] * 4 + [(params.scale_lr_factor, 1.0)] + [(1.0, 1.0)] * blocks
            + [(params.bias_lr_factor, 0.0)] * blocks + [(params.act_lr_factor, 1.0)] * blocks
            + [(1.0, 1.0), (params.bias_lr_factor, 0.0)] + [(1.0, 1.0)] * table)  # fmt: skip


def split_inputs(weights: Weights, c: Array) -> tuple[Array, NDArray[np.int64]]:
    """Return a batch's numeric inputs and its categorical inputs' level codes: its last G columns, G of ``weights``."""
    g = len(weights.levels)
    codes = np.asarray(c[:, c.shape[1] - g :], np.float64)
    numbers = codes.astype(np.int64)
    if not np.array_equal(numbers, codes) or ((numbers < 0) | (numbers >= np.array(weights.levels))).any():
        raise ValueError(f"the last {g} inputs must be the categorical inputs' level codes, 0 .. levels - 1")
    return c[:, : c.shape[1] - g], numbers


def table_rows(weights: Weights, codes: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return each code's row of ``weights.table``: input g's levels start at Σ categories[:g]."""
    starts = np.concatenate([[0], np.cumsum(weights.levels)[:-1]]).astype(np.int64)
    return codes + starts


def _draw(params: RealMLPParams, part: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([params.seed, INIT_SALT, part])))


def embed(weights: Weights, c: Array) -> tuple[Array, Array]:
    """Return the embedded inputs z ((E + 1)·F + D·G per row, before the front scale) and the cosine arguments.

    ``c``: the preprocessed numeric inputs, then the categorical inputs' codes (the last G columns).
    """
    numeric, codes = split_inputs(weights, c)
    arg = 2.0 * math.pi * numeric[:, :, None] * weights.w1 + weights.b1  # (B, F, Q)
    e = np.einsum("bfq,fqe->bfe", np.cos(arg), weights.w2) + weights.b2  # (B, F, E)
    parts = [e.reshape(len(c), -1), numeric]
    if weights.table is not None:
        parts.append(weights.table[table_rows(weights, codes)].reshape(len(c), -1))
    return np.concatenate(parts, axis=1), arg


def selu(v: Array) -> Array:
    """torch.selu."""
    return SELU_SCALE * np.where(v > 0, v, SELU_ALPHA * np.expm1(np.minimum(v, 0.0)))


def selu_grad(v: Array) -> Array:
    """Return the derivative of `selu`."""
    return SELU_SCALE * np.where(v > 0, 1.0, SELU_ALPHA * np.exp(np.minimum(v, 0.0)))


def mish(v: Array) -> Array:
    """Return torch's mish of ``v``: ``v · tanh(softplus(v))``."""
    return v * np.tanh(np.logaddexp(0.0, v))


def mish_grad(v: Array) -> Array:
    """Return the derivative of `mish`: ``tanh(sp) + v · (1 − tanh(sp)²) · sigmoid(v)``, ``sp = softplus(v)``."""
    t = np.tanh(np.logaddexp(0.0, v))
    return t + v * (1.0 - t * t) / (1.0 + np.exp(-v))


def activation(v: Array, act: str) -> Array:
    """Return the hidden layers' activation (``RealMLPParams.act``) of ``v``."""
    if act == "mish":
        return mish(v)
    if act == "relu":
        return np.maximum(v, 0.0)
    return selu(v)


def activation_grad(v: Array, act: str) -> Array:
    """Return the derivative of `activation`."""
    if act == "mish":
        return mish_grad(v)
    if act == "relu":
        return (v > 0).astype(v.dtype)
    return selu_grad(v)


def init(params: RealMLPParams, c: Array, n_targets: int = 1, categories: Sequence[int] = ()) -> Weights:
    """Draw the weights of a network of ``n_targets`` targets; ``c`` are the preprocessed init rows.

    ``categories``: the levels of each categorical input (the last G columns of ``c``, their codes), embedded in
    ``params.embedding_size`` values each, the table N(0, 1) as pytabkit's.

    Embedding: w1 ~ embed_sigma · N(0, 1), b1 ~ U(−π, π), w2 ~ U(−1/√Q, 1/√Q), b2 ~ U(−¼, ¼). Each weight matrix
    ~ N(0, 1), rescaled so every output column of ``h · W / sqrt(fan_in)`` has unit population std on the init rows;
    each bias "he+5": minus a random convex combination (Exp(1) weights) of that column's values on 5 random init rows.
    The output layer has ``n_classes`` columns (``n_targets`` other than 1: experimental, not covered by the
    documentation).
    """
    if n_targets < 1:
        raise ValueError(f"n_targets must be an integer of at least 1, got {n_targets!r}")
    if any(m < 1 for m in categories) or len(categories) >= c.shape[1]:
        raise ValueError(f"categories must be positive level counts of the last of {c.shape[1]} inputs, at least one "
                         f"numeric input before them; got {list(categories)}")  # fmt: skip
    n, f = c.shape[0], c.shape[1] - len(categories)
    q, e, d = params.frequencies, params.embed, params.embedding_size
    draw = _draw(params, 0)
    w1 = params.embed_sigma * draw.standard_normal((f, q))
    b1 = math.pi * (2.0 * draw.random((f, q)) - 1.0)
    w2 = (2.0 * draw.random((f, q, e)) - 1.0) / math.sqrt(q)
    b2 = (2.0 * draw.random((f, e)) - 1.0) / 4.0
    table = None
    if categories:
        cats = np.random.Generator(np.random.PCG64(np.random.SeedSequence([params.seed, CATEGORY_SALT])))
        table = cats.standard_normal((sum(categories), d))
    width = (e + 1) * f + d * len(categories)
    weights = Weights(
        w1, b1, w2, b2, np.ones(width), [], [], [], np.empty(0), np.empty(0), table, tuple(categories) or None
    )
    h = embed(weights, c)[0]
    heads = 1 if params.own_last_block else n_targets
    sizes = [*params.widths, params.n_classes * heads]
    for i, size in enumerate(sizes):
        part = _draw(params, 1 + i)
        w = part.standard_normal((h.shape[1], size))
        w = w / (h @ w / math.sqrt(h.shape[1])).std(axis=0)
        pre = h @ w / math.sqrt(h.shape[1])
        picks = part.integers(0, n, (size, 5))
        mix = part.exponential(1.0, (size, 5))
        mix /= mix.sum(axis=1, keepdims=True)
        b = -(mix * pre[picks, np.arange(size)[:, None]]).sum(axis=1)
        if i < len(params.widths):
            weights.weight.append(w)
            weights.bias.append(b)
            weights.act.append(np.ones(size))
            h = activation(pre + b, params.act)  # a = 1 at init: the parametric activation is the activation itself
        else:
            weights.out, weights.out_bias = w, b
    if params.own_last_block:
        weights.weight[-1] = np.tile(weights.weight[-1], (1, n_targets))
        weights.bias[-1], weights.act[-1] = np.tile(weights.bias[-1], n_targets), np.tile(weights.act[-1], n_targets)
        weights.out, weights.out_bias = np.tile(weights.out, (1, n_targets)), np.tile(weights.out_bias, n_targets)
    return weights


def schedules(t: float) -> tuple[float, float]:
    """Return (coslog4(t), flat_cos(t)): the learning-rate multiplier and the dropout and weight-decay multiplier."""
    coslog4 = 0.5 * (1.0 - math.cos(2.0 * math.pi * math.log2(1.0 + 15.0 * t)))
    flat_cos = 1.0 if t < 0.5 else 0.5 * (1.0 + math.cos(math.pi * (2.0 * t - 1.0)))
    return coslog4, flat_cos


def multipliers(params: RealMLPParams, t: float) -> tuple[float, float, float]:
    """Return the learning-rate, dropout and weight-decay multipliers at training fraction ``t``.

    ``params.lr_schedule``: ``"coslog4"`` (pytabkit's), ``"cosine"`` (``(1 + cos πt) / 2``) or ``"constant"``;
    ``dropout_schedule`` and ``wd_schedule``: ``"flat_cos"`` (pytabkit's) or ``"constant"``.
    """
    coslog4, flat_cos = schedules(t)
    rates = {"coslog4": coslog4, "cosine": 0.5 * (1.0 + math.cos(math.pi * t)), "constant": 1.0}
    return (rates[params.lr_schedule], flat_cos if params.dropout_schedule == "flat_cos" else 1.0,
            flat_cos if params.wd_schedule == "flat_cos" else 1.0)  # fmt: skip


def dropout_masks(params: RealMLPParams, step: int, t: float, rows: NDArray[np.int64],
                  dtype: type[np.floating]) -> list[Array]:  # fmt: skip
    """Return each hidden block's scaled mask for step ``step`` at training fraction ``t`` (p = dropout · flat_cos(t)).

    Element ``(row, u)`` is kept when ``rng.element_bits(derive_key(seed, DROPOUT_SALT, step, block), row, u)`` is at
    least ``round(p · 2^24)``; kept elements are scaled by 1 / (1 − p).
    """
    p = params.dropout * multipliers(params, t)[1]
    threshold = round(p * (1 << rng.UNIFORM_BITS))
    scale = dtype(1.0 / (1.0 - p))
    masks = []
    for block, width in enumerate(params.widths):
        bits = rng.element_bits(rng.derive_key(params.seed, DROPOUT_SALT, step, block), rows, width)
        masks.append(np.where(bits >= threshold, scale, dtype(0.0)))
    return masks


@dataclass
class Cache:
    """What the backward pass needs.

    The clipped inputs (the categorical inputs' codes after them), the cosine arguments, the embedding z, per block its
    input, pre-activation and mask, and the output layer's input.
    """

    c: Array
    arg: Array
    z: Array
    inputs: list[Array]
    pre: list[Array]
    mask: list[Array]
    last: Array


def forward(weights: Weights, c: Array, masks: Sequence[Array] | None, act: str = "selu") -> tuple[Array, Cache]:
    """Return the logits (B, K · C) of the preprocessed inputs ``c``; ``masks`` None means inference."""
    z, arg = embed(weights, c)
    h = z * weights.scale
    cache = Cache(c, arg, z, [], [], [], h)
    for i, w in enumerate(weights.weight):
        pre = h @ w / math.sqrt(w.shape[0]) + weights.bias[i]
        mixed = pre + (activation(pre, act) - pre) * weights.act[i]
        mask: Array = np.ones_like(mixed) if masks is None else masks[i]
        if mask.shape != mixed.shape:  # an own last block repeats its mask for every target
            mask = np.tile(mask, (1, mixed.shape[1] // mask.shape[1]))
        cache.inputs.append(h)
        cache.pre.append(pre)
        cache.mask.append(mask)
        h = mixed * mask
    cache.last = h
    k = own_targets(weights)
    if k == 1:
        return h @ weights.out / math.sqrt(weights.out.shape[0]) + weights.out_bias, cache
    units, classes = weights.out.shape[0], weights.out.shape[1] // k  # target k's logits from its own units only
    logits = np.einsum("bku,ukc->bkc", h.reshape(len(h), k, units), weights.out.reshape(units, k, classes))
    return (logits / math.sqrt(units)).reshape(len(h), k * classes) + weights.out_bias, cache


def loss_and_logit_grad(
    params: RealMLPParams, logits: Array, y: NDArray[Any], weights: NDArray[np.floating] | None = None
) -> tuple[float, Array]:
    """Return the mean cross-entropy against the smoothed labels and its gradient with respect to the logits.

    ``y``: the classes, ``(B,)`` (more columns: experimental, not covered by the documentation). The loss is the mean
    over the rows, so each logit gradient carries 1 / B. With ``params.objective`` ``"mse"`` (C = 1) ``y`` holds float
    targets: the loss is the mean of ``(o − y)²`` and each gradient ``2 (o − y) / B``. ``weights``: one per row;
    the mean over the rows becomes the weighted mean ``Σ w l / Σ w`` (each gradient ``w / Σ w`` in place of ``1 / B``).
    """
    b = logits.shape[0]
    w = None if weights is None else np.asarray(weights, np.float64)
    if w is not None and w.shape != (b,):
        raise ValueError(f"weights {w.shape} do not fit {b} rows")
    if params.objective == "mse":
        target = np.asarray(y, np.float64).reshape(b, -1)
        if target.shape[1] != logits.shape[1]:
            raise ValueError(f"targets {np.shape(y)} do not fit logits {logits.shape}")
        diff = logits - target
        if w is None:
            return float((diff**2).mean()), 2.0 * diff / diff.size
        total = float(w.sum()) or math.inf  # a batch of weight 0: loss 0 and gradient 0
        scale = (w / (total * diff.shape[1]))[:, None]
        return float((w * (diff**2).mean(axis=1)).sum() / total), 2.0 * diff * scale
    y = np.asarray(y).reshape(b, -1)
    k = y.shape[1]
    if k < 1 or logits.shape[1] % k:
        raise ValueError(f"{k} targets need logits of a multiple of {k} columns, got {logits.shape[1]}")
    classes = logits.shape[1] // k
    target = np.full((b, k, classes), params.label_smoothing / classes)
    target[np.arange(b)[:, None], np.arange(k), y] += 1.0 - params.label_smoothing
    z = logits.reshape(b, k, classes)
    shifted = z - z.max(axis=2, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=2, keepdims=True))
    if w is None:
        grad = (np.exp(log_p) - target) / (b * k)
        return float(-(log_p * target).sum(axis=2).mean()), grad.reshape(b, k * classes)
    total = float(w.sum()) or math.inf  # a batch of weight 0: loss 0 and gradient 0
    grad = (np.exp(log_p) - target) * (w / (total * k))[:, None, None]
    per_row = -(log_p * target).sum(axis=2).mean(axis=1)
    return float((w * per_row).sum() / total), grad.reshape(b, k * classes)


def backward(weights: Weights, cache: Cache, d_logits: Array, act: str = "selu") -> Weights:
    """Return the gradient of every parameter, given the gradient of the loss with respect to the logits."""
    gain = 1.0 / math.sqrt(weights.out.shape[0])
    n, k = len(weights.weight), own_targets(weights)
    if k == 1:
        d_out, dh = cache.last.T @ d_logits * gain, d_logits @ weights.out.T * gain
    else:  # the block-diagonal output layer of an own last block
        b, units, classes = len(d_logits), weights.out.shape[0], weights.out.shape[1] // k
        h3, d3 = cache.last.reshape(b, k, units), d_logits.reshape(b, k, classes)
        d_out = np.einsum("bku,bkc->ukc", h3, d3).reshape(units, k * classes) * gain
        dh = np.einsum("bkc,ukc->bku", d3, weights.out.reshape(units, k, classes)).reshape(b, k * units) * gain
    empty = np.empty(0)
    grads = Weights(empty, empty, empty, empty, empty, [empty] * n, [empty] * n, [empty] * n, d_out,
                    d_logits.sum(axis=0), None, weights.categories)  # fmt: skip
    for i in reversed(range(len(weights.weight))):
        w, pre = weights.weight[i], cache.pre[i]
        dy = dh * cache.mask[i]
        grads.act[i] = (dy * (activation(pre, act) - pre)).sum(axis=0)
        dpre = dy * (1.0 + (activation_grad(pre, act) - 1.0) * weights.act[i])
        gain = 1.0 / math.sqrt(w.shape[0])
        grads.weight[i] = cache.inputs[i].T @ dpre * gain
        grads.bias[i] = dpre.sum(axis=0)
        dh = dpre @ w.T * gain
    grads.scale = (dh * cache.z).sum(axis=0)
    dz = dh * weights.scale
    numeric, codes = split_inputs(weights, cache.c)
    b, f = numeric.shape
    de = dz[:, : weights.w2.shape[2] * f].reshape(b, f, -1)  # the embedding part; the clipped inputs are not parameters
    if weights.table is not None:
        grads.table = np.zeros_like(weights.table)
        np.add.at(
            grads.table,
            table_rows(weights, codes),
            dz[:, (weights.w2.shape[2] + 1) * f :].reshape(b, len(weights.levels), -1),
        )
    cos = np.cos(cache.arg)
    grads.w2 = np.einsum("bfq,bfe->fqe", cos, de)
    grads.b2 = de.sum(axis=0)
    darg = -np.sin(cache.arg) * np.einsum("bfe,fqe->bfq", de, weights.w2)
    grads.w1 = (darg * 2.0 * math.pi * numeric[:, :, None]).sum(axis=0)
    grads.b1 = darg.sum(axis=0)
    return grads


def predict(logits: Array, n_targets: int | None = None, objective: str = "cross_entropy") -> Array:
    """Return the expected class ``Σ c · softmax_c``: ``(B,)`` for ``n_targets`` None, else ``(B, K)`` of K targets.

    With ``objective`` ``"mse"`` the output itself.
    """
    if objective == "mse":
        return logits[:, 0] if n_targets is None else logits
    k = 1 if n_targets is None else n_targets
    if k < 1 or logits.shape[1] % k:
        raise ValueError(f"{k} targets need logits of a multiple of {k} columns, got {logits.shape[1]}")
    b, classes = logits.shape[0], logits.shape[1] // k
    z = logits.reshape(b, k, classes)
    p = np.exp(z - z.max(axis=2, keepdims=True))
    p /= p.sum(axis=2, keepdims=True)
    expected: Array = (p * np.arange(classes, dtype=np.float64)).sum(axis=2)
    return expected[:, 0] if n_targets is None else expected


@dataclass
class AdamState:
    """Adam's moments for every parameter array (in `Weights.arrays` order) and the step count."""

    m: list[Array]
    v: list[Array]
    step: int = 0


def adam_state(weights: Weights) -> AdamState:
    """Return zero moments."""
    return AdamState([np.zeros_like(x) for x in weights.arrays()], [np.zeros_like(x) for x in weights.arrays()])


def adam_step(params: RealMLPParams, weights: Weights, grads: Weights, state: AdamState, t: float) -> None:
    """Apply one step in place at training fraction ``t``, as pytabkit's AdamOptimizer.

    With lr_t = learning_rate · coslog4(t) and wd_t = weight_decay · flat_cos(t): first the decoupled decay
    ``p ← p · (1 − wd_t · lr_t · f_wd² · f_lr²)`` (the factors enter twice, as in pytabkit), then torch's Adam with the
    learning rate lr_t · f_lr.
    """
    state.step += 1
    lr_mult, _, wd_mult = multipliers(params, t)
    lr_t, wd_t = params.learning_rate * lr_mult, params.weight_decay * wd_mult
    correction1 = 1 - params.beta1**state.step
    correction2 = 1 - params.beta2**state.step
    each = factors(params, len(weights.weight), table=weights.table is not None)
    arrays = zip(weights.arrays(), grads.arrays(), state.m, state.v, each, strict=True)
    for p, grad, m, v, (f_lr, f_wd) in arrays:
        if f_wd != 0.0 and wd_t != 0.0:
            p[...] = p * (1.0 - wd_t * lr_t * f_wd * f_wd * f_lr * f_lr)
        m[...] = params.beta1 * m + (1 - params.beta1) * grad
        v[...] = params.beta2 * v + (1 - params.beta2) * grad * grad
        p[...] = p - (lr_t * f_lr / correction1) * m / (np.sqrt(v) / np.sqrt(correction2) + params.eps)
