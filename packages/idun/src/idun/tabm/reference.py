"""The NumPy reference of TabM: initialisation, forward, backward, AdamW (the Metal backend's correctness oracle).

Plain NumPy in any float dtype (float64 for gradient checks, float32 to bound the Metal backend's error). Shapes: ``B``
rows, ``k`` members, ``d`` inputs, ``w`` width, ``c`` classes, ``K`` targets. One block is

    u = h ⊙ r              (B, k, d_in)    h = the block's input; for block 0 the row's inputs, shared by every member
    t = u · weightᵀ        (B, k, w)       weight shared by the members
    z = t ⊙ s + bias       (B, k, w)
    a = relu(z) ⊙ m        (B, k, w)       m = the dropout mask, already scaled by 1 / (1 - p); all ones in inference

and the head is ``o[b, j] = a[b, j] · head[j] + head_bias[j]`` per member ``j``, ``c`` outputs (more heads:
experimental, not covered by the documentation). The loss is the mean cross-entropy over all ``B × k`` member outputs
against the row's class; the prediction is the expected class of the member-averaged softmax.

With a temporal encoder (``params.encoder`` not ``"none"``, `idun.tabm.encoder`) each row also has a sequence
``(S, C)``; the encoder's output ``(B, H)`` is prepended to the row's inputs, so block 0 reads ``H + d`` inputs, and the
backward pass carries block 0's input gradient on into the encoder.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun import rng
from idun.tabm import embeddings as emb
from idun.tabm import encoder as enc
from idun.tabm.params import TabMParams

Array = NDArray[np.floating]
# A training loss given by the caller: (the batch's member outputs (B, k, c), its absolute rows (B,)) -> (the loss, its
# gradient with respect to the outputs, in their shape).
Loss = Callable[[NDArray[np.floating], NDArray[np.int64]], tuple[float, NDArray[np.floating]]]
DROPOUT_SALT = 5  # derive_key(seed, DROPOUT_SALT, step, block): the dropout masks' key
INIT_SALT = 3  # PCG64(SeedSequence([seed, INIT_SALT, block, part])): the initial weights


@dataclass
class Weights:
    """TabM's parameters.

    Per block ``i``: ``weight[i]`` (w, d_in), ``r[i]`` (k, d_in), ``s[i]`` and ``bias[i]`` (k, w). The head:
    ``head`` (k, w, K · c) and ``head_bias`` (k, K · c) for K targets. ``encoder``: the temporal encoder's, or None.
    ``embedding``: the piecewise-linear embeddings' of the first F inputs, or None; ``bins`` their fixed encoding
    (`idun.tabm.embeddings.Bins`, F rows), trained by nothing, so not among the `arrays`.
    """

    weight: list[Array]
    r: list[Array]
    s: list[Array]
    bias: list[Array]
    head: Array
    head_bias: Array
    encoder: enc.EncoderWeights | None = None
    embedding: emb.EmbeddingWeights | None = None
    bins: emb.Bins | None = None

    def arrays(self) -> list[Array]:
        """Return every parameter array in a fixed order.

        Weight, r, s, bias of each block, the head's two, then the encoder's (`idun.tabm.encoder.EncoderWeights.arrays`)
        or the embeddings' (`idun.tabm.embeddings.EmbeddingWeights.arrays`) when there are.
        """
        tail = [] if self.encoder is None else self.encoder.arrays()
        tail += [] if self.embedding is None else self.embedding.arrays()
        return [*self.weight, *self.r, *self.s, *self.bias, self.head, self.head_bias, *tail]

    def astype(self, dtype: type[np.floating]) -> "Weights":
        """Return a copy in ``dtype``."""

        def cast(arrays: Sequence[Array]) -> list[Array]:
            return [a.astype(dtype) for a in arrays]

        encoder = None if self.encoder is None else self.encoder.astype(dtype)
        embedding = None if self.embedding is None else self.embedding.astype(dtype)
        return Weights(cast(self.weight), cast(self.r), cast(self.s), cast(self.bias), self.head.astype(dtype),
                       self.head_bias.astype(dtype), encoder, embedding, self.bins)  # fmt: skip


def init(
    params: TabMParams, n_inputs: int, targets: int = 1, sequence_inputs: int = 0, bins: emb.Bins | None = None
) -> Weights:
    """Draw float32 weights as `tabm` 0.0.3 initialises them, from idun's own generators (Torch's cannot be reproduced).

    Weights and biases uniform on ±1/√d_in (one bias draw shared by every member), block 0's ``r`` random signs, every
    other adapter 1, the head uniform on ±1/√w with ``targets · params.outputs`` outputs per member (one linear layer
    onto them, as a Torch head of that width; ``targets`` = 1 draws the single-target head). With an encoder, block 0
    reads its ``params.encoded`` outputs and the ``n_inputs`` inputs, and the encoder's weights come from
    `idun.tabm.encoder.init` for ``sequence_inputs`` inputs per step (0, the default, without an encoder). With
    ``params.num_emb_type`` ``"pwl"``, ``bins`` (`idun.tabm.embeddings.fit_bins` of the first F ≤ ``n_inputs`` inputs)
    embeds those F inputs: block 0 reads their F · ``d_embedding`` embedded values, then the other inputs, its weight
    uniform on ±1/√ of that width and its ``r`` N(0, 1) per member and column (one draw for a column's embedded
    values), and the embeddings' weights come from `idun.tabm.embeddings.init`.
    """
    if targets < 1:
        raise ValueError(f"targets must be a positive int, got {targets!r}")
    if (params.encoder == "none") != (sequence_inputs == 0):
        raise ValueError(f"encoder {params.encoder!r} with {sequence_inputs} sequence inputs per step")
    if (params.num_emb_type == "none") != (bins is None):
        raise ValueError(f"num_emb_type {params.num_emb_type!r} with {'no' if bins is None else ''} bins")
    embedded = 0 if bins is None else int(bins.scale.shape[0])
    if embedded > n_inputs:
        raise ValueError(f"bins for {embedded} columns of {n_inputs} inputs")
    widths = [params.d_embedding] * embedded + [1] * (n_inputs - embedded)  # block 0's columns per input

    def draw(block: int, part: int) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence([params.seed, INIT_SALT, block, part])))

    k, w = params.members, params.width
    weights = Weights([], [], [], [], np.empty(0), np.empty(0))
    for i in range(params.blocks):
        d_in = params.encoded + sum(widths) if i == 0 else w
        bound = 1.0 / np.sqrt(d_in)
        weights.weight.append(draw(i, 0).uniform(-bound, bound, (w, d_in)).astype(np.float32))
        weights.bias.append(np.tile(draw(i, 1).uniform(-bound, bound, w).astype(np.float32), (k, 1)))
        first: NDArray[Any]
        if i > 0:
            first = np.ones((k, d_in))
        elif bins is None:
            first = 2 * draw(i, 2).integers(0, 2, (k, d_in)) - 1  # random signs
        else:
            first = np.repeat(draw(i, 2).normal(size=(k, len(widths))), widths, axis=1)  # one draw per column
        weights.r.append(first.astype(np.float32))
        weights.s.append(np.ones((k, w), np.float32))
    bound = 1.0 / np.sqrt(w)
    head = draw(params.blocks, 0)
    outputs = targets * params.outputs
    weights.head = head.uniform(-bound, bound, (k, w, outputs)).astype(np.float32)
    weights.head_bias = head.uniform(-bound, bound, (k, outputs)).astype(np.float32)
    if sequence_inputs:
        weights.encoder = enc.init(params, sequence_inputs)
    if bins is not None:
        weights.embedding = emb.init(embedded, bins.components, params.d_embedding, params.seed)
        weights.bins = bins
    return weights


def dropout_masks(params: TabMParams, step: int, rows: NDArray[np.int64], dtype: type[np.floating]) -> list[Array]:
    """Return each block's scaled dropout mask for training step ``step``: ``(B, k, w)``, entries 0 or 1 / (1 - p).

    Element ``(row, j, u)`` is kept when ``rng.element_bits(derive_key(seed, DROPOUT_SALT, step, block), row, j·w + u)``
    is at least ``round(p · 2^24)``. The draws depend on the absolute row, not on the batch's composition.
    """
    k, w = params.members, params.width
    threshold = round(params.dropout * (1 << rng.UNIFORM_BITS))
    scale = dtype(1.0 / (1.0 - params.dropout))
    masks = []
    for block in range(params.blocks):
        bits = rng.element_bits(rng.derive_key(params.seed, DROPOUT_SALT, step, block), rows, k * w)
        masks.append(np.where(bits >= threshold, scale, dtype(0.0)).reshape(len(rows), k, w))
    return masks


@dataclass
class Cache:
    """What the backward pass needs from the forward pass.

    Per block h, u, t, z and the mask; the head's input; the encoder's cache, if any; the embedded inputs and their
    encoding, if any.
    """

    h: list[Array]
    u: list[Array]
    t: list[Array]
    z: list[Array]
    mask: list[Array]
    last: Array
    encoder: enc.EncoderCache | None = None
    embedding: tuple[Array, Array] | None = None


def forward(
    weights: Weights, x: Array, masks: Sequence[Array] | None, sequences: Array | None = None
) -> tuple[Array, Cache]:
    """Return the logits ``(B, k, K · c)`` for inputs ``x`` (B, d) and the cache; ``masks`` None means inference.

    ``sequences`` ``(B, S, C)``: each row's sequence, for (and only for) a network with an encoder.
    """
    encoder_cache = None
    if (weights.encoder is None) != (sequences is None):
        raise ValueError("a network with an encoder takes sequences, and one without takes none")
    if weights.encoder is not None and sequences is not None:
        encoded, encoder_cache = enc.forward(weights.encoder, np.asarray(sequences, x.dtype))
        x = np.concatenate([encoded, x], axis=1)
    embedding_cache = None
    if weights.embedding is not None and weights.bins is not None:
        n = weights.bins.scale.shape[0]
        embedded, q = emb.forward(weights.embedding, weights.bins, x[:, :n])
        embedding_cache = (x[:, :n], q)
        x = np.concatenate([embedded, x[:, n:]], axis=1)
    k = weights.r[0].shape[0]
    h = np.broadcast_to(x[:, None, :], (x.shape[0], k, x.shape[1]))
    cache = Cache([], [], [], [], [], h, encoder_cache, embedding_cache)
    for i, weight in enumerate(weights.weight):
        u = h * weights.r[i]
        t = u @ weight.T
        z = t * weights.s[i] + weights.bias[i]
        mask = masks[i] if masks is not None else np.ones_like(z)
        for store, value in zip((cache.h, cache.u, cache.t, cache.z, cache.mask), (h, u, t, z, mask), strict=True):
            store.append(value)
        h = np.maximum(z, 0) * mask
    cache.last = h
    return np.einsum("bjw,jwc->bjc", h, weights.head) + weights.head_bias, cache


def label_matrix(y: NDArray[Any], n_rows: int, targets: int, n_classes: int) -> NDArray[np.int64]:
    """Return labels ``y`` as a ``(n_rows, targets)`` array; raise ``ValueError`` unless they fit.

    ``y`` is ``(n_rows, targets)``, or ``(n_rows,)`` for one target, with every class in ``0 .. n_classes − 1``.
    """
    y = np.asarray(y)
    if y.shape != (n_rows, targets) and not (targets == 1 and y.shape == (n_rows,)):
        raise ValueError(f"need labels of shape ({n_rows}, {targets})" + (f" or ({n_rows},)" if targets == 1 else "")
                         + f", got {y.shape}")  # fmt: skip
    if not np.issubdtype(y.dtype, np.integer) or (y.size and not (y.min() >= 0 and y.max() < n_classes)):
        raise ValueError(f"labels must be integer classes 0 .. {n_classes - 1}")
    return y.reshape(n_rows, targets).astype(np.int64, copy=False)


def loss_and_logit_grad(
    logits: Array, y: NDArray[Any], objective: str = "cross_entropy", weights: NDArray[np.floating] | None = None
) -> tuple[float, Array]:
    """Return the loss and its gradient with respect to the logits ``(B, k, c)``.

    ``y`` is ``(B,)``, the classes (more columns: experimental, not covered by the documentation). The loss is the mean
    cross-entropy over the ``B × k`` member outputs, so the logit gradient is ``(softmax − one-hot) / (B · k)``. With
    ``objective`` ``"mse"`` (c = 1) ``y`` holds float targets, the loss is the mean of ``(o − y)²`` over the ``B × k``
    outputs and the gradient ``2 (o − y) / (B · k)``. With ``"hl_gauss"`` (one target) ``y`` holds the soft labels ``(B,
    n_bins)`` (`idun.tabm.hl_gauss.labels`), the loss is the mean over the ``B × k`` member outputs of ``−Σ p log
    softmax`` and the gradient ``(softmax − p) / (B · k)``. ``weights``: one per row; every mean over rows
    becomes the weighted mean ``Σ w l / Σ w`` (the gradients ``w / Σ w`` in place of ``1 / B``).
    """
    if weights is not None:
        return _weighted_loss(logits, y, objective, np.asarray(weights, np.float64))
    if objective == "hl_gauss":
        b, k, outputs = logits.shape
        soft = np.asarray(y, np.float64)
        if soft.shape != (b, outputs):
            raise ValueError(f"soft labels {soft.shape} do not fit logits {logits.shape}")
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_p = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        return float(-(soft[:, None, :] * log_p).sum(axis=-1).mean()), (np.exp(log_p) - soft[:, None, :]) / (b * k)
    if objective == "mse":
        b, k, outputs = logits.shape
        target = np.asarray(y, np.float64).reshape(b, -1)
        if target.shape[1] != outputs:
            raise ValueError(f"targets {np.shape(y)} do not fit logits {logits.shape}")
        diff = logits - target[:, None, :]
        return float((diff**2).mean()), 2.0 * diff / diff.size
    y = np.asarray(y)
    labels = y[:, None] if y.ndim == 1 else y
    b, k, outputs = logits.shape
    if labels.ndim != 2 or labels.shape[0] != b or labels.shape[1] < 1 or outputs % labels.shape[1] != 0:
        raise ValueError(f"labels {y.shape} do not fit logits {logits.shape}")
    targets = labels.shape[1]
    per_target = logits.reshape(b, k, targets, outputs // targets)
    shifted = per_target - per_target.max(axis=-1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    index = (np.arange(b)[:, None, None], np.arange(k)[None, :, None], np.arange(targets)[None, None, :],
             labels[:, None, :])  # fmt: skip
    grad = np.exp(log_p)
    grad[index] -= 1
    return float(-log_p[index].mean()), (grad / (b * k * targets)).reshape(b, k, outputs)


def _weighted_loss(logits: Array, y: NDArray[Any], objective: str, w: NDArray[np.float64]) -> tuple[float, Array]:
    """Return `loss_and_logit_grad`'s weighted loss and gradient: each row's losses and gradients times ``w / Σ w``."""
    b, k, outputs = logits.shape
    if w.shape != (b,):
        raise ValueError(f"weights {w.shape} do not fit {b} rows")
    total = float(w.sum()) or math.inf  # a batch of weight 0: loss 0 and gradient 0
    if objective == "hl_gauss":
        soft = np.asarray(y, np.float64)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_p = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        per_row: Array = -(soft[:, None, :] * log_p).sum(axis=-1).mean(axis=1)
        grad = (np.exp(log_p) - soft[:, None, :]) * (w / (total * k))[:, None, None]
        return float((w * per_row).sum() / total), grad
    if objective == "mse":
        diff = logits - np.asarray(y, np.float64).reshape(b, -1)[:, None, :]
        squares: Array = (diff**2).reshape(b, -1).mean(axis=1)
        return float((w * squares).sum() / total), 2.0 * diff * (w / (total * k * outputs))[:, None, None]
    y = np.asarray(y)
    labels = y[:, None] if y.ndim == 1 else y
    targets = labels.shape[1]
    per_target = logits.reshape(b, k, targets, outputs // targets)
    shifted = per_target - per_target.max(axis=-1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    index = (np.arange(b)[:, None, None], np.arange(k)[None, :, None], np.arange(targets)[None, None, :],
             labels[:, None, :])  # fmt: skip
    grad = np.exp(log_p)
    grad[index] -= 1
    entropy: Array = -log_p[index].reshape(b, -1).mean(axis=1)
    scale = (w / (total * k * targets))[:, None, None, None]
    return float((w * entropy).sum() / total), (grad * scale).reshape(b, k, outputs)


def host_loss(loss: Loss, outputs: Array, rows: NDArray[np.int64]) -> tuple[float, Array]:
    """Return ``loss(outputs, rows)`` checked: the loss as a float, its gradient in ``outputs``' shape and dtype.

    ``outputs`` is the batch's member outputs ``(B, k, c)`` and ``rows`` its absolute row indices. Raises
    ``ValueError`` unless the gradient has ``outputs``' shape and the loss and the gradient are finite.
    """
    value, gradient = loss(outputs, rows)
    d_outputs = np.asarray(gradient, outputs.dtype)
    if d_outputs.shape != outputs.shape:
        raise ValueError(f"the loss returned a gradient of shape {d_outputs.shape} for outputs of shape "
                         f"{outputs.shape}")  # fmt: skip
    if not (math.isfinite(value) and bool(np.isfinite(d_outputs).all())):
        raise ValueError("the loss returned a loss or a gradient that is not finite")
    return float(value), d_outputs


def backward(weights: Weights, cache: Cache, d_logits: Array) -> Weights:
    """Return the gradient of every parameter, given the gradient of the loss with respect to the logits."""
    d_head = np.einsum("bjw,bjc->jwc", cache.last, d_logits)
    d_head_bias = d_logits.sum(axis=0)
    da = np.einsum("bjc,jwc->bjw", d_logits, weights.head)
    blocks = len(weights.weight)
    grads = Weights([np.empty(0)] * blocks, [np.empty(0)] * blocks, [np.empty(0)] * blocks, [np.empty(0)] * blocks,
                    d_head, d_head_bias)  # fmt: skip
    for i in reversed(range(blocks)):
        dz = da * cache.mask[i] * (cache.z[i] > 0)
        grads.bias[i] = dz.sum(axis=0)
        grads.s[i] = (dz * cache.t[i]).sum(axis=0)
        dt = dz * weights.s[i]
        grads.weight[i] = np.einsum("bjw,bjd->wd", dt, cache.u[i])
        du = dt @ weights.weight[i]
        grads.r[i] = (du * cache.h[i]).sum(axis=0)
        da = du * weights.r[i]
    if weights.encoder is not None and cache.encoder is not None:
        hidden = weights.encoder.w_hh.shape[1]
        grads.encoder = enc.backward(weights.encoder, cache.encoder, da[:, :, :hidden].sum(axis=1))  # Σ over members
    if weights.embedding is not None and cache.embedding is not None:
        x, q = cache.embedding
        width = weights.embedding.ple.shape[0] * weights.embedding.ple.shape[2]
        grads.embedding = emb.backward(x, q, da[:, :, :width].sum(axis=1))  # Σ over members
    return grads


def predict(
    logits: Array, targets: int | None = None, objective: str = "cross_entropy", centres: Array | None = None
) -> Array:
    """Return the expected class of the member-averaged softmax: ``Σ_c c · mean_j softmax(o[b, j])_c``.

    ``targets`` None: the logits ``(B, k, c)`` and the result ``(B,)`` (an integer: experimental, not covered by the
    documentation). With ``objective`` ``"mse"`` it is the mean of the members' outputs; with ``"hl_gauss"`` (one
    target) the expected bin centre, ``Σ_c centres[c] · mean_j softmax(o[b, j])_c``.
    """
    b, k, outputs = logits.shape
    if objective == "mse":
        mean = logits.mean(axis=1)
        return mean[:, 0] if targets is None else mean
    n = 1 if targets is None else targets
    if n < 1 or outputs % n != 0:
        raise ValueError(f"{outputs} outputs are not {n} targets' classes")
    if objective == "hl_gauss" and (n != 1 or centres is None or np.shape(centres) != (outputs,)):
        raise ValueError(f"HL-Gauss predicts one target from {outputs} bin centres")
    values = np.asarray(centres, np.float64) if objective == "hl_gauss" else np.arange(outputs // n)
    per_target = logits.reshape(b, k, n, outputs // n)
    shifted = per_target - per_target.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=-1, keepdims=True)
    expected = (p.mean(axis=1) * values).sum(axis=-1)
    return expected[:, 0] if targets is None else expected


@dataclass
class AdamState:
    """AdamW's moments for every parameter array (in `Weights.arrays` order) and the step count."""

    m: list[Array]
    v: list[Array]
    step: int = 0


def adam_state(weights: Weights) -> AdamState:
    """Return zero moments."""
    return AdamState([np.zeros_like(x) for x in weights.arrays()], [np.zeros_like(x) for x in weights.arrays()])


def _scaled(grads: Weights, scale: float) -> Weights:
    """Return the gradients times ``scale``."""
    n, a = len(grads.weight), [x * scale for x in grads.arrays()]
    tail = a[4 * n + 2 :]
    encoder = None if grads.encoder is None else enc.EncoderWeights(*tail)
    embedding = None if grads.embedding is None else emb.EmbeddingWeights(*tail)
    return Weights(a[:n], a[n : 2 * n], a[2 * n : 3 * n], a[3 * n : 4 * n], a[4 * n], a[4 * n + 1], encoder, embedding,
                   grads.bins)  # fmt: skip


def clip_scale(most: float, squares: Sequence[float]) -> float:
    """Return ``torch.nn.utils.clip_grad_norm_``'s factor: min(1, most / (norm + 1e-6)), norm √ of the arrays' sums.

    ``squares``: each gradient array's sum of squares, in `Weights.arrays` order, added in that order.
    """
    norm = math.sqrt(sum(squares))
    return min(1.0, most / (norm + 1e-6))


def adamw_step(
    params: TabMParams, weights: Weights, grads: Weights, state: AdamState, rate: float | None = None
) -> None:
    """Apply one AdamW step in place as ``torch.optim.AdamW``: decoupled decay, then the bias-corrected update.

    ``rate``: this step's learning rate (a schedule's), ``params.learning_rate`` by default. With
    ``params.gradient_clipping_norm`` set, the gradients are first scaled as ``clip_grad_norm_`` scales them
    (`clip_scale`).
    """
    if params.gradient_clipping_norm > 0.0:
        scale = clip_scale(params.gradient_clipping_norm, [float(np.sum(g * g)) for g in grads.arrays()])
        grads = _scaled(grads, scale)
    lr = params.learning_rate if rate is None else rate
    state.step += 1
    correction1 = 1 - params.beta1**state.step
    correction2 = 1 - params.beta2**state.step
    for p, grad, m, v in zip(weights.arrays(), grads.arrays(), state.m, state.v, strict=True):
        p[...] = p * (1 - lr * params.weight_decay)
        m[...] = params.beta1 * m + (1 - params.beta1) * grad
        v[...] = params.beta2 * v + (1 - params.beta2) * grad * grad
        p[...] = p - (lr / correction1) * m / (np.sqrt(v) / np.sqrt(correction2) + params.eps)
