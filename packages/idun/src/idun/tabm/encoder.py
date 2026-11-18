"""TabM's temporal encoder, the NumPy reference: a per-step Linear and GELU, then a GRU over the steps.

Shapes: ``B`` rows, ``S`` steps (oldest first), ``C`` inputs per step (`idun.tabm.Sequences`: the channels and the
present flag), ``E`` embedding units, ``H`` hidden units. Per step ``s``:

    p_s = W_e x_s + b_e                    (B, E)    the embedding's pre-activation
    e_s = GELU(p_s) = p_s Φ(p_s)           (B, E)    the exact GELU (Φ the standard normal CDF)
    r = sigmoid(W_ir e_s + b_ir + W_hr h + b_hr)    the GRU in Torch's gate order (r, z, n): the rows of W_ih, W_hh,
    z = sigmoid(W_iz e_s + b_iz + W_hz h + b_hz)    b_ih and b_hh are the r, z and n gates' H rows each
    n = tanh(W_in e_s + b_in + r ⊙ (W_hn h + b_hn))
    h' = (1 − z) ⊙ n + z ⊙ h               (B, H)    from h = 0 before the first step

and the encoder's output is the last ``h``. Plain NumPy in any float dtype, as `idun.tabm.reference`; the Metal backend
runs the same arithmetic in float32.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from idun._jit import njit
from idun.tabm.params import TabMParams

Array = NDArray[np.floating]
ENCODER_SALT = 7  # PCG64(SeedSequence([seed, ENCODER_SALT, part])): the encoder's initial weights
GATES = 3  # r, z, n


@dataclass
class EncoderWeights:
    """The encoder's parameters.

    ``embed`` (E, C), ``embed_bias`` (E,), ``w_ih`` (3H, E), ``w_hh`` (3H, H), ``b_ih`` and ``b_hh`` (3H,).
    """

    embed: Array
    embed_bias: Array
    w_ih: Array
    w_hh: Array
    b_ih: Array
    b_hh: Array

    def arrays(self) -> list[Array]:
        """Return the parameter arrays in a fixed order: embed, embed_bias, w_ih, w_hh, b_ih, b_hh."""
        return [self.embed, self.embed_bias, self.w_ih, self.w_hh, self.b_ih, self.b_hh]

    def astype(self, dtype: type[np.floating]) -> "EncoderWeights":
        """Return a copy in ``dtype``."""
        return EncoderWeights(*(a.astype(dtype) for a in self.arrays()))


def init(params: TabMParams, inputs: int) -> EncoderWeights:
    """Draw float32 weights for ``inputs`` inputs per step as Torch initialises a Linear and a GRU.

    The Linear's weight and bias are uniform on ±1/√inputs, every GRU weight and bias uniform on ±1/√H; each array is
    one draw of idun's own generators (``part`` 0..5 in `EncoderWeights.arrays` order).
    """
    if inputs < 1:
        raise ValueError(f"inputs must be a positive int, got {inputs!r}")
    e, h = params.encoder_embed, params.encoder_hidden

    def draw(part: int, bound: float, shape: tuple[int, ...]) -> Array:
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([params.seed, ENCODER_SALT, part])))
        return generator.uniform(-bound, bound, shape).astype(np.float32)

    linear, gru = 1.0 / math.sqrt(inputs), 1.0 / math.sqrt(h)
    shapes = [(e, inputs), (e,), (GATES * h, e), (GATES * h, h), (GATES * h,), (GATES * h,)]
    bounds = [linear, linear, gru, gru, gru, gru]
    return EncoderWeights(*(draw(part, b, s) for part, (b, s) in enumerate(zip(bounds, shapes, strict=True))))


@njit()
def _phi(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Φ(x), elementwise (a flat float64 array): ½ erfc(−x / √2)."""
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = 0.5 * math.erfc(-x[i] / math.sqrt(2.0))
    return out


def normal_cdf(x: Array) -> Array:
    """Return Φ(x) elementwise in ``x``'s dtype (computed in float64)."""
    flat = np.ascontiguousarray(x, np.float64).reshape(-1)
    return _phi(flat).reshape(np.shape(x)).astype(x.dtype)


def gelu(x: Array) -> Array:
    """Return the exact GELU, ``x Φ(x)``."""
    return x * normal_cdf(x)


def gelu_grad(x: Array) -> Array:
    """Return the GELU's derivative, ``Φ(x) + x φ(x)``."""
    return normal_cdf(x) + x * np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def sigmoid(x: Array) -> Array:
    """Return the logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class EncoderCache:
    """What the backward pass needs, per step: x, p, e, the hidden state before the step, r, z, n and W_hn h + b_hn."""

    x: Array  # (B, S, C)
    p: Array  # (B, S, E)
    e: Array
    h: Array  # (B, S, H): the state each step starts from
    r: Array
    z: Array
    n: Array
    hn: Array


def forward(weights: EncoderWeights, x: Array) -> tuple[Array, EncoderCache]:
    """Return the last hidden state ``(B, H)`` of sequences ``x`` (B, S, C) and the cache."""
    b, steps, _ = x.shape
    hidden = weights.w_hh.shape[1]
    p = x @ weights.embed.T + weights.embed_bias
    e = gelu(p)
    gi = e @ weights.w_ih.T + weights.b_ih  # (B, S, 3H): the inputs' part of every gate, all steps at once
    cache = EncoderCache(x, p, e, *(np.empty((b, steps, hidden), x.dtype) for _ in range(5)))
    h = np.zeros((b, hidden), x.dtype)
    for s in range(steps):
        gh = h @ weights.w_hh.T + weights.b_hh
        r = sigmoid(gi[:, s, :hidden] + gh[:, :hidden])
        z = sigmoid(gi[:, s, hidden : 2 * hidden] + gh[:, hidden : 2 * hidden])
        hn = gh[:, 2 * hidden :]
        n = np.tanh(gi[:, s, 2 * hidden :] + r * hn)
        for store, value in zip((cache.h, cache.r, cache.z, cache.n, cache.hn), (h, r, z, n, hn), strict=True):
            store[:, s] = value
        h = (1.0 - z) * n + z * h
    return h, cache


def backward(weights: EncoderWeights, cache: EncoderCache, d_out: Array) -> EncoderWeights:
    """Return the gradient of every encoder parameter, given the gradient ``(B, H)`` of the loss at its output."""
    steps, hidden = cache.h.shape[1], cache.h.shape[2]
    d_gi = np.empty((*cache.h.shape[:2], GATES * hidden), d_out.dtype)
    d_gh = np.empty_like(d_gi)
    dh = d_out
    for s in reversed(range(steps)):
        h, r, z, n, hn = cache.h[:, s], cache.r[:, s], cache.z[:, s], cache.n[:, s], cache.hn[:, s]
        dn = dh * (1.0 - z) * (1.0 - n * n)  # through tanh: the n gate's pre-activation
        dz = dh * (h - n) * z * (1.0 - z)
        dr = dn * hn * r * (1.0 - r)
        d_gi[:, s] = np.concatenate([dr, dz, dn], axis=1)
        d_gh[:, s] = np.concatenate([dr, dz, dn * r], axis=1)
        dh = dh * z + d_gh[:, s] @ weights.w_hh
    dp = (d_gi @ weights.w_ih) * gelu_grad(cache.p)
    return EncoderWeights(
        np.einsum("bse,bsc->ec", dp, cache.x),
        dp.sum(axis=(0, 1)),
        np.einsum("bsg,bse->ge", d_gi, cache.e),
        np.einsum("bsg,bsh->gh", d_gh, cache.h),
        d_gi.sum(axis=(0, 1)),
        d_gh.sum(axis=(0, 1)),
    )
