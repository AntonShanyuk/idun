"""TabM's piecewise-linear numeric embeddings, the NumPy reference.

rtdl_num_embeddings' `PiecewiseLinearEmbeddings` in its version "B" without the activation, as pytabkit's TabM builds
it for ``num_emb_type="pwl"``. Feature f's bin edges are the distinct quantiles 0, 1/T, ..., 1 of its training values
(`fit_bins`, T = ``num_emb_n_bins``: at most T + 1 edges, so T_f ≤ T bins). Its encoding has P components, P the largest
T_f, laid out as rtdl lays them: the feature's last bin in the last component, its other bins first, zeros between.
With ``scale`` = 1 / width and ``shift`` = −left / width of the bin in each component (0 and 0 in the unused ones),

    q_p = x · scale[f, p] + shift[f, p]      then q_0 clamped above at 1, q_1 .. q_{P−2} to [0, 1] and q_{P−1} below
                                             at 0 (unclamped for a feature of one bin; nothing clamped when P = 1)
    e_f = x · a_f + c_f + q @ W_f            (d,)  a_f, c_f: rtdl's LinearEmbeddings; W_f (P, d): zero at init

and the network's block 0 reads e_0, ..., e_{F−1} (F · d columns, feature-major) where it read the F inputs. Plain
NumPy in any float dtype, as `idun.tabm.reference`; the GPU backends run the same arithmetic in float32.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]
EMBED_SALT = 13  # PCG64(SeedSequence([seed, EMBED_SALT, part])): the embeddings' initial weights


@dataclass
class Bins:
    """The encoding's fixed layout: ``scale`` and ``shift`` ``(F, P)`` float32, ``single`` ``(F,)`` bool."""

    scale: NDArray[np.float32]
    shift: NDArray[np.float32]
    single: NDArray[np.bool_]

    @property
    def components(self) -> int:
        """P: the largest bin count of the features."""
        return int(self.scale.shape[1])


@dataclass
class EmbeddingWeights:
    """The embeddings' parameters: ``linear`` and ``linear_bias`` ``(F, d)``, ``ple`` ``(F, P, d)``."""

    linear: Array
    linear_bias: Array
    ple: Array

    def arrays(self) -> list[Array]:
        """Return the parameter arrays in a fixed order: linear, linear_bias, ple."""
        return [self.linear, self.linear_bias, self.ple]

    def astype(self, dtype: type[np.floating]) -> "EmbeddingWeights":
        """Return a copy in ``dtype``."""
        return EmbeddingWeights(*(a.astype(dtype) for a in self.arrays()))


def fit_bins(x: NDArray[np.floating], n_bins: int) -> Bins:
    """Return the bins of the columns of ``x`` ``(n, F)``: rtdl's `compute_bins` on the finite values, in float64.

    Each column's edges are the distinct values among its quantiles at 0, 1/n_bins, ..., 1 (NumPy's linear
    interpolation, as Torch's). A column with fewer than two distinct finite values raises ``ValueError``, as rtdl.
    """
    if n_bins < 2:
        raise ValueError(f"num_emb_n_bins must be at least 2, got {n_bins!r}")
    values = np.asarray(x, np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"need a 2-D array with at least one column, got shape {values.shape}")
    levels = np.linspace(0.0, 1.0, n_bins + 1)
    edges = []
    for f in range(values.shape[1]):
        column = values[:, f][np.isfinite(values[:, f])]
        found = np.unique(np.quantile(column, levels)) if column.size else column
        if found.size < 2:
            raise ValueError(f"input column {f} has fewer than two distinct finite values on the rows it is binned "
                             "on: a piecewise-linear embedding needs at least one bin")  # fmt: skip
        edges.append(found)
    return bins_of(edges)


def bins_of(edges: list[NDArray[np.floating]]) -> Bins:
    """Return the encoding's layout for each feature's increasing bin edges (rtdl's `_PiecewiseLinearEncodingImpl`)."""
    counts = [len(e) - 1 for e in edges]
    p = max(counts)
    scale, shift = np.zeros((len(edges), p), np.float32), np.zeros((len(edges), p), np.float32)
    for f, e in enumerate(edges):
        e32 = np.asarray(e, np.float32)  # rtdl's bins are the data's float32 tensors
        width = np.diff(e32)
        if not (width > 0).all():
            raise ValueError(f"feature {f}'s bin edges must increase")
        w, b = (np.float32(1.0) / width), -e32[:-1] / width
        scale[f, -1], shift[f, -1] = w[-1], b[-1]
        scale[f, : counts[f] - 1], shift[f, : counts[f] - 1] = w[:-1], b[:-1]
    return Bins(scale, shift, np.array([c == 1 for c in counts]))


def init(n_features: int, components: int, d_embedding: int, seed: int) -> EmbeddingWeights:
    """Draw float32 weights as rtdl initialises version "B": the linear term uniform on ±1/√d, the PLE weights zero."""
    bound = 1.0 / np.sqrt(d_embedding)

    def draw(part: int) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, EMBED_SALT, part])))

    shape = (n_features, d_embedding)
    return EmbeddingWeights(draw(0).uniform(-bound, bound, shape).astype(np.float32),
                            draw(1).uniform(-bound, bound, shape).astype(np.float32),
                            np.zeros((n_features, components, d_embedding), np.float32))  # fmt: skip


def encode(bins: Bins, x: Array) -> Array:
    """Return the piecewise-linear encoding ``(B, F, P)`` of inputs ``x`` ``(B, F)`` in their dtype."""
    q = x[:, :, None] * bins.scale.astype(x.dtype) + bins.shift.astype(x.dtype)
    if q.shape[-1] > 1:
        q[..., 0] = np.minimum(q[..., 0], 1.0)
        q[..., 1:-1] = np.clip(q[..., 1:-1], 0.0, 1.0)
        q[..., -1] = np.where(bins.single, q[..., -1], np.maximum(q[..., -1], 0.0))
    return q


def forward(weights: EmbeddingWeights, bins: Bins, x: Array) -> tuple[Array, Array]:
    """Return the embedded inputs ``(B, F · d)``, feature-major, and the encoding ``(B, F, P)`` for `backward`."""
    q = encode(bins, x)
    e = x[:, :, None] * weights.linear + weights.linear_bias + np.einsum("bfp,fpd->bfd", q, weights.ple)
    return e.reshape(x.shape[0], -1), q


def backward(x: Array, q: Array, d_embedded: Array) -> EmbeddingWeights:
    """Return the gradients of the weights given the gradient of the loss with respect to the embedded inputs."""
    b, f, _ = q.shape
    d = d_embedded.reshape(b, f, -1)
    return EmbeddingWeights((x[:, :, None] * d).sum(axis=0), d.sum(axis=0), np.einsum("bfp,bfd->fpd", q, d))


__all__ = ["EMBED_SALT", "Bins", "EmbeddingWeights", "backward", "bins_of", "encode", "fit_bins", "forward", "init"]
