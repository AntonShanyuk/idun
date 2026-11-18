"""Gradient directions of a trained TabM, for the trees in front of which they go (JARF, Rauniyar 2025).

A network responds to combinations of its inputs; a tree splits one input at a time. The directions a trained TabM is
most sensitive to are the top eigenvectors of its expected gradient outer product (EGOP) ``H = E[g gᵀ]``, ``g`` the
gradient of the predicted expected class with respect to the inputs. Projected onto them, the inputs give the trees
oblique splits as ordinary features (`idun.jarf`'s feature front and `idun.jarf.JARFRegressor`).

The gradient runs on the float reference (`idun.tabm.reference`), the EGOP is a float64 sum in a fixed order (rows in
chunks of `CHUNK`, models in the given order), so it repeats its bits on one machine; ``eigh`` is LAPACK's and may
differ in its last bits between machines, which is why a `Directions` is computed once and stored with the model that
uses it. Single-target classifiers without a temporal encoder only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from idun.nn import inputs
from idun.tabm import reference

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from idun.nn.inputs import Data
    from idun.tabm.train import TabMModel

FORMAT = "idun-directions"
FORMAT_VERSION = 1
CHUNK = 512  # rows per gradient batch of `egop`
PROJECT_ROWS = 50_000  # rows per matmul of `Directions.project` (a fixed shape, so the float32 sums repeat)


def check_surrogate(model: TabMModel) -> None:
    """Raise ``ValueError`` unless ``model`` is a single-target classifier without a temporal encoder."""
    if model.targets != 1 or model.multi_target or model.params.objective != "cross_entropy":
        raise ValueError("the directions need a single-target TabM classifier (objective cross_entropy)")
    if model.sequence_inputs or model.weights.encoder is not None:
        raise ValueError("the directions need a TabM without a temporal encoder")


def input_gradients(model: TabMModel, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Return d(expected class)/d(inputs) ``(B, d)`` of ``model`` at the inputs ``x`` ``(B, d)``.

    The expected class is ``Σ_c c · mean_j softmax(o_j)_c``; its gradient runs back through the blocks as
    `reference.backward` runs the loss's (inference: no dropout), and the input's gradient is the members' sum, since
    every member reads ``x``. The arithmetic is in the dtype of the weights and ``x``.
    """
    check_surrogate(model)
    weights = model.weights
    if x.ndim != 2 or x.shape[1] != model.n_inputs:
        raise ValueError(f"x must be (rows, {model.n_inputs}), got {x.shape}")
    logits, cache = reference.forward(weights, x, None)
    k, c = logits.shape[1], logits.shape[2]
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    values = np.arange(c, dtype=p.dtype)
    expected = (p * values).sum(axis=-1, keepdims=True)
    d_logits = p * (values - expected) / k
    da = np.einsum("bjc,jwc->bjw", d_logits, weights.head)
    for i in reversed(range(len(weights.weight))):
        dt = da * (cache.z[i] > 0) * weights.s[i]
        da = (dt @ weights.weight[i]) * weights.r[i]
    gradient: NDArray[np.floating[Any]] = da.sum(axis=1)
    return gradient


def egop(models: Sequence[TabMModel], data: Data, rows: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    """Return the EGOP ``(d, d)``: the mean of ``g gᵀ`` over ``rows`` of ``data`` and ``models``, in a fixed order."""
    if not models:
        raise ValueError("need at least one model")
    d = inputs.n_inputs(data)
    for model in models:
        check_surrogate(model)
        if model.n_inputs != d:
            raise ValueError(f"the data has {d} inputs; a model expects {model.n_inputs}")
    index = np.asarray(rows, np.int64)
    if index.ndim != 1 or index.size == 0:
        raise ValueError("rows must be a non-empty 1-D array of row indices")
    h = np.zeros((d, d))
    for model in models:
        for start in range(0, index.size, CHUNK):
            x = inputs.dense_inputs(data, index[start : start + CHUNK])
            g = input_gradients(model, x).astype(np.float64)
            h += g.T @ g
    return h / (len(models) * index.size)


def sensitivity(h: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the EGOP's diagonal: each input's mean squared gradient, the per-input ranking."""
    return np.ascontiguousarray(np.diagonal(h))


def signed(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the columns of ``v`` each with its largest-magnitude entry positive."""
    pivot = v[np.abs(v).argmax(axis=0), np.arange(v.shape[1])]
    return v * np.sign(pivot)


def top_directions(h: NDArray[np.float64], k: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``h``'s top ``k`` eigenvectors as columns ``(d, k)``, signs fixed, and every eigenvalue, descending."""
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"h must be square, got {h.shape}")
    if not 1 <= k <= h.shape[0]:
        raise ValueError(f"k must be in 1..{h.shape[0]}, got {k}")
    eigenvalues, vectors = np.linalg.eigh(h)
    order = np.argsort(eigenvalues)[::-1]
    return signed(vectors[:, order[:k]]), eigenvalues[order]


def model_digest(model: TabMModel) -> str:
    """Return the sha256 of the model's weights (every array's bytes, in `reference.Weights.arrays` order)."""
    digest = hashlib.sha256()
    for array in model.weights.arrays():
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Directions:
    """The directions of a fitted TabM, stored once and carried by the model that uses them.

    ``vectors`` ``(d, k)`` float32 (orthonormal columns, largest eigenvalue first); ``eigenvalues`` every eigenvalue
    of the EGOP, largest first; ``surrogates`` the digests of the models the EGOP came from.
    """

    vectors: NDArray[np.float32]
    eigenvalues: NDArray[np.float64]
    surrogates: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.dtype != np.float32 or not np.isfinite(self.vectors).all():
            raise ValueError("vectors must be a finite float32 (n_inputs, k) matrix")
        if self.eigenvalues.shape != (self.vectors.shape[0],) or self.eigenvalues.dtype != np.float64:
            raise ValueError(f"eigenvalues must be float64 ({self.vectors.shape[0]},), got {self.eigenvalues.shape}")
        if not 1 <= self.k <= self.n_inputs:
            raise ValueError(f"k must be in 1..{self.n_inputs}, got {self.k}")

    @property
    def n_inputs(self) -> int:
        """d: the inputs the directions are over."""
        return int(self.vectors.shape[0])

    @property
    def k(self) -> int:
        """The number of directions."""
        return int(self.vectors.shape[1])

    @property
    def trace(self) -> float:
        """The EGOP's trace: the sum of every eigenvalue."""
        return float(self.eigenvalues.sum())

    @property
    def top_share(self) -> float:
        """The share of the EGOP's trace the k directions hold."""
        return float(self.eigenvalues[: self.k].sum() / self.eigenvalues.sum())

    @property
    def digest(self) -> str:
        """The sha256 of the vectors' bytes: the id a binning carries."""
        return hashlib.sha256(np.ascontiguousarray(self.vectors).tobytes()).hexdigest()

    def project(self, x: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        """Return ``x`` ``(n, d)`` projected on the directions, ``(n, k)`` float32, `PROJECT_ROWS` rows per matmul."""
        x32 = np.asarray(x, np.float32)
        if x32.ndim != 2 or x32.shape[1] != self.n_inputs:
            raise ValueError(f"x must be (rows, {self.n_inputs}), got {x32.shape}")
        out = np.empty((x32.shape[0], self.k), np.float32)
        for start in range(0, x32.shape[0], PROJECT_ROWS):
            out[start : start + PROJECT_ROWS] = x32[start : start + PROJECT_ROWS] @ self.vectors
        return out

    def save(self, path: str | Path) -> None:
        """Write the directions to ``path`` (an uncompressed ``.npz`` without object arrays)."""
        meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "surrogates": list(self.surrogates)}
        with Path(path).open("wb") as handle:
            np.savez(handle, allow_pickle=False, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
                     vectors=self.vectors, eigenvalues=self.eigenvalues)  # fmt: skip

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read directions written by `save`; anything else raises ``ValueError``."""
        with np.load(path, allow_pickle=False) as archive:
            if "meta" not in archive.files:
                raise ValueError(f"{path} is not a {FORMAT} file")
            meta: dict[str, Any] = json.loads(archive["meta"].tobytes())
            if meta.get("format") != FORMAT or meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"{path} is not a {FORMAT} v{FORMAT_VERSION} file")
            vectors, eigenvalues = archive["vectors"], archive["eigenvalues"]
        return cls(vectors, eigenvalues, tuple(meta["surrogates"]))


def fit_directions(
    models: Sequence[TabMModel], data: Data, *, rows: NDArray[np.integer[Any]], k: int = 16
) -> Directions:
    """Return the top ``k`` directions of ``models``' EGOP over ``rows`` of ``data``."""
    vectors, eigenvalues = top_directions(egop(models, data, rows), k)
    return Directions(vectors.astype(np.float32), eigenvalues, tuple(model_digest(m) for m in models))
