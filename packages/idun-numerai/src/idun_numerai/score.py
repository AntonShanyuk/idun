"""Numerai's per-era scores in NumPy.

- `ncorr(predictions, targets, neutralizers)`: NCORR, as Numerai's `numerai_tools.scoring.neutral_correlation` defines
  it: the prediction is tie-kept ranked and gaussianized, its least-squares fit on the era's neutralizer columns and a
  constant is removed, and the rest is correlated (Pearson) with the centred target. The fit is removed through an
  orthonormal basis of the centred neutralizers from their SVD, without the directions whose singular value is at most
  ``RCOND`` of the largest.
- `corr(predictions, targets)`: CORR, Numerai's `numerai_tools.scoring.numerai_corr`: the gaussianized prediction and
  the centred target, each raised to the power 1.5 with its sign kept, then Pearson.
- `per_era(score, era, *columns)`: one score per era of row-aligned columns, ``era`` sorted (a packed dataset's
  ``era``).

No pandas in the port: numerai-tools is the reference these are checked against, per era in the tests, not a
dependency. Each score takes one era's rows as arrays (float64 inside) and raises ``ValueError`` for mismatched shapes
or a value that is not finite, where numerai-tools would assert, and for a constant prediction or target, whose score
is undefined (numerai-tools asserts for NCORR's prediction and gives NaN otherwise).
"""

import itertools
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, rankdata

from idun_numerai.inputs import era_bounds

RCOND = 1e-6  # relative singular-value cutoff of the neutralizers' span (numerai-tools' least squares uses 1e-6)


def _column(values: ArrayLike, name: str, n: int | None = None) -> NDArray[np.float64]:
    array = np.asarray(values, np.float64)
    if array.ndim != 1 or array.size < 2 or (n is not None and array.size != n):
        raise ValueError(f"{name} must be one value per row of the era (at least 2 rows), got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} has a value that is not finite (NaN or inf)")
    if array.min() == array.max():
        raise ValueError(f"{name} is constant in the era: its score is undefined")
    return array


def _cosine(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return the correlation of two centred vectors (Python floats: a zero vector raises ``ZeroDivisionError``)."""
    return float(a @ b) / math.sqrt(float(a @ a) * float(b @ b))


def _pow15(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sign(x) * np.abs(x) ** 1.5


def gaussianized(predictions: ArrayLike) -> NDArray[np.float64]:
    """Return one era's ``predictions`` tie-kept ranked and gaussianized: ``Φ⁻¹((rank − ½) / n)``, ties averaged."""
    p = _column(predictions, "predictions")
    return np.asarray(norm.ppf((rankdata(p) - 0.5) / p.size), np.float64)


def ncorr(predictions: ArrayLike, targets: ArrayLike, neutralizers: ArrayLike) -> float:
    """Return the NCORR of one era: ``predictions`` and ``targets`` ``(n,)``, ``neutralizers`` ``(n, m)``."""
    g = gaussianized(predictions)
    t = _column(targets, "targets", g.size)
    n = np.asarray(neutralizers, np.float64)
    if n.ndim != 2 or n.shape[0] != g.size or n.shape[1] < 1:
        raise ValueError(f"neutralizers must be (n_rows, m) with n_rows {g.size} and m >= 1, got shape {n.shape}")
    if not np.isfinite(n).all():
        raise ValueError("neutralizers has a value that is not finite (NaN or inf)")
    u, sv, _ = np.linalg.svd(n - n.mean(axis=0), full_matrices=False)
    basis = u[:, sv > sv[0] * RCOND]
    g = g - g.mean()
    return _cosine(g - basis @ (basis.T @ g), t - t.mean())


def corr(predictions: ArrayLike, targets: ArrayLike) -> float:
    """Return the CORR of one era (Numerai's `numerai_corr`): ``predictions`` and ``targets`` ``(n,)``."""
    g = _pow15(gaussianized(predictions))
    t = _column(targets, "targets", g.size)
    t = _pow15(t - t.mean())
    return _cosine(g - g.mean(), t - t.mean())


def per_era(score: Callable[..., float], era: ArrayLike, *columns: ArrayLike) -> NDArray[np.float64]:
    """Return ``score`` of each era, in era order, on the rows of that era of each of ``columns`` (row-aligned)."""
    arrays = [np.asarray(c) for c in columns]
    if not arrays:
        raise ValueError("per_era needs at least one column")
    if any(a.shape[0] != arrays[0].shape[0] for a in arrays):
        raise ValueError(f"the columns must have the same rows, got {[a.shape for a in arrays]}")
    bounds = era_bounds(np.asarray(era), arrays[0].shape[0])
    return np.array([score(*(a[lo:hi] for a in arrays)) for lo, hi in itertools.pairwise(bounds)], np.float64)


__all__ = ["corr", "gaussianized", "ncorr", "per_era"]
