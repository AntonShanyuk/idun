"""HL-Gauss (Imani & White 2018): regression as classification over uniform bins, with Gaussian-smoothed labels.

A target z, clamped to ``[low, high]``, gets the mass of N(z, sigma) in each of ``n_bins`` uniform bins of that
interval, renormalised over the bins: ``cdf = Φ((edges − z) / sigma); p = diff(cdf) / (cdf[-1] − cdf[0])``. The
network's ``n_bins`` outputs are trained on these soft labels with cross-entropy, and its prediction is the expected
bin centre under its softmax. Φ(x) = ½ erfc(−x / √2), in float64 on the host: both backends train on the same labels
(Metal gets them as float32), and Metal's shading language has no ``erf``.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from idun._jit import njit


def edges(n_bins: int, low: float, high: float) -> NDArray[np.float64]:
    """Return the ``n_bins + 1`` bin edges, ``linspace(low, high, n_bins + 1)``."""
    if n_bins < 1 or not low < high:
        raise ValueError(f"need n_bins >= 1 and low < high, got {n_bins}, {low}, {high}")
    return np.linspace(low, high, n_bins + 1)


def centres(n_bins: int, low: float, high: float) -> NDArray[np.float64]:
    """Return the bins' midpoints, the values the prediction averages."""
    e = edges(n_bins, low, high)
    return 0.5 * (e[1:] + e[:-1])


@njit()
def _fill(z: NDArray[np.float64], e: NDArray[np.float64], sigma: float, out: NDArray[np.float64]) -> None:
    low, high, scale = e[0], e[-1], 1.0 / (sigma * math.sqrt(2.0))
    n_bins = e.size - 1
    for i in range(z.size):
        zi = min(max(z[i], low), high)
        below = 0.5 * math.erfc(-(e[0] - zi) * scale)
        first = below
        for c in range(n_bins):
            above = 0.5 * math.erfc(-(e[c + 1] - zi) * scale)
            out[i, c] = above - below
            below = above
        total = below - first
        for c in range(n_bins):
            out[i, c] /= total


def labels(z: ArrayLike, n_bins: int, low: float, high: float, sigma: float) -> NDArray[np.float64]:
    """Return the ``(len(z), n_bins)`` soft labels of targets ``z``; each row sums to 1.

    Raises ``ValueError`` for a target that is not finite or a sigma that is not positive.
    """
    values = np.ascontiguousarray(z, np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("HL-Gauss targets must be finite")
    if not sigma > 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    out = np.empty((values.size, n_bins), np.float64)
    _fill(values, edges(n_bins, low, high), float(sigma), out)
    return out
