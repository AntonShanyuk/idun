"""The training loop's pieces both networks share: the epoch order, the fit report and the softmax.

The epoch order: each epoch a fresh permutation of the training rows from ``PCG64(SeedSequence([seed, epoch,
ORDER_SALT]))``.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from idun.nn.inputs import MatrixData

ORDER_SALT = 2


def epoch_order(seed: int, epoch: int, n: int) -> NDArray[np.int64]:
    """Return the order in which epoch ``epoch`` visits the ``n`` training rows (positions 0 .. n - 1)."""
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch, ORDER_SALT])))
    return generator.permutation(n).astype(np.int64)


@dataclass
class FitReport:
    """Per-epoch mean losses and wall-clock seconds of a fit, and its total.

    With a validation set, ``valid_score`` is its score after each epoch and ``best_epoch`` the epoch (from 0) whose
    weights the model kept.
    """

    epoch_loss: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    seconds: float = 0.0
    valid_score: list[float] = field(default_factory=list)
    best_epoch: int | None = None


@dataclass(frozen=True)
class Validation:
    """A network's validation rows: ``rows`` (contiguous) of ``data``, a `MatrixData` with their labels or targets.

    After every epoch the network scores them, lower being better: the error rate of the most probable class
    (classification), or the mean squared error of the prediction (regression, on the targets as given).
    """

    data: "MatrixData"
    rows: slice

    def bounds(self) -> tuple[int, int]:
        """Return the rows as ``(lo, hi)``; a range that is empty or not contiguous raises ``ValueError``."""
        lo, hi, _ = self.rows.indices(self.data.n_rows)
        if self.rows.step not in (None, 1) or not 0 <= lo < hi:
            raise ValueError(f"validation rows must be a contiguous non-empty range, got {self.rows}")
        return lo, hi


def score(found: NDArray[np.floating], labels: NDArray[Any], n_classes: int) -> float:
    """Return a validation score: the error rate of ``found`` probabilities ``(n, c)``, or the MSE of ``found``."""
    if n_classes == 1:
        d = np.asarray(found, np.float64).reshape(-1) - np.asarray(labels, np.float64)
        return float(np.mean(d * d))
    return float(np.mean(np.argmax(found, axis=1) != labels))


def softmax(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return the softmax over the last axis, in float64 (the networks' class probabilities from their logits)."""
    z = np.asarray(logits, np.float64)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def checked_weights(weight: Any, n: int) -> NDArray[np.float32] | None:
    """Return ``n`` training rows' weights as float32: one per row, finite, non-negative, not all zero (or None)."""
    if weight is None:
        return None
    weights = np.ascontiguousarray(weight, np.float32)
    if weights.shape != (n,):
        raise ValueError(f"weight must hold one weight per training row ({n}), got shape {weights.shape}")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    if not (weights > 0).any():
        raise ValueError("every sample weight is zero")
    return weights


__all__ = ["ORDER_SALT", "FitReport", "Validation", "epoch_order", "score", "softmax"]
