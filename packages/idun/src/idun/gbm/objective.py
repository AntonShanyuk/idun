"""Custom objectives: the gradient hook of the boosting loop (docs/architecture/determinism.md §2, §5).

An objective writes each training row's gradient ``dL/dpred`` (float32) on the host, every iteration. The gradients
steer the whole tree: they are quantized for the split search, and their exact fixed-point sums give the leaf values.
`Objective` is a loss with a unit hessian (leaf ``−Σg / (n + λ)``; the split search counts rows); `NewtonObjective`
writes its own hessians too, which the split search reads (second-order splits) and the leaves divide by
(``−Σg / (Σh + λ)``, one Newton step per leaf). `MulticlassObjective` (`Softmax`) writes every class's gradients and
hessians from every class's scores, for `train_multiclass`.

An objective is called with the training rows in order (row ``i`` of the arrays is training row ``rows.start + i``),
so it may hold per-row data of its own (weights, groups) aligned with them. It must write every row, in the bag or
out, with finite values, and must be a pure function of its arguments and that data: a fit is then reproducible bit
for bit.
"""

import math
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray


class Objective(Protocol):
    """A loss with a unit hessian: ``gradients`` writes ``dL/dpred`` per row into ``out``."""

    constant_hessian: Literal[True]

    def gradients(self, pred: NDArray[np.float32], y: NDArray[np.float32], out: NDArray[np.float32]) -> None:
        """Write the gradient of the loss at ``pred`` for targets ``y`` into ``out`` (float32)."""


class NewtonObjective(Protocol):
    """A loss with its own hessian: ``gradients`` writes ``dL/dpred`` into ``grad`` and ``d²L/dpred²`` into ``hess``."""

    constant_hessian: Literal[False]

    def gradients(
        self, pred: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        """Write the loss's gradient and hessian at ``pred`` for targets ``y`` (float32; hessians >= 0)."""


class MulticlassObjective(Protocol):
    """A loss over K classes' scores: ``gradients`` writes every class's gradient and hessian from all K scores."""

    n_classes: int

    def gradients(
        self, preds: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        """Write ``(K, n)`` gradients and hessians at the ``(K, n)`` scores ``preds`` for class ids ``y``."""


class Softmax:
    """LightGBM's ``multiclass``: the softmax of K raw scores per row; labels are class ids ``0 .. K − 1``.

    ``p = softmax(s)`` per row, ``g_k = p_k − [y = k]`` and ``h_k = K / (K − 1) · p_k (1 − p_k)`` (LightGBM's
    factor), computed in float64. `start` is the log of each class's share of the labels (weighted, with weights).
    """

    def __init__(self, n_classes: int) -> None:
        if n_classes < 2:
            raise ValueError(f"a softmax needs at least two classes, got {n_classes!r}")
        self.n_classes = n_classes

    def gradients(
        self, preds: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        """Write every class's gradient and hessian at the scores ``preds``."""
        k = self.n_classes
        shifted = preds.astype(np.float64) - preds.max(axis=0).astype(np.float64)  # exp stays finite
        p = np.exp(shifted)
        p /= p.sum(axis=0)
        grad[:] = p - (y[None, :] == np.arange(k, dtype=np.float32)[:, None])
        hess[:] = k / (k - 1) * p * (1.0 - p)

    def start(self, y: NDArray[np.float32], weight: NDArray[np.float32] | None = None) -> NDArray[np.float64]:
        """Return the log of each class's (weighted) share of ``y``: the constant scores that fit the labels best."""
        weights = None if weight is None else np.asarray(weight, np.float64)
        counts = np.bincount(np.asarray(y, np.int64), weights=weights, minlength=self.n_classes).astype(np.float64)
        if counts.size != self.n_classes or (counts <= 0.0).any():
            raise ValueError(f"a softmax start needs rows of every class (with positive weight), got {counts.tolist()}")
        shares: NDArray[np.float64] = np.log(counts / counts.sum())
        return shares


class LogLoss:
    """Binary log loss on raw scores (LightGBM's ``binary``): a `NewtonObjective` for labels 0 and 1.

    ``g = p − y`` and ``h = p (1 − p)`` with ``p = 1 / (1 + e^−s)``, computed in float64; `start` is the log-odds
    of the labels' mean, the score `train`'s ``boost_from_average`` starts from.
    """

    constant_hessian: Literal[False] = False

    def gradients(
        self, pred: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        """Write the loss's gradient and hessian at the scores ``pred``."""
        p = 1.0 / (1.0 + np.exp(-pred.astype(np.float64)))
        grad[:] = p - y
        hess[:] = p * (1.0 - p)

    @staticmethod
    def start(y: NDArray[np.float32], weight: NDArray[np.float32] | None = None) -> float:
        """Return the log-odds of ``y``'s mean (weighted by ``weight``): the constant score that fits them best."""
        if weight is None:
            mean = float(np.mean(y, dtype=np.float64))
        else:
            mean = float(np.average(np.asarray(y, np.float64), weights=np.asarray(weight, np.float64)))
        if not 0.0 < mean < 1.0:
            raise ValueError(f"log loss needs rows of both classes (with positive weight), the labels' mean is {mean}")
        return math.log(mean / (1.0 - mean))
