"""Numerai's labels: the 5 classes of a target, ``rint(4 · y)``.

Numerai's targets take the values 0, 0.25, 0.5, 0.75 and 1, so a network can train on them as 5 classes. `classes`
turns a target into the compact class array that `idun.tabm.train_tabm` and `idun.realmlp.train_realmlp` take, and
`expected` turns class probabilities back into a prediction, the expected class. `per_era_z` standardizes a target
within each era, a regression label for the ``"mse"`` objective.
"""

import itertools

import numpy as np
from numpy.typing import NDArray

from idun.nn.inputs import MISSING_CLASS, class_array, target_array

N_CLASSES = 5
BLOCK_ROWS = 1 << 20  # rows per conversion step: no float64 copy of a whole memory-mapped target


def classes(targets: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Return the ``(n_rows, 1)`` class array of a target ``(n_rows,)`` (more targets: experimental, not documented).

    Class ``rint(4 · y)``; a missing target (NaN) is `idun.nn.inputs.MISSING_CLASS`, and any other value than the 5
    bins raises ``ValueError``. The array is page-aligned (`idun.nn.inputs.class_array`), so Metal wraps it without a
    copy.
    """
    array = np.asarray(targets)
    rows = array[None, :] if array.ndim == 1 else array
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError(f"targets must be (n_rows,) or (K, n_rows) with K >= 1, got shape {array.shape}")
    out = class_array(rows.shape[1], rows.shape[0])
    for start in range(0, rows.shape[1], BLOCK_ROWS):
        block = np.asarray(rows[:, start : start + BLOCK_ROWS], np.float64)
        scaled = 4.0 * block
        binned = np.rint(scaled)
        missing = np.isnan(block)
        valid = (np.abs(scaled - binned) < 1e-6) & (binned >= 0) & (binned < N_CLASSES)
        if not (valid | missing).all():
            raise ValueError("the targets are not the 5 bins 0, 0.25, ..., 1 (or NaN for a missing target)")
        out[start : start + block.shape[1]] = np.where(missing, MISSING_CLASS, binned).T.astype(np.uint8)
    return out


def expected(proba: NDArray[np.floating]) -> NDArray[np.float32]:
    """Return the expected class ``Σ_c c · p_c`` of class probabilities ``(n, N_CLASSES)``."""
    p = np.asarray(proba, np.float64)
    if p.ndim != 2 or p.shape[1] != N_CLASSES:
        raise ValueError(f"need (n, {N_CLASSES}) class probabilities, got shape {p.shape}")
    return (p @ np.arange(N_CLASSES, dtype=np.float64)).astype(np.float32)


def era_bounds(era: NDArray[np.integer], n_rows: int) -> NDArray[np.int64]:
    """Return the row bounds ``[0, ..., n_rows]`` of the eras of a sorted ``era`` (one value per row; each era one run).

    ``ValueError`` unless ``era`` has ``n_rows`` >= 1 values in sorted order.
    """
    keys = np.asarray(era)
    if n_rows < 1 or keys.shape != (n_rows,):
        raise ValueError(f"era must have one value per row ({n_rows}), got shape {keys.shape}")
    if keys.size > 1 and (np.diff(keys) < 0).any():
        raise ValueError("era must be sorted, so that each era is one run of rows")
    return np.concatenate([[0], np.flatnonzero(keys[1:] != keys[:-1]) + 1, [keys.size]]).astype(np.int64)


def per_era_z(targets: NDArray[np.floating], era: NDArray[np.integer]) -> NDArray[np.float32]:
    """Return the per-era z of a target ``(n_rows,)``, an ``(n_rows, 1)`` target array (more: experimental).

    Within each era, over its rows with a target: ``(y − mean) / sd`` in float64 (sd with ddof 0), 0 where the sd is
    0, then float32; a missing target (NaN) stays NaN. ``era``: one value per row, sorted (a packed dataset's
    ``era``), so each era is one run of rows. The array is page-aligned (`idun.nn.inputs.target_array`), so Metal wraps
    it without a copy.
    """
    array = np.asarray(targets)
    rows = array[None, :] if array.ndim == 1 else array
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError(f"targets must be (n_rows,) or (K, n_rows) with K >= 1, got shape {array.shape}")
    out = target_array(rows.shape[1], rows.shape[0])
    for lo, hi in itertools.pairwise(era_bounds(era, rows.shape[1])):
        for k in range(rows.shape[0]):
            values = np.asarray(rows[k, lo:hi], np.float64)
            present = ~np.isnan(values)
            if not present.any():
                continue
            v = values if present.all() else values[present]
            sd = v.std()
            z = (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)
            if present.all():
                out[lo:hi, k] = z
            else:
                out[lo:hi, k][present] = z
    return out


__all__ = ["BLOCK_ROWS", "N_CLASSES", "classes", "era_bounds", "expected", "per_era_z"]
