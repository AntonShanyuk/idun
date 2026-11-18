"""The networks' inputs and labels from a packed dataset or a matrix.

A packed dataset's inputs per row, in this order: the numeric columns as float32 with NaN set to 0; the 4-bit codes
mapped to ``(code − 2) / 2`` ∈ {−1, −½, 0, ½, 1} (or their raw values); the categorical column one-hot over its levels.

Labels are given, never derived from a target: a packed dataset's classes come as a compact ``(n_rows, 1)`` uint8
class array (`class_array`, `class_labels`), its regression target as an ``(n_rows, 1)`` float32 target array
(`target_array`, `checked_targets`), and `MatrixData` carries its own (the generic interface's input source).
"""

import mmap
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import prange
from numpy.typing import NDArray

from idun._jit import njit
from idun.data import PackedDataset
from idun.metal.runtime import PAGE_BYTES

MISSING_CLASS = 255  # in a compact class array: no label on this row (never allowed on a training row)
CODES_BLOCK_ROWS = 4096  # rows per parallel task of `codes_row_major`
CHECK_ROWS = 1 << 20  # training rows per step of `checked_targets` (no K-wide float64 copy of every training row)


@dataclass(frozen=True)
class MatrixData:
    """A network's rows as a matrix: float32 inputs ``x`` and a label per row.

    The label is a class 0..n_classes − 1 (int64), or with ``n_classes`` 1 a regression target (float32).
    """

    x: NDArray[np.float32]  # (n_rows, n_inputs), finite
    y: NDArray[np.int64] | NDArray[np.float32]  # (n_rows,)
    n_classes: int

    def __post_init__(self) -> None:
        if self.x.ndim != 2 or self.x.dtype != np.float32 or not np.isfinite(self.x).all():
            raise ValueError("x must be a finite float32 (n_rows, n_inputs) matrix")
        if self.n_classes == 1:
            if self.y.shape != (self.x.shape[0],) or self.y.dtype != np.float32 or not np.isfinite(self.y).all():
                raise ValueError(f"y must be finite float32 targets, one per row ({self.x.shape[0]})")
            return
        if self.y.shape != (self.x.shape[0],) or self.y.dtype != np.int64:
            raise ValueError(f"y must be int64 classes, one per row ({self.x.shape[0]})")
        if self.y.size and not (int(self.y.min()) >= 0 and int(self.y.max()) < self.n_classes):
            raise ValueError(f"the classes must be 0..{self.n_classes - 1}")

    @property
    def n_rows(self) -> int:
        """Rows of the matrix."""
        return int(self.x.shape[0])


Data = PackedDataset | MatrixData


def n_inputs(data: Data) -> int:
    """Return the inputs per row: named factors + obfuscated features + countries (1,086 for the v6 cache)."""
    if isinstance(data, MatrixData):
        return int(data.x.shape[1])
    return len(data.named_features) + len(data.obfuscated_features) + len(data.countries)


def dense_inputs(data: Data, rows: NDArray[np.int64], *, raw_codes: bool = False) -> NDArray[np.float32]:
    """Return the ``(len(rows), n_inputs)`` float32 input matrix of ``rows`` (absolute row indices, any order).

    ``raw_codes``: the obfuscated codes as their values 0..4 (for RealMLP, which scales its inputs itself) instead
    of TabM's ``(code − 2) / 2``.
    """
    if isinstance(data, MatrixData):
        return np.ascontiguousarray(data.x[np.asarray(rows, np.int64)])
    rows = np.asarray(rows, np.int64)
    if rows.ndim != 1 or (rows.size and not (rows.min() >= 0 and rows.max() < data.n_rows)):
        raise ValueError(f"rows must be a 1-D array of row indices in [0, {data.n_rows})")
    named = np.nan_to_num(np.asarray(data.named_values[:, rows], np.float32).T, nan=0.0)
    packed = np.asarray(data.obfuscated[:, rows >> 1])
    codes = np.where((rows & 1) == 1, packed >> 4, packed & 0x0F)
    obfuscated = codes.T.astype(np.float32) if raw_codes else (codes.T.astype(np.float32) - 2.0) / 2.0
    country = np.zeros((rows.size, len(data.countries)), np.float32)
    country[np.arange(rows.size), np.asarray(data.country[rows], np.int64)] = 1.0
    return np.concatenate([named, obfuscated, country], axis=1)


def labels(data: MatrixData, rows: NDArray[np.int64]) -> NDArray[Any]:
    """Return the labels of a `MatrixData`'s ``rows``; a packed dataset's classes are passed as a class array."""
    if not isinstance(data, MatrixData):
        raise TypeError("a packed dataset carries no labels: pass its classes (an (n_rows, K) uint8 class array)")
    return data.y[np.asarray(rows, np.int64)]


def _mapped(n_rows: int, n_targets: int, dtype: type[np.generic]) -> NDArray[Any]:
    """Return an ``(n_rows, n_targets)`` array in anonymous memory: page-aligned, its last page mapped."""
    if n_rows < 1 or n_targets < 1:
        raise ValueError(f"need at least one row and one target, got {n_rows} x {n_targets}")
    count = n_rows * n_targets
    nbytes = count * np.dtype(dtype).itemsize
    region = mmap.mmap(-1, -(-nbytes // PAGE_BYTES) * PAGE_BYTES)
    return np.frombuffer(region, dtype, count=count).reshape(n_rows, n_targets)


def class_array(n_rows: int, n_targets: int) -> NDArray[np.uint8]:
    """Return an ``(n_rows, n_targets)`` uint8 array of `MISSING_CLASS` that Metal can wrap without a copy.

    Anonymous memory, so page-aligned with its last page mapped; the memory is returned when the array (and every
    buffer wrapping it) is gone.
    """
    out: NDArray[np.uint8] = _mapped(n_rows, n_targets, np.uint8)
    out[...] = MISSING_CLASS
    return out


def target_array(n_rows: int, n_targets: int) -> NDArray[np.float32]:
    """Return an ``(n_rows, n_targets)`` float32 array of NaN (no target) that Metal can wrap without a copy.

    The regression counterpart of `class_array`: row r's target at ``[r, 0]``, in anonymous memory.
    """
    out: NDArray[np.float32] = _mapped(n_rows, n_targets, np.float32)
    out[...] = np.nan
    return out


def check_label_source(data: Data, objective: str, classes: object, targets: object) -> None:
    """Raise ``ValueError`` unless the labels given suit ``data`` and the training objective.

    A `MatrixData` carries its own labels and takes neither array; a packed dataset takes ``classes`` for
    ``"cross_entropy"`` and ``targets`` for the regression objectives (``"mse"``, TabM's ``"hl_gauss"``), never both.
    """
    if isinstance(data, MatrixData):
        if classes is not None or targets is not None:
            raise ValueError("a MatrixData carries its own labels and takes no classes or targets")
        return
    if objective in ("mse", "hl_gauss"):
        if targets is None or classes is not None:
            raise ValueError(f"a packed dataset needs its targets for objective {objective!r} (an (n_rows, K) float32 "
                             "target array), not classes")  # fmt: skip
    elif classes is None or targets is not None:
        raise ValueError(f"a packed dataset needs its classes for objective {objective!r} (an (n_rows, K) uint8 class "
                         "array), not targets")  # fmt: skip


def checked_targets(targets: NDArray[np.float32], n_rows: int, lo: int, hi: int) -> NDArray[np.float32]:
    """Return ``targets`` as an ``(n_rows, 1)`` float32 target array without a copy, its training rows checked.

    Raises ``ValueError`` for another shape or dtype, or unless every target of the training rows ``[lo, hi)`` is
    finite (NaN may mark rows outside them), checked `CHECK_ROWS` rows at a time.
    """
    array = np.asarray(targets)
    if array.ndim != 2 or array.shape[0] != n_rows or array.shape[1] < 1 or array.dtype != np.float32:
        raise ValueError(f"targets must be an ({n_rows}, K) float32 array with K >= 1 (target k of row r at [r, k]), "
                         f"got a {array.dtype} array of shape {array.shape}")  # fmt: skip
    for start in range(lo, hi, CHECK_ROWS):
        if not np.isfinite(array[start : min(start + CHECK_ROWS, hi)]).all():
            raise ValueError("every target of the training rows must be finite (NaN may mark only other rows)")
    return array


def class_labels(classes: NDArray[np.uint8], rows: NDArray[np.int64], n_classes: int) -> NDArray[np.int64]:
    """Return the classes of ``rows``, ``(len(rows), 1)``, from a compact ``(n_rows, 1)`` uint8 class array.

    Row r's class is at ``[r, 0]``: 0..n_classes − 1, or `MISSING_CLASS` where row r has no label. Raises
    ``ValueError`` for another shape or dtype, or unless every class of ``rows`` is below ``n_classes``.
    """
    classes = np.asarray(classes)
    if classes.ndim != 2 or classes.shape[1] < 1 or classes.dtype != np.uint8:
        raise ValueError(f"classes must be an (n_rows, K) uint8 array with K >= 1, got a {classes.dtype} array of "
                         f"shape {classes.shape}")  # fmt: skip
    out = classes[np.asarray(rows, np.int64)]
    if out.size and int(out.max()) >= n_classes:
        raise ValueError(f"the classes of these rows are not 0..{n_classes - 1} ({MISSING_CLASS}: a missing label)")
    return out.astype(np.int64)


def codes_row_major(data: PackedDataset, out: NDArray[np.uint8]) -> None:
    """Write ``data``'s obfuscated codes row-major into ``out``, an ``(n_rows, ld)`` uint8 array of zeros.

    Two codes per byte: feature ``2j`` in the low nibble of byte ``j``, ``2j + 1`` in the high one. (The cache keeps
    them feature-major, two rows per byte, so a row's codes lie in as many cache lines as there are features.)
    """
    n_obf = len(data.obfuscated_features)
    if out.dtype != np.uint8 or out.ndim != 2 or out.shape[0] != data.n_rows or out.shape[1] < (n_obf + 1) // 2:
        raise ValueError(f"out must be a uint8 array of {data.n_rows} rows of at least {(n_obf + 1) // 2} bytes")
    _transpose_codes(data.obfuscated, data.n_rows, CODES_BLOCK_ROWS, out)


@njit(parallel=True)
def _transpose_codes(obf: NDArray[np.uint8], n_rows: int, block: int, out: NDArray[np.uint8]) -> None:
    for task in prange((n_rows + block - 1) // block):
        lo = task * block
        hi = min(lo + block, n_rows)
        for feature in range(obf.shape[0]):  # the even feature of a byte before the odd one
            column, j, shift = obf[feature], feature >> 1, 4 * (feature & 1)
            for row in range(lo, hi):
                code = (column[row >> 1] >> (4 * (row & 1))) & 0x0F
                out[row, j] = code if shift == 0 else out[row, j] | (code << 4)


__all__ = [
    "CHECK_ROWS",
    "CODES_BLOCK_ROWS",
    "MISSING_CLASS",
    "MatrixData",
    "check_label_source",
    "checked_targets",
    "class_array",
    "class_labels",
    "codes_row_major",
    "dense_inputs",
    "labels",
    "n_inputs",
    "target_array",
]
