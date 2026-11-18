"""What the booster needs to know about the binned data, and how features map onto histogram bins.

`BinnedData` is the structural interface the booster and backends read: `idun.data.PackedDataset` satisfies it, and
tests satisfy it with small in-memory arrays. Features are in this order: named (uint8 bins), obfuscated (4-bit codes),
categorical; any kind may have no columns.

`take_rows` copies chosen rows of any `BinnedData` into memory as a `BinnedRows` under the same binning id: a row
subset, a row sample, or row copies as integer weights, for a fit whose rows are not one contiguous slice.
"""

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray

NAMED = 0  # uint8 bins 0..254, 255 = missing
OBFUSCATED = 1  # 4-bit codes 0..4, two rows per byte
CATEGORICAL = 2  # uint8 category codes

NAMED_HIST_BINS = 256  # 255 real bins + the missing bin
MISSING_BIN = 255
OBFUSCATED_BINS = 5
MAX_CATEGORIES = 64  # a categorical split is a uint64 bitmask
TAKE_COLUMNS = 64  # obfuscated columns unpacked at a time by `take_rows`


class BinnedData(Protocol):
    """Binned feature columns of a panel (column-major), plus the names that tie a model to its binning."""

    n_rows: int
    binning: str
    feature_names: tuple[str, ...]
    named_features: tuple[str, ...]
    obfuscated_features: tuple[str, ...]
    categories: tuple[tuple[str, ...], ...]  # each categorical column's levels
    named_n_bins: tuple[int, ...]
    named_edges: NDArray[np.float32]  # (n_named, 255)
    named_bins: NDArray[np.uint8]  # (n_named, n_rows)
    obfuscated: NDArray[np.uint8]  # (n_obfuscated, ceil(n_rows / 2))
    categorical: NDArray[np.uint8]  # (n_categorical, n_rows), rows contiguous (a stride of 0 for one column is fine)
    target: NDArray[np.float32]  # (n_rows,)


@dataclass(frozen=True)
class FeatureLayout:
    """Per feature (loader order): its kind, its column within that kind, and its histogram and real bin counts."""

    kind: NDArray[np.int8]
    column: NDArray[np.int32]
    hist_bins: NDArray[np.int32]  # histogram cells: 256 named, 5 obfuscated, one per category
    real_bins: NDArray[np.int32]  # bins a split can separate: the binning's count, 5, or the category count

    @classmethod
    def of(cls, data: BinnedData) -> Self:
        """Lay out ``data``'s features; raises ``ValueError`` if a count does not fit the model format."""
        n_named, n_obfuscated = len(data.named_features), len(data.obfuscated_features)
        levels = [len(column) for column in data.categories]
        if len(data.feature_names) != n_named + n_obfuscated + len(levels):
            raise ValueError("feature_names must be the named, then the obfuscated, then the categorical features")
        if not all(1 <= n <= MAX_CATEGORIES for n in levels):
            raise ValueError(f"{levels} categories; a categorical split holds 1..{MAX_CATEGORIES} per column")
        if data.categorical.shape != (len(levels), data.n_rows):
            raise ValueError(f"categorical must be (n_categorical, n_rows) = ({len(levels)}, {data.n_rows})")
        if len(data.named_n_bins) != n_named or not all(1 <= b <= MISSING_BIN for b in data.named_n_bins):
            raise ValueError(f"named_n_bins must give 1..{MISSING_BIN} real bins per named feature")
        counts = [n_named, n_obfuscated, len(levels)]
        kind = np.repeat(np.array([NAMED, OBFUSCATED, CATEGORICAL], np.int8), counts)
        column = np.concatenate([np.arange(n) for n in counts]).astype(np.int32)
        hist_bins = np.concatenate([[NAMED_HIST_BINS] * n_named, [OBFUSCATED_BINS] * n_obfuscated, levels]).astype(
            np.int32
        )
        real_bins = np.concatenate([data.named_n_bins, [OBFUSCATED_BINS] * n_obfuscated, levels]).astype(np.int32)
        return cls(kind=kind, column=column, hist_bins=hist_bins, real_bins=real_bins)

    @property
    def n_features(self) -> int:
        """Number of features."""
        return int(self.kind.size)

    def offsets(self, features: NDArray[np.int32]) -> tuple[NDArray[np.int32], int]:
        """First histogram cell of each feature in ``features``, and the cells they need in total."""
        sizes = self.hist_bins[features]
        offsets = np.zeros(features.size, np.int32)
        np.cumsum(sizes[:-1], out=offsets[1:])
        return offsets, int(sizes.sum())


@dataclass
class BinnedRows:
    """Rows of binned data held in memory (`BinnedData`), under the binning id of the data they were taken from.

    A plain dataclass, not frozen: the protocol's members are settable attributes.
    """

    n_rows: int
    binning: str
    feature_names: tuple[str, ...]
    named_features: tuple[str, ...]
    obfuscated_features: tuple[str, ...]
    categories: tuple[tuple[str, ...], ...]
    named_n_bins: tuple[int, ...]
    named_edges: NDArray[np.float32]
    named_bins: NDArray[np.uint8]
    obfuscated: NDArray[np.uint8]
    categorical: NDArray[np.uint8]
    target: NDArray[np.float32]


def take_rows(data: BinnedData, rows: NDArray[np.integer]) -> BinnedRows:
    """Return the rows ``rows`` of ``data``, in the order given, as binned data in memory.

    ``rows`` is a 1-D integer array of row indices in ``[0, data.n_rows)``: any order, and a row named twice is there
    twice (integer row weights are row copies). The features, their names, edges and categories and the ``binning`` id
    are ``data``'s, so a booster trained on the result predicts ``data`` and any data binned the same way; ``target``
    holds the rows' targets. The result is a dataset of its own: its rows are numbered from 0, and a fit's per-row
    draws go by a row's number in the data it trains on. Raises ``ValueError`` for rows that are not such an array or
    name no row.
    """
    index = np.asarray(rows)
    if index.ndim != 1 or index.size == 0 or not np.issubdtype(index.dtype, np.integer):
        raise ValueError(
            "rows must be a 1-D integer array naming at least one row (np.flatnonzero turns a mask into one)"
        )
    index = index.astype(np.int64)
    if index.min() < 0 or index.max() >= data.n_rows:
        raise ValueError(f"rows must lie in [0, {data.n_rows}); got {int(index.min())}..{int(index.max())}")
    m = index.size
    half, shift = index >> 1, ((index & 1) << 2).astype(np.uint8)  # a row's byte, and its nibble's shift
    packed = np.zeros((len(data.obfuscated_features), (m + 1) // 2), np.uint8)
    for lo in range(0, packed.shape[0], TAKE_COLUMNS):
        codes = (np.asarray(data.obfuscated[lo : lo + TAKE_COLUMNS])[:, half] >> shift) & 0x0F
        packed[lo : lo + TAKE_COLUMNS] = codes[:, 0::2]
        packed[lo : lo + TAKE_COLUMNS, : m // 2] |= codes[:, 1::2] << 4
    return BinnedRows(
        n_rows=m,
        binning=data.binning,
        feature_names=tuple(data.feature_names),
        named_features=tuple(data.named_features),
        obfuscated_features=tuple(data.obfuscated_features),
        categories=tuple(data.categories),
        named_n_bins=tuple(data.named_n_bins),
        named_edges=data.named_edges,
        named_bins=np.ascontiguousarray(np.asarray(data.named_bins)[:, index]),
        obfuscated=packed,
        categorical=np.ascontiguousarray(np.asarray(data.categorical)[:, index]),
        target=np.asarray(data.target, np.float32)[index],
    )
