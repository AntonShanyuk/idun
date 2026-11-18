"""Quantile bins for the 22 named features (docs/architecture/data-layout.md §2).

Each refit gets its own binning, fitted on a sample of its own training window so later rows never leak into the
edges (LightGBM: ``bin_construct_sample_cnt = 200000``, ``max_bin = 255``):

1. `sample_rows`: 200,000 row indices drawn without replacement from the window by ``PCG64([seed, refit])``.
2. `quantile_edges`: per feature, the distinct finite sample values. With at most 255 of them every value gets its own
   bin; otherwise cuts sit at the ``i / 255`` quantiles (by count) of the sample. Each upper edge is the float32
   midpoint between the two distinct values it separates, and never rounds onto the upper one.
3. `apply_bins`: a value goes to the first bin whose upper edge is ≥ the value; NaN goes to bin 255.

This is plain quantile binning, not LightGBM's ``BinMapper``.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from idun.data.layout import MAX_NAMED_BINS, MISSING_BIN

SAMPLE_ROWS = 200_000


def sample_rows(window_rows: int, size: int, *, seed: int, refit: int) -> NDArray[np.int64]:
    """Sorted, distinct row indices in ``[0, window_rows)``: all of them when ``size >= window_rows``."""
    if window_rows < 1 or size < 1:
        raise ValueError(f"window_rows and size must be positive, got {window_rows} and {size}")
    if size >= window_rows:
        return np.arange(window_rows, dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64([seed, refit]))
    return np.sort(generator.choice(window_rows, size=size, replace=False)).astype(np.int64)


def quantile_edges(values: NDArray[np.float32], max_bins: int = MAX_NAMED_BINS) -> NDArray[np.float32]:
    """Strictly increasing float32 upper edges (``n_bins - 1`` of them, ``n_bins <= max_bins``) for a sample.

    Non-finite sample values are ignored. With no finite value there is a single bin and no edge.
    """
    if values.dtype != np.float32:
        raise TypeError(f"values must be float32, got {values.dtype}")
    if max_bins < 1:
        raise ValueError(f"max_bins must be positive, got {max_bins}")
    distinct, counts = np.unique(values[np.isfinite(values)], return_counts=True)
    if distinct.size <= max_bins:
        cuts = np.arange(max(distinct.size - 1, 0))
    else:
        cumulative = np.cumsum(counts)
        total = int(cumulative[-1])
        quantiles = (total * np.arange(1, max_bins, dtype=np.int64) + max_bins - 1) // max_bins  # ceil(total * i / max)
        cuts = np.unique(np.searchsorted(cumulative, quantiles, side="left"))
        cuts = cuts[cuts < distinct.size - 1]
    lower, upper = distinct[cuts], distinct[cuts + 1]
    middle = ((lower.astype(np.float64) + upper.astype(np.float64)) / 2).astype(np.float32)
    return np.where(middle < upper, middle, lower)


def edge_table(edges: Sequence[NDArray[np.float32]], max_bins: int = MAX_NAMED_BINS) -> NDArray[np.float32]:
    """Stack per-feature edges into a ``(features, max_bins)`` table padded with +inf (so +inf ends every row)."""
    table = np.full((len(edges), max_bins), np.inf, np.float32)
    for row, feature_edges in enumerate(edges):
        if feature_edges.size >= max_bins:
            raise ValueError(f"feature {row} has {feature_edges.size} edges; at most {max_bins - 1} fit")
        table[row, : feature_edges.size] = feature_edges
    return table


def apply_bins(values: NDArray[np.float32], edges_row: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Bin of each value under one row of an `edge_table`: first edge ≥ value; NaN → `MISSING_BIN`."""
    if edges_row.shape != (MAX_NAMED_BINS,) or not np.isposinf(edges_row[-1]):
        raise ValueError("edges_row must be one +inf-padded row of an edge_table")
    bins = np.searchsorted(edges_row, values, side="left").astype(np.uint8)
    bins[np.isnan(values)] = MISSING_BIN
    return bins
