"""idun_gbm's feature front: a TabM's directions appended to binned data as named features (JARF), and kept subsets.

`extend` projects a packed dataset's dense inputs (`idun.nn.inputs.dense_inputs`, the networks' view of the rows) on
a `idun.tabm.Directions`, bins each projection at the quantiles ``1..254 / 255`` of the rows given for the edges (the
distinct ones; a value falls in the first bin whose edge is at or above it) and appends the ``k`` columns to the named
features; `keep` reduces the obfuscated features to a named subset. Both return a `Front`, a `BinnedData` whose
``binning`` id carries the directions' digest and the kept set, so a booster refuses data extended another way. The
trees then split on the projections as on any other feature.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from idun.data import binning
from idun.nn import inputs
from idun.tabm import directions as tabm_directions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from idun.data import PackedDataset
    from idun.gbm.features import BinnedData
    from idun.nn.inputs import Data
    from idun.tabm.directions import Directions

MAX_BINS = 255  # bins per projection: the edges are the distinct quantiles at 1..254 / 255
QUANTILES = np.arange(1, MAX_BINS) / MAX_BINS
PREFIX = "tabm_direction"


@dataclass
class Front:
    """Binned data with named features appended or obfuscated features dropped (`idun.gbm.BinnedData`).

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


def direction_names(k: int, prefix: str = PREFIX) -> tuple[str, ...]:
    """Return the names of ``k`` appended projections: ``{prefix}_0 .. {prefix}_{k-1}``."""
    return tuple(f"{prefix}_{j}" for j in range(k))


def project_rows(data: Data, found: Directions, rows: NDArray[np.integer]) -> NDArray[np.float32]:
    """Return the dense inputs of ``rows`` projected on ``found``, ``(len(rows), k)`` float32.

    Rows go through `idun.nn.inputs.dense_inputs` and `Directions.project` in chunks of
    `idun.tabm.directions.PROJECT_ROWS`, one matmul each, so the float32 sums repeat.
    """
    if found.n_inputs != inputs.n_inputs(data):
        raise ValueError(f"the data has {inputs.n_inputs(data)} inputs; the directions are over {found.n_inputs}")
    index = np.asarray(rows, np.int64)
    out = np.empty((index.size, found.k), np.float32)
    step = tabm_directions.PROJECT_ROWS
    for start in range(0, index.size, step):
        out[start : start + step] = found.project(inputs.dense_inputs(data, index[start : start + step]))
    return out


def quantile_cuts(values: NDArray[np.float32]) -> list[NDArray[np.float32]]:
    """Return each column's bin edges: its distinct quantiles at ``1..254 / 255`` (bin b holds values ≤ edge b)."""
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"values must be a (rows, columns) matrix with rows, got {values.shape}")
    return [np.unique(np.quantile(values[:, j], QUANTILES)).astype(np.float32) for j in range(values.shape[1])]


def bin_projections(projection: NDArray[np.float32], cuts: Sequence[NDArray[np.float32]]) -> NDArray[np.uint8]:
    """Return the bins ``(k, n)`` of a projection ``(n, k)``: the first edge at or above each value."""
    if projection.ndim != 2 or projection.shape[1] != len(cuts):
        raise ValueError(f"projection {projection.shape} does not fit {len(cuts)} columns of edges")
    bins = [np.searchsorted(c, projection[:, j], side="left") for j, c in enumerate(cuts)]
    return np.stack(bins).astype(np.uint8).reshape(len(cuts), projection.shape[0])


def apply(
    data: PackedDataset, found: Directions, edges: NDArray[np.float32], n_bins: Sequence[int], *, prefix: str = PREFIX
) -> Front:
    """Return ``data`` with its rows' projections on ``found`` binned at the given edges and appended as named features.

    ``edges`` ``(k, 255)``, +inf padded as a booster stores them (`Front.named_edges`' rows for the directions);
    ``n_bins`` the real bins per direction. `extend` fits the edges and calls this; a `JARFModel` calls it with the
    edges its booster was trained on.
    """
    names = direction_names(found.k, prefix)
    if set(names) & set(data.feature_names):
        raise ValueError(f"the data already has features named {prefix}_*")
    if edges.shape != (found.k, MAX_BINS) or len(n_bins) != found.k:
        raise ValueError(f"need {found.k} rows of {MAX_BINS} edges and {found.k} bin counts")
    every = np.arange(data.n_rows, dtype=np.int64)
    cuts = [edges[j, : n_bins[j] - 1] for j in range(found.k)]
    bins = bin_projections(project_rows(data, found, every), cuts)
    n_named = len(data.named_features)
    return Front(
        n_rows=data.n_rows,
        binning=f"{data.binning}+jarf:{found.digest[:16]}",
        feature_names=(*data.named_features, *names, *data.feature_names[n_named:]),
        named_features=(*data.named_features, *names),
        obfuscated_features=data.obfuscated_features,
        categories=data.categories,
        named_n_bins=(*data.named_n_bins, *(int(b) for b in n_bins)),
        named_edges=np.vstack([np.asarray(data.named_edges, np.float32), np.asarray(edges, np.float32)]),
        named_bins=np.vstack([np.asarray(data.named_bins), bins]),
        obfuscated=data.obfuscated,
        categorical=data.categorical,
        target=data.target,
    )


def extend(data: PackedDataset, found: Directions, *, edge_rows: NDArray[np.integer], prefix: str = PREFIX) -> Front:
    """Return ``data`` with its rows' projections on ``found`` appended as ``k`` named features.

    The edges come from the projections of ``edge_rows`` (a sample of the training rows); every row is binned (`apply`).
    """
    cuts = quantile_cuts(project_rows(data, found, edge_rows))
    return apply(data, found, binning.edge_table(cuts), [c.size + 1 for c in cuts], prefix=prefix)


def keep(data: BinnedData, obfuscated: Sequence[str]) -> Front:
    """Return ``data`` with its obfuscated features reduced to ``obfuscated`` (names; the data's order is kept)."""
    wanted = list(obfuscated)
    if len(set(wanted)) != len(wanted):
        raise ValueError("a feature is named twice")
    unknown = [name for name in wanted if name not in data.obfuscated_features]
    if unknown:
        raise ValueError(f"not obfuscated features of the data: {unknown}")
    index = [i for i, name in enumerate(data.obfuscated_features) if name in set(wanted)]
    names = tuple(data.obfuscated_features[i] for i in index)
    n_named, n_obfuscated = len(data.named_features), len(data.obfuscated_features)
    identity = hashlib.sha256(repr(names).encode()).hexdigest()[:16]
    return Front(
        n_rows=data.n_rows,
        binning=f"{data.binning}+keep:{identity}",
        feature_names=(*data.named_features, *names, *data.feature_names[n_named + n_obfuscated :]),
        named_features=data.named_features,
        obfuscated_features=names,
        categories=data.categories,
        named_n_bins=data.named_n_bins,
        named_edges=data.named_edges,
        named_bins=data.named_bins,
        obfuscated=np.ascontiguousarray(np.asarray(data.obfuscated)[index]),
        categorical=data.categorical,
        target=data.target,
    )
