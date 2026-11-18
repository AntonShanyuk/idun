"""Tiny in-memory binned panels for unit tests: they satisfy `idun.gbm.BinnedData` without a cache on disk."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from idun.data.layout import pack_nibbles


@dataclass
class TinyData:
    """A binned panel held in memory: named bins, packed obfuscated codes, categorical columns (the country first)."""

    n_rows: int
    binning: str
    feature_names: tuple[str, ...]
    named_features: tuple[str, ...]
    obfuscated_features: tuple[str, ...]
    named_n_bins: tuple[int, ...]
    named_edges: NDArray[np.float32]
    named_bins: NDArray[np.uint8]
    obfuscated: NDArray[np.uint8]
    categories: tuple[tuple[str, ...], ...]
    categorical: NDArray[np.uint8]  # (n_categorical, n_rows)
    country: NDArray[np.uint8]  # the first categorical column's codes (zeros without one)
    target: NDArray[np.float32]
    codes: NDArray[np.uint8]  # unpacked obfuscated codes (n_obfuscated, n_rows), for oracles


def make_tiny(
    n_rows: int = 300,
    *,
    n_named: int = 2,
    n_obfuscated: int = 4,
    n_countries: int = 5,
    more_categorical: tuple[int, ...] = (),
    named_bins: int = 7,
    missing: float = 0.1,
    seed: int = 0,
) -> TinyData:
    """A random panel whose target depends on a few features, so trees have something to find.

    The country is the first categorical column (``n_countries`` 0 leaves it out); ``more_categorical`` adds columns
    with those level counts, drawn after every other array, so the panels without them keep their values.
    """
    rng = np.random.default_rng(seed)
    named = rng.integers(0, named_bins, size=(n_named, n_rows)).astype(np.uint8)
    named[rng.random((n_named, n_rows)) < missing] = 255
    codes = rng.integers(0, 5, size=(n_obfuscated, n_rows)).astype(np.uint8)
    country = rng.integers(0, max(n_countries, 1), size=n_rows).astype(np.uint8)
    first = named[0] if n_named else np.zeros(n_rows, np.uint8)
    signal = (
        (codes[0].astype(np.float64) / 4 if n_obfuscated else 0.0)
        + 0.3 * (country % 2)
        + np.where(first == 255, 0.2, first / named_bins)
    )
    target = np.clip(np.round((signal / 2 + rng.normal(0, 0.15, n_rows)) * 4) / 4, 0, 1).astype(np.float32)
    extra = [rng.integers(0, n, size=n_rows).astype(np.uint8) for n in more_categorical]
    target = (target + sum(0.1 * (column % 2) for column in extra)).astype(np.float32)
    columns = ([country] if n_countries else []) + extra
    levels = ((n_countries,) if n_countries else ()) + more_categorical
    names_named = tuple(f"feature_named_{i}" for i in range(n_named))
    names_obf = tuple(f"feature_obf_{i}" for i in range(n_obfuscated))
    names_cat = (("feature_country",) if n_countries else ()) + tuple(f"feature_cat_{i}" for i in range(len(extra)))
    edges = np.full((n_named, 255), np.inf, np.float32)
    edges[:, : named_bins - 1] = np.arange(named_bins - 1, dtype=np.float32) + 0.5
    return TinyData(
        n_rows=n_rows,
        binning="r0",
        feature_names=(*names_named, *names_obf, *names_cat),
        named_features=names_named,
        obfuscated_features=names_obf,
        named_n_bins=(named_bins,) * n_named,
        named_edges=edges,
        named_bins=named,
        obfuscated=pack_nibbles(codes),
        categories=tuple(tuple(f"C{i}" for i in range(n)) for n in levels),
        categorical=np.array(columns, np.uint8).reshape(len(columns), n_rows),
        country=country if n_countries else np.zeros(n_rows, np.uint8),
        target=target,
        codes=codes,
    )
