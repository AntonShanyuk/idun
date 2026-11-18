"""`PackedDataset`: the read-only, memory-mapped reader of the packed cache (docs/architecture/data-layout.md §2).

Opening a cache checks its format version, that no source file changed since it was built, and that every array file
has its recorded size. ``verify=True`` (or `PackedDataset.verify`) also recomputes every checksum. Nothing is copied
into memory: each array is an ``np.memmap`` view, read-only, and columns are contiguous.
"""

import json
import os
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from idun.data import layout
from idun.data.manifest import check_fresh, read_manifest, sha256_file
from idun.data.refits import Refit
from idun.errors import CacheError

COPY_ENV = "IDUN_CACHE_COPY"  # "1": a copied cache, checked by its checksums instead of its source files


class PackedDataset:
    """One packed cache with one named-feature binning selected. Write one with `idun.data.PackedWriter`.

    Rows are in date order. Features are in the order named
    (sorted), obfuscated (sorted), country.

    Attributes
    ----------
    n_rows : int
        Rows in the panel.
    binning : str
        The selected named-feature binning: ``r<k>``, fitted on refit ``k``'s training window.
    target, era, ticker, country : NDArray
        Per row: float32 target, int32 days since 1970-01-01, int32 index into `tickers`, uint8 index into `countries`.
    obfuscated : NDArray[np.uint8]
        ``(n_obfuscated, ceil(n_rows / 2))``: two rows per byte (`idun.data.layout.unpack_nibbles`).
    named_values, named_bins : NDArray
        ``(n_named, n_rows)``: raw float32 values (NaN kept), and their bins under `binning` (255 = missing).
    named_edges : NDArray[np.float32]
        ``(n_named, 255)`` upper bin edges under `binning`, +inf padded; `named_n_bins` real bins per feature.
    """

    def __init__(self, path: Path, manifest: dict[str, Any], binning: str) -> None:
        if binning not in manifest["binnings"]:
            raise CacheError(f"{path} has no binning {binning!r}; it has {sorted(manifest['binnings'])}")
        self.path = path
        self.binning = binning
        self.n_rows = n = int(manifest["n_rows"])
        features = manifest["features"]
        self.named_features: tuple[str, ...] = tuple(features["named"])
        self.obfuscated_features: tuple[str, ...] = tuple(features["obfuscated"])
        self.feature_names = (*self.named_features, *self.obfuscated_features, features["country"])
        self.countries: tuple[str, ...] = tuple(manifest["countries"])
        self.refits = tuple(Refit(**refit) for refit in manifest["refits"])
        self.seed = int(manifest["seed"])
        self._manifest = manifest
        strides = manifest["strides"]
        n_named, n_obfuscated = len(self.named_features), len(self.obfuscated_features)
        self.target: NDArray[np.float32] = _map(path / layout.TARGET_FILE, np.float32, (n,))
        self.era: NDArray[np.int32] = _map(path / layout.ERA_FILE, np.int32, (n,))
        self.ticker: NDArray[np.int32] = _map(path / layout.TICKER_FILE, np.int32, (n,))
        self.country: NDArray[np.uint8] = _map(path / layout.COUNTRY_FILE, np.uint8, (n,))
        # idun_gbm reads the country as the one categorical column (BinnedData): a zero-stride view, nothing copied
        self.categories: tuple[tuple[str, ...], ...] = (self.countries,)
        self.categorical: NDArray[np.uint8] = self.country[np.newaxis, :]
        obfuscated = _map(path / layout.OBFUSCATED_FILE, np.uint8, (n_obfuscated, strides["obfuscated"]))
        self.obfuscated: NDArray[np.uint8] = obfuscated[:, : layout.packed_bytes(n)]
        values = _map(path / layout.NAMED_VALUES_FILE, np.float32, (n_named, strides["named_values"] // 4))
        self.named_values: NDArray[np.float32] = values[:, :n]
        bins = _map(path / layout.named_bins_file(binning), np.uint8, (n_named, strides["named_bins"]))
        self.named_bins: NDArray[np.uint8] = bins[:, :n]
        edges = np.fromfile(path / layout.named_edges_file(binning), np.float32)
        self.named_edges = edges.reshape(n_named, layout.MAX_NAMED_BINS)
        self.named_n_bins: tuple[int, ...] = tuple(manifest["binnings"][binning]["n_bins"])

    @classmethod
    def open(cls, path: str | Path, *, binning: str, verify: bool = False) -> Self:
        """Open the cache at ``path`` with ``binning`` selected (``"r0"``, ``"r1"``, ...: one per walk-forward window).

        Raises `CacheError` when the cache is missing, of another format version, stale (a source file changed),
        truncated, or (with ``verify``) when a checksum differs. With the environment variable ``IDUN_CACHE_COPY=1``
        the cache is a copy on a machine without the source files: the source check is replaced by
        the checksum check of every array file, so a copy is never opened unchecked.
        """
        path = Path(path)
        manifest = read_manifest(path)
        copy = os.environ.get(COPY_ENV) == "1"
        try:
            if not copy:
                check_fresh(manifest)
            for name, array in manifest["arrays"].items():
                file = path / name
                if not file.is_file() or file.stat().st_size != array["bytes"]:
                    raise CacheError(f"{file} is missing or truncated; rebuild the cache")
            dataset = cls(path, manifest, binning)
        except KeyError as missing:
            raise CacheError(f"{path / layout.MANIFEST} lacks {missing}; rebuild the cache") from None
        if verify or copy:
            dataset.verify()
        return dataset

    def verify(self) -> None:
        """Recompute the sha256 of every array file and raise `CacheError` on any mismatch."""
        for name, array in self._manifest["arrays"].items():
            if sha256_file(self.path / name) != array["sha256"]:
                raise CacheError(f"{self.path / name}: checksum mismatch; the cache is corrupt, rebuild it")

    @cached_property
    def tickers(self) -> tuple[str, ...]:
        """Sorted ticker names; `ticker` indexes into them."""
        names: list[str] = json.loads((self.path / layout.TICKERS_FILE).read_text())
        return tuple(names)

    def obfuscated_codes(self, feature: int, lo: int = 0, hi: int | None = None) -> NDArray[np.uint8]:
        """Unpacked codes (0..4) of obfuscated feature ``feature`` for rows ``[lo, hi)``; all rows by default."""
        hi = self.n_rows if hi is None else hi
        if hi > self.n_rows:
            raise ValueError(f"row {hi} is past the panel's {self.n_rows} rows")
        return layout.unpack_nibbles(self.obfuscated[feature], lo, hi)

    def __repr__(self) -> str:
        return f"PackedDataset({str(self.path)!r}, binning={self.binning!r}, n_rows={self.n_rows})"


def _map(path: Path, dtype: type[np.generic], shape: tuple[int, ...]) -> NDArray[Any]:
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)
