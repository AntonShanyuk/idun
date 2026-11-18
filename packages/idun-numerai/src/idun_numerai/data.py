"""`build_cache`: Numerai's v6 parquet files streamed into idun's packed dataset (docs/architecture/data-layout.md §2).

The panel is never materialized. ``train.parquet`` then ``validation.parquet`` are read twice:

1. **rows**, in record batches: the rows with a value of the chosen target, their era day, target, ticker and country.
   The rows must already be in date order (the v6 files are): a stable sort by date is then a no-op, and the builder
   refuses unsorted input instead of re-sorting it.
2. **features**, one row group at a time and ``columns_per_read`` columns per read: the same rows' obfuscated values
   (checked to be exactly 0..4) and named values. Reading all 1,040 columns of a row group at once would hold ~3 GB
   of decoded arrow buffers. Rows are staged in column-major blocks and written with ``pwrite`` at each column's
   offset, so the build stays under ~2.5 GB.

`idun.data.PackedWriter` stages and writes the columns, fits each refit's named-feature binning on a sample of its
training window and applies it to every row, checksums every array file and writes the manifest; the cache is built
in ``<out>.partial`` and renamed to ``<out>`` only when complete. This module knows Numerai's files and columns; the
walk-forward refits are the caller's (`idun_numerai.refits`).
"""

import datetime as dt
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from idun.data import PackedWriter
from idun.data.manifest import fingerprint_sources
from idun.errors import CacheError
from idun_numerai._version import __version__
from idun_numerai.refits import refit_table

PARQUET_FILES = ("train.parquet", "validation.parquet")
SOURCE_FILES = (*PARQUET_FILES, "features.json")
COUNTRY = "feature_country"
CODE_LEVELS = 5  # obfuscated values are exactly 0..4: they are their own bins
SAMPLE_ROWS = 200_000  # rows per binning sample
SCAN_BATCH_ROWS = 65_536
COLUMNS_PER_READ = 32
BLOCK_ROWS = 131_072

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Features:
    named: tuple[str, ...]
    obfuscated: tuple[str, ...]


@dataclass(frozen=True)
class _Rows:
    era: NDArray[np.int32]
    target: NDArray[np.float32]
    tickers: tuple[str, ...]
    ticker: NDArray[np.int32]
    countries: tuple[str, ...]
    country: NDArray[np.uint8]


def build_cache(
    v6_dir: str | Path,
    out_dir: str | Path,
    *,
    seed: int,
    target: str,
    refit_dates: Sequence[str],
    purge: int,
    sample_rows: int = SAMPLE_ROWS,
    columns_per_read: int = COLUMNS_PER_READ,
    block_rows: int = BLOCK_ROWS,
) -> Path:
    """Build the packed cache of the v6 panel in ``out_dir`` and return its path.

    Parameters
    ----------
    v6_dir
        Directory with ``train.parquet``, ``validation.parquet`` and ``features.json``. Only read.
    out_dir
        Must not exist yet. The cache is assembled in ``<out_dir>.partial`` and renamed when complete.
    seed
        Seeds the binning samples (``PCG64([seed, refit])``).
    target
        The target column the cache holds; it keeps the rows with a value of it (no default).
    refit_dates, purge
        The walk-forward refits (`idun_numerai.refits.refit_table`): their ISO dates, strictly increasing, and the
        eras left out before each. Each refit gets a binning of its own, fitted on its training rows.
    sample_rows
        Rows of each refit's training window the binning is fitted on (`SAMPLE_ROWS`; tests shrink it).
    columns_per_read, block_rows
        Feature columns per parquet read, and rows per staged write block (even). They change speed and memory,
        never the output.

    Raises
    ------
    FileExistsError
        ``out_dir`` or its ``.partial`` sibling exists.
    ValueError
        The input breaks an assumption: rows out of date order, an obfuscated value outside 0..4, a null key column.
    CacheError
        A source file changed while the cache was being built.
    """
    v6_dir, out_dir = Path(v6_dir), Path(out_dir)
    if columns_per_read < 1:
        raise ValueError(f"columns_per_read must be positive, got {columns_per_read}")
    for path in (out_dir, out_dir.with_name(out_dir.name + ".partial")):
        if path.exists():
            raise FileExistsError(f"{path} exists; delete it to rebuild the cache")
    sources = fingerprint_sources(v6_dir, SOURCE_FILES)
    features = _read_features(v6_dir)
    files = [v6_dir / name for name in PARQUET_FILES]
    rows = _scan_rows(files, target)
    refits = refit_table(rows.era, refit_dates, purge)
    _log.info("rows: %d with a target; refits %s", rows.era.size, refits)
    writer = PackedWriter(out_dir, n_rows=int(rows.era.size), codes=features.obfuscated, numeric=features.named,
                          categorical=COUNTRY, block_rows=block_rows)  # fmt: skip
    with writer:
        _write_features(files, features, rows.target, writer, target=target, columns_per_read=columns_per_read)
        writer.write_binnings(refits, seed=seed, sample_rows=sample_rows)
        writer.write_rows(target=rows.target, era=rows.era, ticker=rows.ticker, tickers=rows.tickers,
                          country=rows.country, countries=rows.countries)  # fmt: skip
        _check_unchanged(v6_dir, sources)
        built = {
            "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "builder": {"idun_numerai": __version__, "numpy": np.__version__, "pyarrow": pa.__version__},
        }
        return writer.finish({"source_dir": str(v6_dir.resolve()), "sources": sources, **built, "purge": purge})


def _check_unchanged(v6_dir: Path, sources: list[dict[str, Any]]) -> None:
    if fingerprint_sources(v6_dir, SOURCE_FILES) != sources:
        raise CacheError(f"a source file in {v6_dir} changed during the build")


def _read_features(v6_dir: Path) -> _Features:
    """Return the feature lists: named = signals except country, obfuscated = all except signals (sorted)."""
    sets = json.loads((v6_dir / "features.json").read_text())["feature_sets"]
    signals = set(sets["signals"])
    if COUNTRY not in signals:
        raise ValueError(f"features.json: {COUNTRY} is not in the signals set")
    features = _Features(
        named=tuple(sorted(signals - {COUNTRY})),
        obfuscated=tuple(sorted(name for name in sets["all"] if name not in signals)),
    )
    if not features.named or not features.obfuscated:
        raise ValueError("features.json: the named or the obfuscated feature list is empty")
    return features


def _floats(column: pa.Array) -> NDArray[np.float32]:
    """Return a numeric column as float32, nulls as NaN."""
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float32)


def _with_target(batch: pa.RecordBatch, name: str) -> tuple[pa.RecordBatch, NDArray[np.float32]]:
    """Keep the batch's rows with a target (not NaN); return them and their targets."""
    target = _floats(batch.column(name))
    keep = ~np.isnan(target)
    if keep.all():
        return batch, target
    return batch.filter(pa.array(keep)), target[keep]


class _Codes:
    """Strings to codes: provisional ids in order of first appearance, re-coded at the end as sorted-list indices."""

    def __init__(self, column: str) -> None:
        self._column = column
        self._ids: dict[str, int] = {}
        self._parts: list[NDArray[np.int64]] = []

    def add(self, values: pa.Array) -> None:
        if values.null_count:
            raise ValueError(f"{self._column} has nulls on rows with a target")
        encoded = pc.dictionary_encode(values)
        ids = [self._ids.setdefault(value, len(self._ids)) for value in encoded.dictionary.to_pylist()]
        self._parts.append(np.asarray(ids, np.int64)[encoded.indices.to_numpy(zero_copy_only=False)])

    def finish(self) -> tuple[tuple[str, ...], NDArray[np.int64]]:
        names = sorted(self._ids)
        rank = np.empty(len(names), np.int64)
        rank[[self._ids[name] for name in names]] = np.arange(len(names))
        return tuple(names), rank[np.concatenate(self._parts)]


def _scan_rows(files: Sequence[Path], target_name: str) -> _Rows:
    """Pass 1: era day, target, ticker and country of every row with a target, in file order."""
    eras, targets = [], []
    tickers, countries = _Codes("numerai_ticker"), _Codes(COUNTRY)
    for path in files:
        reader = pq.ParquetFile(path)
        if target_name not in reader.schema_arrow.names:
            raise ValueError(f"{path}: no target column {target_name!r}")
        columns = ["date", "numerai_ticker", COUNTRY, target_name]
        for raw in reader.iter_batches(batch_size=SCAN_BATCH_ROWS, columns=columns):
            batch, target = _with_target(raw, target_name)
            dates = batch.column("date")
            if dates.null_count:
                raise ValueError(f"{path}: date has nulls on rows with a target")
            eras.append(np.asarray(pc.cast(pc.cast(dates, pa.date32()), pa.int32()).to_numpy(), np.int32))
            targets.append(target)
            tickers.add(batch.column("numerai_ticker"))
            countries.add(batch.column(COUNTRY))
        _log.info("pass 1: %s scanned", path.name)
    era = np.concatenate(eras)
    if era.size == 0:
        raise ValueError("no row has a target")
    if bool(np.any(np.diff(era) < 0)):
        raise ValueError("rows are not in date order; the streaming builder needs date-ordered parquet files")
    ticker_names, ticker = tickers.finish()
    country_names, country = countries.finish()
    if len(country_names) > 256:
        raise ValueError(f"{len(country_names)} countries do not fit a uint8 code")
    return _Rows(
        era=era,
        target=np.concatenate(targets),
        tickers=ticker_names,
        ticker=ticker.astype(np.int32),
        countries=country_names,
        country=country.astype(np.uint8),
    )


def _read_group(reader: pq.ParquetFile, group: int, names: Sequence[str]) -> NDArray[np.float32]:
    """Read some numeric columns of one row group as a ``(columns, rows)`` float32 array, nulls as NaN."""
    table = reader.read_row_group(group, columns=list(names))
    return np.stack([_floats(table.column(name)) for name in names])


def _obfuscated_codes(values: NDArray[np.float32], names: Sequence[str], where: str) -> NDArray[np.uint8]:
    """Return obfuscated values as uint8 codes; each must be exactly one of 0..4."""
    with np.errstate(invalid="ignore"):
        codes = values.astype(np.uint8)
    if codes.max(initial=0) < CODE_LEVELS and np.array_equal(codes, values):  # NaN never compares equal
        return codes
    bad = (codes >= CODE_LEVELS) | (codes != values)
    feature, row = (int(i) for i in np.argwhere(bad)[0])
    raise ValueError(f"{where}: {names[feature]} = {values[feature, row]} in kept row {row}, not one of 0..4")


def _write_features(
    files: Sequence[Path],
    features: _Features,
    targets: NDArray[np.float32],
    writer: PackedWriter,
    *,
    target: str,
    columns_per_read: int,
) -> None:
    """Pass 2: obfuscated codes and raw named values of the kept rows, one row group at a time, into ``writer``."""
    at = 0
    for path in files:
        reader = pq.ParquetFile(path)
        for group in range(reader.num_row_groups):
            values = _read_group(reader, group, [target])[0]
            has_target = ~np.isnan(values)
            kept = int(np.count_nonzero(has_target))
            keep = slice(None) if kept == values.size else has_target  # all kept: no filtering copy
            if not np.array_equal(values[keep], targets[at : at + kept]):
                raise ValueError(f"{path}: pass 2 read different rows than pass 1 in row group {group}")
            codes = np.empty((len(features.obfuscated), kept), np.uint8)
            for start in range(0, len(features.obfuscated), columns_per_read):
                names = features.obfuscated[start : start + columns_per_read]
                where = f"{path.name} row group {group}"
                values = _read_group(reader, group, names)[:, keep]
                codes[start : start + len(names)] = _obfuscated_codes(values, names, where)
            writer.add(codes, _read_group(reader, group, features.named)[:, keep])
            at += kept
        _log.info("pass 2: %s written", path.name)
    writer.end_features()


__all__ = ["SAMPLE_ROWS", "SOURCE_FILES", "build_cache"]
