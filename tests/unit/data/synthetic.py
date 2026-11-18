"""A synthetic Numerai-v6-shaped directory, and its expected panel computed independently of idun.data.

The files mimic the real ones where it matters to the builder: features.json feature sets in unsorted order, parquet
columns in unsorted order, obfuscated values stored as float32 0..4, named values with nulls and ties, string dates,
several small row groups, file order within an era that is not ticker order, and validation rows without a target.
"""

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from idun_numerai.data import build_cache

FIRST_ERA = dt.date(2003, 1, 3)
EPOCH = dt.date(1970, 1, 1)
# A protocol scaled to the synthetic panel: 26 weekly eras, refits at eras 13 and 18, purge 2.
TARGET = "target_masked"  # NaN on the live era, as a real target
REFIT_DATES = ("2003-04-04", "2003-05-09")
PURGE = 2
SAMPLE_ROWS = 40


@dataclass
class Synthetic:
    """The source directory and the panel the cache builder must read from it: rows with a target, in file order."""

    v6: Path
    named: list[str]
    obfuscated: list[str]
    era: NDArray[np.int32]
    ticker: NDArray[np.str_]
    country: NDArray[np.str_]
    target: NDArray[np.float32]
    named_values: NDArray[np.float32]  # (rows, named), loader order, NaN for nulls
    codes: NDArray[np.uint8]  # (rows, obfuscated), loader order


def write_v6(root: Path, *, eras: int = 26, tickers: int = 15, seed: int = 0, row_group_size: int = 23) -> Synthetic:
    rng = np.random.default_rng(seed)
    named = ["feature_zeta_factor", "feature_alpha_factor", "feature_tied_factor"]
    obfuscated = [f"feature_obf_{i}" for i in (3, 0, 4, 1, 6, 2, 5)]
    country_of = {f"T{t:03d}": str(c) for t, c in enumerate(rng.choice(["US", "JP", "GB", "DE"], tickers))}
    root.mkdir(parents=True)
    signals = [*named, "feature_country"]
    (root / "features.json").write_text(
        json.dumps({"feature_sets": {"all": [*obfuscated, *signals][::-1], "signals": signals[::-1]}})
    )
    split = eras * 2 // 3
    parts: list[dict[str, NDArray[np.generic]]] = []
    for name, era_range in (("train.parquet", range(split)), ("validation.parquet", range(split, eras))):
        per_era = [rng.permutation(sorted(country_of))[: rng.integers(tickers // 2, tickers + 1)] for _ in era_range]
        n = sum(len(names) for names in per_era)
        date = np.concatenate(
            [[str(FIRST_ERA + dt.timedelta(weeks=e))] * len(t) for e, t in zip(era_range, per_era, strict=True)]
        )
        ticker = np.concatenate(per_era)
        country = np.array([country_of[t] for t in ticker])
        values = rng.normal(size=(n, len(named))).astype(np.float32)
        values[:, 2] = rng.integers(0, 6, n).astype(np.float32) / 4  # six distinct values: ties
        missing = rng.random((n, len(named))) < 0.1
        codes = rng.integers(0, 5, size=(n, len(obfuscated))).astype(np.uint8)
        target = rng.choice(np.array([0, 0.25, 0.5, 0.75, 1], np.float32), n)
        has_target = np.ones(n, bool)
        if name == "validation.parquet":
            has_target[n - len(per_era[-1]) :] = False  # the live era: no target yet
        columns = {
            "numerai_ticker": pa.array(ticker),
            "date": pa.array(date),
            "feature_country": pa.array(country),
            **{c: pa.array(values[:, j], mask=missing[:, j]) for j, c in enumerate(named)},
            **{c: pa.array(codes[:, j].astype(np.float32)) for j, c in enumerate(obfuscated)},
            TARGET: pa.array(target, mask=~has_target),
            "target": pa.array(target),
        }
        order = [list(columns)[i] for i in rng.permutation(len(columns))]  # not the cache's (sorted) column order
        pq.write_table(pa.table({c: columns[c] for c in order}), root / name, row_group_size=row_group_size)
        values[missing] = np.nan
        days = np.array([(dt.date.fromisoformat(d) - EPOCH).days for d in date], np.int32)
        parts.append(
            {
                "era": days[has_target],
                "ticker": ticker[has_target],
                "country": country[has_target],
                "target": target[has_target],
                "named_values": values[has_target][:, [named.index(c) for c in sorted(named)]],
                "codes": codes[has_target][:, [obfuscated.index(c) for c in sorted(obfuscated)]],
            }
        )
    joined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    return Synthetic(v6=root, named=sorted(named), obfuscated=sorted(obfuscated), **joined)


def build_small(v6: Path, out: Path, *, seed: int = 7, columns_per_read: int = 3, block_rows: int = 16) -> Path:
    """Build a cache of a synthetic directory with small walk-forward windows."""
    return build_cache(
        v6,
        out,
        seed=seed,
        target=TARGET,
        sample_rows=SAMPLE_ROWS,
        refit_dates=REFIT_DATES,
        purge=PURGE,
        columns_per_read=columns_per_read,
        block_rows=block_rows,
    )
