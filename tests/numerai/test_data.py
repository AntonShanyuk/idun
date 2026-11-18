import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from idun.data import PackedDataset, binning, layout, manifest
from idun.errors import CacheError
from idun_numerai import data as builder
from idun_numerai.refits import refit_table
from tests.unit.data.synthetic import PURGE, REFIT_DATES, SAMPLE_ROWS, TARGET, Synthetic, build_small, write_v6


def rewrite(path: Path, column: str, values: pa.Array) -> None:
    """Replace one column of a parquet file (keeping its row groups small)."""
    table = pq.read_table(path)
    pq.write_table(table.set_column(table.schema.get_field_index(column), column, values), path, row_group_size=23)


def test_the_cache_holds_the_loader_panel(synthetic: Synthetic, cache_dir: Path) -> None:
    data = PackedDataset.open(cache_dir, binning="r0", verify=True)
    assert data.n_rows == synthetic.target.size
    assert data.named_features == tuple(synthetic.named)
    assert data.obfuscated_features == tuple(synthetic.obfuscated)
    assert data.feature_names == (*synthetic.named, *synthetic.obfuscated, "feature_country")
    np.testing.assert_array_equal(data.era, synthetic.era)
    np.testing.assert_array_equal(data.target, synthetic.target)
    np.testing.assert_array_equal(np.array(data.tickers)[data.ticker], synthetic.ticker)
    assert data.countries == tuple(sorted(set(synthetic.country)))
    np.testing.assert_array_equal(np.array(data.countries)[data.country], synthetic.country)
    for f in range(len(synthetic.obfuscated)):
        np.testing.assert_array_equal(data.obfuscated_codes(f), synthetic.codes[:, f])
    np.testing.assert_array_equal(data.named_values, synthetic.named_values.T)  # NaN where the source had nulls


def test_refits_and_binnings_follow_the_protocol(synthetic: Synthetic, cache_dir: Path) -> None:
    expected_refits = refit_table(synthetic.era, REFIT_DATES, PURGE)
    for index, refit in enumerate(expected_refits):
        data = PackedDataset.open(cache_dir, binning=f"r{index}")
        assert data.refits == expected_refits
        sample = binning.sample_rows(refit.train_hi, SAMPLE_ROWS, seed=7, refit=index)
        assert sample[-1] < refit.train_hi  # edges see the training window only
        for j in range(len(synthetic.named)):
            edges = binning.quantile_edges(np.ascontiguousarray(synthetic.named_values[sample, j]))
            np.testing.assert_array_equal(data.named_edges[j, : edges.size], edges)
            assert np.all(np.isposinf(data.named_edges[j, edges.size :]))
            assert data.named_n_bins[j] == edges.size + 1
            expected_bins = np.searchsorted(edges, synthetic.named_values[:, j], side="left")
            expected_bins[np.isnan(synthetic.named_values[:, j])] = layout.MISSING_BIN
            np.testing.assert_array_equal(data.named_bins[j], expected_bins)


def test_files_are_page_strided_and_zero_padded(synthetic: Synthetic, cache_dir: Path) -> None:
    n = synthetic.target.size
    assert n % 2 == 1, "the synthetic panel should have an odd row count to exercise the half-filled last byte"
    stride = layout.column_stride(layout.packed_bytes(n))
    raw = np.fromfile(cache_dir / layout.OBFUSCATED_FILE, np.uint8).reshape(len(synthetic.obfuscated), stride)
    assert np.all(raw[:, layout.packed_bytes(n) - 1] >> 4 == 0)
    assert not raw[:, layout.packed_bytes(n) :].any()
    bins = np.fromfile(cache_dir / layout.named_bins_file("r0"), np.uint8).reshape(len(synthetic.named), -1)
    assert bins.shape[1] == layout.column_stride(n)
    assert not bins[:, n:].any()


@pytest.mark.parametrize(("columns_per_read", "block_rows"), [(1, 2), (2, 4), (1_000, 1_000)])
def test_batching_never_changes_the_bytes(
    synthetic: Synthetic, cache_dir: Path, tmp_path: Path, columns_per_read: int, block_rows: int
) -> None:
    other = build_small(synthetic.v6, tmp_path / "v6", columns_per_read=columns_per_read, block_rows=block_rows)
    assert manifest.read_manifest(other)["arrays"] == manifest.read_manifest(cache_dir)["arrays"]


def test_the_manifest_records_the_build(synthetic: Synthetic, cache_dir: Path) -> None:
    built = manifest.read_manifest(cache_dir)
    assert built["source_dir"] == str(synthetic.v6.resolve())
    assert built["sources"] == manifest.fingerprint_sources(synthetic.v6, builder.SOURCE_FILES)
    assert (built["seed"], built["sample_rows"], built["purge"], built["n_rows"]) == (7, 40, 2, synthetic.target.size)
    assert sorted(built["binnings"]) == ["r0", "r1"]
    assert set(built["arrays"]) == {p.name for p in cache_dir.iterdir()} - {layout.MANIFEST}
    assert json.loads((cache_dir / layout.TICKERS_FILE).read_text()) == sorted(set(synthetic.ticker))


def test_existing_outputs_are_never_overwritten(synthetic: Synthetic, cache_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="delete it"):
        build_small(synthetic.v6, cache_dir)
    (tmp_path / "v6.partial").mkdir()
    with pytest.raises(FileExistsError, match="partial"):
        build_small(synthetic.v6, tmp_path / "v6")


@pytest.mark.parametrize(
    ("columns_per_read", "block_rows", "message"),
    [(0, 2, "columns_per_read must be positive"), (1, 3, "block_rows must be even"), (1, 0, "block_rows must be even")],
)
def test_batch_sizes_are_checked(
    synthetic: Synthetic, tmp_path: Path, columns_per_read: int, block_rows: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_small(synthetic.v6, tmp_path / "v6", columns_per_read=columns_per_read, block_rows=block_rows)


@pytest.mark.parametrize(
    ("value", "message"), [(5.0, "not one of 0..4"), (2.5, "not one of 0..4"), (None, "not one of 0..4")]
)
def test_obfuscated_values_must_be_codes(tmp_path: Path, value: float | None, message: str) -> None:
    v6 = write_v6(tmp_path / "v6").v6
    table = pq.read_table(v6 / "validation.parquet", columns=["feature_obf_4"])
    values = table["feature_obf_4"].to_pylist()
    values[3] = value
    rewrite(v6 / "validation.parquet", "feature_obf_4", pa.array(values, pa.float32()))
    with pytest.raises(ValueError, match=message):
        build_small(v6, tmp_path / "cache")
    assert not list(tmp_path.glob("cache*")), "a failed build leaves nothing behind"


def test_rows_must_be_in_date_order(tmp_path: Path) -> None:
    v6 = write_v6(tmp_path / "v6").v6
    dates = pq.read_table(v6 / "train.parquet", columns=["date"])["date"].to_pylist()
    dates[0] = "2010-01-01"
    rewrite(v6 / "train.parquet", "date", pa.array(dates))
    with pytest.raises(ValueError, match="date order"):
        build_small(v6, tmp_path / "cache")


@pytest.mark.parametrize("column", ["numerai_ticker", "feature_country", "date"])
def test_key_columns_must_not_be_null(tmp_path: Path, column: str) -> None:
    v6 = write_v6(tmp_path / "v6").v6
    values = pq.read_table(v6 / "train.parquet", columns=[column])[column].to_pylist()
    values[5] = None
    rewrite(v6 / "train.parquet", column, pa.array(values, pa.string()))
    with pytest.raises(ValueError, match="nulls"):
        build_small(v6, tmp_path / "cache")


def test_rows_without_any_target_are_rejected(tmp_path: Path) -> None:
    v6 = write_v6(tmp_path / "v6").v6
    for name in builder.PARQUET_FILES:
        n = pq.read_metadata(v6 / name).num_rows
        rewrite(v6 / name, TARGET, pa.nulls(n, pa.float32()))
    with pytest.raises(ValueError, match="no row has a target"):
        build_small(v6, tmp_path / "cache")


def test_a_missing_target_column_is_rejected(synthetic: Synthetic, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no target column 'target_none'"):
        builder.build_cache(synthetic.v6, tmp_path / "cache", seed=0, target="target_none", refit_dates=REFIT_DATES,
                            purge=PURGE, sample_rows=SAMPLE_ROWS)  # fmt: skip


def test_too_many_countries_are_rejected(tmp_path: Path) -> None:
    v6 = write_v6(tmp_path / "v6", tickers=40).v6
    n = pq.read_metadata(v6 / "train.parquet").num_rows
    assert n > 256
    rewrite(v6 / "train.parquet", "feature_country", pa.array([f"C{i:03d}" for i in range(n)]))
    with pytest.raises(ValueError, match="uint8"):
        build_small(v6, tmp_path / "cache")


@pytest.mark.parametrize(
    ("sets", "message"),
    [
        ({"all": ["feature_a", "feature_b"], "signals": ["feature_a"]}, "not in the signals"),
        ({"all": ["feature_country", "feature_a"], "signals": ["feature_country", "feature_a"]}, "empty"),
    ],
)
def test_feature_sets_are_checked(tmp_path: Path, sets: dict[str, list[str]], message: str) -> None:
    v6 = write_v6(tmp_path / "v6").v6
    (v6 / "features.json").write_text(json.dumps({"feature_sets": sets}))
    with pytest.raises(ValueError, match=message):
        build_small(v6, tmp_path / "cache")


def test_a_source_changing_during_the_build_fails(
    synthetic: Synthetic, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real = manifest.fingerprint_sources

    def drifting(source_dir: Path, names: tuple[str, ...]) -> list[dict[str, object]]:
        calls.append(source_dir)
        prints = real(source_dir, names)
        if len(calls) > 1:
            prints[0] = {**prints[0], "mtime_ns": 0}
        return prints

    monkeypatch.setattr(builder, "fingerprint_sources", drifting)
    with pytest.raises(CacheError, match="changed during the build"):
        build_small(synthetic.v6, tmp_path / "cache")
    assert not list(tmp_path.glob("cache*"))
