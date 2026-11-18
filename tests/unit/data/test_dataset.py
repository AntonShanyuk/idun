import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from idun.data import PackedDataset, layout
from idun.errors import CacheError
from tests.unit.data.synthetic import Synthetic, build_small, write_v6


@pytest.fixture
def own_cache(tmp_path: Path) -> tuple[Synthetic, Path]:
    """A source directory and cache this test may damage."""
    synthetic = write_v6(tmp_path / "v6", eras=20)
    return synthetic, build_small(synthetic.v6, tmp_path / "cache")


def test_arrays_are_read_only_views(cache_dir: Path) -> None:
    data = PackedDataset.open(cache_dir, binning="r1")
    for array in (
        data.target,
        data.era,
        data.ticker,
        data.country,
        data.obfuscated,
        data.named_values,
        data.named_bins,
    ):
        assert not array.flags.writeable
    assert data.obfuscated.shape == (len(data.obfuscated_features), layout.packed_bytes(data.n_rows))
    assert data.named_values.shape == data.named_bins.shape == (len(data.named_features), data.n_rows)
    assert data.named_edges.shape == (len(data.named_features), layout.MAX_NAMED_BINS)
    assert all(row.flags.c_contiguous for row in (data.obfuscated[0], data.named_values[1], data.named_bins[2]))
    assert data.seed == 7
    assert repr(data) == f"PackedDataset({str(cache_dir)!r}, binning='r1', n_rows={data.n_rows})"


def test_obfuscated_codes_slices_rows(synthetic: Synthetic, cache_dir: Path) -> None:
    data = PackedDataset.open(cache_dir, binning="r0")
    np.testing.assert_array_equal(data.obfuscated_codes(2, 5, 12), synthetic.codes[5:12, 2])
    with pytest.raises(ValueError, match="past the panel"):
        data.obfuscated_codes(0, 0, data.n_rows + 1)


def test_unknown_binning_raises(cache_dir: Path) -> None:
    with pytest.raises(CacheError, match=r"no binning 'r9'; it has \['r0', 'r1'\]"):
        PackedDataset.open(cache_dir, binning="r9")


def test_missing_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(CacheError, match=r"write one with idun\.data\.PackedWriter"):
        PackedDataset.open(tmp_path, binning="r0")


def test_stale_cache_raises(own_cache: tuple[Synthetic, Path]) -> None:
    synthetic, cache = own_cache
    stat = (synthetic.v6 / "train.parquet").stat()
    os.utime(synthetic.v6 / "train.parquet", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000))
    with pytest.raises(CacheError, match="stale cache"):
        PackedDataset.open(cache, binning="r0")


def test_truncated_array_raises(own_cache: tuple[Synthetic, Path]) -> None:
    _, cache = own_cache
    with (cache / layout.TARGET_FILE).open("r+b") as handle:
        handle.truncate(8)
    with pytest.raises(CacheError, match="truncated"):
        PackedDataset.open(cache, binning="r0")


def test_corruption_is_found_by_verify(own_cache: tuple[Synthetic, Path]) -> None:
    _, cache = own_cache
    with (cache / layout.OBFUSCATED_FILE).open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0x01]))
    PackedDataset.open(cache, binning="r0")  # sizes still match: only checksums see it
    with pytest.raises(CacheError, match=r"obf\.u4: checksum mismatch"):
        PackedDataset.open(cache, binning="r0", verify=True)


@pytest.mark.parametrize("key", ["arrays", "strides", "source_dir"])
def test_incomplete_manifest_raises(own_cache: tuple[Synthetic, Path], key: str) -> None:
    _, cache = own_cache
    manifest = json.loads((cache / layout.MANIFEST).read_text())
    del manifest[key]
    (cache / layout.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(CacheError, match=f"lacks '{key}'"):
        PackedDataset.open(cache, binning="r0")


def test_a_moved_source_directory_raises(own_cache: tuple[Synthetic, Path], tmp_path: Path) -> None:
    synthetic, cache = own_cache
    shutil.move(synthetic.v6, tmp_path / "moved")
    with pytest.raises(CacheError, match="missing"):
        PackedDataset.open(cache, binning="r0")


def test_a_copy_without_its_sources_opens_by_its_checksums(
    own_cache: tuple[Synthetic, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic, cache = own_cache
    shutil.move(synthetic.v6, tmp_path / "elsewhere")  # a second machine: the sources are not there
    monkeypatch.setenv("IDUN_CACHE_COPY", "1")
    assert PackedDataset.open(cache, binning="r0").n_rows > 0
    with (cache / layout.OBFUSCATED_FILE).open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0x01]))
    with pytest.raises(CacheError, match=r"obf\.u4: checksum mismatch"):  # a copy is never opened unchecked
        PackedDataset.open(cache, binning="r0")
    monkeypatch.setenv("IDUN_CACHE_COPY", "0")
    with pytest.raises(CacheError, match="missing"):
        PackedDataset.open(cache, binning="r0")
