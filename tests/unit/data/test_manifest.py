import hashlib
import json
import os
from pathlib import Path

import pytest

from idun.data import manifest
from idun.data.layout import FORMAT, FORMAT_VERSION, MANIFEST
from idun.errors import CacheError

NAMES = ("train.parquet", "validation.parquet", "features.json")


def sources(root: Path) -> Path:
    root.mkdir()
    for name in NAMES:
        (root / name).write_bytes(name.encode())
    return root


def test_fingerprints_name_size_and_mtime(tmp_path: Path) -> None:
    root = sources(tmp_path / "v6")
    os.utime(root / "train.parquet", ns=(1, 1_000))
    prints = manifest.fingerprint_sources(root, NAMES)
    assert [p["name"] for p in prints] == list(NAMES)
    assert prints[0] == {"name": "train.parquet", "bytes": len(b"train.parquet"), "mtime_ns": 1_000}


def test_missing_source_raises(tmp_path: Path) -> None:
    root = sources(tmp_path / "v6")
    (root / "features.json").unlink()
    with pytest.raises(CacheError, match=r"features\.json is missing"):
        manifest.fingerprint_sources(root, NAMES)


def test_sha256_reads_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manifest, "_HASH_CHUNK", 3)
    data = bytes(range(256)) * 5
    (tmp_path / "blob").write_bytes(data)
    assert manifest.sha256_file(tmp_path / "blob") == hashlib.sha256(data).hexdigest()


def test_manifest_round_trips_with_its_format_stamp(tmp_path: Path) -> None:
    manifest.write_manifest(tmp_path, {"n_rows": 3})
    assert manifest.read_manifest(tmp_path) == {"format": FORMAT, "format_version": FORMAT_VERSION, "n_rows": 3}


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(CacheError, match=r"write one with idun\.data\.PackedWriter"):
        manifest.read_manifest(tmp_path)


@pytest.mark.parametrize("stamp", [{"format": "other", "format_version": FORMAT_VERSION}, {"format": FORMAT}])
def test_another_format_or_version_raises(tmp_path: Path, stamp: dict[str, object]) -> None:
    (tmp_path / MANIFEST).write_text(json.dumps(stamp))
    with pytest.raises(CacheError, match="Rebuild"):
        manifest.read_manifest(tmp_path)


def test_freshness_tracks_the_sources(tmp_path: Path) -> None:
    root = sources(tmp_path / "v6")
    built = {"source_dir": str(root), "sources": manifest.fingerprint_sources(root, NAMES)}
    manifest.check_fresh(built)
    stat = (root / "validation.parquet").stat()
    os.utime(root / "validation.parquet", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    with pytest.raises(CacheError, match=r"stale cache: validation\.parquet changed"):
        manifest.check_fresh(built)
