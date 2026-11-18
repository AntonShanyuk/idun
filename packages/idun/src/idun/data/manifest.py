"""The cache manifest: what the cache was built from, how, and the checksum of every array file.

The manifest is the cache's single source of metadata. The reader trusts nothing else: it compares the source
fingerprints against the files on disk (a stale cache fails loudly) and the array sizes against the files it maps.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from idun.data.layout import FORMAT, FORMAT_VERSION, MANIFEST
from idun.errors import CacheError

_HASH_CHUNK = 8 * 1024 * 1024


def fingerprint_sources(source_dir: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    """Name, size and modification time of each source file ``names``; a change to any of them makes a cache stale."""
    prints = []
    for name in names:
        path = source_dir / name
        if not path.is_file():
            raise CacheError(f"source file {path} is missing")
        stat = path.stat()
        prints.append({"name": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return prints


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    """Write ``manifest.json`` (format and version are stamped here)."""
    stamped = {"format": FORMAT, "format_version": FORMAT_VERSION, **manifest}
    (cache_dir / MANIFEST).write_text(json.dumps(stamped, indent=1) + "\n")


def read_manifest(cache_dir: Path) -> dict[str, Any]:
    """Read ``manifest.json`` and check its format and version."""
    path = cache_dir / MANIFEST
    if not path.is_file():
        raise CacheError(f"no cache at {cache_dir}: {MANIFEST} is missing (write one with idun.data.PackedWriter)")
    manifest: dict[str, Any] = json.loads(path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
        raise CacheError(
            f"{path}: format {manifest.get('format')!r} version {manifest.get('format_version')!r}; this idun reads "
            f"{FORMAT!r} version {FORMAT_VERSION}. Rebuild the cache."
        )
    return manifest


def check_fresh(manifest: dict[str, Any]) -> None:
    """Raise `CacheError` when a source file changed since the cache was built."""
    built = {source["name"]: source for source in manifest["sources"]}
    for now in fingerprint_sources(Path(manifest["source_dir"]), list(built)):
        if built.get(now["name"]) != now:
            raise CacheError(
                f"stale cache: {now['name']} changed since the cache was built "
                f"({built.get(now['name'])} → {now}). Rebuild the cache."
            )
