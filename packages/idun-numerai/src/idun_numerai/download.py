"""`download`: Numerai's dataset files from Numerai's API, through `numerapi`.

The public data needs no credentials. The files land in one directory under their own names (``v6.0/train.parquet``
as ``train.parquet``), the layout `idun_numerai.data.build_cache` reads. Each file is written to ``<name>.partial``
and renamed when complete, so an interrupted download leaves no file that looks whole.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from idun_numerai.data import SOURCE_FILES

VERSION = "v6.0"  # Numerai's dataset version
SCORING_FILES = ("validation_neutralizer.parquet",)  # the neutralizer of `idun_numerai.score.ncorr`


def download(
    out_dir: str | Path, *, files: Sequence[str] = SOURCE_FILES, version: str = VERSION, api: Any = None
) -> list[Path]:
    """Download ``files`` of dataset ``version`` into ``out_dir``; return their paths.

    ``files`` defaults to what `idun_numerai.data.build_cache` reads. ``api`` is a `numerapi.NumerAPI` (a new one when
    None). ``ValueError`` if Numerai lists no such file (nothing is downloaded); ``FileExistsError`` if a file is
    already in ``out_dir``: a dataset's validation file grows every round, so an old copy is never reused silently.
    """
    if not files:
        raise ValueError("no files to download")
    out = Path(out_dir)
    if present := sorted(name for name in files if (out / name).exists()):
        raise FileExistsError(f"{out} already holds {present}")
    if api is None:
        from numerapi import NumerAPI  # noqa: PLC0415 (only a download needs it)

        api = NumerAPI()  # type: ignore[no-untyped-call]  # numerapi has no annotations here
    listed = set(api.list_datasets())
    if missing := sorted(f"{version}/{name}" for name in files if f"{version}/{name}" not in listed):
        raise ValueError(f"Numerai lists no {missing}")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in files:
        partial = out / f"{name}.partial"
        partial.unlink(missing_ok=True)  # an interrupted download's: numerapi would keep a file that exists
        api.download_dataset(f"{version}/{name}", str(partial))
        partial.rename(out / name)
        paths.append(out / name)
    return paths


__all__ = ["VERSION", "download"]
