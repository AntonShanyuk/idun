from pathlib import Path

import pytest

from idun_numerai import download
from idun_numerai.data import SOURCE_FILES


class FakeAPI:
    """numerapi.NumerAPI's two calls, writing a file's remote name into it."""

    def __init__(self, listed: list[str]) -> None:
        self.listed = listed
        self.downloads: list[tuple[str, str]] = []

    def list_datasets(self) -> list[str]:
        return self.listed

    def download_dataset(self, filename: str, dest_path: str) -> str:
        self.downloads.append((filename, dest_path))
        Path(dest_path).write_text(filename)
        return dest_path


def test_the_cache_builders_files_land_under_their_own_names(tmp_path: Path) -> None:
    api = FakeAPI([f"v6.0/{name}" for name in (*SOURCE_FILES, "live.parquet")])
    paths = download.download(tmp_path / "v6", api=api)
    assert paths == [tmp_path / "v6" / name for name in SOURCE_FILES]
    assert [p.read_text() for p in paths] == [f"v6.0/{name}" for name in SOURCE_FILES]
    assert [d for d, _ in api.downloads] == [f"v6.0/{name}" for name in SOURCE_FILES]
    assert all(dest.endswith(".partial") for _, dest in api.downloads)
    assert not list((tmp_path / "v6").glob("*.partial"))


def test_an_unlisted_file_downloads_nothing(tmp_path: Path) -> None:
    api = FakeAPI(["v6.0/train.parquet"])
    with pytest.raises(ValueError, match=r"Numerai lists no \['v6.0/validation_neutralizer.parquet'\]"):
        download.download(tmp_path, files=("train.parquet", *download.SCORING_FILES), api=api)
    assert api.downloads == []


def test_a_file_already_there_is_not_replaced(tmp_path: Path) -> None:
    (tmp_path / "features.json").write_text("old")
    api = FakeAPI([f"v6.0/{name}" for name in SOURCE_FILES])
    with pytest.raises(FileExistsError, match=r"features\.json"):
        download.download(tmp_path, api=api)
    assert api.downloads == []
    assert (tmp_path / "features.json").read_text() == "old"


def test_no_files_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no files"):
        download.download(tmp_path, files=(), api=FakeAPI([]))


def test_an_interrupted_download_leaves_only_a_partial_file(tmp_path: Path) -> None:
    class Failing(FakeAPI):
        def download_dataset(self, filename: str, dest_path: str) -> str:
            Path(dest_path).write_text("half")
            raise ConnectionError(filename)

    with pytest.raises(ConnectionError):
        download.download(tmp_path, files=("train.parquet",), api=Failing(["v6.0/train.parquet"]))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.parquet.partial"]


def test_a_retry_replaces_the_partial_file(tmp_path: Path) -> None:
    (tmp_path / "train.parquet.partial").write_text("half")
    download.download(tmp_path, files=("train.parquet",), api=FakeAPI(["v6.0/train.parquet"]))
    assert (tmp_path / "train.parquet").read_text() == "v6.0/train.parquet"


def test_numerapi_is_the_default_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAPI(["v6.0/features.json"])
    monkeypatch.setattr("numerapi.NumerAPI", lambda: api)
    download.download(tmp_path, files=("features.json",))
    assert api.downloads == [("v6.0/features.json", str(tmp_path / "features.json.partial"))]
