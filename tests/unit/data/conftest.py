"""Session fixtures: one synthetic v6 directory and one cache built from it (helpers in synthetic.py)."""

from pathlib import Path

import pytest

from tests.unit.data.synthetic import Synthetic, build_small, write_v6


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Synthetic:
    return write_v6(tmp_path_factory.mktemp("synthetic") / "v6")


@pytest.fixture(scope="session")
def cache_dir(synthetic: Synthetic, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_small(synthetic.v6, tmp_path_factory.mktemp("cache") / "v6")


@pytest.fixture(autouse=True)
def _no_copy_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic: a worker's IDUN_CACHE_COPY=1 must not change what these tests check."""
    monkeypatch.delenv("IDUN_CACHE_COPY", raising=False)
