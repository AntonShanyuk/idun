"""Fixtures of the CUDA tests: the fake libraries, and tests that run against them and against a real GPU."""

import pytest

from idun.cuda import runtime
from tests.unit.cuda.fake_driver import Fakes, install


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    """The runtime on fresh fake libraries, with no device or guard mode named by the environment."""
    monkeypatch.delenv(runtime.GUARD_ENV, raising=False)
    monkeypatch.delenv(runtime.DEVICE_ENV, raising=False)
    return install(monkeypatch)


@pytest.fixture(params=["fake", pytest.param("gpu", marks=pytest.mark.cuda)])
def anywhere(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run a test against the fake libraries, and against a real GPU where the ``cuda`` tests are selected."""
    if request.param == "fake":
        install(monkeypatch)
    return str(request.param)
