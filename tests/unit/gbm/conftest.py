"""Fixtures of idun_gbm's tests: the GPU backends a test runs on."""

from collections.abc import Iterator

import pytest

from idun.cuda import runtime
from tests.unit.gbm.backends import emulated

GPUS = [
    pytest.param("metal", marks=pytest.mark.metal),
    pytest.param("cuda", marks=pytest.mark.cuda),
    pytest.param("cuda-emulated", marks=pytest.mark.metal),
]


@pytest.fixture(params=GPUS)
def gpu(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The name of a GPU backend: Metal, CUDA on a GPU, and CUDA over the fake driver with Metal running the kernels.

    The last runs `idun.cuda` and the CUDA backend whole on a Mac (`tests.unit.gbm.backends.emulated`); what it
    cannot run is the CUDA kernel source.
    """
    if request.param != "cuda-emulated":
        yield str(request.param)
        return
    monkeypatch.delenv(runtime.GUARD_ENV, raising=False)
    monkeypatch.delenv(runtime.DEVICE_ENV, raising=False)
    emulation = emulated.install(monkeypatch)
    yield "cuda"
    emulation.close()  # collects first: a backend the test left closes its context under this driver
