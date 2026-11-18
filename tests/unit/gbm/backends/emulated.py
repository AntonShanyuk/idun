"""The CUDA backend without a GPU: the fake driver, its kernels run by the Metal kernels of the same names.

`idun.cuda.device.CudaDevice` and the passes above it then run whole on a Mac (the stream's queue, the mirrors, what
is uploaded and downloaded, and when), on device memory the host reaches by copies only. What still needs an NVIDIA
GPU is the CUDA kernel source itself.
"""

import gc
from collections.abc import Mapping
from functools import partial

import numpy as np
import pytest

from idun.device import Signature
from idun.gbm.backends import metal as metal_module
from idun.gbm.backends.device import TABLE
from idun.metal.device import MetalDevice
from idun.metal.runtime import Buffer
from tests.unit.cuda import fake_driver
from tests.unit.cuda.fake_driver import FakeDriver, Kernel, Launch


class Emulation:
    """A Metal device that runs the fake driver's launches on copies of the fake device memory.

    ``source`` and ``table``: the Metal library whose kernels stand in for the CUDA ones of the same names (idun_gbm's
    by default; the networks' tests pass theirs).
    """

    def __init__(self, driver: FakeDriver, source: str | None = None, table: Mapping[str, Signature] = TABLE) -> None:
        self.driver = driver
        self.table = table
        self.metal = MetalDevice(metal_module.KERNELS.read_text() if source is None else source, table)
        self.twins: dict[int, Buffer] = {}  # a fake allocation's address: its Metal copy

    def twin(self, driver: FakeDriver, pointer: int) -> tuple[Buffer, int]:
        assert pointer in driver.memory, "a kernel is bound to a buffer's start (guard mode is not emulated)"
        nbytes = len(driver.memory[pointer])
        if pointer not in self.twins or self.twins[pointer].nbytes != nbytes:
            self.twins[pointer] = self.metal.alloc(nbytes)
        return self.twins[pointer], nbytes

    def run(self, name: str, driver: FakeDriver, launch: Launch) -> None:
        """Run kernel ``name`` on Metal with the launch's memory, grid and parameters; copy back what it may write."""
        bound = [self.twin(driver, pointer) for pointer in launch.pointers]
        for pointer, (buffer, nbytes) in zip(launch.pointers, bound, strict=True):
            self.metal.write(buffer, np.uint8, (nbytes,))[:] = driver.view(pointer, np.uint8, nbytes)
        buffers = [buffer for buffer, _ in bound]
        self.metal.encode(name, buffers, launch.params, launch.grid, launch.block, launch.shared_bytes)
        self.metal.flush()
        for index in self.table[name].writes:
            buffer, nbytes = bound[index]
            driver.view(launch.pointers[index], np.uint8, nbytes)[:] = self.metal.read(buffer, np.uint8, (nbytes,))

    def close(self) -> None:
        """Collect what the test left while this driver can still close it, then require nothing left open.

        A backend the garbage collector reaches later would close its context under another test's driver.
        """
        gc.collect()
        left = self.driver.live()
        self.metal.close()
        assert left == {}, f"the test left CUDA resources open: {left}"


def install(
    monkeypatch: pytest.MonkeyPatch,
    devices: int = 1,
    source: str | None = None,
    table: Mapping[str, Signature] = TABLE,
) -> Emulation:
    """Make `idun.cuda.runtime` use a fresh fake driver that runs a library's kernels (idun_gbm's by default) on Metal.

    Close the result at the end.
    """
    fakes = fake_driver.install(monkeypatch, devices)
    emulation = Emulation(fakes.driver, source, table)
    for name, signature in table.items():
        fakes.driver.kernels[name] = Kernel(signature.buffers, signature.params, partial(emulation.run, name))
    return emulation
