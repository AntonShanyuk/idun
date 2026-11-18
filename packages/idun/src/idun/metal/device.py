"""The Metal `idun.device.Device`: a context with one kernel library, lazily encoded passes and unified memory.

- **Memory:** the GPU's memory is the host's, so `read` and `write` return views of it and nothing is copied.
  Page-aligned memory-mapped data is wrapped without a copy; other data arrays are copied into shared buffers, and
  `placements` records which, so nothing happens silently.
- **Batches:** a pass is encoded into the open command batch and runs when the batch is committed: at `flush`, before
  `read` returns a buffer that a pending pass writes, and before `write` returns one that a pending pass uses.
"""

import ctypes
import mmap
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun.device import Buffer, Signature, extent, signature_of
from idun.errors import BackendUnavailableError
from idun.metal import runtime

PAGE = runtime.PAGE_BYTES


def _mapping(array: NDArray[Any]) -> mmap.mmap | None:
    """Return the ``mmap`` an array's memory comes from, if any."""
    base: Any = array
    while base is not None:
        if isinstance(base, mmap.mmap):
            return base
        if isinstance(base, memoryview):  # np.frombuffer over a map: the view's object is the map
            base = base.obj
            continue
        base = getattr(base, "_mmap", None) or getattr(base, "base", None)
    return None


def _own(buffer: Buffer) -> runtime.Buffer:
    if not isinstance(buffer, runtime.Buffer):
        raise TypeError(f"a Metal device takes its own buffers, got {type(buffer).__name__}")
    return buffer


class MetalDevice:
    """A Metal context with ``source`` loaded as its library; ``table`` declares the kernels (`idun.device.Device`).

    With ``profile``, every pass is committed on its own and its GPU time is added to ``kernel_seconds``: the
    per-kernel timing table. It changes timing only, never results. Apple7-family GPUs (M1) and newer.
    """

    def __init__(self, source: str, table: Mapping[str, Signature], *, profile: bool = False) -> None:
        if not runtime.probe().supports_apple7:
            raise BackendUnavailableError("the Metal backend needs an Apple7-family GPU or newer (M1 and later)")
        self.ctx = runtime.Context()
        self.ctx.load(source)
        self.table = table
        self.kernels: dict[str, runtime.Kernel] = {}
        self._pending: set[int] = set()  # the buffers the open batch's passes use
        self._written: set[int] = set()  # those of them a pass writes
        self._flight: list[tuple[set[int], set[int]]] = []  # submitted batches not waited for: (used, written)
        self.placements: dict[str, str] = {}
        self.profile = profile
        self.kernel_seconds: dict[str, float] = {}

    @property
    def gpu_seconds(self) -> float:
        """GPU time of every committed batch so far."""
        return self.ctx.gpu_seconds

    def kernel(self, name: str) -> runtime.Kernel:
        """Return the compute pipeline of kernel ``name`` (made on first use)."""
        if name not in self.kernels:
            self.kernels[name] = self.ctx.kernel(name)
        return self.kernels[name]

    def alloc(self, nbytes: int) -> runtime.Buffer:
        """Return a zero-filled shared buffer of ``nbytes`` (at least 1)."""
        return self.ctx.alloc(nbytes)

    def place(self, name: str, array: NDArray[Any]) -> tuple[runtime.Buffer, int]:
        """Return a buffer holding ``array`` (1-D, or 2-D with contiguous rows) and its row stride in bytes.

        A page-aligned memory map with contiguous rows is wrapped; anything else is copied with rows of a multiple of 16
        elements and 16 readable elements past the end (kernels may load 16 bytes at once). An empty array gets a small
        zeroed buffer that no kernel reads.
        """
        if array.size == 0:
            self.placements[name] = "empty"
            return self.ctx.alloc(PAGE), 0
        stride = int(array.strides[0]) if array.ndim == 2 else 0
        row = array.shape[-1] * array.itemsize
        extent_bytes = (array.shape[0] - 1) * stride + row if array.ndim == 2 else array.nbytes
        address = int(array.__array_interface__["data"][0])
        rounded = -(-max(extent_bytes, 1) // PAGE) * PAGE
        mapping = _mapping(array)
        if mapping is not None and address % PAGE == 0 and array.strides[-1] == array.itemsize:
            start = int(np.frombuffer(mapping, np.uint8).__array_interface__["data"][0])
            if address == start and rounded <= -(-len(mapping) // PAGE) * PAGE:
                self.placements[name] = "wrapped"
                return self.ctx.wrap(array, rounded), stride
        width = -(-array.shape[-1] // 16) * 16 + 16
        dense = np.zeros((*array.shape[:-1], width), array.dtype)
        dense[..., : array.shape[-1]] = array
        buffer = self.ctx.alloc(dense.nbytes)
        buffer.view(dense.dtype.type, dense.shape)[...] = dense
        self.placements[name] = "copied"
        return buffer, int(dense.strides[0]) if dense.ndim == 2 else 0

    def copy_in(self, array: NDArray[Any]) -> runtime.Buffer:
        """Return a new buffer holding ``array``."""
        dense = np.ascontiguousarray(array)
        buffer = self.ctx.alloc(dense.nbytes)
        buffer.view(dense.dtype.type, dense.shape)[...] = dense
        return buffer

    def read(self, buffer: Buffer, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return a read-only view of the buffer's memory; a pending pass that writes it is committed first."""
        own = _own(buffer)
        extent(dtype, shape, own.nbytes)
        self._settle(own.handle, writers=True)
        if own.handle in self._written:
            self.flush()
        view = own.view(dtype, shape)
        view.flags.writeable = False
        return view

    def write(self, buffer: Buffer, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return a view of the buffer's memory to change; a pending pass that uses it is committed first."""
        own = _own(buffer)
        extent(dtype, shape, own.nbytes)
        self._settle(own.handle, writers=False)
        if own.handle in self._pending:
            self.flush()
        return own.view(dtype, shape)

    def _settle(self, handle: int, *, writers: bool) -> None:
        """Wait for the submitted batches up to the newest one that writes (or, not ``writers``, uses) the buffer."""
        last = max((i for i, (used, written) in enumerate(self._flight) if handle in (written if writers else used)),
                   default=None)  # fmt: skip
        if last is not None:
            self.ctx.wait(keep=len(self._flight) - last - 1)  # batches run in commit order
            del self._flight[: last + 1]

    def fill(self, buffer: Buffer, value: int, nbytes: int) -> None:
        """Set the buffer's first ``nbytes`` bytes to ``value`` (a pending pass that uses it is committed first)."""
        self.write(buffer, np.uint8, (nbytes,))[:] = value

    def encode(
        self,
        name: str,
        buffers: Sequence[Buffer],
        params: ctypes.Structure | None,
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        local_bytes: int = 0,
    ) -> None:
        """Encode kernel ``name`` into the open batch (opened if none is); ``local_bytes`` of threadgroup memory."""
        signature = signature_of(self.table, name, len(buffers), params)
        own = [_own(buffer) for buffer in buffers]
        if self.profile:
            self.flush()
        if not self.ctx.is_open:
            self.ctx.begin()
        self.ctx.dispatch(self.kernel(name), own, params, groups, threads, local_bytes)
        self._pending.update(b.handle for b in own)
        self._written.update(own[index].handle for index in signature.writes)
        if self.profile:
            seconds = self.ctx.commit()
            self._pending.clear()
            self._written.clear()
            self.kernel_seconds[name] = self.kernel_seconds.get(name, 0.0) + seconds

    def submit(self) -> None:
        """Commit the open batch without waiting for it (a profiling device has committed every pass already)."""
        if self.ctx.is_open:
            self.ctx.commit_async()
            self._flight.append((set(self._pending), set(self._written)))
        self._pending.clear()
        self._written.clear()

    def flush(self) -> None:
        """Commit the open batch, if any, and wait for it and every submitted batch."""
        if self.ctx.is_open:
            self.ctx.commit()  # waits for the batches committed before it too
        elif self._flight:
            self.ctx.wait()
        self._flight.clear()
        self._pending.clear()
        self._written.clear()

    def close(self) -> None:
        """Free every buffer and the context."""
        self.ctx.close()
