"""The CUDA `idun.device.Device`: a context with one kernel module, and device memory the host reaches by copies.

- **Memory:** a buffer lives on the GPU. The host sees it through a page-locked mirror, made the first time `read`
  or `write` asks for the buffer; a buffer the host never asks for has none. `read` and `write` download only the
  bytes a kernel or a fill may have changed since the mirror last held them, and a pass uploads only the leading
  bytes `write` handed out since the last upload. Data given to `place` and `copy_in` is uploaded once and has no
  mirror; `placements` records how each placed array went up.
- **Order:** uploads, fills and passes queue on the context's one stream and run in the order they were encoded.
  The host waits at `flush`, at a `read` or `write` that has to download, and at a `write` of a buffer whose upload
  is still queued.
"""

import ctypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun.cuda import runtime
from idun.device import Buffer, Signature, extent, signature_of

CHUNK_BYTES = 64 << 20  # of placed data per upload
TAIL_BYTES = 16  # readable zeros after placed data: kernels may load 16 bytes at once
ROW_ALIGNMENT = 16


@dataclass(eq=False)
class CudaBuffer:
    """Device memory and what the host knows of it."""

    memory: runtime.Buffer
    mirror: runtime.HostMemory | None = None  # page-locked host memory of the same size, once the host asked
    valid: int = 0  # leading bytes of the mirror that hold the buffer's current values
    dirty: int = 0  # leading bytes the host may have changed since the last upload (at most `valid`)
    zero: bool = False  # the device memory is all zero bytes and the host changed nothing since

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        return self.memory.nbytes


def _own(buffer: Buffer) -> CudaBuffer:
    if not isinstance(buffer, CudaBuffer):
        raise TypeError(f"a CUDA device takes its own buffers, got {type(buffer).__name__}")
    return buffer


def _address(array: NDArray[Any]) -> int:
    return int(array.__array_interface__["data"][0])


class CudaDevice:
    """A CUDA context with ``source`` loaded as its module; ``table`` declares the kernels (`idun.device.Device`).

    ``device`` and ``guard`` are `idun.cuda.runtime.Context`'s. With ``profile``, the host waits for every pass and
    its GPU time is added to ``kernel_seconds``: the per-kernel timing table. It changes timing only, never results.
    ``uploaded_bytes`` and ``downloaded_bytes`` count what crossed between host and device.
    """

    def __init__(
        self,
        source: str,
        table: Mapping[str, Signature],
        *,
        profile: bool = False,
        device: int | None = None,
        guard: bool | None = None,
    ) -> None:
        self.ctx = runtime.Context(device, guard=guard)
        try:
            self.warnings = self.ctx.load(source)
        except BaseException:
            self.ctx.close()
            raise
        self.table = table
        self.kernels: dict[str, runtime.Kernel] = {}
        self.placements: dict[str, str] = {}
        self.profile = profile
        self.kernel_seconds: dict[str, float] = {}
        self.uploaded_bytes = 0
        self.downloaded_bytes = 0
        self._in_flight: set[CudaBuffer] = set()  # buffers whose upload from the mirror is queued, not waited for

    @property
    def gpu_seconds(self) -> float:
        """GPU time of the passes waited for so far, as `idun.cuda.runtime.Context.sync` measures it."""
        return self.ctx.gpu_seconds

    def kernel(self, name: str) -> runtime.Kernel:
        """Return kernel ``name`` of the module (looked up on first use)."""
        if name not in self.kernels:
            self.kernels[name] = self.ctx.kernel(name)
        return self.kernels[name]

    def _sync(self) -> float:
        seconds = self.ctx.sync()
        self._in_flight.clear()
        return seconds

    def alloc(self, nbytes: int) -> CudaBuffer:
        """Return a zero-filled buffer of ``nbytes`` (at least 1)."""
        return CudaBuffer(self.ctx.alloc(nbytes), zero=True)

    def place(self, name: str, array: NDArray[Any]) -> tuple[CudaBuffer, int]:
        """Return a buffer holding ``array`` (1-D, or 2-D with contiguous rows) and its row stride in bytes.

        The array's memory goes up as it lies (``"uploaded"``) when its rows are contiguous and, for a 2-D array,
        start a multiple of 16 bytes apart; anything else is first repacked with such rows (``"repacked"``: a
        transient host copy). 16 readable zero bytes follow the data. An empty array gets a small zeroed buffer that no
        kernel reads (``"empty"``).
        """
        if array.size == 0:
            self.placements[name] = "empty"
            return self.alloc(TAIL_BYTES), 0
        rows = array.ndim == 2
        row_bytes = array.shape[-1] * array.itemsize
        stride = int(array.strides[0]) if rows else 0
        if array.strides[-1] == array.itemsize and (not rows or (stride >= row_bytes and stride % ROW_ALIGNMENT == 0)):
            self.placements[name] = "uploaded"
            data = array
            nbytes = (array.shape[0] - 1) * stride + row_bytes if rows else array.nbytes
        else:
            self.placements[name] = "repacked"
            width = -(-array.shape[-1] // ROW_ALIGNMENT) * ROW_ALIGNMENT + ROW_ALIGNMENT
            data = np.zeros((*array.shape[:-1], width), array.dtype)
            data[..., : array.shape[-1]] = array
            stride = int(data.strides[0]) if rows else 0
            nbytes = data.nbytes
        memory = self.ctx.alloc(nbytes + TAIL_BYTES, zero=False)
        self.ctx.fill(memory, 0, TAIL_BYTES, nbytes)
        for start in range(0, nbytes, CHUNK_BYTES):
            count = min(CHUNK_BYTES, nbytes - start)
            self.ctx.upload(memory, _address(data) + start, count, start)
            self._sync()
        self.uploaded_bytes += nbytes
        return CudaBuffer(memory), stride

    def copy_in(self, array: NDArray[Any]) -> CudaBuffer:
        """Return a new buffer holding ``array`` (uploaded before the call returns)."""
        dense = np.ascontiguousarray(array)
        memory = self.ctx.alloc(dense.nbytes, zero=False)
        self.ctx.upload(memory, _address(dense), dense.nbytes)
        self._sync()
        self.uploaded_bytes += dense.nbytes
        return CudaBuffer(memory)

    def _mirror(self, own: CudaBuffer, count: int) -> runtime.HostMemory:
        """Return the buffer's mirror with its first ``count`` bytes current, downloading only what is not."""
        if own.mirror is None:
            own.mirror = self.ctx.alloc_host(own.nbytes)
        if count > own.valid and own.zero:
            if own in self._in_flight:
                self._sync()
            ctypes.memset(own.mirror.address, 0, own.nbytes)
            own.valid = own.nbytes
        if count > own.valid:
            missing = count - own.valid
            self.ctx.download(own.memory, own.mirror.address + own.valid, missing, own.valid)
            self._sync()
            self.downloaded_bytes += missing
            own.valid = count
        return own.mirror

    def read(self, buffer: Buffer, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return the buffer's current values as a read-only view of its mirror (downloaded if a pass changed them)."""
        own = _own(buffer)
        view = self._mirror(own, extent(dtype, shape, own.nbytes)).view(dtype, shape)
        view.flags.writeable = False
        return view

    def write(self, buffer: Buffer, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return the buffer's current values as a view of its mirror to change; the next pass uploads them."""
        own = _own(buffer)
        count = extent(dtype, shape, own.nbytes)
        mirror = self._mirror(own, count)
        if own in self._in_flight:  # the queued upload still reads the mirror
            self._sync()
        own.dirty = max(own.dirty, count)
        own.zero = False
        return mirror.view(dtype, shape)

    def _upload(self, own: CudaBuffer) -> None:
        """Queue the upload of what `write` handed out since the last one."""
        if own.dirty and own.mirror is not None:
            self.ctx.upload(own.memory, own.mirror.address, own.dirty)
            self.uploaded_bytes += own.dirty
            own.dirty = 0
            self._in_flight.add(own)

    def fill(self, buffer: Buffer, value: int, nbytes: int) -> None:
        """Set the buffer's first ``nbytes`` bytes to ``value`` on the device, after the passes encoded so far."""
        own = _own(buffer)
        if not 0 <= nbytes <= own.nbytes:
            raise ValueError(f"a fill of {nbytes} bytes does not fit a buffer of {own.nbytes}")
        if nbytes == 0 or (own.zero and value == 0):
            return
        if own.dirty > nbytes:  # the host's changes past ``nbytes`` survive the fill
            self._upload(own)
        own.dirty = 0
        self.ctx.fill(own.memory, value, nbytes)
        own.valid = 0
        own.zero = value == 0 and nbytes == own.nbytes

    def encode(
        self,
        name: str,
        buffers: Sequence[Buffer],
        params: ctypes.Structure | None,
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        local_bytes: int = 0,
    ) -> None:
        """Queue kernel ``name`` after the uploads its buffers need; ``local_bytes`` of dynamic shared memory."""
        signature = signature_of(self.table, name, len(buffers), params)
        own = [_own(buffer) for buffer in buffers]
        kernel = self.kernel(name)
        for bound in own:
            self._upload(bound)
        if self.profile:
            self._sync()
        self.ctx.launch(kernel, [bound.memory for bound in own], params, groups, threads, local_bytes)
        for index in signature.writes:
            own[index].valid = 0
            own[index].zero = False
        if self.profile:
            self.kernel_seconds[name] = self.kernel_seconds.get(name, 0.0) + self._sync()

    def submit(self) -> None:
        """Start the encoded passes: they are on the stream already, so nothing waits (a read synchronises it)."""

    def flush(self) -> None:
        """Wait until every encoded pass, fill and transfer has run."""
        self._sync()

    def close(self) -> None:
        """Free every buffer, every mirror and the context."""
        self.ctx.close()
