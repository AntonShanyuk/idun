"""A device with memory of its own, played on Metal: it checks what a discrete GPU needs from the host.

On Metal the host and the GPU share memory, so a learner that forgets to ask for a buffer again after a pass, writes
an array it was handed too late, or misdeclares what a kernel writes still gets the right numbers. On a GPU with its
own memory it would not. `MirroredDevice` wraps a device and gives the host copies only:

- ``read`` and ``write`` return views of a host mirror, downloaded when a pass or a fill changed the device's copy
  since the last download. An array handed out earlier keeps its old values (a missing ``read`` shows as stale data).
- What ``write`` handed out is uploaded before the next pass that binds the buffer, and before a fill. A host write
  after that never reaches the device, and the next use of the buffer raises `HostAccessError`.
- Every pass runs on its own. A buffer that the kernel's signature declares read-only and that changed raises
  `HostAccessError` (buffers above `CHECK_BYTES` are not checked).
- Data given to ``place`` has no mirror: the host cannot read or write it.

Results are those of the wrapped device, bit for bit. `open_device` returns one instead of the plain Metal device when
the environment variable ``IDUN_METAL_MIRRORED`` is ``1``: the test suites run a second time that way.
"""

import ctypes
import os
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun.device import Buffer, Device, Signature, extent, signature_of
from idun.errors import IdunError
from idun.metal.device import MetalDevice

ENV = "IDUN_METAL_MIRRORED"
CHECK_BYTES = 64 << 20  # read-only buffers up to this size are compared around every pass


class HostAccessError(IdunError):
    """The host or a kernel touched a buffer in a way that a device with memory of its own cannot serve."""


@dataclass(eq=False)
class MirroredBuffer:
    """A wrapped device's buffer, the host's mirror of it and what each side changed since the last copy."""

    inner: Buffer
    owner: object
    placed: bool = False
    mirror: NDArray[np.uint8] | None = None
    dirty: int = 0  # leading bytes of the mirror the host may have changed since the last upload
    newer: bool = True  # the device's copy changed since the last download (or was never downloaded)
    handed: list[NDArray[Any]] = field(default_factory=list)  # arrays from write() since the last upload
    retired: tuple[NDArray[np.uint8], int] | None = None  # the mirror uploaded last and its checksum then

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        return self.inner.nbytes


class MirroredDevice:
    """``inner`` with discrete-memory semantics for the host (`idun.device.Device`); ``table`` declares the kernels."""

    def __init__(self, inner: Device, table: Mapping[str, Signature]) -> None:
        self.inner = inner
        self.table = table
        self._buffers: list[MirroredBuffer] = []

    @property
    def placements(self) -> Mapping[str, str]:
        """How each array given to `place` reached the wrapped device."""
        return self.inner.placements

    @property
    def gpu_seconds(self) -> float:
        """GPU time of the wrapped device's finished passes."""
        return self.inner.gpu_seconds

    @property
    def kernel_seconds(self) -> Mapping[str, float]:
        """GPU seconds per kernel of the wrapped device (when it profiles)."""
        return self.inner.kernel_seconds

    def _new(self, inner: Buffer, *, placed: bool = False) -> MirroredBuffer:
        buffer = MirroredBuffer(inner, self, placed)
        self._buffers.append(buffer)
        return buffer

    def _own(self, buffer: Buffer, *, host: bool) -> MirroredBuffer:
        if not isinstance(buffer, MirroredBuffer) or buffer.owner is not self:
            raise TypeError(f"a mirrored device takes its own buffers, got {type(buffer).__name__}")
        if host and buffer.placed:
            raise HostAccessError("the host cannot read or write data given to place()")
        self._check_retired(buffer)
        return buffer

    @staticmethod
    def _check_retired(buffer: MirroredBuffer) -> None:
        if buffer.retired is not None and zlib.crc32(buffer.retired[0]) != buffer.retired[1]:
            buffer.retired = None
            raise HostAccessError("the host wrote an array from write() after the pass that used the buffer")

    def _download(self, buffer: MirroredBuffer) -> NDArray[np.uint8]:
        if buffer.newer or buffer.mirror is None:
            buffer.mirror = np.array(self.inner.read(buffer.inner, np.uint8, (buffer.nbytes,)))
            buffer.newer = False
        return buffer.mirror

    def _upload(self, buffer: MirroredBuffer) -> None:
        if buffer.dirty and buffer.mirror is not None:
            self.inner.write(buffer.inner, np.uint8, (buffer.dirty,))[:] = buffer.mirror[: buffer.dirty]
            for array in buffer.handed:
                array.flags.writeable = False
            buffer.retired = (buffer.mirror, zlib.crc32(buffer.mirror))
            buffer.mirror = buffer.mirror.copy()  # later views do not alias the arrays handed out before
        buffer.handed.clear()
        buffer.dirty = 0

    def alloc(self, nbytes: int) -> MirroredBuffer:
        """Return a zero-filled buffer of ``nbytes`` (at least 1)."""
        return self._new(self.inner.alloc(nbytes))

    def place(self, name: str, array: NDArray[Any]) -> tuple[MirroredBuffer, int]:
        """Return a buffer holding data ``array`` and its row stride in bytes; the host cannot reach it again."""
        buffer, stride = self.inner.place(name, array)
        return self._new(buffer, placed=True), stride

    def copy_in(self, array: NDArray[Any]) -> MirroredBuffer:
        """Return a new buffer holding ``array``."""
        return self._new(self.inner.copy_in(array))

    def read(self, buffer: Buffer, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return the buffer's current values as a read-only array (downloaded if the device's copy changed)."""
        own = self._own(buffer, host=True)
        count = extent(dtype, shape, own.nbytes)
        view = self._download(own)[:count].view(dtype).reshape(shape)
        view.flags.writeable = False
        return view

    def write(self, buffer: Buffer, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return the buffer's current values as an array to change; uploaded before the next pass that binds it."""
        own = self._own(buffer, host=True)
        count = extent(dtype, shape, own.nbytes)
        view = self._download(own)[:count].view(dtype).reshape(shape)
        own.dirty = max(own.dirty, count)
        own.handed.append(view)
        return view

    def fill(self, buffer: Buffer, value: int, nbytes: int) -> None:
        """Set the buffer's first ``nbytes`` bytes to ``value`` on the device (the host's changes go first)."""
        own = self._own(buffer, host=True)
        self._upload(own)
        self.inner.fill(own.inner, value, nbytes)
        own.newer = True

    def encode(
        self,
        name: str,
        buffers: Sequence[Buffer],
        params: ctypes.Structure | None,
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        local_bytes: int = 0,
    ) -> None:
        """Upload what the host changed, run kernel ``name`` on its own and check its write declarations."""
        signature = signature_of(self.table, name, len(buffers), params)
        own = [self._own(buffer, host=False) for buffer in buffers]
        for buffer in own:
            self._upload(buffer)
        written = [own[index] for index in signature.writes]
        watched = {
            index: self._checksum(buffer)
            for index, buffer in enumerate(own)
            if buffer.nbytes <= CHECK_BYTES and all(buffer is not other for other in written)
        }
        self.inner.encode(name, [buffer.inner for buffer in own], params, groups, threads, local_bytes)
        self.inner.flush()
        for index, before in watched.items():
            if self._checksum(own[index]) != before:
                raise HostAccessError(f"kernel {name} wrote buffer {index}, which its signature declares read-only")
        for buffer in written:
            buffer.newer = True

    def _checksum(self, buffer: MirroredBuffer) -> int:
        return zlib.crc32(self.inner.read(buffer.inner, np.uint8, (buffer.nbytes,)))

    def submit(self) -> None:
        """Start the encoded passes (each has run already)."""
        self.inner.flush()

    def flush(self) -> None:
        """Wait until every encoded pass has run (each already has)."""
        self.inner.flush()

    def close(self) -> None:
        """Check the arrays handed out last, then close the wrapped device."""
        try:
            for buffer in self._buffers:
                self._check_retired(buffer)
        finally:
            self._buffers.clear()
            self.inner.close()


def open_device(source: str, table: Mapping[str, Signature], *, profile: bool = False) -> Device:
    """Return a Metal device for ``source``; a `MirroredDevice` over it when ``IDUN_METAL_MIRRORED`` is ``1``.

    ``ValueError`` when the variable is set to anything else.
    """
    mode = os.environ.get(ENV)
    if mode not in (None, "1"):
        raise ValueError(f"{ENV} must be 1 or unset, got {mode!r}")
    device = MetalDevice(source, table, profile=profile)
    return MirroredDevice(device, table) if mode == "1" else device
