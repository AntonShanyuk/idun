"""What a learner asks of a GPU device, whichever device runs its kernels.

A learner drives its kernels through a `Device`. It holds buffers it cannot look into, names its kernels in a table
of `Signature`, and reaches a buffer's values from the host through these calls only:

- `Device.read`: the buffer's current values (passes that write it finish first). The array is read-only and shows
  them until the next pass that writes the buffer; ask again after it.
- `Device.write`: the buffer's current values in an array the host may change. The changes reach the device before
  the next pass that uses the buffer; the array must not be written once that pass is encoded (ask again).
- `Device.fill`: the first bytes of a buffer set to one value, on the device.
- `Device.copy_in` and `Device.place`: a new buffer holding an array.

A device that shares its memory with the host (Metal) hands out that memory and copies nothing. A device with memory
of its own downloads at `read` what a kernel changed and uploads before a pass what `write` handed out, which is why
every kernel declares the buffers it writes.
"""

import ctypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Signature:
    """A kernel's buffers: how many it binds, which of them it writes (by index), and its parameter structure."""

    buffers: int
    writes: tuple[int, ...]
    params: type[ctypes.Structure] | None

    def __post_init__(self) -> None:
        if self.buffers < 0 or any(not 0 <= index < self.buffers for index in self.writes):
            raise ValueError(f"writes {self.writes} are not indices of {self.buffers} buffers")
        if len(set(self.writes)) != len(self.writes):
            raise ValueError(f"writes {self.writes} name a buffer twice")


def signature_of(table: Mapping[str, Signature], name: str, n_buffers: int, params: object) -> Signature:
    """Return kernel ``name``'s signature; ``ValueError`` unless the call binds what the signature declares."""
    if name not in table:
        raise ValueError(f"kernel {name!r} is not in the kernel table")
    signature = table[name]
    if n_buffers != signature.buffers:
        raise ValueError(f"kernel {name} binds {signature.buffers} buffers, got {n_buffers}")
    expected = type(None) if signature.params is None else signature.params
    if type(params) is not expected:
        raise ValueError(f"kernel {name} takes {expected.__name__} parameters, got {type(params).__name__}")
    return signature


def extent(dtype: type[np.generic], shape: Sequence[int], nbytes: int) -> int:
    """Return the bytes of a ``shape`` array of ``dtype``; ``ValueError`` when a buffer of ``nbytes`` is smaller."""
    count = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if count > nbytes:
        raise ValueError(f"a {tuple(shape)} {np.dtype(dtype)} array needs {count} bytes; the buffer has {nbytes}")
    return count


class Buffer(Protocol):
    """Device memory. Only the device that made it reads, writes or binds it."""

    @property
    def nbytes(self) -> int:
        """Size in bytes."""


class Device(Protocol):
    """A GPU context with one kernel library loaded: buffers, passes over them, and the host's access to them.

    Passes may run later than they are encoded; `flush`, `read` and `write` wait for the ones they depend on.
    Results never depend on when passes run.
    """

    @property
    def placements(self) -> Mapping[str, str]:
        """How each array given to `place` reached the device, by name."""

    @property
    def gpu_seconds(self) -> float:
        """GPU time of the finished passes."""

    @property
    def kernel_seconds(self) -> Mapping[str, float]:
        """GPU seconds per kernel; filled only by a profiling device (which runs every pass on its own)."""

    def alloc(self, nbytes: int) -> Buffer:
        """Return a zero-filled buffer of ``nbytes`` (at least 1)."""

    def place(self, name: str, array: NDArray[Any]) -> tuple[Buffer, int]:
        """Return a buffer holding data ``array`` (1-D, or 2-D with contiguous rows) and its row stride in bytes.

        The host never reads or writes this buffer again; `placements` records under ``name`` how it got there.
        """

    def copy_in(self, array: NDArray[Any]) -> Buffer:
        """Return a new buffer holding ``array``."""

    def read(self, buffer: Buffer, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return the buffer's current values as a read-only ``shape`` array of ``dtype``."""

    def write(self, buffer: Buffer, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return the buffer's current values as a ``shape`` array of ``dtype`` that the host may change."""

    def fill(self, buffer: Buffer, value: int, nbytes: int) -> None:
        """Set the buffer's first ``nbytes`` bytes to ``value`` (0..255)."""

    def encode(
        self,
        name: str,
        buffers: Sequence[Buffer],
        params: ctypes.Structure | None,
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        local_bytes: int = 0,
    ) -> None:
        """Encode kernel ``name`` over ``groups`` thread groups of ``threads`` threads.

        ``local_bytes`` (a multiple of 16) sizes the scratch memory each group's threads share.
        """

    def submit(self) -> None:
        """Start the encoded passes without waiting for them: the host may encode the next ones meanwhile.

        A later `read` of a buffer they write, `write` or `fill` of a buffer they use, or `flush`, waits for them.
        """

    def flush(self) -> None:
        """Wait until every encoded pass has run."""

    def close(self) -> None:
        """Release the device's memory and context. Arrays from `read` and `write` must not be used afterwards."""
