"""ctypes binding to libidun_metal, the Metal host runtime (C ABI: native/include/idun_metal.h).

It exposes the device probe, runtime compilation of MSL source with idun's fixed compile options, and a `Context`:
compute kernels, shared and no-copy buffers, and command batches with GPU timestamps. Nothing here falls back: a missing
library, an ABI mismatch or a missing GPU raises :class:`~idun.errors.BackendUnavailableError`.
"""

import ctypes
import functools
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun.errors import BackendUnavailableError, IdunError, KernelCompileError

ABI_VERSION = 6
PAGE_BYTES = 16 * 1024  # Apple silicon's page: wrapped memory must start and end on one
LIBRARY_ENV = "IDUN_METAL_LIB"
LIBRARY_NAME = "libidun_metal.dylib"

_IDUN_OK = 0
_IDUN_ERR_NO_DEVICE = -1
_IDUN_ERR_COMPILE = -2
_ERR_LEN = 1 << 16
MAX_BUFFERS = 30


class _IdunDeviceInfo(ctypes.Structure):
    _fields_ = (
        ("name", ctypes.c_char * 128),
        ("max_threadgroup_memory_bytes", ctypes.c_uint64),
        ("recommended_working_set_bytes", ctypes.c_uint64),
        ("max_buffer_bytes", ctypes.c_uint64),
        ("max_threads_per_threadgroup", ctypes.c_uint32),
        ("page_size_bytes", ctypes.c_uint32),
        ("has_unified_memory", ctypes.c_uint32),
        ("supports_apple7", ctypes.c_uint32),
        ("supports_apple9", ctypes.c_uint32),
        ("supports_metal4", ctypes.c_uint32),
    )


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """What the system default Metal device offers. Field meanings follow ``idun_device_info`` in idun_metal.h."""

    name: str
    max_threadgroup_memory_bytes: int
    recommended_working_set_bytes: int
    max_buffer_bytes: int
    max_threads_per_threadgroup: int
    page_size_bytes: int
    has_unified_memory: bool
    supports_apple7: bool
    supports_apple9: bool
    supports_metal4: bool


def library_path() -> Path:
    """Return the path of libidun_metal: ``$IDUN_METAL_LIB`` if set, else the package's copy.

    The macOS arm64 wheel installs that copy; in a source checkout ``make native`` builds it; the pure
    ``py3-none-any`` wheel has none.
    """
    override = os.environ.get(LIBRARY_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "_native" / LIBRARY_NAME


def check_abi(found: int, path: Path) -> None:
    """Raise unless the library at ``path`` reports the ABI version this binding was written for."""
    if found != ABI_VERSION:
        raise BackendUnavailableError(
            f"{path} implements Metal runtime ABI {found}, this binding expects {ABI_VERSION}: "
            "rebuild with `make native`"
        )


def error_for(code: int, message: str) -> IdunError:
    """Map an ``IDUN_ERR_*`` code and its message to the exception the binding raises."""
    if code == _IDUN_ERR_NO_DEVICE:
        return BackendUnavailableError(message)
    if code == _IDUN_ERR_COMPILE:
        return KernelCompileError(message)
    return IdunError(f"libidun_metal error {code}: {message}")


@functools.cache
def load_library() -> ctypes.CDLL:
    """Load libidun_metal once per process and declare its C signatures.

    Raises
    ------
    BackendUnavailableError
        Not on macOS, the library is missing or cannot load (an older macOS), or its ABI version does not match.
    """
    if sys.platform != "darwin":
        raise BackendUnavailableError(f"the Metal backend needs macOS, this is {sys.platform}")
    path = library_path()
    if not path.is_file():
        raise BackendUnavailableError(
            f"{path} does not exist: this installation has no Metal library (the pure py3-none-any wheel, for "
            "Linux and Intel Macs, which runs the CPU backends). On Apple silicon install the macosx arm64 wheel; in "
            "a source checkout build it with `make native`"
        )
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise BackendUnavailableError(
            f"{path} cannot be loaded: the Metal backend needs Apple silicon and macOS 15 or later ({exc})"
        ) from exc
    lib.idun_abi_version.argtypes = ()
    lib.idun_abi_version.restype = ctypes.c_int32
    lib.idun_source_prelude.argtypes = ()
    lib.idun_source_prelude.restype = ctypes.c_char_p
    lib.idun_probe.argtypes = (ctypes.POINTER(_IdunDeviceInfo), ctypes.c_char_p, ctypes.c_size_t)
    lib.idun_probe.restype = ctypes.c_int32
    lib.idun_compile_check.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t)
    lib.idun_compile_check.restype = ctypes.c_int32
    err = (ctypes.c_char_p, ctypes.c_size_t)
    handle = ctypes.c_void_p
    lib.idun_context_create.argtypes = err
    lib.idun_context_create.restype = handle
    lib.idun_context_destroy.argtypes = (handle,)
    lib.idun_context_destroy.restype = None
    lib.idun_library_load.argtypes = (handle, ctypes.c_char_p, *err)
    lib.idun_library_load.restype = ctypes.c_int32
    lib.idun_kernel_get.argtypes = (handle, ctypes.c_char_p, *err)
    lib.idun_kernel_get.restype = handle
    lib.idun_kernel_max_threads.argtypes = (handle,)
    lib.idun_kernel_max_threads.restype = ctypes.c_uint32
    lib.idun_kernel_simd_width.argtypes = (handle,)
    lib.idun_kernel_simd_width.restype = ctypes.c_uint32
    lib.idun_buffer_alloc.argtypes = (handle, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), *err)
    lib.idun_buffer_alloc.restype = handle
    lib.idun_buffer_wrap.argtypes = (handle, ctypes.c_void_p, ctypes.c_size_t, *err)
    lib.idun_buffer_wrap.restype = handle
    lib.idun_buffer_free.argtypes = (handle,)
    lib.idun_buffer_free.restype = None
    lib.idun_batch_begin.argtypes = (handle, *err)
    lib.idun_batch_begin.restype = ctypes.c_int32
    uint3 = ctypes.c_uint32 * 3
    lib.idun_dispatch.argtypes = (
        handle,
        handle,
        ctypes.POINTER(handle),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        uint3,
        uint3,
        *err,
    )
    lib.idun_dispatch.restype = ctypes.c_int32
    lib.idun_batch_commit_wait.argtypes = (handle, ctypes.POINTER(ctypes.c_double), *err)
    lib.idun_batch_commit_wait.restype = ctypes.c_int32
    lib.idun_batch_commit.argtypes = (handle, *err)
    lib.idun_batch_commit.restype = ctypes.c_int32
    lib.idun_batch_wait.argtypes = (handle, ctypes.c_uint32, ctypes.POINTER(ctypes.c_double), *err)
    lib.idun_batch_wait.restype = ctypes.c_int32
    check_abi(int(lib.idun_abi_version()), path)
    return lib


def source_prelude() -> str:
    """Return the text the runtime prepends to every kernel library before compiling it."""
    prelude: bytes = load_library().idun_source_prelude()
    return prelude.decode()


def side_by_side(bytes_each: int) -> int:
    """Return how many jobs, each referencing ``bytes_each`` bytes on the GPU, fit the device's working set at once.

    At least 1. Jobs beyond it thrash: on an 8 GB Apple M1 (5.7 GB recommended working set) two RealMLP fits of about
    4.2 GB each ran 16x slower than one, the GPU re-mapping their memory at every switch between them.
    """
    if bytes_each < 1:
        raise ValueError(f"bytes_each must be positive, got {bytes_each}")
    return max(1, probe().recommended_working_set_bytes // bytes_each)


def workers_side_by_side(requested: int | None, default: int, bytes_each: int) -> int:
    """Return how many jobs to run side by side: ``requested``, or ``default`` capped by `side_by_side` when None.

    A ``requested`` count beyond what fits raises ``ValueError`` (it would thrash, not run faster).
    """
    room = side_by_side(bytes_each)
    if requested is None:
        return min(default, room)
    if requested > room:
        need = requested * bytes_each / 2**30
        raise ValueError(f"{requested} jobs side by side need about {need:.1f} GiB of the GPU's working set, which "
                         f"holds {room} (more thrash); pass at most {room}")  # fmt: skip
    return requested


def probe() -> DeviceInfo:
    """Describe the system default Metal device.

    Raises
    ------
    BackendUnavailableError
        No Metal device, or the runtime library cannot be loaded.
    """
    raw = _IdunDeviceInfo()
    err = ctypes.create_string_buffer(_ERR_LEN)
    code = int(load_library().idun_probe(ctypes.byref(raw), err, _ERR_LEN))
    if code != _IDUN_OK:
        raise error_for(code, err.value.decode())
    return DeviceInfo(
        name=raw.name.decode(),
        max_threadgroup_memory_bytes=raw.max_threadgroup_memory_bytes,
        recommended_working_set_bytes=raw.recommended_working_set_bytes,
        max_buffer_bytes=raw.max_buffer_bytes,
        max_threads_per_threadgroup=raw.max_threads_per_threadgroup,
        page_size_bytes=raw.page_size_bytes,
        has_unified_memory=bool(raw.has_unified_memory),
        supports_apple7=bool(raw.supports_apple7),
        supports_apple9=bool(raw.supports_apple9),
        supports_metal4=bool(raw.supports_metal4),
    )


def compile_check(source: str) -> None:
    """Compile MSL ``source`` with idun's fixed options and prelude, and discard the result.

    Raises
    ------
    KernelCompileError
        The source does not compile; the message is the compiler log.
    BackendUnavailableError
        No Metal device, or the runtime library cannot be loaded.
    """
    err = ctypes.create_string_buffer(_ERR_LEN)
    code = int(load_library().idun_compile_check(source.encode(), err, _ERR_LEN))
    if code != _IDUN_OK:
        raise error_for(code, err.value.decode())


def _check(code: int, err: ctypes.Array[ctypes.c_char]) -> None:
    if code != _IDUN_OK:
        raise error_for(code, err.value.decode())


def _handle(value: int | None, err: ctypes.Array[ctypes.c_char]) -> int:
    if not value:
        raise IdunError(f"libidun_metal: {err.value.decode()}")
    return value


@dataclass(frozen=True)
class Kernel:
    """A compute pipeline of the context's library."""

    name: str
    handle: int
    max_threads: int
    simd_width: int


class Buffer:
    """A shared-storage Metal buffer: allocated by the device, or wrapping host memory without a copy."""

    def __init__(self, handle: int, nbytes: int, address: int, owner: object = None) -> None:
        self.handle, self.nbytes, self.address = handle, nbytes, address
        self._owner = owner  # keeps wrapped host memory alive as long as the buffer

    def view(self, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return a NumPy view of the buffer's memory (unified memory: no copy), valid until the context closes."""
        count = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if count > self.nbytes:
            raise ValueError(
                f"a {tuple(shape)} {np.dtype(dtype)} view needs {count} bytes; the buffer has {self.nbytes}"
            )
        raw = (ctypes.c_char * count).from_address(self.address)
        return np.frombuffer(raw, dtype=dtype).reshape(shape)


class Context:
    """A Metal device context: one library, its kernels, buffers, and at most one open command batch.

    Buffers live until `close`, so NumPy views of them stay valid for the context's lifetime.
    """

    def __init__(self) -> None:
        self._lib = load_library()
        err = ctypes.create_string_buffer(_ERR_LEN)
        handle = self._lib.idun_context_create(err, _ERR_LEN)
        if not handle:
            raise BackendUnavailableError(err.value.decode())
        self._handle: int | None = handle
        self._buffers: list[Buffer] = []
        self._open = False
        self.gpu_seconds = 0.0

    @property
    def handle(self) -> int:
        """The native context handle; raises once the context is closed."""
        if self._handle is None:
            raise IdunError("the Metal context is closed")
        return self._handle

    def load(self, source: str) -> None:
        """Compile ``source`` as the context's library (prelude prepended; see `source_prelude`)."""
        err = ctypes.create_string_buffer(_ERR_LEN)
        _check(self._lib.idun_library_load(self.handle, source.encode(), err, _ERR_LEN), err)

    def kernel(self, name: str) -> Kernel:
        """Return the compute pipeline of kernel function ``name``."""
        err = ctypes.create_string_buffer(_ERR_LEN)
        handle = _handle(self._lib.idun_kernel_get(self.handle, name.encode(), err, _ERR_LEN), err)
        return Kernel(name, handle, self._lib.idun_kernel_max_threads(handle), self._lib.idun_kernel_simd_width(handle))

    def alloc(self, nbytes: int) -> Buffer:
        """Allocate a zero-filled shared buffer of ``nbytes`` (at least 1)."""
        err = ctypes.create_string_buffer(_ERR_LEN)
        contents = ctypes.c_void_p()
        handle = _handle(
            self._lib.idun_buffer_alloc(self.handle, max(nbytes, 1), ctypes.byref(contents), err, _ERR_LEN), err
        )
        buffer = Buffer(handle, max(nbytes, 1), int(contents.value or 0))
        self._buffers.append(buffer)
        return buffer

    def wrap(self, array: NDArray[Any], nbytes: int) -> Buffer:
        """Wrap ``nbytes`` of ``array``'s memory, from its first element, without a copy.

        The address and ``nbytes`` must be multiples of the page size (16 KiB), and the memory must be mapped up to
        ``nbytes`` (a memory-mapped file's last page is). The buffer keeps ``array`` alive.
        """
        err = ctypes.create_string_buffer(_ERR_LEN)
        address = int(array.__array_interface__["data"][0])
        handle = _handle(self._lib.idun_buffer_wrap(self.handle, address, nbytes, err, _ERR_LEN), err)
        buffer = Buffer(handle, nbytes, address, owner=array)
        self._buffers.append(buffer)
        return buffer

    def wrap_mapped(self, array: NDArray[Any]) -> tuple[Buffer, int]:
        """Wrap a page-aligned memory-mapped array (1-D, or 2-D with contiguous rows) without a copy.

        Returns the buffer and the row stride in elements (0 for 1-D). The packed cache's arrays qualify: each file
        starts on a page and its last page is mapped. Raises ``ValueError`` for anything else (fail loudly; a copy is
        the caller's decision).
        """
        address = int(array.__array_interface__["data"][0])
        if address % PAGE_BYTES or array.strides[-1] != array.itemsize or array.ndim not in (1, 2):
            raise ValueError("wrap_mapped needs a page-aligned array with contiguous rows (1-D or 2-D)")
        stride = int(array.strides[0]) if array.ndim == 2 else 0
        extent = (array.shape[0] - 1) * stride + array.shape[1] * array.itemsize if array.ndim == 2 else array.nbytes
        return self.wrap(array, -(-max(extent, 1) // PAGE_BYTES) * PAGE_BYTES), stride // array.itemsize

    @contextmanager
    def batch(self, *, wait: bool = True) -> Iterator[None]:
        """Encode dispatches into one command buffer; committed and waited for on exit (added to `gpu_seconds`).

        ``wait=False`` commits it without waiting (`commit_async`). An exception inside closes the context.
        """
        self.begin()
        try:
            yield
        except BaseException:
            self._open = False
            self.close()
            raise
        if wait:
            self.commit()
        else:
            self.commit_async()

    def begin(self) -> None:
        """Open a command batch."""
        err = ctypes.create_string_buffer(_ERR_LEN)
        _check(self._lib.idun_batch_begin(self.handle, err, _ERR_LEN), err)
        self._open = True

    @property
    def is_open(self) -> bool:
        """Whether a batch is open."""
        return self._open

    def dispatch(
        self,
        kernel: Kernel,
        buffers: Sequence[Buffer],
        params: ctypes.Structure | None,
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        threadgroup_bytes: int = 0,
    ) -> None:
        """Encode ``kernel`` over ``groups`` threadgroups of ``threads`` threads into the open batch.

        ``threadgroup_bytes`` (a multiple of 16) sizes the kernel's dynamic ``threadgroup`` buffer at index 0.
        """
        if len(buffers) > MAX_BUFFERS:
            raise ValueError(f"at most {MAX_BUFFERS} buffers per dispatch")
        err = ctypes.create_string_buffer(_ERR_LEN)
        handles = (ctypes.c_void_p * max(len(buffers), 1))(*(b.handle for b in buffers))
        size = ctypes.sizeof(params) if params is not None else 0
        pointer = ctypes.addressof(params) if params is not None else None
        uint3 = ctypes.c_uint32 * 3
        code = self._lib.idun_dispatch(
            self.handle,
            kernel.handle,
            handles,
            len(buffers),
            pointer,
            size,
            threadgroup_bytes,
            uint3(*groups),
            uint3(*threads),
            err,
            _ERR_LEN,
        )
        _check(code, err)

    def commit(self) -> float:
        """Commit the open batch, wait for it (and any committed before it), and return their GPU seconds."""
        err = ctypes.create_string_buffer(_ERR_LEN)
        seconds = ctypes.c_double()
        self._open = False
        _check(self._lib.idun_batch_commit_wait(self.handle, ctypes.byref(seconds), err, _ERR_LEN), err)
        self.gpu_seconds += seconds.value
        return seconds.value

    def commit_async(self) -> None:
        """Commit the open batch without waiting: the GPU runs it while the host encodes the next one.

        Batches run in commit order, each after the previous one finished. Collect them with `wait`; the host must not
        touch a buffer a committed batch uses until then.
        """
        err = ctypes.create_string_buffer(_ERR_LEN)
        self._open = False
        _check(self._lib.idun_batch_commit(self.handle, err, _ERR_LEN), err)

    def wait(self, keep: int = 0) -> float:
        """Wait, oldest first, until at most ``keep`` committed batches are unfinished; return their GPU seconds."""
        if keep < 0:
            raise ValueError(f"keep must be at least 0, got {keep}")
        err = ctypes.create_string_buffer(_ERR_LEN)
        seconds = ctypes.c_double()
        _check(self._lib.idun_batch_wait(self.handle, keep, ctypes.byref(seconds), err, _ERR_LEN), err)
        self.gpu_seconds += seconds.value
        return seconds.value

    def close(self) -> None:
        """Free every buffer and the context. Views of the buffers must not be used afterwards."""
        if self._handle is None:
            return
        # committed batches may still read wrapped host memory: let them finish first (nothing reads their results)
        self._lib.idun_batch_wait(self._handle, 0, None, None, 0)
        for buffer in self._buffers:
            self._lib.idun_buffer_free(buffer.handle)
        self._buffers.clear()
        self._lib.idun_context_destroy(self._handle)
        self._handle = None

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
