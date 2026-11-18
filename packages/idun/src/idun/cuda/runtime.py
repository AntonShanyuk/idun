"""ctypes binding to the CUDA driver API (libcuda) and to NVRTC, NVIDIA's runtime compiler: the CUDA host runtime.

idun ships no native library for CUDA. The driver's library comes with NVIDIA's driver; NVRTC comes with the CUDA
toolkit or with the ``nvidia-cuda-nvrtc`` pip packages. The module exposes the device probe, compilation of CUDA C++
source to a binary for the device (or PTX, for the compile checks) with idun's fixed options, and a `Context`: one
kernel module, device memory, page-locked host memory, and one stream on which transfers, fills and launches run in the
order they were queued.

Nothing here falls back: a missing library, a missing symbol or a missing GPU raises
:class:`~idun.errors.BackendUnavailableError` naming it, and every failed driver call raises naming the call.
Linux only.
"""

import ctypes
import functools
import os
import platform
import re
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun.device import extent
from idun.errors import BackendUnavailableError, IdunError, KernelCompileError

DRIVER_ENV = "IDUN_CUDA_LIB"
NVRTC_ENV = "IDUN_NVRTC_LIB"
DEVICE_ENV = "IDUN_CUDA_DEVICE"
GUARD_ENV = "IDUN_CUDA_GUARD"
DRIVER_NAMES = ("libcuda.so.1", "libcuda.so")
NVRTC_NAMES = ("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so")
OPTIONS = ("--fmad=false", "--std=c++17")
"""The compile options every kernel source gets: no fused multiply-add (a product and a sum round separately, as
on the CPU), C++17. NVRTC's defaults keep the rest of IEEE arithmetic (no fast math, exact division and roots)."""
GUARD_BYTES = 1024  # of pattern before and after every allocation in guard mode
GUARD_VALUE = 0xA5
SHARED_DEFAULT_BYTES = 48 * 1024  # a kernel's dynamic shared memory without opting in to more
MAX_ARGUMENTS = 31

_SUCCESS = 0
_NVRTC_COMPILATION = 6
_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
_ATTRIBUTE_MAX_GRID = (5, 6, 7)  # blocks along x, y and z
_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
_ATTRIBUTE_CAPABILITY_MAJOR = 75
_ATTRIBUTE_CAPABILITY_MINOR = 76
_ATTRIBUTE_MAX_SHARED_OPTIN = 97
_FUNCTION_MAX_DYNAMIC_SHARED = 8

_INT = ctypes.c_int
_UINT = ctypes.c_uint
_SIZE = ctypes.c_size_t
_HANDLE = ctypes.c_void_p
_DEVICE_POINTER = ctypes.c_uint64
_STRING = ctypes.c_char_p
_OUT_INT = ctypes.POINTER(_INT)
_OUT_SIZE = ctypes.POINTER(_SIZE)
_OUT_HANDLE = ctypes.POINTER(_HANDLE)
_OUT_STRING = ctypes.POINTER(_STRING)

DRIVER_API: Mapping[str, tuple[Any, ...]] = {
    "cuInit": (_UINT,),
    "cuDriverGetVersion": (_OUT_INT,),
    "cuDeviceGetCount": (_OUT_INT,),
    "cuDeviceGet": (_OUT_INT, _INT),
    "cuDeviceGetName": (_STRING, _INT, _INT),
    "cuDeviceTotalMem_v2": (_OUT_SIZE, _INT),
    "cuDeviceGetAttribute": (_OUT_INT, _INT, _INT),
    "cuCtxCreate_v2": (_OUT_HANDLE, _UINT, _INT),
    "cuCtxDestroy_v2": (_HANDLE,),
    "cuCtxSetCurrent": (_HANDLE,),
    "cuModuleLoadDataEx": (_OUT_HANDLE, _STRING, _UINT, _HANDLE, _HANDLE),
    "cuModuleUnload": (_HANDLE,),
    "cuModuleGetFunction": (_OUT_HANDLE, _HANDLE, _STRING),
    "cuFuncSetAttribute": (_HANDLE, _INT, _INT),
    "cuMemAlloc_v2": (ctypes.POINTER(_DEVICE_POINTER), _SIZE),
    "cuMemFree_v2": (_DEVICE_POINTER,),
    "cuMemAllocHost_v2": (_OUT_HANDLE, _SIZE),
    "cuMemFreeHost": (_HANDLE,),
    "cuMemcpyHtoDAsync_v2": (_DEVICE_POINTER, _HANDLE, _SIZE, _HANDLE),
    "cuMemcpyDtoHAsync_v2": (_HANDLE, _DEVICE_POINTER, _SIZE, _HANDLE),
    "cuMemsetD8Async": (_DEVICE_POINTER, ctypes.c_ubyte, _SIZE, _HANDLE),
    "cuStreamCreate": (_OUT_HANDLE, _UINT),
    "cuStreamDestroy_v2": (_HANDLE,),
    "cuStreamSynchronize": (_HANDLE,),
    "cuLaunchKernel": (_HANDLE, _UINT, _UINT, _UINT, _UINT, _UINT, _UINT, _UINT, _HANDLE, _OUT_HANDLE, _HANDLE),
    "cuEventCreate": (_OUT_HANDLE, _UINT),
    "cuEventRecord": (_HANDLE, _HANDLE),
    "cuEventElapsedTime": (ctypes.POINTER(ctypes.c_float), _HANDLE, _HANDLE),
    "cuEventDestroy_v2": (_HANDLE,),
    "cuGetErrorName": (_INT, _OUT_STRING),
    "cuGetErrorString": (_INT, _OUT_STRING),
}
"""The driver functions this binding calls, with their argument types; each returns a ``CUresult``."""

NVRTC_API: Mapping[str, tuple[Any, ...]] = {
    "nvrtcVersion": (_OUT_INT, _OUT_INT),
    "nvrtcGetNumSupportedArchs": (_OUT_INT,),
    "nvrtcGetSupportedArchs": (_OUT_INT,),
    "nvrtcCreateProgram": (_OUT_HANDLE, _STRING, _STRING, _INT, _HANDLE, _HANDLE),
    "nvrtcCompileProgram": (_HANDLE, _INT, _OUT_STRING),
    "nvrtcGetProgramLogSize": (_HANDLE, _OUT_SIZE),
    "nvrtcGetProgramLog": (_HANDLE, _STRING),
    "nvrtcGetPTXSize": (_HANDLE, _OUT_SIZE),
    "nvrtcGetPTX": (_HANDLE, _STRING),
    "nvrtcGetCUBINSize": (_HANDLE, _OUT_SIZE),
    "nvrtcGetCUBIN": (_HANDLE, _STRING),
    "nvrtcDestroyProgram": (_OUT_HANDLE,),
    "nvrtcGetErrorString": (_INT,),
}
"""NVRTC's functions this binding calls; each returns an ``nvrtcResult``, `TEXT_RESULTS` a string."""

TEXT_RESULTS = frozenset({"nvrtcGetErrorString"})

_ENTRY = re.compile(rb"^\.visible \.entry (\w+)\(", re.MULTILINE)
_current = threading.local()  # .context: the `Context` current on this thread, as this module last made it


def bind(library: Any, api: Mapping[str, tuple[Any, ...]], path: str) -> Any:
    """Declare the C signatures of ``api`` on ``library``; a missing function refuses the library by name."""
    for name, arguments in api.items():
        try:
            function = getattr(library, name)
        except AttributeError:
            raise BackendUnavailableError(f"{path} does not export {name}: this binding cannot use it") from None
        function.argtypes = arguments
        function.restype = _STRING if name in TEXT_RESULTS else _INT
    return library


def _open(
    what: str,
    candidates: Sequence[str],
    env: str,
    api: Mapping[str, tuple[Any, ...]],
    hint: str,
    first: Callable[[str], Sequence[str]] | None = None,
) -> Any:
    """Bind the first of ``candidates`` that loads, after the libraries ``first`` names for it (loaded globally)."""
    if platform.system() != "Linux":
        raise BackendUnavailableError(f"the CUDA backend needs Linux, this is {platform.system()}")
    failures = []
    for candidate in candidates:
        try:
            for needed in first(candidate) if first else ():
                ctypes.CDLL(needed, mode=ctypes.RTLD_GLOBAL)
            library = ctypes.CDLL(candidate)
        except OSError as exc:
            failures.append(f"{candidate}: {exc}")
            continue
        return bind(library, api, candidate)
    raise BackendUnavailableError(
        f"{what} cannot be loaded ({'; '.join(failures)}): {hint}, or set {env} to the library's path"
    )


def driver_candidates() -> list[str]:
    """Return the names the driver library is looked for under: ``$IDUN_CUDA_LIB`` alone if set, else the loader's."""
    override = os.environ.get(DRIVER_ENV)
    return [override] if override else list(DRIVER_NAMES)


def nvrtc_candidates() -> list[str]:
    """Return where NVRTC is looked for: ``$IDUN_NVRTC_LIB`` alone if set; else pip's packages, then the loader.

    The ``nvidia-cuda-nvrtc`` packages install ``nvidia/*/lib/libnvrtc.so.N`` on the import path; the newest
    version found goes first.
    """
    override = os.environ.get(NVRTC_ENV)
    if override:
        return [override]
    found = {str(path) for entry in sys.path if entry for path in Path(entry).glob("nvidia/*/lib/libnvrtc.so.*")}
    return [*sorted(found, key=lambda path: (_soname_version(path), path), reverse=True), *NVRTC_NAMES]


def _soname_version(path: str) -> tuple[int, ...]:
    return tuple(int(part) for part in path.rpartition(".so.")[2].split(".") if part.isdigit())


def nvrtc_builtins(path: str) -> list[str]:
    """Return the builtins libraries of its own version beside the NVRTC library at ``path``.

    NVRTC opens ``libnvrtc-builtins.so.N.M`` by name at its first compilation, and the loader looks for that name
    on its search path only, where a pip package's folder is not. Loaded beforehand, the loader has it already.
    A bare library name has none: the loader's path serves both.
    """
    if Path(path).parent == Path():
        return []
    library = Path(path).resolve()  # a link without a version names the file that has one
    version = _soname_version(library.name)[:1]
    found = (str(beside) for beside in library.parent.glob("libnvrtc-builtins.so.*"))
    return sorted(beside for beside in found if _soname_version(beside)[:1] == version)


@functools.cache
def load_driver() -> Any:
    """Load the CUDA driver library once per process and declare its C signatures.

    Raises
    ------
    BackendUnavailableError
        Not on Linux, the library cannot be loaded, or it lacks a function this binding calls.
    """
    return _open("the CUDA driver library", driver_candidates(), DRIVER_ENV, DRIVER_API, "install NVIDIA's driver")


@functools.cache
def load_nvrtc() -> Any:
    """Load NVRTC once per process and declare its C signatures.

    Raises
    ------
    BackendUnavailableError
        Not on Linux, the library cannot be loaded, or it lacks a function this binding calls.
    """
    hint = "install the CUDA toolkit or the nvidia-cuda-nvrtc pip package of the driver's CUDA version"
    return _open("NVRTC, the CUDA runtime compiler", nvrtc_candidates(), NVRTC_ENV, NVRTC_API, hint, nvrtc_builtins)


def driver_error(name: str, code: int) -> IdunError:
    """Return the exception for driver call ``name`` that returned ``code``, with the driver's name for the code."""
    driver = load_driver()
    label, text = _STRING(), _STRING()
    driver.cuGetErrorName(code, ctypes.byref(label))
    driver.cuGetErrorString(code, ctypes.byref(text))
    known = f"{(label.value or b'an unknown error').decode()} ({code}): {(text.value or b'no description').decode()}"
    return IdunError(f"CUDA driver: {name} failed with {known}")


def _cu(name: str, *arguments: Any) -> None:
    code = int(getattr(load_driver(), name)(*arguments))
    if code != _SUCCESS:
        raise driver_error(name, code)


def nvrtc_error(name: str, code: int) -> IdunError:
    """Return the exception for NVRTC call ``name`` that returned ``code``, with NVRTC's text for the code."""
    text: bytes = load_nvrtc().nvrtcGetErrorString(code) or b"an unknown error"
    return IdunError(f"NVRTC: {name} failed with {text.decode()} ({code})")


def _nvrtc(name: str, *arguments: Any) -> None:
    code = int(getattr(load_nvrtc(), name)(*arguments))
    if code != _SUCCESS:
        raise nvrtc_error(name, code)


def initialise() -> None:
    """Initialise the driver (idempotent); `BackendUnavailableError` when it has no usable GPU."""
    code = int(load_driver().cuInit(0))
    if code != _SUCCESS:
        raise BackendUnavailableError(f"the CUDA backend has no usable GPU: {driver_error('cuInit', code)}")


def device_ordinal(device: int | None = None) -> int:
    """Return the device to use: ``device``, else ``$IDUN_CUDA_DEVICE``, else 0; ``ValueError`` unless it exists."""
    initialise()
    count = _INT()
    _cu("cuDeviceGetCount", ctypes.byref(count))
    if device is None:
        named = os.environ.get(DEVICE_ENV, "0")
        if not named.isdigit():
            raise ValueError(f"{DEVICE_ENV} must be a device number, got {named!r}")
        device = int(named)
    if not 0 <= device < count.value:
        raise ValueError(f"CUDA device {device} does not exist: the driver lists {count.value}")
    return device


def device_ordinals() -> list[int]:
    """Return the devices that work side by side is spread over: ``$IDUN_CUDA_DEVICE`` alone if set, else all."""
    if os.environ.get(DEVICE_ENV):
        return [device_ordinal()]
    return list(range(probe().device_count))


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """What one CUDA device offers, and the driver that serves it."""

    ordinal: int
    name: str
    total_memory_bytes: int
    compute_capability: tuple[int, int]
    multiprocessors: int
    max_threads_per_block: int
    max_shared_memory_bytes: int  # per block, for a kernel that opts in
    max_grid: tuple[int, int, int]  # blocks a launch may have along x, y and z
    driver_version: tuple[int, int]  # the CUDA version the driver implements
    device_count: int


def probe(device: int | None = None) -> DeviceInfo:
    """Describe a CUDA device (`device_ordinal` picks it) without creating a context on it.

    Raises
    ------
    BackendUnavailableError
        No usable GPU, or the driver library cannot be loaded.
    """
    ordinal = device_ordinal(device)
    handle, count, version, memory = _INT(), _INT(), _INT(), _SIZE()
    _cu("cuDeviceGet", ctypes.byref(handle), ordinal)
    _cu("cuDeviceGetCount", ctypes.byref(count))
    _cu("cuDriverGetVersion", ctypes.byref(version))
    _cu("cuDeviceTotalMem_v2", ctypes.byref(memory), handle)
    name = ctypes.create_string_buffer(256)
    _cu("cuDeviceGetName", name, len(name), handle)

    def attribute(which: int) -> int:
        value = _INT()
        _cu("cuDeviceGetAttribute", ctypes.byref(value), which, handle)
        return value.value

    grid = _ATTRIBUTE_MAX_GRID
    return DeviceInfo(
        ordinal=ordinal,
        name=name.value.decode(),
        total_memory_bytes=memory.value,
        compute_capability=(attribute(_ATTRIBUTE_CAPABILITY_MAJOR), attribute(_ATTRIBUTE_CAPABILITY_MINOR)),
        multiprocessors=attribute(_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        max_threads_per_block=attribute(_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
        max_shared_memory_bytes=attribute(_ATTRIBUTE_MAX_SHARED_OPTIN),
        max_grid=(attribute(grid[0]), attribute(grid[1]), attribute(grid[2])),
        driver_version=(version.value // 1000, version.value % 1000 // 10),
        device_count=count.value,
    )


def nvrtc_version() -> tuple[int, int]:
    """Return NVRTC's version (major, minor)."""
    major, minor = _INT(), _INT()
    _nvrtc("nvrtcVersion", ctypes.byref(major), ctypes.byref(minor))
    return major.value, minor.value


def supported_architectures() -> list[int]:
    """Return the compute capabilities NVRTC compiles for, as major * 10 + minor, ascending."""
    count = _INT()
    _nvrtc("nvrtcGetNumSupportedArchs", ctypes.byref(count))
    found = (_INT * max(count.value, 1))()
    _nvrtc("nvrtcGetSupportedArchs", found)
    return sorted(found[: count.value])


def architecture(capability: tuple[int, int]) -> str:
    """Return NVRTC's architecture option value for a device of ``capability``: the newest it knows, not newer.

    `BackendUnavailableError` when NVRTC no longer compiles for a device that old.
    """
    wanted = capability[0] * 10 + capability[1]
    usable = [arch for arch in supported_architectures() if arch <= wanted]
    if not usable:
        version = ".".join(map(str, nvrtc_version()))
        raise BackendUnavailableError(
            f"NVRTC {version} does not compile for compute capability {capability[0]}.{capability[1]}: install the "
            "nvidia-cuda-nvrtc package of an older CUDA version"
        )
    return f"compute_{usable[-1]}"


def binary_architecture(capability: tuple[int, int]) -> str:
    """Return the real architecture NVRTC builds a device of ``capability``'s binary for: ``sm_XY``.

    The newest NVRTC knows of the device's major version and not above its minor: a binary for sm_XY runs on every
    device X.Z with Z >= Y. `BackendUnavailableError` when NVRTC knows none.
    """
    wanted = capability[0] * 10 + capability[1]
    usable = [arch for arch in supported_architectures() if arch // 10 == capability[0] and arch <= wanted]
    if not usable:
        version = ".".join(map(str, nvrtc_version()))
        raise BackendUnavailableError(
            f"NVRTC {version} builds no binary for compute capability {capability[0]}.{capability[1]}: install the "
            "nvidia-cuda-nvrtc package of a CUDA version that supports it"
        )
    return f"sm_{usable[-1]}"


def compile_ptx(source: str, arch: str, *, name: str = "idun.cu") -> tuple[bytes, str]:
    """Compile CUDA C++ ``source`` for ``arch`` (``compute_75``) with `OPTIONS`; return the PTX and the warnings.

    Raises
    ------
    KernelCompileError
        The source does not compile; the message is the compiler log.
    BackendUnavailableError
        NVRTC cannot be loaded.
    """
    return _compile(source, arch, name, "PTX")


def compile_cubin(source: str, arch: str, *, name: str = "idun.cu") -> tuple[bytes, str]:
    """Compile CUDA C++ ``source`` to a binary for the real architecture ``arch`` (``sm_75``); return it, the warnings.

    A binary, unlike PTX, loads on a driver older than NVRTC (CUDA's minor version compatibility: a driver of the same
    major version runs it); the PTX of an NVRTC newer than the driver is refused
    (``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``). Errors as `compile_ptx`'s.
    """
    return _compile(source, arch, name, "CUBIN")


def _compile(source: str, arch: str, name: str, output: str) -> tuple[bytes, str]:
    """Compile ``source`` for ``arch`` with `OPTIONS`; return NVRTC's ``output`` (``"PTX"`` or ``"CUBIN"``), the log."""
    program = _HANDLE()
    _nvrtc("nvrtcCreateProgram", ctypes.byref(program), source.encode(), name.encode(), 0, None, None)
    try:
        options = [option.encode() for option in (*OPTIONS, f"--gpu-architecture={arch}")]
        code = int(load_nvrtc().nvrtcCompileProgram(program, len(options), (_STRING * len(options))(*options)))
        size = _SIZE()
        _nvrtc("nvrtcGetProgramLogSize", program, ctypes.byref(size))
        log = ctypes.create_string_buffer(max(size.value, 1))
        _nvrtc("nvrtcGetProgramLog", program, log)
        text = log.value.decode(errors="replace").strip()
        if code == _NVRTC_COMPILATION:
            raise KernelCompileError(text or f"NVRTC could not compile {name} and gave no log")
        if code != _SUCCESS:
            failed = nvrtc_error("nvrtcCompileProgram", code)
            raise IdunError(f"{failed}: {text}") if text else failed
        _nvrtc(f"nvrtcGet{output}Size", program, ctypes.byref(size))
        image = ctypes.create_string_buffer(max(size.value, 1))
        _nvrtc(f"nvrtcGet{output}", program, image)
        return (image.value if output == "PTX" else image.raw[: size.value]), text  # a binary holds zero bytes
    finally:
        _nvrtc("nvrtcDestroyProgram", ctypes.byref(program))


def entry_points(ptx: bytes) -> list[str]:
    """Return the kernels a PTX module exports, in source order."""
    return [name.decode() for name in _ENTRY.findall(ptx)]


def guard_mode() -> bool:
    """Whether ``$IDUN_CUDA_GUARD`` asks for guard mode (``1``); ``ValueError`` for any other value."""
    mode = os.environ.get(GUARD_ENV)
    if mode not in (None, "1"):
        raise ValueError(f"{GUARD_ENV} must be 1 or unset, got {mode!r}")
    return mode == "1"


@dataclass(eq=False)
class Kernel:
    """A kernel of the context's module."""

    name: str
    handle: int
    shared_bytes: int = SHARED_DEFAULT_BYTES  # the dynamic shared memory it may ask for so far


@dataclass(eq=False)
class Buffer:
    """Device memory of a context. ``pointer`` is where its ``nbytes`` start; ``base`` is what the driver allocated."""

    pointer: int
    nbytes: int
    base: int


class HostMemory:
    """Page-locked host memory of a context: the driver copies from and to it without waiting."""

    def __init__(self, address: int, nbytes: int) -> None:
        self.address, self.nbytes = address, nbytes

    def view(self, dtype: type[np.generic], shape: Sequence[int]) -> NDArray[Any]:
        """Return a NumPy view of the memory, valid until the context closes."""
        raw = (ctypes.c_char * extent(dtype, shape, self.nbytes)).from_address(self.address)
        return np.frombuffer(raw, dtype=dtype).reshape(shape)


class Context:
    """A CUDA context on one device: a kernel module, device and page-locked memory, and one stream.

    `upload`, `download`, `fill` and `launch` queue work on the stream and return; the work runs in that order.
    `sync` waits for all of it. A download is complete, and its host memory readable, only after `sync`. Page-locked
    memory given to `upload` must keep its values until `sync`; other memory is copied before `upload` returns.

    In guard mode (``guard=True``, or ``$IDUN_CUDA_GUARD=1`` when ``guard`` is None) every allocation lies between
    two bands of a known pattern, every launch is waited for, and a band that changed raises naming the kernel and
    the buffer: a kernel wrote outside its buffers.
    """

    def __init__(self, device: int | None = None, *, guard: bool | None = None) -> None:
        self.info = probe(device)
        self.guard = guard_mode() if guard is None else guard
        self.gpu_seconds = 0.0
        self._handle: int | None = None
        self._stream: int | None = None
        self._events: list[int] = []
        self._module: int | None = None
        self._buffers: dict[int, Buffer] = {}
        self._host: dict[int, HostMemory] = {}
        self._queued = False  # work is on the stream that no sync has waited for
        self._timing = False  # the start event of the launches since the last wait is recorded
        self._waited = 0.0  # GPU seconds of the launches guard mode has waited for since the last sync
        handle, device_handle = _HANDLE(), _INT()
        _cu("cuDeviceGet", ctypes.byref(device_handle), self.info.ordinal)
        _cu("cuCtxCreate_v2", ctypes.byref(handle), 0, device_handle)
        self._handle = handle.value
        _current.context = self  # creating a context makes it current on this thread
        try:
            stream = _HANDLE()
            _cu("cuStreamCreate", ctypes.byref(stream), 0)
            self._stream = stream.value
            for _ in range(2):
                event = _HANDLE()
                _cu("cuEventCreate", ctypes.byref(event), 0)
                self._events.append(int(event.value or 0))
        except IdunError:
            self._release()
            raise

    def _use(self) -> None:
        """Make the context current on this thread; a driver call only when another one was."""
        if self._handle is None:
            raise IdunError("the CUDA context is closed")
        if getattr(_current, "context", None) is not self:
            _cu("cuCtxSetCurrent", self._handle)
            _current.context = self

    def load(self, source: str) -> str:
        """Compile ``source`` for the device and load it as the context's module; return the compiler's warnings."""
        self._use()
        if self._module is not None:
            raise IdunError("the CUDA context already has its module")
        binary, warnings = compile_cubin(source, binary_architecture(self.info.compute_capability))
        module = _HANDLE()
        _cu("cuModuleLoadDataEx", ctypes.byref(module), binary, 0, None, None)
        self._module = module.value
        return warnings

    def kernel(self, name: str) -> Kernel:
        """Return kernel ``name`` of the context's module."""
        self._use()
        if self._module is None:
            raise IdunError("the CUDA context has no module: load a source first")
        function = _HANDLE()
        _cu("cuModuleGetFunction", ctypes.byref(function), self._module, name.encode())
        return Kernel(name, int(function.value or 0))

    def alloc(self, nbytes: int, *, zero: bool = True) -> Buffer:
        """Allocate device memory of ``nbytes`` (at least 1), zero-filled unless ``zero`` is False."""
        self._use()
        size = max(nbytes, 1)
        band = GUARD_BYTES if self.guard else 0
        base = _DEVICE_POINTER()
        _cu("cuMemAlloc_v2", ctypes.byref(base), size + 2 * band)
        buffer = Buffer(base.value + band, size, base.value)
        self._buffers[buffer.base] = buffer
        if band:
            _cu("cuMemsetD8Async", buffer.base, GUARD_VALUE, band, self._stream)
            _cu("cuMemsetD8Async", buffer.pointer + size, GUARD_VALUE, band, self._stream)
            self._queued = True
        if zero:
            self.fill(buffer, 0, size)
        return buffer

    def free(self, buffer: Buffer) -> None:
        """Free a buffer (queued work finishes first)."""
        self._span(buffer, 0, 0)
        self.sync()
        _cu("cuMemFree_v2", buffer.base)
        del self._buffers[buffer.base]

    def alloc_host(self, nbytes: int) -> HostMemory:
        """Allocate page-locked host memory of ``nbytes`` (at least 1); its contents are undefined."""
        self._use()
        address = _HANDLE()
        _cu("cuMemAllocHost_v2", ctypes.byref(address), max(nbytes, 1))
        memory = HostMemory(int(address.value or 0), max(nbytes, 1))
        self._host[memory.address] = memory
        return memory

    def _span(self, buffer: Buffer, offset: int, nbytes: int) -> int:
        """Return the device address of ``nbytes`` at ``offset`` of a live buffer of this context."""
        if self._buffers.get(buffer.base) is not buffer:
            raise ValueError("the buffer is not a live buffer of this context")
        if offset < 0 or nbytes < 0 or offset + nbytes > buffer.nbytes:
            raise ValueError(f"{nbytes} bytes at offset {offset} do not fit a buffer of {buffer.nbytes}")
        return buffer.pointer + offset

    def upload(self, buffer: Buffer, address: int, nbytes: int, offset: int = 0) -> None:
        """Queue a copy of ``nbytes`` at host ``address`` to ``offset`` of ``buffer``."""
        self._use()
        target = self._span(buffer, offset, nbytes)
        if nbytes:
            _cu("cuMemcpyHtoDAsync_v2", target, address, nbytes, self._stream)
            self._queued = True

    def download(self, buffer: Buffer, address: int, nbytes: int, offset: int = 0) -> None:
        """Queue a copy of ``nbytes`` at ``offset`` of ``buffer`` to host ``address`` (complete after `sync`)."""
        self._use()
        source = self._span(buffer, offset, nbytes)
        if nbytes:
            _cu("cuMemcpyDtoHAsync_v2", address, source, nbytes, self._stream)
            self._queued = True

    def fill(self, buffer: Buffer, value: int, nbytes: int, offset: int = 0) -> None:
        """Queue setting ``nbytes`` bytes at ``offset`` of ``buffer`` to ``value`` (0..255)."""
        self._use()
        if not 0 <= value <= 255:
            raise ValueError(f"a fill value is a byte, got {value}")
        target = self._span(buffer, offset, nbytes)
        if nbytes:
            _cu("cuMemsetD8Async", target, value, nbytes, self._stream)
            self._queued = True

    def launch(
        self,
        kernel: Kernel,
        buffers: Sequence[Buffer],
        params: ctypes.Structure | None,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_bytes: int = 0,
    ) -> None:
        """Queue ``kernel`` over ``grid`` blocks of ``block`` threads.

        The kernel's arguments are the buffers' pointers in order, then ``params`` by value (read before the call
        returns). ``shared_bytes`` sizes its dynamic shared memory (``extern __shared__``); above 48 KB the kernel
        is opted in to it first.
        """
        self._use()
        if len(buffers) + (params is not None) > MAX_ARGUMENTS:
            raise ValueError(f"a launch takes at most {MAX_ARGUMENTS} arguments")
        if min(*grid, *block) < 1:
            raise ValueError(f"grid {grid} and block {block} need at least one block and one thread each way")
        if any(blocks > most for blocks, most in zip(grid, self.info.max_grid, strict=True)):
            raise ValueError(
                f"kernel {kernel.name} asks for a grid of {grid} blocks; the device runs at most {self.info.max_grid}"
            )
        pointers = (_DEVICE_POINTER * max(len(buffers), 1))(*(self._span(b, 0, 0) for b in buffers))
        if shared_bytes > kernel.shared_bytes:
            if shared_bytes > self.info.max_shared_memory_bytes:
                raise ValueError(
                    f"kernel {kernel.name} asks for {shared_bytes} bytes of shared memory; the device gives a block "
                    f"at most {self.info.max_shared_memory_bytes}"
                )
            _cu("cuFuncSetAttribute", kernel.handle, _FUNCTION_MAX_DYNAMIC_SHARED, shared_bytes)
            kernel.shared_bytes = shared_bytes
        slots = [ctypes.addressof(pointers) + 8 * index for index in range(len(buffers))]
        if params is not None:
            slots.append(ctypes.addressof(params))
        arguments = (_HANDLE * max(len(slots), 1))(*slots)
        if not self._timing:
            _cu("cuEventRecord", self._events[0], self._stream)
            self._timing = True
            self._queued = True
        _cu("cuLaunchKernel", kernel.handle, *grid, *block, shared_bytes, self._stream, arguments, None)
        if self.guard:
            self._waited += self._wait(f"kernel {kernel.name}")

    def sync(self, *, after: str = "the queued work") -> float:
        """Wait for everything queued; return the GPU seconds of the launches since the last `sync`.

        The seconds run from the start of the first of those launches to the end of the last, as the stream saw
        them: transfers and fills queued between two launches count. In guard mode every launch is waited for and
        the bands are checked after it, so the seconds are the sum of the launches'.
        """
        self._use()
        seconds = self._waited + self._wait(after)
        self._waited = 0.0
        return seconds

    def _wait(self, after: str) -> float:
        if not self._queued:
            return 0.0
        if self._timing:
            _cu("cuEventRecord", self._events[1], self._stream)
        _cu("cuStreamSynchronize", self._stream)
        self._queued = False
        seconds = 0.0
        if self._timing:
            self._timing = False
            milliseconds = ctypes.c_float()
            _cu("cuEventElapsedTime", ctypes.byref(milliseconds), self._events[0], self._events[1])
            seconds = milliseconds.value / 1e3
            self.gpu_seconds += seconds
        if self.guard and self._buffers:
            self._check_bands(after)
        return seconds

    def _check_bands(self, after: str) -> None:
        bands = np.empty((len(self._buffers), 2, GUARD_BYTES), np.uint8)
        address = int(bands.__array_interface__["data"][0])
        for index, buffer in enumerate(self._buffers.values()):
            for side, start in enumerate((buffer.base, buffer.pointer + buffer.nbytes)):
                target = address + (2 * index + side) * GUARD_BYTES
                _cu("cuMemcpyDtoHAsync_v2", target, start, GUARD_BYTES, self._stream)
        _cu("cuStreamSynchronize", self._stream)
        for index, buffer in enumerate(self._buffers.values()):
            for side, where in enumerate(("before its start", "past its end")):
                changed = np.flatnonzero(bands[index, side] != GUARD_VALUE)
                if changed.size:
                    offset = int(changed[0]) - GUARD_BYTES if side == 0 else buffer.nbytes + int(changed[0])
                    raise IdunError(
                        f"guard mode: after {after}, buffer {index} (in allocation order, {buffer.nbytes} bytes) "
                        f"was written {where}, first at byte offset {offset}"
                    )

    def _release(self) -> IdunError | None:
        """Free the memory, the events, the module, the stream and the context; return the first failure."""
        driver = load_driver()
        calls: list[tuple[str, int]] = [("cuMemFree_v2", buffer.base) for buffer in self._buffers.values()]
        calls += [("cuMemFreeHost", memory.address) for memory in self._host.values()]
        calls += [("cuEventDestroy_v2", event) for event in self._events]
        last = (
            ("cuModuleUnload", self._module),
            ("cuStreamDestroy_v2", self._stream),
            ("cuCtxDestroy_v2", self._handle),
        )
        calls += [(name, handle) for name, handle in last if handle is not None]
        failure = None
        for name, handle in calls:  # every call is made, whatever the ones before returned
            code = int(getattr(driver, name)(handle))
            if code != _SUCCESS and failure is None:
                failure = driver_error(name, code)
        self._buffers.clear()
        self._host.clear()
        self._events.clear()
        self._module = self._stream = self._handle = None
        if getattr(_current, "context", None) is self:
            _current.context = None
        return failure

    def close(self) -> None:
        """Wait for the queued work, then free the memory, the module, the stream and the context."""
        if self._handle is None:
            return
        self._use()
        try:
            self.sync(after="the last work before close")
        finally:
            failure = self._release()
        if failure is not None:
            raise failure

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
