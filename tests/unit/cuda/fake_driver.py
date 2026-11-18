"""Stand-ins for the CUDA driver library and NVRTC, so that the CUDA runtime's own lines run without either.

`FakeDriver` keeps device memory as host memory, hands out its real addresses as device pointers and records every
call. Its stream is a queue: launches, fills, event records and copies from or to page-locked memory run only when
the stream is synchronised, in the order they were queued, so a missing synchronisation or a host write that comes
too late shows as wrong values. A copy from other host memory is staged at once, and one to other host memory waits
for the queue first, as the driver documents. Kernels are emulated by Python functions registered under their names.

`FakeNvrtc` "compiles" a source to a PTX-like text naming its target and its ``__global__`` functions, or for a real
architecture (``sm_XY``) to a binary-like one as well (with a zero byte, as an ELF image has); ``#error`` in the source
fails the compilation and ``#warning`` leaves a line in the log. The driver refuses the PTX of an NVRTC newer than
itself, as a real driver does, and loads a binary.

Both check each call against the argument types the binding declared, as ctypes would.
"""

import ctypes
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from idun.cuda import runtime

ERRORS = {
    1: ("CUDA_ERROR_INVALID_VALUE", "invalid argument"),
    2: ("CUDA_ERROR_OUT_OF_MEMORY", "out of memory"),
    3: ("CUDA_ERROR_NOT_INITIALIZED", "initialization error"),
    100: ("CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected"),
    101: ("CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal"),
    201: ("CUDA_ERROR_INVALID_CONTEXT", "invalid device context"),
    209: ("CUDA_ERROR_NO_BINARY_FOR_GPU", "no kernel image is available for execution on the device"),
    222: ("CUDA_ERROR_UNSUPPORTED_PTX_VERSION", "the provided PTX was compiled with an unsupported toolchain."),
    400: ("CUDA_ERROR_INVALID_HANDLE", "invalid resource handle"),
    500: ("CUDA_ERROR_NOT_FOUND", "named symbol not found"),
    600: ("CUDA_ERROR_NOT_READY", "device not ready"),
}
NVRTC_ERRORS = {
    4: b"NVRTC_ERROR_INVALID_PROGRAM",
    5: b"NVRTC_ERROR_INVALID_OPTION",
    6: b"NVRTC_ERROR_COMPILATION",
    7: b"NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
}
DEVICE_GARBAGE, HOST_GARBAGE = 0xCD, 0xEE  # what fresh memory holds: nothing is zero until the runtime zeroes it
SHARED_DEFAULT = 48 * 1024

_GLOBAL = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
_ENTRY = re.compile(rb"^\.visible \.entry (\w+)\(", re.MULTILINE)
_TARGET = re.compile(rb"^\.target sm_(\d+)", re.MULTILINE)
_MAKER = re.compile(rb"^// fake NVRTC (\d+)\.(\d+)", re.MULTILINE)


def _out(argument: Any) -> Any:
    """Return the ctypes object an out-argument points to (``byref(x)`` gives ``x``; an array is itself)."""
    return getattr(argument, "_obj", argument)


class Function:
    """One C function of a fake library: callable, and carrying the attributes a ctypes function has."""

    def __init__(self, library: "Library", name: str, implementation: Callable[..., Any]) -> None:
        self.library, self.name, self.implementation = library, name, implementation
        self.argtypes: tuple[Any, ...] | None = None
        self.restype: Any = None

    def __call__(self, *arguments: Any) -> Any:
        with self.library.lock:
            self.library.calls.append(self.name)
            assert self.argtypes is not None, f"{self.name} was called before its signature was declared"
            assert len(arguments) == len(self.argtypes), f"{self.name} takes {len(self.argtypes)} arguments"
            for argtype, argument in zip(self.argtypes, arguments, strict=True):
                argtype.from_param(argument)  # raises as a ctypes call would
            if self.library.fail.get(self.name):
                return self.library.fail.pop(self.name)
            return self.implementation(*arguments)


class Library:
    """A fake library: every method named like a function of the binding's tables becomes a `Function`."""

    def __init__(self) -> None:
        self.calls: list[str] = []  # every call's name, in order
        self.fail: dict[str, int] = {}  # the code a function's next call returns, doing nothing else
        self.lock = threading.RLock()  # one call at a time: threads may share the library, as they share the driver
        for name in dir(type(self)):
            if name in runtime.DRIVER_API or name in runtime.NVRTC_API:
                setattr(self, name, Function(self, name, getattr(self, name)))

    def count(self, name: str) -> int:
        """Return how often function ``name`` was called."""
        return self.calls.count(name)


@dataclass
class Launch:
    """One recorded kernel launch: what the driver read from the call's arguments when it was made."""

    name: str
    pointers: tuple[int, ...]
    params: Any  # a copy of the parameter structure, or None
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_bytes: int


@dataclass
class Kernel:
    """A kernel's emulation: how many buffers it takes, its parameter structure, and what it does."""

    buffers: int
    params: type[ctypes.Structure] | None
    run: Callable[["FakeDriver", Launch], None]


@dataclass
class _Function:
    name: str
    context: int
    shared_max: int = SHARED_DEFAULT


@dataclass
class _Module:
    entries: list[str]
    context: int


@dataclass
class FakeDevice:
    """What a fake device reports."""

    name: str = "Fake T4"
    total_memory: int = 15 << 30
    capability: tuple[int, int] = (7, 5)
    attributes: dict[int, int] = field(
        default_factory=lambda: {1: 1024, 5: 2**31 - 1, 6: 65535, 7: 65535, 8: 48 * 1024, 16: 40, 97: 64 * 1024}
    )


class FakeDriver(Library):
    """The driver library's stand-in. ``kernels`` holds the emulations; the rest is state the tests may read."""

    def __init__(self, devices: int = 1) -> None:
        super().__init__()
        self.devices = [FakeDevice() for _ in range(devices)]
        self.version = 12040
        self.kernels: dict[str, Kernel] = {}
        self.launches: list[Launch] = []
        self.transfers: list[tuple[str, int]] = []  # ("up" | "down", bytes), as queued
        self.launch_seconds = 0.25  # of GPU time per launch, on the fake clock
        self.clock = 0.0
        self.queues: dict[int, list[Callable[[], None]]] = {}  # context -> its stream's work, in order
        self.memory: dict[int, Any] = {}  # device allocations by address
        self.host: dict[int, Any] = {}  # page-locked allocations by address
        self.contexts: dict[int, int] = {}  # handle -> device
        self.created: list[int] = []  # the device of every context made, in order
        self.streams: dict[int, int] = {}  # handle -> context
        self.events: dict[int, float | None] = {}  # handle -> the clock when last recorded
        self.modules: dict[int, _Module] = {}
        self.functions: dict[int, _Function] = {}
        self.current: dict[int, int | None] = {}  # thread -> context
        self._initialised = False
        self._next = 0x1000

    # -- what tests use ---------------------------------------------------------------------------------------------

    @property
    def queue(self) -> list[Callable[[], None]]:
        """The work queued in the calling thread's current context and not yet run (a context's work is its own)."""
        context = self._context()
        return [] if context is None else self.queues.setdefault(context, [])

    def live(self) -> dict[str, int]:
        """Return how many resources of each kind are alive."""
        kinds = {"memory": self.memory, "host": self.host, "contexts": self.contexts, "streams": self.streams,
                 "events": self.events, "modules": self.modules}  # fmt: skip
        return {kind: len(held) for kind, held in kinds.items() if held}

    def view(self, pointer: int, dtype: type[np.generic], count: int) -> NDArray[Any]:
        """Return ``count`` values of ``dtype`` at device address ``pointer``, which must lie in one allocation."""
        nbytes = count * np.dtype(dtype).itemsize
        assert self._device_span(pointer, nbytes), f"{nbytes} bytes at {pointer:#x} are not inside one allocation"
        return np.frombuffer((ctypes.c_char * nbytes).from_address(pointer), dtype=dtype)

    # -- helpers ----------------------------------------------------------------------------------------------------

    def _handle(self) -> int:
        self._next += 0x10
        return self._next

    def _context(self) -> int | None:
        return self.current.get(threading.get_ident())

    def _device_span(self, pointer: int, nbytes: int) -> bool:
        return any(base <= pointer and pointer + nbytes <= base + len(held) for base, held in self.memory.items())

    def _pinned(self, address: int) -> bool:
        return any(base <= address < base + len(held) for base, held in self.host.items())

    def _stream(self, stream: int | None) -> int:
        """Return the error a stream call gets (0 when the stream belongs to the thread's current context)."""
        if self._owned():
            return self._owned()
        if stream not in self.streams:
            return 400
        return 0 if self.streams[stream] == self._context() else 201

    def _owned(self) -> int:
        return 0 if self._initialised and self._context() is not None else (201 if self._initialised else 3)

    def _run(self) -> None:
        queue = self.queue
        while queue:
            queue.pop(0)()

    # -- the driver API ---------------------------------------------------------------------------------------------

    def cuInit(self, flags: int) -> int:
        if not self.devices:
            return 100
        self._initialised = True
        return 0

    def cuDriverGetVersion(self, version: Any) -> int:
        _out(version).value = self.version
        return 0

    def cuDeviceGetCount(self, count: Any) -> int:
        if not self._initialised:
            return 3
        _out(count).value = len(self.devices)
        return 0

    def cuDeviceGet(self, device: Any, ordinal: int) -> int:
        if not 0 <= ordinal < len(self.devices):
            return 101
        _out(device).value = ordinal
        return 0

    def cuDeviceGetName(self, name: Any, length: int, device: Any) -> int:
        name.value = self.devices[_value(device)].name.encode()[: length - 1]
        return 0

    def cuDeviceTotalMem_v2(self, nbytes: Any, device: Any) -> int:
        _out(nbytes).value = self.devices[_value(device)].total_memory
        return 0

    def cuDeviceGetAttribute(self, value: Any, attribute: int, device: Any) -> int:
        described = self.devices[_value(device)]
        known = described.attributes | {75: described.capability[0], 76: described.capability[1]}
        if attribute not in known:
            return 1
        _out(value).value = known[attribute]
        return 0

    def cuCtxCreate_v2(self, context: Any, flags: int, device: Any) -> int:
        if not self._initialised:
            return 3
        handle = self._handle()
        self.contexts[handle] = _value(device)
        self.created.append(_value(device))
        self.current[threading.get_ident()] = handle
        _out(context).value = handle
        return 0

    def cuCtxDestroy_v2(self, context: int) -> int:
        if context not in self.contexts:
            return 201
        del self.contexts[context]
        self.queues.pop(context, None)
        self.current = {thread: None if held == context else held for thread, held in self.current.items()}
        return 0

    def cuCtxSetCurrent(self, context: int) -> int:
        if context not in self.contexts:
            return 201
        self.current[threading.get_ident()] = context
        return 0

    def cuModuleLoadDataEx(self, module: Any, image: bytes, options: int, names: Any, values: Any) -> int:
        if self._owned():
            return self._owned()
        context = self._context()
        assert context is not None
        maker = _MAKER.search(image)  # PTX (".version"): the driver JIT-compiles it, so it is not newer than itself
        ptx = maker is not None and b"\n.version " in image
        if ptx and maker is not None and int(maker.group(1)) * 1000 + int(maker.group(2)) * 10 > self.version:
            return 222
        target = _TARGET.search(image)
        capability = self.devices[self.contexts[context]].capability
        if target is None or int(target.group(1)) > capability[0] * 10 + capability[1]:
            return 209
        handle = self._handle()
        self.modules[handle] = _Module([name.decode() for name in _ENTRY.findall(image)], context)
        _out(module).value = handle
        return 0

    def cuModuleUnload(self, module: int) -> int:
        if module not in self.modules:
            return 400
        del self.modules[module]
        return 0

    def cuModuleGetFunction(self, function: Any, module: int, name: bytes) -> int:
        if self._owned():
            return self._owned()
        held = self.modules.get(module)
        if held is None or held.context != self._context():
            return 400
        if name.decode() not in held.entries:
            return 500
        handle = self._handle()
        self.functions[handle] = _Function(name.decode(), held.context)
        _out(function).value = handle
        return 0

    def cuFuncSetAttribute(self, function: int, attribute: int, value: int) -> int:
        if function not in self.functions or attribute != 8:
            return 1
        self.functions[function].shared_max = value
        return 0

    def cuMemAlloc_v2(self, pointer: Any, nbytes: int) -> int:
        if self._owned():
            return self._owned()
        if nbytes == 0:
            return 1
        held = (ctypes.c_char * nbytes)()
        ctypes.memset(held, DEVICE_GARBAGE, nbytes)
        self.memory[ctypes.addressof(held)] = held
        _out(pointer).value = ctypes.addressof(held)
        return 0

    def cuMemFree_v2(self, pointer: int) -> int:
        if pointer not in self.memory:
            return 1
        assert not self.queue, "device memory was freed while work was queued"
        del self.memory[pointer]
        return 0

    def cuMemAllocHost_v2(self, address: Any, nbytes: int) -> int:
        if self._owned():
            return self._owned()
        held = (ctypes.c_char * nbytes)()
        ctypes.memset(held, HOST_GARBAGE, nbytes)
        self.host[ctypes.addressof(held)] = held
        _out(address).value = ctypes.addressof(held)
        return 0

    def cuMemFreeHost(self, address: int) -> int:
        if address not in self.host:
            return 1
        assert not self.queue, "page-locked memory was freed while work was queued"
        del self.host[address]
        return 0

    def cuMemcpyHtoDAsync_v2(self, target: int, source: int, nbytes: int, stream: int) -> int:
        if self._stream(stream):
            return self._stream(stream)
        if not self._device_span(target, nbytes):
            return 1
        self.transfers.append(("up", nbytes))
        staged: Any = source if self._pinned(source) else ctypes.string_at(source, nbytes)
        self.queue.append(lambda: _move(target, staged, nbytes))
        return 0

    def cuMemcpyDtoHAsync_v2(self, target: int, source: int, nbytes: int, stream: int) -> int:
        if self._stream(stream):
            return self._stream(stream)
        if not self._device_span(source, nbytes):
            return 1
        self.transfers.append(("down", nbytes))
        self.queue.append(lambda: _move(target, source, nbytes))
        if not self._pinned(target):
            self._run()
        return 0

    def cuMemsetD8Async(self, target: int, value: int, nbytes: int, stream: int) -> int:
        if self._stream(stream):
            return self._stream(stream)
        if not self._device_span(target, nbytes):
            return 1
        self.queue.append(lambda: _set(target, value, nbytes))
        return 0

    def cuStreamCreate(self, stream: Any, flags: int) -> int:
        if self._owned():
            return self._owned()
        handle = self._handle()
        context = self._context()
        assert context is not None
        self.streams[handle] = context
        _out(stream).value = handle
        return 0

    def cuStreamDestroy_v2(self, stream: int) -> int:
        if stream not in self.streams:
            return 400
        del self.streams[stream]
        return 0

    def cuStreamSynchronize(self, stream: int) -> int:
        if self._stream(stream):
            return self._stream(stream)
        self._run()
        return 0

    def cuLaunchKernel(
        self, function: int, gx: int, gy: int, gz: int, bx: int, by: int, bz: int, shared: int, stream: int,
        arguments: Any, extra: Any,
    ) -> int:  # fmt: skip
        if self._stream(stream):
            return self._stream(stream)
        held = self.functions.get(function)
        if held is None or held.context != self._context():
            return 400
        if min(gx, gy, gz, bx, by, bz) < 1 or bx * by * bz > 1024 or shared > held.shared_max:
            return 1
        kernel = self.kernels[held.name]
        pointers = tuple(ctypes.c_uint64.from_address(arguments[index]).value for index in range(kernel.buffers))
        params = None
        if kernel.params is not None:
            params = kernel.params.from_buffer_copy(
                ctypes.string_at(arguments[kernel.buffers], ctypes.sizeof(kernel.params))
            )
        launch = Launch(held.name, pointers, params, (gx, gy, gz), (bx, by, bz), shared)
        self.launches.append(launch)

        def run() -> None:
            kernel.run(self, launch)
            self.clock += self.launch_seconds

        self.queue.append(run)
        return 0

    def cuEventCreate(self, event: Any, flags: int) -> int:
        if self._owned():
            return self._owned()
        handle = self._handle()
        self.events[handle] = None
        _out(event).value = handle
        return 0

    def cuEventRecord(self, event: int, stream: int) -> int:
        if self._stream(stream):
            return self._stream(stream)
        if event not in self.events:
            return 400

        def record() -> None:
            self.events[event] = self.clock

        self.queue.append(record)
        return 0

    def cuEventElapsedTime(self, milliseconds: Any, start: int, end: int) -> int:
        first, last = self.events.get(start), self.events.get(end)
        if first is None or last is None or self.queue:
            return 600
        _out(milliseconds).value = (last - first) * 1e3
        return 0

    def cuEventDestroy_v2(self, event: int) -> int:
        if event not in self.events:
            return 400
        del self.events[event]
        return 0

    def cuGetErrorName(self, code: int, name: Any) -> int:
        if code not in ERRORS:
            return 1
        _out(name).value = ERRORS[code][0].encode()
        return 0

    def cuGetErrorString(self, code: int, text: Any) -> int:
        if code not in ERRORS:
            return 1
        _out(text).value = ERRORS[code][1].encode()
        return 0


def _value(argument: Any) -> int:
    return int(getattr(argument, "value", argument))


def _move(target: int, source: Any, nbytes: int) -> None:
    ctypes.memmove(target, source, nbytes)


def _set(target: int, value: int, nbytes: int) -> None:
    ctypes.memset(target, value, nbytes)


@dataclass
class _Program:
    source: str
    name: str
    log: bytes = b""
    ptx: bytes = b""
    cubin: bytes = b""  # compiled for a real architecture only


class FakeNvrtc(Library):
    """NVRTC's stand-in. ``compiled`` lists the options of every compilation; without ``builtins`` none succeeds."""

    def __init__(self) -> None:
        super().__init__()
        self.version = (13, 0)
        self.architectures = [75, 80, 86, 89, 90, 100, 120]
        self.builtins = True  # its builtins library is loaded
        self.compiled: list[list[str]] = []
        self.programs: dict[int, _Program] = {}
        self._next = 0x100

    def nvrtcVersion(self, major: Any, minor: Any) -> int:
        _out(major).value, _out(minor).value = self.version
        return 0

    def nvrtcGetNumSupportedArchs(self, count: Any) -> int:
        _out(count).value = len(self.architectures)
        return 0

    def nvrtcGetSupportedArchs(self, found: Any) -> int:
        for index, value in enumerate(self.architectures):
            found[index] = value
        return 0

    def nvrtcCreateProgram(self, program: Any, source: bytes, name: bytes, headers: int, a: Any, b: Any) -> int:
        self._next += 1
        self.programs[self._next] = _Program(source.decode(), name.decode())
        _out(program).value = self._next
        return 0

    def nvrtcCompileProgram(self, program: Any, count: int, options: Any) -> int:
        held = self.programs.get(_value(program))
        if held is None:
            return 4
        given = [options[index].decode() for index in range(count)]
        self.compiled.append(given)
        arch = [option.split("_")[1] for option in given if option.startswith(("--gpu-architecture=compute_",
                                                                               "--gpu-architecture=sm_"))]  # fmt: skip
        if any(not option.startswith("--") for option in given) or not arch or int(arch[0]) not in self.architectures:
            return 5
        real = any(option.startswith("--gpu-architecture=sm_") for option in given)
        if not self.builtins:
            held.log = b"nvrtc: error: failed to open libnvrtc-builtins.so.13.0."
            return 7
        if "#error" in held.source:
            held.log = f"{held.name}(1): error: #error directive".encode()
            return 6
        if "#warning" in held.source:
            held.log = f"{held.name}(1): warning: #warning directive".encode()
        entries = "".join(f".visible .entry {name}(\n)\n{{\n  ret;\n}}\n" for name in _GLOBAL.findall(held.source))
        maker = f"// fake NVRTC {self.version[0]}.{self.version[1]}"
        held.ptx = f"{maker}\n.version 8.5\n.target sm_{arch[0]}\n.address_size 64\n\n{entries}".encode()
        held.cubin = b"\x7fELF\x00" + f"\n{maker}\n.target sm_{arch[0]}\n{entries}".encode() if real else b""
        return 0

    def nvrtcGetProgramLogSize(self, program: Any, size: Any) -> int:
        _out(size).value = len(self.programs[_value(program)].log) + 1
        return 0

    def nvrtcGetProgramLog(self, program: Any, log: Any) -> int:
        log.value = self.programs[_value(program)].log
        return 0

    def nvrtcGetPTXSize(self, program: Any, size: Any) -> int:
        _out(size).value = len(self.programs[_value(program)].ptx) + 1
        return 0

    def nvrtcGetPTX(self, program: Any, ptx: Any) -> int:
        ptx.value = self.programs[_value(program)].ptx
        return 0

    def nvrtcGetCUBINSize(self, program: Any, size: Any) -> int:
        _out(size).value = len(self.programs[_value(program)].cubin)  # 0 after a virtual architecture, as NVRTC's
        return 0

    def nvrtcGetCUBIN(self, program: Any, cubin: Any) -> int:
        image = self.programs[_value(program)].cubin
        ctypes.memmove(cubin, image, len(image))
        return 0

    def nvrtcDestroyProgram(self, program: Any) -> int:
        del self.programs[_value(_out(program))]
        return 0

    def nvrtcGetErrorString(self, code: int) -> bytes | None:
        return NVRTC_ERRORS.get(code)


@dataclass
class Fakes:
    """The two fake libraries the runtime is using."""

    driver: FakeDriver
    nvrtc: FakeNvrtc


def install(monkeypatch: pytest.MonkeyPatch, devices: int = 1) -> Fakes:
    """Make `idun.cuda.runtime` use fresh fake libraries until the test ends."""
    fakes = Fakes(FakeDriver(devices), FakeNvrtc())
    fakes.driver.kernels.update(KERNELS)
    runtime.bind(fakes.driver, runtime.DRIVER_API, "the fake driver")
    runtime.bind(fakes.nvrtc, runtime.NVRTC_API, "the fake NVRTC")
    monkeypatch.setattr(runtime, "load_driver", lambda: fakes.driver)
    monkeypatch.setattr(runtime, "load_nvrtc", lambda: fakes.nvrtc)
    return fakes


# -- toy kernels: a source for a real compiler, and the same kernels emulated for the fake driver -------------------


class Params(ctypes.Structure):
    """Mirror of the toy source's ``Params``."""

    _fields_ = (("n", ctypes.c_uint32), ("add", ctypes.c_uint32))


SOURCE = """
struct Params { unsigned int n; unsigned int add; };
extern "C" __global__ void add_to(const unsigned int *x, unsigned int *out, Params p) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.n) { out[i] = x[i] + p.add; }
}
extern "C" __global__ void bump(unsigned int *x) { x[blockIdx.x * blockDim.x + threadIdx.x] += 1u; }
extern "C" __global__ void shared_sum(unsigned int *out, Params p) {
  extern __shared__ unsigned int scratch[];
  scratch[threadIdx.x] = threadIdx.x + p.add;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int total = 0u;
    for (unsigned int i = 0u; i < blockDim.x; i++) { total += scratch[i]; }
    out[blockIdx.x] = total;
  }
}
"""


def _add_to(driver: FakeDriver, launch: Launch) -> None:
    x, out = (driver.view(pointer, np.uint32, launch.params.n) for pointer in launch.pointers)
    out[:] = x + np.uint32(launch.params.add)


def _bump(driver: FakeDriver, launch: Launch) -> None:
    driver.view(launch.pointers[0], np.uint32, launch.grid[0] * launch.block[0])[:] += np.uint32(1)


def _shared_sum(driver: FakeDriver, launch: Launch) -> None:
    threads = launch.block[0]
    assert launch.shared_bytes >= 4 * threads, "the kernel's scratch does not fit its shared memory"
    total = threads * (threads - 1) // 2 + threads * launch.params.add
    driver.view(launch.pointers[0], np.uint32, launch.grid[0])[:] = np.uint32(total)


KERNELS = {"add_to": Kernel(2, Params, _add_to), "bump": Kernel(1, None, _bump),
           "shared_sum": Kernel(1, Params, _shared_sum)}  # fmt: skip
