"""The CUDA runtime: the binding, the compiler and the context, against the fake libraries and against a real GPU."""

import ctypes
import platform
import re
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numba.cuda.cudadrv import enums

import idun
from idun.cuda import runtime
from idun.errors import BackendUnavailableError, IdunError, KernelCompileError
from tests.unit.cuda.fake_driver import (
    DEVICE_GARBAGE,
    SOURCE,
    FakeDriver,
    FakeNvrtc,
    Fakes,
    Kernel,
    Launch,
    Params,
    install,
)

N = 64
GRID, BLOCK = (1, 1, 1), (N, 1, 1)


def address(array: np.ndarray[Any, Any]) -> int:
    return int(array.__array_interface__["data"][0])


def values(ctx: runtime.Context, buffer: runtime.Buffer, count: int) -> np.ndarray[Any, Any]:
    """Download ``count`` uint32 of a buffer into ordinary memory and wait."""
    out = np.empty(count, np.uint32)
    ctx.download(buffer, address(out), 4 * count)
    ctx.sync()
    return out


def loaded(**options: Any) -> runtime.Context:
    ctx = runtime.Context(**options)
    ctx.load(SOURCE)
    return ctx


# -- finding and binding the libraries --------------------------------------------------------------------------------


def test_a_library_without_a_function_is_refused_by_name() -> None:
    whole = FakeDriver()
    library = types.SimpleNamespace(
        **{name: getattr(whole, name) for name in runtime.DRIVER_API if "Launch" not in name}
    )
    with pytest.raises(BackendUnavailableError, match=r"libfake\.so does not export cuLaunchKernel"):
        runtime.bind(library, runtime.DRIVER_API, "libfake.so")


def test_the_binding_declares_every_signature() -> None:
    driver, nvrtc = FakeDriver(), FakeNvrtc()
    runtime.bind(driver, runtime.DRIVER_API, "driver")
    runtime.bind(nvrtc, runtime.NVRTC_API, "nvrtc")
    for library, api in ((driver, runtime.DRIVER_API), (nvrtc, runtime.NVRTC_API)):
        for name, arguments in api.items():
            function = getattr(library, name)
            assert function.argtypes == arguments
            assert function.restype is (ctypes.c_char_p if name == "nvrtcGetErrorString" else ctypes.c_int)


@pytest.mark.parametrize(
    ("load", "env", "names", "what"),
    [
        (runtime.load_driver, runtime.DRIVER_ENV, runtime.DRIVER_NAMES, "the CUDA driver library"),
        (runtime.load_nvrtc, runtime.NVRTC_ENV, runtime.NVRTC_NAMES, "NVRTC, the CUDA runtime compiler"),
    ],
)
def test_a_missing_library_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, load: Any, env: str, names: tuple[str, ...], what: str
) -> None:
    def missing(name: str) -> Any:
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(ctypes, "CDLL", missing)
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(BackendUnavailableError) as refused:
        load.__wrapped__()
    assert str(refused.value).startswith(f"{what} cannot be loaded (")
    assert all(f"{name}: cannot open shared object file" in str(refused.value) for name in names)
    assert f"or set {env} to the library's path" in str(refused.value)
    monkeypatch.setenv(env, "/opt/elsewhere/lib.so")
    with pytest.raises(BackendUnavailableError, match=r"\(/opt/elsewhere/lib.so: /opt/elsewhere/lib.so: cannot open"):
        load.__wrapped__()


def test_the_first_library_that_opens_is_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = []

    def cdll(name: str) -> Any:
        opened.append(name)
        if name == runtime.DRIVER_NAMES[0]:
            raise OSError("not there")
        return FakeDriver()

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(ctypes, "CDLL", cdll)
    monkeypatch.delenv(runtime.DRIVER_ENV, raising=False)
    library = runtime.load_driver.__wrapped__()
    assert opened == list(runtime.DRIVER_NAMES[:2])
    assert library.cuInit.argtypes == (ctypes.c_uint,)


def test_nvrtc_builtins_beside_the_library_are_loaded_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened = []

    def cdll(name: str, mode: int = 0) -> Any:
        opened.append((Path(name).name, mode))
        if name.endswith("libnvrtc-builtins.so.12.4"):
            raise OSError("wrong ELF class")
        return FakeNvrtc()

    old, new = tmp_path / "nvidia/cuda_nvrtc/lib", tmp_path / "nvidia/cu13/lib"
    for folder, names in (
        (old, ["libnvrtc.so.12", "libnvrtc-builtins.so.12.4"]),
        (new, ["libnvrtc.so.13", "libnvrtc-builtins.so.13.0", "libnvrtc-builtins.so.12.9", "libnvrtc.so"]),
    ):
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).touch()
    assert runtime.nvrtc_builtins(str(new / "libnvrtc.so.13")) == [str(new / "libnvrtc-builtins.so.13.0")]
    assert runtime.nvrtc_builtins(str(new / "libnvrtc.so")) == []  # no version to match
    (old / "libnvrtc.so").symlink_to("libnvrtc.so.12")
    assert runtime.nvrtc_builtins(str(old / "libnvrtc.so")) == [str(old / "libnvrtc-builtins.so.12.4")]
    assert runtime.nvrtc_builtins("libnvrtc.so.13") == []  # the loader's path serves both
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(ctypes, "CDLL", cdll)
    monkeypatch.setenv(runtime.NVRTC_ENV, str(new / "libnvrtc.so.13"))
    runtime.load_nvrtc.__wrapped__()
    assert opened == [("libnvrtc-builtins.so.13.0", ctypes.RTLD_GLOBAL), ("libnvrtc.so.13", 0)]
    monkeypatch.setenv(runtime.NVRTC_ENV, str(old / "libnvrtc.so.12"))
    with pytest.raises(BackendUnavailableError, match=r"libnvrtc.so.12: wrong ELF class\)"):
        runtime.load_nvrtc.__wrapped__()  # a library whose builtins do not load cannot compile
    assert opened[2:] == [("libnvrtc-builtins.so.12.4", ctypes.RTLD_GLOBAL)]


def test_only_linux_has_a_cuda_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    for load in (runtime.load_driver, runtime.load_nvrtc):
        with pytest.raises(BackendUnavailableError, match="needs Linux, this is Darwin"):
            load.__wrapped__()


def test_nvrtc_is_looked_for_in_pip_packages_newest_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, new = tmp_path / "nvidia/cuda_nvrtc/lib", tmp_path / "nvidia/cu13/lib"
    for folder, names in ((old, ["libnvrtc.so.12"]), (new, ["libnvrtc.so.13", "libnvrtc-builtins.so.13.0"])):
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).touch()
    monkeypatch.setattr(sys, "path", ["", str(tmp_path)])
    monkeypatch.delenv(runtime.NVRTC_ENV, raising=False)
    assert runtime.nvrtc_candidates() == [
        str(new / "libnvrtc.so.13"),
        str(old / "libnvrtc.so.12"),
        *runtime.NVRTC_NAMES,
    ]
    monkeypatch.setenv(runtime.NVRTC_ENV, "/somewhere/libnvrtc.so")
    assert runtime.nvrtc_candidates() == ["/somewhere/libnvrtc.so"]
    monkeypatch.setenv(runtime.DRIVER_ENV, "/somewhere/libcuda.so")
    assert runtime.driver_candidates() == ["/somewhere/libcuda.so"]


# -- the device -------------------------------------------------------------------------------------------------------


def test_the_ids_passed_to_the_driver_are_its_headers() -> None:
    """The fake driver answers the ids the binding sends, whatever they are: numba's copy of cuda.h's ids decides."""
    assert (
        runtime._SUCCESS,
        runtime._ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        runtime._ATTRIBUTE_MAX_GRID,
        runtime._ATTRIBUTE_MULTIPROCESSOR_COUNT,
        runtime._ATTRIBUTE_CAPABILITY_MAJOR,
        runtime._ATTRIBUTE_CAPABILITY_MINOR,
        runtime._ATTRIBUTE_MAX_SHARED_OPTIN,
        runtime._FUNCTION_MAX_DYNAMIC_SHARED,
    ) == (
        enums.CUDA_SUCCESS,
        enums.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        (
            enums.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
            enums.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
            enums.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
        ),
        enums.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        enums.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        enums.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        enums.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        enums.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    )


def test_probe_describes_the_device(fakes: Fakes) -> None:
    assert runtime.probe() == runtime.DeviceInfo(
        ordinal=0,
        name="Fake T4",
        total_memory_bytes=15 << 30,
        compute_capability=(7, 5),
        multiprocessors=40,
        max_threads_per_block=1024,
        max_shared_memory_bytes=64 * 1024,
        max_grid=(2**31 - 1, 65535, 65535),
        driver_version=(12, 4),
        device_count=1,
    )
    assert fakes.driver.live() == {}  # no context was made


def test_no_gpu_is_a_backend_that_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, devices=0)
    with pytest.raises(BackendUnavailableError, match=r"no usable GPU.*cuInit failed with CUDA_ERROR_NO_DEVICE"):
        runtime.probe()


def test_the_device_is_named_by_argument_or_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, devices=2)
    monkeypatch.delenv(runtime.DEVICE_ENV, raising=False)
    assert runtime.device_ordinal() == 0
    assert runtime.device_ordinal(1) == 1
    monkeypatch.setenv(runtime.DEVICE_ENV, "1")
    assert runtime.device_ordinal() == 1
    assert runtime.probe().ordinal == 1
    assert runtime.device_ordinal(0) == 0  # the argument wins
    with pytest.raises(ValueError, match="CUDA device 2 does not exist: the driver lists 2"):
        runtime.device_ordinal(2)
    monkeypatch.setenv(runtime.DEVICE_ENV, "first")
    with pytest.raises(ValueError, match="IDUN_CUDA_DEVICE must be a device number, got 'first'"):
        runtime.device_ordinal()


def test_work_side_by_side_is_spread_over_every_gpu_unless_one_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, devices=3)
    monkeypatch.delenv(runtime.DEVICE_ENV, raising=False)
    assert runtime.device_ordinals() == [0, 1, 2]
    monkeypatch.setenv(runtime.DEVICE_ENV, "2")
    assert runtime.device_ordinals() == [2]
    monkeypatch.setenv(runtime.DEVICE_ENV, "3")
    with pytest.raises(ValueError, match="CUDA device 3 does not exist: the driver lists 3"):
        runtime.device_ordinals()


# -- errors -----------------------------------------------------------------------------------------------------------


def everything(fakes: Fakes) -> None:
    """Make every driver call the binding has, at least once."""
    fakes.driver.kernels["shared_sum"] = Kernel(1, Params, lambda driver, launch: None)
    first = loaded()
    second = runtime.Context()
    buffer = first.alloc(4 * N)  # the first context is made current again
    pinned = first.alloc_host(4 * N)
    first.upload(buffer, pinned.address, 4 * N)
    first.launch(first.kernel("shared_sum"), [buffer], Params(N, 0), GRID, BLOCK, 60_000)
    first.download(buffer, pinned.address, 4 * N)
    first.sync()
    first.free(buffer)
    second.close()
    first.close()


@pytest.mark.parametrize("name", [name for name in runtime.DRIVER_API if not name.startswith("cuGetError")])
def test_every_driver_error_names_its_call(fakes: Fakes, name: str) -> None:
    everything(fakes)
    assert name in fakes.driver.calls  # the scenario reaches the call
    fakes.driver.fail[name] = 2
    with pytest.raises(IdunError, match=rf"{name} failed with CUDA_ERROR_OUT_OF_MEMORY \(2\): out of memory"):
        everything(fakes)


def test_an_error_the_driver_cannot_name_keeps_its_code(fakes: Fakes) -> None:
    fakes.driver.fail["cuDeviceGetCount"] = 4242
    with pytest.raises(IdunError, match=r"cuDeviceGetCount failed with an unknown error \(4242\): no description"):
        runtime.probe()


def compiler(fakes: Fakes) -> None:
    """Make every NVRTC call the binding has."""
    runtime.nvrtc_version()
    runtime.architecture((7, 5))
    runtime.compile_ptx(SOURCE, "compute_75")
    runtime.compile_cubin(SOURCE, "sm_75")


@pytest.mark.parametrize("name", [name for name in runtime.NVRTC_API if name != "nvrtcGetErrorString"])
def test_every_compiler_error_names_its_call(fakes: Fakes, name: str) -> None:
    compiler(fakes)
    assert name in fakes.nvrtc.calls
    fakes.nvrtc.fail[name] = 5
    with pytest.raises(IdunError, match=rf"NVRTC: {name} failed with NVRTC_ERROR_INVALID_OPTION \(5\)"):
        compiler(fakes)
    fakes.nvrtc.fail[name] = 77
    with pytest.raises(IdunError, match=rf"NVRTC: {name} failed with an unknown error \(77\)"):
        compiler(fakes)


# -- the compiler -----------------------------------------------------------------------------------------------------


def test_a_source_compiles_with_the_fixed_options(fakes: Fakes) -> None:
    ptx, warnings = runtime.compile_ptx(SOURCE, "compute_75")
    assert fakes.nvrtc.compiled == [["--fmad=false", "--std=c++17", "--gpu-architecture=compute_75"]]
    assert runtime.entry_points(ptx) == ["add_to", "bump", "shared_sum"]
    assert warnings == ""
    assert runtime.nvrtc_version() == (13, 0)
    assert fakes.nvrtc.programs == {}  # the program was destroyed


def test_warnings_are_returned_and_errors_raised_with_the_log(fakes: Fakes) -> None:
    _, warnings = runtime.compile_ptx("#warning careful\n" + SOURCE, "compute_75", name="toy.cu")
    assert warnings == "toy.cu(1): warning: #warning directive"
    with pytest.raises(KernelCompileError, match=r"toy.cu\(1\): error: #error directive"):
        runtime.compile_ptx("#error no\n", "compute_75", name="toy.cu")
    assert fakes.nvrtc.programs == {}


def test_a_compiler_failure_that_is_not_the_source_carries_the_log(fakes: Fakes) -> None:
    fakes.nvrtc.builtins = False
    reason = r"NVRTC_ERROR_BUILTIN_OPERATION_FAILURE \(7\): nvrtc: error: failed to open libnvrtc-builtins.so.13.0."
    with pytest.raises(IdunError, match=f"NVRTC: nvrtcCompileProgram failed with {reason}") as failed:
        runtime.compile_ptx(SOURCE, "compute_75")
    assert not isinstance(failed.value, KernelCompileError)
    assert fakes.nvrtc.programs == {}


def test_the_architecture_is_the_newest_the_compiler_knows_and_the_device_runs(fakes: Fakes) -> None:
    assert runtime.supported_architectures() == [75, 80, 86, 89, 90, 100, 120]
    assert runtime.architecture((7, 5)) == "compute_75"
    assert runtime.architecture((8, 7)) == "compute_86"
    assert runtime.architecture((12, 1)) == "compute_120"
    with pytest.raises(BackendUnavailableError, match=r"NVRTC 13.0 does not compile for compute capability 6.1"):
        runtime.architecture((6, 1))


def test_the_binary_is_for_the_newest_real_architecture_of_the_devices_major_version(fakes: Fakes) -> None:
    assert runtime.binary_architecture((7, 5)) == "sm_75"
    assert runtime.binary_architecture((8, 7)) == "sm_86"  # an 8.6 binary runs on 8.7
    assert runtime.binary_architecture((12, 1)) == "sm_120"
    with pytest.raises(BackendUnavailableError, match=r"NVRTC 13.0 builds no binary for compute capability 6.1"):
        runtime.binary_architecture((6, 1))
    fakes.nvrtc.architectures = [75, 80]
    with pytest.raises(BackendUnavailableError, match=r"compute capability 9\.0"):  # no 9.x: PTX is not loaded instead
        runtime.binary_architecture((9, 0))


def test_a_newer_nvrtc_than_the_driver_loads_as_a_binary(fakes: Fakes) -> None:
    fakes.nvrtc.version, fakes.driver.version = (13, 5), 13000  # NVRTC 13.5 under a CUDA 13.0 driver
    ctx = runtime.Context()
    assert ctx.load(SOURCE) == ""
    assert fakes.nvrtc.compiled[-1] == ["--fmad=false", "--std=c++17", "--gpu-architecture=sm_75"]
    assert ctx.kernel("add_to") is not None
    ptx, _ = runtime.compile_ptx(SOURCE, "compute_75")  # the PTX of that NVRTC: the driver refuses it
    binary, _ = runtime.compile_cubin(SOURCE, "sm_75")
    assert b"\x00" in binary  # read as bytes, not as a string
    module = ctypes.c_void_p()
    assert fakes.driver.cuModuleLoadDataEx(ctypes.byref(module), ptx, 0, None, None) == 222
    assert fakes.driver.cuModuleLoadDataEx(ctypes.byref(module), binary, 0, None, None) == 0
    ctx.close()


def test_a_module_newer_than_the_device_is_refused_by_the_driver(fakes: Fakes) -> None:
    fakes.nvrtc.architectures = [80]
    fakes.driver.devices[0].capability = (8, 0)
    ctx = runtime.Context()
    fakes.driver.devices[0].capability = (7, 5)  # as if the compiler had aimed too high
    with pytest.raises(IdunError, match="cuModuleLoadDataEx failed with CUDA_ERROR_NO_BINARY_FOR_GPU"):
        ctx.load(SOURCE)
    ctx.close()


# -- launches ---------------------------------------------------------------------------------------------------------


def test_a_launch_passes_the_buffers_pointers_then_the_parameters_by_value(fakes: Fakes) -> None:
    ctx = loaded()
    x, out = ctx.alloc(4 * N), ctx.alloc(4 * N)
    start = np.arange(N, dtype=np.uint32)
    ctx.upload(x, address(start), 4 * N)
    params = Params(N, 5)
    ctx.launch(ctx.kernel("add_to"), [x, out], params, (1, 2, 3), (N, 1, 1))
    params.add = 99  # the driver read the structure during the call
    ctx.launch(ctx.kernel("bump"), [out], None, GRID, BLOCK)
    assert fakes.driver.launches == [
        Launch("add_to", (x.pointer, out.pointer), fakes.driver.launches[0].params, (1, 2, 3), (N, 1, 1), 0),
        Launch("bump", (out.pointer,), None, GRID, BLOCK, 0),
    ]
    assert (fakes.driver.launches[0].params.n, fakes.driver.launches[0].params.add) == (N, 5)
    np.testing.assert_array_equal(values(ctx, out, N), start + 6)
    ctx.close()


def test_queued_work_runs_in_order_and_only_when_waited_for(fakes: Fakes) -> None:
    ctx = loaded()
    x = ctx.alloc(4 * N)
    ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
    ctx.fill(x, 2, 4 * N)
    ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
    assert fakes.driver.count("cuStreamSynchronize") == 0
    assert (fakes.driver.view(x.pointer, np.uint8, 4 * N) == DEVICE_GARBAGE).all()  # not even zeroed yet
    ctx.sync()
    np.testing.assert_array_equal(fakes.driver.view(x.pointer, np.uint32, N), 0x02020203)
    assert fakes.driver.count("cuStreamSynchronize") == 1
    ctx.close()


def test_memory_is_zeroed_and_copies_go_both_ways(anywhere: str) -> None:
    with loaded() as ctx:
        x, out = ctx.alloc(4 * N), ctx.alloc(4 * N)
        assert (x.nbytes, ctx.alloc(0).nbytes) == (4 * N, 1)
        np.testing.assert_array_equal(values(ctx, x, N), 0)
        start = np.arange(N, dtype=np.uint32)
        ctx.upload(x, address(start), 4 * N)
        start[:] = 0  # ordinary memory is copied before upload returns
        ctx.launch(ctx.kernel("add_to"), [x, out], Params(N, 7), GRID, BLOCK)
        np.testing.assert_array_equal(values(ctx, out, N), np.arange(N) + 7)
        pinned = ctx.alloc_host(4 * N)
        ctx.download(out, pinned.address, 4 * N)
        ctx.sync()
        view = pinned.view(np.uint32, (N,))
        np.testing.assert_array_equal(view, np.arange(N) + 7)
        view += 1
        ctx.upload(x, pinned.address + 4 * 8, 4 * 8, 4 * 16)  # eight values to offset 16
        ctx.fill(x, 0xFF, 4, 4)
        ctx.sync()
        expected = np.arange(N, dtype=np.uint32)
        expected[16:24] = np.arange(8, 16) + 8
        expected[1] = 0xFFFFFFFF
        np.testing.assert_array_equal(values(ctx, x, N), expected)
        raw = np.empty(8, np.uint32)
        ctx.download(x, address(raw), 4 * 8, 4 * 16)
        ctx.sync()
        np.testing.assert_array_equal(raw, expected[16:24])
        with pytest.raises(ValueError, match=r"a \(65,\) uint32 array needs 260 bytes; the buffer has 256"):
            pinned.view(np.uint32, (N + 1,))


def test_spans_outside_a_buffer_and_foreign_buffers_are_refused(fakes: Fakes) -> None:
    ctx, other = loaded(), runtime.Context()
    x, foreign = ctx.alloc(16), other.alloc(16)
    for call in (ctx.upload, ctx.download):
        with pytest.raises(ValueError, match="9 bytes at offset 8 do not fit a buffer of 16"):
            call(x, 0, 9, 8)
    with pytest.raises(ValueError, match="17 bytes at offset 0 do not fit a buffer of 16"):
        ctx.fill(x, 0, 17)
    with pytest.raises(ValueError, match="-1 bytes at offset 0"):
        ctx.fill(x, 0, -1)
    with pytest.raises(ValueError, match="a fill value is a byte, got 256"):
        ctx.fill(x, 256, 1)
    with pytest.raises(ValueError, match="not a live buffer of this context"):
        ctx.fill(foreign, 0, 1)
    with pytest.raises(ValueError, match="not a live buffer of this context"):
        ctx.launch(ctx.kernel("bump"), [foreign], None, GRID, BLOCK)
    calls = len(fakes.driver.calls)
    ctx.upload(x, 0, 0)  # nothing to copy: no driver call
    ctx.download(x, 0, 0)
    ctx.fill(x, 1, 0)
    assert len(fakes.driver.calls) == calls
    ctx.free(x)
    with pytest.raises(ValueError, match="not a live buffer of this context"):
        ctx.free(x)
    other.close()
    ctx.close()


def test_a_launch_is_checked_before_the_driver_sees_it(fakes: Fakes) -> None:
    ctx = loaded()
    x = ctx.alloc(4 * N)
    bump = ctx.kernel("bump")
    with pytest.raises(ValueError, match="at most 31 arguments"):
        ctx.launch(bump, [x] * 31, Params(1, 1), GRID, BLOCK)
    for grid, block in (((0, 1, 1), BLOCK), (GRID, (N, 0, 1))):
        with pytest.raises(ValueError, match="need at least one block and one thread each way"):
            ctx.launch(bump, [x], None, grid, block)
    for grid in ((1, 65536, 1), (1, 1, 65536), (2**31, 1, 1)):
        with pytest.raises(ValueError, match=r"kernel bump asks for a grid of .* blocks; the device runs at most"):
            ctx.launch(bump, [x], None, grid, BLOCK)
    assert fakes.driver.launches == []
    ctx.close()


def test_shared_memory_above_the_default_is_opted_in_to_once(anywhere: str) -> None:
    with loaded() as ctx:
        kernel, out = ctx.kernel("shared_sum"), ctx.alloc(4 * 3)
        limit = ctx.info.max_shared_memory_bytes
        assert limit >= 60_000
        for shared, add in ((4 * N, 1), (60_000, 2), (50_000, 3)):
            ctx.launch(kernel, [out], Params(0, add), (3, 1, 1), BLOCK, shared)
            np.testing.assert_array_equal(values(ctx, out, 3), N * (N - 1) // 2 + N * add)
        assert kernel.shared_bytes == 60_000
        with pytest.raises(ValueError, match=f"kernel shared_sum asks for {limit + 16} bytes of shared memory"):
            ctx.launch(kernel, [out], Params(0, 1), (3, 1, 1), BLOCK, limit + 16)


def test_the_opt_in_is_one_driver_call(fakes: Fakes) -> None:
    ctx = loaded()
    kernel, out = ctx.kernel("shared_sum"), ctx.alloc(4)
    ctx.launch(kernel, [out], Params(0, 1), GRID, BLOCK, 48 * 1024)
    assert fakes.driver.count("cuFuncSetAttribute") == 0
    ctx.launch(kernel, [out], Params(0, 1), GRID, BLOCK, 60_000)
    ctx.launch(kernel, [out], Params(0, 1), GRID, BLOCK, 56_000)
    assert fakes.driver.count("cuFuncSetAttribute") == 1
    assert [launch.shared_bytes for launch in fakes.driver.launches] == [48 * 1024, 60_000, 56_000]
    ctx.close()


def test_gpu_seconds_are_those_of_the_launches_waited_for(fakes: Fakes) -> None:
    ctx = loaded()
    x = ctx.alloc(4 * N)
    assert ctx.sync() == 0.0  # the fill alone: no launch
    calls = len(fakes.driver.calls)
    assert ctx.sync() == 0.0
    assert len(fakes.driver.calls) == calls  # nothing queued: no driver call
    for _ in range(2):
        ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
    assert ctx.gpu_seconds == 0.0
    assert ctx.sync() == 0.5
    ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
    assert ctx.sync() == 0.25
    assert ctx.gpu_seconds == 0.75
    ctx.close()


def test_guard_mode_waits_inside_a_launch_and_sync_still_returns_the_seconds(fakes: Fakes) -> None:
    ctx = loaded(guard=True)
    x = ctx.alloc(4 * N)
    ctx.sync()
    for _ in range(2):
        ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
    assert not fakes.driver.queue  # each launch was waited for, its bands read
    assert ctx.gpu_seconds == 0.5
    assert ctx.sync() == 0.5
    assert ctx.sync() == 0.0
    ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
    assert ctx.sync() == 0.25
    assert ctx.gpu_seconds == 0.75
    ctx.close()


# -- guard mode -------------------------------------------------------------------------------------------------------


def overrun(at: int) -> Callable[[FakeDriver, Launch], None]:
    def run(driver: FakeDriver, launch: Launch) -> None:
        ctypes.c_uint32.from_address(launch.pointers[0] + at).value = 1

    return run


@pytest.mark.parametrize(
    ("at", "where"),
    [(4 * N, "past its end, first at byte offset 256"), (-4, "before its start, first at byte offset -4")],
)
def test_guard_bands_catch_a_write_outside_the_buffer(fakes: Fakes, at: int, where: str) -> None:
    fakes.driver.kernels["bump"] = Kernel(1, None, overrun(at))
    ctx = loaded(guard=True)
    first, x = ctx.alloc(8), ctx.alloc(4 * N)
    ctx.launch(ctx.kernel("add_to"), [first, x], Params(2, 1), GRID, BLOCK)  # stays inside: nothing is raised
    with pytest.raises(
        IdunError, match=rf"guard mode: after kernel bump, buffer 1 \(.*, 256 bytes\) was written {where}"
    ):
        ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
    ctx.sync()  # nothing queued: no check
    ctx.fill(first, 0, 1)
    with pytest.raises(IdunError, match="guard mode: after the last work before close"):
        ctx.close()
    assert fakes.driver.live() == {}


def test_without_guard_mode_there_are_no_bands_and_no_waits(fakes: Fakes) -> None:
    ctx = loaded()
    x = ctx.alloc(4 * N)
    assert (ctx.guard, x.pointer) == (False, x.base)
    ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
    assert fakes.driver.count("cuStreamSynchronize") == 0
    ctx.close()
    guarded = loaded(guard=True)
    y = guarded.alloc(4 * N)
    assert y.pointer == y.base + runtime.GUARD_BYTES
    guarded.close()


def test_a_real_overrun_is_caught_in_guard_mode(anywhere: str) -> None:
    with loaded(guard=True) as ctx:
        x = ctx.alloc(4 * N)
        ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
        np.testing.assert_array_equal(values(ctx, x, N), 1)
        with pytest.raises(IdunError, match=r"after kernel bump, buffer 0 .* was written past its end, first at byte"):
            ctx.launch(ctx.kernel("bump"), [x], None, GRID, (N + 1, 1, 1))  # one thread too many


def test_guard_mode_comes_from_the_environment(fakes: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not runtime.guard_mode()
    monkeypatch.setenv(runtime.GUARD_ENV, "1")
    ctx = runtime.Context()
    assert ctx.guard
    assert not runtime.Context(guard=False).guard
    monkeypatch.setenv(runtime.GUARD_ENV, "yes")
    with pytest.raises(ValueError, match="IDUN_CUDA_GUARD must be 1 or unset, got 'yes'"):
        runtime.Context()


# -- contexts ---------------------------------------------------------------------------------------------------------


def test_a_context_is_made_current_only_when_another_one_was(fakes: Fakes) -> None:
    first = loaded()
    assert fakes.driver.count("cuCtxSetCurrent") == 0  # creating it made it current
    first.alloc(8)
    second = loaded()
    second.alloc(8)
    assert fakes.driver.count("cuCtxSetCurrent") == 0
    x = first.alloc(4 * N)
    first.launch(first.kernel("bump"), [x], None, GRID, BLOCK)
    assert fakes.driver.count("cuCtxSetCurrent") == 1
    second.alloc(8)
    assert fakes.driver.count("cuCtxSetCurrent") == 2
    np.testing.assert_array_equal(values(first, x, N), 1)
    assert fakes.driver.count("cuCtxSetCurrent") == 3
    second.close()
    first.close()
    assert fakes.driver.live() == {}


def test_a_context_works_from_another_thread(anywhere: str) -> None:
    with loaded() as ctx:
        x = ctx.alloc(4 * N)
        seen: list[Any] = []

        def work() -> None:
            ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
            seen.append(values(ctx, x, N))

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        np.testing.assert_array_equal(seen[0], 1)
        ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
        np.testing.assert_array_equal(values(ctx, x, N), 2)


def test_close_frees_what_was_allocated(fakes: Fakes) -> None:
    ctx = loaded()
    buffers = [ctx.alloc(4 * N) for _ in range(3)]
    ctx.alloc_host(64)
    ctx.free(buffers[1])
    ctx.launch(ctx.kernel("bump"), [buffers[0]], None, GRID, BLOCK)
    assert fakes.driver.live() == {"memory": 2, "host": 1, "contexts": 1, "streams": 1, "events": 2, "modules": 1}
    ctx.close()  # waits for the launch first
    assert fakes.driver.live() == {}
    assert fakes.driver.count("cuMemAlloc_v2") == fakes.driver.count("cuMemFree_v2") == 3
    assert fakes.driver.count("cuMemAllocHost_v2") == fakes.driver.count("cuMemFreeHost") == 1
    calls = len(fakes.driver.calls)
    ctx.close()
    assert len(fakes.driver.calls) == calls
    with pytest.raises(IdunError, match="the CUDA context is closed"):
        ctx.alloc(8)


def test_close_frees_the_rest_when_one_free_fails(fakes: Fakes) -> None:
    ctx = loaded()
    ctx.alloc(8)
    ctx.alloc_host(8)
    ctx.sync()
    fakes.driver.fail["cuMemFree_v2"] = 1
    fakes.driver.fail["cuStreamDestroy_v2"] = 400
    with pytest.raises(IdunError, match="cuMemFree_v2 failed with CUDA_ERROR_INVALID_VALUE"):
        ctx.close()
    assert fakes.driver.live() == {"memory": 1, "streams": 1}  # what the driver refused; everything else is gone


def test_a_context_that_cannot_be_completed_is_destroyed(fakes: Fakes) -> None:
    fakes.driver.fail["cuEventCreate"] = 2
    with pytest.raises(IdunError, match="cuEventCreate failed"):
        runtime.Context()
    assert fakes.driver.live() == {}


def test_a_context_has_one_module_and_its_kernels(fakes: Fakes) -> None:
    ctx = runtime.Context()
    with pytest.raises(IdunError, match="has no module: load a source first"):
        ctx.kernel("bump")
    assert ctx.load("#warning old\n" + SOURCE) == "idun.cu(1): warning: #warning directive"
    with pytest.raises(IdunError, match="already has its module"):
        ctx.load(SOURCE)
    assert ctx.kernel("bump").name == "bump"
    with pytest.raises(IdunError, match="cuModuleGetFunction failed with CUDA_ERROR_NOT_FOUND"):
        ctx.kernel("absent")
    ctx.close()


# -- a real compiler and a real GPU -----------------------------------------------------------------------------------


def programs() -> dict[str, str]:
    """Every CUDA program the package compiles: idun_gbm's source, TabM's for each activation storage and RealMLP's
    at the default embedding sizes."""
    from idun.gbm.backends import cuda as gbm_cuda  # noqa: PLC0415
    from idun.realmlp import RealMLPParams  # noqa: PLC0415
    from idun.realmlp import cuda as realmlp_cuda  # noqa: PLC0415
    from idun.tabm import cuda as tabm_cuda  # noqa: PLC0415

    found = {"gbm.cu": gbm_cuda.KERNELS.read_text()}
    for storage in tabm_cuda.STORAGES:
        found[f"tabm, {storage}"] = tabm_cuda.source(storage)
    defaults = RealMLPParams()  # the kernels depend on the frequencies and the embedding size alone
    found["rmlp"] = realmlp_cuda.source(defaults.frequencies, defaults.embed)
    return found


def test_every_kernel_source_is_in_a_program() -> None:
    texts = programs().values()
    for source in sorted(Path(idun.__file__).parent.rglob("*.cu")):
        assert any(source.read_text() in text for text in texts), source.name


@pytest.mark.nvrtc
def test_every_kernel_source_compiles_without_a_warning() -> None:
    """Each CUDA program of the package, for the oldest GPU served, with the options that ship."""
    for name, text in programs().items():
        ptx, warnings = runtime.compile_ptx(text, "compute_75", name=name)
        assert warnings == "", name
        assert runtime.entry_points(ptx) == re.findall(r"__global__\s+void\s+(\w+)\s*\(", text), name
        binary, warnings = runtime.compile_cubin(text, "sm_75", name=name)  # what a context loads
        assert warnings == "", name
        assert binary.startswith(b"\x7fELF"), name


@pytest.mark.cuda
def test_the_gpu_is_described_and_times_its_launches() -> None:
    info = runtime.probe()
    assert info.name
    assert info.compute_capability >= (7, 5)
    assert info.max_shared_memory_bytes >= 64 * 1024
    assert runtime.architecture(info.compute_capability).startswith("compute_")
    assert runtime.binary_architecture(info.compute_capability).startswith("sm_")
    with loaded() as ctx:
        x = ctx.alloc(4 * N)
        ctx.launch(ctx.kernel("bump"), [x], None, GRID, BLOCK)
        assert ctx.sync() > 0
        assert ctx.gpu_seconds > 0
