import ctypes
import dataclasses
import mmap
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from idun.errors import BackendUnavailableError, IdunError, KernelCompileError
from idun.metal import runtime

pytestmark = pytest.mark.metal

GOOD_KERNEL = """
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device uint* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1u; }
"""


@pytest.fixture
def fresh_library() -> Iterator[None]:
    """Forget the cached library handle around tests that change how it is located."""
    runtime.load_library.cache_clear()
    yield
    runtime.load_library.cache_clear()


def test_device_meets_the_metal_backends_requirements() -> None:
    # the Metal backends need an Apple7-family GPU (M1) or newer
    info = runtime.probe()
    assert info.name
    assert info.has_unified_memory
    assert info.supports_apple7
    assert info.max_threadgroup_memory_bytes >= 32 * 1024
    assert info.max_threads_per_threadgroup >= 1024
    assert info.page_size_bytes == 16 * 1024
    assert info.max_buffer_bytes >= 4 * 2**30  # the obfuscated codes (3.5 GB) are one wrapped buffer


def test_prelude_turns_fp_contraction_off() -> None:
    assert runtime.source_prelude().startswith("#pragma METAL fp contract(off)")


def test_valid_kernel_compiles() -> None:
    runtime.compile_check(GOOD_KERNEL)


def test_compile_error_raises_with_compiler_log() -> None:
    with pytest.raises(KernelCompileError, match="undeclared_symbol"):
        runtime.compile_check(GOOD_KERNEL.replace("x[i] += 1u;", "x[i] += undeclared_symbol;"))


@pytest.mark.usefixtures("fresh_library")
def test_missing_library_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(runtime.LIBRARY_ENV, str(tmp_path / "missing.dylib"))
    with pytest.raises(BackendUnavailableError, match=r"no Metal library \(the pure py3-none-any wheel.*make native"):
        runtime.probe()


@pytest.mark.usefixtures("fresh_library")
def test_a_library_that_cannot_load_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    unloadable = tmp_path / "libidun_metal.dylib"
    unloadable.write_bytes(b"not a Mach-O file")  # as dyld refuses a library built for a newer macOS
    monkeypatch.setenv(runtime.LIBRARY_ENV, str(unloadable))
    with pytest.raises(BackendUnavailableError, match="cannot be loaded: the Metal backend needs Apple silicon"):
        runtime.load_library()


@pytest.mark.usefixtures("fresh_library")
def test_wrong_platform_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(BackendUnavailableError, match="needs macOS"):
        runtime.load_library()


def test_default_library_path_is_inside_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.LIBRARY_ENV, raising=False)
    path = runtime.library_path()
    assert path.name == runtime.LIBRARY_NAME
    assert path.parent == Path(runtime.__file__).resolve().parent.parent / "_native"


def test_abi_mismatch_raises() -> None:
    runtime.check_abi(runtime.ABI_VERSION, Path("lib.dylib"))
    with pytest.raises(BackendUnavailableError, match="ABI 99"):
        runtime.check_abi(99, Path("lib.dylib"))


@pytest.mark.parametrize(
    ("code", "error"), [(-1, BackendUnavailableError), (-2, KernelCompileError), (-3, IdunError), (-42, IdunError)]
)
def test_error_codes_map_to_exceptions(code: int, error: type[IdunError]) -> None:
    raised = runtime.error_for(code, "detail")
    assert type(raised) is error
    assert "detail" in str(raised)


def test_c_abi_rejects_null_arguments_and_accepts_a_null_error_buffer() -> None:
    lib = runtime.load_library()
    err = ctypes.create_string_buffer(256)
    assert lib.idun_probe(None, err, 256) == -3
    assert b"out is NULL" in err.value
    assert lib.idun_compile_check(None, err, 256) == -3
    assert b"source is NULL" in err.value
    assert lib.idun_compile_check(b"not msl at all", None, 0) == -2  # err may be NULL when err_len is 0


ADD_KERNEL = """
#include <metal_stdlib>
using namespace metal;
struct P { uint n; float k; };
kernel void scale_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], constant P& p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) { if (i < p.n) { x[i] = x[i] * p.k + y[i]; } }
"""


class Scale(ctypes.Structure):
    _fields_ = (("n", ctypes.c_uint32), ("k", ctypes.c_float))


@pytest.fixture
def ctx() -> Iterator[runtime.Context]:
    with runtime.Context() as context:
        context.load(ADD_KERNEL)
        yield context


def test_a_batch_runs_dispatches_in_order(ctx: runtime.Context) -> None:
    kernel = ctx.kernel("scale_add")
    assert kernel.max_threads >= 256
    assert kernel.simd_width == 32
    x, y = ctx.alloc(4 * 1000), ctx.alloc(4 * 1000)
    assert not x.view(np.float32, (1000,)).any()  # allocations are zero-filled
    x.view(np.float32, (1000,))[:] = 1.0
    y.view(np.float32, (1000,))[:] = np.arange(1000, dtype=np.float32)
    with ctx.batch():
        ctx.dispatch(kernel, [x, y], Scale(1000, 2.0), (4, 1, 1), (256, 1, 1))
        ctx.dispatch(kernel, [x, y], Scale(1000, 3.0), (4, 1, 1), (256, 1, 1))  # sees the first dispatch's writes
    np.testing.assert_array_equal(x.view(np.float32, (1000,)), (2 + np.arange(1000)) * 3 + np.arange(1000))
    assert ctx.gpu_seconds > 0


def test_committed_batches_run_in_order_while_the_host_encodes(ctx: runtime.Context) -> None:
    kernel = ctx.kernel("scale_add")
    x, y = ctx.alloc(4 * 1000), ctx.alloc(4 * 1000)
    x.view(np.float32, (1000,))[:] = 1.0
    y.view(np.float32, (1000,))[:] = 1.0
    expected = np.ones(1000, np.float32)
    for k in (2.0, 3.0, 5.0):
        ctx.begin()
        ctx.dispatch(kernel, [x, y], Scale(1000, k), (4, 1, 1), (256, 1, 1))  # reads the previous batch's writes
        ctx.commit_async()
        assert ctx.wait(keep=1) >= 0.0  # leaves the newest batch running
        expected = expected * np.float32(k) + np.float32(1.0)
    assert ctx.wait() > 0  # the last one
    assert ctx.wait() == 0.0  # nothing left
    np.testing.assert_array_equal(x.view(np.float32, (1000,)), expected)
    ctx.begin()
    ctx.dispatch(kernel, [x, y], Scale(1000, 2.0), (4, 1, 1), (256, 1, 1))
    ctx.commit_async()
    ctx.begin()
    ctx.dispatch(kernel, [x, y], Scale(1000, 3.0), (4, 1, 1), (256, 1, 1))
    ctx.commit()  # waits for both
    np.testing.assert_array_equal(x.view(np.float32, (1000,)), (expected * 2 + 1) * 3 + 1)
    with pytest.raises(IdunError, match="no open batch"):
        ctx.commit_async()
    with pytest.raises(ValueError, match="at least 0"):
        ctx.wait(-1)


def test_closing_waits_for_committed_batches() -> None:
    n = 4 * 2**20
    with runtime.Context() as context:
        context.load(ADD_KERNEL)
        x, y = context.alloc(4 * n), context.alloc(4 * n)
        for _ in range(4):
            context.begin()
            context.dispatch(context.kernel("scale_add"), [x, y], Scale(n, 2.0), (n // 256, 1, 1), (256, 1, 1))
            context.commit_async()
    # closed without wait: the destroy waited, so nothing ran on freed buffers (the validation layer would object)


def test_wrapped_memory_is_shared_without_a_copy(ctx: runtime.Context) -> None:
    page = runtime.probe().page_size_bytes
    region = mmap.mmap(-1, 2 * page)
    host = np.frombuffer(region, np.float32)
    host[:] = 1.0
    x = ctx.wrap(host, 2 * page)
    y = ctx.alloc(len(region))
    with ctx.batch():
        ctx.dispatch(ctx.kernel("scale_add"), [x, y], Scale(host.size, 5.0), (host.size // 256, 1, 1), (256, 1, 1))
    assert (host == 5.0).all()  # the GPU wrote into the host's own pages
    with pytest.raises(IdunError, match="page-aligned"):
        ctx.wrap(host[1:], page)


def test_misuse_raises(ctx: runtime.Context) -> None:
    kernel = ctx.kernel("scale_add")
    x = ctx.alloc(64)
    with pytest.raises(IdunError, match="no kernel function"):
        ctx.kernel("missing")
    with pytest.raises(IdunError, match="no open batch"):
        ctx.dispatch(kernel, [x, x], Scale(1, 1.0), (1, 1, 1), (1, 1, 1))
    with pytest.raises(IdunError, match="no open batch"):
        ctx.commit()
    ctx.begin()
    with pytest.raises(IdunError, match="already open"):
        ctx.begin()
    with pytest.raises(IdunError, match="maximum"):
        ctx.dispatch(kernel, [x, x], Scale(1, 1.0), (1, 1, 1), (4096, 1, 1))
    with pytest.raises(ValueError, match="at most"):
        ctx.dispatch(kernel, [x] * 31, None, (1, 1, 1), (1, 1, 1))
    ctx.commit()
    with pytest.raises(ValueError, match="needs 128 bytes"):
        x.view(np.float32, (32,))
    with pytest.raises(KernelCompileError, match="undeclared"):
        ctx.load("kernel void k(device uint* x [[buffer(0)]]) { x[0] = undeclared; }")


class _RusageInfo(ctypes.Structure):  # struct rusage_info_v0, <sys/resource.h>
    _fields_ = (("uuid", ctypes.c_uint8 * 16), *((name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime")))  # fmt: skip


def _footprint() -> int:
    info = _RusageInfo()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    assert libc.proc_pid_rusage(ctypes.c_int(os.getpid()), ctypes.c_int(0), ctypes.byref(info)) == 0
    return int(info.phys_footprint)


def test_a_closed_context_returns_the_memory_its_batches_used() -> None:
    """Command buffers retain the buffers they bind; they must not outlive their batch (an undrained autorelease pool
    kept every fit's buffers alive in a Python process, about 1 GB per fit)."""
    n = 16 * 2**20  # 64 MB per buffer

    def one_context() -> None:
        with runtime.Context() as context:
            context.load(ADD_KERNEL)
            x, y = context.alloc(4 * n), context.alloc(4 * n)
            with context.batch():
                context.dispatch(context.kernel("scale_add"), [x, y], Scale(n, 2.0), (n // 256, 1, 1), (256, 1, 1))

    one_context()
    before = _footprint()
    for _ in range(5):
        one_context()
    assert _footprint() - before < 4 * n  # a leak keeps 2 x 64 MB per context: 640 MB here


def test_a_failing_batch_closes_the_context() -> None:
    context = runtime.Context()
    context.load(ADD_KERNEL)
    with pytest.raises(RuntimeError, match="boom"), context.batch():
        raise RuntimeError("boom")
    with pytest.raises(IdunError, match="closed"):
        context.alloc(4)
    context.close()  # idempotent


def test_c_abi_v2_rejects_bad_arguments() -> None:
    lib = runtime.load_library()
    err = ctypes.create_string_buffer(256)
    lib.idun_context_destroy(None)
    assert lib.idun_library_load(None, b"", err, 256) == -3
    assert not lib.idun_kernel_get(None, b"k", err, 256)
    assert lib.idun_kernel_max_threads(None) == 0
    assert lib.idun_kernel_simd_width(None) == 0
    contents = ctypes.c_void_p()
    assert not lib.idun_buffer_alloc(None, 4, ctypes.byref(contents), err, 256)
    assert not lib.idun_buffer_wrap(None, None, 0, err, 256)
    lib.idun_buffer_free(None)
    assert lib.idun_batch_begin(None, err, 256) == -3
    assert lib.idun_batch_commit_wait(None, None, err, 256) == -3
    assert lib.idun_batch_commit(None, err, 256) == -3
    assert b"idun_batch_commit: ctx is NULL" in err.value
    assert lib.idun_batch_wait(None, 0, None, err, 256) == -3
    uint3 = ctypes.c_uint32 * 3
    assert lib.idun_dispatch(None, None, None, 0, None, 0, 0, uint3(1, 1, 1), uint3(1, 1, 1), err, 256) == -3
    handle = lib.idun_context_create(err, 256)
    assert not lib.idun_kernel_get(handle, b"k", err, 256)
    assert b"no library" in err.value
    assert lib.idun_library_load(handle, ADD_KERNEL.encode(), err, 256) == 0
    kernel = lib.idun_kernel_get(handle, b"scale_add", err, 256)
    assert lib.idun_batch_begin(handle, err, 256) == 0
    buffers = (ctypes.c_void_p * 2)(None, None)
    assert lib.idun_dispatch(handle, kernel, buffers, 2, None, 0, 0, uint3(1, 1, 1), uint3(1, 1, 1), err, 256) == -3
    assert b"buffer is NULL" in err.value
    assert lib.idun_dispatch(handle, kernel, buffers, 2, None, 0, 8, uint3(1, 1, 1), uint3(1, 1, 1), err, 256) == -3
    assert (
        lib.idun_dispatch(handle, kernel, buffers, 0, None, 0, 1 << 20, uint3(1, 1, 1), uint3(1, 1, 1), err, 256) == -3
    )
    assert b"threadgroup memory" in err.value
    assert lib.idun_batch_commit(handle, err, 256) == 0  # an empty batch
    assert lib.idun_batch_wait(handle, 0, None, None, 0) == 0  # gpu_seconds and err may be NULL
    assert lib.idun_batch_commit_wait(handle, None, err, 256) == -4
    assert b"idun_batch_commit_wait: no open batch" in err.value
    assert lib.idun_batch_begin(handle, err, 256) == 0
    lib.idun_context_destroy(handle)  # drops the open batch


def test_jobs_side_by_side_are_capped_by_the_working_set(monkeypatch: pytest.MonkeyPatch) -> None:
    info = runtime.probe()
    monkeypatch.setattr(runtime, "probe", lambda: dataclasses.replace(info, recommended_working_set_bytes=10 * 2**30))
    assert runtime.side_by_side(4 * 2**30) == 2
    assert runtime.side_by_side(20 * 2**30) == 1  # at least one
    assert runtime.workers_side_by_side(None, 3, 4 * 2**30) == 2  # the default, capped
    assert runtime.workers_side_by_side(None, 1, 1) == 1
    assert runtime.workers_side_by_side(2, 3, 4 * 2**30) == 2
    with pytest.raises(ValueError, match="pass at most 2"):
        runtime.workers_side_by_side(3, 3, 4 * 2**30)
    with pytest.raises(ValueError, match="positive"):
        runtime.side_by_side(0)
