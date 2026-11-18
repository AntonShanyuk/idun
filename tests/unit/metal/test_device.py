"""The Metal device: buffers, lazily committed passes and the host's access to shared memory."""

import ctypes
import dataclasses
import mmap
from pathlib import Path

import numpy as np
import pytest

from idun.device import Signature
from idun.errors import BackendUnavailableError
from idun.metal import runtime
from idun.metal.device import PAGE, MetalDevice

pytestmark = pytest.mark.metal


class Params(ctypes.Structure):
    _fields_ = (("n", ctypes.c_uint32), ("add", ctypes.c_uint32))


SOURCE = """
#include <metal_stdlib>
using namespace metal;
struct Params { uint n; uint add; };
kernel void add_to(device const uint *x [[buffer(0)]], device uint *out [[buffer(1)]],
                   constant Params &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.n) { out[i] = x[i] + p.add; }
}
kernel void bump(device uint *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1u; }
"""
TABLE = {"add_to": Signature(2, (1,), Params), "bump": Signature(1, (0,), None)}
N = 64
GRID, BLOCK = (1, 1, 1), (N, 1, 1)


def make(*, profile: bool = False) -> MetalDevice:
    return MetalDevice(SOURCE, TABLE, profile=profile)


def pending(device: MetalDevice) -> bool:
    """Whether a command batch is open: passes are encoded and not yet committed."""
    return device.ctx.is_open


def test_a_pass_runs_when_its_output_is_read() -> None:
    device = make()
    x, out = device.copy_in(np.arange(N, dtype=np.uint32)), device.alloc(4 * N)
    assert out.nbytes >= 4 * N
    device.encode("add_to", [x, out], Params(N, 5), GRID, BLOCK)
    assert pending(device)
    assert not device.read(x, np.uint32, (N,)).flags.writeable  # an input of the pending pass: no commit
    assert pending(device)
    values = device.read(out, np.uint32, (N,))
    assert not pending(device)
    np.testing.assert_array_equal(values, np.arange(N) + 5)
    with pytest.raises(ValueError, match="read-only"):
        values[0] = 1
    assert device.gpu_seconds > 0
    assert device.kernel_seconds == {}
    device.close()


def test_write_and_fill_commit_a_pending_pass_that_uses_the_buffer() -> None:
    device = make()
    x, out, other = device.copy_in(np.zeros(N, np.uint32)), device.alloc(4 * N), device.alloc(4 * N)
    device.encode("add_to", [x, out], Params(N, 2), GRID, BLOCK)
    device.write(other, np.uint32, (N,))[:] = 9  # not used by the pending pass: no commit
    assert pending(device)
    device.write(x, np.uint32, (N,))[:] = 7  # the pending pass read the zeros first
    assert not pending(device)
    np.testing.assert_array_equal(device.read(out, np.uint32, (N,)), 2)
    device.encode("add_to", [x, out], Params(N, 2), GRID, BLOCK)
    device.fill(x, 1, 4 * N)
    assert not pending(device)
    np.testing.assert_array_equal(device.read(out, np.uint32, (N,)), 9)
    np.testing.assert_array_equal(device.read(x, np.uint32, (N,)), 0x01010101)
    device.encode("bump", [x], None, GRID, BLOCK)
    device.flush()
    device.flush()  # nothing open: a no-op
    np.testing.assert_array_equal(device.read(x, np.uint32, (N,)), 0x01010102)
    device.close()


def test_profile_commits_every_pass_and_times_it_by_kernel() -> None:
    device = make(profile=True)
    x, out = device.copy_in(np.arange(N, dtype=np.uint32)), device.alloc(4 * N)
    device.encode("bump", [x], None, GRID, BLOCK)  # stays out of the next pass's time: committed on its own
    for _ in range(2):
        device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)
        assert not pending(device)
    assert set(device.kernel_seconds) == {"bump", "add_to"}
    assert all(seconds > 0 for seconds in device.kernel_seconds.values())
    np.testing.assert_array_equal(device.read(out, np.uint32, (N,)), np.arange(N) + 2)
    device.close()


def test_calls_that_do_not_match_the_table_or_the_buffers_raise() -> None:
    device, stranger = make(), make()
    x, out = device.alloc(4 * N), device.alloc(4 * N)
    with pytest.raises(ValueError, match="kernel 'blur' is not in the kernel table"):
        device.encode("blur", [x], None, GRID, BLOCK)
    with pytest.raises(ValueError, match="kernel add_to binds 2 buffers, got 1"):
        device.encode("add_to", [x], Params(N, 1), GRID, BLOCK)
    with pytest.raises(ValueError, match="kernel bump takes NoneType parameters, got Params"):
        device.encode("bump", [x], Params(N, 1), GRID, BLOCK)
    with pytest.raises(TypeError, match="a Metal device takes its own buffers, got str"):
        device.encode("bump", ["x"], None, GRID, BLOCK)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="got int"):
        device.read(3, np.uint32, (N,))  # type: ignore[arg-type]
    for access in (device.read, device.write):
        with pytest.raises(ValueError, match=r"needs \d+ bytes; the buffer has"):
            access(out, np.uint32, (out.nbytes,))
    assert not pending(device)
    device.close()
    stranger.close()


def test_place_wraps_page_aligned_maps_and_copies_everything_else(tmp_path: Path) -> None:
    device = make()
    path = tmp_path / "codes.u1"
    np.arange(2 * PAGE, dtype=np.uint8).tofile(path)
    mapped = np.memmap(path, np.uint8, "r", shape=(2, PAGE))
    wrapped, stride = device.place("mapped", mapped[:, :100])
    assert stride == PAGE
    np.testing.assert_array_equal(wrapped.view(np.uint8, (2, PAGE))[:, :100], mapped[:, :100])
    rows = np.arange(60, dtype=np.uint8).reshape(3, 20)
    copied, stride = device.place("rows", rows)
    assert stride == 48  # 20 bytes padded to 32, and 16 readable bytes past the end
    np.testing.assert_array_equal(copied.view(np.uint8, (3, 48))[:, :20], rows)
    assert not copied.view(np.uint8, (3, 48))[:, 20:].any()
    flat, stride = device.place("flat", np.arange(5, dtype=np.uint16))
    assert stride == 0
    np.testing.assert_array_equal(flat.view(np.uint16, (5,)), np.arange(5))
    floats_path = tmp_path / "values.f32"
    np.arange(2 * PAGE, dtype=np.float32).tofile(floats_path)
    floats = np.memmap(floats_path, np.float32, "r", shape=(4, PAGE // 2))
    wrapped_floats, stride = device.place("floats", floats)  # any element type: wrapped, stride in bytes
    assert stride == 2 * PAGE
    np.testing.assert_array_equal(wrapped_floats.view(np.float32, (4, PAGE // 2)), floats)
    anonymous = mmap.mmap(-1, 2 * PAGE)
    labels = np.frombuffer(anonymous, np.uint8, count=3 * 100).reshape(100, 3)  # anonymous memory, as class arrays
    labels[:] = 7
    wrapped_labels, stride = device.place("labels", labels)
    assert stride == 3
    np.testing.assert_array_equal(wrapped_labels.view(np.uint8, (100, 3)), 7)
    offset, _ = device.place("offset", mapped[:, 1:100])  # not at the start of the map: copied
    np.testing.assert_array_equal(offset.view(np.uint8, (2, 128))[:, :99], mapped[:, 1:100])
    empty, stride = device.place("empty", np.zeros((0, 4), np.uint8))
    assert (empty.nbytes, stride) == (PAGE, 0)
    assert device.placements == {
        "mapped": "wrapped",
        "rows": "copied",
        "flat": "copied",
        "floats": "wrapped",
        "labels": "wrapped",
        "offset": "copied",
        "empty": "empty",
    }
    np.testing.assert_array_equal(
        device.read(device.copy_in(np.arange(12.0).reshape(3, 4)[:, ::2]), np.float64, (3, 2)),
        [[0, 2], [4, 6], [8, 10]],
    )
    device.close()


def test_submitted_passes_run_while_the_host_goes_on_and_are_waited_for_as_needed() -> None:
    device = make()
    x = device.copy_in(np.arange(N, dtype=np.uint32))
    first, second, other = device.alloc(4 * N), device.alloc(4 * N), device.alloc(4 * N)
    device.encode("add_to", [x, first], Params(N, 1), GRID, BLOCK)
    device.submit()  # committed, not waited for
    assert not pending(device)
    device.encode("add_to", [first, second], Params(N, 10), GRID, BLOCK)  # runs after the first, on the GPU
    device.submit()
    assert len(device._flight) == 2
    np.testing.assert_array_equal(device.read(first, np.uint32, (N,)), np.arange(N) + 1)
    assert len(device._flight) == 1  # waited for the batch that writes it, not the one after
    device.write(other, np.uint32, (N,))[:] = 7  # no submitted batch uses it: no wait
    assert len(device._flight) == 1
    device.write(x, np.uint32, (N,))[:] = 0  # the first batch used it, and it is done
    assert len(device._flight) == 1
    np.testing.assert_array_equal(device.read(second, np.uint32, (N,)), np.arange(N) + 11)
    assert device._flight == []
    device.encode("bump", [other], None, GRID, BLOCK)
    device.submit()
    device.write(other, np.uint32, (N,))  # a submitted batch uses it: waited for first
    assert device._flight == []
    device.encode("bump", [other], None, GRID, BLOCK)
    device.submit()
    device.submit()  # nothing open: nothing to commit
    device.flush()
    assert device._flight == []
    np.testing.assert_array_equal(device.read(other, np.uint32, (N,)), 9)
    device.close()


def test_a_profiling_device_submits_nothing_more() -> None:
    device = make(profile=True)
    x, out = device.copy_in(np.arange(N, dtype=np.uint32)), device.alloc(4 * N)
    device.encode("add_to", [x, out], Params(N, 2), GRID, BLOCK)
    device.submit()
    assert device._flight == []
    np.testing.assert_array_equal(device.read(out, np.uint32, (N,)), np.arange(N) + 2)
    device.close()


def test_a_gpu_older_than_apple7_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    old = dataclasses.replace(runtime.probe(), supports_apple7=False)
    monkeypatch.setattr(runtime, "probe", lambda: old)
    with pytest.raises(BackendUnavailableError, match="Apple7-family GPU or newer"):
        make()
