"""The mirrored device: host mistakes that unified memory hides show as stale data or raise."""

import ctypes

import numpy as np
import pytest

from idun.device import Signature
from idun.metal import mirrored
from idun.metal.device import MetalDevice
from idun.metal.mirrored import HostAccessError, MirroredDevice, open_device

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


def make(table: dict[str, Signature] = TABLE) -> MirroredDevice:
    return MirroredDevice(MetalDevice(SOURCE, table), table)


def test_results_are_the_wrapped_devices() -> None:
    device = make()
    x, out = device.copy_in(np.arange(N, dtype=np.uint32)), device.alloc(4 * N)
    assert out.nbytes >= 4 * N
    device.encode("add_to", [x, out], Params(N, 5), GRID, BLOCK)
    values = device.read(out, np.uint32, (N,))
    np.testing.assert_array_equal(values, np.arange(N) + 5)
    assert not values.flags.writeable
    device.write(x, np.uint32, (N,))[:8] = 100  # the rest of the buffer keeps the device's values
    device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)
    device.flush()
    np.testing.assert_array_equal(device.read(out, np.uint32, (N,)), np.r_[np.full(8, 101), np.arange(8, N) + 1])
    device.fill(x, 0, 4 * N)
    device.encode("bump", [x], None, GRID, BLOCK)
    device.submit()  # every pass has run on its own
    np.testing.assert_array_equal(device.read(x, np.uint32, (N,)), 1)
    assert device.gpu_seconds > 0
    assert device.kernel_seconds == {}
    assert device.placements == {}
    device.close()


def test_an_array_read_before_a_pass_is_stale_after_it() -> None:
    device = make()
    x = device.copy_in(np.zeros(N, np.uint32))
    before = device.read(x, np.uint32, (N,))
    device.encode("bump", [x], None, GRID, BLOCK)
    device.flush()
    np.testing.assert_array_equal(before, 0)  # a missing read: on a device with its own memory the host sees this
    np.testing.assert_array_equal(device.read(x, np.uint32, (N,)), 1)
    np.testing.assert_array_equal(device.inner.read(x.inner, np.uint32, (N,)), 1)
    device.close()


def test_a_host_write_after_the_pass_that_used_the_buffer_raises() -> None:
    device = make()
    x, out = device.alloc(4 * N), device.alloc(4 * N)
    whole = device.write(x, np.uint32, (N,))
    part = whole[4:8]  # a view of the handed-out array, as a learner's per-model slice is
    whole[:] = 3
    device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)
    with pytest.raises(ValueError, match="read-only"):
        whole[0] = 4  # the array itself is frozen at the upload
    part[:] = 4  # a view made before stays writable: the write never reaches the device
    np.testing.assert_array_equal(device.inner.read(x.inner, np.uint32, (N,)), 3)
    with pytest.raises(HostAccessError, match=r"wrote an array from write\(\) after the pass that used the buffer"):
        device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)
    device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)  # reported once
    late = device.write(out, np.uint32, (N,))[:4]
    device.fill(out, 0, 4 * N)
    late[:] = 5
    with pytest.raises(HostAccessError, match="after the pass that used the buffer"):
        device.close()


def test_a_kernel_that_writes_a_buffer_declared_read_only_raises() -> None:
    wrong = TABLE | {"add_to": Signature(2, (), Params)}
    device = make(wrong)
    x, out = device.copy_in(np.arange(N, dtype=np.uint32)), device.alloc(4 * N)
    with pytest.raises(HostAccessError, match="kernel add_to wrote buffer 1, which its signature declares read-only"):
        device.encode("add_to", [x, out], Params(N, 5), GRID, BLOCK)
    device.close()


def test_a_missing_write_declaration_leaves_the_hosts_copy_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mirrored, "CHECK_BYTES", 0)  # as for a buffer too large to compare around every pass
    device = make(TABLE | {"bump": Signature(1, (), None)})
    x = device.copy_in(np.zeros(N, np.uint32))
    np.testing.assert_array_equal(device.read(x, np.uint32, (N,)), 0)
    device.encode("bump", [x], None, GRID, BLOCK)
    np.testing.assert_array_equal(device.read(x, np.uint32, (N,)), 0)  # no download: the device holds ones
    np.testing.assert_array_equal(device.inner.read(x.inner, np.uint32, (N,)), 1)
    device.close()


def test_placed_data_and_foreign_buffers_are_refused() -> None:
    device, stranger = make(), make()
    placed, stride = device.place("rows", np.arange(60, dtype=np.uint8).reshape(3, 20))
    assert stride == 48
    assert device.placements == {"rows": "copied"}
    for access in (device.read, device.write):
        with pytest.raises(HostAccessError, match=r"cannot read or write data given to place\(\)"):
            access(placed, np.uint8, (3, 48))
    with pytest.raises(HostAccessError, match=r"place\(\)"):
        device.fill(placed, 0, 16)
    device.encode("bump", [placed], None, GRID, BLOCK)  # kernels bind placed data
    theirs = stranger.alloc(4 * N)
    with pytest.raises(TypeError, match="a mirrored device takes its own buffers, got MirroredBuffer"):
        device.read(theirs, np.uint32, (N,))
    with pytest.raises(TypeError, match="got Buffer"):
        device.encode("bump", [theirs.inner], None, GRID, BLOCK)
    with pytest.raises(ValueError, match="kernel bump binds 1 buffers, got 2"):
        device.encode("bump", [placed, placed], None, GRID, BLOCK)
    with pytest.raises(ValueError, match=r"needs \d+ bytes; the buffer has"):
        device.write(device.alloc(16), np.uint32, (1 << 20,))
    device.close()
    stranger.close()


def test_open_device_follows_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(mirrored.ENV, raising=False)
    plain = open_device(SOURCE, TABLE)
    assert isinstance(plain, MetalDevice)
    assert not plain.profile
    plain.close()
    monkeypatch.setenv(mirrored.ENV, "1")
    wrapped = open_device(SOURCE, TABLE, profile=True)
    assert isinstance(wrapped, MirroredDevice)
    assert isinstance(wrapped.inner, MetalDevice)
    assert wrapped.inner.profile
    x = wrapped.copy_in(np.zeros(N, np.uint32))
    wrapped.encode("bump", [x], None, GRID, BLOCK)
    assert wrapped.kernel_seconds["bump"] > 0
    wrapped.close()
    for value in ("0", "true", ""):
        monkeypatch.setenv(mirrored.ENV, value)
        with pytest.raises(ValueError, match="IDUN_METAL_MIRRORED must be 1 or unset"):
            open_device(SOURCE, TABLE)
