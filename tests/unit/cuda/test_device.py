"""The CUDA device: buffers on the GPU, their page-locked mirrors, and what crosses between the two."""

from typing import Any

import numpy as np
import pytest

from idun.cuda import device as device_module
from idun.cuda.device import TAIL_BYTES, CudaBuffer, CudaDevice
from idun.device import Signature
from idun.errors import IdunError, KernelCompileError
from tests.unit.cuda.fake_driver import SOURCE, Fakes, Params

TABLE = {
    "add_to": Signature(2, (1,), Params),
    "bump": Signature(1, (0,), None),
    "shared_sum": Signature(1, (0,), Params),
}
N = 64
GRID, BLOCK = (1, 1, 1), (N, 1, 1)
U32 = (np.uint32, (N,))


def make(**options: Any) -> CudaDevice:
    return CudaDevice(SOURCE, TABLE, **options)


def raw(device: CudaDevice, buffer: CudaBuffer) -> np.ndarray[Any, Any]:
    """Return every byte of a buffer as the device holds it."""
    return np.array(device.read(buffer, np.uint8, (buffer.nbytes,)))


def test_a_pass_runs_and_its_output_is_read(anywhere: str) -> None:
    device = make()
    x, out = device.copy_in(np.arange(N, dtype=np.uint32)), device.alloc(4 * N)
    assert (x.nbytes, out.nbytes) == (4 * N, 4 * N)
    device.encode("add_to", [x, out], Params(N, 5), GRID, BLOCK)
    values = device.read(out, *U32)
    np.testing.assert_array_equal(values, np.arange(N) + 5)
    with pytest.raises(ValueError, match="read-only"):
        values[0] = 1
    assert device.gpu_seconds > 0
    assert device.kernel_seconds == {}
    assert device.warnings == ""
    assert (device.uploaded_bytes, device.downloaded_bytes) == (4 * N, 4 * N)
    np.testing.assert_array_equal(device.read(x, *U32), np.arange(N))  # data given to copy_in: downloaded once
    device.read(x, *U32)
    assert (device.uploaded_bytes, device.downloaded_bytes) == (4 * N, 8 * N)
    device.flush()
    device.flush()  # nothing queued: a no-op
    device.encode("add_to", [x, out], Params(N, 7), GRID, BLOCK)
    device.submit()  # the launch is on the stream already: the read synchronises it
    np.testing.assert_array_equal(device.read(out, *U32), np.arange(N) + 7)
    device.close()


def test_only_what_changed_crosses(fakes: Fakes) -> None:
    device = make()
    x, out = device.alloc(4 * N), device.alloc(4 * N)
    transfers = fakes.driver.transfers
    assert not device.read(x, *U32).any()  # a buffer still zero is known without a download
    device.write(x, np.uint32, (8,))[:] = np.arange(8)
    assert transfers == []  # nothing goes up until a pass uses the buffer
    device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)
    assert transfers == [("up", 32)]  # the eight values handed out, not the buffer
    device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)
    assert transfers == [("up", 32)]
    np.testing.assert_array_equal(device.read(out, np.uint32, (8,)), np.arange(8) + 1)
    assert transfers == [("up", 32), ("down", 32)]
    np.testing.assert_array_equal(device.read(out, *U32)[8:], 1)
    assert transfers == [("up", 32), ("down", 32), ("down", 4 * N - 32)]  # only what the mirror lacked
    device.read(out, *U32)
    np.testing.assert_array_equal(device.read(x, *U32)[:9], [0, 1, 2, 3, 4, 5, 6, 7, 0])  # an input: still current
    assert len(transfers) == 3
    assert (device.uploaded_bytes, device.downloaded_bytes) == (32, 4 * N)
    device.close()


def test_a_buffer_a_pass_wrote_is_downloaded_again(anywhere: str) -> None:
    device = make()
    x, out = device.alloc(4 * N), device.alloc(4 * N)
    for expected in (1, 2):
        device.encode("bump", [x], None, GRID, BLOCK)
        np.testing.assert_array_equal(device.read(x, *U32), expected)
    assert device.downloaded_bytes == 8 * N
    values = device.write(x, *U32)  # current values, for the host to change
    np.testing.assert_array_equal(values, 2)
    values[::2] = 10
    device.encode("bump", [x], None, GRID, BLOCK)
    device.encode("add_to", [x, out], Params(N, 5), GRID, BLOCK)
    np.testing.assert_array_equal(device.read(out, *U32).reshape(-1, 2), [[16, 8]] * (N // 2))
    np.testing.assert_array_equal(device.read(x, *U32).reshape(-1, 2), [[11, 3]] * (N // 2))
    assert (device.uploaded_bytes, device.downloaded_bytes) == (4 * N, 16 * N)
    device.close()


def test_a_host_write_waits_for_the_upload_still_queued(fakes: Fakes) -> None:
    device = make()
    x, out, other = device.alloc(4 * N), device.alloc(4 * N), device.alloc(4 * N)
    device.write(x, *U32)[:] = 7
    device.encode("add_to", [x, out], Params(N, 2), GRID, BLOCK)
    waits = fakes.driver.count("cuStreamSynchronize")
    device.write(other, *U32)[:] = 1  # another buffer: no wait
    assert fakes.driver.count("cuStreamSynchronize") == waits
    device.write(x, *U32)[:] = 100  # the queued upload has not read the mirror yet
    assert fakes.driver.count("cuStreamSynchronize") == waits + 1
    np.testing.assert_array_equal(device.read(out, *U32), 9)
    device.encode("add_to", [x, out], Params(N, 2), GRID, BLOCK)
    np.testing.assert_array_equal(device.read(out, *U32), 102)
    device.close()


def test_fill_runs_on_the_device_in_order(anywhere: str) -> None:
    device = make()
    x, out = device.alloc(4 * N), device.alloc(4 * N)
    device.write(x, *U32)[:] = 5
    device.encode("add_to", [x, out], Params(N, 2), GRID, BLOCK)
    device.fill(x, 1, 8)  # after the pass; the host's values past the fill stay
    np.testing.assert_array_equal(device.read(out, *U32), 7)
    np.testing.assert_array_equal(device.read(x, *U32), [0x01010101] * 2 + [5] * (N - 2))
    device.write(x, np.uint32, (4,))[:] = 9
    device.fill(x, 2, 8)  # covers part of what the host changed: the rest still goes up
    np.testing.assert_array_equal(device.read(x, np.uint32, (5,)), [0x02020202, 0x02020202, 9, 9, 5])
    device.write(x, np.uint32, (2,))[:] = 3
    device.fill(x, 4, 8)  # covers all of it
    np.testing.assert_array_equal(device.read(x, np.uint32, (3,)), [0x04040404, 0x04040404, 9])
    device.fill(x, 0, 0)
    for nbytes in (-1, 4 * N + 1):
        with pytest.raises(ValueError, match=rf"a fill of {nbytes} bytes does not fit a buffer of 256"):
            device.fill(x, 0, nbytes)
    device.close()


def test_a_zeroed_buffer_needs_no_fill_and_no_download(fakes: Fakes) -> None:
    device = make()
    x, out = device.alloc(4 * N), device.alloc(4 * N)
    fills = fakes.driver.count("cuMemsetD8Async")
    device.fill(x, 0, 4 * N)  # already zero
    assert fakes.driver.count("cuMemsetD8Async") == fills
    device.write(x, *U32)[:] = 7
    device.encode("add_to", [x, out], Params(N, 2), GRID, BLOCK)
    device.fill(x, 0, 4 * N)
    assert not device.read(x, *U32).any()  # known to be zero, and cleared only after the queued upload read it
    device.fill(x, 0, 8)
    assert fakes.driver.count("cuMemsetD8Async") == fills + 1
    assert fakes.driver.transfers == [("up", 4 * N)]
    np.testing.assert_array_equal(device.read(out, *U32), 9)
    device.encode("bump", [x], None, GRID, BLOCK)
    device.fill(x, 0, 8)  # a pass wrote the buffer since: a real fill
    np.testing.assert_array_equal(device.read(x, *U32), [0, 0] + [1] * (N - 2))
    assert fakes.driver.count("cuMemsetD8Async") == fills + 2
    device.close()


def test_place_uploads_rows_as_they_lie_or_repacks_them(anywhere: str) -> None:
    device = make()
    wide = np.arange(96, dtype=np.uint8).reshape(3, 32)
    placed, stride = device.place("wide", wide)
    assert (stride, placed.nbytes) == (32, 96 + TAIL_BYTES)
    np.testing.assert_array_equal(raw(device, placed), [*range(96), *[0] * TAIL_BYTES])
    view, stride = device.place("view", wide[:, :20])  # rows 32 bytes apart: the bytes between them go up too
    assert (stride, view.nbytes) == (32, 2 * 32 + 20 + TAIL_BYTES)
    np.testing.assert_array_equal(raw(device, view)[: 2 * 32 + 20], np.arange(2 * 32 + 20))
    assert not raw(device, view)[2 * 32 + 20 :].any()
    rows = np.arange(60, dtype=np.uint8).reshape(3, 20)
    copied, stride = device.place("rows", rows)
    assert stride == 48  # 20 bytes padded to 32, and 16 readable bytes past the end
    packed = raw(device, copied)[: 3 * 48].reshape(3, 48)
    np.testing.assert_array_equal(packed[:, :20], rows)
    assert not packed[:, 20:].any()
    flat, stride = device.place("flat", np.arange(5, dtype=np.uint16))
    assert (stride, flat.nbytes) == (0, 10 + TAIL_BYTES)
    np.testing.assert_array_equal(device.read(flat, np.uint16, (5,)), np.arange(5))
    for name, array in (("columns", wide[:, ::2]), ("reversed", wide[::-1]), ("every other", wide[::2].T)):
        buffer, stride = device.place(name, array)
        assert stride == -(-array.shape[1] // 16) * 16 + 16
        got = raw(device, buffer)[: array.shape[0] * stride].reshape(array.shape[0], stride)
        np.testing.assert_array_equal(got[:, : array.shape[1]], array)
    empty, stride = device.place("empty", np.zeros((0, 4), np.uint8))
    assert (empty.nbytes, stride) == (TAIL_BYTES, 0)
    assert not raw(device, empty).any()
    assert device.placements == {
        "wide": "uploaded",
        "view": "uploaded",
        "rows": "repacked",
        "flat": "uploaded",
        "columns": "repacked",
        "reversed": "repacked",
        "every other": "repacked",
        "empty": "empty",
    }
    np.testing.assert_array_equal(
        device.read(device.copy_in(np.arange(12.0).reshape(3, 4)[:, ::2]), np.float64, (3, 2)),
        [[0, 2], [4, 6], [8, 10]],
    )
    device.close()


def test_placed_data_goes_up_in_chunks(fakes: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "CHUNK_BYTES", 40)
    device = make()
    placed, _ = device.place("wide", np.arange(96, dtype=np.uint8).reshape(3, 32))
    assert fakes.driver.transfers == [("up", 40), ("up", 40), ("up", 16)]
    assert device.uploaded_bytes == 96
    np.testing.assert_array_equal(raw(device, placed)[:96], np.arange(96))
    device.close()


@pytest.mark.parametrize("guard", [False, True])
def test_profile_waits_for_every_pass_and_times_it_by_kernel(anywhere: str, guard: bool) -> None:
    device = make(profile=True, guard=guard)
    x, out = device.copy_in(np.arange(N, dtype=np.uint32)), device.alloc(4 * N)
    device.encode("bump", [x], None, GRID, BLOCK)
    for _ in range(2):
        device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)
    assert set(device.kernel_seconds) == {"bump", "add_to"}
    assert all(seconds > 0 for seconds in device.kernel_seconds.values())
    assert device.gpu_seconds == pytest.approx(sum(device.kernel_seconds.values()))
    np.testing.assert_array_equal(device.read(out, *U32), np.arange(N) + 2)
    device.close()


def test_profile_times_each_pass_on_its_own(fakes: Fakes) -> None:
    device = make(profile=True)
    x, out = device.alloc(4 * N), device.alloc(4 * N)
    device.encode("bump", [x], None, GRID, BLOCK)
    fakes.driver.launch_seconds = 1.0
    for _ in range(2):
        device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)
    assert device.kernel_seconds == {"bump": 0.25, "add_to": 2.0}
    assert fakes.driver.count("cuStreamSynchronize") == 4  # for the buffers' fills, then after each pass
    device.close()


def test_a_pass_can_ask_for_shared_memory(anywhere: str) -> None:
    device = make()
    out = device.alloc(4 * 3)
    device.encode("shared_sum", [out], Params(0, 2), (3, 1, 1), BLOCK, 4 * N)
    np.testing.assert_array_equal(device.read(out, np.uint32, (3,)), N * (N - 1) // 2 + 2 * N)
    device.close()


def test_calls_that_do_not_match_the_table_or_the_buffers_raise(fakes: Fakes) -> None:
    device, stranger = make(), make()
    x, out = device.alloc(4 * N), device.alloc(4 * N)
    with pytest.raises(ValueError, match="kernel 'blur' is not in the kernel table"):
        device.encode("blur", [x], None, GRID, BLOCK)
    with pytest.raises(ValueError, match="kernel add_to binds 2 buffers, got 1"):
        device.encode("add_to", [x], Params(N, 1), GRID, BLOCK)
    with pytest.raises(ValueError, match="kernel bump takes NoneType parameters, got Params"):
        device.encode("bump", [x], Params(N, 1), GRID, BLOCK)
    with pytest.raises(TypeError, match="a CUDA device takes its own buffers, got str"):
        device.encode("bump", ["x"], None, GRID, BLOCK)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="got int"):
        device.read(3, np.uint32, (N,))  # type: ignore[arg-type]
    for access in (device.read, device.write):
        with pytest.raises(ValueError, match=r"needs \d+ bytes; the buffer has"):
            access(out, np.uint32, (out.nbytes,))
    with pytest.raises(ValueError, match="not a live buffer of this context"):
        device.encode("bump", [stranger.alloc(4 * N)], None, GRID, BLOCK)
    assert fakes.driver.launches == []
    device.close()
    stranger.close()


def test_a_source_that_does_not_compile_leaves_nothing_behind(fakes: Fakes) -> None:
    with pytest.raises(KernelCompileError, match="#error directive"):
        CudaDevice("#error no\n", TABLE)
    assert fakes.driver.live() == {}
    device = CudaDevice("#warning old\n" + SOURCE, TABLE)
    assert device.warnings == "idun.cu(1): warning: #warning directive"
    device.close()


def test_close_frees_the_buffers_and_their_mirrors(fakes: Fakes) -> None:
    device = make()
    x, out = device.alloc(4 * N), device.alloc(4 * N)
    device.place("rows", np.arange(60, dtype=np.uint8).reshape(3, 20))
    device.write(x, *U32)[:] = 1
    device.encode("add_to", [x, out], Params(N, 1), GRID, BLOCK)
    device.read(out, *U32)
    device.encode("bump", [out], None, GRID, BLOCK)  # still queued at close
    assert fakes.driver.live() == {"memory": 3, "host": 2, "contexts": 1, "streams": 1, "events": 2, "modules": 1}
    device.close()
    assert fakes.driver.live() == {}
    assert fakes.driver.count("cuMemAlloc_v2") == fakes.driver.count("cuMemFree_v2") == 3
    assert fakes.driver.count("cuMemAllocHost_v2") == fakes.driver.count("cuMemFreeHost") == 2
    assert fakes.driver.count("cuModuleGetFunction") == 2  # a kernel is looked up once


def test_guard_mode_names_the_kernel_that_wrote_outside_its_buffer(anywhere: str) -> None:
    device = make(guard=True)
    x = device.alloc(4 * N)
    device.encode("bump", [x], None, GRID, BLOCK)
    np.testing.assert_array_equal(device.read(x, *U32), 1)
    with pytest.raises(IdunError, match=r"guard mode: after kernel bump, buffer 0 .* was written past its end"):
        device.encode("bump", [x], None, GRID, (N + 1, 1, 1))
    device.close()
