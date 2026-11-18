"""The device-neutral passes of idun_gbm: what holds without a GPU (the kernels run in the backends' own tests)."""

import ctypes
import itertools
import re
from typing import Any

import numpy as np
import pytest

from idun.gbm.backends import device as passes
from idun.gbm.backends import metal as metal_module

METAL = metal_module.KERNELS
KERNEL = re.compile(r"kernel void (\w+)\((.*?)\)\s*\{", re.DOTALL)
BOUND = re.compile(r"(device|constant)\s+(const\s+)?([\w ]+?)\s*[*&]\s*\w+\s*\[\[buffer\((\d+)\)\]\]")


def test_param_structs_match_the_kernel_source() -> None:
    sizes = {
        passes.CountParams: 4,
        passes.RngParams: 12,
        passes.ModelParams: 8,
        passes.QuantizeParams: 16,
        passes.HistogramParams: 32,
        passes.ObfuscatedParams: 40,
        passes.SingleObfuscatedParams: 40,
        passes.Entry: 16,
        passes.PartitionParams: 24,
        passes.LeafParams: 24,
        passes.UpdateParams: 12,
        passes.MultiParams: 56,
        passes.PredictParams: 28,
        passes.LeafSplitParams: 20,
        passes.NodePredictParams: 28,
    }
    source = METAL.read_text()
    for struct, size in sizes.items():
        assert ctypes.sizeof(struct) == size
        assert f"struct {struct.__name__} {{" in source
        for field in struct._fields_:
            assert f" {field[0]};" in source, (struct.__name__, field[0])


def test_the_kernel_table_matches_the_metal_source() -> None:
    """Each kernel's buffers, the ones it may write (not ``const``) and its parameter structure, as declared."""
    declared = {}
    for name, arguments in KERNEL.findall(METAL.read_text()):
        bound = [(kind, not const, kind_type, int(index)) for kind, const, kind_type, index in BOUND.findall(arguments)]
        assert [index for *_, index in bound] == list(range(len(bound))), name
        buffers = [writable for kind, writable, _, _ in bound if kind == "device"]
        params = [kind_type for kind, _, kind_type, _ in bound if kind == "constant"]
        assert [kind for kind, *_ in bound] == ["device"] * len(buffers) + ["constant"] * len(params), name
        writes = tuple(index for index, writable in enumerate(buffers) if writable)
        declared[name] = (len(buffers), writes, params[0] if params else None)
    table = {
        name: (signature.buffers, signature.writes, signature.params.__name__ if signature.params else None)
        for name, signature in passes.TABLE.items()
    }
    assert table == declared
    assert len(table) == 20


def test_leaf_partials_are_added_exactly_in_int64() -> None:
    # (model, group, leaf, [hi, mid, low]): hi is a signed 32-bit sum stored as uint, mid and low unsigned 32-bit sums
    hi = np.array([[-3, 2**19], [5, -(2**19)]], np.int32)
    mid = np.array([[2**28 - 1, 0], [7, 65535]], np.uint32)
    low = np.array([[0, 2**28 - 1], [1, 2**16]], np.uint32)
    parts = np.stack([hi.view(np.uint32), mid, low], axis=-1)
    want = [sum(int(hi[g, k]) * 2**32 + int(mid[g, k]) * 2**16 + int(low[g, k]) for g in range(2)) for k in range(2)]
    np.testing.assert_array_equal(passes._leaf_total(parts[None]), [want])


def test_chunks_pack_model_slots_up_to_the_cap() -> None:
    entries, starts = passes.chunks(np.array([3, 0, 6, 2]), 4)
    assert entries == [(0, 0, 3, 0), (2, 0, 1, 3), (2, 1, 5, 0), (2, 5, 6, 0), (3, 0, 2, 1)]
    assert starts == [0, 2, 3, 5]
    for first, end in itertools.pairwise(starts):
        assert sum(stop - slot for _, slot, stop, _ in entries[first:end]) <= 4
    entries, starts = passes.chunks(np.array([3, 0, 6, 2]), 4, share=False)  # every model starts a chunk
    assert entries == [(0, 0, 3, 0), (2, 0, 4, 0), (2, 4, 6, 0), (3, 0, 2, 0)]
    assert starts == [0, 1, 2, 3, 4]
    assert passes.chunks(np.zeros(3, np.int64), 8) == ([], [0, 0])


def test_groups_cover_the_rows_and_the_blocks_are_the_kernels() -> None:
    assert passes._groups(0) == (1, 1, 1)
    assert passes._groups(256) == (1, 1, 1)
    assert passes._groups(257) == (2, 1, 1)
    assert passes._groups(65, 32) == (3, 1, 1)
    assert passes.BLOCK == (256, 1, 1)
    assert passes.LANE_BLOCK == (32, 1, 1)


class Closable:
    """A device that only counts its closes: the backends' constructors use nothing else."""

    closed = 0

    def close(self) -> None:
        self.closed += 1


def test_backends_refuse_no_models_and_close_their_device() -> None:
    device: Any = Closable()
    with pytest.raises(ValueError, match="need at least one model, got 0"):
        passes.DeviceMany(device, 0)
    for backend in (passes.DeviceMany(device, 3), passes.DeviceBackend(device), passes.DeviceMultiTarget(device)):
        before = device.closed
        backend.close()
        assert device.closed == before + 1
    assert passes.DeviceMany(device, 3).k == 3
