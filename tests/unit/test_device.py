"""The device protocol's helpers: kernel signatures and array extents."""

import ctypes

import numpy as np
import pytest

from idun.device import Signature, extent, signature_of


class Params(ctypes.Structure):
    _fields_ = (("n", ctypes.c_uint32),)


class Other(ctypes.Structure):
    _fields_ = (("n", ctypes.c_uint32),)


TABLE = {"scale": Signature(2, (1,), Params), "clear": Signature(1, (0,), None)}


def test_a_signature_names_its_written_buffers_by_index() -> None:
    assert Signature(3, (0, 2), Params).writes == (0, 2)
    assert Signature(0, (), None).buffers == 0
    for writes in ((3,), (-1,)):
        with pytest.raises(ValueError, match="not indices of 3 buffers"):
            Signature(3, writes, None)
    with pytest.raises(ValueError, match="not indices of -1 buffers"):
        Signature(-1, (), None)
    with pytest.raises(ValueError, match="name a buffer twice"):
        Signature(3, (1, 1), None)


def test_signature_of_checks_a_call_against_the_table() -> None:
    assert signature_of(TABLE, "scale", 2, Params(3)) is TABLE["scale"]
    assert signature_of(TABLE, "clear", 1, None) is TABLE["clear"]
    with pytest.raises(ValueError, match="kernel 'blur' is not in the kernel table"):
        signature_of(TABLE, "blur", 1, None)
    with pytest.raises(ValueError, match="kernel scale binds 2 buffers, got 3"):
        signature_of(TABLE, "scale", 3, Params(3))
    with pytest.raises(ValueError, match="kernel scale takes Params parameters, got Other"):
        signature_of(TABLE, "scale", 2, Other(3))
    with pytest.raises(ValueError, match="kernel scale takes Params parameters, got NoneType"):
        signature_of(TABLE, "scale", 2, None)
    with pytest.raises(ValueError, match="kernel clear takes NoneType parameters, got Params"):
        signature_of(TABLE, "clear", 1, Params(3))


def test_extent_is_the_arrays_bytes_and_refuses_a_smaller_buffer() -> None:
    assert extent(np.float32, (3, 5), 60) == 60
    assert extent(np.uint8, (7,), 4096) == 7
    assert extent(np.int64, (0,), 1) == 0
    with pytest.raises(ValueError, match=r"a \(3, 5\) float32 array needs 60 bytes; the buffer has 59"):
        extent(np.float32, (3, 5), 59)
