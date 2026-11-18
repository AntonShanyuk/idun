import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.typing import NDArray

from idun.data import layout

CODES = hnp.arrays(np.uint8, st.integers(0, 40), elements=st.integers(0, 15))


def test_column_stride_rounds_up_to_a_page() -> None:
    page = layout.PAGE_BYTES
    assert [layout.column_stride(b) for b in (1, page - 1, page, page + 1)] == [page, page, page, 2 * page]
    # the real panel: 6,947,317 rows, two per byte
    assert layout.column_stride(layout.packed_bytes(6_947_317)) == 213 * page
    with pytest.raises(ValueError, match="positive"):
        layout.column_stride(0)


def test_packed_bytes() -> None:
    assert [layout.packed_bytes(n) for n in (0, 1, 2, 3, 6_947_317)] == [0, 1, 1, 2, 3_473_659]
    with pytest.raises(ValueError, match="non-negative"):
        layout.packed_bytes(-1)


def test_rows_pair_into_bytes_low_nibble_first() -> None:
    packed = layout.pack_nibbles(np.array([1, 2, 3, 4, 15], np.uint8))
    assert packed.tolist() == [0x21, 0x43, 0x0F]  # odd tail: the high nibble is zero


@given(CODES)
def test_pack_unpack_round_trip(codes: NDArray[np.uint8]) -> None:
    packed = layout.pack_nibbles(codes)
    assert packed.shape == (layout.packed_bytes(codes.size),)
    np.testing.assert_array_equal(layout.unpack_nibbles(packed, 0, codes.size), codes)


@given(CODES, st.data())
def test_unpack_any_row_range(codes: NDArray[np.uint8], data: st.DataObject) -> None:
    lo = data.draw(st.integers(0, codes.size))
    hi = data.draw(st.integers(lo, codes.size))
    np.testing.assert_array_equal(layout.unpack_nibbles(layout.pack_nibbles(codes), lo, hi), codes[lo:hi])


@given(hnp.arrays(np.uint8, st.tuples(st.integers(1, 4), st.integers(0, 9)), elements=st.integers(0, 15)), st.data())
def test_columns_pack_along_the_last_axis(codes: NDArray[np.uint8], data: st.DataObject) -> None:
    packed = layout.pack_nibbles(codes)
    assert packed.flags.c_contiguous
    lo = data.draw(st.integers(0, codes.shape[1]))
    for row in range(codes.shape[0]):
        np.testing.assert_array_equal(packed[row], layout.pack_nibbles(np.ascontiguousarray(codes[row])))
    np.testing.assert_array_equal(layout.unpack_nibbles(packed, lo, codes.shape[1]), codes[:, lo:])


def test_pack_rejects_wide_or_mistyped_codes() -> None:
    with pytest.raises(ValueError, match="4 bits"):
        layout.pack_nibbles(np.array([16], np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        layout.pack_nibbles(np.array([1], np.int64))


@pytest.mark.parametrize(("lo", "hi"), [(-1, 2), (3, 2), (0, 7)])
def test_unpack_rejects_rows_outside_the_bytes(lo: int, hi: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        layout.unpack_nibbles(np.zeros(3, np.uint8), lo, hi)


def test_file_names() -> None:
    assert layout.binning_name(2) == "r2"
    assert layout.named_bins_file("r2") == "named.r2.u8"
    assert layout.named_edges_file("r2") == "named.r2.edges.f32"
