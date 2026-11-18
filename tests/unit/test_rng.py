import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from idun import rng

U32 = st.integers(min_value=0, max_value=rng.U32_MAX)

# Known answers from an independent C implementation of lowbias32 (uint32_t arithmetic).
LOWBIAS32_KAT = [
    (0x00000000, 0x00000000),
    (0x00000001, 0x688990C0),
    (0x00000002, 0xD1132181),
    (0x7FFFFFFF, 0x8D29FFB8),
    (0x80000000, 0xCC4B4124),
    (0xFFFFFFFF, 0x6768824A),
    (0x075BCD15, 0xA8F1DB88),
    (0x006A01F4, 0xED0A7DFD),
]
# (seed, iteration, key, bits for rows 0, 1, 2 and 6_947_316), same C program.
SCHEDULE_KAT = [
    (0, 0, 0xAE6F80F1, (0x944FB5, 0xB2FCF0, 0xFDD0CA, 0xA892C9)),
    (0, 1, 0xA07C7A97, (0xC67C68, 0x8BEAEB, 0x26CC07, 0x7128B7)),
    (1, 0, 0x8E374FE0, (0xB98A61, 0xC0951F, 0xDBB74C, 0x622C98)),
    (42, 599, 0x8F0A117C, (0xA934F5, 0x47C75D, 0x14FCA7, 0x766D31)),
    (rng.U32_MAX, rng.U32_MAX, 0x8FAD8C62, (0x1F0BAA, 0x8B2229, 0x896672, 0xF2B976)),
]
KAT_ROWS = (0, 1, 2, 6_947_316)


def lowbias32_oracle(x: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Vectorized NumPy lowbias32 in native uint32 arithmetic: an implementation independent of idun's."""
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x7FEB352D)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x846CA68B)
        return x ^ (x >> np.uint32(16))


@pytest.mark.parametrize(("x", "expected"), LOWBIAS32_KAT)
def test_lowbias32_known_answers(x: int, expected: int) -> None:
    assert rng.lowbias32(x) == expected


@pytest.mark.parametrize(("seed", "iteration", "key", "bits"), SCHEDULE_KAT)
def test_schedule_known_answers(seed: int, iteration: int, key: int, bits: tuple[int, ...]) -> None:
    assert rng.iteration_key(seed, iteration) == key
    assert rng.uniform_bits(key, 3).tolist() == list(bits[:3])
    # the last row (6,947,316: the panel's last index) through the scalar path; uniform_bits shares the formula
    assert [rng.lowbias32(key ^ row) >> (32 - rng.UNIFORM_BITS) for row in KAT_ROWS] == list(bits)


@given(st.lists(U32, min_size=1, max_size=64))
def test_lowbias32_matches_numpy_oracle(xs: list[int]) -> None:
    expected = lowbias32_oracle(np.array(xs, dtype=np.uint32))
    assert [rng.lowbias32(x) for x in xs] == expected.tolist()


@given(key=U32, n_rows=st.integers(min_value=0, max_value=2_000))
def test_uniform_bits_matches_numpy_oracle(key: int, n_rows: int) -> None:
    rows = np.arange(n_rows, dtype=np.uint32)
    expected = lowbias32_oracle(np.uint32(key) ^ rows) >> np.uint32(8)
    draws = rng.uniform_bits(key, n_rows)
    assert draws.dtype == np.uint32
    np.testing.assert_array_equal(draws, expected)


def test_uniform_bits_are_exact_float32_uniforms() -> None:
    draws = rng.uniform_bits(rng.iteration_key(7, 3), 1 << 16)  # small: this also runs as pure Python
    assert int(draws.max()) < 1 << rng.UNIFORM_BITS
    u = draws.astype(np.float32) * np.float32(2.0**-rng.UNIFORM_BITS)
    np.testing.assert_array_equal(u.astype(np.float64), draws.astype(np.float64) * 2.0**-rng.UNIFORM_BITS)
    assert abs(float(u.mean()) - 0.5) < 5e-3
    counts = np.bincount((u * 16).astype(np.int64), minlength=16)
    assert counts.min() > 0.95 * len(u) / 16


def test_streams_differ_across_seeds_and_iterations() -> None:
    keys = {rng.iteration_key(seed, it) for seed in range(8) for it in range(64)}
    assert len(keys) == 8 * 64


@pytest.mark.parametrize("bad", [-1, rng.U32_MAX + 1])
def test_out_of_range_inputs_raise(bad: int) -> None:
    with pytest.raises(ValueError, match="uint32"):
        rng.lowbias32(bad)
    with pytest.raises(ValueError, match="seed"):
        rng.iteration_key(bad, 0)
    with pytest.raises(ValueError, match="iteration"):
        rng.iteration_key(0, bad)
    with pytest.raises(ValueError, match="key"):
        rng.uniform_bits(bad, 1)


def test_row_count_is_bounded() -> None:
    with pytest.raises(ValueError, match="n_rows"):
        rng.uniform_bits(0, -1)
    with pytest.raises(ValueError, match="n_rows"):
        rng.uniform_bits(0, rng.MAX_ROWS + 1)


def test_row_bits_is_the_per_row_form_of_uniform_bits() -> None:
    key = rng.iteration_key(42, 599)
    expected = rng.uniform_bits(key, 5)
    assert [int(rng.row_bits(np.uint64(key), np.uint64(row))) for row in range(5)] == expected.tolist()


def test_derive_key_chains_counters_and_extends_iteration_key() -> None:
    assert rng.derive_key(7, 3) == rng.iteration_key(7, 3)
    assert rng.derive_key(7) == rng.lowbias32(7 ^ rng.SEED_SALT)
    assert rng.derive_key(7, 3, 9) == rng.lowbias32(rng.iteration_key(7, 3) ^ 9)
    with pytest.raises(ValueError, match="counter"):
        rng.derive_key(7, -1)


def test_element_bits_hash_the_row_then_the_element() -> None:
    key = rng.derive_key(1, 2, 3)
    rows = np.array([0, 5, 2**32 - 1], np.int64)
    bits = rng.element_bits(key, rows, 4)
    assert bits.shape == (3, 4)
    assert bits.dtype == np.uint32
    for i, row in enumerate(rows):
        for e in range(4):
            assert bits[i, e] == rng.lowbias32(rng.lowbias32(key ^ int(row)) ^ e) >> 8
    np.testing.assert_array_equal(rng.element_bits(key, rows[::-1], 4), bits[::-1])
    with pytest.raises(ValueError, match="uint32 row indices"):
        rng.element_bits(key, np.array([-1]), 2)
    with pytest.raises(ValueError, match="n_elements"):
        rng.element_bits(key, rows, -1)
