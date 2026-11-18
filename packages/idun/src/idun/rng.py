"""Counter-based random draws, identical bit for bit on every backend (docs/architecture/determinism.md).

Stochastic rounding of the quantized gradients needs one uniform draw per (seed, iteration, row). A stateful device RNG
would tie the draws to thread scheduling. A counter-based hash makes each draw a pure function of its counters, so the
CPU, Metal and CUDA backends draw the same numbers and build bit-identical histograms.

The hash is ``lowbias32`` (Chris Wellons, hash-prospector, public domain). The schedule::

    key = lowbias32(lowbias32(seed ^ 0x9E3779B9) ^ iteration)  # host side, once per boosting iteration
    bits = lowbias32(key ^ row) >> 8  # per row, on the device
    u = bits * 2**-24  # exact in float32, in [0, 1)

The XOR with the golden-ratio constant moves seed 0 off lowbias32's fixed point at 0. Device kernels implement exactly
these lines; the unit tests pin known answers computed by an independent C implementation.
"""

import numpy as np
from numba import prange
from numpy.typing import NDArray

from idun._jit import njit

U32_MAX = 0xFFFF_FFFF
SEED_SALT = 0x9E37_79B9
UNIFORM_BITS = 24
MAX_ROWS = 1 << 32

# Constants are uint64 holding uint32 values: the product of two values below 2**32 fits in 64 bits, and the mask
# restores 32-bit wrap-around. numba freezes module-level numpy scalars as typed constants, so nothing widens to float.
_MASK = np.uint64(U32_MAX)
_MUL1 = np.uint64(0x7FEB_352D)
_MUL2 = np.uint64(0x846C_A68B)
_SHIFT15 = np.uint64(15)
_SHIFT16 = np.uint64(16)
_SHIFT_UNIFORM = np.uint64(32 - UNIFORM_BITS)


@njit()
def _lowbias32(x: np.uint64) -> np.uint64:
    x ^= x >> _SHIFT16
    x = (x * _MUL1) & _MASK
    x ^= x >> _SHIFT15
    x = (x * _MUL2) & _MASK
    x ^= x >> _SHIFT16
    return x


@njit()
def row_bits(key: np.uint64, row: np.uint64) -> np.uint64:
    """Return the `UNIFORM_BITS` draw of one row, ``lowbias32(key ^ row) >> 8``: the form kernels call (jitted)."""
    return _lowbias32(key ^ row) >> _SHIFT_UNIFORM


@njit(parallel=True)
def _fill_uniform_bits(key: np.uint64, out: NDArray[np.uint32]) -> None:
    for row in prange(out.size):
        out[row] = row_bits(key, np.uint64(row))


def _check_u32(name: str, value: int) -> None:
    if not 0 <= value <= U32_MAX:
        raise ValueError(f"{name} must be a uint32 (0..{U32_MAX}), got {value}")


def lowbias32(x: int) -> int:
    """Hash a 32-bit unsigned integer with lowbias32.

    Parameters
    ----------
    x
        Value in ``[0, 2**32)``.

    Returns
    -------
    int
        The hash, in ``[0, 2**32)``.
    """
    _check_u32("x", x)
    return int(_lowbias32(np.uint64(x)))


def iteration_key(seed: int, iteration: int) -> int:
    """Return the per-iteration key that the row draws of boosting iteration ``iteration`` are derived from.

    Parameters
    ----------
    seed
        The fit's seed, in ``[0, 2**32)``.
    iteration
        Zero-based boosting iteration, in ``[0, 2**32)``.

    Returns
    -------
    int
        ``lowbias32(lowbias32(seed ^ SEED_SALT) ^ iteration)``.
    """
    _check_u32("seed", seed)
    _check_u32("iteration", iteration)
    return int(_lowbias32(_lowbias32(np.uint64(seed ^ SEED_SALT)) ^ np.uint64(iteration)))


def derive_key(seed: int, *counters: int) -> int:
    """Return the key of the draws named by ``counters``, hashing them in one after the other.

    ``key = lowbias32(seed ^ SEED_SALT)``, then ``key = lowbias32(key ^ c)`` for each counter, so
    ``derive_key(seed, iteration)`` equals :func:`iteration_key`. The networks key their dropout masks by
    ``(seed, DROPOUT_SALT, step, block)``.
    """
    _check_u32("seed", seed)
    key = _lowbias32(np.uint64(seed ^ SEED_SALT))
    for counter in counters:
        _check_u32("counter", counter)
        key = _lowbias32(key ^ np.uint64(counter))
    return int(key)


@njit()
def element_bits_one(key: np.uint64, row: np.uint64, element: np.uint64) -> np.uint64:
    """Return one element's `UNIFORM_BITS` draw: ``lowbias32(lowbias32(key ^ row) ^ element) >> 8`` (jitted)."""
    return _lowbias32(_lowbias32(key ^ row) ^ element) >> _SHIFT_UNIFORM


@njit(parallel=True)
def _fill_element_bits(key: np.uint64, rows: NDArray[np.int64], out: NDArray[np.uint32]) -> None:
    for i in prange(out.shape[0]):
        for e in range(out.shape[1]):
            out[i, e] = element_bits_one(key, np.uint64(rows[i]), np.uint64(e))


def element_bits(key: int, rows: NDArray[np.int64], n_elements: int) -> NDArray[np.uint32]:
    """Draw ``UNIFORM_BITS`` bits for elements ``0 .. n_elements - 1`` of each of ``rows`` (absolute row indices).

    Returns a ``(len(rows), n_elements)`` ``uint32`` array: ``lowbias32(lowbias32(key ^ row) ^ element) >> 8``. The
    draws depend on the row's absolute index, never on its position in a batch.
    """
    _check_u32("key", key)
    rows = np.asarray(rows, np.int64)
    if rows.ndim != 1 or (rows.size and not (rows.min() >= 0 and rows.max() <= U32_MAX)):
        raise ValueError("rows must be a 1-D array of uint32 row indices")
    if not 0 <= n_elements <= U32_MAX:
        raise ValueError(f"n_elements must be in 0..{U32_MAX}, got {n_elements}")
    out = np.empty((rows.size, n_elements), np.uint32)
    _fill_element_bits(np.uint64(key), rows, out)
    return out


def uniform_bits(key: int, n_rows: int) -> NDArray[np.uint32]:
    """Draw ``UNIFORM_BITS`` random bits for each row ``0 .. n_rows - 1``.

    Parameters
    ----------
    key
        An :func:`iteration_key`.
    n_rows
        Number of rows, at most ``2**32``.

    Returns
    -------
    numpy.ndarray
        ``uint32`` array whose element ``row`` is ``lowbias32(key ^ row) >> 8``. Multiply by ``2**-24`` for a uniform in
        ``[0, 1)``; the product is exact in float32.
    """
    _check_u32("key", key)
    if not 0 <= n_rows <= MAX_ROWS:
        raise ValueError(f"n_rows must be in 0..{MAX_ROWS}, got {n_rows}")
    out = np.empty(n_rows, dtype=np.uint32)
    _fill_uniform_bits(np.uint64(key), out)
    return out
