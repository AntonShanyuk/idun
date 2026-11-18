import os

import numpy as np
from numba import prange
from numpy.typing import NDArray

from idun._jit import njit


@njit()
def _muladd(a: NDArray[np.float32], b: NDArray[np.float32], c: NDArray[np.float32], out: NDArray[np.float32]) -> None:
    for i in range(a.size):
        out[i] = a[i] * b[i] + c[i]


@njit(parallel=True)
def _double(x: NDArray[np.int64], out: NDArray[np.int64]) -> None:
    for i in prange(x.size):
        out[i] = 2 * x[i]


def test_fastmath_is_off_and_options_are_pinned() -> None:
    options = getattr(_muladd, "targetoptions", None)
    if options is None:  # the coverage run: numba hands back the Python function
        assert os.environ.get("NUMBA_DISABLE_JIT") == "1"
        return
    assert options["fastmath"] is False
    assert options["nogil"] is True
    assert getattr(_double, "targetoptions", {})["parallel"] is True


def test_no_fused_multiply_add() -> None:
    # determinism.md: CPU float math must round a * b, then round + c, exactly as the Metal kernels do (contract off).
    rng = np.random.default_rng(1)
    a, b, c = (rng.standard_normal(1 << 16).astype(np.float32) for _ in range(3))
    out = np.empty_like(a)
    _muladd(a, b, c, out)
    np.testing.assert_array_equal(out, (a * b).astype(np.float32) + c)


def test_parallel_loop() -> None:
    x = np.arange(1000, dtype=np.int64)
    out = np.empty_like(x)
    _double(x, out)
    np.testing.assert_array_equal(out, 2 * x)
