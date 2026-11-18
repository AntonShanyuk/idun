import os
import subprocess
import sys
import threading

import numpy as np
from numba import prange
from numpy.typing import NDArray

from idun import _jit
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
    assert getattr(getattr(_double, "__wrapped__", None), "targetoptions", {})["parallel"] is True


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


def test_parallel_functions_take_turns() -> None:
    x = np.arange(1000, dtype=np.int64)
    out = np.zeros_like(x)
    with _jit._PARALLEL:
        other = threading.Thread(target=_double, args=(x, out))
        other.start()
        other.join(timeout=0.5)
        assert other.is_alive()
    other.join()
    np.testing.assert_array_equal(out, 2 * x)


# Three threads drawing at once, as the fits of a seed bag side by side do; numba's workqueue layer aborts the process
# (exit -6) when two of them run a parallel loop at the same time.
SIDE_BY_SIDE = """
import threading
from idun import rng
rng.uniform_bits(0, 1000)
start = threading.Barrier(3)
def draw():
    start.wait()
    for key in range(2000):
        rng.uniform_bits(key, 1000)
threads = [threading.Thread(target=draw) for _ in range(3)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
"""


def test_threads_running_parallel_loops_survive_numbas_workqueue_layer() -> None:
    env = {**os.environ, "NUMBA_THREADING_LAYER": "workqueue"}
    done = subprocess.run(
        [sys.executable, "-c", SIDE_BY_SIDE], env=env, capture_output=True, text=True, timeout=300, check=False
    )
    assert done.returncode == 0, done.stderr
