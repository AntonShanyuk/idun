"""The one place idun compiles Python with numba.

Every jitted function goes through :func:`njit`, so the options that bit-exact parity with the GPU backends depends
on are set once. ``fastmath`` stays off: with it, LLVM contracts ``a * b + c`` into a fused multiply-add and the CPU
results stop matching the Metal kernels, which are compiled with contraction off (docs/architecture/determinism.md).
Compiled code is cached next to the source. Under ``NUMBA_DISABLE_JIT=1`` (the coverage run) numba hands back the
Python function itself.
"""

from collections.abc import Callable
from typing import Any, cast

import numba


def njit[F: Callable[..., Any]](*, parallel: bool = False) -> Callable[[F], F]:
    """Return numba's ``njit`` decorator with idun's fixed options.

    Parameters
    ----------
    parallel
        Compile ``numba.prange`` loops as parallel loops.

    Returns
    -------
    Callable
        A decorator that compiles a function in nopython mode.
    """

    def decorate(func: F) -> F:
        return cast("F", numba.njit(cache=True, fastmath=False, nogil=True, parallel=parallel)(func))

    return decorate
