"""The one place idun compiles Python with numba.

Every jitted function goes through :func:`njit`, so the options that bit-exact parity with the GPU backends depends
on are set once. ``fastmath`` stays off: with it, LLVM contracts ``a * b + c`` into a fused multiply-add and the CPU
results stop matching the Metal kernels, which are compiled with contraction off (docs/architecture/determinism.md).
Compiled code is cached next to the source. Under ``NUMBA_DISABLE_JIT=1`` (the coverage run) numba hands back the
Python function itself.

Parallel functions take turns: numba's workqueue threading layer, the only one numba has without OpenMP or TBB (as on
a Mac with a plain install), aborts the process when two threads run parallel loops at once, and idun runs fits side
by side on threads (`idun.gbm.train_seeds`), each of which can run some on the host (extra trees' threshold draws, a
validation set's scores).
"""

import functools
import threading
from collections.abc import Callable
from typing import Any, cast

import numba

_PARALLEL = threading.Lock()  # held while a parallel function runs


def njit[F: Callable[..., Any]](*, parallel: bool = False) -> Callable[[F], F]:
    """Return numba's ``njit`` decorator with idun's fixed options.

    Parameters
    ----------
    parallel
        Compile ``numba.prange`` loops as parallel loops. The function then holds a process-wide lock while it runs,
        so no two threads run idun's parallel loops at once; it is called from Python, not from other jitted code.

    Returns
    -------
    Callable
        A decorator that compiles a function in nopython mode.
    """

    def decorate(func: F) -> F:
        compiled = numba.njit(cache=True, fastmath=False, nogil=True, parallel=parallel)(func)
        if not parallel:
            return cast("F", compiled)

        @functools.wraps(compiled, updated=())
        def taking_turns(*args: Any, **kwargs: Any) -> Any:
            with _PARALLEL:
                return compiled(*args, **kwargs)

        return cast("F", taking_turns)

    return decorate
