"""Lags in a panel: for each row, the row of the same group a whole number of time steps earlier.

A panel's rows each carry a time (an integer, such as days since an epoch) and a group (an integer id, such as a
ticker). Lag L of a row is the row of the same group whose time is exactly ``time − L · step``; lag 0 is the row
itself. Lags are what a sequence model reads as a row's history (`idun.tabm.Sequences`).
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

ABSENT = -1  # in a lag index: the group has no row at that time
_TIME_BITS = 32  # a lag key is group · 2^32 + (time − the first time)


def lag_index(
    time: NDArray[np.integer], group: NDArray[np.integer], lags: Sequence[int], step: int
) -> NDArray[np.int32]:
    """Return the ``(n_rows, len(lags))`` int32 index of each row's lags, `ABSENT` (−1) where there is no such row.

    Column ``i`` of row ``r`` is the row ``q`` of the same group, ``group[q] == group[r]``, at
    ``time[q] == time[r] − lags[i] · step``: a positive lag always points to an earlier time.

    Raises ``ValueError`` unless ``time`` and ``group`` are 1-D integer arrays of the same length (below 2^31 rows) with
    one row per (group, time) pair, group ids that fit in 32 bits and times that span less than 2^32, ``step`` is a
    positive int and every lag a non-negative int.
    """
    time, group = np.asarray(time), np.asarray(group)
    if time.ndim != 1 or group.shape != time.shape:
        raise ValueError(f"time and group must be 1-D arrays of one length, got shapes {time.shape} and {group.shape}")
    if not (np.issubdtype(time.dtype, np.integer) and np.issubdtype(group.dtype, np.integer)):
        raise ValueError(f"time and group must be integer arrays, got {time.dtype} and {group.dtype}")
    if time.size >= 1 << 31:
        raise ValueError(f"at most 2^31 - 1 rows, got {time.size}")
    if group.size and not (int(group.min()) >= -(1 << 31) and int(group.max()) < 1 << 31):
        raise ValueError("the group ids must fit in 32 bits")
    if not _count(step) or step < 1:
        raise ValueError(f"step must be a positive int, got {step!r}")
    if not all(_count(lag) and lag >= 0 for lag in lags):
        raise ValueError(f"lags must be non-negative ints, got {list(lags)}")
    out = np.full((time.size, len(lags)), ABSENT, np.int32)
    if time.size == 0:
        return out
    since = time.astype(np.int64) - int(time.min())
    if int(since.max()) >= 1 << _TIME_BITS:
        raise ValueError("the times must span less than 2^32")
    key = (group.astype(np.int64) << _TIME_BITS) + since
    order = np.argsort(key, kind="stable")
    ordered = key[order]
    if np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("every (group, time) pair must name one row")
    for i, lag in enumerate(lags):
        back = lag * step
        query = key - back
        at = np.minimum(np.searchsorted(ordered, query), time.size - 1)
        found = (since >= back) & (ordered[at] == query)  # since >= back: the query stays in the row's group
        out[found, i] = order[at[found]]
    return out


def _count(value: object) -> bool:
    """Whether ``value`` is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)
