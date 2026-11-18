"""A walk-forward refit table on Numerai's eras; the dates and the purge are the caller's.

Each refit trains on every row dated before its date minus ``purge`` eras and predicts until the next refit date.
The arithmetic, on era days:

    k = eras.searchsorted(date); cutoff = eras[k - purge]
    train = [0, rows.searchsorted(cutoff)); predict = [rows.searchsorted(date), rows.searchsorted(next date) or n)
"""

import datetime as dt
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from idun.data import Refit

EPOCH = dt.date(1970, 1, 1)


def to_days(date: str) -> int:
    """Days since 1970-01-01 of an ISO date (``YYYY-MM-DD``)."""
    return (dt.date.fromisoformat(date) - EPOCH).days


def to_date(days: int) -> str:
    """ISO date of a day count since 1970-01-01."""
    return (EPOCH + dt.timedelta(days=days)).isoformat()


def refit_table(era_days: NDArray[np.int32], dates: Sequence[str], purge: int) -> tuple[Refit, ...]:
    """Row ranges of each refit on a panel whose rows are sorted by era day.

    ``dates``: the refits' ISO dates, strictly increasing; ``purge``: the eras left out between a refit's training rows
    and its date, at least the target's horizon in eras (no defaults).
    """
    if era_days.size == 0 or bool(np.any(np.diff(era_days) < 0)):
        raise ValueError("era_days must be non-empty and sorted")
    starts = [to_days(date) for date in dates]
    if starts != sorted(set(starts)):
        raise ValueError(f"refit dates must be strictly increasing, got {list(dates)}")
    if purge < 0:
        raise ValueError(f"purge must be non-negative, got {purge}")
    eras = np.unique(era_days)
    table = []
    for i, (date, start) in enumerate(zip(dates, starts, strict=True)):
        k = int(np.searchsorted(eras, start))
        if k - purge < 0:
            raise ValueError(f"refit {date}: fewer than {purge} eras before it, nothing to train on")
        cutoff = eras[k - purge]
        pred_hi = int(np.searchsorted(era_days, starts[i + 1])) if i + 1 < len(starts) else int(era_days.size)
        table.append(
            Refit(date, int(np.searchsorted(era_days, cutoff)), int(np.searchsorted(era_days, start)), pred_hi)
        )
    return tuple(table)


__all__ = ["refit_table", "to_date", "to_days"]
