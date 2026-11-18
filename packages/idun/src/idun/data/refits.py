"""`Refit`: one walk-forward window of a date-sorted packed dataset.

A packed dataset's manifest lists its windows, and each binning is fitted on one window's training rows. Which dates
the windows start on is the caller's choice (`idun_numerai.refits` builds them on Numerai's eras).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Refit:
    """One refit: train on rows ``[0, train_hi)``, predict rows ``[pred_lo, pred_hi)`` of the date-sorted panel."""

    date: str
    train_hi: int
    pred_lo: int
    pred_hi: int
