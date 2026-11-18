import numpy as np
import pytest

from idun.data import Refit
from idun_numerai import refits


def weekly_panel(eras: int, rows_per_era: list[int]) -> np.ndarray:
    first = refits.to_days("2003-01-03")
    return np.repeat(first + 7 * np.arange(eras, dtype=np.int32), rows_per_era).astype(np.int32)


def test_day_conversions_round_trip() -> None:
    assert refits.to_days("1970-01-01") == 0
    assert refits.to_days("2003-01-03") == 12_055
    assert refits.to_date(refits.to_days("2026-07-10")) == "2026-07-10"


def test_rows_follow_the_purged_walk_forward() -> None:
    sizes = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    days = weekly_panel(10, sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)])  # first row of each era
    table = refits.refit_table(days, ("2003-01-31", "2003-02-21"), purge=2)  # eras 4 and 7
    assert table == (
        Refit("2003-01-31", train_hi=int(starts[2]), pred_lo=int(starts[4]), pred_hi=int(starts[7])),
        Refit("2003-02-21", train_hi=int(starts[5]), pred_lo=int(starts[7]), pred_hi=int(days.size)),
    )


def test_a_refit_date_between_eras_starts_at_the_next_era() -> None:
    days = weekly_panel(6, [2] * 6)
    (refit,) = refits.refit_table(days, ("2003-01-29",), purge=1)  # a Wednesday before era 4 (2003-01-31)
    assert (refit.train_hi, refit.pred_lo, refit.pred_hi) == (6, 8, 12)


@pytest.mark.parametrize(
    ("days", "dates", "purge", "message"),
    [
        (np.array([], np.int32), ("2003-01-03",), 0, "sorted"),
        (np.array([5, 4], np.int32), ("1970-01-05",), 0, "sorted"),
        (weekly_panel(4, [1] * 4), ("2003-01-17", "2003-01-10"), 0, "increasing"),
        (weekly_panel(4, [1] * 4), ("2003-01-17", "2003-01-17"), 0, "increasing"),
        (weekly_panel(4, [1] * 4), ("2003-01-17",), -1, "non-negative"),
        (weekly_panel(4, [1] * 4), ("2003-01-17",), 3, "fewer than 3 eras"),
    ],
)
def test_bad_inputs_raise(days: np.ndarray, dates: tuple[str, ...], purge: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        refits.refit_table(days, dates, purge)
