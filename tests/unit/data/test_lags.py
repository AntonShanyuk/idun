import random
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from idun.data import lag_index
from idun.data.lags import ABSENT


def oracle(time: np.ndarray, group: np.ndarray, lags: list[int], step: int) -> np.ndarray:
    """Look every lag up in a dict of (group, time) -> row."""
    where = {(int(g), int(t)): r for r, (g, t) in enumerate(zip(group, time, strict=True))}
    return np.array([[where.get((int(g), int(t) - lag * step), ABSENT) for lag in lags]
                     for g, t in zip(group, time, strict=True)], np.int32).reshape(len(time), len(lags))  # fmt: skip


def test_a_weekly_panel() -> None:
    # two tickers on weekly days 0, 7, 14, 21; ticker 1 misses day 7
    time = np.array([0, 0, 7, 14, 14, 21, 21], np.int32)
    group = np.array([0, 1, 0, 0, 1, 0, 1], np.int32)
    got = lag_index(time, group, [2, 1, 0], step=7)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, [[-1, -1, 0], [-1, -1, 1], [-1, 0, 2], [0, 2, 3], [1, -1, 4], [2, 3, 5],
                                        [-1, 4, 6]])  # fmt: skip


@settings(max_examples=60, deadline=None)
@given(
    cells=st.sets(st.tuples(st.integers(-3, 3), st.integers(-20, 40)), max_size=60),
    lags=st.lists(st.integers(0, 6), max_size=5),
    step=st.integers(1, 7),
    shuffle=st.randoms(use_true_random=False),
)
def test_matches_a_lookup_of_every_pair(
    cells: set[tuple[int, int]], lags: list[int], step: int, shuffle: random.Random
) -> None:
    pairs = sorted(cells)
    shuffle.shuffle(pairs)  # any row order
    group = np.array([g for g, _ in pairs], np.int64).reshape(-1)
    time = np.array([t for _, t in pairs], np.int64).reshape(-1)
    got = lag_index(time, group, lags, step)
    np.testing.assert_array_equal(got, oracle(time, group, lags, step))
    rows = np.nonzero(got >= 0)
    lagged = got[rows]
    assert (group[lagged] == group[rows[0]]).all()
    assert (time[lagged] == time[rows[0]] - np.asarray(lags, np.int64)[rows[1]] * step).all()


def test_a_lag_never_crosses_into_another_group() -> None:
    # group 1 starts where group 0 would continue in key order: a lag past a group's first time finds nothing
    time = np.array([0, 1, 5], np.int32)
    group = np.array([1, 1, 0], np.int32)
    np.testing.assert_array_equal(lag_index(time, group, [1, 3], step=1), [[-1, -1], [0, -1], [-1, -1]])


def test_large_times_and_negative_groups() -> None:
    time = np.array([2**31 - 2, 2**31 - 1, 2**31 - 3], np.int64)
    group = np.array([-(2**31), -(2**31), 2**31 - 1], np.int64)
    np.testing.assert_array_equal(lag_index(time, group, [1], step=1), [[-1], [0], [-1]])


def test_no_rows_and_no_lags() -> None:
    assert lag_index(np.zeros(0, np.int32), np.zeros(0, np.int32), [0, 1], 7).shape == (0, 2)
    assert lag_index(np.arange(3), np.zeros(3, np.int32), [], 7).shape == (3, 0)


@pytest.mark.parametrize(
    ("time", "group", "lags", "step", "message"),
    [
        (np.zeros((2, 2), np.int32), np.zeros((2, 2), np.int32), [1], 1, "1-D"),
        (np.zeros(2, np.int32), np.zeros(3, np.int32), [1], 1, "1-D"),
        (np.zeros(2, np.float32), np.zeros(2, np.int32), [1], 1, "integer"),
        (np.arange(2), np.array([0, 2**31]), [1], 1, "32 bits"),
        (np.arange(2), np.zeros(2, np.int32), [1], 0, "step"),
        (np.arange(2), np.zeros(2, np.int32), [1], True, "step"),
        (np.arange(2), np.zeros(2, np.int32), [-1], 1, "non-negative"),
        (np.arange(2), np.zeros(2, np.int32), [1.0], 1, "non-negative"),
        (np.array([0, 2**32]), np.zeros(2, np.int32), [1], 1, "span"),
        (np.zeros(2, np.int32), np.zeros(2, np.int32), [1], 1, "one row"),
    ],
)
def test_bad_panels_raise(time: np.ndarray, group: np.ndarray, lags: Any, step: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        lag_index(time, group, lags, step)
