import dataclasses

import pytest

from idun.data import Refit


def test_a_refit_is_a_frozen_row_window() -> None:
    refit = Refit("2003-04-04", train_hi=10, pred_lo=14, pred_hi=20)
    assert dataclasses.asdict(refit) == {"date": "2003-04-04", "train_hi": 10, "pred_lo": 14, "pred_hi": 20}
    with pytest.raises(dataclasses.FrozenInstanceError):
        refit.train_hi = 11  # type: ignore[misc]
