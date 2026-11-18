import numpy as np
import pytest

from idun.nn import training
from idun.nn.inputs import MatrixData


def test_epoch_order_is_a_seeded_permutation() -> None:
    order = training.epoch_order(5, 1, 100)
    np.testing.assert_array_equal(np.sort(order), np.arange(100))
    np.testing.assert_array_equal(order, training.epoch_order(5, 1, 100))
    assert not np.array_equal(order, training.epoch_order(5, 2, 100))
    assert not np.array_equal(order, training.epoch_order(6, 1, 100))


def test_a_fit_report_starts_empty() -> None:
    report = training.FitReport()
    assert (report.epoch_loss, report.epoch_seconds, report.seconds) == ([], [], 0.0)


def test_softmax_rows_sum_to_one_and_keep_the_order() -> None:
    p = training.softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 0.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (np.argsort(p[0]) == [0, 1, 2]).all()
    np.testing.assert_allclose(p[1], [0.5, 0.5, 0.0], atol=1e-12)  # large logits do not overflow


def test_a_validation_score_is_the_error_rate_or_the_squared_error() -> None:
    probs = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    assert training.score(probs, np.array([0, 0, 0]), 2) == pytest.approx(1 / 3)
    assert training.score(np.array([1.0, 2.0]), np.array([1.5, 2.0], np.float32), 1) == pytest.approx(0.125)
    x = np.zeros((10, 2), np.float32)
    valid = training.Validation(MatrixData(x, np.zeros(10, np.float32), 1), slice(2, 8))
    assert valid.bounds() == (2, 8)
    with pytest.raises(ValueError, match="contiguous non-empty"):
        training.Validation(valid.data, slice(5, 5)).bounds()
