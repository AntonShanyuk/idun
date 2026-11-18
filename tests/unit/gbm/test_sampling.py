import numpy as np
import pytest

from idun.gbm import sampling


def test_bag_is_a_bernoulli_draw_per_row() -> None:
    bag = sampling.bag_mask(100_000, 0.8, seed=0, iteration=3)
    assert bag.dtype == np.uint8
    assert set(np.unique(bag)) == {0, 1}
    assert abs(bag.mean() - 0.8) < 0.01
    np.testing.assert_array_equal(bag, sampling.bag_mask(100_000, 0.8, seed=0, iteration=3))
    assert not np.array_equal(bag, sampling.bag_mask(100_000, 0.8, seed=0, iteration=4))
    assert not np.array_equal(bag, sampling.bag_mask(100_000, 0.8, seed=1, iteration=3))


def test_a_full_bag_keeps_every_row() -> None:
    assert sampling.bag_mask(1_000, 1.0, seed=0, iteration=0).all()


@pytest.mark.parametrize(
    ("n", "fraction", "expected"), [(1040, 0.1, 104), (5, 0.1, 1), (5, 0.3, 2), (10, 0.25, 3), (7, 1.0, 7)]
)
def test_feature_count_rounds_half_up_like_lightgbm(n: int, fraction: float, expected: int) -> None:
    assert sampling.feature_count(n, fraction) == expected


def test_feature_sample_is_sorted_distinct_and_seeded() -> None:
    sample = sampling.feature_sample(1040, 0.1, seed=0, tree=5)
    assert sample.dtype == np.int32
    assert sample.size == 104
    assert np.all(np.diff(sample) > 0)
    np.testing.assert_array_equal(sample, sampling.feature_sample(1040, 0.1, seed=0, tree=5))
    assert not np.array_equal(sample, sampling.feature_sample(1040, 0.1, seed=0, tree=6))
    np.testing.assert_array_equal(sampling.feature_sample(9, 1.0, seed=0, tree=0), np.arange(9))


@pytest.mark.parametrize(("n", "fraction"), [(0, 0.5), (5, 0.0), (5, 1.5)])
def test_bad_requests_raise(n: int, fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        sampling.feature_count(n, fraction)
    if n:
        with pytest.raises(ValueError, match="fraction"):
            sampling.bag_mask(n, fraction, seed=0, iteration=0)
