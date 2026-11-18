import mmap

import numpy as np
import pytest

from idun.nn.inputs import MISSING_CLASS
from idun_numerai import inputs


def test_classes_are_the_target_bins_and_missing_targets_are_marked() -> None:
    target = np.array([0.0, 0.25, 0.5, 0.75, 1.0, np.nan], np.float32)
    got = inputs.classes(target)
    assert got.shape == (6, 1)
    assert got.dtype == np.dtype(np.uint8)
    assert got.__array_interface__["data"][0] % mmap.PAGESIZE == 0  # page-aligned: Metal wraps it without a copy
    assert got[:, 0].tolist() == [0, 1, 2, 3, 4, MISSING_CLASS]
    two = inputs.classes(np.stack([target, 1 - target]))
    assert two.shape == (6, 2)
    assert two[:5, 1].tolist() == [4, 3, 2, 1, 0]


def test_classes_are_computed_in_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    target = np.tile(np.array([0.0, 0.5, 1.0], np.float32), 7)
    whole = inputs.classes(target)
    monkeypatch.setattr(inputs, "BLOCK_ROWS", 4)
    np.testing.assert_array_equal(inputs.classes(target), whole)


def test_other_values_and_shapes_raise() -> None:
    with pytest.raises(ValueError, match="5 bins"):
        inputs.classes(np.array([0.3], np.float32))
    with pytest.raises(ValueError, match="5 bins"):
        inputs.classes(np.array([1.25], np.float32))
    with pytest.raises(ValueError, match=r"\(K, n_rows\)"):
        inputs.classes(np.zeros((2, 3, 4), np.float32))


def test_the_expected_class_is_the_probability_weighted_class() -> None:
    proba = np.array([[1, 0, 0, 0, 0], [0, 0, 0.5, 0, 0.5]], np.float32)
    np.testing.assert_allclose(inputs.expected(proba), [0.0, 3.0])
    with pytest.raises(ValueError, match=r"\(n, 5\)"):
        inputs.expected(np.zeros((2, 3)))


def z_by_era(y: np.ndarray, era: np.ndarray) -> np.ndarray:
    """A plain per-era z in float64 (sd with ddof 0, 0 where the sd is 0): the oracle."""
    out = np.full(len(y), np.nan)
    for u in np.unique(era):
        i = np.where(era == u)[0]
        v = y[i].astype("float64")
        s = v.std()
        out[i] = (v - v.mean()) / s if s > 0 else 0.0
    return out


def test_per_era_z_is_the_oracle_bit_for_bit() -> None:
    generator = np.random.default_rng(9)
    era = np.repeat(np.arange(7, dtype=np.int32) * 7, [50, 1, 30, 200, 2, 40, 77])
    targets = generator.normal(0.1, 0.8, (3, era.size)).astype(np.float32)
    targets[1, 50] = 0.25  # the era of one row: its sd is 0
    targets[2, 81:281] = 0.75  # a constant era
    z = inputs.per_era_z(targets, era)
    assert z.shape == (era.size, 3)
    assert z.dtype == np.dtype(np.float32)
    assert z.__array_interface__["data"][0] % mmap.PAGESIZE == 0  # page-aligned: Metal wraps it without a copy
    for k in range(3):
        np.testing.assert_array_equal(z[:, k], z_by_era(targets[k], era).astype(np.float32))
    assert float(z[50, 1]) == 0.0
    assert float(np.abs(z[81:281, 2]).max()) == 0.0
    np.testing.assert_array_equal(inputs.per_era_z(targets[0], era)[:, 0], z[:, 0])  # one target: (n_rows,)


def test_per_era_z_keeps_missing_targets_missing() -> None:
    generator = np.random.default_rng(10)
    era = np.repeat(np.arange(3, dtype=np.int32), [40, 60, 20])
    target = generator.normal(size=era.size).astype(np.float32)
    target[[3, 50, 51]] = np.nan
    target[100:] = np.nan  # an era without any target
    z = inputs.per_era_z(target, era)[:, 0]
    missing = np.isnan(target)
    np.testing.assert_array_equal(np.isnan(z), missing)
    for lo, hi in ((0, 40), (40, 100)):
        present = ~missing[lo:hi]
        want = z_by_era(target[lo:hi][present], era[lo:hi][present]).astype(np.float32)
        np.testing.assert_array_equal(z[lo:hi][present], want)


def test_per_era_z_needs_sorted_eras_and_matching_shapes() -> None:
    target = np.zeros(6, np.float32)
    with pytest.raises(ValueError, match="sorted"):
        inputs.per_era_z(target, np.array([0, 0, 2, 1, 1, 1]))
    with pytest.raises(ValueError, match="one value per row"):
        inputs.per_era_z(target, np.zeros(5, np.int32))
    with pytest.raises(ValueError, match=r"\(n_rows,\) or \(K, n_rows\)"):
        inputs.per_era_z(np.zeros((1, 2, 3), np.float32), np.zeros(3, np.int32))


def test_era_bounds_are_the_runs_of_a_sorted_era() -> None:
    np.testing.assert_array_equal(inputs.era_bounds(np.array([2, 2, 5, 7, 7, 7]), 6), [0, 2, 3, 6])
    np.testing.assert_array_equal(inputs.era_bounds(np.array([4]), 1), [0, 1])
    with pytest.raises(ValueError, match="one value per row"):
        inputs.era_bounds(np.array([1, 2]), 3)
    with pytest.raises(ValueError, match="one value per row"):
        inputs.era_bounds(np.array([], np.int32), 0)
    with pytest.raises(ValueError, match="sorted"):
        inputs.era_bounds(np.array([3, 1]), 2)
