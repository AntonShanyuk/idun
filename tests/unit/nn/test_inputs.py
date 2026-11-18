import mmap
from pathlib import Path

import numpy as np
import pytest

from idun.data import PackedDataset
from idun.nn import inputs


def test_dense_inputs_map_named_values_codes_and_country(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    rows = np.array([5, 0, 1, data.n_rows - 1, 2, 2], np.int64)  # any order, odd and even rows, repeats
    x = inputs.dense_inputs(data, rows)
    n_named, n_obf = len(data.named_features), len(data.obfuscated_features)
    assert x.shape == (rows.size, inputs.n_inputs(data)) == (rows.size, n_named + n_obf + len(data.countries))
    assert x.dtype == np.float32
    named = np.asarray(data.named_values)[:, rows].T
    assert np.isnan(np.asarray(data.named_values)).any()  # the synthetic panel has missing values to map
    np.testing.assert_array_equal(x[:, :n_named], np.nan_to_num(named, nan=0.0))
    for f in range(n_obf):
        codes = data.obfuscated_codes(f)[rows].astype(np.float32)
        np.testing.assert_array_equal(x[:, n_named + f], (codes - 2) / 2)
    country = x[:, n_named + n_obf :]
    np.testing.assert_array_equal(country.sum(axis=1), 1.0)
    np.testing.assert_array_equal(country.argmax(axis=1), np.asarray(data.country)[rows])


def test_a_matrix_carries_its_labels_and_a_packed_dataset_none(nn_cache: Path) -> None:
    matrix = inputs.MatrixData(np.zeros((4, 2), np.float32), np.array([0, 2, 1, 2], np.int64), 3)
    np.testing.assert_array_equal(inputs.labels(matrix, np.array([3, 0])), [2, 0])
    with pytest.raises(TypeError, match="pass its classes"):
        inputs.labels(PackedDataset.open(nn_cache, binning="r1"), np.arange(3))  # type: ignore[arg-type]


def test_bad_rows_raise(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    with pytest.raises(ValueError, match="row indices"):
        inputs.dense_inputs(data, np.array([data.n_rows]))
    with pytest.raises(ValueError, match="row indices"):
        inputs.dense_inputs(data, np.array([[0]]))


def test_raw_codes_are_the_codes_themselves(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    rows = np.array([4, 1, 0, 7], np.int64)
    codes = np.s_[len(data.named_features) : len(data.named_features) + len(data.obfuscated_features)]
    raw = inputs.dense_inputs(data, rows, raw_codes=True)
    centred = inputs.dense_inputs(data, rows)
    np.testing.assert_array_equal(raw[:, codes], 2 * centred[:, codes] + 2)
    np.testing.assert_array_equal(np.delete(raw, codes, axis=1), np.delete(centred, codes, axis=1))


@pytest.mark.parametrize("block", [inputs.CODES_BLOCK_ROWS, 5])  # one task, or many (odd blocks share a byte)
def test_codes_row_major_are_the_codes(nn_cache: Path, block: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inputs, "CODES_BLOCK_ROWS", block)
    data = PackedDataset.open(nn_cache, binning="r1")
    n_obf = len(data.obfuscated_features)
    out = np.zeros((data.n_rows, 16), np.uint8)
    inputs.codes_row_major(data, out)
    unpacked = np.empty((data.n_rows, 32), np.uint8)
    unpacked[:, 0::2], unpacked[:, 1::2] = out & 0x0F, out >> 4
    codes = np.s_[len(data.named_features) : len(data.named_features) + n_obf]
    want = inputs.dense_inputs(data, np.arange(data.n_rows), raw_codes=True)[:, codes]
    np.testing.assert_array_equal(unpacked[:, :n_obf], want)
    assert not unpacked[:, n_obf:].any()
    with pytest.raises(ValueError, match="uint8 array"):
        inputs.codes_row_major(data, np.zeros((data.n_rows, 3), np.uint8))


def test_class_arrays_are_page_aligned_and_missing_until_set() -> None:
    classes = inputs.class_array(1000, 7)
    assert classes.shape == (1000, 7)
    assert classes.dtype == np.dtype(np.uint8)
    assert classes.flags.c_contiguous
    assert classes.__array_interface__["data"][0] % mmap.PAGESIZE == 0  # the system's page: 16 KiB on Apple silicon
    assert (classes == inputs.MISSING_CLASS).all()
    with pytest.raises(ValueError, match="at least one row"):
        inputs.class_array(0, 3)


def test_class_labels_read_a_compact_class_array() -> None:
    rows = np.array([4, 1, 0, 7, 9], np.int64)
    classes = np.ascontiguousarray(np.arange(20).reshape(10, 2) % 3, np.uint8)  # K = 2, classes 0..2
    got = inputs.class_labels(classes, rows, 3)
    assert got.dtype == np.dtype(np.int64)
    assert got.shape == (5, 2)
    np.testing.assert_array_equal(got, classes[rows])
    classes[3, 1] = inputs.MISSING_CLASS  # a row without its second label: fine unless it is read
    np.testing.assert_array_equal(inputs.class_labels(classes, rows[:3], 3), got[:3])
    for bad in (inputs.MISSING_CLASS, 3):
        classes[7, 0] = bad
        with pytest.raises(ValueError, match=r"not 0\.\.2"):
            inputs.class_labels(classes, rows, 3)
    wrongs: list[np.ndarray] = [classes[:, 0], classes.astype(np.int64), np.zeros((10, 0), np.uint8)]
    for wrong in wrongs:
        with pytest.raises(ValueError, match=r"\(n_rows, K\) uint8"):
            inputs.class_labels(wrong, rows, 3)


def test_target_arrays_are_page_aligned_and_missing_until_set() -> None:
    targets = inputs.target_array(1000, 3)
    assert targets.shape == (1000, 3)
    assert targets.dtype == np.dtype(np.float32)
    assert targets.flags.c_contiguous
    assert targets.__array_interface__["data"][0] % mmap.PAGESIZE == 0
    assert np.isnan(targets).all()
    with pytest.raises(ValueError, match="at least one row"):
        inputs.target_array(5, 0)


def test_checked_targets_check_the_shape_and_every_training_row(monkeypatch: pytest.MonkeyPatch) -> None:
    targets = inputs.target_array(10, 2)
    targets[2:8] = 0.5
    assert inputs.checked_targets(targets, 10, 2, 8) is targets  # no copy; NaN outside the training rows is fine
    wrongs: list[np.ndarray] = [targets[:, 0], targets.astype(np.float64), targets[:9], np.zeros((10, 0), np.float32)]
    for wrong in wrongs:
        with pytest.raises(ValueError, match=r"targets must be an \(10, K\) float32"):
            inputs.checked_targets(wrong, 10, 2, 8)
    monkeypatch.setattr(inputs, "CHECK_ROWS", 2)  # the bad value lies in the last chunk
    for bad in (np.nan, np.inf, -np.inf):
        targets[7, 1] = bad
        with pytest.raises(ValueError, match="must be finite"):
            inputs.checked_targets(targets, 10, 2, 8)


def test_the_label_source_must_suit_the_data_and_the_objective(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    matrix = inputs.MatrixData(np.zeros((4, 2), np.float32), np.zeros(4, np.float32), 1)
    classes, targets = inputs.class_array(4, 1), inputs.target_array(4, 1)
    labels: tuple[object, object]
    inputs.check_label_source(matrix, "mse", None, None)
    inputs.check_label_source(data, "cross_entropy", classes, None)
    inputs.check_label_source(data, "mse", None, targets)
    for labels in [(classes, None), (None, targets)]:
        with pytest.raises(ValueError, match="carries its own labels"):
            inputs.check_label_source(matrix, "mse", *labels)
    for labels in [(None, None), (None, targets), (classes, targets)]:
        with pytest.raises(ValueError, match="needs its classes for objective 'cross_entropy'"):
            inputs.check_label_source(data, "cross_entropy", *labels)
    for labels in [(None, None), (classes, None), (classes, targets)]:
        with pytest.raises(ValueError, match="needs its targets for objective 'mse'"):
            inputs.check_label_source(data, "mse", *labels)
