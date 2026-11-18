import numpy as np
import pyarrow as pa
import pytest

from idun.nn.table import TableEncoder


def table() -> np.ndarray:
    return np.array([[1.0, 0.0, 2.0], [3.0, 1.0, np.nan], [5.0, 2.0, 4.0], [7.0, 1.0, 6.0]])


def test_numeric_values_are_standardized_and_categorical_columns_one_hot() -> None:
    encoder = TableEncoder(categorical=[1], standardize=True).fit(table())
    out = encoder.transform(table())
    assert out.dtype == np.float32
    assert out.shape == (4, encoder.n_inputs) == (4, 2 + 3)
    np.testing.assert_allclose(out[:, 0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:, 0].std(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(out[:, 2:], np.eye(3, dtype=np.float32)[[0, 1, 2, 1]])
    raw = TableEncoder(categorical=[1]).fit(table()).transform(table())
    np.testing.assert_array_equal(raw[:, 1], [2.0, 0.0, 4.0, 6.0])  # NaN -> 0, not scaled


def test_the_spec_rebuilds_the_encoder_and_columns_are_checked() -> None:
    encoder = TableEncoder(categorical=["b"], standardize=True).fit(pa.table({"a": table()[:, 0], "b": table()[:, 1]}))
    again = TableEncoder.from_spec(encoder.spec())
    frame = pa.table({"b": [2.0, 0.0], "a": [5.0, 1.0]})  # the same columns in another order
    np.testing.assert_array_equal(again.transform(frame), encoder.transform(frame))
    with pytest.raises(ValueError, match="lacks the columns"):
        encoder.transform(pa.table({"a": [1.0]}))
    with pytest.raises(ValueError, match="fitted on 3 columns"):
        TableEncoder().fit(table()).transform(table()[:, :2])
    with pytest.raises(ValueError, match="levels the data was not built with"):
        encoder.transform(pa.table({"a": [1.0], "b": [9.0]}))


def test_wide_categorical_columns_become_codes_last() -> None:
    x = np.column_stack([np.arange(6.0), np.array([0, 1, 2, 3, 2, 1.0]), np.array([5, 6, 5, 6, 5, 6.0]), np.ones(6)])
    encoder = TableEncoder(categorical=[1, 2], max_one_hot_levels=3).fit(x)  # column 1: 4 levels, column 2: 2
    assert encoder.coded_ == ("x1",)
    assert encoder.coded_levels == (4,)
    out = encoder.transform(x)
    assert out.shape == (6, encoder.n_inputs) == (6, 2 + 2 + 1)
    np.testing.assert_array_equal(out[:, 2:4], np.eye(2, dtype=np.float32)[[0, 1, 0, 1, 0, 1]])  # one-hot first
    np.testing.assert_array_equal(out[:, 4], [0, 1, 2, 3, 2, 1])  # then the codes
    again = TableEncoder.from_spec(encoder.spec())
    np.testing.assert_array_equal(again.transform(x), out)
    plain = TableEncoder(categorical=[1, 2]).fit(x)
    assert plain.coded_ == ()
    assert "coded" not in plain.spec()  # without coded columns, the spec of earlier versions
