import dataclasses

import numpy as np
import pytest

from idun import gbm
from idun.data.layout import unpack_nibbles
from idun.gbm.features import CATEGORICAL, NAMED, OBFUSCATED, BinnedRows, FeatureLayout, take_rows
from tests.unit.gbm.tiny import make_tiny


def test_layout_follows_loader_order() -> None:
    data = make_tiny(n_named=2, n_obfuscated=3, n_countries=7, named_bins=9)
    layout = FeatureLayout.of(data)
    assert layout.n_features == 6
    assert layout.kind.tolist() == [NAMED, NAMED, OBFUSCATED, OBFUSCATED, OBFUSCATED, CATEGORICAL]
    assert layout.column.tolist() == [0, 1, 0, 1, 2, 0]
    assert layout.hist_bins.tolist() == [256, 256, 5, 5, 5, 7]
    assert layout.real_bins.tolist() == [9, 9, 5, 5, 5, 7]


def test_offsets_pack_the_sampled_features() -> None:
    layout = FeatureLayout.of(make_tiny(n_named=2, n_obfuscated=3, n_countries=7))
    offsets, cells = layout.offsets(np.array([1, 3, 4, 5], np.int32))
    assert offsets.tolist() == [0, 256, 261, 266]
    assert cells == 273


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"feature_names": ("a",)}, "feature_names"),
        ({"categories": ((),)}, "categories"),
        ({"categories": (tuple(str(i) for i in range(65)),)}, "categories"),
        ({"categorical": np.zeros((2, 300), np.uint8)}, "categorical must be"),
        ({"named_n_bins": (300, 3)}, "named_n_bins"),
        ({"named_n_bins": (3,)}, "named_n_bins"),
    ],
)
def test_layouts_that_do_not_fit_the_model_format_raise(change: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FeatureLayout.of(dataclasses.replace(make_tiny(n_named=2), **change))  # type: ignore[arg-type]


@pytest.mark.parametrize("count", [1, 2, 37, 400])
def test_take_rows_copies_the_rows_named_in_their_order(count: int) -> None:
    data = make_tiny(301, n_named=2, n_obfuscated=70, n_countries=6, more_categorical=(3,), seed=4)
    rows = np.random.default_rng(count).integers(0, 301, count)  # any order, repeats among them
    taken = take_rows(data, rows)
    assert isinstance(taken, BinnedRows)
    assert taken.n_rows == count
    np.testing.assert_array_equal(taken.named_bins, data.named_bins[:, rows])
    assert taken.obfuscated.shape == (70, (count + 1) // 2)
    np.testing.assert_array_equal(unpack_nibbles(taken.obfuscated, 0, count), data.codes[:, rows])
    if count % 2:
        assert not (taken.obfuscated[:, -1] >> 4).any()  # the odd tail's spare nibble is zero
    np.testing.assert_array_equal(taken.categorical, data.categorical[:, rows])
    np.testing.assert_array_equal(taken.target, data.target[rows])
    assert taken.named_bins.flags.c_contiguous
    assert taken.categorical.flags.c_contiguous
    for name in ("binning", "feature_names", "named_features", "obfuscated_features", "categories", "named_n_bins"):
        assert getattr(taken, name) == getattr(data, name)
    assert taken.named_edges is data.named_edges
    assert FeatureLayout.of(taken).real_bins.tolist() == FeatureLayout.of(data).real_bins.tolist()


def test_take_rows_of_a_kind_without_columns() -> None:
    data = make_tiny(50, n_named=3, n_obfuscated=0, n_countries=0)
    taken = take_rows(data, np.array([49, 0, 0], np.int32))
    assert taken.obfuscated.shape == (0, 2)
    assert taken.categorical.shape == (0, 3)
    np.testing.assert_array_equal(taken.named_bins, data.named_bins[:, [49, 0, 0]])


def test_a_fit_on_the_first_rows_taken_is_the_fit_on_that_slice() -> None:
    data = make_tiny(260, n_named=2, n_obfuscated=5, n_countries=7, seed=8)
    params = gbm.Params(num_trees=4, max_depth=3, max_leaves=8, min_data_in_leaf=5, feature_fraction=0.7,
                        bagging_fraction=0.8, bagging_freq=1, seed=2)  # fmt: skip
    taken = take_rows(data, np.arange(240))  # the rows keep their indices, which key the per-row draws
    fit = gbm.train(params, taken, rows=slice(0, taken.n_rows), backend="cpu")
    expected = gbm.train(params, data, rows=slice(0, 240), backend="cpu")
    for name in ("kind", "feature", "threshold", "missing_left", "cat_left", "leaf_value"):
        np.testing.assert_array_equal(getattr(fit.forest, name), getattr(expected.forest, name), err_msg=name)
    everything = slice(0, 260)
    np.testing.assert_array_equal(fit.predict(data, rows=everything, backend="cpu"),
                                  expected.predict(data, rows=everything, backend="cpu"))  # fmt: skip
    np.testing.assert_array_equal(fit.predict(taken, rows=slice(0, 240), backend="cpu"),
                                  expected.predict(data, rows=slice(0, 240), backend="cpu"))  # fmt: skip


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (np.zeros((2, 2), np.int64), "1-D integer array"),
        (np.array([], np.int64), "1-D integer array"),
        (np.array([0.0, 1.0]), "1-D integer array"),
        (np.array([True, False]), "1-D integer array"),
        (np.array([0, 300]), r"rows must lie in \[0, 300\); got 0..300"),
        (np.array([-1, 5]), r"rows must lie in \[0, 300\); got -1..5"),
    ],
)
def test_take_rows_refuses_rows_it_cannot_take(rows: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        take_rows(make_tiny(), rows)
