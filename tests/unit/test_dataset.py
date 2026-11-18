from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from scipy import sparse

from idun import gbm
from idun.data import PackedDataset, binning
from idun.dataset import Dataset, n_columns
from idun.gbm.features import CATEGORICAL, NAMED, OBFUSCATED, FeatureLayout
from tests.unit.data.synthetic import SAMPLE_ROWS


def table(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=n), rng.integers(0, 5, n), rng.integers(0, 4, n), rng.uniform(size=n)])
    x[::13, 0] = np.nan
    y = (x[:, 1] / 4 + (x[:, 2] == 1) * 0.5 + rng.normal(0, 0.1, n)).astype(np.float32)
    return x, y


class FrameLike:
    """What `Dataset` reads from a pandas DataFrame: ``columns``, ``to_numpy`` and ``frame[name].to_numpy()``."""

    def __init__(self, columns: dict[str, Any]) -> None:
        self.columns = list(columns)
        self._data = columns

    def to_numpy(self) -> Any:
        return np.column_stack(list(self._data.values()))

    def __getitem__(self, name: str) -> Any:
        return type("Series", (), {"to_numpy": lambda _: np.asarray(self._data[name])})()


def test_columns_are_laid_out_by_kind_and_binned() -> None:
    x, y = table()
    data = Dataset(x, y, codes=[1], categorical=[2])
    assert data.feature_names == ("x0", "x3", "x1", "x2")
    assert (data.named_features, data.obfuscated_features, data.categorical_features) == (
        ("x0", "x3"),
        ("x1",),
        ("x2",),
    )
    layout = FeatureLayout.of(data)
    assert layout.kind.tolist() == [NAMED, NAMED, OBFUSCATED, CATEGORICAL]
    assert data.categories == (("0.0", "1.0", "2.0", "3.0"),)
    np.testing.assert_array_equal(data.categorical[0], x[:, 2].astype(np.uint8))
    sample = binning.sample_rows(600, binning.SAMPLE_ROWS, seed=0, refit=0)
    edges = binning.quantile_edges(np.ascontiguousarray(x[sample, 0].astype(np.float32)))
    np.testing.assert_array_equal(
        data.named_bins[0], binning.apply_bins(x[:, 0].astype(np.float32), data.named_edges[0])
    )
    assert data.named_n_bins[0] == edges.size + 1
    assert (data.named_bins[0][np.isnan(x[:, 0])] == 255).all()
    np.testing.assert_array_equal(data.target, y)
    assert "2 numeric, 1 codes, 1 categorical" in repr(data)


def test_arrow_and_dataframe_inputs_give_the_same_dataset() -> None:
    x, y = table(200)
    columns = {name: x[:, i] for i, name in enumerate(["a", "b", "c", "d"])}
    from_arrow = Dataset(pa.table(columns), y, codes=["b"], categorical=["c"])
    from_frame = Dataset(FrameLike(columns), y, codes=["b"], categorical=["c"])
    from_array = Dataset(x, y, feature_names=["a", "b", "c", "d"], codes=["b"], categorical=["c"])
    for other in (from_frame, from_array):
        assert other.feature_names == from_arrow.feature_names
        assert other.binning == from_arrow.binning
        np.testing.assert_array_equal(other.named_bins, from_arrow.named_bins)
        np.testing.assert_array_equal(other.categorical, from_arrow.categorical)
        np.testing.assert_array_equal(other.obfuscated, from_arrow.obfuscated)


def test_numeric_columns_bin_exactly_as_the_packed_cache(nn_cache: Path) -> None:
    for refit in (0, 1):
        data = PackedDataset.open(nn_cache, binning=f"r{refit}")
        window = data.refits[refit].train_hi
        sample = binning.sample_rows(window, SAMPLE_ROWS, seed=data.seed, refit=refit)  # the synthetic cache's sample
        ours = Dataset(np.asarray(data.named_values).T, feature_names=data.named_features, bin_rows=sample)
        np.testing.assert_array_equal(ours.named_bins, data.named_bins)
        np.testing.assert_array_equal(ours.named_edges, data.named_edges)
        assert ours.named_n_bins == tuple(data.named_n_bins)


def test_transform_and_subset_reuse_the_binning() -> None:
    x, y = table()
    data = Dataset(x, y, codes=[1], categorical=[2])
    later = data.transform(x[100:200], y[100:200])
    part = data.subset(slice(100, 200))
    assert later.binning == part.binning == data.binning
    for name in ("named_bins", "obfuscated", "categorical", "target"):
        np.testing.assert_array_equal(getattr(later, name), getattr(part, name), err_msg=name)
    rows = np.array([5, 2, 9, 2])
    np.testing.assert_array_equal(data.subset(rows).categorical[0], data.categorical[0, rows])
    with pytest.raises(ValueError, match="levels the data was not built with"):
        data.transform(np.column_stack([x[:3, :2], [7, 7, 7], x[:3, 3]]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"categorical": [5]}, "out of range"),
        ({"categorical": ["nope"]}, "not among the columns"),
        ({"codes": [1], "categorical": [1]}, "both categorical and codes"),
        ({"codes": [0]}, "missing values"),
        ({"codes": [3]}, "integers 0..4"),
        ({"categorical": [0]}, "missing values"),
    ],
)
def test_columns_that_do_not_fit_their_kind_fail_loudly(kwargs: dict[str, Any], message: str) -> None:
    x, y = table()
    with pytest.raises(ValueError, match=message):
        Dataset(x, y, **kwargs)


def test_shapes_and_level_counts_fail_loudly() -> None:
    x, y = table()
    with pytest.raises(ValueError, match="2-D"):
        Dataset(x[:, 0])
    with pytest.raises(ValueError, match="one value per row"):
        Dataset(x, y[:-1])
    with pytest.raises(ValueError, match="feature_names"):
        Dataset(x, feature_names=["a"])
    with pytest.raises(ValueError, match="65 levels"):
        Dataset(np.arange(130.0).reshape(-1, 1) % 65, categorical=[0])
    with pytest.raises(ValueError, match="no rows"):
        Dataset(np.zeros((0, 2)))


def test_inputs_are_checked_as_scikit_learn_checks_them() -> None:
    x, y = table()
    with pytest.raises(ValueError, match="Reshape your data"):
        Dataset(x[:, 0])
    with pytest.raises(ValueError, match=r"2-D \(rows, columns\), got shape \(600, 4, 1\)"):
        Dataset(x[:, :, None])
    with pytest.raises(TypeError, match="sparse input is not supported"):
        Dataset(sparse.csr_array(np.nan_to_num(x)))
    with pytest.raises(ValueError, match="Complex data not supported"):
        Dataset(x + 1j)
    with pytest.raises(ValueError, match="Complex data not supported"):
        Dataset(x, y + 1j)
    with pytest.raises(ValueError, match=r"0 feature\(s\) \(shape=\(4, 0\)\)"):
        Dataset(np.zeros((4, 0)))
    with pytest.raises(ValueError, match=r"0 feature\(s\)"):
        Dataset(pa.table({}))
    with pytest.raises(ValueError, match="finite"):
        Dataset(x, np.where(np.arange(y.size) == 3, np.inf, y))
    data = Dataset(x, y)
    with pytest.raises(ValueError, match="built with 4 columns"):
        data.transform(x[:, :3])
    with pytest.raises(ValueError, match="Reshape your data"):
        data.transform(x[0])


def test_column_counts_of_every_input_kind() -> None:
    x, _ = table()
    assert n_columns(x) == 4
    assert n_columns(pa.table({"a": [1.0], "b": [2.0]})) == 2
    assert n_columns(FrameLike({"a": np.zeros(3)})) == 1
    assert n_columns([("a", np.zeros(3)), ("b", np.ones(3))]) == 2
    with pytest.raises(ValueError, match="Reshape your data"):
        n_columns(x[:, 0])


def test_a_column_list_is_a_table() -> None:
    x, y = table()
    columns = [("a", x[:, 0]), ("b", x[:, 1]), ("c", x[:, 2]), ("d", x[:, 3])]
    from_list = Dataset(columns, y, codes=["b"], categorical=["c"])
    from_array = Dataset(x, y, codes=[1], categorical=[2], feature_names=["a", "b", "c", "d"])
    assert from_list.binning == from_array.binning
    assert from_list.columns == ("a", "b", "c", "d")
    np.testing.assert_array_equal(from_list.named_bins, from_array.named_bins)
    np.testing.assert_array_equal(from_list.obfuscated, from_array.obfuscated)
    new = from_list.transform([(name, values[:10]) for name, values in columns])
    np.testing.assert_array_equal(new.named_bins, from_array.named_bins[:, :10])
    with pytest.raises(ValueError, match="same length"):
        Dataset([("a", x[:, 0]), ("b", x[:5, 1])], y)
    with pytest.raises(ValueError, match="1-D"):
        Dataset([("a", x)], y)
    with pytest.raises(ValueError, match="named twice"):
        Dataset([("a", x[:, 0]), ("a", x[:, 1])], y)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("metal", marks=pytest.mark.metal)])
def test_idun_gbm_trains_on_a_dataset_and_predicts_new_rows(backend: str) -> None:
    x, y = table(800)
    data = Dataset(x[:600], y[:600], codes=[1], categorical=[2])
    params = gbm.Params(num_trees=6, max_depth=3, max_leaves=8, min_data_in_leaf=10, seed=1)
    model = gbm.train(params, data, rows=slice(0, 600), backend=backend)
    pred = model.predict(data.transform(x[600:]), rows=slice(0, 200), backend=backend)
    assert np.corrcoef(pred, y[600:])[0, 1] > 0.5
    cpu = gbm.train(params, data, rows=slice(0, 600), backend="cpu")
    np.testing.assert_array_equal(cpu.forest.leaf_value, model.forest.leaf_value)
    with pytest.raises(ValueError, match="binning"):
        model.predict(Dataset(x[600:], codes=[1], categorical=[2]), rows=slice(0, 200), backend=backend)


def test_max_bin_and_subsample_for_bin_set_the_numeric_edges() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5000, 3)).astype(np.float32)
    default = Dataset(x, np.zeros(5000))
    same = Dataset(x, np.zeros(5000), max_bin=255, subsample_for_bin=binning.SAMPLE_ROWS)
    np.testing.assert_array_equal(same.named_bins, default.named_bins)  # the defaults passed explicitly bin the same
    assert same.binning == default.binning
    coarse = Dataset(x, np.zeros(5000), max_bin=16)
    assert max(coarse.named_n_bins) <= 16
    assert int(coarse.named_bins.max()) <= 15
    assert coarse.binning != default.binning  # a model refuses data binned the other way
    sampled = Dataset(x, np.zeros(5000), subsample_for_bin=100)
    assert max(sampled.named_n_bins) <= 100  # 100 sampled values: at most that many distinct
    for bad, message in (
        ({"max_bin": 1}, "max_bin must be 2 to 255"),
        ({"max_bin": 256}, "max_bin must be 2 to 255"),
        ({"subsample_for_bin": 0}, "must be positive"),
        ({"subsample_for_bin": 10, "bin_rows": np.arange(10)}, "bin_rows or subsample_for_bin"),
    ):
        with pytest.raises(ValueError, match=message):
            Dataset(x, np.zeros(5000), **bad)
