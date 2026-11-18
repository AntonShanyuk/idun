import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from idun import gbm
from idun.gbm.tree import Forest, NodeForest
from tests.unit.gbm.tiny import TinyData, make_tiny


@pytest.fixture(scope="module")
def trained() -> tuple[gbm.Booster, TinyData]:
    data = make_tiny(300, seed=21)
    params = gbm.Params(
        num_trees=4, max_depth=3, max_leaves=8, min_data_in_leaf=8, bagging_fraction=0.8, bagging_freq=1
    )
    return gbm.train(params, data, rows=slice(0, 250), backend="cpu"), data


def test_save_and_load_predict_identically(trained: tuple[gbm.Booster, TinyData], tmp_path: Path) -> None:
    booster, data = trained
    booster.save(tmp_path / "model.igbm")
    assert (tmp_path / "model.igbm").is_file()  # the exact name: no .npz suffix appended
    loaded = gbm.Booster.load(tmp_path / "model.igbm")
    assert loaded.params == booster.params
    assert (loaded.binning, loaded.train_rows, loaded.backend) == ("r0", (0, 250), "cpu")
    assert loaded.fit_report is None
    expected = booster.predict(data, rows=slice(0, 300), backend="cpu")
    np.testing.assert_array_equal(loaded.predict(data, rows=slice(0, 300), backend="cpu"), expected)
    with np.load(tmp_path / "model.igbm", allow_pickle=False) as archive:
        meta = json.loads(archive["meta"].tobytes())
    assert (meta["format"], meta["format_version"], meta["n_trees"], meta["growth"]) == ("idun-gbm", 3, 4, "depthwise")


def test_predict_with_the_first_trees(trained: tuple[gbm.Booster, TinyData]) -> None:
    booster, data = trained
    none = booster.predict(data, rows=slice(0, 10), backend="cpu", num_trees=0)
    assert not none.any()
    partial = booster.predict(data, rows=slice(0, 300), backend="cpu", num_trees=2)
    full = booster.predict(data, rows=slice(0, 300), backend="cpu")
    assert not np.array_equal(partial, full)


def test_predict_with_a_window_of_trees(trained: tuple[gbm.Booster, TinyData]) -> None:
    booster, data = trained
    rows = slice(0, 300)
    later = booster.predict(data, rows=rows, backend="cpu", start_tree=2)
    alone = gbm.Booster(dataclasses.replace(booster.params, num_trees=2), booster.forest.trees(2, 4),
                        binning=booster.binning, feature_names=booster.feature_names, categories=booster.categories,
                        named_edges=booster.named_edges, named_n_bins=booster.named_n_bins,
                        train_rows=booster.train_rows, backend="cpu")  # fmt: skip
    np.testing.assert_array_equal(later, alone.predict(data, rows=rows, backend="cpu"))
    one = booster.predict(data, rows=rows, backend="cpu", start_tree=1, num_trees=1)
    np.testing.assert_array_equal(one, booster.predict(data, rows=rows, backend="cpu", start_tree=1, num_trees=1))
    for bad in ({"start_tree": 5}, {"num_trees": 5}, {"start_tree": 3, "num_trees": 2}):
        with pytest.raises(ValueError, match="outside the forest"):
            booster.predict(data, rows=rows, backend="cpu", **bad)
    with pytest.raises(ValueError, match="num_trees must be non-negative"):
        booster.predict(data, rows=rows, backend="cpu", num_trees=-1)


def test_split_importances_count_each_features_splits(trained: tuple[gbm.Booster, TinyData]) -> None:
    booster, data = trained
    found = booster.feature_importance()
    assert found.shape == (len(data.feature_names),)
    used = booster.forest.kind != 0
    assert found.sum() == used.sum()
    for feature in np.unique(booster.forest.feature[used]):
        assert found[feature] == ((booster.forest.feature == feature) & used).sum()
    with pytest.raises(ValueError, match="kind must be 'split'"):
        booster.feature_importance("gain")


def test_the_best_tree_and_the_scores_are_saved(tmp_path: Path) -> None:
    data = make_tiny(800, seed=22)
    params = gbm.Params(num_trees=30, max_depth=3, max_leaves=8, min_data_in_leaf=8, learning_rate=0.4)
    booster = gbm.train(params, data, rows=slice(0, 500), backend="cpu", valid=[gbm.Validation(data, slice(500, 800))],
                        early_stopping_rounds=2)  # fmt: skip
    booster.save(tmp_path / "es.igbm")
    loaded = gbm.Booster.load(tmp_path / "es.igbm")
    assert (loaded.best_iteration, loaded.evals) == (booster.best_iteration, booster.evals)
    assert loaded.best_iteration is not None


def test_predict_refuses_other_data(trained: tuple[gbm.Booster, TinyData]) -> None:
    booster, data = trained
    with pytest.raises(ValueError, match="binning"):
        booster.predict(dataclasses.replace(data, binning="r1"), rows=slice(0, 10), backend="cpu")
    other = dataclasses.replace(data, categories=(tuple(f"X{i}" for i in range(5)),))
    with pytest.raises(ValueError, match="features or categories"):
        booster.predict(other, rows=slice(0, 10), backend="cpu")
    with pytest.raises(ValueError, match="contiguous"):
        booster.predict(data, rows=slice(0, 10, 2), backend="cpu")


def test_load_refuses_anything_else(trained: tuple[gbm.Booster, TinyData], tmp_path: Path) -> None:
    booster, _ = trained
    booster.save(tmp_path / "good.igbm")
    with np.load(tmp_path / "good.igbm", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    np.savez(tmp_path / "missing.npz", **{k: v for k, v in arrays.items() if k != "cat_left"})
    with pytest.raises(ValueError, match=r"not a \.igbm"):
        gbm.Booster.load(tmp_path / "missing.npz")
    meta = json.loads(arrays["meta"].tobytes())
    np.savez(
        tmp_path / "version.npz",
        **{**arrays, "meta": np.frombuffer(json.dumps({**meta, "format_version": 4}).encode(), np.uint8)},
    )
    with pytest.raises(ValueError, match="not supported"):
        gbm.Booster.load(tmp_path / "version.npz")
    # a version-1 file names one categorical column, the country
    v1 = {k: v for k, v in meta.items() if k != "categories"} | {"format_version": 1, "countries": ["C0", "C1"]}
    np.savez(tmp_path / "v1.npz", **{**arrays, "meta": np.frombuffer(json.dumps(v1).encode(), np.uint8)})
    assert gbm.Booster.load(tmp_path / "v1.npz").categories == (("C0", "C1"),)
    # a version-2 file has no growth: depth-wise
    params = {k: v for k, v in meta["params"].items() if k != "growth"}
    v2 = {k: v for k, v in meta.items() if k != "growth"} | {"format_version": 2, "params": params}
    np.savez(tmp_path / "v2.npz", **{**arrays, "meta": np.frombuffer(json.dumps(v2).encode(), np.uint8)})
    assert gbm.Booster.load(tmp_path / "v2.npz").params == booster.params
    np.savez(tmp_path / "dtype.npz", **{**arrays, "leaf_value": arrays["leaf_value"].astype(np.float64)})
    with pytest.raises(ValueError, match="dtypes"):
        gbm.Booster.load(tmp_path / "dtype.npz")


def test_the_forest_must_match_the_params(trained: tuple[gbm.Booster, TinyData]) -> None:
    booster, _ = trained
    with pytest.raises(ValueError, match="shape"):
        gbm.Booster(
            booster.params,
            Forest.empty(4, 2),
            binning="r0",
            feature_names=booster.feature_names,
            categories=booster.categories,
            named_edges=booster.named_edges,
            named_n_bins=booster.named_n_bins,
            train_rows=(0, 1),
            backend="cpu",
        )


@pytest.fixture(scope="module")
def leafwise() -> tuple[gbm.Booster, TinyData]:
    data = make_tiny(600, n_countries=6, seed=22)
    params = gbm.Params(num_trees=4, growth="leafwise", max_depth=-1, max_leaves=11, min_data_in_leaf=8,
                        bagging_fraction=0.8, bagging_freq=1)  # fmt: skip
    return gbm.train(params, data, rows=slice(0, 500), backend="cpu"), data


def test_leafwise_models_save_and_load(leafwise: tuple[gbm.Booster, TinyData], tmp_path: Path) -> None:
    booster, data = leafwise
    booster.save(tmp_path / "model.igbm")
    loaded = gbm.Booster.load(tmp_path / "model.igbm")
    assert loaded.params == booster.params
    assert isinstance(loaded.forest, NodeForest)
    for name in ("left", "right", "leaf_value"):
        np.testing.assert_array_equal(getattr(loaded.forest, name), getattr(booster.forest, name))
    np.testing.assert_array_equal(
        loaded.predict(data, rows=slice(500, 600), backend="cpu"),
        booster.predict(data, rows=slice(500, 600), backend="cpu"),
    )
    with np.load(tmp_path / "model.igbm", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    assert {"left", "right"} <= set(arrays)
    loop = arrays["right"].copy()
    loop[0, 1] = 1  # node 1 points at itself: a walk would never end
    np.savez(tmp_path / "loop.npz", **{**arrays, "right": loop})
    with pytest.raises(ValueError, match="points back up"):
        gbm.Booster.load(tmp_path / "loop.npz")
    np.savez(tmp_path / "heap.npz", **{k: v for k, v in arrays.items() if k not in ("left", "right")})
    with pytest.raises(ValueError, match="leafwise model"):
        gbm.Booster.load(tmp_path / "heap.npz")


def test_a_leafwise_model_needs_node_lists(leafwise: tuple[gbm.Booster, TinyData]) -> None:
    booster, _ = leafwise
    for forest in (Forest.empty(4, 3), NodeForest.empty(4, 12)):
        with pytest.raises(ValueError, match="node-list trees"):
            gbm.Booster(booster.params, forest, binning="r0", feature_names=booster.feature_names,
                        categories=booster.categories, named_edges=booster.named_edges,
                        named_n_bins=booster.named_n_bins, train_rows=(0, 1), backend="cpu")  # fmt: skip
