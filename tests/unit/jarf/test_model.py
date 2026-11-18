from pathlib import Path

import numpy as np
import pytest

from idun import gbm
from idun.data import PackedDataset
from idun.jarf import JARFModel, front
from idun.nn import inputs
from idun.tabm import directions


def random_directions(d: int, k: int, seed: int = 1) -> directions.Directions:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    eigenvalues = np.sort(np.random.default_rng(seed + 1).uniform(size=d))[::-1]
    return directions.Directions(q.astype(np.float32), eigenvalues, ("s0",))


PARAMS = gbm.Params(num_trees=3, max_depth=2, max_leaves=4, min_data_in_leaf=5)


def fitted(data: PackedDataset, kept: tuple[str, ...] | None) -> tuple[JARFModel, front.Front]:
    found = random_directions(inputs.n_inputs(data), 2)
    extended = front.extend(data, found, edge_rows=np.arange(0, data.n_rows, 3))
    made = extended if kept is None else front.keep(extended, kept)
    booster = gbm.train(PARAMS, made, rows=slice(0, data.n_rows), backend="cpu")
    return JARFModel(booster, found, kept), made


def test_the_model_rebuilds_its_front_and_predicts(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    model, made = fitted(data, None)
    assert model.n_named == len(data.named_features)
    rebuilt = model.front(data)
    assert rebuilt.binning == made.binning
    assert rebuilt.feature_names == made.feature_names
    np.testing.assert_array_equal(rebuilt.named_bins, made.named_bins)
    np.testing.assert_array_equal(rebuilt.named_edges, made.named_edges)
    assert rebuilt.named_n_bins == made.named_n_bins
    rows = slice(0, data.n_rows)
    np.testing.assert_array_equal(model.predict(data, rows=rows, backend="cpu"),
                                  model.booster.predict(made, rows=rows, backend="cpu"))  # fmt: skip


def test_a_kept_subset_is_rebuilt_too(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    kept = (data.obfuscated_features[0], data.obfuscated_features[3])
    model, made = fitted(data, kept)
    rebuilt = model.front(data)
    assert rebuilt.obfuscated_features == kept
    assert rebuilt.binning == made.binning
    np.testing.assert_array_equal(rebuilt.obfuscated, made.obfuscated)
    np.testing.assert_array_equal(model.predict(data, rows=slice(0, 7), backend="cpu"),
                                  model.booster.predict(made, rows=slice(0, 7), backend="cpu"))  # fmt: skip


def test_save_and_load(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    model, _ = fitted(data, (data.obfuscated_features[1],))
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = JARFModel.load(path)
    assert loaded.kept == model.kept
    assert loaded.prefix == model.prefix
    assert loaded.directions.surrogates == model.directions.surrogates
    np.testing.assert_array_equal(loaded.directions.vectors, model.directions.vectors)
    assert loaded.booster.feature_names == model.booster.feature_names
    np.testing.assert_array_equal(loaded.predict(data, rows=slice(0, 9), backend="cpu"),
                                  model.predict(data, rows=slice(0, 9), backend="cpu"))  # fmt: skip
    model.directions.save(tmp_path / "d.npz")
    with pytest.raises(ValueError, match="idun-jarf"):
        JARFModel.load(tmp_path / "d.npz")
    np.savez(tmp_path / "x.npz", a=np.zeros(1))
    with pytest.raises(ValueError, match="idun-jarf"):
        JARFModel.load(tmp_path / "x.npz")


def test_the_booster_must_hold_the_direction_columns(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    plain = gbm.train(PARAMS, data, rows=slice(0, data.n_rows), backend="cpu")
    with pytest.raises(ValueError, match="direction columns"):
        JARFModel(plain, random_directions(inputs.n_inputs(data), 2))
