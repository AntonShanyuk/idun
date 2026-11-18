from pathlib import Path
from typing import Any

import numpy as np
import pytest

import idun
from idun.dataset import Dataset
from idun.errors import NotFittedError
from idun.gbm import GBMRegressor
from idun.jarf import JARFRegressor
from idun.jarf import estimator as jarf
from idun.nn.inputs import MatrixData
from idun.tabm import TabMClassifier, directions
from tests.unit.gbm.backends import emulated

TINY_TABM: dict[str, Any] = {"members": 2, "blocks": 1, "width": 8, "epochs": 2, "batch_size": 64}
SETTINGS: dict[str, Any] = {
    "num_trees": 4, "max_depth": 2, "max_leaves": 4, "min_data_in_leaf": 5, "backend": "cpu",
    "surrogate_params": TINY_TABM,
    "surrogate_backend": "reference", "n_surrogates": 2, "surrogate_classes": 5, "k": 2, "egop_rows": 50,
    "categorical": ["d"], "codes": ["c"],
}  # fmt: skip
COLUMNS = ["a", "b", "c", "d"]


def table(n: int = 240, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=n), rng.uniform(size=n), rng.integers(0, 5, n), rng.integers(0, 3, n)])
    y = (x[:, 2] / 4 + 0.5 * (x[:, 3] == 1) + 0.3 * x[:, 0] + rng.normal(0, 0.1, n)).astype(np.float32)
    return x.astype(np.float64), y


def frame(x: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [(name, x[:, i]) for i, name in enumerate(COLUMNS)]


def test_quantile_classes() -> None:
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.5])
    np.testing.assert_array_equal(jarf.quantile_classes(values, 5), [0, 1, 2, 3, 4, 2])
    continuous = np.random.default_rng(1).normal(size=1000)
    classes = jarf.quantile_classes(continuous, 4)
    assert classes.dtype == np.int64
    assert set(classes) == {0, 1, 2, 3}
    assert abs(np.bincount(classes) - 250).max() <= 1
    with pytest.raises(ValueError, match="at least 2"):
        jarf.quantile_classes(values, 1)


def test_the_estimator_gives_the_functional_path_bits() -> None:
    x, y = table()
    model = JARFRegressor(**SETTINGS).fit(frame(x), y)
    pred = model.predict(frame(x))
    assert pred.shape == (240,)
    assert model.directions_.k == 2
    assert model.kept_ == ("a", "b", "c", "d")
    assert model.n_features_in_ == 4
    assert list(model.feature_names_in_) == COLUMNS
    assert np.corrcoef(pred, y)[0, 1] > 0.5
    # the same by hand
    labels = jarf.quantile_classes(y.astype(np.float64), 5)
    surrogates = [TabMClassifier(**TINY_TABM, seed=s, backend="reference", categorical=["d"]).fit(frame(x), labels)
                  for s in (0, 1)]  # fmt: skip
    encoder = surrogates[0].encoder_
    inputs = encoder.transform(frame(x))
    rows = np.sort(np.random.default_rng(0).choice(240, 50, replace=False))
    models: list[Any] = [m for s in surrogates for m in s.models_]
    h = directions.egop(models, MatrixData(inputs, np.zeros(240, np.float32), 1), rows)
    vectors, eigenvalues = directions.top_directions(h, 2)
    found = directions.Directions(vectors.astype(np.float32), eigenvalues, ())
    np.testing.assert_array_equal(model.directions_.vectors, found.vectors)
    np.testing.assert_array_equal(model.sensitivity_, np.diagonal(h))
    projected = found.project(inputs)
    columns = [*frame(x), ("tabm_direction_0", projected[:, 0]), ("tabm_direction_1", projected[:, 1])]
    data = Dataset(columns, y, categorical=["d"], codes=["c"])
    trees = GBMRegressor(num_trees=4, max_depth=2, max_leaves=4, min_data_in_leaf=5, backend="cpu").fit(data)
    np.testing.assert_array_equal(trees.predict(data), pred)
    np.testing.assert_array_equal(model.predict(frame(x), per_seed=True)[:, 0], pred)


def test_fitted_surrogates_are_taken_as_they_are() -> None:
    x, y = table()
    labels = jarf.quantile_classes(y.astype(np.float64), 5)
    surrogates = [TabMClassifier(**TINY_TABM, seed=s, backend="reference", categorical=["d"]).fit(frame(x), labels)
                  for s in (0, 1)]  # fmt: skip
    own = JARFRegressor(**SETTINGS).fit(frame(x), y)
    given = JARFRegressor(**SETTINGS, surrogates=surrogates).fit(frame(x), y)
    np.testing.assert_array_equal(given.directions_.vectors, own.directions_.vectors)
    assert given.directions_.surrogates == own.directions_.surrogates
    np.testing.assert_array_equal(given.predict(frame(x)), own.predict(frame(x)))
    with pytest.raises(ValueError, match="fitted TabMClassifiers"):
        JARFRegressor(**SETTINGS, surrogates=[TabMClassifier(**TINY_TABM)]).fit(frame(x), y)
    other = TabMClassifier(**TINY_TABM, backend="reference").fit(x[:, :3], labels)  # another table
    with pytest.raises(ValueError, match="different tables"):
        JARFRegressor(**SETTINGS, surrogates=[surrogates[0], other]).fit(frame(x), y)


def test_keep_drops_the_least_sensitive_columns() -> None:
    x, y = table()
    model = JARFRegressor(**SETTINGS, keep=2).fit(frame(x), y)
    assert len(model.kept_) == 3  # two numeric or codes columns and the categorical one
    assert "d" in model.kept_
    numeric = model.encoder_.numeric_
    strongest = {numeric[i] for i in np.argsort(-model.sensitivity_[: len(numeric)], kind="stable")[:2]}
    assert set(model.kept_) == strongest | {"d"}
    assert model.gbm_.template_.columns == (*model.kept_, "tabm_direction_0", "tabm_direction_1")
    assert model.predict(frame(x)).shape == (240,)
    with pytest.raises(ValueError, match="keep must be"):
        JARFRegressor(**SETTINGS, keep=4).fit(frame(x), y)


def test_save_and_idun_load_give_the_same_predictions(tmp_path: Path) -> None:
    x, y = table()
    model = JARFRegressor(**SETTINGS).fit(frame(x), y)
    model.save(tmp_path / "jarf.npz")
    loaded = idun.load(tmp_path / "jarf.npz")
    assert isinstance(loaded, JARFRegressor)
    assert loaded.get_params() == model.get_params()
    assert loaded.kept_ == model.kept_
    assert loaded.directions_.surrogates == model.directions_.surrogates
    np.testing.assert_array_equal(loaded.directions_.vectors, model.directions_.vectors)
    np.testing.assert_array_equal(loaded.sensitivity_, model.sensitivity_)
    np.testing.assert_array_equal(loaded.predict(frame(x)), model.predict(frame(x)))
    plain = GBMRegressor(num_trees=2, max_depth=2, max_leaves=4, backend="cpu").fit(x, y)  # a format-1 file loads
    plain.save(tmp_path / "gbm.npz")
    np.testing.assert_array_equal(idun.load(tmp_path / "gbm.npz").predict(x), plain.predict(x))


def test_fit_and_predict_refuse_what_they_cannot_use() -> None:
    x, y = table()
    model = JARFRegressor(**SETTINGS)
    with pytest.raises(NotFittedError):
        model.predict(frame(x))
    with pytest.raises(ValueError, match="requires y"):
        model.fit(frame(x))
    with pytest.raises(ValueError, match="finite"):
        model.fit(frame(x), np.where(np.arange(240) == 3, np.nan, y))
    with pytest.raises(ValueError, match="tabm_direction_"):
        model.fit([*frame(x), ("tabm_direction_0", y)], y)
    with pytest.raises(ValueError, match="n_surrogates"):
        JARFRegressor(**SETTINGS | {"n_surrogates": 0}).fit(frame(x), y)
    with pytest.raises(ValueError, match="egop_rows"):
        JARFRegressor(**SETTINGS | {"egop_rows": 0}).fit(frame(x), y)
    capped = JARFRegressor(**SETTINGS | {"k": 99}).fit(frame(x), y)  # k at most the encoder's inputs
    assert capped.directions_.k == capped.encoder_.n_inputs
    with pytest.raises(ValueError, match="k must be"):
        JARFRegressor(**SETTINGS | {"k": 0}).fit(frame(x), y)
    fitted = model.fit(frame(x), y)
    with pytest.raises(ValueError, match="expecting 4 features"):
        fitted.predict(x[:, :3])


def test_params_follow_scikit_learns_conventions() -> None:
    model = JARFRegressor(k=8, keep=3)
    params = model.get_params()
    assert params["k"] == 8
    assert params["keep"] == 3
    assert params["num_trees"] == 100
    assert params["surrogates"] is None
    assert model.set_params(k=32).k == 32
    assert repr(model) == "JARFRegressor(k=32, keep=3)"
    with pytest.raises(ValueError, match="unknown parameters"):
        model.set_params(nothing=1)


@pytest.mark.cuda
def test_on_cuda_jarf_runs_whole_and_its_trees_are_the_cpus_given_the_same_surrogates() -> None:
    x, y = table()
    whole = JARFRegressor(**(SETTINGS | {"backend": "cuda", "surrogate_backend": "cuda"})).fit(frame(x), y)
    pred = whole.predict(frame(x))
    assert pred.shape == (240,)
    assert np.isfinite(pred).all()
    labels = jarf.quantile_classes(y.astype(np.float64), 5)
    surrogates = [TabMClassifier(**TINY_TABM, seed=s, backend="cuda", categorical=["d"]).fit(frame(x), labels)
                  for s in (0, 1)]  # fmt: skip
    on_gpu = JARFRegressor(**(SETTINGS | {"backend": "cuda"}), surrogates=surrogates).fit(frame(x), y)
    on_cpu = JARFRegressor(**SETTINGS, surrogates=surrogates).fit(frame(x), y)
    np.testing.assert_array_equal(on_gpu.directions_.vectors, on_cpu.directions_.vectors)
    np.testing.assert_array_equal(on_gpu.predict(frame(x)), on_cpu.predict(frame(x)))  # idun_gbm's bits are the CPU's
    np.testing.assert_array_equal(on_gpu.predict(frame(x)), pred)  # the whole fit's surrogates are these, run to run


@pytest.mark.metal
def test_on_the_emulated_cuda_jarfs_trees_are_the_cpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """idun_gbm's part on the CUDA device over the fake driver (the Metal kernels in the CUDA ones' place)."""
    x, y = table()
    labels = jarf.quantile_classes(y.astype(np.float64), 5)
    surrogates = [TabMClassifier(**TINY_TABM, seed=s, backend="metal", categorical=["d"]).fit(frame(x), labels)
                  for s in (0, 1)]  # fmt: skip
    emulation = emulated.install(monkeypatch)
    try:
        on_gpu = JARFRegressor(**(SETTINGS | {"backend": "cuda"}), surrogates=surrogates).fit(frame(x), y)
        on_cpu = JARFRegressor(**SETTINGS, surrogates=surrogates).fit(frame(x), y)
        np.testing.assert_array_equal(on_gpu.predict(frame(x)), on_cpu.predict(frame(x)))
    finally:
        emulation.close()
