"""End to end: every learner on both of its backends, through the Numerai flow and through the generic interface.

Numerai flow: a small synthetic packed cache. Generic interface: the estimators on a plain array with codes and
categorical columns. Each trains on the first rows, predicts the rest, saves, loads and predicts again (the same bits);
then the two backends against each other: idun_gbm's CPU and Metal fits are bit-identical, the networks' Metal fits
follow their float64 references. These guard the layering that puts the Numerai layer on top of a generic one.
"""

from pathlib import Path

import numpy as np
import pytest

import idun
from idun import gbm
from idun.data import PackedDataset
from idun.gbm.tree import fields_of
from idun.realmlp import RealMLPClassifier, RealMLPModel, RealMLPParams, RealMLPRegressor, train_realmlp
from idun.tabm import TabMClassifier, TabMModel, TabMParams, TabMRegressor, train_tabm
from idun_numerai.inputs import classes as numerai_classes

TRAIN, TEST = slice(0, 200), slice(200, None)  # the synthetic cache has 277 rows
TREES = gbm.Params(num_trees=8, max_depth=3, max_leaves=8, min_data_in_leaf=5, seed=1)
LEAFWISE_TREES = gbm.Params(num_trees=8, growth="leafwise", max_depth=-1, max_leaves=12, min_data_in_leaf=5, seed=1)
GROWTH = {"depthwise": TREES, "leafwise": LEAFWISE_TREES}
TINY_TABM = TabMParams(n_classes=5, members=2, blocks=2, width=16, batch_size=32, epochs=2, seed=3, learning_rate=1e-2)
TINY_REALMLP = RealMLPParams(
    n_classes=5, blocks=2, hidden=64, frequencies=4, embed=2, batch_size=32, epochs=2, init_rows=60, seed=2
)
GBM_BACKENDS = ["cpu", pytest.param("metal", marks=pytest.mark.metal)]
NN_BACKENDS = ["reference", pytest.param("metal", marks=pytest.mark.metal)]


@pytest.fixture(scope="module")
def data(nn_cache: Path) -> PackedDataset:
    return PackedDataset.open(nn_cache, binning="r1")


@pytest.mark.parametrize("growth", GROWTH)
@pytest.mark.parametrize("backend", GBM_BACKENDS)
def test_gbm_trains_predicts_and_round_trips(data: PackedDataset, backend: str, growth: str, tmp_path: Path) -> None:
    model = gbm.train(GROWTH[growth], data, rows=TRAIN, backend=backend)
    pred = model.predict(data, rows=TEST, backend=backend)
    assert pred.shape == (data.n_rows - 200,)
    assert np.isfinite(pred).all()
    assert np.unique(pred).size > 1
    model.save(tmp_path / "model.igbm")
    np.testing.assert_array_equal(
        gbm.Booster.load(tmp_path / "model.igbm").predict(data, rows=TEST, backend=backend), pred
    )


@pytest.mark.metal
@pytest.mark.parametrize("growth", GROWTH)
def test_gbm_backends_are_bit_identical(data: PackedDataset, growth: str) -> None:
    cpu, metal = (gbm.train(GROWTH[growth], data, rows=TRAIN, backend=b) for b in ("cpu", "metal"))
    for field in fields_of(cpu.forest):
        np.testing.assert_array_equal(getattr(cpu.forest, field), getattr(metal.forest, field))
    np.testing.assert_array_equal(
        cpu.predict(data, rows=TEST, backend="cpu"), metal.predict(data, rows=TEST, backend="metal")
    )


@pytest.mark.parametrize("backend", NN_BACKENDS)
def test_tabm_trains_predicts_and_round_trips(data: PackedDataset, backend: str, tmp_path: Path) -> None:
    model = train_tabm(TINY_TABM, data, rows=TRAIN, backend=backend, classes=numerai_classes(data.target))
    pred = model.predict(data, rows=TEST, backend=backend)
    assert pred.shape == (data.n_rows - 200,)
    assert np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 4)).all()  # the expected class of 5
    model.save(tmp_path / "tabm.npz")
    np.testing.assert_array_equal(TabMModel.load(tmp_path / "tabm.npz").predict(data, rows=TEST, backend=backend), pred)


@pytest.mark.parametrize("backend", NN_BACKENDS)
def test_realmlp_trains_predicts_and_round_trips(data: PackedDataset, backend: str, tmp_path: Path) -> None:
    model = train_realmlp(TINY_REALMLP, data, rows=TRAIN, backend=backend, classes=numerai_classes(data.target))
    pred = model.predict(data, rows=TEST, backend=backend)
    assert pred.shape == (data.n_rows - 200,)
    assert np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 4)).all()
    model.save(tmp_path / "realmlp.npz")
    loaded = RealMLPModel.load(tmp_path / "realmlp.npz")
    np.testing.assert_array_equal(loaded.predict(data, rows=TEST, backend=backend), pred)


@pytest.mark.metal
def test_the_networks_metal_fits_follow_their_references(data: PackedDataset) -> None:
    labels = numerai_classes(data.target)
    for train in (lambda b: train_tabm(TINY_TABM, data, rows=TRAIN, backend=b, classes=labels),
                  lambda b: train_realmlp(TINY_REALMLP, data, rows=TRAIN, backend=b, classes=labels)):  # fmt: skip
        reference, metal = train("reference"), train("metal")
        np.testing.assert_allclose(metal.predict(data, rows=TEST, backend="metal"),
                                   reference.predict(data, rows=TEST, backend="reference"), atol=1e-3)  # fmt: skip


def generic_table(n: int = 500, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=n), rng.integers(0, 5, n), rng.integers(0, 6, n), rng.uniform(size=n)])
    x[::9, 3] = np.nan
    y = (x[:, 1] / 4 + (x[:, 2] % 2) * 0.4 + 0.2 * x[:, 0] + rng.normal(0, 0.1, n)).astype(np.float32)
    return x, y


@pytest.mark.parametrize("growth", [{"max_depth": 3, "max_leaves": 8}, {"growth": "leafwise", "max_depth": -1,
                                                                         "max_leaves": 12}])  # fmt: skip
@pytest.mark.parametrize("backend", GBM_BACKENDS)
def test_gbm_estimator_trains_predicts_and_round_trips(backend: str, growth: dict[str, object], tmp_path: Path) -> None:
    x, y = generic_table()
    model = gbm.GBMRegressor(num_trees=10, min_data_in_leaf=10, n_seeds=2, backend=backend, codes=[1],
                             categorical=[2], **growth).fit(x[:400], y[:400])  # type: ignore[arg-type]  # fmt: skip
    pred = model.predict(x[400:])
    assert pred.shape == (100,)
    assert np.corrcoef(pred, y[400:])[0, 1] > 0.6
    model.save(tmp_path / "model.idun")
    np.testing.assert_array_equal(idun.load(tmp_path / "model.idun").predict(x[400:]), pred)


@pytest.mark.metal
def test_gbm_estimator_backends_are_bit_identical() -> None:
    x, y = generic_table()
    fits = [gbm.GBMRegressor(num_trees=10, max_depth=3, max_leaves=8, min_data_in_leaf=10, backend=b, codes=[1],
                             categorical=[2]).fit(x, y) for b in ("cpu", "metal")]  # fmt: skip
    np.testing.assert_array_equal(fits[0].predict(x), fits[1].predict(x))


CLASSIFIERS = [
    (TabMClassifier, {"members": 2, "blocks": 2, "width": 16, "batch_size": 32, "epochs": 6, "learning_rate": 1e-2}),
    (RealMLPClassifier, {"blocks": 2, "hidden": 64, "frequencies": 4, "embed": 2, "batch_size": 32, "epochs": 4,
                         "init_rows": 60}),
]  # fmt: skip


@pytest.mark.parametrize("backend", NN_BACKENDS)
@pytest.mark.parametrize(("estimator", "settings"), CLASSIFIERS)
def test_network_classifiers_train_predict_and_round_trip(
    estimator: type, settings: dict[str, object], backend: str, tmp_path: Path
) -> None:
    x, y = generic_table()
    labels = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3]))  # three classes
    model = estimator(**settings, backend=backend, categorical=[2]).fit(x[:400], labels[:400])
    proba = model.predict_proba(x[400:])
    assert proba.shape == (100, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (model.predict(x[400:]) == labels[400:]).mean() > 0.5  # chance is a third
    model.save(tmp_path / "model.idun")
    np.testing.assert_array_equal(idun.load(tmp_path / "model.idun").predict_proba(x[400:]), proba)


@pytest.mark.parametrize("backend", NN_BACKENDS)
@pytest.mark.parametrize(
    ("estimator", "settings"), [(TabMRegressor, CLASSIFIERS[0][1]), (RealMLPRegressor, CLASSIFIERS[1][1])]
)
def test_network_regressors_train_predict_and_round_trip(
    estimator: type, settings: dict[str, object], backend: str, tmp_path: Path
) -> None:
    x, y = generic_table()
    model = estimator(**settings, backend=backend, categorical=[2]).fit(x[:400], y[:400])
    pred = model.predict(x[400:])
    assert pred.shape == (100,)
    assert np.corrcoef(pred, y[400:])[0, 1] > 0.6
    model.save(tmp_path / "model.idun")
    np.testing.assert_array_equal(idun.load(tmp_path / "model.idun").predict(x[400:]), pred)
