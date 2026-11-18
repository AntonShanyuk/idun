from pathlib import Path
from typing import Any

import numpy as np
import pytest

import idun
from idun.nn.inputs import MatrixData
from idun.nn.table import TableEncoder
from idun.tabm import TabMClassifier, TabMParams, TabMRegressor, Validation, train_tabm

TINY: dict[str, Any] = {"members": 2, "blocks": 1, "width": 16, "batch_size": 32, "epochs": 4, "learning_rate": 1e-2}
GPU = [pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]  # each, marked


def table(n: int = 240, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=n), rng.integers(0, 4, n), rng.uniform(size=n)])
    y = (x[:, 0] + 0.5 * (x[:, 1] == 3) > 0.2).astype(np.int64) + (x[:, 2] > 0.8)
    return x, y


def test_the_estimator_gives_the_functional_layers_bits() -> None:
    x, y = table()
    model = TabMClassifier(**TINY, backend="reference", categorical=[1]).fit(x, y)
    encoder = TableEncoder(categorical=[1], standardize=TabMClassifier.standardize).fit(x)
    data = MatrixData(encoder.transform(x), y, 3)
    fit = train_tabm(TabMParams(**TINY, n_classes=3, seed=0), data, rows=slice(0, 240), backend="reference")
    np.testing.assert_array_equal(
        model.predict_proba(x), fit.predict_proba(data, rows=slice(0, 240), backend="reference")
    )
    assert model.get_params()["n_seeds"] == 1
    assert model.n_features_in_ == 3


@pytest.mark.parametrize("gpu_backend", GPU)
def test_gpu_follows_the_reference(gpu_backend: str) -> None:
    x, y = table()
    reference, metal = (TabMClassifier(**TINY, backend=b).fit(x, y) for b in ("reference", gpu_backend))
    np.testing.assert_allclose(metal.predict_proba(x), reference.predict_proba(x), atol=1e-3)


def test_the_regressor_gives_the_functional_layers_bits_scaled_back() -> None:
    x, _ = table()
    target = (x[:, 0] * 3.0 + 10.0).astype(np.float64)
    model = TabMRegressor(**TINY, backend="reference", categorical=[1]).fit(x, target)
    assert model.target_mean_ == pytest.approx(target.mean())
    encoder = TableEncoder(categorical=[1], standardize=TabMRegressor.standardize).fit(x)
    scaled = ((target - target.mean()) / target.std()).astype(np.float32)
    data = MatrixData(encoder.transform(x), scaled, 1)
    fit = train_tabm(
        TabMParams(**TINY, n_classes=1, objective="mse", seed=0), data, rows=slice(0, 240), backend="reference"
    )
    want = fit.predict(data, rows=slice(0, 240), backend="reference") * target.std() + target.mean()
    np.testing.assert_allclose(model.predict(x), want, rtol=1e-6)


@pytest.mark.parametrize("gpu_backend", GPU)
def test_the_gpu_regressor_follows_the_reference(gpu_backend: str) -> None:
    x, _ = table()
    target = x[:, 0] - x[:, 2]
    reference, metal = (TabMRegressor(**TINY, backend=b).fit(x, target) for b in ("reference", gpu_backend))
    np.testing.assert_allclose(metal.predict(x), reference.predict(x), atol=1e-2 * float(target.std()))


def test_the_hl_gauss_regressor_gives_the_functional_layers_bits_scaled_back() -> None:
    x, _ = table()
    target = (x[:, 0] * 3.0 + 10.0).astype(np.float64)
    hl: dict[str, Any] = {"objective": "hl_gauss", "n_bins": 32, "hl_low": -3.0, "hl_high": 3.0, "hl_sigma": 0.15}
    model = TabMRegressor(**TINY, **hl, backend="reference", categorical=[1]).fit(x, target)
    assert {k: model.get_params()[k] for k in hl} == hl
    encoder = TableEncoder(categorical=[1], standardize=TabMRegressor.standardize).fit(x)
    scaled = ((target - target.mean()) / target.std()).astype(np.float32)
    data = MatrixData(encoder.transform(x), scaled, 1)
    params = TabMParams(**TINY, **hl, n_classes=1, seed=0)
    fit = train_tabm(params, data, rows=slice(0, 240), backend="reference")
    want = fit.predict(data, rows=slice(0, 240), backend="reference") * target.std() + target.mean()
    np.testing.assert_allclose(model.predict(x), want, rtol=1e-6)
    assert getattr(model.models_[0], "params", None) == params
    assert "objective" not in TabMClassifier().get_params()
    with pytest.raises(ValueError, match="objective"):
        TabMRegressor(**TINY, objective="hinge", backend="reference").fit(x, target)


def test_an_eval_set_keeps_each_networks_best_epoch(tmp_path: Path) -> None:
    x, y = table(320, seed=4)
    labels = np.where(y > np.median(y), "hi", "lo")
    model = TabMClassifier(**{**TINY, "epochs": 6}, backend="reference", patience=2)
    model.fit(x[:240], labels[:240], eval_set=(x[240:], labels[240:]))
    best = model.best_epoch_
    assert isinstance(best, int)
    scores = model.valid_score_
    assert scores[best] == min(scores)
    encoder = TableEncoder(standardize=True).fit(x[:240])
    classes = np.unique(labels[:240])
    data = MatrixData(encoder.transform(x[:240]), np.searchsorted(classes, labels[:240]).astype(np.int64), 2)
    valid = Validation(MatrixData(encoder.transform(x[240:]), np.searchsorted(classes, labels[240:]).astype(np.int64),
                                  2), slice(0, 80))  # fmt: skip
    alone = train_tabm(model.network_params(0, 2), data, rows=slice(0, 240), backend="reference", valid=valid,
                       patience=2)  # fmt: skip
    assert alone.report.valid_score == scores
    model.save(tmp_path / "m.npz")
    loaded = idun.load(tmp_path / "m.npz")
    assert (loaded.best_epoch_, loaded.valid_score_) == (best, scores)
    with pytest.raises(ValueError, match="one validation set"):
        TabMClassifier(**TINY, backend="reference").fit(x, labels, eval_set=[(x, labels), (x, labels)])
    with pytest.raises(ValueError, match="labels outside the training classes"):
        TabMClassifier(**TINY, backend="reference").fit(x, labels, eval_set=(x, np.full(x.shape[0], "mid")))
    regressor = TabMRegressor(**TINY, backend="reference").fit(x[:240], y[:240], eval_set=[(x[240:], y[240:])])
    assert len(regressor.valid_score_) == TINY["epochs"]
    unvalidated = TabMRegressor(**TINY, backend="reference").fit(x, y)
    assert (unvalidated.best_epoch_, unvalidated.valid_score_) == (None, [])
