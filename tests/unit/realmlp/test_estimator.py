from typing import Any

import numpy as np
import pytest

from idun.nn.inputs import MatrixData
from idun.nn.table import TableEncoder
from idun.realmlp import RealMLPClassifier, RealMLPParams, RealMLPRegressor, train_realmlp

TINY: dict[str, Any] = {
    "blocks": 1,
    "hidden": 64,
    "frequencies": 4,
    "embed": 2,
    "batch_size": 32,
    "epochs": 3,
    "init_rows": 60,
}
GPU = [pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]  # each, marked


def table(n: int = 240, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=n), rng.integers(0, 4, n), rng.uniform(size=n)])
    y = (x[:, 0] + 0.5 * (x[:, 1] == 3) > 0.2).astype(np.int64) + (x[:, 2] > 0.8)
    return x, y


def test_the_estimator_gives_the_functional_layers_bits() -> None:
    x, y = table()
    model = RealMLPClassifier(**TINY, backend="reference", categorical=[1]).fit(x, y)
    encoder = TableEncoder(categorical=[1], standardize=RealMLPClassifier.standardize).fit(x)
    data = MatrixData(encoder.transform(x), y, 3)
    fit = train_realmlp(RealMLPParams(**TINY, n_classes=3, seed=0), data, rows=slice(0, 240), backend="reference")
    np.testing.assert_array_equal(
        model.predict_proba(x), fit.predict_proba(data, rows=slice(0, 240), backend="reference")
    )
    assert model.get_params()["n_seeds"] == 1
    assert model.n_features_in_ == 3


@pytest.mark.parametrize("gpu_backend", GPU)
def test_gpu_follows_the_reference(gpu_backend: str) -> None:
    x, y = table()
    reference, metal = (RealMLPClassifier(**TINY, backend=b).fit(x, y) for b in ("reference", gpu_backend))
    np.testing.assert_allclose(metal.predict_proba(x), reference.predict_proba(x), atol=1e-3)


def test_the_regressor_gives_the_functional_layers_bits_scaled_back() -> None:
    x, _ = table()
    target = (x[:, 0] * 3.0 + 10.0).astype(np.float64)
    model = RealMLPRegressor(**TINY, backend="reference", categorical=[1]).fit(x, target)
    assert model.target_mean_ == pytest.approx(target.mean())
    encoder = TableEncoder(categorical=[1], standardize=RealMLPRegressor.standardize).fit(x)
    scaled = ((target - target.mean()) / target.std()).astype(np.float32)
    data = MatrixData(encoder.transform(x), scaled, 1)
    fit = train_realmlp(
        RealMLPParams(**TINY, n_classes=1, objective="mse", seed=0), data, rows=slice(0, 240), backend="reference"
    )
    want = fit.predict(data, rows=slice(0, 240), backend="reference") * target.std() + target.mean()
    np.testing.assert_allclose(model.predict(x), want, rtol=1e-6)


@pytest.mark.parametrize("gpu_backend", GPU)
def test_the_gpu_regressor_follows_the_reference(gpu_backend: str) -> None:
    x, _ = table()
    target = x[:, 0] - x[:, 2]
    reference, metal = (RealMLPRegressor(**TINY, backend=b).fit(x, target) for b in ("reference", gpu_backend))
    np.testing.assert_allclose(metal.predict(x), reference.predict(x), atol=1e-2 * float(target.std()))


def test_fewer_rows_than_a_batch_train_on_one_batch_of_every_row() -> None:
    x, y = table(n=40)
    model = RealMLPClassifier(**TINY, backend="reference").fit(x, y)  # 40 rows: one batch of 32 per epoch
    small = {**TINY, "batch_size": 64}
    wide = RealMLPClassifier(**small, backend="reference").fit(x, y)
    encoder = TableEncoder(categorical=(), standardize=False).fit(x)
    data = MatrixData(encoder.transform(x), np.searchsorted(np.unique(y), y).astype(np.int64), 3)
    params = RealMLPParams(**{**small, "batch_size": 40}, n_classes=3, seed=0)
    fit = train_realmlp(params, data, rows=slice(0, 40), backend="reference")
    np.testing.assert_array_equal(
        wide.predict_proba(x), fit.predict_proba(data, rows=slice(0, 40), backend="reference")
    )
    assert model.predict(x).shape == (40,)


def test_the_activation_and_the_widths_are_parameters() -> None:
    model = RealMLPRegressor(act="mish", hidden_sizes=[64, 32], backend="reference")
    assert model.get_params()["act"] == "mish"
    assert RealMLPRegressor().get_params()["act"] == "selu"
    assert model.network_params(0, 1).widths == (64, 32)
    assert RealMLPRegressor().network_params(0, 1).widths == (256, 256, 256)
