from pathlib import Path
from typing import Any

import numpy as np
import pytest

import idun
from idun.errors import NotFittedError
from idun.realmlp import RealMLPClassifier, RealMLPRegressor
from idun.realmlp.train import RealMLPModel
from idun.tabm import TabMClassifier, TabMRegressor
from idun.tabm.train import TabMModel

TINY: dict[str, Any] = {"members": 2, "blocks": 1, "width": 16, "batch_size": 32, "epochs": 6, "learning_rate": 1e-2}


def table(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=n), rng.integers(0, 3, n), rng.normal(size=n)])
    y = np.where(x[:, 0] + (x[:, 1] == 2) > 0.5, "up", np.where(x[:, 0] < -0.5, "down", "flat"))
    return x, y


def test_labels_map_to_classes_and_back() -> None:
    x, y = table()
    model = TabMClassifier(**TINY, backend="reference", categorical=[1]).fit(x, y)
    assert list(model.classes_) == ["down", "flat", "up"]
    proba = model.predict_proba(x)
    assert proba.shape == (300, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert set(model.predict(x)) <= {"down", "flat", "up"}
    assert (model.predict(x) == y).mean() > 0.6


def test_a_seed_bag_averages_its_networks(tmp_path: Path) -> None:
    x, y = table()
    bag = TabMClassifier(**TINY, backend="reference", n_seeds=2, seed=5).fit(x, y)
    alone = [TabMClassifier(**TINY, backend="reference", seed=s).fit(x, y) for s in (5, 6)]
    np.testing.assert_allclose(bag.predict_proba(x), np.mean([m.predict_proba(x) for m in alone], axis=0), rtol=1e-6)
    each = bag.predict_proba(x, per_seed=True)
    assert each.shape == (300, 2, 3)
    assert each.dtype == np.dtype(np.float32)
    for i, model in enumerate(alone):
        np.testing.assert_array_equal(each[:, i], model.predict_proba(x))
    bag.save(tmp_path / "bag.idun")
    loaded = idun.load(tmp_path / "bag.idun")
    assert isinstance(loaded, TabMClassifier)
    np.testing.assert_array_equal(loaded.predict_proba(x), bag.predict_proba(x))
    assert list(loaded.classes_) == list(bag.classes_)


def test_bad_inputs_fail_loudly() -> None:
    x, y = table()
    with pytest.raises(ValueError, match="two classes"):
        TabMClassifier(**TINY, backend="reference").fit(x, np.zeros(300))
    with pytest.raises(ValueError, match="1-D labels"):
        TabMClassifier(**TINY, backend="reference").fit(x, y[:, None])
    with pytest.raises(ValueError, match="rows"):
        TabMClassifier(**TINY, backend="reference").fit(x, y[:-1])
    with pytest.raises(ValueError, match="n_seeds"):
        TabMClassifier(**TINY, backend="reference", n_seeds=0).fit(x, y)
    with pytest.raises(NotFittedError, match="not fitted"):
        TabMClassifier().predict(x)
    with pytest.raises(ValueError, match="Unknown label type: continuous"):
        TabMClassifier(**TINY, backend="reference").fit(x, x[:, 0])
    with pytest.raises(ValueError, match="got 1 class"):
        TabMClassifier(**TINY, backend="reference").fit(x[:1], y[:1])
    with pytest.raises(ValueError, match="requires y to be passed"):
        TabMClassifier(**TINY, backend="reference").fit(x, None)
    with pytest.raises(ValueError, match="Complex data"):
        TabMClassifier(**TINY, backend="reference").fit(x, np.arange(300) % 2 + 1j)
    with pytest.raises(ValueError, match="Complex data"):
        TabMRegressor(**TINY, backend="reference").fit(x, x[:, 0] + 1j)
    with pytest.raises(ValueError, match="requires y to be passed"):
        TabMRegressor(**TINY, backend="reference").fit(x, None)
    fitted = TabMClassifier(**{**TINY, "epochs": 1}, backend="reference").fit(x, y)
    with pytest.raises(ValueError, match="X has 2 features, but TabMClassifier is expecting 3"):
        fitted.predict(x[:, :2])
    assert TabMClassifier(**TINY, backend="reference").fit(x, (x[:, 0] > 0).astype(float)).classes_.tolist() == [0, 1]


def test_a_regressor_scales_its_target_and_round_trips(tmp_path: Path) -> None:
    x, _ = table()
    target = 50.0 + 5.0 * x[:, 0]
    model = TabMRegressor(**TINY, backend="reference", n_seeds=2).fit(x, target)
    pred = model.predict(x)
    assert np.corrcoef(pred, target)[0, 1] > 0.9
    assert abs(float(pred.mean()) - 50.0) < 2.0  # back in the target's units
    each = model.predict(x, per_seed=True)
    assert each.shape == (300, 2)
    assert each.dtype == np.dtype(np.float32)
    alone = TabMRegressor(**TINY, backend="reference", seed=1).fit(x, target)
    np.testing.assert_array_equal(each[:, 1], alone.predict(x))  # seed 1, the bag's second network
    np.testing.assert_allclose(each.mean(axis=1), pred, rtol=0, atol=1e-4)
    model.save(tmp_path / "reg.idun")
    loaded = idun.load(tmp_path / "reg.idun")
    assert isinstance(loaded, TabMRegressor)
    np.testing.assert_array_equal(loaded.predict(x), pred)
    with pytest.raises(ValueError, match="finite targets"):
        TabMRegressor(**TINY, backend="reference").fit(x, np.full(300, np.nan))
    with pytest.raises(ValueError, match="not fitted"):
        TabMRegressor().predict(x)


@pytest.mark.parametrize("estimator", ["TabMClassifier", "TabMRegressor", "RealMLPClassifier", "RealMLPRegressor"])
def test_sample_weight_reaches_the_networks(estimator: str) -> None:
    x, labels = table()
    regressor = estimator.endswith("Regressor")
    y: Any = x[:, 0] * 2 + x[:, 2] if regressor else labels
    settings: dict[str, Any] = TINY if estimator.startswith("TabM") else {"blocks": 1, "hidden": 16, "epochs": 4}
    make = {"TabMClassifier": TabMClassifier, "TabMRegressor": TabMRegressor, "RealMLPClassifier": RealMLPClassifier,
            "RealMLPRegressor": RealMLPRegressor}[estimator]  # fmt: skip
    plain = make(**settings, backend="reference").fit(x, y)
    ones = make(**settings, backend="reference").fit(x, y, np.ones(300))
    weight = np.where(x[:, 0] > 0, 3.0, 0.5)
    weighted = make(**settings, backend="reference").fit(x, y, sample_weight=weight)

    def found(model: Any) -> np.ndarray:
        return np.asarray(model.predict(x) if regressor else model.predict_proba(x))

    np.testing.assert_allclose(found(ones), found(plain), rtol=1e-5, atol=1e-6)
    assert not np.allclose(found(weighted), found(plain), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="sample_weight must hold one weight per row"):
        make(**settings, backend="reference").fit(x, y, np.ones(299))
    with pytest.raises(ValueError, match="non-negative"):
        make(**settings, backend="reference").fit(x, y, -np.ones(300))


def test_tabm_embeds_the_numeric_columns() -> None:
    x, y = table()
    model = TabMClassifier(**TINY, backend="reference", categorical=[1], num_emb_type="pwl", num_emb_n_bins=8,
                           d_embedding=4).fit(x, y)  # fmt: skip
    network = model.models_[0]
    assert isinstance(network, TabMModel)
    bins = network.weights.bins
    assert bins is not None
    assert bins.scale.shape[0] == 2  # columns 0 and 2; the categorical column's one-hot block enters as it is
    assert network.weights.weight[0].shape[1] == 2 * 4 + 3
    assert (model.predict(x) == y).mean() > 0.6
    with pytest.raises(ValueError, match="X has none"):
        TabMClassifier(**TINY, backend="reference", categorical=[0], num_emb_type="pwl").fit(x[:, 1:2], y)


def test_realmlp_embeds_wide_categorical_columns_as_pytabkit() -> None:
    rng = np.random.default_rng(2)
    x = np.column_stack([rng.normal(size=300), rng.integers(0, 12, 300), rng.integers(0, 3, 300)]).astype(np.float64)
    y = np.where(x[:, 0] + (x[:, 1] > 6) > 0.5, "up", "down")
    settings: dict[str, Any] = {"blocks": 1, "hidden": 64, "epochs": 30, "batch_size": 32, "backend": "reference",
                                "categorical": [1, 2]}  # fmt: skip
    model = RealMLPClassifier(**settings, max_one_hot_cat_size=9, embedding_size=4).fit(x, y)
    network = model.models_[0]
    assert isinstance(network, RealMLPModel)
    assert network.weights.categories == (12,)  # 12 levels embedded; 3 levels (4 with pytabkit's missing one) one-hot
    assert network.n_inputs == 1 + 3 + 1
    assert (model.predict(x) == y).mean() > 0.9  # observed 0.993; one-hot 0.997
    plain = RealMLPClassifier(**settings).fit(x, y)  # max_one_hot_cat_size=-1: every column one-hot
    plain_network = plain.models_[0]
    assert isinstance(plain_network, RealMLPModel)
    assert plain_network.weights.categories is None
    assert plain_network.n_inputs == 1 + 12 + 3
    with pytest.raises(ValueError, match="max_one_hot_cat_size"):
        RealMLPClassifier(**settings, max_one_hot_cat_size=-2).fit(x, y)
