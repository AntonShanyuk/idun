from pathlib import Path
from typing import Any

import numpy as np
import pytest

import idun
from idun import gbm
from idun.dataset import Dataset
from idun.gbm import GBMRegressor

SMALL: dict[str, Any] = {"num_trees": 6, "max_depth": 3, "max_leaves": 8, "min_data_in_leaf": 10, "seed": 3}


def table(n: int = 700, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.normal(size=n), rng.integers(0, 5, n), rng.integers(0, 4, n), rng.uniform(size=n)])
    x[::11, 0] = np.nan
    y = (x[:, 1] / 4 + (x[:, 2] == 2) * 0.5 + np.nan_to_num(x[:, 0]) * 0.2 + rng.normal(0, 0.1, n)).astype(np.float32)
    return x, y


def test_the_estimator_gives_the_functional_layers_bits() -> None:
    x, y = table()
    model = GBMRegressor(**SMALL, backend="cpu", codes=[1], categorical=[2]).fit(x[:500], y[:500])
    data = Dataset(x[:500], y[:500], codes=[1], categorical=[2])
    booster = gbm.train(model.params(), data, rows=slice(0, 500), backend="cpu")
    test = data.transform(x[500:])
    np.testing.assert_array_equal(model.predict(x[500:]), booster.predict(test, rows=slice(0, 200), backend="cpu"))
    np.testing.assert_array_equal(model.predict(test), model.predict(x[500:]))
    assert model.n_features_in_ == 4
    assert list(model.feature_names_in_) == ["x0", "x1", "x2", "x3"]
    assert np.corrcoef(model.predict(x[500:]), y[500:])[0, 1] > 0.6


def test_a_seed_bag_predicts_the_mean_of_its_separate_fits_or_each_ones() -> None:
    x, y = table()
    bag = GBMRegressor(**SMALL, backend="cpu", n_seeds=3, bagging_fraction=0.7, bagging_freq=1).fit(x, y)
    data = Dataset(x, y)
    fits = [gbm.train(gbm.Params(**{**SMALL, "seed": 3 + i}, bagging_fraction=0.7, bagging_freq=1), data,
                      rows=slice(0, 700), backend="cpu") for i in range(3)]  # fmt: skip
    for ours, theirs in zip(bag.boosters_, fits, strict=True):
        np.testing.assert_array_equal(ours.forest.leaf_value, theirs.forest.leaf_value)
    np.testing.assert_array_equal(bag.predict(x), gbm.predict_mean(fits, data, rows=slice(0, 700), backend="cpu"))
    each = bag.predict(x, per_seed=True)
    assert each.shape == (700, 3)
    assert each.dtype == np.dtype(np.float32)
    for i, fit in enumerate(fits):
        np.testing.assert_array_equal(each[:, i], fit.predict(data, rows=slice(0, 700), backend="cpu"))
    assert GBMRegressor(**SMALL, backend="cpu").fit(x, y).predict(x, per_seed=True).shape == (700, 1)


def test_params_follow_scikit_learns_conventions() -> None:
    model = GBMRegressor(num_trees=5)
    params = model.get_params()
    assert params["num_trees"] == 5
    assert params["categorical_split"] == "sorted"
    assert params["growth"] == "depthwise"
    assert set(params) >= {"n_seeds", "backend", "categorical", "codes"}
    assert model.set_params(learning_rate=0.05) is model
    assert model.params().learning_rate == 0.05
    assert GBMRegressor(**model.get_params()).get_params() == model.get_params()
    with pytest.raises(ValueError, match="unknown parameters"):
        model.set_params(nope=1)
    with pytest.raises(ValueError, match=r"invalid Params\.max_depth"):
        GBMRegressor(max_depth=0, backend="cpu").fit(*table(50))


def test_save_and_idun_load_give_the_same_predictions(tmp_path: Path) -> None:
    x, y = table()
    model = GBMRegressor(**SMALL, backend="cpu", n_seeds=2, codes=[1], categorical=[2]).fit(x[:500], y[:500])
    model.save(tmp_path / "model.idun")
    loaded = idun.load(tmp_path / "model.idun")
    assert isinstance(loaded, GBMRegressor)
    assert loaded.get_params() == {**model.get_params(), "codes": [1], "categorical": [2]}
    np.testing.assert_array_equal(loaded.predict(x[500:]), model.predict(x[500:]))
    assert loaded.template_.binning == model.template_.binning


def test_fit_and_predict_refuse_what_they_cannot_use() -> None:
    x, y = table()
    with pytest.raises(ValueError, match="carries its target"):
        GBMRegressor(backend="cpu").fit(Dataset(x, y), y)
    with pytest.raises(ValueError, match=r"requires y to be passed.*a Dataset built with its target"):
        GBMRegressor(backend="cpu").fit(x)
    with pytest.raises(ValueError, match="not fitted"):
        GBMRegressor().predict(x)
    with pytest.raises(ValueError, match="n_seeds"):
        GBMRegressor(backend="cpu", n_seeds=0).fit(x, y)
    model = GBMRegressor(**SMALL, backend="cpu").fit(Dataset(x, y))
    with pytest.raises(ValueError, match="X has 2 features, but GBMRegressor is expecting 4 features"):
        model.predict(x[:, :2])


def test_gpu_and_cpu_estimators_are_bit_identical(gpu: str) -> None:
    x, y = table()
    cpu, fast = (GBMRegressor(**SMALL, backend=b, codes=[1], categorical=[2]).fit(x, y) for b in ("cpu", gpu))
    np.testing.assert_array_equal(cpu.predict(x), fast.predict(x))


LEAFWISE: dict[str, Any] = {"num_trees": 6, "growth": "leafwise", "max_depth": -1, "max_leaves": 15,
                            "min_data_in_leaf": 10, "seed": 3}  # fmt: skip


@pytest.mark.parametrize(
    "backend", ["cpu", pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_a_leafwise_estimator_fits_saves_and_loads(backend: str, tmp_path: Path) -> None:
    x, y = table()
    model = GBMRegressor(**LEAFWISE, backend=backend, n_seeds=2, codes=[1], categorical=[2]).fit(x[:500], y[:500])
    data = Dataset(x[:500], y[:500], codes=[1], categorical=[2])
    booster = gbm.train(model.params(), data, rows=slice(0, 500), backend=backend)
    np.testing.assert_array_equal(model.boosters_[0].forest.right, booster.forest.right)  # type: ignore[union-attr]
    model.save(tmp_path / "model.idun")
    loaded = idun.load(tmp_path / "model.idun")
    assert isinstance(loaded, GBMRegressor)
    assert loaded.get_params()["growth"] == "leafwise"
    np.testing.assert_array_equal(loaded.predict(x[500:]), model.predict(x[500:]))
    assert np.corrcoef(model.predict(x[500:]), y[500:])[0, 1] > 0.6


def test_eval_sets_stop_early_and_read_as_lightgbms(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    x, y = table(1200, seed=4)
    settings = {**SMALL, "num_trees": 60, "learning_rate": 0.4}
    model = GBMRegressor(**settings, backend="cpu", early_stopping_rounds=3, eval_metric="rmse", verbose=5)
    model.fit(x[:700], y[:700], eval_set=[(x[700:1000], y[700:1000]), (x[1000:], y[1000:])], eval_names=["a", "b"])
    best = model.best_iteration_
    assert isinstance(best, int)
    result = model.evals_result_
    assert isinstance(result, dict)
    assert list(result) == ["a", "b"]
    assert list(result["a"]) == ["rmse"]
    assert len(result["a"]["rmse"]) == best + 1 + 3
    assert "a's rmse:" in capsys.readouterr().out
    data = Dataset(x[:700], y[:700])
    alone = gbm.train(model.params(), data, rows=slice(0, 700), backend="cpu",
                      valid=[gbm.Validation(data.transform(x[700:1000], y[700:1000]), slice(0, 300))], metric="rmse",
                      early_stopping_rounds=3)  # fmt: skip
    test = data.transform(x[1000:])
    np.testing.assert_array_equal(model.predict(x[1000:]), alone.predict(test, rows=slice(0, 200), backend="cpu"))
    np.testing.assert_array_equal(model.predict(x[1000:], num_iteration=2),
                                  alone.predict(test, rows=slice(0, 200), backend="cpu", num_trees=2))  # fmt: skip
    window = alone.predict(test, rows=slice(0, 200), backend="cpu", start_tree=1, num_trees=2)
    np.testing.assert_array_equal(model.predict(x[1000:], start_iteration=1, num_iteration=2), window)
    importances = model.feature_importances_
    assert importances.shape == (model.n_features_in_,)
    assert importances.sum() == (alone.forest.kind != 0).sum()
    model.save(tmp_path / "es.npz")
    loaded = idun.load(tmp_path / "es.npz")
    assert (loaded.best_iteration_, loaded.evals_result_) == (best, result)


def test_a_bags_models_stop_each_on_its_own() -> None:
    x, y = table(900, seed=5)
    bag = GBMRegressor(**{**SMALL, "num_trees": 40, "learning_rate": 0.4}, backend="cpu", n_seeds=2,
                       bagging_fraction=0.7, bagging_freq=1, early_stopping_rounds=2)  # fmt: skip
    bag.fit(x[:600], y[:600], eval_set=[(x[600:], y[600:])])
    best, result = bag.best_iteration_, bag.evals_result_
    assert isinstance(best, list)
    assert isinstance(result, list)
    assert len(best) == len(result) == 2
    assert bag.feature_importances_.sum() == sum((b.forest.kind != 0).sum() for b in bag.boosters_)


def test_eval_settings_fail_loudly() -> None:
    x, y = table(400)
    with pytest.raises(ValueError, match="eval_names must name each"):
        GBMRegressor(**SMALL, backend="cpu").fit(x[:300], y[:300], eval_set=[(x[300:], y[300:])], eval_names=[])
    with pytest.raises(ValueError, match="verbose must be 0"):
        GBMRegressor(**SMALL, backend="cpu", verbose=-1).fit(x, y)
    with pytest.raises(ValueError, match="needs a validation set"):
        GBMRegressor(**SMALL, backend="cpu", early_stopping_rounds=2).fit(x, y)
    unstopped = GBMRegressor(**SMALL, backend="cpu").fit(x, y)
    assert unstopped.best_iteration_ is None
    assert unstopped.evals_result_ == {}


def test_the_regularisers_reach_the_params() -> None:
    settings: dict[str, Any] = {"lambda_l1": 0.5, "min_gain_to_split": 0.1, "min_sum_hessian_in_leaf": 2.0,
                                 "max_delta_step": 0.3}  # fmt: skip
    model = GBMRegressor(**SMALL, **settings, backend="cpu")
    params = model.params()
    assert {name: getattr(params, name) for name in settings} == settings
    x, y = table(500)
    data = Dataset(x, y)
    alone = gbm.train(params, data, rows=slice(0, 500), backend="cpu")
    np.testing.assert_array_equal(model.fit(x, y).predict(x), alone.predict(data.transform(x), rows=slice(0, 500),
                                                                            backend="cpu"))  # fmt: skip


def labelled(n: int = 1500, seed: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = np.where(x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.5, n) > 0, "yes", "no")
    return x, y


def test_the_classifier_fits_log_loss_from_the_log_odds(tmp_path: Path) -> None:
    x, y = labelled()
    model = gbm.GBMClassifier(**{**SMALL, "num_trees": 30, "learning_rate": 0.3}, backend="cpu")
    model.fit(x[:1000], y[:1000], eval_set=[(x[1000:], y[1000:])])
    assert model.classes_.tolist() == ["no", "yes"]
    data = Dataset(x[:1000], (y[:1000] == "yes").astype(np.float32))
    alone = gbm.train(model.params(), data, rows=slice(0, 1000), backend="cpu", objective=gbm.LogLoss(),
                      boost_from_average=True)  # fmt: skip
    test = data.transform(x[1000:])
    score = alone.predict(test, rows=slice(0, 500), backend="cpu")
    np.testing.assert_array_equal(model.decision_function(x[1000:]), score)
    p = 1.0 / (1.0 + np.exp(-score.astype(np.float64)))
    np.testing.assert_array_equal(model.predict_proba(x[1000:]), np.column_stack([1 - p, p]))
    np.testing.assert_array_equal(model.predict(x[1000:]), np.where(p > 0.5, "yes", "no"))
    assert model.score(x[1000:], y[1000:]) > 0.75
    assert list(model.evals_result_["valid_0"]) == ["binary_logloss"]  # type: ignore[call-overload]
    model.save(tmp_path / "clf.npz")
    loaded = idun.load(tmp_path / "clf.npz")
    assert isinstance(loaded, gbm.GBMClassifier)
    assert loaded.classes_.tolist() == ["no", "yes"]
    np.testing.assert_array_equal(loaded.predict_proba(x[1000:]), model.predict_proba(x[1000:]))


def test_the_classifier_refuses_what_it_cannot_fit() -> None:
    x, y = labelled(300)
    with pytest.raises(
        ValueError, match=r"Only binary classification is supported\. GBMClassifier fits two classes, y has 3"
    ):
        gbm.GBMClassifier(**SMALL, backend="cpu").fit(x, np.arange(300) % 3)
    with pytest.raises(ValueError, match="labels outside the training classes"):
        gbm.GBMClassifier(**SMALL, backend="cpu").fit(x[:200], y[:200], eval_set=[(x[200:], np.full(100, "maybe"))])
    with pytest.raises(ValueError, match="needs the targets 0 and 1"):
        gbm.GBMClassifier(**SMALL, backend="cpu").fit(Dataset(x, np.arange(300, dtype=np.float32)))
    model = gbm.GBMClassifier(**SMALL, backend="cpu").fit(Dataset(x, (y == "yes").astype(np.float32)))
    assert model.classes_.tolist() == [0.0, 1.0]


def test_importances_follow_the_columns_of_x() -> None:
    x, y = table(600, seed=7)
    model = GBMRegressor(**SMALL, backend="cpu", codes=[1], categorical=[2]).fit(x, y)
    booster = model.boosters_[0]
    assert list(booster.feature_names) != list(model.feature_names_in_)  # the binned order groups the kinds
    per_name = dict(zip(booster.feature_names, booster.feature_importance(), strict=True))
    assert list(model.feature_importances_) == [per_name[name] for name in model.feature_names_in_]


def test_contributions_follow_the_columns_and_add_up_to_the_prediction() -> None:
    x, y = table(600, seed=7)
    model = GBMRegressor(**SMALL, backend="cpu", codes=[1], categorical=[2]).fit(x, y)
    found = model.predict(x, pred_contrib=True)
    assert found.shape == (600, 5)
    np.testing.assert_allclose(found.sum(axis=1), model.predict(x), atol=1e-5)
    booster = model.boosters_[0]
    raw = booster.predict_contributions(model.template_.transform(x), rows=slice(0, 600))
    for column, name in enumerate(model.feature_names_in_):  # the boosters' order groups the kinds
        np.testing.assert_array_equal(found[:, column], raw[:, list(booster.feature_names).index(name)])
    assert np.all(found[:, 0] == 0) == (model.feature_importances_[0] == 0)
    window = model.predict(x, pred_contrib=True, num_iteration=3, start_iteration=2)
    np.testing.assert_allclose(window.sum(axis=1), model.predict(x, num_iteration=3, start_iteration=2), atol=1e-5)


def test_a_bags_contributions_are_its_models_mean_and_its_leaves_come_per_seed() -> None:
    x, y = table(500, seed=2)
    model = GBMRegressor(**SMALL, backend="cpu", n_seeds=3).fit(x, y)
    each = model.predict(x, pred_contrib=True, per_seed=True)
    assert each.shape == (500, 3, 5)
    np.testing.assert_allclose(model.predict(x, pred_contrib=True), each.mean(axis=1))
    np.testing.assert_allclose(each.sum(axis=2), model.predict(x, per_seed=True), atol=1e-5)
    leaves = model.predict(x, pred_leaf=True, per_seed=True)
    assert leaves.shape == (500, 3, 6)
    with pytest.raises(ValueError, match="per_seed=True"):
        model.predict(x, pred_leaf=True)
    with pytest.raises(ValueError, match="not both"):
        model.predict(x, pred_contrib=True, pred_leaf=True)


def test_the_classifiers_contributions_add_up_to_its_raw_score() -> None:
    x, y = labelled(800)
    model = gbm.GBMClassifier(**SMALL, backend="cpu").fit(x, y)
    found = model.predict(x, pred_contrib=True)
    np.testing.assert_allclose(found.sum(axis=1), model.decision_function(x), atol=1e-5)
    assert model.predict(x, pred_leaf=True).shape == (800, 6)


def test_sample_weights_reach_the_trees_as_trains_weight() -> None:
    x, y = table(600, seed=4)
    w = np.random.default_rng(1).choice([0.0, 0.5, 2.0], 600)
    model = GBMRegressor(**SMALL, backend="cpu").fit(x, y, w)
    data = model.template_.transform(x, y)
    direct = gbm.train(model.params(), data, rows=slice(0, 600), backend="cpu", weight=w.astype(np.float32))
    np.testing.assert_array_equal(model.predict(x), direct.predict(data, rows=slice(0, 600), backend="cpu"))
    plain = GBMRegressor(**SMALL, backend="cpu").fit(x, y)
    as_list = GBMRegressor(**SMALL, backend="cpu").fit(x, y, sample_weight=[1.0] * 600)
    np.testing.assert_array_equal(as_list.predict(x), plain.predict(x))  # unit weights: the unweighted model
    with pytest.raises(ValueError, match="one weight per row"):
        GBMRegressor(**SMALL, backend="cpu").fit(x, y, sample_weight=np.ones(599))


def test_the_estimators_bin_with_their_max_bin() -> None:
    x, y = table(600, seed=2)
    model = GBMRegressor(**SMALL, backend="cpu", max_bin=8).fit(x, y)
    assert max(model.template_.named_n_bins) <= 8
    assert model.get_params()["max_bin"] == 8
    plain = GBMRegressor(**SMALL, backend="cpu").fit(x, y)
    assert model.template_.binning != plain.template_.binning


def test_eval_sample_weights_weigh_the_validation_scores() -> None:
    x, y = table(800, seed=5)
    w = np.where(np.arange(200) < 100, 0.0, 1.0)
    model = GBMRegressor(**SMALL, backend="cpu").fit(x[:600], y[:600], eval_set=[(x[600:], y[600:])],
                                                     eval_sample_weight=[w])  # fmt: skip
    half = GBMRegressor(**SMALL, backend="cpu").fit(x[:600], y[:600], eval_set=[(x[700:], y[700:])])
    np.testing.assert_allclose(model.evals_result_["valid_0"]["l2"], half.evals_result_["valid_0"]["l2"], rtol=1e-12)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="one entry per eval set"):
        GBMRegressor(**SMALL, backend="cpu").fit(x[:600], y[:600], eval_set=[(x[600:], y[600:])],
                                                 eval_sample_weight=[w, w])  # fmt: skip
