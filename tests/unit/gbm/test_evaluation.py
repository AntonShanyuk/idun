"""idun.gbm.evaluation: validation sets scored after every tree, early stopping, callbacks."""

import dataclasses
import math
from typing import Literal

import numpy as np
import pytest

from idun import gbm
from idun.gbm import evaluation
from idun.gbm.tree import FIELDS, NODE_FIELDS
from tests.unit.gbm.tiny import make_tiny

PARAMS = gbm.Params(num_trees=40, max_depth=4, max_leaves=16, min_data_in_leaf=5, learning_rate=0.3, seed=1)
BACKENDS = ["cpu", pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]


def test_the_metrics_are_lightgbms() -> None:
    y, pred = np.array([1.0, 2.0, 4.0], np.float32), np.array([2.0, 2.0, 2.0], np.float32)
    assert evaluation.metric_of("l2").fn(y, pred) == pytest.approx(5 / 3)
    assert evaluation.metric_of("rmse").fn(y, pred) == pytest.approx(math.sqrt(5 / 3))
    assert evaluation.metric_of("l1").fn(y, pred) == pytest.approx(1.0)
    custom = evaluation.Metric("r", lambda y, p: float(np.corrcoef(y, p)[0, 1]), higher_is_better=True)
    assert evaluation.metric_of(custom) is custom
    with pytest.raises(ValueError, match="metric must be one of"):
        evaluation.metric_of("ndcg")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("growth", ["depthwise", "leafwise"])
def test_an_early_stopped_model_is_the_model_of_its_trees(
    backend: str, growth: Literal["depthwise", "leafwise"]
) -> None:
    data = make_tiny(3000, n_countries=3, seed=3)
    params = dataclasses.replace(PARAMS, growth=growth)
    valid = [gbm.Validation(data, slice(2000, 3000)), gbm.Validation(data, slice(1500, 2000), name="later")]
    model = gbm.train(params, data, rows=slice(0, 1500), backend=backend, valid=valid, early_stopping_rounds=4)
    assert model.best_iteration is not None
    kept = model.best_iteration + 1
    assert kept < PARAMS.num_trees
    assert model.n_trees == model.params.num_trees == kept
    assert len(model.evals["valid_0"]) == len(model.evals["later"]) == kept + 4  # trained 4 past the best
    assert model.evals["valid_0"].index(min(model.evals["valid_0"])) == model.best_iteration
    same = gbm.train(dataclasses.replace(params, num_trees=kept), data, rows=slice(0, 1500), backend=backend)
    for field in NODE_FIELDS if growth == "leafwise" else FIELDS:
        np.testing.assert_array_equal(getattr(model.forest, field), getattr(same.forest, field))
    for name, rows in (("valid_0", slice(2000, 3000)), ("later", slice(1500, 2000))):
        for i in (0, kept // 2, kept - 1):
            pred = model.predict(data, rows=rows, backend="cpu", num_trees=i + 1)
            assert model.evals[name][i] == evaluation.METRICS["l2"].fn(data.target[rows], pred)  # the same float


def test_scores_without_early_stopping_keep_every_tree_and_a_callback_may_stop() -> None:
    data = make_tiny(1200, n_countries=3, seed=4)
    seen: list[evaluation.Evaluation] = []
    model = gbm.train(PARAMS, data, rows=slice(0, 900), backend="cpu", valid=[gbm.Validation(data, slice(900, 1200))],
                      metric="rmse", callback=seen.append)  # fmt: skip
    assert model.n_trees == PARAMS.num_trees
    assert model.best_iteration is None
    assert [e.iteration for e in seen] == list(range(PARAMS.num_trees))
    assert seen[-1].metric == "rmse"
    assert seen[-1].scores == {"valid_0": model.evals["valid_0"][-1]}
    stopped = gbm.train(PARAMS, data, rows=slice(0, 900), backend="cpu", callback=lambda e: e.iteration == 6)
    assert stopped.n_trees == 7  # a callback alone: no scores, the trees up to its stop
    assert stopped.evals == {}


def test_a_custom_metric_better_when_higher() -> None:
    data = make_tiny(1200, n_countries=3, seed=5)
    corr = gbm.Metric("corr", lambda y, p: float(np.corrcoef(y, p)[0, 1]) if p.std() > 0 else -1.0,
                      higher_is_better=True)  # fmt: skip
    model = gbm.train(PARAMS, data, rows=slice(0, 900), backend="cpu", valid=[gbm.Validation(data, slice(900, 1200))],
                      metric=corr, early_stopping_rounds=3)  # fmt: skip
    scores = model.evals["valid_0"]
    assert model.best_iteration == scores.index(max(scores))


def test_seed_bags_stop_each_fit_on_its_own() -> None:
    data = make_tiny(1500, n_countries=3, seed=6)
    params = dataclasses.replace(PARAMS, bagging_fraction=0.7, bagging_freq=1)
    valid = [gbm.Validation(data, slice(1000, 1500))]
    bag = gbm.train_seeds(params, data, rows=slice(0, 1000), backend="cpu", seeds=[1, 2], valid=valid,
                          early_stopping_rounds=3)  # fmt: skip
    for seed, model in zip([1, 2], bag, strict=True):
        alone = gbm.train(dataclasses.replace(params, seed=seed), data, rows=slice(0, 1000), backend="cpu",
                          valid=valid, early_stopping_rounds=3)  # fmt: skip
        assert (model.best_iteration, model.evals) == (alone.best_iteration, alone.evals)


def test_bad_validation_sets_fail_loudly() -> None:
    data = make_tiny(600, n_countries=3, seed=7)
    other = make_tiny(600, n_countries=3, seed=8)
    fit = {"params": PARAMS, "data": data, "rows": slice(0, 400), "backend": "cpu"}
    with pytest.raises(ValueError, match="needs a validation set"):
        gbm.train(**fit, early_stopping_rounds=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs a validation set and a positive count"):
        gbm.train(**fit, valid=[gbm.Validation(data, slice(400, 600))], early_stopping_rounds=0)  # type: ignore[arg-type]
    if other.binning != data.binning:
        with pytest.raises(ValueError, match="is binned as"):
            gbm.train(**fit, valid=[gbm.Validation(other, slice(0, 100))])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contiguous non-empty range"):
        gbm.train(**fit, valid=[gbm.Validation(data, slice(500, 500))])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="y must hold one float32 per row"):
        gbm.train(**fit, valid=[gbm.Validation(data, slice(400, 600), y=np.zeros(3, np.float32))])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="distinct names"):
        gbm.train(**fit, valid=[gbm.Validation(data, slice(400, 500), name="a"),  # type: ignore[arg-type]
                                gbm.Validation(data, slice(500, 600), name="a")])  # fmt: skip


def test_log_evaluation_prints_lightgbms_lines(capsys: pytest.CaptureFixture[str]) -> None:
    log = evaluation.log_evaluation(2)
    log(evaluation.Evaluation(0, "l2", {"valid_0": 0.5}))
    log(evaluation.Evaluation(1, "l2", {"valid_0": 0.25, "b": 1.0}))
    assert capsys.readouterr().out == "[2]\tvalid_0's l2: 0.25\tb's l2: 1\n"
    with pytest.raises(ValueError, match="period must be a positive integer"):
        evaluation.log_evaluation(0)


def test_the_binary_metrics_read_raw_scores() -> None:
    y = np.array([0, 1, 1, 0, 1], np.float32)
    s = np.array([-1.0, 2.0, 0.5, 0.5, -0.2], np.float32)
    p = 1.0 / (1.0 + np.exp(-s.astype(np.float64)))
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert evaluation.METRICS["binary_logloss"].fn(y, s) == pytest.approx(logloss)
    # pairs (pos, neg): (2.0 > -1, 0.5), (0.5 > -1, = 0.5 tie), (-0.2 > -1, < 0.5): 1 + 1 + 1 + 0.5 + 1 + 0 = 4.5 of 6
    assert evaluation.METRICS["auc"].fn(y, s) == pytest.approx(4.5 / 6)
    assert evaluation.METRICS["auc"].higher_is_better
    assert math.isnan(evaluation.METRICS["auc"].fn(np.ones(3, np.float32), s[:3]))


def test_the_multiclass_metrics_read_every_class_score() -> None:
    scores = np.array([[2.0, 0.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], np.float32)  # (K, n)
    y = np.array([0, 1, 0, 1], np.float32)
    s = scores.astype(np.float64)
    p = np.exp(s) / np.exp(s).sum(axis=0)
    expected = float(np.mean(-np.log(p[y.astype(np.int64), np.arange(4)])))
    assert evaluation.METRICS["multi_logloss"].fn(y, scores) == pytest.approx(expected, rel=1e-12)
    assert evaluation.METRICS["multi_error"].fn(y, scores) == 0.5  # row 2 picks class 2; row 3 ties: class 0 first
    huge = np.array([[1000.0], [0.0]], np.float32)
    assert math.isfinite(evaluation.METRICS["multi_logloss"].fn(np.array([1.0], np.float32), huge))


def test_a_metric_needs_its_kind_of_fit() -> None:
    data = make_tiny(200, seed=3)
    with pytest.raises(ValueError, match="multi-class"):
        evaluation.Evaluator([], data.binning, "multi_logloss", None, None)
    with pytest.raises(ValueError, match="multi-class"):
        evaluation.Evaluator([], data.binning, "l2", None, None, k=3)
    evaluation.Evaluator([], data.binning, evaluation.Metric("mine", lambda y, p: 0.0), None, None, k=3)


def test_weighted_metrics_are_lightgbms_weighted_means() -> None:
    rng = np.random.default_rng(8)
    y = (rng.random(300) < 0.4).astype(np.float32)
    pred = rng.normal(size=300).astype(np.float32)
    w = rng.choice([0.0, 0.5, 2.0], 300)
    d = pred.astype(np.float64) - y
    metrics = evaluation.METRICS
    assert metrics["l2"].fn(y, pred, w) == pytest.approx(np.sum(w * d * d) / np.sum(w), rel=1e-12)  # type: ignore[call-arg]
    assert metrics["l1"].fn(y, pred, w) == pytest.approx(np.sum(w * np.abs(d)) / np.sum(w), rel=1e-12)  # type: ignore[call-arg]
    positive, negative = np.flatnonzero(y > 0.5), np.flatnonzero(y < 0.5)
    pairs = sum(w[i] * w[j] * (1.0 if pred[i] > pred[j] else 0.5 if pred[i] == pred[j] else 0.0)
                for i in positive for j in negative)  # fmt: skip
    expected = pairs / (w[positive].sum() * w[negative].sum())
    assert metrics["auc"].fn(y, pred, w) == pytest.approx(expected, rel=1e-12)  # type: ignore[call-arg]
    ones = np.ones(300)
    for name in ("l2", "rmse", "l1", "binary_logloss"):  # unit weights: the unweighted score, bit for bit
        assert metrics[name].fn(y, pred, ones) == metrics[name].fn(y, pred), name  # type: ignore[call-arg]
    assert metrics["auc"].fn(y, pred, ones) == pytest.approx(metrics["auc"].fn(y, pred), rel=1e-12)  # type: ignore[call-arg]
    scores, labels = rng.normal(size=(3, 300)).astype(np.float32), rng.integers(0, 3, 300).astype(np.float32)
    errors = (np.argmax(scores, axis=0) != labels).astype(np.float64)
    assert metrics["multi_error"].fn(labels, scores, w) == pytest.approx(np.sum(w * errors) / np.sum(w))  # type: ignore[call-arg]


def test_a_weighted_validation_set_scores_and_stops_with_its_weights() -> None:
    data = make_tiny(800, seed=6)
    w = np.where(np.arange(200) < 100, 0.0, 1.0).astype(np.float32)  # the second half alone counts
    params = gbm.Params(num_trees=40, max_depth=3, max_leaves=8, min_data_in_leaf=5, learning_rate=0.3)
    weighted = gbm.train(params, data, rows=slice(0, 600), backend="cpu",
                         valid=[gbm.Validation(data, slice(600, 800), weight=w)], early_stopping_rounds=3)  # fmt: skip
    half = gbm.train(params, data, rows=slice(0, 600), backend="cpu",
                     valid=[gbm.Validation(data, slice(700, 800))], early_stopping_rounds=3)  # fmt: skip
    assert weighted.evals["valid_0"] == pytest.approx(half.evals["valid_0"], rel=1e-12)
    assert weighted.n_trees == half.n_trees
    valid, mine = [gbm.Validation(data, slice(600, 800), weight=w)], evaluation.Metric("mine", lambda y, p: 0.0)
    with pytest.raises(ValueError, match="needs a built-in metric"):
        gbm.train(params, data, rows=slice(0, 600), backend="cpu", valid=valid, metric=mine)
    for bad in (np.ones(5, np.float32), np.full(200, -1.0, np.float32), np.zeros(200, np.float32)):
        with pytest.raises(ValueError, match="weight"):
            gbm.train(params, data, rows=slice(0, 600), backend="cpu",
                      valid=[gbm.Validation(data, slice(600, 800), weight=bad)])  # fmt: skip
