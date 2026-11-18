import dataclasses
import gc
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from idun import gbm
from idun.cuda import runtime as cuda_runtime
from idun.gbm.train import fit_bytes, leaf_values
from idun.gbm.tree import NodeForest
from tests import oracle
from tests.unit.gbm.backends import emulated
from tests.unit.gbm.test_objective import Logistic, Weighted  # Logistic has no start; Weighted: the weighted L2
from tests.unit.gbm.tiny import make_tiny

FIELDS = ("kind", "feature", "threshold", "missing_left", "cat_left", "leaf_value")
NODE_FIELDS = (*FIELDS, "left", "right")


@pytest.mark.parametrize(
    "params",
    [
        gbm.Params(
            num_trees=3,
            max_depth=3,
            max_leaves=8,
            min_data_in_leaf=5,
            lambda_l2=1.0,
            feature_fraction=0.6,
            bagging_fraction=0.7,
            bagging_freq=2,
            cat_l2=0.0,
            cat_smooth=1.0,
            min_data_per_group=3,
            seed=4,
        ),
        gbm.Params(
            num_trees=2, max_depth=4, max_leaves=5, min_data_in_leaf=3, categorical="one_vs_rest", grad_quant_bins=8
        ),
    ],
)
def test_agrees_with_the_oracle(params: gbm.Params) -> None:
    data = make_tiny(250, n_countries=8, seed=12)
    booster = gbm.train(params, data, rows=slice(10, 240), backend="cpu")
    expected = oracle.fit(params, data, 10, 240)
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(booster.forest, name), getattr(expected, name), err_msg=name)


def test_training_predictions_equal_inference() -> None:
    data = make_tiny(300, seed=13)
    params = gbm.Params(
        num_trees=5, max_depth=3, max_leaves=8, min_data_in_leaf=10, bagging_fraction=0.5, bagging_freq=1
    )
    booster = gbm.train(params, data, rows=slice(0, 300), backend="cpu")
    expected = oracle.fit(params, data, 0, 300)
    np.testing.assert_array_equal(booster.predict(data, rows=slice(0, 300), backend="cpu"), expected.pred)


def test_the_leaf_cap_keeps_the_best_splits() -> None:
    data = make_tiny(400, seed=14)
    booster = gbm.train(
        gbm.Params(num_trees=3, max_depth=4, max_leaves=3, min_data_in_leaf=5), data, rows=slice(0, 400), backend="cpu"
    )
    assert ((booster.forest.kind != 0).sum(axis=1) <= 2).all()  # 3 leaves = 2 splits
    assert (booster.forest.kind != 0).any()


def test_zero_gradients_grow_empty_trees() -> None:
    data = make_tiny(100, seed=15)
    booster = gbm.train(
        gbm.Params(num_trees=2, max_depth=2, max_leaves=4),
        data,
        rows=slice(0, 100),
        backend="cpu",
        y=np.zeros(100, np.float32),
    )
    assert not booster.forest.kind.any()
    assert not booster.forest.leaf_value.any()


def test_the_fit_report_times_every_phase() -> None:
    data = make_tiny(100, seed=16)
    report = gbm.train(
        gbm.Params(num_trees=2, max_depth=2, max_leaves=4), data, rows=slice(0, 100), backend="cpu"
    ).fit_report
    assert report is not None
    assert set(report.seconds) == {"upload", "gradients", "histograms", "splits", "partition", "leaves"}
    assert report.total_seconds > 0
    assert (report.backend, report.train_rows) == ("cpu", (0, 100))


def test_leaf_values_use_the_exact_sums() -> None:
    values = leaf_values(np.array([2**32, -(2**33), 0]), np.array([3, 1, 0]), 1.0, 0.5)
    assert values.dtype == np.float32
    assert values.tolist() == [-0.125, 0.5, 0.0]  # -(1)/(3+1)*0.5, 2/(1+1)*0.5, empty leaf
    assert leaf_values(np.array([0]), np.array([0]), 0.0, 1.0).tolist() == [0.0]  # 0/0 never escapes


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"rows": slice(0, 100, 2)}, ValueError, "contiguous"),
        ({"rows": slice(50, 50)}, ValueError, "no row"),
        ({"y": np.zeros(7, np.float32)}, ValueError, "one float32 per training row"),
        ({"backend": "tpu"}, ValueError, "unknown backend"),
        ({"y": np.full(100, 1e30, np.float32)}, OverflowError, "fixed-point"),
    ],
)
def test_bad_requests_fail_loudly(kwargs: dict[str, object], error: type[Exception], message: str) -> None:
    data = make_tiny(100, seed=17)
    call = {"rows": slice(0, 100), "backend": "cpu", **kwargs}
    with pytest.raises(error, match=message):
        gbm.train(gbm.Params(num_trees=1, max_depth=1, max_leaves=2), data, **call)  # type: ignore[arg-type]


SEEDS = gbm.Params(
    num_trees=3, max_depth=3, max_leaves=8, min_data_in_leaf=5, lambda_l2=1.0, feature_fraction=0.6,
    bagging_fraction=0.7, bagging_freq=1, cat_l2=0.0, cat_smooth=1.0, min_data_per_group=3, seed=0,
)  # fmt: skip


SIDE_BY_SIDE = [
    ("cpu", 1),
    pytest.param("metal", 3, marks=pytest.mark.metal),
    pytest.param("cuda", 2, marks=pytest.mark.cuda),
]


def same_forest(a: gbm.Booster, b: gbm.Booster) -> bool:
    return all(np.array_equal(getattr(a.forest, name), getattr(b.forest, name)) for name in (*FIELDS, "leaf_count"))


@pytest.mark.parametrize(("backend", "workers"), SIDE_BY_SIDE)
def test_a_seed_bag_is_the_separate_fits_in_seed_order(backend: str, workers: int) -> None:
    data = make_tiny(400, n_countries=5, seed=21)
    seeds = [5, 1, 9, 2]
    bag = gbm.train_seeds(SEEDS, data, rows=slice(0, 400), backend=backend, seeds=seeds, workers=workers)
    separate = [gbm.train(dataclasses.replace(SEEDS, seed=s), data, rows=slice(0, 400), backend=backend) for s in seeds]
    assert [model.params.seed for model in bag] == seeds
    assert all(same_forest(a, b) for a, b in zip(bag, separate, strict=True))
    assert not same_forest(bag[0], bag[1])


def test_a_bag_predicts_the_mean_of_its_models() -> None:
    data = make_tiny(300, n_countries=5, seed=22)
    bag = gbm.train_seeds(SEEDS, data, rows=slice(0, 200), backend="cpu", seeds=[0, 1, 2], workers=1)
    want = np.mean([m.predict(data, rows=slice(200, 300), backend="cpu").astype(np.float64) for m in bag], axis=0)
    np.testing.assert_array_equal(
        gbm.predict_mean(bag, data, rows=slice(200, 300), backend="cpu"), want.astype(np.float32)
    )
    with pytest.raises(ValueError, match="at least one model"):
        gbm.predict_mean([], data, rows=slice(200, 300), backend="cpu")


@pytest.mark.parametrize(
    ("seeds", "workers", "backend", "message"),
    [([], 1, "cpu", "distinct"), ([1, 1], 1, "cpu", "distinct"), ([1, 2], 2, "cpu", "workers"),
     ([1, 2], 0, "metal", "workers"), ([1, 2], 0, "cuda", "workers")],
)  # fmt: skip
def test_bad_seed_bags_fail_loudly(seeds: list[int], workers: int, backend: str, message: str) -> None:
    data = make_tiny(100, n_countries=3, seed=23)
    with pytest.raises(ValueError, match=message):
        gbm.train_seeds(SEEDS, data, rows=slice(0, 100), backend=backend, seeds=seeds, workers=workers)


@pytest.mark.metal
@pytest.mark.parametrize(("devices", "named", "used"), [(2, None, {0, 1}), (3, None, {0, 1, 2}), (2, "1", {1})])
def test_cuda_seeds_side_by_side_are_spread_over_the_gpus(
    monkeypatch: pytest.MonkeyPatch, devices: int, named: str | None, used: set[int]
) -> None:
    emulation = emulated.install(monkeypatch, devices)
    monkeypatch.delenv(cuda_runtime.GUARD_ENV, raising=False)
    if named is None:
        monkeypatch.delenv(cuda_runtime.DEVICE_ENV, raising=False)
    else:
        monkeypatch.setenv(cuda_runtime.DEVICE_ENV, named)
    data = make_tiny(400, n_countries=5, seed=21)
    seeds = [5, 1, 9, 2, 7, 3]
    bag = gbm.train_seeds(SEEDS, data, rows=slice(0, 400), backend="cuda", seeds=seeds, workers=3)
    made = emulation.driver.created
    assert set(made) == used
    assert len(made) == len(seeds)  # a context per fit
    assert emulation.driver.live() == {}  # each fit's backend was closed
    separate = [gbm.train(dataclasses.replace(SEEDS, seed=s), data, rows=slice(0, 400), backend="cpu") for s in seeds]
    assert [model.params.seed for model in bag] == seeds
    assert all(same_forest(a, b) for a, b in zip(bag, separate, strict=True))
    before = len(emulation.driver.created)
    default = gbm.train_seeds(SEEDS, data, rows=slice(0, 400), backend="cuda", seeds=seeds[:2])  # one fit a GPU
    assert set(emulation.driver.created[before:]) == ({1} if named else {0, 1})
    assert len(emulation.driver.created) == before + 2
    assert all(same_forest(a, b) for a, b in zip(default, separate[:2], strict=True))
    gc.collect()
    emulation.close()


def test_seed_bags_never_exceed_the_gpus_room(monkeypatch: pytest.MonkeyPatch) -> None:
    from idun.metal import runtime  # noqa: PLC0415

    data = make_tiny(100, n_countries=3, seed=24)
    assert fit_bytes(data) > 32 * data.n_rows
    monkeypatch.setattr(runtime, "side_by_side", lambda bytes_each: 1)
    with pytest.raises(ValueError, match="pass at most 1"):
        gbm.train_seeds(SEEDS, data, rows=slice(0, 100), backend="metal", seeds=[1, 2], workers=2)


@pytest.mark.parametrize(
    "shape",
    [{"more_categorical": (3, 17)}, {"n_countries": 0}, {"n_obfuscated": 0}, {"n_named": 0, "more_categorical": (6,)}],
)
def test_any_number_of_columns_per_kind_agrees_with_the_oracle(shape: dict[str, object]) -> None:
    data = make_tiny(400, seed=13, **shape)  # type: ignore[arg-type]
    params = gbm.Params(
        num_trees=3, max_depth=3, max_leaves=6, min_data_in_leaf=5, cat_smooth=1.0, min_data_per_group=3
    )
    booster = gbm.train(params, data, rows=slice(0, 380), backend="cpu")
    expected = oracle.fit(params, data, 0, 380)
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(booster.forest, name), getattr(expected, name), err_msg=name)


LEAFWISE = [
    gbm.Params(
        num_trees=4, growth="leafwise", max_depth=-1, max_leaves=9, min_data_in_leaf=5, lambda_l2=1.0,
        feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=2, cat_l2=0.0, cat_smooth=1.0, min_data_per_group=3,
        seed=4,
    ),
    gbm.Params(num_trees=3, growth="leafwise", max_depth=3, max_leaves=7, min_data_in_leaf=3,
               categorical="one_vs_rest", grad_quant_bins=8),
    gbm.Params(num_trees=2, growth="leafwise", max_depth=-1, max_leaves=40, min_data_in_leaf=2, cat_smooth=1.0,
               min_data_per_group=2),  # more leaves than the data supports: trees stop early
]  # fmt: skip


@pytest.mark.parametrize("params", LEAFWISE)
def test_leafwise_agrees_with_the_oracle(params: gbm.Params) -> None:
    data = make_tiny(250, n_countries=8, more_categorical=(3,), seed=12)
    booster = gbm.train(params, data, rows=slice(10, 240), backend="cpu")
    expected = oracle.fit(params, data, 10, 240)
    for name in NODE_FIELDS:
        np.testing.assert_array_equal(getattr(booster.forest, name), getattr(expected, name), err_msg=name)
    np.testing.assert_array_equal(booster.predict(data, rows=slice(10, 240), backend="cpu"), expected.pred)
    np.testing.assert_array_equal(
        booster.predict(data, rows=slice(0, 250), backend="cpu"), oracle.predict(expected, data, 0, 250)
    )


def test_leafwise_trees_are_lopsided_and_capped() -> None:
    data = make_tiny(3000, n_countries=8, seed=18)
    params = gbm.Params(num_trees=5, growth="leafwise", max_depth=-1, max_leaves=24, min_data_in_leaf=10)
    forest = gbm.train(params, data, rows=slice(0, 3000), backend="cpu").forest
    assert isinstance(forest, NodeForest)
    splits = (forest.kind != 0).sum(axis=1)
    assert (splits == 23).all()  # 24 leaves
    depths = [_depth(forest, t) for t in range(forest.n_trees)]
    assert max(depths) > 5  # deeper than a balanced 24-leaf tree: no depth limit

    limited_params = dataclasses.replace(params, max_depth=3, max_leaves=8)
    limited = gbm.train(limited_params, data, rows=slice(0, 3000), backend="cpu").forest
    assert isinstance(limited, NodeForest)
    assert max(_depth(limited, t) for t in range(limited.n_trees)) <= 3
    assert ((limited.kind != 0).sum(axis=1) <= 7).all()


def _depth(forest: NodeForest, tree: int) -> int:
    """The deepest leaf of a node-list tree."""
    deepest, stack = 0, [(0, 1)]
    while stack:
        node, depth = stack.pop()
        for child in (int(forest.left[tree, node]), int(forest.right[tree, node])):
            if child >= 0:
                stack.append((child, depth + 1))
            else:
                deepest = max(deepest, depth)
    return deepest


@pytest.mark.parametrize("depth", [1, 3])
def test_leafwise_with_room_for_the_whole_tree_predicts_as_depthwise(depth: int) -> None:
    """Best-first growth with ``2^D`` leaves and depth ``D`` makes every split depth-wise growth makes."""
    data = make_tiny(800, n_countries=6, seed=19)
    depthwise = gbm.Params(num_trees=4, max_depth=depth, max_leaves=2**depth, min_data_in_leaf=8, lambda_l2=1.0,
                           bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, cat_smooth=1.0)  # fmt: skip
    leafwise = dataclasses.replace(depthwise, growth="leafwise")
    a = gbm.train(depthwise, data, rows=slice(0, 700), backend="cpu")
    b = gbm.train(leafwise, data, rows=slice(0, 700), backend="cpu")
    for rows in (slice(0, 700), slice(700, 800)):
        np.testing.assert_array_equal(
            a.predict(data, rows=rows, backend="cpu"), b.predict(data, rows=rows, backend="cpu")
        )
    assert sorted(a.forest.leaf_value[0].tolist()) == sorted(b.forest.leaf_value[0].tolist())


def test_leafwise_zero_gradients_grow_empty_trees() -> None:
    data = make_tiny(100, seed=15)
    params = gbm.Params(num_trees=2, growth="leafwise", max_depth=-1, max_leaves=31)
    booster = gbm.train(params, data, rows=slice(0, 100), backend="cpu", y=np.zeros(100, np.float32))
    assert isinstance(booster.forest, NodeForest)
    assert not booster.forest.kind.any()
    assert (booster.forest.left == -1).all()
    assert not booster.forest.leaf_value.any()


@pytest.mark.parametrize(("backend", "workers"), SIDE_BY_SIDE)
def test_a_leafwise_seed_bag_is_the_separate_fits(backend: str, workers: int) -> None:
    params = dataclasses.replace(SEEDS, growth="leafwise", max_depth=-1, max_leaves=12)
    data = make_tiny(400, n_countries=5, seed=21)
    bag = gbm.train_seeds(params, data, rows=slice(0, 400), backend=backend, seeds=[3, 0, 7], workers=workers)
    for seed, model in zip([3, 0, 7], bag, strict=True):
        alone = gbm.train(dataclasses.replace(params, seed=seed), data, rows=slice(0, 400), backend=backend)
        for name in NODE_FIELDS:
            np.testing.assert_array_equal(getattr(model.forest, name), getattr(alone.forest, name), err_msg=name)


@pytest.mark.parametrize(("backend", "workers"), SIDE_BY_SIDE)
def test_a_seed_bag_drawing_and_scoring_on_the_host_is_the_separate_fits(backend: str, workers: int) -> None:
    # Extra trees draw each node's thresholds, and a validation set scores each tree, with numba's parallel loops in
    # every fit's thread: numba's workqueue threading layer aborts the process when two threads run them at once.
    params = dataclasses.replace(SEEDS, extra_trees=True, num_trees=20)
    data = make_tiny(600, n_countries=5, seed=21)
    valid = [gbm.Validation(data, slice(400, 600))]
    bag = gbm.train_seeds(params, data, rows=slice(0, 400), backend=backend, seeds=[3, 0, 7], workers=workers,
                          valid=valid)  # fmt: skip
    for seed, model in zip([3, 0, 7], bag, strict=True):
        alone = gbm.train(dataclasses.replace(params, seed=seed), data, rows=slice(0, 400), backend=backend,
                          valid=valid)  # fmt: skip
        assert same_forest(model, alone)


REGULARISED: list[dict[str, Any]] = [
    {"lambda_l1": 0.5},
    {"lambda_l1": 2.0, "lambda_l2": 1.0},
    {"min_gain_to_split": 0.5},
    {"min_sum_hessian_in_leaf": 30.0},
    {"max_delta_step": 0.02},
    {"lambda_l1": 0.3, "max_delta_step": 0.1, "min_gain_to_split": 0.01, "min_sum_hessian_in_leaf": 12.0},
]


@pytest.mark.parametrize("settings", REGULARISED)
@pytest.mark.parametrize(("growth", "categorical"), [("depthwise", "sorted"), ("depthwise", "one_vs_rest"),
                                                      ("leafwise", "sorted")])  # fmt: skip
def test_the_regularisers_agree_with_the_oracle(settings: dict[str, Any], growth: str, categorical: str) -> None:
    data = make_tiny(400, n_countries=8, seed=31)
    base = gbm.Params(num_trees=4, max_depth=3 if growth == "depthwise" else -1, max_leaves=8, min_data_in_leaf=5,
                      learning_rate=0.5, cat_smooth=1.0, cat_l2=0.0, min_data_per_group=3, bagging_fraction=0.8,
                      bagging_freq=1, growth=growth, categorical=categorical)  # type: ignore[arg-type]  # fmt: skip
    params = dataclasses.replace(base, **settings)
    booster = gbm.train(params, data, rows=slice(0, 380), backend="cpu")
    expected = oracle.fit(params, data, 0, 380)
    fields = NODE_FIELDS if growth == "leafwise" else FIELDS
    for name in fields:
        np.testing.assert_array_equal(getattr(booster.forest, name), getattr(expected, name), err_msg=name)
    plain = gbm.train(base, data, rows=slice(0, 380), backend="cpu")
    assert not all(np.array_equal(getattr(booster.forest, n), getattr(plain.forest, n)) for n in fields)  # it acts
    if "max_delta_step" in settings:
        assert np.abs(booster.forest.leaf_value).max() <= np.float32(settings["max_delta_step"] * params.learning_rate)


def test_the_regularisers_fail_loudly_and_train_multi_refuses_them() -> None:
    for name in ("lambda_l1", "min_gain_to_split", "min_sum_hessian_in_leaf", "max_delta_step"):
        bad: dict[str, Any] = {name: -1.0}
        with pytest.raises(ValueError, match=f"invalid Params.{name}"):
            gbm.Params(**bad)
    data = make_tiny(200, seed=32)
    with pytest.raises(ValueError, match="train_multi does not take lambda_l1 yet"):
        gbm.train_multi(gbm.Params(lambda_l1=1.0), data, rows=slice(0, 200), backend="cpu", ys=[data.target[:200]])
    old = gbm.Params().to_dict()
    for name in ("lambda_l1", "min_gain_to_split", "min_sum_hessian_in_leaf", "max_delta_step"):
        del old[name]
    assert gbm.Params.from_dict(old) == gbm.Params()  # a file without them loads with the defaults


CONTINUED = ["cpu", pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("backend", CONTINUED)
@pytest.mark.parametrize("growth", ["depthwise", "leafwise"])
def test_a_continued_model_is_the_model_of_all_its_trees(backend: str, growth: str) -> None:
    data = make_tiny(1600, n_countries=5, seed=33)
    params = gbm.Params(num_trees=24, max_depth=4 if growth == "depthwise" else -1, max_leaves=12, min_data_in_leaf=5,
                        bagging_fraction=0.7, bagging_freq=3, feature_fraction=0.7, seed=4,
                        growth=growth)  # type: ignore[arg-type]  # fmt: skip
    rows = slice(0, 1200)
    whole = gbm.train(params, data, rows=rows, backend=backend)
    head = gbm.train(dataclasses.replace(params, num_trees=13), data, rows=rows, backend=backend)  # mid-period: 13 % 3
    rest = gbm.train(dataclasses.replace(params, num_trees=11), data, rows=rows, backend=backend, init_model=head)
    assert (rest.n_trees, rest.params.num_trees) == (24, 24)
    assert same_forest(rest, whole)
    valid = [gbm.Validation(data, slice(1200, 1600))]
    scored = gbm.train(dataclasses.replace(params, num_trees=11), data, rows=rows, backend=backend, init_model=head,
                       valid=valid, early_stopping_rounds=50)  # fmt: skip
    pred = whole.predict(data, rows=slice(1200, 1600), backend="cpu")
    d = pred.astype(np.float64) - data.target[1200:1600]
    assert scored.evals["valid_0"][-1] == float(np.mean(d * d))  # the sets start from init_model's predictions
    assert scored.best_iteration is not None
    assert scored.best_iteration >= 13  # the model's tree index


def test_continued_training_needs_the_same_tree_shape() -> None:
    data = make_tiny(400, seed=34)
    head = gbm.train(gbm.Params(num_trees=3, max_depth=3, max_leaves=8, min_data_in_leaf=5), data, rows=slice(0, 400),
                     backend="cpu")  # fmt: skip
    with pytest.raises(ValueError, match="another shape"):
        gbm.train(gbm.Params(num_trees=3, max_depth=4, max_leaves=8, min_data_in_leaf=5), data, rows=slice(0, 400),
                  backend="cpu", init_model=head)  # fmt: skip
    with pytest.raises(ValueError, match="not with init_model"):
        gbm.train(head.params, data, rows=slice(0, 400), backend="cpu", init_model=head, boost_from_average=True)


def test_boost_from_average_starts_at_the_mean_in_the_first_tree() -> None:
    data = make_tiny(800, seed=35)
    params = gbm.Params(num_trees=6, max_depth=3, max_leaves=8, min_data_in_leaf=5, learning_rate=0.3)
    rows = slice(0, 800)
    one = gbm.train(dataclasses.replace(params, num_trees=1), data, rows=rows, backend="cpu", boost_from_average=True)
    mean = np.float32(np.mean(data.target[:800], dtype=np.float64))
    plain = gbm.train(dataclasses.replace(params, num_trees=1), data, rows=rows, backend="cpu",
                      y=(data.target[:800] - mean).astype(np.float32))  # fmt: skip
    np.testing.assert_array_equal(one.forest.leaf_value[0], (plain.forest.leaf_value[0] + mean).astype(np.float32))
    with pytest.raises(ValueError, match="needs its start"):
        gbm.train(params, data, rows=rows, backend="cpu", boost_from_average=True, objective=SquaredError())


class SquaredError:
    """L2 as an `Objective`, without a start."""

    constant_hessian: Literal[True] = True

    def gradients(self, pred: NDArray[np.float32], y: NDArray[np.float32], out: NDArray[np.float32]) -> None:
        out[:] = pred - y


WEIGHTED = gbm.Params(num_trees=4, max_depth=3, max_leaves=8, min_data_in_leaf=5, lambda_l2=1.0, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, seed=3)  # fmt: skip
WEIGHTED_LEAFWISE = dataclasses.replace(WEIGHTED, growth="leafwise", max_depth=-1, max_leaves=6)


def row_weights(n: int, seed: int = 5) -> NDArray[np.float32]:
    return np.random.default_rng(seed).choice(np.array([0.0, 0.5, 1.0, 3.0], np.float32), n)


class WeightedLogLoss:
    """`LogLoss` with row weights written by hand: its float32 gradient and hessian, each times the weight."""

    constant_hessian: Literal[False] = False

    def __init__(self, weights: NDArray[np.float32]) -> None:
        self.weights = weights

    def gradients(
        self, pred: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        gbm.LogLoss().gradients(pred, y, grad, hess)
        grad *= self.weights
        hess *= self.weights


@pytest.mark.parametrize("params", [WEIGHTED, WEIGHTED_LEAFWISE, dataclasses.replace(WEIGHTED, second_order=True),
                                    dataclasses.replace(WEIGHTED_LEAFWISE, second_order=True)])  # fmt: skip
def test_weights_are_the_weighted_l2_and_agree_with_the_oracle(params: gbm.Params) -> None:
    data = make_tiny(250, n_countries=8, seed=12)
    w = row_weights(230)
    weighted = gbm.train(params, data, rows=slice(10, 240), backend="cpu", weight=w)
    hooked = gbm.train(params, data, rows=slice(10, 240), backend="cpu", objective=Weighted(w))
    expected = oracle.fit(params, data, 10, 240, Weighted(w))
    fields = NODE_FIELDS if params.leafwise else FIELDS
    for name in (*fields, "leaf_count"):
        np.testing.assert_array_equal(getattr(weighted.forest, name), getattr(hooked.forest, name), err_msg=name)
    for name in fields:
        np.testing.assert_array_equal(getattr(weighted.forest, name), getattr(expected, name), err_msg=name)


@pytest.mark.parametrize("params", [WEIGHTED, WEIGHTED_LEAFWISE])
def test_unit_weights_give_the_unweighted_model(params: gbm.Params) -> None:
    data = make_tiny(250, n_countries=8, seed=12)
    ones = np.ones(230, np.float32)
    label = (data.target[10:240] > 0.5).astype(np.float32)
    for objective, y in ((None, None), (gbm.LogLoss(), label)):
        plain = gbm.train(params, data, rows=slice(10, 240), backend="cpu", y=y, objective=objective)
        unit = gbm.train(params, data, rows=slice(10, 240), backend="cpu", y=y, objective=objective, weight=ones)
        assert same_forest(plain, unit)


def test_weights_scale_an_objectives_gradients_and_hessians() -> None:
    data = make_tiny(250, n_countries=8, seed=12)
    w, label = row_weights(230, seed=8), (data.target[10:240] > 0.5).astype(np.float32)
    weighted = gbm.train(WEIGHTED, data, rows=slice(10, 240), backend="cpu", y=label, objective=gbm.LogLoss(),
                         weight=w)  # fmt: skip
    by_hand = gbm.train(WEIGHTED, data, rows=slice(10, 240), backend="cpu", y=label, objective=WeightedLogLoss(w))
    assert same_forest(weighted, by_hand)


def test_a_weighted_fit_starts_from_the_weighted_mean() -> None:
    data = make_tiny(300, n_countries=6, seed=4)
    w = row_weights(300, seed=2) + np.float32(0.25)
    root = dataclasses.replace(WEIGHTED, num_trees=1, min_data_in_leaf=300, lambda_l2=0.0, bagging_fraction=1.0,
                               bagging_freq=0)  # fmt: skip  # one leaf: its step is 0 at the weighted mean
    model = gbm.train(root, data, rows=slice(0, 300), backend="cpu", weight=w, boost_from_average=True)
    mean = np.average(np.asarray(data.target[:300], np.float64), weights=w.astype(np.float64))
    np.testing.assert_allclose(model.predict(data, rows=slice(0, 1), backend="cpu"), mean, rtol=1e-6)
    label = np.array([0, 1, 1, 0], np.float32)
    assert gbm.LogLoss.start(label, weight=np.array([3, 1, 1, 3], np.float32)) == pytest.approx(np.log(0.25 / 0.75))


def test_a_bag_and_lockstep_models_take_the_weights() -> None:
    data = make_tiny(300, n_countries=6, seed=9)
    w = row_weights(280, seed=3)
    rows = slice(10, 290)
    bag = gbm.train_seeds(WEIGHTED, data, rows=rows, backend="cpu", seeds=[1, 4], weight=w)
    for seed, model in zip((1, 4), bag, strict=True):
        alone = gbm.train(dataclasses.replace(WEIGHTED, seed=seed), data, rows=rows, backend="cpu", weight=w)
        assert same_forest(model, alone)
    ys = [np.asarray(data.target[rows], np.float32), (data.target[rows] > 0.5).astype(np.float32)]
    many = gbm.train_many(WEIGHTED, data, rows=rows, backend="cpu", ys=ys, weight=w)
    for y, model in zip(ys, many, strict=True):
        assert same_forest(model, gbm.train(WEIGHTED, data, rows=rows, backend="cpu", y=y, weight=w))


def test_a_gpu_fits_weights_as_the_cpu_does(gpu: str) -> None:
    data = make_tiny(250, n_countries=8, seed=12)
    w = row_weights(230)
    for params in (WEIGHTED, WEIGHTED_LEAFWISE):
        cpu = gbm.train(params, data, rows=slice(10, 240), backend="cpu", weight=w)
        assert same_forest(gbm.train(params, data, rows=slice(10, 240), backend=gpu, weight=w), cpu)


@pytest.mark.parametrize(
    ("weight", "message"),
    [
        (np.ones(10, np.float32), "one float32 per training row"),
        (np.full(100, -1.0, np.float32), "finite and non-negative"),
        (np.full(100, np.nan, np.float32), "finite and non-negative"),
        (np.full(100, np.inf, np.float32), "finite and non-negative"),
        (np.zeros(100, np.float32), "every sample weight is zero"),
    ],
)
def test_bad_weights_fail_loudly(weight: NDArray[np.float32], message: str) -> None:
    data = make_tiny(100, seed=17)
    with pytest.raises(ValueError, match=message):
        gbm.train(WEIGHTED, data, rows=slice(0, 100), backend="cpu", weight=weight)


def test_a_weighted_start_needs_the_objectives_weighted_start() -> None:
    class Started(Logistic):
        @staticmethod
        def start(y: NDArray[np.float32]) -> float:
            return 0.0

    data = make_tiny(100, seed=17)
    label = (data.target[:100] > 0.5).astype(np.float32)
    with pytest.raises(ValueError, match=r"start\(y, weight\)"):
        gbm.train(WEIGHTED, data, rows=slice(0, 100), backend="cpu", y=label, objective=Started(),
                  weight=np.ones(100, np.float32), boost_from_average=True)  # fmt: skip


def three_classes(data: Any, rows: slice) -> NDArray[np.float32]:
    """Class ids 0..2 of the rows: the target's terciles."""
    t = np.asarray(data.target, np.float64)
    return np.digitize(t[rows], np.quantile(t, [1 / 3, 2 / 3])).astype(np.float32)


def head(model: gbm.Booster, trees: int) -> gbm.Booster:
    """The model of its first ``trees`` trees."""
    return gbm.Booster(dataclasses.replace(model.params, num_trees=trees), model.forest.head(trees),
                       binning=model.binning, feature_names=model.feature_names, categories=model.categories,
                       named_edges=model.named_edges, named_n_bins=model.named_n_bins, train_rows=model.train_rows,
                       backend=model.backend)  # fmt: skip


class OneClass:
    """`Softmax`'s gradient and hessian of class ``k``, the other classes' scores held at ``scores``."""

    constant_hessian: Literal[False] = False

    def __init__(self, k: int, scores: NDArray[np.float32], labels: NDArray[np.float32], weights: Any = None) -> None:
        self.k, self.scores, self.labels, self.weights = k, scores, labels, weights

    def gradients(
        self, pred: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        scores = self.scores.copy()
        scores[self.k] = pred
        g, h = np.empty_like(scores), np.empty_like(scores)
        gbm.Softmax(len(scores)).gradients(scores, self.labels, g, h)
        grad[:], hess[:] = g[self.k], h[self.k]
        if self.weights is not None:
            grad *= self.weights
            hess *= self.weights


SECOND_ORDER = dataclasses.replace(WEIGHTED, second_order=True)
SECOND_LEAFWISE = dataclasses.replace(WEIGHTED_LEAFWISE, second_order=True)


@pytest.mark.parametrize(("params", "weighted"), [(WEIGHTED, False), (WEIGHTED_LEAFWISE, False), (WEIGHTED, True),
                                                  (SECOND_ORDER, True), (SECOND_LEAFWISE, False)])  # fmt: skip
def test_each_class_tree_is_a_continued_fit_under_the_others_scores(params: gbm.Params, weighted: bool) -> None:
    data = make_tiny(400, n_countries=6, seed=21)
    rows = slice(20, 380)
    labels, w = three_classes(data, rows), row_weights(360, seed=4) if weighted else None
    models = gbm.train_multiclass(params, data, rows=rows, backend="cpu", y=labels, n_classes=3, weight=w)
    assert len(models) == 3
    fields = NODE_FIELDS if params.leafwise else FIELDS
    for t in range(params.num_trees):
        scores = np.stack([head(m, t).predict(data, rows=rows, backend="cpu") if t else
                           np.zeros(360, np.float32) for m in models])  # fmt: skip
        for k, model in enumerate(models):
            one = dataclasses.replace(params, num_trees=1)
            again = gbm.train(one, data, rows=rows, backend="cpu", y=labels, objective=OneClass(k, scores, labels, w),
                              init_model=head(model, t) if t else None)  # fmt: skip
            for name in fields:
                np.testing.assert_array_equal(getattr(again.forest, name)[t], getattr(model.forest, name)[t],
                                              err_msg=f"{name} of class {k}'s tree {t}")  # fmt: skip


def test_a_gpu_fits_the_classes_as_the_cpu_does(gpu: str) -> None:
    data = make_tiny(400, n_countries=6, seed=21)
    labels = three_classes(data, slice(0, 400))
    for params in (WEIGHTED, WEIGHTED_LEAFWISE):
        cpu = gbm.train_multiclass(params, data, rows=slice(0, 400), backend="cpu", y=labels, n_classes=3)
        fast = gbm.train_multiclass(params, data, rows=slice(0, 400), backend=gpu, y=labels, n_classes=3)
        for one, other in zip(fast, cpu, strict=True):
            assert same_forest(one, other)


def test_a_multiclass_fit_starts_at_the_class_shares_and_stops_together() -> None:
    data = make_tiny(600, n_countries=6, seed=22)
    labels = three_classes(data, slice(0, 600))
    root = dataclasses.replace(WEIGHTED, num_trees=1, min_data_in_leaf=600, lambda_l2=0.0, bagging_fraction=1.0,
                               bagging_freq=0)  # fmt: skip
    models = gbm.train_multiclass(root, data, rows=slice(0, 400), backend="cpu", y=labels[:400], n_classes=3,
                                  boost_from_average=True)  # fmt: skip
    shares = np.log(np.bincount(labels[:400].astype(np.int64), minlength=3) / 400)
    first = np.array([m.predict(data, rows=slice(0, 1), backend="cpu")[0] for m in models])
    np.testing.assert_allclose(first, shares, atol=1e-5)  # one leaf each: its step is 0 at the shares
    long = dataclasses.replace(WEIGHTED, num_trees=60, learning_rate=0.5)
    valid = gbm.Validation(data, slice(400, 600), y=labels[400:])
    stopped = gbm.train_multiclass(long, data, rows=slice(0, 400), backend="cpu", y=labels[:400], n_classes=3,
                                   valid=[valid], early_stopping_rounds=3)  # fmt: skip
    best = stopped[0].best_iteration
    assert best is not None
    assert {m.n_trees for m in stopped} == {best + 1}
    assert stopped[0].n_trees < 60
    history = stopped[0].evals["valid_0"]
    assert min(history) == history[best]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"y": np.full(400, 3.0, np.float32)}, "class ids 0 .. 2"),
        ({"y": np.full(400, 0.5, np.float32)}, "class ids 0 .. 2"),
        ({"y": np.zeros(10, np.float32)}, "one class id per training row"),
        ({"objective": gbm.Softmax(4)}, "4 classes"),
        ({"metric": "l2", "callback": lambda _: None}, "is for multi-class fits|takes"),
    ],
)
def test_bad_multiclass_fits_fail_loudly(change: dict[str, Any], message: str) -> None:
    data = make_tiny(400, n_countries=6, seed=21)
    settings: dict[str, Any] = {"y": three_classes(data, slice(0, 400))} | change
    with pytest.raises(ValueError, match=message):
        gbm.train_multiclass(WEIGHTED, data, rows=slice(0, 400), backend="cpu", n_classes=3, **settings)


RANDOMISED = [
    dataclasses.replace(WEIGHTED, extra_trees=True),
    dataclasses.replace(WEIGHTED, feature_fraction_bynode=0.5),
    dataclasses.replace(WEIGHTED_LEAFWISE, extra_trees=True, feature_fraction_bynode=0.6),
    dataclasses.replace(WEIGHTED_LEAFWISE, feature_fraction_bynode=0.3, feature_fraction=1.0),
]


@pytest.mark.parametrize("params", RANDOMISED)
def test_extra_trees_and_bynode_sampling_agree_with_the_oracle(params: gbm.Params) -> None:
    data = make_tiny(300, n_countries=8, more_categorical=(3,), seed=14)
    model = gbm.train(params, data, rows=slice(10, 290), backend="cpu")
    expected = oracle.fit(params, data, 10, 290)
    for name in NODE_FIELDS if params.leafwise else FIELDS:
        np.testing.assert_array_equal(getattr(model.forest, name), getattr(expected, name), err_msg=name)
    plain = gbm.train(dataclasses.replace(params, extra_trees=False, feature_fraction_bynode=1.0), data,
                      rows=slice(10, 290), backend="cpu")  # fmt: skip
    assert not same_forest(model, plain)  # the draws choose other splits


def test_the_draws_follow_the_seed() -> None:
    data = make_tiny(400, seed=3)
    params = dataclasses.replace(WEIGHTED, extra_trees=True, num_trees=6)
    model = gbm.train(params, data, rows=slice(0, 400), backend="cpu")
    assert (model.forest.kind != 0).any()
    again = gbm.train(params, data, rows=slice(0, 400), backend="cpu")
    assert same_forest(model, again)  # drawn from the seed: the same trees every time
    other = gbm.train(dataclasses.replace(params, seed=4), data, rows=slice(0, 400), backend="cpu")
    assert not same_forest(model, other)


@pytest.mark.parametrize("params", RANDOMISED[1:3])
def test_a_continued_randomised_fit_is_the_longer_fit(params: gbm.Params) -> None:
    data = make_tiny(300, n_countries=6, seed=15)
    longer = gbm.train(dataclasses.replace(params, num_trees=6), data, rows=slice(0, 300), backend="cpu")
    first = gbm.train(dataclasses.replace(params, num_trees=2), data, rows=slice(0, 300), backend="cpu")
    both = gbm.train(
        dataclasses.replace(params, num_trees=4), data, rows=slice(0, 300), backend="cpu", init_model=first
    )
    assert same_forest(both, longer)


def test_a_gpu_draws_as_the_cpu_does(gpu: str) -> None:
    data = make_tiny(300, n_countries=8, seed=14)
    for params in RANDOMISED:
        cpu = gbm.train(params, data, rows=slice(10, 290), backend="cpu")
        assert same_forest(gbm.train(params, data, rows=slice(10, 290), backend=gpu), cpu)


def test_bad_bynode_fractions_fail_loudly() -> None:
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="feature_fraction_bynode"):
            dataclasses.replace(WEIGHTED, feature_fraction_bynode=bad)
    with pytest.raises(ValueError, match="wrong type"):
        dataclasses.replace(WEIGHTED, extra_trees=1)  # type: ignore[arg-type]
