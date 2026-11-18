import dataclasses
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from idun import gbm
from idun.gbm.train import newton_leaf_values
from tests import oracle
from tests.unit.gbm.tiny import TinyData, make_tiny

FIELDS = ("kind", "feature", "threshold", "missing_left", "cat_left", "leaf_value")
BACKENDS = ["cpu", pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]
PARAMS = gbm.Params(
    num_trees=4, max_depth=3, max_leaves=8, min_data_in_leaf=5, lambda_l2=1.0, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, seed=3,
)  # fmt: skip
LEAFWISE = dataclasses.replace(PARAMS, growth="leafwise", max_depth=-1, max_leaves=6)
ROWS = slice(10, 240)


class SquaredError:
    """The L2 objective written by hand: must reproduce the built-in device gradients exactly."""

    constant_hessian: Literal[True] = True

    def gradients(self, pred: NDArray[np.float32], y: NDArray[np.float32], out: NDArray[np.float32]) -> None:
        np.subtract(pred, y, out=out)


class Clipped:
    """L2 with its gradient clipped at ±delta and a unit hessian (not L2: the leaves must follow it)."""

    constant_hessian: Literal[True] = True

    def gradients(self, pred: NDArray[np.float32], y: NDArray[np.float32], out: NDArray[np.float32]) -> None:
        np.clip(pred - y, np.float32(-0.2), np.float32(0.2), out=out)


class Logistic:
    """Binary cross-entropy on the logit ``pred``: ``g = p − y``, ``h = p (1 − p)``."""

    constant_hessian: Literal[False] = False

    def gradients(
        self, pred: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        p = (1.0 / (1.0 + np.exp(-pred.astype(np.float64)))).astype(np.float32)
        np.subtract(p, y, out=grad)
        np.multiply(p, np.float32(1.0) - p, out=hess)


class Weighted:
    """L2 with a weight per training row, held by the objective: ``g = w (pred − y)``, ``h = w``."""

    constant_hessian: Literal[False] = False

    def __init__(self, weights: NDArray[np.float32]) -> None:
        self.weights = weights

    def gradients(
        self, pred: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        np.multiply(self.weights, pred - y, out=grad)
        hess[:] = self.weights


class Towards:
    """L2 towards the objective's own labels: it ignores ``y``."""

    constant_hessian: Literal[True] = True

    def __init__(self, labels: NDArray[np.float32]) -> None:
        self.labels = labels

    def gradients(self, pred: NDArray[np.float32], y: NDArray[np.float32], out: NDArray[np.float32]) -> None:
        np.subtract(pred, self.labels, out=out)


def panel() -> tuple[TinyData, NDArray[np.float32]]:
    """Return a panel and a 0/1 label of its training rows."""
    data = make_tiny(250, n_countries=8, seed=12)
    return data, (data.target[ROWS] > 0.5).astype(np.float32)


def weights() -> NDArray[np.float32]:
    return np.random.default_rng(5).choice(np.array([0.0, 0.5, 1.0, 3.0], np.float32), ROWS.stop - ROWS.start)


def same(left: object, right: object, fields: tuple[str, ...] = FIELDS) -> None:
    for name in fields:
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_hand_written_l2_is_the_built_in_fit(backend: str) -> None:
    data = make_tiny(200, seed=11)
    params = gbm.Params(num_trees=3, max_depth=3, max_leaves=8, min_data_in_leaf=5)
    built_in = gbm.train(params, data, rows=slice(0, 200), backend=backend)
    hooked = gbm.train(params, data, rows=slice(0, 200), backend=backend, objective=SquaredError())
    same(hooked.forest, built_in.forest)


SECOND = [dataclasses.replace(p, second_order=True) for p in (PARAMS, LEAFWISE)]  # the hessians' split search


@pytest.mark.parametrize("params", [PARAMS, LEAFWISE, *SECOND])
@pytest.mark.parametrize("make", [Clipped, Logistic, lambda: Weighted(weights())])
def test_a_custom_objective_agrees_with_the_oracle(params: gbm.Params, make: object) -> None:
    data, label = panel()
    booster = gbm.train(params, data, rows=ROWS, backend="cpu", y=label, objective=make())  # type: ignore[operator]
    expected = oracle.fit(params, data, ROWS.start, ROWS.stop, make(), label)  # type: ignore[operator]
    same(booster.forest, expected, (*FIELDS, "left", "right") if params.leafwise else FIELDS)


@pytest.mark.parametrize("params", [PARAMS, LEAFWISE, *SECOND])
@pytest.mark.parametrize("make", [Clipped, Logistic, lambda: Weighted(weights())])
def test_a_gpu_fits_a_custom_objective_as_the_cpu_does(gpu: str, params: gbm.Params, make: object) -> None:
    data, label = panel()
    cpu = gbm.train(params, data, rows=ROWS, backend="cpu", y=label, objective=make())  # type: ignore[operator]
    fast = gbm.train(params, data, rows=ROWS, backend=gpu, y=label, objective=make())  # type: ignore[operator]
    same(fast.forest, cpu.forest)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_leaves_follow_the_objective_not_the_labels(backend: str) -> None:
    data, label = panel()
    plain = gbm.train(PARAMS, data, rows=ROWS, backend=backend, y=label)
    hooked = gbm.train(PARAMS, data, rows=ROWS, backend=backend, objective=Towards(label))  # y: the panel's target
    same(hooked.forest, plain.forest)


@pytest.mark.parametrize("params", [PARAMS, SECOND[0]])
def test_unit_hessians_give_the_constant_hessian_leaves(params: gbm.Params) -> None:
    data, label = panel()
    unit = Weighted(np.ones(ROWS.stop - ROWS.start, np.float32))
    newton = gbm.train(params, data, rows=ROWS, backend="cpu", y=label, objective=unit)
    same(newton.forest, gbm.train(params, data, rows=ROWS, backend="cpu", y=label).forest)


def test_second_order_splits_are_opt_in() -> None:
    data, label = panel()
    first = gbm.train(PARAMS, data, rows=ROWS, backend="cpu", y=label, objective=Logistic())
    expected = oracle.fit(dataclasses.replace(PARAMS, second_order=False), data, ROWS.start, ROWS.stop, Logistic(),
                          label)  # fmt: skip
    same(first.forest, expected)  # the default: first-order splits under a NewtonObjective
    second = gbm.train(SECOND[0], data, rows=ROWS, backend="cpu", y=label, objective=Logistic())
    assert not np.array_equal(second.forest.feature, first.forest.feature) or not np.array_equal(
        second.forest.threshold, first.forest.threshold
    )  # the hessians choose other splits here


def test_newton_leaf_values_divide_by_the_hessian_sums() -> None:
    one = 2**32
    values = newton_leaf_values(
        np.array([one, -2 * one, 0, one]), np.array([one, one // 2, 0, 0]), np.array([3, 1, 0, 2]), 1.0, 0.5
    )
    assert values.dtype == np.float32
    assert values.tolist() == [-0.25, np.float32(2 / 1.5 * 0.5), 0.0, -0.5]  # the third leaf is empty
    with pytest.raises(ValueError, match="Newton step is undefined"):
        newton_leaf_values(np.array([one]), np.array([0]), np.array([2]), 0.0, 0.5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lockstep_models_take_one_objective_or_one_each(backend: str) -> None:
    data, label = panel()
    target = np.asarray(data.target[ROWS], np.float32)
    separate = [
        gbm.train(PARAMS, data, rows=ROWS, backend=backend, y=y, objective=objective)
        for y, objective in ((label, Logistic()), (target, Clipped()), (target, Logistic()))
    ]
    each = gbm.train_many(PARAMS, data, rows=ROWS, backend=backend, ys=[label, target],
                          objective=[Logistic(), Clipped()])  # fmt: skip
    same(each[0].forest, separate[0].forest)
    same(each[1].forest, separate[1].forest)
    shared = gbm.train_many(PARAMS, data, rows=ROWS, backend=backend, ys=[label, target], objective=Logistic())
    same(shared[0].forest, separate[0].forest)
    same(shared[1].forest, separate[2].forest)


@pytest.mark.parametrize(
    ("backend", "workers"),
    [("cpu", 1), pytest.param("metal", 3, marks=pytest.mark.metal), pytest.param("cuda", 2, marks=pytest.mark.cuda)],
)
def test_a_seed_bag_takes_an_objective(backend: str, workers: int) -> None:
    data, label = panel()
    bag = gbm.train_seeds(PARAMS, data, rows=ROWS, backend=backend, seeds=[4, 9, 2], y=label, workers=workers,
                          objective=Logistic())  # fmt: skip
    for seed, model in zip([4, 9, 2], bag, strict=True):
        alone = gbm.train(dataclasses.replace(PARAMS, seed=seed), data, rows=ROWS, backend=backend, y=label,
                          objective=Logistic())  # fmt: skip
        same(model.forest, alone.forest)


class Broken:
    """A `Logistic` whose hessians are ``fill``."""

    constant_hessian: Literal[False] = False

    def __init__(self, fill: float, **params: float) -> None:
        self.fill, self.params = fill, params  # params: the fit's own settings

    def gradients(
        self, pred: NDArray[np.float32], y: NDArray[np.float32], grad: NDArray[np.float32], hess: NDArray[np.float32]
    ) -> None:
        np.subtract(pred, y, out=grad)
        hess[:] = self.fill


class NoGradients:
    constant_hessian: Literal[True] = True


class NoFlag:
    def gradients(self, pred: object, y: object, out: object) -> None:
        raise AssertionError


@pytest.mark.parametrize(
    ("objective", "error", "message"),
    [
        (Broken(-1.0), ValueError, "finite and non-negative"),
        (Broken(float("nan")), ValueError, "finite and non-negative"),
        (Broken(float("inf")), ValueError, "finite and non-negative"),
        (Broken(1e9), OverflowError, "max h"),
        (Broken(0.0), ValueError, "Newton step is undefined"),
        (Broken(0.0, max_delta_step=0.5), ValueError, "Newton step is undefined"),
        (NoGradients(), ValueError, "an objective needs"),
        (NoFlag(), ValueError, "an objective needs"),
    ],
)
def test_bad_objectives_fail_loudly(objective: object, error: type[Exception], message: str) -> None:
    data = make_tiny(100, seed=17)
    params = gbm.Params(num_trees=2, max_depth=2, max_leaves=4, min_data_in_leaf=5, lambda_l2=0.0,
                        **getattr(objective, "params", {}))  # fmt: skip
    with pytest.raises(error, match=message):
        gbm.train(params, data, rows=slice(0, 100), backend="cpu", objective=objective)  # type: ignore[arg-type]


def test_the_objectives_must_match_the_targets() -> None:
    data, label = panel()
    with pytest.raises(ValueError, match="1 objectives for 2 targets"):
        gbm.train_many(PARAMS, data, rows=ROWS, backend="cpu", ys=[label, label], objective=[Logistic()])


def test_log_loss_is_lightgbms_binary_objective() -> None:
    loss = gbm.LogLoss()
    pred = np.array([-2.0, 0.0, 3.0], np.float32)
    y = np.array([0.0, 1.0, 1.0], np.float32)
    grad, hess = np.empty(3, np.float32), np.empty(3, np.float32)
    loss.gradients(pred, y, grad, hess)
    p = 1.0 / (1.0 + np.exp(-pred.astype(np.float64)))
    np.testing.assert_array_equal(grad, (p - y).astype(np.float32))
    np.testing.assert_array_equal(hess, (p * (1.0 - p)).astype(np.float32))
    assert loss.start(np.array([0, 0, 0, 1], np.float32)) == pytest.approx(np.log(1 / 3))
    with pytest.raises(ValueError, match="needs rows of both classes"):
        loss.start(np.ones(3, np.float32))


def test_softmax_is_lightgbms_multiclass_objective() -> None:
    scores = np.array([[0.0, 2.0, -1.0], [1.0, 2.0, 0.5], [-1.0, 2.0, 3.0]], np.float32)  # (K, n)
    labels = np.array([1.0, 0.0, 2.0], np.float32)
    grad, hess = np.empty_like(scores), np.empty_like(scores)
    gbm.Softmax(3).gradients(scores, labels, grad, hess)
    p = np.exp(scores.astype(np.float64)) / np.exp(scores.astype(np.float64)).sum(axis=0)
    onehot = np.eye(3)[labels.astype(np.int64)].T
    np.testing.assert_array_equal(grad, (p - onehot).astype(np.float32))
    np.testing.assert_array_equal(hess, (1.5 * p * (1 - p)).astype(np.float32))  # K / (K - 1)
    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-6)
    big = np.array([[1000.0], [0.0]], np.float32)  # stable: the largest score is subtracted first
    two_grad, two_hess = np.empty_like(big), np.empty_like(big)
    gbm.Softmax(2).gradients(big, np.array([0.0], np.float32), two_grad, two_hess)
    assert np.isfinite(two_grad).all()
    assert np.isfinite(two_hess).all()
    shares = gbm.Softmax(3).start(np.array([0, 1, 1, 2], np.float32), weight=np.array([2, 1, 1, 4], np.float32))
    np.testing.assert_allclose(shares, np.log([0.25, 0.25, 0.5]))
    with pytest.raises(ValueError, match="every class"):
        gbm.Softmax(3).start(np.array([0, 1, 1], np.float32))
    with pytest.raises(ValueError, match="at least two classes"):
        gbm.Softmax(1)
