"""train_many: batched fits are identical to separate fits, model by model."""

import numpy as np
import pytest

from idun import gbm
from idun.gbm.backends import CPUBackend, Lockstep
from idun.gbm.features import FeatureLayout
from tests.unit.gbm.tiny import make_tiny

FIELDS = ("kind", "feature", "threshold", "missing_left", "cat_left", "leaf_value")
BACKENDS = ["cpu", pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]


def targets(data: object, k: int, seed: int = 0) -> list[np.ndarray]:
    base = np.asarray(data.target, np.float32)  # type: ignore[attr-defined]
    generator = np.random.default_rng(seed)
    return [np.clip(base + generator.normal(0, 0.2, base.size), 0, 1).astype(np.float32) for _ in range(k)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_batched_model_equals_its_separate_fit(backend: str) -> None:
    data = make_tiny(600, n_countries=7, seed=31)
    params = gbm.Params(
        num_trees=4, max_depth=3, max_leaves=6, min_data_in_leaf=8, lambda_l2=1.0, feature_fraction=0.6,
        bagging_fraction=0.8, bagging_freq=1, cat_l2=0.0, cat_smooth=1.0, min_data_per_group=4,
    )  # fmt: skip
    ys = [y[20:580] for y in targets(data, 3)]
    models = gbm.train_many(params, data, rows=slice(20, 580), backend=backend, ys=ys, seeds=[0, 9])
    assert len(models) == 6
    for index, model in enumerate(models):
        seed, y = (0, 9)[index // 3], ys[index % 3]
        alone = gbm.train(
            gbm.Params(**{**params.to_dict(), "seed": seed}), data, rows=slice(20, 580), backend="cpu", y=y
        )
        for name in (*FIELDS, "leaf_count"):
            np.testing.assert_array_equal(getattr(model.forest, name), getattr(alone.forest, name), err_msg=name)
        assert model.params.seed == seed
    assert models[0].fit_report is models[1].fit_report  # one report per group


def test_defaults_and_bad_targets() -> None:
    data = make_tiny(200, seed=32)
    params = gbm.Params(num_trees=2, max_depth=2, max_leaves=4, min_data_in_leaf=5, seed=3)
    (model,) = gbm.train_many(params, data, rows=slice(0, 200), backend="cpu", ys=[data.target])
    alone = gbm.train(params, data, rows=slice(0, 200), backend="cpu")
    np.testing.assert_array_equal(model.forest.leaf_value, alone.forest.leaf_value)
    with pytest.raises(ValueError, match="one float32 per training row"):
        gbm.train_many(params, data, rows=slice(0, 200), backend="cpu", ys=[data.target[:10]])


def test_lockstep_checks_its_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Lockstep([])
    group = Lockstep([CPUBackend(), CPUBackend()])
    data = make_tiny(50)
    with pytest.raises(ValueError, match="1 targets for 2 models"):
        group.start(data, FeatureLayout.of(data), 0, np.zeros((1, 50), np.float32))
    assert group.gpu_seconds is None
    assert group.kernel_seconds == {}


def test_many_models_with_deep_trees_span_chunks(gpu: str) -> None:
    """Nine models on a GPU with up to 16 slots each (kernel histograms_obfuscated, K > 1): chunks of (model, slot)
    pairs cross model boundaries."""
    data = make_tiny(3000, n_countries=9, seed=33)
    params = gbm.Params(
        num_trees=3, max_depth=6, max_leaves=40, min_data_in_leaf=5, lambda_l2=1.0, feature_fraction=0.8,
        bagging_fraction=0.9, bagging_freq=1, cat_l2=0.0, cat_smooth=1.0, min_data_per_group=3, seed=4,
    )  # fmt: skip
    ys = targets(data, 9, seed=5)
    batched = gbm.train_many(params, data, rows=slice(0, 3000), backend=gpu, ys=ys)
    reference = gbm.train_many(params, data, rows=slice(0, 3000), backend="cpu", ys=ys)
    assert (batched[0].forest.kind != 0).sum() > 30
    for left, right in zip(batched, reference, strict=True):
        for name in (*FIELDS, "leaf_count"):
            np.testing.assert_array_equal(getattr(left.forest, name), getattr(right.forest, name), err_msg=name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_leafwise_batched_models_equal_their_separate_fits(backend: str) -> None:
    """Models of one seed split different leaves at each step; each is still its separate fit."""
    data = make_tiny(2000, n_countries=7, seed=34)
    params = gbm.Params(
        num_trees=3, growth="leafwise", max_depth=-1, max_leaves=20, min_data_in_leaf=8, lambda_l2=1.0,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, cat_l2=0.0, cat_smooth=1.0, min_data_per_group=4,
    )  # fmt: skip
    ys = targets(data, 4, seed=6)
    ys[3] = np.zeros_like(ys[3])  # a model that never splits, beside models that do
    models = gbm.train_many(params, data, rows=slice(0, 2000), backend=backend, ys=ys)
    for model, y in zip(models, ys, strict=True):
        alone = gbm.train(params, data, rows=slice(0, 2000), backend="cpu", y=y)
        for name in (*FIELDS, "left", "right"):
            np.testing.assert_array_equal(getattr(model.forest, name), getattr(alone.forest, name), err_msg=name)
    assert not models[3].forest.kind.any()
