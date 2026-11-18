"""Multi-target boosting (experimental): one shared tree per iteration, per-target leaves."""

import dataclasses
import warnings

import numpy as np
import pytest

from idun import gbm
from idun.errors import MultiTargetQualityWarning
from idun.gbm.backends.cpu import CPUMultiTarget
from tests.unit.gbm.tiny import make_tiny

STRUCTURE = ("kind", "feature", "threshold", "missing_left", "cat_left")
PARAMS = gbm.Params(
    num_trees=4, max_depth=3, max_leaves=7, min_data_in_leaf=6, lambda_l2=1.0, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, cat_l2=0.0, cat_smooth=1.0, min_data_per_group=4, seed=2,
)  # fmt: skip


def noisy(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    return [np.clip(y + generator.normal(0, 0.25, y.size), 0, 1).astype(np.float32) for _ in range(k)]


def test_one_target_is_the_single_model() -> None:
    data = make_tiny(500, n_countries=6, seed=41)
    (multi,) = gbm.train_multi(PARAMS, data, rows=slice(0, 500), backend="cpu", ys=[data.target])
    single = gbm.train(PARAMS, data, rows=slice(0, 500), backend="cpu")
    for name in (*STRUCTURE, "leaf_value", "leaf_count"):
        np.testing.assert_array_equal(getattr(multi.forest, name), getattr(single.forest, name), err_msg=name)


def test_targets_share_the_structure_and_keep_their_leaves() -> None:
    data = make_tiny(800, n_countries=6, seed=42)
    ys = noisy(np.asarray(data.target), 3)
    backend = CPUMultiTarget()
    models = gbm.train_multi(PARAMS, data, rows=slice(0, 800), backend=backend, ys=ys)
    assert len(models) == 3
    for model in models[1:]:
        for name in STRUCTURE:
            assert getattr(model.forest, name) is getattr(models[0].forest, name)
    assert not np.array_equal(models[0].forest.leaf_value, models[1].forest.leaf_value)
    assert (models[0].forest.kind != 0).sum() >= 8
    for target, model in enumerate(models):  # training predictions = the running predictions, bit for bit
        np.testing.assert_array_equal(model.predict(data, rows=slice(0, 800), backend="cpu"), backend.pred[target])


def test_shared_splits_sum_the_targets_gains() -> None:
    """Brute force: the chosen root split maximizes the summed per-target gain over every candidate."""
    data = make_tiny(600, n_named=1, n_obfuscated=3, n_countries=4, missing=0.0, seed=43)
    ys = noisy(np.asarray(data.target), 2, seed=1)
    params = gbm.Params(num_trees=1, max_depth=1, max_leaves=2, min_data_in_leaf=20, lambda_l2=2.0)
    backend = CPUMultiTarget()
    (first, _) = gbm.train_multi(params, data, rows=slice(0, 600), backend=backend, ys=ys)
    q = backend.q.astype(np.int64)  # the quantized gradients of the only iteration
    scales = np.abs(-np.stack(ys)).max(axis=1) / 2  # pred = 0 before the first tree: g = -y
    codes = data.codes
    best: tuple[float, tuple[int, int] | None] = (-np.inf, None)
    for f in range(codes.shape[0]):
        for t in range(4):
            left = codes[f] <= t
            n_left, n_right = int(left.sum()), int((~left).sum())
            if min(n_left, n_right) < 20:
                continue
            gain = 0.0
            for target in range(2):
                total = q[target].sum() * scales[target]
                g_left = q[target][left].sum() * scales[target]
                gain += g_left**2 / (n_left + 2.0) + (total - g_left) ** 2 / (n_right + 2.0) - total**2 / (600 + 2.0)
            if gain > best[0] + 1e-9:
                best = (gain, (1 + f, t))
    assert (int(first.forest.feature[0, 0]), int(first.forest.threshold[0, 0])) == best[1]


def test_own_last_level_at_depth_one_is_each_targets_separate_fit() -> None:
    """With one level, every split is a target's own: K separate fits, bit for bit (same bag and feature draws)."""
    data = make_tiny(700, n_countries=5, seed=45)
    ys = noisy(np.asarray(data.target), 3, seed=2)
    params = dataclasses.replace(PARAMS, max_depth=1, max_leaves=2, num_trees=6)
    models = gbm.train_multi(params, data, rows=slice(0, 700), backend="cpu", ys=ys, own_last_level=True)
    for y, model in zip(ys, models, strict=True):
        single = gbm.train(params, data, rows=slice(0, 700), backend="cpu", y=y)
        for name in (*STRUCTURE, "leaf_value", "leaf_count"):
            np.testing.assert_array_equal(getattr(model.forest, name), getattr(single.forest, name), err_msg=name)


def test_own_last_level_shares_the_upper_levels_only() -> None:
    data = make_tiny(900, n_countries=6, seed=46)
    ys = noisy(np.asarray(data.target), 3, seed=3)
    backend = CPUMultiTarget()
    models = gbm.train_multi(PARAMS, data, rows=slice(0, 900), backend=backend, ys=ys, own_last_level=True)
    upper, last = slice(0, 3), slice(3, 7)  # depth 3: levels 0-1 shared, level 2 per target
    for model in models[1:]:
        for name in STRUCTURE:
            np.testing.assert_array_equal(
                getattr(model.forest, name)[:, upper], getattr(models[0].forest, name)[:, upper]
            )
    assert any(
        not np.array_equal(models[0].forest.feature[:, last], model.forest.feature[:, last]) for model in models[1:]
    )
    for target, model in enumerate(models):  # training predictions = the running predictions (per-target last level)
        np.testing.assert_array_equal(model.predict(data, rows=slice(0, 900), backend="cpu"), backend.pred[target])
    (one,) = gbm.train_multi(PARAMS, data, rows=slice(0, 900), backend="cpu", ys=ys[:1], own_last_level=True)
    single = gbm.train(PARAMS, data, rows=slice(0, 900), backend="cpu", y=ys[0])
    for name in (*STRUCTURE, "leaf_value", "leaf_count"):
        np.testing.assert_array_equal(getattr(one.forest, name), getattr(single.forest, name), err_msg=name)


def test_bad_targets_and_backends_raise() -> None:
    data = make_tiny(100, seed=44)
    with pytest.raises(ValueError, match="one float32 per training row"):
        gbm.train_multi(PARAMS, data, rows=slice(0, 100), backend="cpu", ys=[data.target[:5]])
    with pytest.raises(ValueError, match="unknown multi-target backend"):
        gbm.train_multi(PARAMS, data, rows=slice(0, 100), backend="tpu", ys=[data.target])
    leafwise = gbm.Params(num_trees=1, growth="leafwise", max_depth=-1, max_leaves=4)
    with pytest.raises(ValueError, match="depth-wise trees only"):
        gbm.train_multi(leafwise, data, rows=slice(0, 100), backend="cpu", ys=[data.target])


def test_more_than_one_target_warns_about_quality_and_one_does_not() -> None:
    data = make_tiny(300, n_countries=6, seed=44)
    with pytest.warns(MultiTargetQualityWarning, match="2 targets with one shared tree structure"):
        gbm.train_multi(PARAMS, data, rows=slice(0, 300), backend="cpu", ys=noisy(np.asarray(data.target), 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error", MultiTargetQualityWarning)
        gbm.train_multi(PARAMS, data, rows=slice(0, 300), backend="cpu", ys=[data.target])


def test_the_per_node_draws_are_not_taken_yet() -> None:
    data = make_tiny(300, n_countries=6, seed=41)
    for change in ({"extra_trees": True}, {"feature_fraction_bynode": 0.5}):
        with pytest.raises(ValueError, match="does not take"):
            gbm.train_multi(dataclasses.replace(PARAMS, **change), data, rows=slice(0, 300), backend="cpu",
                            ys=noisy(np.asarray(data.target), 2))  # fmt: skip
