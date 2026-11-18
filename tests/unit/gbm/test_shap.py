"""idun.gbm.shap: TreeSHAP contributions, against exact Shapley values by brute force and adding up to predictions."""

import itertools
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from idun import gbm
from idun.gbm import shap
from idun.gbm.backends.cpu import _bin, _goes_right
from idun.gbm.features import FeatureLayout
from tests.unit.gbm.tiny import TinyData, make_tiny


def fitted(growth: Literal["depthwise", "leafwise"], seed: int = 3, **settings: Any) -> tuple[gbm.Booster, TinyData]:
    data = make_tiny(900, n_countries=5, seed=seed)
    params = gbm.Params(num_trees=6, max_depth=3 if growth == "depthwise" else -1, max_leaves=8, min_data_in_leaf=5,
                        growth=growth, cat_smooth=1.0, min_data_per_group=3, seed=seed, **settings)  # fmt: skip
    return gbm.train(params, data, rows=slice(0, 700), backend="cpu"), data


def exact_shapley(nodes: shap.Nodes, go_right: list[bool], n_features: int) -> np.ndarray:
    """Shapley values of one tree's output by enumerating every feature subset (the covers weigh what is unknown)."""
    split, _, _, _, _, left, right, value, cover = nodes

    def expected(m: int, known: frozenset[int]) -> float:
        if left[m] < 0:
            return float(value[m])
        if int(split[m]) in known:
            return expected(int(right[m] if go_right[m] else left[m]), known)
        both = cover[left[m]] * expected(int(left[m]), known) + cover[right[m]] * expected(int(right[m]), known)
        return float(both / cover[m])

    phi = np.zeros(n_features + 1)
    for i in range(n_features):
        others = [f for f in range(n_features) if f != i]
        for size in range(n_features):
            for subset in itertools.combinations(others, size):
                weight = math.factorial(size) * math.factorial(n_features - size - 1) / math.factorial(n_features)
                phi[i] += weight * (expected(0, frozenset((*subset, i))) - expected(0, frozenset(subset)))
    phi[n_features] = expected(0, frozenset())
    return phi


def decisions(nodes: shap.Nodes, data: TinyData, row: int) -> list[bool]:
    split, kind, threshold, missing_left, cat_left, left, *_ = nodes
    layout = FeatureLayout.of(data)
    found = []
    for m in range(left.size):
        if left[m] < 0:
            found.append(False)
            continue
        f = int(split[m])
        b = _bin(layout.kind[f], layout.column[f], np.int64(row), data.named_bins, data.obfuscated, data.categorical)
        found.append(bool(_goes_right(kind[m], b, threshold[m], missing_left[m], cat_left[m])))
    return found


@pytest.mark.parametrize("growth", ["depthwise", "leafwise"])
def test_each_trees_contributions_are_its_exact_shapley_values(growth: Literal["depthwise", "leafwise"]) -> None:
    model, data = fitted(growth)
    n_features = len(data.feature_names)
    for tree in range(model.n_trees):
        one = model.forest.trees(tree, tree + 1)
        nodes = shap.tree_nodes(one, 0)
        found = shap.contributions(one, data, 700, 706, n_features)
        for row in range(700, 706):
            want = exact_shapley(nodes, decisions(nodes, data, row), n_features)
            np.testing.assert_allclose(found[row - 700], want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(("growth", "settings"), [("depthwise", {}), ("leafwise", {}),
                                                  ("depthwise", {"bagging_fraction": 0.7, "bagging_freq": 1}),
                                                  ("leafwise", {"categorical": "one_vs_rest"})])  # fmt: skip
def test_the_contributions_and_the_expected_value_add_up_to_the_prediction(
    growth: Literal["depthwise", "leafwise"], settings: dict[str, Any]
) -> None:
    model, data = fitted(growth, seed=5, **settings)
    found = model.predict_contributions(data, rows=slice(0, 900))
    assert found.shape == (900, len(data.feature_names) + 1)
    pred = model.predict(data, rows=slice(0, 900), backend="cpu").astype(np.float64)
    np.testing.assert_allclose(found.sum(axis=1), pred, atol=1e-5)
    assert np.all(found[:, -1] == found[0, -1])  # one expected value for every row


def test_the_leaf_counts_are_each_trees_in_bag_rows() -> None:
    model, _ = fitted("depthwise", bagging_fraction=0.6, bagging_freq=1)
    assert model.forest.leaf_count is not None
    totals = model.forest.leaf_count.sum(axis=1)
    assert np.all(totals < 700)
    assert np.all(totals > 300)  # about 60 % of the 700 training rows each time
    plain, _ = fitted("leafwise")
    assert plain.forest.leaf_count is not None
    assert np.all(plain.forest.leaf_count.sum(axis=1) == 700)


def test_the_counts_are_saved_and_a_model_without_them_has_no_contributions(tmp_path: Path) -> None:
    model, data = fitted("leafwise")
    model.save(tmp_path / "m.igbm")
    loaded = gbm.Booster.load(tmp_path / "m.igbm")
    np.testing.assert_array_equal(loaded.predict_contributions(data, rows=slice(0, 50)),
                                  model.predict_contributions(data, rows=slice(0, 50)))  # fmt: skip
    with np.load(tmp_path / "m.igbm", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "leaf_count"}
    with (tmp_path / "old.igbm").open("wb") as handle:
        np.savez(handle, allow_pickle=False, **arrays)
    old = gbm.Booster.load(tmp_path / "old.igbm")  # as a file without leaf counts
    assert old.forest.leaf_count is None
    np.testing.assert_array_equal(old.predict(data, rows=slice(0, 50), backend="cpu"),
                                  model.predict(data, rows=slice(0, 50), backend="cpu"))  # fmt: skip
    with pytest.raises(ValueError, match="no leaf counts"):
        old.predict_contributions(data, rows=slice(0, 50))


@pytest.mark.parametrize("growth", ["depthwise", "leafwise"])
def test_the_leaf_indices_give_the_prediction_tree_by_tree(growth: Literal["depthwise", "leafwise"]) -> None:
    model, data = fitted(growth)
    leaves = model.predict_leaves(data, rows=slice(0, 900))
    assert leaves.shape == (900, model.n_trees)
    total = np.zeros(900, np.float32)
    for tree in range(model.n_trees):
        total = (total + model.forest.leaf_value[tree, leaves[:, tree]]).astype(np.float32)  # in tree order
    np.testing.assert_array_equal(total, model.predict(data, rows=slice(0, 900), backend="cpu"))
