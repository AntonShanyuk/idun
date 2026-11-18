import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from idun.gbm import Params, split
from idun.gbm.features import MISSING_BIN, FeatureLayout
from idun.gbm.split import CATEGORY, NUMERIC, Split, best_split, leaf_gain
from tests import oracle
from tests.unit.gbm import split_reference
from tests.unit.gbm.tiny import TinyData, make_tiny


def histogram(
    data: TinyData, q: NDArray[np.int64], rows: NDArray[np.bool_], features: NDArray[np.int32]
) -> NDArray[np.int32]:
    """Histograms the slow way: np.add.at over the selected rows."""
    layout = FeatureLayout.of(data)
    offsets, cells = layout.offsets(features)
    bins, _, _ = oracle.feature_bins(data)
    hist = np.zeros((cells, 2), np.int64)
    for f, offset in zip(features, offsets, strict=True):
        np.add.at(hist[:, 0], offset + bins[f, rows], q[rows])
        np.add.at(hist[:, 1], offset + bins[f, rows], 1)
    return hist.astype(np.int32)


def as_tuple(split: Split | None) -> tuple[float, int, int, int, bool, int, int] | None:
    if split is None:
        return None
    return (
        split.gain,
        split.feature,
        split.kind,
        split.threshold,
        split.missing_left,
        split.cat_left,
        split.left_count,
    )


@settings(max_examples=300)
@given(
    seed=st.integers(0, 10_000),
    n_rows=st.integers(20, 300),
    named_bins=st.integers(1, 30),
    n_countries=st.integers(1, 20),
    categorical=st.sampled_from(["sorted", "one_vs_rest"]),
    min_data=st.integers(1, 20),
    lam=st.sampled_from([0.0, 2.0]),
    cat_smooth=st.sampled_from([0.0, 2.0]),
    group=st.integers(1, 20),
    data=st.data(),
)
def test_matches_the_exhaustive_search(
    seed: int,
    n_rows: int,
    named_bins: int,
    n_countries: int,
    categorical: str,
    min_data: int,
    lam: float,
    cat_smooth: float,
    group: int,
    data: st.DataObject,
) -> None:
    tiny = make_tiny(n_rows, named_bins=named_bins, n_countries=n_countries, missing=0.3, seed=seed)
    params = Params(
        min_data_in_leaf=min_data,
        lambda_l2=lam,
        categorical=categorical,  # type: ignore[arg-type]
        cat_smooth=cat_smooth,
        cat_l2=1.0,
        min_data_per_group=group,
        max_cat_threshold=5,
    )
    rng = np.random.default_rng(seed)
    q = rng.integers(-2, 3, n_rows).astype(np.int64)
    rows = rng.random(n_rows) < 0.8
    n_features = FeatureLayout.of(tiny).n_features
    features = np.array(sorted(data.draw(st.sets(st.integers(0, n_features - 1), min_size=1))), np.int32)
    scale = 0.37
    found = best_split(histogram(tiny, q, rows, features), features, FeatureLayout.of(tiny), params, scale)
    bins, kinds, real = oracle.feature_bins(tiny)
    assert as_tuple(found) == oracle.best_split(rows, q, bins, kinds, real, features.tolist(), params, scale)


def test_a_clear_numeric_split_is_found() -> None:
    tiny = make_tiny(200, n_named=1, n_obfuscated=2, missing=0.0, seed=1)
    q = np.where(tiny.codes[1] >= 3, 2, -2).astype(np.int64)  # obfuscated feature 1 separates the gradient
    features = np.arange(4, dtype=np.int32)
    split = best_split(
        histogram(tiny, q, np.ones(200, bool), features), features, FeatureLayout.of(tiny), Params(), 1.0
    )
    assert split is not None
    assert (split.feature, split.kind, split.threshold) == (2, NUMERIC, 2)
    assert split.left_count == int((tiny.codes[1] <= 2).sum())
    assert split.left_count + split.right_count == 200


def test_missing_rows_go_where_they_fit() -> None:
    tiny = make_tiny(300, n_named=1, n_obfuscated=1, missing=0.3, seed=2)
    named = tiny.named_bins[0]
    q = np.where((named >= 4) & (named != 255), 2, -2).astype(np.int64)
    q[named == 255] = 2  # missing rows behave like the high bins: they must go right
    features = np.array([0], np.int32)
    split = best_split(
        histogram(tiny, q, np.ones(300, bool), features), features, FeatureLayout.of(tiny), Params(), 1.0
    )
    assert split is not None
    assert (split.threshold, split.missing_left) == (3, False)


def test_equal_gains_go_to_the_lowest_feature() -> None:
    tiny = make_tiny(100, n_named=1, n_obfuscated=2, missing=0.0, seed=3)
    tiny.obfuscated[1] = tiny.obfuscated[0]  # two identical columns
    tiny.codes[1] = tiny.codes[0]
    q = np.where(tiny.codes[0] >= 2, 1, -1).astype(np.int64)
    features = np.array([1, 2], np.int32)
    split = best_split(
        histogram(tiny, q, np.ones(100, bool), features), features, FeatureLayout.of(tiny), Params(), 1.0
    )
    assert split is not None
    assert split.feature == 1


def test_no_split_without_gain_or_rows() -> None:
    tiny = make_tiny(100, seed=4)
    features = np.arange(FeatureLayout.of(tiny).n_features, dtype=np.int32)
    layout = FeatureLayout.of(tiny)
    flat = histogram(tiny, np.zeros(100, np.int64), np.ones(100, bool), features)
    assert best_split(flat, features, layout, Params(), 1.0) is None  # all gradients zero: no gain anywhere
    signal = histogram(tiny, np.where(tiny.codes[0] > 1, 2, -2).astype(np.int64), np.ones(100, bool), features)
    assert best_split(signal, features, layout, Params(min_data_in_leaf=51), 1.0) is None  # 100 < 2 * 51
    assert best_split(signal, features, layout, Params(), 0.0) is None  # scale 0: every gain is 0


def test_categorical_splits_carry_a_bitmask() -> None:
    tiny = make_tiny(400, n_named=1, n_obfuscated=1, n_countries=6, seed=5)
    q = np.where(np.isin(tiny.country, [1, 4]), 2, -2).astype(np.int64)
    features = np.array([2], np.int32)
    for method in ("sorted", "one_vs_rest"):
        params = Params(categorical=method, cat_l2=0.0, min_data_per_group=1)
        split = best_split(
            histogram(tiny, q, np.ones(400, bool), features), features, FeatureLayout.of(tiny), params, 1.0
        )
        assert split is not None
        assert split.kind == CATEGORY
        assert split.cat_left in ((1 << 1) | (1 << 4), 0b101101) if method == "sorted" else split.cat_left in (2, 16)


def test_leaf_gain() -> None:
    assert float(leaf_gain(4, 3, 0.5, 1.0)) == 1.0  # (4 * 0.5)^2 / (3 + 1)
    np.testing.assert_array_equal(leaf_gain(np.array([2, -2]), np.array([1, 1]), 1.0, 0.0), [4.0, 4.0])


def test_a_split_object_is_frozen() -> None:
    split = Split(1.0, 0, NUMERIC, 0, True, 0, 1, 1)
    with pytest.raises(AttributeError):
        split.gain = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("second", [False, True])
@pytest.mark.parametrize("missing", [True, False])
@pytest.mark.parametrize(("min_data", "lam"), [(1, 0.0), (20, 5.0), (400, 0.0), (3_000, 1.0)])
def test_the_compiled_numeric_search_is_the_numpy_one(missing: bool, min_data: int, lam: float, second: bool) -> None:
    rng = np.random.default_rng(min_data + int(missing))
    bins = 256 if missing else 5
    for n, n_features, spread in ((1, 1, 3), (3, 7, 1), (16, 24, 40), (5, 2, 0)):
        counts = rng.integers(0, 60, (n, n_features, bins)) * rng.integers(0, 2, (n, n_features, bins))
        q = rng.integers(-spread, spread + 1, (n, n_features, bins)) * counts  # spread 0: every gain ties
        q_h = rng.integers(0, 3, (n, n_features, bins)) * counts  # hessians' sums, zeros among them
        cells = np.stack([q, counts, q_h] if second else [q, counts], axis=-1).astype(np.int32)
        cells[:, 1:] = cells[:, :1]  # features equal to the first: ties across features too
        total_q, total_c = cells[:, 0, :, 0].sum(axis=1, dtype=np.int64), cells[:, 0, :, 1].sum(axis=1, dtype=np.int64)
        total_h = cells[:, 0, :, 2 if second else 1].sum(axis=1, dtype=np.int64)
        for scale, h_scale in ((0.37, 0.125), (1e-3, 0.05)):
            hessian = (2, h_scale) if second else (1, 1.0)
            parent = leaf_gain(total_q, np.multiply(total_h, hessian[1]), scale, lam)
            params = Params(min_data_in_leaf=min_data, lambda_l2=lam, min_sum_hessian_in_leaf=0.3 if second else 1e-3)
            found = split._numeric_block(cells, total_q, total_c, total_h, parent, scale, hessian, params,
                                         missing=missing)  # fmt: skip
            expected = split_reference.numeric_block(cells, total_q, total_c, total_h, parent, scale, hessian, params,
                                                     missing=missing)  # fmt: skip
            for one, other in zip(found, expected, strict=True):
                assert one.dtype == other.dtype
                np.testing.assert_array_equal(one, other)  # -inf where no candidate is valid, on both


@pytest.mark.parametrize("second", [False, True])
@pytest.mark.parametrize("categorical", ["sorted", "one_vs_rest"])
def test_the_compiled_categorical_search_is_the_python_one(categorical: str, second: bool) -> None:
    rng = np.random.default_rng(7)
    splits = 0
    for case in range(400):
        levels = int(rng.integers(1, 64))
        counts = rng.integers(0, 400, levels) * (rng.random(levels) < 0.8)
        q = rng.integers(-3, 4, levels) * counts if case % 4 else np.zeros(levels, np.int64)  # some all tied
        q_h = rng.integers(0, 3, levels) * counts  # hessians' sums, zeros among them
        cells = np.stack([q, counts, q_h] if second else [q, counts], axis=-1).astype(np.int32)
        total = (int(q.sum()), int(counts.sum()), int((q_h if second else counts).sum()))
        hessian = (2, 0.125) if second else (1, 1.0)
        params = Params(
            categorical=categorical,  # type: ignore[arg-type]
            min_data_in_leaf=int(rng.integers(1, 200)),
            lambda_l2=float(rng.choice([0.0, 5.0])),
            cat_l2=float(rng.choice([0.0, 10.0])),
            cat_smooth=float(rng.choice([0.0, 1.0, 10.0])),
            max_cat_threshold=int(rng.integers(1, 40)),
            min_data_per_group=int(rng.integers(1, 300)),
        )
        scale = float(rng.choice([0.37, 1e-3]))
        if total[1] < 2 * params.min_data_in_leaf:  # best_splits searches splittable nodes only
            continue
        parent = float(leaf_gain(total[0], total[2] * hessian[1], scale, params.lambda_l2))
        expected = split_reference.categorical(cells, total, parent, scale, hessian, params)
        found = split._categorical(cells, total, parent, scale, hessian, params)
        assert found == expected, case
        splits += expected is not None
    assert splits > 100  # most cases have a split to compare


def test_the_compiled_search_restricted_per_node_is_the_numpy_one() -> None:
    rng = np.random.default_rng(31)
    for missing, bins in ((True, 256), (False, 5)):
        n, n_features = 6, 9
        counts = rng.integers(0, 60, (n, n_features, bins)) * rng.integers(0, 2, (n, n_features, bins))
        q = rng.integers(-3, 4, (n, n_features, bins)) * counts
        cells = np.stack([q, counts], axis=-1).astype(np.int32)
        total_q, total_c = cells[:, 0, :, 0].sum(axis=1, dtype=np.int64), cells[:, 0, :, 1].sum(axis=1, dtype=np.int64)
        cells[:, 1:] = cells[:, :1]
        params = Params(min_data_in_leaf=5, lambda_l2=1.0)
        parent = leaf_gain(total_q, total_c, 0.37, 1.0)
        highest = (MISSING_BIN if missing else bins) - 1
        search = split.NodeSearch(rng.integers(0, 2, (n, n_features)).astype(np.uint8),
                                  rng.integers(-2, highest, (n, n_features)))  # fmt: skip
        found = split._numeric_block(cells, total_q, total_c, total_c, parent, 0.37, (1, 1.0), params,
                                     missing=missing, search=search)  # fmt: skip
        expected = split_reference.numeric_block(cells, total_q, total_c, total_c, parent, 0.37, (1, 1.0), params,
                                                 missing=missing, search=search)  # fmt: skip
        for one, other in zip(found, expected, strict=True):
            np.testing.assert_array_equal(one, other)
        assert np.isinf(found[0][search.allowed == 0]).all()
