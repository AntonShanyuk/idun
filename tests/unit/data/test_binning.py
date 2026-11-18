import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.typing import NDArray

from idun.data import binning
from idun.data.layout import MAX_NAMED_BINS, MISSING_BIN

SAMPLES = hnp.arrays(
    np.float32,
    st.integers(0, 600),
    elements=st.one_of(
        st.floats(-4, 4, width=32),
        st.sampled_from([0.0, 0.25, 1.5, float("nan"), float("inf"), float("-inf")]),
    ),
)


def bins_of(values: NDArray[np.float32], edges: NDArray[np.float32]) -> NDArray[np.uint8]:
    return binning.apply_bins(values, binning.edge_table([edges])[0])


def oracle_bin(value: float, edges: NDArray[np.float32]) -> int:
    """Bin by the definition: the number of upper edges strictly below the value; NaN is missing."""
    if np.isnan(value):
        return MISSING_BIN
    return sum(1 for edge in edges.tolist() if edge < value)


def test_sample_rows_is_a_sorted_distinct_subset() -> None:
    rows = binning.sample_rows(1_000, 100, seed=0, refit=2)
    assert rows.dtype == np.int64
    assert rows.size == 100
    assert np.all(np.diff(rows) > 0)
    assert rows[0] >= 0
    assert rows[-1] < 1_000


def test_sample_rows_depends_on_seed_and_refit_only() -> None:
    first = binning.sample_rows(1_000, 100, seed=0, refit=0)
    np.testing.assert_array_equal(first, binning.sample_rows(1_000, 100, seed=0, refit=0))
    assert not np.array_equal(first, binning.sample_rows(1_000, 100, seed=0, refit=1))
    assert not np.array_equal(first, binning.sample_rows(1_000, 100, seed=1, refit=0))


def test_a_small_window_is_used_whole() -> None:
    np.testing.assert_array_equal(binning.sample_rows(5, 200_000, seed=0, refit=0), np.arange(5))


@pytest.mark.parametrize(("window", "size"), [(0, 5), (5, 0)])
def test_sample_rows_rejects_empty_requests(window: int, size: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        binning.sample_rows(window, size, seed=0, refit=0)


def test_few_distinct_values_each_get_a_bin_at_midpoints() -> None:
    edges = binning.quantile_edges(np.array([3, 1, 1, 2, 3, np.nan], np.float32))
    np.testing.assert_array_equal(edges, np.array([1.5, 2.5], np.float32))
    assert bins_of(np.array([1, 2, 3, 1.5, 0, 99, np.nan], np.float32), edges).tolist() == [0, 1, 2, 0, 0, 2, 255]


def test_adjacent_float32_values_stay_in_separate_bins() -> None:
    low = np.float32(1.0)
    high = np.nextafter(low, np.float32(2))
    edges = binning.quantile_edges(np.array([low, high], np.float32))
    assert edges.tolist() == [low]  # the midpoint rounds onto `high` in float32, so the edge falls back to `low`
    assert bins_of(np.array([low, high], np.float32), edges).tolist() == [0, 1]


def test_many_distinct_values_get_balanced_quantile_bins() -> None:
    values = np.random.default_rng(0).normal(size=200_000).astype(np.float32)
    edges = binning.quantile_edges(values)
    assert edges.size == MAX_NAMED_BINS - 1
    counts = np.bincount(bins_of(values, edges), minlength=MAX_NAMED_BINS)
    # 200,000 / 255 = 784.3 rows per bin; the sample has a few float32 ties, which can move a cut by a row
    assert counts.min() >= 783
    assert counts.max() <= 786


def test_heavy_ties_merge_quantile_cuts() -> None:
    values = np.concatenate([np.zeros(10_000), np.arange(300)]).astype(np.float32)  # one value holds 97 % of the rows
    edges = binning.quantile_edges(values)
    assert edges.size < MAX_NAMED_BINS - 1
    assert edges[0] == np.float32(0.5)


@given(SAMPLES, st.integers(1, 300))
def test_edges_are_strictly_increasing_and_bounded(values: NDArray[np.float32], max_bins: int) -> None:
    edges = binning.quantile_edges(values, max_bins)
    assert edges.dtype == np.float32
    assert edges.size <= max_bins - 1
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0)
    distinct = np.unique(values[np.isfinite(values)])
    if distinct.size <= max_bins:
        assert edges.size == max(distinct.size - 1, 0)


@given(SAMPLES)
def test_bins_follow_the_definition_and_keep_order(values: NDArray[np.float32]) -> None:
    edges = binning.quantile_edges(values)
    bins = bins_of(values, edges)
    assert bins.tolist() == [oracle_bin(v, edges) for v in values.tolist()]
    present = ~np.isnan(values)
    order = np.argsort(values[present], kind="stable")
    assert np.all(np.diff(bins[present][order].astype(int)) >= 0)
    assert np.all(bins[np.isnan(values)] == MISSING_BIN)
    assert np.all(bins[~np.isnan(values)] <= edges.size)


def test_edge_table_pads_with_infinity() -> None:
    table = binning.edge_table([np.array([1, 2], np.float32), np.array([], np.float32)])
    assert table.shape == (2, MAX_NAMED_BINS)
    assert table[0, :2].tolist() == [1, 2]
    assert np.all(np.isposinf(table[0, 2:]))
    assert np.all(np.isposinf(table[1]))


def test_edge_table_rejects_too_many_edges() -> None:
    with pytest.raises(ValueError, match="at most 254"):
        binning.edge_table([np.arange(MAX_NAMED_BINS, dtype=np.float32)])


def test_infinities_go_to_the_outer_bins() -> None:
    edges = np.array([0.0, 1.0], np.float32)
    assert bins_of(np.array([-np.inf, np.inf], np.float32), edges).tolist() == [0, 2]


def test_inputs_are_checked() -> None:
    with pytest.raises(TypeError, match="float32"):
        binning.quantile_edges(np.zeros(3))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_bins"):
        binning.quantile_edges(np.zeros(3, np.float32), 0)
    with pytest.raises(ValueError, match="edge_table"):
        binning.apply_bins(np.zeros(3, np.float32), np.zeros(MAX_NAMED_BINS, np.float32))
