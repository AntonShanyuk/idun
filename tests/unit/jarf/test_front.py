from pathlib import Path

import numpy as np
import pytest

from idun import gbm
from idun.data import PackedDataset
from idun.jarf import front
from idun.nn import inputs
from idun.tabm import directions


def random_directions(d: int, k: int, seed: int = 1) -> directions.Directions:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    eigenvalues = np.sort(np.random.default_rng(seed + 1).uniform(size=d))[::-1]
    return directions.Directions(q.astype(np.float32), eigenvalues, ("s0",))


def test_project_rows_chunks_the_dense_inputs(nn_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    found = random_directions(inputs.n_inputs(data), 3)
    rows = np.array([4, 0, data.n_rows - 1, 7, 7], np.int64)
    whole = found.project(inputs.dense_inputs(data, rows))
    np.testing.assert_array_equal(front.project_rows(data, found, rows), whole)
    np.testing.assert_array_equal(front.project_rows(data, found, rows), front.project_rows(data, found, rows))
    monkeypatch.setattr(directions, "PROJECT_ROWS", 2)
    # another chunk shape is another BLAS summation order of the float32 dot products: the last bit may move, which
    # is why the chunk is a constant and the directions are stored with the model that uses them
    np.testing.assert_allclose(front.project_rows(data, found, rows), whole, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="inputs"):
        front.project_rows(data, random_directions(5, 2), rows)


def test_quantile_cuts_and_bins() -> None:
    values = np.column_stack([np.arange(1000, dtype=np.float32), np.zeros(1000, np.float32)])
    cuts = front.quantile_cuts(values)
    assert len(cuts) == 2
    assert cuts[0].dtype == np.float32
    assert cuts[0].size == 254  # distinct quantiles
    assert cuts[1].size == 1  # a constant column has one distinct quantile
    np.testing.assert_allclose(cuts[0], np.quantile(values[:, 0], np.arange(1, 255) / 255), rtol=1e-6)
    bins = front.bin_projections(values, cuts)
    assert bins.shape == (2, 1000)
    assert bins.dtype == np.uint8
    assert bins[0, 0] == 0
    assert bins[0, -1] == 254
    assert (np.diff(bins[0].astype(int)) >= 0).all()
    np.testing.assert_array_equal(bins[1], 0)
    with pytest.raises(ValueError, match="does not fit"):
        front.bin_projections(values, cuts[:1])
    with pytest.raises(ValueError, match="with rows"):
        front.quantile_cuts(values[:0])


def test_extend_appends_binned_projections(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    k = 3
    found = random_directions(inputs.n_inputs(data), k)
    edge_rows = np.arange(0, data.n_rows, 2)
    extended = front.extend(data, found, edge_rows=edge_rows)
    names = ("tabm_direction_0", "tabm_direction_1", "tabm_direction_2")
    assert extended.n_rows == data.n_rows
    assert extended.binning == f"{data.binning}+jarf:{found.digest[:16]}"
    assert extended.named_features == (*data.named_features, *names)
    assert extended.obfuscated_features == data.obfuscated_features
    assert extended.feature_names == (*data.named_features, *names, *data.obfuscated_features, data.feature_names[-1])
    n_named = len(data.named_features)
    assert extended.named_edges.shape == (n_named + k, 255)
    assert np.isposinf(extended.named_edges[-1, -1])
    np.testing.assert_array_equal(extended.named_edges[:n_named], data.named_edges)
    np.testing.assert_array_equal(extended.named_bins[:n_named], data.named_bins)
    assert extended.named_n_bins[:n_named] == data.named_n_bins
    projected = front.project_rows(data, found, np.arange(data.n_rows))  # the same matmul shapes as extend
    cuts = front.quantile_cuts(front.project_rows(data, found, edge_rows))
    for j in range(k):
        assert extended.named_n_bins[n_named + j] == cuts[j].size + 1
        np.testing.assert_array_equal(extended.named_edges[n_named + j, : cuts[j].size], cuts[j])
        np.testing.assert_array_equal(extended.named_bins[n_named + j], np.searchsorted(cuts[j], projected[:, j]))
    assert extended.obfuscated is data.obfuscated
    assert extended.target is data.target
    params = gbm.Params(num_trees=3, max_depth=2, max_leaves=4, feature_fraction=1.0)
    booster = gbm.train(params, extended, rows=slice(0, extended.n_rows), backend="cpu")
    assert booster.feature_names == extended.feature_names
    assert booster.predict(extended, rows=slice(0, 5), backend="cpu").shape == (5,)
    with pytest.raises(ValueError, match="binning"):
        booster.predict(data, rows=slice(0, 5), backend="cpu")
    with pytest.raises(ValueError, match="already has features"):
        front.extend(data, found, edge_rows=edge_rows, prefix=data.obfuscated_features[0].rsplit("_", 1)[0])


def test_keep_reduces_the_obfuscated_features(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    wanted = [data.obfuscated_features[2], data.obfuscated_features[0]]
    kept = front.keep(data, wanted)
    assert kept.obfuscated_features == (data.obfuscated_features[0], data.obfuscated_features[2])  # the data's order
    assert kept.feature_names == (*data.named_features, *kept.obfuscated_features, data.feature_names[-1])
    assert kept.binning.startswith(f"{data.binning}+keep:")
    assert kept.obfuscated.shape == (2, data.obfuscated.shape[1])
    np.testing.assert_array_equal(kept.obfuscated[0], data.obfuscated[0])
    np.testing.assert_array_equal(kept.obfuscated[1], data.obfuscated[2])
    assert kept.named_bins is data.named_bins
    booster = gbm.train(
        gbm.Params(num_trees=2, max_depth=2, max_leaves=4), kept, rows=slice(0, kept.n_rows), backend="cpu"
    )
    assert booster.predict(kept, rows=slice(0, 4), backend="cpu").shape == (4,)
    assert front.keep(data, data.obfuscated_features).binning != kept.binning
    found = random_directions(inputs.n_inputs(data), 2)
    both = front.keep(front.extend(data, found, edge_rows=np.arange(data.n_rows)), wanted)
    assert both.named_features[-2:] == ("tabm_direction_0", "tabm_direction_1")
    assert both.obfuscated_features == kept.obfuscated_features
    assert "+jarf:" in both.binning
    assert "+keep:" in both.binning
    with pytest.raises(ValueError, match="not obfuscated features"):
        front.keep(data, ["nothing"])
    with pytest.raises(ValueError, match="twice"):
        front.keep(data, [wanted[0], wanted[0]])


def test_direction_names() -> None:
    assert front.direction_names(2) == ("tabm_direction_0", "tabm_direction_1")
    assert front.direction_names(1, "d") == ("d_0",)
