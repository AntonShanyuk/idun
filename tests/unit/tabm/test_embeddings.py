import numpy as np
import pytest

from idun.tabm import embeddings as emb


def test_bins_are_the_distinct_quantiles_laid_out_as_rtdl_lays_them() -> None:
    x = np.column_stack([np.arange(9.0), np.repeat([0.0, 1.0, 2.0], 3), np.r_[np.zeros(8), 1.0]])
    bins = emb.fit_bins(x, 4)
    assert bins.components == 4  # column 0: edges 0, 2, 4, 6, 8; column 1: 0, 1, 2 (two bins); column 2: 0, 1
    np.testing.assert_allclose(bins.scale[0], [0.5] * 4)
    np.testing.assert_allclose(bins.shift[0], [0.0, -1.0, -2.0, -3.0])
    np.testing.assert_allclose(bins.scale[1], [1.0, 0.0, 0.0, 1.0])  # the last bin in the last component
    np.testing.assert_allclose(bins.shift[1], [0.0, 0.0, 0.0, -1.0])
    np.testing.assert_allclose(bins.scale[2], [0.0, 0.0, 0.0, 1.0])
    assert bins.single.tolist() == [False, False, True]
    for bad, message in ((np.ones((5, 1)), "fewer than two distinct"), (np.full((5, 1), np.nan), "fewer than two")):
        with pytest.raises(ValueError, match=message):
            emb.fit_bins(bad, 4)
    with pytest.raises(ValueError, match="at least 2"):
        emb.fit_bins(x, 1)


def test_the_encoding_clamps_as_rtdl() -> None:
    bins = emb.bins_of([np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 3.0]), np.array([-1.0, 1.0])])
    x = np.array([[-1.0, -1.0, -3.0], [0.5, 2.0, 0.0], [2.5, 4.0, 5.0]])
    q = emb.encode(bins, x)
    np.testing.assert_allclose(q[0], [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # below: q_0 negative
    np.testing.assert_allclose(q[1], [[0.5, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    np.testing.assert_allclose(q[2], [[1.0, 1.0, 0.5], [1.0, 0.0, 1.5], [0.0, 0.0, 3.0]])  # above: the last unclamped
    one = emb.bins_of([np.array([0.0, 2.0])])  # P = 1: nothing clamped
    np.testing.assert_allclose(emb.encode(one, np.array([[-2.0], [4.0]]))[:, 0, 0], [-1.0, 2.0])


def test_init_is_rtdls_version_b_and_seeded() -> None:
    weights = emb.init(3, 4, 8, seed=2)
    assert [a.shape for a in weights.arrays()] == [(3, 8), (3, 8), (3, 4, 8)]
    assert all(a.dtype == np.float32 for a in weights.arrays())
    bound = 1 / np.sqrt(8)
    assert np.abs(weights.linear).max() <= bound
    assert np.abs(weights.linear_bias).max() <= bound
    assert not weights.ple.any()  # zero: at init the embedding is the linear one
    again = emb.init(3, 4, 8, seed=2)
    assert all(np.array_equal(a, b) for a, b in zip(weights.arrays(), again.arrays(), strict=True))
    assert not np.array_equal(emb.init(3, 4, 8, seed=3).linear, weights.linear)


def test_gradients_match_finite_differences() -> None:
    generator = np.random.default_rng(5)
    x = generator.normal(size=(40, 3))
    bins = emb.fit_bins(x, 5)
    weights = emb.init(3, bins.components, 4, seed=1).astype(np.float64)
    weights.ple[...] = generator.normal(size=weights.ple.shape)
    probe = generator.normal(size=(40, 12))

    def loss(w: emb.EmbeddingWeights) -> float:
        return float((emb.forward(w, bins, x)[0] * probe).sum())

    embedded, q = emb.forward(weights, bins, x)
    assert embedded.shape == (40, 12)
    np.testing.assert_allclose(embedded[:, 4:8], x[:, 1:2] * weights.linear[1] + weights.linear_bias[1]
                               + q[:, 1] @ weights.ple[1])  # fmt: skip
    grads = emb.backward(x, q, probe)
    h = 1e-6
    for array, grad in zip(weights.arrays(), grads.arrays(), strict=True):
        numeric = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            keep = array[index]
            array[index] = keep + h
            up = loss(weights)
            array[index] = keep - h
            numeric[index] = (up - loss(weights)) / (2 * h)
            array[index] = keep
        np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-8)
