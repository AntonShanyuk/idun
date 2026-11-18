from pathlib import Path

import numpy as np
import pytest

from idun.nn.inputs import MatrixData
from idun.nn.training import FitReport
from idun.tabm import TabMModel, TabMParams, directions, reference

TINY = TabMParams(members=3, blocks=2, width=5, dropout=0.0, n_classes=5, seed=11, batch_size=4096, epochs=10)
N_INPUTS = 7


def tiny_model(dtype: type[np.floating] = np.float64, params: TabMParams = TINY, seed: int = 3) -> TabMModel:
    generator = np.random.default_rng(seed)
    weights = reference.init(params, n_inputs=N_INPUTS).astype(dtype)
    for array in weights.arrays():  # off the initial pattern (adapters 1, biases equal)
        array[...] = array + generator.normal(0, 0.3, array.shape).astype(dtype)
    return TabMModel(params, weights, N_INPUTS, (0, 10), FitReport())


def expected_class(model: TabMModel, x: np.ndarray) -> np.ndarray:
    logits, _ = reference.forward(model.weights, x, None)
    return reference.predict(logits)


def test_input_gradients_match_finite_differences() -> None:
    model = tiny_model()
    x = np.random.default_rng(5).normal(size=(4, N_INPUTS))
    grad = directions.input_gradients(model, x)
    assert grad.shape == (4, N_INPUTS)
    assert grad.dtype == np.float64
    h = 1e-6
    numeric = np.empty_like(grad)
    for index in np.ndindex(x.shape):
        keep = x[index]
        x[index] = keep + h
        up = expected_class(model, x)[index[0]]
        x[index] = keep - h
        down = expected_class(model, x)[index[0]]
        x[index] = keep
        numeric[index] = (up - down) / (2 * h)
    assert np.abs(grad - numeric).max() / max(np.abs(numeric).max(), 1e-12) < 1e-6


def test_input_gradients_keep_float32_arithmetic_and_check_the_shape() -> None:
    model = tiny_model(np.float32)
    x = np.random.default_rng(5).normal(size=(3, N_INPUTS)).astype(np.float32)
    assert directions.input_gradients(model, x).dtype == np.float32
    with pytest.raises(ValueError, match="rows, 7"):
        directions.input_gradients(model, x[:, :5])


def test_only_single_target_classifiers_without_an_encoder() -> None:
    mse = tiny_model(
        params=TabMParams(
            members=3, blocks=2, width=5, n_classes=1, objective="mse", seed=11, batch_size=4096, epochs=10
        )
    )
    with pytest.raises(ValueError, match="single-target TabM classifier"):
        directions.check_surrogate(mse)
    two = reference.init(TINY, N_INPUTS, targets=2)
    multi = TabMModel(TINY, two, N_INPUTS, (0, 10), FitReport(), multi_target=True)
    with pytest.raises(ValueError, match="single-target"):
        directions.check_surrogate(multi)
    encoder_params = TabMParams(members=3, blocks=2, width=5, n_classes=5, seed=11, encoder="gru", encoder_embed=4,
                                encoder_hidden=6, batch_size=4096, epochs=10)  # fmt: skip
    encoded = TabMModel(encoder_params, reference.init(encoder_params, N_INPUTS, sequence_inputs=3), N_INPUTS,
                        (0, 10), FitReport(), sequence_inputs=3)  # fmt: skip
    with pytest.raises(ValueError, match="without a temporal encoder"):
        directions.check_surrogate(encoded)
    directions.check_surrogate(tiny_model())


def matrix(n_rows: int, seed: int = 9) -> MatrixData:
    generator = np.random.default_rng(seed)
    x = generator.normal(size=(n_rows, N_INPUTS)).astype(np.float32)
    return MatrixData(x, generator.integers(0, 5, n_rows), 5)


def test_egop_is_the_mean_outer_product_over_rows_and_models() -> None:
    models = [tiny_model(np.float32, seed=s) for s in (1, 2)]
    data = matrix(1200)
    rows = np.arange(1200)
    h = directions.egop(models, data, rows)
    assert h.shape == (N_INPUTS, N_INPUTS)
    assert h.dtype == np.float64
    np.testing.assert_allclose(h, h.T)
    assert np.linalg.eigvalsh(h).min() > -1e-12
    expected = np.zeros((N_INPUTS, N_INPUTS))
    for model in models:
        g = directions.input_gradients(model, data.x).astype(np.float64)
        expected += g.T @ g
    np.testing.assert_allclose(h, expected / (2 * 1200), rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(directions.sensitivity(h), np.diagonal(h))
    np.testing.assert_array_equal(directions.egop(models, data, rows), h)  # repeats its bits
    with pytest.raises(ValueError, match="at least one model"):
        directions.egop([], data, rows)
    with pytest.raises(ValueError, match="non-empty"):
        directions.egop(models, data, np.array([], np.int64))
    with pytest.raises(ValueError, match="inputs"):
        directions.egop(models, MatrixData(data.x[:, :5].copy(), data.y, 5), rows)


def test_top_directions_are_orthonormal_ordered_and_signed() -> None:
    generator = np.random.default_rng(2)
    a = generator.normal(size=(N_INPUTS, 20))
    h = a @ a.T / 20
    vectors, eigenvalues = directions.top_directions(h, 3)
    assert vectors.shape == (N_INPUTS, 3)
    assert eigenvalues.shape == (N_INPUTS,)
    assert (np.diff(eigenvalues) <= 0).all()
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(h @ vectors, vectors * eigenvalues[:3], atol=1e-10)
    pivots = vectors[np.abs(vectors).argmax(axis=0), np.arange(3)]
    assert (pivots > 0).all()
    with pytest.raises(ValueError, match="square"):
        directions.top_directions(a, 3)
    with pytest.raises(ValueError, match="k must be"):
        directions.top_directions(h, 8)


def test_fit_save_load_project(tmp_path: Path) -> None:
    models = [tiny_model(np.float32, seed=s) for s in (1, 2, 3)]
    data = matrix(300)
    found = directions.fit_directions(models, data, rows=np.arange(300), k=4)
    assert found.n_inputs == N_INPUTS
    assert found.k == 4
    assert found.vectors.dtype == np.float32
    assert 0 < found.top_share <= 1
    assert found.trace == pytest.approx(float(found.eigenvalues.sum()))
    assert found.surrogates == tuple(directions.model_digest(m) for m in models)
    assert len(set(found.surrogates)) == 3
    assert len(found.digest) == 64
    projected = found.project(data.x)
    assert projected.shape == (300, 4)
    assert projected.dtype == np.float32
    np.testing.assert_array_equal(projected, data.x @ found.vectors)
    with pytest.raises(ValueError, match="rows, 7"):
        found.project(data.x[:, :3])
    path = tmp_path / "d.npz"
    found.save(path)
    loaded = directions.Directions.load(path)
    np.testing.assert_array_equal(loaded.vectors, found.vectors)
    np.testing.assert_array_equal(loaded.eigenvalues, found.eigenvalues)
    assert loaded.surrogates == found.surrogates
    assert loaded.digest == found.digest
    other = tmp_path / "other.npz"
    np.savez(other, vectors=found.vectors)
    with pytest.raises(ValueError, match="idun-directions"):
        directions.Directions.load(other)
    models[0].save(other)
    with pytest.raises(ValueError, match="idun-directions v1"):
        directions.Directions.load(other)


def test_directions_are_checked() -> None:
    v = np.eye(N_INPUTS, 2, dtype=np.float32)
    e = np.ones(N_INPUTS)
    directions.Directions(v, e, ())
    with pytest.raises(ValueError, match="float32"):
        directions.Directions(v.astype(np.float64), e, ())
    with pytest.raises(ValueError, match="eigenvalues"):
        directions.Directions(v, e[:3], ())
    with pytest.raises(ValueError, match="k must be"):
        directions.Directions(np.zeros((N_INPUTS, 0), np.float32), e, ())


def test_project_chunks_do_not_change_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    found = directions.Directions(np.eye(N_INPUTS, 3, dtype=np.float32), np.ones(N_INPUTS), ())
    x = np.random.default_rng(1).normal(size=(10, N_INPUTS)).astype(np.float32)
    whole = found.project(x)
    monkeypatch.setattr(directions, "PROJECT_ROWS", 4)
    np.testing.assert_array_equal(found.project(x), whole)
