"""`train_tabm` and `TabMModel` with a temporal encoder: fits, predictions, model files, mismatched inputs."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from idun.data import PackedDataset, lag_index
from idun.tabm import Sequences, TabMParams, train
from idun_numerai.inputs import classes as numerai_classes

TINY = TabMParams(n_classes=5, members=2, blocks=2, width=16, dropout=0.1, batch_size=32, epochs=2, seed=3,
                  learning_rate=1e-2, encoder="gru", encoder_embed=8, encoder_hidden=6)  # fmt: skip
GPU = [pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]  # each, marked


def sequences(data: PackedDataset, channels: int = 3) -> Sequences:
    """Random channels and each row's lags 2, 1 and 0 at the panel's era spacing."""
    era = np.asarray(data.era)
    step = int(np.diff(np.unique(era)).min())
    index = lag_index(era, np.asarray(data.ticker), [2, 1, 0], step)
    values = np.random.default_rng(0).normal(size=(data.n_rows, channels)).astype(np.float32)
    return Sequences(values, index)


@pytest.mark.parametrize("gpu_backend", GPU)
def test_gpu_training_follows_the_reference(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    seq = sequences(data)
    assert (seq.index[:, :2] >= 0).any()  # some rows have a history
    fits = {name: train.train_tabm(TINY, data, rows=slice(10, 110), backend=name, classes=numerai_classes(data.target),
                                   sequences=seq) for name in ("reference", gpu_backend)}  # fmt: skip
    ref, gpu = fits["reference"], fits[gpu_backend]
    assert ref.sequence_inputs == gpu.sequence_inputs == 4
    np.testing.assert_allclose(gpu.report.epoch_loss, ref.report.epoch_loss, rtol=1e-5)
    for a, b in zip(gpu.weights.arrays(), ref.weights.arrays(), strict=True):
        assert np.abs(a - b).max() / max(np.abs(b).max(), 1e-30) < 1e-4
    rows = slice(100, 160)
    np.testing.assert_allclose(gpu.predict(data, rows=rows, backend=gpu_backend, sequences=seq),
                               ref.predict(data, rows=rows, backend="reference", sequences=seq), atol=2e-4)  # fmt: skip
    proba = gpu.predict_proba(data, rows=rows, backend=gpu_backend, chunk=16, sequences=seq)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("gpu_backend", GPU)
def test_two_gpu_fits_give_identical_bits(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    seq = sequences(data)
    params = dataclasses.replace(TINY, micro_batch=8)
    fits = [train.train_tabm(params, data, rows=slice(0, 100), backend=gpu_backend,
                             classes=numerai_classes(data.target), sequences=seq) for _ in range(2)]  # fmt: skip
    assert [a.tobytes() for a in fits[0].weights.arrays()] == [a.tobytes() for a in fits[1].weights.arrays()]


@pytest.mark.parametrize("gpu_backend", GPU)
def test_a_gpu_fit_with_an_encoder_resumes_bit_for_bit(nn_cache: Path, tmp_path: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    seq, labels = sequences(data), numerai_classes(data.target)
    params = dataclasses.replace(TINY, micro_batch=8, epochs=3)
    whole = train.train_tabm(params, data, rows=slice(0, 100), backend=gpu_backend, classes=labels, sequences=seq)

    def stop(epoch: int, loss: float, seconds: float) -> None:
        raise RuntimeError(f"stopped after epoch {epoch + 1}")

    with pytest.raises(RuntimeError, match="after epoch 1"):
        train.train_tabm(params, data, rows=slice(0, 100), backend=gpu_backend, classes=labels, sequences=seq,
                         checkpoint=tmp_path / "fit.ckpt.npz", on_epoch=stop)  # fmt: skip
    resumed = train.train_tabm(params, data, rows=slice(0, 100), backend=gpu_backend, classes=labels, sequences=seq,
                               checkpoint=tmp_path / "fit.ckpt.npz")  # fmt: skip
    assert [a.tobytes() for a in resumed.weights.arrays()] == [a.tobytes() for a in whole.weights.arrays()]
    assert resumed.report.epoch_loss == whole.report.epoch_loss


def test_an_encoder_model_round_trips_through_npz(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    seq = sequences(data)
    model = train.train_tabm(TINY, data, rows=slice(0, 64), backend="reference", classes=numerai_classes(data.target),
                             sequences=seq)  # fmt: skip
    model.save(tmp_path / "t.npz")
    with np.load(tmp_path / "t.npz") as archive:
        meta = json.loads(archive["meta"].tobytes())
    assert (meta["format_version"], meta["multi_target"], meta["sequence_inputs"]) == (3, False, 4)
    back = train.TabMModel.load(tmp_path / "t.npz")
    assert (back.params, back.sequence_inputs, back.multi_target) == (TINY, 4, False)
    assert all(np.array_equal(a, b) for a, b in zip(back.weights.arrays(), model.weights.arrays(), strict=True))
    first = slice(0, 20)
    np.testing.assert_array_equal(back.predict(data, rows=first, backend="reference", sequences=seq),
                                  model.predict(data, rows=first, backend="reference", sequences=seq))  # fmt: skip


def test_a_model_without_an_encoder_writes_an_earlier_file(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    plain = dataclasses.replace(TINY, encoder="none", encoder_embed=32, encoder_hidden=64)  # the defaults
    model = train.train_tabm(plain, data, rows=slice(0, 64), backend="reference", classes=numerai_classes(data.target))
    model.save(tmp_path / "p.npz")
    with np.load(tmp_path / "p.npz") as archive:
        meta = json.loads(archive["meta"].tobytes())
    assert meta["format_version"] == 1
    assert not {"encoder", "encoder_embed", "encoder_hidden", "sequence_inputs"} & {*meta, *meta["params"]}
    assert train.TabMModel.load(tmp_path / "p.npz").params == plain
    unused = dataclasses.replace(plain, encoder_embed=8)  # an unused setting is kept
    dataclasses.replace(model, params=unused).save(tmp_path / "u.npz")
    assert train.TabMModel.load(tmp_path / "u.npz").params == unused


def test_sequences_come_with_an_encoder_and_fit_the_data(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    seq, labels = sequences(data), numerai_classes(data.target)
    with pytest.raises(ValueError, match="takes the rows' sequences"):
        train.train_tabm(TINY, data, rows=slice(0, 64), backend="reference", classes=labels)
    with pytest.raises(ValueError, match="takes no sequences"):
        train.train_tabm(dataclasses.replace(TINY, encoder="none"), data, rows=slice(0, 64), backend="reference",
                         classes=labels, sequences=seq)  # fmt: skip
    short = Sequences(seq.channels[:-1], seq.index[:-1].clip(-1, data.n_rows - 2))
    with pytest.raises(ValueError, match="rows; the data has"):
        train.train_tabm(TINY, data, rows=slice(0, 64), backend="reference", classes=labels, sequences=short)
    model = train.train_tabm(TINY, data, rows=slice(0, 64), backend="reference", classes=labels, sequences=seq)
    with pytest.raises(ValueError, match="takes the rows' sequences"):
        model.predict(data, rows=slice(0, 10), backend="reference")
    with pytest.raises(ValueError, match="inputs per step; the model reads 4"):
        model.predict(data, rows=slice(0, 10), backend="reference", sequences=sequences(data, channels=2))


def test_the_model_s_parts_must_agree(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    model = train.train_tabm(TINY, data, rows=slice(0, 64), backend="reference", classes=numerai_classes(data.target),
                             sequences=sequences(data))  # fmt: skip
    with pytest.raises(ValueError, match="do not agree"):
        dataclasses.replace(model, sequence_inputs=0)
    with pytest.raises(ValueError, match="do not agree"):
        dataclasses.replace(model, params=dataclasses.replace(TINY, encoder="none"))
    with pytest.raises(ValueError, match="reads 4 inputs per step, not 5"):
        dataclasses.replace(model, sequence_inputs=5)
