import dataclasses
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from idun.data import PackedDataset
from idun.errors import MultiTargetQualityWarning
from idun.nn import inputs
from idun.nn.inputs import MISSING_CLASS, MatrixData
from idun.nn.training import Validation
from idun.tabm import TabMParams, reference, train
from idun.tabm.params import scheduled_rate
from idun_numerai.inputs import classes as numerai_classes

TINY = TabMParams(
    n_classes=5, members=2, blocks=2, width=16, dropout=0.1, batch_size=32, epochs=2, seed=3, learning_rate=1e-2
)
GPU = [pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]  # each, marked


def test_batches_cover_every_row_once_per_epoch_and_keep_the_ragged_last() -> None:
    plan = list(train.batches(TINY, lo=1000, n=70))
    assert [len(rows) for _, _, rows in plan] == [32, 32, 6] * 2
    assert [step for _, step, _ in plan] == list(range(6))
    assert [epoch for epoch, _, _ in plan] == [0, 0, 0, 1, 1, 1]
    for epoch in (0, 1):
        seen = np.concatenate([rows for e, _, rows in plan if e == epoch])
        np.testing.assert_array_equal(np.sort(seen), np.arange(1000, 1070))


@pytest.mark.parametrize("gpu_backend", GPU)
def test_gpu_training_follows_the_reference(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    epochs: list[tuple[int, float, float]] = []
    fits = {name: train.train_tabm(TINY, data, rows=slice(10, 110), backend=name,
                                   on_epoch=lambda e, loss, s: epochs.append((e, loss, s)),
                                   classes=numerai_classes(data.target))
            for name in ("reference", gpu_backend)}  # fmt: skip
    assert [e for e, _, _ in epochs] == [0, 1, 0, 1]
    ref, gpu = fits["reference"], fits[gpu_backend]
    assert ref.train_rows == gpu.train_rows == (10, 110)
    np.testing.assert_allclose(gpu.report.epoch_loss, ref.report.epoch_loss, rtol=1e-5)
    for got, want in zip(gpu.weights.arrays(), ref.weights.arrays(), strict=True):
        assert np.abs(got - want).max() / np.abs(want).max() < 1e-4
    pred = {name: fits[name].predict(data, rows=slice(0, 50), backend=name, chunk=16) for name in fits}
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], rtol=1e-4)
    assert len(gpu.report.epoch_seconds) == 2
    assert gpu.report.seconds >= sum(gpu.report.epoch_seconds) * 0.99


@pytest.mark.parametrize("gpu_backend", GPU)
def test_micro_batches_pass_through_training_and_prediction(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    split = dataclasses.replace(TINY, micro_batch=12)  # batches of 32, 32, 32, 4 rows: micro-batches 12, 12, 8 and 4
    ref = train.train_tabm(
        split, data, rows=slice(10, 110), backend="reference", classes=numerai_classes(data.target)
    )  # ignores micro_batch
    gpu = train.train_tabm(split, data, rows=slice(10, 110), backend=gpu_backend, classes=numerai_classes(data.target))
    assert gpu.params.micro_batch == 12
    np.testing.assert_allclose(gpu.report.epoch_loss, ref.report.epoch_loss, rtol=1e-5)
    for got, want in zip(gpu.weights.arrays(), ref.weights.arrays(), strict=True):
        assert np.abs(got - want).max() / np.abs(want).max() < 1e-4
    pred = {name: model.predict(data, rows=slice(0, 50), backend=name, chunk=16) for name, model in
            (("reference", ref), (gpu_backend, gpu))}  # fmt: skip
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], rtol=1e-4)


def test_a_model_round_trips_through_npz(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = TabMParams(n_classes=5, members=2, blocks=1, width=8, batch_size=64, epochs=1)
    model = train.train_tabm(params, data, rows=slice(0, 64), backend="reference", classes=numerai_classes(data.target))
    model.save(tmp_path / "m.npz")
    back = train.TabMModel.load(tmp_path / "m.npz")
    assert back.params == params
    assert (back.n_inputs, back.train_rows) == (model.n_inputs, (0, 64))
    assert back.report == model.report
    assert all(np.array_equal(a, b) for a, b in zip(back.weights.arrays(), model.weights.arrays(), strict=True))
    np.savez(tmp_path / "bad.npz", meta=np.frombuffer(b'{"format": "x"}', np.uint8))
    with pytest.raises(ValueError, match="not a idun-tabm"):
        train.TabMModel.load(tmp_path / "bad.npz")


def test_bad_arguments_raise(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    with pytest.raises(ValueError, match="unknown backend"):
        train.make_backend("rocm", TINY, 5, 1)
    with pytest.raises(ValueError, match="non-empty contiguous"):
        train.train_tabm(TINY, data, rows=slice(5, 5), backend="reference", classes=numerai_classes(data.target))
    with pytest.raises(ValueError, match="non-empty contiguous"):
        train.train_tabm(TINY, data, rows=slice(0, 10, 2), backend="reference", classes=numerai_classes(data.target))
    model = train.train_tabm(TabMParams(n_classes=5, members=1, blocks=1, width=4, batch_size=16, epochs=1), data,
                             rows=slice(0, 16), backend="reference", classes=numerai_classes(data.target))  # fmt: skip
    with pytest.raises(ValueError, match="contiguous slice"):
        model.predict(data, rows=slice(0, 10, 2), backend="reference")
    model.n_inputs += 1
    with pytest.raises(ValueError, match="the model expects"):
        model.predict(data, rows=slice(0, 10), backend="reference")


# -- multi-target heads -----------------------------------------------------------------------------------------------


def three_targets(data: PackedDataset) -> np.ndarray:
    """K = 3 targets on the 5 bins, the first the panel's own (as `test_inputs` builds K targets)."""
    target = np.asarray(data.target)
    return np.stack([target, 1.0 - target, np.roll(target, 7)]).astype(np.float32)


@pytest.mark.parametrize(
    ("backend", "params"),
    [("reference", TINY),
     *(pytest.param(gpu.values[0], params, marks=gpu.marks) for gpu in GPU
       for params in (TINY, dataclasses.replace(TINY, micro_batch=12), dataclasses.replace(TINY, storage="bf16")))],
)  # fmt: skip
def test_one_class_column_is_the_single_target_network(nn_cache: Path, backend: str, params: TabMParams) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    labels = numerai_classes(data.target)
    partial = labels.copy()
    partial[110:, 0] = MISSING_CLASS  # rows outside the training rows may lack a label
    fits = [train.train_tabm(params, data, rows=slice(10, 110), backend=backend, classes=c) for c in (labels, partial)]
    assert [(fit.multi_target, fit.targets) for fit in fits] == [(False, 1), (False, 1)]
    assert fits[0].report.epoch_loss == fits[1].report.epoch_loss
    for a, b in zip(fits[0].weights.arrays(), fits[1].weights.arrays(), strict=True):
        np.testing.assert_array_equal(a, b)
    pred = [fit.predict(data, rows=slice(0, 50), backend=backend, chunk=16) for fit in fits]
    assert pred[0].shape == (50,)
    np.testing.assert_array_equal(pred[0], pred[1])


@pytest.mark.parametrize("gpu_backend", GPU)
@pytest.mark.parametrize("micro", [0, 12])
def test_multi_target_gpu_training_follows_the_reference(nn_cache: Path, micro: int, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = dataclasses.replace(TINY, micro_batch=micro)
    targets = three_targets(data)
    fits = {name: train.train_tabm(params, data, rows=slice(10, 110), backend=name, classes=numerai_classes(targets))
            for name in ("reference", gpu_backend)}  # fmt: skip
    ref, gpu = fits["reference"], fits[gpu_backend]
    assert gpu.targets == ref.targets == 3
    assert gpu.weights.head.shape == (TINY.members, TINY.width, 15)
    np.testing.assert_allclose(gpu.report.epoch_loss, ref.report.epoch_loss, rtol=1e-5)
    for got, want in zip(gpu.weights.arrays(), ref.weights.arrays(), strict=True):
        assert np.abs(got - want).max() / np.abs(want).max() < 1e-4
    pred = {name: fits[name].predict(data, rows=slice(0, 50), backend=name, chunk=16) for name in fits}
    assert pred[gpu_backend].shape == pred["reference"].shape == (50, 3)
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], rtol=1e-4)
    again = train.train_tabm(params, data, rows=slice(10, 110), backend=gpu_backend, classes=numerai_classes(targets))
    assert again.report.epoch_loss == gpu.report.epoch_loss
    assert all(a.tobytes() == b.tobytes() for a, b in zip(again.weights.arrays(), gpu.weights.arrays(), strict=True))


def test_a_multi_target_model_round_trips_through_npz(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = TabMParams(n_classes=5, members=2, blocks=1, width=8, batch_size=64, epochs=1)
    model = train.train_tabm(
        params, data, rows=slice(0, 64), backend="reference", classes=numerai_classes(three_targets(data)[:2])
    )
    model.save(tmp_path / "m.npz")
    with np.load(tmp_path / "m.npz") as archive:
        assert json.loads(archive["meta"].tobytes())["format_version"] == 2
    back = train.TabMModel.load(tmp_path / "m.npz")
    assert (back.multi_target, back.targets, back.params) == (True, 2, params)
    assert all(np.array_equal(a, b) for a, b in zip(back.weights.arrays(), model.weights.arrays(), strict=True))
    pred = back.predict(data, rows=slice(0, 20), backend="reference")
    assert pred.shape == (20, 2)
    np.testing.assert_array_equal(pred, model.predict(data, rows=slice(0, 20), backend="reference"))
    single = train.train_tabm(
        params, data, rows=slice(0, 64), backend="reference", classes=numerai_classes(data.target)
    )
    single.save(tmp_path / "s.npz")
    with np.load(tmp_path / "s.npz") as archive:
        assert json.loads(archive["meta"].tobytes())["format_version"] == 1  # single-target files are unchanged
    assert not train.TabMModel.load(tmp_path / "s.npz").multi_target
    np.savez(tmp_path / "v5.npz", meta=np.frombuffer(b'{"format": "idun-tabm", "format_version": 5}', np.uint8))
    with pytest.raises(ValueError, match="v1 to v4"):
        train.TabMModel.load(tmp_path / "v5.npz")


def test_bad_classes_raise(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = TabMParams(n_classes=5, members=1, blocks=1, width=4, batch_size=16, epochs=1)
    labels = numerai_classes(data.target)
    bads: list[np.ndarray] = [labels[:-1], labels[:, 0], labels.astype(np.int64)]  # a row short, 1-D, int64
    for bad in bads:
        with pytest.raises(ValueError, match=r"classes must be an \(\d+, K\) uint8"):
            train.train_tabm(params, data, rows=slice(0, 16), backend="reference", classes=bad)
    high = labels.copy()
    high[3, 0] = 5
    with pytest.raises(ValueError, match=r"not 0\.\.4"):
        train.train_tabm(params, data, rows=slice(0, 16), backend="reference", classes=high)
    with pytest.raises(ValueError, match="needs its classes"):
        train.train_tabm(params, data, rows=slice(0, 16), backend="reference")
    matrix = MatrixData(np.zeros((16, 3), np.float32), np.zeros(16, np.int64), 5)
    with pytest.raises(ValueError, match="carries its own labels"):
        train.train_tabm(params, matrix, rows=slice(0, 16), backend="reference", classes=labels[:16])
    model = train.train_tabm(params, data, rows=slice(0, 16), backend="reference",
                             classes=numerai_classes(three_targets(data)))  # fmt: skip
    with pytest.raises(ValueError, match="not the targets' 5 outputs each"):
        dataclasses.replace(model, multi_target=False)


def test_more_than_one_target_warns_about_quality_and_one_does_not(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = dataclasses.replace(TINY, epochs=1)
    with pytest.warns(MultiTargetQualityWarning, match="3 target heads on one trunk"):
        train.train_tabm(
            params, data, rows=slice(0, 64), backend="reference", classes=numerai_classes(three_targets(data))
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", MultiTargetQualityWarning)
        train.train_tabm(
            params, data, rows=slice(0, 64), backend="reference", classes=numerai_classes(data.target[None, :])
        )


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_regression_on_a_matrix_learns_the_targets(backend: str, tmp_path: Path) -> None:
    generator = np.random.default_rng(2)
    x = generator.normal(size=(300, 4)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1] + generator.normal(0, 0.1, 300)).astype(np.float32)
    params = dataclasses.replace(TINY, n_classes=1, objective="mse", epochs=8)
    model = train.train_tabm(params, MatrixData(x, y, 1), rows=slice(0, 250), backend=backend)
    pred = model.predict(MatrixData(x, y, 1), rows=slice(250, 300), backend=backend)
    assert pred.shape == (50,)
    assert np.corrcoef(pred, y[250:])[0, 1] > 0.8
    model.save(tmp_path / "reg.npz")
    np.testing.assert_array_equal(
        train.TabMModel.load(tmp_path / "reg.npz").predict(MatrixData(x, y, 1), rows=slice(250, 300), backend=backend),
        pred,
    )
    with pytest.raises(ValueError, match="classification"):
        model.predict_proba(MatrixData(x, y, 1), rows=slice(0, 10), backend=backend)


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_the_weights_after_each_epoch_are_the_shorter_fits(backend: str) -> None:
    generator = np.random.default_rng(3)
    x = generator.normal(size=(120, 4)).astype(np.float32)
    data = MatrixData(x, (x[:, 0] > 0).astype(np.int64) + (x[:, 1] > 1), 3)
    params = dataclasses.replace(TINY, n_classes=3, epochs=3)
    seen: dict[int, list[bytes]] = {}

    def keep(epoch: int, weights: reference.Weights) -> None:
        seen[epoch] = [a.tobytes() for a in weights.arrays()]

    full = train.train_tabm(params, data, rows=slice(0, 120), backend=backend, on_weights=keep)
    assert sorted(seen) == [0, 1, 2]
    assert seen[2] == [a.tobytes() for a in full.weights.arrays()]
    for epochs in (1, 2):
        short = train.train_tabm(dataclasses.replace(params, epochs=epochs), data, rows=slice(0, 120), backend=backend)
        assert seen[epochs - 1] == [a.tobytes() for a in short.weights.arrays()]


# -- checkpoints ------------------------------------------------------------------------------------------------------


class Stop(Exception):  # noqa: N818
    """Raised from ``on_epoch``: a fit stopped after an epoch."""


def stop_after(epochs: int) -> Callable[[int, float, float], None]:
    def report(epoch: int, loss: float, seconds: float) -> None:
        if epoch + 1 == epochs:
            raise Stop

    return report


def three_classes(seed: int = 4) -> MatrixData:
    x = np.random.default_rng(seed).normal(size=(120, 4)).astype(np.float32)
    return MatrixData(x, (x[:, 0] > 0).astype(np.int64) + (x[:, 1] > 1), 3)


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_a_fit_resumed_from_its_checkpoint_is_the_uninterrupted_fit(backend: str, tmp_path: Path) -> None:
    data, params = three_classes(), dataclasses.replace(TINY, n_classes=3, epochs=3)
    whole = train.train_tabm(params, data, rows=slice(0, 120), backend=backend)
    path = tmp_path / "fit.ckpt.npz"
    for epochs in (1, 2):  # stopped after epoch 1, resumed and stopped after epoch 2
        with pytest.raises(Stop):
            train.train_tabm(params, data, rows=slice(0, 120), backend=backend, checkpoint=path,
                             on_epoch=stop_after(epochs))  # fmt: skip
    seen: list[int] = []
    resumed = train.train_tabm(params, data, rows=slice(0, 120), backend=backend, checkpoint=path,
                               on_epoch=lambda epoch, loss, seconds: seen.append(epoch))  # fmt: skip
    assert seen == [2]
    assert [a.tobytes() for a in resumed.weights.arrays()] == [a.tobytes() for a in whole.weights.arrays()]
    assert resumed.report.epoch_loss == whole.report.epoch_loss
    assert len(resumed.report.epoch_seconds) == 3
    assert resumed.report.seconds >= sum(resumed.report.epoch_seconds) - 1e-6
    again = train.train_tabm(params, data, rows=slice(0, 120), backend=backend, checkpoint=path,
                             on_epoch=lambda epoch, loss, seconds: seen.append(epoch))  # fmt: skip
    assert seen == [2]  # a finished checkpoint trains nothing
    assert [a.tobytes() for a in again.weights.arrays()] == [a.tobytes() for a in whole.weights.arrays()]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fit.ckpt.npz"]


def test_a_checkpoint_belongs_to_one_fit(tmp_path: Path) -> None:
    data, params = three_classes(), dataclasses.replace(TINY, n_classes=3, epochs=2)
    path = tmp_path / "fit.ckpt.npz"
    with pytest.raises(Stop):
        train.train_tabm(params, data, rows=slice(0, 120), backend="reference", checkpoint=path, on_epoch=stop_after(1))
    with pytest.raises(ValueError, match=r"another fit .*params"):
        train.train_tabm(dataclasses.replace(params, seed=4), data, rows=slice(0, 120), backend="reference",
                         checkpoint=path)  # fmt: skip
    with pytest.raises(ValueError, match=r"another fit .*train_rows"):
        train.train_tabm(params, data, rows=slice(0, 100), backend="reference", checkpoint=path)
    model = train.train_tabm(params, data, rows=slice(0, 120), backend="reference")
    model.save(tmp_path / "model.npz")
    with pytest.raises(ValueError, match="not a idun-tabm-checkpoint"):
        train.train_tabm(params, data, rows=slice(0, 120), backend="reference", checkpoint=tmp_path / "model.npz")
    np.savez(tmp_path / "bare.npz", x=np.zeros(1))
    with pytest.raises(ValueError, match="not a idun-tabm-checkpoint"):
        train.train_tabm(params, data, rows=slice(0, 120), backend="reference", checkpoint=tmp_path / "bare.npz")


# -- regression on a packed dataset -----------------------------------------------------------------------------------

REG = dataclasses.replace(TINY, n_classes=1, objective="mse")


def packed_targets(data: PackedDataset, n_targets: int) -> np.ndarray:
    """K continuous targets of the cache's rows as a page-aligned (n_rows, K) float32 target array."""
    out = inputs.target_array(data.n_rows, n_targets)
    out[...] = np.random.default_rng(6).normal(size=out.shape).astype(np.float32)
    return out


@pytest.mark.parametrize(
    ("backend", "params"),
    [("reference", REG),
     *(pytest.param(gpu.values[0], params, marks=gpu.marks) for gpu in GPU
       for params in (REG, dataclasses.replace(REG, micro_batch=12)))],
)  # fmt: skip
def test_regression_on_a_packed_dataset_is_the_matrix_fit(nn_cache: Path, backend: str, params: TabMParams) -> None:
    """TabM builds every batch on the host: a packed dataset's targets give the dense matrix's fit, bit for bit."""
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, 1)
    targets[110:, 0] = np.nan  # rows outside the training rows may lack a target
    packed = train.train_tabm(params, data, rows=slice(10, 110), backend=backend, targets=targets)
    matrix = MatrixData(inputs.dense_inputs(data, np.arange(data.n_rows)), np.nan_to_num(targets[:, 0]), 1)
    dense = train.train_tabm(params, matrix, rows=slice(10, 110), backend=backend)
    assert (packed.multi_target, packed.targets) == (False, 1)
    assert packed.report.epoch_loss == dense.report.epoch_loss
    for a, b in zip(packed.weights.arrays(), dense.weights.arrays(), strict=True):
        np.testing.assert_array_equal(a, b)
    pred = packed.predict(data, rows=slice(110, 160), backend=backend, chunk=16)
    assert pred.shape == (50,)
    np.testing.assert_array_equal(pred, dense.predict(matrix, rows=slice(110, 160), backend=backend, chunk=16))


@pytest.mark.parametrize("gpu_backend", GPU)
def test_multi_target_regression_on_the_gpu_follows_the_reference(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, 3)
    with pytest.warns(MultiTargetQualityWarning):
        fits = {b: train.train_tabm(REG, data, rows=slice(0, 100), backend=b, targets=targets)
                for b in ("reference", gpu_backend)}  # fmt: skip
    np.testing.assert_allclose(fits[gpu_backend].report.epoch_loss, fits["reference"].report.epoch_loss, rtol=1e-4)
    pred = {b: fit.predict(data, rows=slice(100, 150), backend=b, chunk=16) for b, fit in fits.items()}
    assert pred[gpu_backend].shape == (50, 3)
    assert fits[gpu_backend].multi_target
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], atol=1e-4)


def test_bad_targets_raise(nn_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, 2)
    bads: list[np.ndarray] = [targets[:-1], targets[:, 0], targets.astype(np.float64)]
    for bad in bads:
        with pytest.raises(ValueError, match=r"targets must be an \(\d+, K\) float32"):
            train.train_tabm(REG, data, rows=slice(0, 16), backend="reference", targets=bad)
    monkeypatch.setattr(inputs, "CHECK_ROWS", 4)  # the bad row lies in the third chunk of training rows
    broken = packed_targets(data, 2)
    broken[9, 0] = np.nan
    with pytest.raises(ValueError, match="every target of the training rows must be finite"):
        train.train_tabm(REG, data, rows=slice(0, 16), backend="reference", targets=broken)
    with pytest.raises(ValueError, match="needs its targets for objective 'mse'"):
        train.train_tabm(REG, data, rows=slice(0, 16), backend="reference", classes=numerai_classes(data.target))
    with pytest.raises(ValueError, match="needs its classes for objective 'cross_entropy'"):
        train.train_tabm(TINY, data, rows=slice(0, 16), backend="reference", targets=targets)
    matrix = MatrixData(np.zeros((16, 3), np.float32), np.zeros(16, np.float32), 1)
    with pytest.raises(ValueError, match="carries its own labels"):
        train.train_tabm(REG, matrix, rows=slice(0, 16), backend="reference", targets=targets[:16])


HLG = dataclasses.replace(TINY, n_classes=1, objective="hl_gauss", n_bins=16)


@pytest.mark.parametrize(
    ("backend", "params"),
    [("reference", HLG),
     *(pytest.param(gpu.values[0], params, marks=gpu.marks) for gpu in GPU
       for params in (HLG, dataclasses.replace(HLG, micro_batch=12)))],
)  # fmt: skip
def test_hl_gauss_on_a_packed_dataset_is_the_matrix_fit(nn_cache: Path, backend: str, params: TabMParams) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, 1)
    targets[0, 0] = 9.0  # beyond the support: clamped to the last bin
    packed = train.train_tabm(params, data, rows=slice(0, 100), backend=backend, targets=targets)
    matrix = MatrixData(inputs.dense_inputs(data, np.arange(data.n_rows)), targets[:, 0].copy(), 1)
    dense = train.train_tabm(params, matrix, rows=slice(0, 100), backend=backend)
    assert packed.report.epoch_loss == dense.report.epoch_loss
    assert packed.weights.head.shape == (params.members, params.width, 16)
    pred = packed.predict(data, rows=slice(100, 150), backend=backend, chunk=16)
    np.testing.assert_array_equal(pred, dense.predict(matrix, rows=slice(100, 150), backend=backend, chunk=16))
    assert pred.shape == (50,)
    assert (np.abs(pred) < 4.0).all()  # an expected bin centre lies in the support


@pytest.mark.parametrize("gpu_backend", GPU)
def test_hl_gauss_training_on_the_gpu_follows_the_reference(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, 1)
    fits = {b: train.train_tabm(HLG, data, rows=slice(0, 100), backend=b, targets=targets)
            for b in ("reference", gpu_backend)}  # fmt: skip
    np.testing.assert_allclose(fits[gpu_backend].report.epoch_loss, fits["reference"].report.epoch_loss, rtol=1e-4)
    pred = {b: fit.predict(data, rows=slice(100, 150), backend=b, chunk=16) for b, fit in fits.items()}
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], atol=1e-4)


def test_an_hl_gauss_model_round_trips_and_trains_one_target(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, 2)
    model = train.train_tabm(HLG, data, rows=slice(0, 64), backend="reference", targets=targets[:, :1].copy())
    model.save(tmp_path / "hlg.npz")
    loaded = train.TabMModel.load(tmp_path / "hlg.npz")
    assert loaded.params == HLG
    np.testing.assert_array_equal(loaded.predict(data, rows=slice(64, 96), backend="reference"),
                                  model.predict(data, rows=slice(64, 96), backend="reference"))  # fmt: skip
    with pytest.raises(ValueError, match="predict_proba needs a single-target classification model"):
        model.predict_proba(data, rows=slice(64, 96), backend="reference")
    with pytest.raises(ValueError, match="HL-Gauss trains one target"):  # before the multi-target warning
        train.train_tabm(HLG, data, rows=slice(0, 16), backend="reference", targets=targets)
    with pytest.raises(ValueError, match="needs its targets for objective 'hl_gauss'"):
        train.train_tabm(HLG, data, rows=slice(0, 16), backend="reference", classes=numerai_classes(data.target))


def test_the_learning_rate_schedules() -> None:
    base = dataclasses.replace(TINY, learning_rate=0.1)
    assert scheduled_rate(base, 5, 100) is None  # constant: the network's own rate, untouched
    cosine = dataclasses.replace(base, lr_schedule="cosine")
    assert scheduled_rate(cosine, 0, 100) == pytest.approx(0.1)
    assert scheduled_rate(cosine, 50, 100) == pytest.approx(0.05)
    linear = dataclasses.replace(base, lr_schedule="linear", lr_warmup_steps=10)
    assert scheduled_rate(linear, 0, 110) == pytest.approx(0.01)  # warm-up: (step + 1) / 10 of the rate
    assert scheduled_rate(linear, 9, 110) == pytest.approx(0.1)
    assert scheduled_rate(linear, 60, 110) == pytest.approx(0.05)
    warm = dataclasses.replace(base, lr_warmup_steps=4)
    assert scheduled_rate(warm, 10, 100) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="out of range: lr_schedule"):
        dataclasses.replace(base, lr_schedule="step")
    with pytest.raises(ValueError, match="out of range: lr_warmup_steps"):
        dataclasses.replace(base, lr_warmup_steps=-1)


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_a_step_at_a_set_rate_is_a_step_of_that_learning_rate(backend: str) -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(32, 6)).astype(np.float32), rng.integers(0, 5, 32).astype(np.uint8)
    rows = np.arange(32)
    set_rate, own_rate = (
        train.make_backend(backend, TINY, 6, 32),
        train.make_backend(backend, dataclasses.replace(TINY, learning_rate=3e-3), 6, 32),
    )
    try:
        weights = reference.init(TINY, 6)
        for net in (set_rate, own_rate):
            net.set_weights(weights)
        set_rate.set_learning_rate(3e-3)
        for step in range(3):
            assert set_rate.train_step(x, y, rows, step) == own_rate.train_step(x, y, rows, step)
        for a, b in zip(set_rate.weights().arrays(), own_rate.weights().arrays(), strict=True):
            np.testing.assert_array_equal(a, b)
    finally:
        set_rate.close()
        own_rate.close()


def test_a_schedule_trains_and_needs_the_seeded_plan(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    labels = numerai_classes(data.target)
    flat = train.train_tabm(TINY, data, rows=slice(10, 110), backend="reference", classes=labels)
    cosine = train.train_tabm(dataclasses.replace(TINY, lr_schedule="cosine", lr_warmup_steps=2), data,
                              rows=slice(10, 110), backend="reference", classes=labels)  # fmt: skip
    assert not np.array_equal(flat.weights.head, cosine.weights.head)
    with pytest.raises(ValueError, match="not with batches="):
        train.train_tabm(dataclasses.replace(TINY, lr_schedule="cosine"), data, rows=slice(10, 110),
                         backend="reference", classes=labels, batches=lambda epoch: [np.arange(10, 42)])  # fmt: skip


def matrix(n: int, seed: int, n_classes: int) -> MatrixData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6)).astype(np.float32)
    signal = x[:, 0] + 0.5 * x[:, 1]
    if n_classes == 1:
        return MatrixData(x, (signal + rng.normal(0, 0.3, n)).astype(np.float32), 1)
    return MatrixData(x, np.digitize(signal, [-0.5, 0.5]).astype(np.int64), n_classes)


@pytest.mark.parametrize("n_classes", [3, 1])
def test_a_validated_fit_keeps_its_best_epochs_weights(n_classes: int) -> None:
    params = dataclasses.replace(TINY, epochs=6, n_classes=n_classes, objective="cross_entropy" if n_classes > 1
                                 else "mse")  # fmt: skip
    data, valid = matrix(160, 1, n_classes), Validation(matrix(80, 2, n_classes), slice(0, 80))
    seen: dict[int, reference.Weights] = {}
    model = train.train_tabm(params, data, rows=slice(0, 160), backend="reference", valid=valid,
                             on_weights=seen.__setitem__)  # fmt: skip
    report = model.report
    assert len(report.valid_score) == 6
    assert report.best_epoch == int(np.argmin(report.valid_score))
    for got, want in zip(model.weights.arrays(), seen[report.best_epoch].arrays(), strict=True):
        np.testing.assert_array_equal(got, want)
    best = train.TabMModel(params, seen[report.best_epoch], 6, (0, 160), report)
    if n_classes == 1:
        pred = best.predict(valid.data, rows=slice(0, 80), backend="reference")
        assert report.valid_score[report.best_epoch] == pytest.approx(np.mean((pred - valid.data.y) ** 2), rel=1e-6)
    else:
        proba = best.predict_proba(valid.data, rows=slice(0, 80), backend="reference")
        assert report.valid_score[report.best_epoch] == np.mean(np.argmax(proba, axis=1) != valid.data.y)


def test_patience_stops_after_that_many_epochs_without_a_better_one() -> None:
    params = dataclasses.replace(TINY, epochs=40, n_classes=1, objective="mse", learning_rate=0.05)
    data, valid = matrix(160, 3, 1), Validation(matrix(80, 4, 1), slice(0, 80))
    model = train.train_tabm(params, data, rows=slice(0, 160), backend="reference", valid=valid, patience=2)
    report = model.report
    assert report.best_epoch is not None
    assert len(report.valid_score) == len(report.epoch_loss) == min(40, report.best_epoch + 3)
    assert min(report.valid_score[report.best_epoch + 1 :], default=np.inf) >= report.valid_score[report.best_epoch]


def test_bad_validation_settings_fail_loudly() -> None:
    data = matrix(100, 5, 3)
    fit = {"rows": slice(0, 100), "backend": "reference"}
    params = dataclasses.replace(TINY, n_classes=3)
    with pytest.raises(ValueError, match="patience needs a validation set"):
        train.train_tabm(params, data, patience=3, **fit)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="has 1 classes, the network 3"):
        train.train_tabm(params, data, valid=Validation(matrix(40, 6, 1), slice(0, 40)), **fit)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not with a checkpoint"):
        train.train_tabm(params, data, valid=Validation(data, slice(0, 50)), checkpoint="x.npz", **fit)  # type: ignore[arg-type]


@pytest.mark.parametrize("gpu_backend", GPU)
def test_a_validated_fit_on_the_gpu(gpu_backend: str) -> None:
    params = dataclasses.replace(TINY, epochs=3, n_classes=3)
    model = train.train_tabm(params, matrix(160, 7, 3), rows=slice(0, 160), backend=gpu_backend,
                             valid=Validation(matrix(80, 8, 3), slice(0, 80)), patience=1)  # fmt: skip
    assert 2 <= len(model.report.valid_score) <= 3
    assert model.report.best_epoch == int(np.argmin(model.report.valid_score))


# -- sample weights ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("gpu_backend", GPU)
def test_weighted_gpu_training_follows_the_reference_and_unit_weights_change_nothing(
    nn_cache: Path, gpu_backend: str
) -> None:
    data, rows = PackedDataset.open(nn_cache, binning="r1"), slice(10, 110)
    classes = numerai_classes(data.target)
    weight = np.random.default_rng(6).choice(np.array([0.0, 0.5, 1.0, 3.0], np.float32), 100)
    fits = {name: train.train_tabm(TINY, data, rows=rows, backend=name, classes=classes, weight=weight)
            for name in ("reference", gpu_backend)}  # fmt: skip
    np.testing.assert_allclose(fits[gpu_backend].report.epoch_loss, fits["reference"].report.epoch_loss, rtol=1e-5)
    for got, want in zip(fits[gpu_backend].weights.arrays(), fits["reference"].weights.arrays(), strict=True):
        assert np.abs(got - want).max() / np.abs(want).max() < 1e-4
    plain = train.train_tabm(TINY, data, rows=rows, backend=gpu_backend, classes=classes)
    ones = train.train_tabm(
        TINY, data, rows=rows, backend=gpu_backend, classes=classes, weight=np.ones(100, np.float32)
    )
    assert [a.tobytes() for a in ones.weights.arrays()] == [a.tobytes() for a in plain.weights.arrays()]
    assert ones.report.epoch_loss == plain.report.epoch_loss
    assert not np.array_equal(fits[gpu_backend].weights.arrays()[0], plain.weights.arrays()[0])


def test_bad_weights_raise() -> None:
    data, params = three_classes(), dataclasses.replace(TINY, n_classes=3)
    for bad, message in (
        (np.ones(119), "one weight per training row"),
        (np.full(120, np.nan), "finite"),
        (np.full(120, -1.0), "non-negative"),
        (np.zeros(120), "every sample weight is zero"),
    ):
        with pytest.raises(ValueError, match=message):
            train.train_tabm(params, data, rows=slice(0, 120), backend="reference", weight=bad)

    def loss(outputs: np.ndarray, rows: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.0, np.zeros_like(outputs)

    with pytest.raises(ValueError, match="pass no weight with loss="):
        train.train_tabm(params, data, rows=slice(0, 120), backend="reference", loss=loss, weight=np.ones(120))


def test_zero_weights_leave_rows_out_of_the_loss() -> None:
    data, params = three_classes(), dataclasses.replace(TINY, n_classes=3, epochs=1)
    weight = np.ones(120, np.float32)
    weight[60:] = 0.0
    report: list[float] = []
    train.train_tabm(params, data, rows=slice(0, 120), backend="reference", weight=weight,
                     on_epoch=lambda epoch, loss, seconds: report.append(loss))  # fmt: skip
    assert np.isfinite(report).all()


# -- piecewise-linear embeddings --------------------------------------------------------------------------------------

PWL = dataclasses.replace(TINY, n_classes=3, num_emb_type="pwl", num_emb_n_bins=8, d_embedding=4)


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_an_embedded_fit_follows_the_reference_and_round_trips(backend: str, tmp_path: Path) -> None:
    data = three_classes()
    model = train.train_tabm(PWL, data, rows=slice(0, 120), backend=backend, embedded=3)
    assert model.weights.bins is not None
    assert model.weights.bins.scale.shape == (3, 8)  # four inputs, the first three embedded; 8 quantile bins each
    want = train.train_tabm(PWL, data, rows=slice(0, 120), backend="reference", embedded=3)
    for got, expected in zip(model.weights.arrays(), want.weights.arrays(), strict=True):
        assert np.abs(got - expected).max() / max(np.abs(expected).max(), 1e-30) < 1e-4
    model.save(tmp_path / "m.npz")
    with np.load(tmp_path / "m.npz") as archive:
        assert json.loads(archive["meta"].tobytes())["format_version"] == 4
    loaded = train.TabMModel.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(loaded.predict(data, rows=slice(0, 120), backend=backend),
                                  model.predict(data, rows=slice(0, 120), backend=backend))  # fmt: skip
    plain = train.train_tabm(dataclasses.replace(TINY, n_classes=3), data, rows=slice(0, 120), backend="reference")
    plain.save(tmp_path / "p.npz")
    with np.load(tmp_path / "p.npz") as archive:
        meta = json.loads(archive["meta"].tobytes())
    assert meta["format_version"] == 1  # without embeddings, the file of earlier versions
    assert "num_emb_type" not in meta["params"]


def test_an_embedded_fit_resumes_from_its_checkpoint(tmp_path: Path) -> None:
    data, params = three_classes(), dataclasses.replace(PWL, epochs=3)
    whole = train.train_tabm(params, data, rows=slice(0, 120), backend="reference")
    path = tmp_path / "fit.ckpt.npz"
    with pytest.raises(Stop):
        train.train_tabm(params, data, rows=slice(0, 120), backend="reference", checkpoint=path,
                         on_epoch=stop_after(1))  # fmt: skip
    resumed = train.train_tabm(params, data, rows=slice(0, 120), backend="reference", checkpoint=path)
    assert [a.tobytes() for a in resumed.weights.arrays()] == [a.tobytes() for a in whole.weights.arrays()]
    with pytest.raises(ValueError, match="bins"):  # other bins: another fit
        train.train_tabm(dataclasses.replace(params, num_emb_n_bins=5), data, rows=slice(0, 120),
                         backend="reference", checkpoint=path)  # fmt: skip


def test_bad_embedding_settings_raise() -> None:
    data = three_classes()
    for embedded in (0, 5):
        with pytest.raises(ValueError, match="embedded must be"):
            train.train_tabm(PWL, data, rows=slice(0, 120), backend="reference", embedded=embedded)
    with pytest.raises(ValueError, match=r"needs params\.num_emb_type"):
        train.train_tabm(dataclasses.replace(TINY, n_classes=3), data, rows=slice(0, 120), backend="reference",
                         embedded=2)  # fmt: skip
    constant = MatrixData(np.column_stack([np.ones(120), data.x[:, 1:]]).astype(np.float32), data.y, 3)
    with pytest.raises(ValueError, match="fewer than two distinct"):
        train.train_tabm(PWL, constant, rows=slice(0, 120), backend="reference")
