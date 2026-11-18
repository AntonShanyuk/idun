import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from idun.data import PackedDataset
from idun.errors import MultiTargetQualityWarning
from idun.nn import inputs
from idun.nn.inputs import MatrixData
from idun.nn.training import Validation
from idun.realmlp import metal as realmlp_metal
from idun.realmlp import reference as ref
from idun.realmlp import train as rt
from idun.realmlp.params import RealMLPParams
from idun_numerai.inputs import classes as numerai_classes

SMALL = RealMLPParams(
    n_classes=5, blocks=2, hidden=64, frequencies=4, embed=3, batch_size=32, epochs=3, init_rows=60, seed=2
)
TINY = RealMLPParams(n_classes=5, blocks=1, hidden=8, frequencies=2, embed=2, batch_size=64, epochs=1, init_rows=50)
OWN = dataclasses.replace(SMALL, own_last_block=True)
GPU = [pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]  # each, marked


def target_file(data: PackedDataset, path: Path, shift: float = 0.0) -> np.memmap:
    """Three targets of the cache's rows, (3, n_rows) float32, memory-mapped (page-aligned): y, 1 - y, y of row - 1."""
    y = np.asarray(data.target, np.float32)
    np.stack([y, 1 - y, np.roll(y, 1) + np.float32(shift)]).tofile(path)
    return np.memmap(path, np.float32, "r", shape=(3, data.n_rows))


def class_array_of(targets: np.ndarray) -> np.ndarray:
    """The classes rint(4 · y) of a (K, n_rows) target array as a compact (n_rows, K) uint8 array Metal can wrap."""
    return numerai_classes(targets)


def test_init_rows_are_a_seeded_sorted_draw() -> None:
    rows = rt.init_rows(SMALL, lo=1000, n=500, n_inputs=8)
    assert rows.size == 60
    assert np.all(np.diff(rows) > 0)
    assert rows.min() >= 1000
    assert rows.max() < 1500
    np.testing.assert_array_equal(rows, rt.init_rows(SMALL, 1000, 500, 8))
    assert rt.init_rows(SMALL, 0, 40, 8).size == 40  # fewer rows than init_rows: all of them


def test_init_rows_zero_takes_pytabkits_count_for_the_table() -> None:
    assert rt.pytabkit_init_rows(100) == 43_289  # floor(2^28 / 6,201)
    assert rt.pytabkit_init_rows(10**8) == 1
    pytabkit = dataclasses.replace(SMALL, init_rows=0)
    assert rt.init_rows(pytabkit, 0, 50_000, 100).size == 43_289
    assert rt.init_rows(pytabkit, 0, 5000, 100).size == 5000  # a smaller table: every training row
    np.testing.assert_array_equal(rt.init_rows(pytabkit, 0, 50_000, 100), rt.init_rows(
        dataclasses.replace(SMALL, init_rows=43_289), 0, 50_000, 100))  # fmt: skip


def test_the_initialisation_reads_the_preprocessed_rows_in_float64() -> None:
    """RealMLP's scales and biases are fitted on unrounded init rows; float32 rows would move every initial weight."""
    generator = np.random.default_rng(5)
    x = (3.0 * generator.normal(size=(300, 6)) + 1.0).astype(np.float32)
    data = MatrixData(x, generator.integers(0, 5, 300), 5)
    raw = rt.raw_inputs(data, rt.init_rows(TINY, 0, 300, 6))
    pre = ref.fit_preprocess(raw)
    want = ref.init(TINY, ref.preprocess(TINY, pre, raw), 1).astype(np.float32)
    rounded = ref.init(TINY, ref.preprocess(TINY, pre, raw).astype(np.float32), 1).astype(np.float32)
    got = rt._initial_model(TINY, data, (0, 300), None, ()).weights
    assert all(np.array_equal(a, b) for a, b in zip(got.arrays(), want.arrays(), strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(rounded.arrays(), want.arrays(), strict=True))


@pytest.mark.parametrize("gpu_backend", GPU)
def test_gpu_training_follows_the_reference_and_repeats_bit_for_bit(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    epochs: list[int] = []
    ref_fit = rt.train_realmlp(
        SMALL, data, rows=slice(0, 200), backend="reference", classes=numerai_classes(data.target)
    )
    gpu = [rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend=gpu_backend,
                            on_epoch=lambda e, *_: epochs.append(e), classes=numerai_classes(data.target))
           for _ in range(2)]  # fmt: skip
    assert epochs == [0, 1, 2, 0, 1, 2]
    np.testing.assert_allclose(gpu[0].report.epoch_loss, ref_fit.report.epoch_loss, rtol=1e-5)
    for got, want in zip(gpu[0].weights.arrays(), ref_fit.weights.arrays(), strict=True):
        assert np.abs(got - want).max() / max(np.abs(want).max(), 1e-30) < 1e-4  # observed 7e-7
    assert all(np.array_equal(a, b) for a, b in zip(gpu[0].weights.arrays(), gpu[1].weights.arrays(), strict=True))
    assert gpu[0].report.epoch_loss == gpu[1].report.epoch_loss
    pred = {name: fit.predict(data, rows=slice(200, 277), backend=name, chunk=64)
            for name, fit in (("reference", ref_fit), (gpu_backend, gpu[0]))}  # fmt: skip
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], atol=1e-4)


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_one_target_with_its_own_last_block_is_the_single_target_fit(nn_cache: Path, backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    labels = numerai_classes(data.target)  # one column: the single-target network
    fits = [
        rt.train_realmlp(params, data, rows=slice(0, 200), backend=backend, classes=labels) for params in (SMALL, OWN)
    ]
    assert fits[1].report.epoch_loss == fits[0].report.epoch_loss
    assert all(np.array_equal(a, b) for a, b in zip(fits[1].weights.arrays(), fits[0].weights.arrays(), strict=True))
    assert [fit.n_targets for fit in fits] == [None, 1]  # an own last block is a network of K targets, here 1
    pred = [fit.predict(data, rows=slice(200, 277), backend=backend, chunk=64) for fit in fits]
    assert (pred[0].shape, pred[1].shape) == ((77,), (77, 1))
    np.testing.assert_array_equal(pred[0], pred[1][:, 0])


@pytest.mark.parametrize("gpu_backend", GPU)
def test_k_target_training_follows_the_reference_and_repeats_bit_for_bit(
    nn_cache: Path, tmp_path: Path, gpu_backend: str
) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = target_file(data, tmp_path / "t.f32")
    ref_fit = rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend="reference", classes=numerai_classes(targets))
    gpu = [
        rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend=gpu_backend, classes=numerai_classes(targets))
        for _ in range(2)
    ]
    assert ref_fit.n_targets == gpu[0].n_targets == 3
    assert gpu[0].weights.out.shape == (64, 15)
    np.testing.assert_allclose(gpu[0].report.epoch_loss, ref_fit.report.epoch_loss, rtol=1e-5)
    for got, want in zip(gpu[0].weights.arrays(), ref_fit.weights.arrays(), strict=True):
        assert np.abs(got - want).max() / max(np.abs(want).max(), 1e-30) < 1e-4
    assert all(np.array_equal(a, b) for a, b in zip(gpu[0].weights.arrays(), gpu[1].weights.arrays(), strict=True))
    assert gpu[0].report.epoch_loss == gpu[1].report.epoch_loss
    pred = {name: fit.predict(data, rows=slice(200, 277), backend=name, chunk=64)
            for name, fit in (("reference", ref_fit), (gpu_backend, gpu[0]))}  # fmt: skip
    assert pred[gpu_backend].shape == pred["reference"].shape == (77, 3)
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], atol=1e-4)


@pytest.mark.parametrize("gpu_backend", GPU)
def test_own_last_block_training_follows_the_reference_and_repeats_bit_for_bit(
    nn_cache: Path, tmp_path: Path, gpu_backend: str
) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = target_file(data, tmp_path / "t.f32")
    ref_fit = rt.train_realmlp(OWN, data, rows=slice(0, 200), backend="reference", classes=numerai_classes(targets))
    gpu = [
        rt.train_realmlp(OWN, data, rows=slice(0, 200), backend=gpu_backend, classes=numerai_classes(targets))
        for _ in range(2)
    ]
    assert ref_fit.n_targets == gpu[0].n_targets == 3
    assert (gpu[0].weights.weight[-1].shape, gpu[0].weights.out.shape) == ((64, 192), (64, 15))
    np.testing.assert_allclose(gpu[0].report.epoch_loss, ref_fit.report.epoch_loss, rtol=1e-5)
    for got, want in zip(gpu[0].weights.arrays(), ref_fit.weights.arrays(), strict=True):
        assert np.abs(got - want).max() / max(np.abs(want).max(), 1e-30) < 1e-4
    assert all(np.array_equal(a, b) for a, b in zip(gpu[0].weights.arrays(), gpu[1].weights.arrays(), strict=True))
    assert gpu[0].report.epoch_loss == gpu[1].report.epoch_loss
    pred = {name: fit.predict(data, rows=slice(200, 277), backend=name)
            for name, fit in (("reference", ref_fit), (gpu_backend, gpu[0]))}  # fmt: skip
    assert pred[gpu_backend].shape == pred["reference"].shape == (77, 3)
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], atol=1e-4)
    np.testing.assert_array_equal(
        gpu[0].predict(data, rows=slice(200, 277), backend=gpu_backend, chunk=64), pred[gpu_backend]
    )


def test_an_own_last_block_model_round_trips_and_checks_its_shapes(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    own = dataclasses.replace(TINY, own_last_block=True)
    model = rt.train_realmlp(own, data, rows=slice(0, 128), backend="reference",
                             classes=numerai_classes(target_file(data, tmp_path / "t.f32")))  # fmt: skip
    model.save(tmp_path / "m.npz")
    back = rt.RealMLPModel.load(tmp_path / "m.npz")
    assert back.params == own
    assert (back.n_targets, back.weights.weight[-1].shape, back.weights.out.shape) == (3, (42, 24), (8, 15))
    np.testing.assert_array_equal(back.predict(data, rows=slice(130, 160), backend="reference"),
                                  model.predict(data, rows=slice(130, 160), backend="reference"))  # fmt: skip
    c = model.inputs(data, np.arange(50)).astype(np.float64)
    with pytest.raises(ValueError, match="needs a model of a"):  # one target, not as an array
        rt.RealMLPModel(own, model.preprocess, ref.init(TINY, c), model.n_inputs, (0, 128), model.report)
    shared = ref.init(TINY, c, 3)  # one last block for all three targets
    with pytest.raises(ValueError, match="is not 3 x 8"):
        rt.RealMLPModel(own, model.preprocess, shared, model.n_inputs, (0, 128), model.report, True)
    with pytest.raises(ValueError, match="this backend's has 3 x 8"):
        rt.ReferenceRealMLP(own, 3).set_weights(shared)
    assert [rt.predict_chunk(p, k) for p, k in ((TINY, 3), (own, None), (own, 3), (OWN, 100), (OWN, 5000))] == [
        4096, 4096, 4096, 2560, 64]  # fmt: skip


def test_a_k_target_model_round_trips_and_version_1_files_load(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    model = rt.train_realmlp(TINY, data, rows=slice(0, 128), backend="reference",
                             classes=numerai_classes(target_file(data, tmp_path / "t.f32")))  # fmt: skip
    model.save(tmp_path / "m.npz")
    back = rt.RealMLPModel.load(tmp_path / "m.npz")
    assert (back.multi_target, back.n_targets, back.weights.out.shape) == (True, 3, (8, 15))
    assert all(np.array_equal(a, b) for a, b in zip(back.weights.arrays(), model.weights.arrays(), strict=True))
    pred = back.predict(data, rows=slice(130, 160), backend="reference")
    assert pred.shape == (30, 3)
    np.testing.assert_array_equal(pred, model.predict(data, rows=slice(130, 160), backend="reference"))
    single = rt.train_realmlp(TINY, data, rows=slice(0, 128), backend="reference", classes=numerai_classes(data.target))
    single.save(tmp_path / "s.npz")
    with np.load(tmp_path / "s.npz") as archive:  # the file as format version 1 wrote it (no "multi_target")
        arrays = dict(archive)
    meta = json.loads(arrays["meta"].tobytes())
    del meta["multi_target"]
    arrays["meta"] = np.frombuffer(json.dumps({**meta, "format_version": 1}).encode(), np.uint8)
    np.savez(tmp_path / "v1.npz", **arrays)
    old = rt.RealMLPModel.load(tmp_path / "v1.npz")
    assert (old.multi_target, old.n_targets) == (False, None)
    assert old.predict(data, rows=slice(130, 160), backend="reference").shape == (30,)
    with pytest.raises(ValueError, match="multi_target is False"):
        rt.RealMLPModel(TINY, model.preprocess, model.weights, model.n_inputs, (0, 128), model.report)
    model.weights.out = model.weights.out[:, :14]
    with pytest.raises(ValueError, match="not K targets of 5 classes"):
        rt.RealMLPModel(TINY, model.preprocess, model.weights, model.n_inputs, (0, 128), model.report, True)
    with pytest.raises(ValueError, match="this backend has 1 targets"):
        rt.ReferenceRealMLP(TINY).set_weights(back.weights)


def test_a_model_round_trips_through_npz(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = RealMLPParams(
        n_classes=5, blocks=1, hidden=8, frequencies=2, embed=2, batch_size=64, epochs=1, init_rows=50
    )
    model = rt.train_realmlp(
        params, data, rows=slice(0, 128), backend="reference", classes=numerai_classes(data.target)
    )
    model.save(tmp_path / "m.npz")
    back = rt.RealMLPModel.load(tmp_path / "m.npz")
    assert back.params == params
    assert (back.n_inputs, back.train_rows, back.report) == (model.n_inputs, (0, 128), model.report)
    np.testing.assert_array_equal(back.preprocess.median, model.preprocess.median)
    np.testing.assert_array_equal(back.preprocess.factor, model.preprocess.factor)
    assert all(np.array_equal(a, b) for a, b in zip(back.weights.arrays(), model.weights.arrays(), strict=True))
    np.testing.assert_array_equal(back.predict(data, rows=slice(130, 160), backend="reference"),
                                  model.predict(data, rows=slice(130, 160), backend="reference"))  # fmt: skip
    np.savez(tmp_path / "bad.npz", meta=np.frombuffer(b'{"format": "x"}', np.uint8))
    with pytest.raises(ValueError, match="not a idun-realmlp"):
        rt.RealMLPModel.load(tmp_path / "bad.npz")


def test_bad_arguments_raise(nn_cache: Path, tmp_path: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    labels = numerai_classes(data.target)
    bads: list[np.ndarray] = [labels[:-1], labels.astype(np.int64), labels[:, 0]]  # a row short, int64, 1-D
    for bad in bads:
        with pytest.raises(ValueError, match=r"classes must be an \(277, K\) uint8"):
            rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend="reference", classes=bad)
    high = labels.copy()
    high[150, 0] = 5  # a training row's class at n_classes
    with pytest.raises(ValueError, match=r"must be 0\.\.4"):
        rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend="reference", classes=high)
    with pytest.raises(ValueError, match="needs its classes"):
        rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend="reference")
    matrix = MatrixData(np.zeros((200, 3), np.float32), np.zeros(200, np.int64), 5)
    with pytest.raises(ValueError, match="carries its own labels"):
        rt.train_realmlp(SMALL, matrix, rows=slice(0, 200), backend="reference", classes=labels)
    with pytest.raises(ValueError, match="unknown backend"):
        rt.make_backend("rocm", SMALL, 5, 1)
    with pytest.raises(ValueError, match="own_last_block needs a network of K targets"):
        rt.ReferenceRealMLP(OWN)
    with pytest.raises(ValueError, match="at least batch_size"):
        rt.train_realmlp(SMALL, data, rows=slice(0, 20), backend="reference", classes=numerai_classes(data.target))
    with pytest.raises(ValueError, match="at least batch_size"):
        rt.train_realmlp(SMALL, data, rows=slice(0, 200, 2), backend="reference", classes=numerai_classes(data.target))
    params = RealMLPParams(
        n_classes=5, blocks=1, hidden=8, frequencies=2, embed=2, batch_size=64, epochs=1, init_rows=50
    )
    model = rt.train_realmlp(params, data, rows=slice(0, 64), backend="reference", classes=numerai_classes(data.target))
    with pytest.raises(ValueError, match="contiguous slice"):
        model.predict(data, rows=slice(0, 10, 2), backend="reference")
    model.n_inputs += 1
    with pytest.raises(ValueError, match="the model expects"):
        model.predict(data, rows=slice(0, 10), backend="reference")


def test_the_reference_backend_needs_weights_first() -> None:
    backend = rt.ReferenceRealMLP(SMALL)
    with pytest.raises(RuntimeError, match="set_weights first"):
        backend.weights()


@pytest.mark.parametrize("n_targets", [1, 3])
@pytest.mark.parametrize("backend", ["reference", *GPU])
@pytest.mark.parametrize("params", [SMALL, OWN], ids=["shared", "own_last_block"])
def test_classes_train_as_the_targets_do(
    nn_cache: Path, tmp_path: Path, backend: str, n_targets: int, params: RealMLPParams
) -> None:
    """The port's classes and a class array missing labels outside the training rows: the same fit, bit for bit."""
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = target_file(data, tmp_path / "t.f32")[:n_targets]
    classes = class_array_of(targets)
    classes[200:, -1] = inputs.MISSING_CLASS  # rows outside the training rows may lack a target
    fits = [rt.train_realmlp(params, data, rows=slice(0, 200), backend=backend, classes=numerai_classes(targets)),
            rt.train_realmlp(params, data, rows=slice(0, 200), backend=backend, classes=classes)]  # fmt: skip
    assert fits[0].report.epoch_loss == fits[1].report.epoch_loss
    assert all(np.array_equal(a, b) for a, b in zip(fits[0].weights.arrays(), fits[1].weights.arrays(), strict=True))
    k = None if n_targets == 1 and not params.own_last_block else n_targets  # one shared target: single-target
    assert (fits[1].multi_target, fits[1].n_targets) == (k is not None, k)
    pred = [fit.predict(data, rows=slice(200, 277), backend=backend, chunk=64) for fit in fits]
    assert pred[1].shape == ((77,) if k is None else (77, n_targets))
    np.testing.assert_array_equal(pred[0], pred[1])


def test_bad_classes_raise(nn_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = target_file(data, tmp_path / "t.f32")
    classes = class_array_of(targets)
    bads: list[np.ndarray] = [
        classes[:-1],
        classes[:, 0],
        classes.astype(np.int16),
        np.zeros((data.n_rows, 0), np.uint8),
    ]
    for bad in bads:
        with pytest.raises(ValueError, match=r"classes must be an \(277, K\) uint8 array"):
            rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend="reference", classes=bad)
    monkeypatch.setattr(rt, "CHECK_ROWS", 64)  # the bad row lies in the third chunk of training rows
    for value in (inputs.MISSING_CLASS, 5):
        broken = class_array_of(targets)
        broken[150, 2] = value
        with pytest.raises(ValueError, match=r"classes of the training rows must be 0\.\.4"):
            rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend="reference", classes=broken)


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_snapshots_hand_over_weights_without_changing_the_training(nn_cache: Path, tmp_path: Path, backend: str
                                                                   ) -> None:  # fmt: skip
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = target_file(data, tmp_path / "t.f32")
    plain = rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend=backend, classes=numerai_classes(targets))
    seen: list[tuple[float, ref.Weights]] = []
    snapped = rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend=backend, classes=numerai_classes(targets),
                               snapshots=(0.2, 0.5, 1.0), on_snapshot=lambda f, w: seen.append((f, w)))  # fmt: skip
    assert [f for f, _ in seen] == [0.2, 0.5, 1.0]  # 18 steps: after 4 (mid-epoch), 9 (mid-epoch) and 18
    assert snapped.report.epoch_loss == plain.report.epoch_loss
    for weights in (snapped.weights, seen[-1][1]):
        assert all(np.array_equal(a, b) for a, b in zip(weights.arrays(), plain.weights.arrays(), strict=True))
    assert not np.array_equal(seen[0][1].weight[0], plain.weights.weight[0])


@pytest.mark.parametrize("gpu_backend", GPU)
def test_gpu_snapshots_follow_the_reference(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    seen: dict[str, list[ref.Weights]] = {"reference": [], gpu_backend: []}
    for name, kept in seen.items():

        def keep(_: float, weights: ref.Weights, kept: list[ref.Weights] = kept) -> None:
            kept.append(weights)

        rt.train_realmlp(
            SMALL,
            data,
            rows=slice(0, 200),
            backend=name,
            snapshots=(0.25, 0.75),
            on_snapshot=keep,
            classes=numerai_classes(data.target),
        )
    for got, want in zip(seen[gpu_backend], seen["reference"], strict=True):
        for a, b in zip(got.arrays(), want.arrays(), strict=True):
            assert np.abs(a - b).max() / max(np.abs(b).max(), 1e-30) < 1e-4


def test_snapshot_steps_are_checked() -> None:
    def keep(_: float, __: ref.Weights) -> None:
        pass

    assert rt.snapshot_steps(rt.SCHEDULE_ZEROS, 150, keep) == {30: 0.2, 70: 7 / 15, 150: 1.0}
    assert rt.snapshot_steps((), 10, None) == {}
    with pytest.raises(ValueError, match="need on_snapshot"):
        rt.snapshot_steps((1.0,), 10, None)
    for bad in ((0.0,), (1.2,), (0.5, 0.5), (0.6, 0.4)):
        with pytest.raises(ValueError, match="increasing training fractions"):
            rt.snapshot_steps(bad, 10, keep)
    for bad in ((0.50, 0.52), (0.01, 1.0)):  # one step for both; step 0 (before any training)
        with pytest.raises(ValueError, match="distinct steps"):
            rt.snapshot_steps(bad, 10, keep)
    for f in rt.SCHEDULE_ZEROS:  # the learning rate is 0 there
        assert ref.schedules(f)[0] == pytest.approx(0.0, abs=1e-12)


def test_more_than_one_target_warns_about_quality_and_one_does_not(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    two = np.stack([np.asarray(data.target), 1.0 - np.asarray(data.target)]).astype(np.float32)
    with pytest.warns(MultiTargetQualityWarning, match="2 targets on one network"):
        rt.train_realmlp(TINY, data, rows=slice(0, 128), backend="reference", classes=numerai_classes(two))
    with warnings.catch_warnings():
        warnings.simplefilter("error", MultiTargetQualityWarning)
        rt.train_realmlp(
            TINY, data, rows=slice(0, 128), backend="reference", classes=numerai_classes(data.target[None, :])
        )


def same_fit(a: rt.RealMLPModel, b: rt.RealMLPModel) -> bool:
    return a.report.epoch_loss == b.report.epoch_loss and all(
        np.array_equal(x, y) for x, y in zip(a.weights.arrays(), b.weights.arrays(), strict=True)
    )


@pytest.mark.parametrize(("backend", "workers"), [("reference", 1), pytest.param("metal", 2, marks=pytest.mark.metal),
                                                  pytest.param("cuda", 1, marks=pytest.mark.cuda)])  # fmt: skip
def test_a_seed_bag_is_the_separate_fits_in_seed_order(nn_cache: Path, backend: str, workers: int) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    seeds, params = [4, 1, 7], dataclasses.replace(SMALL, epochs=1)  # Metal needs hidden in multiples of 64
    bag = rt.train_realmlp_seeds(
        params,
        data,
        rows=slice(0, 128),
        backend=backend,
        seeds=seeds,
        workers=workers,
        classes=numerai_classes(data.target),
    )
    separate = [rt.train_realmlp(dataclasses.replace(params, seed=s), data, rows=slice(0, 128), backend=backend,
                                 classes=numerai_classes(data.target)) for s in seeds]  # fmt: skip
    assert [model.params.seed for model in bag] == seeds
    assert all(same_fit(a, b) for a, b in zip(bag, separate, strict=True))
    assert not same_fit(bag[0], bag[1])
    want = np.mean([m.predict(data, rows=slice(128, 200), backend=backend) for m in bag], axis=0, dtype=np.float64)
    got = rt.predict_mean(bag, data, rows=slice(128, 200), backend=backend)
    np.testing.assert_array_equal(got, want.astype(np.float32))


@pytest.mark.parametrize(
    ("seeds", "workers", "backend", "message"),
    [([], 1, "reference", "distinct"), ([2, 2], 1, "reference", "distinct"), ([1, 2], 2, "reference", "workers"),
     ([1, 2], 0, "metal", "workers")],
)  # fmt: skip
def test_bad_seed_bags_fail_loudly(nn_cache: Path, seeds: list[int], workers: int, backend: str, message: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    with pytest.raises(ValueError, match=message):
        rt.train_realmlp_seeds(
            TINY,
            data,
            rows=slice(0, 128),
            backend=backend,
            seeds=seeds,
            workers=workers,
            classes=numerai_classes(data.target),
        )
    with pytest.raises(ValueError, match="at least one model"):
        rt.predict_mean([], data, rows=slice(0, 10), backend="reference")


def test_shared_codes_are_for_the_metal_backend_only(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    codes = realmlp_metal.shared_codes(data)
    with pytest.raises(ValueError, match="does not read them"):
        rt.train_realmlp(
            TINY, data, rows=slice(0, 128), backend="reference", codes=codes, classes=numerai_classes(data.target)
        )


def test_seed_bags_never_exceed_the_gpus_room(nn_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from idun.metal import runtime  # noqa: PLC0415

    data = PackedDataset.open(nn_cache, binning="r1")
    assert rt.network_bytes(data, 0) > data.n_rows * realmlp_metal.codes_stride(len(data.obfuscated_features))
    monkeypatch.setattr(runtime, "side_by_side", lambda bytes_each: 1)
    with pytest.raises(ValueError, match="pass at most 1"):
        rt.train_realmlp_seeds(
            SMALL,
            data,
            rows=slice(0, 128),
            backend="metal",
            seeds=[1, 2],
            workers=2,
            classes=numerai_classes(data.target),
        )


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_regression_on_a_matrix_learns_the_targets(backend: str, tmp_path: Path) -> None:
    generator = np.random.default_rng(3)
    x = generator.normal(size=(400, 4)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1] + generator.normal(0, 0.1, 400)).astype(np.float32)
    params = dataclasses.replace(SMALL, n_classes=1, objective="mse", epochs=6, label_smoothing=0.0)
    data = MatrixData(x, y, 1)
    model = rt.train_realmlp(params, data, rows=slice(0, 320), backend=backend)
    pred = model.predict(data, rows=slice(320, 400), backend=backend)
    assert pred.shape == (80,)
    assert np.corrcoef(pred, y[320:])[0, 1] > 0.8
    model.save(tmp_path / "reg.npz")
    np.testing.assert_array_equal(
        rt.RealMLPModel.load(tmp_path / "reg.npz").predict(data, rows=slice(320, 400), backend=backend), pred
    )
    with pytest.raises(ValueError, match="classification"):
        model.predict_proba(data, rows=slice(0, 10), backend=backend)


# -- regression on a packed dataset -----------------------------------------------------------------------------------

REG = dataclasses.replace(SMALL, n_classes=1, objective="mse", label_smoothing=0.0)


def packed_targets(data: PackedDataset, n_targets: int) -> np.ndarray:
    """K continuous targets of the cache's rows as a page-aligned (n_rows, K) float32 target array."""
    out = inputs.target_array(data.n_rows, n_targets)
    out[...] = np.random.default_rng(5).normal(size=out.shape).astype(np.float32)
    return out


def test_regression_on_a_packed_dataset_is_the_matrix_fit_on_the_reference(nn_cache: Path) -> None:
    """The reference backend builds both batches on the host: the packed fit is the matrix fit, bit for bit."""
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, 1)
    targets[200:, 0] = np.nan  # rows outside the training rows may lack a target
    packed = rt.train_realmlp(REG, data, rows=slice(0, 200), backend="reference", targets=targets)
    matrix = MatrixData(inputs.dense_inputs(data, np.arange(data.n_rows), raw_codes=True),
                        np.nan_to_num(targets[:, 0]), 1)  # fmt: skip
    dense = rt.train_realmlp(REG, matrix, rows=slice(0, 200), backend="reference")
    assert packed.report.epoch_loss == dense.report.epoch_loss
    assert all(np.array_equal(a, b) for a, b in zip(packed.weights.arrays(), dense.weights.arrays(), strict=True))
    np.testing.assert_array_equal(packed.predict(data, rows=slice(200, 277), backend="reference"),
                                  dense.predict(matrix, rows=slice(200, 277), backend="reference"))  # fmt: skip


@pytest.mark.parametrize("gpu_backend", GPU)
@pytest.mark.parametrize("n_targets", [1, 3])
def test_gpu_regression_on_a_packed_dataset_follows_the_reference(
    nn_cache: Path, n_targets: int, gpu_backend: str
) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, n_targets)
    targets[200:, -1] = np.nan
    ref_fit = rt.train_realmlp(REG, data, rows=slice(0, 200), backend="reference", targets=targets)
    gpu = [rt.train_realmlp(REG, data, rows=slice(0, 200), backend=gpu_backend, targets=targets) for _ in range(2)]
    np.testing.assert_allclose(gpu[0].report.epoch_loss, ref_fit.report.epoch_loss, rtol=1e-5)
    for got, want in zip(gpu[0].weights.arrays(), ref_fit.weights.arrays(), strict=True):
        assert np.abs(got - want).max() / max(np.abs(want).max(), 1e-30) < 1e-4
    assert all(np.array_equal(a, b) for a, b in zip(gpu[0].weights.arrays(), gpu[1].weights.arrays(), strict=True))
    assert gpu[0].report.epoch_loss == gpu[1].report.epoch_loss
    pred = {name: fit.predict(data, rows=slice(200, 277), backend=name, chunk=64)
            for name, fit in (("reference", ref_fit), (gpu_backend, gpu[0]))}  # fmt: skip
    assert pred[gpu_backend].shape == ((77,) if n_targets == 1 else (77, n_targets))
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], atol=1e-4)


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_a_regression_seed_bag_is_the_separate_fits(nn_cache: Path, backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, 1)
    params = dataclasses.replace(REG, epochs=1)
    bag = rt.train_realmlp_seeds(params, data, rows=slice(0, 200), backend=backend, seeds=[3, 1], targets=targets)
    for seed, model in zip([3, 1], bag, strict=True):
        alone = rt.train_realmlp(dataclasses.replace(params, seed=seed), data, rows=slice(0, 200), backend=backend,
                                 targets=targets)  # fmt: skip
        assert all(np.array_equal(a, b) for a, b in zip(model.weights.arrays(), alone.weights.arrays(), strict=True))


def test_bad_targets_raise(nn_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    targets = packed_targets(data, 2)
    bads: list[np.ndarray] = [targets[:-1], targets[:, 0], targets.astype(np.float64)]
    for bad in bads:
        with pytest.raises(ValueError, match=r"targets must be an \(277, K\) float32"):
            rt.train_realmlp(REG, data, rows=slice(0, 200), backend="reference", targets=bad)
    monkeypatch.setattr(inputs, "CHECK_ROWS", 64)  # the bad row lies in the third chunk of training rows
    for value in (np.nan, np.inf):
        broken = packed_targets(data, 2)
        broken[150, 1] = value
        with pytest.raises(ValueError, match="every target of the training rows must be finite"):
            rt.train_realmlp(REG, data, rows=slice(0, 200), backend="reference", targets=broken)
    with pytest.raises(ValueError, match="needs its targets for objective 'mse'"):
        rt.train_realmlp(REG, data, rows=slice(0, 200), backend="reference", classes=numerai_classes(data.target))
    with pytest.raises(ValueError, match="needs its classes for objective 'cross_entropy'"):
        rt.train_realmlp(SMALL, data, rows=slice(0, 200), backend="reference", targets=targets)
    matrix = MatrixData(np.zeros((200, 3), np.float32), np.zeros(200, np.float32), 1)
    with pytest.raises(ValueError, match="carries its own labels"):
        rt.train_realmlp(REG, matrix, rows=slice(0, 200), backend="reference", targets=targets)


@pytest.mark.parametrize("gpu_backend", GPU)
def test_other_schedules_on_the_gpu_follow_the_reference(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    params = dataclasses.replace(SMALL, lr_schedule="cosine", dropout_schedule="constant", wd_schedule="constant")
    labels = numerai_classes(data.target)
    fits = {name: rt.train_realmlp(params, data, rows=slice(10, 170), backend=name, classes=labels)
            for name in ("reference", gpu_backend)}  # fmt: skip
    for got, want in zip(fits[gpu_backend].weights.arrays(), fits["reference"].weights.arrays(), strict=True):
        assert np.abs(got - want).max() / max(np.abs(want).max(), 1e-12) < 1e-4
    plain = rt.train_realmlp(SMALL, data, rows=slice(10, 170), backend="reference",
                                classes=numerai_classes(data.target))  # fmt: skip
    assert not np.array_equal(plain.weights.out, fits["reference"].weights.out)


def matrix(n: int, seed: int, n_classes: int) -> MatrixData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    signal = x[:, 0] - 0.5 * x[:, 2]
    if n_classes == 1:
        return MatrixData(x, (signal + rng.normal(0, 0.3, n)).astype(np.float32), 1)
    return MatrixData(x, np.digitize(signal, [-0.4, 0.4]).astype(np.int64), n_classes)


@pytest.mark.parametrize("n_classes", [3, 1])
def test_a_validated_fit_keeps_its_best_epochs_weights(n_classes: int) -> None:
    epochs = 5
    params = dataclasses.replace(SMALL, epochs=epochs, n_classes=n_classes,
                                 objective="cross_entropy" if n_classes > 1 else "mse")  # fmt: skip
    data, valid = matrix(192, 1, n_classes), Validation(matrix(96, 2, n_classes), slice(0, 96))
    seen: dict[float, ref.Weights] = {}
    ends = [(e + 1) / epochs for e in range(epochs)]
    model = rt.train_realmlp(params, data, rows=slice(0, 192), backend="reference", valid=valid, snapshots=ends,
                             on_snapshot=seen.__setitem__)  # fmt: skip
    report = model.report
    assert len(report.valid_score) == epochs
    assert report.best_epoch == int(np.argmin(report.valid_score))
    kept = seen[(report.best_epoch + 1) / epochs]
    for got, want in zip(model.weights.arrays(), kept.arrays(), strict=True):
        np.testing.assert_array_equal(got, want)


def test_patience_stops_a_realmlp_after_that_many_epochs_without_a_better_one() -> None:
    params = dataclasses.replace(SMALL, epochs=30, n_classes=3, lr_schedule="constant")
    model = rt.train_realmlp(params, matrix(192, 3, 3), rows=slice(0, 192), backend="reference",
                             valid=Validation(matrix(96, 4, 3), slice(0, 96)), patience=2)  # fmt: skip
    report = model.report
    assert report.best_epoch is not None
    assert len(report.valid_score) == len(report.epoch_loss) == min(30, report.best_epoch + 3)
    with pytest.raises(ValueError, match="patience needs a validation set"):
        rt.train_realmlp(params, matrix(64, 5, 3), rows=slice(0, 64), backend="reference", patience=2)
    with pytest.raises(ValueError, match="has 1 classes, the network 3"):
        rt.train_realmlp(params, matrix(64, 5, 3), rows=slice(0, 64), backend="reference",
                         valid=Validation(matrix(32, 6, 1), slice(0, 32)))  # fmt: skip


# -- sample weights ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("gpu_backend", GPU)
def test_weighted_gpu_training_follows_the_reference_and_unit_weights_change_nothing(
    nn_cache: Path, gpu_backend: str
) -> None:
    data, rows = PackedDataset.open(nn_cache, binning="r1"), slice(0, 200)
    classes = numerai_classes(data.target)
    weight = np.random.default_rng(6).choice(np.array([0.0, 0.5, 1.0, 3.0], np.float32), 200)
    fits = {name: rt.train_realmlp(SMALL, data, rows=rows, backend=name, classes=classes, weight=weight)
            for name in ("reference", gpu_backend)}  # fmt: skip
    np.testing.assert_allclose(fits[gpu_backend].report.epoch_loss, fits["reference"].report.epoch_loss, rtol=1e-5)
    for got, want in zip(fits[gpu_backend].weights.arrays(), fits["reference"].weights.arrays(), strict=True):
        assert np.abs(got - want).max() / max(np.abs(want).max(), 1e-30) < 1e-4
    plain = rt.train_realmlp(SMALL, data, rows=rows, backend=gpu_backend, classes=classes)
    ones = rt.train_realmlp(SMALL, data, rows=rows, backend=gpu_backend, classes=classes,
                            weight=np.ones(200, np.float32))  # fmt: skip
    assert same_fit(ones, plain)  # GPU batches: unit weights are the unweighted fit
    assert not same_fit(fits[gpu_backend], plain)


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_weighted_host_batches_follow_the_reference(backend: str) -> None:
    generator = np.random.default_rng(3)
    x = generator.normal(size=(400, 4)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1] + generator.normal(0, 0.1, 400)).astype(np.float32)
    params = dataclasses.replace(SMALL, n_classes=1, objective="mse", label_smoothing=0.0)
    data, rows = MatrixData(x, y, 1), slice(0, 320)
    weight = generator.choice(np.array([0.0, 0.5, 1.0, 3.0], np.float32), 320)
    fit = rt.train_realmlp(params, data, rows=rows, backend=backend, weight=weight)
    want = rt.train_realmlp(params, data, rows=rows, backend="reference", weight=weight)
    for got, expected in zip(fit.weights.arrays(), want.weights.arrays(), strict=True):
        assert np.abs(got - expected).max() / max(np.abs(expected).max(), 1e-30) < 1e-4
    plain = rt.train_realmlp(params, data, rows=rows, backend=backend)
    ones = rt.train_realmlp(params, data, rows=rows, backend=backend, weight=np.ones(320, np.float32))
    if backend == "reference":  # float64 sums in another order: close
        np.testing.assert_allclose(ones.report.epoch_loss, plain.report.epoch_loss, rtol=1e-12)
    else:
        assert same_fit(ones, plain)
    for bad, message in (
        (np.ones(319), "one weight per training row"),
        (np.full(320, np.inf), "finite"),
        (np.full(320, -1.0), "non-negative"),
        (np.zeros(320), "every sample weight is zero"),
    ):
        with pytest.raises(ValueError, match=message):
            rt.train_realmlp(params, data, rows=rows, backend=backend, weight=bad)


# -- categorical embeddings -------------------------------------------------------------------------------------------


def categorical_matrix(n: int = 300, seed: int = 4) -> MatrixData:
    generator = np.random.default_rng(seed)
    numeric, codes = generator.normal(size=(n, 3)), generator.integers(0, 12, n)
    y = (numeric[:, 0] + (codes % 3 == 0) > 0.5).astype(np.int64) + (codes > 8)
    return MatrixData(np.column_stack([numeric, codes]).astype(np.float32), y, 3)


@pytest.mark.parametrize("backend", ["reference", *GPU])
def test_categorical_inputs_train_and_round_trip(backend: str, tmp_path: Path) -> None:
    data, params = categorical_matrix(), dataclasses.replace(SMALL, n_classes=3, embedding_size=4)
    model = rt.train_realmlp(params, data, rows=slice(0, 300), backend=backend, categories=(12,))
    assert model.weights.table is not None
    assert model.weights.table.shape == (12, 4)
    assert len(model.preprocess.median) == 3  # the codes are not preprocessed
    want = rt.train_realmlp(params, data, rows=slice(0, 300), backend="reference", categories=(12,))
    for got, expected in zip(model.weights.arrays(), want.weights.arrays(), strict=True):
        assert np.abs(got - expected).max() / max(np.abs(expected).max(), 1e-30) < 1e-4
    model.save(tmp_path / "m.npz")
    with np.load(tmp_path / "m.npz") as archive:
        assert json.loads(archive["meta"].tobytes())["format_version"] == 3
    loaded = rt.RealMLPModel.load(tmp_path / "m.npz")
    assert loaded.weights.categories == (12,)
    np.testing.assert_array_equal(loaded.predict(data, rows=slice(0, 300), backend=backend),
                                  model.predict(data, rows=slice(0, 300), backend=backend))  # fmt: skip


def test_bad_categories_raise(nn_cache: Path) -> None:
    data, params = categorical_matrix(), dataclasses.replace(SMALL, n_classes=3)
    for bad, message in (((11,), "level codes"), ((0,), "positive level counts"), ((12, 2, 3, 4), "positive level")):
        with pytest.raises(ValueError, match=message):
            rt.train_realmlp(params, data, rows=slice(0, 300), backend="reference", categories=bad)
    packed = PackedDataset.open(nn_cache, binning="r1")
    with pytest.raises(ValueError, match="packed dataset has none"):
        rt.train_realmlp(SMALL, packed, rows=slice(0, 200), backend="reference",
                         classes=numerai_classes(packed.target), categories=(3,))  # fmt: skip
