"""`train_tabm` with a loss and a batch plan of the caller's: the built-in fits again, per-group losses, checkpoints."""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from idun.data import PackedDataset
from idun.nn.inputs import MatrixData
from idun.nn.training import epoch_order
from idun.tabm import BatchPlan, Loss, TabMParams, reference, train
from tests.unit.tabm.test_train import Stop, stop_after
from tests.unit.tabm.test_train_sequences import sequences

TINY = TabMParams(members=2, blocks=2, width=16, dropout=0.1, batch_size=32, epochs=2, seed=3, learning_rate=1e-2,
                  n_classes=1, objective="mse")  # fmt: skip
GPU = [pytest.param("metal", marks=pytest.mark.metal), pytest.param("cuda", marks=pytest.mark.cuda)]  # each, marked
BACKENDS = ["reference", *GPU]
GROUP_SIZES = (14, 9, 16, 11, 15, 8, 13, 10, 12, 12)  # 120 rows in ten groups
ROWS = slice(10, 130)


def panel(seed: int = 6) -> tuple[MatrixData, NDArray[np.int64]]:
    """Return 140 rows of a linear regression problem and the rows' groups: ten contiguous ones over `ROWS`."""
    generator = np.random.default_rng(seed)
    x = generator.normal(size=(140, 5)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1] + generator.normal(0, 0.3, 140)).astype(np.float32)
    groups = np.full(140, -1, np.int64)
    groups[ROWS] = np.repeat(np.arange(len(GROUP_SIZES)), GROUP_SIZES)
    return MatrixData(x, y, 1), groups


def squared_error(y: NDArray[Any]) -> Loss:
    """The built-in MSE as a loss: the mean of ``(o − y)²`` over the member outputs, in the reference's operations."""

    def loss(outputs: NDArray[np.floating], rows: NDArray[np.int64]) -> tuple[float, NDArray[np.floating]]:
        diff = outputs - y[rows].astype(outputs.dtype)[:, None, None]
        return float((diff.astype(np.float64) ** 2).mean()), 2.0 * diff / diff.size

    return loss


def cross_entropy(labels: NDArray[np.int64]) -> Loss:
    """The built-in cross-entropy as a loss."""

    def loss(outputs: NDArray[np.floating], rows: NDArray[np.int64]) -> tuple[float, NDArray[np.floating]]:
        return reference.loss_and_logit_grad(outputs, labels[rows])

    return loss


def group_pearson(y: NDArray[Any], groups: NDArray[np.int64], ridge: float = 0.05) -> Loss:
    """Minus the mean, over the batch's groups and the members, of the correlation of a member's outputs with ``y``.

    The correlation is taken within each group, so every group in a batch needs two rows or more that differ. It
    ignores the outputs' level and scale; ``ridge`` times their mean square pins both (a direction the loss ignores
    gets a gradient of rounding size, which AdamW normalises into a drift: the head's bias then differs between
    float32 and float64).
    """

    def loss(outputs: NDArray[np.floating], rows: NDArray[np.int64]) -> tuple[float, NDArray[np.floating]]:
        out, ids = outputs[:, :, 0].astype(np.float64), groups[rows]
        gradient, total, present = np.zeros_like(out), 0.0, np.unique(ids)
        for group in present:
            at = ids == group
            p = out[at] - out[at].mean(axis=0)
            t = y[rows[at]].astype(np.float64)
            t -= t.mean()
            norm_p, norm_t = np.sqrt((p**2).sum(axis=0)), np.sqrt((t**2).sum())
            r = (p * t[:, None]).sum(axis=0) / (norm_p * norm_t)
            total += float(r.sum())
            gradient[at] = t[:, None] / (norm_p * norm_t) - r * p / norm_p**2
        scale = -1.0 / (present.size * out.shape[1])
        value = scale * total + ridge * float((out**2).mean())
        return value, (scale * gradient + ridge * 2.0 * out / out.size)[:, :, None].astype(outputs.dtype)

    return loss


def seeded_shuffle(params: TabMParams, rows: slice) -> BatchPlan:
    """The default plan as a batch plan: the seeded shuffle of ``rows`` in batches of ``params.batch_size``."""

    def plan(epoch: int) -> list[NDArray[np.int64]]:
        order = rows.start + epoch_order(params.seed, epoch, rows.stop - rows.start)
        return [order[i : i + params.batch_size] for i in range(0, order.size, params.batch_size)]

    return plan


def whole_groups(groups: NDArray[np.int64], seed: int = 1) -> BatchPlan:
    """Batches of two whole groups each, the groups paired in an order drawn from ``(seed, epoch)``."""
    names = np.unique(groups[groups >= 0])

    def plan(epoch: int) -> list[NDArray[np.int64]]:
        order = np.random.default_rng([seed, epoch]).permutation(names)
        return [np.flatnonzero(np.isin(groups, order[i : i + 2])) for i in range(0, order.size, 2)]

    return plan


def bits(model: train.TabMModel) -> list[bytes]:
    return [a.tobytes() for a in model.weights.arrays()]


def close(got: train.TabMModel, want: train.TabMModel, weights: float = 1e-4, losses: float = 1e-5) -> None:
    """Assert two fits agree: the epoch losses within ``losses`` (absolute plus relative: a correlation's are near 0),
    every weight array within ``weights`` of its largest entry. The defaults are Metal against the float64 reference.
    """
    np.testing.assert_allclose(got.report.epoch_loss, want.report.epoch_loss, rtol=losses, atol=losses)
    for a, b in zip(got.weights.arrays(), want.weights.arrays(), strict=True):
        assert np.abs(a - b).max() / np.abs(b).max() < weights


# -- loss -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("micro", [0, 12])
def test_a_hand_written_mse_is_the_built_in_fit(backend: str, micro: int) -> None:
    data, _ = panel()
    params = dataclasses.replace(TINY, micro_batch=micro)
    built_in = train.train_tabm(params, data, rows=ROWS, backend=backend)
    hooked = train.train_tabm(params, data, rows=ROWS, backend=backend, loss=squared_error(data.y))
    if backend == "reference":
        assert bits(hooked) == bits(built_in)
        assert hooked.report.epoch_loss == built_in.report.epoch_loss
    else:  # the kernel multiplies by 1 / (B k) where the loss divides
        close(hooked, built_in, 1e-6, 1e-6)
    np.testing.assert_allclose(hooked.predict(data, rows=slice(0, 140), backend=backend),
                               built_in.predict(data, rows=slice(0, 140), backend=backend), atol=1e-5)  # fmt: skip


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_hand_written_cross_entropy_is_the_built_in_fit(backend: str) -> None:
    x = np.random.default_rng(4).normal(size=(120, 4)).astype(np.float32)
    labels = (x[:, 0] > 0).astype(np.int64) + (x[:, 1] > 1)
    data, params = MatrixData(x, labels, 3), dataclasses.replace(TINY, n_classes=3, objective="cross_entropy")
    built_in = train.train_tabm(params, data, rows=slice(0, 120), backend=backend)
    hooked = train.train_tabm(params, data, rows=slice(0, 120), backend=backend, loss=cross_entropy(labels))
    if backend == "reference":
        assert bits(hooked) == bits(built_in)
    else:
        close(hooked, built_in, 1e-6, 1e-6)
    assert hooked.predict_proba(data, rows=slice(0, 10), backend=backend).shape == (10, 3)


def test_the_group_loss_returns_its_gradient() -> None:
    data, groups = panel()
    loss = group_pearson(data.y, groups)
    rows = np.flatnonzero(np.isin(groups, [2, 5]))
    outputs = np.random.default_rng(9).normal(size=(rows.size, 2, 1))
    value, gradient = loss(outputs, rows)
    first = groups[rows] == 2
    expected = -np.mean([np.corrcoef(outputs[at, j, 0], data.y[rows[at]])[0, 1] for at in (first, ~first)
                         for j in range(2)])  # fmt: skip
    assert value == pytest.approx(expected + 0.05 * (outputs**2).mean())
    assert group_pearson(data.y, groups, ridge=0.0)(outputs, rows)[0] == pytest.approx(expected)
    for index in [(0, 0, 0), (7, 1, 0), (rows.size - 1, 0, 0)]:
        step = np.zeros_like(outputs)
        step[index] = 1e-6
        numeric = (loss(outputs + step, rows)[0] - loss(outputs - step, rows)[0]) / 2e-6
        assert gradient[index] == pytest.approx(numeric, rel=1e-5, abs=1e-9)


@pytest.mark.parametrize("gpu_backend", GPU)
@pytest.mark.parametrize("micro", [0, 12])
def test_gpu_follows_the_reference_under_a_group_loss(micro: int, gpu_backend: str) -> None:
    data, groups = panel()
    params = dataclasses.replace(TINY, micro_batch=micro, epochs=6)
    seen: dict[str, list[bytes]] = {"reference": [], gpu_backend: []}

    def watched(name: str) -> Loss:
        inner = group_pearson(data.y, groups)

        def loss(outputs: NDArray[np.floating], rows: NDArray[np.int64]) -> tuple[float, NDArray[np.floating]]:
            assert outputs.shape == (rows.size, 2, 1)
            assert outputs.dtype == (np.float64 if name == "reference" else np.float32)
            seen[name].append(rows.tobytes())
            return inner(outputs, rows)

        return loss

    plan = whole_groups(groups)
    fits = {name: train.train_tabm(params, data, rows=ROWS, backend=name, loss=watched(name), batches=plan)
            for name in seen}  # fmt: skip
    planned = [np.asarray(rows, np.int64).tobytes() for epoch in range(6) for rows in plan(epoch)]
    assert seen["reference"] == seen[gpu_backend] == planned  # the loss sees the planned rows, in the plan's order
    assert len(planned) == 30
    close(fits[gpu_backend], fits["reference"])
    assert fits["reference"].report.epoch_loss[-1] < fits["reference"].report.epoch_loss[0] < 0.0
    pred = {name: fits[name].predict(data, rows=slice(0, 140), backend=name) for name in fits}
    np.testing.assert_allclose(pred[gpu_backend], pred["reference"], atol=2e-4)
    assert np.corrcoef(pred["reference"][ROWS], data.y[ROWS])[0, 1] > 0.5  # the members' mean follows the targets


@pytest.mark.parametrize("gpu_backend", GPU)
def test_a_loss_on_a_packed_dataset_with_an_encoder(nn_cache: Path, gpu_backend: str) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    seq, y = sequences(data), np.asarray(data.target, np.float32)
    params = dataclasses.replace(TINY, micro_batch=8, encoder="gru", encoder_embed=8, encoder_hidden=6)
    rows = slice(10, 110)
    built_in = train.train_tabm(params, data, rows=rows, backend="reference", targets=y[:, None], sequences=seq)
    fits = {name: train.train_tabm(params, data, rows=rows, backend=name, loss=squared_error(y), sequences=seq)
            for name in ("reference", gpu_backend)}  # fmt: skip
    assert bits(fits["reference"]) == bits(built_in)
    close(fits[gpu_backend], fits["reference"])
    assert fits[gpu_backend].sequence_inputs == 4
    assert not fits[gpu_backend].multi_target


# -- batches ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("micro", [0, 12])
def test_the_seeded_shuffle_as_a_batch_plan_is_the_default_fit(backend: str, micro: int) -> None:
    data, _ = panel()
    params = dataclasses.replace(TINY, micro_batch=micro)
    default = train.train_tabm(params, data, rows=ROWS, backend=backend)
    planned = train.train_tabm(params, data, rows=ROWS, backend=backend, batches=seeded_shuffle(params, ROWS))
    assert bits(planned) == bits(default)
    assert planned.report.epoch_loss == default.report.epoch_loss


def test_a_plan_decides_the_steps_and_their_rows() -> None:
    plan = list(
        train._plan(TINY, 10, 130, 1, lambda epoch: [np.arange(10, 13), np.full(epoch + 1, 129, np.int32)], steps=7)
    )
    assert [(epoch, step, rows.tolist()) for epoch, step, rows in plan] == [(1, 7, [10, 11, 12]), (1, 8, [129, 129])]
    assert all(rows.dtype == np.int64 for _, _, rows in plan)
    default = list(train._plan(TINY, 10, 130, 0, None, steps=0))
    assert [(epoch, step) for epoch, step, _ in default] == [(e, e * 4 + i) for e in range(2) for i in range(4)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_batches_of_unequal_sizes_train(backend: str) -> None:
    data, groups = panel()
    sizes: list[int] = []

    def plan(epoch: int) -> list[NDArray[np.int64]]:  # one group per batch, the first one twice, the last one left out
        batches = [np.flatnonzero(groups == group) for group in (0, *range(len(GROUP_SIZES) - 1))]
        sizes.extend(len(rows) for rows in batches)
        return batches

    params = dataclasses.replace(TINY, epochs=3)
    model = train.train_tabm(params, data, rows=ROWS, backend=backend, batches=plan)
    assert sizes == [14, *GROUP_SIZES[:-1]] * 3
    assert len(model.report.epoch_loss) == 3
    assert model.report.epoch_loss[2] < model.report.epoch_loss[0]
    again = train.train_tabm(params, data, rows=ROWS, backend=backend, batches=plan)
    assert bits(again) == bits(model)


# -- checkpoints ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_fit_with_a_loss_and_a_plan_resumes_bit_for_bit(backend: str, tmp_path: Path) -> None:
    data, groups = panel()
    params = dataclasses.replace(TINY, epochs=3, micro_batch=12)
    hooks = {"loss": group_pearson(data.y, groups), "batches": whole_groups(groups)}
    whole = train.train_tabm(params, data, rows=ROWS, backend=backend, **hooks)  # type: ignore[arg-type]
    path = tmp_path / "fit.ckpt.npz"
    for epochs in (1, 2):  # stopped after epoch 1, resumed and stopped after epoch 2
        with pytest.raises(Stop):
            train.train_tabm(params, data, rows=ROWS, backend=backend, checkpoint=path, on_epoch=stop_after(epochs),
                             **hooks)  # type: ignore[arg-type]  # fmt: skip
    seen: list[int] = []
    resumed = train.train_tabm(params, data, rows=ROWS, backend=backend, checkpoint=path,
                               on_epoch=lambda epoch, loss, seconds: seen.append(epoch),
                               **hooks)  # type: ignore[arg-type]  # fmt: skip
    assert seen == [2]
    assert bits(resumed) == bits(whole)
    assert resumed.report.epoch_loss == whole.report.epoch_loss


def test_a_checkpoint_records_that_a_loss_or_a_plan_was_given(tmp_path: Path) -> None:
    data, groups = panel()
    loss, plan = group_pearson(data.y, groups), whole_groups(groups)
    path = tmp_path / "fit.ckpt.npz"
    with pytest.raises(Stop):
        train.train_tabm(TINY, data, rows=ROWS, backend="reference", checkpoint=path, on_epoch=stop_after(1),
                         loss=loss, batches=plan)  # fmt: skip
    with pytest.raises(ValueError, match=r"another fit \(it differs in batches, loss\)"):
        train.train_tabm(TINY, data, rows=ROWS, backend="reference", checkpoint=path)
    with pytest.raises(ValueError, match=r"another fit \(it differs in loss\)"):
        train.train_tabm(TINY, data, rows=ROWS, backend="reference", checkpoint=path, batches=plan)
    plain = tmp_path / "plain.ckpt.npz"
    with pytest.raises(Stop):
        train.train_tabm(TINY, data, rows=ROWS, backend="reference", checkpoint=plain, on_epoch=stop_after(1))
    with pytest.raises(ValueError, match=r"another fit \(it differs in batches\)"):
        train.train_tabm(TINY, data, rows=ROWS, backend="reference", checkpoint=plain, batches=plan)


# -- what is refused --------------------------------------------------------------------------------------------------


def test_bad_hooks_raise(nn_cache: Path) -> None:
    data, groups = panel()
    loss = squared_error(data.y)

    def fit(**arguments: object) -> train.TabMModel:
        return train.train_tabm(TINY, data, rows=ROWS, backend="reference", **arguments)  # type: ignore[arg-type]

    packed = PackedDataset.open(nn_cache, binning="r1")
    for labels in ({"targets": np.zeros((packed.n_rows, 1), np.float32)},
                   {"classes": np.zeros((packed.n_rows, 1), np.uint8)}):  # fmt: skip
        with pytest.raises(ValueError, match="a loss holds its own labels: pass no classes or targets"):
            train.train_tabm(TINY, packed, rows=slice(0, 64), backend="reference", loss=loss,
                             **labels)  # fmt: skip
    hl = dataclasses.replace(TINY, objective="hl_gauss", n_bins=8)
    with pytest.raises(ValueError, match="a loss holds its own labels"):
        train.train_tabm(hl, data, rows=ROWS, backend="reference", loss=loss)
    with pytest.raises(ValueError, match=r"gradient of shape \(32, 2\) for outputs of shape \(32, 2, 1\)"):
        fit(loss=lambda outputs, rows: (0.0, outputs[:, :, 0]))
    with pytest.raises(ValueError, match="not finite"):
        fit(loss=lambda outputs, rows: (float("inf"), outputs))
    with pytest.raises(ValueError, match=r"batches\(0\) gave no batch"):
        fit(batches=lambda epoch: [])
    with pytest.raises(ValueError, match=r"batches\(1\) gave no batch"):
        fit(batches=lambda epoch: [np.arange(10, 20)][epoch:])
    for bad, what in (
        (np.arange(10, 43), r"a int64 array of shape \(33,\)"),  # more than params.batch_size rows
        (np.zeros(0, np.int64), r"a int64 array of shape \(0,\)"),
        (np.arange(10.0, 20.0), r"a float64 array of shape \(10,\)"),
        (np.arange(10, 20).reshape(2, 5), r"a int64 array of shape \(2, 5\)"),
        (groups[ROWS] == 0, r"a bool array of shape \(120,\)"),
    ):
        with pytest.raises(ValueError, match=r"batches\(0\) must give 1-D integer arrays of 1\.\.32 rows.*got " + what):
            fit(batches=lambda epoch, bad=bad: [bad])
    for outside in ([9, 10], [129, 130], [-1]):
        with pytest.raises(ValueError, match=r"batches\(0\) names rows outside the training rows \[10, 130\)"):
            fit(batches=lambda epoch, outside=outside: [np.array(outside)])


def test_a_step_needs_labels_or_a_loss_and_hl_gauss_takes_no_loss() -> None:
    data, _ = panel()
    net = train.ReferenceTabM(TINY, 5)
    rows = np.arange(10, 20)
    with pytest.raises(ValueError, match="needs the batch's labels or a loss"):
        train._step(net, None, data.x[rows], None, rows, 0, None)
    before = [a.copy() for a in net.weights().arrays()]
    value = train._step(net, squared_error(data.y), data.x[rows], None, rows, 0, None)
    assert value > 0.0
    assert any(not np.array_equal(a, b) for a, b in zip(net.weights().arrays(), before, strict=True))
    hl = train.ReferenceTabM(dataclasses.replace(TINY, objective="hl_gauss", n_bins=8), 5)
    with pytest.raises(ValueError, match="HL-Gauss network takes no loss"):
        hl.loss_step(data.x[rows], squared_error(data.y), rows, 0)
