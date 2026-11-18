"""Training RealMLP on the packed cache (docs/architecture/networks.md §8): the loop, the model, three backends.

The loop: the preprocessing and the data-dependent initialisation are fitted on ``init_rows`` random training rows; each
epoch is a fresh permutation of the training rows in batches of ``batch_size``, the last partial batch dropped; step k
of K = ⌊n / batch_size⌋ · epochs runs at training fraction t = k / K (the schedules); the final weights are kept. Inputs
are `idun.nn.inputs.dense_inputs(raw_codes=True)` through `idun.realmlp.reference.preprocess`. Every draw is idun's own:
the init rows from ``PCG64(SeedSequence([seed, INIT_ROWS_SALT]))``, the epoch order from `idun.nn.training.epoch_order`,
the weights from `idun.realmlp.reference.init`, the dropout masks from the absolute row and the step. pytabkit fits its
preprocessing on a subsample of ``floor(2^28 / (62F + 1))`` rows (43,289 for F = 100), the default (``init_rows`` 0,
`pytabkit_init_rows`); drawing them directly from the training rows is the same distribution. On Metal each batch is
built on the GPU from the wrapped cache (`MetalRealMLP.bind`), on CUDA from the uploaded one (`CudaRealMLP`); the
reference backend gathers the next batch on a worker thread.

Labels: a packed dataset's come as a compact ``(n_rows, 1)`` uint8 class array (``classes``), or for the ``"mse"``
objective as an ``(n_rows, 1)`` float32 target array (``targets``); more than one column is experimental, not covered
by the documentation. A `MatrixData` carries its own labels.
"""

import dataclasses
import itertools
import json
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from idun.data import PackedDataset
from idun.errors import MultiTargetQualityWarning
from idun.metal import runtime
from idun.nn import inputs
from idun.nn.inputs import Data
from idun.nn.training import FitReport, Validation, checked_weights, epoch_order, score, softmax
from idun.realmlp import reference as ref
from idun.realmlp.cuda import CudaRealMLP
from idun.realmlp.device import DeviceRealMLP
from idun.realmlp.metal import MetalRealMLP, check_classes, check_own, codes_stride, shared_codes
from idun.realmlp.params import RealMLPParams

INIT_ROWS_SALT = 11
PYTABKIT_INIT_BUDGET = 1 << 28  # pytabkit's 1 GB preprocessing budget at 4 bytes a value (`pytabkit_init_rows`)
CHECK_ROWS = 1 << 20  # training rows per chunk when the classes are checked (no K-wide copy of every training row)
BACKENDS = ("metal", "cuda", "reference")
GPU_BACKENDS = ("metal", "cuda")  # the networks that build their batches on the GPU from the bound cache
FORMAT = "idun-realmlp"
FORMAT_VERSION = 2  # 2 adds "multi_target"; a version-1 file is a single-target model
CATEGORY_VERSION = 3  # a model with categorical embeddings: meta "categories", the table after the other arrays


class Backend(Protocol):
    """What the loop needs from a backend."""

    def set_weights(self, weights: ref.Weights) -> None:
        """Copy ``weights`` in and reset the optimizer."""

    def weights(self) -> ref.Weights:
        """Return the current weights (float32, unpadded)."""

    def train_step(
        self, c: NDArray[np.floating], y: NDArray[Any], rows: NDArray[np.integer], step: int, t: float,
        weights: NDArray[np.floating] | None = None,
    ) -> float:  # fmt: skip
        """Run training step ``step`` at training fraction ``t`` on one preprocessed batch; return its mean loss.

        ``y``: the classes, ``(B,)`` for a single-target backend, ``(B, K)`` for one of K targets. ``weights``: each
        row's weight, the loss their weighted mean.
        """

    def predict(self, c: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the expected class of each preprocessed row: ``(n,)``, or ``(n, K)`` for K targets."""

    def inference_logits(self, c: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the inference logits ``(n, K · C)`` of preprocessed rows (no dropout)."""

    def close(self) -> None:
        """Release the backend's resources."""


class ReferenceRealMLP:
    """The NumPy reference as a backend (float64); ``n_targets`` as for `MetalRealMLP`."""

    def __init__(self, params: RealMLPParams, n_targets: int | None = None) -> None:
        check_own(params, n_targets)
        self.params, self.n_targets = params, n_targets
        self._weights: ref.Weights | None = None
        self._state: ref.AdamState | None = None

    def set_weights(self, weights: ref.Weights) -> None:
        """Copy ``weights`` in (float64) and reset Adam."""
        if weights.out.shape[-1] != self.params.n_classes * (self.n_targets or 1):
            raise ValueError(f"the output layer has {weights.out.shape[-1]} outputs; this backend has "
                             f"{self.n_targets or 1} targets of {self.params.n_classes} classes")  # fmt: skip
        own = self.n_targets if self.params.own_last_block else 1
        if ref.own_targets(weights) != own:
            raise ValueError(f"the last hidden block has {weights.act[-1].size} units; this backend's has "
                             f"{own} x {self.params.widths[-1]}")  # fmt: skip
        self._weights = weights.astype(np.float64)
        self._state = ref.adam_state(self._weights)

    def _ready(self) -> tuple[ref.Weights, ref.AdamState]:
        if self._weights is None or self._state is None:
            raise RuntimeError("set_weights first: RealMLP's initialisation needs the init rows")
        return self._weights, self._state

    def weights(self) -> ref.Weights:
        """Return the current weights as float32."""
        return self._ready()[0].astype(np.float32)

    def train_step(
        self, c: NDArray[np.floating], y: NDArray[Any], rows: NDArray[np.integer], step: int, t: float,
        weights: NDArray[np.floating] | None = None,
    ) -> float:  # fmt: skip
        """Run one training step; return the (weighted) mean loss."""
        current, state = self._ready()
        masks = ref.dropout_masks(self.params, step, t, np.asarray(rows, np.int64), np.float64)
        logits, cache = ref.forward(current, np.asarray(c, np.float64), masks, self.params.act)
        labels = np.asarray(y, np.float64 if self.params.objective == "mse" else np.int64)
        loss, d_logits = ref.loss_and_logit_grad(self.params, logits, labels, weights)
        ref.adam_step(self.params, current, ref.backward(current, cache, d_logits, self.params.act), state, t)
        return loss

    def predict(self, c: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the expected class of each row (inference): ``(n,)``, or ``(n, K)``."""
        logits = ref.forward(self._ready()[0], np.asarray(c, np.float64), None, self.params.act)[0]
        return ref.predict(logits, self.n_targets, self.params.objective)

    def inference_logits(self, c: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the inference logits ``(n, K · C)`` in float64."""
        return ref.forward(self._ready()[0], np.asarray(c, np.float64), None, self.params.act)[0]

    def close(self) -> None:
        """Nothing to release."""


def predict_chunk(params: RealMLPParams, n_targets: int | None) -> int:
    """Return the rows per prediction pass: 4,096, or fewer for a wide own last block (at most 2^24 of its units)."""
    if not params.own_last_block or n_targets is None:
        return 4096
    return max(64, min(4096, (1 << 24) // (n_targets * params.widths[-1]) // 64 * 64))


def make_backend(name: str, params: RealMLPParams, n_inputs: int, max_batch: int, n_targets: int | None = None,
                 categories: Sequence[int] = ()) -> Backend:  # fmt: skip
    """Construct the backend called ``name`` (``"metal"``, ``"cuda"``, ``"reference"``) for ``n_targets`` (None: 1).

    ``categories``: the levels of the categorical inputs, the last of the ``n_inputs`` (the reference reads them from
    the weights it is given).
    """
    if name == "metal":
        return MetalRealMLP(params, n_inputs, max_batch, n_targets, categories)
    if name == "cuda":
        return CudaRealMLP(params, n_inputs, max_batch, n_targets, categories)
    if name == "reference":
        return ReferenceRealMLP(params, n_targets)
    raise ValueError(f"unknown backend {name!r}; the backends are {BACKENDS}")


def pytabkit_init_rows(n_inputs: int) -> int:
    """Return pytabkit's preprocessing subsample for ``n_inputs`` inputs: ``floor(2^28 / (62F + 1))`` rows.

    pytabkit 1.7.3 fits RealMLP-TD's preprocessing on at most ``ram_limit_gb · 2^30 / (4 · n_forward)`` rows
    (``_fit_transform_subsample`` in ``models/nn_models/base.py``), with a 1 GB limit and ``n_forward = 62F + 1`` values
    per row for its default network: 43,289 rows for F = 100, and every row of a table that has fewer.
    """
    return max(1, PYTABKIT_INIT_BUDGET // (62 * n_inputs + 1))


def init_rows(params: RealMLPParams, lo: int, n: int, n_inputs: int) -> NDArray[np.int64]:
    """Return the sorted absolute rows that fit the preprocessing and the initialisation.

    ``params.init_rows`` of the ``n`` training rows from ``lo``, or with 0 `pytabkit_init_rows` of ``n_inputs``; all of
    them when there are fewer.
    """
    count = min(params.init_rows or pytabkit_init_rows(n_inputs), n)
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([params.seed, INIT_ROWS_SALT])))
    return lo + np.sort(generator.choice(n, count, replace=False)).astype(np.int64)


def raw_inputs(data: Data, rows: NDArray[np.int64]) -> NDArray[np.float64]:
    """Return RealMLP's unpreprocessed inputs of ``rows``: named factors (NaN → 0), raw codes, country one-hot."""
    return inputs.dense_inputs(data, rows, raw_codes=True).astype(np.float64)


def model_inputs(params: RealMLPParams, pre: ref.Preprocess, x: NDArray[np.float64]) -> NDArray[np.float32]:
    """Return raw inputs ``x`` preprocessed: the numeric ones (``pre``'s), then the categorical inputs' codes as are."""
    return _preprocessed(params, pre, x).astype(np.float32)


def _preprocessed(params: RealMLPParams, pre: ref.Preprocess, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """`model_inputs` in float64, as `ref.init` takes them: its data-dependent scales and biases see unrounded rows."""
    numeric = len(pre.median)
    return np.concatenate([ref.preprocess(params, pre, x[:, :numeric]), x[:, numeric:]], axis=1)


@dataclass
class RealMLPModel:
    """A trained RealMLP: parameters, preprocessing statistics, weights, the inputs it expects, the rows it saw.

    ``multi_target``: experimental, not covered by the documentation. With categorical inputs (``weights.categories``,
    the last of the ``n_inputs``), ``preprocess`` holds the numeric inputs' statistics only.
    """

    params: RealMLPParams
    preprocess: ref.Preprocess
    weights: ref.Weights
    n_inputs: int
    train_rows: tuple[int, int]
    report: FitReport
    multi_target: bool = False

    def __post_init__(self) -> None:
        width, classes = self.weights.out.shape[-1], self.params.n_classes
        if width < classes or width % classes or (width != classes and not self.multi_target):
            raise ValueError(f"an output layer of {width} outputs is not K targets of {classes} classes"
                             f"{'' if self.multi_target else ' with K = 1 (multi_target is False)'}")  # fmt: skip
        if self.params.own_last_block and not self.multi_target:
            raise ValueError("own_last_block needs a model of a (K, n_rows) target array (multi_target)")
        own = width // classes if self.params.own_last_block else 1
        if ref.own_targets(self.weights) != own:
            raise ValueError(f"a last hidden block of {self.weights.act[-1].size} units is not {own} x "
                             f"{self.params.widths[-1]}")  # fmt: skip
        numeric = self.n_inputs - len(self.weights.levels)
        if len(self.preprocess.median) != numeric or (self.weights.table is None) != (not self.weights.levels):
            count = len(self.weights.levels)
            raise ValueError(f"preprocessing of {len(self.preprocess.median)} inputs and {count} categorical inputs "
                             f"do not make {self.n_inputs}")  # fmt: skip

    @property
    def n_targets(self) -> int | None:
        """None for a single-target model (an integer: experimental, not covered by the documentation)."""
        return self.weights.out.shape[-1] // self.params.n_classes if self.multi_target else None

    def inputs(self, data: Data, rows: NDArray[np.int64]) -> NDArray[np.float32]:
        """Return the preprocessed inputs of ``rows``: the numeric ones, then the categorical inputs' codes."""
        return model_inputs(self.params, self.preprocess, raw_inputs(data, rows))

    def predict(self, data: Data, *, rows: slice, backend: str, chunk: int | None = None,
                codes: NDArray[np.uint8] | None = None) -> NDArray[np.float32]:  # fmt: skip
        """Return the expected class of each row in ``rows`` (a contiguous slice), ``chunk`` rows per pass.

        The shape is ``(n,)``. ``chunk`` None: `predict_chunk` (the bits do not depend on it). ``codes``:
        `idun.realmlp.metal.shared_codes` of ``data``, read instead of a fresh copy (the GPU backends only).
        """
        _check_codes(codes, backend)
        chunk = predict_chunk(self.params, self.n_targets) if chunk is None else chunk
        if inputs.n_inputs(data) != self.n_inputs:
            raise ValueError(f"the data has {inputs.n_inputs(data)} inputs; the model expects {self.n_inputs}")
        lo, hi, step = rows.indices(data.n_rows)
        if step != 1:
            raise ValueError("rows must be a contiguous slice")
        net = make_backend(backend, self.params, self.n_inputs, chunk, self.n_targets, self.weights.levels)
        try:
            net.set_weights(self.weights)
            if isinstance(net, DeviceRealMLP) and isinstance(data, PackedDataset):  # batches built on the GPU
                net.bind(data, self.preprocess, codes=codes)
                return net.predict_rows(lo, hi)
            n = max(hi - lo, 0)
            out = np.empty((n,) if self.n_targets is None else (n, self.n_targets), np.float32)
            for start in range(lo, hi, chunk):
                stop = min(start + chunk, hi)
                out[start - lo : stop - lo] = net.predict(self.inputs(data, np.arange(start, stop)))
            return out
        finally:
            net.close()

    def predict_proba(self, data: Data, *, rows: slice, backend: str, chunk: int | None = None) -> NDArray[np.float32]:
        """Return each row's class probabilities ``(n, n_classes)`` (single-target; batches built on the host)."""
        if self.multi_target or self.params.objective != "cross_entropy":
            raise ValueError("predict_proba needs a single-target classification model")
        chunk = predict_chunk(self.params, self.n_targets) if chunk is None else chunk
        if inputs.n_inputs(data) != self.n_inputs:
            raise ValueError(f"the data has {inputs.n_inputs(data)} inputs; the model expects {self.n_inputs}")
        lo, hi, step = rows.indices(data.n_rows)
        if step != 1:
            raise ValueError("rows must be a contiguous slice")
        net = make_backend(backend, self.params, self.n_inputs, chunk, self.n_targets, self.weights.levels)
        try:
            net.set_weights(self.weights)
            out = np.empty((max(hi - lo, 0), self.params.n_classes), np.float32)
            for start in range(lo, hi, chunk):
                stop = min(start + chunk, hi)
                out[start - lo : stop - lo] = softmax(net.inference_logits(self.inputs(data, np.arange(start, stop))))
            return out
        finally:
            net.close()

    def save(self, path: str | Path) -> None:
        """Write the model to ``path`` (an uncompressed ``.npz`` without object arrays)."""
        meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "params": asdict(self.params),
                "n_inputs": self.n_inputs, "train_rows": list(self.train_rows), "blocks": len(self.weights.weight),
                "report": asdict(self.report), "multi_target": self.multi_target}  # fmt: skip
        if self.weights.categories:
            meta.update(format_version=CATEGORY_VERSION, categories=list(self.weights.categories))
        arrays = {f"p{i:02d}": a for i, a in enumerate(self.weights.arrays())}
        with Path(path).open("wb") as handle:
            np.savez(handle, allow_pickle=False, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
                     median=self.preprocess.median, factor=self.preprocess.factor, **arrays)  # fmt: skip

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a model written by `save` (format version 1, 2 or 3); anything else raises ``ValueError``."""
        with np.load(path, allow_pickle=False) as archive:
            meta: dict[str, Any] = json.loads(archive["meta"].tobytes())
            if meta.get("format") != FORMAT or meta.get("format_version") not in (1, FORMAT_VERSION, CATEGORY_VERSION):
                raise ValueError(f"{path} is not a {FORMAT} v1, v2 or v3 file")
            n, categories = int(meta["blocks"]), tuple(int(m) for m in meta.get("categories", ()))
            arrays = [archive[f"p{i:02d}"] for i in range(5 + 3 * n + 2 + bool(categories))]
            pre = ref.Preprocess(archive["median"], archive["factor"])
        w1, b1, w2, b2, scale = arrays[:5]
        weights = ref.Weights(w1, b1, w2, b2, scale, arrays[5 : 5 + n], arrays[5 + n : 5 + 2 * n],
                              arrays[5 + 2 * n : 5 + 3 * n], arrays[5 + 3 * n], arrays[5 + 3 * n + 1],
                              arrays[-1] if categories else None, categories or None)  # fmt: skip
        return cls(RealMLPParams(**meta["params"]), pre, weights, int(meta["n_inputs"]), tuple(meta["train_rows"]),
                   FitReport(**meta["report"]), bool(meta.get("multi_target", False)))  # fmt: skip


SCHEDULE_ZEROS = (0.2, 7 / 15, 1.0)  # coslog4 is 0 at t = 1/15, 0.2, 7/15 and 1: its cycles' ends after the first


def snapshot_steps(snapshots: Sequence[float], total: int, on_snapshot: Callable[[float, ref.Weights], None] | None
                   ) -> dict[int, float]:  # fmt: skip
    """Return {steps done: fraction} for `train_realmlp`'s ``snapshots``; ``ValueError`` for a bad request."""
    fractions = list(snapshots)
    if fractions and on_snapshot is None:
        raise ValueError("snapshots need on_snapshot")
    if any(not 0.0 < f <= 1.0 for f in fractions) or any(b <= a for a, b in itertools.pairwise(fractions)):
        raise ValueError(f"snapshots must be increasing training fractions in (0, 1], got {fractions}")
    marks = {round(f * total): f for f in fractions}
    if len(marks) != len(fractions) or 0 in marks:
        raise ValueError(f"snapshots {fractions} fall on fewer than {len(fractions)} distinct steps of {total}")
    return marks


def _checked_classes(classes: NDArray[np.uint8], data: Data, lo: int, hi: int, n_classes: int) -> NDArray[np.uint8]:
    """Return the ``(n_rows, K)`` uint8 ``classes``; ``ValueError`` unless rows [lo, hi) are below ``n_classes``."""
    array = check_classes(classes, data.n_rows)
    for start in range(lo, hi, CHECK_ROWS):
        if int(array[start : min(start + CHECK_ROWS, hi)].max()) >= n_classes:
            raise ValueError(f"the classes of the training rows must be 0..{n_classes - 1}, not "
                             f"{inputs.MISSING_CLASS} (a missing label) or above")  # fmt: skip
    return array


def _checked_labels(
    data: Data, params: RealMLPParams, classes: NDArray[np.uint8] | None, targets: NDArray[np.float32] | None, lo: int,
    hi: int,
) -> tuple[NDArray[np.uint8] | None, NDArray[np.float32] | None, int]:  # fmt: skip
    """Return the checked ``classes`` and ``targets`` (one of them, or neither for a `MatrixData`) and K."""
    inputs.check_label_source(data, params.objective, classes, targets)
    classes = None if classes is None else _checked_classes(classes, data, lo, hi, params.n_classes)
    targets = None if targets is None else inputs.checked_targets(targets, data.n_rows, lo, hi)
    labelled = classes if targets is None else targets
    return classes, targets, 1 if labelled is None else int(labelled.shape[1])


def _end_epoch(report: FitReport, epoch: int, losses: list[float], begin: float,
               on_epoch: Callable[[int, float, float], None] | None) -> None:  # fmt: skip
    report.epoch_loss.append(float(np.mean(losses)))
    report.epoch_seconds.append(time.perf_counter() - begin)
    if on_epoch is not None:
        on_epoch(epoch, report.epoch_loss[-1], report.epoch_seconds[-1])


def _snapshot_hook(net: Backend, marks: dict[int, float], on_snapshot: Callable[[float, ref.Weights], None] | None
                   ) -> Callable[[int], None]:  # fmt: skip
    """Return the call that hands ``on_snapshot`` the weights once a marked number of steps has run."""

    def snapshot(done: int) -> None:
        if done in marks and on_snapshot is not None:
            on_snapshot(marks[done], net.weights())

    return snapshot


Weigh = Callable[[NDArray[np.int64]], NDArray[np.float32] | None]  # a batch's row weights, or None (unweighted)


def _device_epoch(net: DeviceRealMLP, order: NDArray[np.int64], first: int, total: int, batch: int,
                 marks: dict[int, float], snapshot: Callable[[int], None], weigh: Weigh) -> list[float]:  # fmt: skip
    """Run one epoch's pipelined steps (global steps ``first`` ..); a snapshot ends a run of them (the same bits)."""
    per_epoch, losses, start = len(order) // batch, [], 0
    for stop in sorted({m - first for m in marks if first < m < first + per_epoch} | {per_epoch}):
        rows = [order[i * batch : (i + 1) * batch] for i in range(start, stop)]
        losses += net.train_steps_rows(
            (part, first + i, (first + i) / total, weigh(part)) for i, part in enumerate(rows, start)
        )
        snapshot(first + stop)
        start = stop
    return losses


def _host_epoch(net: Backend, pool: ThreadPoolExecutor,
                gather: Callable[[NDArray[np.int64]], tuple[NDArray[np.float32], NDArray[Any]]],
                order: NDArray[np.int64], first: int, total: int, batch: int,
                snapshot: Callable[[int], None], weigh: Weigh) -> list[float]:  # fmt: skip
    """Run one epoch's steps on batches gathered on the host (global steps ``first`` ..), the next during this one."""
    per_epoch, losses = len(order) // batch, []
    plan = [order[i * batch : (i + 1) * batch] for i in range(per_epoch)]
    pending = pool.submit(gather, plan[0])
    for i, rows in enumerate(plan):
        c, y = pending.result()
        if i + 1 < per_epoch:
            pending = pool.submit(gather, plan[i + 1])  # gathered while the GPU runs this step
        weights = weigh(rows)
        if weights is None:
            losses.append(net.train_step(c, y, rows, first + i, (first + i) / total))
        else:
            losses.append(net.train_step(c, y, rows, first + i, (first + i) / total, weights))
        snapshot(first + i + 1)
    return losses


def train_realmlp(
    params: RealMLPParams,
    data: Data,
    *,
    rows: slice,
    backend: str,
    on_epoch: Callable[[int, float, float], None] | None = None,
    classes: NDArray[np.uint8] | None = None,
    snapshots: Sequence[float] = (),
    on_snapshot: Callable[[float, ref.Weights], None] | None = None,
    codes: NDArray[np.uint8] | None = None,
    targets: NDArray[np.float32] | None = None,
    valid: Validation | None = None,
    patience: int | None = None,
    weight: NDArray[np.floating] | None = None,
    categories: Sequence[int] = (),
) -> RealMLPModel:
    """Train RealMLP on the contiguous training rows ``rows``; ``on_epoch(epoch, loss, seconds)`` reports.

    ``codes``: `idun.realmlp.metal.shared_codes` of ``data``, which a GPU backend places instead of making its own
    row-major copy (`train_realmlp_seeds` shares one between seeds); the bits are the same. The GPU backends only.

    ``classes``: a packed dataset's labels, a compact ``(n_rows, 1)`` uint8 class array (row r's class at
    ``[r, 0]``); the Metal backend wraps it without a copy, so it must be page-aligned with contiguous rows
    (`inputs.class_array`, or a memory-mapped file); the CUDA backend uploads it. Every class of the training rows must
    be below ``params.n_classes`` (`inputs.MISSING_CLASS`, 255, may mark rows outside them); ``ValueError`` otherwise,
    before any training. One column (the model predicts ``(n,)``); more is experimental, not covered by the
    documentation. ``targets``: instead, for ``params.objective`` ``"mse"``, a packed dataset's regression target, an
    ``(n_rows, 1)`` float32 target array (`inputs.target_array`), wrapped by the
    Metal backend as ``classes`` are, so page-aligned, and finite on every training row (``ValueError`` otherwise,
    before any training); the model predicts the output values. A `MatrixData` carries its own labels and takes neither.


    ``snapshots``: training fractions in (0, 1], increasing; once the first ``round(f · total)`` steps have run,
    ``on_snapshot(f, weights)`` gets that moment's weights (float32), so one run gives the networks of several
    stopping points (for example the schedule's zeros, `SCHEDULE_ZEROS`). The training is unchanged, bit for bit.

    ``weight``: one finite, non-negative weight per training row (``(hi − lo,)``, row ``lo + i`` at ``i``; not all
    zero). Each step's loss is then its batch's weighted mean, ``Σ w l / Σ w`` over the batch's rows, in place of the
    plain mean; the batches and the dropout are unchanged, a row of weight 0 only
    leaves its batch's loss, and unit weights give the unweighted fit bit for bit on a GPU. A batch whose rows all
    weigh 0 has loss 0 and gradient 0 (AdamW still steps on its moments). The reported losses are the weighted
    means; a validation set's score is not weighted.
    """
    lo, hi, step = rows.indices(data.n_rows)
    n = hi - lo
    if step != 1 or n < params.batch_size:
        raise ValueError(f"rows must be a contiguous slice of at least batch_size ({params.batch_size}) rows")
    categories = _checked_categories(data, categories, lo, hi)
    classes, targets, width = _checked_labels(data, params, classes, targets, lo, hi)
    row_weights = checked_weights(weight, n)
    n_targets = width if width > 1 or params.own_last_block else None  # one target: the single-target network
    check_own(params, n_targets)
    _check_codes(codes, backend)
    if n_targets is not None and n_targets > 1:
        warnings.warn(
            f"train_realmlp trains {n_targets} targets on one network (experimental): less accurate than separate "
            "networks, and the heads' outputs come out nearly alike",
            MultiTargetQualityWarning,
            stacklevel=2,
        )
    _check_validation(params, valid, patience, n_targets)
    n_inputs = inputs.n_inputs(data)
    begin = time.perf_counter()
    model = _initial_model(params, data, (lo, hi), n_targets, categories)

    def labels_of(batch: NDArray[np.int64]) -> NDArray[Any]:
        y: NDArray[Any]
        if targets is not None:
            y = targets[batch]  # finite (checked above)
        elif classes is None:
            return inputs.labels(data, batch)  # type: ignore[arg-type]  # a MatrixData (checked above)
        else:
            y = inputs.class_labels(classes, batch, params.n_classes)
        return y if n_targets is not None else y[:, 0]

    def gather(batch: NDArray[np.int64]) -> tuple[NDArray[np.float32], NDArray[Any]]:
        return model.inputs(data, batch), labels_of(batch)

    def weigh(batch: NDArray[np.int64]) -> NDArray[np.float32] | None:
        return None if row_weights is None else row_weights[batch - lo]

    per_epoch = n // params.batch_size
    total = per_epoch * params.epochs
    marks = snapshot_steps(snapshots, total, on_snapshot)
    net = make_backend(backend, params, n_inputs, params.batch_size, n_targets, categories)
    snapshot = _snapshot_hook(net, marks, on_snapshot)
    try:
        net.set_weights(model.weights)
        if isinstance(net, DeviceRealMLP) and isinstance(data, PackedDataset):  # GPU batches
            if valid is not None:
                raise ValueError("a validation set scores host batches: not with GPU batches of a packed dataset")
            net.bind(data, model.preprocess, classes, codes, targets=targets)
            for epoch in range(params.epochs):
                epoch_begin = time.perf_counter()
                losses = _device_epoch(net, lo + epoch_order(params.seed, epoch, n), epoch * per_epoch, total,
                                      params.batch_size, marks, snapshot, weigh)  # fmt: skip
                _end_epoch(model.report, epoch, losses, epoch_begin, on_epoch)
            model.weights = net.weights()
            model.report.seconds = time.perf_counter() - begin
            return model
        keeper = None if valid is None else _Keeper(net, model, valid, params.batch_size, patience)
        _host_fit(net, model, gather, (lo, n, per_epoch, total), snapshot, on_epoch, keeper, weigh)
        model.report.seconds = time.perf_counter() - begin
        return model
    finally:
        net.close()


def _host_fit(
    net: Backend,
    model: "RealMLPModel",
    gather: Callable[[NDArray[np.int64]], tuple[NDArray[np.float32], NDArray[Any]]],
    plan: tuple[int, int, int, int],
    snapshot: Callable[[int], None],
    on_epoch: Callable[[int, float, float], None] | None,
    keeper: "_Keeper | None",
    weigh: Weigh,
) -> None:
    """Train every epoch on host batches (``plan``: the first row, the rows, the steps an epoch, the steps in all)."""
    params = model.params
    lo, n, per_epoch, total = plan
    with ThreadPoolExecutor(max_workers=1) as pool:
        for epoch in range(params.epochs):
            epoch_begin = time.perf_counter()
            losses = _host_epoch(net, pool, gather, lo + epoch_order(params.seed, epoch, n), epoch * per_epoch,
                                 total, params.batch_size, snapshot, weigh)  # fmt: skip
            _end_epoch(model.report, epoch, losses, epoch_begin, on_epoch)
            if keeper is not None and keeper.after(epoch):
                break  # no better epoch for `patience` epochs: the best one's weights
    model.weights = net.weights() if keeper is None or keeper.best is None else keeper.best


def _initial_model(params: RealMLPParams, data: Data, rows: tuple[int, int], n_targets: int | None,
                   categories: tuple[int, ...]) -> RealMLPModel:  # fmt: skip
    """Return the untrained model: the preprocessing and the weights fitted on the init rows of the training rows."""
    lo, hi = rows
    n_inputs = inputs.n_inputs(data)
    x_init = raw_inputs(data, init_rows(params, lo, hi - lo, n_inputs))
    pre = ref.fit_preprocess(x_init[:, : n_inputs - len(categories)])
    weights = ref.init(params, _preprocessed(params, pre, x_init), n_targets or 1, categories).astype(np.float32)
    return RealMLPModel(params, pre, weights, n_inputs, (lo, hi), FitReport(), n_targets is not None)


def _checked_categories(data: Data, categories: Sequence[int], lo: int, hi: int) -> tuple[int, ...]:
    """Return ``categories`` as a tuple; ``ValueError`` unless the data's last inputs hold valid codes of them."""
    found = tuple(categories)
    if found and isinstance(data, PackedDataset):
        raise ValueError("categorical inputs train on host batches of a MatrixData: a packed dataset has none")
    if not found:
        return found
    n_inputs = inputs.n_inputs(data)
    if any(m < 1 for m in found) or len(found) >= n_inputs:
        raise ValueError(f"categories must be positive level counts of the last of the {n_inputs} inputs, at least "
                         f"one numeric input before them; got {list(found)}")  # fmt: skip
    for start in range(lo, hi, CHECK_ROWS):
        codes = raw_inputs(data, np.arange(start, min(start + CHECK_ROWS, hi)))[:, n_inputs - len(found) :]
        if not np.array_equal(codes, np.round(codes)) or ((codes < 0) | (codes >= np.array(found))).any():
            raise ValueError(
                f"the last {len(found)} inputs must be the categorical inputs' level codes, 0 .. levels - 1"
            )
    return found


def _check_validation(
    params: RealMLPParams, valid: Validation | None, patience: int | None, n_targets: int | None
) -> None:
    """Refuse a validation set or a patience `train_realmlp` cannot use (one target, the network's classes)."""
    if patience is not None and (valid is None or patience < 1):
        raise ValueError(f"patience needs a validation set and a positive count, got {patience!r}")
    if valid is None:
        return
    valid.bounds()
    if n_targets is not None:
        raise ValueError("a validation set scores one target (yet)")
    if valid.data.n_classes != params.n_classes:
        raise ValueError(f"the validation set has {valid.data.n_classes} classes, the network {params.n_classes}")


@dataclass
class _Keeper:
    """Scores the validation rows after each epoch, keeps the best epoch's weights, says when patience runs out."""

    net: Backend
    model: "RealMLPModel"
    valid: Validation
    chunk: int
    patience: int | None
    best: ref.Weights | None = None

    def after(self, epoch: int) -> bool:
        """Score epoch ``epoch``; return whether training stops after it."""
        report = self.model.report
        lo, hi = self.valid.bounds()
        parts = []
        for start in range(lo, hi, self.chunk):
            c = self.model.inputs(self.valid.data, np.arange(start, min(start + self.chunk, hi)))
            if self.model.params.objective == "cross_entropy":
                parts.append(softmax(self.net.inference_logits(c)))
            else:
                parts.append(np.asarray(self.net.predict(c), np.float64).reshape(-1))
        report.valid_score.append(score(np.concatenate(parts), self.valid.data.y[lo:hi], self.valid.data.n_classes))
        if report.best_epoch is None or report.valid_score[-1] < report.valid_score[report.best_epoch]:
            report.best_epoch, self.best = epoch, self.net.weights()
            return False
        return self.patience is not None and epoch - report.best_epoch >= self.patience


SEED_WORKERS = 2  # networks in flight in train_realmlp_seeds: the fastest on an Apple M4


def train_realmlp_seeds(
    params: RealMLPParams,
    data: PackedDataset,
    *,
    rows: slice,
    backend: str,
    seeds: Sequence[int],
    workers: int | None = None,
    classes: NDArray[np.uint8] | None = None,
    codes: NDArray[np.uint8] | None = None,
    targets: NDArray[np.float32] | None = None,
) -> list[RealMLPModel]:
    """Train a seed bag: one network per seed in ``seeds``, each identical to `train_realmlp` with that seed.

    Up to ``workers`` networks train side by side, each on its own backend and thread, all reading one row-major copy
    of the codes (`idun.realmlp.metal.shared_codes`, made once). The models come back in ``seeds`` order, bit for bit as
    separate fits. Side by side is a modest gain for RealMLP (its 256-row steps nearly fill the GPU's time already):
    on an Apple M4 four seeds took 1.16x less time with two networks in flight, 1.13x with three. Each network
    references about 4.2 GB on the GPU at v6, so ``workers`` None takes `SEED_WORKERS`, or fewer when the GPU's working
    set cannot hold them (`runtime.side_by_side`: one on an 8 GB M1, where two networks ran 16x slower than one); a
    larger explicit ``workers`` raises ``ValueError``. ``workers`` > 1 needs ``backend="metal"``. A bag predicts the
    mean of its networks' predictions (`predict_mean`). ``codes``: a `shared_codes` copy to use (for example one the
    caller also predicts with); None makes one for
    a GPU backend. ``classes`` or ``targets``: the labels, as for `train_realmlp`.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct and at least one, got {list(seeds)}")
    if workers is not None and (workers < 1 or (workers > 1 and backend != "metal")):
        raise ValueError(f"workers must be 1, or > 1 with backend 'metal'; got {workers} with {backend!r}")
    _check_codes(codes, backend)
    if backend == "metal":
        labels = sum(np.asarray(a).nbytes for a in (classes, targets) if a is not None)
        workers = runtime.workers_side_by_side(workers, SEED_WORKERS, network_bytes(data, labels))
    workers = 1 if workers is None else workers
    if codes is None and backend in GPU_BACKENDS:
        codes = shared_codes(data)

    def fit(seed: int) -> RealMLPModel:
        return train_realmlp(dataclasses.replace(params, seed=seed), data, rows=rows, backend=backend,
                             classes=classes, codes=codes, targets=targets)  # fmt: skip

    if workers == 1:
        return [fit(seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=min(workers, len(seeds))) as pool:
        return list(pool.map(fit, seeds))


def network_bytes(data: PackedDataset, label_bytes: int) -> int:
    """Estimate the bytes one bound Metal network references on the GPU: codes, named values, countries, labels."""
    codes = data.n_rows * codes_stride(len(data.obfuscated_features))
    return codes + np.asarray(data.named_values).nbytes + np.asarray(data.country).nbytes + label_bytes + 2**26


def predict_mean(models: Sequence[RealMLPModel], data: PackedDataset, *, rows: slice, backend: str,
                 codes: NDArray[np.uint8] | None = None) -> NDArray[np.float32]:  # fmt: skip
    """Return a bag's prediction: the mean of ``models``' predictions of ``rows`` (float64 sum in order)."""
    if not models:
        raise ValueError("a bag needs at least one model")
    total = models[0].predict(data, rows=rows, backend=backend, codes=codes).astype(np.float64)
    for model in models[1:]:
        total += model.predict(data, rows=rows, backend=backend, codes=codes)
    return (total / len(models)).astype(np.float32)


def _check_codes(codes: NDArray[np.uint8] | None, backend: str) -> None:
    if codes is not None and backend not in GPU_BACKENDS:
        raise ValueError(f"codes are the GPU backends' row-major copy; backend {backend!r} does not read them")
