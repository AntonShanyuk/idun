"""Training TabM on the packed cache: the loop, the model, three backends.

The loop: each epoch a fresh permutation of the training rows, batches of ``batch_size`` rows, the ragged last batch
kept, one AdamW step per batch, final weights after ``epochs`` epochs. Every draw is idun's own and counter-based: the
epoch order from ``PCG64(SeedSequence([seed, epoch, ORDER_SALT]))``, the initial weights from `reference.init`, the
dropout masks from the absolute row and the global step. The next batch's inputs are gathered on a worker thread while
the backend runs the current step.

Backends: ``"metal"`` (`idun.tabm.metal.MetalTabM`, fp32 on an Apple GPU), ``"cuda"`` (`idun.tabm.cuda.CudaTabM`,
fp32 on an NVIDIA GPU) and ``"reference"`` (`ReferenceTabM`, the NumPy reference in float64; the oracle for small
problems). No automatic choice between them.

Labels: a packed dataset's come as an ``(n_rows, 1)`` uint8 class array (``classes``), or for the ``"mse"`` objective
as an ``(n_rows, 1)`` float32 target array (``targets``); more columns are experimental, not covered by the
documentation. A `MatrixData` carries its own labels.

Hooks: `train_tabm` takes a training loss (``loss``, a host callable on a batch's member outputs: `Loss`) and a batch
plan (``batches``, a function of the epoch: `BatchPlan`) in place of the built-in loss and the seeded shuffle.
"""

import hashlib
import json
import os
import time
import warnings
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from idun.errors import MultiTargetQualityWarning
from idun.nn import inputs
from idun.nn.inputs import Data
from idun.nn.training import FitReport, Validation, checked_weights, epoch_order, score, softmax
from idun.tabm import embeddings as emb
from idun.tabm import hl_gauss, reference
from idun.tabm.cuda import CudaTabM
from idun.tabm.encoder import EncoderWeights
from idun.tabm.metal import MetalTabM
from idun.tabm.params import TabMParams, scheduled_rate
from idun.tabm.reference import Loss
from idun.tabm.sequences import Sequences

BACKENDS = ("metal", "cuda", "reference")
FORMAT = "idun-tabm"
FORMAT_VERSION = 1  # single-target models; multi-target ones are version 2 (meta "multi_target")
MULTI_TARGET_VERSION = 2
ENCODER_VERSION = 3  # models with a temporal encoder (meta "multi_target" and "sequence_inputs")
ENCODER_FIELDS = ("encoder", "encoder_embed", "encoder_hidden")  # left out of a file when all are the defaults
EMBEDDING_VERSION = 4  # models with piecewise-linear embeddings (their encoding in bins_scale, bins_shift, bins_single)
EMBEDDING_FIELDS = ("num_emb_type", "num_emb_n_bins", "d_embedding")  # left out of a file when all are the defaults
EMB_SAMPLE_ROWS = 50_000  # the embeddings' bins: at most this many training rows, drawn with the seed
CHECKPOINT_FORMAT = "idun-tabm-checkpoint"
CHECKPOINT_VERSION = 1

# A batch plan given by the caller: the epoch -> its batches, each a 1-D integer array of absolute training rows.
BatchPlan = Callable[[int], Sequence[NDArray[np.integer]]]


class Backend(Protocol):
    """What the loop needs from a backend."""

    def set_weights(self, weights: reference.Weights) -> None:
        """Copy ``weights`` in and reset the optimizer."""

    def weights(self) -> reference.Weights:
        """Return the current weights (float32, unpadded)."""

    def set_learning_rate(self, rate: float | None) -> None:
        """Set the next steps' learning rate; None is ``params.learning_rate``."""

    def train_step(self, x: NDArray[np.floating], y: NDArray[Any], rows: NDArray[np.integer], step: int,
                   seq: NDArray[np.floating] | None = None,
                   weights: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run training step ``step`` on one batch (labels ``(B, K)``, or ``(B,)`` for one target); return its loss.

        ``seq``: the rows' sequences ``(B, S, C)`` for a network with an encoder; ``weights``: each row's weight, the
        loss their weighted mean.
        """

    def predict(self, x: NDArray[np.floating], seq: NDArray[np.floating] | None = None) -> NDArray[np.floating]:
        """Return the expected class of each row: ``(B,)`` for one target, ``(B, K)`` for K > 1."""

    def close(self) -> None:
        """Release the backend's resources."""


class InferenceBackend(Backend, Protocol):
    """A backend that also returns its logits and its training state, as `predict_proba` and checkpoints need.

    And that steps under a loss of the caller's. Both of ours.
    """

    def loss_step(self, x: NDArray[np.floating], loss: Loss, rows: NDArray[np.integer], step: int,
                  seq: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run training step ``step`` on one batch under the caller's ``loss``; return the loss it gave.

        ``loss(outputs, rows)`` gets the batch's member outputs ``(B, k, K · c)`` in the backend's precision and
        returns the loss and its gradient with respect to them.
        """

    def training_state(self) -> tuple[reference.Weights, reference.AdamState]:
        """Return copies of the weights and AdamW's state in the backend's own precision."""

    def set_training_state(self, weights: reference.Weights, state: reference.AdamState) -> None:
        """Copy in what `training_state` returned, so that training continues exactly."""

    def inference_logits(
        self, x: NDArray[np.floating], seq: NDArray[np.floating] | None = None
    ) -> NDArray[np.floating]:
        """Return the inference logits ``(B, k, K · c)`` (no dropout)."""


class ReferenceTabM:
    """The NumPy reference as a backend (float64): forward, backward and AdamW of `idun.tabm.reference`.

    It always runs the whole batch: ``params.micro_batch`` (a Metal memory option) does not apply. ``targets`` (K) is
    the number of target heads; ``sequence_shape`` ``(S, C)`` the rows' sequences of a network with an encoder.
    """

    def __init__(
        self,
        params: TabMParams,
        n_inputs: int,
        targets: int = 1,
        sequence_shape: tuple[int, int] | None = None,
        bins_shape: tuple[int, int] | None = None,
    ) -> None:
        check_hl_gauss(params, targets)
        if (params.encoder == "none") != (sequence_shape is None):
            raise ValueError(f"encoder {params.encoder!r} with sequences of shape {sequence_shape}")
        if (params.num_emb_type == "none") != (bins_shape is None):
            raise ValueError(f"num_emb_type {params.num_emb_type!r} with bins of shape {bins_shape}")
        self.params, self.targets = params, targets
        inputs = 0 if sequence_shape is None else sequence_shape[1]
        bins = None if bins_shape is None else emb.bins_of([np.arange(bins_shape[1] + 1.0)] * bins_shape[0])
        self._weights = reference.init(params, n_inputs, targets, inputs, bins).astype(np.float64)  # set_weights next
        self._state = reference.adam_state(self._weights)
        self._rate: float | None = None

    def set_learning_rate(self, rate: float | None) -> None:
        """Set the next steps' learning rate (a schedule's); None is ``params.learning_rate``."""
        self._rate = rate

    def set_weights(self, weights: reference.Weights) -> None:
        """Copy ``weights`` in (float64) and reset AdamW."""
        self._weights = weights.astype(np.float64)
        self._state = reference.adam_state(self._weights)

    def weights(self) -> reference.Weights:
        """Return the current weights as float32."""
        return self._weights.astype(np.float32)

    def training_state(self) -> tuple[reference.Weights, reference.AdamState]:
        """Return copies of the weights and AdamW's state (float64)."""
        state = reference.AdamState([m.copy() for m in self._state.m], [v.copy() for v in self._state.v],
                                    self._state.step)  # fmt: skip
        return self._weights.astype(np.float64), state

    def set_training_state(self, weights: reference.Weights, state: reference.AdamState) -> None:
        """Copy in what `training_state` returned (float64)."""
        self._weights = weights.astype(np.float64)
        self._state = reference.AdamState([np.array(m, np.float64) for m in state.m],
                                          [np.array(v, np.float64) for v in state.v], state.step)  # fmt: skip

    def train_step(self, x: NDArray[np.floating], y: NDArray[Any], rows: NDArray[np.integer], step: int,
                   seq: NDArray[np.floating] | None = None,
                   weights: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run one training step; return the (weighted) mean loss."""
        if self.params.objective == "mse":
            y = np.asarray(y, np.float64).reshape(len(rows), self.targets)
        elif self.params.objective == "hl_gauss":
            y = soft_labels(self.params, y)
        else:
            y = reference.label_matrix(y, len(rows), self.targets, self.params.n_classes)
        masks = reference.dropout_masks(self.params, step, np.asarray(rows, np.int64), np.float64)
        logits, cache = reference.forward(self._weights, np.asarray(x, np.float64), masks, _wide(seq))
        loss, d_logits = reference.loss_and_logit_grad(logits, y, self.params.objective, weights)
        reference.adamw_step(self.params, self._weights, reference.backward(self._weights, cache, d_logits),
                             self._state, self._rate)  # fmt: skip
        return loss

    def loss_step(self, x: NDArray[np.floating], loss: Loss, rows: NDArray[np.integer], step: int,
                  seq: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run one training step under the caller's ``loss`` (outputs in float64); return the loss it gave."""
        if self.params.objective == "hl_gauss":
            raise ValueError("an HL-Gauss network takes no loss")
        index = np.asarray(rows, np.int64)
        masks = reference.dropout_masks(self.params, step, index, np.float64)
        outputs, cache = reference.forward(self._weights, np.asarray(x, np.float64), masks, _wide(seq))
        value, d_outputs = reference.host_loss(loss, outputs, index)
        reference.adamw_step(self.params, self._weights, reference.backward(self._weights, cache, d_outputs),
                             self._state, self._rate)  # fmt: skip
        return value

    def predict(self, x: NDArray[np.floating], seq: NDArray[np.floating] | None = None) -> NDArray[np.floating]:
        """Return the expected class of each row (inference): ``(B,)`` for one target, ``(B, K)`` for K > 1."""
        logits = reference.forward(self._weights, np.asarray(x, np.float64), None, _wide(seq))[0]
        return reference.predict(logits, None if self.targets == 1 else self.targets, self.params.objective,
                                 bin_centres(self.params))  # fmt: skip

    def inference_logits(
        self, x: NDArray[np.floating], seq: NDArray[np.floating] | None = None
    ) -> NDArray[np.floating]:
        """Return the inference logits ``(B, k, K · c)`` in float64."""
        return reference.forward(self._weights, np.asarray(x, np.float64), None, _wide(seq))[0]

    def close(self) -> None:
        """Nothing to release."""


def _wide(seq: NDArray[np.floating] | None) -> NDArray[np.float64] | None:
    """Return ``seq`` in float64, or None."""
    return None if seq is None else np.asarray(seq, np.float64)


def check_hl_gauss(params: TabMParams, targets: int) -> None:
    """Raise ``ValueError`` for an HL-Gauss network of more than one target (not supported yet)."""
    if params.objective == "hl_gauss" and targets != 1:
        raise ValueError(f"HL-Gauss trains one target; got {targets}")


def soft_labels(params: TabMParams, y: NDArray[Any]) -> NDArray[np.float64]:
    """Return HL-Gauss's ``(B, n_bins)`` soft labels of the float targets ``y`` (`idun.tabm.hl_gauss.labels`)."""
    return hl_gauss.labels(y, params.n_bins, params.hl_low, params.hl_high, params.hl_sigma)


def bin_centres(params: TabMParams) -> NDArray[np.float64] | None:
    """Return HL-Gauss's bin centres, or None for another objective."""
    if params.objective != "hl_gauss":
        return None
    return hl_gauss.centres(params.n_bins, params.hl_low, params.hl_high)


def make_backend(
    name: str,
    params: TabMParams,
    n_inputs: int,
    max_batch: int,
    targets: int = 1,
    sequence_shape: tuple[int, int] | None = None,
    bins_shape: tuple[int, int] | None = None,
) -> InferenceBackend:
    """Construct the backend called ``name`` (``"metal"``, ``"cuda"`` or ``"reference"``) with ``targets`` target heads.

    ``sequence_shape`` ``(S, C)``: the rows' sequences, for a network with an encoder. ``bins_shape`` ``(F, P)``: the
    embedded inputs and their encoding's components, for a network with piecewise-linear embeddings.
    """
    if name == "metal":
        return MetalTabM(params, n_inputs, max_batch, targets, sequence_shape, bins_shape)
    if name == "cuda":
        return CudaTabM(params, n_inputs, max_batch, targets, sequence_shape, bins_shape)
    if name == "reference":
        return ReferenceTabM(params, n_inputs, targets, sequence_shape, bins_shape)
    raise ValueError(f"unknown backend {name!r}; the backends are {BACKENDS}")


def batches(params: TabMParams, lo: int, n: int, first: int = 0) -> Iterator[tuple[int, int, NDArray[np.int64]]]:
    """Yield (epoch, global step, absolute rows) of each batch from epoch ``first`` on; an epoch's last may be short."""
    steps_per_epoch = -(-n // params.batch_size)
    for epoch in range(first, params.epochs):
        order = epoch_order(params.seed, epoch, n)
        for i in range(steps_per_epoch):
            rows = lo + order[i * params.batch_size : (i + 1) * params.batch_size]
            yield epoch, epoch * steps_per_epoch + i, rows


def _plan(params: TabMParams, lo: int, hi: int, first: int, custom: BatchPlan | None,
          steps: int) -> Iterator[tuple[int, int, NDArray[np.int64]]]:  # fmt: skip
    """Yield (epoch, global step, absolute rows) of each batch from epoch ``first`` on.

    The seeded shuffle (`batches`), or with ``custom`` the batches it gives for each epoch, checked: 1-D integer
    arrays of 1 .. ``params.batch_size`` rows inside ``[lo, hi)``, at least one per epoch (``ValueError`` otherwise).
    Their global steps count on from ``steps``, the steps taken before epoch ``first``.
    """
    if custom is None:
        yield from batches(params, lo, hi - lo, first)
        return
    for epoch in range(first, params.epochs):
        plan = [np.asarray(rows) for rows in custom(epoch)]
        if not plan:
            raise ValueError(f"batches({epoch}) gave no batch")
        for index in plan:
            if (index.ndim != 1 or not 1 <= index.size <= params.batch_size
                    or not np.issubdtype(index.dtype, np.integer)):  # fmt: skip
                raise ValueError(f"batches({epoch}) must give 1-D integer arrays of 1..{params.batch_size} rows "
                                 f"(params.batch_size); got a {index.dtype} array of shape {index.shape}")  # fmt: skip
            if index.min() < lo or index.max() >= hi:
                raise ValueError(f"batches({epoch}) names rows outside the training rows [{lo}, {hi})")
            yield epoch, steps, index.astype(np.int64)
            steps += 1


@dataclass
class TabMModel:
    """A trained TabM: parameters, weights, the inputs it expects, the rows it saw.

    ``multi_target`` and `targets`: experimental, not covered by the documentation. ``sequence_inputs``: with a
    temporal encoder, the inputs per step of the sequences it reads (`idun.tabm.Sequences.inputs`); 0 without one.
    """

    params: TabMParams
    weights: reference.Weights
    n_inputs: int
    train_rows: tuple[int, int]
    report: FitReport
    multi_target: bool = False
    sequence_inputs: int = 0

    def __post_init__(self) -> None:
        outputs, each = self.weights.head.shape[-1], self.params.outputs
        if outputs % each != 0 or not (self.multi_target or outputs == each):
            raise ValueError(f"a head of {outputs} outputs is not the targets' {each} outputs each")
        encoder = self.weights.encoder
        if (self.params.encoder == "none") != (encoder is None) or (encoder is None) != (self.sequence_inputs == 0):
            raise ValueError(f"encoder {self.params.encoder!r}, sequence_inputs {self.sequence_inputs} and the weights "
                             "do not agree")  # fmt: skip
        if encoder is not None and encoder.embed.shape[1] != self.sequence_inputs:
            raise ValueError(f"the encoder reads {encoder.embed.shape[1]} inputs per step, not {self.sequence_inputs}")
        embedding, bins = self.weights.embedding, self.weights.bins
        if (self.params.num_emb_type == "none") != (embedding is None) or (embedding is None) != (bins is None):
            raise ValueError(f"num_emb_type {self.params.num_emb_type!r} and the weights' embeddings do not agree")
        if bins is not None and not bins.scale.shape[0] <= self.n_inputs:
            raise ValueError(f"embeddings of {bins.scale.shape[0]} inputs; the model has {self.n_inputs}")

    @property
    def targets(self) -> int:
        """K: the number of targets the model predicts."""
        return int(self.weights.head.shape[-1] // self.params.outputs)

    def _chunks(
        self, data: Data, rows: slice, backend: str, chunk: int,
        each: Callable[[InferenceBackend, NDArray[np.float32], NDArray[np.float32] | None], NDArray[np.floating]],
        width: int, sequences: Sequences | None,
    ) -> NDArray[np.float32]:  # fmt: skip
        """Return ``each(net, inputs, sequences)`` of every ``chunk`` rows of ``rows`` (a contiguous slice).

        As ``(n, width)``; ``sequences`` the rows' sequences of a model with an encoder.
        """
        if inputs.n_inputs(data) != self.n_inputs:
            raise ValueError(f"the data has {inputs.n_inputs(data)} inputs; the model expects {self.n_inputs}")
        shape = check_sequences(self.params, data, sequences)
        if shape is not None and shape[1] != self.sequence_inputs:
            raise ValueError(f"the sequences have {shape[1]} inputs per step; the model reads {self.sequence_inputs}")
        lo, hi, step = rows.indices(data.n_rows)
        if step != 1:
            raise ValueError("rows must be a contiguous slice")
        bins = self.weights.bins
        net = make_backend(backend, self.params, self.n_inputs, chunk, self.targets, shape,
                           None if bins is None else bins.scale.shape)  # fmt: skip
        try:
            net.set_weights(self.weights)
            out = np.empty((max(hi - lo, 0), width), np.float32)
            for start in range(lo, hi, chunk):
                stop = min(start + chunk, hi)
                batch = np.arange(start, stop)
                seq = None if sequences is None else sequences.gather(batch)
                part = each(net, inputs.dense_inputs(data, batch), seq)
                out[start - lo : stop - lo] = np.reshape(part, (stop - start, width))
            return out
        finally:
            net.close()

    def predict(
        self, data: Data, *, rows: slice, backend: str, chunk: int = 4096, sequences: Sequences | None = None
    ) -> NDArray[np.float32]:
        """Return the expected class of each row in ``rows`` (a contiguous slice), ``chunk`` rows per pass.

        ``(n,)``. ``sequences``: the rows' sequences, for (and only for) a model with a temporal encoder.
        """
        out = self._chunks(data, rows, backend, chunk, lambda net, x, seq: net.predict(x, seq), self.targets,
                           sequences)  # fmt: skip
        return out if self.multi_target else out[:, 0]

    def predict_proba(
        self, data: Data, *, rows: slice, backend: str, chunk: int = 4096, sequences: Sequences | None = None
    ) -> NDArray[np.float32]:
        """Return each row's class probabilities ``(n, n_classes)``: the members' mean softmax (one target only)."""
        if self.targets != 1 or self.params.objective != "cross_entropy":
            raise ValueError("predict_proba needs a single-target classification model")

        def proba(net: InferenceBackend, x: NDArray[np.float32], seq: NDArray[np.float32] | None) -> NDArray[Any]:
            return softmax(net.inference_logits(x, seq)).mean(axis=1)

        return self._chunks(data, rows, backend, chunk, proba, self.params.n_classes, sequences)

    def save(self, path: str | Path) -> None:
        """Write the model to ``path`` (an uncompressed ``.npz`` without object arrays)."""
        params, defaults = asdict(self.params), asdict(TabMParams())
        for fields in (ENCODER_FIELDS, EMBEDDING_FIELDS):  # without the feature, as earlier versions wrote
            if all(params[name] == defaults[name] for name in fields):
                for name in fields:
                    del params[name]
        meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "params": params, "n_inputs": self.n_inputs,
                "train_rows": list(self.train_rows), "report": asdict(self.report)}  # fmt: skip
        if self.multi_target:
            meta.update(format_version=MULTI_TARGET_VERSION, multi_target=True)
        if self.sequence_inputs:
            meta.update(format_version=ENCODER_VERSION, multi_target=self.multi_target,
                        sequence_inputs=self.sequence_inputs)  # fmt: skip
        arrays = {f"p{i:02d}": a for i, a in enumerate(self.weights.arrays())}
        if self.weights.bins is not None:
            meta.update(format_version=EMBEDDING_VERSION, multi_target=self.multi_target)
            bins = self.weights.bins
            arrays.update(bins_scale=bins.scale, bins_shift=bins.shift, bins_single=bins.single.astype(np.uint8))
        with Path(path).open("wb") as handle:
            np.savez(handle, allow_pickle=False, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a model written by `save`; anything else raises ``ValueError``."""
        with np.load(path, allow_pickle=False) as archive:
            meta: dict[str, Any] = json.loads(archive["meta"].tobytes())
            versions = (FORMAT_VERSION, MULTI_TARGET_VERSION, ENCODER_VERSION, EMBEDDING_VERSION)
            if meta.get("format") != FORMAT or meta.get("format_version") not in versions:
                raise ValueError(f"{path} is not a {FORMAT} v1 to v4 file")
            arrays = [archive[f"p{i:02d}"] for i in range(sum(name.startswith("p") for name in archive.files))]
            bins = None
            if "bins_scale" in archive.files:
                bins = emb.Bins(archive["bins_scale"], archive["bins_shift"], archive["bins_single"].astype(bool))
        params = TabMParams(**meta["params"])
        weights = weights_from_arrays(
            arrays, params.blocks, bins
        )  # a v3 file's end with the encoder's, a v4's embeddings
        multi = meta["format_version"] == MULTI_TARGET_VERSION or bool(meta.get("multi_target", False))
        return cls(params, weights, int(meta["n_inputs"]), tuple(meta["train_rows"]), FitReport(**meta["report"]),
                   multi_target=multi, sequence_inputs=int(meta.get("sequence_inputs", 0)))  # fmt: skip


def weights_from_arrays(
    arrays: list[NDArray[np.floating]], blocks: int, bins: emb.Bins | None = None
) -> reference.Weights:
    """Return the `reference.Weights` of arrays in `reference.Weights.arrays` order.

    The encoder's last, if any; or, with ``bins``, the embeddings'.
    """
    n, tail = blocks, arrays[4 * blocks + 2 :]
    encoder = EncoderWeights(*tail) if tail and bins is None else None
    embedding = emb.EmbeddingWeights(*tail) if bins is not None else None
    return reference.Weights(arrays[:n], arrays[n : 2 * n], arrays[2 * n : 3 * n], arrays[3 * n : 4 * n],
                             arrays[4 * n], arrays[4 * n + 1], encoder, embedding, bins)  # fmt: skip


# -- checkpoints ------------------------------------------------------------------------------------------------------


def _write_checkpoint(
    path: Path, fit: dict[str, Any], epochs: int, net: InferenceBackend, report: FitReport, seconds: float
) -> None:
    """Write the state after ``epochs`` epochs (``seconds`` of training so far) to ``path`` atomically.

    A temporary file, synced, then renamed: a stop in the middle leaves the previous checkpoint.
    """
    report.seconds = seconds
    weights, state = net.training_state()
    meta = {"format": CHECKPOINT_FORMAT, "format_version": CHECKPOINT_VERSION, "fit": fit, "epochs": epochs,
            "adam_step": state.step, "report": asdict(report)}  # fmt: skip
    arrays = {f"{kind}{i:02d}": a for kind, group in (("w", weights.arrays()), ("m", state.m), ("v", state.v))
              for i, a in enumerate(group)}  # fmt: skip
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez(handle, allow_pickle=False, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _read_checkpoint(
    path: Path, fit: dict[str, Any], blocks: int, bins: emb.Bins | None = None
) -> tuple[reference.Weights, reference.AdamState, int, FitReport]:
    """Return a checkpoint's weights, AdamW state, epochs done and report; ``ValueError`` unless it is this fit's."""
    with np.load(path, allow_pickle=False) as archive:
        if "meta" not in archive.files:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} file")
        meta: dict[str, Any] = json.loads(archive["meta"].tobytes())
        if meta.get("format") != CHECKPOINT_FORMAT or meta.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} v{CHECKPOINT_VERSION} file")
        if meta["fit"] != fit:
            changed = sorted(k for k in fit.keys() | meta["fit"].keys() if fit.get(k) != meta["fit"].get(k))
            raise ValueError(f"{path} is the checkpoint of another fit (it differs in {', '.join(changed)})")
        n = sum(name.startswith("w") for name in archive.files)
        group = {kind: [archive[f"{kind}{i:02d}"] for i in range(n)] for kind in "wmv"}
    state = reference.AdamState(group["m"], group["v"], int(meta["adam_step"]))
    return weights_from_arrays(group["w"], blocks, bins), state, int(meta["epochs"]), FitReport(**meta["report"])


def _begin(
    net: InferenceBackend, path: Path | None, fit: dict[str, Any], params: TabMParams, weights: reference.Weights
) -> tuple[int, FitReport, int]:
    """Load ``path``'s checkpoint into ``net`` if it exists, else ``weights``.

    Return the first epoch, the report so far and the training steps taken.
    """
    if path is not None and path.is_file():
        saved, state, first, report = _read_checkpoint(path, json.loads(json.dumps(fit)), params.blocks,
                                                       weights.bins)  # fmt: skip
        net.set_training_state(saved, state)
        return first, report, state.step
    net.set_weights(weights)
    return 0, FitReport(), 0


def check_sequences(params: TabMParams, data: Data, sequences: Sequences | None) -> tuple[int, int] | None:
    """Return the sequences' ``(S, C)``, or None; ``ValueError`` unless they come with an encoder and fit ``data``."""
    if (params.encoder == "none") != (sequences is None):
        raise ValueError(f"encoder {params.encoder!r} takes " + ("no sequences" if sequences is not None else
                         "the rows' sequences (idun.tabm.Sequences)"))  # fmt: skip
    if sequences is None:
        return None
    if sequences.n_rows != data.n_rows:
        raise ValueError(f"the sequences have {sequences.n_rows} rows; the data has {data.n_rows}")
    return sequences.steps, sequences.inputs


def _batch_labels(
    params: TabMParams, data: Data, rows: NDArray[np.int64], classes: NDArray[np.uint8] | None,
    targets: NDArray[np.float32] | None, n_targets: int,
) -> NDArray[Any]:  # fmt: skip
    """Return a batch's labels: its float targets, its classes, or a `MatrixData`'s own; ``(B,)`` for one target."""
    y: NDArray[Any]
    if targets is not None:
        y = np.asarray(targets)[rows]  # finite (checked before training)
    elif classes is None:
        return inputs.labels(data, rows)  # type: ignore[arg-type]  # a MatrixData (checked before training)
    else:
        y = inputs.class_labels(classes, rows, params.n_classes)
    return y[:, 0] if n_targets == 1 else y


def _label_width(
    data: Data, objective: str, classes: NDArray[np.uint8] | None, targets: NDArray[np.float32] | None, lo: int, hi: int
) -> int:
    """Return K, the targets of the labels given (1 for a `MatrixData`); ``ValueError`` unless they fit the data.

    Float targets are checked here on every training row; classes when their batch is gathered.
    """
    inputs.check_label_source(data, objective, classes, targets)
    if targets is not None:
        return int(inputs.checked_targets(targets, data.n_rows, lo, hi).shape[1])
    if classes is None:
        return 1
    array = np.asarray(classes)
    if array.ndim != 2 or array.shape[0] != data.n_rows or array.shape[1] < 1 or array.dtype != np.uint8:
        raise ValueError(f"classes must be an ({data.n_rows}, K) uint8 array with K >= 1, got a {array.dtype} array "
                         f"of shape {array.shape}")  # fmt: skip
    return int(array.shape[1])


def _fit_targets(
    params: TabMParams, data: Data, loss: Loss | None, classes: NDArray[np.uint8] | None,
    targets: NDArray[np.float32] | None, lo: int, hi: int,
) -> int:  # fmt: skip
    """Return K, the targets of a fit: the labels' (`_label_width`), or 1 under a loss, which holds its own labels.

    ``ValueError`` for labels that do not fit, for labels beside a loss, and for an HL-Gauss network with a loss or
    with more than one target.
    """
    if loss is None:
        n_targets = _label_width(data, params.objective, classes, targets, lo, hi)
        check_hl_gauss(params, n_targets)
        return n_targets
    if classes is not None or targets is not None or params.objective == "hl_gauss":
        raise ValueError('a loss holds its own labels: pass no classes or targets, with objective "mse" (one output '
                         'per member) or "cross_entropy" (class logits)')  # fmt: skip
    return 1


def _check_validation(
    params: TabMParams,
    valid: Validation | None,
    patience: int | None,
    n_targets: int,
    sequences: Sequences | None,
    checkpoint: str | Path | None,
) -> None:
    """Refuse a validation set or a patience `train_tabm` cannot use (one target, no sequences, no checkpoint)."""
    if patience is not None and (valid is None or patience < 1):
        raise ValueError(f"patience needs a validation set and a positive count, got {patience!r}")
    if valid is None:
        return
    valid.bounds()
    if n_targets != 1 or sequences is not None or checkpoint is not None:
        raise ValueError("a validation set scores one target without sequences, and not with a checkpoint (yet)")
    if valid.data.n_classes != (params.n_classes if params.objective == "cross_entropy" else 1):
        raise ValueError(f"the validation set has {valid.data.n_classes} classes, the network {params.n_classes}")


def _schedule(params: TabMParams, n: int, batches: BatchPlan | None) -> Callable[[int], float | None] | None:
    """Return each step's learning rate under ``params``' schedule, or None for the constant rate."""
    if params.lr_schedule == "constant" and params.lr_warmup_steps == 0:
        return None
    if batches is not None:
        raise ValueError("a learning-rate schedule counts the seeded plan's steps: not with batches=")
    total = params.epochs * -(-n // params.batch_size)
    return lambda step: scheduled_rate(params, step, total)


@dataclass
class _Keeper:
    """Scores the validation rows after each epoch, keeps the best epoch's weights, says when patience runs out."""

    net: InferenceBackend
    params: TabMParams
    valid: Validation
    chunk: int
    report: FitReport
    patience: int | None
    best: reference.Weights | None = None

    def after(self, epoch: int) -> bool:
        """Score epoch ``epoch``; return whether training stops after it."""
        report = self.report
        report.valid_score.append(_validation_score(self.net, self.params, self.valid, self.chunk))
        if report.best_epoch is None or report.valid_score[-1] < report.valid_score[report.best_epoch]:
            report.best_epoch, self.best = epoch, self.net.weights()
            return False
        return self.patience is not None and epoch - report.best_epoch >= self.patience


def _validation_score(net: InferenceBackend, params: TabMParams, valid: Validation, chunk: int) -> float:
    """Score the validation rows: the error rate of the members' mean softmax, or the prediction's squared error."""
    lo, hi = valid.bounds()
    parts = []
    for start in range(lo, hi, chunk):
        x = inputs.dense_inputs(valid.data, np.arange(start, min(start + chunk, hi)))
        if params.objective == "cross_entropy":
            parts.append(softmax(net.inference_logits(x, None)).mean(axis=1))
        else:
            parts.append(np.asarray(net.predict(x, None), np.float64).reshape(-1))
    return score(np.concatenate(parts), valid.data.y[lo:hi], valid.data.n_classes)


def _step(net: InferenceBackend, loss: Loss | None, x: NDArray[np.float32], y: NDArray[Any] | None,
          rows: NDArray[np.int64], step: int, seq: NDArray[np.float32] | None,
          weights: NDArray[np.float32] | None = None) -> float:  # fmt: skip
    """Run training step ``step`` on one batch: under ``loss``, or the built-in loss on the labels ``y`` (weighted)."""
    if loss is not None:
        return net.loss_step(x, loss, rows, step, seq)
    if y is None:
        raise ValueError("a training step needs the batch's labels or a loss")
    if weights is None:
        return net.train_step(x, y, rows, step, seq)
    return net.train_step(x, y, rows, step, seq, weights)


def _warn_targets(n_targets: int) -> None:
    """Warn that K > 1 heads on one trunk cost accuracy (`MultiTargetQualityWarning`)."""
    if n_targets > 1:
        warnings.warn(
            f"train_tabm trains {n_targets} target heads on one trunk (experimental): less accurate than separate "
            "networks, and the heads' outputs come out nearly alike",
            MultiTargetQualityWarning,
            stacklevel=3,
        )


def _network(
    backend: str, params: TabMParams, data: Data, rows: tuple[int, int], most: int, n_targets: int,
    shape: tuple[int, int] | None, embedded: int | None,
) -> tuple[InferenceBackend, emb.Bins | None]:  # fmt: skip
    """Return the fit's backend for batches of up to ``most`` rows, and its embeddings' bins (or None)."""
    bins = _embedding_bins(params, data, *rows, embedded)
    net = make_backend(backend, params, inputs.n_inputs(data), most, n_targets, shape,
                       None if bins is None else bins.scale.shape)  # fmt: skip
    return net, bins


def _embedding_bins(params: TabMParams, data: Data, lo: int, hi: int, embedded: int | None) -> emb.Bins | None:
    """Return the embeddings' bins of the first ``embedded`` inputs (all by default), or None without embeddings.

    Fitted on the training rows, or `EMB_SAMPLE_ROWS` of them drawn with the seed when there are more.
    """
    if params.num_emb_type == "none":
        if embedded is not None:
            raise ValueError(f"embedded={embedded!r} needs params.num_emb_type 'pwl'")
        return None
    n_inputs = inputs.n_inputs(data)
    count = n_inputs if embedded is None else embedded
    if not 1 <= count <= n_inputs:
        raise ValueError(f"embedded must be 1..{n_inputs} inputs (the first ones), got {embedded!r}")
    rows = np.arange(lo, hi)
    if rows.size > EMB_SAMPLE_ROWS:
        draw = np.random.Generator(np.random.PCG64(np.random.SeedSequence([params.seed, emb.EMBED_SALT, 2])))
        rows = np.sort(draw.choice(rows, EMB_SAMPLE_ROWS, replace=False))
    return emb.fit_bins(inputs.dense_inputs(data, rows)[:, :count], params.num_emb_n_bins)


def _bins_digest(bins: emb.Bins) -> str:
    """SHA-256 of the bins' arrays: a checkpoint's fit records which bins it trained with."""
    h = hashlib.sha256()
    for array in (bins.scale, bins.shift, bins.single.astype(np.uint8)):
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def _row_weights(weight: NDArray[Any] | None, lo: int, hi: int, loss: Loss | None) -> NDArray[np.float32] | None:
    """Return the training rows' weights checked: one per row, finite, non-negative, not all zero; not with a loss."""
    if weight is None:
        return None
    if loss is not None:
        raise ValueError("a loss weighs its own rows: pass no weight with loss=")
    return checked_weights(weight, hi - lo)


def _end_epoch(
    net: InferenceBackend, report: FitReport, epoch: int, losses: list[float], seconds: float, total: float,
    checkpoint: tuple[Path | None, dict[str, Any]], on_epoch: Callable[[int, float, float], None] | None,
    on_weights: Callable[[int, reference.Weights], None] | None,
) -> None:  # fmt: skip
    """Record epoch ``epoch`` (its steps' mean loss, its ``seconds``), write the checkpoint, call the callbacks.

    ``checkpoint``: the file (or None) and the fit's record; ``total``: the fit's training seconds so far.
    """
    report.epoch_loss.append(float(np.mean(losses)))
    report.epoch_seconds.append(seconds)
    path, fit = checkpoint
    if path is not None:
        _write_checkpoint(path, fit, epoch + 1, net, report, total)
    if on_epoch is not None:
        on_epoch(epoch, report.epoch_loss[-1], report.epoch_seconds[-1])
    if on_weights is not None:
        on_weights(epoch, net.weights())


def train_tabm(
    params: TabMParams,
    data: Data,
    *,
    rows: slice,
    backend: str,
    on_epoch: Callable[[int, float, float], None] | None = None,
    classes: NDArray[np.uint8] | None = None,
    on_weights: Callable[[int, reference.Weights], None] | None = None,
    targets: NDArray[np.float32] | None = None,
    sequences: Sequences | None = None,
    checkpoint: str | Path | None = None,
    loss: Loss | None = None,
    batches: BatchPlan | None = None,
    valid: Validation | None = None,
    patience: int | None = None,
    weight: NDArray[np.floating] | None = None,
    embedded: int | None = None,
) -> TabMModel:
    """Train TabM on the contiguous training rows ``rows`` of ``data``; ``on_epoch(epoch, loss, seconds)`` reports.

    ``classes``: the labels of a packed dataset, an ``(n_rows, 1)`` uint8 class array (`inputs.class_array`), every
    training row's class below ``params.n_classes``; the model predicts ``(n,)`` (more than one column is
    experimental, not covered by the documentation). ``targets``: instead, for ``params.objective`` ``"mse"`` or
    ``"hl_gauss"``, a packed dataset's regression target, an ``(n_rows, 1)`` float32 target array
    (`inputs.target_array`), finite on every training row (checked before any training). With ``"mse"`` the
    model predicts the members' mean; with ``"hl_gauss"`` it trains on the target's soft labels
    (`idun.tabm.hl_gauss`) and predicts the expected bin centre. A `MatrixData` carries its own labels and takes
    neither.

    ``on_weights(epoch, weights)``: after each epoch, that moment's weights (float32). With a constant learning rate
    (the default ``lr_schedule``) an epoch's batch order depends only on the seed and the epoch, so the weights after
    epoch ``e`` are those of the same fit with ``epochs = e + 1``, bit for bit: one run gives the networks of every
    shorter training.

    ``sequences``: for a network with a temporal encoder (``params.encoder`` ``"gru"``), every row's sequence
    (`idun.tabm.Sequences`, the same rows as ``data``); each batch's sequences are gathered with its inputs. Pass the
    same sequences (or ones built the same way) to `TabMModel.predict`.

    ``checkpoint``: a file for the training state. After each epoch the weights, AdamW's moments and step count, the
    epochs done and the report so far are written there (atomically: a temporary file, synced, then renamed), before
    ``on_epoch`` and ``on_weights`` are called. If the file exists when the call starts, training resumes after its last
    epoch, and the result is the uninterrupted fit's, bit for bit (the learning rate is constant, an epoch's order and
    the dropout masks are counter-based; a test). So a fit stopped at any moment, or by an exception from ``on_epoch``,
    loses at most the epoch it was in. The fit must be the same: the parameters, the rows, the inputs, the targets, the
    sequences' shape and the backend are recorded and checked (``ValueError`` if they differ); the data itself is not,
    so pass the same data. The file stays after the fit; remove it when the model is saved.

    ``loss``: a training loss of the caller's in place of the built-in one. ``loss(outputs, rows)`` is called once per
    step on the host with the batch's member outputs ``(B, k, c)`` (float64 on the reference, float32 on Metal) and
    its absolute row indices ``(B,)``, and returns the loss (a float, reported per epoch as their mean) and its
    gradient with respect to ``outputs``, in their shape. The loss holds whatever it needs, indexed by row: labels,
    groups, weights; pass no ``classes`` or ``targets`` (a `MatrixData`'s own labels are not read). The gradient is
    used as returned, so a mean over the batch carries its ``1 / (B · k)``. ``params.objective`` still declares the
    outputs and the prediction: ``"mse"`` is one output per member (c = 1) and the model predicts the members' mean;
    ``"cross_entropy"`` is ``n_classes`` logits and the model predicts the expected class; ``"hl_gauss"`` takes no
    loss. One target. A gradient of another shape, or a loss or a gradient that is not finite, raises ``ValueError``.
    On Metal a step is then two command buffers with the host between them, and a micro-batched fit
    (``params.micro_batch``) runs each micro-batch's forward pass twice.

    ``batches``: a batch plan of the caller's in place of the seeded shuffle. ``batches(epoch)`` returns the epoch's
    batches in order, each a 1-D integer array of 1 .. ``params.batch_size`` absolute row indices inside ``rows``
    (any number of batches of any such sizes; rows may repeat or be left out); the checks raise ``ValueError``. Each
    batch is one AdamW step, and the dropout masks go by the count of steps taken. It must be a pure function of the
    epoch: a resumed fit calls it for the epochs left. On Metal the buffers are sized for ``params.batch_size`` rows.
    A checkpoint records that a loss or a plan was given, not which: pass the same ones when resuming.

    ``weight``: one finite, non-negative weight per training row (``(hi − lo,)``, row ``lo + i`` at ``i``; not all
    zero). Each step's loss is then its batch's weighted mean, ``Σ w l / Σ w`` over the batch's rows (and the mean
    over the targets), in place of the plain mean; the batches and the dropout are unchanged, a row of weight 0 only
    leaves its batch's loss, and unit weights give the unweighted fit bit for bit on a GPU. A batch whose rows all
    weigh 0 has loss 0 and gradient 0 (AdamW still steps on its moments). The reported losses are the weighted
    means; a validation set's score is not weighted. Not with ``loss`` (which weighs its own rows); a
    checkpoint records that weights were given, not which.
    """
    lo, hi, step = rows.indices(data.n_rows)
    if step != 1 or hi <= lo:
        raise ValueError("rows must be a non-empty contiguous slice")
    n_targets = _fit_targets(params, data, loss, classes, targets, lo, hi)
    weights = _row_weights(weight, lo, hi, loss)
    shape = check_sequences(params, data, sequences)
    sequence_inputs = 0 if shape is None else shape[1]
    _warn_targets(n_targets)
    _check_validation(params, valid, patience, n_targets, sequences, checkpoint)
    n_inputs = inputs.n_inputs(data)
    most = params.batch_size if batches is not None else min(params.batch_size, hi - lo)
    net, bins = _network(backend, params, data, (lo, hi), most, n_targets, shape, embedded)
    begin = epoch_begin = time.perf_counter()
    losses: list[float] = []
    path = None if checkpoint is None else Path(checkpoint)
    fit = {"params": asdict(params), "n_rows": data.n_rows, "n_inputs": n_inputs, "train_rows": [lo, hi],
           "targets": n_targets, "sequences": None if shape is None else list(shape), "backend": backend,
           **{name: True for name, hook in (("loss", loss), ("batches", batches), ("weight", weights))
              if hook is not None},
           **({} if bins is None else {"embedded": int(bins.scale.shape[0]), "bins": _bins_digest(bins)})}  # fmt: skip
    try:
        first_epoch, report, steps = _begin(
            net, path, fit, params, reference.init(params, n_inputs, n_targets, sequence_inputs, bins)
        )
        spent = report.seconds
        keeper = None if valid is None else _Keeper(net, params, valid, most, report, patience)
        with ThreadPoolExecutor(max_workers=1) as pool:

            def gather(
                rows: NDArray[np.int64],
            ) -> tuple[NDArray[np.float32], NDArray[Any] | None, NDArray[np.float32] | None]:
                y = None if loss is not None else _batch_labels(params, data, rows, classes, targets, n_targets)
                seq = None if sequences is None else sequences.gather(rows)
                return inputs.dense_inputs(data, rows), y, seq

            plan = _plan(params, lo, hi, first_epoch, batches, steps)
            schedule = _schedule(params, hi - lo, batches)
            current = next(plan, None)
            pending = pool.submit(gather, current[2]) if current is not None else None
            while current is not None and pending is not None:
                epoch, global_step, batch_rows = current
                upcoming = next(plan, None)
                x, y, seq = pending.result()
                if upcoming is not None:
                    pending = pool.submit(gather, upcoming[2])  # gathered while the GPU runs this step
                if schedule is not None:
                    net.set_learning_rate(schedule(global_step))
                losses.append(_step(net, loss, x, y, batch_rows, global_step, seq,
                                    None if weights is None else weights[batch_rows - lo]))  # fmt: skip
                if upcoming is None or upcoming[0] != epoch:
                    now = time.perf_counter()
                    _end_epoch(net, report, epoch, losses, now - epoch_begin, spent + now - begin,
                               (path, fit), on_epoch, on_weights)  # fmt: skip
                    losses, epoch_begin = [], now
                    if keeper is not None and keeper.after(epoch):
                        break  # no better epoch for `patience` epochs: the best one's weights
                current = upcoming
        report.seconds = spent + time.perf_counter() - begin
        kept = net.weights() if keeper is None or keeper.best is None else keeper.best
        return TabMModel(params, kept, n_inputs, (lo, hi), report, multi_target=n_targets > 1,
                         sequence_inputs=sequence_inputs)  # fmt: skip
    finally:
        net.close()
