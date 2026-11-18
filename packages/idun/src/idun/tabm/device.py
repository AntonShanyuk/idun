"""TabM on a GPU device: the network of `idun.tabm.reference` as kernels driven through `idun.device.Device`.

A `DeviceTabM` owns padded parameter, gradient, AdamW and activation buffers for batches of up to ``max_batch`` rows on
its device (`idun.tabm.metal` builds one on Metal). It names its kernels in `TABLE` and reaches their buffers from the
host through the device's ``read``, ``write`` and ``fill`` only, so it runs the same passes on a GPU with memory of its
own. Widths are padded to multiples of 64 (the GEMM tiles) and the batch so that ``rows · k`` is one too; padded
entries are zero and stay zero. A training step (forward, loss, backward, AdamW) is one batch of passes. Determinism:
fixed-order reductions and counter-based dropout masks keyed by the absolute row, so two runs give identical bits.

With ``TabMParams.micro_batch`` > 0 a batch runs as micro-batches of that many rows (the last may be smaller): the
activation buffers hold one micro-batch, each micro-batch has its own small input, logit and loss buffers, and the step
encodes forward and backward of micro-batch 0, 1, ... in order, the first storing every parameter gradient and each
later one adding to it, then one AdamW step. Every row's forward and backward is the whole batch's (the logit gradient
keeps the scale 1 / (B · k), the dropout masks are keyed by the absolute row); only the batch sums of the gradients
are added in another (fixed) order.

A step under a loss of the caller's (`loss_step`) is two command buffers with the host between them: the forward
passes, the loss and its output gradient on the host, then the backward passes from that gradient and AdamW. The
activation buffers hold one micro-batch, so a micro-batched step encodes each micro-batch's forward pass again before
its backward pass (the dropout masks are counter-based, so the two passes are equal).

``targets`` other than 1: experimental, not covered by the documentation. The GPU keeps the head (and its gradient and
moments) transposed; `set_weights` and `weights` convert from and to `reference.Weights`.

HL-Gauss (``objective="hl_gauss"``, one target): the head has ``n_bins`` outputs, which the head kernels compute as
``n_bins / 8`` groups of 8 (`HEAD_GROUP`); the labels are the soft labels of `idun.tabm.hl_gauss`, ``(B, n_bins)``
float32, built on the host from the float targets; the loss is ``hl_gauss_ce`` and the prediction ``predict_centre``.

A temporal encoder (``params.encoder`` ``"gru"``, `idun.tabm.encoder`) runs first in each micro-batch's forward pass:
``seq_forward`` reads the micro-batch's sequences, ``(rows_pad · S, ld_in)`` float32 (pair ``b · S + s``, its inputs
padded to 64), and writes the last GRU state into the first ``H`` columns of the batch inputs, whose snapshot inputs
follow. Its backward pass comes after block 0's: the input gradient ``dT_0 · W_0[:, :H]`` (``seq_input_grad``) summed
over the members (``seq_member_sum``), back through time (``seq_backward``), then the weight gradients as transposed
GEMMs over `SEQ_CHUNKS` chunks of pairs and the bias gradients as column sums over the same chunks, each added up in
chunk order. The encoder computes in float32 whatever the activation storage.
"""

import ctypes
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun import rng
from idun.device import Buffer, Device, Signature
from idun.nn.device import BK, METAL_GEMM, TILE, AdamParams, GemmLaunch, GemmParams, round_up
from idun.tabm import hl_gauss, reference
from idun.tabm.embeddings import Bins, EmbeddingWeights
from idun.tabm.encoder import EncoderWeights
from idun.tabm.params import TabMParams

MAX_CLASSES = 8
HEAD_GROUP = 8  # HL-Gauss's head runs its bins in groups of this many outputs (MAX_CLASSES in tabm.metal)
STORAGES = ("fp32", "bf16")  # TabMParams.storage: the activations' bytes are 4 or 2
CHUNKS = 32  # the two-stage batch sums: partials per chunk of rows, then a fixed-order sum
HBB_ROWS = 4  # batch rows per thread of head_block_backward (HBB_ROWS in tabm.metal)
TARGETS_PER_GROUP = 16  # head_forward, head_backward: SIMD-groups of different targets per threadgroup
SEQ_ROWS = 4  # rows per threadgroup of seq_forward and seq_backward (SEQ_ROWS in tabm.metal)
SEQ_CHUNKS = 32  # the encoder's gradient sums: partials per chunk of (row, step) pairs, then a fixed-order sum
ENCODER_KERNELS = ("seq_forward", "seq_input_grad", "seq_member_sum", "seq_backward", "seq_weight_grad",
                   "seq_column_partials")  # fmt: skip
EMBEDDING_KERNELS = ("emb_forward", "emb_member_sum", "emb_weight_grad")  # and seq_input_grad, the encoder's


class NetParams(ctypes.Structure):
    """Mirror of ``NetParams`` in tabm.metal."""

    _fields_ = [
        *(
            (n, ctypes.c_uint32)
            for n in ("batch", "rows_pad", "members", "width", "ldw", "classes", "key", "threshold")
        ),
        ("scale", ctypes.c_float),
        ("inv_count", ctypes.c_float),
        ("din", ctypes.c_uint32),
        ("chunk", ctypes.c_uint32),
        ("chunks", ctypes.c_uint32),
        ("accumulate", ctypes.c_uint32),
        ("targets", ctypes.c_uint32),
        ("low", ctypes.c_float),
        ("bin_width", ctypes.c_float),
    ]


class FusedParams(ctypes.Structure):
    """Mirror of ``FusedParams`` in tabm.metal: the GEMM's and the output stage's parameters."""

    _fields_ = [("g", GemmParams), ("n", NetParams)]


class SeqParams(ctypes.Structure):
    """Mirror of ``SeqParams`` in tabm.metal."""

    _fields_ = [(n, ctypes.c_uint32) for n in ("batch", "rows_pad", "steps", "inputs", "ld_in", "embed", "ld_e",
                                                "hidden", "ld_h", "ld_g", "members", "ldd")]  # fmt: skip


class EmbParams(ctypes.Structure):
    """Mirror of ``EmbParams`` in tabm.metal."""

    _fields_ = [(n, ctypes.c_uint32) for n in ("batch", "rows_pad", "features", "components", "dim", "ld_raw", "ldd",
                                                "ld_e", "members", "accumulate")]  # fmt: skip


class MemberGemmParams(ctypes.Structure):
    """Mirror of ``MemberGemmParams`` in tabm.metal: a GEMM per member, member j's operands ``j · step`` further on."""

    _fields_ = [("g", GemmParams), *((n, ctypes.c_uint32) for n in ("a_step", "b_step", "c_step"))]


def to_bf16(x: NDArray[np.floating]) -> NDArray[np.uint16]:
    """Round float32 values to bfloat16 bits, to nearest with ties to even (finite inputs)."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(bits: NDArray[np.uint16]) -> NDArray[np.float32]:
    """Widen bfloat16 bits to float32 (exact)."""
    return (np.asarray(bits, np.uint32) << 16).view(np.float32)


TABLE: dict[str, Signature] = {
    "adamw": Signature(4, (0, 2, 3), AdamParams),
    "gemm_forward": Signature(8, (2, 7), FusedParams),
    "gemm_backward": Signature(9, (7, 8), FusedParams),
    "gemm_tn_members": Signature(3, (2,), MemberGemmParams),
    "member_weight_grad": Signature(3, (2,), NetParams),
    "member_adapter_grad": Signature(3, (2,), NetParams),
    "head_forward": Signature(4, (3,), NetParams),
    "softmax_ce": Signature(5, (2, 3), NetParams),
    "hl_gauss_ce": Signature(5, (2, 3), NetParams),
    "mse_loss": Signature(5, (2, 3), NetParams),
    "head_backward": Signature(4, (2, 3), NetParams),
    "head_block_backward": Signature(8, (6, 7), NetParams),
    "member_sums": Signature(4, (2, 3), NetParams),
    "sum_partials": Signature(2, (1,), NetParams),
    "predict_class": Signature(2, (1,), NetParams),
    "predict_mean": Signature(2, (1,), NetParams),
    "predict_centre": Signature(2, (1,), NetParams),
    "seq_forward": Signature(12, (7, 8, 9, 10, 11), SeqParams),
    "seq_input_grad": Signature(3, (2,), GemmParams),
    "seq_member_sum": Signature(3, (2,), SeqParams),
    "seq_backward": Signature(9, (6, 7, 8), SeqParams),
    "seq_weight_grad": Signature(3, (2,), MemberGemmParams),
    "seq_column_partials": Signature(2, (1,), NetParams),
    "emb_forward": Signature(8, (7,), EmbParams),
    "emb_member_sum": Signature(3, (2,), EmbParams),
    "emb_weight_grad": Signature(8, (5, 6, 7), EmbParams),
}  # the buffers each kernel binds, the ones it writes, its parameters (the tests hold it to the kernel sources)


@dataclass
class Slice:
    """One micro-batch's own buffers: inputs, absolute row indices, labels, logits, per-output losses, predictions.

    Per row: K labels and K predictions; per member output: K · c logits and K losses.
    """

    x: Buffer
    rows: Buffer
    labels: Buffer
    logits: Buffer
    loss: Buffer
    pred: Buffer
    weights: Buffer  # each row's weight in the loss, (rows,): ones unless the step is weighted
    seq: Buffer | None = None  # the sequences of a network with an encoder, (pairs, ld_in)
    raw: Buffer | None = None  # the embedded inputs of a network with embeddings, (rows, F)
    d_outputs: Buffer | None = None  # the output gradient of a host loss (`MetalTabM.loss_step`)


def check_network(params: TabMParams, max_batch: int, targets: int, sequence_shape: tuple[int, int] | None,
                  bins_shape: tuple[int, int] | None = None) -> None:  # fmt: skip
    """Raise ``ValueError`` unless the kernels can build this network."""
    if (params.num_emb_type == "none") != (bins_shape is None):
        raise ValueError(f"num_emb_type {params.num_emb_type!r} with bins of shape {bins_shape}")
    if bins_shape is not None and not (bins_shape[0] >= 1 and bins_shape[1] >= 1):
        raise ValueError(f"embeddings need at least one input and one component, got bins of shape {bins_shape}")
    if params.n_classes > MAX_CLASSES:
        raise ValueError(f"at most {MAX_CLASSES} classes, got {params.n_classes}")
    if params.objective == "hl_gauss" and targets != 1:
        raise ValueError(f"HL-Gauss trains one target; got {targets}")
    if max_batch < 1:
        raise ValueError("max_batch must be at least 1")
    if targets < 1:
        raise ValueError(f"targets must be a positive int, got {targets!r}")
    if (params.encoder == "none") != (sequence_shape is None):
        raise ValueError(f"encoder {params.encoder!r} with sequences of shape {sequence_shape}")
    if sequence_shape is not None and not (sequence_shape[0] >= 1 and sequence_shape[1] >= 1):
        raise ValueError(f"sequences need at least one step and one input, got shape {sequence_shape}")


class DeviceTabM:
    """One TabM network on ``device``: weights, AdamW state and the buffers of a batch of up to ``max_batch`` rows.

    The network owns ``device`` and closes it; the device's library holds `TABLE`'s kernels for ``params.storage``.

    ``targets`` (K) is the number of target heads: ``K · n_classes`` outputs per member. ``rows_pad`` is the padded rows
    of one micro-batch (of the whole batch when ``params.micro_batch`` is 0 or at least ``max_batch``), ``slice_rows``
    the rows a micro-batch takes from a batch and ``capacity`` the most rows one call accepts. ``sequence_shape``
    ``(S, C)``: the steps and the inputs per step of the rows' sequences, for (and only for) a network with an encoder.
    ``bins_shape`` ``(F, P)``: the embedded inputs and the encoding's components, for (and only for) a network with
    piecewise-linear embeddings (the first F of the ``n_inputs`` inputs). ``GEMM`` is how the device's library launches
    its GEMM kernels (Metal's here; CUDA's in `idun.tabm.cuda`).
    """

    GEMM: GemmLaunch = METAL_GEMM

    def __init__(
        self,
        device: Device,
        params: TabMParams,
        n_inputs: int,
        max_batch: int,
        targets: int = 1,
        sequence_shape: tuple[int, int] | None = None,
        bins_shape: tuple[int, int] | None = None,
    ) -> None:
        check_network(params, max_batch, targets, sequence_shape, bins_shape)
        self.params, self.n_inputs, self.targets = params, n_inputs, targets
        self.sequence_shape, self.bins_shape = sequence_shape, bins_shape
        self.encoded = params.encoded  # the encoder's inputs, before the snapshot's in each row
        self.embedded, self.components = bins_shape or (0, 0)  # F embedded inputs, P components each
        self.emb_width = self.embedded * params.d_embedding  # block 0's embedded columns, before the other inputs
        self.block0 = self._block0_width()
        k, w, n_out = params.members, params.width, targets * params.outputs
        self.hl = params.objective == "hl_gauss"
        self.label_width = params.n_bins if self.hl else targets  # soft labels per row, or one label per target
        self.ldw, self.ldd = round_up(w, TILE), round_up(self.block0, TILE)
        whole = not 0 < params.micro_batch < max_batch  # one micro-batch: the whole batch
        # rows · k fills whole tiles, and rows is a whole number of GEMM k-steps (gemm_tn_members sums over rows)
        self.rows_pad = round_up(max_batch if whole else params.micro_batch, math.lcm(BK, TILE // math.gcd(k, TILE)))
        self.slice_rows = self.rows_pad if whole else params.micro_batch
        self.capacity = round_up(max_batch, self.slice_rows)
        self.m = self.rows_pad * k
        self.device = device
        self.act_bytes = 2 if params.storage == "bf16" else 4
        self.head = 4 * params.blocks  # the head's index in the parameter lists; its bias follows, then the encoder's
        self.din = [self.ldd] + [self.ldw] * (params.blocks - 1)
        shapes = ([(self.ldw, d) for d in self.din] + [(k, d) for d in self.din] + [(k, self.ldw)] * params.blocks
                  + [(k, self.ldw)] * params.blocks + [(k, n_out, self.ldw), (k, n_out)])  # fmt: skip
        if sequence_shape is not None:
            shapes += self._encoder_shapes(sequence_shape)
        if bins_shape is not None:  # EmbeddingWeights.arrays: linear, linear_bias, ple
            shapes += [(self.embedded, params.d_embedding)] * 2 + [(self.embedded, self.components, params.d_embedding)]
        self.shapes: list[tuple[int, ...]] = shapes
        self.param = [self._zeros(s) for s in shapes]
        self.grad = [self._zeros(s) for s in shapes]
        self.moment1 = [self._zeros(s) for s in shapes]
        self.moment2 = [self._zeros(s) for s in shapes]
        self.adam_steps = 0
        self._rate: float | None = None  # a schedule's learning rate (`set_learning_rate`)
        b, m = self.rows_pad, self.m
        self.slices = [
            Slice(self._acts((b, self.ldd)), self._zeros((b,)), self._zeros((b, self.label_width)),
                  self._zeros((m, n_out)), self._zeros((m, targets)), self._zeros((b, targets)), self._zeros((b,)))
            for _ in range(self.capacity // self.slice_rows)
        ]  # fmt: skip
        self.t = [self._acts((m, self.ldw)) for _ in range(params.blocks)]
        self.a = [self._acts((m, self.ldw)) for _ in range(params.blocks)]
        self.d_logits = self._zeros((m, n_out))
        self.dz = self._acts((m, self.ldw))
        self.dt = [
            self._acts((m, self.ldw)),
            self._acts((m, self.ldw)),
        ]  # block i's reads one, block i - 1's the other
        self.products = self._zeros((k, self.ldw, max(self.ldd, self.ldw)))  # G_j of one block (gemm_tn_members)
        self.parts = [self._zeros((CHUNKS, k, self.ldw)) for _ in range(2)]
        self.sizes: list[int] = []  # rows of each micro-batch of the loaded batch
        self.weight_sum: float | None = None  # the loaded batch's weight sum (None: unweighted, B)
        if sequence_shape is not None:
            self._encoder_buffers()
        self._bins: Bins | None = None
        if bins_shape is not None:
            self._embedding_buffers()

    def _block0_width(self) -> int:
        """Return block 0's input width: the encoder's outputs, the embedded values, then the other inputs."""
        if self.embedded > self.n_inputs:
            raise ValueError(f"bins for {self.embedded} inputs of {self.n_inputs}")
        return self.encoded + self.emb_width + self.n_inputs - self.embedded

    def _embedding_buffers(self) -> None:
        """Allocate the embeddings' buffers: per micro-batch its embedded inputs; the encoding; the input gradients."""
        f, c = self.embedded, self.components
        for part in self.slices:
            part.raw = self._zeros((self.rows_pad, f))
        self.ld_e = round_up(self.emb_width, TILE)
        self.bins_buffers = [self._zeros((f, c)), self._zeros((f, c)), self._zeros((f,))]  # scale, shift, single
        self.d_emb_inputs = self._zeros((self.m, self.ld_e))  # dU = dT_0 · W_0[:, :ld_e]
        self.d_emb = self._zeros((self.rows_pad, self.ld_e))

    def _encoder_shapes(self, sequence_shape: tuple[int, int]) -> list[tuple[int, ...]]:
        """Set the encoder's padded widths; return its parameters' padded shapes (`EncoderWeights.arrays` order)."""
        p = self.params
        self.steps, self.seq_inputs = sequence_shape
        self.ld_in, self.ld_e = round_up(self.seq_inputs, TILE), round_up(p.encoder_embed, TILE)
        self.ld_h, self.ld_g = round_up(p.encoder_hidden, TILE), round_up(3 * p.encoder_hidden, TILE)
        return [(self.ld_e, self.ld_in), (self.ld_e,), (self.ld_g, self.ld_e), (self.ld_g, self.ld_h), (self.ld_g,),
                (self.ld_g,)]  # fmt: skip

    def _encoder_buffers(self) -> None:
        """Allocate the encoder's buffers: per micro-batch its sequences, shared its activations and gradients.

        The (row, step) pairs of a micro-batch, ``rows_pad · S``, are summed in `SEQ_CHUNKS` chunks of ``chunk_pairs``
        (a whole number of GEMM k-steps); the pair buffers have room for every chunk, the pairs past the last zero.
        """
        pairs = self.rows_pad * self.steps
        self.chunk_pairs = round_up(-(-pairs // SEQ_CHUNKS), BK)
        self.n_chunks = -(-pairs // self.chunk_pairs)
        room = self.n_chunks * self.chunk_pairs
        for part in self.slices:
            part.seq = self._zeros((room, self.ld_in))
        self.pre, self.emb, self.d_pre = (self._zeros((room, self.ld_e)) for _ in range(3))
        self.hprev = self._zeros((room, self.ld_h))
        self.gates = self._zeros((room, 4, self.ld_h))
        self.d_gi, self.d_gh = self._zeros((room, self.ld_g)), self._zeros((room, self.ld_g))
        self.d_inputs = self._zeros((self.m, self.ld_h))  # dU = dT_0 · W_0[:, :ld_h]
        self.d_encoded = self._zeros((self.rows_pad, self.ld_h))
        widest = max(self.ld_e * self.ld_in, self.ld_g * max(self.ld_e, self.ld_h))
        self.seq_parts = self._zeros((self.n_chunks, widest))

    def _zeros(self, shape: tuple[int, ...]) -> Buffer:
        return self.device.alloc(4 * int(np.prod(shape)))

    def _acts(self, shape: tuple[int, ...]) -> Buffer:
        """Return an activation buffer: ``act_t`` elements (4 or 2 bytes each)."""
        return self.device.alloc(self.act_bytes * int(np.prod(shape)))

    # -- parameters ---------------------------------------------------------------------------------------------------

    def _blocks(self) -> int:
        return self.params.blocks

    def _fill(self, buffers: list[Buffer], arrays: list[NDArray[np.floating]]) -> None:
        """Copy unpadded arrays (`reference.Weights.arrays` order, the head ``(k, w, K · c)``) into padded buffers."""
        arrays = list(arrays)
        arrays[self.head] = arrays[self.head].transpose(0, 2, 1)
        for buffer, shape, array in zip(buffers, self.shapes, arrays, strict=True):
            view = self.device.write(buffer, np.float32, shape)
            view[...] = 0.0
            view[tuple(slice(0, n) for n in array.shape)] = array

    def set_weights(self, weights: reference.Weights) -> None:
        """Copy ``weights`` (unpadded, any float dtype) into the padded parameter buffers (the head transposed)."""
        if (weights.encoder is None) != (self.sequence_shape is None):
            raise ValueError("the weights' encoder does not match this network's")
        if (weights.bins is None) != (self.bins_shape is None) or (
            weights.bins is not None and weights.bins.scale.shape != self.bins_shape
        ):
            raise ValueError("the weights' embeddings do not match this network's")
        if weights.bins is not None:
            bins, (scale, shift, single) = weights.bins, self.bins_buffers
            self.device.write(scale, np.float32, bins.scale.shape)[...] = bins.scale
            self.device.write(shift, np.float32, bins.shift.shape)[...] = bins.shift
            self.device.write(single, np.uint32, bins.single.shape)[...] = bins.single
            self._bins = bins
        self._fill(self.param, weights.arrays())
        self.adam_steps = 0
        for buffer, shape in zip([*self.moment1, *self.moment2], self.shapes * 2, strict=True):
            self.device.fill(buffer, 0, 4 * int(np.prod(shape)))

    def training_state(self) -> tuple[reference.Weights, reference.AdamState]:
        """Return the weights, AdamW's moments (unpadded float32 copies, `reference.Weights.arrays` order) and steps."""
        moments = [self._unpadded(buffers).arrays() for buffers in (self.moment1, self.moment2)]
        return self.weights(), reference.AdamState(moments[0], moments[1], self.adam_steps)

    def set_training_state(self, weights: reference.Weights, state: reference.AdamState) -> None:
        """Copy in what `training_state` returned: the next step is the one the saved network would have taken."""
        self.set_weights(weights)
        self._fill(self.moment1, state.m)
        self._fill(self.moment2, state.v)
        self.adam_steps = state.step

    def _unpadded(self, buffers: list[Buffer]) -> reference.Weights:
        p, n, n_out = self.params, self._blocks(), self.targets * self.params.outputs
        d0 = self.block0
        real = ([(p.width, d0)] + [(p.width, p.width)] * (n - 1) + [(p.members, d0)]
                + [(p.members, p.width)] * (n - 1) + [(p.members, p.width)] * (2 * n)
                + [(p.members, n_out, p.width), (p.members, n_out)])  # fmt: skip
        if self.sequence_shape is not None:
            e, h, c = p.encoder_embed, p.encoder_hidden, self.seq_inputs
            real += [(e, c), (e,), (3 * h, e), (3 * h, h), (3 * h,), (3 * h,)]
        if self.bins_shape is not None:
            real += self.shapes[self.head + 2 :]
        arrays = [self.device.read(b, np.float32, s)[tuple(slice(0, d) for d in r)].copy()
                  for b, s, r in zip(buffers, self.shapes, real, strict=True)]  # fmt: skip
        arrays[self.head] = np.ascontiguousarray(arrays[self.head].transpose(0, 2, 1))
        encoder = None if self.sequence_shape is None else EncoderWeights(*arrays[self.head + 2 :])
        embedding = None if self.bins_shape is None else EmbeddingWeights(*arrays[self.head + 2 :])
        return reference.Weights(arrays[:n], arrays[n : 2 * n], arrays[2 * n : 3 * n], arrays[3 * n : 4 * n],
                                 arrays[4 * n], arrays[4 * n + 1], encoder, embedding, self._bins)  # fmt: skip

    def weights(self) -> reference.Weights:
        """Return the current weights, unpadded, float32."""
        return self._unpadded(self.param)

    def gradients(self) -> reference.Weights:
        """Return the gradients of the last backward pass, unpadded, float32."""
        return self._unpadded(self.grad)

    # -- one batch ----------------------------------------------------------------------------------------------------

    def _load(
        self,
        x: NDArray[np.floating],
        rows: NDArray[np.integer],
        y: NDArray[Any] | None,
        seq: NDArray[np.floating] | None = None,
        weights: NDArray[np.floating] | None = None,
    ) -> None:
        """Split a batch into micro-batches of ``slice_rows`` rows and copy each into its `Slice`.

        ``y``: the labels, ``(B, K)`` or ``(B,)`` for K = 1; None for inference. ``seq``: the rows' sequences
        ``(B, S, C)``, for (and only for) a network with an encoder. ``weights``: each row's weight in the loss
        (None: one each).
        """
        n = x.shape[0]
        if not 1 <= n <= self.capacity or x.shape[1] != self.n_inputs or rows.shape != (n,):
            raise ValueError(f"need 1..{self.capacity} rows of {self.n_inputs} inputs and one row index per row")
        if (seq is None) != (self.sequence_shape is None):
            raise ValueError("a network with an encoder takes sequences, and one without takes none")
        if seq is not None and seq.shape != (n, *self.steps_inputs()):
            raise ValueError(f"need sequences of shape ({n}, {self.steps}, {self.seq_inputs}), got {seq.shape}")
        y = None if y is None else self._labels(y, n)
        self.sizes = []
        if weights is not None and np.shape(weights) != (n,):
            raise ValueError(f"need one weight per row ({n}), got shape {np.shape(weights)}")
        self.weight_sum = None if weights is None else float(np.sum(weights, dtype=np.float64))
        f, start = self.embedded, self.encoded + self.emb_width  # the inputs block 0 reads as they are start there
        for part, lo in zip(self.slices, range(0, n, self.slice_rows), strict=False):
            hi = min(lo + self.slice_rows, n)
            if self.act_bytes == 2:
                bits = self.device.write(part.x, np.uint16, (self.rows_pad, self.ldd))
                bits[...] = 0
                bits[: hi - lo, start : self.block0] = to_bf16(x[lo:hi, f:])
            else:
                view = self.device.write(part.x, np.float32, (self.rows_pad, self.ldd))
                view[...] = 0.0
                view[: hi - lo, start : self.block0] = x[lo:hi, f:]
            if part.raw is not None:
                raw = self.device.write(part.raw, np.float32, (self.rows_pad, f))
                raw[...] = 0.0
                raw[: hi - lo] = x[lo:hi, :f]
            if seq is not None and part.seq is not None:
                pairs = self.device.write(part.seq, np.float32, (self.n_chunks * self.chunk_pairs, self.ld_in))
                pairs[...] = 0.0
                pairs[: (hi - lo) * self.steps, : self.seq_inputs] = seq[lo:hi].reshape(-1, self.seq_inputs)
            index = self.device.write(part.rows, np.uint32, (self.rows_pad,))
            index[...] = 0
            index[: hi - lo] = rows[lo:hi]
            kind = np.uint32 if self.params.objective == "cross_entropy" else np.float32  # classes, or float labels
            labels = self.device.write(part.labels, kind, (self.rows_pad, self.label_width))
            labels[...] = 0
            if y is not None:
                labels[: hi - lo] = y[lo:hi]
            each = self.device.write(part.weights, np.float32, (self.rows_pad,))
            each[...] = 0.0
            each[: hi - lo] = 1.0 if weights is None else weights[lo:hi]
            self.sizes.append(hi - lo)

    def _labels(self, y: NDArray[Any], n: int) -> NDArray[Any]:
        """Return a batch's labels as the loss kernel reads them: targets, soft labels or classes."""
        if self.params.objective == "mse":
            return np.asarray(y, np.float32).reshape(n, self.targets)  # regression targets
        if self.hl:
            p = self.params
            return hl_gauss.labels(y, p.n_bins, p.hl_low, p.hl_high, p.hl_sigma).astype(np.float32)  # (n, n_bins)
        return reference.label_matrix(y, n, self.targets, self.params.n_classes)

    def steps_inputs(self) -> tuple[int, int]:
        """Return the sequences' (S, C)."""
        if self.sequence_shape is None:
            raise ValueError("this network has no encoder")
        return self.sequence_shape

    def _seq(self, q: int) -> SeqParams:
        """Return micro-batch ``q``'s `SeqParams`."""
        p = self.params
        return SeqParams(self.sizes[q], self.rows_pad, self.steps, self.seq_inputs, self.ld_in, p.encoder_embed,
                         self.ld_e, p.encoder_hidden, self.ld_h, self.ld_g, p.members, self.ldd)  # fmt: skip

    def _emb(self, q: int) -> EmbParams:
        """Return micro-batch ``q``'s `EmbParams`: weight gradients stored for micro-batch 0, added for later ones."""
        return EmbParams(self.sizes[q], self.rows_pad, self.embedded, self.components, self.params.d_embedding,
                         self.embedded, self.ldd, self.ld_e, self.params.members, int(q > 0))  # fmt: skip

    def _sums(self, din: int, q: int) -> NetParams:
        """Return the `NetParams` of an encoder gradient sum over the chunks of pairs.

        ``din`` columns; the sums are stored for micro-batch 0 and added for the later ones.
        """
        return NetParams(0, 0, 1, 0, 0, 0, 0, 0, 1.0, 0.0, din, self.chunk_pairs, self.n_chunks, int(q > 0), 1, 0.0,
                         0.0)  # fmt: skip

    def _net(self, q: int, key: int = 0, threshold: int = 0, scale: float = 1.0, din: int = 0, *,
             head: bool = False) -> NetParams:  # fmt: skip
        """Return micro-batch ``q``'s `NetParams`: its rows, the whole batch's logit scale, accumulating when q > 0.

        ``head``: the head kernels' view of the outputs, ``targets × classes``: an HL-Gauss head as groups of
        `HEAD_GROUP` outputs. Otherwise one HL-Gauss target has ``n_bins`` classes.
        """
        p = self.params
        batch = self.sizes[q]
        classes, targets = p.outputs, self.targets
        if self.hl and head:
            classes, targets = HEAD_GROUP, p.n_bins // HEAD_GROUP
        width = (p.hl_high - p.hl_low) / p.n_bins
        rows = sum(self.sizes) if self.weight_sum is None else self.weight_sum  # the batch's B, or its weight sum
        inv_count = 1.0 / (rows * p.members * self.targets) if rows > 0 else 0.0  # weight 0: no gradient
        return NetParams(batch, self.rows_pad, p.members, p.width, self.ldw, classes, key, threshold, scale,
                         inv_count, din, -(-batch // CHUNKS), CHUNKS,
                         int(q > 0), targets, p.hl_low, width)  # fmt: skip

    def _dropout(self, step: int | None, block: int) -> tuple[int, int, float]:
        """Return (key, threshold, scale) of a block's dropout mask; (0, 0, 1) keeps everything."""
        if step is None or self.params.dropout == 0.0:
            return 0, 0, 1.0
        p = self.params
        key = rng.derive_key(p.seed, reference.DROPOUT_SALT, step, block)
        return key, round(p.dropout * (1 << rng.UNIFORM_BITS)), float(np.float32(1.0 / (1.0 - p.dropout)))

    def _gemm(self, name: str, buffers: list[Buffer], mnk: tuple[int, int, int], ld: tuple[int, int, int],
              div: int, ldr: int, net: NetParams | None = None) -> None:  # fmt: skip
        """Dispatch a GEMM; ``net`` set means a fused output stage (`FusedParams`)."""
        m, n, k = mnk
        gemm = GemmParams(m, n, k, *ld, self.params.members, div, ldr)
        params = gemm if net is None else FusedParams(gemm, net)
        self.device.encode(name, buffers, params, *self.GEMM.grid(n, m))

    def _grid2(self, name: str, buffers: list[Buffer], params: ctypes.Structure, nx: int, ny: int) -> None:
        self.device.encode(name, buffers, params, (-(-nx // 32), -(-ny // 8), 1), (32, 8, 1))

    def _reduce(self, q: int, name: str, buffers: list[Buffer], width: int, outs: list[Buffer]) -> None:
        """Run micro-batch ``q``'s two-stage batch sum into ``outs`` (each ``(k, width)``; added to them when q > 0).

        ``name`` writes per-chunk partials (one ``parts`` buffer per output), then ``sum_partials`` adds them in order.
        """
        net = self._net(q, din=width)
        k = self.params.members
        self.device.encode(name, buffers, net, (-(-width // 32), -(-k // 8), CHUNKS), (32, 8, 1))
        for part, out in zip(self.parts, outs, strict=False):
            self.device.encode("sum_partials", [part, out], net, (-(-k * width // 256), 1, 1), (256, 1, 1))

    def _encoder_arrays(self) -> list[Buffer]:
        """Return the encoder's parameter buffers: embed, embed_bias, w_ih, w_hh, b_ih, b_hh."""
        return self.param[self.head + 2 : self.head + 8]

    def _encode_forward(self, q: int, step: int | None) -> None:
        """Encode micro-batch ``q``'s forward pass into its logits (the encoder's first, if any)."""
        k, part = self.params.members, self.slices[q]
        if part.seq is not None:
            self.device.encode("seq_forward", [part.seq, *self._encoder_arrays(), self.pre, self.emb,
                              self.hprev, self.gates, part.x], self._seq(q), (1, self.rows_pad // SEQ_ROWS, 1),
                              (self.params.encoder_hidden, SEQ_ROWS, 1))  # fmt: skip
        if part.raw is not None:
            self._grid2("emb_forward", [part.raw, *self.bins_buffers, *self.param[self.head + 2 : self.head + 5],
                        part.x], self._emb(q), self.emb_width, self.rows_pad)  # fmt: skip
        h, div = part.x, k
        for i in range(self._blocks()):
            weight, r, s, bias = (self.param[g * self._blocks() + i] for g in range(4))
            self._gemm("gemm_forward", [h, weight, self.t[i], r, s, bias, part.rows, self.a[i]],
                       (self.m, self.ldw, self.din[i]), (self.din[i], self.din[i], self.ldw), div, self.din[i],
                       self._net(q, *self._dropout(step, i)))  # fmt: skip
            h, div = self.a[i], 1
        head, head_bias, net = self.param[self.head], self.param[self.head + 1], self._net(q, head=True)
        per_group = min(net.targets, TARGETS_PER_GROUP)
        self.device.encode("head_forward", [self.a[-1], head, head_bias, part.logits], net,
                          (-(-self.rows_pad // 32), -(-net.targets // per_group), k), (32, per_group, 1))  # fmt: skip

    def _encode_backward(self, q: int, step: int, d_logits: Buffer | None = None) -> None:
        """Encode micro-batch ``q``'s loss and backward pass: store the gradients (q = 0) or add to them.

        ``d_logits``: the micro-batch's output gradient ``(m, K · c)``, padding rows zero, in place of the loss
        kernel's (a host loss; no loss kernel runs).
        """
        n, k, part = self._blocks(), self.params.members, self.slices[q]
        if d_logits is None:
            d_logits = self.d_logits
            loss = {"mse": "mse_loss", "hl_gauss": "hl_gauss_ce"}.get(self.params.objective, "softmax_ce")
            self.device.encode(loss, [part.logits, part.labels, d_logits, part.loss, part.weights], self._net(q),
                              (-(-self.m * self.targets // 256), 1, 1), (256, 1, 1))  # fmt: skip
        head = self._net(q, head=True)
        per_group = min(head.targets, TARGETS_PER_GROUP)
        self.device.encode("head_backward", [self.a[-1], d_logits, self.grad[self.head],
                                                         self.grad[self.head + 1]],
                          head, (-(-(self.ldw // 4 + 1) // 32), -(-head.targets // per_group), k),
                          (32, per_group, 1))  # fmt: skip
        top, cur = n - 1, 0
        self._grid2("head_block_backward", [d_logits, self.param[self.head], self.t[top], self.param[2 * n + top],
                    self.param[3 * n + top], part.rows, self.dz, self.dt[cur]],
                    self._net(q, *self._dropout(step, top)), self.ldw, k * -(-self.rows_pad // HBB_ROWS))  # fmt: skip
        for i in reversed(range(n)):
            weight, r = self.param[i], self.param[n + i]
            d_weight, d_r, d_s, d_bias = (self.grad[g * n + i] for g in range(4))
            self._reduce(q, "member_sums", [self.dz, self.t[i], *self.parts], self.ldw, [d_bias, d_s])
            h, ldh, h_step = (part.x, self.ldd, 0) if i == 0 else (self.a[i - 1], k * self.ldw, self.ldw)
            self._weight_grads(q, self.dt[cur], h, ldh, h_step, weight, r, d_weight, d_r, self.din[i])
            if i == 0 and self.slices[q].seq is not None:
                self._encode_encoder_backward(q, self.dt[cur])
            if i == 0 and self.slices[q].raw is not None:
                self._encode_embedding_backward(q, self.dt[cur])
            if i > 0:
                below = [self.t[i - 1], self.param[2 * n + i - 1], self.param[3 * n + i - 1], part.rows, self.dz,
                         self.dt[1 - cur]]  # fmt: skip
                self._gemm("gemm_backward", [self.dt[cur], weight, r, *below], (self.m, self.din[i], self.ldw),
                           (self.ldw, self.din[i], self.din[i]), 1, self.din[i],
                           self._net(q, *self._dropout(step, i - 1)))  # fmt: skip
            cur = 1 - cur

    def _encode_encoder_backward(self, q: int, dt0: Buffer) -> None:
        """Encode the encoder's backward pass from block 0's ``dT_0``: its gradients stored (q = 0) or added to."""
        p, part, seq = self.params, self.slices[q], self._seq(q)
        k, n = p.members, self._blocks()
        if part.seq is None:
            raise ValueError("this network has no encoder")
        gemm = GemmParams(self.m, self.ld_h, self.ldw, self.ldw, self.ldd, self.ld_h, k, 1, 0)
        self.device.encode("seq_input_grad", [dt0, self.param[0], self.d_inputs], gemm,
                          *self.GEMM.grid(self.ld_h, self.m))  # fmt: skip
        self._grid2("seq_member_sum", [self.d_inputs, self.param[n], self.d_encoded], seq, self.ld_h, self.rows_pad)
        _, _, w_ih, w_hh, _, _ = self._encoder_arrays()
        self.device.encode("seq_backward", [self.d_encoded, w_ih, w_hh, self.pre, self.hprev, self.gates,
                          self.d_gi, self.d_gh, self.d_pre], seq, (1, self.rows_pad // SEQ_ROWS, 1),
                          (p.encoder_hidden, SEQ_ROWS, 1))  # fmt: skip
        d_embed, d_embed_bias, d_ih, d_hh, d_bih, d_bhh = self.grad[self.head + 2 : self.head + 8]
        sequences = part.seq
        for a, b, out, (rows, cols) in ((self.d_pre, sequences, d_embed, (self.ld_e, self.ld_in)),
                                        (self.d_gi, self.emb, d_ih, (self.ld_g, self.ld_e)),
                                        (self.d_gh, self.hprev, d_hh, (self.ld_g, self.ld_h))):  # fmt: skip
            chunk = self.chunk_pairs
            gemm = GemmParams(rows, cols, chunk, rows, cols, cols, 1, 1, 0)
            self.device.encode("seq_weight_grad", [a, b, self.seq_parts],
                              MemberGemmParams(gemm, chunk * rows, chunk * cols, rows * cols),
                              *self.GEMM.grid(cols, rows, self.n_chunks))  # fmt: skip
            self._sum_chunks(rows * cols, q, out)
        for a, out, cols in ((self.d_pre, d_embed_bias, self.ld_e), (self.d_gi, d_bih, self.ld_g),
                             (self.d_gh, d_bhh, self.ld_g)):  # fmt: skip
            self._grid2("seq_column_partials", [a, self.seq_parts], self._sums(cols, q), cols, self.n_chunks)
            self._sum_chunks(cols, q, out)

    def _encode_embedding_backward(self, q: int, dt0: Buffer) -> None:
        """Encode the embeddings' backward pass from block 0's ``dT_0``: their gradients stored (q = 0) or added to."""
        part, k, n = self.slices[q], self.params.members, self._blocks()
        if part.raw is None:
            raise ValueError("this network has no embeddings")
        gemm = GemmParams(self.m, self.ld_e, self.ldw, self.ldw, self.ldd, self.ld_e, k, 1, 0)
        self.device.encode("seq_input_grad", [dt0, self.param[0], self.d_emb_inputs], gemm,
                          *self.GEMM.grid(self.ld_e, self.m))  # fmt: skip
        emb = self._emb(q)
        self._grid2("emb_member_sum", [self.d_emb_inputs, self.param[n], self.d_emb], emb, self.ld_e, self.rows_pad)
        total = (self.components + 2) * self.emb_width
        self.device.encode("emb_weight_grad", [part.raw, *self.bins_buffers, self.d_emb,
                          *self.grad[self.head + 2 : self.head + 5]], emb, (-(-total // 256), 1, 1),
                          (256, 1, 1))  # fmt: skip

    def _sum_chunks(self, size: int, q: int, out: Buffer) -> None:
        """Add the chunks' partials in `seq_parts` (``size`` values each) in chunk order into ``out``."""
        self.device.encode("sum_partials", [self.seq_parts, out], self._sums(size, q),
                          (-(-size // 256), 1, 1), (256, 1, 1))  # fmt: skip

    def _weight_grads(self, q: int, dt: Buffer, h: Buffer, ldh: int, h_step: int,
                      weight: Buffer, r: Buffer, d_weight: Buffer, d_r: Buffer,
                      din: int) -> None:  # fmt: skip
        """Encode a block's weight and adapter gradients from the members' products G_j = dT_jᵀ · h_j.

        Member j's rows of ``dt`` are b · k + j; its rows of ``h`` start ``j · h_step`` elements in, ``ldh`` apart
        (block 0's batch inputs: every member's, ``h_step`` 0). dW = Σ_j G_j ⊙ r_j and d_r_j = Σ_u W ⊙ G_j, stored
        for micro-batch 0 and added for the later ones.
        """
        k = self.params.members
        gemm = GemmParams(self.ldw, din, self.rows_pad, k * self.ldw, ldh, din, k, 1, din)
        self.device.encode("gemm_tn_members", [dt, h, self.products],
                          MemberGemmParams(gemm, self.ldw, h_step, self.ldw * din),
                          *self.GEMM.grid(din, self.ldw, k))  # fmt: skip
        net = self._net(q, din=din)
        self._grid2("member_weight_grad", [self.products, r, d_weight], net, din, self.ldw)
        self._grid2("member_adapter_grad", [self.products, weight, d_r], net, din, k)

    def set_learning_rate(self, rate: float | None) -> None:
        """Set the next steps' learning rate (a schedule's); None is ``params.learning_rate``."""
        self._rate = rate

    def _clip_scale(self) -> float:
        """Return clip_grad_norm_'s factor for the gradients: 1 unless ``params.gradient_clipping_norm`` is set.

        Clipping runs the encoded passes and reads every gradient back: their norm is summed on the host in float64,
        array by array in `reference.Weights.arrays` order (the padding is zero), as `reference.clip_scale`.
        """
        most = self.params.gradient_clipping_norm
        if most == 0.0:
            return 1.0
        self.device.flush()
        squares = [float(np.sum(np.square(self.device.read(g, np.float32, s), dtype=np.float64), dtype=np.float64))
                   for g, s in zip(self.grad, self.shapes, strict=True)]  # fmt: skip
        return reference.clip_scale(most, squares)

    def _encode_adam(self) -> None:
        p = self.params
        scale = self._clip_scale()
        lr = p.learning_rate if self._rate is None else self._rate
        self.adam_steps += 1
        correction1 = 1 - p.beta1**self.adam_steps
        correction2 = 1 - p.beta2**self.adam_steps
        for param, grad, m1, m2, shape in zip(self.param, self.grad, self.moment1, self.moment2, self.shapes,
                                              strict=True):  # fmt: skip
            size = int(np.prod(shape))
            adam = AdamParams(size, 1 - lr * p.weight_decay, p.beta1, p.beta2, lr / correction1,
                              math.sqrt(correction2), p.eps, scale)  # fmt: skip
            self.device.encode("adamw", [param, grad, m1, m2], adam, (-(-size // 256), 1, 1),
                              (256, 1, 1))  # fmt: skip

    # -- public steps -------------------------------------------------------------------------------------------------

    def _gather(self, field: str, per_row: int) -> NDArray[np.float32]:
        """Return the loaded batch's rows of a `Slice` buffer, ``per_row`` floats each, micro-batches concatenated."""
        read = self.device.read
        return np.concatenate([read(getattr(part, field), np.float32, (self.rows_pad * per_row,))[: size * per_row]
                               for part, size in zip(self.slices, self.sizes, strict=False)])  # fmt: skip

    def forward(self, x: NDArray[np.floating], rows: NDArray[np.integer], step: int | None,
                seq: NDArray[np.floating] | None = None) -> NDArray[np.float32]:  # fmt: skip
        """Return the logits ``(B, k, K · c)``; ``step`` None means inference (no dropout).

        ``seq``: the sequences ``(B, S, C)`` of a network with an encoder.
        """
        self._load(x, rows, None, seq)
        for q in range(len(self.sizes)):
            self._encode_forward(q, step)
        self.device.flush()
        p = self.params
        n_out = self.targets * p.outputs
        return self._gather("logits", p.members * n_out).reshape(x.shape[0], p.members, n_out)

    def backward(self, x: NDArray[np.floating], y: NDArray[Any], rows: NDArray[np.integer], step: int,
                 seq: NDArray[np.floating] | None = None,
                 weights: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run forward and backward of training step ``step`` (no update); return the mean loss. See `gradients`.

        ``y``: the labels, ``(B, K)``, or ``(B,)`` for one target (so in `train_step`); ``seq`` as in `forward`;
        ``weights`` as in `train_step`.
        """
        return self._step(x, y, rows, step, seq, update=False, weights=weights)

    def train_step(self, x: NDArray[np.floating], y: NDArray[Any], rows: NDArray[np.integer], step: int,
                   seq: NDArray[np.floating] | None = None,
                   weights: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run training step ``step`` (forward, backward, AdamW) in one command buffer; return the mean loss.

        ``weights``: each row's weight, the loss their weighted mean (None: the plain mean).
        """
        return self._step(x, y, rows, step, seq, update=True, weights=weights)

    def _step(self, x: NDArray[np.floating], y: NDArray[Any], rows: NDArray[np.integer], step: int,
              seq: NDArray[np.floating] | None, *, update: bool,
              weights: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Forward and backward of each micro-batch in order, then AdamW when ``update``; one command buffer."""
        self._load(x, rows, y, seq, weights)
        for q in range(len(self.sizes)):
            self._encode_forward(q, step)
            self._encode_backward(q, step)
        if update:
            self._encode_adam()
        self.device.flush()
        loss = self._gather("loss", self.params.members * self.targets)
        if weights is None:
            return float(np.sum(loss, dtype=np.float64) / loss.size)
        per_row = loss.reshape(len(rows), -1).astype(np.float64).mean(axis=1)
        total = float(np.sum(weights, dtype=np.float64)) or math.inf  # a batch of weight 0: loss 0
        return float(np.sum(np.asarray(weights, np.float64) * per_row) / total)

    def loss_step(self, x: NDArray[np.floating], loss: reference.Loss, rows: NDArray[np.integer], step: int,
                  seq: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run training step ``step`` under the caller's ``loss``; return the loss it gave.

        Forward in one command buffer; ``loss(outputs, rows)`` on the host, ``outputs`` the batch's member outputs
        ``(B, k, K · c)`` in float32 (`reference.host_loss` checks what it returns); then one command buffer with the
        backward pass from the returned gradient and AdamW. A micro-batched step runs each micro-batch's forward pass
        again before its backward pass. HL-Gauss networks take no loss (``ValueError``).
        """
        if self.hl:
            raise ValueError("an HL-Gauss network takes no loss")
        self._load(x, rows, None, seq)
        for q in range(len(self.sizes)):
            self._encode_forward(q, step)
        self.device.flush()
        k, n_out = self.params.members, self.targets * self.params.outputs
        outputs = self._gather("logits", k * n_out).reshape(x.shape[0], k, n_out)
        value, d_outputs = reference.host_loss(loss, outputs, np.asarray(rows, np.int64))
        buffers, lo = [], 0
        for part, size in zip(self.slices, self.sizes, strict=False):
            if part.d_outputs is None:
                part.d_outputs = self._zeros((self.m, n_out))
            view = self.device.write(part.d_outputs, np.float32, (self.m, n_out))
            view[...] = 0.0
            view[: size * k] = d_outputs[lo : lo + size].reshape(size * k, n_out)
            buffers.append(part.d_outputs)
            lo += size
        for q, buffer in enumerate(buffers):
            if len(buffers) > 1:
                self._encode_forward(q, step)  # the activation buffers hold the last micro-batch's
            self._encode_backward(q, step, buffer)
        self._encode_adam()
        self.device.flush()
        return value

    def inference_logits(self, x: NDArray[np.floating], seq: NDArray[np.floating] | None = None) -> NDArray[np.float32]:
        """Return the inference logits ``(B, k, K · c)`` (no dropout), at most `capacity` rows."""
        return self.forward(x, np.arange(x.shape[0]), None, seq)

    def predict(self, x: NDArray[np.floating], seq: NDArray[np.floating] | None = None) -> NDArray[np.float32]:
        """Return the expected class of the member-averaged softmax for each row of ``x`` (inference).

        ``(B,)`` for one target, ``(B, K)`` for K > 1 (each target's expected class). The members' mean output for
        ``"mse"``; the expected bin centre for ``"hl_gauss"``. ``seq``: the rows' sequences ``(B, S, C)`` of a network
        with an encoder.
        """
        out = np.empty((x.shape[0], self.targets), np.float32)
        for lo in range(0, x.shape[0], self.capacity):
            hi = min(lo + self.capacity, x.shape[0])
            self._load(x[lo:hi], np.zeros(hi - lo, np.uint32), None, None if seq is None else seq[lo:hi])
            for q, size in enumerate(self.sizes):
                self._encode_forward(q, None)
                name = {"mse": "predict_mean", "hl_gauss": "predict_centre"}.get(self.params.objective,
                                                                                 "predict_class")  # fmt: skip
                self.device.encode(name, [self.slices[q].logits, self.slices[q].pred], self._net(q),
                                   (-(-size * self.targets // 256), 1, 1), (256, 1, 1))  # fmt: skip
            self.device.flush()
            out[lo:hi] = self._gather("pred", self.targets).reshape(hi - lo, self.targets)
        return out[:, 0] if self.targets == 1 else out

    def close(self) -> None:
        """Release the device and its buffers."""
        self.device.close()
