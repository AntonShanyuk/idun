"""RealMLP on a GPU device: the network of `idun.realmlp.reference` as kernels driven through `idun.device.Device`.

A `DeviceRealMLP` owns padded parameter, gradient, Adam and activation buffers for batches of up to ``max_batch``
preprocessed rows (`idun.realmlp.reference.preprocess`), given by the host or gathered on the GPU from the cache placed
on the device (`DeviceRealMLP.bind`; `idun.realmlp.metal` builds the network on Metal). It names its kernels in `TABLE`
and reaches their buffers from the host through the device's ``read``, ``write`` and ``fill`` only. The batch is padded
to the GEMM tile and the embedded width (E + 1)·F to a multiple of 64; padding stays zero. A training step (embedding,
forward, loss, backward, Adam with the step's schedule values) is one batch of passes; every reduction runs in a fixed
order, and dropout masks are keyed by the absolute row.

``n_targets``: None is the single-target network (more targets: experimental, not covered by the documentation). The
labels come from a compact ``(n_rows, 1)`` uint8 array of classes (`idun.nn.inputs.class_array`), or for the
``"mse"`` objective from an ``(n_rows, 1)`` float32 target array (`idun.nn.inputs.target_array`).
"""

import ctypes
import math
import mmap
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun import rng
from idun.data import PackedDataset
from idun.device import Buffer, Device, Signature
from idun.nn import inputs
from idun.nn.device import METAL_GEMM, TILE, AdamParams, GemmLaunch, GemmParams, round_up
from idun.realmlp import reference as ref
from idun.realmlp.params import ACTIVATIONS, RealMLPParams

MAX_SMALL = 8  # the kernels are built for at most this many classes per target and embedding values per input


class RmlpParams(ctypes.Structure):
    """Mirror of ``RmlpParams`` in rmlp.metal."""

    _fields_ = [
        *((n, ctypes.c_uint32) for n in ("batch", "rows_pad", "inputs", "freqs", "embed", "ldz", "hidden", "classes",
                                         "targets", "units", "own", "key", "threshold")),
        *((n, ctypes.c_float) for n in ("scale", "gain", "smoothing", "inv_batch")),
        ("activation", ctypes.c_uint32),  # ACTIVATIONS' index of RealMLPParams.act
    ]  # fmt: skip


class GatherParams(ctypes.Structure):
    """Mirror of ``GatherParams`` in rmlp.metal."""

    _fields_ = [
        *((n, ctypes.c_uint32) for n in ("batch", "rows_pad", "n_named", "n_obf", "n_countries", "named_ld",
                                         "codes_ld", "targets", "target_ld")),
        ("inv_clip_sq", ctypes.c_float),
    ]  # fmt: skip


class CatParams(ctypes.Structure):
    """Mirror of ``CatParams`` in rmlp.metal."""

    _fields_ = [(n, ctypes.c_uint32) for n in ("batch", "rows_pad", "features", "dim", "ldz", "offset", "rows")]


class RmlpFused(ctypes.Structure):
    """Mirror of ``RmlpFused`` in rmlp.metal."""

    _fields_ = [("g", GemmParams), ("n", RmlpParams)]


TABLE: dict[str, Signature] = {
    "adamw": Signature(4, (0, 2, 3), AdamParams),
    "rmlp_embed": Signature(6, (5,), RmlpParams),
    "rmlp_forward_scaled": Signature(8, (2, 7), RmlpFused),
    "rmlp_forward": Signature(8, (2, 7), RmlpFused),
    "rmlp_out_forward": Signature(4, (3,), RmlpParams),
    "rmlp_loss": Signature(5, (2, 3), RmlpParams),
    "rmlp_mse": Signature(5, (2, 3), RmlpParams),
    "rmlp_out_backward": Signature(4, (2, 3), RmlpParams),
    "rmlp_out_block_backward": Signature(7, (5, 6), RmlpParams),
    "rmlp_out_block_backward_own": Signature(7, (5, 6), RmlpParams),
    "rmlp_input_backward": Signature(8, (6, 7), RmlpFused),
    "rmlp_embed_input_grad": Signature(4, (2,), RmlpFused),
    "rmlp_weight_grad": Signature(4, (2,), RmlpFused),
    "rmlp_block_sums": Signature(5, (3, 4), RmlpParams),
    "rmlp_scale_grad": Signature(3, (2,), RmlpParams),
    "rmlp_embed_backward": Signature(10, (6, 7, 8, 9), RmlpParams),
    "rmlp_predict": Signature(2, (1,), RmlpParams),
    "rmlp_predict_value": Signature(2, (1,), RmlpParams),
    "rmlp_gather": Signature(7, (6,), GatherParams),
    "rmlp_class_labels": Signature(3, (2,), GatherParams),
    "rmlp_target_values": Signature(3, (2,), GatherParams),
    "rmlp_cat_embed": Signature(4, (3,), CatParams),
    "rmlp_cat_backward": Signature(5, (4,), CatParams),
}  # the buffers each kernel binds, the ones it writes, its parameters (the tests hold it to the kernel sources)


def codes_stride(n_obf: int) -> int:
    """Return the row stride in bytes of the row-major codes: two codes per byte, rounded up to 16 bytes."""
    return round_up(-(-n_obf // 2), 16)


def check_classes(classes: NDArray[np.uint8], n_rows: int) -> NDArray[np.uint8]:
    """Return ``classes`` as an ``(n_rows, K)`` uint8 array without a copy; raise ``ValueError`` for anything else."""
    array = np.asarray(classes)
    if array.ndim != 2 or array.shape[0] != n_rows or array.shape[1] < 1 or array.dtype != np.uint8:
        raise ValueError(f"classes must be an ({n_rows}, K) uint8 array with K >= 1 (class k of row r at [r, k]), "
                         f"got a {array.dtype} array of shape {array.shape}")  # fmt: skip
    return array


def shared_codes(data: PackedDataset) -> NDArray[np.uint8]:
    """Return ``data``'s codes row-major (`inputs.codes_row_major`) in memory a Metal device wraps without a copy.

    ``n_rows`` × `codes_stride` bytes (3.6 GB at v6) of anonymous, page-aligned memory. Networks bound with it
    (`DeviceRealMLP.bind` ``codes=``) all read this one copy, so several seeds can train side by side on an 8 GB Mac;
    without it each network makes its own. The memory is returned when the array and every buffer wrapping it are gone.
    """
    ld = codes_stride(len(data.obfuscated_features))
    nbytes = data.n_rows * ld
    region = mmap.mmap(-1, -(-nbytes // mmap.PAGESIZE) * mmap.PAGESIZE)  # page-aligned
    out = np.frombuffer(region, np.uint8, count=nbytes).reshape(data.n_rows, ld)
    inputs.codes_row_major(data, out)
    return out


def check_own(params: RealMLPParams, n_targets: int | None) -> None:
    """Raise ``ValueError`` for an own last block (``params.own_last_block``) without K targets (``n_targets`` None)."""
    if params.own_last_block and n_targets is None:
        raise ValueError("own_last_block needs a network of K targets (n_targets, or train_realmlp's classes)")


def check_network(params: RealMLPParams, max_batch: int, n_targets: int | None, categories: Sequence[int] = (),
                  n_inputs: int | None = None) -> None:  # fmt: skip
    """Raise ``ValueError`` unless the kernels can build this network."""
    if any(m < 1 for m in categories) or (n_inputs is not None and len(categories) >= n_inputs):
        raise ValueError(f"categories must be positive level counts of the last of the inputs, at least one numeric "
                         f"input before them; got {list(categories)}")  # fmt: skip
    if params.n_classes > MAX_SMALL or params.embed > MAX_SMALL:
        raise ValueError(f"at most {MAX_SMALL} classes and {MAX_SMALL} embedding values per input")
    if any(width % TILE for width in params.widths):
        raise ValueError(f"every hidden width must be a multiple of {TILE}, got {list(params.widths)}")
    if max_batch < 1:
        raise ValueError("max_batch must be at least 1")
    if n_targets is not None and n_targets < 1:
        raise ValueError(f"n_targets must be None or an integer of at least 1, got {n_targets!r}")
    check_own(params, n_targets)


class DeviceRealMLP:
    """One RealMLP on ``device``: weights, Adam state and the buffers of a batch of up to ``max_batch`` rows.

    ``n_targets``: None for the single-target network, or K >= 1 heads on the trunk (see the module docstring).
    ``categories``: the levels of each of the G categorical inputs, the last G of the ``n_inputs`` (their codes; host
    batches only), embedded in ``params.embedding_size`` values each. The network owns ``device`` and closes it; the
    device's library holds `TABLE`'s kernels for the params' sizes. ``GEMM`` is how that library launches its GEMM
    kernels (Metal's here; CUDA's in `idun.realmlp.cuda`).
    """

    GEMM: GemmLaunch = METAL_GEMM

    def __init__(
        self,
        device: Device,
        params: RealMLPParams,
        n_inputs: int,
        max_batch: int,
        n_targets: int | None = None,
        categories: Sequence[int] = (),
    ) -> None:
        check_network(params, max_batch, n_targets, categories, n_inputs)
        self.params, self.n_inputs, self.n_targets = params, n_inputs, n_targets
        self.categories = tuple(categories)
        self.n_numeric = n_inputs - len(self.categories)  # F: the inputs before the categorical ones
        self.k = n_targets or 1  # the targets K; the output layer has K · C outputs
        self.own = params.own_last_block
        # each hidden layer's units: its width, and K times it for an own last block
        self.widths = list(params.widths)
        self.units = [*self.widths[:-1], self.widths[-1] * (self.k if self.own else 1)]
        f, q, e, h = self.n_numeric, params.frequencies, params.embed, self.widths[-1]
        c = params.n_classes * self.k
        self.z_width = (e + 1) * f + params.embedding_size * len(self.categories)  # block 0's inputs
        self.ldz = round_up(self.z_width, TILE)
        self.rows_pad = round_up(max_batch, TILE)
        self.device = device
        fan_in = [self.ldz, *self.widths[:-1]]
        weights = [(d, u) for d, u in zip(fan_in, self.units, strict=True)]
        units = [(u,) for u in self.units]
        self.shapes: list[tuple[int, ...]] = [(f, q), (f, q), (f, q, e), (f, e), (self.ldz,), *weights, *units, *units,
                                              (h, c), (c,)]  # fmt: skip
        self.out = 5 + 3 * len(self.widths)  # the output layer's index in the parameter lists; its bias, then the table
        if self.categories:  # the table, last: Weights.arrays
            self.shapes.append((sum(self.categories), params.embedding_size))
        self.param = [self._zeros(s) for s in self.shapes]
        self.grad = [self._zeros(s) for s in self.shapes]
        self.moment1 = [self._zeros(s) for s in self.shapes]
        self.moment2 = [self._zeros(s) for s in self.shapes]
        self.adam_steps = 0
        b = self.rows_pad
        self.c, self.z, self.dzs = self._zeros((b, f)), self._zeros((b, self.ldz)), self._zeros((b, self.ldz))
        if self.categories:
            self.cat_codes = self._zeros((b, len(self.categories)))
            self.cat_starts = self._zeros((len(self.categories),))
            starts = np.concatenate([[0], np.cumsum(self.categories)[:-1]])
            self.device.write(self.cat_starts, np.uint32, (len(self.categories),))[...] = starts
        self.pre = [self._zeros((b, u)) for u in self.units]
        self.h = [self._zeros((b, u)) for u in self.units]
        self.dy = [self._zeros((b, u)) for u in self.units]
        self.dpre = [self._zeros((b, u)) for u in self.units]
        self.logits, self.d_logits = self._zeros((b, c)), self._zeros((b, c))
        self.labels, self.pred = self._zeros((b, self.k)), self._zeros((b, self.k))
        # two row-index and loss buffers: a pipelined step writes one while the GPU still reads the other's step
        self._row_slots = [self._zeros((b,)) for _ in range(2)]
        self._loss_slots = [self._zeros((b, self.k)) for _ in range(2)]
        self._weight_slots = [self._zeros((b,)) for _ in range(2)]  # each row's weight in the loss: ones unweighted
        self.rows, self.loss, self.row_weights = self._row_slots[0], self._loss_slots[0], self._weight_slots[0]
        self.weight_sum: float | None = None  # the loaded batch's weight sum (None: unweighted, its rows)
        self._host_weights: NDArray[np.float32] | None = None
        self.batch = 0
        self._cache: dict[str, Any] | None = None

    def _zeros(self, shape: tuple[int, ...]) -> Buffer:
        return self.device.alloc(4 * int(np.prod(shape)))

    # -- parameters ---------------------------------------------------------------------------------------------------

    def _real_shapes(self) -> list[tuple[int, ...]]:
        """Return the unpadded shapes of the parameter arrays (`idun.realmlp.reference.Weights.arrays` order)."""
        z = self.z_width
        return [*self.shapes[:4], (z,), (z, self.units[0]), *self.shapes[6:]]

    def set_weights(self, weights: ref.Weights) -> None:
        """Copy ``weights`` (unpadded, any float dtype) into the padded buffers and reset Adam."""
        got = [tuple(a.shape) for a in weights.arrays()]
        if got != self._real_shapes() or weights.levels != self.categories:
            raise ValueError(f"weights of shapes {got} for categories {list(weights.levels)}; this network's are "
                             f"{self._real_shapes()} for {list(self.categories)}")  # fmt: skip
        for buffer, shape, array in zip(self.param, self.shapes, weights.arrays(), strict=True):
            view = self.device.write(buffer, np.float32, shape)
            view[...] = 0.0
            view[tuple(slice(0, n) for n in array.shape)] = array
        self.adam_steps = 0
        for buffer, shape in zip([*self.moment1, *self.moment2], self.shapes * 2, strict=True):
            self.device.fill(buffer, 0, 4 * int(np.prod(shape)))

    def _unpadded(self, buffers: list[Buffer]) -> ref.Weights:
        n = len(self.widths)
        arrays = [self.device.read(b, np.float32, s)[tuple(slice(0, d) for d in r)].copy()
                  for b, s, r in zip(buffers, self.shapes, self._real_shapes(), strict=True)]  # fmt: skip
        w1, b1, w2, b2, scale = arrays[:5]
        table = arrays[5 + 3 * n + 2] if self.categories else None
        return ref.Weights(w1, b1, w2, b2, scale, arrays[5 : 5 + n], arrays[5 + n : 5 + 2 * n],
                           arrays[5 + 2 * n : 5 + 3 * n], arrays[5 + 3 * n], arrays[5 + 3 * n + 1], table,
                           self.categories or None)  # fmt: skip

    def weights(self) -> ref.Weights:
        """Return the current weights, unpadded, float32."""
        return self._unpadded(self.param)

    def gradients(self) -> ref.Weights:
        """Return the gradients of the last backward pass, unpadded, float32."""
        return self._unpadded(self.grad)

    # -- one batch ----------------------------------------------------------------------------------------------------

    def _set_weights(self, n: int, weights: NDArray[np.floating] | None) -> None:
        """Write the batch's ``n`` row weights (ones for None) and their sum; padding rows weigh 0."""
        if weights is not None and np.shape(weights) != (n,):
            raise ValueError(f"need one weight per row ({n}), got shape {np.shape(weights)}")
        each = self.device.write(self.row_weights, np.float32, (self.rows_pad,))
        each[...] = 0.0
        each[:n] = 1.0 if weights is None else weights
        self._host_weights = None if weights is None else np.asarray(weights, np.float32)
        self.weight_sum = None if weights is None else float(np.sum(weights, dtype=np.float64))

    def _load(self, c: NDArray[np.floating], rows: NDArray[np.integer], y: NDArray[Any] | None,
              weights: NDArray[np.floating] | None = None) -> None:  # fmt: skip
        n = c.shape[0]
        if not 1 <= n <= self.rows_pad or c.shape[1] != self.n_inputs or rows.shape != (n,):
            raise ValueError(f"need 1..{self.rows_pad} rows of {self.n_inputs} inputs and one row index per row")
        if y is not None and np.shape(y) != ((n,) if self.n_targets is None else (n, self.k)):
            raise ValueError(f"need labels of shape {(n,) if self.n_targets is None else (n, self.k)}, got "
                             f"{np.shape(y)}")  # fmt: skip
        view = self.device.write(self.c, np.float32, (self.rows_pad, self.n_numeric))
        view[...] = 0.0
        view[:n] = c[:, : self.n_numeric]
        if self.categories:
            codes = self.device.write(self.cat_codes, np.uint32, (self.rows_pad, len(self.categories)))
            codes[...] = 0
            codes[:n] = self._codes(c[:, self.n_numeric :])
        index = self.device.write(self.rows, np.uint32, (self.rows_pad,))
        index[...] = 0
        index[:n] = rows
        kind = np.float32 if self.params.objective == "mse" else np.uint32  # targets or classes, 4 bytes each
        labels = self.device.write(self.labels, kind, (self.rows_pad, self.k))
        labels[...] = 0
        if y is not None:
            labels[:n] = np.reshape(y, (n, self.k))
        self.batch = n
        self._set_weights(n, weights)

    def _net(self, block: int = 0, step: int | None = None, t: float = 0.0, gain: float = 1.0, units: int | None = None
             ) -> RmlpParams:  # fmt: skip
        """Return the kernels' parameters; ``step`` set gives block ``block``'s dropout at training fraction ``t``.

        ``units``: the layer's units (None: the block's width). ``hidden`` is block ``block``'s width: its dropout
        mask's units, and for the output layer's kernels (block -1, the last) the width they read.
        """
        hidden = self.widths[block]
        p = self.params
        key, threshold, scale = 0, 0, 1.0
        drop = p.dropout * ref.multipliers(p, t)[1] if step is not None else 0.0
        if step is not None and drop > 0.0:
            key = rng.derive_key(p.seed, ref.DROPOUT_SALT, step, block)
            threshold = round(drop * (1 << rng.UNIFORM_BITS))
            scale = float(np.float32(1.0 / (1.0 - drop)))
        return RmlpParams(self.batch, self.rows_pad, self.n_numeric, p.frequencies, p.embed, self.ldz, hidden,
                          p.n_classes, self.k, hidden if units is None else units, int(self.own), key, threshold,
                          scale, gain, p.label_smoothing,
                          self._inv_batch(),
                          ACTIVATIONS.index(p.act))  # fmt: skip

    def _codes(self, values: NDArray[np.floating]) -> NDArray[np.uint32]:
        """Return a batch's level codes as uint32; ``ValueError`` unless each is a level of its input."""
        codes = np.asarray(values, np.float64)
        numbers = codes.astype(np.int64)
        if not np.array_equal(numbers, codes) or ((numbers < 0) | (numbers >= np.array(self.categories))).any():
            raise ValueError(f"the last {len(self.categories)} inputs must be the categorical inputs' level codes, "
                             "0 .. levels - 1")  # fmt: skip
        return numbers.astype(np.uint32)

    def _cat(self) -> CatParams:
        p = self.params
        return CatParams(self.batch, self.rows_pad, len(self.categories), p.embedding_size, self.ldz,
                         (p.embed + 1) * self.n_numeric, sum(self.categories))  # fmt: skip

    def _grid2(self, name: str, buffers: list[Buffer], params: ctypes.Structure, nx: int, ny: int) -> None:
        self.device.encode(name, buffers, params, (-(-nx // 32), -(-ny // 8), 1), (32, 8, 1))

    def _grid(self, name: str, buffers: list[Buffer], net: RmlpParams, nx: int, ny: int = 1) -> None:
        if ny == 1:
            self.device.encode(name, buffers, net, (-(-nx // 256), 1, 1), (256, 1, 1))
        else:
            self.device.encode(name, buffers, net, (-(-nx // 32), -(-ny // 8), 1), (32, 8, 1))

    def _encode_forward(self, step: int | None, t: float) -> None:
        p, n = self.params, len(self.widths)
        w1, b1, w2, b2, scale = self.param[:5]
        self._grid("rmlp_embed", [self.c, w1, b1, w2, b2, self.z], self._net(), self.n_numeric, self.rows_pad)
        if self.categories:
            table = self.param[self.out + 2]
            self._grid2("rmlp_cat_embed", [self.cat_codes, self.cat_starts, table, self.z], self._cat(),
                        len(self.categories) * p.embedding_size, self.rows_pad)  # fmt: skip
        inputs, width = self.z, self.ldz
        for i, units in enumerate(self.units):
            weight, bias, act = self.param[5 + i], self.param[5 + n + i], self.param[5 + 2 * n + i]
            net = self._net(i, step, t, 1.0 / math.sqrt(width if i else self.z_width), units)
            name = "rmlp_forward_scaled" if i == 0 else "rmlp_forward"
            # the scaled load reads R = scale as row 0 of a one-member ensemble (members 1, div 1, ldr 0)
            gemm = GemmParams(self.rows_pad, units, width, width, units, units, 1, 1, 0)
            buffers = [inputs, weight, self.pre[i], scale, bias, act, self.rows, self.h[i]]
            self.device.encode(name, buffers, RmlpFused(gemm, net), *self.GEMM.grid(units, self.rows_pad))
            inputs, width = self.h[i], self.widths[i]
        out, out_bias = self.param[self.out], self.param[self.out + 1]
        self._grid("rmlp_out_forward", [self.h[-1], out, out_bias, self.logits],
                   self._net(-1, gain=1.0 / math.sqrt(self.widths[-1]), units=self.units[-1]),
                   self.rows_pad * p.n_classes * self.k)  # fmt: skip

    def _encode_backward(self, step: int, t: float) -> None:
        p, n = self.params, len(self.widths)
        out_gain = 1.0 / math.sqrt(self.widths[-1])
        loss = "rmlp_mse" if self.params.objective == "mse" else "rmlp_loss"
        self._grid(loss, [self.logits, self.labels, self.d_logits, self.loss, self.row_weights], self._net(),
                   self.rows_pad * self.k)  # fmt: skip
        top = n - 1
        self._grid("rmlp_out_backward", [self.h[-1], self.d_logits, self.grad[self.out], self.grad[self.out + 1]],
                   self._net(top, gain=out_gain, units=self.units[top]),
                   (self.widths[top] + 1) * p.n_classes * self.k)  # fmt: skip
        top_act = self.param[5 + 2 * n + top]
        buffers = [self.d_logits, self.param[self.out], self.pre[top], top_act, self.rows, self.dy[top],
                   self.dpre[top]]  # fmt: skip
        # threadgroups of 32 units × 8 rows, as the kernel's tiles of W and d_logits assume (OB_U × OB_B)
        name = "rmlp_out_block_backward_own" if self.own else "rmlp_out_block_backward"
        self._grid(name, buffers, self._net(top, step, t, out_gain, self.units[top]), self.units[top], self.rows_pad)
        scale = self.param[4]
        for i in reversed(range(n)):
            width, units = self.ldz if i == 0 else self.widths[i - 1], self.units[i]
            gain = 1.0 / math.sqrt(self.z_width if i == 0 else self.widths[i - 1])
            weight = self.param[5 + i]
            d_weight, d_bias, d_act = self.grad[5 + i], self.grad[5 + n + i], self.grad[5 + 2 * n + i]
            self._grid("rmlp_block_sums", [self.dy[i], self.dpre[i], self.pre[i], d_bias, d_act],
                       self._net(i, units=units), units)  # fmt: skip
            inputs = self.z if i == 0 else self.h[i - 1]
            # dW = gain · inputsᵀ · dpre; layer 0 scales row r by scale[r] (members 1 selects it)
            gemm = GemmParams(width, units, self.rows_pad, width, units, units, 1 if i == 0 else 0, 1, 0)
            self.device.encode("rmlp_weight_grad", [inputs, self.dpre[i], d_weight, scale],
                              RmlpFused(gemm, self._net(gain=gain)), *self.GEMM.grid(units, width))  # fmt: skip
            # the input gradient dpre · Wᵀ · gain (an own last block's sums over its K targets' units): into block
            # i - 1's backward, or the scaled embedding's gradient
            gemm = GemmParams(self.rows_pad, width, units, units, units, width, 0, 1, 0)
            if i > 0:
                act_below = self.param[5 + 2 * n + i - 1]
                self.device.encode("rmlp_input_backward", [self.dpre[i], weight, self.pre[i - 1], scale,
                                  act_below, self.rows, self.dy[i - 1], self.dpre[i - 1]],
                                  RmlpFused(gemm, self._net(i - 1, step, t, gain, self.units[i - 1])),
                                  *self.GEMM.grid(width, self.rows_pad))  # fmt: skip
            else:
                self.device.encode("rmlp_embed_input_grad", [self.dpre[0], weight, self.dzs, scale],
                                  RmlpFused(gemm, self._net(gain=gain)),
                                  *self.GEMM.grid(width, self.rows_pad))  # fmt: skip
        self._grid("rmlp_scale_grad", [self.dzs, self.z, self.grad[4]], self._net(), self.ldz)
        w1, b1, w2 = self.param[:3]
        self.device.encode("rmlp_embed_backward", [self.c, w1, b1, w2, self.dzs, scale, *self.grad[:4]],
                          self._net(), (-(-p.frequencies // 16), -(-self.n_numeric // 16), 1), (16, 16, 1))  # fmt: skip
        if self.categories:
            d_table = self.grad[self.out + 2]
            self._grid2("rmlp_cat_backward", [self.cat_codes, self.cat_starts, self.dzs, scale, d_table],
                        self._cat(), p.embedding_size, sum(self.categories))  # fmt: skip

    def _encode_adam(self, t: float) -> None:
        p = self.params
        self.adam_steps += 1
        lr_mult, _, wd_mult = ref.multipliers(p, t)
        lr_t, wd_t = p.learning_rate * lr_mult, p.weight_decay * wd_mult
        correction1 = 1 - p.beta1**self.adam_steps
        correction2 = 1 - p.beta2**self.adam_steps
        each = ref.factors(p, len(self.widths), table=bool(self.categories))
        arrays = zip(self.param, self.grad, self.moment1, self.moment2, self.shapes, each, strict=True)
        for param, grad, m1, m2, shape, (f_lr, f_wd) in arrays:
            size = int(np.prod(shape))
            decay = 1.0 - wd_t * lr_t * f_wd * f_wd * f_lr * f_lr if f_wd != 0.0 else 1.0
            adam = AdamParams(size, decay, p.beta1, p.beta2, lr_t * f_lr / correction1, math.sqrt(correction2), p.eps,
                              1.0)  # fmt: skip
            self.device.encode("adamw", [param, grad, m1, m2], adam, (-(-size // 256), 1, 1), (256, 1, 1))

    # -- batches gathered on the GPU from the packed cache ------------------------------------------------------------

    def bind(self, data: PackedDataset, preprocess: ref.Preprocess, classes: NDArray[np.uint8] | None = None,
             codes: NDArray[np.uint8] | None = None, *,
             targets: NDArray[np.float32] | None = None) -> None:  # fmt: skip
        """Wrap ``data``'s mapped arrays without a copy, copy its codes row-major and upload ``preprocess``.

        ``codes``: ``data``'s row-major codes from `shared_codes`, wrapped instead of copied (several networks, one
        copy). Without them the network makes its own the same way and places it, so a device with memory of its own
        keeps no host copy beside its buffer. The bits are the same either way.

        From then on the ``*_rows`` methods build each batch on the GPU from its row indices alone. The cache keeps
        the codes feature-major, one row's codes in as many cache lines as there are features; the GPU reads a
        row-major copy (`inputs.codes_row_major`, ``n_rows`` × `codes_stride` bytes: 3.6 GB at v6), made on the CPU
        so the GPU never needs both copies resident (an 8 GB M1 ran out of GPU memory when it did).

        The labels come from ``classes``, a compact ``(n_rows, 1)`` uint8 array of classes (row r's at ``[r, 0]``),
        wrapped without a copy, so page-aligned with contiguous rows
        (`inputs.class_array` or a memory-mapped file); a network bound without them only predicts. The classes of the
        rows a step trains on must be below ``n_classes`` (`train_realmlp` checks its training rows;
        `inputs.MISSING_CLASS` marks a row without that label). A regression network (objective ``"mse"``) takes
        ``targets`` instead, an ``(n_rows, 1)`` float32 target array wrapped the same way (`inputs.target_array`; read
        by kernel ``rmlp_target_values``), finite on the rows a step trains on (`train_realmlp` checks them).
        """
        if self.categories:
            raise ValueError("categorical inputs train on host batches: a packed dataset has none")
        regression = self.params.objective == "mse"
        if (classes if regression else targets) is not None:
            raise ValueError(f"objective {self.params.objective!r} takes its labels as "
                             f"{'targets (float32)' if regression else 'classes (uint8)'}")  # fmt: skip
        n_named, n_obf, n_countries = len(data.named_features), len(data.obfuscated_features), len(data.countries)
        if n_named + n_obf + n_countries != self.n_inputs:
            raise ValueError(f"the data has {n_named + n_obf + n_countries} inputs; this network takes {self.n_inputs}")
        labels: NDArray[Any] | None = None
        if classes is not None:
            labels = check_classes(classes, data.n_rows)
        elif targets is not None:
            labels = inputs.checked_targets(targets, data.n_rows, 0, 0)  # shape and dtype; train_realmlp checks values
        if labels is not None and labels.shape[1] != self.k:
            name = "targets" if regression else "classes"
            raise ValueError(f"this network has {self.k} targets; {name} has {labels.shape[1]} columns")
        if labels is not None:
            target, target_ld = self._wrapped("target", labels)
        else:
            target, target_ld = None, 0
        codes_ld = codes_stride(n_obf)
        if codes is not None:
            if codes.shape != (data.n_rows, codes_ld) or codes.dtype != np.uint8:
                raise ValueError(f"codes must be shared_codes(data): ({data.n_rows}, {codes_ld}) uint8, "
                                 f"got a {codes.dtype} array of shape {codes.shape}")  # fmt: skip
        else:  # its own copy, made on the host and placed like the shared one (data the host never reaches again)
            codes = shared_codes(data)
        codes_buffer, _ = self._wrapped("codes", codes)
        named, named_ld = self._wrapped("named", data.named_values)
        country, _ = self._wrapped("country", data.country)
        stats = []
        for values in (preprocess.median, preprocess.factor):
            buffer = self._zeros((self.n_inputs,))
            self.device.write(buffer, np.float32, (self.n_inputs,))[...] = np.asarray(values, np.float32)
            stats.append(buffer)
        self._cache = {"codes": codes_buffer, "named": named, "country": country, "target": target,
                       "median": stats[0], "factor": stats[1], "n_rows": data.n_rows,
                       "shape": (n_named, n_obf, n_countries, named_ld, codes_ld, self.k, target_ld)}  # fmt: skip

    def _wrapped(self, name: str, array: NDArray[Any]) -> tuple[Buffer, int]:
        """Place ``array`` on the device as it lies; return the buffer and its row stride in elements.

        1-D, or 2-D with contiguous rows. A Metal device wraps a page-aligned memory map; ``ValueError`` when it would
        have to copy instead, a second copy in the memory the host shares (fail loudly: a copy is the caller's
        decision). A device with memory of its own uploads the array (``"uploaded"``, or ``"repacked"`` through a
        transient host copy) and keeps no host copy of it.
        """
        buffer, stride = self.device.place(name, array)
        if self.device.placements[name] == "copied":
            raise ValueError(f"{name}: the device takes a page-aligned memory-mapped array with contiguous rows (1-D "
                             "or 2-D) as it lies; this one it would copy")  # fmt: skip
        return buffer, stride // array.itemsize

    def _load_rows(
        self, rows: NDArray[np.integer], *, labels: bool = True, weights: NDArray[np.floating] | None = None
    ) -> None:
        if self._cache is None:
            raise RuntimeError("bind(data, preprocess) first")
        if labels and self._cache["target"] is None:
            raise RuntimeError(
                f"bind(data, preprocess, ...) with the labels of the network's {self.k} targets to train"
            )
        rows = np.asarray(rows)
        if (
            not 1 <= rows.size <= self.rows_pad
            or rows.ndim != 1
            or rows.min() < 0
            or rows.max() >= self._cache["n_rows"]
        ):
            raise ValueError(f"need 1..{self.rows_pad} row indices of the bound data")
        index = self.device.write(self.rows, np.uint32, (self.rows_pad,))
        index[...] = 0
        index[: rows.size] = rows
        self.batch = rows.size
        self._set_weights(rows.size, weights)

    def _encode_gather(self, *, labels: bool = True) -> None:
        cache = self._cache
        if cache is None:
            raise RuntimeError("bind(data, preprocess) first")
        params = GatherParams(self.batch, self.rows_pad, *cache["shape"], 1.0 / (self.params.clip * self.params.clip))
        self.device.encode("rmlp_gather", [cache["codes"], cache["named"], cache["country"], self.rows,
                          cache["median"], cache["factor"], self.c], params,
                          (-(-self.n_inputs // 32), -(-self.rows_pad // 8), 1), (32, 8, 1))  # fmt: skip
        if labels:
            name = "rmlp_target_values" if self.params.objective == "mse" else "rmlp_class_labels"
            self.device.encode(name, [cache["target"], self.rows, self.labels], params,
                              (-(-self.rows_pad * self.k // 256), 1, 1), (256, 1, 1))  # fmt: skip

    def _shaped(self, values: NDArray[Any]) -> NDArray[Any]:
        """Return ``values`` (rows × K) as ``(n,)`` for the single-target network, else ``(n, K)``."""
        return values[:, 0] if self.n_targets is None else values

    def gathered(self, rows: NDArray[np.integer]) -> tuple[NDArray[np.float32], NDArray[Any]]:
        """Return the preprocessed inputs and labels (``(n,)``, or ``(n, K)``) the GPU builds for ``rows`` (checks).

        The labels are uint32 classes, or float32 targets for the ``"mse"`` objective.
        """
        self._load_rows(rows)
        self._encode_gather()
        n = self.batch
        kind = np.float32 if self.params.objective == "mse" else np.uint32
        return (self.device.read(self.c, np.float32, (self.rows_pad, self.n_inputs))[:n].copy(),
                self._shaped(self.device.read(self.labels, kind, (self.rows_pad, self.k))[:n].copy()))  # fmt: skip

    def train_step_rows(self, rows: NDArray[np.integer], step: int, t: float,
                        weights: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run step ``step`` at training fraction ``t`` on the bound data's ``rows``, gathered on the GPU.

        ``weights``: each row's weight, the loss their weighted mean (None: the plain mean).
        """
        return self.train_steps_rows([(rows, step, t, weights)])[0]

    def train_steps_rows(self, steps: Iterable[tuple[Any, ...]]) -> list[float]:
        """Run each ``(rows, step, t)`` or ``(rows, step, t, weights)`` of ``steps`` as `train_step_rows` does.

        Returns the steps' (weighted) mean losses.

        Pipelined: each step's passes are submitted without waiting (`idun.device.Device.submit`), so the host
        encodes a step while the GPU runs the one before it, and reading a step's loss waits for that step only. The
        results are those of one `train_step_rows` call per step.
        """
        losses: list[float] = []
        running: tuple[Buffer, int, NDArray[np.float32] | None] | None = None  # loss buffer, size, weights on the GPU
        for i, (rows, step, t, *rest) in enumerate(steps):
            self.rows, self.loss = self._row_slots[i % 2], self._loss_slots[i % 2]  # the step before last's: done
            self.row_weights = self._weight_slots[i % 2]
            self._load_rows(rows, weights=rest[0] if rest else None)
            self._encode_gather()
            self._encode_forward(step, t)
            self._encode_backward(step, t)
            self._encode_adam(t)
            self.device.submit()
            if running is not None:
                losses.append(self._loss_of(*running))  # waits for the step before this one
            running = (self.loss, self.batch * self.k, self._host_weights)
        self.device.flush()
        if running is not None:
            losses.append(self._loss_of(*running))
        return losses

    def predict_rows(self, lo: int, hi: int) -> NDArray[np.float32]:
        """Return the expected class (``(n,)``, or ``(n, K)``) of the bound data's rows [lo, hi)."""
        out = np.empty((max(hi - lo, 0), self.k), np.float32)
        for start in range(lo, hi, self.rows_pad):
            stop = min(start + self.rows_pad, hi)
            self._load_rows(np.arange(start, stop), labels=False)
            self._encode_gather(labels=False)
            self._encode_forward(None, 0.0)
            self._encode_predict()
            predicted = self.device.read(self.pred, np.float32, (self.rows_pad, self.k))
            out[start - lo : stop - lo] = predicted[: stop - start]
        return self._shaped(out)

    def _encode_predict(self) -> None:
        name = "rmlp_predict_value" if self.params.objective == "mse" else "rmlp_predict"
        self._grid(name, [self.logits, self.pred], self._net(), self.rows_pad * self.k)

    # -- public steps -------------------------------------------------------------------------------------------------

    def forward(self, c: NDArray[np.floating], rows: NDArray[np.integer], step: int | None, t: float = 0.0
                ) -> NDArray[np.float32]:  # fmt: skip
        """Return the logits ``(B, K · C)`` (target k's classes at k · C).

        ``step`` None means inference, otherwise the dropout of step ``step`` at training fraction ``t``.
        """
        self._load(c, rows, None)
        self._encode_forward(step, t)
        width = self.params.n_classes * self.k
        return self.device.read(self.logits, np.float32, (self.rows_pad, width))[: self.batch].copy()

    def backward(self, c: NDArray[np.floating], y: NDArray[Any], rows: NDArray[np.integer], step: int,
                 t: float, weights: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run forward and backward of step ``step`` (no update); return the mean loss. See `gradients`.

        ``y``: the classes, ``(B,)`` for the single-target network, else ``(B, K)``; ``weights`` as `train_step`'s.
        """
        self._load(c, rows, y, weights)
        self._encode_forward(step, t)
        self._encode_backward(step, t)
        return self._mean_loss()

    def train_step(self, c: NDArray[np.floating], y: NDArray[Any], rows: NDArray[np.integer], step: int,
                   t: float, weights: NDArray[np.floating] | None = None) -> float:  # fmt: skip
        """Run step ``step`` at training fraction ``t`` (forward, backward, Adam) in one batch of passes.

        ``weights``: each row's weight, the loss their weighted mean (None: the plain mean).
        """
        self._load(c, rows, y, weights)
        self._encode_forward(step, t)
        self._encode_backward(step, t)
        self._encode_adam(t)
        return self._mean_loss()

    def _inv_batch(self) -> float:
        """Return 1 / (B · K), or 1 / (Σ w · K) for a weighted batch (0 when its weights sum to 0: no gradient)."""
        rows = self.batch if self.weight_sum is None else self.weight_sum
        return 1.0 / (rows * self.k) if rows > 0 else 0.0

    def _mean_loss(self) -> float:
        return self._loss_of(self.loss, self.batch * self.k, self._host_weights)

    def _loss_of(self, loss: Buffer, n: int, weights: NDArray[np.float32] | None = None) -> float:
        """Return the mean of the first ``n`` per-(row, target) losses in ``loss`` (summed in float64).

        Under ``weights``, the rows' weighted mean.
        """
        values = self.device.read(loss, np.float32, (n,))
        if weights is None:
            return float(np.sum(values, dtype=np.float64) / n)
        per_row = values.reshape(weights.size, -1).astype(np.float64).mean(axis=1)
        total = float(np.sum(weights, dtype=np.float64)) or math.inf  # a batch of weight 0: loss 0
        return float(np.sum(weights.astype(np.float64) * per_row) / total)

    def inference_logits(self, c: NDArray[np.floating]) -> NDArray[np.float32]:
        """Return the inference logits ``(B, K · C)`` of preprocessed rows (no dropout), at most `max_batch` rows."""
        return self.forward(c, np.arange(c.shape[0]), None)

    def predict(self, c: NDArray[np.floating]) -> NDArray[np.float32]:
        """Return the expected class (``(n,)``, or ``(n, K)``) of each preprocessed row, ``max_batch`` rows at once."""
        out = np.empty((c.shape[0], self.k), np.float32)
        for lo in range(0, c.shape[0], self.rows_pad):
            hi = min(lo + self.rows_pad, c.shape[0])
            self._load(c[lo:hi], np.zeros(hi - lo, np.uint32), None)
            self._encode_forward(None, 0.0)
            self._encode_predict()
            out[lo:hi] = self.device.read(self.pred, np.float32, (self.rows_pad, self.k))[: hi - lo]
        return self._shaped(out)

    def close(self) -> None:
        """Release the device and its buffers."""
        self.device.close()
