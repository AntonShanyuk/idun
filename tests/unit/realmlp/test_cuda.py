"""CudaRealMLP: its program and kernel table against the CUDA source, and on the emulated CUDA the Metal network's
bits with the bound cache.

The network against the float64 reference, and run to run, is tests/unit/realmlp/test_metal.py's (each test once per
GPU backend; the ``cuda`` ones where an NVIDIA GPU is).

Over the fake driver the CUDA launches run the Metal kernels of the same names, so what the emulated tests hold is the
host side on a device with memory of its own (uploads, repacks, downloads); the CUDA kernels run where the ``cuda``
tests run.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from idun.data import PackedDataset
from idun.nn import cuda as nn_cuda
from idun.nn import device as nn_device
from idun.nn import inputs
from idun.realmlp import RealMLPParams, cuda, device, metal
from idun.realmlp import reference as ref
from idun.realmlp import train as realmlp_train
from idun_numerai.inputs import classes as numerai_classes
from tests.unit.gbm.backends import emulated
from tests.unit.tabm.test_cuda import declared

SMALL = RealMLPParams(
    blocks=3,
    hidden=64,
    frequencies=4,
    embed=3,
    dropout=0.2,
    seed=6,
    learning_rate=0.01,
    n_classes=5,
    epochs=5,
    init_rows=3986,
)
OWN = dataclasses.replace(SMALL, own_last_block=True)
PARAMS = dataclasses.replace(SMALL, blocks=2, init_rows=80)
N_INPUTS, BATCH = 9, 20


def problem(n_targets: int | None = None, params: RealMLPParams = SMALL
            ) -> tuple[ref.Weights, np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Weights off the initial pattern, a batch, its labels ((BATCH,), or (BATCH, K) for K targets) and its rows."""
    generator = np.random.default_rng(5)
    x = generator.normal(size=(200, N_INPUTS))
    c = ref.preprocess(params, ref.fit_preprocess(x), x)
    weights = ref.init(params, c, n_targets or 1)
    for array in weights.arrays():  # off the initial pattern
        array[...] = array + generator.normal(0, 0.1, array.shape)
    y = generator.integers(0, 5, BATCH if n_targets is None else (BATCH, n_targets))
    return weights.astype(np.float32), c[:BATCH].astype(np.float32), y, np.arange(900, 900 + BATCH, dtype=np.int64)


def relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30))


def test_the_program_is_the_embeddings_sizes_then_the_shared_kernels_and_realmlps() -> None:
    shared, own = (nn_cuda.KERNELS / "common.cu").read_text(), (nn_cuda.KERNELS / "rmlp.cu").read_text()
    assert cuda.source(4, 3) == "#define RMLP_FREQS 4\n#define RMLP_EMBED 3\n" + shared + own


def test_the_cuda_network_launches_common_cus_gemm() -> None:
    assert cuda.CudaRealMLP.GEMM is nn_cuda.GEMM
    assert device.DeviceRealMLP.GEMM is nn_device.METAL_GEMM


def test_the_cuda_kernels_are_the_kernel_table() -> None:
    table = {
        name: (signature.buffers, signature.writes, signature.params.__name__ if signature.params else None)
        for name, signature in device.TABLE.items()
    }
    assert declared(cuda.source(4, 3)) == table


@pytest.mark.metal
def test_on_the_emulated_cuda_the_bound_network_is_the_metal_ones(monkeypatch: pytest.MonkeyPatch,
                                                                  nn_cache: Path) -> None:  # fmt: skip
    data = PackedDataset.open(nn_cache, binning="r1")
    n_inputs = inputs.n_inputs(data)
    pre = ref.fit_preprocess(realmlp_train.raw_inputs(data, np.arange(80)))
    weights = ref.init(PARAMS, ref.preprocess(PARAMS, pre, realmlp_train.raw_inputs(data, np.arange(80))), 2)
    labels = numerai_classes(np.stack([data.target, data.target[::-1].copy()]))
    draw = np.random.default_rng(3)
    steps = [(draw.choice(data.n_rows, 16, replace=False), step, step / 5) for step in range(5)]
    emulation = emulated.install(monkeypatch, source=metal.source(PARAMS.frequencies, PARAMS.embed),
                                 table=device.TABLE)  # fmt: skip
    monkeypatch.setattr(cuda.CudaRealMLP, "GEMM", nn_device.METAL_GEMM)  # the Metal kernels run on their own grid
    found = []
    try:
        made = realmlp_train.make_backend("cuda", PARAMS, n_inputs, 16, 2)
        assert isinstance(made, cuda.CudaRealMLP)
        made.close()
        for network in (metal.MetalRealMLP, cuda.CudaRealMLP):
            net = network(PARAMS, n_inputs, max_batch=16, n_targets=2)
            net.set_weights(weights.astype(np.float32))
            net.bind(data, pre, labels)
            losses = net.train_steps_rows(steps)
            one = net.train_step_rows(steps[0][0], 5, 0.9)
            c, y = net.gathered(steps[1][0])
            backward = net.backward(c, y, steps[1][0], 6, 0.9)
            gradients = net.gradients().arrays()
            logits = net.forward(c, steps[1][0], None)
            predicted = net.predict_rows(0, data.n_rows)
            found.append(([*losses, one, backward], [*gradients, logits, predicted, *net.weights().arrays()]))
            net.close()
    finally:
        emulation.close()
    (metal_losses, metal_arrays), (cuda_losses, cuda_arrays) = found
    assert metal_losses == cuda_losses
    for left, right in zip(metal_arrays, cuda_arrays, strict=True):
        np.testing.assert_array_equal(left, right)
