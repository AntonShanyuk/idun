"""CudaTabM: its program and kernel table against the CUDA source, and on the emulated CUDA the Metal network's bits.

The network against the float64 reference, and run to run, is tests/unit/tabm/test_metal.py's (each test once per GPU
backend; the ``cuda`` ones where an NVIDIA GPU is).

Over the fake driver the CUDA launches run the Metal kernels of the same names, so what the emulated tests hold is the
host side on a device with memory of its own (uploads, downloads, mirrors); the CUDA kernels run where the ``cuda``
tests run.
"""

import dataclasses
import re

import numpy as np
import pytest

from idun.nn import cuda as nn_cuda
from idun.nn import device as nn_device
from idun.tabm import TabMParams, cuda, device, metal, reference, train
from tests.unit.gbm.backends import emulated

KERNEL = re.compile(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{', re.DOTALL)
ARGUMENT = re.compile(r"^(const\s+)?([\w ]+?)\s*(\*)?\s*\w+$")
TINY = TabMParams(
    members=3,
    blocks=3,
    width=40,
    dropout=0.25,
    seed=11,
    learning_rate=1e-2,
    weight_decay=0.1,
    n_classes=5,
    batch_size=4096,
    epochs=10,
)
N_INPUTS, BATCH, STEPS, SEQ_INPUTS = 37, 10, 5, 7


def declared(text: str) -> dict[str, tuple[int, tuple[int, ...], str | None]]:
    """Each kernel of a program: its buffers, the ones it may write (not ``const``) and its parameter structure."""
    found = {}
    for name, arguments in KERNEL.findall(text):
        parsed = [ARGUMENT.match(argument.strip()) for argument in arguments.split(",")]
        assert all(parsed), name
        kinds = [(bool(match.group(3)), not match.group(1), match.group(2)) for match in parsed if match]
        buffers = [writable for pointer, writable, _ in kinds if pointer]
        params = [kind_type for pointer, _, kind_type in kinds if not pointer]
        assert [pointer for pointer, *_ in kinds] == [True] * len(buffers) + [False] * len(params), name
        writes = tuple(index for index, writable in enumerate(buffers) if writable)
        found[name] = (len(buffers), writes, params[0] if params else None)
    return found


def problem(params: TabMParams = TINY, seed: int = 3, *, sequences: bool = False) -> tuple[
    reference.Weights, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray
]:  # fmt: skip
    generator = np.random.default_rng(seed)
    weights = reference.init(params, N_INPUTS, sequence_inputs=SEQ_INPUTS if sequences else 0)
    for array in weights.arrays():  # off the initial pattern, so every gradient path is exercised
        array[...] = array + generator.normal(0, 0.2, array.shape).astype(np.float32)
    x = generator.normal(size=(BATCH, N_INPUTS)).astype(np.float32)
    seq = generator.normal(size=(BATCH, STEPS, SEQ_INPUTS)).astype(np.float32) if sequences else None
    return weights, x, seq, generator.integers(0, 5, BATCH), np.arange(500, 500 + BATCH, dtype=np.int64)


def relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-30))


def test_the_program_is_the_storages_prelude_then_the_shared_kernels_and_tabms() -> None:
    shared, own = (nn_cuda.KERNELS / "common.cu").read_text(), (nn_cuda.KERNELS / "tabm.cu").read_text()
    assert cuda.source("fp32") == shared + own
    assert cuda.source("bf16") == "#define ACT_BF16 1\n" + shared + own
    with pytest.raises(ValueError, match="unknown storage 'fp16'"):
        cuda.source("fp16")


def test_the_cuda_network_launches_common_cus_gemm() -> None:
    assert cuda.CudaTabM.GEMM is nn_cuda.GEMM
    assert device.DeviceTabM.GEMM is nn_device.METAL_GEMM


def test_the_cuda_kernels_are_the_kernel_table() -> None:
    table = {
        name: (signature.buffers, signature.writes, signature.params.__name__ if signature.params else None)
        for name, signature in device.TABLE.items()
    }
    assert declared(cuda.source("fp32")) == table


@pytest.mark.metal
@pytest.mark.parametrize(
    ("storage", "encoder", "micro"), [("fp32", "none", 0), ("bf16", "none", 4), ("fp32", "gru", 4)]
)  # fmt: skip
def test_on_the_emulated_cuda_the_network_is_the_metal_ones(
    monkeypatch: pytest.MonkeyPatch, storage: str, encoder: str, micro: int
) -> None:
    params = dataclasses.replace(TINY, storage=storage, encoder=encoder, micro_batch=micro, blocks=2)
    shape = (STEPS, SEQ_INPUTS) if encoder != "none" else None
    weights, x, seq, y, rows = problem(params, sequences=shape is not None)
    emulation = emulated.install(monkeypatch, source=metal.source(storage), table=device.TABLE)
    monkeypatch.setattr(cuda.CudaTabM, "GEMM", nn_device.METAL_GEMM)  # the Metal kernels run on their own grid
    found = []
    try:
        made = train.make_backend("cuda", params, N_INPUTS, BATCH, 1, shape)
        assert isinstance(made, cuda.CudaTabM)
        made.close()
        for network in (metal.MetalTabM, cuda.CudaTabM):
            net = network(params, N_INPUTS, BATCH, sequence_shape=shape)
            net.set_weights(weights)
            losses = [net.train_step(x, y, rows, step, seq) for step in range(2)]
            losses.append(net.backward(x, y, rows, 2, seq))
            gradients = net.gradients().arrays()
            logits, predicted = net.forward(x, rows, None, seq), net.predict(x, seq)
            losses.append(net.loss_step(x, lambda out, r: (float(np.square(out).mean()), 2 * out / out.size), rows,
                                        3, seq))  # fmt: skip
            found.append((losses, [*gradients, logits, predicted, *net.weights().arrays()]))
            net.close()
    finally:
        emulation.close()
    (metal_losses, metal_arrays), (cuda_losses, cuda_arrays) = found
    assert metal_losses == cuda_losses
    for left, right in zip(metal_arrays, cuda_arrays, strict=True):
        np.testing.assert_array_equal(left, right)
