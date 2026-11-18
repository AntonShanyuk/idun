"""DeviceTabM: its kernel table against the Metal sources, and the same bits on a device with memory of its own.

The network's arithmetic against the float64 reference is tests/unit/tabm/test_metal.py's (on the plain device, and
on the mirrored one under ``make test-metal-mirrored``).
"""

import dataclasses
import re

import numpy as np
import pytest

from idun.metal.device import MetalDevice
from idun.metal.mirrored import MirroredDevice
from idun.nn.metal import KERNELS
from idun.tabm import TabMParams, device, metal, reference

KERNEL = re.compile(r"kernel void (\w+)\((.*?)\)\s*\{", re.DOTALL)
BOUND = re.compile(r"(device|constant)\s+(const\s+)?(\w+)\s*[*&]\s*\w+\s*\[\[buffer\((\d+)\)\]\]")
TINY = TabMParams(members=3, blocks=2, width=40, dropout=0.25, seed=11, learning_rate=1e-2, weight_decay=0.1,
                  micro_batch=4, n_classes=5, batch_size=4096, epochs=10)  # fmt: skip
N_INPUTS, BATCH, STEPS, SEQ_INPUTS = 37, 10, 5, 7


def test_the_kernel_table_matches_the_metal_sources() -> None:
    """Each kernel's buffers, the ones it may write (not ``const``) and its parameter structure, as declared."""
    declared = {}
    for name in ("common.metal", "tabm.metal"):
        for kernel, arguments in KERNEL.findall((KERNELS / name).read_text()):
            bound = [
                (kind, not const, kind_type, int(index)) for kind, const, kind_type, index in BOUND.findall(arguments)
            ]
            assert [index for *_, index in bound] == list(range(len(bound))), kernel
            buffers = [writable for kind, writable, _, _ in bound if kind == "device"]
            params = [kind_type for kind, _, kind_type, _ in bound if kind == "constant"]
            writes = tuple(index for index, writable in enumerate(buffers) if writable)
            declared[kernel] = (len(buffers), writes, params[0] if params else None)
    table = {
        name: (signature.buffers, signature.writes, signature.params.__name__ if signature.params else None)
        for name, signature in device.TABLE.items()
    }
    assert table == declared
    assert set(device.ENCODER_KERNELS) <= set(table)


def problem(params: TabMParams, sequences: bool) -> tuple[reference.Weights, np.ndarray, np.ndarray | None,
                                                          np.ndarray, np.ndarray]:  # fmt: skip
    generator = np.random.default_rng(7)
    weights = reference.init(params, N_INPUTS, sequence_inputs=SEQ_INPUTS if sequences else 0)
    for array in weights.arrays():
        array[...] = array + generator.normal(0, 0.2, array.shape).astype(np.float32)
    x = generator.normal(size=(BATCH, N_INPUTS)).astype(np.float32)
    seq = generator.normal(size=(BATCH, STEPS, SEQ_INPUTS)).astype(np.float32) if sequences else None
    return weights, x, seq, generator.integers(0, 5, BATCH), np.arange(100, 100 + BATCH, dtype=np.int64)


@pytest.mark.metal
@pytest.mark.parametrize(("storage", "encoder"), [("fp32", "none"), ("bf16", "none"), ("fp32", "gru")])
def test_a_device_with_memory_of_its_own_gives_the_same_bits(storage: str, encoder: str) -> None:
    params = dataclasses.replace(TINY, storage=storage, encoder=encoder)
    shape = (STEPS, SEQ_INPUTS) if encoder != "none" else None
    weights, x, seq, y, rows = problem(params, shape is not None)
    found = []
    for mirrored in (False, True):
        inner = MetalDevice(metal.source(storage), device.TABLE)
        net = device.DeviceTabM(MirroredDevice(inner, device.TABLE) if mirrored else inner, params, N_INPUTS, BATCH,
                                sequence_shape=shape)  # fmt: skip
        net.set_weights(weights)
        losses = [net.train_step(x, y, rows, step, seq) for step in range(2)]
        backward = net.backward(x, y, rows, 2, seq)
        gradients = net.gradients().arrays()
        logits = net.forward(x, rows, None, seq)
        predicted = net.predict(x, seq)
        loss = net.loss_step(x, lambda out, r: (float(np.square(out).mean()), 2 * out / out.size), rows, 3, seq)
        state = net.training_state()
        found.append((losses, backward, gradients, logits, predicted, loss, state[0].arrays(), state[1].m))
        net.close()
    plain, own = found
    assert plain[0] == own[0]
    assert plain[1] == own[1]
    assert plain[5] == own[5]
    for left, right in zip((*plain[2], plain[3], plain[4], *plain[6], *plain[7]),
                           (*own[2], own[3], own[4], *own[6], *own[7]), strict=True):  # fmt: skip
        np.testing.assert_array_equal(left, right)
