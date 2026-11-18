"""DeviceRealMLP: its kernel table against the Metal sources, and the same bits on a device with memory of its own.

The network's arithmetic against the float64 reference is tests/unit/realmlp/test_metal.py's (on the plain device, and
on the mirrored one under ``make test-metal-mirrored``).
"""

import mmap
import re
from pathlib import Path

import numpy as np
import pytest

from idun.data import PackedDataset
from idun.metal.device import MetalDevice
from idun.metal.mirrored import MirroredDevice
from idun.nn import inputs
from idun.nn.metal import KERNELS
from idun.realmlp import RealMLPParams, device, metal
from idun.realmlp import reference as ref
from idun.realmlp import train as realmlp_train
from idun_numerai.inputs import classes as numerai_classes

KERNEL = re.compile(r"kernel void (\w+)\((.*?)\)\s*\{", re.DOTALL)
BOUND = re.compile(r"(device|constant)\s+(const\s+)?(\w+)\s*[*&]\s*\w+\s*\[\[buffer\((\d+)\)\]\]")
PARAMS = RealMLPParams(blocks=2, hidden=64, frequencies=4, embed=3, dropout=0.2, seed=6, learning_rate=0.01,
                       init_rows=80, n_classes=5, epochs=5)  # fmt: skip


def test_the_kernel_table_matches_the_metal_sources() -> None:
    """Each kernel's buffers, the ones it may write (not ``const``) and its parameter structure, as declared."""
    declared = {}
    for name in ("common.metal", "rmlp.metal"):
        for kernel, arguments in KERNEL.findall((KERNELS / name).read_text()):
            bound = [(kind, not const, kind_type, int(i)) for kind, const, kind_type, i in BOUND.findall(arguments)]
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


def test_shared_codes_are_the_row_major_codes_on_whole_os_pages(nn_cache: Path) -> None:
    """On any host: the copy starts on a page of the OS (4 KiB on x86-64 Linux, 16 KiB on Apple silicon)."""
    data = PackedDataset.open(nn_cache, binning="r1")
    codes = device.shared_codes(data)
    want = np.zeros_like(codes)
    inputs.codes_row_major(data, want)
    np.testing.assert_array_equal(codes, want)
    assert codes.__array_interface__["data"][0] % mmap.PAGESIZE == 0
    assert codes.shape == (data.n_rows, device.codes_stride(len(data.obfuscated_features)))


@pytest.mark.metal
def test_a_device_with_memory_of_its_own_gives_the_same_bits(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    n_inputs = inputs.n_inputs(data)
    pre = ref.fit_preprocess(realmlp_train.raw_inputs(data, np.arange(80)))
    weights = ref.init(PARAMS, ref.preprocess(PARAMS, pre, realmlp_train.raw_inputs(data, np.arange(80))), 2)
    labels = numerai_classes(np.stack([data.target, data.target[::-1].copy()]))
    draw = np.random.default_rng(3)
    steps = [(draw.choice(data.n_rows, 16, replace=False), step, step / 5) for step in range(5)]
    found = []
    for mirrored in (False, True):
        inner = MetalDevice(metal.source(PARAMS.frequencies, PARAMS.embed), device.TABLE)
        net = device.DeviceRealMLP(MirroredDevice(inner, device.TABLE) if mirrored else inner, PARAMS, n_inputs,
                                   max_batch=16, n_targets=2)  # fmt: skip
        net.set_weights(weights.astype(np.float32))
        net.bind(data, pre, labels)
        losses = net.train_steps_rows(steps)  # submitted without waiting, one step after another
        one = net.train_step_rows(steps[0][0], 5, 0.9)
        c, y = net.gathered(steps[1][0])
        backward = net.backward(c, y, steps[1][0], 6, 0.9)
        gradients = net.gradients().arrays()
        logits = net.forward(c, steps[1][0], None)
        predicted = net.predict_rows(0, data.n_rows)
        found.append(([*losses, one, backward], [*gradients, logits, predicted, *net.weights().arrays()]))
        net.close()
    (plain_losses, plain_arrays), (own_losses, own_arrays) = found
    assert plain_losses == own_losses
    for left, right in zip(plain_arrays, own_arrays, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.metal
def test_an_array_the_device_would_copy_is_refused(nn_cache: Path) -> None:
    data = PackedDataset.open(nn_cache, binning="r1")
    pre = ref.fit_preprocess(realmlp_train.raw_inputs(data, np.arange(80)))
    net = metal.MetalRealMLP(PARAMS, inputs.n_inputs(data), max_batch=16, n_targets=1)
    in_memory = np.asarray(numerai_classes(data.target)).copy()  # not page-aligned anonymous memory
    with pytest.raises(ValueError, match="page-aligned memory-mapped"):
        net.bind(data, pre, in_memory)
    assert net.device.placements["target"] == "copied"
    net.close()
