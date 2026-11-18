"""Metal backend: what is Metal's own (the kernels against the CPU backend: tests/unit/gbm/backends/test_gpu.py)."""

from pathlib import Path

import numpy as np
import pytest

from idun.data.layout import pack_nibbles
from idun.gbm.backends import CPUBackend, MetalBackend, get_backend, get_many
from idun.gbm.backends import metal as metal_module
from idun.gbm.features import FeatureLayout
from idun.gbm.multi import get_multi
from tests.unit.gbm.tiny import make_tiny

pytestmark = pytest.mark.metal


def test_a_bag_of_no_models_is_refused_before_a_device_is_made() -> None:
    with pytest.raises(ValueError, match="need at least one model, got 0"):
        metal_module.MetalMany(0)


def test_the_backends_by_name() -> None:
    assert isinstance(get_backend("metal"), MetalBackend)
    assert isinstance(get_many("metal", 2), metal_module.MetalMany)
    assert isinstance(get_multi("metal"), metal_module.MetalMultiTarget)


def test_memory_mapped_columns_are_wrapped_and_arrays_copied(tmp_path: Path) -> None:
    data = make_tiny(100, n_obfuscated=3, seed=5)
    page = 16 * 1024
    path = tmp_path / "packed.u4"
    np.zeros(3 * page, np.uint8).tofile(path)
    mapped = np.memmap(path, np.uint8, "r+", shape=(3, page))
    mapped[:, :50] = pack_nibbles(data.codes)
    data.obfuscated = np.memmap(path, np.uint8, "r", shape=(3, page))[:, :50]
    cpu, metal = CPUBackend(), MetalBackend()
    layout = FeatureLayout.of(data)
    for backend in (cpu, metal):
        backend.start(data, layout, 0, data.target)
    assert metal.placements == {"named_bins": "copied", "obfuscated": "wrapped", "categorical": "copied"}
    features = np.arange(layout.n_features, dtype=np.int32)
    offsets, cells = layout.offsets(features)
    for backend in (cpu, metal):
        backend.quantize(np.ones(100, np.uint8), 0, np.float32(0.0), 2)
    np.testing.assert_array_equal(
        metal.histograms(features, offsets, cells, np.zeros(1, np.int16)),
        cpu.histograms(features, offsets, cells, np.zeros(1, np.int16)),
    )
    assert metal.gpu_seconds > 0
