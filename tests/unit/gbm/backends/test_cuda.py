"""The CUDA backend of idun_gbm: its kernel source against the kernel table, and what is CUDA's own.

The kernels against the CPU backend, on a GPU and over the fake driver: tests/unit/gbm/backends/test_gpu.py.
"""

import re

import numpy as np
import pytest

from idun import gbm
from idun.cuda import runtime
from idun.cuda.device import CudaDevice
from idun.errors import BackendUnavailableError
from idun.gbm.backends import CPUBackend, CudaBackend, CudaMany, get_backend, get_many
from idun.gbm.backends import cuda as cuda_module
from idun.gbm.backends import device as passes
from idun.gbm.features import FeatureLayout
from idun.gbm.multi import get_multi
from tests.unit.gbm.backends import emulated
from tests.unit.gbm.tiny import make_tiny

KERNEL = re.compile(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{', re.DOTALL)
ARGUMENT = re.compile(r"^(const\s+)?([\w ]+?)\s*(\*)?\s*\w+$")


def declared() -> dict[str, tuple[int, tuple[int, ...], str | None]]:
    """Each kernel of the source: its buffers, the ones it may write (not ``const``) and its parameter structure."""
    found = {}
    for name, arguments in KERNEL.findall(cuda_module.KERNELS.read_text()):
        parsed = [ARGUMENT.match(argument.strip()) for argument in arguments.split(",")]
        assert all(parsed), name
        kinds = [(bool(match.group(3)), not match.group(1), match.group(2)) for match in parsed if match]
        buffers = [writable for pointer, writable, _ in kinds if pointer]
        params = [kind_type for pointer, _, kind_type in kinds if not pointer]
        assert [pointer for pointer, *_ in kinds] == [True] * len(buffers) + [False] * len(params), name
        assert len(params) <= 1, name
        writes = tuple(index for index, writable in enumerate(buffers) if writable)
        found[name] = (len(buffers), writes, params[0] if params else None)
    return found


def test_the_kernels_have_no_fused_multiply_add() -> None:
    """idun_gbm's bits are the CPU's: no kernel fuses a * b + c (the networks' GEMM may: their contract differs)."""
    text = cuda_module.KERNELS.read_text()
    assert not re.findall(r"\b(?:fmaf?|__fmaf?_\w+)\s*\(", text)


def test_the_cuda_kernels_are_the_kernel_table() -> None:
    table = {
        name: (signature.buffers, signature.writes, signature.params.__name__ if signature.params else None)
        for name, signature in passes.TABLE.items()
    }
    found = declared()
    assert found == table
    assert list(found) == list(table)  # in the table's order, as the Metal source is
    assert len(re.findall(r"__global__", cuda_module.KERNELS.read_text())) == len(found)  # each one is extern "C"


def test_param_structs_match_the_cuda_source() -> None:
    source = cuda_module.KERNELS.read_text()
    structs = {signature.params for signature in passes.TABLE.values()} - {None}
    assert len(structs) > 10
    for struct in structs:
        assert struct is not None
        declaration = re.search(rf"struct {struct.__name__} \{{(.*?)\}};", source, re.DOTALL)
        assert declaration, struct.__name__
        fields = re.findall(r"(\w+);", re.sub(r"//.*", "", declaration.group(1)))
        assert fields == [field[0] for field in struct._fields_], struct.__name__


def test_a_bag_of_no_models_is_refused_before_a_device_is_made() -> None:
    with pytest.raises(ValueError, match="need at least one model, got 0"):
        CudaMany(0)


def test_without_the_driver_the_backend_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise BackendUnavailableError("the CUDA driver library cannot be loaded")

    monkeypatch.setattr(runtime, "load_driver", missing)
    for make in (lambda: get_backend("cuda"), lambda: get_many("cuda", 2), lambda: get_multi("cuda")):
        with pytest.raises(BackendUnavailableError, match="cannot be loaded"):
            make()


@pytest.mark.metal
def test_a_backend_runs_on_the_gpu_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    emulation = emulated.install(monkeypatch, devices=2)
    monkeypatch.delenv(runtime.DEVICE_ENV, raising=False)
    first, second = CudaBackend(), CudaBackend(device=1)
    devices = [backend._device for backend in (first, second)]
    assert all(isinstance(device, CudaDevice) for device in devices)
    assert [device.ctx.info.ordinal for device in devices if isinstance(device, CudaDevice)] == [0, 1]
    monkeypatch.setenv(runtime.DEVICE_ENV, "1")
    third = get_backend("cuda")
    assert isinstance(third, CudaBackend)
    assert isinstance(third._device, CudaDevice)
    assert third._device.ctx.info.ordinal == 1
    with pytest.raises(ValueError, match="CUDA device 2 does not exist"):
        CudaBackend(device=2)
    for backend in (first, second, third):
        backend.close()
    emulation.close()


@pytest.mark.metal
def test_the_emulation_will_not_close_over_a_backend_left_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Else the garbage collector closes the backend later, under another test's driver (an error of that test)."""
    emulation = emulated.install(monkeypatch)
    backend = CudaBackend()
    with pytest.raises(AssertionError, match=r"the test left CUDA resources open: \{.*'contexts': 1"):
        emulation.close()
    backend.close()
    assert emulation.driver.live() == {}


@pytest.mark.metal
def test_the_single_model_pass_reads_the_cudas_rows_a_block(monkeypatch: pytest.MonkeyPatch) -> None:
    emulation = emulated.install(monkeypatch)
    data = make_tiny(9_000, n_named=2, n_obfuscated=40, seed=5)
    layout = FeatureLayout.of(data)
    features = np.arange(layout.n_features, dtype=np.int32)
    offsets, cells = layout.offsets(features)
    found = []
    shared = passes.DeviceBackend(CudaDevice(cuda_module.KERNELS.read_text(), passes.TABLE))  # the shared default
    for backend in (CudaBackend(), CPUBackend(), shared):
        backend.start(data, layout, 0, data.target[:9_000])
        backend.quantize(np.ones(9_000, np.uint8), 7, np.float32(2 / backend.gradients()), 2)
        backend.start_tree()
        found.append(np.array(backend.histograms(features, offsets, cells, np.zeros(1, np.int16))))
        if not isinstance(backend, CPUBackend):
            backend.close()
    assert cuda_module.ROWS_PER_GROUP == 4096
    one = [launch for launch in emulation.driver.launches if launch.name == "histograms_obfuscated_one"]
    assert [(launch.grid[1], launch.params.rows_per_group) for launch in one] == [(3, 4096), (1, 16352)]
    np.testing.assert_array_equal(found[0], found[1])  # the CUDA backend's blocks of 4,096 rows: the CPU's sums
    np.testing.assert_array_equal(found[2], found[1])  # the shared default's single block
    device = CudaDevice(cuda_module.KERNELS.read_text(), passes.TABLE)
    with pytest.raises(ValueError, match="at least 32"):
        passes.DeviceMany(device, 1, rows_per_group=16)
    device.close()
    emulation.close()


@pytest.mark.metal
@pytest.mark.parametrize("room", [2048, passes.SHARED_HISTOGRAM_BYTES])
def test_named_features_counted_in_shared_memory_fit_as_the_cpu(monkeypatch: pytest.MonkeyPatch, room: int) -> None:
    emulation = emulated.install(monkeypatch)
    monkeypatch.setattr(passes, "SHARED_HISTOGRAM_BYTES", room)  # 2,048 bytes: one slot a chunk, chunks per model
    monkeypatch.setattr(passes, "SHARED_ROWS_PER_GROUP", 512)  # several row groups
    data = make_tiny(3_000, n_named=3, n_obfuscated=4, n_countries=7, more_categorical=(3,), seed=8)
    params = gbm.Params(num_trees=3, max_depth=5, max_leaves=32, min_data_in_leaf=5, feature_fraction=1.0)
    ys = [data.target[:3_000], data.target[:3_000][::-1].copy()]
    found = [gbm.train_many(params, data, rows=slice(0, 3_000), backend=name, ys=ys, seeds=[0, 1])
             for name in ("cuda", "cpu")]  # fmt: skip
    for one, other in zip(*found, strict=True):
        for field in gbm.tree.FIELDS:
            np.testing.assert_array_equal(getattr(one.forest, field), getattr(other.forest, field))
    assert sum(int((model.forest.kind != 0).sum()) for model in found[1]) > 20
    launched = {launch.name for launch in emulation.driver.launches}
    assert "build_histograms_shared" in launched
    assert "build_histograms" not in launched
    emulation.close()


@pytest.mark.metal
@pytest.mark.parametrize(("n_rows", "placement"), [(160, "uploaded"), (100, "repacked")])
def test_columns_go_up_as_they_lie_when_their_rows_are_16_bytes_apart(
    monkeypatch: pytest.MonkeyPatch, n_rows: int, placement: str
) -> None:
    emulation = emulated.install(monkeypatch)
    data = make_tiny(n_rows, n_obfuscated=3, seed=5)
    layout = FeatureLayout.of(data)
    cpu, cuda = CPUBackend(), CudaBackend(profile=True)
    for backend in (cpu, cuda):
        backend.start(data, layout, 0, data.target)
    assert cuda.placements == dict.fromkeys(("named_bins", "obfuscated", "categorical"), placement)
    features = np.arange(layout.n_features, dtype=np.int32)
    offsets, cells = layout.offsets(features)
    for backend in (cpu, cuda):
        backend.quantize(np.ones(n_rows, np.uint8), 0, np.float32(0.0), 2)
    np.testing.assert_array_equal(
        cuda.histograms(features, offsets, cells, np.zeros(1, np.int16)),
        cpu.histograms(features, offsets, cells, np.zeros(1, np.int16)),
    )
    assert cuda.gpu_seconds > 0
    assert set(cuda.kernel_seconds) >= {"quantize_gradients", "build_histograms_shared"}  # CUDA counts in shared memory
    cuda.close()
    emulation.close()


@pytest.mark.cuda
def test_the_source_compiles_for_the_gpu_without_a_warning() -> None:
    gpu = CudaDevice(cuda_module.KERNELS.read_text(), passes.TABLE)
    assert gpu.warnings == ""
    gpu.close()
