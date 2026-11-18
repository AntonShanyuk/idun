"""The CUDA backend: `idun.gbm.backends.device`'s passes over the kernels of ``kernels/cuda/gbm.cu``.

`CudaMany` fits K models of one seed in lockstep, `CudaBackend` is its single-model face (K = 1) and
`CudaMultiTarget` shares one tree among K targets. A kernel has the name, the buffers and the parameter structure
that the kernel table declares and mirrors the Metal kernel of that name operation for operation, so every pass
equals the CPU backend's bit for bit. NVIDIA GPUs on Linux, through the driver library and NVRTC
(`idun.cuda.runtime`).

- **Data:** the panel's columns are uploaded once per fit and stay on the GPU; `placements` records how each went
  up. The host reaches the fit's state (predictions, gradients, histograms) through page-locked copies, and only
  the bytes a pass changed cross again.
- **Order:** passes queue on the device's stream; the host waits only where it needs a result.
"""

from pathlib import Path

from idun.cuda.device import CudaDevice
from idun.device import Device
from idun.gbm.backends.device import TABLE, DeviceBackend, DeviceMany, DeviceMultiTarget

KERNELS = Path(__file__).resolve().parent.parent / "kernels" / "cuda" / "gbm.cu"
# Rows a block of the single-model obfuscated histogram pass reads: on a Tesla T4 whole fits ran 5 % faster with
# 4,096 than with the shared default of 16,384 (more blocks for its 40 SMs); the sums are the same either way.
ROWS_PER_GROUP = 4096
# The named and categorical features counted in shared memory first (build_histograms_shared), not row by row with
# atomics in the histogram.
SHARED_NAMED = True


def _device(*, profile: bool, device: int | None) -> Device:
    return CudaDevice(KERNELS.read_text(), TABLE, profile=profile, device=device)


class CudaMany(DeviceMany):
    """K models of one seed in lockstep on the GPU (`ManyBackend`): shared bag and columns, shared code reads.

    ``device`` is the GPU's ordinal (``$IDUN_CUDA_DEVICE``, else 0). With ``profile``, the host waits for every pass
    and times it (``kernel_seconds``); results do not change.
    """

    name = "cuda"

    def __init__(self, k: int, *, profile: bool = False, device: int | None = None) -> None:
        if k < 1:
            raise ValueError(f"need at least one model, got {k}")
        super().__init__(
            _device(profile=profile, device=device), k, rows_per_group=ROWS_PER_GROUP, shared_named=SHARED_NAMED
        )


class CudaBackend(DeviceBackend):
    """The CUDA backend for one model (`Backend`): `CudaMany` with K = 1."""

    name = "cuda"

    def __init__(self, *, profile: bool = False, device: int | None = None) -> None:
        super().__init__(
            _device(profile=profile, device=device), rows_per_group=ROWS_PER_GROUP, shared_named=SHARED_NAMED
        )


class CudaMultiTarget(DeviceMultiTarget):
    """K targets sharing one tree per iteration (experimental) on the GPU; bit-identical to `CPUMultiTarget`."""

    name = "cuda"

    def __init__(self, *, profile: bool = False, device: int | None = None) -> None:
        super().__init__(_device(profile=profile, device=device))
