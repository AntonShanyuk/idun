"""TabM on CUDA: `idun.tabm.device`'s network on a CUDA device, its kernels ``kernels/cuda/{common,tabm}.cu``.

`source` builds the program for an activation storage (bf16 is stored as 16-bit words, rounded to nearest with ties to
even); `CudaTabM` is `DeviceTabM` on a CUDA device (`idun.cuda.device.CudaDevice`). The kernels are the Metal kernels
of the same names, buffers and launch shapes: the network is the same passes, within float32 error of the float64
reference and identical run to run.
"""

from functools import cache

from idun.cuda.device import CudaDevice
from idun.nn import cuda as nn_cuda
from idun.nn.cuda import KERNELS
from idun.tabm.device import TABLE, DeviceTabM, check_network
from idun.tabm.params import TabMParams

STORAGES = {"fp32": "", "bf16": "#define ACT_BF16 1\n"}  # TabMParams.storage -> the program's prelude


@cache
def source(storage: str = "fp32") -> str:
    """Return the kernel program for activation storage ``storage`` (``"fp32"`` or ``"bf16"``)."""
    if storage not in STORAGES:
        raise ValueError(f"unknown storage {storage!r}; the storages are {tuple(STORAGES)}")
    return STORAGES[storage] + (KERNELS / "common.cu").read_text() + (KERNELS / "tabm.cu").read_text()


class CudaTabM(DeviceTabM):
    """One TabM network on an NVIDIA GPU (`DeviceTabM`).

    ``device`` is the GPU's ordinal (``$IDUN_CUDA_DEVICE``, else 0). With ``profile``, the host waits for every pass and
    times it (the device's ``kernel_seconds``); results do not change.
    """

    GEMM = nn_cuda.GEMM

    def __init__(
        self,
        params: TabMParams,
        n_inputs: int,
        max_batch: int,
        targets: int = 1,
        sequence_shape: tuple[int, int] | None = None,
        bins_shape: tuple[int, int] | None = None,
        *,
        profile: bool = False,
        device: int | None = None,
    ) -> None:
        check_network(params, max_batch, targets, sequence_shape, bins_shape)  # before the device is made
        gpu = CudaDevice(source(params.storage), TABLE, profile=profile, device=device)
        super().__init__(gpu, params, n_inputs, max_batch, targets, sequence_shape, bins_shape)
