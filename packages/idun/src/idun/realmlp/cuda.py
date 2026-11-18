"""RealMLP on CUDA: `idun.realmlp.device`'s network on a CUDA device, its kernels ``kernels/cuda/{common,rmlp}.cu``.

`source` builds the program for the embedding's sizes; `CudaRealMLP` is `DeviceRealMLP` on a CUDA device
(`idun.cuda.device.CudaDevice`). The cache's arrays are uploaded as they lie (or repacked through a transient host
copy) and the device keeps no host copy of them. The kernels are the Metal kernels of the same names, buffers and
launch shapes: the network is the same passes, within float32 error of the float64 reference and identical run to run.
"""

from collections.abc import Sequence
from functools import cache

from idun.cuda.device import CudaDevice
from idun.nn import cuda as nn_cuda
from idun.nn.cuda import KERNELS
from idun.realmlp.device import TABLE, DeviceRealMLP, check_network
from idun.realmlp.params import RealMLPParams


@cache
def source(frequencies: int, embed: int) -> str:
    """Return the kernel program for ``frequencies`` cosines per input and ``embed`` values per input.

    fp32 activations, the shared kernels, then RealMLP's; the embedding kernels are specialised to the two sizes.
    """
    prelude = f"#define RMLP_FREQS {frequencies}\n#define RMLP_EMBED {embed}\n"
    return prelude + (KERNELS / "common.cu").read_text() + (KERNELS / "rmlp.cu").read_text()


class CudaRealMLP(DeviceRealMLP):
    """One RealMLP on an NVIDIA GPU (`DeviceRealMLP`).

    ``device`` is the GPU's ordinal (``$IDUN_CUDA_DEVICE``, else 0). With ``profile``, the host waits for every pass and
    times it (the device's ``kernel_seconds``); results do not change.
    """

    GEMM = nn_cuda.GEMM

    def __init__(
        self,
        params: RealMLPParams,
        n_inputs: int,
        max_batch: int,
        n_targets: int | None = None,
        categories: Sequence[int] = (),
        *,
        profile: bool = False,
        device: int | None = None,
    ) -> None:
        check_network(params, max_batch, n_targets, categories, n_inputs)  # before the device is made
        gpu = CudaDevice(source(params.frequencies, params.embed), TABLE, profile=profile, device=device)
        super().__init__(gpu, params, n_inputs, max_batch, n_targets, categories)
