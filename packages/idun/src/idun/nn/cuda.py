"""What idun_tabm and idun_realmlp share on CUDA: the kernel sources' directory and the GEMM's launch.

``kernels/cuda/common.cu`` holds the GEMM and AdamW kernels both networks use; ``tabm.cu`` (and RealMLP's file) hold
each network's own, the Metal kernels of the same names.
"""

from pathlib import Path

from idun.nn.device import GemmLaunch

KERNELS = Path(__file__).parent / "kernels" / "cuda"
GEMM = GemmLaunch(128, 256)  # common.cu's GEMM: 128 x 128 tiles, 256 threads, each 8 x 8 outputs
