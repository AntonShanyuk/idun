"""What idun_tabm and idun_realmlp share on Metal: the kernel sources' directory (`idun.nn.device` the rest).

``kernels/metal/common.metal`` holds the GEMM and AdamW kernels both networks use; ``tabm.metal`` and ``rmlp.metal``
hold each network's own.
"""

from pathlib import Path

from idun.nn.device import BK, TILE, AdamParams, GemmParams, round_up

KERNELS = Path(__file__).parent / "kernels" / "metal"

__all__ = ["BK", "KERNELS", "TILE", "AdamParams", "GemmParams", "round_up"]
