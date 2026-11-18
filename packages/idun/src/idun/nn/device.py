"""What idun_tabm and idun_realmlp share on every GPU device: the GEMM tile and the shared kernels' parameters.

The kernel sources hold the GEMM and AdamW kernels both networks use (``common.metal``) and each network's own; the
ctypes structures below mirror the shared structures field for field, on every device.
"""

import ctypes
from dataclasses import dataclass

TILE = 64  # the padding of every width and batch: whole tiles of the Metal GEMM
BK = 16  # the GEMM's k-step


@dataclass(frozen=True)
class GemmLaunch:
    """How a device's GEMM kernels are launched: the square output tile a block computes and the block's threads."""

    tile: int
    threads: int

    def grid(self, n: int, m: int, z: int = 1) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return the blocks and threads of a GEMM of ``m`` x ``n`` outputs, ``z`` times (a last tile may be cut)."""
        return (-(-n // self.tile), -(-m // self.tile), z), (self.threads, 1, 1)


METAL_GEMM = GemmLaunch(TILE, 128)  # the Metal kernels': 64 x 64 tiles, 128 threads (4 SIMD groups)


class GemmParams(ctypes.Structure):
    """Mirror of ``GemmParams``."""

    _fields_ = [(n, ctypes.c_uint32) for n in ("M", "N", "K", "lda", "ldb", "ldc", "members", "div", "ldr")]


class AdamParams(ctypes.Structure):
    """Mirror of ``AdamParams``."""

    _fields_ = [("n", ctypes.c_uint32), *((n, ctypes.c_float) for n in ("decay", "beta1", "beta2", "step_size",
                                                                          "root_c2", "eps", "grad_scale"))]  # fmt: skip


def round_up(n: int, to: int) -> int:
    """Return ``n`` rounded up to a multiple of ``to``."""
    return -(-n // to) * to
