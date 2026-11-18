import re

from idun.nn import cuda


def test_the_kernel_directory_holds_the_shared_and_tabms_sources() -> None:
    assert {p.name for p in cuda.KERNELS.glob("*.cu")} >= {"common.cu", "tabm.cu", "rmlp.cu"}


def test_the_gemm_launch_is_common_cus_tile() -> None:
    """The source's square tile, and one thread for each 8 x 8 block of it."""
    text = (cuda.KERNELS / "common.cu").read_text()
    defines = {name: int(value) for name, value in re.findall(r"#define (GM|GN|GK) (\d+)", text)}
    assert defines["GM"] == defines["GN"] == cuda.GEMM.tile == 128
    assert cuda.GEMM.threads == (defines["GM"] // 8) * (defines["GN"] // 8) == 256
    assert defines["GK"] == 8
