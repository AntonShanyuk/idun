import ctypes

from idun.nn import device


def test_struct_sizes_match_the_kernel_sources() -> None:
    assert ctypes.sizeof(device.GemmParams) == 36
    assert ctypes.sizeof(device.AdamParams) == 32


def test_round_up_rounds_to_a_multiple() -> None:
    assert [device.round_up(n, 64) for n in (0, 1, 64, 65)] == [0, 64, 64, 128]
    assert device.TILE % device.BK == 0


def test_a_gemm_launch_covers_its_outputs_in_whole_or_cut_tiles() -> None:
    assert device.GemmLaunch(device.TILE, 128) == device.METAL_GEMM
    assert device.METAL_GEMM.grid(512, 4096) == ((8, 64, 1), (128, 1, 1))
    wide = device.GemmLaunch(128, 256)
    assert wide.grid(1088, 192, 3) == ((9, 2, 3), (256, 1, 1))  # the last tiles cut: 1,088 = 8.5 x 128
