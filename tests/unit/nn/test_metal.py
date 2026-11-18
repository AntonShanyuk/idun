from idun.nn import device, metal


def test_the_kernel_directory_holds_both_networks_sources() -> None:
    assert {p.name for p in metal.KERNELS.glob("*.metal")} == {"common.metal", "rmlp.metal", "tabm.metal"}


def test_the_shared_names_are_the_device_modules() -> None:
    assert (metal.GemmParams, metal.AdamParams, metal.round_up) == (device.GemmParams, device.AdamParams,
                                                                    device.round_up)  # fmt: skip
    assert (metal.TILE, metal.BK) == (device.TILE, device.BK)
