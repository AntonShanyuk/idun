"""RealMLP on Metal: `idun.realmlp.device`'s network on a Metal device, its kernels ``kernels/metal/rmlp.metal``, fp32.

`source` builds the library for the embedding's sizes; `MetalRealMLP` is `DeviceRealMLP` on a Metal device (on the
mirrored one under ``IDUN_METAL_MIRRORED=1``, `idun.metal.mirrored`). The cache's arrays are wrapped without a copy
(their files are page-aligned), and so are `shared_codes`' codes. The rest of what the network needs is in
`idun.realmlp.device` and named here too.
"""

from collections.abc import Sequence
from functools import cache

from idun.metal.mirrored import open_device
from idun.nn.metal import KERNELS
from idun.realmlp.device import (
    MAX_SMALL,
    TABLE,
    CatParams,
    DeviceRealMLP,
    GatherParams,
    RmlpFused,
    RmlpParams,
    check_classes,
    check_network,
    check_own,
    codes_stride,
    shared_codes,
)
from idun.realmlp.params import RealMLPParams

__all__ = [
    "MAX_SMALL", "TABLE", "CatParams", "GatherParams", "MetalRealMLP", "RmlpFused", "RmlpParams", "check_classes",
    "check_network", "check_own", "codes_stride", "shared_codes", "source",
]  # fmt: skip


@cache
def source(frequencies: int, embed: int) -> str:
    """Return the kernel source for ``frequencies`` cosines per input and ``embed`` values per input.

    fp32 activations, the shared kernels, then RealMLP's; the embedding kernels are specialised to the two sizes.
    """
    prelude = f"#define ACT_T float\n#define RMLP_FREQS {frequencies}\n#define RMLP_EMBED {embed}\n"
    return prelude + (KERNELS / "common.metal").read_text() + (KERNELS / "rmlp.metal").read_text()


class MetalRealMLP(DeviceRealMLP):
    """One RealMLP on the Metal GPU (`DeviceRealMLP`).

    With ``profile``, every pass runs and is timed on its own (the device's ``kernel_seconds``); results do not change.
    """

    def __init__(
        self,
        params: RealMLPParams,
        n_inputs: int,
        max_batch: int,
        n_targets: int | None = None,
        categories: Sequence[int] = (),
        *,
        profile: bool = False,
    ) -> None:
        check_network(params, max_batch, n_targets, categories, n_inputs)  # before the device is made
        device = open_device(source(params.frequencies, params.embed), TABLE, profile=profile)
        super().__init__(device, params, n_inputs, max_batch, n_targets, categories)
