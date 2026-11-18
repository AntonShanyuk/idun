"""TabM on Metal: `idun.tabm.device`'s network on a Metal device, its kernels ``kernels/metal/{common,tabm}.metal``.

`source` builds the library for an activation storage; `MetalTabM` is `DeviceTabM` on a Metal device (on the mirrored
one under ``IDUN_METAL_MIRRORED=1``, `idun.metal.mirrored`). The rest of what the network needs is in
`idun.tabm.device` and named here too.
"""

from functools import cache

from idun.metal.mirrored import open_device
from idun.nn.metal import KERNELS
from idun.tabm.device import (
    CHUNKS,
    EMBEDDING_KERNELS,
    ENCODER_KERNELS,
    HEAD_GROUP,
    MAX_CLASSES,
    TABLE,
    DeviceTabM,
    EmbParams,
    FusedParams,
    MemberGemmParams,
    NetParams,
    SeqParams,
    Slice,
    check_network,
    from_bf16,
    to_bf16,
)
from idun.tabm.params import TabMParams

ACT_TYPES = {"fp32": "float", "bf16": "bfloat"}  # TabMParams.storage -> act_t in common.metal

__all__ = [
    "ACT_TYPES", "CHUNKS", "EMBEDDING_KERNELS", "ENCODER_KERNELS", "HEAD_GROUP", "MAX_CLASSES", "TABLE", "EmbParams",
    "FusedParams", "MemberGemmParams", "MetalTabM", "NetParams", "SeqParams", "Slice", "check_network", "from_bf16",
    "source", "to_bf16",
]  # fmt: skip


@cache
def source(storage: str = "fp32") -> str:
    """Return the kernel source for activation storage ``storage`` (``"fp32"`` or ``"bf16"``)."""
    if storage not in ACT_TYPES:
        raise ValueError(f"unknown storage {storage!r}; the storages are {tuple(ACT_TYPES)}")
    return (
        f"#define ACT_T {ACT_TYPES[storage]}\n"
        + (KERNELS / "common.metal").read_text()
        + (KERNELS / "tabm.metal").read_text()
    )


class MetalTabM(DeviceTabM):
    """One TabM network on the Metal GPU (`DeviceTabM`).

    With ``profile``, every pass runs and is timed on its own (the device's ``kernel_seconds``); results do not change.
    """

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
    ) -> None:
        check_network(params, max_batch, targets, sequence_shape, bins_shape)  # before the device is made
        device = open_device(source(params.storage), TABLE, profile=profile)
        super().__init__(device, params, n_inputs, max_batch, targets, sequence_shape, bins_shape)
