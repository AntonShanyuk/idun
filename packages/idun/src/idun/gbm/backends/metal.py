"""The Metal backend: `idun.gbm.backends.device`'s passes over the kernels of ``kernels/metal/gbm.metal``.

`MetalMany` fits K models of one seed in lockstep, `MetalBackend` is its single-model face (K = 1) and
`MetalMultiTarget` shares one tree among K targets. Every pass equals the CPU backend's bit for bit.

- **Data:** the packed cache's memory-mapped columns are wrapped without a copy (their files are page-aligned,
  data-layout.md §2). Arrays that are not page-aligned memory maps (tests' in-memory panels) are copied into shared
  buffers; `placements` records which, so nothing happens silently.
- **Batches:** passes that return nothing to the host are encoded lazily; a pass whose result the host needs commits
  the open batch. A tree therefore costs about ``max_depth + 2`` command buffers, whatever K. Before the host writes a
  buffer that a pending pass uses, the batch is committed first.
"""

from pathlib import Path

from idun.device import Device
from idun.gbm.backends.device import TABLE, DeviceBackend, DeviceMany, DeviceMultiTarget
from idun.metal.mirrored import open_device

KERNELS = Path(__file__).resolve().parent.parent / "kernels" / "metal" / "gbm.metal"


def _device(*, profile: bool) -> Device:
    return open_device(KERNELS.read_text(), TABLE, profile=profile)


class MetalMany(DeviceMany):
    """K models of one seed in lockstep on the GPU (`ManyBackend`): shared bag and columns, shared code reads.

    With ``profile``, every pass is committed on its own and timed (``kernel_seconds``); results do not change.
    """

    name = "metal"

    def __init__(self, k: int, *, profile: bool = False) -> None:
        if k < 1:
            raise ValueError(f"need at least one model, got {k}")
        super().__init__(_device(profile=profile), k)


class MetalBackend(DeviceBackend):
    """The Metal backend for one model (`Backend`): `MetalMany` with K = 1. Apple7 GPUs and newer."""

    name = "metal"

    def __init__(self, *, profile: bool = False) -> None:
        super().__init__(_device(profile=profile))


class MetalMultiTarget(DeviceMultiTarget):
    """K targets sharing one tree per iteration (experimental) on the GPU; bit-identical to `CPUMultiTarget`."""

    name = "metal"

    def __init__(self, *, profile: bool = False) -> None:
        super().__init__(_device(profile=profile))
