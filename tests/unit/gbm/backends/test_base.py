import numpy as np

from idun.gbm.backends import BACKENDS, Backend, CPUBackend, LevelSplits
from idun.gbm.backends.base import Lockstep


def test_the_cpu_backend_implements_the_protocol() -> None:
    backend: Backend = CPUBackend()  # checked by mypy
    assert backend.name == "cpu"
    assert "cpu" in BACKENDS


def test_level_splits_hold_one_entry_per_node() -> None:
    level = LevelSplits(
        kind=np.zeros(4, np.uint8),
        feature=np.zeros(4, np.int32),
        threshold=np.zeros(4, np.uint8),
        missing_left=np.zeros(4, np.uint8),
        cat_left=np.zeros(4, np.uint64),
    )
    assert level.kind.shape == level.cat_left.shape == (4,)


def test_lockstep_asks_a_backend_for_hessians_only_on_their_pass() -> None:
    class Plain:
        """A backend written before the hessians' pass: its histograms take no keyword."""

        name = "plain"

        def histograms(self, features: object, offsets: object, cells: int, slots: object) -> str:
            return "built"

    class Newer(Plain):
        def histograms(
            self, features: object, offsets: object, cells: int, slots: object, *, hessians: bool = False
        ) -> str:
            return "hessians" if hessians else "built"

    group = Lockstep([Plain(), Newer()])  # type: ignore[list-item]
    assert group.histograms(None, None, 0, [None, None]) == ["built", "built"]  # type: ignore[arg-type]
    assert Lockstep([Newer()]).histograms(None, None, 0, [None], hessians=True) == ["hessians"]  # type: ignore[arg-type,list-item]
