"""`JARFModel`: a booster with the directions and the kept features it was trained behind, for the packed data path.

The trees of `idun.gbm` read a `Front` (`idun.jarf.front`); a `JARFModel` keeps what rebuilds that front for new rows
of the same packed dataset: the `Directions`, the direction columns' edges (the booster's own), and the kept obfuscated
features. `predict` builds the front and predicts; `save` / `load` write one ``.npz`` (``idun-jarf`` v1) with the
booster's own file inside.
"""

from __future__ import annotations

import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from idun.gbm.booster import Booster
from idun.jarf import front
from idun.tabm.directions import Directions

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from idun.data import PackedDataset

FORMAT = "idun-jarf"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class JARFModel:
    """A `Booster` trained on a `Front`, with the `Directions` and the kept obfuscated features (None: all)."""

    booster: Booster
    directions: Directions
    kept: tuple[str, ...] | None = None
    prefix: str = front.PREFIX

    def __post_init__(self) -> None:
        names = self.booster.feature_names
        expected = front.direction_names(self.directions.k, self.prefix)
        if not any(names[i : i + len(expected)] == expected for i in range(len(names))):
            raise ValueError(f"the booster's features do not hold the {self.directions.k} direction columns")

    @property
    def n_named(self) -> int:
        """The named features before the direction columns in the booster's features."""
        return self.booster.feature_names.index(f"{self.prefix}_0")

    def front(self, data: PackedDataset) -> front.Front:
        """Return ``data`` extended with its projections binned at the booster's edges, reduced to the kept features."""
        if len(data.named_features) != self.n_named:
            raise ValueError(f"the data has {len(data.named_features)} named features; the model saw {self.n_named}")
        k = self.directions.k
        edges = np.asarray(self.booster.named_edges[self.n_named : self.n_named + k], np.float32)
        n_bins = tuple(int(b) for b in self.booster.named_n_bins[self.n_named : self.n_named + k])
        extended = front.apply(data, self.directions, edges, n_bins, prefix=self.prefix)
        if self.kept is None:
            return extended
        return front.keep(extended, self.kept)

    def predict(self, data: PackedDataset, *, rows: slice, backend: str) -> NDArray[np.float32]:
        """Predict ``rows`` of ``data`` through the front (the booster refuses data extended another way)."""
        return self.booster.predict(self.front(data), rows=rows, backend=backend)

    def save(self, path: str | Path) -> None:
        """Write the model to ``path`` (an uncompressed ``.npz`` without object arrays)."""
        meta: dict[str, Any] = {"format": FORMAT, "format_version": FORMAT_VERSION, "prefix": self.prefix,
                                "kept": None if self.kept is None else list(self.kept),
                                "surrogates": list(self.directions.surrogates)}  # fmt: skip
        with tempfile.TemporaryDirectory() as scratch:
            file = Path(scratch) / "booster.npz"
            self.booster.save(file)
            booster = np.frombuffer(file.read_bytes(), np.uint8)
        with Path(path).open("wb") as handle:
            np.savez(handle, allow_pickle=False, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8),
                     vectors=self.directions.vectors, eigenvalues=self.directions.eigenvalues,
                     booster=booster)  # fmt: skip

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a model written by `save`; anything else raises ``ValueError``."""
        with np.load(path, allow_pickle=False) as archive:
            if "meta" not in archive.files:
                raise ValueError(f"{path} is not a {FORMAT} file")
            meta = json.loads(archive["meta"].tobytes())
            if meta.get("format") != FORMAT or meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"{path} is not a {FORMAT} v{FORMAT_VERSION} file")
            vectors, eigenvalues, booster = archive["vectors"], archive["eigenvalues"], archive["booster"]
        with tempfile.TemporaryDirectory() as scratch:
            file = Path(scratch) / "booster.npz"
            file.write_bytes(io.BytesIO(booster.tobytes()).getvalue())
            trees = Booster.load(file)
        found = Directions(vectors, eigenvalues, tuple(meta["surrogates"]))
        kept = None if meta["kept"] is None else tuple(meta["kept"])
        return cls(trees, found, kept, meta["prefix"])
