"""Per-row sequences for TabM's temporal encoder: a channel matrix and, per row, the rows its steps read.

A row's sequence has ``S`` steps, oldest first. Step ``s`` of row ``r`` reads channel row ``index[r, s]`` of
``channels`` (`idun.data.lag_index` builds such an index from a panel's times and groups), or nothing where the index
is −1: the step's channels are then 0 and its present flag 0. The channel matrix may have a row per panel row or be a
smaller table that many rows share (one row per date, say). The encoder sees each step as its ``C`` channels and the
present flag, ``C + 1`` inputs. The batches' sequences are gathered on the host, as the snapshot inputs are.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CHECK_ROWS = 1 << 20  # rows per step of the finiteness check (no full-size temporary)


@dataclass(frozen=True)
class Sequences:
    """Channels ``(m, C)`` float32 (finite) and each panel row's steps ``(n_rows, S)``, int32 rows of it (−1: none).

    ``m`` need not be ``n_rows``: a step may read any row of the channel matrix. Both are kept as given (a memory map
    stays one). Raises ``ValueError`` for other shapes or dtypes, a channel matrix without rows, a non-finite channel,
    or an index outside ``−1 .. m − 1``.
    """

    channels: NDArray[np.float32]
    index: NDArray[np.int32]

    def __post_init__(self) -> None:
        channels, index = self.channels, self.index
        if channels.ndim != 2 or channels.dtype != np.float32 or channels.shape[0] < 1 or channels.shape[1] < 1:
            raise ValueError(f"channels must be an (m, C) float32 array with m, C >= 1, got a {channels.dtype} "
                             f"array of shape {channels.shape}")  # fmt: skip
        if index.ndim != 2 or index.dtype != np.int32 or index.shape[1] < 1:
            raise ValueError(f"index must be an (n_rows, S) int32 array with S >= 1, got a {index.dtype} "
                             f"array of shape {index.shape}")  # fmt: skip
        for start in range(0, channels.shape[0], CHECK_ROWS):
            if not np.isfinite(channels[start : start + CHECK_ROWS]).all():
                raise ValueError("every channel value must be finite (set a missing value to 0)")
        if index.size and not (int(index.min()) >= -1 and int(index.max()) < channels.shape[0]):
            raise ValueError(f"the index must lie in -1 .. {channels.shape[0] - 1}")

    @property
    def n_rows(self) -> int:
        """Rows of the panel."""
        return int(self.index.shape[0])

    @property
    def steps(self) -> int:
        """S: the steps of every row's sequence."""
        return int(self.index.shape[1])

    @property
    def n_channels(self) -> int:
        """C: the channels of a step."""
        return int(self.channels.shape[1])

    @property
    def inputs(self) -> int:
        """C + 1: a step's inputs to the encoder, its channels and the present flag."""
        return self.n_channels + 1

    def gather(self, rows: NDArray[np.integer]) -> NDArray[np.float32]:
        """Return the ``(len(rows), S, C + 1)`` float32 sequences of ``rows``: each step's channels, then its flag."""
        rows = np.asarray(rows, np.int64)
        if rows.ndim != 1 or (rows.size and not (rows.min() >= 0 and rows.max() < self.n_rows)):
            raise ValueError(f"rows must be a 1-D array of row indices in [0, {self.n_rows})")
        steps = np.asarray(self.index[rows], np.int64)
        present = steps >= 0
        out = np.zeros((rows.size, self.steps, self.inputs), np.float32)
        out[..., : self.n_channels] = np.asarray(self.channels)[np.where(present, steps, 0)]
        out[~present] = 0.0
        out[..., self.n_channels] = present
        return out
