"""`PackedWriter`: writes the packed on-disk dataset that `PackedDataset` reads (docs/architecture/data-layout.md §2).

The writer knows nothing of where the rows come from. Its caller gives the row count and the feature names, then the
feature columns in row blocks (`add`: 4-bit codes and raw numeric values), which are staged column-major and written
with ``pwrite`` at each column's offset, so a table larger than memory streams through. `write_rows` writes the per-row
arrays (target, a sorted time key, an entity id and one categorical column), `write_binnings` fits each window's numeric
edges on a seeded sample of its training rows and bins every row, and `finish` checksums every array, writes the
manifest and renames ``<out>.partial`` to ``<out>``. An exception inside the ``with`` block removes the partial
directory.
"""

import contextlib
import json
import os
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from idun.data import binning, layout
from idun.data.manifest import sha256_file, write_manifest
from idun.data.refits import Refit


class _ColumnFile:
    """A column-major output file of ``columns * stride`` bytes, written with ``pwrite`` at column offsets."""

    def __init__(self, path: Path, columns: int, stride: int) -> None:
        self._stride = stride
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.ftruncate(self._fd, columns * stride)

    def write(self, column: int, offset: int, data: NDArray[Any]) -> None:
        view = memoryview(np.ascontiguousarray(data)).cast("B")
        position = column * self._stride + offset
        while view:
            written = os.pwrite(self._fd, view, position)
            view, position = view[written:], position + written

    def close(self) -> None:
        os.close(self._fd)


class PackedWriter:
    """Write one packed dataset into ``out`` (see the module docstring); use it as a context manager.

    Parameters
    ----------
    out
        Must not exist yet, nor ``<out>.partial``, where the dataset is assembled.
    n_rows
        Rows of the table.
    codes, numeric
        Names of the 4-bit code columns (values 0..4) and of the numeric columns, in their stored order.
    categorical
        Name of the one categorical column.
    block_rows
        Rows per staged write block (even). It changes speed and memory, never the output.
    """

    def __init__(
        self, out: str | Path, *, n_rows: int, codes: Sequence[str], numeric: Sequence[str], categorical: str,
        block_rows: int,
    ) -> None:  # fmt: skip
        if block_rows < 2 or block_rows % 2:
            raise ValueError(f"block_rows must be even and >= 2, got {block_rows}")
        if n_rows < 1 or not codes or not numeric:
            raise ValueError("a packed dataset needs rows, code columns and numeric columns")
        self.out = Path(out)
        self.partial = self.out.with_name(self.out.name + ".partial")
        for path in (self.out, self.partial):
            if path.exists():
                raise FileExistsError(f"{path} exists; delete it to rebuild the dataset")
        self.n_rows, self.codes, self.numeric, self.categorical = n_rows, tuple(codes), tuple(numeric), categorical
        self.strides = {
            "obfuscated": layout.column_stride(layout.packed_bytes(n_rows)),
            "named_values": layout.column_stride(4 * n_rows),
            "named_bins": layout.column_stride(n_rows),
        }
        self._block = (np.empty((len(codes), block_rows), np.uint8), np.empty((len(numeric), block_rows), np.float32))
        self._files: tuple[_ColumnFile, ...] = ()
        self._filled = 0
        self.flushed = 0
        self._manifest: dict[str, Any] = {}

    def __enter__(self) -> Self:
        self.partial.mkdir(parents=True)
        self._files = (
            _ColumnFile(self.partial / layout.OBFUSCATED_FILE, len(self.codes), self.strides["obfuscated"]),
            _ColumnFile(self.partial / layout.NAMED_VALUES_FILE, len(self.numeric), self.strides["named_values"]),
        )
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self._close_features()
        if kind is not None:
            shutil.rmtree(self.partial, ignore_errors=True)

    def _close_features(self) -> None:
        for file in self._files:
            file.close()
        self._files = ()

    def add(self, codes: NDArray[np.uint8], values: NDArray[np.float32]) -> None:
        """Append rows: ``codes`` ``(len(codes), k)`` uint8 in 0..4 and ``values`` ``(len(numeric), k)`` float32."""
        start = 0
        block_codes, block_values = self._block
        while start < codes.shape[1]:
            take = min(block_codes.shape[1] - self._filled, codes.shape[1] - start)
            block_codes[:, self._filled : self._filled + take] = codes[:, start : start + take]
            block_values[:, self._filled : self._filled + take] = values[:, start : start + take]
            self._filled, start = self._filled + take, start + take
            if self._filled == block_codes.shape[1]:
                self._flush()

    def _flush(self) -> None:
        block_codes, block_values = self._block
        packed = layout.pack_nibbles(block_codes[:, : self._filled])
        codes_file, values_file = self._files
        for column in range(packed.shape[0]):
            codes_file.write(column, self.flushed // 2, packed[column])
        for column in range(block_values.shape[0]):
            values_file.write(column, 4 * self.flushed, block_values[column, : self._filled])
        self.flushed += self._filled
        self._filled = 0

    def end_features(self) -> None:
        """Write the last partial block and close the feature files; every row must have been added."""
        self._flush()
        self._close_features()
        if self.flushed != self.n_rows:
            raise ValueError(f"{self.flushed} feature rows were added for {self.n_rows} rows")

    def write_rows(
        self, *, target: NDArray[np.float32], era: NDArray[np.int32], ticker: NDArray[np.int32],
        tickers: Sequence[str], country: NDArray[np.uint8], countries: Sequence[str],
    ) -> None:  # fmt: skip
        """Write the per-row arrays: the target, the sorted time key, the entity id and the categorical column."""
        for name, array in (("target", target), ("era", era), ("ticker", ticker), ("country", country)):
            if array.shape != (self.n_rows,):
                raise ValueError(f"{name} must have one value per row ({self.n_rows}), got shape {array.shape}")
        target.tofile(self.partial / layout.TARGET_FILE)
        era.tofile(self.partial / layout.ERA_FILE)
        ticker.tofile(self.partial / layout.TICKER_FILE)
        country.tofile(self.partial / layout.COUNTRY_FILE)
        (self.partial / layout.TICKERS_FILE).write_text(json.dumps(list(tickers)) + "\n")
        self._manifest["countries"] = list(countries)

    def write_binnings(self, windows: Sequence[Refit], *, seed: int, sample_rows: int) -> None:
        """Per window: fit the numeric edges on a seeded sample of its training rows, then bin every row.

        Columns are read back with ``pread`` (one at a time, for every window) rather than mapped, so the writer's
        resident memory does not grow by the mapped raw values.
        """
        names = [layout.binning_name(index) for index in range(len(windows))]
        samples = [binning.sample_rows(w.train_hi, sample_rows, seed=seed, refit=i) for i, w in enumerate(windows)]
        edges: list[list[NDArray[np.float32]]] = [[] for _ in windows]
        n = self.n_rows
        with contextlib.ExitStack() as stack:
            bins = [_ColumnFile(self.partial / layout.named_bins_file(name), len(self.numeric),
                                self.strides["named_bins"]) for name in names]  # fmt: skip
            for file in bins:
                stack.callback(file.close)
            source = os.open(self.partial / layout.NAMED_VALUES_FILE, os.O_RDONLY)
            stack.callback(os.close, source)
            for j in range(len(self.numeric)):
                raw = os.pread(source, 4 * n, j * self.strides["named_values"])
                if len(raw) != 4 * n:
                    raise OSError(f"short read of {layout.NAMED_VALUES_FILE} column {j}")
                values = np.frombuffer(raw, np.float32)
                for index, sample in enumerate(samples):
                    edges[index].append(binning.quantile_edges(values[sample]))
                    bins[index].write(j, 0, binning.apply_bins(values, binning.edge_table([edges[index][j]])[0]))
        binnings = {}
        for index, name in enumerate(names):
            binning.edge_table(edges[index]).tofile(self.partial / layout.named_edges_file(name))
            binnings[name] = {"refit": index, "sample_rows": int(samples[index].size),
                              "n_bins": [int(e.size) + 1 for e in edges[index]]}  # fmt: skip
        self._manifest |= {"seed": seed, "sample_rows": sample_rows, "refits": [asdict(w) for w in windows],
                           "binnings": binnings}  # fmt: skip

    def finish(self, extra: dict[str, Any]) -> Path:
        """Checksum every array, write the manifest (with the caller's ``extra`` entries) and publish the dataset."""
        arrays = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(self.partial.iterdir())
            if path.name != layout.MANIFEST
        }
        manifest = {**extra, **self._manifest, "n_rows": self.n_rows, "strides": self.strides,
                    "features": {"named": list(self.numeric), "obfuscated": list(self.codes),
                                 "country": self.categorical},
                    "arrays": arrays}  # fmt: skip
        write_manifest(self.partial, manifest)
        self.partial.rename(self.out)
        return self.out
