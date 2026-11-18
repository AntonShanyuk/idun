import json
from pathlib import Path

import numpy as np
import pytest

from idun.data import PackedDataset, PackedWriter, Refit, layout

N = 37


def rows() -> dict[str, object]:
    rng = np.random.default_rng(4)
    return {"target": rng.random(N).astype(np.float32), "era": np.repeat(np.arange(N // 4 + 1), 4)[:N].astype(np.int32),
            "ticker": (np.arange(N) % 5).astype(np.int32), "tickers": ["a", "b", "c", "d", "e"],
            "country": (np.arange(N) % 3).astype(np.uint8), "countries": ["x", "y", "z"]}  # fmt: skip


def write(out: Path, block_rows: int, pieces: list[int]) -> Path:
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 5, (3, N)).astype(np.uint8)
    values = rng.normal(size=(2, N)).astype(np.float32)
    values[0, ::7] = np.nan
    with PackedWriter(out, n_rows=N, codes=["c0", "c1", "c2"], numeric=["v0", "v1"], categorical="k",
                      block_rows=block_rows) as writer:  # fmt: skip
        at = 0
        for size in pieces:
            writer.add(codes[:, at : at + size], values[:, at : at + size])
            at += size
        writer.end_features()
        writer.write_binnings([Refit("w0", 20, 20, 30), Refit("w1", 30, 30, N)], seed=3, sample_rows=16)
        writer.write_rows(**rows())  # type: ignore[arg-type]
        return writer.finish({"source_dir": str(out.parent), "sources": []})


def test_blocks_and_pieces_change_nothing(tmp_path: Path) -> None:
    first = write(tmp_path / "a", 2, [N])
    second = write(tmp_path / "b", 8, [3, 1, 20, 13])
    arrays = [json.loads((p / layout.MANIFEST).read_text())["arrays"] for p in (first, second)]
    assert arrays[0] == arrays[1]
    data = PackedDataset.open(first, binning="r1")
    assert data.n_rows == N
    assert data.named_features == ("v0", "v1")
    assert data.obfuscated_features == ("c0", "c1", "c2")
    assert data.refits[1] == Refit("w1", 30, 30, N)
    assert data.countries == ("x", "y", "z")
    assert not (tmp_path / "a.partial").exists()


def test_bad_writes_fail_loudly_and_leave_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="even"):
        PackedWriter(tmp_path / "x", n_rows=N, codes=["c"], numeric=["v"], categorical="k", block_rows=3)
    with pytest.raises(ValueError, match="needs rows"):
        PackedWriter(tmp_path / "x", n_rows=0, codes=["c"], numeric=["v"], categorical="k", block_rows=2)
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError, match="taken"):
        PackedWriter(tmp_path / "taken", n_rows=N, codes=["c"], numeric=["v"], categorical="k", block_rows=2)

    def short() -> None:
        with PackedWriter(tmp_path / "short", n_rows=N, codes=["c"], numeric=["v"], categorical="k",
                          block_rows=2) as writer:  # fmt: skip
            writer.add(np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.float32))
            writer.end_features()

    with pytest.raises(ValueError, match="1 feature rows were added for 37 rows"):
        short()
    assert not list(tmp_path.glob("short*"))
    with (
        pytest.raises(ValueError, match="one value per row"),
        PackedWriter(tmp_path / "rows", n_rows=N, codes=["c"], numeric=["v"], categorical="k", block_rows=2) as w,
    ):
        w.write_rows(**{**rows(), "era": np.zeros(3, np.int32)})  # type: ignore[arg-type]
