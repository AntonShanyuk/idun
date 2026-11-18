from pathlib import Path

import numpy as np
import pytest

from idun.tabm import Sequences
from idun.tabm import sequences as seq_module


def panel() -> Sequences:
    channels = np.arange(12, dtype=np.float32).reshape(4, 3) + 1.0  # row r: 3r + 1, 3r + 2, 3r + 3
    index = np.array([[-1, 0], [0, 1], [-1, 2], [1, 3]], np.int32)
    return Sequences(channels, index)


def test_shape_properties() -> None:
    s = panel()
    assert (s.n_rows, s.steps, s.n_channels, s.inputs) == (4, 2, 3, 4)


def test_gather_reads_each_step_and_masks_absent_ones() -> None:
    got = panel().gather(np.array([3, 0, 3]))
    assert got.shape == (3, 2, 4)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got[0], [[4, 5, 6, 1], [10, 11, 12, 1]])
    np.testing.assert_array_equal(got[1], [[0, 0, 0, 0], [1, 2, 3, 1]])
    np.testing.assert_array_equal(got[2], got[0])


def test_gather_of_no_rows() -> None:
    assert panel().gather(np.zeros(0, np.int64)).shape == (0, 2, 4)


def test_a_memory_mapped_panel_is_read_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "c.npy"
    np.save(path, panel().channels)
    mapped = Sequences(np.load(path, mmap_mode="r"), panel().index)
    np.testing.assert_array_equal(mapped.gather(np.arange(4)), panel().gather(np.arange(4)))


def test_rows_may_share_the_rows_of_a_smaller_table() -> None:
    table = np.array([[1, 2], [3, 4]], np.float32)  # e.g. one row per era
    s = Sequences(table, np.array([[0], [0], [1], [-1], [1]], np.int32))
    assert (s.n_rows, s.steps, s.n_channels) == (5, 1, 2)
    np.testing.assert_array_equal(s.gather(np.arange(5))[:, 0], [[1, 2, 1], [1, 2, 1], [3, 4, 1], [0, 0, 0], [3, 4, 1]])
    with pytest.raises(ValueError, match=r"-1 \.\. 1"):
        Sequences(table, np.array([[2]], np.int32))


def test_the_finiteness_check_reads_every_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seq_module, "CHECK_ROWS", 1)
    channels = np.zeros((3, 2), np.float32)
    channels[2, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        Sequences(channels, np.zeros((3, 1), np.int32))


@pytest.mark.parametrize(
    ("channels", "index", "message"),
    [
        (np.zeros(3, np.float32), np.zeros((3, 1), np.int32), "channels"),
        (np.zeros((3, 2), np.float64), np.zeros((3, 1), np.int32), "channels"),
        (np.zeros((3, 0), np.float32), np.zeros((3, 1), np.int32), "channels"),
        (np.zeros((3, 2), np.float32), np.zeros((3, 1), np.int64), "index"),
        (np.zeros((0, 2), np.float32), np.zeros((3, 1), np.int32), "channels"),
        (np.zeros((3, 2), np.float32), np.zeros((3, 0), np.int32), "index"),
        (np.full((3, 2), np.nan, np.float32), np.zeros((3, 1), np.int32), "finite"),
        (np.zeros((3, 2), np.float32), np.full((3, 1), -2, np.int32), "-1 .. 2"),
        (np.zeros((3, 2), np.float32), np.full((3, 1), 3, np.int32), "-1 .. 2"),
    ],
)
def test_bad_sequences_raise(channels: np.ndarray, index: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Sequences(channels, index)


@pytest.mark.parametrize("rows", [np.array([4]), np.array([-1]), np.zeros((1, 1), np.int64)])
def test_bad_rows_raise(rows: np.ndarray) -> None:
    with pytest.raises(ValueError, match="rows"):
        panel().gather(rows)
