"""idun._cpus: the CPUs a Linux CPU quota allows, and numba's threads sized to them before numba is imported."""

import ast
import os
import sys
import types
from pathlib import Path

import pytest

import idun
from idun import _cpus


def cgroup(tmp_path: Path, proc: str, files: dict[str, str]) -> tuple[Path, Path]:
    """Write a fake ``/sys/fs/cgroup`` (``files`` under it) and ``/proc/self/cgroup`` (``proc``)."""
    root = tmp_path / "cgroup"
    root.mkdir()
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    (tmp_path / "proc").write_text(proc)
    return root, tmp_path / "proc"


@pytest.mark.parametrize(
    ("proc", "files", "cpus"),
    [
        ("0::/\n", {"cpu.max": "765000 100000\n"}, 8),  # a container's own group at the root: 7.65 CPUs
        ("0::/\n", {"cpu.max": "max 100000\n"}, None),  # no quota
        ("0::/\n", {"cpu.max": "50000 100000\n"}, 1),  # half a CPU is one thread
        ("0::/\n", {}, None),  # the root group has no cpu.max
        ("0::/a/b\n", {"a/cpu.max": "300000 100000", "a/b/cpu.max": "max 100000"}, 3),  # a parent's quota holds
        ("0::/a/b\n", {"a/cpu.max": "300000 100000", "a/b/cpu.max": "200000 100000"}, 2),  # the smallest
        ("0::/docker/abc\n", {"cpu.max": "200000 100000"}, 2),  # a group outside the container: its own is the root
        ("4:cpu,cpuacct:/\n", {"cpu,cpuacct/cpu.cfs_quota_us": "150000", "cpu,cpuacct/cpu.cfs_period_us": "100000"}, 2),
        ("4:cpu,cpuacct:/\n", {"cpu,cpuacct/cpu.cfs_quota_us": "-1", "cpu,cpuacct/cpu.cfs_period_us": "100000"}, None),
        ("3:cpu:/x\n1:name=systemd:/\n", {"cpu/x/cpu.cfs_quota_us": "400000", "cpu/x/cpu.cfs_period_us": "100000"}, 4),
        ("garbled\n0::/\n", {"cpu.max": "200000 100000"}, 2),  # a line not of three fields is passed over
        ("0::/\n", {"cpu.max": "garbled"}, None),
    ],
)
def test_the_quota_is_read_from_either_cgroup_version(
    tmp_path: Path, proc: str, files: dict[str, str], cpus: int | None
) -> None:
    root, path = cgroup(tmp_path, proc, files)
    assert _cpus.quota_cpus(root, path) == cpus


def test_no_cgroup_file_is_no_quota(tmp_path: Path) -> None:
    assert _cpus.quota_cpus(tmp_path, tmp_path / "absent") is None


def test_numba_gets_the_quotas_cpus_only_when_fewer_and_none_is_named() -> None:
    assert _cpus.threads_for(8, 96, named=False) == 8
    assert _cpus.threads_for(8, 96, named=True) is None  # NUMBA_NUM_THREADS set by the user
    assert _cpus.threads_for(None, 96, named=False) is None  # no quota
    assert _cpus.threads_for(16, 8, named=False) is None  # the affinity is the smaller already
    assert _cpus.listed_cpus() >= 1


def unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset NUMBA_NUM_THREADS so that what the test sets is undone after it.

    ``delenv`` of an unset variable records nothing to restore: a value `size_numba` wrote would outlive the test, and
    numba, which re-reads its configuration, refuses a thread count other than the one it started with.
    """
    monkeypatch.setenv(_cpus.ENV, "1")
    monkeypatch.delenv(_cpus.ENV)


def test_the_count_is_set_before_numba_is_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cpus, "quota_cpus", lambda: 8)
    monkeypatch.setattr(_cpus, "listed_cpus", lambda: 96)
    unset(monkeypatch)
    monkeypatch.delitem(sys.modules, "numba", raising=False)
    _cpus.size_numba()
    assert os.environ[_cpus.ENV] == "8"
    monkeypatch.setenv(_cpus.ENV, "32")  # the user's count stands
    _cpus.size_numba()
    assert os.environ[_cpus.ENV] == "32"


def test_numba_imported_first_gets_a_warning_with_the_value_to_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cpus, "quota_cpus", lambda: 8)
    monkeypatch.setattr(_cpus, "listed_cpus", lambda: 96)
    unset(monkeypatch)
    numba = types.SimpleNamespace(config=types.SimpleNamespace(NUMBA_NUM_THREADS=96))
    monkeypatch.setitem(sys.modules, "numba", numba)
    with pytest.warns(RuntimeWarning, match="imported before idun with 96 threads.*set NUMBA_NUM_THREADS=8"):
        _cpus.size_numba()
    assert _cpus.ENV not in os.environ


def test_idun_sizes_numba_before_anything_else_it_imports() -> None:
    assert idun.__file__ is not None
    body = ast.parse(Path(idun.__file__).read_text()).body
    first, second = (ast.unparse(node) for node in body[1:3])  # after the docstring
    assert (first, second) == ("from idun import _cpus", "_cpus.size_numba()")
