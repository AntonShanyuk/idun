"""The CPUs a Linux CPU quota allows the process, and numba's threads sized to them.

numba starts one thread per CPU the process may be scheduled on (its affinity), but a Linux control group can hold
it to fewer with a CPU quota: Docker's ``--cpus``, a Kubernetes limit, a cloud GPU machine's container. Every thread
past the quota waits its turn, and a parallel loop slows by about the ratio (a test of idun_gbm's took 40.6 s with
96 threads on a quota of 7.65 CPUs, and 1.9 s with 8). numba fixes its thread count when it is imported, so `idun`
calls `size_numba` before any of its modules imports numba. The thread count never changes a result: the parallel
loops' sums are exact in any order.
"""

import math
import os
import sys
import warnings
from pathlib import Path

ENV = "NUMBA_NUM_THREADS"
CGROUP = Path("/sys/fs/cgroup")
PROC = Path("/proc/self/cgroup")
V1_CPU = ("cpu", "cpu,cpuacct", "cpuacct,cpu")  # the names the v1 cpu controller's hierarchy is mounted under


def quota_cpus(root: Path = CGROUP, proc: Path = PROC) -> int | None:
    """Return the CPUs the process's CPU quota allows, rounded up; None without a quota (or off Linux).

    cgroup v2: ``cpu.max`` (``"<quota> <period>"``, or ``"max <period>"`` for none) of the process's group and of
    each group above it, the smallest. cgroup v1: ``cpu.cfs_quota_us`` (-1 for none) over ``cpu.cfs_period_us`` in
    the cpu controller's hierarchy. A container sees its own group at the mount's root when the path in ``proc``
    names a group outside it.
    """
    try:
        lines = proc.read_text().splitlines()
    except OSError:
        return None
    found: list[float] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":  # v2
            found += [cpus for folder in _groups(root, path) if (cpus := _v2(folder / "cpu.max")) is not None]
        elif "cpu" in controllers.split(","):
            for mount in (root / name for name in V1_CPU):
                found += [cpus for folder in _groups(mount, path) if (cpus := _v1(folder)) is not None]
    return max(1, math.ceil(min(found))) if found else None


def _groups(mount: Path, path: str) -> list[Path]:
    """Return the process's group under ``mount`` and each group above it, up to the mount."""
    group = mount / path.lstrip("/")
    if not group.is_dir():
        group = mount
    folders = [group]
    while folders[-1] != mount and mount in folders[-1].parents:
        folders.append(folders[-1].parent)
    return folders


def _v2(path: Path) -> float | None:
    try:
        quota, period = path.read_text().split()
    except (OSError, ValueError):
        return None
    return None if quota == "max" else int(quota) / int(period)


def _v1(folder: Path) -> float | None:
    try:
        quota = int((folder / "cpu.cfs_quota_us").read_text())
        period = int((folder / "cpu.cfs_period_us").read_text())
    except (OSError, ValueError):
        return None
    return None if quota < 0 or period <= 0 else quota / period


def listed_cpus() -> int:
    """Return the CPUs the process may be scheduled on: numba's thread count by default."""
    affinity = getattr(os, "sched_getaffinity", None)  # Linux's
    return max(1, len(affinity(0)) if affinity is not None else os.cpu_count() or 1)


def threads_for(allowed: int | None, listed: int, *, named: bool) -> int | None:
    """Return the thread count to give numba: the quota's CPUs when they are fewer than listed, unless one is named."""
    return None if named or allowed is None or allowed >= listed else allowed


def size_numba() -> None:
    """Set NUMBA_NUM_THREADS to the quota's CPUs when numba would start more threads, unless it is set already.

    numba imported before idun has its threads counted already: then a warning names the value to set.
    """
    threads = threads_for(quota_cpus(), listed_cpus(), named=ENV in os.environ)
    if threads is None:
        return
    numba = sys.modules.get("numba")
    if numba is not None:
        warnings.warn(
            f"numba was imported before idun with {numba.config.NUMBA_NUM_THREADS} threads, more than the {threads} "
            f"CPUs the process's CPU quota allows: its parallel loops run slower; set {ENV}={threads} before "
            "numba is imported",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    os.environ[ENV] = str(threads)
