"""Backends: the CPU reference, Metal and CUDA. Nothing selects a backend implicitly."""

from idun.gbm.backends.base import Backend, LeafSplit, LevelSplits, Lockstep, ManyBackend
from idun.gbm.backends.cpu import CPUBackend
from idun.gbm.backends.cuda import CudaBackend, CudaMany
from idun.gbm.backends.metal import MetalBackend, MetalMany

BACKENDS = ("cpu", "metal", "cuda")

__all__ = [
    "BACKENDS",
    "Backend",
    "CPUBackend",
    "CudaBackend",
    "CudaMany",
    "LeafSplit",
    "LevelSplits",
    "Lockstep",
    "ManyBackend",
    "MetalBackend",
    "MetalMany",
    "get_backend",
    "get_many",
]


def get_backend(name: str) -> Backend:
    """Construct the backend called ``name``; raise `BackendUnavailableError` if it cannot run here."""
    if name == "cpu":
        return CPUBackend()
    if name == "metal":
        return MetalBackend()
    if name == "cuda":
        return CudaBackend()
    raise ValueError(f"unknown backend {name!r}; the backends are {BACKENDS}")


def get_many(name: str, k: int) -> ManyBackend:
    """Construct a backend for K models of one seed in lockstep: a GPU shares reads; the CPU reference does not."""
    if k < 1:
        raise ValueError(f"need at least one model, got {k}")
    if name == "metal":
        return MetalMany(k)
    if name == "cuda":
        return CudaMany(k)
    return Lockstep([get_backend(name) for _ in range(k)])
