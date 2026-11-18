"""Idun: tabular learners on the GPU (Metal on Apple silicon, CUDA on NVIDIA), with CPU reference backends.

The learners are `idun.gbm` (idun_gbm), `idun.tabm` (idun_tabm) and `idun.realmlp` (idun_realmlp); `Dataset` holds
tabular data binned for them. The public interface is documented in docs/api/python-api.md.
"""

from idun import _cpus

_cpus.size_numba()  # first: numba counts its threads when it is imported (`idun._cpus`)

from idun._version import __version__  # noqa: E402
from idun.dataset import Dataset  # noqa: E402
from idun.models import load  # noqa: E402

__all__ = ["Dataset", "__version__", "load"]
