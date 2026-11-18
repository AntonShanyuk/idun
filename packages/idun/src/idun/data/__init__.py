"""The packed on-disk dataset: its writer and its memory-mapped reader.

A packed dataset is a date-sorted panel stored column-major (4-bit codes, raw numeric values and their bins per
window, one categorical column, a target), wrapped by Metal without a copy. Format: docs/architecture/data-layout.md.
API: docs/api/python-api.md. idun-numerai builds one from Numerai's files (`idun_numerai.data.build_cache`).
"""

from idun.data.dataset import PackedDataset
from idun.data.lags import lag_index
from idun.data.refits import Refit
from idun.data.writer import PackedWriter

__all__ = ["PackedDataset", "PackedWriter", "Refit", "lag_index"]
