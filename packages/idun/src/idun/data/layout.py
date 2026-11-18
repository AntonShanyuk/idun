"""On-disk layout of the packed cache: file names, column strides, 4-bit packing (docs/architecture/data-layout.md §2).

Every multi-column array is column-major: column ``c`` starts at byte ``c * stride`` of its file, and the stride is the
column's byte length rounded up to a 16 KiB page. A whole file therefore maps page-aligned (Metal wraps it without a
copy), and every column starts on a 16-byte boundary for vector loads.

Obfuscated features pack two **rows** per byte: row ``2i`` in the low nibble of byte ``i``, row ``2i + 1`` in the high
nibble. A column with an odd row count has a zero high nibble in its last byte.
"""

import numpy as np
from numpy.typing import NDArray

FORMAT = "idun-packed"  # the packed format's id, stamped into every manifest
FORMAT_VERSION = 1
PAGE_BYTES = 16 * 1024


OBFUSCATED_LEVELS = 5  # obfuscated values are exactly 0..4: they are their own bins
MAX_NAMED_BINS = 255  # real bins 0..254 for a named feature
MISSING_BIN = 255  # NaN

MANIFEST = "manifest.json"
OBFUSCATED_FILE = "obf.u4"
NAMED_VALUES_FILE = "named.f32"
COUNTRY_FILE = "country.u8"
TARGET_FILE = "y.f32"
ERA_FILE = "era.i32"
TICKER_FILE = "ticker.i32"
TICKERS_FILE = "tickers.json"

_LOW = np.uint8(0x0F)
_FOUR = np.uint8(4)


def binning_name(refit: int) -> str:
    """Name of the named-feature binning fitted on refit ``refit``'s training window: ``r0``, ``r1``, ..."""
    return f"r{refit}"


def named_bins_file(binning: str) -> str:
    """File holding every row's named-feature bins under ``binning``."""
    return f"named.{binning}.u8"


def named_edges_file(binning: str) -> str:
    """File holding the upper bin edges of ``binning``: float32, one row of `MAX_NAMED_BINS` per named feature."""
    return f"named.{binning}.edges.f32"


def column_stride(column_bytes: int) -> int:
    """Byte distance between consecutive columns: ``column_bytes`` rounded up to a whole page (at least one page)."""
    if column_bytes < 1:
        raise ValueError(f"column_bytes must be positive, got {column_bytes}")
    return -(-column_bytes // PAGE_BYTES) * PAGE_BYTES


def packed_bytes(n_rows: int) -> int:
    """Bytes of one 4-bit packed column of ``n_rows`` rows."""
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative, got {n_rows}")
    return (n_rows + 1) // 2


def pack_nibbles(codes: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pack 4-bit codes two rows per byte along the last axis (low nibble = even row); an odd tail is zero-padded."""
    if codes.dtype != np.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype}")
    if codes.size and int(codes.max()) > 0x0F:
        raise ValueError(f"codes must fit in 4 bits, got {int(codes.max())}")
    if codes.shape[-1] % 2:
        codes = np.concatenate([codes, np.zeros((*codes.shape[:-1], 1), np.uint8)], axis=-1)
    return np.ascontiguousarray(codes[..., 0::2] | (codes[..., 1::2] << _FOUR))


def unpack_nibbles(packed: NDArray[np.uint8], lo: int, hi: int) -> NDArray[np.uint8]:
    """Codes of rows ``[lo, hi)`` from columns packed by `pack_nibbles` (last axis = packed bytes)."""
    if not 0 <= lo <= hi <= 2 * packed.shape[-1]:
        raise ValueError(f"rows [{lo}, {hi}) are outside the {2 * packed.shape[-1]} rows these bytes hold")
    first, stop = lo // 2, (hi + 1) // 2
    chunk = packed[..., first:stop]
    codes = np.empty((*chunk.shape[:-1], 2 * chunk.shape[-1]), np.uint8)
    codes[..., 0::2] = chunk & _LOW
    codes[..., 1::2] = chunk >> _FOUR
    start = lo - 2 * first
    return codes[..., start : start + hi - lo]
