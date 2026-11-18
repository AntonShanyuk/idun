"""`Dataset`: tabular data from NumPy, pandas or Arrow, binned once for idun_gbm (docs/api/python-api.md).

Columns are of three kinds, each stored the way idun_gbm's kernels read them (`idun.gbm.BinnedData`):

- **numeric** (every column not named below): uint8 quantile bins with the packed cache's own rules
  (`idun.data.binning`: edges from a fixed sample of rows, at most 255 bins, NaN in bin 255);
- **codes** (``codes=[...]``): integer columns with values 0..4 and no missing value, packed two rows per byte (the
  fastest path of idun_gbm's histograms);
- **categorical** (``categorical=[...]``): one uint8 code per row over the column's sorted levels, at most 64 levels
  (a split is a 64-bit mask) and no missing value.

Nothing is chosen from the data: a column is codes or categorical only when the caller names it, and a value that does
not fit fails loudly. `transform` bins new rows with the stored edges and levels; `subset` selects rows.
"""

import hashlib
import sys
from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from idun.data import binning
from idun.data.layout import MAX_NAMED_BINS, pack_nibbles

MAX_CATEGORIES = 64
CODE_VALUES = 5  # codes are 0..4
Column = int | str


def _array(x: Any) -> NDArray[Any]:
    """Return a 2-D NumPy array of ``x``, refusing sparse, 1-D, complex and column-less input with the fix."""
    if type(x).__module__.startswith("scipy.sparse"):
        raise TypeError("sparse input is not supported: pass a dense array (X.toarray())")
    array = np.asarray(x)
    if array.ndim == 1:
        raise ValueError(
            f"X must be 2-D (rows, columns), got a 1-D array of shape {array.shape}. Reshape your data with "
            "X.reshape(-1, 1) for a single column or X.reshape(1, -1) for a single row."
        )
    if array.ndim != 2:
        raise ValueError(f"X must be 2-D (rows, columns), got shape {array.shape}")
    return array


def _arrow_table(x: Any) -> bool:
    """Whether ``x`` is an Arrow table; pyarrow is an optional input type, so this never imports it."""
    pa = sys.modules.get("pyarrow")  # an Arrow table exists only once pyarrow has been imported
    return pa is not None and isinstance(x, pa.Table)


def _column_list(x: Any) -> bool:
    """Whether ``x`` is a list of ``(name, 1-D array)`` pairs, the fourth input kind (columns built by the caller)."""
    return isinstance(x, list) and all(isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], str) for c in x)


def n_columns(x: Any) -> int:
    """Return the number of columns of a 2-D NumPy array, a pandas DataFrame, an Arrow table or a column list."""
    if _arrow_table(x):
        return int(x.num_columns)
    if _column_list(x):
        return len(x)
    if hasattr(x, "columns") and hasattr(x, "to_numpy") and not isinstance(x, np.ndarray):  # a pandas DataFrame
        return len(x.columns)
    return int(_array(x).shape[1])


def _columns(x: Any, names: Sequence[str] | None) -> list[tuple[str, NDArray[Any]]]:
    """``(name, 1-D array)`` per column of a 2-D NumPy array, a pandas DataFrame, an Arrow table or a column list."""
    if _arrow_table(x):
        columns = [(name, x.column(name).to_numpy(zero_copy_only=False)) for name in x.column_names]
        shape = (x.num_rows, x.num_columns)
    elif _column_list(x):
        columns = [(name, np.asarray(values)) for name, values in x]
        lengths = {values.shape for _, values in columns}
        if len(lengths) > 1 or any(len(shape) != 1 for shape in lengths):
            raise ValueError(f"every column must be a 1-D array of the same length, got shapes {sorted(lengths)}")
        if len({name for name, _ in columns}) != len(columns):
            raise ValueError("a column is named twice")
        shape = (columns[0][1].size if columns else 0, len(columns))
    elif hasattr(x, "columns") and hasattr(x, "to_numpy") and not isinstance(x, np.ndarray):  # a pandas DataFrame
        columns = [(str(name), np.asarray(x[name].to_numpy())) for name in x.columns]
        shape = (len(columns[0][1]) if columns else len(x.to_numpy()), len(columns))
    else:
        array = _array(x)
        labels = list(names) if names is not None else [f"x{i}" for i in range(array.shape[1])]
        if len(labels) != array.shape[1]:
            raise ValueError(f"{len(labels)} feature_names for {array.shape[1]} columns")
        columns, shape = [(label, array[:, i]) for i, label in enumerate(labels)], array.shape
    if not columns:
        raise ValueError(f"Found array with 0 feature(s) (shape={shape}) while a minimum of 1 is required.")
    if any(values.dtype.kind == "c" for _, values in columns):
        raise ValueError("Complex data not supported: X must be real")
    return columns


def columns_of(x: Any, names: Sequence[str] | None = None) -> list[tuple[str, NDArray[Any]]]:
    """Return ``(name, 1-D array)`` per column of a table (`_columns`, the public doorway for `idun.jarf`)."""
    return _columns(x, names)


def resolve_columns(selected: Sequence[Column], names: Sequence[str], what: str) -> list[str]:
    """Return the names of ``selected`` columns (names or positions among ``names``); unknown ones raise."""
    return _resolve(selected, list(names), what)


def _resolve(selected: Sequence[Column], names: list[str], what: str) -> list[str]:
    """Column names of ``selected`` (names or positions); unknown ones raise."""
    out = []
    for column in selected:
        if isinstance(column, int) and not isinstance(column, bool):
            if not 0 <= column < len(names):
                raise ValueError(f"{what} column {column} is out of range for {len(names)} columns")
            out.append(names[column])
        elif column in names:
            out.append(str(column))
        else:
            raise ValueError(f"{what} column {column!r} is not among the columns {names}")
    if len(set(out)) != len(out):
        raise ValueError(f"{what} names a column twice: {out}")
    return out


def _missing(values: NDArray[Any]) -> NDArray[np.bool_]:
    if values.dtype.kind == "f":
        return np.isnan(values)
    if values.dtype.kind == "O":
        return np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in values], bool)
    return np.zeros(values.shape, bool)


def _codes(name: str, values: NDArray[Any]) -> NDArray[np.uint8]:
    if _missing(values).any():
        raise ValueError(f"codes column {name!r} has missing values; codes are integers 0..4")
    as_int = np.asarray(values).astype(np.int64)
    if not np.array_equal(as_int, values) or as_int.min(initial=0) < 0 or as_int.max(initial=0) >= CODE_VALUES:
        raise ValueError(f"codes column {name!r} must hold integers 0..4; bin it as numeric instead")
    return as_int.astype(np.uint8)


def _text(values: NDArray[Any]) -> NDArray[np.str_]:
    """Return a categorical column's values as strings: levels compare as text, sorted lexicographically."""
    return np.asarray(values).astype(str)


def _levels(name: str, values: NDArray[Any]) -> tuple[str, ...]:
    if _missing(values).any():
        raise ValueError(f"categorical column {name!r} has missing values: give them a level of their own first")
    levels = tuple(str(v) for v in np.unique(_text(values)))
    if len(levels) > MAX_CATEGORIES:
        raise ValueError(f"categorical column {name!r} has {len(levels)} levels; at most {MAX_CATEGORIES} fit")
    return levels


def _encode(name: str, values: NDArray[Any], levels: tuple[str, ...]) -> NDArray[np.uint8]:
    text, known = _text(values), np.array(levels)
    at = np.minimum(np.searchsorted(known, text), len(levels) - 1)
    unseen = text != known[at]
    if unseen.any():
        raise ValueError(f"categorical column {name!r} has levels the data was not built with: "
                         f"{sorted(set(text[unseen].tolist()))[:10]}")  # fmt: skip
    return at.astype(np.uint8)


class Dataset:
    """Binned columns of a table, satisfying `idun.gbm.BinnedData`; see the module docstring.

    Parameters
    ----------
    X
        A 2-D NumPy array, a pandas DataFrame or an Arrow table.
    y
        Optional target per row (numeric; it becomes ``target``, float32). Without it ``target`` is zeros.
    categorical, codes
        Columns (names, or positions for an array) of those kinds; every other column is numeric.
    feature_names
        Column names for a NumPy array (default ``x0``, ``x1``, ...).
    bin_rows
        The rows whose values fit the numeric bin edges (default: a sample of ``subsample_for_bin`` rows drawn with
        ``seed``, as the packed cache draws its own).
    seed
        Seed of that sample.
    max_bin
        At most this many bins per numeric column, 2 to 255 (LightGBM's ``max_bin``; 255 by default, the packed
        cache's), NaN in a bin of its own beyond them.
    subsample_for_bin
        The rows sampled for the edges (LightGBM's; `idun.data.binning.SAMPLE_ROWS`, 200,000, by default); every row
        when the table has no more. Not with ``bin_rows``.
    """

    def __init__(
        self,
        X: Any,  # noqa: N803 - scikit-learn's name for the feature matrix
        y: Any = None,
        *,
        categorical: Sequence[Column] = (),
        codes: Sequence[Column] = (),
        feature_names: Sequence[str] | None = None,
        bin_rows: NDArray[np.integer] | None = None,
        seed: int = 0,
        max_bin: int = MAX_NAMED_BINS,
        subsample_for_bin: int = binning.SAMPLE_ROWS,
    ) -> None:
        if not 2 <= max_bin <= MAX_NAMED_BINS:
            raise ValueError(f"max_bin must be 2 to {MAX_NAMED_BINS}, got {max_bin!r}")
        if subsample_for_bin < 1:
            raise ValueError(f"subsample_for_bin must be positive, got {subsample_for_bin!r}")
        if bin_rows is not None and subsample_for_bin != binning.SAMPLE_ROWS:
            raise ValueError("bin_rows or subsample_for_bin: the rows of the edges are one or the other")
        columns = _columns(X, feature_names)
        names = [name for name, _ in columns]
        self.columns: tuple[str, ...] = tuple(names)
        cat_names, code_names = _resolve(categorical, names, "categorical"), _resolve(codes, names, "codes")
        if set(cat_names) & set(code_names):
            raise ValueError(f"columns {sorted(set(cat_names) & set(code_names))} are both categorical and codes")
        by_name = dict(columns)
        numeric = [n for n in names if n not in cat_names and n not in code_names]
        self.n_rows = len(columns[0][1]) if columns else 0
        if self.n_rows == 0:
            raise ValueError("X has no rows")
        sample = (
            binning.sample_rows(self.n_rows, subsample_for_bin, seed=seed, refit=0) if bin_rows is None else bin_rows
        )
        values = [np.asarray(by_name[n], np.float32) for n in numeric]
        edges = [binning.quantile_edges(np.ascontiguousarray(v[sample]), max_bin) for v in values]
        self._set(
            numeric=numeric,
            code_names=code_names,
            cat_names=cat_names,
            edge_rows=binning.edge_table(edges),
            n_bins=tuple(e.size + 1 for e in edges),
            categories=tuple(_levels(n, by_name[n]) for n in cat_names),
            by_name=by_name,
            y=y,
        )

    def _set(
        self,
        *,
        numeric: list[str],
        code_names: list[str],
        cat_names: list[str],
        edge_rows: NDArray[np.float32],
        n_bins: tuple[int, ...],
        categories: tuple[tuple[str, ...], ...],
        by_name: dict[str, NDArray[Any]],
        y: Any,
    ) -> None:
        """Fill the binned arrays of ``by_name``'s rows under the given edges and levels."""
        n = self.n_rows = len(next(iter(by_name.values())))
        categories = tuple(categories)
        self.named_features = tuple(numeric)
        self.obfuscated_features = tuple(code_names)
        self.categorical_features = tuple(cat_names)
        self.feature_names = (*numeric, *code_names, *cat_names)
        self.named_edges = edge_rows
        self.named_n_bins = n_bins
        self.categories = categories
        named = [binning.apply_bins(np.asarray(by_name[c], np.float32), edge_rows[i]) for i, c in enumerate(numeric)]
        self.named_bins = np.array(named, np.uint8).reshape(len(numeric), n)
        codes = np.array([_codes(c, by_name[c]) for c in code_names], np.uint8).reshape(len(code_names), n)
        self.obfuscated = pack_nibbles(codes)
        cats = [_encode(c, by_name[c], levels) for c, levels in zip(cat_names, categories, strict=True)]
        self.categorical = np.array(cats, np.uint8).reshape(len(cat_names), n)
        if y is not None and np.asarray(y).dtype.kind == "c":
            raise ValueError("Complex data not supported: y must be real")
        target = np.zeros(n, np.float32) if y is None else np.asarray(y, np.float32)
        if target.shape != (n,):
            raise ValueError(f"y must have one value per row ({n}), got shape {target.shape}")
        if not np.isfinite(target).all():
            raise ValueError("y has NaN or infinite values; every target must be finite")
        self.target = target
        identity = hashlib.sha256(edge_rows.tobytes() + repr((self.feature_names, categories)).encode()).hexdigest()
        self.binning = f"dataset:{identity[:16]}"  # a model refuses data binned another way

    def transform(self, X: Any, y: Any = None) -> Self:  # noqa: N803 - scikit-learn's name
        """Return new rows ``X`` binned with this dataset's edges and levels (the same columns; unseen levels raise)."""
        if isinstance(X, np.ndarray) and n_columns(X) != len(self.columns):
            raise ValueError(f"X has shape {X.shape}; the data was built with {len(self.columns)} columns")
        columns = dict(_columns(X, list(self.columns) if isinstance(X, np.ndarray) else None))
        missing = [c for c in self.columns if c not in columns]
        if missing:
            raise ValueError(f"X lacks the columns {missing}")
        out = object.__new__(type(self))
        out.columns = self.columns
        out._set(
            numeric=list(self.named_features),
            code_names=list(self.obfuscated_features),
            cat_names=list(self.categorical_features),
            edge_rows=self.named_edges,
            n_bins=self.named_n_bins,
            categories=self.categories,
            by_name={c: columns[c] for c in self.columns},
            y=y,
        )
        return out

    def subset(self, rows: slice | NDArray[np.integer]) -> Self:
        """Return the rows ``rows`` (a slice or row indices) as a dataset with the same binning."""
        index = np.arange(self.n_rows)[rows]
        out = object.__new__(type(self))
        out.__dict__.update(self.__dict__)
        out.n_rows = int(index.size)
        out.named_bins = np.ascontiguousarray(self.named_bins[:, index])
        codes = (self.obfuscated[:, index >> 1] >> ((index & 1) << 2).astype(np.uint8)) & 15
        out.obfuscated = pack_nibbles(np.ascontiguousarray(codes, np.uint8))
        out.categorical = np.ascontiguousarray(self.categorical[:, index])
        out.target = np.ascontiguousarray(self.target[index])
        return out

    def spec(self) -> dict[str, Any]:
        """Return the binning without the rows (JSON-safe; the edges are `named_edges`), for `from_spec`."""
        return {"columns": list(self.columns), "numeric": list(self.named_features),
                "codes": list(self.obfuscated_features), "categorical": list(self.categorical_features),
                "named_n_bins": list(self.named_n_bins), "categories": [list(c) for c in self.categories]}  # fmt: skip

    @classmethod
    def from_spec(cls, spec: dict[str, Any], named_edges: NDArray[np.float32]) -> Self:
        """Return a dataset of no rows with ``spec``'s binning: its `transform` bins new rows as the original."""
        out = object.__new__(cls)
        out.columns = tuple(spec["columns"])
        out._set(
            numeric=list(spec["numeric"]),
            code_names=list(spec["codes"]),
            cat_names=list(spec["categorical"]),
            edge_rows=np.asarray(named_edges, np.float32),
            n_bins=tuple(int(b) for b in spec["named_n_bins"]),
            categories=tuple(tuple(c) for c in spec["categories"]),
            by_name={c: np.zeros(0) for c in out.columns},
            y=None,
        )
        return out

    @property
    def n_features(self) -> int:
        """Number of columns."""
        return len(self.feature_names)

    def __repr__(self) -> str:
        return (
            f"Dataset({self.n_rows} rows: {len(self.named_features)} numeric, {len(self.obfuscated_features)} codes, "
            f"{len(self.categorical_features)} categorical)"
        )
