"""`TableEncoder`: a table to a network's float32 input matrix.

Numeric columns (and codes, which the networks read as numbers) become float32 values with NaN set to 0, standardized
with the training rows' mean and standard deviation when asked (TabM; RealMLP scales its inputs itself). Categorical
columns become one-hot blocks over their sorted levels (as `idun.Dataset` keeps them: at most 64, no missing value,
an unseen level raises). The column order of the matrix: numeric columns in the input's order, then the one-hot blocks.
With ``max_one_hot_levels`` (RealMLP's categorical embeddings), a categorical column of more levels becomes one
column of its level codes (0, 1, ... in the sorted order) instead, those columns last, in the input's order.
"""

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from idun.dataset import Column, _columns, _encode, _levels, _resolve, n_columns


class TableEncoder:
    """Fit on a training table with `fit`; `transform` any table with the same columns."""

    def __init__(
        self, *, categorical: Sequence[Column] = (), standardize: bool = False, max_one_hot_levels: int | None = None
    ) -> None:
        self.categorical = categorical
        self.standardize = standardize
        self.max_one_hot_levels = max_one_hot_levels

    def fit(self, X: Any, feature_names: Sequence[str] | None = None) -> Self:  # noqa: N803 - scikit-learn's name
        """Learn the columns, the categorical levels and (if standardizing) each numeric column's mean and spread."""
        columns = _columns(X, feature_names)
        if not len(columns[0][1]):
            raise ValueError("X has no rows")
        names = [name for name, _ in columns]
        self.columns_ = tuple(names)
        self.categorical_ = tuple(_resolve(self.categorical, names, "categorical"))
        self.numeric_ = tuple(n for n in names if n not in self.categorical_)
        by_name = dict(columns)
        self.levels_ = tuple(_levels(n, by_name[n]) for n in self.categorical_)
        most = self.max_one_hot_levels
        self.coded_ = tuple(n for n, levels in zip(self.categorical_, self.levels_, strict=True)
                            if most is not None and len(levels) > most)  # fmt: skip
        values = self._numeric(by_name)
        mean = np.zeros(values.shape[1], np.float64)
        scale = np.ones(values.shape[1], np.float64)
        if self.standardize and values.shape[0]:
            mean, spread = values.mean(axis=0, dtype=np.float64), values.std(axis=0, dtype=np.float64)
            scale = np.where(spread > 0, spread, 1.0)
        self.mean_, self.scale_ = mean.astype(np.float32), scale.astype(np.float32)
        return self

    def _numeric(self, by_name: dict[str, NDArray[Any]]) -> NDArray[np.float32]:
        cols = [
            np.nan_to_num(np.asarray(by_name[n], np.float32), nan=0.0, posinf=0.0, neginf=0.0) for n in self.numeric_
        ]
        return np.array(cols, np.float32).T.reshape(len(next(iter(by_name.values()))), len(cols))

    @property
    def n_inputs(self) -> int:
        """Columns of the matrix: the numeric columns, each one-hot column's levels and one per coded column."""
        return len(self.numeric_) + sum(1 if n in self.coded_ else len(levels)
                                        for n, levels in zip(self.categorical_, self.levels_, strict=True))  # fmt: skip

    @property
    def coded_levels(self) -> tuple[int, ...]:
        """The levels of each coded column (`coded_`), the matrix's last columns: `train_realmlp`'s ``categories``."""
        return tuple(len(levels) for n, levels in zip(self.categorical_, self.levels_, strict=True)
                     if n in self.coded_)  # fmt: skip

    def transform(self, X: Any) -> NDArray[np.float32]:  # noqa: N803 - scikit-learn's name
        """Return the ``(n_rows, n_inputs)`` float32 matrix of ``X``."""
        if isinstance(X, np.ndarray) and n_columns(X) != len(self.columns_):
            raise ValueError(f"X has shape {X.shape}; the encoder was fitted on {len(self.columns_)} columns")
        by_name = dict(_columns(X, list(self.columns_) if isinstance(X, np.ndarray) else None))
        missing = [c for c in self.columns_ if c not in by_name]
        if missing:
            raise ValueError(f"X lacks the columns {missing}")
        numeric = (self._numeric(by_name) - self.mean_) / self.scale_
        blocks, coded = [numeric.astype(np.float32)], []
        for name, levels in zip(self.categorical_, self.levels_, strict=True):
            codes = _encode(name, by_name[name], levels)
            if name in self.coded_:
                coded.append(codes.astype(np.float32)[:, None])
            else:
                blocks.append(np.eye(len(levels), dtype=np.float32)[codes])
        return np.ascontiguousarray(np.concatenate(blocks + coded, axis=1), np.float32)

    def spec(self) -> dict[str, Any]:
        """Return the fitted state as JSON-safe values, for `from_spec` (the coded columns only when there are)."""
        coded = {"max_one_hot_levels": self.max_one_hot_levels, "coded": list(self.coded_)} if self.coded_ else {}
        return {"columns": list(self.columns_), "categorical": list(self.categorical_), "numeric": list(self.numeric_),
                "levels": [list(levels) for levels in self.levels_], "standardize": self.standardize,
                "mean": self.mean_.tolist(), "scale": self.scale_.tolist(), **coded}  # fmt: skip

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> Self:
        """Return an encoder in the fitted state ``spec``."""
        out = cls(categorical=spec["categorical"], standardize=spec["standardize"],
                  max_one_hot_levels=spec.get("max_one_hot_levels"))  # fmt: skip
        out.coded_ = tuple(spec.get("coded", ()))
        out.columns_, out.categorical_, out.numeric_ = (
            tuple(spec["columns"]),
            tuple(spec["categorical"]),
            tuple(spec["numeric"]),
        )
        out.levels_ = tuple(tuple(levels) for levels in spec["levels"])
        out.mean_ = np.array(spec["mean"], np.float32)
        out.scale_ = np.array(spec["scale"], np.float32)
        return out
