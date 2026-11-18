"""`RealMLPClassifier`, `RealMLPRegressor`: idun_realmlp behind scikit-learn's conventions (`idun.nn.estimator`).

The parameters are `RealMLPParams`' fields (the class count and the objective come from the task) plus ``n_seeds``,
``backend`` (``"metal"`` by default; ``"cuda"``, or ``"reference"`` for the float64 NumPy network), ``categorical``,
``max_one_hot_cat_size`` and ``embedding_size``. RealMLP scales its inputs itself (robust scaling and smooth
clipping); categorical columns are one-hot, or embedded from a learned table when they have
``max_one_hot_cat_size`` levels or more. Batches are built on the host. With fewer training rows than
``batch_size``, a batch holds every row (``train_realmlp`` itself needs at least ``batch_size`` rows, since it drops
an epoch's last partial batch).
"""

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from idun.dataset import Column
from idun.nn.estimator import NetClassifier, NetEstimator, NetRegressor
from idun.nn.inputs import MatrixData
from idun.nn.training import Validation
from idun.realmlp.params import RealMLPParams
from idun.realmlp.train import RealMLPModel, train_realmlp


class _RealMLPSettings(NetEstimator):
    """RealMLP's settings as estimator parameters, shared by `RealMLPClassifier` and `RealMLPRegressor`."""

    params_class = RealMLPParams
    standardize = False

    def __init__(
        self,
        *,
        blocks: int = 3,
        hidden: int = 256,
        frequencies: int = 16,
        embed: int = 3,
        embed_sigma: float = 0.1,
        clip: float = 3.0,
        dropout: float = 0.15,
        label_smoothing: float = 0.1,
        learning_rate: float = 0.04,
        weight_decay: float = 0.02,
        eps: float = 1e-8,
        scale_lr_factor: float = 6.0,
        bias_lr_factor: float = 0.1,
        act_lr_factor: float = 0.1,
        embed_lr_factor: float = 0.1,
        batch_size: int = 256,
        epochs: int = 256,
        init_rows: int = 0,
        lr_schedule: str = "coslog4",
        dropout_schedule: str = "flat_cos",
        wd_schedule: str = "flat_cos",
        act: str = "selu",
        hidden_sizes: Sequence[int] = (),
        patience: int | None = None,
        seed: int = 0,
        n_seeds: int = 1,
        backend: str = "metal",
        categorical: Sequence[Column] = (),
        max_one_hot_cat_size: int = -1,
        embedding_size: int = 8,
    ) -> None:
        self.blocks = blocks
        self.hidden = hidden
        self.frequencies = frequencies
        self.embed = embed
        self.embed_sigma = embed_sigma
        self.clip = clip
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.scale_lr_factor = scale_lr_factor
        self.bias_lr_factor = bias_lr_factor
        self.act_lr_factor = act_lr_factor
        self.embed_lr_factor = embed_lr_factor
        self.batch_size = batch_size
        self.epochs = epochs
        self.init_rows = init_rows
        self.lr_schedule = lr_schedule
        self.dropout_schedule = dropout_schedule
        self.wd_schedule = wd_schedule
        self.act = act
        self.hidden_sizes = hidden_sizes
        self.patience = patience
        self.seed = seed
        self.n_seeds = n_seeds
        self.backend = backend
        self.categorical = categorical
        self.max_one_hot_cat_size = max_one_hot_cat_size
        self.embedding_size = embedding_size

    def _max_one_hot_levels(self) -> int | None:
        """Return the most one-hot levels: pytabkit's ``max_one_hot_cat_size`` counts a missing level, so one fewer."""
        size: Any = self.max_one_hot_cat_size  # set_params may have set anything
        if isinstance(size, bool) or not isinstance(size, int) or size < -1:
            raise ValueError(f"max_one_hot_cat_size must be -1 (every categorical column one-hot) or a level count, "
                             f"got {size!r}")  # fmt: skip
        return None if size < 0 else size - 1

    def _train(self, params: RealMLPParams, data: MatrixData, valid: Validation | None,
               weight: NDArray[np.float32] | None) -> RealMLPModel:  # fmt: skip
        params = dataclasses.replace(params, batch_size=min(params.batch_size, data.n_rows))
        return train_realmlp(params, data, rows=slice(0, data.n_rows), backend=self.backend, valid=valid,
                             patience=self.patience, weight=weight, categories=self.encoder_.coded_levels)  # fmt: skip

    @classmethod
    def _load_model(cls, path: Path) -> RealMLPModel:
        return RealMLPModel.load(path)


class RealMLPClassifier(_RealMLPSettings, NetClassifier):
    """RealMLP-TD (Holzmüller et al.) for classification; the defaults are `RealMLPParams`'."""


class RealMLPRegressor(_RealMLPSettings, NetRegressor):
    """RealMLP-TD for regression (the ``"mse"`` objective, a standardized target); defaults as `RealMLPClassifier`."""
