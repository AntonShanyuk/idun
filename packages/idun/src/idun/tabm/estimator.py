"""`TabMClassifier`, `TabMRegressor`: idun_tabm behind scikit-learn's conventions (`idun.nn.estimator`).

The parameters are `TabMParams`' fields (the class count and the objective come from the task) plus ``n_seeds``,
``backend`` (``"metal"`` by default, ``"reference"`` for the float64 NumPy network) and ``categorical``. Inputs are
standardized with the training rows' mean and spread; categorical columns are one-hot.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from idun.dataset import Column
from idun.nn.estimator import NetClassifier, NetEstimator, NetRegressor
from idun.nn.inputs import MatrixData
from idun.nn.training import Validation
from idun.tabm.params import TabMParams
from idun.tabm.train import TabMModel, train_tabm


class _TabMSettings(NetEstimator):
    """TabM's settings as estimator parameters, shared by `TabMClassifier` and `TabMRegressor`."""

    params_class = TabMParams
    standardize = True

    def __init__(
        self,
        *,
        members: int = 32,
        blocks: int = 3,
        width: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 2e-3,
        weight_decay: float = 3e-4,
        eps: float = 1e-8,
        batch_size: int = 256,
        epochs: int = 50,
        storage: str = "fp32",
        micro_batch: int = 0,
        lr_schedule: str = "constant",
        lr_warmup_steps: int = 0,
        patience: int | None = None,
        seed: int = 0,
        n_seeds: int = 1,
        backend: str = "metal",
        categorical: Sequence[Column] = (),
        num_emb_type: str = "none",
        num_emb_n_bins: int = 48,
        d_embedding: int = 16,
        gradient_clipping_norm: float = 0.0,
    ) -> None:
        self.members = members
        self.blocks = blocks
        self.width = width
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eps = eps
        self.batch_size = batch_size
        self.epochs = epochs
        self.storage = storage
        self.micro_batch = micro_batch
        self.lr_schedule = lr_schedule
        self.lr_warmup_steps = lr_warmup_steps
        self.patience = patience
        self.seed = seed
        self.n_seeds = n_seeds
        self.backend = backend
        self.categorical = categorical
        self.num_emb_type = num_emb_type
        self.num_emb_n_bins = num_emb_n_bins
        self.d_embedding = d_embedding
        self.gradient_clipping_norm = gradient_clipping_norm

    def _train(self, params: TabMParams, data: MatrixData, valid: Validation | None,
               weight: NDArray[np.float32] | None) -> TabMModel:  # fmt: skip
        embedded = None
        if params.num_emb_type != "none":  # the numeric columns, which the encoder puts first
            embedded = len(self.encoder_.numeric_)
            if embedded == 0:
                raise ValueError(f"num_emb_type {params.num_emb_type!r} embeds numeric columns; X has none")
        return train_tabm(params, data, rows=slice(0, data.n_rows), backend=self.backend, valid=valid,
                          patience=self.patience, weight=weight, embedded=embedded)  # fmt: skip

    @classmethod
    def _load_model(cls, path: Path) -> TabMModel:
        return TabMModel.load(path)


class TabMClassifier(_TabMSettings, NetClassifier):
    """TabM (Gorishniy et al.) for classification; the defaults are `TabMParams`'."""


class TabMRegressor(_TabMSettings, NetRegressor):
    """TabM for regression on a standardized target; defaults as `TabMClassifier`.

    ``objective``: ``"mse"`` (the members' mean output) or ``"hl_gauss"`` (classification over ``n_bins`` uniform bins
    of ``[hl_low, hl_high]`` of the standardized target with labels smoothed by N(target, ``hl_sigma``); the prediction
    is the expected bin centre, scaled back). See `TabMParams`.
    """

    def __init__(
        self,
        *,
        members: int = 32,
        blocks: int = 3,
        width: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 2e-3,
        weight_decay: float = 3e-4,
        eps: float = 1e-8,
        batch_size: int = 256,
        epochs: int = 50,
        storage: str = "fp32",
        micro_batch: int = 0,
        objective: str = "mse",
        n_bins: int = 64,
        hl_low: float = -4.0,
        hl_high: float = 4.0,
        hl_sigma: float = 0.09375,
        lr_schedule: str = "constant",
        lr_warmup_steps: int = 0,
        patience: int | None = None,
        seed: int = 0,
        n_seeds: int = 1,
        backend: str = "metal",
        categorical: Sequence[Column] = (),
        num_emb_type: str = "none",
        num_emb_n_bins: int = 48,
        d_embedding: int = 16,
        gradient_clipping_norm: float = 0.0,
    ) -> None:
        super().__init__(members=members, blocks=blocks, width=width, dropout=dropout, learning_rate=learning_rate,
                         weight_decay=weight_decay, eps=eps, batch_size=batch_size, epochs=epochs, storage=storage,
                         micro_batch=micro_batch, lr_schedule=lr_schedule, lr_warmup_steps=lr_warmup_steps,
                         patience=patience, seed=seed, n_seeds=n_seeds, backend=backend, categorical=categorical,
                         num_emb_type=num_emb_type, num_emb_n_bins=num_emb_n_bins, d_embedding=d_embedding,
                         gradient_clipping_norm=gradient_clipping_norm)  # fmt: skip
        self.objective = objective  # type: ignore[misc]  # the regressor's objective is a parameter here
        self.n_bins = n_bins
        self.hl_low = hl_low
        self.hl_high = hl_high
        self.hl_sigma = hl_sigma
