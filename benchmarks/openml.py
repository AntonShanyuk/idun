"""idun's learners against the libraries they reimplement, on five OpenML tasks.

    uv run --frozen --group benchmarks python -m benchmarks.openml fetch     # benchmarks/data/, checked by sha256
    uv run --frozen --group benchmarks python -m benchmarks.openml all       # every run, one process each
    uv run --frozen --group benchmarks python -m benchmarks.openml run california idun_gbm --seed 0
    uv run --frozen --group benchmarks python -m benchmarks.openml report    # a Markdown table of the records

Each task is one OpenML dataset (`TASKS`), prepared by `prepare` under fixed column rules and split once into 64 %
training, 16 % validation and 20 % test rows (`split`, a seeded permutation). Every run fits on the training rows and
scores the test rows (RMSE for regression; log-loss and accuracy for classification): idun_gbm against LightGBM,
XGBoost and scikit-learn's HistGradientBoosting at the same nominal settings, and idun_tabm and idun_realmlp against
the `tabm` package and pytabkit's RealMLP on the same input matrix with the same training settings (`LEARNERS`). The
idun learners run at their estimators' defaults on Metal (idun_realmlp with the published runs' `init_rows`); the
Torch references run on Torch's MPS device.

`all` runs each fit in its own process (Torch and the boosting libraries bring their own OpenMP runtimes) and skips a
run whose record exists. Records (`<out>/<task>/<learner>_s<seed>.json`) hold the metrics, the fit's and the
prediction's seconds, the machine and the package versions; the test predictions are saved next to them. Close other
work before timing: nothing here waits for a quiet machine. `--smoke` runs every learner on a few thousand rows into
`<out>/smoke/`, to check an installation in minutes. docs/benchmarks.md has the method and measured results.
"""

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "data"
OUT = ROOT / "benchmarks" / "records"
OPENML_URL = "https://data.openml.org/datasets/{bucket:04d}/{id}/dataset_{id}.pq"
SPLIT_SEED = 20261031
SAMPLE_SEED = 20261031
RARE = 30  # a categorical level seen in fewer rows of the whole table joins the level "rare"
MISSING_LEVEL, RARE_LEVEL = "missing", "rare"
SMOKE_ROWS = (2_000, 500, 500)  # training, validation and test rows of a smoke run
CHUNK = 8192  # rows per inference call of the Torch reference


@dataclasses.dataclass(frozen=True)
class Task:
    """One OpenML dataset: its file, target, task kind and an optional seeded row sample."""

    name: str
    openml_id: int
    sha256: str
    target: str
    kind: str  # "regression", "binary" or "multiclass"
    sample: int | None = None

    @property
    def filename(self) -> str:
        """Return the parquet file's name, as OpenML serves it."""
        return f"dataset_{self.openml_id}.pq"

    @property
    def url(self) -> str:
        """Return where `fetch` downloads it."""
        return OPENML_URL.format(bucket=self.openml_id // 10_000, id=self.openml_id)


TASKS = {
    t.name: t
    for t in (
        Task("california", 44025, "f2445ca0f101cbf48249bcc56b39d66a096df3b58ef1e234e0b0786f0a3ed6ad", "price",
             "regression"),
        Task("diamonds", 42225, "a8bed5e78fc346a17585171d310d949e33b1d18def72660e68f5e9b82f69e6de", "price",
             "regression"),
        Task("adult", 1590, "01f7570bc4c800d473e9381615e6a5b0eca53900196bbf9f751439545c909263", "class", "binary"),
        Task("higgs", 44129, "5c1e141a9eb9869d2ee05831ba87b84d07bb8473b7394dd51871748f2962afc7", "target", "binary",
             sample=500_000),
        Task("covertype", 1596, "93f923a6c2d649e22c6a8cb94ca9590ecfe47b30435527c3a1195079dd55790b", "class",
             "multiclass"),
    )
}  # fmt: skip
GBM_TASKS = ("california", "diamonds")  # idun_gbm is a regressor
LARGE_TASKS = ("higgs", "covertype")  # one seed per network: their fits take most of the machine time
SMALL_SEEDS, LARGE_SEEDS = (0, 1), (0,)
DEFAULTS_TASKS = ("california", "diamonds", "adult")  # pytabkit at its own defaults: reported, not compared


@dataclasses.dataclass
class Prepared:
    """A task's table after the column rules, and its split."""

    X: pd.DataFrame  # numeric columns float64 (NaN kept), categorical columns str
    y: NDArray[Any]  # float64 targets, or int64 classes 0..C-1
    categorical: list[str]
    classes: list[str]  # the sorted class labels (empty for regression)
    train: NDArray[np.int64]
    val: NDArray[np.int64]
    test: NDArray[np.int64]


@dataclasses.dataclass(frozen=True)
class Settings:
    """Where data and records live, and whether runs are smoke runs."""

    data: Path = DATA
    out: Path = OUT
    smoke: bool = False

    @property
    def records(self) -> Path:
        """Return the records' folder: ``out``, or ``out/smoke`` for smoke runs."""
        return self.out / "smoke" if self.smoke else self.out

    def record(self, task: str, learner: str, seed: int) -> Path:
        """Return where a run's record goes."""
        return self.records / task / f"{learner}_s{seed}.json"


def sha256(path: Path) -> str:
    """Return the file's SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(data: Path) -> None:
    """Download each task's parquet file into ``data`` (if absent) and check it against its sha256."""
    import urllib.request  # noqa: PLC0415 - only here

    data.mkdir(parents=True, exist_ok=True)
    for task in TASKS.values():
        path = data / task.filename
        if not path.exists():
            print(f"downloading {task.url}", flush=True)
            urllib.request.urlretrieve(task.url, path)  # noqa: S310 - a fixed https URL
        if (found := sha256(path)) != task.sha256:
            raise RuntimeError(f"{path}: sha256 {found}, expected {task.sha256}; delete it and fetch again")
        print(f"{task.name}: {path.name} ok", flush=True)


def split(n: int) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return sorted training, validation and test rows: 64 %, 16 % and 20 % of a seeded permutation."""
    order = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_test, n_val = round(0.2 * n), round(0.16 * n)
    test, val, train = order[:n_test], order[n_test : n_test + n_val], order[n_test + n_val :]
    return np.sort(train).astype(np.int64), np.sort(val).astype(np.int64), np.sort(test).astype(np.int64)


def _is_text(column: pa.ChunkedArray) -> bool:
    kind = column.type.value_type if pa.types.is_dictionary(column.type) else column.type
    return bool(pa.types.is_string(kind) or pa.types.is_large_string(kind))


def prepare_table(table: pa.Table, target: str, kind: str) -> tuple[pd.DataFrame, NDArray[Any], list[str], list[str]]:
    """Apply the column rules; return the features, the target, the categorical columns and the classes.

    A text column whose values are exactly the levels "0" and "1" is a numeric indicator; any other text column is
    categorical, its missing values the level "missing" and levels seen in fewer than `RARE` rows the level "rare".
    Every other column is numeric (float64, missing values NaN).
    """
    columns: dict[str, Any] = {}
    categorical: list[str] = []
    for name in table.column_names:
        if name == target:
            continue
        column = table.column(name)
        if not _is_text(column):
            columns[name] = column.to_numpy(zero_copy_only=False).astype(np.float64)
            continue
        values = np.asarray(column.to_pylist(), dtype=object)
        present = {v for v in values if v is not None}
        if present == {"0", "1"} and column.null_count == 0:
            columns[name] = (values == "1").astype(np.float64)
            continue
        text = np.where(pd.isna(values), MISSING_LEVEL, values).astype(str)
        levels, counts = np.unique(text, return_counts=True)
        rare = set(levels[counts < RARE].tolist())
        columns[name] = np.where(np.isin(text, list(rare)), RARE_LEVEL, text) if rare else text
        categorical.append(name)
    frame = pd.DataFrame(columns)
    raw = table.column(target)
    if kind == "regression":
        return frame, raw.to_numpy(zero_copy_only=False).astype(np.float64), categorical, []
    labels = np.asarray(raw.to_pylist(), dtype=str)
    classes = sorted(set(labels.tolist()))
    return frame, np.searchsorted(np.array(classes), labels).astype(np.int64), categorical, classes


def prepare(task: Task, data: Path) -> Prepared:
    """Read, sample and split a task (see `prepare_table`); fail loudly on a level outside the training rows."""
    path = data / task.filename
    if not path.exists() or sha256(path) != task.sha256:
        raise RuntimeError(f"{path}: missing or not the expected file; run `fetch`")
    table = pq.read_table(path)
    if task.sample is not None:
        rows = np.sort(np.random.default_rng(SAMPLE_SEED).choice(table.num_rows, task.sample, replace=False))
        table = table.take(pa.array(rows))
    frame, y, categorical, classes = prepare_table(table, task.target, task.kind)
    train, val, test = split(len(frame))
    for name in categorical:
        seen = set(frame[name].to_numpy()[train].tolist())
        unseen = set(frame[name].to_numpy()[np.concatenate([val, test])].tolist()) - seen
        if unseen:
            raise RuntimeError(f"{task.name}: column {name!r} has levels outside the training rows: {sorted(unseen)}")
    return Prepared(frame, y, categorical, classes, train, val, test)


def shrink(data: Prepared) -> Prepared:
    """Return a smoke run's rows: `SMOKE_ROWS` of each part, evenly spaced (a table may be sorted by its target)."""
    train, val, test = (part[np.linspace(0, part.size - 1, min(n, part.size)).astype(np.int64)]
                        for part, n in zip((data.train, data.val, data.test), SMOKE_ROWS, strict=True))  # fmt: skip
    for name in data.categorical:  # a level outside the training rows would fail the run
        seen = set(data.X[name].to_numpy()[train].tolist())
        keep = np.isin(data.X[name].to_numpy(), list(seen))
        val, test = val[keep[val]], test[keep[test]]
    return dataclasses.replace(data, train=train, val=val, test=test)


def rmse(y: NDArray[Any], pred: NDArray[Any]) -> float:
    """Return the root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y, np.float64) - np.asarray(pred, np.float64)) ** 2)))


def log_loss(y: NDArray[np.int64], proba: NDArray[Any]) -> float:
    """Return the mean negative log-likelihood of the classes ``y`` under ``proba`` (clipped at 1e-15)."""
    p = np.asarray(proba, np.float64)
    return float(-np.mean(np.log(np.clip(p[np.arange(y.size), y], 1e-15, 1.0))))


def metrics(kind: str, y: NDArray[Any], out: NDArray[Any]) -> dict[str, float]:
    """Return RMSE for regression; log-loss and accuracy for classification."""
    if kind == "regression":
        return {"rmse": rmse(y, out)}
    return {"log_loss": log_loss(y, out), "accuracy": float(np.mean(np.argmax(out, axis=1) == y))}


def primary(kind: str) -> str:
    """Return a task kind's compared metric (lower is better)."""
    return "rmse" if kind == "regression" else "log_loss"


# ---------------------------------------------------------------------------------------------------------------------
# Learners. Each fits on the training rows and returns the test rows' predictions (regression values, or class
# probabilities), with the fit's and the prediction's seconds. ``epochs`` overrides the networks' epochs (smoke runs).
# ---------------------------------------------------------------------------------------------------------------------
Learner = Callable[[Prepared, int, int | None], tuple[NDArray[Any], float, float]]


def gbm_settings() -> dict[str, Any]:
    """Return the settings the boosting libraries share: `GBMRegressor`'s defaults."""
    from idun.gbm import GBMRegressor  # noqa: PLC0415

    params = GBMRegressor().get_params()
    return {k: params[k] for k in ("num_trees", "learning_rate", "max_depth", "max_leaves", "min_data_in_leaf",
                                   "lambda_l2")}  # fmt: skip


def _categories(data: Prepared, rows: NDArray[np.int64]) -> pd.DataFrame:
    """Return the rows with categorical columns as pandas categories over the training levels (the baselines)."""
    frame = data.X.iloc[rows].copy()
    for name in data.categorical:
        levels = sorted(set(data.X[name].to_numpy()[data.train].tolist()))
        frame[name] = pd.Categorical(frame[name], categories=levels)
    return frame


def _timed(fit: Callable[[], Any], predict: Callable[[], Any]) -> tuple[NDArray[Any], float, float]:
    begin = time.perf_counter()
    fit()
    fitted = time.perf_counter()
    out = predict()
    return np.asarray(out), fitted - begin, time.perf_counter() - fitted


def run_idun_gbm(data: Prepared, seed: int, _: int | None) -> tuple[NDArray[Any], float, float]:
    """`GBMRegressor` at its defaults on Metal."""
    from idun.gbm import GBMRegressor  # noqa: PLC0415

    model = GBMRegressor(seed=seed, categorical=data.categorical)
    x_train, x_test = data.X.iloc[data.train], data.X.iloc[data.test]
    return _timed(lambda: model.fit(x_train, data.y[data.train]), lambda: model.predict(x_test))


def _lightgbm(data: Prepared, seed: int, *, quantized: bool) -> tuple[NDArray[Any], float, float]:
    import lightgbm as lgb  # noqa: PLC0415

    gbm = gbm_settings()
    extra: dict[str, Any] = {"use_quantized_grad": True, "num_grad_quant_bins": 4} if quantized else {}
    model = lgb.LGBMRegressor(n_estimators=gbm["num_trees"], learning_rate=gbm["learning_rate"],
                              max_depth=gbm["max_depth"], num_leaves=gbm["max_leaves"],
                              min_child_samples=gbm["min_data_in_leaf"], reg_lambda=gbm["lambda_l2"], max_bin=255,
                              subsample=1.0, colsample_bytree=1.0, random_state=seed, deterministic=True,
                              verbose=-1, **extra)  # fmt: skip
    x_train, x_test = _categories(data, data.train), _categories(data, data.test)
    return _timed(lambda: model.fit(x_train, data.y[data.train]), lambda: model.predict(x_test))


def run_lightgbm(data: Prepared, seed: int, _: int | None) -> tuple[NDArray[Any], float, float]:
    """LightGBM at idun_gbm's nominal settings (gradients not quantized: LightGBM's default)."""
    return _lightgbm(data, seed, quantized=False)


def run_lightgbm_q4(data: Prepared, seed: int, _: int | None) -> tuple[NDArray[Any], float, float]:
    """LightGBM with gradients quantized into 4 bins, as idun_gbm trains (a control row)."""
    return _lightgbm(data, seed, quantized=True)


def run_xgboost(data: Prepared, seed: int, _: int | None) -> tuple[NDArray[Any], float, float]:
    """XGBoost's histogram trees, depth-wise, at idun_gbm's nominal settings."""
    import xgboost as xgb  # noqa: PLC0415

    gbm = gbm_settings()
    model = xgb.XGBRegressor(n_estimators=gbm["num_trees"], learning_rate=gbm["learning_rate"],
                             max_depth=gbm["max_depth"], max_leaves=gbm["max_leaves"], grow_policy="depthwise",
                             tree_method="hist", max_bin=255, min_child_weight=gbm["min_data_in_leaf"],
                             reg_lambda=gbm["lambda_l2"], reg_alpha=0.0, subsample=1.0, colsample_bytree=1.0,
                             enable_categorical=True, max_cat_to_onehot=4, random_state=seed)  # fmt: skip
    x_train, x_test = _categories(data, data.train), _categories(data, data.test)
    return _timed(lambda: model.fit(x_train, data.y[data.train]), lambda: model.predict(x_test))


def run_hgb(data: Prepared, seed: int, _: int | None) -> tuple[NDArray[Any], float, float]:
    """scikit-learn's HistGradientBoostingRegressor at idun_gbm's nominal settings, without early stopping."""
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

    gbm = gbm_settings()
    model = HistGradientBoostingRegressor(max_iter=gbm["num_trees"], learning_rate=gbm["learning_rate"],
                                          max_depth=gbm["max_depth"], max_leaf_nodes=gbm["max_leaves"],
                                          min_samples_leaf=gbm["min_data_in_leaf"], l2_regularization=gbm["lambda_l2"],
                                          max_bins=255, early_stopping=False, categorical_features="from_dtype",
                                          random_state=seed)  # fmt: skip
    x_train, x_test = _categories(data, data.train), _categories(data, data.test)
    return _timed(lambda: model.fit(x_train, data.y[data.train]), lambda: model.predict(x_test))


def tabm_epochs(epochs: int | None) -> int:
    """Return TabM's epochs: ``epochs``, or `TabMRegressor`'s default."""
    from idun.tabm import TabMRegressor  # noqa: PLC0415

    return int(TabMRegressor().get_params()["epochs"]) if epochs is None else epochs


def realmlp_epochs(epochs: int | None) -> int:
    """Return RealMLP's epochs: ``epochs``, or `RealMLPRegressor`'s default (RealMLP-TD's 256)."""
    from idun.realmlp import RealMLPRegressor  # noqa: PLC0415

    return int(RealMLPRegressor().get_params()["epochs"]) if epochs is None else epochs


def _idun_net(model: Any, data: Prepared) -> tuple[NDArray[Any], float, float]:
    x_train, x_test = data.X.iloc[data.train], data.X.iloc[data.test]
    predict = model.predict if not data.classes else model.predict_proba
    return _timed(lambda: model.fit(x_train, data.y[data.train]), lambda: predict(x_test))


def run_idun_tabm(data: Prepared, seed: int, epochs: int | None) -> tuple[NDArray[Any], float, float]:
    """`TabMRegressor` / `TabMClassifier` at their defaults on Metal."""
    from idun.tabm import TabMClassifier, TabMRegressor  # noqa: PLC0415

    cls = TabMRegressor if not data.classes else TabMClassifier
    return _idun_net(cls(epochs=tabm_epochs(epochs), seed=seed, categorical=data.categorical), data)


def run_idun_realmlp(data: Prepared, seed: int, epochs: int | None) -> tuple[NDArray[Any], float, float]:
    """`RealMLPRegressor` / `RealMLPClassifier` at their defaults on Metal, but the published runs' `init_rows`."""
    from idun.realmlp import RealMLPClassifier, RealMLPRegressor  # noqa: PLC0415

    cls = RealMLPRegressor if not data.classes else RealMLPClassifier
    # init_rows: the published runs'; the default, 0, takes pytabkit's count, every row here
    return _idun_net(cls(epochs=realmlp_epochs(epochs), seed=seed, categorical=data.categorical, init_rows=3986), data)


def encoded(data: Prepared, *, standardize: bool) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return the training and test rows as the idun estimators' input matrix (`TableEncoder` on the training rows)."""
    from idun.nn.table import TableEncoder  # noqa: PLC0415

    encoder = TableEncoder(categorical=data.categorical, standardize=standardize).fit(data.X.iloc[data.train])
    return encoder.transform(data.X.iloc[data.train]), encoder.transform(data.X.iloc[data.test])


def run_tabm_package(data: Prepared, seed: int, epochs: int | None) -> tuple[NDArray[Any], float, float]:
    """Train the `tabm` package's TabM (its defaults: k = 32, 3 x 512, dropout 0.1) on Torch MPS as idun trains it.

    The same input matrix, standardized target, AdamW (2e-3, weight decay 3e-4), batches of 256 rows in a fresh
    permutation each epoch with the ragged last batch kept, the mean loss over rows and members, idun_tabm's epochs and
    the last weights; predictions are the members' mean (softmax probabilities for classification).
    """
    import tabm  # noqa: PLC0415
    import torch  # noqa: PLC0415

    n_epochs = tabm_epochs(epochs)
    x_train, x_test = encoded(data, standardize=True)
    device = torch.device("mps")
    torch.manual_seed(seed)
    regression = not data.classes
    y = data.y[data.train]
    mean, scale = (float(y.mean()), float(y.std()) or 1.0) if regression else (0.0, 1.0)
    target = torch.as_tensor(((y - mean) / scale).astype(np.float32) if regression else y, device=device)
    xt = torch.as_tensor(x_train, device=device)
    model = tabm.TabM.make(n_num_features=x_train.shape[1], d_out=1 if regression else len(data.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    order = torch.Generator().manual_seed(seed)

    def fit() -> None:
        model.train()
        for _ in range(n_epochs):
            for batch in torch.randperm(xt.shape[0], generator=order).split(256):
                rows = batch.to(device)
                out = model(x_num=xt[rows])  # (B, k, d_out)
                if regression:
                    loss = torch.nn.functional.mse_loss(out[..., 0], target[rows][:, None].expand(-1, out.shape[1]))
                else:
                    loss = torch.nn.functional.cross_entropy(out.flatten(0, 1), target[rows].repeat_interleave(
                        out.shape[1]))  # fmt: skip
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        torch.mps.synchronize()

    def predict() -> NDArray[Any]:
        model.eval()
        parts = []
        with torch.inference_mode():
            for lo in range(0, x_test.shape[0], CHUNK):
                out = model(x_num=torch.as_tensor(x_test[lo : lo + CHUNK], device=device))
                parts.append((out[..., 0].mean(1) * scale + mean) if regression else out.softmax(-1).mean(1))
        return np.concatenate([p.float().cpu().numpy() for p in parts])

    return _timed(fit, predict)


def run_pytabkit_realmlp(data: Prepared, seed: int, epochs: int | None) -> tuple[NDArray[Any], float, float]:
    """Train pytabkit's RealMLP-TD on Torch MPS, on idun_realmlp's input matrix and training settings.

    The same matrix (numeric columns and one-hot categorical columns, NaN as 0), idun_realmlp's epochs of 256-row
    batches, no validation rows (so the last epoch's weights). The regressor takes idun_realmlp's block (SELU, learning
    rate 0.04) and leaves its predictions unclamped.
    """
    from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor  # noqa: PLC0415

    x_train, x_test = encoded(data, standardize=False)
    common = {"device": "mps", "random_state": seed, "n_epochs": realmlp_epochs(epochs), "batch_size": 256,
              "val_fraction": 0.0, "verbosity": 0}  # fmt: skip
    if not data.classes:
        model = RealMLP_TD_Regressor(**common, act="selu", lr=0.04, clamp_output=False)
        return _timed(lambda: model.fit(x_train, data.y[data.train]), lambda: model.predict(x_test))
    model = RealMLP_TD_Classifier(**common)
    return _timed(lambda: model.fit(x_train, data.y[data.train]), lambda: model.predict_proba(x_test))


def run_pytabkit_defaults(data: Prepared, seed: int, *, tabm_d: bool) -> tuple[NDArray[Any], float, float]:
    """Train pytabkit's RealMLP-TD or TabM-D at their own defaults on the raw table (reported, not compared)."""
    import pytabkit  # noqa: PLC0415

    name = ("TabM_D_" if tabm_d else "RealMLP_TD_") + ("Regressor" if not data.classes else "Classifier")
    model = getattr(pytabkit, name)(device="mps", random_state=seed, verbosity=0)
    x_train, x_test = _categories(data, data.train), _categories(data, data.test)
    predict = model.predict if not data.classes else model.predict_proba
    return _timed(lambda: model.fit(x_train, data.y[data.train]), lambda: predict(x_test))


LEARNERS: dict[str, Learner] = {
    "idun_gbm": run_idun_gbm,
    "lightgbm": run_lightgbm,
    "lightgbm_q4": run_lightgbm_q4,
    "xgboost": run_xgboost,
    "hgb": run_hgb,
    "idun_tabm": run_idun_tabm,
    "tabm_package": run_tabm_package,
    "idun_realmlp": run_idun_realmlp,
    "pytabkit_realmlp": run_pytabkit_realmlp,
    "pytabkit_realmlp_defaults": lambda d, s, _: run_pytabkit_defaults(d, s, tabm_d=False),
    "pytabkit_tabm_defaults": lambda d, s, _: run_pytabkit_defaults(d, s, tabm_d=True),
}
GBM_LEARNERS = ("idun_gbm", "lightgbm", "lightgbm_q4", "xgboost", "hgb")
GBM_BASELINES = ("lightgbm", "xgboost", "hgb")
NET_PAIRS = (("idun_tabm", "tabm_package"), ("idun_realmlp", "pytabkit_realmlp"))
DEFAULT_ROWS = ("pytabkit_realmlp_defaults", "pytabkit_tabm_defaults")


def plan() -> list[tuple[str, str, int]]:
    """Return every run as (task, learner, seed), in the order `all` runs them."""
    runs = [(task, learner, 0) for task in GBM_TASKS for learner in GBM_LEARNERS]
    for task in TASKS:
        seeds = LARGE_SEEDS if task in LARGE_TASKS else SMALL_SEEDS
        runs += [(task, learner, seed) for pair in NET_PAIRS for learner in pair for seed in seeds]
    runs += [(task, learner, 0) for task in DEFAULTS_TASKS for learner in DEFAULT_ROWS]
    return runs


# ---------------------------------------------------------------------------------------------------------------------
# Runs, records and the report.
# ---------------------------------------------------------------------------------------------------------------------
def now() -> str:
    """Return the time in UTC, ISO 8601."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _sysctl(name: str) -> str:
    out = subprocess.run(["/usr/sbin/sysctl", "-n", name], check=False, capture_output=True, text=True)
    return out.stdout.strip()


def machine() -> dict[str, Any]:
    """Describe the machine a record is measured on."""
    mac = sys.platform == "darwin"
    return {"model": _sysctl("hw.model") if mac else platform.machine(),
            "chip": _sysctl("machdep.cpu.brand_string") if mac else platform.processor(),
            "memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"), "system": platform.platform(),
            "python": sys.version}  # fmt: skip


def versions() -> dict[str, str]:
    """Return the versions of the packages a run could use."""
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    out = {}
    for name in ("idun", "numpy", "scikit-learn", "lightgbm", "xgboost", "torch", "tabm", "pytabkit"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "absent"
    return out


def run(task_name: str, learner: str, seed: int, settings: Settings) -> Path:
    """Fit one learner on one task with one seed; write its record and test predictions; return the record's path."""
    path = settings.record(task_name, learner, seed)
    if path.exists():
        print(f"{path}: done already", flush=True)
        return path
    task = TASKS[task_name]
    data = prepare(task, settings.data)
    epochs = None
    if settings.smoke:
        data, epochs = shrink(data), 1 if "tabm" in learner else 2
    started = now()
    out, fit_seconds, predict_seconds = LEARNERS[learner](data, seed, epochs)
    scores = metrics(task.kind, data.y[data.test], out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), out.astype(np.float32))
    record = {"task": task_name, "openml_id": task.openml_id, "learner": learner, "seed": seed, "smoke": settings.smoke,
              "metrics": scores, "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
              "n_train": int(data.train.size), "n_test": int(data.test.size), "started": started, "finished": now(),
              "machine": machine(), "versions": versions()}  # fmt: skip
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"{task_name} {learner} s{seed}: {scores}, fit {fit_seconds:.1f} s", flush=True)
    return path


def run_all(settings: Settings, tasks: Sequence[str] = (), learners: Sequence[str] = ()) -> None:
    """Run every run of `plan` (or those of ``tasks`` and ``learners``) in its own process; stop at a failure."""
    for task, learner, seed in plan():
        if (tasks and task not in tasks) or (learners and learner not in learners):
            continue
        if settings.record(task, learner, seed).exists():
            continue
        print(f"{now()} {task} {learner} s{seed}", flush=True)
        command = [sys.executable, "-m", "benchmarks.openml", "--data", str(settings.data), "--out", str(settings.out),
                   *(["--smoke"] if settings.smoke else []), "run", task, learner, "--seed", str(seed)]  # fmt: skip
        code = subprocess.run(command, cwd=ROOT, check=False).returncode
        if code != 0:
            raise SystemExit(f"{task} {learner} s{seed} failed with exit code {code}")


def summary(records: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    """Return each (task, learner)'s mean metrics and mean fit seconds over its seeds, with the seed count."""
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        by.setdefault((r["task"], r["learner"]), []).append(r)
    out = {}
    for key, rows in by.items():
        means = {m: float(np.mean([r["metrics"][m] for r in rows])) for m in rows[0]["metrics"]}
        out[key] = {**means, "fit_seconds": float(np.mean([r["fit_seconds"] for r in rows])), "seeds": len(rows)}
    return out


def reference(task: str, learner: str, table: dict[tuple[str, str], dict[str, float]]) -> float | None:
    """Return what a learner is compared with on a task, or None.

    idun_gbm is compared with the median of the three libraries, idun_tabm and idun_realmlp with their Torch
    references; other learners, and a comparison whose records are missing, give None.
    """
    metric = primary(TASKS[task].kind)
    if learner == "idun_gbm":
        values = [table[(task, b)][metric] for b in GBM_BASELINES if (task, b) in table]
        return float(np.median(values)) if len(values) == len(GBM_BASELINES) else None
    pair = dict(NET_PAIRS).get(learner)
    return table[(task, pair)][metric] if pair is not None and (task, pair) in table else None


def report(settings: Settings) -> str:
    """Return a Markdown table of the records.

    One row per task and learner: the seeds, the mean metric, its ratio to what the learner is compared with (see
    `reference`) and the mean fit seconds.
    """
    records = [json.loads(p.read_text()) for p in sorted(settings.records.glob("*/*.json"))]
    if not records:
        raise SystemExit(f"no records in {settings.records}")
    table = summary(records)
    lines = ["| task | learner | seeds | metric | value | ratio | fit seconds |", "|---|---|---|---|---|---|---|"]
    for task, learner in sorted(table, key=lambda k: (list(TASKS).index(k[0]), list(LEARNERS).index(k[1]))):
        row, metric = table[(task, learner)], primary(TASKS[task].kind)
        ref = reference(task, learner, table)
        ratio = f"{row[metric] / ref:.4f}" if ref is not None else ""
        lines.append(f"| {task} | {learner} | {row['seeds']:.0f} | {metric} | {row[metric]:.6g} | {ratio} | "
                     f"{row['fit_seconds']:.1f} |")  # fmt: skip
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: fetch, run, all or report."""
    parser = argparse.ArgumentParser(prog="python -m benchmarks.openml", description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DATA, help="where the OpenML files are (default %(default)s)")
    parser.add_argument("--out", type=Path, default=OUT, help="where records go (default %(default)s)")
    parser.add_argument("--smoke", action="store_true", help="a few thousand rows and one or two epochs per run")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch")
    sub.add_parser("report")
    one = sub.add_parser("run")
    one.add_argument("task", choices=list(TASKS))
    one.add_argument("learner", choices=list(LEARNERS))
    one.add_argument("--seed", type=int, default=0)
    every = sub.add_parser("all")
    every.add_argument("--task", action="append", default=[], choices=list(TASKS), help="only this task (repeatable)")
    every.add_argument("--learner", action="append", default=[], choices=list(LEARNERS),
                       help="only this learner (repeatable)")  # fmt: skip
    args = parser.parse_args(argv)
    settings = Settings(data=args.data, out=args.out, smoke=args.smoke)
    if args.command == "fetch":
        fetch(settings.data)
    elif args.command == "run":
        run(args.task, args.learner, args.seed, settings)
    elif args.command == "all":
        run_all(settings, args.task, args.learner)
    else:
        print(report(settings))


if __name__ == "__main__":
    main()
