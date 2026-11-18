"""The network estimators' shared parts: `NetClassifier` and `NetRegressor` over `NetEstimator`.

``fit(X, y)`` encodes ``X`` with a `TableEncoder` (numeric values, one-hot categorical columns; standardized for TabM)
and trains ``n_seeds`` networks with seeds ``seed``, ``seed + 1``, ... on a `MatrixData`.

- `NetClassifier` maps the labels to classes 0..C − 1 over their sorted values (``classes_``); ``predict_proba`` is the
  mean of the networks' class probabilities and ``predict`` the label of the most probable class.
- `NetRegressor` standardizes the target with the training rows' mean and spread (``target_mean_``,
  ``target_scale_``), trains on the regressor's ``objective`` (``"mse"``; `TabMRegressor` also ``"hl_gauss"``) and
  predicts the networks' mean output, scaled back.

``save`` writes one ``idun-model`` file (the encoder, the labels or the target scale, and each network's own model file
inside it), which `idun.load` reads back. Each network's estimators name its parameter class and how to train and load
one network (`idun.tabm.estimator`, `idun.realmlp.estimator`).
"""

import dataclasses
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from idun._version import __version__
from idun.base import Classifier, Estimator, Regressor
from idun.dataset import Column
from idun.nn.inputs import MatrixData
from idun.nn.table import TableEncoder
from idun.nn.training import FitReport, Validation, checked_weights

MODEL_FORMAT = "idun-model"
MODEL_FORMAT_VERSION = 1


class NetModel(Protocol):
    """A trained network as `TabMModel` and `RealMLPModel` are."""

    report: FitReport

    def predict(self, data: MatrixData, *, rows: slice, backend: str) -> NDArray[np.float32]: ...  # noqa: D102
    def predict_proba(self, data: MatrixData, *, rows: slice, backend: str) -> NDArray[np.float32]: ...  # noqa: D102
    def save(self, path: str | Path) -> None: ...  # noqa: D102


class NetEstimator(Estimator):
    """A network's settings as estimator parameters; subclasses add the task (classification or regression)."""

    params_class: ClassVar[type]
    standardize: ClassVar[bool]
    objective: ClassVar[str]
    seed: int
    n_seeds: int
    backend: str
    categorical: Sequence[Column]

    patience: int | None

    def _train(
        self, params: Any, data: MatrixData, valid: Validation | None, weight: NDArray[np.float32] | None
    ) -> NetModel:
        raise NotImplementedError

    def _max_one_hot_levels(self) -> int | None:
        """Return the most levels a categorical column is one-hot with (None: any); more make a column of codes."""
        return None

    @classmethod
    def _load_model(cls, path: Path) -> NetModel:
        raise NotImplementedError

    def network_params(self, seed: int, n_classes: int) -> Any:
        """Return the network's parameters for ``seed`` and ``n_classes`` (validated by the parameter class).

        Every field the estimator exposes is taken from it; the rest keep the parameter class's defaults.
        """
        exposed = set(self._param_names()) - {"seed", "objective", "n_classes"}  # given below
        names = [f.name for f in dataclasses.fields(self.params_class) if f.name in exposed]
        settings = {n: getattr(self, n) for n in names}
        return self.params_class(**settings, n_classes=n_classes, seed=seed, objective=self.objective)

    def _fit_networks(
        self,
        X: Any,  # noqa: N803 - scikit-learn's name
        y: NDArray[Any],
        n_classes: int,
        eval_set: tuple[Any, NDArray[Any]] | None = None,
        sample_weight: Any = None,
    ) -> None:
        if self.n_seeds < 1:
            raise ValueError(f"n_seeds must be a positive integer, got {self.n_seeds!r}")
        self.encoder_ = TableEncoder(categorical=self.categorical, standardize=self.standardize,
                                     max_one_hot_levels=self._max_one_hot_levels()).fit(X)  # fmt: skip
        x = self.encoder_.transform(X)
        if x.shape[0] != y.size:
            raise ValueError(f"X has {x.shape[0]} rows and y {y.size}")
        weight = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, np.float64)
            if weights.shape != (y.size,):
                raise ValueError(f"sample_weight must hold one weight per row ({y.size}), got shape {weights.shape}")
            weight = checked_weights(weights, y.size)
        data = MatrixData(x, y, n_classes)
        valid = None
        if eval_set is not None:
            rows = self.encoder_.transform(eval_set[0])
            valid = Validation(MatrixData(rows, eval_set[1], n_classes), slice(0, rows.shape[0]))
        self.models_ = [self._train(self.network_params(self.seed + i, n_classes), data, valid, weight)
                        for i in range(self.n_seeds)]  # fmt: skip
        self.n_features_in_ = len(self.encoder_.columns_)
        self.feature_names_in_ = np.array(self.encoder_.columns_, dtype=object)

    @property
    def best_epoch_(self) -> int | list[int | None] | None:
        """The epoch (from 0) whose weights a network kept on the validation set, or each seed's; None without one."""
        self._check_fitted("models_")
        found = [model.report.best_epoch for model in self.models_]
        return found[0] if len(found) == 1 else found

    @property
    def valid_score_(self) -> list[float] | list[list[float]]:
        """The validation score after each epoch (lower is better), or each seed's; empty without a validation set."""
        self._check_fitted("models_")
        found = [list(model.report.valid_score) for model in self.models_]
        return found[0] if len(found) == 1 else found

    @staticmethod
    def _one_set(eval_set: Any) -> tuple[Any, Any] | None:
        """Return the one ``(X, y)`` validation set of ``eval_set`` (a pair or a list of one), or None."""
        if eval_set is None or (isinstance(eval_set, list) and not eval_set):
            return None
        if isinstance(eval_set, list):
            if len(eval_set) != 1:
                raise ValueError(f"the networks score one validation set, got {len(eval_set)}")
            eval_set = eval_set[0]
        X_val, y_val = eval_set  # noqa: N806 - scikit-learn's name
        return X_val, y_val

    def _matrix(self, X: Any, n_classes: int) -> MatrixData:  # noqa: N803 - scikit-learn's name
        self._check_fitted("models_")
        self._check_n_features(X)
        x = self.encoder_.transform(X)
        dummy: NDArray[np.float32] | NDArray[np.int64] = (
            np.zeros(x.shape[0], np.float32) if n_classes == 1 else np.zeros(x.shape[0], np.int64)
        )  # labels are not read when predicting
        return MatrixData(x, dummy, n_classes)

    def _task_meta(self) -> dict[str, Any]:
        raise NotImplementedError

    def _restore_task(self, meta: dict[str, Any]) -> None:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write the fitted estimator to ``path`` (an uncompressed ``.npz``, no object arrays); `idun.load` reads it."""
        self._check_fitted("models_")
        meta = {
            "format": MODEL_FORMAT, "format_version": MODEL_FORMAT_VERSION, "idun_version": __version__,
            "estimator": type(self).__name__,
            "params": {k: list(v) if isinstance(v, tuple) else v for k, v in self.get_params().items()},
            "encoder": self.encoder_.spec(), "task": self._task_meta(),
        }  # fmt: skip
        arrays: dict[str, NDArray[Any]] = {"meta": np.frombuffer(json.dumps(meta).encode(), np.uint8)}
        with tempfile.TemporaryDirectory() as scratch:
            for i, model in enumerate(self.models_):
                file = Path(scratch) / f"m{i}.npz"
                model.save(file)
                arrays[f"m{i}"] = np.frombuffer(file.read_bytes(), np.uint8)
        with Path(path).open("wb") as handle:
            np.savez(handle, allow_pickle=False, **arrays)

    @classmethod
    def _from_archive(cls, meta: dict[str, Any], arrays: dict[str, NDArray[Any]]) -> Self:
        out = cls(**meta["params"])
        out.encoder_ = TableEncoder.from_spec(meta["encoder"])
        out._restore_task(meta["task"])
        models = []
        with tempfile.TemporaryDirectory() as scratch:
            for i in range(sum(1 for name in arrays if name.startswith("m"))):
                file = Path(scratch) / f"m{i}.npz"
                file.write_bytes(arrays[f"m{i}"].tobytes())
                models.append(cls._load_model(file))
        out.models_ = models
        out.n_features_in_ = len(out.encoder_.columns_)
        out.feature_names_in_ = np.array(out.encoder_.columns_, dtype=object)
        return out


class NetClassifier(NetEstimator, Classifier):
    """Classification: labels to classes and back; see the module docstring."""

    objective: ClassVar[str] = "cross_entropy"

    def fit(self, X: Any, y: Any, sample_weight: Any = None, *, eval_set: Any = None) -> Self:  # noqa: N803 - scikit-learn's name
        """Train ``n_seeds`` networks on ``X`` and the labels ``y``; return the estimator.

        ``sample_weight``: one finite, non-negative weight per row (not all zero); each step's loss is its batch's
        weighted mean (`train_tabm`'s ``weight``). ``eval_set``: one ``(X_val, y_val)`` (or a list of one): its rows
        are scored after every epoch (unweighted), each network keeps the weights of its best epoch, and ``patience``
        stops training after that many epochs without a better one (`train_tabm`'s ``valid``).
        """
        self._require_y(y)
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError(f"y must be 1-D labels, got shape {labels.shape}")
        if labels.dtype.kind == "c":
            raise ValueError("Complex data not supported: y must hold class labels")
        if labels.dtype.kind == "f" and not np.array_equal(labels, np.round(labels)):
            raise ValueError("Unknown label type: continuous; a classifier needs class labels (use a regressor)")
        classes = np.unique(labels)
        if classes.size < 2:
            raise ValueError(f"y must hold at least two classes, got {classes.size} class")
        self.classes_ = classes
        found = self._one_set(eval_set)
        encoded = None
        if found is not None:
            valid_labels = np.asarray(found[1])
            if not np.isin(valid_labels, classes).all():
                raise ValueError("eval_set holds labels outside the training classes")
            encoded = (found[0], np.searchsorted(classes, valid_labels).astype(np.int64))
        self._fit_networks(X, np.searchsorted(classes, labels).astype(np.int64), int(classes.size), encoded,
                           sample_weight)  # fmt: skip
        return self

    def predict_proba(self, X: Any, *, per_seed: bool = False) -> NDArray[np.float32]:  # noqa: N803 - scikit-learn's name
        """Return each row's class probabilities ``(n, n_classes)``, the mean over the seed bag's networks.

        ``per_seed``: each network's probabilities instead, ``(n, n_seeds, n_classes)`` in seed order.
        """
        self._check_fitted("models_")
        data = self._matrix(X, int(self.classes_.size))
        rows = slice(0, data.n_rows)
        probs = [m.predict_proba(data, rows=rows, backend=self.backend) for m in self.models_]
        if per_seed:
            return np.stack(probs, axis=1).astype(np.float32)
        return np.mean(probs, axis=0, dtype=np.float64).astype(np.float32)

    def predict(self, X: Any) -> NDArray[Any]:  # noqa: N803 - scikit-learn's name
        """Return the label of each row's most probable class."""
        proba = self.predict_proba(X)  # checks that the estimator is fitted
        return np.asarray(self.classes_[np.argmax(proba, axis=1)])

    def _task_meta(self) -> dict[str, Any]:
        return {"classes": self.classes_.tolist()}

    def _restore_task(self, meta: dict[str, Any]) -> None:
        self.classes_ = np.array(meta["classes"])


class NetRegressor(NetEstimator, Regressor):
    """Regression on the ``"mse"`` objective with a standardized target; see the module docstring."""

    objective: ClassVar[str] = "mse"

    def fit(self, X: Any, y: Any, sample_weight: Any = None, *, eval_set: Any = None) -> Self:  # noqa: N803 - scikit-learn's name
        """Train ``n_seeds`` networks on ``X`` and the targets ``y``; return the estimator.

        ``sample_weight`` and ``eval_set`` (one ``(X_val, y_val)`` or a list of one, its targets standardized as the
        training ones): see `NetClassifier.fit`. The target's standardization is unweighted.
        """
        self._require_y(y)
        if np.asarray(y).dtype.kind == "c":
            raise ValueError("Complex data not supported: y must be real")
        target = np.asarray(y, np.float64)
        if target.ndim != 1 or not np.isfinite(target).all():
            raise ValueError(f"y must be 1-D finite targets, got shape {target.shape}")
        spread = float(target.std())
        self.target_mean_, self.target_scale_ = float(target.mean()), spread if spread > 0 else 1.0
        scaled = ((target - self.target_mean_) / self.target_scale_).astype(np.float32)
        found = self._one_set(eval_set)
        encoded = None
        if found is not None:
            valid_target = np.asarray(found[1], np.float64)
            encoded = (found[0], ((valid_target - self.target_mean_) / self.target_scale_).astype(np.float32))
        self._fit_networks(X, scaled, 1, encoded, sample_weight)
        return self

    def predict(self, X: Any, *, per_seed: bool = False) -> NDArray[np.float32]:  # noqa: N803 - scikit-learn's name
        """Return each row's prediction: the seed bag's mean output, scaled back to the target's units.

        ``per_seed``: each network's prediction instead, ``(n, n_seeds)`` in seed order; column i is what a one-seed fit
        with seed ``seed + i`` predicts.
        """
        data = self._matrix(X, 1)
        rows = slice(0, data.n_rows)
        each = [m.predict(data, rows=rows, backend=self.backend) for m in self.models_]
        outputs = np.column_stack(each) if per_seed else np.mean(each, axis=0)
        return (outputs * self.target_scale_ + self.target_mean_).astype(np.float32)

    def _task_meta(self) -> dict[str, Any]:
        return {"target_mean": self.target_mean_, "target_scale": self.target_scale_}

    def _restore_task(self, meta: dict[str, Any]) -> None:
        self.target_mean_, self.target_scale_ = float(meta["target_mean"]), float(meta["target_scale"])
