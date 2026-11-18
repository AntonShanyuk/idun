"""What every idun estimator shares: scikit-learn's conventions without importing it.

`Estimator` holds the constructor parameters (`get_params`, `set_params`), the checks of inputs that scikit-learn's
``validate_data`` would make (a missing target, a column count other than the fitted one, an unfitted estimator) and
scikit-learn's tags. `Regressor` and `Classifier` add the task and ``score`` (R² and accuracy, as scikit-learn's mixins
define them).
"""

import inspect
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from idun.dataset import n_columns
from idun.errors import NotFittedError


class Estimator:
    """Constructor parameters as attributes; `get_params` and `set_params` as scikit-learn defines them.

    Subclasses name their task in ``_estimator_type`` (``"classifier"`` or ``"regressor"``), which scikit-learn reads
    through `__sklearn_tags__`.
    """

    _estimator_type: ClassVar[str]

    @classmethod
    def _param_names(cls) -> list[str]:
        return [name for name in inspect.signature(cls.__init__).parameters if name != "self"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002, FBT001, FBT002 - scikit-learn's signature
        """Return the constructor parameters (scikit-learn's convention)."""
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> Self:
        """Set constructor parameters; an unknown name raises ``ValueError``."""
        unknown = sorted(set(params) - set(self._param_names()))
        if unknown:
            raise ValueError(f"unknown parameters {unknown}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def _check_fitted(self, attribute: str) -> None:
        if not hasattr(self, attribute):
            raise NotFittedError(f"this {type(self).__name__} is not fitted: call fit first")

    def _require_y(self, y: Any, hint: str = "") -> None:
        if y is None:
            raise ValueError(f"{type(self).__name__} requires y to be passed, but the target y is None{hint}")

    def _check_n_features(self, X: Any) -> None:  # noqa: N803 - scikit-learn's name
        n = n_columns(X)
        if n != self.n_features_in_:
            raise ValueError(
                f"X has {n} features, but {type(self).__name__} is expecting {self.n_features_in_} features as input."
            )

    n_features_in_: int

    def __sklearn_tags__(self) -> Any:
        """Return scikit-learn's tags for this estimator.

        Only scikit-learn calls this, so it is imported here and is never a dependency of idun. Missing values are
        allowed (idun_gbm bins them apart; the networks read them as 0).
        """
        from sklearn.utils import ClassifierTags, InputTags, RegressorTags, Tags, TargetTags  # noqa: PLC0415, TID251

        classifier = self._estimator_type == "classifier"
        return Tags(
            estimator_type=self._estimator_type,
            target_tags=TargetTags(required=True),
            classifier_tags=ClassifierTags() if classifier else None,
            regressor_tags=None if classifier else RegressorTags(),
            input_tags=InputTags(allow_nan=True),
        )

    def __repr__(self) -> str:
        defaults = {n: p.default for n, p in inspect.signature(type(self).__init__).parameters.items() if n != "self"}
        changed = ", ".join(f"{n}={v!r}" for n, v in self.get_params().items() if v != defaults.get(n))
        return f"{type(self).__name__}({changed})"


def _weights(sample_weight: Any, n: int) -> NDArray[np.float64]:
    if sample_weight is None:
        return np.ones(n, np.float64)
    weights = np.asarray(sample_weight, np.float64)
    if weights.shape != (n,):
        raise ValueError(f"sample_weight must have one value per row ({n}), got shape {weights.shape}")
    return weights


class Regressor(Estimator):
    """A regression estimator: ``score`` is the coefficient of determination R² of ``predict``."""

    _estimator_type = "regressor"

    def predict(self, X: Any) -> NDArray[Any]:  # noqa: N803 - scikit-learn's name
        """Return the prediction of each row of ``X``."""
        raise NotImplementedError

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:  # noqa: N803 - scikit-learn's name
        """Return R² = 1 − Σ w (y − ŷ)² / Σ w (y − ȳ)², with ȳ the weighted mean of ``y`` (1.0 when y is constant)."""
        target = np.asarray(y, np.float64)
        weights = _weights(sample_weight, target.size)
        residual = float(np.sum(weights * (target - np.asarray(self.predict(X), np.float64)) ** 2))
        total = float(np.sum(weights * (target - np.average(target, weights=weights)) ** 2))
        if total == 0:
            return 1.0 if residual == 0 else 0.0
        return 1.0 - residual / total


class Classifier(Estimator):
    """A classification estimator: ``score`` is the (weighted) share of rows whose ``predict`` equals ``y``."""

    _estimator_type = "classifier"

    def predict(self, X: Any) -> NDArray[Any]:  # noqa: N803 - scikit-learn's name
        """Return the predicted label of each row of ``X``."""
        raise NotImplementedError

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:  # noqa: N803 - scikit-learn's name
        """Return the accuracy of ``predict(X)`` against the labels ``y``."""
        labels = np.asarray(y)
        weights = _weights(sample_weight, labels.size)
        return float(np.average(np.asarray(self.predict(X)) == labels, weights=weights))
