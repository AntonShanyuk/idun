from typing import Any

import numpy as np
import pytest

from idun.base import Classifier, Estimator, Regressor
from idun.errors import NotFittedError


class Toy(Estimator):
    def __init__(self, *, a: int = 1, b: str = "x") -> None:
        self.a = a
        self.b = b


class Echo(Regressor):
    """Predicts the first column."""

    def __init__(self) -> None:
        pass

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803 - scikit-learn's name
        return np.asarray(X)[:, 0]


class Sign(Classifier):
    """Predicts the sign of the first column as a label."""

    def __init__(self) -> None:
        pass

    def predict(self, X: Any) -> np.ndarray:  # noqa: N803 - scikit-learn's name
        return np.where(np.asarray(X)[:, 0] > 0, "up", "down")


def test_params_and_repr_follow_scikit_learns_conventions() -> None:
    toy = Toy(a=3)
    assert toy.get_params() == {"a": 3, "b": "x"}
    assert toy.set_params(b="y") is toy
    assert toy.b == "y"
    assert repr(toy) == "Toy(a=3, b='y')"
    assert repr(Toy()) == "Toy()"
    with pytest.raises(ValueError, match="unknown parameters"):
        toy.set_params(c=1)
    with pytest.raises(NotFittedError, match="Toy is not fitted"):
        toy._check_fitted("model_")


def test_checks_name_the_estimator_and_the_fix() -> None:
    toy = Toy()
    with pytest.raises(ValueError, match="Toy requires y to be passed, but the target y is None"):
        toy._require_y(None)
    toy._require_y([1])
    toy.n_features_in_ = 3
    toy._check_n_features(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="X has 1 features, but Toy is expecting 3 features as input"):
        toy._check_n_features(np.zeros((2, 1)))


def test_score_is_r2_for_regressors() -> None:
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert Echo().score(x, [1.0, 2.0, 3.0, 4.0]) == 1.0
    assert Echo().score(x, [2.0, 2.0, 4.0, 4.0]) == pytest.approx(1 - 2 / 4)
    # weighted: residuals 1, 0, 1, 0 with weights 1, 1, 0, 2 sum to 1; the weighted mean is 3, so the total is 4
    assert Echo().score(x, [2.0, 2.0, 4.0, 4.0], sample_weight=[1, 1, 0, 2]) == pytest.approx(0.75)
    assert Echo().score(x, [1.0, 1.0, 1.0, 1.0]) == 0.0  # a constant target the prediction misses
    assert Echo().score(np.ones((3, 1)), [1.0, 1.0, 1.0]) == 1.0
    with pytest.raises(ValueError, match="one value per row"):
        Echo().score(x, [1.0, 2.0, 3.0, 4.0], sample_weight=[1.0])


def test_score_is_accuracy_for_classifiers() -> None:
    x = np.array([[1.0], [-1.0], [2.0], [-3.0]])
    assert Sign().score(x, ["up", "down", "down", "down"]) == 0.75
    assert Sign().score(x, ["up", "down", "down", "down"], sample_weight=[1, 1, 2, 0]) == 0.5


def test_scikit_learns_tags() -> None:
    regressor, classifier = Echo().__sklearn_tags__(), Sign().__sklearn_tags__()
    assert (regressor.estimator_type, classifier.estimator_type) == ("regressor", "classifier")
    assert regressor.regressor_tags is not None
    assert classifier.classifier_tags is not None
    assert regressor.input_tags.allow_nan
    assert regressor.target_tags.required
