"""idun_realmlp: RealMLP-TD (Holzmüller et al.) on Metal, with a float64 NumPy reference.

`train_realmlp` fits a `RealMLPModel` with `RealMLPParams` (the defaults are pytabkit's RealMLP-TD settings);
`train_realmlp_seeds` fits a seed bag.
"""

from idun.realmlp.estimator import RealMLPClassifier, RealMLPRegressor
from idun.realmlp.params import RealMLPParams
from idun.realmlp.train import RealMLPModel, predict_mean, train_realmlp, train_realmlp_seeds

__all__ = [
    "RealMLPClassifier",
    "RealMLPModel",
    "RealMLPParams",
    "RealMLPRegressor",
    "predict_mean",
    "train_realmlp",
    "train_realmlp_seeds",
]
