"""idun_tabm: TabM (Gorishniy et al.) on Metal, with a float64 NumPy reference.

`train_tabm` fits a `TabMModel` with `TabMParams`; the defaults are the `tabm` package's ``TabM.make`` defaults plus
fixed training constants (`idun.tabm.params`). `Loss` and `BatchPlan` are the types of its ``loss`` and ``batches``.
"""

from idun.nn.training import Validation
from idun.tabm.directions import Directions, fit_directions
from idun.tabm.estimator import TabMClassifier, TabMRegressor
from idun.tabm.params import TabMParams
from idun.tabm.reference import Loss
from idun.tabm.sequences import Sequences
from idun.tabm.train import BatchPlan, TabMModel, train_tabm

__all__ = ["BatchPlan", "Directions", "Loss", "Sequences", "TabMClassifier", "TabMModel", "TabMParams",
           "TabMRegressor", "Validation", "fit_directions", "train_tabm"]  # fmt: skip
