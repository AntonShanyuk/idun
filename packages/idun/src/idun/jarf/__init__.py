"""JARF (Rauniyar 2025): idun_gbm in front of which go the directions a trained idun_tabm is most sensitive to.

The directions come from `idun.tabm.directions`; this package composes the two learners, which stay independent of
each other: `extend`, `apply` and `keep` build the trees' data (`Front`), `JARFModel` keeps a booster with the
directions and the kept features it was trained behind, and `JARFRegressor` fits the surrogates, the directions and the
trees in one `fit` (docs/api/python-api.md).
"""

from idun.jarf.estimator import JARFRegressor, quantile_classes
from idun.jarf.front import Front, apply, direction_names, extend, keep
from idun.jarf.model import JARFModel

__all__ = ["Front", "JARFModel", "JARFRegressor", "apply", "direction_names", "extend", "keep", "quantile_classes"]
