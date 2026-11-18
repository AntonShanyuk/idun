"""`load`: read a fitted estimator written by its ``save`` (the ``idun-model`` file format, docs/api/python-api.md)."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def load(path: str | Path) -> Any:
    """Return the estimator saved at ``path``; anything but an ``idun-model`` file of a known version raises."""
    from idun.gbm.estimator import MODEL_FORMAT, MODEL_FORMAT_VERSION, GBMClassifier, GBMRegressor  # noqa: PLC0415
    from idun.jarf.estimator import MODEL_FORMAT_VERSION as JARF_VERSION  # noqa: PLC0415
    from idun.jarf.estimator import JARFRegressor  # noqa: PLC0415
    from idun.realmlp.estimator import RealMLPClassifier, RealMLPRegressor  # noqa: PLC0415
    from idun.tabm.estimator import TabMClassifier, TabMRegressor  # noqa: PLC0415

    estimators: dict[str, Any] = {
        cls.__name__: cls
        for cls in (
            GBMClassifier,
            GBMRegressor,
            JARFRegressor,
            RealMLPClassifier,
            RealMLPRegressor,
            TabMClassifier,
            TabMRegressor,
        )
    }
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if "meta" not in arrays:
        raise ValueError(f"{path}: not an idun model file (no meta entry)")
    meta = json.loads(arrays.pop("meta").tobytes())
    versions = (MODEL_FORMAT_VERSION, JARF_VERSION)  # v1: one estimator's file; v2: a JARFRegressor's, trees inside
    if meta.get("format") != MODEL_FORMAT or meta.get("format_version") not in versions:
        raise ValueError(f"{path}: format {meta.get('format')!r} v{meta.get('format_version')!r} is not supported")
    if meta.get("estimator") not in estimators:
        raise ValueError(f"{path}: unknown estimator {meta.get('estimator')!r}")
    return estimators[meta["estimator"]]._from_archive(meta, arrays)
