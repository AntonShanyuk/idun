import json
from pathlib import Path

import numpy as np
import pytest

import idun


def write(path: Path, meta: dict[str, object] | None) -> Path:
    arrays = {"x": np.zeros(2)} if meta is None else {"meta": np.frombuffer(json.dumps(meta).encode(), np.uint8)}
    np.savez(path, **arrays)
    return path


@pytest.mark.parametrize(
    ("meta", "message"),
    [
        (None, "not an idun model file"),
        ({"format": "idun-gbm", "format_version": 2}, "is not supported"),
        ({"format": "idun-model", "format_version": 9}, "is not supported"),
        ({"format": "idun-model", "format_version": 1, "estimator": "Nope"}, "unknown estimator"),
    ],
)
def test_load_refuses_anything_but_a_known_model(tmp_path: Path, meta: dict[str, object] | None, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        idun.load(write(tmp_path / "m.npz", meta))
