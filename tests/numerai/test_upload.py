"""idun_numerai.upload: the predict function's models equal idun's predictions, and it runs without idun."""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from idun import gbm, jarf
from idun.data import PackedDataset
from idun.realmlp import RealMLPParams, train_realmlp
from idun.tabm import TabMParams, directions, reference, train_tabm
from idun_numerai import upload
from idun_numerai.inputs import classes, per_era_z
from tests.unit.data.synthetic import TARGET, Synthetic


@pytest.fixture(scope="module")
def data(cache_dir: Path) -> PackedDataset:
    return PackedDataset.open(cache_dir, binning="r1")


def live(synthetic: Synthetic, data: PackedDataset) -> tuple[pd.DataFrame, slice]:
    """Return the last era with targets as Numerai's live features (validation.parquet's rows), and its cache rows."""
    day = int(data.era[-1])
    table = pq.read_table(synthetic.v6 / "validation.parquet")
    table = table.filter(pc.is_valid(table.column(TARGET)))
    dates = (np.array(table.column("date").to_pylist(), "datetime64[D]") - np.datetime64("1970-01-01", "D")).astype(int)
    frame = table.filter(dates == day).to_pandas()
    frame.index = pd.Index([f"id{i}" for i in range(len(frame))], name="id")
    lo = int(np.searchsorted(data.era, day))
    assert frame["numerai_ticker"].tolist() == [data.tickers[t] for t in data.ticker[lo:]]  # the cache's row order
    return frame, slice(lo, data.n_rows)


def boosters(data: PackedDataset) -> dict[str, Any]:
    train = slice(0, data.refits[-1].train_hi)
    depthwise = gbm.Params(num_trees=12, learning_rate=0.1, max_depth=3, max_leaves=8, min_data_in_leaf=5,
                           feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1)  # fmt: skip
    leafwise = dataclasses.replace(depthwise, growth="leafwise", max_depth=-1, max_leaves=6)
    bag = [dataclasses.replace(depthwise, num_trees=8, max_depth=2, max_leaves=4, seed=s) for s in (0, 1)]
    return {
        "depthwise": gbm.train(depthwise, data, rows=train, backend="cpu"),
        "leafwise": gbm.train(leafwise, data, rows=train, backend="cpu"),
        "bag": [gbm.train(params, data, rows=train, backend="cpu") for params in bag],
    }


@pytest.fixture(scope="module")
def models(data: PackedDataset) -> dict[str, Any]:
    train = slice(0, data.refits[-1].train_hi)
    labels = classes(data.target)
    z = per_era_z(data.target, data.era)
    tabm = TabMParams(n_classes=5, members=4, blocks=2, width=16, batch_size=32, epochs=1)
    realmlp = RealMLPParams(
        n_classes=5, blocks=2, hidden=16, frequencies=4, embed=2, batch_size=32, epochs=1, init_rows=64
    )
    hlg = dataclasses.replace(tabm, objective="hl_gauss", n_classes=1, n_bins=16)
    return {
        **boosters(data),
        "tabm": train_tabm(tabm, data, rows=train, backend="reference", classes=labels),
        "realmlp": train_realmlp(realmlp, data, rows=train, backend="reference", classes=labels),
        "tabm_mse": train_tabm(
            dataclasses.replace(tabm, objective="mse", n_classes=1), data, rows=train, backend="reference", targets=z
        ),
        "realmlp_mse": train_realmlp(
            dataclasses.replace(realmlp, objective="mse", n_classes=1), data, rows=train, backend="reference", targets=z
        ),
        "tabm_hlg": train_tabm(hlg, data, rows=train, backend="reference", targets=z),
    }


def test_each_model_predicts_as_idun(synthetic: Synthetic, data: PackedDataset, models: dict[str, Any]) -> None:
    frame, rows = live(synthetic, data)
    names = ["depthwise", "leafwise", "bag", "tabm", "realmlp", "tabm_mse", "realmlp_mse", "tabm_hlg"]
    raw = upload.predictor([models[n] for n in names], data).raw(frame)  # type: ignore[attr-defined]
    for name in ("depthwise", "leafwise"):
        assert raw[names.index(name)].dtype == np.float32
        assert np.array_equal(raw[names.index(name)], models[name].predict(data, rows=rows, backend="cpu")), name
    bag = gbm.predict_mean(models["bag"], data, rows=rows, backend="cpu")
    assert np.array_equal(raw[2], bag)
    for name in ("tabm", "realmlp", "tabm_mse", "realmlp_mse", "tabm_hlg"):
        np.testing.assert_allclose(raw[names.index(name)], models[name].predict(data, rows=rows, backend="reference"),
                                   rtol=0, atol=1e-6, err_msg=name)  # fmt: skip
    assert all(np.ptp(r) > 0 for r in raw)  # the models do separate the rows


def test_the_prediction_is_the_rank_blend(synthetic: Synthetic, data: PackedDataset, models: dict[str, Any]) -> None:
    frame, _ = live(synthetic, data)
    one = upload.predictor([models["depthwise"]], data)
    out = one(frame, pd.DataFrame(index=frame.index))
    assert list(out.columns) == ["prediction"]
    assert out.index.equals(frame.index)
    raw = one.raw(frame)[0]  # type: ignore[attr-defined]
    assert np.array_equal(out["prediction"].to_numpy(), pd.Series(raw).rank(pct=True).to_numpy())
    two = upload.predictor([models["depthwise"], models["tabm"]], data, weights=[3.0, 1.0])
    first, second = (pd.Series(r).rank(pct=True) for r in two.raw(frame))  # type: ignore[attr-defined]
    expected = ((3.0 * first + second) / 4.0).rank(pct=True).to_numpy()
    np.testing.assert_allclose(two(frame, None)["prediction"].to_numpy(), expected, rtol=0, atol=1e-15)
    assert out["prediction"].min() > 0
    assert out["prediction"].max() == 1.0


def test_rank_averages_ties_as_pandas() -> None:
    values = np.array([3.0, 1.0, 3.0, 2.0, 3.0, 1.0], np.float32)
    assert np.array_equal(upload._rank(values), pd.Series(values).rank(pct=True).to_numpy())


def test_live_features_outside_the_training_data_are_refused(
    synthetic: Synthetic, data: PackedDataset, models: dict[str, Any]
) -> None:
    frame, _ = live(synthetic, data)
    predict = upload.predictor([models["depthwise"]], data)
    obfuscated, country = data.obfuscated_features[0], data.feature_names[-1]
    with pytest.raises(ValueError, match="lacks 1 of the models' features"):
        predict(frame.drop(columns=[obfuscated]), None)
    for value, message in ((5.0, "not one of 0..4"), (np.nan, "not one of 0..4"), (1.5, "not one of 0..4")):
        bad = frame.copy()
        bad.loc[bad.index[2], obfuscated] = value
        with pytest.raises(ValueError, match=message):
            predict(bad, None)
    bad = frame.copy()
    bad.loc[bad.index[0], country] = "XX"
    with pytest.raises(ValueError, match=r"levels the training data has not: \['XX'\]"):
        predict(bad, None)
    bad.loc[bad.index[0], country] = None
    with pytest.raises(ValueError, match="missing values"):
        predict(bad, None)
    assert predict(frame.assign(extra=1.0), None).shape == (len(frame), 1)  # other columns are ignored


def test_models_that_do_not_fit_are_refused(data: PackedDataset, models: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least one model"):
        upload.predictor([], data)
    with pytest.raises(ValueError, match="positive finite"):
        upload.predictor([models["depthwise"]], data, weights=[0.0])
    with pytest.raises(ValueError, match="positive finite"):
        upload.predictor([models["depthwise"]], data, weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="one kind"):
        upload.predictor([[models["depthwise"], models["tabm"]]], data)
    with pytest.raises(ValueError, match="one kind"):
        upload.predictor([[]], data)
    with pytest.raises(TypeError, match="not an idun_gbm"):
        upload.predictor(["model"], data)  # type: ignore[list-item]
    other = gbm.Booster(models["depthwise"].params, models["depthwise"].forest, binning="r1",
                        feature_names=("a", *models["depthwise"].feature_names[1:]),
                        categories=models["depthwise"].categories, named_edges=models["depthwise"].named_edges,
                        named_n_bins=models["depthwise"].named_n_bins, train_rows=(0, 1), backend="cpu")  # fmt: skip
    with pytest.raises(ValueError, match="other features"):
        upload.predictor([other], data)
    tabm = models["tabm"]
    with pytest.raises(ValueError, match="one target"):
        upload.predictor([type(tabm)(tabm.params, tabm.weights, tabm.n_inputs, tabm.train_rows, tabm.report,
                                     multi_target=True)], data)  # fmt: skip
    with pytest.raises(ValueError, match="reads 7 inputs"):
        upload.predictor([type(tabm)(tabm.params, tabm.weights, 7, tabm.train_rows, tabm.report)], data)
    temporal = dataclasses.replace(tabm.params, encoder="gru", encoder_embed=4, encoder_hidden=4)
    weights = reference.init(temporal, tabm.n_inputs, sequence_inputs=3)
    with pytest.raises(ValueError, match="temporal encoder"):
        upload.predictor([type(tabm)(temporal, weights, tabm.n_inputs, tabm.train_rows, tabm.report,
                                     sequence_inputs=3)], data)  # fmt: skip


RUNTIME = """
import sys
for name in ("idun", "idun_numerai", "numba", "llvmlite"):  # none of them exists where Numerai predicts
    sys.modules[name] = None
import cloudpickle, numpy as np, pandas as pd
predict = cloudpickle.loads(open(sys.argv[1], "rb").read())
frame = pd.read_pickle(sys.argv[2])
np.save(sys.argv[3], predict(frame, None)["prediction"].to_numpy())
np.save(sys.argv[4], np.stack(predict.raw(frame)))
assert not any(m == "idun" or m.startswith(("idun.", "idun_numerai", "numba")) for m in sys.modules if sys.modules[m])
"""


def test_the_saved_function_runs_without_idun(
    synthetic: Synthetic, data: PackedDataset, models: dict[str, Any], tmp_path: Path
) -> None:
    frame, _ = live(synthetic, data)
    predict = upload.predictor([models[n] for n in ("depthwise", "leafwise", "bag", "tabm", "realmlp", "tabm_hlg")],
                               data)  # fmt: skip
    path = upload.save(predict, tmp_path / "model.pkl", python=sys.version_info[:2])
    frame.to_pickle(tmp_path / "live.pkl")
    script = tmp_path / "runtime.py"
    script.write_text(RUNTIME)
    paths = [str(tmp_path / n) for n in ("model.pkl", "live.pkl", "prediction.npy", "raw.npy")]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, "-I", str(script), *paths], check=True, env=env, cwd=tmp_path)
    assert path.stat().st_size > 0
    assert np.array_equal(np.load(tmp_path / "prediction.npy"), predict(frame, None)["prediction"].to_numpy())
    assert np.array_equal(np.load(tmp_path / "raw.npy"), np.stack(predict.raw(frame)))  # type: ignore[attr-defined]


def test_save_refuses_another_python(data: PackedDataset, models: dict[str, Any], tmp_path: Path) -> None:
    predict = upload.predictor([models["depthwise"]], data)
    other = (3, sys.version_info[1] + 1)
    with pytest.raises(RuntimeError, match=f"the runtime runs 3.{other[1]}"):
        upload.save(predict, tmp_path / "model.pkl", python=other)
    assert not (tmp_path / "model.pkl").exists()


def test_a_jarf_model_predicts_as_idun(synthetic: Synthetic, data: PackedDataset, models: dict[str, Any]) -> None:
    frame, rows = live(synthetic, data)
    found = directions.fit_directions([models["tabm"]], data, rows=np.arange(0, data.refits[-1].train_hi, 2), k=3)
    extended = jarf.extend(data, found, edge_rows=np.arange(0, data.refits[-1].train_hi, 3))
    params = gbm.Params(num_trees=6, max_depth=2, max_leaves=4, min_data_in_leaf=5)
    train = slice(0, data.refits[-1].train_hi)
    whole = jarf.JARFModel(gbm.train(params, extended, rows=train, backend="cpu"), found)
    kept = (data.obfuscated_features[1], data.obfuscated_features[4])
    reduced = jarf.keep(extended, kept)
    fewer = jarf.JARFModel(gbm.train(params, reduced, rows=train, backend="cpu"), found, kept)
    bag = [whole, jarf.JARFModel(gbm.train(dataclasses.replace(params, seed=1), extended, rows=train, backend="cpu"),
                                 found)]  # fmt: skip
    raw = upload.predictor([whole, fewer, bag], data).raw(frame)  # type: ignore[attr-defined]
    assert raw[0].dtype == np.float32
    assert np.array_equal(raw[0], whole.predict(data, rows=rows, backend="cpu"))
    assert np.array_equal(raw[1], fewer.predict(data, rows=rows, backend="cpu"))
    mean = np.mean([m.predict(data, rows=rows, backend="cpu") for m in bag], axis=0, dtype=np.float64)
    assert np.array_equal(raw[2], mean.astype(np.float32))
    assert all(np.ptp(r) > 0 for r in raw)
    other = jarf.JARFModel(whole.booster, found, kept)  # the booster's features do not fit the kept set
    with pytest.raises(ValueError, match="other features"):
        upload.predictor([other], data)
    with pytest.raises(ValueError, match="one kind"):
        upload.predictor([[whole, models["depthwise"]]], data)
