"""benchmarks/openml.py: the split, the column rules, the metrics, the plan, runs on a toy task and the report."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks import openml


def test_the_split_is_seeded_disjoint_and_complete() -> None:
    train, val, test = openml.split(1_000)
    assert (train.size, val.size, test.size) == (640, 160, 200)
    assert np.array_equal(np.sort(np.concatenate([train, val, test])), np.arange(1_000))
    again = openml.split(1_000)
    assert all(np.array_equal(a, b) for a, b in zip((train, val, test), again, strict=True))


def test_column_rules() -> None:
    n = 100
    table = pa.table({
        "num": pa.array([float(i) if i % 10 else None for i in range(n)]),
        "flag": pa.array(["1" if i % 3 else "0" for i in range(n)]).dictionary_encode(),
        "cat": pa.array(["a" if i < 60 else "b" if i < 95 else None if i < 97 else "c" for i in range(n)]),
        "y": pa.array(["no" if i % 2 else "yes" for i in range(n)]).dictionary_encode(),
    })  # fmt: skip
    frame, y, categorical, classes = openml.prepare_table(table, "y", "binary")
    assert categorical == ["cat"]
    assert np.isnan(frame["num"].to_numpy()[0])
    assert frame["flag"].tolist()[:3] == [0.0, 1.0, 1.0]
    levels = frame["cat"].tolist()
    assert (levels.count("a"), levels.count("b")) == (60, 35)
    assert levels.count(openml.RARE_LEVEL) == 5  # 2 missing and 3 "c": both seen in fewer than 30 rows
    assert classes == ["no", "yes"]
    assert y.tolist()[:2] == [1, 0]
    _, target, _, none = openml.prepare_table(pa.table({"x": [1.0, 2.0], "t": [3, 4]}), "t", "regression")
    assert target.tolist() == [3.0, 4.0]
    assert none == []


def test_metrics() -> None:
    assert openml.rmse(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == 1.0
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    scores = openml.metrics("binary", np.array([0, 1, 1]), proba)
    assert scores["log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.4])))
    assert scores["accuracy"] == pytest.approx(2 / 3)
    assert (openml.primary("regression"), openml.primary("multiclass")) == ("rmse", "log_loss")


def test_the_plan_runs_each_learner_where_it_applies_once() -> None:
    runs = openml.plan()
    assert len(runs) == len(set(runs))
    assert {t for t, learner, _ in runs if learner in openml.GBM_LEARNERS} == set(openml.GBM_TASKS)
    assert all(openml.TASKS[t].kind == "regression" for t in openml.GBM_TASKS)
    for task in openml.TASKS:
        seeds = {s for t, learner, s in runs if t == task and learner == "idun_tabm"}
        assert seeds == set(openml.LARGE_SEEDS if task in openml.LARGE_TASKS else openml.SMALL_SEEDS)
    assert {learner for _, learner, _ in runs} == set(openml.LEARNERS)


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> openml.Settings:
    """A small regression task registered as "toy", its file in ``tmp_path/data``."""
    rng = np.random.default_rng(0)
    n = 4_000
    x = rng.normal(size=(n, 3))
    sector = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    y = x[:, 0] - 2 * x[:, 1] + (sector == "b") + rng.normal(scale=0.1, size=n)
    data = tmp_path / "data"
    data.mkdir()
    path = data / "dataset_1.pq"
    pq.write_table(pa.table({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "sector": sector, "y": y}), path)
    monkeypatch.setitem(openml.TASKS, "toy", openml.Task("toy", 1, openml.sha256(path), "y", "regression"))
    return openml.Settings(data=data, out=tmp_path / "out", smoke=True)


def test_a_run_writes_its_record_and_predictions_once(
    toy: openml.Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def mean(data: openml.Prepared, seed: int, epochs: int | None) -> tuple[Any, float, float]:
        assert (seed, epochs) == (3, 2)  # a smoke run's epochs
        return np.full(data.test.size, data.y[data.train].mean()), 1.5, 0.5

    monkeypatch.setitem(openml.LEARNERS, "mean", mean)
    path = openml.run("toy", "mean", 3, toy)
    assert path == toy.out / "smoke" / "toy" / "mean_s3.json"
    record = json.loads(path.read_text())
    assert (record["task"], record["learner"], record["seed"], record["smoke"]) == ("toy", "mean", 3, True)
    assert (record["n_train"], record["n_test"], record["fit_seconds"]) == (2_000, 500, 1.5)
    assert record["metrics"]["rmse"] > 1.0
    assert np.load(path.with_suffix(".npy")).shape == (500,)
    assert {"idun", "numpy", "torch"} <= set(record["versions"])
    openml.run("toy", "mean", 3, toy)
    assert "done already" in capsys.readouterr().out


def test_a_changed_file_is_refused(toy: openml.Settings) -> None:
    path = toy.data / "dataset_1.pq"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="not the expected file"):
        openml.prepare(openml.TASKS["toy"], toy.data)
    (toy.data / openml.TASKS["california"].filename).write_bytes(b"not parquet")
    with pytest.raises(RuntimeError, match="sha256"):
        openml.fetch(toy.data)  # the planted file is checked, nothing downloaded


@pytest.mark.metal
def test_idun_gbm_runs_and_the_report_lists_it(toy: openml.Settings, capsys: pytest.CaptureFixture[str]) -> None:
    openml.main(["--data", str(toy.data), "--out", str(toy.out), "--smoke", "run", "toy", "idun_gbm", "--seed", "0"])
    record = json.loads(toy.record("toy", "idun_gbm", 0).read_text())
    assert record["metrics"]["rmse"] < 0.5
    capsys.readouterr()
    openml.main(["--data", str(toy.data), "--out", str(toy.out), "--smoke", "report"])
    assert "| toy | idun_gbm | 1 | rmse |" in capsys.readouterr().out


def record(task: str, learner: str, seed: int, value: float, seconds: float) -> dict[str, Any]:
    metric = openml.primary(openml.TASKS[task].kind)
    return {"task": task, "learner": learner, "seed": seed, "metrics": {metric: value}, "fit_seconds": seconds}


def test_the_report_compares_each_learner_with_its_reference(tmp_path: Path) -> None:
    rows = [record("california", "idun_gbm", 0, 0.15, 1.0), record("california", "lightgbm", 0, 0.14, 0.5),
            record("california", "xgboost", 0, 0.13, 0.2), record("california", "hgb", 0, 0.145, 0.5),
            record("adult", "idun_tabm", 0, 0.30, 10.0), record("adult", "idun_tabm", 1, 0.32, 12.0),
            record("adult", "tabm_package", 0, 0.31, 20.0), record("adult", "idun_realmlp", 0, 0.4, 5.0)]  # fmt: skip
    table = openml.summary(rows)
    assert table[("adult", "idun_tabm")] == {"log_loss": pytest.approx(0.31), "fit_seconds": 11.0, "seeds": 2}
    assert openml.reference("california", "idun_gbm", table) == 0.14  # the median of the three libraries
    assert openml.reference("adult", "idun_tabm", table) == 0.31
    assert openml.reference("adult", "idun_realmlp", table) is None  # pytabkit's record is missing
    assert openml.reference("california", "lightgbm", table) is None
    settings = openml.Settings(out=tmp_path)
    for r in rows:
        path = settings.record(r["task"], r["learner"], r["seed"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(r))
    lines = openml.report(settings).splitlines()
    assert lines[2] == "| california | idun_gbm | 1 | rmse | 0.15 | 1.0714 | 1.0 |"
    assert "| adult | idun_tabm | 2 | log_loss | 0.31 | 1.0000 | 11.0 |" in lines
    assert "| adult | idun_realmlp | 1 | log_loss | 0.4 |  | 5.0 |" in lines
    with pytest.raises(SystemExit, match="no records"):
        openml.report(openml.Settings(out=tmp_path / "empty"))
