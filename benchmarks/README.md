# Benchmarks

`openml.py` compares idun's learners with the libraries they reimplement on five OpenML tasks: idun_gbm against
LightGBM, XGBoost and scikit-learn's HistGradientBoosting; idun_tabm against the `tabm` package; idun_realmlp against
pytabkit's RealMLP-TD. The method and the measured results are in [docs/benchmarks.md](../docs/benchmarks.md).

```bash
uv run --frozen --group benchmarks python -m benchmarks.openml fetch          # five parquet files into benchmarks/data
uv run --frozen --group benchmarks python -m benchmarks.openml --smoke all    # every learner on a few thousand rows
uv run --frozen --group benchmarks python -m benchmarks.openml all            # every run: about 7 hours on an M1
uv run --frozen --group benchmarks python -m benchmarks.openml report         # a Markdown table of the records
```

The `benchmarks` dependency group holds the outside libraries (Torch among them); idun itself needs none of them.
Records go to `benchmarks/records/`, one JSON file and one `.npy` of test predictions per run.
