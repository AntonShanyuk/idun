# idun-numerai

Numerai on top of [idun](https://pypi.org/project/idun/): Numerai's data in idun's packed format, walk-forward refits
of your choice, the labels, Numerai's scoring functions in NumPy, and model uploads. It ships the tools, not a
configuration: every learner's parameters are idun's
([the learners' parameters](https://github.com/AntonShanyuk/idun/tree/main/docs/learners)).

```bash
pip install idun-numerai    # Python 3.12–3.14; idun's Metal backend needs Apple silicon with macOS 15 or later
```

## Train and score

```python
from idun import gbm
from idun.data import PackedDataset
from idun_numerai.data import build_cache
from idun_numerai.download import download
from idun_numerai.score import corr, per_era

download("v6")  # Numerai's v6 training and validation files and the feature list (public data, no credentials)
# Streamed into a memory-mapped packed dataset (never the whole panel in memory): one target, and walk-forward
# windows that train on every era up to `purge` eras before each date and predict from it until the next.
build_cache("v6", "cache", seed=0, target="target", refit_dates=("2020-01-03", "2023-01-06"), purge=12)

data = PackedDataset.open("cache", binning="r1")  # the second window's binning, fitted on its training rows
window = data.refits[1]
params = gbm.Params(num_trees=300, learning_rate=0.05, feature_fraction=0.25)  # an illustration, not a tuned setting
model = gbm.train(params, data, rows=slice(0, window.train_hi), backend="metal")
rows = slice(window.pred_lo, window.pred_hi)
scores = per_era(corr, data.era[rows], model.predict(data, rows=rows, backend="metal"), data.target[rows])
```

- `idun_numerai.download`, `idun_numerai.data.build_cache`: the data, checked and packed (4-bit codes, the named
  factors binned per window, the country as a categorical column).
- `idun_numerai.refits`: the walk-forward windows (training rows, a purge, the predicted rows) for the dates and
  purge you give.
- `idun_numerai.inputs`: the labels: classes `rint(4 · y)` for the networks' classification, per-era z for regression.
- `idun_numerai.score`: Numerai's per-era scores (`corr`, `ncorr`, `per_era`) in NumPy and SciPy, as `numerai-tools`
  defines them (checked against it per era in the tests; `numerai-tools` is not a dependency).

The target, the refit dates and the purge are arguments of `build_cache` and `refit_table`, without defaults.

## Model uploads

Numerai's model uploads run a pickled `predict(live_features, live_benchmark_models)` on its servers, where idun is not
installed. `idun_numerai.upload` carries trained models as arrays and predicts with NumPy alone:

```python
from idun_numerai import upload

predict = upload.predictor([model, [tabm_s0, tabm_s1], realmlp], data, weights=[2, 1, 1])
upload.save(predict, "model.pkl")  # run under Python 3.12, the runtime's
```

idun_gbm's upload predictions equal `Booster.predict`'s bit for bit; TabM's and RealMLP's are their float64 reference
backend's. A list is a bag (its mean); several entries are blended by rank.

## Licence

Apache-2.0.
