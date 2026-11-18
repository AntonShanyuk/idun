# JARF: boosted trees behind a network's directions

JARF (Rauniyar, 2025) combines the two kinds of learner: a trained network responds to combinations of the inputs,
while a tree splits one input at a time. JARF takes the directions in input space the network is most sensitive to,
the top eigenvectors of its expected gradient outer product (EGOP), and gives them to the trees as extra features, so
that an axis-aligned split on such a feature acts as an oblique split in the original inputs. The mechanism is on the
[JARF architecture page](../architecture/jarf.md).

`idun.jarf.JARFRegressor` is the whole method as one estimator: idun_tabm networks as the surrogates, idun_gbm as the
trees. Both learners stay independent of each other; `idun.tabm.directions` and `idun.jarf.extend` are the pieces for
building it yourself on a packed dataset.

## A first fit

```python
from idun.jarf import JARFRegressor

model = JARFRegressor(num_trees=500, learning_rate=0.05, k=32, n_surrogates=2, backend="metal").fit(X_train, y_train)
pred = model.predict(X_test)
model.save("jarf.idun")  # idun.load reads it back; the surrogates are not kept, their directions are
```

`fit(X, y)`:

1. trains `n_surrogates` `TabMClassifier`s on `y`'s classes (its distinct values when there are at most
   `surrogate_classes` of them, else its quantile bins), with `surrogate_params` on `surrogate_backend`, or takes fitted
   ones (`surrogates=[...]`);
2. computes one EGOP over the surrogates together, the mean of the gradients' outer products over `egop_rows` rows
   drawn with `seed` and every network, and its top `k` directions;
3. appends the rows' projections on them to the table as numeric columns `tabm_direction_0` to `tabm_direction_{k-1}`,
   keeping every column, or with `keep` the most sensitive non-categorical ones;
4. fits a `GBMRegressor` with the tree parameters on that table.

After `fit`: `directions_`, `sensitivity_` (per encoded input), `kept_` and `gbm_`.

## Parameters

The tree parameters (`num_trees`, `learning_rate`, `max_depth`, ...) and `n_seeds`, `workers`, `backend`,
`categorical` and `codes` are `GBMRegressor`'s, with its defaults: see [idun_gbm](gbm.md). `JARFRegressor` bins its
front table itself, so it does not take `max_bin` or `subsample_for_bin`; nor, yet, `second_order`, `extra_trees` or
`feature_fraction_bynode`.

| parameter | default | what it does |
|---|---|---|
| `k` | `16` | the directions appended (at most the encoded inputs) |
| `egop_rows` | `10000` | the rows the EGOP is computed on, a seeded sample |
| `keep` | `None` | keep only this many of the original non-categorical columns, the most sensitive; `None` keeps all (categorical columns always stay) |
| `n_surrogates` | `1` | the surrogate networks, seeds `seed`, `seed + 1`, ..., read together by the EGOP |
| `surrogate_params` | `None` | `TabMClassifier` keywords for the surrogates; `None` takes its defaults |
| `surrogate_backend` | `"metal"` | the surrogates' backend: `"metal"`, `"cuda"` or `"reference"` |
| `surrogate_classes` | `4` | the most classes the surrogates train on: `y`'s quantile bins when it has more distinct values |
| `surrogates` | `None` | fitted `TabMClassifier`s to take instead of training new ones (their encoders must agree) |
