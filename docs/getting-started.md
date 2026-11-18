# Getting started

## Install

```bash
pip install idun
```

Python 3.12 to 3.14. What you get depends on the platform:

| platform | wheel | backends |
|---|---|---|
| Apple silicon (M1 or newer), macOS 15 or later | `macosx_15_0_arm64`, with the Metal library | `"metal"`, and the CPU (`"cpu"` for idun_gbm, `"reference"` for the networks) |
| Linux with an NVIDIA GPU | the pure wheel | `"cuda"` once NVRTC is installed (below), and the CPU |
| Linux without a GPU, Intel Macs | the pure wheel | the CPU |

### CUDA

The CUDA backend is part of the pure wheel; it needs no native library of idun's, only NVIDIA's driver and NVRTC, its
runtime compiler, which builds the kernels for your GPU the first time a backend is made:

```bash
pip install idun nvidia-cuda-nvrtc
```

You need Linux, an NVIDIA GPU of compute capability 7.5 or newer (Turing: a T4 and later) and its driver. NVRTC can be
the `nvidia-cuda-nvrtc` wheel or the CUDA toolkit's. idun builds a binary for the device itself rather than PTX, so
the driver only needs the same CUDA major version as NVRTC: Kaggle's CUDA 13.0 driver runs with the newest NVRTC 13.x
wheel. Tested on a Tesla T4 and an RTX A5000 under CUDA 13 drivers, with NVRTC 13 and 12; a CUDA 12 driver is
untested. With several GPUs, `IDUN_CUDA_DEVICE=1` puts a process's fits on GPU 1 (idun_gbm's seed bags use every GPU
by themselves).

## A first fit with each learner

```python
import numpy as np

from idun.gbm import GBMRegressor
from idun.jarf import JARFRegressor
from idun.realmlp import RealMLPClassifier
from idun.tabm import TabMRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(25_000, 8))
y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(scale=0.3, size=len(X))
X_train, X_test, y_train = X[:20_000], X[20_000:], y[:20_000]
labels = np.where(y_train > np.median(y_train), "high", "low")

gbm = GBMRegressor(num_trees=300, learning_rate=0.05).fit(X_train, y_train)
tabm = TabMRegressor(epochs=30).fit(X_train, y_train)
mlp = RealMLPClassifier(n_seeds=2).fit(X_train, labels)
jarf = JARFRegressor(num_trees=300).fit(X_train, y_train)

pred = gbm.predict(X_test)
proba = mlp.predict_proba(X_test)
```

[`examples/quickstart.py`](https://github.com/AntonShanyuk/idun/blob/main/examples/quickstart.py) is the same tour
as a script: the backends compared, a refit's bits, saving and loading.

The estimators run on Metal by default; on Linux, pass `backend="cpu"` (idun_gbm), `"cuda"` or `"reference"`
(the networks), as [the next section](#choosing-the-backend) explains. `X` is a 2-D NumPy array, a pandas DataFrame or an Arrow table, with missing values allowed. The estimators follow
scikit-learn's conventions (parameters in the constructor, `fit` returns the estimator, `get_params`, `set_params`,
`score`) without depending on scikit-learn, and pass its `check_estimator` but for two checks
([Python API](api/python-api.md#estimators)).

## Choosing the backend

The estimators run on `"metal"` unless told otherwise, and idun never picks a backend for you: on a machine without
Metal, a Metal fit raises `idun.errors.BackendUnavailableError` naming what is missing instead of quietly running
somewhere slower. Say where each fit runs:

```python
GBMRegressor(backend="cuda")  # an NVIDIA GPU
GBMRegressor(backend="cpu")  # idun_gbm's CPU backend: the same trees, bit for bit
TabMRegressor(backend="reference")  # the networks' float64 NumPy reference: slow, for checks and small tables
```

## Saving and loading

```python
import idun

gbm.save("model.idun")
model = idun.load("model.idun")  # any idun estimator; anything else is refused
```

A model file is an uncompressed `.npz` without pickled objects. Within a major version, a file loads in every later
release.

## The lower-level interface

Each learner has a functional layer under its estimators, for what they do not expose:

```python
import idun
from idun import gbm

data = idun.Dataset(X, y, categorical=["sector"])  # binned once
params = gbm.Params(num_trees=500, learning_rate=0.05, extra_trees=True)
booster = gbm.train(params, data, rows=slice(0, len(y)), backend="metal")
bag = gbm.train_seeds(params, data, rows=slice(0, len(y)), backend="cuda", seeds=[0, 1, 2, 3])
```

`gbm.train` takes custom objectives, validation sets and callbacks; `idun.tabm.train_tabm` and
`idun.realmlp.train_realmlp` train one network from a `TabMParams` or `RealMLPParams` on a matrix or a packed dataset,
with checkpoints, callbacks and your own loss. The learner pages ([idun_gbm](learners/gbm.md),
[idun_tabm](learners/tabm.md), [idun_realmlp](learners/realmlp.md), [JARF](learners/jarf.md)) list every parameter;
the [Python API](api/python-api.md) has the rest.

## Determinism

The same data, parameters and seed give the same bits, run after run: idun_gbm on every backend alike, the networks
on the same GPU. Seeds are explicit (`seed`, and a seed bag's `seed`, `seed + 1`, ...); nothing reads the clock or a
global generator. What is guaranteed where is on the [Determinism and testing](testing.md) page.

## Next

- [Performance guide](performance.md): getting the most out of Metal and CUDA.
- [Benchmarks](benchmarks.md): accuracy against the reference implementations, and speed.
