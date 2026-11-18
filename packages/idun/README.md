# idun

Tabular learning on the GPU: Metal on Apple silicon, CUDA on NVIDIA. idun trains three learners on either, each
with a CPU reference backend:

- **idun_gbm** (`idun.gbm`): histogram gradient boosting with LightGBM's model (depth-wise or leaf-wise trees,
  quantized gradients, categorical splits, seed bags). Its CPU, Metal and CUDA backends build the same model, bit
  for bit.
- **idun_tabm** (`idun.tabm`): TabM (Gorishniy et al., ICLR 2025), a parameter-efficient ensemble of MLPs, for
  classification, regression, and HL-Gauss regression (a target's Gaussian-smoothed bins).
- **idun_realmlp** (`idun.realmlp`): RealMLP-TD (Holzmüller et al., NeurIPS 2024).
- **JARF** (`idun.jarf`): idun_gbm in front of which go the directions a trained idun_tabm is most sensitive to
  (Rauniyar 2025), as a feature front and as one scikit-learn-style estimator.

Every fit is deterministic: the same data, parameters and seed give the same bits, run after run (the networks on the
same GPU). The estimators follow scikit-learn's conventions without depending on it.

idun follows semantic versioning: within a major version a public name, its parameters and their defaults stay, and a
model file loads in every later release.

## Install

```bash
pip install idun            # Apple silicon with macOS 15 or later: the Metal wheel; elsewhere the pure wheel
pip install nvidia-cuda-nvrtc  # Linux with an NVIDIA GPU, for backend="cuda" (or the CUDA toolkit's NVRTC)
pip install idun-numerai    # the Numerai port
```

Python 3.12–3.14; the only dependencies are NumPy and numba. Metal needs an Apple silicon Mac (M1 or newer) with
macOS 15 or later. CUDA (`backend="cuda"`) needs Linux, an NVIDIA GPU of compute capability 7.5 or newer (a T4 and
later) with its driver, and NVRTC (the CUDA toolkit's, or the `nvidia-cuda-nvrtc` wheel of the driver's CUDA
version): the kernels are compiled when a backend is made, so the pure wheel carries the CUDA backend. Its tests
pass on a T4 (Turing) and an RTX A5000 (Ampere) with CUDA 13 drivers, with NVRTC 13 and with NVRTC 12; a CUDA 12
driver is untested. On Linux and Intel Macs the package installs without the Metal library; a backend that cannot
run raises `idun.errors.BackendUnavailableError` rather than falling back.

## Quick start

```python
import idun
from idun.gbm import GBMRegressor
from idun.realmlp import RealMLPRegressor
from idun.tabm import TabMClassifier

gbm = GBMRegressor(num_trees=300, learning_rate=0.05).fit(X_train, y_train)
pred = gbm.predict(X_test)
cpu = GBMRegressor(num_trees=300, learning_rate=0.05, backend="cpu").fit(X_train, y_train)  # the same trees

clf = TabMClassifier(categorical=["sector"]).fit(X_train, labels)  # any labels, two classes or more
proba = clf.predict_proba(X_test)

mlp = RealMLPRegressor(n_seeds=4).fit(X_train, y_train)  # a bag of four seeds; predicts their mean
each = mlp.predict(X_test, per_seed=True)  # (n, 4): each seed's prediction

gbm.save("gbm.idun")
same = idun.load("gbm.idun")
```

`X` is a 2-D NumPy array, a pandas DataFrame or an Arrow table; missing values are allowed. The backend is `"metal"`
by default; pass `backend="cpu"` (idun_gbm) or `backend="reference"` (the networks' float64 reference) to run on the
CPU. Nothing switches backends behind your back.

Below the estimators: `idun.Dataset` bins a table once for reuse; the functional layer (`idun.gbm.train`,
`idun.tabm.train_tabm`, `idun.realmlp.train_realmlp`, seed bags through `train_seeds` and `predict_mean`) trains on
binned data or on `idun.data.PackedDataset`, a memory-mapped, date-sorted panel for data larger than memory, which
Metal reads without a copy. The docstrings document every public name.

## How it compares

On five OpenML tasks on an Apple M1 (8 GB): idun_gbm's RMSE is within 3 % of the median of LightGBM, XGBoost and
scikit-learn's HistGradientBoosting at the same nominal settings; idun_tabm and idun_realmlp are within 2 % of their
reference implementations (the `tabm` package and pytabkit on Torch's MPS device) on every task and better on most, and
idun_tabm fits in 0.45 of the reference's time.

## Licence

Apache-2.0. The NOTICE file lists the work idun builds on, including MLX's Steel GEMM (MIT), whose structure the
networks' GEMM kernel adapts.
