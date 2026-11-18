# idun

> In Norse mythology Idun (Old Norse *Iðunn*, usually glossed as "ever young" or "the rejuvenating one") keeps the
> apples the gods eat to stay young; she is the wife of Bragi, the god of poetry. Snorri Sturluson's *Prose Edda*
> (13th century) and the older skaldic poem *Haustlöng* tell how the giant Thjazi, in the shape of an eagle, forced
> Loki to lure her out of Asgard and carried her off. Without her apples the gods grew old and grey, until Loki, in
> Freyja's falcon shape, flew her back as a nut in his claws; the eagle chased them, and the gods set its feathers alight
> at Asgard's walls. Her apples are often pictured as golden: gold for Metal, apples for Apple silicon.

Gradient boosting, TabM and RealMLP on the GPU: Metal on Apple silicon, CUDA on NVIDIA. Every fit is deterministic,
and idun_gbm builds the same model, bit for bit, on the CPU, on Metal and on CUDA.

- **idun_gbm**: histogram gradient boosting with LightGBM's model and parameter names (depth-wise or leaf-wise trees,
  quantized gradients, categorical splits, custom objectives, multi-class, TreeSHAP).
- **idun_tabm**: TabM (Gorishniy et al., ICLR 2025), a parameter-efficient ensemble of MLPs, for classification,
  regression and HL-Gauss regression.
- **idun_realmlp**: RealMLP-TD (Holzmüller et al., NeurIPS 2024), as pytabkit builds it.
- **JARF**: idun_gbm behind the directions a trained idun_tabm is most sensitive to (Rauniyar 2025).

The estimators follow scikit-learn's conventions without depending on it. idun follows semantic versioning: within a
major version a public name, its parameters and their defaults stay, and a model file loads in every later release.

## Install

```bash
pip install idun                      # Apple silicon with macOS 15 or later: the Metal wheel; elsewhere the pure wheel
pip install idun nvidia-cuda-nvrtc    # Linux with an NVIDIA GPU, for backend="cuda"
pip install idun-numerai              # Numerai's data, labels, scores and model uploads on top of idun
```

| platform | backends |
|---|---|
| Apple silicon (M1 or newer), macOS 15 or later | `"metal"`; the CPU |
| Linux with an NVIDIA GPU of compute capability 7.5 or newer (a T4 and later) and NVRTC | `"cuda"`; the CPU |
| Linux, Intel Macs | the CPU (`"cpu"` for idun_gbm, `"reference"` for the networks) |

Python 3.12 to 3.14. A backend that cannot run raises `idun.errors.BackendUnavailableError`; idun never picks a backend
for you. [Getting started](docs/getting-started.md) has the details, CUDA's included.

## A first fit

```python
from idun.gbm import GBMRegressor
from idun.jarf import JARFRegressor
from idun.realmlp import RealMLPRegressor
from idun.tabm import TabMClassifier

gbm = GBMRegressor(num_trees=300, learning_rate=0.05, backend="metal").fit(X_train, y_train)
tabm = TabMClassifier(categorical=["sector"], backend="cuda").fit(X_train, labels)  # X_train: a DataFrame
mlp = RealMLPRegressor(n_seeds=4).fit(X_train, y_train)  # a bag of four seeds, predicting their mean
jarf = JARFRegressor(num_trees=300, k=16).fit(X_train, y_train)

pred, proba = gbm.predict(X_test), tabm.predict_proba(X_test)
gbm.save("gbm.idun")  # idun.load("gbm.idun") reads any idun estimator back
```

`X` is a 2-D NumPy array, a pandas DataFrame or an Arrow table; missing values are allowed. Under the estimators,
`idun.gbm.train`, `idun.tabm.train_tabm` and `idun.realmlp.train_realmlp` take custom objectives and losses,
validation sets, callbacks, checkpoints and memory-mapped tables larger than memory.

## Speed

On 5.95 million rows of 1,040 features, 600 trees of depth 6 ([Benchmarks](docs/benchmarks.md)):

| | idun | reference |
|---|---|---|
| idun_gbm, one Tesla T4 | 33.9 s | LightGBM: 769 s on 4 CPU cores, 343 s with its OpenCL build on the T4 |
| idun_gbm, Apple M4 | 48.0 s | LightGBM: 187 s on the M4's 10 CPU cores |
| idun_tabm, an epoch, Tesla T4 | 412 s | the `tabm` package on Torch, float32: 528 s |
| idun_tabm, an epoch, Apple M4 | 553 s | the `tabm` package on Torch's MPS backend: 1,386 s |
| idun_realmlp, an epoch, Tesla T4 | 76.5 s | pytabkit on Torch: 205 s |

On five public OpenML tasks the networks are within 2 % of their reference implementations' accuracy, and idun_gbm's
error is about 3 % above the CPU libraries' median at the same settings (its gradients are always quantized). The [Performance guide](docs/performance.md) has what to do with Metal
and CUDA to get there; [Determinism and testing](docs/testing.md) has what is guaranteed, how it is tested, and what is
not tested yet.

## Documentation

- [Getting started](docs/getting-started.md)
- The learners, with every parameter: [idun_gbm](docs/learners/gbm.md), [idun_tabm](docs/learners/tabm.md),
  [idun_realmlp](docs/learners/realmlp.md), [JARF](docs/learners/jarf.md)
- [Data](docs/data.md): tables in memory and packed datasets larger than memory
- [Performance guide](docs/performance.md) and [Benchmarks](docs/benchmarks.md)
- [Determinism and testing](docs/testing.md)
- [Python API](docs/api/python-api.md) and the [architecture notes](docs/architecture/overview.md)
- [Changelog](CHANGELOG.md)

## Numerai

`idun-numerai` builds on idun's public interface: it downloads Numerai's data (`numerapi`), packs it into idun's
memory-mapped format, provides walk-forward refits for the dates you choose and the labels, and scores predictions per
era with Numerai's scoring functions in NumPy (checked against `numerai-tools`). `idun_numerai.upload` turns trained
models into the pickled predict function Numerai's model uploads run, with NumPy alone.
[`examples/numerai.py`](examples/numerai.py) walks through it.

## Contributing

Contributions are welcome, measurements on hardware we have not tested most of all: see
[CONTRIBUTING.md](CONTRIBUTING.md) for the tests, the device suites and the benchmarking rules.

## Licence

Apache-2.0 ([LICENSE](LICENSE)). [NOTICE](NOTICE) lists the work idun builds on, including MLX's Steel GEMM, whose
structure the networks' GEMM kernel adapts.
