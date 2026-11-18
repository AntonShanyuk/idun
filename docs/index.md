# idun

Gradient boosting, TabM and RealMLP on the GPU: Metal on Apple silicon, CUDA on NVIDIA, with a CPU backend for each.
Every fit is deterministic, and idun_gbm builds the same model, bit for bit, on the CPU, on Metal and on CUDA.

- [idun_gbm](learners/gbm.md): histogram gradient boosting with LightGBM's model and parameter names.
- [idun_tabm](learners/tabm.md): TabM (Gorishniy et al., ICLR 2025), a parameter-efficient ensemble of MLPs.
- [idun_realmlp](learners/realmlp.md): RealMLP-TD (Holzmüller et al., NeurIPS 2024), as pytabkit builds it.
- [JARF](learners/jarf.md): idun_gbm behind the directions a trained idun_tabm is most sensitive to.

```bash
pip install idun                      # Apple silicon with macOS 15 or later: the Metal wheel; elsewhere the pure wheel
pip install idun nvidia-cuda-nvrtc    # Linux with an NVIDIA GPU
```

```python
from idun.gbm import GBMRegressor

model = GBMRegressor(num_trees=300, learning_rate=0.05, backend="metal").fit(X_train, y_train)
pred = model.predict(X_test)
```

On 5.95 million rows, 600 trees take 33.9 s on one Tesla T4 and 48.0 s on an Apple M4, against LightGBM's 769 s and
187 s on those machines' CPUs ([Benchmarks](benchmarks.md)).

Start with [Getting started](getting-started.md). The [Performance guide](performance.md) shows how to get the most
out of Metal and CUDA, [Determinism and testing](testing.md) what is guaranteed and what is not tested yet, and the
[architecture notes](architecture/overview.md) how the learners run on the GPU.
