# Provenance

Where idun's code comes from, for the Apache-2.0 release. The rule: reimplement from the papers
and from reading other implementations; no code copied without attribution.

| component | what it reproduces | how | the source's licence | attribution |
|---|---|---|---|---|
| GEMM in `idun/nn/kernels/metal/common.metal` | MLX's Steel GEMM (`mlx/backend/metal/kernels/steel/gemm/`) | **adapted**: the tile loader's thread layout (`loader.h`, `BlockLoader`: `TCOLS`, `TROWS`, `bi`, `bj` and the strided row loop) and the 8 × 8 fragment's lane coordinates (`mma.h`, `get_coord`: three lines, the variable renamed), inside our own kernel: its own tile sizes, 4-wide loads, the BatchEnsemble scaling fused into the tile load, transposed operands and the epilogue | MIT, © 2023 Apple Inc. | NOTICE carries MLX's copyright and permission notice; the kernel's comment names it |
| idun_gbm (numba CPU backend, `gbm.metal`) | LightGBM's model and split rules: histogram split search, the integer-histogram categorical search, gradient quantization, depth-wise and leaf-wise growth | reimplemented in numba and MSL from LightGBM's papers, documentation and code reading, to give its decisions; no LightGBM code (C++) is included | MIT | NOTICE cites LightGBM |
| idun_tabm (`idun.tabm.reference`, `tabm.metal`) | TabM, as the `tabm` package trains it | reimplemented in NumPy and MSL from the paper and the package's behaviour; no `tabm` code (PyTorch) is included | Apache-2.0 | NOTICE cites the paper |
| idun_realmlp (`idun.realmlp.reference`, `rmlp.metal`) | RealMLP-TD with pytabkit 1.7.3's defaults: preprocessing, schedules, the optimizer's factors | reimplemented in NumPy and MSL from the paper and from reading pytabkit's code, to give its results; no pytabkit code (PyTorch) is included | Apache-2.0 | NOTICE cites the paper |
| `idun_numerai.score` | Numerai's per-era scores (numerai-tools' `neutral_correlation`, `numerai_corr`) | written in NumPy and SciPy; numerai-tools is a test reference only, not a dependency | — | — |
| obfuscated-feature histograms in `gbm.metal` | one feature per SIMD lane, and the `code > t` bit masks | ideas from the Numerai community's EFB booster ([forum](https://forum.numer.ai/t/200-trees-per-second-with-cuda/7823)) and [PackBoost](https://github.com/Pranshu-Bahadur/PackBoost); our own MSL, no code from them (CUDA) | — | named here |
| everything else (the runtime, the other kernels, data, RNG, estimators) | — | written for idun | — | — |

## How it was checked

- The GEMM against MLX's `steel/gemm/loader.h` and `mma.h` (its main branch, read when the kernel was written), line by line: the two
  adapted parts above; the rest of the kernel has no counterpart there.
- The other MSL kernels: MLX has no gradient-boosting or BatchEnsemble kernels; `gbm.metal`, `tabm.metal` and
  `rmlp.metal` implement idun's own reference code.
- The Python and numba code: the sources it reproduces are C++ (LightGBM) and PyTorch (`tabm`, pytabkit); comments that
  name them (``as LightGBM's``, ``as pytabkit's``) mark behaviour reproduced, not code taken. A spot check of the
  closest parts (RealMLP's robust scaling and its optimizer step against pytabkit) found NumPy re-expressions of the
  math.
