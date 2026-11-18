# Architecture overview

## Packages and layers

```
packages/idun/src/idun
├── gbm/                     idun_gbm                                                         docs/api/python-api.md
│   ├── __init__.py          public API: Params, train, train_seeds, Booster, FitReport, Objective, BinnedData
│   ├── params.py, sampling.py, split.py, tree.py, train.py, booster.py, features.py, objective.py, multi.py
│   ├── backends/
│   │   ├── base.py          the Backend protocol                                             backend-interface.md
│   │   ├── cpu.py           numba reference backend
│   │   ├── device.py        the GPU passes, once, over idun.device; the kernel table
│   │   ├── metal.py         Metal backend: the passes on a Metal device
│   │   └── cuda.py          CUDA backend: the same passes on a CUDA device
│   ├── kernels/metal/*.metal   MSL kernels, compiled at runtime                              metal-kernels.md
│   └── kernels/cuda/*.cu    CUDA kernels, compiled at runtime by NVRTC
├── tabm/                    idun_tabm: params, reference (float64), metal (MetalTabM), train networks.md
├── realmlp/                 idun_realmlp: params, reference (float64), metal, train          networks.md
├── nn/                      what the networks share: inputs, metal (GEMM tile, structures), training
│   └── kernels/metal/       common.metal (GEMM, AdamW), tabm.metal, rmlp.metal
├── data/                    packed cache builder and memory-mapped reader                    data-layout.md
├── device.py                the Device protocol of the GPU backends, kernel signatures       backend-interface.md
├── metal/
│   ├── runtime.py           ctypes binding to libidun_metal (C ABI)                          backend-interface.md
│   ├── device.py            the Metal device: the GPU's memory is the host's
│   └── mirrored.py          a test device: the host gets copies only
├── cuda/
│   ├── runtime.py           ctypes binding to the CUDA driver library and NVRTC              backend-interface.md
│   └── device.py            the CUDA device: GPU memory and page-locked host mirrors
├── rng.py                   counter-based draws shared by every learner and backend          determinism.md
├── _jit.py                  the only numba entry point (fastmath off)                        determinism.md
└── errors.py                IdunError, BackendUnavailableError, KernelCompileError
packages/idun/native/
├── include/idun_metal.h     C ABI                                                            backend-interface.md
└── metal/idun_metal.mm      Objective-C++ implementation
packages/idun-numerai/       idun_numerai, the Numerai port: download, cache build, labels, refits, scores, uploads
```

Import rules, checked by `lint-imports`:

- `idun.metal`, `idun.cuda` and `idun.device` import no learner (`idun.gbm`, `idun.tabm`, `idun.realmlp`) and not
  `idun.nn`;
- the three learners are independent of each other; `idun.nn` imports no learner, and `idun.gbm` does not use it;
- `idun.data` imports no learner and no device (`idun.metal`, `idun.cuda`, `idun.device`);
- `errors` and `_jit` import nothing from idun;
- `idun` never imports the Numerai port (`idun_numerai`), and ruff bans pandas, lightgbm, sklearn, mlx and torch
  inside it.

## Who does what in one boosting iteration

Python orchestrates; backends own device buffers and run a small kernel set; split finding is host-side and shared.

```
host (Python)                                   backend (CPU: numba │ Metal: kernels)
─────────────────────────────────────────────   ──────────────────────────────────────────────
seed, iteration → bag mask, feature sample
iteration_key(seed, iteration)             ──►  quantize_gradients: g = pred − y, q = SR(g/scale)  → int8 q
for level d in 0..max_depth−1:
    choose the smaller child of each split ──►  build_histograms(level, sampled features)         → int32 (Σq, n)
    sibling = parent − child
    find_splits(histograms)  [host, float64]
                                           ──►  partition_rows(splits)                             → node id per row
leaf sums                                  ──►  leaf_partials (int64 fixed point Σg, n per leaf)
leaf = Σ(y − pred)/(n + λ) · lr  [host]
                                           ──►  update_predictions: pred += leaf[node]  (+ max|g|)
```

Leaf-wise growth (`growth="leafwise"`) replaces the level loop with best-first steps: split the leaf of
highest gain (`split_leaf`), build its smaller child's histogram, subtract for the sibling, search both children; until
`max_leaves` leaves.

On Metal, everything between two host steps is encoded into **one command buffer** and committed once, so each tree
needs about `max_depth + 2` host↔GPU round trips (leaf-wise: about `max_leaves`). Buffers are shared (unified memory): the host reads histograms and
writes splits in place, with no copies.

## Data at rest and in flight (a training window of 5.95 M rows of Numerai's data)

| what | layout | size |
|---|---|---|
| obfuscated features (1,017) | column-major, 4-bit, two rows per byte, 16 KiB-aligned columns | ≈ 3.5 GB on disk, memory-mapped, wrapped by Metal without a copy |
| named features (22) | column-major uint8 bins, one set per refit binning | ≈ 150 MB per refit |
| country | uint8 codes | 7 MB |
| per-row state | `y` f32, `pred` f32, `g` f32, `q` i8, node u8, bag u8 | ≈ 14 B/row ≈ 85 MB |
| histograms | (nodes ≤ 64) × (sampled features) × bins × 8 B | ≤ 14 MB at 100 features |

A tree reads only its sampled columns (`feature_fraction` of about 3 GB a level at 4 bits). That makes the job memory-bandwidth
bound, and every layout choice above aims to move fewer bytes.

## Where to read next

- [apple-silicon.md](apple-silicon.md): which parts of an Apple M4 we use and why.
- [backend-interface.md](backend-interface.md): the Python protocol and the C ABI.
- [determinism.md](determinism.md): the bit-exactness contract.
- [data-layout.md](data-layout.md): the cache format, histogram and tree layouts.
- [metal-kernels.md](metal-kernels.md): kernel designs and the optimization ladder.
- [networks.md](networks.md): idun_tabm and idun_realmlp on Metal.
- [provenance.md](provenance.md): where the code comes from.
