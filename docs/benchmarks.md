# Benchmarks

Two kinds of measurement, both with the reference implementations' rows beside idun's:

- [Accuracy and fit time on public tables](#accuracy-and-fit-time-on-public-tables): five OpenML tasks on an M1,
  against LightGBM, XGBoost, scikit-learn, the `tabm` package and pytabkit. The script is in the repository; rerun it.
- [Speed on a large table](#speed-on-a-large-table): millions of rows on an M1, an M4 and a pair of Tesla T4s,
  speed only.

Every figure is a measurement, made with the version of the day; where a later change touched a kernel, the parity
tests and an A/B timing were rerun ([the optimisation ladders](#optimisation-ladders)). The [gaps](testing.md#the-gaps)
apply: one machine of each kind.

## Accuracy and fit time on public tables

idun's learners against the libraries they reimplement, on five public OpenML tasks. The script is
`benchmarks/openml.py`; every number below comes from it, and you can rerun it on your own Mac. idun's learners are
deterministic, so on the machine below a rerun gives their rows bit for bit, as it does the boosting libraries'
(checked on california, diamonds and adult).

### Method

| task | OpenML id | kind | training rows | test rows | features |
|---|---|---|---|---|---|
| california | 44025 | regression | 13,210 | 4,128 | 8 numeric |
| diamonds | 42225 | regression | 34,522 | 10,788 | 6 numeric, 3 categorical |
| adult | 1590 | binary | 31,259 | 9,768 | 6 numeric, 8 categorical |
| higgs | 44129 | binary | 320,000 | 100,000 | 24 numeric (a seeded sample of 500,000 rows) |
| covertype | 1596 | 7 classes | 371,848 | 116,202 | 54 numeric |

- **Data.** Each task is OpenML's parquet file, checked by sha256. Text columns become categorical, with missing
  values as their own level and levels seen in fewer than 30 rows merged into one. A seeded permutation splits the
  rows once into 64 % training, 16 % validation (unused here) and 20 % test.
- **Boosting.** idun_gbm (`GBMRegressor` at its defaults: 100 trees, learning rate 0.1, depth 6, 64 leaves, 20 rows
  per leaf, gradients quantized into 4 bins) against LightGBM, XGBoost (histogram, depth-wise) and scikit-learn's
  HistGradientBoosting at the same nominal settings, with 255 bins and no early stopping. A second LightGBM row
  quantizes its gradients into 4 bins as idun_gbm does. idun_gbm is a regressor, so only the regression tasks.
- **Networks.** idun_tabm and idun_realmlp at their estimators' defaults against their reference implementations on
  Torch's MPS device, trained the same way on the same input matrix: the `tabm` package's TabM (k = 32, 3 blocks of
  512, AdamW at 2e-3, 256-row batches, 50 epochs, the last weights) and pytabkit's RealMLP-TD (256 epochs, 256-row
  batches, no held-out rows, the last weights; the regressor with the classifier's SELU block at learning rate 0.04,
  as idun_realmlp). idun_realmlp fitted its preprocessing on 3,986 training rows, which the script passes explicitly; the
  estimators' default takes pytabkit's own count, every training row on these tables. Two seeds on
  the three smaller tasks, one on higgs and covertype.
- **Defaults rows.** pytabkit's TabM-D and RealMLP-TD at their own defaults, which hold out a fifth of the training
  rows to pick the best epoch. idun's rows train without a validation set; these rows show what picking the best epoch is worth.
- **Metrics.** RMSE on the test rows for regression; log-loss (and accuracy) for classification. Lower is better.
  Times are the fit's wall-clock seconds.

**Machine and versions.** An Apple M1 with 8 GB on AC power, Python 3.13, NumPy 2.5.4, scikit-learn 1.9.1, LightGBM
4.7.0, XGBoost 3.4.1, Torch 2.14.1, tabm 0.0.3, pytabkit 1.7.3. One run at a time on an otherwise idle machine.

### Boosting

| | california RMSE | fit s | diamonds RMSE | fit s |
|---|---|---|---|---|
| idun_gbm (Metal) | 0.14336 | 0.8 | 557.25 | 2.0 |
| LightGBM | 0.13925 | 0.5 | 542.41 | 0.5 |
| LightGBM, gradients in 4 bins | 0.14061 | 0.4 | 545.79 | 0.5 |
| XGBoost | 0.13855 | 0.2 | 536.59 | 0.2 |
| HistGradientBoosting | 0.14016 | 0.5 | 545.26 | 0.6 |
| idun_gbm / median of the three libraries | 1.029 | | 1.027 | |

idun_gbm's RMSE is about 3 % above the CPU libraries' median at the same nominal settings. A quarter to a third of
that gap comes with 4-bin gradients, as LightGBM's own quantized row shows. On tables this small the CPU libraries fit
faster; idun_gbm is built for panels of millions of rows.

### Networks

| task | metric | idun_tabm | tabm (Torch) | ratio | fit s, idun / Torch |
|---|---|---|---|---|---|
| california | RMSE | 0.14196 | 0.14172 | 1.002 | 72.5 / 161.7 |
| diamonds | RMSE | 558.04 | 565.39 | 0.987 | 187.1 / 414.4 |
| adult | log-loss | 0.32229 | 0.32745 | 0.984 | 176.2 / 389.2 |
| higgs | log-loss | 0.49109 | 0.49250 | 0.997 | 1,767.6 / 3,868.4 |
| covertype | log-loss | 0.18958 | 0.18998 | 0.998 | 2,077.6 / 4,553.6 |

| task | metric | idun_realmlp | RealMLP-TD (pytabkit) | ratio | fit s, idun / Torch |
|---|---|---|---|---|---|
| california | RMSE | 0.15728 | 0.15450 | 1.018 | 83.8 / 45.7 |
| diamonds | RMSE | 561.38 | 567.57 | 0.989 | 224.7 / 122.8 |
| adult | log-loss | 0.44312 | 0.45325 | 0.978 | 225.9 / 118.9 |
| higgs | log-loss | 0.66973 | 0.67588 | 0.991 | 2,091.4 / 1,182.0 |
| covertype | log-loss | 0.15131 | 0.15160 | 0.998 | 2,564.8 / 1,396.4 |

Values are means over the seeds. Each network is within 2 % of its reference on every task, and better on most.
idun_tabm fits in 0.45 of the reference's time; idun_realmlp takes about 1.8 times pytabkit's.

With the last epoch's weights, RealMLP's log-loss on adult and higgs is well above TabM's, on both sides. The defaults
rows pick the best epoch on held-out rows instead: on adult, pytabkit's RealMLP-TD defaults reach 0.318.

| task | metric | RealMLP-TD defaults | fit s | TabM-D defaults | fit s |
|---|---|---|---|---|---|
| california | RMSE | 0.13858 | 49.9 | 0.13773 | 153.6 |
| diamonds | RMSE | 540.07 | 120.7 | 537.31 | 571.1 |
| adult | log-loss | 0.31828 | 115.0 | 0.30220 | 158.4 |

### Running it

```bash
uv run --frozen --group benchmarks python -m benchmarks.openml fetch          # five parquet files, checked by sha256
uv run --frozen --group benchmarks python -m benchmarks.openml --smoke all    # every learner on a few thousand rows
uv run --frozen --group benchmarks python -m benchmarks.openml all            # every run: about 7 hours on an M1
uv run --frozen --group benchmarks python -m benchmarks.openml report         # a Markdown table of your records
```

`all --task california --learner idun_gbm` restricts a run to some tasks and learners (both repeatable). Each fit runs
in its own process and writes `benchmarks/records/<task>/<learner>_s<seed>.json` with its metrics, seconds, machine and
package versions; a run whose record exists is skipped. Close other work before timing: the script does not wait for
a quiet machine.

## Speed on a large table

The large-table speeds are measured on Numerai's v6 data: a training window of 5,948,565 rows and 894,383 predicted
rows. The trees read its 1,040 features (1,017 integer columns of 0 to 4, stored as 4-bit codes, 22 numeric columns
and one categorical column), the networks its 1,086 dense inputs. Only speed is reported.

| machine | what |
|---|---|
| M4 | Apple M4 MacBook Air: fanless, 10-core GPU, 24 GB |
| M1 | Apple M1 MacBook Air: 7-core GPU, 8 GB |
| T4 | a Kaggle session: two NVIDIA Tesla T4 (Turing, compute capability 7.5), 4 vCPUs (Xeon, 2.0 GHz), 33.7 GB |

Whole fits (the backend made, the trees grown, the backend closed) and whole epochs, the median of three, on a quiet
machine with five-minute cool-downs on the M4; the reference rows ran in the same session unless noted.

### idun_gbm: 600 trees

The same configuration throughout: 600 depth-wise trees of depth 6 (64 leaves) with 4-bin gradients, row bagging and
a feature sample per tree. The LightGBM rows (4.7.0) grow its default leaf-wise trees of 63 leaves with quantized
gradients and the same sampling.

| machine | idun_gbm | LightGBM |
|---|---|---|
| T4, CUDA | **33.9 s** on one T4 | 769.1 s on the session's 4 CPUs (22.7×); 342.9 s with its OpenCL build on one T4 |
| M4, Metal | **48.0 s** | 186.5 s on the M4's 10 CPU cores (3.9×; measured on the machine, not in the same session) |
| M1, Metal | about 174 s | — |

- Prediction of the 894,383 rows: 1.17 s on one T4, against LightGBM's 21.8 s on the session's CPUs and 19.9 s with
  OpenCL.
- idun's own CPU backend, the reference the GPU backends are checked against bit for bit, took 529.5 s for 50 of
  those trees on the T4 session's 4 vCPUs (one T4: 4.1 s).
- Leaf-wise growth (63 leaves, no depth limit) took 1.96 times depth-wise's time on the M4 (105.1 against 53.7 s).

Seed bags (`train_seeds`), every model bit for bit its separate fit:

| machine | bag | one after another | side by side |
|---|---|---|---|
| T4 pair | 4 seeds of 200 trees | 53.7 s | 30.2 s, one fit per T4 (the default); two per T4 gained nothing (30.1 s) |
| M4 | a bag of 600-tree fits, on 2.6 million rows | 205.0 s | 86.9 s with three fits in flight (2.36×) |

### idun_tabm: one epoch

The paper's network (32 members, 3 blocks of 512, dropout 0.1, AdamW) in float32, 4,096-row batches: 1,453 steps an
epoch.

| machine | idun_tabm | the `tabm` package 0.0.3 on Torch 2.14.1 |
|---|---|---|
| T4, CUDA | **411.6 s** | 527.7 s in float32; 216.3 s with fp16 autocast and `torch.compile` |
| M4, Metal | **553.0 s** (495.5 s on a cool machine) | 1,386.2 s on Torch's MPS device (2.51×) |
| M1, Metal | about 890 s with `micro_batch=1024` (from a whole fit's time) | — |

idun_tabm has no half-precision path: on a T4's tensor cores Torch's mixed precision is faster than idun's float32.

### idun_realmlp: one epoch

RealMLP-TD's network (3 blocks of 256), 256-row batches: 23,236 steps an epoch.

| machine | idun_realmlp | pytabkit 1.7.3 on Torch 2.14.1 |
|---|---|---|
| T4, CUDA | **76.5 s** | 205.4 s (2.7×) |
| M1, Metal | about 87 s (from a whole fit's time) | — |

On the M4 a step takes about 1.9 ms in bursts (indicative, not a whole epoch). Four seeds side by side on the M4
(`train_realmlp_seeds`, a 2.6-million-row window): 394.9 s with two networks in flight against 457.8 s one after
another (1.16×): its 256-row steps already keep the GPU busy.

## Optimisation ladders

Each step was kept only if the parity tests still passed (bit for bit for idun_gbm, within tolerance of the reference
for the networks) and the timing improved.

| learner, backend | from | step | to |
|---|---|---|---|
| idun_gbm, Metal (M4, ms a tree) | 223 | one thread per row with device atomics, to per-dispatch threadgroup memory, packed counters, bitmask paths for the shallow levels and host work overlapped ([Metal kernels](architecture/metal-kernels.md#4a-the-ladders-log-engineering-measurements)) | 89 |
| idun_gbm, CUDA (T4, 600 trees) | 37.4 s | more blocks for the T4's 40 SMs, the host's split search compiled, the numeric features counted in shared memory ([§4b](architecture/metal-kernels.md#4b-the-same-kernels-on-cuda)) | 32.7 s |
| idun_tabm, Metal (M4, a full 4,096-row step) | 1.265 s (Torch MPS 1.540 s) | element-wise passes fused into the GEMM epilogues (0.875 s), fixed-order batch sums (0.868 s), per-member weight-gradient products | 0.334 s (Torch MPS 0.856 s) |
| idun_tabm, CUDA (T4, an epoch from the step time) | 623.1 s with 64 × 64 GEMM tiles | bf16 storage tried and dropped (706.1 s, slower); 128 × 128 tiles with fused multiply-adds | 401.7 s |
| idun_realmlp, Metal (M1, a 256-row step) | 10.15 ms | the periodic embedding compiled with constants (9.40 ms), batches gathered and preprocessed on the GPU (9.24), pipelined steps (5.60), codes laid out row-major once a fit | 3.79 ms |
| idun_realmlp, CUDA (T4) | | the first port already took 0.41 of pytabkit's time an epoch; no step was needed | |

Tried and not kept: bf16 storage on CUDA (above); batched kernels for RealMLP seed bags (1.07× at four seeds on the
M4, not worth their complexity); one AdamW dispatch for all of RealMLP's arrays (AdamW is already bandwidth-bound).
On Metal, bf16 storage is kept as an option: up to 1.08× faster TabM steps on the M4.
