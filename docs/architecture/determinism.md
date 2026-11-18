# Determinism contract

Every backend produces the same bits as the CPU backend: histograms, splits, leaf values and predictions. The parity
tests compare with `==`. This page lists every rule that makes that true for idun_gbm; the networks' contract is in
[networks.md](networks.md) §3.

## 1. Randomness

| draw | where | definition |
|---|---|---|
| bagging mask (every `bagging_freq` iterations) | host | `Generator(PCG64(SeedSequence([seed, iteration, 0]))).random(n, float32) < bagging_fraction`: one Bernoulli draw per training row, as LightGBM's `bagging.hpp` (`NextFloat() < bagging_fraction`), so the bag size varies around `fraction × n` |
| feature sample (per tree) | host | `Generator(PCG64(SeedSequence([seed, tree, 1]))).choice(n_features, k, replace=False)`, sorted, with `k = max(floor(fraction × n_features + 0.5), 1)` as LightGBM's `col_sampler.hpp` (`RoundInt`): 50 of 100 at fraction 0.5 |
| feature sample (per node, `feature_fraction_bynode` < 1) | host | `Generator(PCG64(SeedSequence([seed, tree, 2, a, b]))).choice(k, m, replace=False)` over the tree's k sampled features, `m = max(floor(fraction × k + 0.5), 1)`; node `(a, b)` is `(heap index, 0)` depth-wise and `(the tree's leaf count when the leaf is searched, leaf id)` leaf-wise |
| one threshold per numeric feature (per node, `extra_trees`) | host | `u = idun.rng.uniform_bits(derive_key(seed, 11, tree, a, b), k)[i]` for the i-th sampled feature, threshold `(u · (real − 1)) >> 24` over its real bins (none below two); both missing directions at it; categorical features keep their search |
| stochastic rounding (per row) | device | `idun.rng`: `key = lowbias32(lowbias32(seed ^ 0x9E3779B9) ^ iteration)`; `bits = lowbias32(key ^ row) >> 8`; `u = bits · 2^-24` |

- `tree` is the tree's index in the model (a continued fit's trees draw as a fit from scratch would).
- `row` is the row's absolute index in the packed matrix, never its position in a bag or node. Draws therefore do not
  depend on partitioning or thread layout.
- `lowbias32` is defined on uint32 with wrap-around multiply. numba emulates it in uint64 with a mask; MSL and CUDA use
  native `uint`. Known answers from an independent C implementation are pinned in `tests/unit/test_rng.py`.

## 2. Gradient quantization (mirrors LightGBM 4.7.0 `gradient_discretizer.cpp`)

LightGBM (`DiscretizeGradients`, constant hessian, stochastic rounding):
`scale = max|g| / (bins / 2)` over **all** training rows (in-bag and out-of-bag: it runs on `num_data_`, and bagging
does not use a subset), `inverse = 1 / scale` in double, then
`q = int8(g ≥ 0 ? g·inverse + u : g·inverse − u)` (C++ cast: truncation toward zero), `u ~ U[0, 1)`; the hessian is
discretized to 1 and its scale is `max|h| = 1`. Leaves are renewed from the float gradients (`RenewIntGradTreeOutput`).

Idun, identical on every backend:

- `g = pred − y` (float32, one subtraction per row), for every training row.
- `m = max|g|` over the training rows; the host computes `scale = m / (bins / 2)` in float64 and passes
  `inv_scale = float32(1 / scale)` down. Devices multiply and never divide.
- `x = g · inv_scale` (float32), then `q = trunc(x + u)` if `g ≥ 0`, else `trunc(x − u)`, with `u = bits · 2^-24` from §1
  (float32, exact), then **clamped to `[−bins/2, bins/2]`**. The clamp only bites when `|x|` is within a rounding of
  `bins/2` and `u > 1 − 2^-23`, where the float32 sum rounds up to `bins/2 + 1`; LightGBM's double arithmetic has the
  same corner at `u > 1 − 2^-53`.
- Out-of-bag rows get `q = 0` and are not counted in histograms.
- Split gains use `G = Σq · scale` and `H = count` (the discretized hessian is 1 per row). If `m = 0` every `q` is 0 and
  no split has positive gain.
- A host objective (`idun.gbm.Objective`, `NewtonObjective`) writes `g` into the same buffer instead; everything
  below reads that buffer, so it must write the same floats for the same inputs.
- Row weights are applied on the host as such an objective: `g · w` and `h · w` in float32 (`h = 1` for the
  built-in L2, `g = pred − y` there as the device computes it): Newton leaves, and second-order splits with
  `Params.second_order`.
- Hessians: the split search counts rows for Σh (first-order splits) under every objective, unless
  `Params.second_order`: then a `NewtonObjective`'s hessians steer the splits too (second-order splits). They
  enter the leaf values either way (§5). They are quantized as the gradients are, with the same draws `u` and the same half range:
  `scale_h = max h / (bins / 2)` over the training rows in float64, `inv_h = float32(1 / scale_h)` (0 when every
  `h` is 0), `q_h = trunc(h · inv_h + u)` (`h ≥ 0`) clamped to `[0, bins/2]`, and 0 out of the bag, by the gradients'
  kernel on the hessians' buffers. LightGBM rounds `h` to `[0, bins]` (`hessian_scale = max h / bins`); the half range
  keeps `q_h` within the int8 and the histogram kernels' bounds of `q`, at half LightGBM's resolution, unbiased.

## 3. Integer histograms

- A cell holds `int32 sum_q` and `int32 count`. For second-order splits the host adds the hessians' `int32 sum_q_h`
  from a second pass of the same kernels over `q_h` (the same slots, so the same cells), as a third channel; the
  sibling subtraction below covers it as it covers the others.
- Bounds: `|q| ≤ 2` with 4 quantization bins and `count ≤ n_rows`. With n ≤ 2^23 rows in a fit,
  `|Σq| ≤ 2^24`, far inside int32. The builder asserts `n_rows < 2^30`.
- Threadgroup and device atomics are relaxed integer adds: order-independent, therefore exact.
- Sibling histograms are `parent − child`, also exact in integers.

## 4. Split finding (host, shared; mirrors LightGBM's integer-histogram split search)

- One implementation for all backends, in float64, from the integer sums: `G = Σq · scale`, and `H = count`
  (first-order splits) or `H = Σq_h · scale_h` (second-order), each side's `H` from its integer sum. On the
  first-order path the search reads the count as the hessian channel at scale 1.0, and `float(count) · 1.0` is
  `float(count)`: the same floats as first-order splits.
- A split is valid only if both children hold ≥ `min_data_in_leaf` in-bag rows, `H ≥ min_sum_hessian_in_leaf` and
  `H + λ_c > 0` (always true for counts; for hessians that quantize to 0 it keeps `G²/0` out), and its gain is > 0, where
  `gain = G_L²/(H_L + λ_c) + G_R²/(H_R + λ_c) − G²/(H + λ)`; `λ_c = λ` for numeric splits and `λ + cat_l2` for sorted
  categorical ones (LightGBM adds `cat_l2` to the children's `l2` only).
- **Numeric** features: thresholds `t = 0 .. n_bins − 2`, bins `≤ t` go left. Named features try the missing bin (255) on
  each side, as LightGBM's two scans for `missing_type = NaN`; obfuscated features have no missing bin.
- **Categorical** (`feature_country`), `categorical="sorted"`, a sorted partition (LightGBM
  `FindBestThresholdCategoricalIntInner`): categories with `count ≥ max(cat_smooth, 1)` (10 by default) are
  stable-sorted by `G_c / (H_c + cat_smooth)` (`H_c` the category's count, or its hessians' sum; with hessians, a
  category whose `H_c + cat_smooth` is 0 is left out too, as its key would be `G_c / 0`) (LightGBM's filter is `count ≥ cat_smooth`; the `1` keeps empty
  categories, whose key is 0/0 when `cat_smooth = 0`, out); two scans, from the low end and from the high end, add one
  category at a time to the left set, for at most `min(max_cat_threshold, (used + 1) // 2)` categories (32 cap); a
  prefix is skipped while the left side has fewer than `min_data_in_leaf` rows, the scan stops once the right side has
  fewer than `min_data_in_leaf` or `min_data_per_group` (100) rows, and a prefix is evaluated only after it gathered ≥
  `min_data_per_group` rows since the last evaluated one. Categories outside the left set, including filtered and unseen
  ones, go right. The one-vs-rest alternative tries each category alone on the left, with `λ_c = λ`.
- Ties are broken by lowest feature index, then lowest threshold bin, then missing-left before missing-right; for a
  categorical feature, the first best prefix in scan order (low end first).
- The search is compiled (`idun._jit.njit`, so without contraction) and evaluates every gain with the operations above
  in that order; tests hold it to the same search written in numpy, bit for bit, ties included.
- Leaf-wise growth splits the leaf whose best split has the highest gain; ties go to the lowest leaf id.
  Leaf ids follow the order of the splits (a leaf's left child keeps its id, the right child takes the next one), so
  the same histograms give the same tree on every backend.

## 5. Leaf values (int64 fixed point)

- Per in-bag row: `g_fx = int64(rint(g · 2^32))`. `g · 2^32` is exact in float32 (a power-of-two scale) and `rint` is
  exact, so the conversion is identical on every backend.
- Per leaf: `S = Σ g_fx`, `n = Σ 1`, as int64. Bound: `|g| < 2^7` (asserted per iteration) and `n < 2^23` keep
  `|S| < 2^62`.
- Host: `leaf = float32(−(S · 2^-32) / (n + λ) · learning_rate)` in float64, rounded once to float32; a leaf no in-bag
  row reaches gets 0. This is LightGBM's renewal (`−Σg / (Σh + λ)`, then shrinkage) with an exact sum.
- `g` is the stored gradient of §2 (the built-in `pred − y`, or what a host objective wrote).
- A `NewtonObjective`: `h_fx = int64(rint(h · 2^32))` per in-bag row and `T = Σ h_fx` per leaf, by the same kernel.
  Hessians are finite, `h ≥ 0` and `max h · rows < 2^31` (checked per iteration), so `T < 2^63`. Host:
  `leaf = float32(−(S · 2^-32) / (T · 2^-32 + λ) · learning_rate)`. A leaf with rows and `T · 2^-32 + λ = 0` is an
  error, not a zero.

## 6. Prediction update and inference

- `pred[row] = pred[row] + leaf[leaf_of(row)]`: one float32 addition per row per tree, in tree order. No reduction is
  involved, so the order is fixed.
- Inference sums trees in index order, one float32 addition each, starting from 0.0 (`boost_from_average=False`).

## 7. Compilation rules

| backend | rule | enforced by |
|---|---|---|
| Metal | `#pragma METAL fp contract(off)` prepended to every library; `mathMode = Safe`; precise math functions | `idun_compile_check` / runtime (native); the `fma_probe` test kernel |
| numba | `fastmath=False` | `idun._jit.njit`, ruff `banned-api`, `tests/unit/test_jit.py` |
| CUDA | NVRTC `--fmad=false`; no `--use_fast_math`; idun_gbm's kernels have no explicit `fmaf` (the networks' GEMM has: their contract is [networks.md](networks.md) §3) | `idun.cuda.runtime.compile_ptx` and `compile_cubin` (fixed options, a test); the `fma_probe` test kernel on the GPU |

Measured: Metal with default options **and** with `MTLMathModeSafe` fused `a*b+c` (244,609 of 2^20 results
differed from separate multiply and add); with the pragma, 0 differed. numba default 0, numba fastmath 244,609.
CUDA on two Tesla T4 with `--fmad=false`: 0 of 2^20 differed on each (a fused product would differ on about
245,000).

## 8. What breaks determinism (do not do this)

- float atomics, or any float reduction on a device;
- a stateful RNG on a device, or draws indexed by thread id instead of row;
- division, `exp`, `log` or `sqrt` inside kernels that feed parity-checked outputs;
- `fastmath=True` in numba, or `-ffast-math` / `--use_fast_math` anywhere;
- host-side sampling from global NumPy state (`np.random.*` functions) instead of a seeded `Generator`.

## 9. Computed once, stored: JARF's directions

`idun.tabm.directions` sits outside the bit-exact contract above in two places, both measured and both bounded by
storing the result rather than recomputing it:

- `numpy.linalg.eigh` is LAPACK's; its vectors may differ in their last bits between Accelerate builds (M1 against
  M4). The EGOP itself is a float64 sum in a fixed order and repeats its bits on one machine.
- A float32 projection (`Directions.project`, the front's `project_rows`) is a BLAS matmul whose summation order
  depends on the matrix's shape: the same row projected in a chunk of another size moved by one last bit (measured in
  `tests/unit/jarf/test_front.py`). So the front projects in fixed chunks of `PROJECT_ROWS` = 50,000 rows, in the
  fixed order of the rows asked for, and the upload runtime does the same.

A `Directions` is therefore computed once and stored with the model that uses it (`JARFModel`, `JARFRegressor`'s
file, the upload's pickle); every fit and prediction from stored directions is bit-exact as idun_gbm's are. The
difference between directions recomputed on another machine and the stored ones is measured, not assumed.
