# idun_tabm and idun_realmlp: design notes

Why the networks are built as they are on Metal, and what they reproduce: TabM (§6) and RealMLP-TD (§8). No MLX or
Torch runs inside `idun`; both serve only as outside benchmarks.

## 1. Where the win is, and against which baseline

- **Against Torch MPS**, a native Metal implementation should win several-fold: Torch MPS runs mid-size MLPs as many
  small unfused fp32 kernels with per-op Python overhead.
- **Against a GPU with fp16 tensor cores**, such as an NVIDIA T4 (about 65 TFLOPS nominal against an Apple M4 GPU's
  about 4 TFLOPS fp32), raw dense FLOPs cannot win. The design has to win by doing less dense work and by removing the
  data movement and overhead around it. That is where the sections below aim.

## 2. The input layer

idun_tabm does not embed the codes. It feeds the packed dataset's dense inputs (`idun.nn.inputs.dense_inputs`): the
numeric columns, the 4-bit codes mapped linearly to `(x − 2) / 2`, and the categorical column one-hot, through a
BatchEnsemble linear layer `((x ⊙ r_k) @ Wᵀ) ⊙ s_k + b_k` with per-member random-sign r_k. That is a dense GEMM over
B·k rows (§6); the packed codes save input bandwidth (0.5 B per feature in the gather), and unified memory makes the
packed panel GPU-visible through the same zero-copy mapping idun_gbm uses: no data loader, no host copies.

## 3. The other layers

- One fused kernel per layer: matmul, bias, activation and dropout (dropout masks from `idun.rng`'s counter-based
  draws, so they are reproducible); one command buffer per training step, no per-op Python.
- Activations and their gradients can be stored in bfloat16 (`TabMParams.storage="bf16"`); weights, AdamW and all
  arithmetic stay fp32.
- **Determinism.** Bit identity with a CPU oracle, as idun_gbm has, would need the CPU to emulate the GPU's exact
  operation order in float training math, for no gain in accuracy; bit identity run to run is cheap because we own the
  accumulation order. So:
  - the same fit on the same machine, with the same inputs and seed, gives identical bits: parameters, predictions,
    losses (tested with `==`);
  - no float atomics: every reduction (the batch sums of weight and bias gradients, softmax sums, loss means) runs in
    an order fixed by the kernel's code and launch configuration, never by scheduling;
  - initialisation, the epoch permutation (on the host, `PCG64(SeedSequence([seed, epoch, 2]))`) and the dropout masks
    (on the device, `lowbias32` keyed by seed, step, layer, absolute row, member and unit) never depend on thread ids
    or timing; the masks are regenerated in the backward pass, not stored;
  - a float64 NumPy reference (`idun.tabm.reference`, `idun.realmlp.reference`) is the correctness oracle, and the
    Metal kernels match it within a stated float32 tolerance. Unlike idun_gbm's parity-checked paths, the network
    kernels may divide and use transcendentals (exp, log, cos), and on CUDA the GEMM accumulates with fused
    multiply-adds (`fmaf`: one rounding a step, the same order every run);
  - not promised: identical bits across Apple GPU generations (a few identical training steps give different weights
    on an M4 and an M1, each identical run to run), or with any Torch run.

## 4. Why TabM fits this hardware

TabM's own trick suits a bandwidth-bound GPU: k = 32 members in one pass with shared weights and rank-1 adapters
(BatchEnsemble), so each weight tile is read once for all members, which raises the arithmetic intensity.

## 5. What is ported

- **TabM:** BatchEnsemble k = 32, a 3 × 512 MLP with ReLU and dropout 0.1, cross-entropy or a regression loss, AdamW
  2e-3 / 3e-4 (§6).
- **RealMLP's long tail:** NTK parametrization with per-layer learning-rate factors, the diagonal input-scaling layer,
  parametric activations, the flat-cos schedule, label smoothing, seed bagging. Each is small and cheap (§8).

## 6. TabM as idun trains it

TabM (Gorishniy et al.) with the `tabm` package's (0.0.3) `TabM.make` defaults, k = 32 members, 3 blocks of 512 and
dropout 0.1 (the `TabMParams` defaults). On Numerai's cache the inputs are the 1,086 dense columns of §2.

- **Blocks.** The k members share each block's weight matrix and differ by rank-1 adapters (`r` on the input, `s` on
  the output) and their own biases; block 0's `r` starts as random signs. One block is `u = h ⊙ r`, `t = u · Wᵀ`,
  `z = t ⊙ s + bias`, `a = relu(z) ⊙ m` (m the dropout mask, scaled by 1 / (1 − p)); block 0's `h` is the row's
  inputs, shared by every member.
- **Head:** one linear layer per member onto the `n_classes` classes.
- **Loss:** the mean cross-entropy over all B × k member outputs against the row's class. **Prediction:** the
  member-averaged softmax (its expected class for a packed dataset's ordered classes).
- **Optimizer:** AdamW at 2e-3 with weight decay 3e-4 on every parameter (one parameter group, as
  `torch.optim.AdamW(model.parameters())`).
- **Loop:** each epoch a fresh permutation of the training rows, batches of `batch_size` (256 by default) rows with
  the ragged last batch kept, one AdamW step per batch, the final weights after `epochs` (50 by default) epochs.
- **Regression** (`objective="mse"`, `n_classes=1`): one output per member, the loss the mean of `(o − y)²` over the
  B × k outputs, the prediction the members' mean.
- **HL-Gauss** (`objective="hl_gauss"`, one target; Imani & White 2018): regression as classification over `n_bins`
  uniform bins of `[hl_low, hl_high]` (64 on [−4, 4] by default, for a standardized target). A target, clamped to the range, is
  labelled with the mass of N(target, `hl_sigma`) in each bin, renormalised; every member trains on soft-label
  cross-entropy and the prediction is the expected bin centre under the members' mean softmax. The labels are built
  on the host in float64 (`idun.tabm.hl_gauss`, Φ through `erfc`), so both backends train on the same ones and the
  Metal kernels need no `erf`; the Metal head computes its `n_bins` outputs as groups of 8, and two kernels add the
  soft-label loss (`hl_gauss_ce`) and the expected centre (`predict_centre`).
- **Temporal encoder** (`encoder="gru"`): each row also has a sequence of S steps, oldest first, each step C channels
  and a present flag (`Sequences`: the steps index a channel table, a row per panel row or a shared table such as
  one row per date; on a panel, `idun.data.lag_index` finds a row's own earlier rows). Per step a shared
  Linear to E units and an exact GELU, then a GRU of H units (Torch's gate order r, z, n; `h' = (1 − z) ⊙ n + z ⊙ h`)
  from a zero state; the last state is prepended to block 0's inputs, which every member shares, so the encoder runs
  once per row and its output gradient is the members' sum of block 0's input gradient over those H columns. On Metal,
  `seq_forward` runs one thread per (hidden unit, row) through all the steps, with the row's state in threadgroup
  memory, and writes the state into the batch inputs; backward, one GEMM gives `dT_0 · W_0[:, :H]`, a kernel sums it
  over the members, `seq_backward` runs back through time with the same layout, and the weight gradients are transposed
  GEMMs over 32 chunks of (row, step) pairs added in chunk order. Metal has no `erf`: its GELU takes Φ from
  Abramowitz & Stegun 7.1.26 (absolute error below 1.5e-7), the float64 reference from `erfc`. The GRU's arithmetic
  is small next to TabM's (about 1.1 GMAC forward per 4,096-row batch at 14 steps against about 145 in the blocks).
- **Piecewise-linear embeddings** (`num_emb_type="pwl"`; `idun.tabm.embeddings`): the first F inputs of a row reach
  the GPU in a buffer of their own; `emb_forward` writes their F · d embedded values into block 0's input columns
  before the other inputs, which the host writes there, one thread per (value, row) recomputing the clamped encoding
  from the fixed bins. Block 0's input gradient for those columns is the encoder's GEMM (`seq_input_grad`, dU =
  dT_0 · W_0[:, :ld_e]) and a sum over the members in order (`emb_member_sum`); each embedding weight's gradient is
  one thread's sum over the batch rows in order (`emb_weight_grad`), stored by the first micro-batch and added to by
  the later ones, so the gradients repeat their bits and follow the float64 reference within float32 error.
- **Row weights** (`train_tabm(weight=)`, `train_realmlp(weight=)`): each loss kernel reads the batch's row weights
  from a buffer the host writes with the rows (ones when unweighted, 0 on padding rows) and scales a row's gradient by
  `inv_count · w`, where `inv_count` is `1 / (Σ w · k)` for a weighted batch and `1 / (B · k)` otherwise; at
  w = 1 the product is `inv_count` itself, so the unweighted step keeps its bits. The host forms the step's loss as the
  weighted mean of the rows' losses in float64. A batch of weight 0 gets `inv_count` 0: no gradient.
- **A loss and a batch plan of the caller's** (`train_tabm(loss=, batches=)`): the loss is a host function of a
  batch's member outputs `(B, k, c)` and its rows, returning its value and its gradient with respect to the outputs;
  the plan gives each epoch's batches as row-index arrays. On the reference the gradient goes where the built-in
  logit gradient goes. On Metal a step becomes two command buffers: the forward pass, then, after the host has
  evaluated the loss on the gathered outputs, the backward pass and AdamW with the returned gradient in the buffer
  the loss kernel would have written. The activation buffers hold one micro-batch, so a micro-batched step encodes
  each micro-batch's forward pass again before its backward pass; the dropout masks are counter-based, so the second
  pass repeats the first bit for bit. With the built-in MSE's gradient computed in the kernel's float32 operations
  the step is the built-in step, bit for bit (a test). The masks and AdamW go by the count of steps taken, whatever
  the plan's batches.

## 7. Sources

The specs of §6 and §8 were read from the packages' code; the line numbers are of these versions.

- `tabm` 0.0.3, `tabm.py`: 1695–1709 (k, blocks, width, dropout), 626–657 (BatchEnsemble linear), 1659–1663 and
  1408–1412 (adapter inits), 495–499 and 1666–1672 (head).
- pytabkit 1.7.3: `pytabkit/models/sklearn/default_params.py` 7–22 and 37–51 (RealMLP-TD's defaults); §8 cites the
  files under `pytabkit/models/` by name and line.

## 8. RealMLP as idun trains it

pytabkit 1.7.3's `RealMLP_TD_Classifier` defaults, with idun's own training loop (pytabkit's files under
`pytabkit/models/` are cited by name and line). On a packed dataset the F inputs are its numeric columns (NaN → 0), its
codes as raw floats 0..4 and its categorical column one-hot; every column numeric (no categorical path).

- **Init rows.** Everything data-dependent is fitted on random rows of the training data, by default pytabkit's
  RAM-limit subsample, `floor(2^28 / (62F + 1))` rows (43,289 at F = 100; every row of a smaller table;
  `_fit_transform_subsample` in base.py, its 1 GB limit and the default network's 62F + 1 values a row; derived by
  reading, not run): the preprocessing statistics and the weight and bias initialisation.
- **Preprocessing** (model-internal, per batch): `x − median`; × `1 / (q75 − q25 + 1e-30)` with the IQR replaced by
  `(max − min) / 2` when 0 and the factor 0 when that is 0 too (pipeline.py 149–157; quantiles as `np.quantile`, linear);
  smooth clip `x / sqrt(1 + x² / 9)` (models.py 71–72).
- **PBLD embedding** per feature (nn.py 556–583, 657–677): `cos(2π · c · w1 + b1) @ W2 + b2`, 16 frequencies → 3 values;
  w1 ~ 0.1·N(0, 1), b1 ~ U(−π, π), W2 ~ U(−¼, ¼), b2 ~ U(−¼, ¼); the 3F embeddings (feature-major) then the F clipped values
  are concatenated: 4F inputs.
- **Categorical embeddings** (opt-in, pytabkit's `embedding_size` 8 for columns of at least `max_one_hot_cat_size`
  levels): the last G inputs are level codes that skip the preprocessing; each looks up `embedding_size` values in a
  learned table (N(0, 1) at the start), which join the front-scaled inputs after the PBLD values. On the GPUs
  `rmlp_cat_embed` writes them into z beside `rmlp_embed`'s, and `rmlp_cat_backward` gives each table value's gradient
  as one thread's sum over the batch's rows in order, so the gradients repeat their bits.
- **Network:** front scale (ones, 4F) → 3 × [Linear 256 (NTK: `x @ W / sqrt(fan_in)`), bias, parametric SELU
  `x + (selu(x) − x) · a` (a per neuron, init 1; `RealMLPParams.act` puts Mish or ReLU in SELU's place), dropout] → Linear C + bias (C the classes). Weights ~ N(0, 1) rescaled so each output
  column has unit population std on the init rows; biases "he+5": `b_j = −Σ_k w_k x[i_k, j]` over 5 random init rows
  with weights Exp(1) normalised to sum 1 (nn.py 166–169, 213–220).
- **Loss:** label smoothing 0.1 over the C classes (the true class 1 − 0.1 + 0.1 / C, the others 0.1 / C), mean cross-entropy (nn.py 829–856,
  metrics.py 29–42). **Prediction:** expected class `Σ c · softmax_c`.
- **Optimizer:** Adam (0.9, 0.95, eps 1e-8); lr `0.04 · coslog4(t) · f_lr`; decoupled decay before each step,
  `p ← p · (1 − wd_t · lr_t · f_wd² · f_lr²)` with wd_t = `0.02 · flat_cos(t)` and lr_t = `0.04 · coslog4(t)` (the
  factors enter twice: optimizers.py 31–36, 55–62). Factors (f_lr, f_wd): weights (1, 1), front scale (6, 1), SELU
  weights (0.1, 1), PBLD parameters (0.1, 1), biases (0.1, 0).
- **Schedules,** per step at t = k / K, K = ⌊n / 256⌋ · epochs (scheduling.py 268–273, 380–381):
  `coslog4(t) = ½ (1 − cos(2π log2(1 + 15t)))` (so the first step has lr 0), `flat_cos(t) = 1` for t < ½ else
  `½ (1 + cos(π(2t − 1)))`; dropout p = `0.15 · flat_cos(t)`.
- **Loop:** batch 256, a fresh permutation each epoch, the last n mod 256 rows dropped, 256 epochs by default (pytabkit's),
  final weights; no clipping, EMA or best-epoch selection.
