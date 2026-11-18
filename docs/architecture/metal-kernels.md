# Metal kernels

idun_gbm's kernels are in `packages/idun/src/idun/gbm/kernels/metal/gbm.metal` and match the CPU backend bit for bit.
§2 is the kernel set in its first, simple form (correctness first); §4 is the optimization ladder built on it and §4a
its measurements.

## 1. Rules for every kernel (CUDA-portable subset)

1. Express everything as threadgroup = block, threadgroup memory = shared memory, SIMD-group = warp (32). Use the mapping
   table in [backend-interface.md](backend-interface.md) §4.
2. **Integer atomics only**, relaxed order. No float atomics, no float reductions ([determinism.md](determinism.md)).
3. Floats: only per-row `+` and `*` on parity-checked paths. No division or transcendentals; the host passes reciprocals.
4. Draws come from the row's absolute index via `lowbias32`, never from thread ids.
5. No Metal-only tricks in kernel logic: no texture sampling, no imageblocks, no tile shaders. Metal-specific tuning
   belongs in launch configuration only (threadgroup size, threadgroup memory size).
6. Parameters go in one `struct` passed with `setBytes`, mirrored by a `ctypes.Structure`, with sizes asserted in a test.
7. Each kernel is named in a unit test and has a bit-exact parity test against the CPU backend.
8. Bounds are checked by the Metal GPU validation layer in `make test-metal-validation`.

## 2. Kernel set (simple versions first)

| kernel | grid | reads | writes |
|---|---|---|---|
| `rng_uniform_bits` | rows | — | uint32 bits (test kernel for parity with `idun.rng`) |
| `fma_probe` | rows | `a`, `b`, `c` | `a * b + c` (test kernel: must round twice) |
| `compute_gradients` | rows | `pred`, `y` | `g`; max\|g\| via a SIMD max and one `atomic_fetch_max` on the float bits per SIMD group |
| `quantize_gradients` | rows | `g`, `bag`, key, `inv_scale` | `q` |
| `build_histograms` | (feature tiles × row chunks) | sampled columns, `q`, `node`, `bag`, node → slot map | int32 histogram slots (device atomics; §4 privatizes them) |
| `partition_rows` | rows | split columns, level splits, `node` | `node` |
| `split_leaf` | rows × models | split columns, one split per model, `node` (leaf ids) | `node` of the split leaf's rows that go right (leaf-wise) |
| `leaf_partials` | 4,096-row chunks | `pred`, `y`, `node`, `bag` | int64 Σg_fx and count per (chunk, leaf); the host sums the chunks. No 64-bit atomics on Metal: each value is split as `hi·2^32 + mid·2^16 + low` into three 32-bit threadgroup atomics that cannot overflow within a chunk |
| `update_predictions` | rows | `node`, leaf values | `pred` (fusing the next max\|g\| in is a §4 step) |
| `predict_trees` | rows | packed columns, tree arrays | float32 sums |
| `predict_nodes` | rows | packed columns, node-list tree arrays | float32 sums (leaf-wise trees) |

## 3. Command structure per tree

```
cb0: quantize_gradients → build_histograms(level 0)                       → wait → host: splits(level 0)
cb1: partition_rows(level 0) → build_histograms(level 1, smaller children)  → wait → host: subtract, splits(level 1)
...
cbD: partition_rows(level D−1) → leaf_partials                              → wait → host: leaf values
cbD+1 (fused into the next tree's cb0): update_predictions
```

A leaf-wise tree has one command buffer per split instead of per level:
`split_leaf(leaf) → build_histograms(smaller child) → wait → host: subtract, splits of both children, next leaf`.

## 4. The optimization ladder

Each step is measured with per-kernel GPU timings on a 5.95-million-row window of Numerai's data, and kept only if
the Metal backend still matches the CPU backend bit for bit:

1. **Vectorized 4-bit reads.** One `uint4` load holds 32 rows of one feature. Each thread unpacks nibbles in registers.
   A naive byte-per-thread histogram reached ≈ 31–42 GB/s (two unprofiled runs) of an Apple M4's ~120 GB/s bus, which
   shows how much load width matters.
2. **Threadgroup-privatized histograms.** Each threadgroup takes a tile of features and accumulates
   `nodes × bins × {Σq, n}` int32 in threadgroup memory, then flushes once with device atomics. For obfuscated features:
   64 nodes × 5 bins × 8 B = 2.5 KiB per feature, so ~12 features per 32 KiB.
3. **Contention relief for 5-bin features.** Five counters per node are hot spots:
   - per-SIMD-group sub-histograms (fewer features per threadgroup);
   - at levels 0–2 (≤ 4 nodes, ≤ 20 counters), accumulation in registers with a `simd_sum` reduction;
   - pick by measurement.
4. **Named features (≤ 256 bins).** 64 nodes × 256 bins × 8 B = 128 KiB does not fit in 32 KiB. Use node-chunked passes
   at deep levels, or device atomics for these 22 columns only. The choice is measured and reported, never silent.
5. **Histogram subtraction**, and only the smaller children are built (the subtraction runs on the host for now).
6. **Fusion:** max\|g\| computed inside `update_predictions`; quantization fused into the root histogram pass.
7. **Command buffers:** one per level, the passes encoded back to back. Untracked hazards, residency sets and Metal 4's
   command API are candidates, not done: encoding does not show up in the profiles.
8. **Host:** vectorized split finding (§4a).
9. **Several models:** K models share each column read: K gradient vectors and K node-id vectors per pass
   (`train_many`).

## 4a. The ladder's log (engineering measurements)

The same window on an Apple M4, depth-wise trees of depth 6, 30 trees: per-kernel GPU time
from a profiling backend (one command buffer per dispatch) and the normal wall-clock. Indicative only, not timed runs.
The CPU–Metal bit-exactness checks are re-run after every kept step.

| step | histogram GPU ms/tree | wall ms/tree | kept |
|---|---|---|---|
| baseline (§2): one thread per row, device atomics for every feature | 194 | 223 | — |
| lane owns column (obfuscated features), one row per iteration, static 30 KiB threadgroup memory | 928 | 952 | no: one SIMD group per core, latency-bound |
| + 32-row blocks: coalesced row metadata broadcast with `simd_shuffle`, one 16-byte load per lane | 525 | 632 | no: still one group per core |
| + dynamic threadgroup memory sized per dispatch (C ABI v3: `threadgroup_bytes`) | 119 + 8 | 174 | yes |
| + one packed counter per (slot, bin) (Σq·2^16 + count), unrolled block loop, skip empty blocks | 76 + 8 | 108 | yes |
| + register counters when a dispatch has one slot (levels 0–1), 16,384-row groups | 66 + 8 | 98 | yes (then replaced) |
| bitmask path for ≤ 4 slots: per block, ballots for slot and the bits of \|q\|, per-lane "code > t" masks, popcounts | — | 112 | no at 4 slots (19.9 vs 11.1 ms per level) |
| bitmask path for ≤ 2 slots (levels 0–2), per-row path above; chunks sized to the slots present | 63 + 7 | 98 | yes (GPU −3 ms) |
| host: the next bag (PCG64, 5.95 M draws, 10.5 ms) drawn on a worker thread while the GPU grows the current tree | 63 + 7 | 91 | yes |
| host: split finding vectorized over all nodes of a level (same float64 expressions) | 64 + 7 | 89 | yes |
| named features and the country in threadgroup counters (3 slots per 16 KiB: deep levels re-scan up to 6 times) | 64 + 7.0 | 98 | no (no gain; reverted) |

Per level (random nodes, half the rows selected below the root): 7.9, 6.7, 9.1, 11.1, 13.5 and 23.9 ms for levels 0–5.
Removing the final device-atomic flush changes nothing; smaller slot chunks only add re-scans. Level 0 reads the sampled
packed columns at about 37 GB/s against ~120 GB/s available, so the deep levels' per-row threadgroup traffic and the
full-panel scan per level (rows outside the build slots are read and skipped) are what remain. Candidates: a per-tree
gather of the sampled columns, rows ordered by node at deep levels, the named/country pass in threadgroup memory
(7 ms).

**One model on the K-model backend.** A single fit runs through the K-model backend (`MetalMany(1)`). The K-model
histogram kernel (`histograms_obfuscated`) made a single fit on that window about 1.2× slower (77.4 against 63.6 ms per tree in
the kernel, the loss at the deep levels), because its per-pair words and chunk entries take threadgroup memory a single
model does not need. So `MetalMany` dispatches the single-model kernel, `histograms_obfuscated_one`, when K = 1: the
same predictions, and a fit within 1 % of the single-model backend's time.

## 4b. The same kernels on CUDA

`packages/idun/src/idun/gbm/kernels/cuda/gbm.cu` holds every kernel of the table under the same name, each the Metal
kernel operation for operation: a threadgroup is a block, threadgroup memory is shared memory sized by the host per
launch, a SIMD-group is a warp. The passes that drive them are the same code (`idun.gbm.backends.device`), so the CUDA
backend gives the CPU's bits; NVRTC compiles the source without contraction (`--fmad=false`). What CUDA does
differently is launch configuration, and one kernel Metal does not use:

- **Rows a block** of the single-model obfuscated pass: 4,096 on CUDA (the shared default is 16,384): more blocks for a
  T4's 40 SMs.
- **The named and categorical features** counted per block in shared memory with integer atomics
  (`build_histograms_shared`: 256 threads over 65,536 rows, one feature and as many slots as 32 KiB holds), then
  added to the histogram, instead of integer atomics row by row in device memory (`build_histograms`). Metal tried the
  same and kept `build_histograms` (three slots per 16 KiB re-scanned the rows at deep levels; no gain).
- **The seed bag**: `train_seeds` runs one fit a GPU at a time on CUDA.

Measured on two Tesla T4 (5.95 million training rows of Numerai's data, depth-wise trees of depth 6, 600 trees; whole fits: the backend made, the trees
grown, the backend closed):

| step | whole fit (median of 3) | kept |
|---|---|---|
| the port as it is (the Metal kernels' launch configuration) | 37.4 s (histograms 26.8 s, the host's split search 4.8 s) | — |
| + 4,096 rows a block for the single-model pass | 35.1 s (whole fits alternated with the defaults: 5.0 % faster) | yes |
| + the host's split search compiled (numba, numpy's operations in numpy's order; bit for bit the numpy search) | split search 4.95 → 1.59 s | yes (6.5 %) |
| + the named features counted in shared memory | that kernel 3.59 → 2.85 s | yes (2.2 %) |
| all three | 32.7 s, 12.6 % less than the port as it is | |

Per kernel (one T4, all three steps): the single-model obfuscated pass 82 % of the GPU time, the named features' pass
11 %, the partition 4 %; a histogram pass costs about the same whatever its slots (3.3 to 6.7 s per 600 passes by
level), since every pass reads every row's codes. Not done, by the measures: compiled kernels kept per process (a
backend is made in 0.2 s), uploads of only the rows a pass reads (the upload is 2 % of a fit), and overlapping the
histogram download with the split search (splitting a level's pass would add a pass's row reads to hide at most the
search's 1.3 ms).

On the same session, LightGBM (leaf-wise, 63 leaves, quantized gradients) fitted the same 600 trees in 769 s on the
session's 4 CPUs and 343 s with LightGBM's own OpenCL build on one T4; idun_gbm took 33.9 s on one T4 (22.7× the CPU
fit).

## 5. Timing and validation

- GPU time per command buffer: `GPUEndTime − GPUStartTime`. Per-kernel time: a profiling mode that commits one kernel
  per command buffer. It is used for the timing table only, never for wall-clock claims.
- `make test-metal-validation` runs every `metal` test under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`.
- `make test-metal-mirrored` runs every `metal` test on the mirrored device
  ([backend-interface.md](backend-interface.md) §2): the host's reads and writes of GPU memory are held to the
  rules of a GPU with memory of its own.
