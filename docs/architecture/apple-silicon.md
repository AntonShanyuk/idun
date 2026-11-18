# Using the Apple silicon: what we use, what we don't, and why

idun_gbm aims to use the Mac's hardware as fully as it can, which LightGBM does not. This page lists an Apple M4's
compute and memory resources and says which ones idun_gbm uses.

## The machine

An Apple M4 in a fanless MacBook Air (passive cooling); other M-series chips differ in core counts and memory. "probe" is `idun.metal.runtime.probe()`:

| resource | value | source |
|---|---|---|
| CPU | 4 performance + 6 efficiency cores | `sysctl hw.perflevel*` |
| GPU | 10 cores, Apple9 family, Metal 4, SIMD width 32 | Apple spec, probe, `Kernel.simd_width` |
| threadgroup memory | 32 KiB per threadgroup; ≤ 1,024 threads per threadgroup | probe |
| unified memory | 24 GB; recommended GPU working set 17.8 GiB; max buffer 13.3 GiB | probe |
| page size | 16 KiB | probe |
| memory bandwidth | ~120 GB/s | Apple spec |

## Resource by resource

| unit | used? | how / why not |
|---|---|---|
| **GPU (Metal compute)** | **yes, the main engine** | Histogram building is scattered integer adds over a byte stream. Thousands of GPU threads keep the memory bus busy where CPU code rarely does. Every hot kernel runs here. |
| **Unified memory** | **yes, everywhere** | No host↔device copies: the packed cache is memory-mapped and wrapped as an `MTLBuffer` without a copy (verified). Per-row state lives in shared buffers that NumPy views directly, so the host reads histograms and writes splits in place. |
| **Bandwidth economy** | **yes, the main lever** | 4-bit features (2 rows/byte), int8 quantized gradients (4× fewer gradient bytes than f32), uint8 node ids, column-major storage so a tree reads only its sampled columns, histogram subtraction. |
| **Threadgroup memory + atomics** | yes | Privatized int32 histograms in 32 KiB, flushed once per threadgroup. Integer atomics are order-independent, hence exact. |
| **SIMD-group functions** | yes | `simd_sum`, `simd_prefix_exclusive_sum` and ballots to cut atomic contention and to compact rows; they map one-to-one to CUDA warp intrinsics. |
| **P-cores** | yes | Python orchestration and host-side split finding (small: nodes × features × ≤ 256 bins). |
| **E-cores** | reference only | The numba CPU backend uses them. Timings are reported at 10 and 4 threads. |
| **CPU NEON SIMD** | via numba/LLVM | Auto-vectorized in the CPU backend; no hand-written intrinsics: the CPU backend is the reference, not the fast path. |
| **SME / AMX (matrix units)** | no | Dense matrix outer products; histogramming is gather/scatter on bytes. Reconsider for the networks. |
| **Neural Engine** | no | Fixed-function inference on fp16 tensors with no general integer scatter-add; not reachable for custom kernels. Reconsider for the networks inference. |
| **Media engines, ray tracing** | no | Irrelevant. |

## Constraints that shape the design

- **Thermals.** A fanless MacBook Air throttles under sustained load, so only sustained-run timings count.
- **Shared GPU.** Another process on the GPU changes timings and takes from the same working set: time on an otherwise
  idle GPU.
- **No Xcode needed.** Kernels compile at runtime from source; GPU timing uses command-buffer
  timestamps (`GPUStartTime` / `GPUEndTime`), and per-kernel timing splits command buffers in a profiling mode.
- **FP contraction.** Metal fuses multiply-add by default, even in Safe math mode. The runtime turns it off for every
  library ([determinism.md](determinism.md)).
