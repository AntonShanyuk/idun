# Performance guide

How to get the most out of Metal and CUDA. Every item is something that was measured; the numbers are on the
[Benchmarks](benchmarks.md) page, from one machine of each kind.

## Choosing the backend

- **idun_gbm** is built for tables of millions of rows. On small tables (tens of thousands of rows) the CPU libraries
  fit faster than any GPU backend ([accuracy and fit time](benchmarks.md#boosting)); on 6 million rows one T4 fits
  600 trees in 34 s and an M4 in 48 s, where LightGBM takes 769 s on 4 CPU cores and 187 s on the M4's 10
  ([600 trees](benchmarks.md#idun_gbm-600-trees)).
- **The networks** gain most on large tables too. On the T4, idun_tabm's epoch took 0.78 of the `tabm` package's in
  float32 and idun_realmlp's 0.37 of pytabkit's; on an M4, idun_tabm took 0.40 of Torch's MPS backend.
- The CPU backends (`"cpu"` for idun_gbm, `"reference"` for the networks) are the references the GPUs are tested
  against, not fast paths: idun_gbm's CPU backend took over 100 times one T4's time on the same session.

## Data

- **Integer columns of 0 to 4 as `codes`.** Stored as 4-bit codes, two rows a byte: half a numeric column's memory
  and the fastest histogram path. A table of mostly such columns is idun_gbm's best case
  ([Data](data.md#idundataset-a-table-binned-for-idun_gbm)).
- **Bin once, reuse.** An `idun.Dataset` is binned once; pass it to several fits and predictions instead of a raw
  table each time.
- **Tables larger than memory: the packed format.** A packed dataset is memory-mapped and column-major; Metal wraps
  its arrays without a copy, and the networks' batches are gathered on the GPU from it
  ([packed datasets](data.md#packed-datasets-tables-larger-than-memory)).

## idun_gbm

- **Depth-wise trees are the fast shape.** Leaf-wise growth of 63 leaves took 1.96 times depth-wise depth 6's time on
  the M4; use `growth="leafwise"` when the model needs it, not by habit.
- **Seed bags side by side.** `n_seeds` (or `train_seeds`) with `workers=None` runs several fits at once: on CUDA one
  fit a GPU (four seeds on two T4s: 30 s against 54 s one after another; two fits a GPU gained nothing), on an M4 up to
  three (2.36× the throughput). Each fit's models are bit for bit its separate fit's.
- **Prediction is fast on the GPU**: 600 trees over 894,383 rows in 1.2 s on a T4, against about 20 s for LightGBM.

## idun_tabm and idun_realmlp

- **Batch size sets the step count.** An epoch is ⌈rows / `batch_size`⌉ steps: the large-table timings use 4,096-row
  batches for TabM, 1,453 steps an epoch on 6 million rows, where the default 256 rows (the paper's) would take
  23,236. A larger batch changes the training as well as its speed; tune it with the learning rate and epochs.
- **Seed bags of RealMLP gain little side by side** (four seeds, two networks in flight on an M4: 1.16×), because its
  256-row steps already keep the GPU busy.
- **On an 8 GB Mac, keep one network at a time and split TabM's batches.** Each RealMLP network references about 4 GB
  of GPU memory on a 6-million-row table: two side by side on an 8 GB M1 ran 16 times slower than one, so
  `train_realmlp_seeds` takes one at a time there by itself. An unsplit 4,096-row TabM batch pushed the memory-mapped
  table out of memory on the M1 (23 s a step); `micro_batch=1024` gave the same update for a third of the memory
  (3.0 GiB down to 0.95 GiB a step) and 0.7 s a step.
- **`storage="bf16"`** stores activations in bfloat16 on the GPU: up to 1.08× faster TabM steps on an M4 (an option,
  measured on a shared GPU). On a T4 it was slower (13 %), so it is not a CUDA speed-up.
- **No half precision on CUDA.** idun computes in float32; Torch's fp16 autocast with `torch.compile` beat it on a
  T4's tensor cores (216 s against 412 s an epoch).

## Metal

- **A fanless Mac throttles.** On the M4 MacBook Air a TabM step took 0.30 s in a burst and 0.38 s through whole
  epochs; the first epoch on a cool machine took 496 s, later ones 553 s. Time whole fits or epochs, not bursts, and
  let the machine cool between runs; compare arms in alternation (before, after, before, after).
- **Measure on a quiet machine.** Other GPU work shares the GPU: fits timed beside another GPU job fell on two
  levels a quarter apart, and a heavy neighbour slows a fit severalfold. Nothing in idun waits for a quiet GPU.

## CUDA

- **Two GPUs.** idun_gbm's seed bags use every GPU the driver lists (`workers=None`). The networks train one network a
  process: for a seed bag on two GPUs, run two processes, one with `IDUN_CUDA_DEVICE=0` and one with
  `IDUN_CUDA_DEVICE=1`, and average their predictions (no speed is recorded for it yet).
- **Compilation is cheap.** NVRTC builds the kernels for the device when a backend is made, about 0.2 s on a T4,
  a small part of a fit.
- **Containers: numba's threads.** In a container with a CPU quota, numba once started one thread per host CPU (96 on
  a 7.65-CPU quota) and a CPU test took 40.6 s instead of 1.9 s. idun sizes numba's threads to the quota when it is
  imported, unless `NUMBA_NUM_THREADS` is set; set it yourself if numba is imported before idun.
- **Drivers.** idun loads binaries built for the device, so any driver of NVRTC's CUDA major version runs them
  ([Getting started](getting-started.md#cuda)).
