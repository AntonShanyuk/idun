# Determinism and testing

## What is guaranteed

| learner | across backends | run to run |
|---|---|---|
| idun_gbm | **bit for bit**: CPU, Metal and CUDA build the same trees and the same predictions, compared with `==` | identical bits |
| idun_tabm, idun_realmlp | within a stated float32 tolerance of the float64 NumPy reference; not bit for bit between Metal, CUDA and the reference, nor between Apple GPU generations | identical bits on the same machine |
| JARF | its trees are idun_gbm's; its directions come from the networks, so they follow the networks' rule | identical bits on the same machine |

How idun_gbm stays bit for bit on every device ([the determinism contract](architecture/determinism.md)):

- every random draw is counter-based and keyed by the seed and the absolute row, tree or node (never by a thread, a
  partition or the time), and the host draws the bags, the feature samples and the per-node draws;
- gradients are quantized to small integers with a seeded stochastic rounding, so every histogram is an exact integer
  sum, whatever the order its threads add in;
- the split search and the leaf values are computed on the host in float64, from the integer histograms;
- CUDA kernels are compiled without fused multiply-add (`--fmad=false`), so each product and sum rounds as on the CPU.

The networks own their accumulation order instead: no float atomics, every reduction in an order fixed by the kernel's
code, the dropout masks regenerated from a counter rather than stored. That makes a fit repeat exactly on one machine,
while the GPUs' float32 arithmetic (division, transcendentals, fused multiply-adds in the GEMM on CUDA) is held to the
reference within a tolerance ([the networks' design notes](architecture/networks.md)).

## How it is tested

| kind | what | where it runs |
|---|---|---|
| unit and property tests | every module (`packages/idun/src/idun/a/b.py` has `tests/unit/a/test_b.py`), with hypothesis for invariants | everywhere, in under a minute |
| oracles | a NumPy brute force against idun_gbm's CPU backend: the exhaustive split search, exact leaves | everywhere |
| parity, `metal` | CPU against Metal, bit for bit, kernel by kernel and whole fits; the networks against their reference | Apple silicon Macs |
| parity, `cuda` | CPU against CUDA, as for Metal; every CUDA kernel named in a test | a machine with an NVIDIA GPU |
| `nvrtc` | every CUDA kernel source compiles under NVRTC, no GPU needed | Linux with NVRTC |
| an emulated CUDA driver | the CUDA runtime's every call, allocation, transfer and error path, against a fake driver behind the same binding | everywhere |
| end to end | every learner on the CPU and Metal (idun_gbm) or the reference and Metal (the networks), through the packed data path and the generic interface; CUDA through the unit tests' parity | everywhere (the Metal parts under their marker) |
| reference checks | idun_tabm and idun_realmlp against the `tabm` package and pytabkit, idun_gbm against LightGBM | the benchmarks ([Benchmarks](benchmarks.md)) |

Tests never skip: a `metal` or `cuda` test fails when its device is missing, and leaving them out is an explicit
`-m "not metal"`. Warnings are errors. Python coverage is held at 90 % (lines and branches).

## The gaps

What has not been tested, stated plainly:

- **GPUs.** CUDA has run on Tesla T4s (Turing, compute capability 7.5) and one RTX A5000 (Ampere) only, under CUDA 13
  drivers; a CUDA 12 driver is untested, and no Ada, Hopper or Blackwell GPU has run the suite. Metal has run on an
  M1 and an M4 only (MacBook Airs): no Pro, Max or Ultra chip, no M2 or M3.
- **Platforms.** No Windows. Intel Macs and Linux are CPU only, and Linux on arm64 has had no GPU.
- **Continuous integration has no GPU.** CI runs the CPU suite, the emulated-driver tests and the NVRTC compile checks;
  GitHub's macOS runners may not run Metal, so the Metal suite runs on real Macs, and the CUDA suite on a Kaggle T4
  pair, before each release.
- **JARF on CUDA** has one end-to-end test.
- **Across backends, the networks** are held to tolerances of the reference, not to each other's bits: a model trained
  on Metal and one trained on CUDA from the same seed differ slightly.
- **Speed** figures come from one machine of each kind ([Benchmarks](benchmarks.md)); others will differ.
- **Limits.** idun_gbm's depth-wise trees go to depth 8 and 256 leaves, and a categorical column has at most 64
  levels; the networks' GPU backends take at most 8 classes (and idun_realmlp at most 8 embedding values per input).

Contributions that close any of these are welcome: see [Contributing](contributing.md).
