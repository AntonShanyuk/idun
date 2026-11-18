# Contributing to idun

Thank you for helping. idun's promise is that a fit is reproducible to the bit and fast on the GPU, and that every
claim about it, correctness or speed, is measured. These guidelines are how a change keeps that promise. Bug reports
with a small reproduction, measurements on hardware we have not tested (the gaps are listed in
[Determinism and testing](testing.md#the-gaps)) and fixes are all welcome.

## Setting up

```bash
git clone https://github.com/AntonShanyuk/idun && cd idun
make setup        # .venv from uv.lock (uv required), and the Metal library on a Mac
make precommit    # the fast gates: format, lint, types, import contracts, the tests under coverage
```

Python 3.12 to 3.14 and [uv](https://docs.astral.sh/uv/). On a Mac, the Metal library builds with the Command Line
Tools' clang (the kernels compile at run time, so Xcode is not needed). Elsewhere `make setup` builds nothing native and `make test` deselects the Metal tests by
marker; the coverage floor of `make precommit` counts the Metal code, so on Linux run `make lint` and `make test`
(CI checks the floor's report there and the floor itself on a Mac). `make help` lists every target.

## The gates

| target | what | when |
|---|---|---|
| `make precommit` | `ruff format --check`, `ruff check`, `mypy` (strict), `lint-imports`, `clang-format`, and the tests with the JIT off under coverage (90 % floor, lines and branches) | before every commit, on a Mac (on Linux: `make lint` and `make test`) |
| `make test` | the tests with numba's JIT on, the semantics that ship | before a pull request |
| `make check` | everything above, plus native coverage and the Metal tests under Apple's validation layers and on the mirrored device | on a Mac, before a pull request that touches Metal |
| `make check-cuda` | the CUDA tests with guard pages around every allocation | on a machine with an NVIDIA GPU, before a pull request that touches CUDA |
| `make cuda-compile` | NVRTC compiles every CUDA source without a warning, in a Linux container | anywhere with Docker |
| `make docs` | the documentation site, warnings as errors | when the docs change |

Never weaken a gate to get a change through (a lower floor, a looser tolerance, a skipped test, an ignore); if a gate
is wrong, say so in the pull request.

## Tests

Every module has its test file: `packages/idun/src/idun/a/b.py` is tested by `tests/unit/a/test_b.py` (a leading
underscore dropped). Tests are hermetic: they generate their data, write only under `tmp_path`, and touch no network.

- **Unit and property tests** for behaviour, with [hypothesis](https://hypothesis.readthedocs.io) for invariants
  (derandomized by default; `HYPOTHESIS_PROFILE=thorough` searches deeper).
- **Oracles.** idun_gbm's CPU backend is checked against brute-force NumPy (`tests/oracle.py`): the exhaustive split
  search, exact leaf values. A new split rule, objective or regulariser gets an oracle test.
- **Parity.** A GPU path is checked against the CPU, bit for bit with `np.testing.assert_array_equal`, kernel by kernel
  and for whole fits. A tolerance is allowed only for the networks, against their float64 reference, with a comment
  saying which float operation it covers.
- **End to end** (`tests/e2e/`): each learner on its CPU (or reference) and Metal backends, through the packed data
  path and the generic interface.
- **Acceptance tests** (`tests/e2e/`): one whole behaviour a user relies on, end to end, through the public
  interface: a learner trained, saved, loaded and predicting on each backend, the estimators' contract, a whole
  configuration of parameters set together. Each states its criterion as assertions on bits, shapes, determinism or
  agreement with a reference, never on a score, and runs on synthetic or public data under the markers of the
  backends it needs. A feature that a user sees comes with one.

Rules that apply everywhere: no skipped tests (a test that cannot run fails); warnings are errors; `xfail` is strict;
seeds are explicit.

### Vendor-specific tests

| marker | needs | without it | run with |
|---|---|---|---|
| `metal` | an Apple silicon Mac with macOS 15 or later | fails | `make test`; `-m "not metal"` leaves them out explicitly |
| `cuda` | Linux, an NVIDIA GPU (compute capability 7.5 or newer), its driver and NVRTC | deselected by name in `make test` | `make check-cuda` |
| `nvrtc` | NVRTC alone, no GPU | deselected by name in `make test` | `make cuda-compile`, or `uv sync --group nvrtc` and `pytest -m nvrtc` on Linux |
| none | the CPU | | everywhere, in CI on Linux x86-64 and arm64, Python 3.12 to 3.14 |

- **Every Metal and CUDA kernel is named in a test**, with a parity test against the CPU (idun_gbm) or the reference
  (the networks). Kernel code cannot be covered by coverage.py, so this is how it is held.
- **The CUDA runtime is tested without a GPU**: `tests/unit/cuda/fake_driver.py` stands in for the driver and NVRTC
  behind the same binding. A change to the runtime gets a test on the fake; a test that should also hold on a real
  GPU takes the `anywhere` fixture and runs on both.
- **CI has no GPU.** Run the `metal` suite on a Mac and the `cuda` suite on an NVIDIA GPU yourself when a change
  touches them, and say in the pull request which machines ran them (the chip or GPU, the macOS or driver version).
  Before each release the maintainers run both suites on real hardware.

## Determinism

A change must not make a fit depend on anything but its data, parameters and seed:

- numba only through `idun._jit.njit` (it fixes the compilation options);
- no float atomics and no float reductions whose order depends on scheduling, on any device; idun_gbm's histograms
  stay integer sums;
- every random draw from `idun.rng` or a seeded `numpy.random.Generator`, keyed by the seed and the absolute row,
  tree, node or step, never by a thread, a partition or the clock;
- CUDA sources compile without contraction (`--fmad=false`) where they must match the CPU.

The contract is [architecture/determinism.md](architecture/determinism.md) for idun_gbm and
[architecture/networks.md](architecture/networks.md) for the networks. Fail loudly: no silent fallbacks, no
automatic choice of backend.

## Benchmarks

A speed or accuracy claim in a pull request, a doc or a changelog needs a measurement anyone can repeat:

- **Absolute numbers**, with the reference rows measured in the same run (LightGBM, XGBoost, the `tabm` package,
  pytabkit, or idun before the change), never a ratio alone.
- **The machine and the versions**: the chip or GPU, memory, the OS and driver, Python and every library involved.
- **A quiet machine**, on AC power, one measurement at a time; a fanless laptop throttles under sustained GPU load, so
  time whole fits or epochs and alternate the arms (before, after, before, after).
- **The method stated before the numbers**: what is timed (a whole fit, an epoch, a kernel), the median of how many
  runs, the data's shape and the configuration.
- `benchmarks/openml.py` is the public accuracy and speed suite ([Benchmarks](benchmarks.md)); extend it rather
  than writing a one-off script.

A change that makes something faster on one backend shows that it is no slower on the others, and that the parity
tests still pass with `==`.

## Documentation and the public surface

The docs move with the code, in the same pull request:

- a public name, parameter or default: `docs/api/python-api.md`, the learner page's parameter table
  (`docs/learners/`) and a `CHANGELOG.md` entry;
- a backend or kernel change: the architecture page it touches;
- public names, parameters and defaults are stable within a major version, and model files load in every
  later release of it.

## Pull requests

- One change per pull request, with the gates' output and, for GPU changes, the machines the device suites ran on.
- Code style: `ruff` and `clang-format` decide formatting; type annotations everywhere (`mypy --strict`).
- Commits carry their author alone: no co-author or generated-by trailers.
- By contributing you agree that your contribution is licensed under the Apache License 2.0, as the project is.
