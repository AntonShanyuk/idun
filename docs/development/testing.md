# Testing

## Test kinds

| kind | where | data | runs in | purpose |
|---|---|---|---|---|
| **unit** | `tests/unit/` mirroring `packages/idun/src/idun/` | synthetic, generated in the test | `make test`, `make coverage`, `make precommit` | behaviour of one module; fast (the suite stays under a minute) |
| **property** | inside unit test modules (hypothesis) | generated | same | invariants over input spaces: round trips, oracle agreement, bounds |
| **oracle** | `tests/oracle.py` + unit tests | tiny synthetic | same | NumPy brute force vs CPU backend: exhaustive split search, exact leaves |
| **parity** | unit and end to end (`metal` marker) | synthetic | `make test` | CPU vs Metal, bit for bit, kernel by kernel and end to end |
| **port** | `tests/numerai/` | synthetic Numerai-shaped data | with unit | `idun_numerai`: data, labels, refits, scores, uploads |
| **end to end** | `tests/e2e/` | synthetic | with unit | every learner on its CPU (or reference) and Metal backends, through the port and the generic interface |

## Rules

1. **Layout.** `packages/idun/src/idun/a/b.py` is tested by `tests/unit/a/test_b.py`; a leading underscore in the module name is
   dropped (`_jit.py` → `test_jit.py`).
2. **Hermetic tests.** No network, no downloaded data, no files outside `tmp_path`: every test generates its data.
3. **No skips.** The `metal` marker labels the Apple GPU's tests; when Metal is unavailable they **fail**. To leave
   them out, say so explicitly with `-m "not metal"`. The `cuda` marker (an NVIDIA GPU, its driver and NVRTC) and the
   `nvrtc` marker (the compiler alone, no GPU) work the same way; the Makefile's `test` and `coverage` deselect both
   by name, and they run in `make check-cuda` and `make cuda-compile`.
4. **Warnings are errors** (`filterwarnings = error`), including numba performance warnings (a `parallel=True` loop that
   did not parallelize).
5. **xfail is strict.** An unexpected pass fails the run, so stale markers get removed.
6. **Determinism.** Seeds are explicit. Hypothesis runs derandomized with no example database by default
   (`HYPOTHESIS_PROFILE=thorough` for deep searches).
7. **Exact comparisons.** Parity tests use `np.testing.assert_array_equal`, not tolerances
   ([determinism.md](../architecture/determinism.md)). A tolerance needs a comment explaining which non-associative
   float operation it covers.
8. **Each Metal and CUDA kernel is named in a test.**
9. **The CUDA runtime is tested without a GPU.** `tests/unit/cuda/fake_driver.py` stands in for the driver and the
   compiler behind the same `ctypes` binding: it records every call, fails any call on request, keeps device memory
   as host bytes and runs a launch as a Python function when the stream is waited for. A test that holds on a real
   GPU as well takes the `anywhere` fixture and runs twice: on the fake, and under the `cuda` marker.

## Coverage

| scope | how | floor |
|---|---|---|
| Python (`idun`, `idun_numerai`) | `NUMBA_DISABLE_JIT=1 pytest --cov`, line + branch | **90 % total** |
| native (`libidun_metal`) | llvm-cov (`-fprofile-instr-generate`), exercised by the `metal` tests | a line report (`make coverage-native`) |
| MSL and CUDA kernels | not measurable | every kernel named in a test, with a bit-exact parity test |

Why the unit suite runs twice:

- coverage.py cannot trace compiled numba code, so `make coverage` runs everything as Python with `NUMBA_DISABLE_JIT=1`;
- integer width and overflow behave differently in Python and in compiled code, so `make test` runs the same tests with
  the JIT on.

Both must pass.

## Markers

| marker | meaning |
|---|---|
| `metal` | needs the Apple GPU (fails without it) |
| `cuda` | needs an NVIDIA GPU, its driver and NVRTC (deselected by name in `make test`; `make check-cuda`) |
| `nvrtc` | needs NVRTC alone, no GPU (deselected by name in `make test`; `make cuda-compile`) |
