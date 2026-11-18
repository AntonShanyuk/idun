# Coding guidelines

The guidelines, with the tool that enforces each one where a tool can. A rule marked **(review)** has no automatic
check; reviewers apply it by hand.

## 1. Principles

1. **Fail loudly.** Never fall back silently: no backend switch, no precision downgrade, no skipped test, no swallowed
   exception. Raise a `idun.errors` exception that names what is missing and how to fix it. **(review; tests assert
   the raises)**
2. **Exact before fast.** A kernel is correct when it matches the CPU reference bit for bit; only then is it optimized,
   and the optimization keeps the parity tests green. ([determinism.md](../architecture/determinism.md))
3. **Bytes are the budget.** The hot path is memory-bound. Justify every byte a kernel reads, and prefer narrower types
   and fewer passes. **(review)**
4. **Docs move with code.** A change to a public interface, file format, kernel contract or design decision updates
   its doc page in the same commit; every public name (`__all__`) is documented in
   [python-api.md](../api/python-api.md). **(review)**

## 2. Python

| rule | enforced by |
|---|---|
| ruff formatting, 120 columns | `ruff format --check` |
| broad lint set: pyflakes, bugbear, pylint, numpy (`NPY`), performance, security, simplify, pathlib, ... | `ruff check` |
| NumPy-style docstrings on every public module, class and function | ruff `D` (numpy convention) |
| full type annotations; `mypy --strict`; NumPy arrays typed as `NDArray[np.<dtype>]` | `mypy` |
| absolute imports only | ruff `TID252` |
| no `print` in the package (use return values, or `logging` in tools) | ruff `T20` |
| banned in `src/`: `mlx`, `torch`, `lightgbm`, `sklearn`, `pandas`, and direct `numba.njit` / `numba.jit` | ruff `TID251` |
| layer contracts (`idun.metal` ↛ `idun.gbm`; `idun.data` ↛ `gbm`, `metal`; `errors`, `_jit` import nothing internal) | `lint-imports` |
| `pragma: no cover` only with a reason: `# pragma: no cover - <why>` | review |

Conventions:

- **Explicit dtypes everywhere.** Write `np.empty(n, dtype=np.int32)`, never rely on a default. When integer width
  matters, write the overflow bound in a comment next to the code. **(review)**
- **Seeded generators only:** `np.random.Generator(np.random.PCG64(...))`. Never call the `np.random.*` module functions.
  **(ruff `NPY002`)**
- **Immutable configuration:** `@dataclass(frozen=True, slots=True)`, validated in `__post_init__`.
- **No hidden global state.** Caches are explicit (`functools.cache` with a documented reset, as `load_library`).
- **numba:** decorate with `idun._jit.njit` only. Use typed module-level constants (`np.uint64(...)`) in integer math.
  Keep jitted functions small and free of Python objects.
- **Errors carry the fix:** `"…/libidun_metal.dylib does not exist: build it with \`make native\`"`.

## 3. Native code (C ABI, Objective-C++)

| rule | enforced by |
|---|---|
| `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror` | CMake flags |
| `-fvisibility=hidden`; only `IDUN_API` functions are exported | CMake flags |
| ARC (`-fobjc-arc`); no manual retain/release except at bridged handle boundaries | CMake flags, review |
| clang-format (Google base, 120 columns) | `make lint` |
| C ABI conventions: status codes, error buffers, opaque handles, `IDUN_ABI_VERSION` bump on change | review, [backend-interface.md](../architecture/backend-interface.md) |
| every exported function exercised by a `metal` test | `make coverage-native` (line report), review |

## 4. MSL kernels

The CUDA-portable rules are in [metal-kernels.md](../architecture/metal-kernels.md) §1: integer atomics only, per-row
`+` and `*` only on float paths, draws from row indices, no Metal-only tricks in kernel logic. Every kernel is named in a
test and has a CPU parity test.

## 5. Data and safety

- Source data is **read-only**: code opens it for reading only, and never writes, moves or deletes anything there.
- Load `.npz` files with `allow_pickle=False`.
- No credentials and no network uploads.

## 6. Commits

- Small, single-purpose commits. The message says what and why.
- `make precommit` (lint and coverage) must pass before you commit, on a Mac; `make check` runs every gate.
