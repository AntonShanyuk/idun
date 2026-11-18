# Development setup

## Prerequisites

| tool | version | check |
|---|---|---|
| macOS on Apple silicon | macOS 15 or later (the native library's minimum), M1 or later | `sw_vers`, `uname -m` → `arm64` |
| Command Line Tools (clang, llvm-cov) | Apple clang 21 | `clang --version`, `xcrun --find llvm-cov` |
| Python | 3.13 | `python3.13 --version` |
| uv | ≥ 0.9.9 | `uv --version` |

Xcode is **not** required: kernels compile at runtime from source.

## First time

```bash
make setup        # uv sync from uv.lock, native library build
make check        # every gate; should pass on a fresh clone
```

## Make targets

| target | what it does |
|---|---|
| `make help` | list the targets |
| `make setup` | create `.venv` from `uv.lock`, build the native library |
| `make native` | CMake release build of `libidun_metal.dylib` into `packages/idun/src/idun/_native/` (macOS 15.0 and later) |
| `make wheels` | Into `dist/`: idun's sdist, its `py3-none-macosx_15_0_arm64` wheel built from the sdist (scikit-build-core runs CMake), its pure `py3-none-any` wheel (`IDUN_PURE_WHEEL=true`: no library, CPU backends only) and idun-numerai's sdist and wheel |
| `make fmt` | ruff format, safe ruff fixes, clang-format |
| `make lint` | format check, ruff, mypy (strict), import contracts, clang-format check |
| `make test` | the tests, numba JIT on (all but the `cuda` and `nvrtc` tests, which need NVIDIA's GPU or compiler) |
| `make coverage` | the tests, JIT off, under coverage: ≥ 90 % total |
| `make coverage-native` | `metal` tests against an llvm-cov instrumented library: a native line coverage report |
| `make test-metal-validation` | `metal` tests under the Metal API and GPU validation layers |
| `make test-metal-mirrored` | `metal` tests on the mirrored device: the host gets copies of GPU memory, as on a GPU with memory of its own |
| `make check-cuda` | the `cuda` tests in guard mode: on a Linux machine with an NVIDIA GPU, its driver and NVRTC |
| `make cuda-compile` | in a Linux container (Docker; the `nvrtc` dependency group): NVRTC compiles every CUDA kernel source without a warning (the `nvrtc` tests), then the CPU unit tests on Linux. No GPU needed |
| `make precommit` | `lint` + `coverage`, the fast gates |
| `make check` / `make build` | every gate; `build` is the gated build (the native library plus every gate) |
| `make docs` | build this documentation with `mkdocs build --strict` (warnings are errors; the `docs` dependency group) |
| `make clean` | remove build outputs (the docs site included) and tool caches; never touches `data/` |

All Python tooling runs through `uv run --frozen`, so the lockfile is the single source of versions.

## Environment variables

| variable | used by | meaning |
|---|---|---|
| `IDUN_METAL_LIB` | `idun.metal.runtime` | load this `libidun_metal.dylib` instead of the packaged one (the coverage build uses it) |
| `IDUN_CACHE_COPY` | `PackedDataset.open` | `1`: a copied cache (no source files on this machine), checked by its checksums instead of its sources |
| `NUMBA_DISABLE_JIT` | numba | `1` in `make coverage` only |
| `HYPOTHESIS_PROFILE` | tests | `default` (derandomized) or `thorough` |
| `MTL_DEBUG_LAYER`, `MTL_SHADER_VALIDATION` | Metal | `1` in `make test-metal-validation` |
| `IDUN_CUDA_LIB`, `IDUN_NVRTC_LIB` | `idun.cuda.runtime` | load this CUDA driver library (`libcuda.so.1`) or this NVRTC library instead of the ones found |
| `IDUN_CUDA_DEVICE` | `idun.cuda.runtime` | the GPU's ordinal (default `0`) |
| `IDUN_CUDA_GUARD` | `idun.cuda.runtime` | `1` in `make check-cuda`: guard bands around every allocation, checked after every launch (any other value is an error) |
| `IDUN_METAL_MIRRORED` | `idun.metal.mirrored.open_device` | `1` in `make test-metal-mirrored`: Metal backends open the mirrored test device (any other value is an error) |

## Adding a dependency

Edit the right `pyproject.toml`: `dependencies` in `packages/idun/pyproject.toml` (or `packages/idun-numerai/`) for a
package's runtime, the workspace's `dev` group for tooling. Then run `uv lock` and commit the `pyproject.toml` and
`uv.lock` together. A new runtime dependency needs a documented reason: it ships to every user.

## Repository layout

```
packages/idun/          distribution `idun`: src/idun (docs/architecture/overview.md), native/ (C ABI header and
                        Objective-C++ runtime), CMakeLists.txt, pyproject.toml
packages/idun-numerai/  distribution `idun-numerai`: the Numerai port
pyproject.toml          the uv workspace: its dependency groups and every tool's settings
tests/unit/             hermetic unit tests, mirroring packages/idun/src/idun
tests/numerai/          tests of the Numerai port on synthetic Numerai-shaped data
tests/e2e/              every learner on its CPU (or reference) and Metal backends, end to end
docs/                   this documentation
data/                   local data and caches, gitignored
```
