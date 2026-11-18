# idun developer entry points (docs/development/setup.md).
#
# `make check` runs every gate: format, lint, types, import contracts, the tests, the coverage floor and the Metal tests
# under the validation layers and on the mirrored device. `make precommit` is the fast subset.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

RUN := uv run --frozen
PKG := packages/idun
NATIVE_SRC := $(shell find $(PKG)/native $(PKG)/src/idun \( -name '*.mm' -o -name '*.h' -o -name '*.metal' -o -name '*.cu' \) -type f)
UNAME := $(shell uname -s)
# Tests that need an NVIDIA GPU (cuda) or its compiler (nvrtc) run where those exist: make check-cuda, make cuda-compile.
# Off macOS the Metal tests are deselected by name too: Metal runs on Apple silicon only.
ifeq ($(UNAME),Darwin)
HERE := 'not cuda and not nvrtc'
else
HERE := 'not metal and not cuda and not nvrtc'
endif
COV_LIB := $(PKG)/build/coverage/lib/libidun_metal.dylib
PROF_DIR := $(PKG)/build/coverage/prof
LINUX_IMAGE := python:3.13-slim
UV_VERSION := $(word 2,$(shell uv --version))

.PHONY: help setup native wheels fmt lint test coverage coverage-native test-metal-validation test-metal-mirrored check-cuda cuda-compile precommit check build docs clean

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-24s %s\n", $$1, $$2}'

setup: ## Create .venv from uv.lock and build the native library
	uv sync --frozen
	$(MAKE) native

native: ## Build libidun_metal.dylib (release) into packages/idun/src/idun/_native (macOS; nothing elsewhere)
ifeq ($(UNAME),Darwin)
	cd $(PKG) && $(RUN) cmake --preset release > /dev/null
	cd $(PKG) && $(RUN) cmake --build --preset release
else
	@echo "native: no Metal library off macOS (the CPU and CUDA backends need no native library)"
endif

wheels: ## Build into dist/: idun's sdist, its macOS arm64 wheel (from the sdist) and pure wheel, idun-numerai's
	rm -rf dist
	uv build $(PKG) --out-dir dist
	IDUN_PURE_WHEEL=true uv build $(PKG) --wheel --out-dir dist
	uv build packages/idun-numerai --out-dir dist

fmt: ## Apply formatters and safe lint fixes
	$(RUN) ruff format
	$(RUN) ruff check --fix
	$(RUN) clang-format -i $(NATIVE_SRC)

lint: ## Format check, ruff, mypy, import contracts, clang-format
	$(RUN) ruff format --check
	$(RUN) ruff check
	$(RUN) mypy
	$(RUN) lint-imports
	$(RUN) clang-format --dry-run --Werror $(NATIVE_SRC)

test: native ## Tests with numba JIT on (the semantics that ship)
	$(RUN) pytest -m $(HERE)

coverage: native ## Tests with JIT off under coverage; the total floor
	NUMBA_DISABLE_JIT=1 $(RUN) pytest -q -m $(HERE) --cov --cov-report=term

coverage-native: ## Metal tests against an llvm-cov instrumented library; native line coverage report
	cd $(PKG) && $(RUN) cmake --preset coverage > /dev/null
	cd $(PKG) && $(RUN) cmake --build --preset coverage
	rm -rf $(PROF_DIR)
	IDUN_METAL_LIB=$(COV_LIB) LLVM_PROFILE_FILE=$(PROF_DIR)/%p.profraw $(RUN) pytest -q -m metal -p no:cacheprovider
	xcrun llvm-profdata merge -sparse -o $(PKG)/build/coverage/native.profdata $(PROF_DIR)/*.profraw
	xcrun llvm-cov report $(COV_LIB) -instr-profile=$(PKG)/build/coverage/native.profdata

test-metal-validation: native ## Metal tests under the Metal API and GPU (shader) validation layers
	MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 $(RUN) pytest -q -m metal -p no:cacheprovider

test-metal-mirrored: native ## Metal tests on the mirrored device: the host reaches GPU memory as on a discrete GPU
	IDUN_METAL_MIRRORED=1 $(RUN) pytest -q -m metal -p no:cacheprovider

check-cuda: ## The CUDA tests in guard mode: on a machine with an NVIDIA GPU, its driver and NVRTC
	IDUN_CUDA_GUARD=1 $(RUN) pytest -q -m cuda -p no:cacheprovider tests/unit

cuda-compile: ## In a Linux container (Docker): NVRTC compiles every CUDA source without a warning; the CPU unit tests
	git ls-files -co --exclude-standard -z packages tests/unit tests/conftest.py tests/oracle.py pyproject.toml uv.lock \
		| COPYFILE_DISABLE=1 tar --no-xattrs --null -T - -cf - \
		| docker run --rm -i -e UV_LINK_MODE=copy $(LINUX_IMAGE) bash -euc 'mkdir /work && cd /work && tar -xf - \
			&& pip install -q --root-user-action=ignore uv==$(UV_VERSION) && uv sync --frozen --group nvrtc \
			&& uv run --frozen pytest -q -m nvrtc -p no:cacheprovider tests/unit \
			&& uv run --frozen pytest -q -m "not metal and not cuda and not nvrtc" -p no:cacheprovider tests/unit'

precommit: lint coverage ## The fast gates

check: lint test coverage coverage-native test-metal-validation test-metal-mirrored ## Every gate

build: check ## The gated build: native library plus every gate

docs: ## The documentation site, warnings as errors
	$(RUN) --group docs mkdocs build --strict

clean: ## Remove build outputs and tool caches
	rm -rf build dist site $(PKG)/build $(PKG)/src/idun/_native .coverage .coverage.* .pytest_cache .mypy_cache .ruff_cache .hypothesis .import_linter_cache
	find packages tests -name __pycache__ -type d -prune -exec rm -rf {} +
