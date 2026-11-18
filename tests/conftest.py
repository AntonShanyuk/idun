"""Shared test configuration: hypothesis profiles (docs/development/testing.md) and the synthetic packed cache."""

import os
from pathlib import Path

import pytest
from hypothesis import settings

from tests.unit.data.synthetic import build_small, write_v6

# `default` is derandomized and writes no example database, so a run (and its coverage) is reproducible.
# `thorough` searches harder: HYPOTHESIS_PROFILE=thorough make test.
settings.register_profile("default", deadline=None, derandomize=True, database=None, print_blob=True)
settings.register_profile("thorough", deadline=None, max_examples=5_000, database=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(scope="session")
def nn_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    synthetic = write_v6(tmp_path_factory.mktemp("nn_synthetic") / "v6")
    return build_small(synthetic.v6, tmp_path_factory.mktemp("nn_cache") / "v6")
