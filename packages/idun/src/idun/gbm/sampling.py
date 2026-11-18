"""Host-side draws: the row bag of an iteration and the feature sample of a tree (docs/architecture/determinism.md §1).

Both come from ``numpy.random.Generator(PCG64(SeedSequence([seed, counter, stream])))``, so they are a pure function of
the seed and the counter. Every backend receives the same mask and the same features by construction. The rules
follow LightGBM: a Bernoulli draw per row (``bagging.hpp``) and ``max(round(fraction × n), 1)`` features
(``col_sampler.hpp``).
"""

import numpy as np
from numpy.typing import NDArray

BAG_STREAM = 0
FEATURE_STREAM = 1


def _generator(seed: int, counter: int, stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, counter, stream])))


def bag_mask(n_rows: int, fraction: float, *, seed: int, iteration: int) -> NDArray[np.uint8]:
    """Return 1 for each of ``n_rows`` training rows drawn into iteration ``iteration``'s bag, else 0."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    draws = _generator(seed, iteration, BAG_STREAM).random(n_rows, dtype=np.float32)
    return (draws < np.float32(fraction)).astype(np.uint8)


def feature_count(n_features: int, fraction: float) -> int:
    """Features per tree: ``max(floor(n_features × fraction + 0.5), 1)`` (LightGBM ``RoundInt``)."""
    if n_features < 1 or not 0.0 < fraction <= 1.0:
        raise ValueError(f"need n_features >= 1 and fraction in (0, 1], got {n_features}, {fraction}")
    return max(int(n_features * fraction + 0.5), 1)


def feature_sample(n_features: int, fraction: float, *, seed: int, tree: int) -> NDArray[np.int32]:
    """Sorted feature indices sampled without replacement for tree ``tree``; all features when ``fraction == 1``."""
    k = feature_count(n_features, fraction)
    if k == n_features:
        return np.arange(n_features, dtype=np.int32)
    chosen = _generator(seed, tree, FEATURE_STREAM).choice(n_features, size=k, replace=False)
    return np.sort(chosen).astype(np.int32)
