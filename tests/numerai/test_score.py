import math

import numpy as np
import pandas as pd
import pytest
from numerai_tools import scoring
from numpy.typing import NDArray

from idun_numerai import score


def era(rng: np.random.Generator, n: int, m: int) -> tuple[NDArray[np.float64], ...]:
    neutralizers = rng.normal(size=(n, m))
    y = np.round(4 * rng.uniform(size=n)) / 4  # Numerai's 5 bins
    p = 0.3 * y + neutralizers[:, 0] + rng.normal(size=n)
    p[:7] = p[7]  # ties are kept
    return p, y, neutralizers


def numerai_ncorr(p: NDArray[np.float64], y: NDArray[np.float64], neutralizers: NDArray[np.float64]) -> float:
    frame, target = pd.DataFrame({"prediction": p}), pd.Series(y, name="target")
    return float(scoring.neutral_correlation(frame, pd.DataFrame(neutralizers), target)["prediction"])


def numerai_corr(p: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    return float(scoring.numerai_corr(pd.DataFrame({"prediction": p}), pd.Series(y, name="target"))["prediction"])


@pytest.mark.parametrize(("seed", "n", "m"), [(0, 400, 12), (1, 1000, 1), (2, 250, 60), (3, 5000, 300)])
def test_ncorr_is_numerai_tools_neutral_correlation(seed: int, n: int, m: int) -> None:
    p, y, neutralizers = era(np.random.default_rng(seed), n, m)
    assert score.ncorr(p, y, neutralizers) == pytest.approx(numerai_ncorr(p, y, neutralizers), abs=1e-12)


def test_ncorr_drops_a_rank_deficient_or_constant_neutralizer_as_numerai_tools_does() -> None:
    p, y, neutralizers = era(np.random.default_rng(4), 300, 3)
    deficient = np.column_stack([neutralizers, 2 * neutralizers[:, 1], np.full(300, 0.5)])
    assert score.ncorr(p, y, deficient) == pytest.approx(score.ncorr(p, y, neutralizers), abs=1e-12)
    assert score.ncorr(p, y, deficient) == pytest.approx(numerai_ncorr(p, y, deficient), abs=1e-12)
    constant = np.ones((300, 1))  # only a constant: NCORR is the correlation of the gaussianized prediction
    assert score.ncorr(p, y, constant) == pytest.approx(numerai_ncorr(p, y, constant), abs=1e-12)
    assert score.ncorr(p, y, constant) == pytest.approx(np.corrcoef(score.gaussianized(p), y)[0, 1], abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_corr_is_numerai_tools_numerai_corr(seed: int) -> None:
    p, y, _ = era(np.random.default_rng(10 + seed), 500, 1)
    assert score.corr(p, y) == pytest.approx(numerai_corr(p, y), abs=1e-12)


def test_gaussianized_keeps_ties_and_the_order() -> None:
    g = score.gaussianized([3.0, 1.0, 1.0, 2.0])
    assert g[1] == g[2]
    np.testing.assert_array_equal(np.argsort(g, kind="stable"), [1, 2, 3, 0])
    np.testing.assert_allclose(score.gaussianized([2.0, 0.0, 1.0]), [0.9674215661017009, -0.9674215661017009, 0.0])


def test_a_constant_prediction_or_target_raises() -> None:
    p, y, neutralizers = era(np.random.default_rng(5), 200, 4)
    flat = np.full(200, 0.5)
    for args in ((flat, y, neutralizers), (p, flat, neutralizers)):
        with pytest.raises(ValueError, match="is constant in the era"):
            score.ncorr(*args)
    with pytest.raises(ValueError, match="predictions is constant"):
        score.corr(flat, y)
    with pytest.raises(AssertionError):  # numerai-tools: undefined too
        numerai_ncorr(flat, y, neutralizers)
    assert math.isnan(numerai_corr(p, flat))


def test_per_era_scores_each_run_of_rows() -> None:
    rng = np.random.default_rng(6)
    eras = [era(rng, n, 4) for n in (50, 80, 65)]
    keys = np.repeat([3, 5, 9], [50, 80, 65])
    p, y, neutralizers = (np.concatenate([e[i] for e in eras]) for i in range(3))
    got = score.per_era(score.ncorr, keys, p, y, neutralizers)
    np.testing.assert_array_equal(got, [score.ncorr(*e) for e in eras])
    np.testing.assert_array_equal(score.per_era(score.corr, keys, p, y), [score.corr(e[0], e[1]) for e in eras])


def test_bad_inputs_raise() -> None:
    p, y, neutralizers = era(np.random.default_rng(7), 40, 2)
    with pytest.raises(ValueError, match="targets must be one value per row"):
        score.ncorr(p, y[:-1], neutralizers)
    with pytest.raises(ValueError, match="neutralizers must be"):
        score.ncorr(p, y, neutralizers[:-1])
    with pytest.raises(ValueError, match="neutralizers must be"):
        score.ncorr(p, y, neutralizers[:, 0])
    missing = p.copy()
    missing[3] = np.nan
    with pytest.raises(ValueError, match="predictions has a value that is not finite"):
        score.corr(missing, y)
    bad_neutralizers = neutralizers.copy()
    bad_neutralizers[0, 0] = np.inf
    with pytest.raises(ValueError, match="neutralizers has a value that is not finite"):
        score.ncorr(p, y, bad_neutralizers)
    with pytest.raises(ValueError, match="at least 2 rows"):
        score.corr(p[:1], y[:1])
    with pytest.raises(ValueError, match="at least one column"):
        score.per_era(score.corr, np.zeros(3))
    with pytest.raises(ValueError, match="the same rows"):
        score.per_era(score.corr, np.zeros(40), p, y[:-1])
    with pytest.raises(ValueError, match="era must be sorted"):
        score.per_era(score.corr, np.arange(40)[::-1], p, y)
