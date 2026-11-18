import math

import numpy as np
import pytest

from idun.tabm import hl_gauss

SIGMA = 0.09375  # 0.75 of the bins' width


def spec(z: np.ndarray) -> np.ndarray:
    return hl_gauss.labels(z, 64, -4.0, 4.0, SIGMA)  # the defaults: 64 bins on [-4, 4]


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_edges_and_centres_are_uniform() -> None:
    assert hl_gauss.edges(64, -4.0, 4.0).tolist() == np.linspace(-4.0, 4.0, 65).tolist()
    centres = hl_gauss.centres(64, -4.0, 4.0)
    assert (centres.size, centres[0], centres[-1]) == (64, -3.9375, 3.9375)
    assert np.allclose(np.diff(centres), 0.125)
    with pytest.raises(ValueError, match="low < high"):
        hl_gauss.edges(8, 1.0, 1.0)


def test_labels_are_the_gaussian_mass_per_bin_renormalised() -> None:
    z = np.array([0.0, 0.3, -1.234, 3.99, -4.0])
    p = spec(z)
    assert p.shape == (5, 64)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert (p >= 0).all()
    e = np.linspace(-4.0, 4.0, 65)
    for row, zi in zip(p, z, strict=True):  # the formula with math.erf, independently
        cdf = np.array([phi((edge - zi) / SIGMA) for edge in e])
        np.testing.assert_allclose(row, np.diff(cdf) / (cdf[-1] - cdf[0]), rtol=1e-9, atol=1e-15)
    assert p[0, 31] == pytest.approx(p[0, 32])  # z = 0 sits on an edge: the two bins share the mass
    assert np.argmax(p[1]) == int((0.3 + 4.0) // 0.125)  # the bin holding z gets the most
    centres = hl_gauss.centres(64, -4.0, 4.0)
    assert abs(p[2] @ centres + 1.234) < 1e-3  # inside the support the expected centre is about z


def test_targets_are_clamped_to_the_support() -> None:
    p = spec(np.array([-4.0, -9.0, 4.0, 1e6]))
    np.testing.assert_array_equal(p[0], p[1])
    np.testing.assert_array_equal(p[2], p[3])
    assert p[0, 0] == pytest.approx(2 * (phi(0.125 / 0.09375) - 0.5))  # the cut-off half renormalised to 1


def test_bad_targets_raise() -> None:
    with pytest.raises(ValueError, match="finite"):
        spec(np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="sigma"):
        hl_gauss.labels(np.zeros(2), 64, -4.0, 4.0, 0.0)
