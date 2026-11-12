import numpy as np
import pytest

from ocr.risks.hurricane import tracks


def test_steering_noise_of_independent_winds():
    a = np.zeros(10)
    a[[0, 2, 5, 9]] = 1.0  # identity covariance over (u250, v250, u850, v850)
    vu, vv, cuv = tracks.predicted(a, zero_off_diagonal=False)
    assert (vu, vv, cuv) == pytest.approx((0.2**2 + 0.8**2, 0.2**2 + 0.8**2, 0.0))


def test_zeroing_the_off_diagonals_removes_the_u_v_correlation():
    cov = np.array(
        [[4.0, 1.0, 2.0, 0.5], [1.0, 3.0, 0.4, 1.5], [2.0, 0.4, 2.0, 0.6], [0.5, 1.5, 0.6, 1.0]]
    )
    L = np.linalg.cholesky(cov)
    a = np.array([L[i, j] for i, j in tracks.LOWER])
    vu, vv, cuv = tracks.predicted(a, zero_off_diagonal=False)
    w = np.array([[0.2, 0, 0.8, 0], [0, 0.2, 0, 0.8]])
    assert (vu, vv, cuv) == pytest.approx(tuple((w @ cov @ w.T)[[0, 1, 0], [0, 1, 1]]))
    assert tracks.predicted(a, zero_off_diagonal=True)[2] == 0.0


def test_compare_removes_the_drift_and_measures_the_rest():
    rng = np.random.default_rng(0)
    n = 20000
    scale = rng.uniform(0.8, 1.0, n)
    noise = rng.multivariate_normal([0, 0], [[4.0, 1.2], [1.2, 2.25]], n)
    x = np.zeros((n, 9))
    x[:, 0] = -1.5 * scale + noise[:, 0]
    x[:, 1] = 2.0 * scale + noise[:, 1]
    x[:, 2] = scale
    x[:, 3:6] = (4.0, 2.25, 1.2)
    x[:, 6:9] = (4.0, 2.25, 0.0)
    got = tracks.compare(x)
    assert got['measured']['sd u (m/s)'] == pytest.approx(2.0, rel=0.03)
    assert got['measured']['r(u, v)'] == pytest.approx(0.4, abs=0.03)
    assert got['predicted, lower factor']['r(u, v)'] == pytest.approx(0.4)
    assert got['predicted, off-diagonals zero']['r(u, v)'] == 0.0
