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
