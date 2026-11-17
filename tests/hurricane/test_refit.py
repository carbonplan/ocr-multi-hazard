import numpy as np
import pytest

from ocr.risks.hurricane import chaz, refit, store


def tutorial_cached() -> bool:
    return (store.CACHE / store.TUTORIAL / chaz.gv.obpath).exists()


def test_refit_on_the_tutorial_environment_recovers_the_shipped_coefficients(monkeypatch):
    monkeypatch.setattr(store, 'OFFLINE', True)
    if not tutorial_cached():
        pytest.skip('needs the tutorial inputs in the local cache')
    got, ref = refit.fit(refit.tutorial_env()), refit.shipped()
    for tag in 'wl':
        np.testing.assert_allclose(got[tag]['params'], ref[tag]['params'], atol=0.06)
    errors = got['errors']
    assert np.isfinite(errors).sum() == got['w']['n'] + got['l']['n']
    assert abs(np.nanmean(errors)) < 0.1


def test_refit_intensity_needs_fine_land(monkeypatch):
    monkeypatch.setattr(chaz.gv, 'land', 'shared')
    with pytest.raises(ValueError):
        chaz.apply_options({'intensity': 'refit'})
    chaz.apply_options({'intensity': 'refit', 'land': 'fine'})
    assert chaz.gv.intensity == 'refit'
    chaz.apply_options({})
    assert chaz.gv.intensity == 'tutorial'


def test_refit_normalization_is_only_for_era5():
    assert chaz.meanstd_for(chaz.ERA5, 'model', 'refit') == refit.key(chaz.ERA5, refit.FILES[2])
    with pytest.raises(ValueError):
        chaz.meanstd_for(chaz.ERA5, 'env', 'refit')
    with pytest.raises(ValueError):
        chaz.meanstd_for('cmip6/UKESM1-0-LL/r1i1p1f2/historical', 'model', 'refit')


def pool_file(v, pi, errors, water):
    n = v.size
    return {
        'errors': errors[None, :],
        'StormMwspd': v[None, :],
        'PIstep': pi[None, :],
        'water': water[None, :],
        'StormYear': np.full(n, 2000),
    }


def test_pi_error_pool_draws_by_regime_wind_and_pi(monkeypatch):
    rng = np.random.default_rng(0)
    n = 6000
    v = rng.uniform(20, 140, n)
    pi = rng.uniform(40, 190, n)
    water = (np.arange(n) % 2).astype(float)
    # the sign marks the environment; land errors are offset by 1000 to tell the pools apart
    errors = np.where(pi > 140, 100.0, -100.0) + v / 1000 + 1000 * (1 - water)
    monkeypatch.setattr(
        chaz, 'pool_pi', chaz.error_pool_pi(pool_file(v, pi, errors, water)), raising=False
    )
    cat1 = chaz.pool_pi[True][0]
    for v0 in (25.0, np.array([75.0]), 135.0):  # get_stochastic passes one-element arrays
        lo = chaz.find_error_pi(v0, np.float64(45.0), True)
        hi = chaz.find_error_pi(v0, 185.0, True)
        assert lo < 0 < hi < 500
        k = chaz.wind_bin(np.ravel(v0), cat1)[0]
        assert chaz.wind_bin(np.array([(lo + 100) * 1000]), cat1)[0] == k
        assert (
            500 < chaz.find_error_pi(v0, 45.0, False) < 1000 < chaz.find_error_pi(v0, 185.0, False)
        )
    assert chaz.find_error_pi(np.nan, 100.0, True) == 0


def test_pi_error_pool_with_tied_pi_falls_back_to_the_wind_bin(monkeypatch):
    n = 400
    v = np.full(n, 50.0)
    pi = np.full(n, 100.0)
    f = pool_file(v, pi, np.linspace(-5, 5, n), (np.arange(n) % 2).astype(float))
    monkeypatch.setattr(chaz, 'pool_pi', chaz.error_pool_pi(f), raising=False)
    for p in (90.0, 100.0, 110.0):
        assert -5 <= chaz.find_error_pi(50.0, p, True) <= 5


def land_step_prediction(intensity, monkeypatch):
    """The land regression's 12-hour change for a storm whose centre is over water at the start,
    land 12 hours on, and water again 18 hours on, with only the land predictor (coefficient 1)."""
    from datetime import datetime
    from types import SimpleNamespace

    monkeypatch.setattr(chaz.gv, 'intensity', intensity)
    bt = SimpleNamespace(
        Time=np.array([[datetime(2017, 9, 1)]] * 4),
        landmaskMean=np.array([[-1.0], [-1.0], [0.0], [-1.0]]),
    )
    z = np.zeros(2)
    one = np.array([0.0, 1.0])
    _, v1 = chaz.getPrediction_v0input_result(
        bt, z, 0.0, one, 1.0, 0, 0, [12], one, ['landmaskMean'], [], 50.0, 0.0
    )
    return v1[1] - 50.0


def test_refit_reads_the_land_change_over_the_step_itself(monkeypatch):
    assert land_step_prediction('refit', monkeypatch) == 1.0
    assert land_step_prediction('refit-pi', monkeypatch) == 1.0
    assert land_step_prediction('tutorial', monkeypatch) == 0.0  # the public code's t+18h
