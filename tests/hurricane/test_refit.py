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


def test_pi_error_pool_draws_from_the_storms_wind_bin_and_pi_tercile(monkeypatch):
    rng = np.random.default_rng(0)
    n = 3000
    v = rng.uniform(20, 140, n)
    pi = rng.uniform(40, 190, n)
    errors = np.where(pi > 140, 100.0, -100.0) + v / 1000  # the sign marks the environment
    f = {
        'errors': errors[None, :],
        'StormMwspd': v[None, :],
        'PIstep': pi[None, :],
        'StormYear': np.full(n, 2000),
    }
    monkeypatch.setattr(chaz, 'pool_pi', chaz.error_pool_pi(f), raising=False)
    cat1 = chaz.pool_pi[0]
    for v0 in (25.0, 75.0, 135.0):
        lo, hi = chaz.find_error_pi(v0, 45.0), chaz.find_error_pi(v0, 185.0)
        assert lo < 0 < hi
        k = chaz.wind_bin(np.array([v0]), cat1)[0]
        assert chaz.wind_bin(np.array([(lo + 100) * 1000]), cat1)[0] == k
    assert chaz.find_error_pi(np.nan, 100.0) == 0
