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


def test_pi_error_pool_empty_wind_bins_take_the_nearest_populated_bin(monkeypatch):
    v = np.r_[np.linspace(20, 25, 60), np.linspace(70, 75, 60)]  # nothing between 30 and 70 kt
    errors = np.r_[np.full(60, -1.0), np.full(60, 1.0)]
    water = np.r_[np.ones(30), np.zeros(30), np.ones(30), np.zeros(30)]
    f = pool_file(v, np.full(v.size, 100.0), errors, water)
    monkeypatch.setattr(chaz, 'pool_pi', chaz.error_pool_pi(f), raising=False)
    for regime in (True, False):
        assert chaz.find_error_pi(35.0, 100.0, regime) == -1.0
        assert chaz.find_error_pi(65.0, 100.0, regime) == 1.0
    with pytest.raises(ValueError):
        chaz.error_pool_pi(pool_file(v, v, errors, np.ones(v.size)))


def test_basin_of_positions():
    lat = np.array([25.0, 15.0, 15.0, 10.0, 15.0, 15.0, 15.0, -15.0, 40.0, np.nan])
    lon = np.array([-90.0, -40.0, 340.0, -95.0, 260.0, 135.0, 88.0, 150.0, 10.0, 0.0])
    want = ['NA', 'NA', 'NA', 'EP', 'EP', 'WP', 'NI', 'SH', 'NA']
    got = chaz.basin(lat, lon)
    assert [chaz.BASINS[i] for i in got[:-1]] == want
    assert got[-1] == -1


def test_basin_error_pool_keeps_to_the_basin_and_falls_back_without_one(monkeypatch):
    rng = np.random.default_rng(1)
    n = 4000
    v = rng.uniform(20, 140, n)
    pi = rng.uniform(40, 190, n)
    water = (np.arange(n) % 4 != 0).astype(float)
    b = np.where(np.arange(n) < 3000, 0, 4)
    b[-200:] = 3  # too few NI steps for a pool of its own
    errors = np.where(b == 0, 1.0, np.where(b == 4, -1.0, 0.0))
    f = pool_file(v, pi, errors, water)
    monkeypatch.setattr(chaz, 'pool_pi', chaz.error_pool_pi(f, b[None, :]), raising=False)
    assert (3, True) not in chaz.pool_pi and (4, False) not in chaz.pool_pi
    for _ in range(20):
        assert chaz.find_error_pi(80.0, 150.0, True, 0) == 1.0
        assert chaz.find_error_pi(80.0, 150.0, True, 4) == -1.0
    draws = {chaz.find_error_pi(80.0, 150.0, True, 3) for _ in range(200)}
    assert draws <= {1.0, -1.0, 0.0} and len(draws) > 1
    assert chaz.find_error_pi(80.0, 150.0, False, 0) == 1.0
    assert len({chaz.find_error_pi(80.0, 150.0, False, 4) for _ in range(200)}) > 1


def test_basin_pool_without_wind_bins_falls_back_to_the_regime_pool(monkeypatch):
    rng = np.random.default_rng(2)
    n_wide = 3000
    ni = np.r_[15.0, np.full(550, 25.0), 80.0]  # wind_bins gives no edges for these
    v = np.r_[rng.uniform(20, 140, n_wide), ni]
    b = np.r_[np.zeros(n_wide, int), np.full(ni.size, 3)]
    water = np.ones(v.size)
    water[: n_wide // 4] = 0
    errors = np.where(b == 3, 7.0, 1.0)
    f = pool_file(v, rng.uniform(40, 190, v.size), errors, water)
    monkeypatch.setattr(chaz, 'pool_pi', chaz.error_pool_pi(f, b[None, :]), raising=False)
    assert (3, True) not in chaz.pool_pi
    assert chaz.find_error_pi(25.0, 100.0, True, 3) in (1.0, 7.0)


def test_refit_years_choose_the_files_and_need_a_refit(monkeypatch):
    assert refit.name('1981-2012') == chaz.ERA5
    assert refit.name('1981-1999') == 'era5-1981-1999'
    assert chaz.meanstd_for(chaz.ERA5, 'model', 'refit-basin', '1981-1999') == refit.key(
        'era5-1981-1999', 'coefficient_meanstd.nc'
    )
    monkeypatch.setattr(store, 'path', lambda k: k)
    chaz.apply_options({'intensity': 'refit-basin', 'land': 'fine', 'fit': '1981-1999'})
    assert str(chaz.intensity_file('errors.nc')).endswith('era5-1981-1999/errors.nc')
    with pytest.raises(ValueError):
        chaz.apply_options({'fit': '1981-1999'})
    chaz.apply_options({})
    assert chaz.run_name('TCGI_CRH_PI', land='fine', intensity='refit-basin', fit='1981-1999') == (
        'TCGI_CRH_PI-land-fine-intensity-refit-basin-fit-1981-1999'
    )
