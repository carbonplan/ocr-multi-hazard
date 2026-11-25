from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from ocr.risks.hurricane import chaz, store

NEIGHBORS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def test_fill_missing_fills_from_neighbours():
    x = np.ones((2, 4, 5))
    x[1, 2, 3] = np.nan
    got = chaz.fill_missing(x, NEIGHBORS)
    assert np.isfinite(got).all()
    assert got[1, 2, 3] == 1.0


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_fill_missing_refuses_a_month_missing_everywhere(bad):
    x = np.ones((2, 4, 5))
    x[0] = bad
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        chaz.fill_missing(x, NEIGHBORS)


def test_fill_missing_fills_infinities_and_prefers_the_first_neighbour():
    x = np.arange(20.0).reshape(1, 4, 5)
    x[0, 2, 2] = np.inf
    assert chaz.fill_missing(x, NEIGHBORS)[0, 2, 2] == x[0, 1, 2]  # fillinNaN's (0, 1): row before


@pytest.fixture
def no_beta(monkeypatch):
    monkeypatch.setattr(chaz.gv, 'uBeta', 0.0)
    monkeypatch.setattr(chaz.gv, 'vBeta', 0.0)


def test_track_moves_with_the_steering_wind_in_metres(no_beta):
    t0 = datetime(2000, 8, 1)
    lon, lat, t = chaz.getTrackPrediction(10.0, 0.0, 10.0, 0.0, 3600.0, 300.0, 20.0, t0)
    metres_per_degree = 2 * np.pi * 6371000 * np.cos(np.radians(20.0)) / 360
    assert lon - 300.0 == pytest.approx(36000 / metres_per_degree)
    assert lat == pytest.approx(20.0)
    assert t == t0 + timedelta(hours=1)


@pytest.mark.parametrize('lat0', [20.0, -20.0])
def test_beta_drift_is_westward_and_poleward(lat0):
    lon, lat, _ = chaz.getTrackPrediction(
        0.0, 0.0, 0.0, 0.0, 3600.0, 300.0, lat0, datetime(2000, 1, 1)
    )
    assert lon < 300.0
    assert abs(lat) > abs(lat0)


def test_to_dataset_keeps_storms_and_members_by_the_notebook_rules():
    n_t, n_m = 30, 2
    # 0 develops; 1 never reaches 35 kt; 2 seeded at 1N; 3 member 0 blows up; 4 seeded at 60N
    lon0 = np.array([300.0, 300.0, 300.0, 300.0, 300.0])
    lat0 = np.array([20.0, 20.0, 1.0, 20.0, 60.0])
    wind = np.full((n_t, 5, n_m), 50.0)
    wind[:, 1] = 20.0
    wind[10, 3, 0] = 400.0
    t = np.array([[datetime(2000, 8, 1) + timedelta(hours=6 * i)] * 5 for i in range(n_t)])
    bt1 = SimpleNamespace(
        stochastic=wind,
        StormLon=np.tile(lon0, (n_t, 1)),
        StormLat=np.tile(lat0, (n_t, 1)),
        Time=t,
        StormYear=np.full(5, 2000),
    )
    ds = chaz.to_dataset(bt1)
    assert ds.sizes == {'ensembleNum': n_m, 'lifelength': n_t, 'stormID': 3}
    assert ds['latitude'].values[0].tolist() == [20.0, 20.0, 20.0]
    w = ds['Mwspd'].values  # member, time, storm
    assert np.isfinite(w[:, :, 0]).all()
    assert np.isnan(w[:, :, 1]).all()
    assert np.isnan(w[0, :, 2]).all() and np.isfinite(w[1, :, 2]).all()


@pytest.fixture
def tutorial_landmask(monkeypatch):
    monkeypatch.setattr(store, 'OFFLINE', True)
    if not (store.CACHE / store.TUTORIAL / 'landmask.nc').exists():
        pytest.skip('needs the CHAZ tutorial files in the local cache')
    monkeypatch.setattr(chaz.gv, 'ldmask', None)
    chaz._load_landmask()


@pytest.mark.parametrize(
    ('lat', 'lon', 'land'),
    [(39, 262, True), (23, 10, True), (29, 320, False), (-21, 134, True), (-31, 260, False)],
)
def test_land_mask_rows_run_south_to_north_on_the_2_degree_grid(tutorial_landmask, lat, lon, land):
    # Kansas, the Sahara, the Atlantic, central Australia, the South Pacific
    assert bool(chaz.gv.ldmask[(lat + 89) // 2, lon // 2] > 0) is land


def test_a_seeded_realization_repeats_exactly(monkeypatch, tmp_path):
    import xarray as xr

    monkeypatch.setattr(store, 'OFFLINE', True)
    if not (store.CACHE / store.FORCING / '2017' / 'A_201712.nc').exists():
        pytest.skip('needs the 2017 forcing in the local cache')
    runs = []
    for k in range(2):
        out = tmp_path / str(k)
        out.mkdir()
        monkeypatch.setattr(chaz, 'run_dir', lambda name, out=out: out)
        t = chaz.run_one((2017, 0, 0.15, 2, 'TCGI_CRH_PI', {}, chaz.ERA5))
        assert t['storms_kept'] > 0
        runs.append(xr.open_dataset(out / '2017_r000.nc').load())
    a, b = runs
    for v in a.data_vars:
        np.testing.assert_array_equal(a[v].values, b[v].values)
    assert a.attrs['random_seed'] == 2017000 and a.attrs['seeding_ratio'] == 0.15


def test_diagonal_steering_zeroes_only_the_off_diagonal_entries(monkeypatch):
    import xarray as xr

    a = xr.DataArray(np.arange(1.0, 11.0)[:, None, None, None] * np.ones((10, 2, 3, 4)))
    monkeypatch.setattr(chaz.gv, 'steering', 'lower')
    assert chaz.steering_factor(a) is a
    monkeypatch.setattr(chaz.gv, 'steering', 'diagonal')
    got = chaz.steering_factor(a)
    assert got[:, 0, 0, 0].values.tolist() == [1, 0, 3, 0, 0, 6, 0, 0, 0, 10]
    assert a[1, 0, 0, 0] == 2  # the forcing is left alone
    assert chaz.run_name('TCGI_CRH_PI', steering='diagonal') == 'TCGI_CRH_PI-steering-diagonal'
    assert chaz.run_name('TCGI_CRH_PI') == 'TCGI_CRH_PI'
    assert (
        chaz.run_name('TCGI_CRH_PI', steering='lower', land='separate')
        == 'TCGI_CRH_PI-land-separate'
    )
    with pytest.raises(ValueError):
        chaz.run_name('TCGI_CRH_PI', land='radius')


@pytest.mark.parametrize(('land', 'shared'), [('shared', True), ('separate', False)])
def test_land_mean_shares_the_centre_cell_array_only_by_default(monkeypatch, land, shared):
    from datetime import datetime

    monkeypatch.setattr(chaz.gv, 'land', land)
    t = np.array([[datetime(2000, 8, 1)], [datetime(2000, 8, 1, 6)]], dtype=object)
    bt = chaz.fst2bt(
        {
            'lon': np.array([[300.0], [299.0]]),
            'lat': np.array([[20.0], [20.5]]),
            'Time': t,
            'ldmask': np.zeros((2, 1)),
        }
    )
    assert (bt.landmask is bt.landmaskMean) is shared


def test_training_normalization_is_the_tutorial_observed_data():
    assert chaz.meanstd_for(chaz.ERA5, 'training') == f'{store.TUTORIAL}/observed_data.nc'
    with pytest.raises(ValueError):
        chaz.meanstd_for('cmip6/UKESM1-0-LL/r1i1p1f2/historical', 'training')


def _write_normalization(path, suffix, y_shape):
    import netCDF4 as nc

    with nc.Dataset(path, 'w') as f:
        f.createDimension('w', 3)
        f.createDimension('l', 4)
        f.createDimension('one', 1)
        for k, (tag, dim) in enumerate((('w', 'w'), ('l', 'l'))):
            for j, name in enumerate(('meanX', 'stdX')):
                f.createVariable(f'{name}_{tag}{suffix}', 'f8', (dim,))[:] = (
                    10 * k + j + np.arange(len(f.dimensions[dim]))
                )
            for j, name in enumerate(('meanY', 'stdY')):
                v = f.createVariable(f'{name}_{tag}{suffix}', 'f8', y_shape)
                v[:] = 100 * k + j


@pytest.mark.parametrize('suffix, y_shape', [('', ()), ('_obs', ('one',))])
def test_read_normalization_reads_both_layouts(tmp_path, suffix, y_shape):
    import netCDF4 as nc

    path = tmp_path / 'n.nc'
    _write_normalization(path, suffix, y_shape)
    with nc.Dataset(path) as f:
        mx_w, sx_w, my_w, sy_w, mx_l, sx_l, my_l, sy_l = chaz.read_normalization(f)
    np.testing.assert_array_equal(mx_w, [0, 1, 2])
    np.testing.assert_array_equal(sx_l, [11, 12, 13, 14])
    assert my_w.shape == () and float(my_w) == 0 and float(sy_w) == 1
    assert float(my_l) == 100 and float(sy_l) == 101


def test_forward_speed_runs_from_each_point_to_the_next_and_leaves_the_last_two():
    from ocr.risks.hurricane import meanstd

    lat = np.array([[0.0], [0.0], [0.0], [0.0], [np.nan]])
    lon = np.array([[100.0], [101.0], [103.0], [104.0], [np.nan]])
    days = np.arange(5.0).reshape(-1, 1) * 0.25
    got = meanstd.forward_speed(lon, lat, days)[:, 0]
    km_per_degree = 2 * np.pi * 6371.0 / 360
    np.testing.assert_allclose(got[:2], np.array([1, 2]) * km_per_degree * 1000 / 21600)
    assert np.isnan(got[2:]).all()
