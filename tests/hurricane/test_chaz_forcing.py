import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocr.risks.hurricane import batch, chaz, forcing, tcgi


def profile(levels, t_k=290.0):
    return xr.DataArray(np.full(len(levels), t_k), dims='level', coords={'level': levels})


def test_saturation_column_uses_the_sst_below_the_last_level_and_needs_150_hpa():
    levels = [100.0, 150.0, 500.0, 925.0, 1000.0]
    msl, sst = xr.DataArray(101000.0), xr.DataArray(300.0)
    w = tcgi.saturation_column(profile(levels), msl, sst)
    warmer = tcgi.saturation_column(profile(levels), msl, xr.DataArray(302.0))
    assert 0 < float(w) < float(warmer)
    # the 1000 hPa level does not enter: the last layer runs from 925 hPa to sea level
    t = profile(levels)
    t[-1] = 200.0
    assert float(tcgi.saturation_column(t, msl, sst)) == pytest.approx(float(w))
    assert np.isnan(float(tcgi.saturation_column(profile(levels), msl, xr.DataArray(np.nan))))
    with pytest.raises(ValueError):
        tcgi.saturation_column(profile([100.0, 200.0, 1000.0]), msl, sst)


def temperature_for(qs, p_pa, kelvin=tcgi.KELVIN_CHAZ):
    """The temperature (K) at which saturation_humidity is `qs` at `p_pa`: Bolton inverted."""
    es = qs * p_pa / (tcgi.EPS + qs * (1 - tcgi.EPS))
    a = np.log(es / 611.2)
    return 243.5 * a / (17.67 - a) + kelvin


def arithmetic_case():
    levels = [100.0, 125.0, 150.0, 500.0, 925.0, 1000.0]
    qs = {150.0: 0.001, 500.0: 0.004, 925.0: 0.008}
    t = [temperature_for(qs[p], p * 100) if p in qs else 250.0 for p in levels]
    t = xr.DataArray(t, dims='level', coords={'level': levels})
    return t, xr.DataArray(101000.0), xr.DataArray(temperature_for(0.016, 101000.0))


def test_saturation_column_layer_arithmetic():
    # half of 0.001 over 125-150 hPa, log-linear layers 150-500 and 500-925 hPa, and 925 hPa to
    # sea level at the SST's 0.016; 1000 hPa does not enter
    t, msl, sst = arithmetic_case()
    assert float(tcgi.saturation_column(t, msl, sst)) == pytest.approx(43.254916326877336)
    # an interior missing level drops its two layers; a missing 150 hPa or SST does not
    gap = t.copy()
    gap.loc[500.0] = np.nan
    half = 0.5 * 0.001 * 2500 / tcgi.G_W
    bottom = 0.016 * (1 - 0.25 - 0.25**2 / 3 - 0.125 / 24) * 8500 / tcgi.G_W
    assert float(tcgi.saturation_column(gap, msl, sst)) == pytest.approx(half + bottom)
    top = t.copy()
    top.loc[150.0] = np.nan
    assert np.isnan(float(tcgi.saturation_column(top, msl, sst)))
    # missing sea-level pressure is taken as 1000 hPa
    at_1000 = float(tcgi.saturation_column(t, xr.DataArray(100000.0), sst))
    assert float(tcgi.saturation_column(t, xr.DataArray(np.nan), sst)) == pytest.approx(at_1000)


def test_saturation_column_with_the_chaz_offset_is_about_one_percent_drier():
    levels = [100.0, 150.0, 200.0, 500.0, 850.0, 925.0, 1000.0]
    msl, sst = xr.DataArray(101000.0), xr.DataArray(301.0)
    chaz_w = float(tcgi.saturation_column(profile(levels, 280.0), msl, sst))
    right = float(tcgi.saturation_column(profile(levels, 280.0), msl, sst, kelvin=tcgi.KELVIN))
    assert 0.985 < chaz_w / right < 0.995


def test_era5_2017_moisture_reproduces_the_tutorial_tcgi(monkeypatch):
    from ocr.risks.hurricane import store

    needed = [
        store.CACHE / store.ERA5 / '2017' / f
        for f in ('pl_tq_1deg.nc', 'pl_wind_025.nc', 'sl_025.nc')
    ]
    needed += [store.CACHE / store.TUTORIAL / f'TCGI_{k}_PI_2017.nc' for k in ('CRH', 'SD')]
    needed += [store.CACHE / store.TUTORIAL / 'PI_2017.nc']
    if not all(f.exists() for f in needed):
        pytest.skip('needs the 2017 ERA5 and tutorial inputs in the local cache')
    monkeypatch.setattr(store, 'OFFLINE', True)
    grid = tcgi.tutorial_grid()
    ours = tcgi._months(tcgi.era5_inputs(2017, grid), 'time')
    ocean = ours['ocean']
    for kind, name in (('CRH', 'TCGI_CRH_PI_2017.nc'), ('SD', 'TCGI_SD_PI_2017.nc')):
        theirs = tcgi._months(store.tutorial(name), 'time')['TCGI'].where(ocean)
        got = tcgi.tcgi(ours['avor'], ours[kind], ours['PI'], ours['vws'], kind).where(ocean)
        assert float(got.sum() / theirs.sum()) == pytest.approx(1.0, abs=2e-4)


def field(value, lat=0.0):
    return xr.DataArray([[value]], dims=('lat', 'lon'), coords={'lat': [lat], 'lon': [0.0]})


def test_tcgi_caps_vorticity_and_scales_with_cell_area():
    args = (field(40.0), field(60.0), field(10.0))  # CRH, PI, shear
    capped = tcgi.tcgi(field(3.7e-5), args[0], args[1], args[2], 'CRH')
    beyond = tcgi.tcgi(field(9e-5), args[0], args[1], args[2], 'CRH')
    assert beyond.item() == pytest.approx(capped.item())
    one_degree = tcgi.tcgi(field(1e-5), *args, 'CRH', cell_deg=1.0)
    two_degree = tcgi.tcgi(field(1e-5), *args, 'CRH', cell_deg=2.0)
    assert two_degree.item() == pytest.approx(4 * one_degree.item())


def test_tcgi_area_follows_latitude():
    args = (field(1e-5, 60.0), field(40.0, 60.0), field(60.0, 60.0), field(10.0, 60.0))
    equator = (field(1e-5), field(40.0), field(60.0), field(10.0))
    ratio = tcgi.tcgi(*args, 'SD').item() / tcgi.tcgi(*equator, 'SD').item()
    assert ratio == pytest.approx(np.cos(np.deg2rad(60.0)))


def test_seeds_round_each_month():
    monthly = np.array([[0.4, 0.4, 0.4, 1.6] + [0.0] * 8])
    assert tcgi.seeds(monthly, 1.0).tolist() == [2.0]  # 0 + 0 + 0 + 2, not round(2.8)


def test_a_matrices_reconstruct_the_covariance():
    lat, lon = forcing.LAT[:3], forcing.LON[:2]
    full = np.array(
        [[4.0, 1.0, 0.5, 0.2], [1.0, 3.0, 0.3, 0.1], [0.5, 0.3, 2.0, 0.4], [0.2, 0.1, 0.4, 1.0]]
    )
    upper = list(zip(*np.triu_indices(4), strict=True))
    data = {
        name: (('month', 'lat', 'lon'), np.full((12, lat.size, lon.size), full[i, j]))
        for name, (i, j) in zip(forcing.COV_PAIRS, upper, strict=True)
    }
    cov = xr.Dataset(data, coords={'month': np.arange(1, 13), 'lat': lat, 'lon': lon})
    a = forcing.a_matrices(2018, cov)
    assert set(a) == set(range(1, 13))
    assert a[2].shape == (10, calendar.monthrange(2018, 2)[1], lat.size, lon.size)
    lower = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)]
    chol = np.zeros((4, 4))
    for k, (i, j) in enumerate(lower):
        chol[i, j] = a[7][k, 10, 1, 1]
    np.testing.assert_allclose(chol @ chol.T, full, rtol=1e-6)


def test_a_matrices_refuse_a_covariance_with_gaps():
    lat, lon = forcing.LAT[:3], forcing.LON[:2]
    data = {
        name: (
            ('month', 'lat', 'lon'),
            np.full((12, lat.size, lon.size), 1.0 if i in (0, 4, 7, 9) else 0.0),
        )
        for i, name in enumerate(forcing.COV_PAIRS)
    }
    data['u850pu850p'][1][5, 1, 1] = np.nan
    cov = xr.Dataset(data, coords={'month': np.arange(1, 13), 'lat': lat, 'lon': lon})
    with pytest.raises(ValueError, match='missing values'):
        forcing.a_matrices(2018, cov)


@pytest.mark.parametrize(
    ('day', 'expected'),
    [
        (datetime(2020, 1, 10), 0.0),
        (datetime(2020, 2, 15), 1.0),
        (datetime(2020, 3, 1), 1.0 + 15 / 29),  # 15 Feb to 15 Mar is 29 days in a leap year
        (datetime(2021, 3, 1), 1.0 + 14 / 28),
        (datetime(2020, 12, 20), 11.0),
    ],
)
def test_monthly_fields_interpolate_between_mid_months(day, expected):
    field = np.arange(12.0).reshape(12, 1, 1)
    assert float(chaz.date_interpolation(day, field)[0, 0]) == pytest.approx(expected)


def test_ocean_needs_sst_at_the_node_and_its_north_and_west_neighbours():
    lat, lon = np.arange(4.0, -1.0, -1.0), np.arange(5.0)
    sst = xr.DataArray(np.ones((5, 5)), dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
    sst[2, 2] = np.nan  # 2N 2E
    grid = xr.Dataset(coords={'lat': lat[1:], 'lon': lon})
    ok = tcgi.ocean(sst, grid)
    missing = {(float(a), float(b)) for a, b in zip(*np.nonzero(~ok.values), strict=True)}
    missing = {(float(ok['lat'][int(i)]), float(ok['lon'][int(j)])) for i, j in missing}
    assert missing == {(2.0, 2.0), (1.0, 2.0), (2.0, 3.0), (1.0, 3.0)}
    with pytest.raises(ValueError, match='north to south'):
        tcgi.ocean(sst.sortby('lat'), grid)


def test_covariance_of_daily_anomalies():
    year, lat, lon = 2018, forcing.LAT[:1], forcing.LON[:1]
    time = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
    rng = np.random.default_rng(0)
    u = rng.normal(size=(time.size, 1, 1))
    daily = xr.Dataset(
        {
            name: (('time', 'lat', 'lon'), u if name == 'u850' else np.zeros_like(u))
            for name in ('u250', 'v250', 'u850', 'v850')
        },
        coords={'time': time, 'lat': lat, 'lon': lon},
    )
    monthly = daily.resample(time='MS').mean()
    cov = forcing.covariance(year, daily, monthly)
    assert float(cov['u250pu250p'].max()) == 0.0
    assert float(cov['u850pu850p'].mean()) == pytest.approx(1.0, rel=0.25)


def test_parse_years():
    assert batch.parse_years('1981-1983,2019') == [1981, 1982, 1983, 2019]


def daily_files(monkeypatch, start: str, end: str):
    """forcing._daily_winds reading synthetic 6-hourly winds from start to end."""
    time = pd.date_range(start, end, freq='6h')
    shape = (time.size, 2, 1, 1)
    ds = xr.Dataset(
        {c: (('time', 'level', 'lat', 'lon'), np.ones(shape)) for c in ('u', 'v')},
        coords={'time': time, 'level': [250, 850], 'lat': forcing.LAT[:1], 'lon': forcing.LON[:1]},
    )
    months = {m: ds.sel(time=ds['time.month'] == m) for m in range(1, 13)}
    monkeypatch.setattr(forcing, 'LAT', forcing.LAT[:1])
    monkeypatch.setattr(forcing, 'LON', forcing.LON[:1])
    monkeypatch.setattr(forcing.store, 'path', lambda key: key)
    monkeypatch.setattr(forcing.store, 'open_era5', lambda key: months[int(key[-5:-3])])


def test_daily_winds_refuse_a_shifted_calendar(monkeypatch):
    daily_files(monkeypatch, '2018-01-02', '2019-01-01 18:00')
    with pytest.raises(ValueError, match='every day'):
        forcing._daily_winds(2018)


def test_daily_winds_accept_the_whole_year(monkeypatch):
    daily_files(monkeypatch, '2018-01-01', '2018-12-31 18:00')
    assert forcing._daily_winds(2018).sizes['time'] == 365
