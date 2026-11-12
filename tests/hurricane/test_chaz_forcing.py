import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocr.risks.hurricane import batch, chaz, forcing, tcgi


def column(values, levels):
    return xr.DataArray(np.asarray(values, float), dims='level', coords={'level': levels})


@pytest.mark.parametrize(
    ('surface', 'pascals'),
    [(950.0, 85000.0), (1013.0, 91300.0), (400.0, 30000.0)],
)
def test_column_to_surface_integrates_to_the_surface(surface, pascals):
    x = column([1, 1, 1], [100.0, 500.0, 1000.0])
    got = tcgi.column_to_surface(x, xr.DataArray(surface), 100) * tcgi.G
    assert float(got) == pytest.approx(pascals)


def test_column_to_surface_skips_missing_levels_below_ground():
    x = column([1, 1, np.nan], [100.0, 500.0, 1000.0])
    got = tcgi.column_to_surface(x, xr.DataArray(450.0), 100) * tcgi.G
    assert float(got) == pytest.approx(35000.0)


def test_column_to_surface_propagates_missing_values_that_contribute():
    x = column([1, np.nan, 1], [100.0, 500.0, 1000.0])
    assert np.isnan(float(tcgi.column_to_surface(x, xr.DataArray(950.0), 100)))


def test_column_to_surface_rejects_a_top_that_is_not_a_level():
    with pytest.raises(ValueError):
        tcgi.column_to_surface(column([1, 1, 1], [100.0, 500.0, 1000.0]), xr.DataArray(950.0), 150)


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
