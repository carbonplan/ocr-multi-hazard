import calendar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocr.risks.hurricane import batch, forcing, tcgi


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
