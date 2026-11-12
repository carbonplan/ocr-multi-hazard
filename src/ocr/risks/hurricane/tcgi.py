"""Tropical cyclone genesis index (TCGI) for CHAZ, from monthly-mean ERA5 or GCM fields.

CHAZ seeds storm precursors at a rate set by TCGI (Tippett et al. 2011; Camargo et al. 2014),
in the ERA5 refit of Fosu et al. (2024, doi:10.63024/dpva-2pa1):

    TCGI = exp(b + b_eta * min(|eta_850| * 1e5, 3.7) + b_mv * MV + b_pi * PI + b_vws * VWS)
           * cos(lat) * cell_area_in_square_degrees

eta_850 is the 850 hPa absolute vorticity (s-1), MV the moisture variable (column relative
humidity CRH in %, or column saturation deficit SD in kg m-2, negative), PI the Bister-Emanuel
potential intensity (m s-1) and VWS the 250-850 hPa vector wind difference (m s-1). The rate is
per month and grid cell. Fosu's b is for 1 x 1 degree boxes; the area factor scales it to the grid.
Fosu et al. print the shear as 200-850 hPa; the tutorial's TCGI, Camargo et al. (2025,
supplement) and Lee et al. (2020, 2023) use 250-850 hPa, and so do we.

The CHAZ tutorial data (Zenodo 11631930) carry the TCGI for ERA5 2017 with its inputs. The
commands below build what the review notebook docs/methods/hurricane-risk/genesis-index.ipynb
reads (files under store.S3_ROOT):

    compare           our inputs and TCGI next to the tutorial's 2017 ones, on its grid
    seeding           our TCGI per month, 1981-2005 and 2017, next to the storms per year in
                      the published ERA5 track files
    fetch-validation  the 6-hourly August 2017 temperatures the notebook tests W* against

ERA5 itself comes from forcing.py fetch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ocr.risks.hurricane import store

# Fosu et al. (2024) section 2.1 prints these to two or three decimals. The digits here are
# recovered from the tutorial's 2017 TCGI (an exact log-linear fit on its own inputs); the printed
# ones move the 2017 global total by -1.4% (CRH) and +2.5% (SD).
COEFFICIENTS = {
    'CRH': {'b': -24.1323, 'eta': 2.512, 'mv': 0.077, 'pi': 0.0622, 'vws': -0.1202},
    'SD': {'b': -18.3563, 'eta': 2.4829, 'mv': 0.0735, 'pi': 0.0798, 'vws': -0.1346},
}
PUBLISHED = {
    'CRH': {'b': -24.132, 'eta': 2.512, 'mv': 0.077, 'pi': 0.062, 'vws': -0.120},
    'SD': {'b': -18.356, 'eta': 2.483, 'mv': 0.073, 'pi': 0.080, 'vws': -0.135},
}
ETA_CAP = 3.7  # 1e-5 s-1
SHEAR_TOP = 250  # hPa
G = 9.80665  # m s-2

# tcpyPI settings of the CHAZ pre-processing tutorial (Tutorial_chaz_pre.ipynb)
PI_KWARGS = {
    'CKCD': 0.9,
    'ascent_flag': 0,
    'diss_flag': 1.0,
    'V_reduc': 0.8,
    'ptop': 30,
    'miss_handle': 0,
}

ERA5_LEVELS = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]


def tcgi(
    eta: xr.DataArray,
    mv: xr.DataArray,
    pi: xr.DataArray,
    vws: xr.DataArray,
    kind: str,
    cell_deg: float = 2.0,
    coefficients: dict = COEFFICIENTS,
) -> xr.DataArray:
    """Genesis rate per month and grid cell; `kind` is 'CRH' or 'SD' and names `mv`."""
    c = coefficients[kind]
    eta_c = np.minimum(np.abs(eta) * 1e5, ETA_CAP)
    rate = np.exp(c['b'] + c['eta'] * eta_c + c['mv'] * mv + c['pi'] * pi + c['vws'] * vws)
    area = np.cos(np.deg2rad(eta['lat'])) * cell_deg**2
    return (rate * area).rename(f'TCGI_{kind}')


def absolute_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Absolute vorticity (s-1) of monthly-mean winds on their native lat-lon grid, by MetPy's
    finite differences (WGS84 grid spacing). It matches the tutorial's 2017 field to 5e-8 s-1 rms;
    ERA5's spectral vorticity differs by 2e-6, enough to move TCGI by tens of percent near the
    equator, where f is small."""
    import metpy.calc

    uv = xr.Dataset({'u': u, 'v': v}).rename(lat='latitude', lon='longitude')
    for c in ('u', 'v'):
        uv[c].attrs['units'] = 'm/s'
    av = metpy.calc.absolute_vorticity(uv['u'].metpy.quantify(), uv['v'].metpy.quantify())
    return av.metpy.dequantify().rename(latitude='lat', longitude='lon').rename('avor')


def wind_shear(u200, v200, u850, v850) -> xr.DataArray:
    return np.hypot(u200 - u850, v200 - v850)


def saturation_vapour_pressure(t: xr.DataArray) -> xr.DataArray:
    """Over liquid water, hPa (Bolton 1980, as in tcpyPI)."""
    tc = t - 273.15
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def saturation_humidity(t: xr.DataArray, p_hpa: xr.DataArray, dilute: bool = True) -> xr.DataArray:
    """Saturation specific humidity, kg kg-1. The dilute form 0.622 e_s / p, without the
    0.378 e_s term, reproduces the tutorial's 2017 W* better than the exact form."""
    es = saturation_vapour_pressure(t)
    return 0.622 * es / (p_hpa if dilute else p_hpa - 0.378 * es)


def column_to_surface(x: xr.DataArray, ps_hpa: xr.DataArray, top_hpa: float) -> xr.DataArray:
    """Integral of x dp / g (kg m-2 per unit of x) from `top_hpa` to the surface: x piecewise
    linear in pressure between levels, cut at the surface pressure, and held at its lowest-level
    value between the lowest level and a higher surface pressure. `top_hpa` must be a level.
    Layers wholly below ground are skipped, so they may be NaN; a missing value in any layer
    that contributes makes the column NaN."""
    if top_hpa not in x['level'].values:
        raise ValueError(f'top_hpa={top_hpa} is not one of the levels')
    x = x.sortby('level').sel(level=slice(top_hpa, None))
    p = x['level'].values * 100.0
    ps = ps_hpa * 100.0
    xl, xh = x.isel(level=slice(None, -1)), x.isel(level=slice(1, None))
    lo = xr.DataArray(p[:-1], dims='level', coords={'level': xl['level']})
    hi = xr.DataArray(p[1:], dims='level', coords={'level': xl['level']})
    xh = xh.assign_coords(level=xl['level'])
    cut = np.minimum(np.maximum(ps, lo), hi)
    x_cut = xl + (xh - xl) * (cut - lo) / (hi - lo)
    width = cut - lo
    layers = xr.where(width > 0, (xl + x_cut) / 2 * width, 0.0).sum('level', skipna=False)
    below = xr.where(ps > p[-1], (ps - p[-1]) * x.isel(level=-1, drop=True), 0.0)
    return (layers + below) / G


def moisture(t: xr.DataArray, w: xr.DataArray, p_bottom_hpa: xr.DataArray, top_hpa: float = 100.0):
    """Column relative humidity (%) and saturation deficit (kg m-2, negative), Bretherton et al.
    (2004): W is the model's column water vapour (ERA5 tcwv, CMIP6 prw), which the tutorial's
    W = W* CRH matches to 3e-4 kg m-2; W* integrates the saturation humidity of the monthly-mean
    temperature profile."""
    ws = column_to_surface(saturation_humidity(t, t['level']), p_bottom_hpa, top_hpa)
    return (100 * w / ws).rename('CRH'), (w - ws).rename('SD')


def potential_intensity(sst_c, msl_hpa, t_c, q_gkg) -> xr.DataArray:
    """tcpyPI vmax (m s-1) with the tutorial's settings; NaN (land, no SST) set to 0 as the tutorial does."""
    from tcpyPI import pi

    t_c = t_c.sortby('level', ascending=False)
    q_gkg = q_gkg.sortby('level', ascending=False)
    vmax, *_ = xr.apply_ufunc(
        pi,
        sst_c,
        msl_hpa,
        t_c['level'],
        t_c,
        q_gkg,
        kwargs=PI_KWARGS,
        input_core_dims=[[], [], ['level'], ['level'], ['level']],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
    )
    return vmax.fillna(0).rename('PI')


# --- ERA5 ---------------------------------------------------------------------------------------


def fetch_validation() -> None:
    """The 6-hourly August 2017 temperatures the notebook averages W* over, in four requests to
    stay under the CDS size limit."""
    import cdsapi

    days = {'a': range(1, 9), 'b': range(9, 17), 'c': range(17, 25), 'd': range(25, 32)}
    for k, ds in days.items():
        dest = store.CACHE / f'{store.VALIDATION}/era5-201708-6h/{k}.nc'
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + '.part')
        cdsapi.Client(quiet=True).retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': ['reanalysis'],
                'variable': ['temperature', 'specific_humidity'],
                'year': ['2017'],
                'month': ['08'],
                'day': [f'{d:02d}' for d in ds],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'pressure_level': [str(p) for p in ERA5_LEVELS if p >= 100],
                'area': [89, 0, -89, 358],
                'grid': ['2.0', '2.0'],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
            },
            str(part),
        )
        part.rename(dest)


def ocean(sst: xr.DataArray, grid: xr.Dataset) -> xr.DataArray:
    """Cells with TCGI: SST defined at the node and its N, W and NW neighbours on the native grid,
    the support of the tutorial's bilinear regrid (99.99% of its 2017 cells agree). N is the
    previous row, as ERA5 orders latitude north to south."""
    if not (sst['lat'].diff('lat') < 0).all():
        raise ValueError('ocean() needs latitudes in ERA5 order, north to south')
    ok = sst.notnull()
    ok = (
        ok
        & ok.shift(lat=1, fill_value=False)
        & ok.roll(lon=1)
        & ok.shift(lat=1, fill_value=False).roll(lon=1)
    )
    return ok.sel(lat=grid['lat'].astype(float), lon=grid['lon'].astype(float))


def tutorial_grid() -> xr.Dataset:
    return store.tutorial('TCGI_CRH_PI_2017.nc')


def era5_inputs(year: int, grid: xr.Dataset) -> xr.Dataset:
    """Our TCGI inputs on the tutorial's grid. Its 2-degree centres (odd latitudes, even
    longitudes) are nodes of the ERA5 grids, so a bilinear regrid there is a point sample."""
    at = {'lat': grid['lat'].astype(float), 'lon': grid['lon'].astype(float)}
    wind_native = store.open_era5(store.path(f'{store.ERA5}/{year}/pl_wind_025.nc'))
    wind = wind_native.sel(at)
    sl_native = store.open_era5(store.path(f'{store.ERA5}/{year}/sl_025.nc'))
    sl = sl_native.sel(at)
    tq = store.open_era5(store.path(f'{store.ERA5}/{year}/pl_tq_1deg.nc')).sel(at)

    out = xr.Dataset()
    w850 = wind_native.sel(level=850)
    out['avor'] = absolute_vorticity(w850['u'], w850['v']).sel(at)
    out['vws'] = wind_shear(
        *(wind[c].sel(level=lev) for lev in (SHEAR_TOP, 850) for c in ('u', 'v'))
    )
    # W* down to mean sea-level pressure, not surface pressure: half the rms against the tutorial's W*
    out['CRH'], out['SD'] = moisture(tq['t'], sl['tcwv'], sl['msl'] / 100)
    out['PI'] = potential_intensity(
        sl['sst'] - 273.15, sl['msl'] / 100, tq['t'] - 273.15, tq['q'] * 1000
    )
    out['ocean'] = ocean(sl_native['sst'], grid)
    return out.drop_vars('level', errors='ignore').transpose('time', 'lat', 'lon')


REFERENCE_YEAR = 2017  # the only year the tutorial publishes


def _months(ds: xr.Dataset, dim: str) -> xr.Dataset:
    """Label the time axis by calendar month after checking it is Jan..Dec of REFERENCE_YEAR.
    The tutorial's files count days from 1 January."""
    t = ds[dim].values
    if np.issubdtype(t.dtype, np.datetime64):
        got = pd.DatetimeIndex(t)
        assert (got.year == REFERENCE_YEAR).all() and (got.month == np.arange(1, 13)).all(), got
    else:
        starts = pd.date_range(f'{REFERENCE_YEAR}-01-01', periods=12, freq='MS').dayofyear - 1
        assert (np.asarray(t) == starts).all(), t
    return ds.rename({dim: 'month'}).assign_coords(month=np.arange(1, 13))


def compare(out: Path = store.CACHE / store.VALIDATION / f'compare_{REFERENCE_YEAR}.nc') -> Path:
    grid = tutorial_grid()
    ours = _months(era5_inputs(REFERENCE_YEAR, grid), 'time')
    theirs = _months(grid, 'time')
    pi_theirs = _months(store.tutorial(f'PI_{REFERENCE_YEAR}.nc'), 'time')
    sd_theirs = _months(store.tutorial(f'TCGI_SD_PI_{REFERENCE_YEAR}.nc'), 'time')
    for other in (pi_theirs, sd_theirs):
        assert (other['lat'] == theirs['lat']).all() and (other['lon'] == theirs['lon']).all()
    ds = xr.Dataset()
    for v in ('avor', 'vws', 'CRH', 'SD'):
        ds[f'{v}_theirs'] = theirs[v]
        ds[f'{v}_ours'] = ours[v]
    ds['PI_theirs'] = pi_theirs['vmax']
    ds['PI_ours'] = ours['PI']
    ds['TCGI_CRH_theirs'] = theirs['TCGI']
    ds['TCGI_SD_theirs'] = sd_theirs['TCGI']
    ds['ocean'] = ours['ocean']
    for kind in ('CRH', 'SD'):
        their_inputs = (ds['avor_theirs'], ds[f'{kind}_theirs'], ds['PI_theirs'], ds['vws_theirs'])
        # their inputs through the formula: checks the formula alone
        ds[f'TCGI_{kind}_formula'] = tcgi(*their_inputs, kind)
        ds[f'TCGI_{kind}_published'] = tcgi(*their_inputs, kind, coefficients=PUBLISHED)
        # our inputs and mask: checks the whole chain
        ours_tcgi = tcgi(ds['avor_ours'], ds[f'{kind}_ours'], ds['PI_ours'], ds['vws_ours'], kind)
        ds[f'TCGI_{kind}_ours'] = ours_tcgi.where(ds['ocean'])
    ds['lat'] = ds['lat'].astype(float)
    ds['lon'] = ds['lon'].astype(float)
    ds.attrs = {
        'year': REFERENCE_YEAR,
        'source': 'ERA5 monthly means (CDS) vs CHAZ tutorial, Zenodo 11631930',
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


# CHAZ seeds rint(ratio * global TCGI) precursors a month (CHAZvBeta TCgiSeeding); get_seeding_ratio
# sets ratio from observed storms 1981-2005 over a survival rate of 0.78.
SEEDING_YEARS = list(range(1981, 2006))
SURVIVAL = 0.78
PUBLISHED_TRACKS = store.PUBLISHED_TRACKS


def seeds(monthly_tcgi: np.ndarray, ratio: float) -> np.ndarray:
    """Seeds per year from (year, month) global TCGI totals, by CHAZ's monthly rounding."""
    return np.rint(ratio * monthly_tcgi).sum(axis=-1)


def published_counts(years: list[int]) -> pd.DataFrame:
    """Storms per year in each of the ten published ERA5 track files (Meiler, shared Sept 2026).
    These are the storms CHAZ kept, not its seeds, so counts differ between files where the seed
    counts cannot."""
    import s3fs

    fs = s3fs.S3FileSystem() if PUBLISHED_TRACKS.startswith('s3://') else None
    counts = {}
    for i in range(10):
        src = f'{PUBLISHED_TRACKS}/global_2019_2ens{i:03d}_pre.nc'
        with fs.open(src, 'rb') if fs else open(src, 'rb') as f:
            year = xr.open_dataset(f, engine='scipy', decode_times=False)['year'].values
        counts[f'file_{i}'] = [int((year == y).sum()) for y in years]
    return pd.DataFrame(counts, index=pd.Index(years, name='year'))


def seeding(out: Path = store.CACHE / store.VALIDATION / 'seeding_1981_2005.csv') -> Path:
    """Our global TCGI per month for SEEDING_YEARS and REFERENCE_YEAR, the published files'
    storms per year, and (in the .json) the observed storm count the public code calibrates to."""
    import netCDF4

    grid = tutorial_grid()
    rows = {}
    for year in [*SEEDING_YEARS, REFERENCE_YEAR]:
        x = era5_inputs(year, grid)
        row = {}
        for kind in ('CRH', 'SD'):
            monthly = (
                tcgi(x['avor'], x[kind], x['PI'], x['vws'], kind)
                .where(x['ocean'])
                .sum(('lat', 'lon'))
                .values
            )
            row.update({f'TCGI_{kind}_m{m:02d}': v for m, v in enumerate(monthly, 1)})
        rows[year] = row
        print(year, flush=True)
    df = pd.DataFrame.from_dict(rows, orient='index').rename_axis('year')
    df = df.join(published_counts(list(df.index)))
    with netCDF4.Dataset(store.path(f'{store.TUTORIAL}/bt_global_predictors.nc')) as bt:
        years = np.asarray(bt['StormYear'][:])
    observed = ((years >= SEEDING_YEARS[0]) & (years <= SEEDING_YEARS[-1])).sum() / len(
        SEEDING_YEARS
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    (out.with_suffix('.json')).write_text(
        json.dumps({'observed_storms_per_year_1981_2005': observed, 'survival_rate': SURVIVAL})
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('compare')
    sub.add_parser('seeding')
    sub.add_parser('fetch-validation')
    a = p.parse_args()
    if a.cmd == 'compare':
        print(compare())
    elif a.cmd == 'seeding':
        print(seeding())
    else:
        fetch_validation()


if __name__ == '__main__':
    main()
