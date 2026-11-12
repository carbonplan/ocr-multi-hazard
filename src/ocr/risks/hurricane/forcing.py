"""CHAZ forcing from ERA5, one year at a time: the files CHAZ's downscaling reads.

Ports the CHAZ pre-processing tutorial (github.com/cl3225/CHAZ, Tutorial_chaz_pre.ipynb) and
writes its file layout, on the tutorial's 2-degree grid (cell centres on odd latitudes):

    TCGI_CRH_PI_{year}.nc, TCGI_SD_PI_{year}.nc   genesis index and its inputs (tcgi.py)
    PI_{year}.nc                                  potential intensity (vmax)
    I_MLR_{year}.nc                               intensity predictors: 250/850 hPa winds, mean
                                                  relative humidity at 500/700/850 hPa, PI
    Cov_{year}_2deg.nc                            covariance of daily 250/850 hPa wind anomalies
    A_{year}{month}.nc                            its Cholesky factor, interpolated to each day

Raw ERA5 goes under store.S3_ROOT/inputs/era5/{year}/ with the CDS request beside each file
(`<file>.request.json`); forcing goes under store.S3_ROOT/forcing/era5/{year}/.

    python -m ocr.risks.hurricane.forcing <command>

    fetch     ERA5 for a year into the cache: from S3 when it is there, else from the CDS
    build     the forcing files for a year, from the cached ERA5
    task      fetch, build and push one year (what each Coiled task runs; year from
              $COILED_BATCH_TASK_INPUT)
    launch    one Coiled batch task per year (batch.py; needs a Coiled login and ~/.cdsapirc)
    manifest  merge the jobs' manifest fragments into manifest.json, on Coiled
    reference / launch-reference
              the tutorial's Zenodo files and a January 2017 wind test the review notebooks
              compare with, fetched and pushed (here, or on Coiled)

Run in the `hurricane` pixi environment.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from ocr.risks.hurricane import batch, chaz, store, tcgi

RAW = store.ERA5
OUT = store.FORCING
LAT = np.arange(-89.0, 90.0, 2.0)
LON = np.arange(0.0, 360.0, 2.0)
DAILY_TIMES = ['00:00', '06:00', '12:00', '18:00']  # daily means of 6-hourly analyses
COV_PAIRS = [
    'u250pu250p',
    'u250pv250p',
    'u250pu850p',
    'u250pv850p',
    'v250pv250p',
    'v250pu850p',
    'v250pv850p',
    'u850pu850p',
    'u850pv850p',
    'v850pv850p',
]


def grid() -> xr.Dataset:
    return xr.Dataset(coords={'lat': LAT, 'lon': LON})


# --- fetch -------------------------------------------------------------------------------------


def requests(year: int) -> dict[str, tuple[str, dict]]:
    base = {'year': [str(year)], 'data_format': 'netcdf', 'download_format': 'unarchived'}
    monthly = {
        **base,
        'product_type': ['monthly_averaged_reanalysis'],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'time': ['00:00'],
    }
    pl, sl = (
        'reanalysis-era5-pressure-levels-monthly-means',
        'reanalysis-era5-single-levels-monthly-means',
    )
    out = {
        f'{RAW}/{year}/pl_wind_025.nc': (
            pl,
            {
                **monthly,
                'pressure_level': ['250', '850'],
                'variable': ['u_component_of_wind', 'v_component_of_wind'],
            },
        ),
        f'{RAW}/{year}/sl_025.nc': (
            sl,
            {
                **monthly,
                'variable': [
                    'sea_surface_temperature',
                    'mean_sea_level_pressure',
                    'surface_pressure',
                    'total_column_water_vapour',
                    'land_sea_mask',
                ],
            },
        ),
        f'{RAW}/{year}/pl_tq_1deg.nc': (
            pl,
            {
                **monthly,
                'variable': ['temperature', 'specific_humidity'],
                'pressure_level': [str(p) for p in tcgi.ERA5_LEVELS],
                'grid': ['1.0', '1.0'],
            },
        ),
        f'{RAW}/{year}/pl_rh_025.nc': (
            pl,
            {**monthly, 'variable': ['relative_humidity'], 'pressure_level': ['500', '700', '850']},
        ),
    }
    for m in range(1, 13):  # one month per request keeps each under the CDS size limit
        out[f'{RAW}/{year}/uv_daily_2deg_{m:02d}.nc'] = (
            'reanalysis-era5-pressure-levels',
            {
                **base,
                'product_type': ['reanalysis'],
                'month': [f'{m:02d}'],
                'day': [f'{d:02d}' for d in range(1, calendar.monthrange(year, m)[1] + 1)],
                'time': DAILY_TIMES,
                'pressure_level': ['250', '850'],
                'variable': ['u_component_of_wind', 'v_component_of_wind'],
                'area': [89, 0, -89, 358],
                'grid': ['2.0', '2.0'],
            },
        )
    return out


def _on_s3(key: str) -> bool:
    return store.on_s3(key)


def check_request(key: str, dataset: str, request: dict) -> None:
    """A reused raw file must come from the request we would send now. Its saved request is
    compared when there is one beside it."""
    saved = store.CACHE / f'{key}.request.json'
    if not saved.exists():
        return
    got = json.loads(saved.read_text())
    if (got['dataset'], got['request']) != (dataset, request):
        raise ValueError(f'{key} was fetched with a different request; delete it to fetch again')


def fetch(year: int, workers: int = 4) -> None:
    """Cache every raw file for `year`: from S3 if it is there, else from the CDS, with its
    request saved beside it. Each CDS download goes to S3 at once, since a spot reclaim can end a
    task hours into the CDS queue."""
    import cdsapi

    todo = requests(year)

    def get(key):
        local = store.CACHE / key
        if local.exists():
            check_request(key, *todo[key])
            return key, 'cached'
        if _on_s3(key):
            store.path(key)
            if _on_s3(f'{key}.request.json'):
                store.path(f'{key}.request.json')
            check_request(key, *todo[key])
            return key, 's3'
        dataset, req = todo[key]
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + '.part')
        cdsapi.Client(quiet=True).retrieve(dataset, req, str(part))
        part.rename(local)
        local.with_name(local.name + '.request.json').write_text(
            json.dumps(
                {'dataset': dataset, 'request': req, 'retrieved': datetime.now(UTC).isoformat()},
                indent=1,
            )
        )
        store.upload(f'{key}.request.json')
        store.upload(key)
        return key, 'cds'

    with ThreadPoolExecutor(workers) as pool:
        for key, how in pool.map(get, todo):
            print(f'{how:6s} {key}', flush=True)


def fetch_validation() -> list[str]:
    """January 2017 winds at every hour and at 00/06/12/18 UTC, on the tutorial's grid: the review
    notebook tests which daily mean the tutorial's daily winds are."""
    import cdsapi

    keys = []
    for name, hours in (('1h', [f'{h:02d}:00' for h in range(24)]), ('6h', DAILY_TIMES)):
        key = f'validation/forcing/uv_201701_{name}_2deg.nc'
        dest = store.CACHE / key
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + '.part')
            cdsapi.Client(quiet=True).retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': ['reanalysis'],
                    'variable': ['u_component_of_wind', 'v_component_of_wind'],
                    'year': ['2017'],
                    'month': ['01'],
                    'day': [f'{d:02d}' for d in range(1, 32)],
                    'time': hours,
                    'pressure_level': ['250', '850'],
                    'area': [89, 0, -89, 358],
                    'grid': ['2.0', '2.0'],
                    'data_format': 'netcdf',
                    'download_format': 'unarchived',
                },
                str(part),
            )
            part.rename(dest)
        keys.append(key)
    return keys


# --- build -------------------------------------------------------------------------------------


def _monthly_winds(year: int) -> xr.Dataset:
    w = store.open_era5(store.path(f'{RAW}/{year}/pl_wind_025.nc')).sel(lat=LAT, lon=LON)
    return xr.Dataset(
        {f'{c}{lev}': w[c].sel(level=lev, drop=True) for c in ('u', 'v') for lev in (250, 850)}
    )


def _daily_winds(year: int) -> xr.Dataset:
    months = [
        store.open_era5(store.path(f'{RAW}/{year}/uv_daily_2deg_{m:02d}.nc')) for m in range(1, 13)
    ]
    w = xr.concat(months, 'time').sortby('time').sortby('lat').sel(lat=LAT, lon=LON)
    w = w.resample(time='1D').mean()
    days = 366 if calendar.isleap(year) else 365
    if w.sizes['time'] != days or any(bool(w[v].isnull().any()) for v in w.data_vars):
        raise ValueError(f'{year}: daily winds cover {w.sizes["time"]} days or have gaps')
    return xr.Dataset(
        {f'{c}{lev}': w[c].sel(level=lev, drop=True) for c in ('u', 'v') for lev in (250, 850)}
    )


def covariance(year: int, daily: xr.Dataset, monthly: xr.Dataset) -> xr.Dataset:
    """Per-month covariance of daily wind anomalies from the monthly means interpolated to each day
    (the tutorial's cell 13)."""
    ndays = daily.sizes['time']
    mid, start = [], 0
    for m in range(1, 13):
        n = calendar.monthrange(year, m)[1]
        mid.append(int(n / 2 + start) - 1)
        start += n
    d = np.arange(ndays)
    anom = {}
    for v in ('u250', 'v250', 'u850', 'v850'):
        clim = interp1d(
            mid, monthly[v].values, bounds_error=False, fill_value='extrapolate', axis=0
        )(d)
        anom[f'{v}p'] = daily[v].values - clim
    month = daily['time.month'].values
    cov = {}
    for pair in COV_PAIRS:
        x, y = anom[pair[:5]], anom[pair[5:]]
        out = np.empty((12,) + x.shape[1:])
        for m in range(1, 13):
            xm, ym = x[month == m], y[month == m]
            n = xm.shape[0]
            out[m - 1] = (np.mean(xm * ym, 0) - np.mean(xm, 0) * np.mean(ym, 0)) * n / (n - 1)
        cov[pair] = (('month', 'lat', 'lon'), out)
    ds = xr.Dataset(
        cov,
        coords={'month': np.arange(1, 13), 'lat': daily['lat'].values, 'lon': daily['lon'].values},
    )
    for v in COV_PAIRS:
        ds[v].attrs['units'] = '(m/s)^2'
    return ds


def a_matrices(year: int, cov: xr.Dataset) -> dict[int, np.ndarray]:
    """Lower Cholesky factors of each day's covariance, as the 10 lower-triangle entries CHAZ reads,
    shape (10, day, lat, lon) per month (the tutorial's cell 14). The tutorial's published A files
    hold the lower factor; its notebook code as written transposes it, zeroing the off-diagonals."""
    c = {}
    for v in COV_PAIRS:
        a = cov[v].values.copy()
        a[:, 0, :] = a[:, 1, :]
        a[:, -1, :] = a[:, -2, :]
        c[v] = a
    upper = list(zip(*np.triu_indices(4)))  # COV_PAIRS order over (u250, v250, u850, v850)
    lower = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)]
    out = {}
    for m in range(1, 13):
        days = [datetime(year, m, d) for d in range(1, calendar.monthrange(year, m)[1] + 1)]
        stack = np.stack(
            [np.stack([chaz.date_interpolation(d, c[v]) for d in days]) for v in COV_PAIRS]
        )  # (10, day, lat, lon)
        mat = np.zeros(stack.shape[1:] + (4, 4))
        for k, (i, j) in enumerate(upper):
            mat[..., i, j] = mat[..., j, i] = stack[k]
        if not np.isfinite(mat).all():
            raise ValueError(f'{year}-{m:02d}: the wind covariance has missing values')
        chol = np.linalg.cholesky(mat)
        out[m] = np.stack([chol[..., i, j] for i, j in lower]).astype('float32')
    return out


def predictors(year: int, monthly: xr.Dataset, pi: xr.DataArray) -> xr.Dataset:
    """The intensity model's monthly predictors (the tutorial's cell 17)."""
    rh = store.open_era5(store.path(f'{RAW}/{year}/pl_rh_025.nc')).sel(lat=LAT, lon=LON)
    ds = monthly.copy()
    ds['hur'] = rh['r'].sel(level=[500, 700, 850]).mean('level')
    ds['PI'] = pi
    return ds.drop_vars('level', errors='ignore').transpose('time', 'lat', 'lon')


def build(year: int) -> list[str]:
    out = store.CACHE / OUT / str(year)
    out.mkdir(parents=True, exist_ok=True)
    x = tcgi.era5_inputs(year, grid())
    written = []

    def save(ds: xr.Dataset, name: str, **kw) -> None:
        part = out / f'{name}.part'
        ds.to_netcdf(part, **kw)
        part.rename(out / name)
        written.append(f'{OUT}/{year}/{name}')

    for kind in ('CRH', 'SD'):
        g = tcgi.tcgi(x['avor'], x[kind], x['PI'], x['vws'], kind).where(x['ocean'])
        ds = xr.Dataset(
            {'avor': x['avor'], 'vws': x['vws'], 'CRH': x['CRH'], 'SD': x['SD'], 'TCGI': g}
        )
        save(ds.drop_vars('ocean', errors='ignore'), f'TCGI_{kind}_PI_{year}.nc')
    # PI is 0 wherever the tutorial's bilinear regrid touches land or missing SST, as TCGI is undefined
    x['PI'] = x['PI'].where(x['ocean'], 0.0)
    save(x[['PI']].rename(PI='vmax'), f'PI_{year}.nc')
    monthly = _monthly_winds(year)
    save(predictors(year, monthly, x['PI']), f'I_MLR_{year}.nc')
    cov = covariance(year, _daily_winds(year), monthly)
    save(cov, f'Cov_{year}_2deg.nc')
    for m, a in a_matrices(year, cov).items():
        ds = xr.Dataset(
            {'A': (('Adim', 'days', 'latitude', 'longitude'), a)},
            coords={
                'latitude': LAT.astype('float32'),
                'longitude': LON.astype('float32'),
                'days': np.arange(a.shape[1], dtype='int32'),
                'Adim': np.arange(10, dtype='int32'),
            },
        )
        save(ds, f'A_{year}{m:02d}.nc', format='NETCDF3_CLASSIC')
    return written


# --- Coiled ------------------------------------------------------------------------------------


def task(year: int) -> None:
    fetch(year)
    build(year)
    raw, forcing = (
        store.push(f'{RAW}/{year}', fragment=f'era5-{year}-raw'),
        store.push(f'{OUT}/{year}', fragment=f'era5-{year}-forcing'),
    )
    print(f'{year}: {len(raw)} raw and {len(forcing)} forcing files on S3', flush=True)


def reference() -> None:
    """The reference files the review notebooks read: the tutorial's Zenodo record and the January
    2017 wind test, pushed to S3 (what `launch-reference` runs)."""
    store.fetch_tutorial()
    store.push(store.TUTORIAL, fragment='tutorial')
    fetch_validation()
    store.push('validation/forcing', fragment='validation-forcing')


def cds_secrets() -> dict[str, str]:
    """The CDS key from ~/.cdsapirc, for Coiled's secret env vars."""
    rc = Path('~/.cdsapirc').expanduser().read_text().splitlines()
    cds = dict(line.split(':', 1) for line in rc if ':' in line)
    return {'CDSAPI_URL': cds['url'].strip(), 'CDSAPI_KEY': cds['key'].strip()}


def launch(years: list[int], vm_type: str, max_workers: int, spot: bool = True) -> None:
    """One Coiled batch task per year; the CDS key goes as secret env vars."""
    secrets = cds_secrets()
    batch.run(
        'forcing',
        ['task'],
        name=f'chaz-forcing-era5-{years[0]}-{years[-1]}',
        over=years,
        vm_type=vm_type,
        max_workers=max_workers,
        spot=spot,
        secret_env=secrets,
    )


def launch_manifest() -> None:
    """Merge the parallel jobs' manifest fragments, on a small Coiled VM."""
    batch.run(
        'store', ['manifest'], name='chaz-manifest', vm_type='m8g.large', spot=False, wait=True
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('fetch', 'build'):
        s = sub.add_parser(name)
        s.add_argument('--year', type=int, nargs='+', required=True)
    t = sub.add_parser('task')
    t.add_argument('--year', type=int, default=None)
    la = sub.add_parser('launch')
    la.add_argument('--years', default='1981-2019')
    la.add_argument('--vm-type', default='m8g.xlarge')
    la.add_argument('--max-workers', type=int, default=8)
    la.add_argument('--on-demand', action='store_true', help='on-demand VMs instead of spot')
    sub.add_parser('manifest', help='merge manifest fragments on Coiled')
    sub.add_parser('reference', help="fetch the notebooks' reference files and push them")
    sub.add_parser('launch-reference', help='run `reference` on Coiled')
    a = p.parse_args()
    if a.cmd == 'fetch':
        for y in a.year:
            fetch(y)
    elif a.cmd == 'build':
        for y in a.year:
            print('\n'.join(build(y)))
    elif a.cmd == 'task':
        task(a.year or int(os.environ['COILED_BATCH_TASK_INPUT']))
    elif a.cmd == 'manifest':
        launch_manifest()
    elif a.cmd == 'reference':
        reference()
    elif a.cmd == 'launch-reference':
        batch.run(
            'forcing', ['reference'], name='chaz-reference', spot=False, secret_env=cds_secrets()
        )
    else:
        launch(batch.parse_years(a.years), a.vm_type, a.max_workers, spot=not a.on_demand)


if __name__ == '__main__':
    main()
