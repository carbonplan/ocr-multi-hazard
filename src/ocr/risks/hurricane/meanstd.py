"""The intensity model's predictor normalization (coefficient_meanstd.nc) for a forcing source.

CHAZ standardizes each intensity predictor by a mean and standard deviation before applying the
regression coefficients (result_w.nc, result_l.nc). For a climate model the CHAZ authors
recompute those means and standard deviations from the model's own fields and keep the
coefficients: CHAZvBeta/getMeanStd.py evaluates the predictors along the observed tracks of
1981–2012, each storm reading the model fields of its year, and writes only the normalization.
This ports it:

- environment along the tracks as chaz.get_predictors reads it from I_MLR: PI (kt) within 500 km,
  250–850 hPa shear components and relative humidity within 200–800 km, land within 300 km
- 12-hour samples as CHAZvBeta/tools/regression4.py builds them: water where the 300 km land
  value is at most -0.5 at both ends of the step, land otherwise; the predictand is the 12-hour
  change in observed wind; only samples with every value finite; population standard deviations

The tutorial's normalization (coefficient_meanstd.nc) is not reproducible from the tutorial's
observed-track file: even the predictors that depend only on the tracks (storm wind, translation
speed, the 12-hour change) differ, so it was built from another track sample. A model's
normalization is therefore the tutorial's, mapped by this port's model and ERA5 values on the same
samples so that each standardized predictor is distributed as the tutorial's gives ERA5 (compose).
For ERA5 it returns the tutorial's file; entries that depend only on the tracks stay as they are.

    python -m ocr.risks.hurricane.meanstd build --forcing forcing/era5 --out validation/meanstd/era5.nc
    python -m ocr.risks.hurricane.meanstd model --model UKESM1-0-LL --member r1i1p1f2
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from ocr.risks.hurricane import chaz, store

YEARS = (1981, 2012)
LEAD = 2  # 6-hourly steps in the 12-hour forecast
KT = 1.94384449
WATER = [
    'StormMwspd',
    'dVdt',
    'trSpeed',
    'dPIwspd',
    'SHRD',
    'rhMean',
    'dPIwspd2',
    'dPIwspd3',
    'dVdt2',
]
LAND = WATER + ['landmaskMean']
ENV = ('PIwspdMean', 'UShearMean', 'VShearMean', 'rhMean', 'landmaskMean')


def observed() -> xr.Dataset:
    return xr.open_dataset(chaz.static(chaz.gv.obpath), decode_times=False)


def along_tracks(forcing: str, years: tuple[int, int] = YEARS) -> dict[str, np.ndarray]:
    """The environmental predictors along the observed tracks of `years`, from the I_MLR files
    under `forcing`, as chaz.get_predictors computes them for synthetic storms."""
    bt = observed()
    lon, lat, days = bt['StormLon'].values, bt['StormLat'].values, bt['Time'].values
    storm_year = bt['StormYear'].values
    chaz._load_landmask()
    land = chaz.gv.ldmask.astype(float).copy()
    land[land == 0] = -1.0
    land[land == 3] = 0.0
    out = {k: np.full(lon.shape, np.nan) for k in ENV}
    t0 = datetime(1800, 1, 1)
    for year in range(years[0], years[1] + 1):
        with xr.open_dataset(store.path(f'{forcing}/{year}/I_MLR_{year}.nc')) as ds:
            xx, yy = np.meshgrid(ds['lon'].values, ds['lat'].values)
            fields = {
                'PIwspdMean': (ds['PI'].values * KT, 0, 500),
                'UShearMean': (ds['u250'].values - ds['u850'].values, 200, 800),
                'VShearMean': (ds['v250'].values - ds['v850'].values, 200, 800),
                'rhMean': (ds['hur'].values, 200, 800),
            }
        londis = 2 * np.pi * 6371.0 * np.cos(yy / 180 * np.pi) / 360
        for s in np.flatnonzero(storm_year == year):
            for it in np.flatnonzero((days[:, s] > 0) & np.isfinite(lon[:, s])):
                date = t0 + timedelta(hours=int(days[it, s] * 24))
                dist = np.hypot(londis * (xx - lon[it, s]), 110 * (yy - lat[it, s]))
                for k, (field, r1, r2) in fields.items():
                    var = chaz.date_interpolation(date, field)
                    ring = (dist <= r2) & (dist >= r1) & np.isfinite(var)
                    out[k][it, s] = var[ring].mean() if ring.any() else np.nan
                ring = (dist <= chaz.gv.landRadius) & np.isfinite(land)
                out['landmaskMean'][it, s] = land[ring].mean()
    return out


def steps(
    lm: np.ndarray, water: bool, years: tuple[int, int] = YEARS, bt: xr.Dataset | None = None
) -> np.ndarray:
    """(storm, end node) of each 12-hour step of the observed tracks of `years` that falls under
    the water regression (land value `lm` at most -0.5 at both ends) or the land regression."""
    bt = observed() if bt is None else bt
    lon, storm_year = bt['StormLon'].values, bt['StormYear'].values
    out = []
    for s in np.flatnonzero((storm_year >= years[0]) & (storm_year <= years[1])):
        for it in range(LEAD, lon.shape[0]):
            if not np.isfinite(lon[it, s]):
                continue
            if water == bool((lm[it, s] <= -0.5) & (lm[it - LEAD, s] <= -0.5)):
                out.append((s, it))
    return np.array(out, dtype=int).reshape(-1, 2)


def samples(
    env: dict[str, np.ndarray], predictors: list[str], water: bool, years: tuple[int, int] = YEARS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """12-hour regression samples (X, Y) at `steps`, and which of them are complete (every value
    finite); the rows are the same for any environment with the same land values."""
    bt = observed()
    v, dvdt, speed = (bt[k].values for k in ('StormMwspd', 'dVdt', 'trSpeed'))
    lm = env['landmaskMean']
    X, Y = [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN windows give NaN samples
        for s, it in steps(lm, water, years, bt):
            i0, w = it - LEAD, slice(it - LEAD, it + 1)
            dpi = np.nanmean(env['PIwspdMean'][w, s]) - v[i0, s]
            row = {
                'StormMwspd': v[i0, s],
                'dVdt': dvdt[i0, s],
                'trSpeed': speed[i0, s],
                'dPIwspd': dpi,
                'SHRD': np.nanmean(np.hypot(env['UShearMean'][w, s], env['VShearMean'][w, s])),
                'rhMean': np.nanmean(env['rhMean'][w, s]),
                'dPIwspd2': dpi**2,
                'dPIwspd3': dpi**3,
                'dVdt2': dvdt[i0, s] ** 2,
                'landmaskMean': lm[it, s] - lm[i0, s],
            }
            X.append([row[p] for p in predictors])
            Y.append(v[it, s] - v[i0, s])
    X, Y = np.array(X), np.array(Y)
    return X, Y, np.isfinite(X).all(axis=1) & np.isfinite(Y)


SETS = (('w', WATER, True, ('d2', 'd3')), ('l', LAND, False, ('d7', 'd8')))
TRACK_ONLY = {'StormMwspd', 'dVdt', 'trSpeed', 'dVdt2'}


def normalization(
    env: dict[str, np.ndarray], years: tuple[int, int] = YEARS, also: dict | None = None
) -> xr.Dataset:
    """coefficient_meanstd.nc's layout: a leading 0 for the constant, then one entry per predictor.
    With `also` (another environment), only the samples complete in both."""
    out = {}
    for tag, preds, water, (dm, ds) in SETS:
        X, Y, ok = samples(env, preds, water, years)
        if also is not None:
            ok &= samples(also, preds, water, years)[2]
        X, Y = X[ok], Y[ok]
        out[f'meanX_{tag}'] = (dm, np.r_[0.0, X.mean(0)])
        out[f'stdX_{tag}'] = (ds, np.r_[0.0, X.std(0)])
        out[f'meanY_{tag}'] = ((), Y.mean())
        out[f'stdY_{tag}'] = ((), Y.std())
        out[f'n_{tag}'] = ((), len(Y))
    ds = xr.Dataset(out)
    ds.attrs.update(
        {
            'predictors_w': ' '.join(WATER),
            'predictors_l': ' '.join(LAND),
            'years': f'{years[0]}-{years[1]}',
            'tracks': chaz.gv.obpath,
        }
    )
    return ds


def build(forcing: str, out: str, years: tuple[int, int] = YEARS) -> None:
    env = along_tracks(forcing, years)
    ds = normalization(env, years)
    ds.attrs['forcing'] = forcing
    local = store.CACHE / out
    local.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(local)
    np.savez_compressed(local.with_suffix('.env.npz'), **env)
    store.push(keys=[out, str(Path(out).with_suffix('.env.npz'))], replace=True)
    print(ds)


ERA5_PORT = 'validation/meanstd/era5.nc'


def compose(model_port: xr.Dataset, era5_port: xr.Dataset, shipped: xr.Dataset) -> xr.Dataset:
    """The model's normalization: each predictor standardized so its distribution over the
    observed tracks matches what the shipped normalization gives ERA5 there. With mean m and
    standard deviation s, s_target = s_shipped s_model / s_ERA5 and
    m_target = m_model + (m_shipped - m_ERA5) s_model / s_ERA5. The ports must be on the same
    samples; predictors from the tracks alone and the predictand keep the shipped values."""
    out = shipped.copy()
    for tag, preds, _, _ in SETS:
        mm, me = model_port[f'meanX_{tag}'].values, era5_port[f'meanX_{tag}'].values
        sm, se = model_port[f'stdX_{tag}'].values, era5_port[f'stdX_{tag}'].values
        ms, ss = shipped[f'meanX_{tag}'].values, shipped[f'stdX_{tag}'].values
        k = np.divide(sm, se, out=np.ones_like(sm), where=se > 0)
        mean, std = mm + (ms - me) * k, ss * k
        for i, p in enumerate(preds, start=1):
            if p in TRACK_ONLY:
                mean[i], std[i] = ms[i], ss[i]
        mean[0], std[0] = ms[0], ss[0]
        out[f'meanX_{tag}'] = (shipped[f'meanX_{tag}'].dims, mean)
        out[f'stdX_{tag}'] = (shipped[f'stdX_{tag}'].dims, std)
    return out


ERA5_ENV = 'validation/meanstd/era5_env.nc'
ENVIRONMENT = {'dPIwspd', 'SHRD', 'rhMean', 'dPIwspd2', 'dPIwspd3'}


def environment_only(port: xr.Dataset, shipped: xr.Dataset) -> xr.Dataset:
    """The shipped normalization with the potential intensity, shear and humidity entries
    replaced by the port's, so those are standardized by the forcing's own statistics; the land
    term, the entries from the tracks alone and the predictand keep the shipped values."""
    out = shipped.copy()
    for tag, preds, _, _ in SETS:
        mean, std = shipped[f'meanX_{tag}'].values.copy(), shipped[f'stdX_{tag}'].values.copy()
        for i, p in enumerate(preds, start=1):
            if p in ENVIRONMENT:
                mean[i] = port[f'meanX_{tag}'].values[i]
                std[i] = port[f'stdX_{tag}'].values[i]
        out[f'meanX_{tag}'] = (shipped[f'meanX_{tag}'].dims, mean)
        out[f'stdX_{tag}'] = (shipped[f'stdX_{tag}'].dims, std)
    return out


def era5_env() -> None:
    with (
        xr.open_dataset(store.path(ERA5_PORT)) as port,
        xr.open_dataset(chaz.static('coefficient_meanstd.nc')) as shipped,
    ):
        ds = environment_only(port.load(), shipped.load())
    ds.attrs.update(
        {
            'method': 'meanstd.environment_only: the tutorial coefficient_meanstd.nc with the '
            'environment entries from this port on ERA5',
            'port': ERA5_PORT,
        }
    )
    local = store.CACHE / ERA5_ENV
    local.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(local)
    store.push(keys=[ERA5_ENV], replace=True)
    print(ds)


def model(model_name: str, member: str) -> None:
    from ocr.risks.hurricane import cmip

    hist = cmip.root('historical', model_name, member)
    env_m = along_tracks(hist)
    with np.load(store.path(str(Path(ERA5_PORT).with_suffix('.env.npz')))) as f:
        env_e = {k: f[k] for k in ENV}
    ports = {
        'model': normalization(env_m, also=env_e).assign_attrs(forcing=hist),
        'era5': normalization(env_e, also=env_m).assign_attrs(forcing=store.FORCING),
    }
    keys = []
    for name, port in ports.items():
        key = f'validation/meanstd/{model_name}_{member}_port_{name}.nc'
        (store.CACHE / key).parent.mkdir(parents=True, exist_ok=True)
        port.to_netcdf(store.CACHE / key)
        keys.append(key)
    np.savez_compressed(store.CACHE / f'validation/meanstd/{model_name}_{member}.env.npz', **env_m)
    keys.append(f'validation/meanstd/{model_name}_{member}.env.npz')
    with xr.open_dataset(chaz.static('coefficient_meanstd.nc')) as shipped:
        ds = compose(ports['model'], ports['era5'], shipped.load())
    ds.attrs.update(
        {
            'method': 'meanstd.compose: the tutorial coefficient_meanstd.nc mapped by this port, '
            'model against ERA5 on the same observed-track samples',
            'model_port': keys[0],
            'era5_port': keys[1],
        }
    )
    key = cmip.meanstd_key(model_name, member)
    local = store.CACHE / key
    local.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(local)
    store.push(keys=[key, *keys], replace=True)
    print(ds)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--forcing', required=True, help='forcing prefix, e.g. forcing/era5')
    b.add_argument('--out', required=True, help='key for the normalization file')
    sub.add_parser('era5-env')
    m = sub.add_parser('model')
    m.add_argument('--model', default='UKESM1-0-LL')
    m.add_argument('--member', default='r1i1p1f2')
    a = p.parse_args()
    if a.cmd == 'era5-env':
        era5_env()
    elif a.cmd == 'build':
        build(a.forcing, a.out)
    else:
        model(a.model, a.member)


if __name__ == '__main__':
    main()
