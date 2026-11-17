"""CHAZ's intensity regressions refit on a forcing's own environment along the observed tracks.

The shipped coefficients (result_w.nc, result_l.nc) and their normalization were fit on the CHAZ
authors' forcing. This fits the same two regressions as CHAZvBeta/tools/regression4.py
(calMultiRegression_coef_mean_water and _land): the same predictors, 12-hour steps of the observed
tracks of meanstd.YEARS, the predictand the 12-hour change in observed wind, every predictor and
the predictand standardized by their sample mean and population standard deviation, then
ordinary least squares with a constant. The environment along the tracks is meanstd's port of
get_predictors on the forcing (PI within 500 km, shear and humidity within 200-800 km).

Land is the land value CHAZ's intensity model reads with chaz --land fine: -1 or 0 at the
storm's position on the 1/12-degree mask. A step is a water step when that value is -1 at both
ends, and the land regression's land predictor is its change over the step. regression4.py fits
on the 300 km land mean instead, which the public intensity model does not read.

The stochastic term draws, for a storm at wind v0, one 12-hour error from the observed steps that
start in v0's 10 kt bin (chaz.get_E1, chaz.findError). The errors are the fitted model's
prediction minus the observed wind 12 hours on, kept at the step's first node, as in the
tutorial's bt_global_predictors.nc. chaz --intensity refit reads all four files. errors.nc also
keeps each step's ring PI (PIstep), for chaz --intensity refit-pi, which draws from the steps
that start in v0's wind bin and the same tercile of ring PI (chaz.error_pool_pi).

    python -m ocr.risks.hurricane.refit fit
    python -m ocr.risks.hurricane.refit check   # the same fit on the tutorial's own environment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from ocr.risks.hurricane import chaz, meanstd, store

OUT = 'validation/refit'
FILES = ('result_w.nc', 'result_l.nc', 'coefficient_meanstd.nc', 'errors.nc')


def key(name: str, file: str) -> str:
    return f'{OUT}/{name}/{file}'


def fine_land(bt: xr.Dataset) -> np.ndarray:
    """chaz.fine_land at each observed node, NaN off the tracks."""
    chaz._load_landmask()
    lat, lon = bt['StormLat'].values, bt['StormLon'].values
    out = np.full(lat.shape, np.nan)
    for i, j in zip(*np.nonzero(np.isfinite(lat) & np.isfinite(lon)), strict=True):
        out[i, j] = chaz.fine_land(lat[i, j], lon[i, j])
    return out


def era5_env() -> dict[str, np.ndarray]:
    """meanstd's ERA5 environment along the tracks, with chaz --land fine's land values."""
    with np.load(store.path(str(Path(meanstd.ERA5_PORT).with_suffix('.env.npz')))) as f:
        env = {k: f[k] for k in meanstd.ENV}
    env['landmaskMean'] = fine_land(meanstd.observed())
    return env


def tutorial_env() -> dict[str, np.ndarray]:
    """The environment the tutorial's observed-track file carries, 300 km land mean included."""
    bt = meanstd.observed()
    return {k: bt[k].values for k in meanstd.ENV}


def fit(env: dict[str, np.ndarray], years: tuple[int, int] = meanstd.YEARS) -> dict:
    """Coefficients, normalization and error pool for each regression, and fit statistics."""
    bt = meanstd.observed()
    errors = np.full(bt['StormMwspd'].shape, np.nan)
    pi_step = np.full(bt['StormMwspd'].shape, np.nan)
    out = {'errors': errors, 'PIstep': pi_step}
    for tag, preds, water, _ in meanstd.SETS:
        X, Y, ok = meanstd.samples(env, preds, water, years)
        rows = meanstd.steps(env['landmaskMean'], water, years, bt)[ok]
        X, Y = X[ok], Y[ok]
        mX, sX, mY, sY = X.mean(0), X.std(0), Y.mean(), Y.std()
        Z = np.c_[np.ones(len(Y)), (X - mX) / sX]
        params = np.linalg.lstsq(Z, (Y - mY) / sY, rcond=None)[0]
        dy = (Z @ params) * sY + mY
        start = rows[:, 1] - meanstd.LEAD, rows[:, 0]
        errors[start] = dy - Y
        pi_step[start] = X[:, preds.index('dPIwspd')] + X[:, preds.index('StormMwspd')]
        out[tag] = {
            'params': params,
            'meanX': np.r_[0.0, mX],
            'stdX': np.r_[0.0, sX],
            'meanY': mY,
            'stdY': sY,
            'n': len(Y),
            'r2': 1 - np.var(dy - Y) / np.var(Y),
            'rmse': float(np.sqrt(np.mean((dy - Y) ** 2))),
        }
    return out


def datasets(f: dict, attrs: dict) -> dict[str, xr.Dataset]:
    bt = meanstd.observed()
    norm = {}
    for tag, _, _, (dm, ds) in meanstd.SETS:
        norm[f'meanX_{tag}'] = (dm, f[tag]['meanX'])
        norm[f'stdX_{tag}'] = (ds, f[tag]['stdX'])
        norm[f'meanY_{tag}'] = ((), f[tag]['meanY'])
        norm[f'stdY_{tag}'] = ((), f[tag]['stdY'])
        norm[f'n_{tag}'] = ((), f[tag]['n'])
    stats = {f'{tag}_{k}': float(f[tag][k]) for tag in 'wl' for k in ('n', 'r2', 'rmse')}
    common = {**attrs, **stats, 'method': __doc__.split('\n\n')[1].replace('\n', ' ')}
    return {
        'result_w.nc': xr.Dataset({'params': ('d1', f['w']['params'])}, attrs=common),
        'result_l.nc': xr.Dataset({'params': ('d1', f['l']['params'])}, attrs=common),
        'coefficient_meanstd.nc': xr.Dataset(
            norm,
            attrs={
                **common,
                'predictors_w': ' '.join(meanstd.WATER),
                'predictors_l': ' '.join(meanstd.LAND),
            },
        ),
        'errors.nc': xr.Dataset(
            {
                'errors': (('d1', 'd2'), f['errors'], {'units': 'kt'}),
                'PIstep': (
                    ('d1', 'd2'),
                    f['PIstep'],
                    {'units': 'kt', 'long_name': 'ring PI averaged over the step'},
                ),
                'StormMwspd': (('d1', 'd2'), bt['StormMwspd'].values, {'units': 'kt'}),
                'StormYear': ('d2', bt['StormYear'].values),
            },
            attrs=common,
        ),
    }


def response(params, meanX, stdX, meanY, stdY, wind, pi) -> np.ndarray:
    """The water regression's 12-hour change (kt) for a storm at `wind` under ring PI `pi` whose
    wind did not change over the previous 12 hours, with translation speed, shear and humidity
    at the normalization's means."""
    d = pi - wind
    x = {
        'StormMwspd': wind,
        'dVdt': 0.0,
        'trSpeed': meanX[3],
        'dPIwspd': d,
        'SHRD': meanX[5],
        'rhMean': meanX[6],
        'dPIwspd2': d**2,
        'dPIwspd3': d**3,
        'dVdt2': 0.0,
    }
    z = [(x[p] - meanX[i]) / stdX[i] for i, p in enumerate(meanstd.WATER, start=1)]
    return (params[0] + np.dot(params[1:], z)) * stdY + meanY


def shipped() -> dict:
    with xr.open_dataset(chaz.static('coefficient_meanstd.nc')) as ms:
        out = {}
        for tag in 'wl':
            with xr.open_dataset(chaz.static(f'result_{tag}.nc')) as r:
                params = r['params'].values
            out[tag] = {
                'params': params,
                'meanX': ms[f'meanX_{tag}'].values,
                'stdX': ms[f'stdX_{tag}'].values,
                'meanY': float(ms[f'meanY_{tag}']),
                'stdY': float(ms[f'stdY_{tag}']),
            }
    return out


def report(f: dict) -> None:
    import pandas as pd

    ref = shipped()
    for tag, preds, _, _ in meanstd.SETS:
        print(
            f'\n{tag}: n {f[tag]["n"]}, R2 {f[tag]["r2"]:.3f}, rmse {f[tag]["rmse"]:.2f} kt '
            f'(meanY {f[tag]["meanY"]:.2f}, stdY {f[tag]["stdY"]:.2f}; shipped '
            f'{ref[tag]["meanY"]:.2f}, {ref[tag]["stdY"]:.2f})'
        )
        print(
            pd.DataFrame(
                {
                    'refit': f[tag]['params'],
                    'shipped': ref[tag]['params'],
                    'mean refit': f[tag]['meanX'],
                    'mean shipped': ref[tag]['meanX'],
                    'std refit': f[tag]['stdX'],
                    'std shipped': ref[tag]['stdX'],
                },
                index=['const', *preds],
            ).round(3)
        )
    winds, pis = (40, 60, 80, 100, 120, 140), (40, 70, 100, 130, 160, 190)
    for name, c in (('refit', f['w']), ('shipped', ref['w'])):
        args = (c['params'], c['meanX'], c['stdX'], c['meanY'], c['stdY'])
        table = pd.DataFrame(
            [[response(*args, w, p) for p in pis] for w in winds],
            index=pd.Index(winds, name='wind'),
            columns=pd.Index(pis, name='ring PI'),
        )
        print(f'\n12-hour change, water regression ({name}), steady storm, mean environment')
        print(table.round(1))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fit', help='refit on ERA5 and push the four files')
    sub.add_parser('check', help='refit on the tutorial file environment and print it')
    a = p.parse_args()
    if a.cmd == 'check':
        report(fit(tutorial_env()))
        return
    f = fit(era5_env())
    report(f)
    attrs = {
        'forcing': store.FORCING,
        'environment': str(Path(meanstd.ERA5_PORT).with_suffix('.env.npz')),
        'land': 'chaz --land fine',
        'years': f'{meanstd.YEARS[0]}-{meanstd.YEARS[1]}',
        'tracks': chaz.gv.obpath,
    }
    keys = []
    for file, ds in datasets(f, attrs).items():
        k = key('era5', file)
        local = store.CACHE / k
        local.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(local)
        keys.append(k)
    store.push(keys=keys, replace=True)
    print('\n'.join(keys))


if __name__ == '__main__':
    main()
