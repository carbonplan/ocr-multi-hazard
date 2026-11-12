"""Which steering noise produced a set of CHAZ tracks, from the tracks' own motion.

CHAZ moves a storm with 0.8 of the 850 hPa plus 0.2 of the 250 hPa wind: the monthly mean
interpolated to the day plus A·ε, with A a factor of the daily wind covariance and ε white noise,
plus beta drift. Track motion minus the mean steering, less a fitted drift that scales with
cos(lat), leaves the A·ε part, whose spread and u-v correlation differ between A read as the lower
Cholesky factor (the tutorial's files, and ours) and A with its off-diagonal entries zero (what
the tutorial notebook's code as written produces).

Steps are 6-hourly, North Atlantic, 10-30N, in the year of the forcing. The steering and A are
averaged over 5x5 cells of the 2-degree grid around each step, to smooth over the grid.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter

from ocr.risks.hurricane import chaz, landfall

DEG = 2 * np.pi * 6371000 / 360.0
EPOCH = datetime.datetime(1950, 1, 1)
LOWER = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)]
OFF_DIAGONAL = [k for k, (i, j) in enumerate(LOWER) if i != j]


def _smooth(x: np.ndarray) -> np.ndarray:
    """5x5 box mean over the last two (lat, lon) axes."""
    size = (1,) * (x.ndim - 2) + (5, 5)
    return uniform_filter(x, size=size, mode='nearest')


class Forcing:
    """One year's monthly steering wind and daily A, smoothed, from CHAZ forcing files: I_MLR
    (u/v 250/850) and A_{year}{month}, as `open(name)` returns them."""

    def __init__(self, year: int, open):
        ds = open(f'I_MLR_{year}.nc')
        self.year = year
        self.lat = ds['lat'].values.astype(float)
        self.lon = ds['lon'].values.astype(float)
        self.steer = {
            k: _smooth(0.8 * ds[f'{k}850'].values + 0.2 * ds[f'{k}250'].values) for k in 'uv'
        }
        self.a = {m: _smooth(open(f'A_{year}{m:02d}.nc')['A'].values) for m in range(1, 13)}


def predicted(a: np.ndarray, zero_off_diagonal: bool) -> tuple[float, float, float]:
    """Variance of u, of v, and their covariance for the steering noise 0.8·A850ε + 0.2·A250ε,
    with a the 10 lower-triangle entries over (u250, v250, u850, v850)."""
    a = a.copy()
    if zero_off_diagonal:
        a[OFF_DIAGONAL] = 0.0
    L = np.zeros((4, 4))
    for k, (i, j) in enumerate(LOWER):
        L[i, j] = a[k]
    cu = 0.2 * L[0] + 0.8 * L[2]
    cv = 0.2 * L[1] + 0.8 * L[3]
    return float(cu @ cu), float(cv @ cv), float(cu @ cv)


def steps(ds: xr.Dataset, f: Forcing) -> np.ndarray:
    """Per 6-hourly step of the North Atlantic tracks in a CHAZ track file: residual u and v
    (m/s) from the mean steering, the cos(lat) scale of beta drift, and the predicted variances
    and covariance for both readings of A."""
    year = ds['year'].values
    sel = np.nonzero(year == f.year)[0]
    lat = ds['latitude'].values[:, sel]
    lon = ds['longitude'].values[:, sel]
    t = ds['time'].values[:, sel]
    fill = t <= landfall.TIME_FILL
    lat, lon = np.where(fill, np.nan, lat), np.where(fill, np.nan, lon)
    out = []
    for s in np.nonzero(landfall.in_north_atlantic(lat[0], landfall.wrap(lon[0])))[0]:
        n = int(np.isfinite(lat[:, s]).sum())
        for k in range(n - 1):
            la0, lo0, la1, lo1 = lat[k, s], lon[k, s], lat[k + 1, s], lon[k + 1, s]
            when = EPOCH + datetime.timedelta(days=float(t[k, s]) + 0.125)
            if when.year != f.year or not (10 <= la0 < 30):
                continue
            mla, mlo = 0.5 * (la0 + la1), 0.5 * (lo0 + lo1)
            iy = int(np.argmin(np.abs(f.lat - mla)))
            ix = int(np.argmin(np.abs(f.lon - mlo % 360)))
            us = chaz.date_interpolation(when, f.steer['u'])[iy, ix]
            vs = chaz.date_interpolation(when, f.steer['v'])[iy, ix]
            u = (lo1 - lo0) * DEG * np.cos(np.deg2rad(mla)) / 21600.0
            v = (la1 - la0) * DEG / 21600.0
            scale = np.cos(np.deg2rad(mla)) / np.cos(np.deg2rad(10.0))
            a = f.a[when.month][:, when.day - 1, iy, ix]
            out.append((u - us, v - vs, scale, *predicted(a, False), *predicted(a, True)))
    return np.array(out).reshape(-1, 9)


def compare(x: np.ndarray) -> dict:
    """Measured spread and u-v correlation of the residual after the fitted drift, against what
    each reading of A predicts."""
    ru, rv, sc = x[:, 0], x[:, 1], x[:, 2]
    eu = ru - (ru * sc).sum() / (sc**2).sum() * sc
    ev = rv - (rv * sc).sum() / (sc**2).sum() * sc
    out = {
        'steps': len(x),
        'sd u (m/s)': eu.std(),
        'sd v (m/s)': ev.std(),
        'r(u, v)': np.corrcoef(eu, ev)[0, 1],
    }
    rows = {'measured': out}
    for name, o in (('lower factor', 3), ('off-diagonals zero', 6)):
        vu, vv, cuv = x[:, o].mean(), x[:, o + 1].mean(), x[:, o + 2].mean()
        rows[f'predicted, {name}'] = {
            'steps': len(x),
            'sd u (m/s)': np.sqrt(vu),
            'sd v (m/s)': np.sqrt(vv),
            'r(u, v)': cuv / np.sqrt(vu * vv),
        }
    return rows


def table(sets: dict[str, np.ndarray]) -> pd.DataFrame:
    """compare() for each named track set, as one table."""
    return pd.concat(
        {name: pd.DataFrame(compare(x)).T for name, x in sets.items()}, names=['tracks', '']
    )
