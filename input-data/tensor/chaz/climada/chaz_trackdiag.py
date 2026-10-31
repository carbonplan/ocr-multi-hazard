"""Where the CHAZ ERA5 US-landfall deficit lives: tracks, intensity, or decay.

The wind-field tests (chaz_observed.py) find CHAZ with about the observed rate
of tropical-storm winds on US land but half the hurricane-force rate. This
compares the CHAZ ERA5 track files with IBTrACS (US agency, 1981-2019)
directly in track space, with no wind model:

  - both are read at 6-hourly nodes and interpolated to hourly
  - one US land mask (Natural Earth 1:10m, rasterized at 1/24 degree)
  - US landfall: the first hour on US land after an hour over water
  - CHAZ rates are scaled so its North Atlantic tropical-storm count
    (34 kt at a node inside chaz_events' outline) equals the observed one,
    which leaves the conditional parts: where storms go and how strong they are

  run    --files 0-9     tables and trackdiag/*.csv under ~/.cache/ocr/chaz-local

Run in the `climada` pixi environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chaz_events as ce

ROOT = Path(os.environ.get('CHAZ_LOCAL_DIR', Path.home() / '.cache' / 'ocr' / 'chaz-local'))
OUT = ROOT / 'trackdiag'
IBTRACS = Path.home() / 'climada' / 'data' / 'IBTrACS.since1980.v04r01.nc'
YEARS = (1981, 2019)
TS, HU, MH = 34.0, 64.0, 96.0  # kt
BOX = (-130.0, 15.0, -60.0, 52.0)  # lon0, lat0, lon1, lat1 of the land raster
RES = 1 / 24
REGIONS = ('TX-AL', 'FL', 'GA-Car', 'VA-ME', 'other')


class USLand:
    """US land and any land on a 1/24 degree raster, looked up by nearest
    cell."""

    def __init__(self):
        import climada.util.coordinates as u_coord
        import shapely

        lon0, lat0, lon1, lat1 = BOX
        self.lons = np.arange(lon0, lon1, RES) + RES / 2
        self.lats = np.arange(lat0, lat1, RES) + RES / 2
        gx, gy = np.meshgrid(self.lons, self.lats)
        us = u_coord.get_country_geometries(['USA'], resolution=10).geometry.iloc[0]
        land = u_coord.get_land_geometry(extent=(lon0, lon1, lat0, lat1), resolution=10)
        shapely.prepare(us)
        shapely.prepare(land)
        self.mask = shapely.contains_xy(us, gx, gy)
        self.land = self.mask | shapely.contains_xy(land, gx, gy)

    def water(self, lat, lon):
        return self._lookup(~self.land, lat, lon, outside=True)

    def __call__(self, lat, lon):
        return self._lookup(self.mask, lat, lon, outside=False)

    def _lookup(self, grid, lat, lon, outside: bool):
        lon = np.where(lon > 180, lon - 360, lon)
        ok = np.isfinite(lat) & np.isfinite(lon)
        i = np.floor((np.where(ok, lat, 0) - BOX[1]) / RES).astype(int)
        j = np.floor((np.where(ok, lon, 0) - BOX[0]) / RES).astype(int)
        inside = ok & (i >= 0) & (i < self.lats.size) & (j >= 0) & (j < self.lons.size)
        out = np.full(np.shape(lat), outside) & ok
        out[inside] = grid[i[inside], j[inside]]
        return out


def region(lat, lon) -> np.ndarray:
    """The coastal regions of eberenz_na2 ead, by landfall point."""
    lon = np.where(lon > 180, lon - 360, lon)
    out = np.full(np.shape(lat), 'other', dtype=object)
    out[(lon >= -82) & (lat >= 36.6)] = 'VA-ME'
    out[(lon >= -87.6) & (lat >= 31) & (lat < 36.6)] = 'GA-Car'
    out[(lon >= -87.6) & (lat < 31)] = 'FL'
    out[(lon < -87.6) & (lat < 33)] = 'TX-AL'
    return out


def unwrap(lon: np.ndarray) -> np.ndarray:
    """Longitudes along axis 0 made continuous, so interpolation never sweeps
    the globe where a track crosses the date line or 0/360."""
    step = np.diff(lon, axis=0)
    step = (step + 180) % 360 - 180
    out = lon.copy()
    out[1:] = lon[:1] + np.cumsum(np.nan_to_num(step), axis=0)
    return np.where(np.isfinite(lon), out, np.nan)


def wrap(lon):
    return (lon + 180) % 360 - 180


def hourly(x: np.ndarray) -> np.ndarray:
    """6-hourly nodes along axis 0 to hourly, linearly; NaN where either
    neighbouring node is missing."""
    n = x.shape[0]
    h = np.arange((n - 1) * 6 + 1)
    i0 = np.minimum(h // 6, n - 2)
    f = ((h - 6 * i0) / 6.0).reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - f) + x[i0 + 1] * f


def landfalls(lat_h, lon_h, land: USLand):
    """First US landfall hour per storm (column), -1 where none: an hour on
    US land right after an hour over water, so storms crossing the border
    over land don't count."""
    on = land(lat_h, lon_h)
    off = land.water(lat_h, lon_h)
    cross = on[1:] & off[:-1]
    has = cross.any(axis=0)
    return np.where(has, np.argmax(cross, axis=0) + 1, -1)


def storm_rows(lat, lon, wind, year, land: USLand, source: str, unit: str) -> pd.DataFrame:
    """One row per (member, storm): North Atlantic TS flag, lifetime peak, and
    US landfall wind, region, 24 h approach peak and 12/24 h decay.

    lat, lon: (node, storm); wind: (member, node, storm) in kt."""
    lon = np.where(lon > 180, lon - 360, lon)
    n_m, _, n_s = wind.shape
    na = ((wind >= TS) & ce.in_north_atlantic(lat, lon)[None] & np.isfinite(lat)[None]).any(axis=1)
    peak = np.nanmax(np.where(np.isfinite(wind), wind, -np.inf), axis=1)
    lat_h, lon_h = hourly(lat), wrap(hourly(unwrap(lon)))
    lf = landfalls(lat_h, lon_h, land)
    cols = np.nonzero(lf >= 0)[0]
    w_h = hourly(np.moveaxis(wind[:, :, cols], 0, 1))  # hour, member, storm
    rows = {
        'source': source,
        'unit': unit,
        'member': np.repeat(np.arange(n_m), n_s),
        'storm': np.tile(np.arange(n_s), n_m),
        'year': np.tile(year, n_m),
        'na_ts': na.ravel(),
        'peak': peak.ravel(),
    }
    df = pd.DataFrame(rows)
    for k in ('lf_wind', 'lf_pre24', 'lf_post12', 'lf_post24', 'lf_lat', 'lf_lon'):
        df[k] = np.nan
    df['lf_region'] = None
    n_h = w_h.shape[0]
    for c, s in enumerate(cols):
        h = lf[s]
        reg = region(lat_h[h, s], lon_h[h, s])
        w = w_h[:, :, c]  # hour, member
        at = w[h]
        pre = np.nanmax(
            np.where(np.isfinite(w[max(h - 24, 0) : h + 1]), w[max(h - 24, 0) : h + 1], -np.inf),
            axis=0,
        )
        p12 = w[h + 12] if h + 12 < n_h else np.full(n_m, np.nan)
        p24 = w[h + 24] if h + 24 < n_h else np.full(n_m, np.nan)
        idx = np.arange(n_m) * n_s + s
        df.loc[idx, 'lf_wind'] = at
        df.loc[idx, 'lf_pre24'] = np.where(np.isfinite(pre), pre, np.nan)
        df.loc[idx, 'lf_post12'] = p12
        df.loc[idx, 'lf_post24'] = p24
        df.loc[idx, 'lf_region'] = reg
        df.loc[idx, 'lf_lat'] = lat_h[h, s]
        df.loc[idx, 'lf_lon'] = lon_h[h, s]
    # a member counts as a storm only where it carries wind; a landfall only
    # where the member is alive and at tropical-storm strength there
    df = df[np.isfinite(df['peak']) & (df['peak'] > 0)]
    df.loc[~(df['lf_wind'] >= TS), ['lf_region']] = None
    return df


def chaz_file(i: int, land: USLand) -> pd.DataFrame:
    ds = xr.open_dataset(ce.CACHE / 'tracks' / 'ERA5' / ce.track_file(i), decode_times=False)
    year = ds['year'].values
    sel = np.nonzero((year >= YEARS[0]) & (year <= YEARS[1]))[0]
    lat = ds['latitude'].values[:, sel]
    lon = ds['longitude'].values[:, sel]
    wind = ds['Mwspd'].values[:, :, sel]
    fill = ds['time'].values[:, sel] <= ce.TIME_FILL
    lat = np.where(fill, np.nan, lat)
    lon = np.where(fill, np.nan, lon)
    wind = np.where(fill[None] | ~np.isfinite(lat)[None], np.nan, wind)
    return storm_rows(lat, lon, wind, year[sel], land, 'CHAZ ERA5', f'ens{i:03d}')


def observed(land: USLand) -> pd.DataFrame:
    d = xr.open_dataset(IBTRACS)
    season = d['season'].values
    keep = np.nonzero((season >= YEARS[0]) & (season <= YEARS[1]))[0]
    t = pd.to_datetime(
        d['iso_time'].values[keep].astype(str).ravel(), errors='coerce'
    ).values.reshape(keep.size, -1)
    hour = pd.DatetimeIndex(t.ravel()).hour.values.reshape(t.shape)
    minute = pd.DatetimeIndex(t.ravel()).minute.values.reshape(t.shape)
    syn = (np.isin(hour, [0, 6, 12, 18])) & (minute == 0)
    lat_all = d['lat'].values[keep]
    lon_all = d['lon'].values[keep]
    w_all = d['usa_wind'].values[keep].astype(float)
    n_nodes = int(syn.sum(axis=1).max())
    lat = np.full((n_nodes, keep.size), np.nan)
    lon = np.full((n_nodes, keep.size), np.nan)
    wind = np.full((1, n_nodes, keep.size), np.nan)
    for s in range(keep.size):
        j = np.nonzero(syn[s])[0]
        lat[: j.size, s] = lat_all[s, j]
        lon[: j.size, s] = lon_all[s, j]
        wind[0, : j.size, s] = w_all[s, j]
    return storm_rows(lat, lon, wind, season[keep], land, 'observed', 'ibtracs')


def summarize(df: pd.DataFrame, member_years: float, scale: float) -> dict:
    na = df[df['na_ts']]
    lf = df[df['lf_region'].notna()]
    big = lf[lf['lf_wind'] >= HU]
    out = {
        'NA TS storms/yr': len(na) / member_years * scale,
        'NA share peak >= 64 kt': (na['peak'] >= HU).mean(),
        'NA share peak >= 96 kt': (na['peak'] >= MH).mean(),
        'US landfalls/yr (>= 34 kt)': len(lf) / member_years * scale,
        'US landfalls/yr >= 64 kt': len(big) / member_years * scale,
        'US landfalls/yr >= 96 kt': (lf['lf_wind'] >= MH).sum() / member_years * scale,
        'landfall share of NA TS storms': len(lf[lf['na_ts']]) / max(len(na), 1),
        'share of landfalls >= 64 kt': (lf['lf_wind'] >= HU).mean(),
        'share of landfalls >= 96 kt': (lf['lf_wind'] >= MH).mean(),
        'landfall wind median (kt)': lf['lf_wind'].median(),
        'peak of landfallers median (kt)': lf['peak'].median(),
        'landfallers peak >= 96 kt share': (lf['peak'] >= MH).mean(),
        'landfall / 24 h approach peak, median': (lf['lf_wind'] / lf['lf_pre24']).median(),
        '12 h after / at landfall (>= 64 kt), median': (big['lf_post12'] / big['lf_wind']).median(),
        '24 h after / at landfall (>= 64 kt), median': (big['lf_post24'] / big['lf_wind']).median(),
    }
    for r in REGIONS[:-1]:
        sub = lf[lf['lf_region'] == r]
        out[f'{r} landfalls/yr'] = len(sub) / member_years * scale
        out[f'{r} landfalls/yr >= 64 kt'] = (sub['lf_wind'] >= HU).sum() / member_years * scale
    return out


def cmd_run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    land = USLand()
    years = YEARS[1] - YEARS[0] + 1
    obs = observed(land)
    files = ce.parse_range(args.files)
    chaz = pd.concat([chaz_file(i, land) for i in files], ignore_index=True)
    n_members = chaz.groupby('unit')['member'].nunique().sum()
    chaz_my = n_members * years
    obs.to_parquet(OUT / 'observed.parquet')
    chaz.to_parquet(OUT / 'chaz.parquet')
    o = summarize(obs, years, 1.0)
    raw = summarize(chaz, chaz_my, 1.0)
    k = o['NA TS storms/yr'] / raw['NA TS storms/yr']
    c = summarize(chaz, chaz_my, k)
    tab = pd.DataFrame({'observed': o, 'CHAZ (raw)': raw, f'CHAZ (x{k:.2f}, NA TS matched)': c})
    tab['CHAZ / observed'] = tab.iloc[:, 2] / tab['observed']
    tab.to_csv(OUT / 'summary.csv')
    pd.set_option('display.width', 200)
    print(
        f'{YEARS[0]}-{YEARS[1]}: IBTrACS (usa) against CHAZ ERA5 files {args.files} '
        f'({chaz_my:,} member-years)'
    )
    print(tab.round(3).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('run', help='track-space comparison with IBTrACS')
    q.add_argument('--files', default='0-9')
    q.set_defaults(func=cmd_run)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
