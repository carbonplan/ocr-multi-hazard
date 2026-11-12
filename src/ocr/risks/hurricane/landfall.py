"""US landfall statistics of CHAZ tracks against IBTrACS, in track space with no wind model.

The rules of chaz_trackdiag.py (Shane98c/chaz-local), so its published-track numbers can be
reproduced and our runs compared on the same footing:

  - CHAZ and IBTrACS (USA_WIND, synoptic 6-hourly records, season 1981-2019) nodes are
    interpolated to hourly, longitudes unwrapped across the date line
  - one US land mask: Natural Earth 1:10m, rasterized at 1/24 degree
  - US landfall: the first hour on US land right after an hour over water; each storm counts
    once, at its wind in that hour (and, in one row, at its strongest crossing)
  - a (storm, member) counts only where it carries wind, a landfall only at 34 kt or more
  - CHAZ rates are also given scaled so its North Atlantic tropical-storm count matches the
    observed one, which leaves where storms go and how strong they are

  run    tables for IBTrACS, the published ERA5 tracks and our runs, under validation/chaz/landfall
"""

from __future__ import annotations

import argparse
import io
import os

import numpy as np
import pandas as pd
import xarray as xr

from ocr.risks.hurricane import store

YEARS = (1981, 2019)
TS, HU, MH = 34.0, 64.0, 96.0  # kt
BOX = (-130.0, 15.0, -60.0, 52.0)  # lon0, lat0, lon1, lat1 of the land raster
RES = 1 / 24
REGIONS = ('TX-AL', 'FL', 'GA-Car', 'VA-ME', 'other')
TIME_FILL = -54786.0  # 1800-01-01 in days since 1950: CHAZ's marker for no data
PUBLISHED = os.environ.get(
    'CHAZ_PUBLISHED_TRACKS', 's3://carbonplan-ocr/ocr-explore/CHAZ/tracks/ERA5'
)
OUT = 'validation/chaz/landfall'


def wrap(lon):
    return (lon + 180) % 360 - 180


class USLand:
    """US land and any land on a 1/24 degree raster, looked up by cell."""

    def __init__(self):
        import cartopy.io.shapereader as shpreader
        import shapely

        lon0, lat0, lon1, lat1 = BOX
        self.lons = np.arange(lon0, lon1, RES) + RES / 2
        self.lats = np.arange(lat0, lat1, RES) + RES / 2
        gx, gy = np.meshgrid(self.lons, self.lats)
        countries = shpreader.Reader(
            shpreader.natural_earth('10m', 'cultural', 'admin_0_countries')
        )
        us = next(r.geometry for r in countries.records() if r.attributes['ADM0_A3'] == 'USA')
        land = shapely.union_all(
            list(shpreader.Reader(shpreader.natural_earth('10m', 'physical', 'land')).geometries())
        )
        land = shapely.clip_by_rect(land, lon0, lat0, lon1, lat1)
        shapely.prepare(us)
        shapely.prepare(land)
        self.mask = shapely.contains_xy(us, gx, gy)
        self.land = self.mask | shapely.contains_xy(land, gx, gy)

    def water(self, lat, lon):
        return self._lookup(~self.land, lat, lon, outside=True)

    def __call__(self, lat, lon):
        return self._lookup(self.mask, lat, lon, outside=False)

    def _lookup(self, grid, lat, lon, outside: bool):
        lon = wrap(lon)
        ok = np.isfinite(lat) & np.isfinite(lon)
        i = np.floor((np.where(ok, lat, 0) - BOX[1]) / RES).astype(int)
        j = np.floor((np.where(ok, lon, 0) - BOX[0]) / RES).astype(int)
        inside = ok & (i >= 0) & (i < self.lats.size) & (j >= 0) & (j < self.lons.size)
        out = np.full(np.shape(lat), outside) & ok
        out[inside] = grid[i[inside], j[inside]]
        return out


def in_north_atlantic(lat, lon) -> np.ndarray:
    """Atlantic side of the Central American divide, 5-60N."""
    lon = wrap(lon)
    atl = (lat > 5) & (lat < 60) & (lon > -100) & (lon < 0)
    pacific = ((lon < -84) & (lat < 9)) | ((lon < -90) & (lat < 15)) | ((lon < -98) & (lat < 18))
    return atl & ~pacific


def region(lat, lon) -> np.ndarray:
    """Coastal regions of the landfall point."""
    lon = wrap(lon)
    out = np.full(np.shape(lat), 'other', dtype=object)
    out[(lon >= -82) & (lat >= 36.6)] = 'VA-ME'
    out[(lon >= -87.6) & (lat >= 31) & (lat < 36.6)] = 'GA-Car'
    out[(lon >= -87.6) & (lat < 31)] = 'FL'
    out[(lon < -87.6) & (lat < 33)] = 'TX-AL'
    return out


def unwrap(lon: np.ndarray) -> np.ndarray:
    """Longitudes along axis 0 made continuous across the date line and 0/360."""
    step = (np.diff(lon, axis=0) + 180) % 360 - 180
    out = lon.copy()
    out[1:] = lon[:1] + np.cumsum(np.nan_to_num(step), axis=0)
    return np.where(np.isfinite(lon), out, np.nan)


def hourly(x: np.ndarray) -> np.ndarray:
    """6-hourly nodes along axis 0 to hourly, linearly; NaN next to a missing node."""
    n = x.shape[0]
    h = np.arange((n - 1) * 6 + 1)
    i0 = np.minimum(h // 6, n - 2)
    f = ((h - 6 * i0) / 6.0).reshape((-1,) + (1,) * (x.ndim - 1))
    out = x[i0] * (1 - f) + x[i0 + 1] * f
    out = np.where(f == 0, x[i0], out)
    return np.where(f == 1, x[i0 + 1], out)


def crossings(lat_h, lon_h, land: USLand) -> np.ndarray:
    """(hour, storm) flags: an hour on US land right after an hour over water."""
    on = land(lat_h, lon_h)
    off = land.water(lat_h, lon_h)
    out = np.zeros(on.shape, bool)
    out[1:] = on[1:] & off[:-1]
    return out


def storm_rows(lat, lon, wind, year, land: USLand, source: str, unit: str) -> pd.DataFrame:
    """One row per (member, storm): North Atlantic TS flag, lifetime peak, and US landfall
    wind, region, 24 h approach peak and 12/24 h decay. lat, lon (node, storm); wind (member,
    node, storm) in kt."""
    lon = wrap(lon)
    n_m, _, n_s = wind.shape
    na = ((wind >= TS) & in_north_atlantic(lat, lon)[None] & np.isfinite(lat)[None]).any(axis=1)
    with np.errstate(all='ignore'):
        peak = np.nanmax(np.where(np.isfinite(wind), wind, -np.inf), axis=1)
    lat_h, lon_h = hourly(lat), wrap(hourly(unwrap(lon)))
    cross = crossings(lat_h, lon_h, land)
    lf = np.where(cross.any(axis=0), np.argmax(cross, axis=0), -1)
    cols = np.nonzero(lf >= 0)[0]
    w_h = hourly(np.moveaxis(wind[:, :, cols], 0, 1))  # hour, member, storm
    df = pd.DataFrame(
        {
            'source': source,
            'unit': unit,
            'member': np.repeat(np.arange(n_m), n_s),
            'storm': np.tile(np.arange(n_s), n_m),
            'year': np.tile(year, n_m),
            'na_ts': na.ravel(),
            'peak': peak.ravel(),
        }
    )
    for k in ('lf_wind', 'lf_max_wind', 'lf_pre24', 'lf_post12', 'lf_post24', 'lf_lat', 'lf_lon'):
        df[k] = np.nan
    df['lf_region'] = None
    n_h = w_h.shape[0]
    for c, s in enumerate(cols):
        h = lf[s]
        w = w_h[:, :, c]  # hour, member
        win = w[max(h - 24, 0) : h + 1]
        with np.errstate(all='ignore'):
            pre = np.nanmax(np.where(np.isfinite(win), win, -np.inf), axis=0)
            at_cross = w[cross[:, s]]
            strongest = np.nanmax(np.where(np.isfinite(at_cross), at_cross, -np.inf), axis=0)
        idx = np.arange(n_m) * n_s + s
        df.loc[idx, 'lf_wind'] = w[h]
        df.loc[idx, 'lf_max_wind'] = strongest
        df.loc[idx, 'lf_pre24'] = np.where(np.isfinite(pre), pre, np.nan)
        df.loc[idx, 'lf_post12'] = w[h + 12] if h + 12 < n_h else np.nan
        df.loc[idx, 'lf_post24'] = w[h + 24] if h + 24 < n_h else np.nan
        df.loc[idx, 'lf_region'] = region(lat_h[h, s], lon_h[h, s])
        df.loc[idx, 'lf_lat'] = lat_h[h, s]
        df.loc[idx, 'lf_lon'] = lon_h[h, s]
    df = df[np.isfinite(df['peak']) & (df['peak'] > 0)]
    df.loc[~(df['lf_wind'] >= TS), ['lf_region']] = None
    return df


def chaz_rows(ds: xr.Dataset, land: USLand, source: str, unit: str) -> pd.DataFrame:
    """storm_rows for the storms of a CHAZ track file in YEARS."""
    year = ds['year'].values
    sel = np.nonzero((year >= YEARS[0]) & (year <= YEARS[1]))[0]
    lat = ds['latitude'].values[:, sel]
    lon = ds['longitude'].values[:, sel]
    wind = ds['Mwspd'].transpose('ensembleNum', 'lifelength', 'stormID').values[:, :, sel]
    fill = ds['time'].values[:, sel] <= TIME_FILL
    lat = np.where(fill, np.nan, lat)
    lon = np.where(fill, np.nan, lon)
    wind = np.where(fill[None] | ~np.isfinite(lat)[None], np.nan, wind)
    return storm_rows(lat, lon, wind, year[sel], land, source, unit)


def observed(land: USLand, csv: bytes) -> pd.DataFrame:
    """IBTrACS since-1980 CSV, every track, synoptic records only, by season."""
    bt = pd.read_csv(
        io.BytesIO(csv),
        skiprows=[1],
        low_memory=False,
        usecols=['SID', 'SEASON', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND'],
    )
    t = pd.to_datetime(bt['ISO_TIME'])
    bt = bt[(t.dt.minute == 0) & (t.dt.hour % 6 == 0) & bt['SEASON'].between(*YEARS)]
    bt = bt.assign(USA_WIND=pd.to_numeric(bt['USA_WIND'], errors='coerce'))
    groups = [g for _, g in bt.groupby('SID', sort=True)]
    n, life = len(groups), max(len(g) for g in groups)
    lat = np.full((life, n), np.nan)
    lon = np.full((life, n), np.nan)
    wind = np.full((1, life, n), np.nan)
    season = np.zeros(n, int)
    for k, g in enumerate(groups):
        lat[: len(g), k], lon[: len(g), k], wind[0, : len(g), k] = g['LAT'], g['LON'], g['USA_WIND']
        season[k] = g['SEASON'].iloc[0]
    return storm_rows(lat, lon, wind, season, land, 'IBTrACS', 'ibtracs')


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
        'US landfalls/yr >= 64 kt at the strongest crossing': (df['lf_max_wind'] >= HU).sum()
        / member_years
        * scale,
        'landfall share of NA TS storms': len(lf[lf['na_ts']]) / max(len(na), 1),
        'share of landfalls >= 64 kt': (lf['lf_wind'] >= HU).mean(),
        'share of landfalls >= 96 kt': (lf['lf_wind'] >= MH).mean(),
        'landfall wind median (kt)': lf['lf_wind'].median(),
        'peak of landfallers median (kt)': lf['peak'].median(),
        'landfall / 24 h approach peak, median': (lf['lf_wind'] / lf['lf_pre24']).median(),
        '12 h after / at landfall (>= 64 kt), median': (big['lf_post12'] / big['lf_wind']).median(),
    }
    for r in REGIONS[:-1]:
        sub = lf[lf['lf_region'] == r]
        out[f'{r} landfalls/yr >= 64 kt'] = (sub['lf_wind'] >= HU).sum() / member_years * scale
    return out


def compare(obs: pd.DataFrame, runs: dict[str, tuple[pd.DataFrame, int]]) -> pd.DataFrame:
    """Observed against each CHAZ set, raw and scaled to the observed NA storm count.
    runs: name -> (rows, member count)."""
    years = YEARS[1] - YEARS[0] + 1
    o = summarize(obs, years, 1.0)
    cols = {'IBTrACS': o}
    for name, (rows, members) in runs.items():
        raw = summarize(rows, members * years, 1.0)
        k = o['NA TS storms/yr'] / raw['NA TS storms/yr']
        cols[f'{name} (raw)'] = raw
        cols[f'{name} (x{k:.2f})'] = summarize(rows, members * years, k)
        cols[f'{name} / IBTrACS'] = {
            key: v / o[key] if o[key] else np.nan for key, v in cols[f'{name} (x{k:.2f})'].items()
        }
    return pd.DataFrame(cols)


def run(tcgi_name: str = 'TCGI_CRH_PI') -> pd.DataFrame:
    fs = store.s3()
    land = USLand()
    csv_key = sorted(
        fs.glob(f'{store.S3_ROOT}/inputs/ibtracs-v04r01/*/ibtracs.since1980.list.v04r01.csv')
    )[-1]
    obs = observed(land, fs.cat(csv_key))
    print('IBTrACS', csv_key.rsplit('/', 2)[-2], len(obs), 'storms', flush=True)

    pub = []
    for i in range(10):
        with fs.open(f'{PUBLISHED}/global_2019_2ens{i:03d}_pre.nc', 'rb') as f:
            ds = xr.open_dataset(f, engine='scipy', decode_times=False).load()
        pub.append(chaz_rows(ds, land, 'published', f'ens{i:03d}'))
        print('published file', i, flush=True)
    pub = pd.concat(pub, ignore_index=True)

    ours = []
    for r in range(10):
        for year in range(YEARS[0], YEARS[1] + 1):
            key = f'{store.RUNS}/{tcgi_name}/{year}_r{r:03d}.nc'
            with xr.open_dataset(store.path(key), decode_times=False) as ds:
                rows = chaz_rows(ds.load(), land, 'ours', f'r{r:03d}')
            rows['storm'] = rows['storm'] + year * 10_000
            ours.append(rows)
        print('our realization', r, flush=True)
    ours = pd.concat(ours, ignore_index=True)

    members = {
        name: int(rows.groupby('unit')['member'].nunique().sum())
        for name, rows in (('published', pub), ('ours', ours))
    }
    tab = compare(obs, {'published': (pub, members['published']), 'ours': (ours, members['ours'])})
    local = store.CACHE / OUT
    local.mkdir(parents=True, exist_ok=True)
    tab.to_csv(local / f'summary_{tcgi_name}.csv')
    for name, rows in (('ibtracs', obs), ('published', pub), (f'ours_{tcgi_name}', ours)):
        rows.to_parquet(local / f'rows_{name}.parquet')
    store.push(OUT, fragment=f'landfall-{tcgi_name}')
    return tab


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--tcgi', default='TCGI_CRH_PI', choices=['TCGI_CRH_PI', 'TCGI_SD_PI'])
    a = p.parse_args()
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 20)
    print(run(a.tcgi).round(3).to_string())


if __name__ == '__main__':
    main()
