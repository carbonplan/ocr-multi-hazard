"""CHAZ track sets against IBTrACS around the globe, year by year.

For each source, one row per year (1981-2019) of counts, summed over the source's members:

  - landfalls by country: the first hour on a country's land right after an hour over sea, on
    the Natural Earth 1:10m country raster at 1/12 degree, counted at the member's wind in that
    hour, at 34, 64 and 96 kt
  - 6-hourly nodes over sea at 64 and 96 kt, in REGIONS and, for North Atlantic storms, in the
    North Atlantic's NA_REGIONS
  - storms first reaching 34 kt, by where they reach it
  - North Atlantic tropical storms and their peaks, US landfalls on the Atlantic and Gulf coasts
    (in all, and at 64 kt by coastal region) and on the Pacific coast, by landfall.storm_rows

A source is IBTrACS (USA_WIND, one member), the CHAZ team's ERA5 track set ('chaz-team', 10
files of 40 members), or one of our ERA5 runs (10 realizations of 40 members). A storm belongs to
the calendar year of its first record: its CHAZ file's year, IBTrACS's first synoptic record, so
Southern Hemisphere seasons split as CHAZ's do. `members` is the member count of each year, so a
rate is a count over members.

`metrics` turns a span of years into rates, scaling the North Atlantic counts so each source's
North Atlantic tropical-storm count matches the observed one; `score` is the mean |ln(source /
IBTrACS)| over SCORED; `bootstrap` resamples whole years, the same years for every source.

    python -m ocr.risks.hurricane.validate task --source <run name | ibtracs | chaz-team>
    python -m ocr.risks.hurricane.validate launch --runs <run name>,<run name>,...
"""

from __future__ import annotations

import argparse
import io
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from ocr.risks.hurricane import batch, landfall, store

YEARS = landfall.YEARS
OUT = 'validation/chaz/validate'
RES = 1 / 12
THRESHOLDS = (34, 64, 96)
COUNTRIES = (
    'USA', 'MEX', 'CUB', 'PHL', 'CHN', 'TWN', 'JPN', 'KOR', 'VNM', 'IND', 'BGD', 'MMR', 'AUS',
    'MDG', 'MOZ',
)  # fmt: skip
REGIONS = {  # (lat0, lat1, lon0, lon1), longitude 0-360
    'WP Philippine Sea': (5, 25, 125, 160),
    'WP South China Sea': (5, 25, 105, 125),
    'WP East China Sea/Japan': (25, 35, 120, 145),
    'WP north of 35N': (35, 60, 100, 180),
    'EP main': (5, 25, 180, 280),
    'EP north of 25N': (25, 40, 200, 250),
    'NI Bay of Bengal': (5, 23, 80, 100),
    'NI Arabian Sea': (5, 25, 50, 78),
    'SI west': (-30, -5, 30, 90),
    'AU': (-30, -5, 90, 160),
    'SP': (-30, -5, 160, 240),
    'SH south of 30S': (-60, -30, 0, 360),
    'NA tropics': (5, 25, 260, 345),
    'NA north of 35N': (35, 60, 260, 360),
}
NA_REGIONS = ('Gulf', 'Caribbean', 'W Atlantic', 'MDR', 'Subtropical', 'north of 45N')
SCORED = (
    [f'Atl+Gulf landfalls >= {t} kt' for t in THRESHOLDS]
    + ['NA share peak >= 64 kt', 'NA share peak >= 96 kt']
    + [f'NA time >= {t}: {r}' for t in (64, 96) for r in NA_REGIONS[:-1]]
    + [
        f'time {r} >= {t}'
        for t in (64, 96)
        for r in (
            'WP Philippine Sea',
            'WP South China Sea',
            'WP East China Sea/Japan',
            'EP main',
            'NI Bay of Bengal',
            'NI Arabian Sea',
            'SI west',
            'AU',
            'SP',
        )  # fmt: skip
    ]
    + [f'landfall {c} >= 64' for c in ('PHL', 'CHN', 'TWN', 'JPN', 'VNM', 'MEX', 'AUS', 'IND')]
)
FLOOR = 0.05  # the ratio a source scores when it has none of something observed


def country_raster(res: float = RES) -> np.ndarray:
    """Global raster from 90N and 180W: 0 sea, 1 a country not in COUNTRIES, 2 + its index."""
    import cartopy.io.shapereader as shpreader
    from rasterio import features
    from rasterio.transform import from_origin

    reader = shpreader.Reader(shpreader.natural_earth('10m', 'cultural', 'admin_0_countries'))
    shapes = []
    for r in reader.records():
        a3 = r.attributes['ADM0_A3']
        shapes.append((r.geometry, COUNTRIES.index(a3) + 2 if a3 in COUNTRIES else 1))
    return features.rasterize(
        shapes,
        out_shape=(round(180 / res), round(360 / res)),
        transform=from_origin(-180, 90, res, res),
        fill=0,
        dtype='int16',
    )


def lookup(grid: np.ndarray, lat, lon) -> np.ndarray:
    """The raster's code at each point; -1 where the point is missing."""
    res = 180 / grid.shape[0]
    ok = np.isfinite(lat) & np.isfinite(lon)
    i = np.clip(np.floor((90 - np.where(ok, lat, 0)) / res).astype(int), 0, grid.shape[0] - 1)
    j = np.floor((landfall.wrap(np.where(ok, lon, 0)) + 180) / res).astype(int) % grid.shape[1]
    out = grid[i, j].astype(int)
    out[~ok] = -1
    return out


def na_region(lat, lon) -> np.ndarray:
    """North Atlantic subregion of each point, '' outside the North Atlantic; later rules win."""
    lon = landfall.wrap(lon)
    r = np.full(np.shape(lat), '', object)
    na = landfall.in_north_atlantic(lat, lon) & np.isfinite(lat)
    r[na & (lat >= 45)] = 'north of 45N'
    r[na & (lat >= 25) & (lat < 45) & (lon >= -60) & (lon < -20)] = 'Subtropical'
    r[na & (lat >= 5) & (lat < 25) & (lon >= -60) & (lon < -15)] = 'MDR'
    r[na & (lat >= 22) & (lat < 45) & (lon >= -82) & (lon < -60)] = 'W Atlantic'
    r[na & (lat >= 9) & (lat < 22) & (lon >= -89) & (lon < -60)] = 'Caribbean'
    r[na & (lat >= 18) & (lat < 31) & (lon >= -98) & (lon < -80.5)] = 'Gulf'
    return r


def in_region(lat, lon, box) -> np.ndarray:
    la0, la1, lo0, lo1 = box
    lon360 = np.mod(lon, 360)
    return (lat >= la0) & (lat < la1) & (lon360 >= lo0) & (lon360 < lo1)


def counts(lat, lon, wind, grid: np.ndarray, land) -> dict:
    """Counts over all members of one set of storms: lat, lon (node, storm) 6-hourly, wind
    (member, node, storm) in kt; `land` is a landfall.USLand."""
    n_m = wind.shape[0]
    out = {'members': n_m}
    lat_h, lon_h = landfall.hourly(lat), landfall.wrap(landfall.hourly(landfall.unwrap(lon)))
    code = lookup(grid, lat_h, lon_h)
    w_h = landfall.hourly(np.moveaxis(wind, 0, 1))  # hour, member, storm
    prev = np.vstack([np.full((1, code.shape[1]), -1), code[:-1]])
    entry = (prev == 0) & (code >= 2)
    for c, a3 in enumerate(COUNTRIES, start=2):
        e = entry & (code == c)
        s = np.nonzero(e.any(axis=0))[0]
        w = w_h[np.argmax(e, axis=0)[s], :, s] if s.size else np.empty((0, n_m))
        for t in THRESHOLDS:
            with np.errstate(invalid='ignore'):
                out[f'landfall {a3} >= {t}'] = int((w >= t).sum())
    sea = lookup(grid, lat, lon) == 0
    for name, box in REGIONS.items():
        here = in_region(lat, lon, box) & sea
        for t in (64, 96):
            with np.errstate(invalid='ignore'):
                out[f'time {name} >= {t}'] = int(((wind >= t) & here[None]).sum())
    with np.errstate(invalid='ignore'):
        ts = wind >= 34
    m, s = np.nonzero(ts.any(axis=1))
    node = np.argmax(ts, axis=1)[m, s]
    first_lat, first_lon = lat[node, s], lon[node, s]
    out['TS formed global'] = int(m.size)
    for name, box in REGIONS.items():
        out[f'TS formed {name}'] = int(in_region(first_lat, first_lon, box).sum())

    rows = landfall.storm_rows(lat, lon, wind, np.zeros(lat.shape[1], int), land, 'v', 'u')
    pacific = rows['crosses'] & (
        landfall.region(rows['lf_lat'].values, rows['lf_lon'].values) == 'Pacific'
    )
    us = rows['lf_region'].notna() & ~pacific
    w = rows['lf_wind']
    out['NA TS'] = int(rows['na_ts'].sum())
    for t in (64, 96):
        out[f'NA peak >= {t}'] = int((rows['na_ts'] & (rows['peak'] >= t)).sum())
    for t in THRESHOLDS:
        out[f'Atl+Gulf landfalls >= {t} kt'] = int((us & (w >= t)).sum())
    for r in landfall.ATLANTIC:
        out[f'{r} landfalls >= 64 kt'] = int((us & (rows['lf_region'] == r) & (w >= 64)).sum())
    out['Pacific coast landfalls >= 34 kt'] = int((pacific & (w >= 34)).sum())

    na_ts = (ts & landfall.in_north_atlantic(lat, lon)[None] & np.isfinite(lat)[None]).any(axis=1)
    reg = na_region(lat, lon)
    for t in (64, 96):
        with np.errstate(invalid='ignore'):
            hit = (wind >= t) & na_ts[:, None, :] & sea[None]
        for r in NA_REGIONS:
            out[f'NA time >= {t}: {r}'] = int((hit & (reg == r)[None]).sum())
    return out


_grid, _land = None, None


def _init() -> None:
    global _grid, _land
    _grid, _land = country_raster(), landfall.USLand()


def _by_year(lat, lon, wind, year, years) -> list[dict]:
    """counts for each of `years`, the years the set covers: a year with no storms counts its
    members and nothing else."""
    rows = []
    for y in years:
        s = year == y
        if not s.any():
            rows.append({'year': y, 'members': wind.shape[0]})
            continue
        rows.append({'year': y, **counts(lat[:, s], lon[:, s], wind[..., s], _grid, _land)})
    return rows


def _run_file(job: tuple[str, int]) -> list[dict]:
    key, file_year = job
    with xr.open_dataset(store.path(key), decode_times=False) as ds:
        lat, lon, wind, year = landfall.chaz_arrays(ds.load())
    return _by_year(lat, lon, wind, year, [file_year])


def _team_file(i: int) -> list[dict]:
    with store.s3().open(f'{landfall.PUBLISHED}/global_2019_2ens{i:03d}_pre.nc', 'rb') as f:
        ds = xr.open_dataset(f, engine='scipy', decode_times=False).load()
    return _by_year(*landfall.chaz_arrays(ds), range(YEARS[0], YEARS[1] + 1))


def _natural_earth() -> None:
    """Download the shapefiles once, before workers read them."""
    import cartopy.io.shapereader as shpreader

    shpreader.natural_earth('10m', 'cultural', 'admin_0_countries')
    shpreader.natural_earth('10m', 'physical', 'land')


def first_record_year(csv: bytes, seasons: tuple[int, int]) -> np.ndarray:
    """Calendar year of each track's first synoptic record, in the order of
    landfall.observed_arrays(csv, seasons)."""
    bt = pd.read_csv(
        io.BytesIO(csv), skiprows=[1], low_memory=False, usecols=['SID', 'SEASON', 'ISO_TIME']
    )
    t = pd.to_datetime(bt['ISO_TIME'])
    bt = bt[(t.dt.minute == 0) & (t.dt.hour % 6 == 0) & bt['SEASON'].between(*seasons)]
    return pd.to_datetime(bt.groupby('SID', sort=True)['ISO_TIME'].min()).dt.year.values


def annual(source: str, workers: int | None = None) -> pd.DataFrame:
    """Counts by year for `source`: 'ibtracs', 'chaz-team', or a run name under runs/era5."""
    if source == 'ibtracs':
        _init()
        csv = landfall.ibtracs_csv(store.s3())
        # a Southern Hemisphere season starts the July before its year: one more season, so
        # storms that start late in the last year are in
        seasons = (YEARS[0], YEARS[1] + 1)
        lat, lon, wind, _ = landfall.observed_arrays(csv, seasons)
        year = first_record_year(csv, seasons)
        if year.size != lat.shape[1]:
            raise ValueError(f'{year.size} first-record years for {lat.shape[1]} tracks')
        parts = [_by_year(lat, lon, wind, year, range(YEARS[0], YEARS[1] + 1))]
    else:
        _natural_earth()
        if source == 'chaz-team':
            # a file holds every year: fewer at once
            fn, jobs, workers = _team_file, range(10), workers or 5
        else:
            fn = _run_file
            jobs = [
                (f'{store.RUNS}/{source}/{y}_r{r:03d}.nc', y)
                for r in range(10)
                for y in range(YEARS[0], YEARS[1] + 1)
            ]
            for key, _ in jobs:
                store.path(key)
        with ProcessPoolExecutor(workers or os.cpu_count(), initializer=_init) as pool:
            parts = list(pool.map(fn, jobs))
    df = pd.DataFrame([row for part in parts for row in part])
    return df.groupby('year').sum().fillna(0)


def key(source: str) -> str:
    return f'{OUT}/annual_{source}.parquet'


def load(source: str) -> pd.DataFrame:
    return pd.read_parquet(store.path(key(source)))


def task(source: str) -> None:
    df = annual(source)
    local = store.CACHE / key(source)
    local.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local)
    store.push(keys=[key(source)], fragment=f'validate-{source}', replace=True)
    print(source, df[['members', 'NA TS', 'TS formed global']].sum().to_dict(), flush=True)


def launch(sources: list[str], vm_type: str = 'm8g.4xlarge') -> None:
    batch.run(
        'validate',
        ['task'],
        name=f'chaz-validate-{len(sources)}',
        over=sources,
        vm_type=vm_type,
        max_workers=len(sources),
    )


def metrics(tables: dict[str, pd.DataFrame], years) -> pd.DataFrame:
    """Rates over `years` (a year may repeat), one column per source. `tables` maps a label to
    `annual` counts and must include 'IBTrACS'. Per year: storms reaching 34 kt, time over sea in
    REGIONS, landfalls by country; scaled to the observed North Atlantic tropical-storm count:
    Atlantic and Gulf landfalls, North Atlantic time over sea, Pacific-coast landfalls; and the
    North Atlantic shares of storms peaking at 64 and 96 kt."""
    years = list(years)
    sums = {label: t.loc[years].sum() for label, t in tables.items()}
    obs = sums['IBTrACS']
    out = {}
    for label, s in sums.items():
        rate = s / s['members']  # members summed over years: member-years
        k = (obs['NA TS'] / obs['members']) / (s['NA TS'] / s['members'])
        m = {'NA TS storms/yr': rate['NA TS']}
        for c in s.index:
            if c.startswith(('landfall ', 'time ', 'TS formed ')):
                m[c] = rate[c]
            elif c.startswith(('Atl+Gulf ', 'NA time ', 'Pacific ')) or c in (
                f'{r} landfalls >= 64 kt' for r in landfall.ATLANTIC
            ):
                m[c] = rate[c] * k
        for t in (64, 96):
            m[f'NA share peak >= {t} kt'] = s[f'NA peak >= {t}'] / s['NA TS']
        out[label] = m
    return pd.DataFrame(out)


def score(table: pd.DataFrame) -> pd.Series:
    """Mean |ln(source / IBTrACS)| over the SCORED metrics IBTrACS has; a source's zero counts
    as FLOOR."""
    obs = table.loc[SCORED, 'IBTrACS']
    ratio = table.loc[SCORED].drop(columns='IBTrACS').div(obs.where(obs > 0), axis=0)
    return np.abs(np.log(ratio.clip(lower=FLOOR))).mean()


def bootstrap(tables: dict[str, pd.DataFrame], years, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Scores of `n` resamples of whole years, the same years for every source. Every SCORED
    metric must be observed over `years`; a resample without it leaves it out."""
    years = list(years)
    obs = metrics(tables, years).loc[SCORED, 'IBTrACS']
    if (obs <= 0).any():
        raise ValueError(f'no observed events for {list(obs[obs <= 0].index)}')
    rng = np.random.default_rng(seed)
    return pd.DataFrame([score(metrics(tables, rng.choice(years, len(years)))) for _ in range(n)])


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('task')
    t.add_argument('--source', help='default: $COILED_BATCH_TASK_INPUT')
    la = sub.add_parser('launch')
    la.add_argument('--runs', required=True, help='comma-separated run names under runs/era5')
    la.add_argument('--vm-type', default='m8g.4xlarge')
    a = p.parse_args()
    if a.cmd == 'task':
        task(a.source or os.environ['COILED_BATCH_TASK_INPUT'])
    else:
        launch(['ibtracs', 'chaz-team', *a.runs.split(',')], a.vm_type)


if __name__ == '__main__':
    main()
