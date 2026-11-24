"""CHAZ intensity members weighted by their first landfall, to match observed landfall rates.

A member's first landfall is the first hour its track enters land from sea on validate's country
raster with the member's wind interpolated to that hour at 34 kt or more. It has:

  - a zone: the country (validate.COUNTRIES, the US split by landfall.region into its Atlantic
    coastal regions, the Pacific coast and 'other'), or 'other land' for any other country
  - a basin: where the landfall point is (`basin`)
  - a class: the member's wind there, 34-64, 64-96 or >= 96 kt

`weights` sets one weight per (key, class), where the key is the zone, the basin, or the zone
with 'other land' split by basin (SCHEMES): the observed rate of first landfalls over the fit
years divided by CHAZ's, shrunk toward 1 as (n_obs + k) / (chaz_expected + k), with
chaz_expected CHAZ's rate times the observed member-years. Members with no landfall at 34 kt
weigh 1. A weight applies to everything its member counts toward.

`task` writes, for a run or IBTrACS, each member's first landfall (records_{source}.parquet) and
validate.counts summed over the members of each (zone, basin, class) group, year by year
(grouped_{source}.parquet). Any weights of those groups then give the weighted validate.annual
table (`weighted_annual`) without another pass over the tracks.

    python -m ocr.risks.hurricane.reweight task --source <run name | ibtracs>
    python -m ocr.risks.hurricane.reweight launch --run <run name>
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from ocr.risks.hurricane import batch, landfall, store, validate

OUT = 'validation/chaz/reweight'
CLASSES = ((34.0, '34-64'), (64.0, '64-96'), (96.0, '>=96'))
GROUP = ['zone', 'basin', 'cls']
NONE = ''  # zone, basin and class of members with no landfall at 34 kt
SCHEMES = ('zone', 'zone, other land by basin', 'basin')
K = 2.0


def basin(lat, lon) -> np.ndarray:
    """NA, EP, WP, NI north of the equator; SI, AU, SP south of it, split at 90E and 160E;
    'other' for the rest of the Northern Hemisphere outside the North Atlantic."""
    lat = np.asarray(lat, float)
    lon360 = np.mod(lon, 360)
    out = np.full(lat.shape, 'other', object)
    nh = lat >= 0
    out[nh & (lon360 >= 30) & (lon360 < 100)] = 'NI'
    out[nh & (lon360 >= 100) & (lon360 < 180)] = 'WP'
    out[nh & (lon360 >= 180) & (lon360 < 300)] = 'EP'
    out[landfall.in_north_atlantic(lat, lon)] = 'NA'
    out[~nh & (lon360 < 90)] = 'SI'
    out[~nh & (lon360 >= 90) & (lon360 < 160)] = 'AU'
    out[~nh & (lon360 >= 160)] = 'SP'
    return out


def zone(code, lat, lon) -> np.ndarray:
    """Zone of landfall points with raster `code` (1 other land, 2 + index in COUNTRIES)."""
    code = np.asarray(code)
    out = np.full(code.shape, 'other land', object)
    for c, a3 in enumerate(validate.COUNTRIES, start=2):
        out[code == c] = a3
    us = code == validate.COUNTRIES.index('USA') + 2
    out[us] = 'USA ' + landfall.region(np.asarray(lat)[us], np.asarray(lon)[us]).astype(object)
    return out


def wind_class(w) -> np.ndarray:
    out = np.full(np.shape(w), NONE, object)
    for lo, name in CLASSES:
        with np.errstate(invalid='ignore'):
            out[np.asarray(w) >= lo] = name
    return out


def first_landfall(lat, lon, wind, grid: np.ndarray) -> pd.DataFrame:
    """One row per (member, storm) with a first landfall: its hour from the first node, wind,
    position, zone, basin and class. lat, lon (node, storm) 6-hourly; wind (member, node, storm)."""
    lat_h, lon_h = landfall.hourly(lat), landfall.wrap(landfall.hourly(landfall.unwrap(lon)))
    code = validate.lookup(grid, lat_h, lon_h)
    w_h = landfall.hourly(np.moveaxis(wind, 0, 1))  # hour, member, storm
    prev = np.vstack([np.full((1, code.shape[1]), -1), code[:-1]])
    entry = (prev == 0) & (code >= 1)
    with np.errstate(invalid='ignore'):
        hit = entry[:, None, :] & (w_h >= CLASSES[0][0])
    m, s = np.nonzero(hit.any(axis=0))
    h = np.argmax(hit, axis=0)[m, s]
    la, lo = lat_h[h, s], lon_h[h, s]
    w = w_h[h, m, s]
    return pd.DataFrame(
        {
            'member': m,
            'storm': s,
            'hour': h,
            'lf_wind': w,
            'lf_lat': la,
            'lf_lon': lo,
            'zone': zone(code[h, s], la, lo),
            'basin': basin(la, lo),
            'cls': wind_class(w),
        }
    )


def grouped_counts(lat, lon, wind, grid, land) -> tuple[pd.DataFrame, pd.DataFrame]:
    """first_landfall, and validate.counts summed over the members of each (zone, basin, class)
    group, one row per group, the members without a landfall as the group NONE."""
    n_m, n_s = wind.shape[0], wind.shape[2]
    lf = first_landfall(lat, lon, wind, grid)
    label = np.full((n_m, n_s), -1)
    keys = [(NONE, NONE, NONE)]
    if len(lf):
        g, idx = np.unique(lf[GROUP].agg('|'.join, axis=1), return_inverse=True)
        keys += [tuple(k.split('|')) for k in g]
        label[lf['member'].values, lf['storm'].values] = idx
    onehot = (label[None] == np.arange(-1, len(keys) - 1)[:, None, None]).astype(float)
    c = validate.counts(lat, lon, wind, grid, land, weight=onehot)
    c.pop('members')
    table = pd.DataFrame(c)
    table[GROUP] = pd.DataFrame(keys, columns=GROUP)
    return lf, table


def key(df: pd.DataFrame, scheme: str) -> pd.Series:
    """The weight key of each row of a records or grouped table under `scheme`."""
    if scheme == 'zone':
        return df['zone']
    if scheme == 'basin':
        return df['basin'].where(df['zone'] != NONE, NONE)
    if scheme == 'zone, other land by basin':
        return df['zone'].where(df['zone'] != 'other land', 'other land ' + df['basin'])
    raise ValueError(scheme)


def weights(
    records: pd.DataFrame,
    obs: pd.DataFrame,
    members: pd.Series,
    obs_members: pd.Series,
    years,
    scheme: str,
    k: float = K,
) -> pd.DataFrame:
    """Weight per (key, class) from first landfalls over `years`: records and obs are
    first_landfall rows with a year column; members and obs_members, member counts by year."""
    years = list(years)

    def tally(df):
        df = df[df['year'].isin(years)]
        return df.groupby([key(df, scheme), df['cls']]).size()

    n_obs, n_chaz = tally(obs), tally(records)
    idx = n_obs.index.union(n_chaz.index)
    n_obs, n_chaz = n_obs.reindex(idx, fill_value=0), n_chaz.reindex(idx, fill_value=0)
    expected = n_chaz * obs_members.loc[years].sum() / members.loc[years].sum()
    out = pd.DataFrame({'n_obs': n_obs, 'chaz_expected': expected})
    out['weight'] = (out['n_obs'] + k) / (out['chaz_expected'] + k)
    out.index.names = ['key', 'cls']
    return out


def weighted_annual(
    grouped: pd.DataFrame, w: pd.DataFrame | None, scheme: str, members: pd.Series
) -> pd.DataFrame:
    """validate.annual counts with each group's members weighted by `w` (from `weights`; None
    weighs every member 1). grouped: grouped_counts rows with a year column."""
    factor = np.ones(len(grouped))
    if w is not None:
        k = pd.DataFrame({'key': key(grouped, scheme).values, 'cls': grouped['cls'].values})
        factor = k.merge(w['weight'].reset_index(), on=['key', 'cls'], how='left')['weight']
        factor = np.where(grouped['zone'].values == NONE, 1.0, factor.fillna(1.0).to_numpy())
    cols = grouped.columns.difference([*GROUP, 'year'])
    out = grouped[cols].mul(factor, axis=0).groupby(grouped['year'].values).sum()
    out.index.name = 'year'
    return out.assign(members=members.reindex(out.index))


def _years(lat, lon, wind, year, years, unit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs, tabs = [], []
    for y in years:
        s = year == y
        if not s.any():
            continue
        lf, table = grouped_counts(
            lat[:, s], lon[:, s], wind[..., s], validate._grid, validate._land
        )
        recs.append(lf.assign(year=y, unit=unit))
        tabs.append(table.assign(year=y))
    return pd.concat(recs, ignore_index=True), pd.concat(tabs, ignore_index=True)


def _run_file(job: tuple[str, int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path, year, unit = job
    with xr.open_dataset(store.path(path), decode_times=False) as ds:
        lat, lon, wind, file_year = landfall.chaz_arrays(ds.load())
    return _years(lat, lon, wind, file_year, [year], unit)


def build(source: str, workers: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(records, grouped) for `source`: 'ibtracs' or a run name under runs/era5."""
    years = range(validate.YEARS[0], validate.YEARS[1] + 1)
    if source == 'ibtracs':
        validate._init()
        csv = landfall.ibtracs_csv(store.s3())
        seasons = (validate.YEARS[0], validate.YEARS[1] + 1)
        lat, lon, wind, _ = landfall.observed_arrays(csv, seasons)
        year = validate.first_record_year(csv, seasons)
        parts = [_years(lat, lon, wind, year, years, 'ibtracs')]
    else:
        validate.natural_earth()
        jobs = [
            (f'{store.RUNS}/{source}/{y}_r{r:03d}.nc', y, f'r{r:03d}')
            for r in range(10)
            for y in years
        ]
        for path, _, _ in jobs:
            store.path(path)
        with ProcessPoolExecutor(workers or os.cpu_count(), initializer=validate._init) as pool:
            parts = list(pool.map(_run_file, jobs))
    records = pd.concat([p[0] for p in parts], ignore_index=True)
    grouped = pd.concat([p[1] for p in parts], ignore_index=True)
    grouped = grouped.groupby(['year', *GROUP], as_index=False).sum()
    return records, grouped


def keys(source: str) -> dict[str, str]:
    return {name: f'{OUT}/{name}_{source}.parquet' for name in ('records', 'grouped')}


def task(source: str) -> None:
    records, grouped = build(source)
    k = keys(source)
    for name, df in (('records', records), ('grouped', grouped)):
        local = store.CACHE / k[name]
        local.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local)
    store.push(keys=list(k.values()), fragment=f'reweight-{source}', replace=True)
    print(source, len(records), 'records', len(grouped), 'group-years', flush=True)


def launch(run: str, vm_type: str = 'm8g.4xlarge') -> None:
    batch.run(
        'reweight',
        ['task'],
        name='chaz-reweight',
        over=['ibtracs', run],
        vm_type=vm_type,
        max_workers=2,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('task')
    t.add_argument('--source', help='default: $COILED_BATCH_TASK_INPUT')
    la = sub.add_parser('launch')
    la.add_argument('--run', required=True, help='run name under runs/era5')
    la.add_argument('--vm-type', default='m8g.4xlarge')
    a = p.parse_args()
    if a.cmd == 'task':
        task(a.source or os.environ['COILED_BATCH_TASK_INPUT'])
    else:
        launch(a.run, a.vm_type)


if __name__ == '__main__':
    main()
