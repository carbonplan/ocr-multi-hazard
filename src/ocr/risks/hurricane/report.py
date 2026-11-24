"""North Atlantic metrics of Emanuel (2025), as John et al. (2026) apply them to CHAZ.

For each source, one row per year of counts, summed over the source's members:

  - North Atlantic storms: a storm (for CHAZ, an intensity member) whose lifetime maximum reaches
    40 kt, in the North Atlantic where it first reaches 40 kt (landfall.in_north_atlantic)
  - `lmi >= v`: those storms with a lifetime maximum of at least v kt, every 5 kt from 40 kt
  - `month m`: those storms by the month in which they first reach 40 kt
  - `dv c`: their 6-hour intensity changes, every step with a wind at both ends, in 5 kt bins
    centred on c, from -60 to +60 kt, the end bins open
  - US landfalls: those storms whose track first crosses onto US Atlantic or Gulf land
    (landfall.crossings, timed by landfall.crossing_time), the member still carrying a wind at
    the crossing; `lf lmi >= v` their lifetime maxima, `lf wind >= v` their wind at the crossing,
    `lf dv c` their change over the 6 hours before it

A source is IBTrACS (one member; with `thin`, its winds put in CHAZ's 12-hourly form), the CHAZ
team's ERA5 set, one of our ERA5 runs, or RAFT (John et al.'s public set, Zenodo 18879988).

`card` scores the sources against an observed table over a span of years with skill.emanuel.
Where the papers leave a choice open, it reads them as follows, with the alternatives as options:

  - intensity histograms: exceedance counts over the period. `scale='observed'` scales the
    synthetic counts to the observed total (John et al., section 2.3), which makes the first bin
    match; `scale='fixed'` sets the synthetic frequency to 12 storms and 3 landfalls a year
    (Emanuel's Table 2; John et al., section 3). The landfall histogram is of lifetime maxima (their
    Figure 1 caption), and `lf wind` is the reading by wind at the crossing
  - 6-hour changes: the synthetic histogram scaled so its peak bin equals the observed one
    (Emanuel's Table 2), then each histogram accumulated from both ends towards its own peak bin
    (Emanuel's Figure 6b). `log_changes` applies ln(1 + count), the form John et al. (section 2.3)
    give for this metric
  - annual cycle: monthly counts, the synthetic total scaled to the observed total
  - Rk': US landfalls over North Atlantic storms, unscaled; we count landfalls on US land rather
    than on coastal gates, whose coordinates are not published

    python -m ocr.risks.hurricane.report task --source <run name | ibtracs | ibtracs-thin |
        chaz-team | raft>
    python -m ocr.risks.hurricane.report launch --runs <run name>,<run name>,...
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from ocr.risks.hurricane import batch, landfall, skill, store

OUT = 'validation/chaz/report'
TS = 40.0  # kt: Emanuel's and John et al.'s storms
LMI = np.arange(40, 205, 5)
DV = np.arange(-60, 65, 5)  # bin centres, kt per 6 hours
RAFT_URL = 'https://zenodo.org/api/records/18879988/files/RAFT_global_v20250604.nc/content'
RAFT_KEY = f'{OUT}/raft_na.npz'
FIXED = {'na': 12.0, 'us': 3.0}  # storms a year, scale='fixed'
EPOCH = np.datetime64('1950-01-01')


def chaz_form(wind: np.ndarray, phase: int = 0) -> np.ndarray:
    """6-hourly winds as CHAZ stores them: every other node kept, from node `phase`, and each
    node between two kept ones their mean (NaN if either is missing); nodes outside the kept
    span are NaN."""
    out = np.full_like(wind, np.nan)
    kept = np.arange(phase, wind.shape[1], 2)
    out[:, kept] = wind[:, kept]
    mid = kept[:-1] + 1
    out[:, mid] = (wind[:, mid - 1] + wind[:, mid + 1]) / 2
    return out


def _bin(values: np.ndarray) -> np.ndarray:
    """Index of each change's 5 kt bin; a change halfway between two centres goes up."""
    return np.floor(np.clip(values, DV[0], DV[-1]) / 5 + 0.5).astype(int) - DV[0] // 5


def counts(lat, lon, wind, month, land) -> dict:
    """Counts over all members of one set of storms: lat, lon (node, storm) 6-hourly, wind
    (member, node, storm) in kt, month (node, storm) of each node; `land` is a
    landfall.USLand."""
    n_m, n_n, n_s = wind.shape
    s_idx = np.arange(n_s)
    with np.errstate(invalid='ignore'):
        reach = wind >= TS
    storm = reach.any(axis=1)
    node = np.argmax(reach, axis=1)
    na = storm & landfall.in_north_atlantic(lat[node, s_idx], lon[node, s_idx])
    with np.errstate(all='ignore'):
        lmi = np.nanmax(np.where(np.isfinite(wind), wind, -np.inf), axis=1)
    out = {'members': n_m, 'na': int(na.sum())}
    for v in LMI:
        out[f'lmi >= {v}'] = int((na & (lmi >= v)).sum())
    first_month = month[node, s_idx]
    for m in range(1, 13):
        out[f'month {m}'] = int((na & (first_month == m)).sum())
    dv = np.diff(wind, axis=1)
    keep = na[:, None, :] & np.isfinite(dv)
    hist = np.bincount(_bin(dv[keep]), minlength=DV.size)
    for c, n in zip(DV, hist):
        out[f'dv {c}'] = int(n)

    lat_h, lon_h = landfall.hourly(lat), landfall.wrap(landfall.hourly(landfall.unwrap(lon)))
    h, s = np.nonzero(landfall.crossings(lat_h, lon_h, land))
    atlantic = landfall.region(lat_h[h, s], lon_h[h, s]) != 'Pacific'
    first = np.full(n_s, np.iinfo(int).max)
    np.minimum.at(first, s[atlantic], h[atlantic])
    storms = np.nonzero(first < np.iinfo(int).max)[0]
    lf_wind = np.full((n_m, n_s), np.nan)
    lf_dv = np.full((n_m, n_s), np.nan)
    if storms.size:
        t = landfall.crossing_time(lat_h, lon_h, first[storms], storms, land)
        lf_wind[:, storms] = landfall.wind_at(wind, t, storms)
        before = landfall.wind_at(wind, np.maximum(t - 6, 0), storms)
        lf_dv[:, storms] = np.where(t >= 6, lf_wind[:, storms] - before, np.nan)
    us = na & np.isfinite(lf_wind)
    out['us'] = int(us.sum())
    for v in LMI:
        out[f'lf lmi >= {v}'] = int((us & (lmi >= v)).sum())
        with np.errstate(invalid='ignore'):
            out[f'lf wind >= {v}'] = int((us & (lf_wind >= v)).sum())
    ok = us & np.isfinite(lf_dv)
    hist = np.bincount(_bin(lf_dv[ok]), minlength=DV.size)
    for c, n in zip(DV, hist):
        out[f'lf dv {c}'] = int(n)
    return out


def _month(days: np.ndarray) -> np.ndarray:
    """Calendar month of days since 1950-01-01; 0 where missing."""
    ok = np.isfinite(days)
    stamps = EPOCH + np.where(ok, days * 86400, 0).astype('timedelta64[s]')
    month = stamps.astype('datetime64[M]').astype(int) % 12 + 1
    return np.where(ok, month, 0)


def _by_year(lat, lon, wind, month, year, years) -> list[dict]:
    rows = []
    for y in years:
        s = year == y
        c = counts(lat[:, s], lon[:, s], wind[..., s], month[:, s], _land) if s.any() else {}
        rows.append({'year': y, **c, 'members': wind.shape[0]})
    return rows


def _chaz(ds: xr.Dataset, years) -> tuple:
    lat, lon, wind, year = landfall.chaz_arrays(ds, years)
    sel = (ds['year'].values >= years[0]) & (ds['year'].values <= years[1])
    days = np.where(np.isfinite(lat), ds['time'].values[:, sel], np.nan)
    return lat, lon, wind, _month(days), year


def _run_file(job: tuple[str, int]) -> list[dict]:
    key, file_year = job
    with xr.open_dataset(store.path(key), decode_times=False) as ds:
        arrays = _chaz(ds.load(), landfall.YEARS)
    return _by_year(*arrays, [file_year])


def _team_file(i: int) -> list[dict]:
    with store.s3().open(f'{landfall.PUBLISHED}/global_2019_2ens{i:03d}_pre.nc', 'rb') as f:
        ds = xr.open_dataset(f, engine='scipy', decode_times=False).load()
    years = landfall.YEARS
    return _by_year(*_chaz(ds, years), range(years[0], years[1] + 1))


def observed(seasons: tuple[int, int], thin: bool = False) -> list[dict]:
    """IBTrACS by the year of each track's first record; with `thin`, the mean over the two
    phases of chaz_form."""
    csv = landfall.ibtracs_csv(store.s3())
    lat, lon, wind, _ = landfall.observed_arrays(csv, seasons)
    bt = landfall.observed_records(csv, seasons)
    start = bt.groupby('SID', sort=True)['time'].min().values.astype('datetime64[s]')
    hours = 6 * np.arange(lat.shape[0])[:, None].astype('timedelta64[h]')
    stamps = start[None] + hours
    days = (stamps - EPOCH) / np.timedelta64(1, 'D')
    month = np.where(np.isfinite(lat), _month(days), 0)
    year = start.astype('datetime64[Y]').astype(int) + 1970
    years = range(seasons[0], seasons[1])
    if not thin:
        return _by_year(lat, lon, wind, month, year, years)
    phases = [_by_year(lat, lon, chaz_form(wind, p), month, year, years) for p in (0, 1)]
    return [
        {k: (a[k] + b[k]) / 2 if k not in ('year', 'members') else a[k] for k in a}
        for a, b in zip(*phases)
    ]


def raft_extract(path: str):
    """The North Atlantic storms of the RAFT file at `path`, saved locally under RAFT_KEY: lat,
    lon, vmax (timestep, storm), month of each step and the storm's year. Returns the local
    path."""
    import netCDF4

    with netCDF4.Dataset(path) as d:
        vmax = np.ma.filled(d['vmax'][:].astype(float), np.nan)
        lat = np.ma.filled(d['lat'][:].astype(float), np.nan)
        lon = np.ma.filled(d['lon'][:].astype(float), np.nan)
        reach = vmax >= TS
        node = np.argmax(reach, axis=1)
        idx = np.arange(vmax.shape[0])
        na = reach.any(axis=1) & landfall.in_north_atlantic(lat[idx, node], lon[idx, node])
        jday = np.ma.filled(d['jday'][:].astype(float), np.nan)[na]
        year = np.asarray(d['year'][:])[na]
    vmax, lat, lon = vmax[na], lat[na], lon[na]
    ok = np.isfinite(vmax) & np.isfinite(lat) & (vmax > 0)
    lat, lon, vmax = (
        np.where(ok, lat, np.nan),
        np.where(ok, lon, np.nan),
        np.where(ok, vmax, np.nan),
    )
    # jday counts from 1 January of the storm's year and runs past 365 into the next
    new_year = (year - 1970).astype('datetime64[Y]').astype('datetime64[D]')
    days = (new_year - EPOCH.astype('datetime64[D]')).astype(float)[:, None] + jday - 1
    local = store.CACHE / RAFT_KEY
    local.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(local, lat=lat.T, lon=lon.T, vmax=vmax.T, month=_month(days).T, year=year)
    return local


def raft(years: tuple[int, int] = (1980, 2018), path=None) -> list[dict]:
    """RAFT's counts from the extraction at `path`, by default the published one."""
    with np.load(path or store.path(RAFT_KEY)) as f:
        lat, lon, wind, month, year = (f[k] for k in ('lat', 'lon', 'vmax', 'month', 'year'))
    _init()
    return _by_year(lat, lon, wind[None], month, year, range(years[0], years[1] + 1))


_land = None


def _init() -> None:
    global _land
    _land = landfall.USLand()


def annual(source: str, workers: int | None = None, raft_path=None) -> pd.DataFrame:
    """Counts by year for `source`: 'ibtracs' or 'ibtracs-thin' (1980-2019), 'raft'
    (1980-2018, from the extraction at `raft_path` if given), 'chaz-team', or a run name under
    runs/era5."""
    if source in ('ibtracs', 'ibtracs-thin'):
        _init()
        # one more season, so Southern Hemisphere storms of the last year are in
        rows = observed((1980, landfall.YEARS[1] + 1), thin=source == 'ibtracs-thin')
    elif source == 'raft':
        rows = raft(path=raft_path)
    else:
        from ocr.risks.hurricane import validate

        validate.natural_earth()
        if source == 'chaz-team':
            fn, jobs, workers = _team_file, range(10), workers or 5
        else:
            fn = _run_file
            years = range(landfall.YEARS[0], landfall.YEARS[1] + 1)
            jobs = [(f'{store.RUNS}/{source}/{y}_r{r:03d}.nc', y) for r in range(10) for y in years]
            for k, _ in jobs:
                store.path(k)
        with ProcessPoolExecutor(workers or os.cpu_count(), initializer=_init) as pool:
            rows = [row for part in pool.map(fn, jobs) for row in part]
    return pd.DataFrame(rows).fillna(0).groupby('year').sum()


def key(source: str) -> str:
    return f'{OUT}/annual_{source}.parquet'


def load(source: str) -> pd.DataFrame:
    return pd.read_parquet(store.path(key(source)))


def task(source: str) -> None:
    keys, raft_path = [], None
    if source == 'raft':
        import urllib.request

        path = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'raft.nc')
        urllib.request.urlretrieve(RAFT_URL, path)
        raft_path = raft_extract(path)
        keys.append(RAFT_KEY)
    df = annual(source, raft_path=raft_path)
    local = store.CACHE / key(source)
    local.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local)
    store.push(keys=[*keys, key(source)], fragment=f'report-{source}', replace=True)
    print(source, df[['members', 'na', 'us']].sum().to_dict(), flush=True)


def launch(sources: list[str], vm_type: str = 'm8g.4xlarge') -> None:
    batch.run(
        'report',
        ['task'],
        name=f'chaz-report-{len(sources)}',
        over=sources,
        vm_type=vm_type,
        max_workers=len(sources),
    )


def _columns(table: pd.Series, prefix: str, values) -> np.ndarray:
    return np.array([table[f'{prefix} {v}'] for v in values], float)


def double_sided(hist: np.ndarray, peak: int) -> np.ndarray:
    """Cumulative counts from each end towards bin `peak`: below it, the counts at or below each
    bin; from it up, the counts at or above."""
    out = np.empty_like(hist, dtype=float)
    out[:peak] = np.cumsum(hist[:peak])
    out[peak:] = np.cumsum(hist[peak:][::-1])[::-1]
    return out


def series(
    obs: pd.Series,
    syn: pd.Series,
    n_years: int,
    scale: str = 'observed',
    log_changes: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(observed, synthetic) values of each metric from annual counts summed over `n_years`."""
    out = {}
    for name, prefix, total in (
        ('intensity histogram', 'lmi >=', 'na'),
        ('intensity histogram (landfall)', 'lf lmi >=', 'us'),
        ('wind at landfall', 'lf wind >=', 'us'),
    ):
        o, s = _columns(obs, prefix, LMI), _columns(syn, prefix, LMI)
        target = obs[total] if scale == 'observed' else FIXED[total] * n_years
        out[name] = (o, s * target / syn[total])
    for name, prefix in (
        ('6 h intensity change', 'dv'),
        ('6 h intensity change (landfall)', 'lf dv'),
    ):
        o, s = _columns(obs, prefix, DV), _columns(syn, prefix, DV)
        s = s * o.max() / s.max()
        pair = (double_sided(o, int(np.argmax(o))), double_sided(s, int(np.argmax(s))))
        out[name] = tuple(np.log1p(x) for x in pair) if log_changes else pair
    o, s = _columns(obs, 'month', range(1, 13)), _columns(syn, 'month', range(1, 13))
    out['annual cycle'] = (o, s * o.sum() / s.sum())
    return out


def card(
    tables: dict[str, pd.DataFrame],
    years,
    observed_label: str = 'IBTrACS',
    scale: str = 'observed',
    log_changes: bool = False,
) -> pd.DataFrame:
    """D', K', r2 and chi2' of every metric for every source against `observed_label` over
    `years`, and each source's Rk'. Rows (metric, test), one column per source; `scale` and
    `log_changes` as for `series`."""
    years = list(years)
    sums = {label: t.loc[years].sum() for label, t in tables.items()}
    obs = sums[observed_label]
    out = {}
    for label, syn in sums.items():
        if label == observed_label:
            continue
        col = {}
        for metric, (o, s) in series(obs, syn, len(years), scale, log_changes).items():
            for test, value in skill.emanuel(o, s).items():
                col[(metric, test)] = value
        col[('coastal crossing-to-basin ratio', "Rk'")] = (
            skill.ratio_skill(syn['us'], syn['na'], obs['us'], obs['na'])
            if obs['us'] > 0 and obs['na'] > 0 and syn['na'] > 0
            else np.nan
        )
        out[label] = col
    return pd.DataFrame(out)


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
        launch(['ibtracs', 'ibtracs-thin', 'chaz-team', 'raft', *a.runs.split(',')], a.vm_type)


if __name__ == '__main__':
    main()
