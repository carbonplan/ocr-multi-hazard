"""CONUS Holland winds on CLIMADA 6.1 for the public CMIP6 CHAZ tracks.

Chia-Ying Lee's TROPICONA tutorial (Zenodo 12533388, CC BY 4.0) holds 25
CMIP6 CHAZ track files with the naming of the files behind the published
CMIP6 maps: {model}_Global_2100_2ens00{0,1}_{CRH|SD}_compressed.nc, two of the
ten track realizations per model and genesis index, 40 intensity members,
1951-2100. The scenario isn't stated in the files.

This runs the CMIP6 wind step of the published maps (Meiler et al. 2025,
risk-model-various/CHAZ_wind_calc_split_ens.py) on CONUS cells. It uses the
same members (0, 5, ..., 35), the base and end-of-century windows and a 1 h
step, with two fixes the published maps lack:

  - CLIMADA 6.1, whose Holland models count translational velocity once
    (PR #833)
  - central pressure re-estimated from wind with longitudes wrapped to
    -180..180. CLIMADA's CHAZ reader passes 0-360 longitudes to a regression
    fit on -180..180, which makes western-hemisphere storms 12.7 hPa too deep.
    --pressure raw keeps the reader's value.

Only storms with tropical-storm winds within 5 degrees of the CONUS box are
computed, cropped to that reach (chaz_events.crop_to_reach).

  run      --files all --workers 12     h08/<file>_<tag>.hdf5 and timing.csv
  compare  --file CESM2_..._SD          per-track CONUS peak wind, wrapped vs raw
  summary                               CONUS storm rates, base against end of century

Everything stays under ~/.cache/ocr/chaz-local/tropicona_h08. Run in the
`climada` pixi environment.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chaz_events as ce

ROOT = Path(os.environ.get('CHAZ_LOCAL_DIR', Path.home() / '.cache' / 'ocr' / 'chaz-local'))
TRACKS = ROOT / 'tropicona'
OUT = ROOT / 'tropicona_h08'
MEMBERS = list(range(0, 40, 5))
PERIODS = {'base': (1995, 2014), 'fut2': (2081, 2100)}
TIME_STEP_H = 1.0

log = logging.getLogger('chaz_tropicona')


def track_files(sel: str) -> list[Path]:
    files = sorted(TRACKS.glob('*_compressed.nc'))
    if sel == 'all':
        return files
    keys = sel.split(',')
    return [f for f in files if any(k in f.name for k in keys)]


def subset(path: Path) -> tuple[Path, pd.DataFrame]:
    """Storms in the two windows that reach CONUS in a selected member,
    written with padding winds blanked (chaz_events.build_events does the same
    for the ERA5 files)."""
    sub = OUT / 'subsets' / path.name
    events = OUT / 'subsets' / f'{path.stem}.parquet'
    if sub.exists() and events.exists():
        return sub, pd.read_parquet(events)
    sub.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(path, decode_times=False).isel(ensembleNum=MEMBERS)
    year = ds['year'].values
    in_window = np.zeros(year.size, bool)
    for y0, y1 in PERIODS.values():
        in_window |= (year >= y0) & (year <= y1)
    lat, lon = ds['latitude'].values, ds['longitude'].values
    w = ds['Mwspd'].values
    strong = w >= ce.INTENSITY_THRES * ce.KT_PER_MS
    reach = (strong & ce.in_conus_reach(lat, lon)[None]).any(axis=1) & in_window[None]
    keep = np.nonzero(reach.any(axis=0))[0]
    fill = ds['time'].values <= ce.TIME_FILL
    ds['Mwspd'] = ds['Mwspd'].where(~fill[None])
    ds = ds.isel(stormID=keep).assign_coords(ensembleNum=MEMBERS)
    ds.to_netcdf(sub)
    m, s = np.nonzero(reach[:, keep])
    df = pd.DataFrame(
        {
            'pos': s.astype('int32'),
            'member': np.asarray(MEMBERS)[m].astype('int16'),
            'year': year[keep][s].astype('int16'),
        }
    )
    df['period'] = np.where(df['year'] <= PERIODS['base'][1], 'base', 'fut2')
    df.to_parquet(events, index=False)
    return sub, df


def read_tracks(path: Path, pressure: str):
    from climada.hazard import TCTracks

    sub, events = subset(path)
    tracks = TCTracks.from_simulations_chaz(str(sub), ensemble_nums=MEMBERS)
    # the reader names tracks <file>-<storm position>-<member>
    wanted = {
        f'{sub.name}-{p}-{m}': per
        for p, m, per in zip(events['pos'], events['member'], events['period'])
    }
    kept = []
    for tr in tracks.data:
        if tr.sid in wanted:
            tr.attrs['period'] = wanted[tr.sid]
            if pressure == 'wrapped':
                tr = ce.fix_pressure(tr)
            kept.append(ce.crop_to_reach(tr))
    tracks.data = kept
    tracks.equal_timestep(TIME_STEP_H)
    return tracks


def cmd_run(args):
    from importlib.metadata import version

    from climada.hazard import Centroids

    logging.getLogger('climada').setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'h08').mkdir(exist_ok=True)
    cent = Centroids.from_hdf5(str(ce._local(ce.CENTROIDS_KEY)))
    timing_path = OUT / 'timing.csv'
    for path in track_files(args.files):
        out = OUT / 'h08' / f'{path.stem}_{args.pressure}.hdf5'
        if out.exists() and not args.force:
            print(f'exists: {out.name}')
            continue
        t0 = time.perf_counter()
        tracks = read_tracks(path, args.pressure)
        t_read = time.perf_counter() - t0
        t1 = time.perf_counter()
        haz = ce.winds_parallel(tracks, cent, args.workers, args.max_memory_gb)
        t_wind = time.perf_counter() - t1
        period = {tr.sid: tr.attrs['period'] for tr in tracks.data}
        periods = list(period.values())
        # workers return in completion order, so name events from the hazard
        haz.event_name = [f'{n}|{period[n]}' for n in haz.event_name]
        haz.write_hdf5(str(out))
        row = {
            'file': path.stem,
            'pressure': args.pressure,
            'climada': version('climada'),
            'time_step_h': TIME_STEP_H,
            'tracks': tracks.size,
            'base': periods.count('base'),
            'fut2': periods.count('fut2'),
            'nonzeros': haz.intensity.nnz,
            'max_ms': float(haz.intensity.max()),
            's_read': t_read,
            's_wind': t_wind,
            'workers': args.workers,
            'core_s_per_track': t_wind * args.workers / max(tracks.size, 1),
        }
        pd.DataFrame([row]).to_csv(
            timing_path, mode='a', header=not timing_path.exists(), index=False
        )
        print(
            f'{path.stem} [{args.pressure}]: {tracks.size:,} tracks (base {row["base"]:,}, '
            f'fut2 {row["fut2"]:,}), read {t_read:.0f}s, winds {t_wind:.0f}s '
            f'({row["core_s_per_track"]:.2f} core-s/track), max {row["max_ms"]:.1f} m/s',
            flush=True,
        )


def cmd_compare(args):
    from climada.hazard import Hazard

    stem = track_files(args.file)[0].stem
    peaks = {}
    for p in ('wrapped', 'raw'):
        haz = Hazard.from_hdf5(str(OUT / 'h08' / f'{stem}_{p}.hdf5'))
        peaks[p] = pd.Series(haz.intensity.max(axis=1).toarray().ravel(), index=haz.event_name)
    df = pd.DataFrame(peaks).dropna()
    df = df[df['raw'] > 0]
    ratio = df['raw'] / df['wrapped'].where(df['wrapped'] > 0)
    hurricane = df['raw'] >= 33
    print(f'{stem}: {len(df):,} tracks with CONUS wind under the reader pressure')
    for name, m in (('all', slice(None)), ('raw peak >= 33 m/s', hurricane)):
        r = ratio[m].dropna()
        print(
            f'  {name}: raw / wrapped peak CONUS wind median {r.median():.3f} '
            f'(5-95% {r.quantile(0.05):.3f}-{r.quantile(0.95):.3f}), n={len(r):,}'
        )
    print(
        f'  tracks reaching 33 m/s on CONUS: raw {int((df["raw"] >= 33).sum()):,}, '
        f'wrapped {int((df["wrapped"] >= 33).sum()):,}'
    )


def cmd_summary(args):
    """Per model and genesis index: storms per member-year with CONUS winds of
    tropical-storm and hurricane strength, base against end of century."""
    import h5py

    rows = []
    for path in sorted((OUT / 'h08').glob(f'*_{args.pressure}.hdf5')):
        stem = path.name.removesuffix(f'_{args.pressure}.hdf5')
        model, _, rest = stem.partition('_Global_2100_')
        ens, tcgi = rest.split('_')[:2]
        with h5py.File(path) as f:
            names = [n.decode() if isinstance(n, bytes) else n for n in f['event_name'][:]]
            indptr = f['intensity/indptr'][:]
            data = f['intensity/data'][:]
        peak = np.array(
            [data[a:b].max() if b > a else 0.0 for a, b in zip(indptr[:-1], indptr[1:])]
        )
        period = np.array([n.rsplit('|', 1)[1] for n in names])
        for per, (y0, y1) in PERIODS.items():
            member_years = len(MEMBERS) * (y1 - y0 + 1)
            m = period == per
            rows.append(
                {
                    'model': model,
                    'tcgi': tcgi,
                    'realization': ens,
                    'period': per,
                    'ts_per_yr': float((peak[m] >= ce.INTENSITY_THRES).sum()) / member_years,
                    'hu_per_yr': float((peak[m] >= 33.0).sum()) / member_years,
                }
            )
    df = pd.DataFrame(rows)
    g = df.groupby(['tcgi', 'model', 'period'])[['ts_per_yr', 'hu_per_yr']].mean().unstack('period')
    g[('hu_per_yr', 'change')] = g[('hu_per_yr', 'fut2')] / g[('hu_per_yr', 'base')] - 1
    print(
        'storms per member-year with CONUS winds >= 17.5 m/s (ts) and >= 33 m/s (hu), '
        f'pressure {args.pressure}, mean over the realizations'
    )
    print(g.round(3).to_string())
    df.to_csv(OUT / f'summary_{args.pressure}.csv', index=False)


def main():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s: %(message)s')
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('run', help='CONUS winds for the selected track files')
    q.add_argument('--files', default='all', help="'all' or comma-separated name fragments")
    q.add_argument('--pressure', default='wrapped', choices=['wrapped', 'raw'])
    q.add_argument('--workers', type=int, default=max((os.cpu_count() or 2) - 6, 1))
    q.add_argument('--max-memory-gb', type=float, default=24.0)
    q.add_argument('--force', action='store_true')
    q.set_defaults(func=cmd_run)
    q = sub.add_parser('compare', help='peak CONUS winds with and without the longitude fix')
    q.add_argument('--file', default='CESM2_Global_2100_2ens000_SD')
    q.set_defaults(func=cmd_compare)
    q = sub.add_parser('summary', help='CONUS storm rates per model, base against end of century')
    q.add_argument('--pressure', default='wrapped', choices=['wrapped', 'raw'])
    q.set_defaults(func=cmd_summary)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
