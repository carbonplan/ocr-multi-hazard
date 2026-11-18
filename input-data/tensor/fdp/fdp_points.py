# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "boto3",
#   "numpy",
#   "obstore>=0.12",
#   "pandas",
#   "pyproj",
#   "rasterio",
#   "requests",
#   "universal-pathlib>=0.3.10",
# ]
# ///
"""Build the presence/absence training points for each FDP model run.

Each run is written to `input/flood/fdp-refit/runs/<run>/points.csv` with `x, y, pa, huc2,
nlcd` (EPSG:5070 metres) and a `points.json` summary:

- `original`: the released Collins et al. (2022) points, unchanged, tagged
  with the HUC-2 and land-cover class of their cell. The summary audits the
  released absences against the paper's sampling rules.
- `noaa-200612-202512`: every NOAA Storm Events Flood / Flash Flood report
  beginning December 2006 through December 2025 in CONUS, at its begin
  coordinates, one per 100 m cell. That is the construction that matches
  99.6% of the released presence cells.
- `noaa-200612-202005`: the same construction over the released points'
  window, so that comparing runs separates the effect of this construction
  from that of the added years.

`--seeds 1 2 ...` adds `<run>-seed<k>` runs whose absences are drawn with seed
SEED + k; seed 0 is the unsuffixed run. `--noaa-from <run>` reuses the NOAA
files that run recorded, kept under `input/flood/raw/noaa-storm-events/`, so runs
built later share its presences even after NOAA reissues a year's file. Runs
already built (both files present; points.json is written last) are skipped
unless `--overwrite` is given.

Absences for the NOAA runs are resampled with the paper's rules (Collins et al.
2022, section 2): within each HUC-2, as many
absences as presences in each land cover class, outside the combined 100-year
floodplain, and never in a cell holding a presence. Absences sit at cell centres, as the released ones do.

Loads the HUC, land-cover and floodplain grids whole (~4 GB as uint8).

`--coiled` runs `build` with the same arguments on a Coiled VM.

Usage:
  python fdp_points.py build [--runs original noaa-200612-202512 ...] [--seeds 0 1 2]
      [--noaa-from noaa-200612-202512] [--overwrite] [--coiled]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys

import numpy as np
import pandas as pd
import rasterio
import requests
from fdp import (
    CACHE,
    REFIT,
    ROOT,
    delete,
    exists,
    get_file,
    launch,
    put_bytes,
    put_file,
    read_bytes,
)
from fdp_inputs import HUC_RASTER, HUCS, predictor_path, read_manifest, response_path
from pyproj import Transformer
from rasterio.windows import Window
from upath import UPath

RUNS = REFIT / 'runs'
NOAA = 'https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/'
# Every NOAA file a run used: NCEI drops a year's file when it reissues it, so `--noaa-from`
# reads them here.
NOAA_ARCHIVE = ROOT / 'input/flood/raw/noaa-storm-events'
NOAA_START, NOAA_END = pd.Timestamp('2006-12-01'), pd.Timestamp('2026-01-01')
ORIGINAL_END = pd.Timestamp('2020-06-01')
FLOOD_TYPES = ['Flood', 'Flash Flood']
# Run -> end of its report window (exclusive).
NOAA_RUNS = {'noaa-200612-202512': NOAA_END, 'noaa-200612-202005': ORIGINAL_END}
SEED = 20261022
NODATA = 255


def run_path(run: str, name: str) -> UPath:
    return RUNS / run / name


def seeded(run: str, seed: int) -> str:
    return run if seed == 0 else f'{run}-seed{seed}'


class Grid:
    def __init__(self, manifest: dict):
        self.shape = tuple(manifest['grid']['shape'])
        self.transform = rasterio.Affine(*manifest['grid']['transform'])
        self.windows = manifest['huc2']['windows']

    def cells(self, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = np.floor((np.asarray(y) - self.transform.f) / self.transform.e).astype('int64')
        col = np.floor((np.asarray(x) - self.transform.c) / self.transform.a).astype('int64')
        inside = (row >= 0) & (row < self.shape[0]) & (col >= 0) & (col < self.shape[1])
        return row, col, inside

    def centres(self, row, col) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.transform.c + (col + 0.5) * self.transform.a,
            self.transform.f + (row + 0.5) * self.transform.e,
        )


def read_uint8(path: UPath, grid: Grid, offset=(0, 0), strip: int = 4096) -> np.ndarray:
    """Read a categorical raster on `grid` from S3 as uint8, nodata as NODATA."""
    print(f'  reading {path}')
    (height, width), (dr, dc) = grid.shape, offset
    with rasterio.open(str(path)) as src:
        out = np.empty((height, width), dtype='uint8')
        for r0 in range(0, height, strip):
            h = min(strip, height - r0)
            block = src.read(1, window=Window(dc, dr + r0, width, h), masked=True)
            valid = block.compressed()
            if valid.size and (valid.min() < 0 or valid.max() >= NODATA):
                sys.exit(f'{path}: values outside 0..{NODATA - 1}')
            out[r0 : r0 + h] = block.filled(NODATA).astype('uint8')
    return out


def tag(points: pd.DataFrame, grid: Grid, hucs: np.ndarray, nlcd: np.ndarray) -> pd.DataFrame:
    row, col, inside = grid.cells(points.x, points.y)
    points = points.assign(row=row, col=col)[inside].copy()
    points['huc2'] = hucs[points.row.to_numpy(), points.col.to_numpy()]
    points['nlcd'] = nlcd[points.row.to_numpy(), points.col.to_numpy()]
    return points


# ---------------------------------------------------------------------------
# NOAA reports
# ---------------------------------------------------------------------------


def noaa_year(name: str) -> int:
    return int(re.search(r'_v1\.0_d(\d{4})_', name).group(1))


def noaa_details(pinned: list[dict] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    raw = CACHE / 'noaa'
    raw.mkdir(parents=True, exist_ok=True)
    if pinned:
        latest = {noaa_year(f['file']): f['file'] for f in pinned}
        hashes = {f['file']: f['sha256'] for f in pinned}
    else:
        index = requests.get(NOAA, timeout=60)
        index.raise_for_status()
        latest, hashes = {}, {}
        for name in sorted(
            set(re.findall(r'StormEvents_details-ftp_v1\.0_d\d{4}_c\d+\.csv\.gz', index.text))
        ):
            year = noaa_year(name)
            if NOAA_START.year <= year < NOAA_END.year:
                latest[year] = name  # names sort by creation date within a year
    if sorted(latest) != list(range(NOAA_START.year, NOAA_END.year)):
        sys.exit(f'NOAA index is missing years: {sorted(latest)}')

    frames, provenance = [], []
    for year, name in sorted(latest.items()):
        path, archived = raw / name, NOAA_ARCHIVE / name
        if not path.exists() and exists(archived):
            get_file(archived, path)
        elif not path.exists():
            r = requests.get(NOAA + name, timeout=300)
            r.raise_for_status()
            path.with_suffix('.part').write_bytes(r.content)
            path.with_suffix('.part').rename(path)
        if not exists(archived):
            put_file(path, archived)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in hashes and digest != hashes[name]:
            sys.exit(f'{path}: sha256 differs from the pinned run')
        provenance.append({'file': name, 'bytes': path.stat().st_size, 'sha256': digest})
        df = pd.read_csv(
            path,
            low_memory=False,
            usecols=[
                'EVENT_ID',
                'EVENT_TYPE',
                'STATE_FIPS',
                'BEGIN_DATE_TIME',
                'BEGIN_LAT',
                'BEGIN_LON',
            ],
        )
        frames.append(df[df.EVENT_TYPE.isin(FLOOD_TYPES)])
    events = pd.concat(frames, ignore_index=True)
    if not events.EVENT_ID.is_unique:
        sys.exit('duplicate NOAA EVENT_ID across details files')
    events['date'] = pd.to_datetime(events.BEGIN_DATE_TIME, format='%d-%b-%y %H:%M:%S')
    return events, provenance


def noaa_cells(events: pd.DataFrame, grid: Grid, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    window = events[(events.date >= NOAA_START) & (events.date < end)]
    conus = window[(window.STATE_FIPS < 60) & ~window.STATE_FIPS.isin([2, 15])]
    located = conus.dropna(subset=['BEGIN_LAT', 'BEGIN_LON'])
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(
        located.BEGIN_LON.to_numpy(), located.BEGIN_LAT.to_numpy()
    )
    reports = located.assign(x=x, y=y).sort_values(['date', 'EVENT_ID'])
    row, col, inside = grid.cells(reports.x, reports.y)
    reports = reports.assign(row=row, col=col)[inside]
    cells = reports.drop_duplicates(['row', 'col'])
    counts = {
        'reports': len(window),
        'conus': len(conus),
        'located': len(located),
        'on_grid': len(reports),
        'unique_cells': len(cells),
    }
    return cells[['x', 'y', 'row', 'col']].reset_index(drop=True), counts


# ---------------------------------------------------------------------------
# absences
# ---------------------------------------------------------------------------


def window_mask(points: pd.DataFrame, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    inside = (points.row >= r0) & (points.row < r0 + h) & (points.col >= c0) & (points.col < c0 + w)
    mask = np.zeros((h, w), dtype=bool)
    mask[points.row[inside].to_numpy() - r0, points.col[inside].to_numpy() - c0] = True
    return mask


def sample_absences(
    presence: pd.DataFrame,
    grid: Grid,
    hucs,
    nlcd,
    floodplain,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict]]:
    """Draw absences per HUC and land cover class, outside the floodplain."""
    rows, cols, table = [], [], []
    for huc in HUCS:
        r0, c0, h, w = grid.windows[huc]
        eligible = (
            (hucs[r0 : r0 + h, c0 : c0 + w] == int(huc))
            & ~window_mask(presence, r0, c0, h, w)
            & (floodplain[r0 : r0 + h, c0 : c0 + w] == 0)
        )
        classes = nlcd[r0 : r0 + h, c0 : c0 + w]

        need = presence[(presence.huc2 == int(huc)) & (presence.nlcd != NODATA)].nlcd.value_counts()
        for cls, n in sorted(need.items()):
            idx = np.flatnonzero(eligible & (classes == cls))
            take = rng.choice(idx, size=min(n, idx.size), replace=False)
            r, c = np.divmod(take, w)
            rows.append(r + r0)
            cols.append(c + c0)
            table.append(
                {
                    'huc2': huc,
                    'nlcd': int(cls),
                    'presence': int(n),
                    'absence': int(take.size),
                    'eligible_cells': int(idx.size),
                }
            )
        print(f'    {huc}: {int(need.sum())} absences')
    row, col = np.concatenate(rows), np.concatenate(cols)
    x, y = grid.centres(row, col)
    return pd.DataFrame({'x': x, 'y': y, 'row': row, 'col': col}), table


def audit(presence: pd.DataFrame, absence: pd.DataFrame, floodplain) -> dict:
    """How the released absences compare with the paper's stated rules."""
    p = presence.groupby(['huc2', 'nlcd']).size().rename('presence')
    a = absence.groupby(['huc2', 'nlcd']).size().rename('absence')
    by_class = pd.concat([p, a], axis=1).fillna(0).astype(int).reset_index()
    p_cells = set(zip(presence.row, presence.col))
    return {
        'absence_in_floodplain': int(
            (floodplain[absence.row.to_numpy(), absence.col.to_numpy()] == 1).sum()
        ),
        'absence_on_presence_cell': sum(
            (r, c) in p_cells for r, c in zip(absence.row, absence.col)
        ),
        'huc_class_groups': len(by_class),
        'huc_class_groups_equal': int((by_class.presence == by_class.absence).sum()),
        'by_huc_class': by_class.to_dict(orient='records'),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def write_run(run: str, presence: pd.DataFrame, absence: pd.DataFrame, summary: dict) -> None:
    points = pd.concat([presence.assign(pa=1), absence.assign(pa=0)], ignore_index=True)
    points = points[points.huc2 != 0]
    buf = io.StringIO()
    points[['x', 'y', 'pa', 'huc2', 'nlcd']].to_csv(buf, index=False, float_format='%.4f')
    summary['points'] = {
        'presence': int((points.pa == 1).sum()),
        'absence': int((points.pa == 0).sum()),
        'by_huc': {
            f'{h:02d}': {
                'presence': int(((points.huc2 == h) & (points.pa == 1)).sum()),
                'absence': int(((points.huc2 == h) & (points.pa == 0)).sum()),
            }
            for h in range(1, 19)
        },
    }
    delete(run_path(run, 'points.json'))
    put_bytes(run_path(run, 'points.csv'), buf.getvalue().encode())
    put_bytes(run_path(run, 'points.json'), json.dumps(summary, indent=2).encode())
    print(
        f'  {summary["points"]["presence"]} presence, {summary["points"]["absence"]} absence'
        f' -> {run_path(run, "points.csv")}'
    )


def read_summary(run: str) -> dict:
    return json.loads(read_bytes(run_path(run, 'points.json')))


def released(kind: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(read_bytes(response_path(kind))))


def cmd_build(args) -> None:
    if args.coiled:
        argv = ['fdp_points', 'build', '--runs', *args.runs, '--seeds', *map(str, args.seeds)]
        argv += ['--noaa-from', args.noaa_from] if args.noaa_from else []
        argv += ['--overwrite'] if args.overwrite else []
        launch('fdp-points', argv, vm_type=args.vm_type)
        return
    targets = [('original', 0)] if 'original' in args.runs else []
    targets += [(run, seed) for run in NOAA_RUNS if run in args.runs for seed in args.seeds]
    built = [
        (r, k)
        for r, k in targets
        if all(exists(run_path(seeded(r, k), name)) for name in ('points.csv', 'points.json'))
    ]
    if built and not args.overwrite:
        names = ', '.join(seeded(r, k) for r, k in built)
        print(f'already built, skipped: {names} (--overwrite replaces them)')
        targets = [t for t in targets if t not in built]
        if not targets:
            return
    pinned = read_summary(args.noaa_from)['noaa_files'] if args.noaa_from else None

    manifest = read_manifest()
    grid = Grid(manifest)
    hucs = read_uint8(HUC_RASTER, grid)
    nlcd = read_uint8(predictor_path('nlcd'), grid, manifest['predictors']['nlcd']['offset'])
    floodplain = read_uint8(
        predictor_path('floodplain'), grid, manifest['predictors']['floodplain']['offset']
    )

    pres = tag(released('presence'), grid, hucs, nlcd)
    absn = tag(released('absence'), grid, hucs, nlcd)

    if ('original', 0) in targets:
        print('original:')
        summary = {
            'run': 'original',
            'source': 'Collins et al. (2022) released points, doi:10.5066/P954TTQN',
            'outside_huc_raster': {
                'presence': int((pres.huc2 == 0).sum()),
                'absence': int((absn.huc2 == 0).sum()),
            },
            'audit': audit(pres[pres.huc2 != 0], absn[absn.huc2 != 0], floodplain),
        }
        write_run('original', pres, absn, summary)

    runs = [run for run in NOAA_RUNS if any(r == run for r, _ in targets)]
    if runs:
        events, provenance = noaa_details(pinned)
        rebuilt, _ = noaa_cells(events, grid, ORIGINAL_END)
        released_cells = set(zip(pres.row, pres.col))
        overlap = sum(c in released_cells for c in zip(rebuilt.row, rebuilt.col))
    for run in runs:
        end = NOAA_RUNS[run]
        presence, counts = noaa_cells(events, grid, end)
        presence = tag(presence[['x', 'y']], grid, hucs, nlcd)
        for seed in [k for r, k in targets if r == run]:
            name = seeded(run, seed)
            print(f'{name}:')
            print('  sampling absences')
            rng = np.random.default_rng(SEED + seed)
            absence, table = sample_absences(
                presence[presence.huc2 != 0],
                grid,
                hucs,
                nlcd,
                floodplain,
                rng,
            )
            absence = tag(absence[['x', 'y']], grid, hucs, nlcd)
            summary = {
                'run': name,
                'window': [str(NOAA_START.date()), str((end - pd.Timedelta(days=1)).date())],
                'event_types': FLOOD_TYPES,
                'location': 'BEGIN_LAT/BEGIN_LON, first report per 100 m cell',
                'counts': counts
                | {
                    'outside_huc_raster': int((presence.huc2 == 0).sum()),
                    'presence_without_land_cover': int(
                        ((presence.huc2 != 0) & (presence.nlcd == NODATA)).sum()
                    ),
                },
                'released_window_check': {
                    'rebuilt_cells_dec2006_may2020': len(rebuilt),
                    'released_cells': len(released_cells),
                    'overlap': overlap,
                },
                'absence_sampling': {
                    'seed': SEED + seed,
                    'by_huc_class': table,
                },
                'noaa_files': provenance,
            }
            write_run(name, presence, absence, summary)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='write points.csv and points.json for each run')
    b.add_argument(
        '--runs', nargs='+', default=['original', *NOAA_RUNS], choices=['original', *NOAA_RUNS]
    )
    b.add_argument('--seeds', nargs='+', type=int, default=[0], help='absence draws per NOAA run')
    b.add_argument('--noaa-from', help='reuse the NOAA files recorded by this run')
    b.add_argument('--overwrite', action='store_true')
    b.add_argument('--coiled', action='store_true', help='run on a Coiled VM')
    b.add_argument('--vm-type', default='m7i.2xlarge')
    args = p.parse_args()
    {'build': cmd_build}[args.cmd](args)


if __name__ == '__main__':
    main()
