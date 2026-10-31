"""How the published frequency correction responds when the winds are fixed.

The published maps weight every storm in a basin by the observed annual count
over N, the number of synthetic storms with any wind on a cell inside the
basin (CHAZ-hazard-maps CHAZ-pre-processing/freq_corr_era5.py,
`intensity.max(axis=1).getnnz()`). Winds are stored from 17.5 m/s, so N
depends on the wind construction. Weaker winds can drop storms out of N and
raise each remaining storm's weight. chaz_events instead counts storms whose
own track carries 17.5 m/s inside simple outlines, which doesn't depend on the
wind field.

This counts N her way on global winds for the same tracks under the reader's
pressure (chaz_rerun wind, the published layout) and the wrapped-longitude
pressure (chaz_rerun wind --pressure wrapped), and sets them beside the
chaz_events count for those tracks.

  count --tasks 0:0,0:13,0:15,0:16     table of N per basin, freqnorm/count.csv

Run in the `climada` pixi environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chaz_events as ce
import chaz_rerun as cr

OUT = (
    Path(os.environ.get('CHAZ_LOCAL_DIR', Path.home() / '.cache' / 'ocr' / 'chaz-local'))
    / 'freqnorm'
)

# freq_corr_era5.py, as published
EP_NA_BOUNDARY_LINE = [
    (-100.0, 60.0),
    (-100.0, 18.0),
    (-90.0, 18.0),
    (-90.0, 15.0),
    (-85.0, 15.0),
    (-85.0, 9.0),
    (-75.0, 9.0),
    (-75.0, 5.0),
]
YRLY_FREQ_IB_LIT = {'EP': 14.5, 'NA': 10.8, 'NI': 2.0, 'SI': 12.3, 'SP': 9.3, 'WP': 22.5}
BOXES = {
    'NI': (30.0, 100.0, 5.0, 60.0),
    'SI': (10.0, 135.0, -60.0, -5.0),
    'SP': (135.0, -120.0, -60.0, -5.0),
    'WP': (100.0, 180.0, 5.0, 60.0),
}


def basin_cells(lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """Her basin_split_haz: polygons for EP and NA, open boxes for the rest."""
    import shapely
    from shapely.geometry import Polygon

    polys = {
        'EP': Polygon([(-180, 60), (-180, 5)] + EP_NA_BOUNDARY_LINE[::-1]),
        'NA': Polygon(EP_NA_BOUNDARY_LINE + [(0, 5), (0, 60)]),
    }
    out = {b: shapely.contains_xy(p, lon, lat) for b, p in polys.items()}
    for b, (lo0, lo1, la0, la1) in BOXES.items():
        in_lat = (lat > la0) & (lat < la1)
        in_lon = ((lon > lo0) | (lon < lo1)) if lo1 < lo0 else ((lon > lo0) & (lon < lo1))
        out[b] = in_lat & in_lon
    return out


def counts(path: Path, cells: dict[str, np.ndarray]) -> dict[str, int]:
    from climada.hazard import Hazard

    haz = Hazard.from_hdf5(str(path))
    inten = haz.intensity.tocsc()
    return {
        b: int((inten[:, np.nonzero(m)[0]].tocsr().getnnz(axis=1) > 0).sum())
        for b, m in cells.items()
    }


def cmd_count(args):
    from climada.hazard import Centroids

    OUT.mkdir(parents=True, exist_ok=True)
    s3 = ce._s3()
    cent = Centroids.from_hdf5(str(ce._fetch(s3, cr.CENTROIDS_KEY)))
    lon = np.where(cent.lon > 180, cent.lon - 360, cent.lon)
    cells = basin_cells(cent.lat, lon)
    tasks = [tuple(int(v) for v in t.split(':')) for t in args.tasks.split(',')]
    years = cr.YEAR_RANGE[1] - cr.YEAR_RANGE[0] + 1
    rows = []
    for i, m in tasks:
        for pressure in ('raw', 'wrapped'):
            key = f'{cr.present_dir(pressure)}/{cr.wind_name(i, m)}'
            if not s3.exists(f'{ce.BUCKET}/{key}'):
                print(f'missing: {key}')
                continue
            for b, n in counts(ce._fetch(s3, key), cells).items():
                rows.append(
                    {'file': i, 'member': m, 'count': f'footprint, {pressure}', 'basin': b, 'n': n}
                )
        ev = pd.read_parquet(ce._fetch(s3, ce.events_key(i)))
        ev = ev[(ev['member'] == m) & ev['year'].between(*cr.YEAR_RANGE)]
        rows.append(
            {
                'file': i,
                'member': m,
                'count': 'track (chaz_events)',
                'basin': 'NA',
                'n': int(ev['in_na'].sum()),
            }
        )
        rows.append(
            {
                'file': i,
                'member': m,
                'count': 'track (chaz_events)',
                'basin': 'EP',
                'n': int(ev['in_ep'].sum()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'count.csv', index=False)
    runs = df.groupby('count')[['file', 'member']].apply(lambda d: len(d.drop_duplicates()))
    tab = df.groupby(['count', 'basin'])['n'].sum().unstack('basin')
    per_my = tab.div(runs * years, axis=0)
    print(
        f'storms per member-year counted toward the basin frequency, {len(tasks)} member runs, '
        f'{cr.YEAR_RANGE[0]}-{cr.YEAR_RANGE[1]}'
    )
    print(per_my.round(2).to_string())
    if {'footprint, raw', 'footprint, wrapped'} <= set(tab.index):
        print(
            '\nweight change if the published correction is rerun on wrapped winds (N_raw / N_wrapped):'
        )
        print((tab.loc['footprint, raw'] / tab.loc['footprint, wrapped']).round(3).to_string())
    if 'track (chaz_events)' in tab.index and 'footprint, wrapped' in tab.index:
        print(
            '\nchaz_events weights relative to the published rule on wrapped winds (N_track / N_wrapped):'
        )
        print(
            (tab.loc['track (chaz_events)'] / tab.loc['footprint, wrapped'])[['NA', 'EP']]
            .round(3)
            .to_string()
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('count', help='basin storm counts under each wind construction')
    q.add_argument('--tasks', default='0:0,0:13,0:15,0:16', help='file:member pairs')
    q.set_defaults(func=cmd_count)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
