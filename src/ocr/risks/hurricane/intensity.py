"""How CHAZ intensity changes over open water, against IBTrACS, for North Atlantic storms.

Over water is where the intensity model's land term does not act: both ends of a 12-hour step on
an ocean cell of CHAZ's own 2-degree land mask, the cell get_predictors picks (nearest by its
distance in km). A step starts at an even node of a CHAZ track, where the model stores its
predictions; IBTrACS steps start at every 6-hourly record. Members count when they reach 34 kt
inside the North Atlantic, as in landfall.py.

  run    the step and member tables for IBTrACS, the published ERA5 tracks and one of our runs,
         under validation/chaz/intensity
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from ocr.risks.hurricane import chaz, landfall, store

OUT = 'validation/chaz/intensity'
STEP = 2  # nodes: 12 hours


def ocean_mask() -> np.ndarray:
    """CHAZ's 2-degree land mask as the intensity model reads it, True over ocean; rows are
    latitudes -89..89, columns longitudes 0..358."""
    chaz._load_landmask()
    return np.asarray(chaz.gv.ldmask) == 0


GRID_LAT = np.arange(-89.0, 90.0, 2.0)
GRID_LON = np.arange(0.0, 360.0, 2.0)


def model_cell(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the 2-degree cell get_predictors reads for each point: the nearest by
    its distance, with east-west kilometres scaled by the cell's latitude, ties to the lower row
    then column, longitudes 0-360 without wrapping. Searches the cells around each point."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float) % 360
    r0 = np.floor((lat + 89) / 2).astype(int)
    c0 = np.floor(lon / 2).astype(int)
    best = np.full(lat.shape, np.inf)
    row = np.zeros(lat.shape, int)
    col = np.zeros(lat.shape, int)
    for dr in (-1, 0, 1, 2):
        for dc in (0, 1):
            r = np.clip(r0 + dr, 0, GRID_LAT.size - 1)
            c = np.clip(c0 + dc, 0, GRID_LON.size - 1)
            glat, glon = GRID_LAT[r], GRID_LON[c]
            dx = 2 * np.pi * 6371.0 * np.cos(np.deg2rad(glat)) / 360 * (glon - lon)
            dy = 110 * (glat - lat)
            d = np.sqrt(dx * dx + dy * dy)
            better = (d < best) | ((d == best) & ((r < row) | ((r == row) & (c < col))))
            best, row, col = (
                np.where(better, d, best),
                np.where(better, r, row),
                np.where(better, c, col),
            )
    return row, col


def over_water(ocean: np.ndarray, lat, lon) -> np.ndarray:
    ok = np.isfinite(lat) & np.isfinite(lon)
    row, col = model_cell(np.where(ok, lat, 0), np.where(ok, lon, 0))
    return ok & ocean[row, col]


def tables(lat, lon, wind, ocean, source: str, every_node: bool):
    """(steps, members) for the members of one track set that reach 34 kt in the North Atlantic.

    steps: one row per 12-hour step over water, with the wind at its start (v0), the change over
    it (dv) and hours since the member's lifetime peak at its start. members: peak, hours with
    wind, hours from peak to the last wind, and whether the last wind is over water."""
    n_m, n_n, n_s = wind.shape
    in_na = landfall.in_north_atlantic(lat, landfall.wrap(lon)) & np.isfinite(lat)
    with np.errstate(invalid='ignore'):
        keep = ((wind >= landfall.TS) & in_na[None]).any(axis=1)  # (member, storm)
    water = over_water(ocean, lat, lon)
    alive = np.isfinite(wind)
    with np.errstate(invalid='ignore'):
        peak = np.max(np.where(alive, wind, -np.inf), axis=1)
    peak_node = np.argmax(np.where(alive, wind, -np.inf), axis=1)
    first = np.argmax(alive, axis=1)
    last = n_n - 1 - np.argmax(alive[:, ::-1], axis=1)

    starts = np.arange(0, n_n - STEP, 1 if every_node else STEP)
    v0, v1 = wind[:, starts], wind[:, starts + STEP]
    ok = (
        np.isfinite(v0)
        & np.isfinite(v1)
        & water[starts][None]
        & water[starts + STEP][None]
        & keep[:, None, :]
    )
    m, k, s = np.nonzero(ok)
    steps = pd.DataFrame(
        {
            'source': source,
            'v0': v0[m, k, s].astype('float32'),
            'dv': (v1 - v0)[m, k, s].astype('float32'),
            'h_since_peak': (6 * (starts[k] - peak_node[m, s])).astype('int16'),
        }
    )
    m, s = np.nonzero(keep)
    members = pd.DataFrame(
        {
            'source': source,
            'peak': peak[m, s].astype('float32'),
            'hours_with_wind': (6 * (last - first)[m, s]).astype('int16'),
            'peak_to_end_h': (6 * (last - peak_node)[m, s]).astype('int16'),
            'ends_over_water': water[last[m, s], s],
        }
    )
    return steps, members


def run(run_name: str = 'TCGI_CRH_PI') -> None:
    fs = store.s3()
    ocean = ocean_mask()
    out = {}

    lat, lon, wind, _ = landfall.observed_arrays(landfall.ibtracs_csv(fs))
    out['ibtracs'] = tables(lat, lon, wind, ocean, 'IBTrACS', every_node=True)

    parts = []
    for i in range(10):
        with fs.open(f'{landfall.PUBLISHED}/global_2019_2ens{i:03d}_pre.nc', 'rb') as f:
            ds = xr.open_dataset(f, engine='scipy', decode_times=False).load()
        lat, lon, wind, _ = landfall.chaz_arrays(ds)
        parts.append(tables(lat, lon, wind, ocean, 'published', every_node=False))
        print('published file', i, flush=True)
    out['published'] = tuple(pd.concat(x, ignore_index=True) for x in zip(*parts, strict=True))

    parts = []
    for r in range(10):
        for year in range(landfall.YEARS[0], landfall.YEARS[1] + 1):
            key = f'{store.RUNS}/{run_name}/{year}_r{r:03d}.nc'
            with xr.open_dataset(store.path(key), decode_times=False) as ds:
                lat, lon, wind, _ = landfall.chaz_arrays(ds.load())
            parts.append(tables(lat, lon, wind, ocean, 'ours', every_node=False))
        print('our realization', r, flush=True)
    out[f'ours_{run_name}'] = tuple(
        pd.concat(x, ignore_index=True) for x in zip(*parts, strict=True)
    )

    local = store.CACHE / OUT
    local.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (steps, members) in out.items():
        for kind, df in (('steps', steps), ('members', members)):
            written.append(f'{OUT}/{kind}_{name}.parquet')
            df.to_parquet(store.CACHE / written[-1])
        print(name, len(steps), 'water steps', len(members), 'members', flush=True)
    store.push(fragment=f'intensity-{run_name}', replace=True, keys=written)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--run', default='TCGI_CRH_PI', help='run under runs/era5 (chaz.run_name)')
    a = p.parse_args()
    run(a.run)


if __name__ == '__main__':
    main()
