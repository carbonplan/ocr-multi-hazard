"""How CHAZ intensity changes over open water, against IBTrACS, for North Atlantic storms.

Over water is where the intensity model's land term does not act: both ends of a 12-hour step on
an ocean cell of CHAZ's own 2-degree land mask, the cell get_predictors picks (nearest by its
distance in km). A step starts at an even node of a CHAZ track, where the model stores its
predictions; IBTrACS steps start at every 6-hourly record. Members count when they reach 34 kt
inside the North Atlantic, as in landfall.py.

Each step also records where it starts: the distance to the nearest US coast, the US coastal
region that coast belongs to (landfall.region), and the share of land among the 2-degree cells
within 500 km, the ring whose mean potential intensity the model reads (land cells carry 0).

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


RING_KM = 500.0
LOOKUP = 0.25  # degrees, the grid ring land shares are looked up on
LOOKUP_BOX = (0.0, 60.0, 240.0, 360.0)  # lat0, lat1, lon0, lon1


class Coast:
    """Distance to the US coast and ring land share at any point of the North Atlantic."""

    def __init__(self, ocean: np.ndarray):
        from scipy.spatial import cKDTree

        land = landfall.USLand()
        us = land.mask
        water = ~land.land
        edge = np.zeros_like(us)
        edge[1:-1, 1:-1] = us[1:-1, 1:-1] & (
            water[:-2, 1:-1] | water[2:, 1:-1] | water[1:-1, :-2] | water[1:-1, 2:]
        )
        i, j = np.nonzero(edge)
        self.coast_lat, self.coast_lon = land.lats[i], land.lons[j]
        self.coast_region = landfall.region(self.coast_lat, self.coast_lon)
        self.tree = cKDTree(self._xyz(self.coast_lat, self.coast_lon))
        self.lut_lat = np.arange(LOOKUP_BOX[0], LOOKUP_BOX[1] + LOOKUP, LOOKUP)
        self.lut_lon = np.arange(LOOKUP_BOX[2], LOOKUP_BOX[3] + LOOKUP, LOOKUP)
        self.lut = self._ring_land(ocean)

    @staticmethod
    def _xyz(lat, lon):
        la, lo = np.deg2rad(lat), np.deg2rad(lon)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)

    def _ring_land(self, ocean: np.ndarray) -> np.ndarray:
        """Share of land among the 2-degree cells within RING_KM, by get_predictors' distance."""
        glon, glat = np.meshgrid(GRID_LON, GRID_LAT)
        londis = 2 * np.pi * 6371.0 * np.cos(np.deg2rad(glat)) / 360
        land = ~ocean
        out = np.zeros((self.lut_lat.size, self.lut_lon.size), 'float32')
        for a, lat in enumerate(self.lut_lat):
            for b, lon in enumerate(self.lut_lon):
                d = np.hypot(londis * (glon - lon), 110 * (glat - lat))
                ring = d <= RING_KM
                out[a, b] = land[ring].mean() if ring.any() else np.nan
        return out

    def __call__(self, lat, lon):
        """Distance to the US coast (km), its region, and the ring land share; NaN outside."""
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float) % 360
        ok = np.isfinite(lat) & np.isfinite(lon)
        dist = np.full(lat.shape, np.nan, 'float32')
        reg = np.full(lat.shape, None, object)
        chord, idx = self.tree.query(self._xyz(lat[ok], lon[ok]))
        dist[ok] = 2 * 6371.0 * np.arcsin(np.clip(chord / 2, 0, 1))
        reg[ok] = self.coast_region[idx]
        a = np.round((lat - LOOKUP_BOX[0]) / LOOKUP).astype('int64', copy=False)
        b = np.round((lon - LOOKUP_BOX[2]) / LOOKUP).astype('int64', copy=False)
        inside = ok & (a >= 0) & (a < self.lut_lat.size) & (b >= 0) & (b < self.lut_lon.size)
        ring = np.full(lat.shape, np.nan, 'float32')
        ring[inside] = self.lut[a[inside], b[inside]]
        return dist, reg, ring


def over_water(ocean: np.ndarray, lat, lon) -> np.ndarray:
    ok = np.isfinite(lat) & np.isfinite(lon)
    row, col = model_cell(np.where(ok, lat, 0), np.where(ok, lon, 0))
    return ok & ocean[row, col]


def tables(lat, lon, wind, ocean, source: str, every_node: bool, coast: Coast | None = None):
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
    if coast is not None:
        node = starts[k]
        dist, reg, ring = coast(lat[node, s], lon[node, s])
        steps['lat'] = lat[node, s].astype('float32')
        steps['lon'] = landfall.wrap(lon[node, s]).astype('float32')
        steps['coast_km'] = dist
        steps['coast_region'] = pd.Categorical(reg)
        steps['ring_land'] = ring
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
    coast = Coast(ocean)
    out = {}

    lat, lon, wind, _ = landfall.observed_arrays(landfall.ibtracs_csv(fs))
    out['ibtracs'] = tables(lat, lon, wind, ocean, 'IBTrACS', every_node=True, coast=coast)

    parts = []
    for i in range(10):
        with fs.open(f'{landfall.PUBLISHED}/global_2019_2ens{i:03d}_pre.nc', 'rb') as f:
            ds = xr.open_dataset(f, engine='scipy', decode_times=False).load()
        lat, lon, wind, _ = landfall.chaz_arrays(ds)
        parts.append(tables(lat, lon, wind, ocean, 'published', every_node=False, coast=coast))
        print('published file', i, flush=True)
    out['published'] = tuple(pd.concat(x, ignore_index=True) for x in zip(*parts, strict=True))

    parts = []
    for r in range(10):
        for year in range(landfall.YEARS[0], landfall.YEARS[1] + 1):
            key = f'{store.RUNS}/{run_name}/{year}_r{r:03d}.nc'
            with xr.open_dataset(store.path(key), decode_times=False) as ds:
                lat, lon, wind, _ = landfall.chaz_arrays(ds.load())
            parts.append(tables(lat, lon, wind, ocean, 'ours', every_node=False, coast=coast))
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
