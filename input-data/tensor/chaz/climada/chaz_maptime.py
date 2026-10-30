"""Time the published map step: CLIMADA's per-cell ranking on one tile.

The published map scripts (CHAZ-hazard-maps main/compute_*_era5_parallel.py)
load the global frequency-corrected hazard, select a lon/lat tile, and call
local_exceedance_intensity and local_return_period. Both loop over the tile's
cells, and for each cell they densify its column over every event row
(getcol(i).toarray()). Selecting a tile keeps every row. The ERA5 file holds
1,395,323 storms six times over (one copy per basin), so each cell's cost
grows with that row count as well as with the tile's nonzeros.

This times the same loop body on a tile cut from the cached CONUS crop of her
ERA5 set (chaz_published.py). The tile is padded with empty rows up to her row
counts, so it matches what her scripts see, and is also timed with the empty
rows dropped. A real CLIMADA call on a small tile checks the timing against
the replica. A numpy ranking (the algorithm of chaz_tail_test.Ranked) is
timed on the same tile and checked against CLIMADA's values.

  run   --tile -90,-75,25,40 --sample 200    timing table -> ~/.cache/ocr/chaz-local/maptime/

Run in the `climada` pixi environment.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

CROP = Path.home() / '.cache' / 'ocr' / 'chaz-events' / 'published' / 'conus' / 'era5_conus.hdf5'
OUT = Path.home() / '.cache' / 'ocr' / 'chaz-local' / 'maptime'
MAPS_REPO = Path.home() / 'dev' / 'carbonplan' / 'CHAZ-hazard-maps'

ROWS = {
    'pruned': None,  # events with wind in the tile
    'one copy': 1_395_323,  # her ERA5 catalogue
    'six copies': 8_371_938,  # the frequency-corrected global file
}
RETURN_PERIODS = (10, 25, 50, 100, 250, 1000)
THRESHOLDS = (33.0, 50.0)
METHOD = 'extrapolate_constant'
GLOBAL_CELLS = 929_389


def parse_tile(s: str) -> tuple[float, float, float, float]:
    lon0, lon1, lat0, lat1 = (float(v) for v in s.split(','))
    return lon0, lon1, lat0, lat1


def load_tile(tile):
    from climada.hazard import Hazard

    haz = Hazard.from_hdf5(str(CROP))
    haz = haz.select(extent=tile)
    keep = np.nonzero(haz.intensity.getnnz(axis=1) > 0)[0]
    return haz, keep


def padded(haz, keep, n_rows):
    """The tile's columns over n_rows events: its events with wind, then empty
    rows, the way a tile of the global file looks."""
    from climada.hazard import Hazard

    inten = haz.intensity[keep].tocsr()
    freq = haz.frequency[keep]
    if n_rows is not None and n_rows > inten.shape[0]:
        extra = n_rows - inten.shape[0]
        inten = sparse.vstack([inten, sparse.csr_matrix((extra, inten.shape[1]))], format='csr')
        freq = np.concatenate([freq, np.full(extra, freq.mean())])
    out = Hazard(
        haz_type=haz.haz_type,
        units=haz.units,
        centroids=haz.centroids,
        event_id=np.arange(inten.shape[0]),
        frequency=freq,
        intensity=inten,
        fraction=sparse.csr_matrix(inten.shape),
    )
    return out


def time_loop(haz, cells, metric: str) -> float:
    """Seconds per cell for CLIMADA's per-cell loop body, on the given cells."""
    import climada.util.interpolation as u_interp

    t0 = time.perf_counter()
    for i in cells:
        col = haz.intensity.getcol(i).toarray().flatten()
        if metric == 'exceedance':
            u_interp.preprocess_and_interpolate_ev(
                1 / np.array(RETURN_PERIODS),
                None,
                haz.frequency,
                col,
                log_frequency=True,
                log_values=True,
                value_threshold=0,
                method=METHOD,
                y_asymptotic=0.0,
            )
        else:
            u_interp.preprocess_and_interpolate_ev(
                None,
                np.array(THRESHOLDS),
                haz.frequency,
                col,
                log_frequency=True,
                log_values=True,
                value_threshold=0,
                method=METHOD,
                y_asymptotic=np.nan,
            )
    return (time.perf_counter() - t0) / len(cells)


def numpy_exceedance(intensity, frequency, return_periods=RETURN_PERIODS) -> np.ndarray:
    """(n_cells, len(T)) winds by sorting every nonzero once; the algorithm of
    chaz_tail_test.Ranked (log-log interpolation, constant extrapolation)."""
    coo = intensity.tocoo()
    keep = coo.data > 0
    r, c, v = coo.row[keep], coo.col[keep], coo.data[keep].astype('float64')
    order = np.lexsort((-v, c))
    col, val, freq = c[order], v[order], frequency[r[order]]
    n = intensity.shape[1]
    cells = np.arange(n)
    start = np.searchsorted(col, cells, side='left')
    end = np.searchsorted(col, cells, side='right')
    cs = np.cumsum(freq)
    cumfreq = cs - np.concatenate([[0.0], cs])[start][col]
    scale = 1.0 / (cumfreq.max() * 1.01 + 1e-12)
    keys = col + cumfreq * scale
    has = end > start
    out = np.zeros((n, len(return_periods)))
    for k, T in enumerate(return_periods):
        f = 1.0 / T
        pos = np.searchsorted(keys, cells + f * scale, side='left')
        top = has & (pos <= start)
        out[top, k] = val[start[top]]
        inside = has & (pos > start) & (pos < end)
        p = pos[inside]
        f0, f1 = np.log(cumfreq[p - 1]), np.log(cumfreq[p])
        v0, v1 = np.log(val[p - 1]), np.log(val[p])
        w = (np.log(f) - f0) / (f1 - f0)
        out[inside, k] = np.exp(v0 + w * (v1 - v0))
    return out


def cmd_run(args):
    import logging

    logging.getLogger('climada').setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    tile = parse_tile(args.tile)
    t0 = time.perf_counter()
    haz, keep = load_tile(tile)
    nz_cells = np.nonzero(haz.intensity.getnnz(axis=0) > 0)[0]
    print(
        f'tile {tile}: {haz.intensity.shape[1]:,} cells ({nz_cells.size:,} with wind), '
        f'{keep.size:,} events with wind, {haz.intensity.nnz:,} nonzeros; '
        f'loaded in {time.perf_counter() - t0:.0f}s'
    )
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(nz_cells, size=min(args.sample, nz_cells.size), replace=False))

    rows = []
    for label, n_rows in ROWS.items():
        h = padded(haz, keep, n_rows)
        for metric in ('exceedance', 'return_period'):
            s = time_loop(h, sample, metric)
            rows.append(
                {
                    'rows': label,
                    'n_rows': h.intensity.shape[0],
                    'metric': metric,
                    's_per_cell': s,
                    'tile_hours': s * nz_cells.size / 3600,
                    'global_hours': s * GLOBAL_CELLS / 3600,
                }
            )
            print(
                f'{label:>10} ({h.intensity.shape[0]:>9,} rows) {metric:>13}: '
                f'{s * 1e3:8.1f} ms/cell, tile {s * nz_cells.size / 3600:6.2f} h, '
                f'global {s * GLOBAL_CELLS / 3600:8.1f} h'
            )
        del h

    # check the replica against a real CLIMADA call, and numpy against CLIMADA
    small = parse_tile(args.check_tile)
    hs = haz.select(extent=small)
    ks = np.nonzero(hs.intensity.getnnz(axis=1) > 0)[0]
    nz_small = np.nonzero(hs.intensity.getnnz(axis=0) > 0)[0]
    for label in ('pruned', 'one copy'):
        h = padded(hs, ks, ROWS[label])
        t1 = time.perf_counter()
        gdf, _, _ = h.local_exceedance_intensity(return_periods=RETURN_PERIODS, method=METHOD)
        real = (time.perf_counter() - t1) / nz_small.size
        replica = time_loop(h, nz_small[: args.sample], 'exceedance')
        print(
            f'check {small} {label}: CLIMADA {real * 1e3:.1f} ms/cell, '
            f'replica {replica * 1e3:.1f} ms/cell over {nz_small.size:,} cells'
        )
        if label == 'pruned':
            climada_vals = gdf[[str(T) for T in RETURN_PERIODS]].to_numpy()
    h = padded(hs, ks, None)
    t1 = time.perf_counter()
    np_vals = numpy_exceedance(h.intensity, h.frequency)
    t_np_small = time.perf_counter() - t1
    diff = np.nanmax(np.abs(np_vals - climada_vals))
    print(f'numpy vs CLIMADA on {small}: max abs difference {diff:.2e} m/s ({t_np_small:.2f}s)')

    h = padded(haz, keep, None)
    t1 = time.perf_counter()
    numpy_exceedance(h.intensity, h.frequency)
    t_np = time.perf_counter() - t1
    print(
        f'numpy ranking, whole tile ({nz_cells.size:,} cells, {h.intensity.nnz:,} nonzeros): {t_np:.1f}s'
    )

    t1 = time.perf_counter()
    sys.path.insert(0, str(MAPS_REPO))
    from main.hazard_map_utils import gdf_to_raster

    gdf_to_raster(
        gdf,
        OUT / 'raster_check.nc',
        variable_prefix='rp',
        grid_res=0.05,
        method='linear',
    )
    t_raster = time.perf_counter() - t1
    print(f'gdf_to_raster on the {nz_small.size:,}-cell check tile: {t_raster:.1f}s')

    df = pd.DataFrame(rows)
    df.attrs = {}
    name = 'maptime_' + '_'.join(f'{v:g}' for v in tile) + '.csv'
    df.to_csv(OUT / name, index=False)
    print(f'-> {OUT / name}')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('run', help='time the per-cell ranking on one tile')
    q.add_argument('--tile', default='-90,-75,25,40', help='lon_min,lon_max,lat_min,lat_max')
    q.add_argument('--sample', type=int, default=200, help='cells timed per configuration')
    q.add_argument(
        '--check-tile', default='-82,-80,26,28', help='small tile for the CLIMADA and numpy checks'
    )
    q.set_defaults(func=cmd_run)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
