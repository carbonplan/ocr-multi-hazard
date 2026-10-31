"""Score the CHAZ ERA5 CONUS sets against observed storms run through the same winds.

The observed storms are the IBTrACS v04r01 tracks (US agency) of 1981-2019
that put tropical-storm winds within 5 degrees of the CONUS box. They get
Holland (2008) winds from the installed CLIMADA at 0.5 h steps on the CONUS
cells of chaz_events, under one of three conventions:

  track    IBTrACS central pressure, radius of maximum wind and outer
           pressure, missing values estimated; every record
  chaz     the CHAZ reader's conventions with both fixes: 6-hourly records,
           pressure from wind with longitudes in -180..180, radius estimated,
           environmental pressure 1010 hPa
  chaz360  the same with longitudes in 0-360, as the reader passes them

track is the most realistic wind the record supports. chaz builds winds the
way the corrected CHAZ set does, so comparing the two sets isolates CHAZ's
storm climatology from the wind construction.

The CHAZ side is the sets written by chaz_events levels (published, era5_10f,
era5_10f_wrapped), each with its per-event frequencies. The metric is storms
per year whose peak wind on US land cells (LitPop value > 0) reaches 17.5, 33,
43 or 50 m/s, nationally and in the coastal regions of eberenz_na2 ead.
Observed rates carry a 90% Poisson interval.

  wind     --conv track         observed/obs_<conv>.hdf5
  compare                       rate table, observed/compare.csv
  ead      65.4 89.2            event-based damage per region, observed/ead.csv

Outputs go to ~/.cache/ocr/chaz-local/observed. wind runs in the `climada`
pixi environment, compare in either.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chaz_events as ce

ROOT = Path(os.environ.get('CHAZ_LOCAL_DIR', Path.home() / '.cache' / 'ocr' / 'chaz-local'))
OUT = ROOT / 'observed'
IBTRACS = Path.home() / 'climada' / 'data' / 'IBTrACS.since1980.v04r01.nc'
YEARS = (1981, 2019)
SETS = ('published', 'era5_10f', 'era5_10f_wrapped')
OBS = ('track', 'chaz', 'chaz360')
THRESHOLDS = (17.5, 33.0, 43.0, 50.0)
CONVENTIONS = {
    'chaz': SimpleNamespace(records='6h', pressure='wind180', penv='1010'),
    'chaz360': SimpleNamespace(records='6h', pressure='wind360', penv='1010'),
}

log = logging.getLogger('chaz_observed')


def reaching(tr) -> bool:
    strong = tr['max_sustained_wind'].values >= ce.INTENSITY_THRES * ce.KT_PER_MS
    return bool((strong & ce.in_conus_reach(tr['lat'].values, tr['lon'].values)).any())


def cmd_wind(args):
    import eberenz_na2
    from climada.hazard import Centroids, TCTracks
    from climada.hazard.tc_tracks import _estimate_pressure

    logging.getLogger('climada').setLevel(logging.WARNING)
    OUT.mkdir(parents=True, exist_ok=True)
    tracks = TCTracks.from_ibtracs_netcdf(
        provider='usa', year_range=YEARS, estimate_missing=True, file_name=str(IBTRACS)
    )
    n_all = tracks.size
    tracks.data = [tr for tr in tracks.data if reaching(tr)]
    if args.conv in CONVENTIONS:
        conv = CONVENTIONS[args.conv]
        tracks.data = [eberenz_na2._records(tr, conv, _estimate_pressure) for tr in tracks.data]
        # 6-hourly records can drop a storm's only nodes within reach
        tracks.data = [
            tr for tr in tracks.data if ce.in_conus_reach(tr['lat'].values, tr['lon'].values).any()
        ]
    tracks.data = [ce.crop_to_reach(tr) for tr in tracks.data]
    tracks.equal_timestep(ce.TIME_STEP_H)
    if args.conv in CONVENTIONS:
        for tr in tracks.data:
            tr['radius_max_wind'] = ('time', np.zeros(tr.sizes['time']))
    cent = Centroids.from_hdf5(str(ce._local(ce.CENTROIDS_KEY)))
    haz = ce.winds_parallel(tracks, cent, args.workers, args.max_memory_gb)
    haz.frequency = np.full(haz.size, 1.0 / (YEARS[1] - YEARS[0] + 1))
    out = OUT / f'obs_{args.conv}.hdf5'
    haz.write_hdf5(str(out))
    print(
        f'{args.conv}: {tracks.size} of {n_all} storms {YEARS[0]}-{YEARS[1]} reach CONUS; '
        f'{haz.intensity.nnz:,} nonzeros, max {haz.intensity.max():.1f} m/s -> {out}'
    )


def read_set(path: Path):
    """Intensity (events x cells), per-event frequency and cell coordinates of
    a CLIMADA hazard file; the published crop carries no centroids and uses
    the published levels' points."""
    import h5py
    import xarray as xr
    from scipy import sparse

    with h5py.File(path) as f:
        inten = sparse.csr_matrix(
            (f['intensity/data'][:], f['intensity/indices'][:], f['intensity/indptr'][:])
        )
        freq = f['frequency'][:]
        if 'centroids' in f:
            # CLIMADA 6 writes the centroids as a pandas table; the
            # coordinates sit in the float block as geometry.x and geometry.y
            items = [i.decode() for i in f['centroids/block0_items'][:]]
            v = f['centroids/block0_values'][:]
            lat, lon = v[:, items.index('geometry.y')], v[:, items.index('geometry.x')]
        else:
            pts = xr.open_dataset(ce.CACHE / 'hazard' / 'ERA5' / 'levels' / 'published.nc')
            lat, lon = pts['lat'].values, pts['lon'].values
    return inten, freq, np.asarray(lat, float), np.asarray(lon, float)


def us_cells():
    import xarray as xr

    d = xr.open_dataset(ce.CACHE / 'tail' / 'published_decompose.nc')
    lat, lon = d['lat'].values, d['lon'].values
    us = d['litpop_value_us'].values > 0
    return dict(zip(ce.cell_key(lat[us], lon[us]), np.ones(us.sum(), bool)))


def regions(lat, lon) -> dict[str, np.ndarray]:
    """eberenz_na2 ead's coastal regions."""
    r = {
        'TX-AL': (lon < -87.6) & (lat < 33),
        'FL': (lon >= -87.6) & (lat < 31),
        'GA-Car': (lon >= -87.6) & (lat >= 31) & (lat < 36.6),
        'VA-ME': (lon >= -82) & (lat >= 36.6),
    }
    return {'US': np.ones(lat.size, bool), **r}


def rates(inten, freq, lat, lon, us) -> dict[tuple[str, float], float]:
    keys = ce.cell_key(lat, lon)
    on_us = np.array([k in us for k in keys])
    out = {}
    for name, m in regions(lat, lon).items():
        cols = np.nonzero(m & on_us)[0]
        peak = inten[:, cols].max(axis=1).toarray().ravel()
        for thr in THRESHOLDS:
            out[(name, thr)] = float(freq[peak >= thr].sum())
    return out


def poisson_90(n: int, years: int) -> tuple[float, float]:
    from scipy.stats import chi2

    lo = chi2.ppf(0.05, 2 * n) / 2 if n else 0.0
    hi = chi2.ppf(0.95, 2 * (n + 1)) / 2
    return lo / years, hi / years


def cmd_compare(args):
    us = us_cells()
    years = YEARS[1] - YEARS[0] + 1
    rows = []
    for name in OBS:
        path = OUT / f'obs_{name}.hdf5'
        if not path.exists():
            continue
        r = rates(*read_set(path), us)
        for (reg, thr), v in r.items():
            n = round(v * years)
            lo, hi = poisson_90(n, years)
            rows.append(
                {
                    'source': f'observed {name}',
                    'region': reg,
                    'thr': thr,
                    'rate': v,
                    'lo': lo,
                    'hi': hi,
                }
            )
    for name in SETS:
        path = ce.CACHE / 'hazard' / 'ERA5' / 'sets' / f'{name}.hdf5'
        r = rates(*read_set(path), us)
        rows += [
            {'source': f'CHAZ {name}', 'region': reg, 'thr': thr, 'rate': v}
            for (reg, thr), v in r.items()
        ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'compare.csv', index=False)
    order = [f'observed {n}' for n in OBS] + [f'CHAZ {n}' for n in SETS]
    pd.set_option('display.width', 200)
    for thr in THRESHOLDS:
        t = df[df['thr'] == thr].copy()
        t['cell'] = [
            f'{r:.2f}' if np.isnan(lo) else f'{r:.2f} ({lo:.2f}-{hi:.2f})'
            for r, lo, hi in zip(t['rate'], t.get('lo', np.nan), t.get('hi', np.nan))
        ]
        tab = t.pivot(index='source', columns='region', values='cell')
        tab = tab.reindex([o for o in order if o in tab.index])[
            list(regions(np.zeros(1), np.zeros(1)))
        ]
        print(f'\nstorms per year with peak wind >= {thr} m/s on US land cells')
        print(tab.to_string())


def cmd_ead(args):
    """Expected annual damage from the events themselves: the impact
    function on every cell's wind, times LitPop value, times event frequency.
    Observed storms and CHAZ sets go through the same function and values."""
    import xarray as xr

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chaz_damage import _mdr

    d = xr.open_dataset(ce.CACHE / 'tail' / 'published_decompose.nc')
    value_at = dict(zip(ce.cell_key(d['lat'].values, d['lon'].values), d['litpop_value_us'].values))
    sources = [(f'observed {n}', OUT / f'obs_{n}.hdf5') for n in OBS]
    sources += [(f'CHAZ {n}', ce.CACHE / 'hazard' / 'ERA5' / 'sets' / f'{n}.hdf5') for n in SETS]
    rows = []
    for name, path in sources:
        if not path.exists():
            continue
        inten, freq, lat, lon = read_set(path)
        value = np.array([value_at.get(k, 0.0) for k in ce.cell_key(lat, lon)])
        for vh in args.v_halfs:
            dmg = inten.copy().astype('float64')
            dmg.data = _mdr(dmg.data, vh)
            per_cell = np.asarray(dmg.T @ freq).ravel() * value
            for reg, m in regions(lat, lon).items():
                rows.append(
                    {'source': name, 'v_half': vh, 'region': reg, 'ead_bn': per_cell[m].sum() / 1e9}
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'ead.csv', index=False)
    tab = df.pivot_table(index=['v_half', 'source'], columns='region', values='ead_bn', sort=False)
    pd.set_option('display.width', 200)
    print('expected annual damage, $bn/yr on LitPop 2020 values')
    print(tab[list(regions(np.zeros(1), np.zeros(1)))].round(2).to_string())


def main():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(name)s: %(message)s')
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('wind', help='winds for the observed storms that reach CONUS')
    q.add_argument('--conv', default='track', choices=OBS)
    q.add_argument('--workers', type=int, default=max((os.cpu_count() or 2) - 6, 1))
    q.add_argument('--max-memory-gb', type=float, default=24.0)
    q.set_defaults(func=cmd_wind)
    q = sub.add_parser('compare', help='rates against the CHAZ sets')
    q.set_defaults(func=cmd_compare)
    q = sub.add_parser('ead', help='event-based damage of observed storms and CHAZ sets')
    q.add_argument('v_halfs', nargs='+', type=float)
    q.set_defaults(func=cmd_ead)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
