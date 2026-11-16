"""A regional calibration of where CHAZ forms storms, as a fixed multiplier map on the genesis
index.

CHAZ seeds storms in proportion to the genesis index (TCGI). Where its storms reach tropical-storm
strength can differ from where observed storms do. The multiplier is the observed density of
tropical-storm formation (the first IBTrACS USA_WIND record at 34 kt or more, 1981-2019) over the
same density from a CHAZ run (the first 6-hourly node at which each intensity member reaches
34 kt), both on the 2-degree TCGI grid, smoothed with a Gaussian of SIGMA degrees and clipped to
CLIP. chaz.py applies it to the index when it seeds storms and when it computes the seeding ratio
(--genesis calibrated), so the global count stays calibrated and a climate model's change in the
index still drives its change in genesis.

    python -m ocr.risks.hurricane.calibrate genesis --run TCGI_CRH_PI-land-fine-normalization-env
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from ocr.risks.hurricane import landfall, store

YEARS = (1981, 2019)
SIGMA = 5.0  # degrees
CLIP = (0.25, 4.0)
FLOOR = 0.02  # where the smoothed CHAZ density is below FLOOR / (SIGMA / 2)**2 a year, the map is 1
LAT = np.arange(-89.0, 90.0, 2.0)
LON = np.arange(0.0, 360.0, 2.0)
OUT = 'validation/genesis'


def key(tcgi_name: str) -> str:
    return f'{OUT}/multiplier_{tcgi_name}.nc'


def formation(lat, lon, wind) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latitude, longitude and count weight (1 per storm, 1/members per CHAZ member) where each
    member first reaches 34 kt."""
    n_m = wind.shape[0]
    with np.errstate(invalid='ignore'):
        ts = wind >= landfall.TS
    reached = ts.any(axis=1)  # (member, storm)
    node = np.argmax(ts, axis=1)
    m, s = np.nonzero(reached)
    return lat[node[m, s], s], lon[node[m, s], s], np.full(m.size, 1.0 / n_m)


def density(lat, lon, weight, years: int) -> np.ndarray:
    """Formation per year on the 2-degree grid (lat, lon), before smoothing."""
    lon = (np.asarray(lon) + 1.0) % 360.0 - 1.0
    h, _, _ = np.histogram2d(
        lat,
        lon,
        bins=[np.arange(-90.0, 91.0, 2.0), np.arange(-1.0, 360.0, 2.0)],
        weights=weight,
    )
    return h / years


def smooth(d: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(d, SIGMA / 2.0, mode=['constant', 'wrap'])


def genesis(run_name: str) -> xr.Dataset:
    fs = store.s3()
    n_years = YEARS[1] - YEARS[0] + 1
    obs = density(*formation(*landfall.observed_arrays(landfall.ibtracs_csv(fs))[:3]), n_years)
    parts = []
    for r in range(10):
        for year in range(YEARS[0], YEARS[1] + 1):
            k = f'runs/era5/{run_name}/{year}_r{r:03d}.nc'
            with xr.open_dataset(store.path(k), decode_times=False) as ds:
                parts.append(formation(*landfall.chaz_arrays(ds.load(), YEARS)[:3]))
    lat, lon, w = (np.concatenate(x) for x in zip(*parts, strict=True))
    chaz_d = density(lat, lon, w, n_years * 10)
    so, sc = smooth(obs), smooth(chaz_d)
    with np.errstate(divide='ignore', invalid='ignore'):
        m = np.where(sc > FLOOR * (SIGMA / 2.0) ** -2, so / sc, 1.0)
    m = np.clip(np.nan_to_num(m, nan=1.0), *CLIP)
    return xr.Dataset(
        {
            'multiplier': (('lat', 'lon'), m),
            'observed': (('lat', 'lon'), obs, {'units': 'storms per year'}),
            'chaz': (('lat', 'lon'), chaz_d, {'units': 'storms per year per member'}),
        },
        coords={'lat': LAT, 'lon': LON},
        attrs={
            'method': __doc__.split('\n\n')[1].replace('\n', ' '),
            'run': f'runs/era5/{run_name}',
            'years': f'{YEARS[0]}-{YEARS[1]}',
            'sigma_deg': SIGMA,
            'clip': f'{CLIP[0]}, {CLIP[1]}',
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('genesis')
    g.add_argument('--run', required=True, help='CHAZ ERA5 run under runs/era5')
    g.add_argument('--tcgi', default='TCGI_CRH_PI')
    a = p.parse_args()
    ds = genesis(a.run)
    k = key(a.tcgi)
    local = store.CACHE / k
    local.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(local)
    store.push(keys=[k], replace=True)
    m = ds['multiplier']
    print(k, 'multiplier quantiles', np.round(np.quantile(m, [0.01, 0.1, 0.5, 0.9, 0.99]), 2))
    for name, (la, lo) in {
        'Gulf 25N 90W': (25, 270),
        'Caribbean 15N 75W': (15, 286),
        'MDR 12N 40W': (13, 320),
        'W Atlantic 28N 72W': (27, 288),
        'WP 15N 135E': (15, 134),
        'S China Sea 15N 115E': (15, 114),
        'EP 14N 105W': (13, 254),
        'Bay of Bengal 15N 88E': (15, 88),
        'S Indian 15S 70E': (-15, 70),
        'Australia 15S 120E': (-15, 120),
        'S Pacific 15S 170E': (-15, 170),
    }.items():
        print(f'  {name}: {float(m.sel(lat=la, lon=lo, method="nearest")):.2f}')


if __name__ == '__main__':
    main()
