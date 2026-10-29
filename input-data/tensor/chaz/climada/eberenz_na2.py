"""Reproduce the Eberenz et al. (2021) NA2 v_half and refit it on other CLIMADA versions.

Eberenz et al. (https://doi.org/10.5194/nhess-21-393-2021) fit v_half for the
USA and Canada on CLIMADA 1.4.1 against 43 EM-DAT events. CLIMADA ships their
per-event table (tc_impf_cal_v01_EDR.csv: IBTrACS IDs, damages normalized to
2014, per-event fits) and climada_papers the simulated damage per event under
the default curve. Their processed tracks and exposure are not published, so
both are rebuilt here, and the fit can be repeated on winds from any CLIMADA
version and track convention.

  prep   the 43 events, the paper's per-event damages, LitPop at 300 arcsec
         rescaled to the paper's 2014 totals, and the IBTrACS file
  wind   Holland (2008) winds for the 43 storms under the installed CLIMADA
  fit    TDR and RMSF fits, Table A2 statistics, per-event ratios to the paper
  ead    CONUS EAD of the corrected-wind ERA5 rebuild against the served store

The original fit rests on two conventions the paper does not state: CLIMADA
1.4.1 replaced every radius of maximum wind with a function of central
pressure (--radius p141), and the processed tracks held 6-hourly synoptic
records only (--records 6h). CLIMADA's CHAZ reader hands the wind model
6-hourly records with no radius, central pressure estimated from wind with
longitudes in 0-360 (--pressure wind360) and environmental pressure 1010 hPa
(--penv 1010).

  E="pixi run -e climada python input-data/tensor/chaz/climada/eberenz_na2.py"
  $E prep
  $E wind --radius p141 --records 6h
  uv run --no-project --python 3.10 --with climada==4.0.1 --with 'numpy<2' \\
    --with 'pandas<2.2' --with 'xarray<2024.3' --with 'scipy<1.13' --with 'shapely<2.1' \\
    --with 'geopandas<1' --with 'pyproj<3.7' --with 'dask<2024.3' --with pyarrow \\
    python input-data/tensor/chaz/climada/eberenz_na2.py wind --radius p141 --records 6h
  pixi run python input-data/tensor/chaz/climada/eberenz_na2.py fit 4.0.1_p141_6h 6.1.0_p141_6h
  pixi run python input-data/tensor/chaz/climada/eberenz_na2.py ead 70.2 68.0

wind runs under CLIMADA 6.1 or 4.0.1, the last release before the 4.1.0 wind
fix; fit and ead read chaz_damage.py and run in the default environment.
Everything lands in ~/.cache/ocr/eberenz-na2 (EBERENZ_NA2_DIR).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy import sparse

WORK = Path(os.environ.get('EBERENZ_NA2_DIR', Path.home() / '.cache' / 'ocr' / 'eberenz-na2'))
CHAZ_EVENTS = Path(
    os.environ.get('CHAZ_EVENTS_DIR', Path.home() / '.cache' / 'ocr' / 'chaz-events')
)

REGION = 'NA2'
ISO = {'USA': 840, 'CAN': 124}
# produced capital, climada_papers 202004_metadata_countries_v1_2.csv
TOTAL_2014 = {'USA': 5.56e13, 'CAN': 6.59e12}
# Luis, renumbered between IBTrACS v04r00 and v04r01
RENAMED = {'1995240N11337': '1995241N11333'}
IBTRACS = Path.home() / 'climada' / 'data' / 'IBTrACS.since1980.v04r01.nc'
IBTRACS_URL = (
    'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/'
    'v04r01/access/netcdf/IBTrACS.since1980.v04r01.nc'
)
PAPER_DAMAGE_URL = (
    'https://raw.githubusercontent.com/CLIMADA-project/climada_papers/'
    '8f616a090cc87b985067475b83f0ff6a518da220/archive/202005_tropical_cyclone_calibration/'
    'data/plot_data/tc_cal_202002_default_IF.csv'
)

DEFAULT_VHALF = 74.7
TABLE_A2 = {74.7: (2.11, 9.5), 86.0: (1.16, 8.7), 89.2: (1.00, 8.7)}  # NA2: TDR, RMSF
PUBLISHED_FIT = (89.2, 86.0)  # TDR, RMSF

YRLY_FREQ_IB = {'NA': 10.8, 'EP': 14.5}  # chaz_events.YRLY_FREQ_IB
# storm counts behind the published per-event frequencies, README section 4
PUBLISHED_STORMS = {'NA': 207_708, 'EP': 319_096}


def read_events(work: Path) -> list[dict]:
    with open(work / 'events.csv') as f:
        return list(csv.DictReader(f))


def wind_tag(version: str, args) -> str:
    tag = f'{version}_{args.radius}_{args.records}'
    if args.pressure != 'track':
        tag += f'_{args.pressure}'
    if args.penv != 'track':
        tag += f'_penv{args.penv}'
    if args.time_step != 1:
        tag += f'_dt{args.time_step:g}'
    return tag


def cmd_prep(args) -> None:
    import logging

    import climada
    import xarray as xr
    from climada.util.api_client import Client
    from scipy.spatial import cKDTree

    logging.getLogger('climada').setLevel(logging.WARNING)
    out = args.dir
    out.mkdir(parents=True, exist_ok=True)
    if not IBTRACS.exists():
        IBTRACS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IBTRACS_URL, IBTRACS)
    urllib.request.urlretrieve(PAPER_DAMAGE_URL, out / 'paper_default_impf.csv')

    cal = Path(climada.__file__).parent / 'data' / 'system' / 'tc_impf_cal_v01_EDR.csv'
    with open(cal, encoding='ISO-8859-1') as f:
        rows = [r for r in csv.DictReader(f) if r['cal_region2'] == REGION]
    ib = xr.open_dataset(IBTRACS)
    name = dict(zip(ib['sid'].values.astype(str), ib['name'].values.astype(str)))
    for r in rows:
        r['name'] = name[RENAMED.get(r['ibtracsID'], r['ibtracsID'])].title()
    with open(out / 'events.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(
        f'{len(rows)} {REGION} events, {sum(r["country"] == "CAN" for r in rows)} of them Canadian'
    )

    client = Client()
    api = client.get_centroids(
        res_arcsec_land=150, res_arcsec_ocean=1800, extent=(-180, 180, -60, 60)
    ).gdf
    tree = cKDTree(np.c_[api.geometry.x.values, api.geometry.y.values])
    lat_all, lon_all, val_all, iso_all = [], [], [], []
    for iso, num in ISO.items():
        exp = client.get_litpop(country=iso, exponents=(1, 1))
        g = exp.gdf
        lat, lon, val = g.geometry.y.values, g.geometry.x.values, g['value'].values
        print(
            f'{iso}: LitPop {exp.ref_year} at 150 arcsec, {val.sum():.3e} -> {TOTAL_2014[iso]:.3e}'
        )
        val = val * TOTAL_2014[iso] / val.sum()
        iy, ix = np.floor(lat * 12).astype(np.int64), np.floor(lon * 12).astype(np.int64)
        u, inv = np.unique(iy * 100000 + (ix + 50000), return_inverse=True)
        lat_all.append((u // 100000 + 0.5) / 12)
        lon_all.append((u % 100000 - 50000 + 0.5) / 12)
        val_all.append(np.bincount(inv, weights=val))
        iso_all.append(np.full(u.size, num))
    lat, lon = np.concatenate(lat_all), np.concatenate(lon_all)
    _, j = tree.query(np.c_[lon, lat])
    # border cells appear once per country; the hazard gets each coordinate once
    key = np.round(lat * 24).astype(np.int64) * 100000 + np.round(lon * 24).astype(np.int64)
    _, first, col = np.unique(key, return_index=True, return_inverse=True)
    np.savez(
        out / 'exposure.npz',
        lat=lat,
        lon=lon,
        value=np.concatenate(val_all),
        iso=np.concatenate(iso_all),
        dist_coast=api['dist_coast'].values[j],
        first=first,
        col=col,
    )
    print(f'{lat.size:,} exposure cells at 300 arcsec, {first.size:,} hazard cells')


def _records(tr, args, estimate_pressure):
    """Edits made at the native records, before interpolation."""
    if args.records == '6h':
        t = tr.time.to_index()
        tr = tr.isel(time=np.nonzero((t.hour % 6 == 0) & (t.minute == 0))[0])
    if args.pressure != 'track':
        lon = tr['lon'].values % 360
        if args.pressure == 'wind180':
            lon = np.where(lon > 180, lon - 360, lon)
        est = estimate_pressure(
            np.full(lon.size, -1.0), tr['lat'].values, lon, tr['max_sustained_wind'].values
        )
        tr['central_pressure'].values[:] = np.where(
            np.isnan(est), tr['central_pressure'].values, est
        )
    if args.penv == '1010':
        tr['environmental_pressure'].values[:] = 1010.0
    return tr


def cmd_wind(args) -> None:
    import logging
    from importlib.metadata import version

    from climada.hazard import Centroids, TCTracks, TropCyclone
    from climada.hazard.tc_tracks import _estimate_pressure

    logging.getLogger('climada').setLevel(logging.WARNING)
    tag = wind_tag(version('climada'), args)
    back = {v: k for k, v in RENAMED.items()}
    ids = [RENAMED.get(r['ibtracsID'], r['ibtracsID']) for r in read_events(args.dir)]
    e = np.load(args.dir / 'exposure.npz')
    lat, lon, dist_coast = (e[k][e['first']] for k in ('lat', 'lon', 'dist_coast'))
    on_land = np.ones(lat.size, bool)
    try:
        cent = Centroids(lat=lat, lon=lon, dist_coast=dist_coast, on_land=on_land)
    except TypeError:  # CLIMADA < 5
        cent = Centroids.from_lat_lon(lat, lon)
        cent.dist_coast, cent.on_land = dist_coast, on_land

    tracks = TCTracks.from_ibtracs_netcdf(
        provider='usa', storm_id=ids, estimate_missing=True, file_name=str(IBTRACS)
    )
    missing = set(ids) - {t.sid for t in tracks.data}
    if missing:
        raise SystemExit(f'not in {IBTRACS.name}: {sorted(missing)}')
    tracks.data = [_records(tr, args, _estimate_pressure) for tr in tracks.data]
    tracks.equal_timestep(time_step_h=args.time_step)
    for tr in tracks.data:
        if args.radius == 'p141':
            pc = np.minimum(tr['central_pressure'].values, tr['environmental_pressure'].values)
            tr['radius_max_wind'] = ('time', np.interp(pc, [950, 980, 1020], [15.0, 25.0, 50.0]))
        elif args.radius == 'est':
            tr['radius_max_wind'] = ('time', np.zeros(tr.sizes['time']))

    haz = TropCyclone.from_tracks(tracks, centroids=cent)
    coo = haz.intensity.tocoo()
    sparse.save_npz(
        args.dir / f'wind_{tag}.npz',
        sparse.csr_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape),
    )
    (args.dir / f'wind_{tag}.txt').write_text('\n'.join(back.get(n, n) for n in haz.event_name))
    print(f'{tag}: {coo.shape[0]} storms, peak {coo.data.max():.1f} m/s')


def cmd_fit(args) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chaz_damage import IMPF_NODES, V_THRESH, _mdd

    # CLIMADA's impact function is linear in its node values, so each event's
    # damage at every v_half is its exposure weights on the 5 m/s nodes times
    # the sigmoid at the nodes
    lo, hi, step = IMPF_NODES
    nodes = np.arange(lo, hi + step / 2, step)
    v_halfs = np.round(V_THRESH + 0.1 * np.arange(1, 3001), 1)  # the paper's sweep, 25.8-325.7
    node_mdd = _mdd(nodes[:, None], v_halfs[None, :])
    at = {v: int(np.argmin(np.abs(v_halfs - v))) for v in (DEFAULT_VHALF, *TABLE_A2)}

    events = read_events(args.dir)
    target = np.array([float(r['emdat_impact_scaled']) for r in events])
    shipped = np.array([float(r['v_half']) for r in events])
    with open(args.dir / 'paper_default_impf.csv') as f:
        paper = {r['unique_ID']: float(r['climada_impact']) for r in csv.DictReader(f)}
    paper = np.array([paper[r['unique_ID']] for r in events])
    e = np.load(args.dir / 'exposure.npz')
    cells = [np.nonzero(e['iso'] == ISO[r['country']])[0] for r in events]

    cols = (
        ' '.join(f'{f"TDR@{v}":>8s}' for v in TABLE_A2)
        + ' '
        + ' '.join(f'{f"RMSF@{v}":>9s}' for v in TABLE_A2)
    )
    print(f'{"":24s} {cols}   TDR fit  RMSF fit   n   per-event median (IQR)')
    print(
        f'{"paper":24s} '
        + ' '.join(f'{t:8.2f}' for t, _ in TABLE_A2.values())
        + ' '
        + ' '.join(f'{r:9.1f}' for _, r in TABLE_A2.values())
        + f'   {PUBLISHED_FIT[0]:7.1f}  {PUBLISHED_FIT[1]:8.1f}  {len(events):2d}   '
        + f'{np.median(shipped):.1f} ({np.percentile(shipped, 25):.1f}-{np.percentile(shipped, 75):.1f})'
    )
    ratio = {}
    for tag in args.tags:
        m = sparse.load_npz(args.dir / f'wind_{tag}.npz').tocsr()
        row = {n: i for i, n in enumerate((args.dir / f'wind_{tag}.txt').read_text().split())}
        weights = np.zeros((len(events), nodes.size))
        for k, r in enumerate(events):
            w = m[row[r['ibtracsID']]].toarray().ravel()[e['col'][cells[k]]]
            x = np.clip(w, lo, hi)
            j = np.clip(np.floor(x / step), 0, nodes.size - 2).astype(int)
            f = (x - nodes[j]) / step
            val = e['value'][cells[k]]
            weights[k] = np.bincount(j, val * (1 - f), nodes.size) + np.bincount(
                j + 1, val * f, nodes.size
            )
        dmg = weights @ node_mdd
        ok = dmg[:, at[DEFAULT_VHALF]] > 0
        tdr = dmg[ok].sum(axis=0) / target[ok].sum()
        with np.errstate(divide='ignore'):
            lr = np.log(dmg / target[:, None])
        rmsf = np.exp(np.sqrt(np.mean(lr[ok] ** 2, axis=0)))
        per_event = v_halfs[np.argmin(np.abs(lr), axis=1)]
        pe = per_event[ok]
        print(
            f'{tag:24s} '
            + ' '.join(f'{tdr[at[v]]:8.2f}' for v in TABLE_A2)
            + ' '
            + ' '.join(f'{rmsf[at[v]]:9.1f}' for v in TABLE_A2)
            + f'   {v_halfs[np.argmin(np.abs(np.log(tdr)))]:7.1f}  {v_halfs[np.argmin(rmsf)]:8.1f}  {ok.sum():2d}   '
            + f'{np.median(pe):.1f} ({np.percentile(pe, 25):.1f}-{np.percentile(pe, 75):.1f})'
        )
        ratio[tag] = dmg[:, at[DEFAULT_VHALF]] / paper
        with open(args.dir / f'fit_{tag}.csv', 'w', newline='') as f:
            out = csv.writer(f)
            out.writerow(
                [
                    'ibtracsID',
                    'name',
                    'country',
                    'emdat_impact_scaled',
                    'paper_at_74.7',
                    'sim_at_74.7',
                    'v_half',
                ]
            )
            for k, r in enumerate(events):
                out.writerow(
                    [
                        r['ibtracsID'],
                        r['name'],
                        r['country'],
                        target[k],
                        paper[k],
                        dmg[k, at[DEFAULT_VHALF]],
                        per_event[k] if ok[k] else '',
                    ]
                )

    print(f'\nsimulated / paper damage at v_half {DEFAULT_VHALF}, largest events in the paper')
    width = max(len(t) for t in args.tags) + 2
    print(f'{"":18s} {"paper $bn":>9s}' + ''.join(f'{t:>{width}s}' for t in args.tags))
    for k in np.argsort(-paper)[: args.events]:
        label = f'{events[k]["name"]} {events[k]["year"]}' + (
            ' CAN' if events[k]['country'] == 'CAN' else ''
        )
        print(
            f'{label:18s} {paper[k] / 1e9:9.1f}'
            + ''.join(f'{ratio[t][k]:{width}.2f}' for t in args.tags)
        )
    with np.errstate(divide='ignore'):
        logs = {t: np.log(v) for t, v in ratio.items()}
    print(
        f'{"median":18s} {"":9s}'
        + ''.join(f'{np.exp(np.median(logs[t])):{width}.2f}' for t in args.tags)
    )
    print(
        f'{"within 25%":18s} {"":9s}'
        + ''.join(
            f'{f"{int((np.abs(logs[t]) < np.log(1.25)).sum())}/{len(events)}":>{width}s}'
            for t in args.tags
        )
    )


def cmd_ead(args) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import xarray as xr
    from chaz_damage import BANDS, ead_from_levels

    pub = xr.open_dataset(CHAZ_EVENTS / 'levels' / 'published.nc')
    new = xr.open_dataset(CHAZ_EVENTS / 'levels' / 'era5_10f.nc')
    value = xr.open_dataset(CHAZ_EVENTS / 'tail' / 'published_decompose.nc')[
        'litpop_value_us'
    ].values
    lat, lon = new['lat'].values, new['lon'].values
    assert np.allclose(lat, pub['lat'].values) and np.allclose(lon, pub['lon'].values)
    vp, vn = (
        np.stack([np.nan_to_num(d[b].values.astype('float64')) for b in BANDS]) for d in (pub, new)
    )
    # EAD is linear in frequency: give the rebuild's events the published per-event frequencies
    freq = {b: YRLY_FREQ_IB[b] / n for b, n in PUBLISHED_STORMS.items()}
    scale = np.where(
        lon > -100, freq['NA'] / float(new.attrs['w_na']), freq['EP'] / float(new.attrs['w_ep'])
    )

    us = value > 0
    served = ead_from_levels(vp, args.served)[us] * value[us]
    regions = {
        'TX-AL': (lon < -87.6) & (lat < 33),
        'FL': (lon >= -87.6) & (lat < 31),
        'GA-Car': (lon >= -87.6) & (lat >= 31) & (lat < 36.6),
        'VA-ME': (lon >= -82) & (lat >= 36.6),
    }
    regions['other'] = ~np.any(list(regions.values()), axis=0)
    top = np.argsort(-served)
    top = top[np.cumsum(served[top]) <= 0.9 * served.sum()]
    print(
        f'served ERA5 store x {args.served}: {served.sum() / 1e9:.1f} $bn/yr over US cells (LitPop 2020)'
    )
    print(
        f'{"":26s} {"national":>8s}'
        + ''.join(f'{r:>8s}' for r in regions)
        + '   cells with 90% of loss: median (5-95%)'
    )
    print(
        f'{"share of served loss":26s} {"":8s}'
        + ''.join(f'{served[m[us]].sum() / served.sum():8.1%}' for m in regions.values())
    )
    for vh in (args.served, *args.v_halfs):
        loss = ead_from_levels(vn, vh)[us] * scale[us] * value[us]
        r = loss[top] / served[top]
        print(
            f'{f"corrected winds x {vh}":26s} {loss.sum() / served.sum():8.3f}'
            + ''.join(f'{loss[m[us]].sum() / served[m[us]].sum():8.2f}' for m in regions.values())
            + f'   {np.median(r):.2f} ({np.percentile(r, 5):.2f}-{np.percentile(r, 95):.2f})'
        )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument(
        '--dir', type=Path, default=WORK, help='working directory (default %(default)s)'
    )
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prep', help='events, paper damages, exposure, IBTrACS')
    w = sub.add_parser('wind', help='winds for the 43 storms under the installed CLIMADA')
    w.add_argument(
        '--radius',
        choices=('track', 'p141', 'est'),
        default='p141',
        help='track: IBTrACS radius where given; p141: CLIMADA 1.4.1 pressure rule at every record; '
        "est: CLIMADA's estimate from pressure, as for CHAZ tracks",
    )
    w.add_argument(
        '--records',
        choices=('all', '6h'),
        default='6h',
        help='all: every IBTrACS record (3-hourly and landfall); 6h: 00/06/12/18 UTC only',
    )
    w.add_argument(
        '--pressure',
        choices=('track', 'wind360', 'wind180'),
        default='track',
        help="track: IBTrACS central pressure; wind360/wind180: CLIMADA's regression on wind, "
        'with longitudes in 0-360 (what its CHAZ reader passes) or -180-180 (what it was fit on)',
    )
    w.add_argument(
        '--penv',
        choices=('track', '1010'),
        default='track',
        help="track: IBTrACS outer-isobar pressure; 1010: CLIMADA's default, as for CHAZ tracks",
    )
    w.add_argument(
        '--time-step', type=float, default=1.0, help='hours between interpolated records'
    )
    f = sub.add_parser('fit', help='fit v_half on saved winds and compare with the paper')
    f.add_argument('tags', nargs='+', help='wind file tags, e.g. 4.0.1_p141_6h')
    f.add_argument('--events', type=int, default=10, help='largest paper events to list')
    e = sub.add_parser(
        'ead', help='CONUS EAD of the corrected-wind rebuild against the served store'
    )
    e.add_argument('v_halfs', nargs='+', type=float)
    e.add_argument('--served', type=float, default=89.2, help='v_half of the served store')
    args = ap.parse_args(argv)
    {'prep': cmd_prep, 'wind': cmd_wind, 'fit': cmd_fit, 'ead': cmd_ead}[args.cmd](args)


if __name__ == '__main__':
    main()
