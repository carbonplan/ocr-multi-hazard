"""Intensity members that run past 300 kt, which CHAZ drops: where they start, and how often on
the same tracks under different predictor normalizations.

The water regression's cubic term in (potential intensity - wind) makes a storm whose wind is far
above the potential intensity around it strengthen, and the strengthening widens the gap. Each
task draws one realization's tracks once and runs the 40 intensity members on them under each
normalization of NORMS, with the same random draws to start, so differences between them come
from the normalization. For every member CHAZ would drop it records the onset (the step after the
last one that did not strengthen before the wind first passes 300 kt) and, at the largest 12-hour
jump, every predictor's standardized value and its contribution to the regression.

    python -m ocr.risks.hurricane.runaways run        on Coiled (launch), or on a cached machine
    python -m ocr.risks.hurricane.runaways launch
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd

from ocr.risks.hurricane import batch, chaz, store

U = 'cmip6/UKESM1-0-LL/r1i1p1f2'
UK_MAPPED = 'forcing/cmip6/UKESM1-0-LL/r1i1p1f2/coefficient_meanstd.nc'
UK_PORT = 'validation/meanstd/UKESM1-0-LL_r1i1p1f2_port_model.nc'
ERA5_PORT = 'validation/meanstd/era5.nc'
NORMS = {
    'era5': {'tutorial': None, 'ERA5 port': ERA5_PORT},
    f'{U}/historical': {'mapped': UK_MAPPED, 'tutorial': None, 'model port': UK_PORT},
    f'{U}/ssp585': {'mapped': UK_MAPPED, 'tutorial': None},
}
LABEL = {'era5': 'ERA5', f'{U}/historical': 'UKESM 1995–2014', f'{U}/ssp585': 'UKESM 2081–2100'}
YEARS = {'era5': [2005, 2010], f'{U}/historical': [2005, 2010], f'{U}/ssp585': [2090, 2095]}
TCGI = 'TCGI_CRH_PI'
MEMBERS = 40
LT = 2  # 6-hourly steps per 12-hour prediction
OUT = 'validation/chaz/cmip/runaways'


def predictor_values(bt, it, iS, v0, dvdt, preds):
    """The predictors chaz.getPrediction_v0input_result reads for one 12-hour step."""
    sl = slice(it, it + LT + 1)
    pi = np.nanmean(bt.PIwspdMean[sl, iS])
    vals = {}
    for var in preds:
        if var == 'SHRD':
            vals[var] = np.nanmean(np.sqrt(bt.UShearMean[sl, iS] ** 2 + bt.VShearMean[sl, iS] ** 2))
        elif var == 'dPIwspd':
            vals[var] = pi - v0
        elif var == 'rhMean':
            vals[var] = np.nanmean(bt.rhMean[sl, iS])
        elif var == 'StormMwspd':
            vals[var] = v0
        elif var == 'dVdt':
            vals[var] = dvdt
        elif var == 'landmaskMean':
            vals[var] = chaz.gv.landDelta * (
                bt.landmaskMean[it + LT + 1, iS] - bt.landmaskMean[it, iS]
            )
        elif var == 'dPIwspd2':
            vals[var] = (pi - v0) ** 2
        elif var == 'dPIwspd3':
            vals[var] = (pi - v0) ** 3
        elif var == 'dVdt2':
            vals[var] = dvdt**2
        else:
            vals[var] = getattr(bt, var)[it, iS]
    return vals, pi


def analyse(bt, source, year, r, norm):
    """Counts as chaz.to_dataset drops members, and one row per dropped member."""
    S = bt.stochastic.copy()
    S[S == 0] = np.nan
    with np.errstate(all='ignore'):
        max5 = np.nanmax(S[0:21], axis=0)
        peak = np.nanmax(S, axis=0)
    undeveloped = max5 < 35
    over = (peak > 300) & ~undeveloped
    lat0, lon0 = bt.StormLat[0], bt.StormLon[0]
    xbin, ybin, bmap = chaz.getbasinMap()
    basin = chaz.defineBasin(lon0, lat0, xbin, ybin, bmap)
    valid = (np.abs(lat0) >= 2) & (basin > 0) & np.isfinite(lon0)
    kept = int((~undeveloped[valid] & ~over[valid] & (peak[valid] > 0)).sum())
    rows = []
    for iS, m in zip(*np.nonzero(over)):
        w = S[:, iS, m]
        t300 = int(np.argmax(w > 300))
        even = np.arange(0, t300, LT)
        jumps = np.array([w[i + LT] - w[i] for i in even])
        flat = even[jumps <= 0]
        onset = int(flat.max() + LT) if flat.size else 0
        pi_on = np.nanmean(bt.PIwspdMean[onset : onset + LT + 1, iS])
        it = int(even[np.nanargmax(jumps)])
        v0 = w[it]
        dvdt = 0.0 if it == 0 else w[it] - w[it - LT]
        preds, result, meanX, meanY, stdX, stdY = chaz._regression(it, LT, iS)
        vals, pi = predictor_values(bt, it, iS, v0, dvdt, preds)
        row = {
            'source': LABEL[source],
            'year': year,
            'r': r,
            'norm': norm,
            'storm': iS,
            'member': m,
            'basin': chaz.BASIN_NAMES[int(basin[iS])],
            'valid': bool(valid[iS]),
            'lat0': lat0[iS],
            'lon0': lon0[iS],
            'month': bt.StormInitMonth[iS],
            't_onset': onset,
            'v_onset': w[onset],
            'PI_onset': pi_on,
            'gap_onset': w[onset] - pi_on,
            'lat_onset': bt.StormLat[onset, iS],
            'lon_onset': bt.StormLon[onset, iS],
            't300': t300,
            'it_jump': it,
            'lat_jump': bt.StormLat[it, iS],
            'v0': v0,
            'jump': w[it + LT] - v0,
            'error': bt.error[it, iS, m],
            'regression': 'water' if preds is chaz.PREDICTORS_W else 'land',
            'PI_ring': pi,
        }
        dy = result[0]
        for k, var in enumerate(preds, start=1):
            z = (vals[var] - meanX[k]) / stdX[k]
            row[f'x_{var}'] = vals[var]
            row[f'z_{var}'] = z
            row[f'c_{var}'] = z * result[k]
            dy += z * result[k]
        row['dV_model'] = dy * stdY + meanY
        rows.append(row)
    count = {
        'source': LABEL[source],
        'year': year,
        'r': r,
        'norm': norm,
        'dropped': int((over & valid[:, None]).sum()),
        'kept': kept,
        'storms': int(valid.sum()),
    }
    return count, rows


def environment(bt, source, year, r):
    """Ring potential intensity, shear and humidity at every track point."""
    ok = np.isfinite(bt.PIwspdMean) & np.isfinite(bt.StormLon)
    shear = np.sqrt(bt.UShearMean**2 + bt.VShearMean**2)
    return pd.DataFrame(
        {
            'source': LABEL[source],
            'year': year,
            'r': r,
            'PI_ring': bt.PIwspdMean[ok],
            'shear': shear[ok],
            'rh': bt.rhMean[ok],
            'water': (bt.landmaskMean <= chaz.gv.landThreshold)[ok],
            'lat': bt.StormLat[ok],
        }
    )


def work(task):
    source, year, r, ratio = task
    chaz.gv.source, chaz.gv.meanstd = source, None
    chaz.yearTC = year
    chaz.gv.TCGIinput = TCGI
    for key, choices in chaz.OPTIONS.items():
        setattr(chaz.gv, key, choices[0])
    seed = year * 1000 + r
    random.seed(seed)
    np.random.seed(seed)
    chaz._load_landmask()
    fst = chaz.getBam(*chaz.getSeeding(year, ratio))
    chaz.calPredictors(fst)
    env = environment(chaz.bt, source, year, r)
    counts, rows = [], []
    for norm, key in NORMS[source].items():
        chaz.gv.meanstd = key
        random.seed(seed + 1)
        np.random.seed(seed + 1)
        chaz.calIntensity(MEMBERS)
        c, rr = analyse(chaz.bt, source, year, r, norm)
        counts.append(c)
        rows.extend(rr)
    print(
        f'{LABEL[source]} {year} r{r}: ' + ', '.join(f'{c["norm"]} {c["dropped"]}' for c in counts),
        flush=True,
    )
    return counts, rows, env


def prefetch():
    for source, years in YEARS.items():
        for name in chaz.STATIC:
            chaz.static(name)
        for key in NORMS[source].values():
            if key:
                store.path(key)
        for y in years:
            for name in [
                f'{TCGI}_{y}.nc',
                f'I_MLR_{y}.nc',
                *(f'A_{y}{m:02d}.nc' for m in range(1, 13)),
            ]:
                chaz.forcing(name, y, source)


def run(realizations: int) -> None:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    prefetch()
    ratio = {s: chaz.seeding_ratio(TCGI, s) for s in YEARS}  # each run's own, as chaz sets it
    print('seeding ratios', ratio, flush=True)
    tasks = [
        (s, y, r, ratio[s]) for s, ys in YEARS.items() for y in ys for r in range(realizations)
    ]
    counts, rows, envs = [], [], []
    with ProcessPoolExecutor(
        os.cpu_count(), mp_context=mp.get_context('spawn'), max_tasks_per_child=1
    ) as pool:
        for c, rr, e in pool.map(work, tasks):
            counts.extend(c)
            rows.extend(rr)
            envs.append(e)
    local = store.CACHE / OUT
    local.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame(counts)
    counts.to_csv(local / 'counts.csv', index=False)
    pd.DataFrame(rows).to_parquet(local / 'dropped.parquet')
    pd.concat(envs, ignore_index=True).to_parquet(local / 'environment.parquet')
    store.push(prefix=OUT, fragment='cmip-runaways', replace=True)
    g = counts.groupby(['source', 'norm'])[['dropped', 'kept']].sum()
    g['dropped, %'] = 100 * g['dropped'] / (g['dropped'] + g['kept'])
    print(g.round(3).to_string(), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--realizations', type=int, default=4)
    la = sub.add_parser('launch')
    la.add_argument('--realizations', type=int, default=4)
    la.add_argument('--vm-type', default='m8g.4xlarge')
    a = p.parse_args()
    if a.cmd == 'run':
        run(a.realizations)
    else:
        batch.run(
            'runaways',
            ['run', '--realizations', str(a.realizations)],
            name='chaz-cmip-runaways',
            vm_type=a.vm_type,
            wait=True,
        )


if __name__ == '__main__':
    main()
