"""CHAZ forcing from one CMIP6 model member, from the public CMIP6 Zarr store on AWS (cmip6-pds).

The same files forcing.py writes for ERA5, on the same 2-degree grid, under
forcing/cmip6/{model}/{member}/{experiment}/{year}/. Following Fosu et al. (2024, section 2.2):

- the genesis index reads bias-corrected fields: for 850 hPa absolute vorticity (its magnitude),
  250-850 hPa shear, PI, CRH and SD, the monthly ERA5-minus-model climatology over 1981-2010 is
  added to every model month, historical and future, before TCGI
- the track and intensity models read the model's own fields: monthly 250/850 hPa winds,
  500/700/850 hPa relative humidity and PI (I_MLR), and the daily wind covariance (A)
- the intensity predictors are standardized with the model's normalization (meanstd.py)

Each field is put on the 2-degree grid as for ERA5: vorticity from the model's monthly winds on
their own grid, then bilinear; everything else bilinear first (periodic in longitude, each
variable from its own coordinates), then computed. SST is the model's surface temperature where
its land fraction is under 50%, so land never enters the bilinear PI inputs. The daily wind
covariance is computed per model month on the model's wind grid (all of a 360-day calendar's 30
days), then regridded, interpolated to each real calendar day and factored, as CHAZvBeta does.

    python -m ocr.risks.hurricane.cmip <command>

    raw       one year's model fields: genesis inputs before correction, I_MLR, PI, Cov, A
    correct   the bias correction from the 1981-2010 climatologies, then TCGI for every year
    launch    one Coiled task per year for `raw`, then `correct`
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

from ocr.risks.hurricane import batch, forcing, store, tcgi

MODEL = 'UKESM1-0-LL'
MEMBER = 'r1i1p1f2'
INSTITUTION = {'UKESM1-0-LL': 'MOHC', 'MIROC6': 'MIROC'}
LAND_FRACTION = ('piControl', 'r1i1p1f2')  # UKESM's historical members carry no sftlf
HISTORICAL = (1981, 2014)
FUTURE = {'ssp585': (2081, 2100)}
CLIMATOLOGY = (1981, 2010)
GENESIS = ['avor', 'vws', 'PI', 'CRH', 'SD']
LAT, LON = forcing.LAT, forcing.LON


def experiment(year: int, future: str = 'ssp585') -> str:
    return 'historical' if year <= 2014 else future


def source(exp: str, model: str = MODEL, member: str = MEMBER) -> str:
    return f'cmip6/{model}/{member}/{exp}'


def root(exp: str, model: str = MODEL, member: str = MEMBER) -> str:
    return f'forcing/{source(exp, model, member)}'


def meanstd_key(model: str = MODEL, member: str = MEMBER) -> str:
    return f'forcing/cmip6/{model}/{member}/coefficient_meanstd.nc'


# --- reading -------------------------------------------------------------------------------------


def _open(var: str, table: str, exp: str, model: str = MODEL, member: str = MEMBER) -> xr.DataArray:
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    mip = 'CMIP' if exp in ('historical', 'piControl') else 'ScenarioMIP'
    base = f'cmip6-pds/CMIP6/{mip}/{INSTITUTION[model]}/{model}/{exp}/{member}/{table}/{var}'
    grid = sorted(fs.ls(base))[0]
    version = sorted(fs.ls(grid))[-1]
    ds = xr.open_zarr(fs.get_mapper(version), consolidated=True, use_cftime=True)
    da = ds[var]
    if 'plev' in da.dims:
        da = da.assign_coords(plev=(da['plev'] / 100).round()).rename(plev='level')
    return da


def _year(da: xr.DataArray, year: int) -> xr.DataArray:
    return da.sel(time=str(year)).load()


def regrid(da: xr.DataArray) -> xr.DataArray:
    """Bilinear onto the 2-degree centres, periodic in longitude; NaN wherever a corner is NaN."""
    da = da.drop_vars([c for c in da.coords if c not in da.dims and c != 'time'], errors='ignore')
    da = da.sortby('lat')
    lon = da['lon']
    da = xr.concat(
        [
            da.isel(lon=[-1]).assign_coords(lon=lon[-1:] - 360),
            da,
            da.isel(lon=[0]).assign_coords(lon=lon[:1] + 360),
        ],
        'lon',
    )
    return da.interp(lat=LAT, lon=LON)


def hold_lowest(da: xr.DataArray) -> xr.DataArray:
    """Levels below ground, missing in CMIP6, take the value of the lowest level above ground."""
    da = da.sortby('level')
    levels = [da.isel(level=0, drop=True)]
    for i in range(1, da.sizes['level']):
        levels.append(da.isel(level=i, drop=True).fillna(levels[-1]))
    return xr.concat(levels, 'level').assign_coords(level=da['level'].values)


def era5_ocean() -> xr.DataArray:
    """The 2-degree cells our ERA5 TCGI covers (the same in every month and year)."""
    with xr.open_dataset(store.path(f'{store.FORCING}/1981/TCGI_CRH_PI_1981.nc')) as ds:
        ocean = ds['TCGI'].notnull().all('time').load()
    return ocean.assign_coords(lat=LAT, lon=LON)


def vorticity(native: xr.Dataset) -> xr.DataArray:
    """Absolute vorticity of the monthly 850 hPa winds on the model's grid, without the pole rows
    (where grid spacing in longitude vanishes), bilinear to 2 degrees; the 2-degree rows beyond the
    model's last row take the nearest row."""
    w = native.sel(lat=slice(-89.99, 89.99))
    av = regrid(tcgi.absolute_vorticity(w['u850'], w['v850']))
    return av.bfill('lat', limit=1).ffill('lat', limit=1)


def extend_ocean(sst: xr.DataArray, steps: int = 2) -> xr.DataArray:
    """SST carried `steps` cells inland on the model grid, each step the mean of a missing cell's
    ocean neighbours, so 2-degree cells on a coast the model's coarser land mask cuts still get an
    SST, and only from ocean values."""
    for _ in range(steps):
        nb = [sst.roll(lon=k, roll_coords=False) for k in (-1, 1)]
        nb += [sst.shift(lat=k) for k in (-1, 1)]
        stack = xr.concat(nb, 'nb')
        sst = sst.fillna(stack.mean('nb', skipna=True))
    return sst


def ocean_fraction(model: str = MODEL) -> xr.DataArray:
    exp, member = LAND_FRACTION
    return _open('sftlf', 'fx', exp, model, member).load()


# --- raw fields for one year ----------------------------------------------------------------------


def daily_winds(year: int, exp: str, model: str = MODEL, member: str = MEMBER) -> xr.Dataset:
    """Daily 250/850 hPa winds on the model's wind grid, every day of the model's year. Where
    850 hPa is below ground (high terrain) it holds the lowest level above, from 700 or 500 hPa."""
    out, filled = {}, 0
    for c in ('u', 'v'):
        da = _year(_open(f'{c}a', 'day', exp, model, member).sel(level=[250, 500, 700, 850]), year)
        filled += int(da.sel(level=850).isnull().sum())
        da = hold_lowest(da)
        for lev in (250, 850):
            out[f'{c}{lev}'] = da.sel(level=lev, drop=True)
    ds = xr.Dataset(out).sortby('lat')
    ds.attrs['below_ground_850_filled'] = filled
    if ds.sizes['time'] not in (360, 365, 366) or any(bool(ds[v].isnull().any()) for v in ds):
        raise ValueError(f'{year}: daily winds incomplete ({ds.sizes["time"]} days)')
    return ds


def monthly_means(daily: xr.Dataset) -> xr.Dataset:
    m = daily.groupby('time.month').mean('time')
    return m.rename(month='time')


def raw(
    year: int,
    model: str = MODEL,
    member: str = MEMBER,
    future: str = 'ssp585',
    kelvin: float = tcgi.KELVIN_CHAZ,
) -> list[str]:
    exp = experiment(year, future)
    key = f'{root(exp, model, member)}/{year}'
    out = store.CACHE / key
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def save(ds: xr.Dataset, name: str, **kw) -> None:
        part = out / f'{name}.part'
        ds.to_netcdf(part, **kw)
        part.rename(out / name)
        written.append(f'{key}/{name}')

    daily = daily_winds(year, exp, model, member)
    native = monthly_means(daily)
    months = pd.date_range(f'{year}-01-01', periods=12, freq='MS')
    winds = regrid(native).assign_coords(time=months)

    ocean = era5_ocean()
    g = xr.Dataset(coords={'time': months, 'lat': LAT, 'lon': LON})
    g['avor'] = vorticity(native).assign_coords(time=months)
    g['vws'] = tcgi.wind_shear(winds['u250'], winds['v250'], winds['u850'], winds['v850'])

    def monthly(var: str, levels=None) -> xr.DataArray:
        da = _open(var, 'Amon', exp, model, member)
        if levels is not None:
            da = da.sel(level=levels)
        da = _year(da, year)
        if da.sizes['time'] != 12:
            raise ValueError(f'{year}: {var} has {da.sizes["time"]} months')
        return da

    ta = monthly('ta')
    hus = monthly('hus')
    filled = int(ta.isnull().sum())
    ta, hus = hold_lowest(ta), hold_lowest(hus)
    t2, q2 = regrid(ta).assign_coords(time=months), regrid(hus).assign_coords(time=months)
    psl = regrid(monthly('psl')).assign_coords(time=months) / 100
    prw = regrid(monthly('prw')).assign_coords(time=months)
    ts = monthly('ts')
    sea = ocean_fraction(model).reindex_like(ts, method='nearest', tolerance=1e-3) < 50
    sst_k = regrid(extend_ocean(ts.where(sea))).assign_coords(time=months)
    sst = sst_k - 273.15
    hur = regrid(monthly('hur', [500, 700, 850]).mean('level')).assign_coords(time=months)

    # W*'s bottom layer reads the SST; where the model's sea does not reach a cell, its surface
    # temperature, as the model's ts is what the CHAZ team's TCGI code reads there
    surface_k = sst_k.fillna(regrid(ts).assign_coords(time=months))
    g['CRH'], g['SD'] = tcgi.moisture(t2, prw, psl, surface_k, kelvin)
    pi, flag = tcgi.potential_intensity(sst, psl, t2 - 273.15, q2 * 1000, flags=True)
    # no PI (cold water, or a cell the model's ocean does not reach) is 0, as in ERA5's PI files
    g['PI'] = pi.where(sst.notnull()).fillna(0.0).where(ocean | sst.notnull())
    g['pi_flag'] = flag.where(sst.notnull())
    g = g.drop_vars('level', errors='ignore').transpose('time', 'lat', 'lon')
    g.attrs.update(
        {
            'model': model,
            'member': member,
            'experiment': exp,
            'below_ground_ta_filled': filled,
            'below_ground_850_wind_filled': daily.attrs['below_ground_850_filled'],
            'calendar': str(daily['time'].dt.calendar),
        }
    )
    save(g, f'genesis_raw_{year}.nc')

    # the intensity model reads PI on ERA5's ocean cells only, 0 elsewhere, as for ERA5
    pi0 = g['PI'].where(ocean, 0.0)
    save(pi0.to_dataset(name='vmax'), f'PI_{year}.nc')
    mlr = winds.copy()
    mlr['hur'] = hur
    mlr['PI'] = pi0
    save(
        mlr.drop_vars('level', errors='ignore').transpose('time', 'lat', 'lon'), f'I_MLR_{year}.nc'
    )

    cov = regrid(forcing.covariance(year, daily, native)).transpose('month', 'lat', 'lon')
    save(cov, f'Cov_{year}_2deg.nc')
    for m, a in forcing.a_matrices(year, cov).items():
        ds = xr.Dataset(
            {'A': (('Adim', 'days', 'latitude', 'longitude'), a)},
            coords={
                'latitude': LAT.astype('float32'),
                'longitude': LON.astype('float32'),
                'days': np.arange(a.shape[1], dtype='int32'),
                'Adim': np.arange(10, dtype='int32'),
            },
        )
        save(ds, f'A_{year}{m:02d}.nc', format='NETCDF3_CLASSIC')
    return written


def raw_task(year: int, model: str = MODEL, member: str = MEMBER) -> None:
    raw(year, model, member)
    exp = experiment(year)
    files = store.push(
        f'{root(exp, model, member)}/{year}',
        fragment=f'cmip6-{model}-{member}-{year}',
        replace=True,
    )
    print(f'{year}: {len(files)} files on S3', flush=True)


# --- bias correction -------------------------------------------------------------------------------


def era5_climatology(years: tuple[int, int] = CLIMATOLOGY) -> xr.Dataset:
    parts = []
    for y in range(years[0], years[1] + 1):
        g = xr.open_dataset(store.path(f'{store.FORCING}/{y}/TCGI_CRH_PI_{y}.nc'))
        p = xr.open_dataset(store.path(f'{store.FORCING}/{y}/PI_{y}.nc'))
        ds = g[['avor', 'vws', 'CRH', 'SD']].assign(PI=p['vmax'], ocean=g['TCGI'].notnull())
        parts.append(ds.load())
    e = xr.concat(parts, 'time').assign_coords(lat=LAT, lon=LON)
    ocean = e['ocean'].all('time')
    varies = int((e['ocean'].any('time') & ~ocean).sum())
    print(f'ERA5 ocean mask: {int(ocean.sum())} cells always ocean, {varies} only some months')
    e['avor'] = np.abs(e['avor'])
    clim = e[GENESIS].groupby('time.month').mean('time')
    return clim.assign(ocean=ocean)


def model_climatology(model: str, member: str, years: tuple[int, int] = CLIMATOLOGY) -> xr.Dataset:
    parts = [
        xr.open_dataset(store.path(f'{root("historical", model, member)}/{y}/genesis_raw_{y}.nc'))[
            GENESIS
        ].load()
        for y in range(years[0], years[1] + 1)
    ]
    m = xr.concat(parts, 'time')
    m['avor'] = np.abs(m['avor'])
    return m.groupby('time.month').mean('time')


def correct(model: str = MODEL, member: str = MEMBER, future: str = 'ssp585') -> None:
    """Bias-correct every year's genesis inputs and write TCGI_{CRH,SD}_PI_{year}.nc; the
    correction and a per-year count of values outside their physical range go beside them."""
    era5 = era5_climatology()
    ocean = era5['ocean']
    delta = era5[GENESIS] - model_climatology(model, member)
    base = f'forcing/cmip6/{model}/{member}'
    local = store.CACHE / base
    local.mkdir(parents=True, exist_ok=True)
    delta.assign(ocean=ocean).to_netcdf(local / 'genesis_correction.nc')
    keys = [f'{base}/genesis_correction.nc']
    rows = []
    for exp, (y0, y1) in (('historical', HISTORICAL), (future, FUTURE[future])):
        for year in range(y0, y1 + 1):
            g = xr.open_dataset(
                store.path(f'{root(exp, model, member)}/{year}/genesis_raw_{year}.nc')
            )
            g = g.load()
            month = g['time.month']
            x = g[GENESIS].copy()
            x['avor'] = np.abs(x['avor'])
            c = x + delta.sel(month=month).drop_vars('month')
            o = ocean.broadcast_like(c['PI'])
            row = {'experiment': exp, 'year': year}
            for v, bad in (
                ('avor', c['avor'] < 0),
                ('vws', c['vws'] < 0),
                ('PI', c['PI'] < 0),
                ('CRH', (c['CRH'] < 0) | (c['CRH'] > 100)),
                ('SD', c['SD'] > 0),
            ):
                row[f'{v}_out_of_range'] = int((bad & o).sum())
            missing = {v: int((x[v].isnull() & o).sum()) for v in GENESIS}
            if any(missing.values()):
                raise ValueError(
                    f'{exp} {year}: model fields missing on ERA5 ocean cells: {missing}'
                )
            row['PI_flag_not_1_on_ocean'] = int(((g['pi_flag'] != 1) & g['PI'].notnull() & o).sum())
            rows.append(row)
            c['avor'] = c['avor'].clip(min=0)
            c['vws'] = c['vws'].clip(min=0)
            c['PI'] = c['PI'].clip(min=0)
            c['CRH'] = c['CRH'].clip(min=0)
            out = store.CACHE / root(exp, model, member) / str(year)
            for kind in ('CRH', 'SD'):
                t = tcgi.tcgi(c['avor'], c[kind], c['PI'], c['vws'], kind).where(ocean)
                t_raw = tcgi.tcgi(x['avor'], x[kind], g['PI'], x['vws'], kind).where(ocean)
                ds = xr.Dataset(
                    {
                        'avor': c['avor'],
                        'vws': c['vws'],
                        'CRH': c['CRH'],
                        'SD': c['SD'],
                        'PI': c['PI'],
                        'TCGI': t,
                        'TCGI_uncorrected': t_raw,
                    }
                )
                name = f'TCGI_{kind}_PI_{year}.nc'
                ds.to_netcdf(out / name)
                keys.append(f'{root(exp, model, member)}/{year}/{name}')
            print(exp, year, row, flush=True)
    pd.DataFrame(rows).to_csv(local / 'genesis_correction_ranges.csv', index=False)
    keys.append(f'{base}/genesis_correction_ranges.csv')
    store.push(keys=keys, replace=True)


# --- Coiled ---------------------------------------------------------------------------------------


def years(future: str = 'ssp585') -> list[int]:
    return [
        *range(HISTORICAL[0], HISTORICAL[1] + 1),
        *range(FUTURE[future][0], FUTURE[future][1] + 1),
    ]


def launch_raw(vm_type: str, max_workers: int, spot: bool = True) -> None:
    batch.run(
        'cmip',
        ['raw-task'],
        name=f'chaz-cmip6-{MODEL}-{MEMBER}-raw',
        over=years(),
        vm_type=vm_type,
        max_workers=max_workers,
        spot=spot,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('raw')
    r.add_argument('--year', type=int, required=True)
    sub.add_parser('raw-task')
    sub.add_parser('correct')
    la = sub.add_parser('launch-raw')
    la.add_argument('--vm-type', default='m8g.xlarge')
    la.add_argument('--max-workers', type=int, default=20)
    la.add_argument('--on-demand', action='store_true')
    lc = sub.add_parser('launch-correct')
    lc.add_argument('--vm-type', default='m8g.xlarge')
    a = p.parse_args()
    if a.cmd == 'raw':
        print(raw(a.year))
    elif a.cmd == 'raw-task':
        raw_task(int(os.environ['COILED_BATCH_TASK_INPUT']))
    elif a.cmd == 'correct':
        correct()
    elif a.cmd == 'launch-raw':
        launch_raw(a.vm_type, a.max_workers, not a.on_demand)
    else:
        batch.run(
            'cmip',
            ['correct'],
            name=f'chaz-cmip6-{MODEL}-{MEMBER}-correct',
            vm_type=a.vm_type,
            wait=True,
        )


if __name__ == '__main__':
    main()
