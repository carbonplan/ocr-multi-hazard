"""Served store for CarbonPlan's refit of the USGS Flood Damage Probability model.

Mosaics the per-watershed outputs of `input-data/tensor/fdp/fdp_train.py`
(`fdp.tif`, `contributions.tif`, `contributions.json` and, when every
watershed has one, `se.tif`) onto the USGS FDP grid, cropped to the watersheds
included, and writes one multiscale icechunk store at
`OutputPaths(Risk.FLOOD, version).store`:

    {0..7}/
      fdp              (y, x)          float32  probability
      fdp_se           (y, x)          float32  standard error of fdp
      contribution     (driver, y, x)  float16  change in probability; per cell,
                                                the watershed's baseline + the
                                                sum over drivers = fdp
      dominant_driver  (y, x)          uint8    index of the largest
                                                contribution, 255 for no data

No data is NaN in every float variable. Levels coarsen by mean in float32, so
the sum identity holds at every level up to float16 rounding; contribution is
cast to float16 as each level is copied into the store, since topozarr's kernel
has no float16. dominant_driver is recomputed from each level's float32
contributions. Root attributes carry the driver names, the per-watershed
baselines and the run the store was built from.

Run: pixi run -e flood flood-retrained build --run noaa-200612-202512 --hucs 03 --version vX.Y.Z
"""

import itertools
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import icechunk
import numpy as np
import obstore
import rasterio
import typer
import xarray as xr
import xproj  # noqa: F401 - registers the .proj accessor
import zarr
from topozarr import create_pyramid

from ocr.config import OutputPaths, Risk, StorageConfig
from ocr.risks.flood.etl.flood import STORE_PREFIX as USGS_STORE

RUNS = 'ocr-explore/FDP/runs'
HUCS = [f'{i:02d}' for i in range(1, 19)]
CRS = 'EPSG:5070'
LEVELS = 8  # 100 m (level 0) to 12.8 km (level 7)

FDP = 'fdp'
SE = 'fdp_se'
CONTRIBUTION = 'contribution'
DOMINANT = 'dominant_driver'
NO_DRIVER = 255
# float16 keeps 11 significant bits
F16_EPS = 2.0**-11

app = typer.Typer(add_completion=False, no_args_is_help=True)


def run_prefix(run: str, distance: str, huc: str) -> str:
    return f'{RUNS}/{run}/{distance}/huc{huc}'


def read_grid(cfg: StorageConfig) -> rasterio.Affine:
    """Transform of the USGS FDP grid, from the cell centres in the input store."""
    repo = icechunk.Repository.open(cfg.icechunk_storage(USGS_STORE))
    ds = xr.open_zarr(repo.readonly_session('main').store, consolidated=False)
    x, y = ds['x'].values, ds['y'].values
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    return rasterio.Affine(dx, 0.0, float(x[0]) - dx / 2, 0.0, dy, float(y[0]) - dy / 2)


def cell_offset(t: rasterio.Affine, grid: rasterio.Affine) -> tuple[int, int]:
    """Row and column of `t`'s origin on `grid`, which must share its lattice."""
    if (t.a, t.e) != (grid.a, grid.e):
        raise ValueError(f'resolution {(t.a, t.e)} differs from the grid {(grid.a, grid.e)}')
    row, col = (t.f - grid.f) / grid.e, (t.c - grid.c) / grid.a
    if abs(row - round(row)) > 1e-6 or abs(col - round(col)) > 1e-6:
        raise ValueError(f'origin {(t.c, t.f)} is off the grid lattice')
    return round(row), round(col)


@dataclass
class Watershed:
    fdp: np.ndarray
    contribution: np.ndarray
    se: np.ndarray | None
    offset: tuple[int, int]
    meta: dict


def _exists(cfg: StorageConfig, key: str) -> bool:
    try:
        obstore.head(cfg.object_store(), key)
    except FileNotFoundError:
        return False
    return True


def _read_band(uri: str) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(uri) as src:
        return src.read(1, masked=True).filled(np.nan).astype('float32'), src.transform


def read_watershed(cfg: StorageConfig, prefix: str, grid: rasterio.Affine) -> Watershed:
    store = cfg.object_store()
    meta = json.loads(bytes(obstore.get(store, f'{prefix}/contributions.json').bytes()))
    fdp, transform = _read_band(cfg.uri(f'{prefix}/fdp.tif'))
    with rasterio.open(cfg.uri(f'{prefix}/contributions.tif')) as src:
        raw = src.read()
        if src.nodata != meta['nodata'] or raw.shape[0] != len(meta['drivers']):
            raise ValueError(f'{prefix}/contributions.tif: unexpected nodata or band count')
    contribution = raw * np.float32(meta['scale'])
    contribution[raw == meta['nodata']] = np.nan
    del raw
    se = None
    if _exists(cfg, f'{prefix}/se.tif'):
        se, _ = _read_band(cfg.uri(f'{prefix}/se.tif'))
    return Watershed(fdp, contribution, se, cell_offset(transform, grid), meta)


def mosaic(
    sheds: dict[str, Watershed], grid: rasterio.Affine, run: str, distance: str
) -> xr.Dataset:
    drivers = next(iter(sheds.values())).meta['drivers']
    if any(s.meta['drivers'] != drivers for s in sheds.values()):
        raise ValueError('watersheds disagree on driver order')
    with_se = [huc for huc, s in sheds.items() if s.se is not None]
    if with_se and len(with_se) != len(sheds):
        raise ValueError(f'only HUCs {with_se} have se.tif; run `fdp_train.py se` for the rest')

    r0 = min(s.offset[0] for s in sheds.values())
    c0 = min(s.offset[1] for s in sheds.values())
    r1 = max(s.offset[0] + s.fdp.shape[0] for s in sheds.values())
    c1 = max(s.offset[1] + s.fdp.shape[1] for s in sheds.values())
    fdp = np.full((r1 - r0, c1 - c0), np.nan, dtype='float32')
    contribution = np.full((len(drivers), *fdp.shape), np.nan, dtype='float32')
    se = np.full_like(fdp, np.nan) if with_se else None
    for s in sheds.values():
        r, c = s.offset[0] - r0, s.offset[1] - c0
        h, w = s.fdp.shape
        valid = np.isfinite(s.fdp)
        fdp[r : r + h, c : c + w][valid] = s.fdp[valid]
        contribution[:, r : r + h, c : c + w][:, valid] = s.contribution[:, valid]
        if se is not None:
            se[r : r + h, c : c + w][valid] = s.se[valid]

    data = {
        FDP: (('y', 'x'), fdp),
        CONTRIBUTION: (('driver', 'y', 'x'), contribution, {'drivers': drivers}),
    }
    if se is not None:
        data[SE] = (
            ('y', 'x'),
            se,
            {'long_name': 'standard error of fdp', 'method': 'infinitesimal jackknife'},
        )
    x = grid.c + grid.a * (np.arange(c0, c1) + 0.5)
    y = grid.f + grid.e * (np.arange(r0, r1) + 0.5)
    ds = xr.Dataset(
        data,
        coords={'x': x, 'y': y, 'driver': np.arange(len(drivers), dtype='int8')},
        attrs={
            'title': 'CarbonPlan refit of the USGS Flood Damage Probability model',
            'drivers': drivers,
            'baselines': {huc: s.meta['baseline'] for huc, s in sheds.items()},
            'attribution': next(iter(sheds.values())).meta['method'],
            # contributions.tif stores each contribution in steps of this size
            'source_scale': next(iter(sheds.values())).meta['scale'],
            'run': run,
            'distance': distance,
            'hucs': sorted(sheds),
        },
    )
    return ds.proj.assign_crs(spatial_ref=CRS)


def _blocks(shape: tuple[int, ...], block: tuple[int, ...]):
    """Slices covering `shape` in steps of `block`."""
    ranges = [range(0, n, b) for n, b in zip(shape, block, strict=True)]
    for starts in itertools.product(*ranges):
        yield tuple(slice(s, min(s + b, n)) for s, b, n in zip(starts, block, shape, strict=True))


def _like(group: zarr.Group, name: str, src: zarr.Array, dtype: str, fill_value) -> zarr.Array:
    return group.create_array(
        name,
        shape=src.shape,
        dtype=dtype,
        chunks=src.chunks,
        shards=src.shards,
        fill_value=fill_value,
        compressors=src.compressors,
        dimension_names=src.metadata.dimension_names,
        attributes=dict(src.attrs),
    )


def dominant(contribution: np.ndarray) -> np.ndarray:
    """Index of the largest contribution along axis 0, NO_DRIVER where any is NaN."""
    valid = np.isfinite(contribution).all(axis=0)
    filled = np.where(np.isfinite(contribution), contribution, -np.inf)
    return np.where(valid, filled.argmax(axis=0), NO_DRIVER).astype('uint8')


def copy_level(src: zarr.Group, dst: zarr.Group, drivers: list[str]) -> None:
    """Copy one float32 pyramid level, casting contribution to float16 and adding dominant_driver."""
    for name, arr in src.arrays():
        if name == CONTRIBUTION:
            out = _like(dst, name, arr, 'float16', np.nan)
        else:
            out = _like(dst, name, arr, str(arr.dtype), arr.fill_value)
        for block in _blocks(arr.shape, arr.shards or arr.chunks):
            out[block] = arr[block]

    like = src[FDP]
    out = _like(dst, DOMINANT, like, 'uint8', NO_DRIVER)
    out.attrs.update(
        {
            'flag_values': list(range(len(drivers))),
            'flag_meanings': ' '.join(drivers),
            'long_name': 'predictor with the largest contribution',
        }
    )
    contribution = src[CONTRIBUTION]
    for block in _blocks(like.shape, like.shards or like.chunks):
        out[block] = dominant(contribution[(slice(None), *block)])


def write_store(
    cfg: StorageConfig, ds: xr.Dataset, out: OutputPaths, overwrite: bool, levels: int = LEVELS
) -> str:
    """Write the pyramid and return the snapshot id. Refuses an existing repo unless overwrite, which adds a commit on main."""
    storage = cfg.icechunk_storage(out.store)
    if icechunk.Repository.exists(storage) and not overwrite:
        raise FileExistsError(f'{cfg.uri(out.store)} exists; pass --overwrite')
    pyramid = create_pyramid(ds, levels=levels, x_dim='x', y_dim='y', method='mean')
    with tempfile.TemporaryDirectory(prefix='flood-retrained-') as tmp:
        local = Path(tmp) / 'pyramid.zarr'
        pyramid.write(str(local))
        src = zarr.open_group(str(local), mode='r')
        repo = icechunk.Repository.open_or_create(storage)
        session = repo.writable_session('main')
        root = zarr.open_group(session.store, mode='w', attributes={**src.attrs, **ds.attrs})
        for level in range(levels):
            group = src[str(level)]
            copy_level(
                group,
                root.create_group(str(level), attributes=dict(group.attrs)),
                ds.attrs['drivers'],
            )
        return session.commit(
            f'FDP refit {ds.attrs["run"]} {ds.attrs["distance"]}, HUCs {",".join(ds.attrs["hucs"])}'
        )


def verify(cfg: StorageConfig, out: OutputPaths, window: int = 1024) -> list[str]:
    """Check each level's value ranges, the sum identity and the dominant driver in a central window."""
    repo = icechunk.Repository.open(cfg.icechunk_storage(out.store))
    root = zarr.open_group(repo.readonly_session('main').store, mode='r')
    baselines = list(root.attrs['baselines'].values())
    levels = sorted(int(k) for k in root.group_keys())
    fails = []
    for level in levels:
        group = root[str(level)]
        ny, nx = group[FDP].shape
        y0, x0 = max(0, (ny - window) // 2), max(0, (nx - window) // 2)
        win = (slice(y0, y0 + window), slice(x0, x0 + window))
        p = group[FDP][win]
        c = group[CONTRIBUTION][(slice(None), *win)].astype('float32')
        d = group[DOMINANT][win]
        valid = np.isfinite(p)
        if not valid.any():
            fails.append(f'level {level}: sample window is empty')
            continue
        if np.isfinite(c[:, ~valid]).any() or (d[~valid] != NO_DRIVER).any():
            fails.append(f'level {level}: attribution set where fdp is not')
        if not np.isfinite(c[:, valid]).all():
            fails.append(f'level {level}: attribution missing where fdp is set')
            continue
        residual = np.abs(p[valid] - c[:, valid].sum(axis=0, dtype='float64') - np.mean(baselines))
        tol = (
            F16_EPS * np.abs(c[:, valid]).sum(axis=0).max()
            + len(c) * root.attrs['source_scale']
            + np.ptp(baselines)
            + 1e-5
        )
        if residual.max() > tol:
            fails.append(f'level {level}: sum identity off by {residual.max():.2e} (> {tol:.1e})')
        # float16 rounding can tie drivers that differ in float32, so check the
        # chosen driver holds the largest value rather than matching argmax
        chosen = np.take_along_axis(c[:, valid], d[valid][None].astype('intp'), axis=0)[0]
        if (d[valid] == NO_DRIVER).any() or (chosen != c[:, valid].max(axis=0)).any():
            fails.append(f'level {level}: dominant_driver disagrees with contribution')
        if SE in group:
            se = group[SE][win][valid]
            if not np.isfinite(se).all() or se.min() < 0 or se.max() > 0.5:
                fails.append(f'level {level}: fdp_se missing or outside [0, 0.5]')
        typer.echo(
            f'/{level}: shape={(ny, nx)} fdp {np.nanmin(p):.3f}..{np.nanmax(p):.3f} max residual {residual.max():.1e}'
        )
    return fails


@app.command('build')
def build_cmd(
    run: str = typer.Option(..., help='training run, e.g. noaa-200612-202512'),
    version: str = typer.Option(..., help='output version, e.g. v2.1.0'),
    hucs: list[str] = typer.Option(HUCS, help='two-digit HUC codes; all 18 by default'),
    distance: str = typer.Option('planar', help='distance predictor variant'),
    overwrite: bool = typer.Option(False, '--overwrite'),
) -> None:
    cfg, out = StorageConfig(), OutputPaths(Risk.FLOOD, version)
    grid = read_grid(cfg)
    sheds = {}
    for huc in sorted(hucs):
        prefix = run_prefix(run, distance, huc)
        typer.echo(f'reading {cfg.uri(prefix)}')
        sheds[huc] = read_watershed(cfg, prefix, grid)
    ds = mosaic(sheds, grid, run, distance)
    typer.echo(f'window {dict(ds.sizes)}')
    snapshot = write_store(cfg, ds, out, overwrite)
    typer.echo(f'wrote {cfg.uri(out.store)} : {snapshot}')
    if fails := verify(cfg, out):
        raise typer.Exit(_report(fails))
    typer.echo('OK')


@app.command('verify')
def verify_cmd(version: str = typer.Option(...)) -> None:
    if fails := verify(StorageConfig(), OutputPaths(Risk.FLOOD, version)):
        raise typer.Exit(_report(fails))
    typer.echo('OK')


def _report(fails: list[str]) -> int:
    typer.echo('FAIL', err=True)
    for f in fails:
        typer.echo(f'  - {f}', err=True)
    return 1


if __name__ == '__main__':
    app()
