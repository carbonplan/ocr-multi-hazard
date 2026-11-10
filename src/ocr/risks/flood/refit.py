"""Flood `score` stage: CarbonPlan's refit of the USGS Flood Damage Probability model.

`input-data/tensor/fdp/fdp_train.py` fits one random forest per HUC-2 watershed (in R) and uploads
each watershed's probability (`fdp.tif`), standard error (`se.tif`) and per-driver attributions
(`contributions.tif`, `contributions.json`) under `RUNS/<run>/<distance>/huc<NN>/`. `score` mosaics
RUN's watersheds into group "0" of the flood store, on their shared EPSG:5070 100 m grid cropped
to their union:

    fdp              (y, x)          int16  probability of flood damage
    fdp_se           (y, x)          int16  standard error of fdp (infinitesimal jackknife)
    contribution     (driver, y, x)  int16  change in fdp credited to each driver; per cell,
                                            the watershed's baseline + the sum over drivers = fdp
    dominant_driver  (y, x)          uint8  driver with the largest contribution

The first three are packed in steps of STEP (scale_factor), with fill NODATA. The `driver`
coordinate is the driver's index; its flag_meanings name them in order. These layers are chunked
for map tiles, and the pyramid stage adds coarser levels of them. Group "point" holds the same
packed values once more at full resolution, chunked for reading every layer of one cell at once:

    values  (layer, y, x)  int16  fdp, fdp_se, then each driver's contribution, as named by
                                  the `layer` coordinate's flag_meanings

Group "0" attributes carry the driver order, each watershed's baseline and the run. At most one
band of one watershed, or a few-shard block, is in memory at a time, so CONUS fits a small VM.
"""

import json
from dataclasses import dataclass
from typing import Annotated, Literal

import dask.array
import icechunk
import numpy as np
import obstore
import rasterio
import xarray as xr
import xproj  # noqa: F401 - registers the .proj accessor
import zarr
from annotated_types import MinLen
from topozarr import attach_geozarr_metadata, recommend_encoding

from ocr.config import OutputPaths, StorageConfig

RUNS = 'input/flood/fdp-refit/runs'
RUN = 'noaa-200612-202512'
DISTANCE = 'planar'
HUCS = tuple(f'{i:02d}' for i in range(1, 19))
CRS = 'EPSG:5070'
SOURCE = 'https://doi.org/10.5066/P954TTQN'

FDP = 'fdp'
SE = 'fdp_se'
CONTRIBUTION = 'contribution'
DOMINANT = 'dominant_driver'
STEP = 1e-4  # contributions.tif's scale
NODATA = -32768  # contributions.tif's nodata
NO_DRIVER = 255
PACKED = {
    'dtype': 'int16',
    'scale_factor': STEP,
    '_FillValue': NODATA,
    'fill_value': NODATA,  # zarr's, for chunks never written
}
POINT = 'point'
VALUES = 'values'
POINT_CHUNK = 32  # cells per side: one click reads 32 x 32 x 13 int16 (26 KB raw)
POINT_SHARD = 512
CHECK_CELLS = 1000  # per watershed, read back from the store for the sum identity


@dataclass(frozen=True)
class Watershed:
    huc: str
    prefix: str
    transform: rasterio.Affine
    shape: tuple[int, int]
    meta: dict  # contributions.json: drivers, baseline, scale, nodata, method

    def uri(self, cfg: StorageConfig, name: str) -> str:
        return cfg.uri(f'{self.prefix}/{name}')


def read_watershed(
    cfg: StorageConfig, huc: str, run: str = RUN, distance: str = DISTANCE
) -> Watershed:
    prefix = f'{RUNS}/{run}/{distance}/huc{huc}'
    meta = json.loads(
        bytes(obstore.get(cfg.object_store(), f'{prefix}/contributions.json').bytes())
    )
    with rasterio.open(cfg.uri(f'{prefix}/fdp.tif')) as src:
        if src.crs != CRS:
            raise ValueError(f'HUC {huc}: fdp.tif is in {src.crs}, not {CRS}')
        return Watershed(huc, prefix, src.transform, src.shape, meta)


@dataclass(frozen=True)
class Grid:
    transform: rasterio.Affine
    shape: tuple[int, int]
    offsets: dict[str, tuple[int, int]]  # huc -> (row, col) of its window


def union_grid(sheds: list[Watershed]) -> Grid:
    """The watersheds' shared lattice, cropped to their union. Errors if they don't share one."""
    t0 = sheds[0].transform
    offsets = {}
    for s in sheds:
        t = s.transform
        if (t.a, t.b, t.d, t.e) != (t0.a, t0.b, t0.d, t0.e):
            raise ValueError(f'HUC {s.huc}: resolution {(t.a, t.e)} differs from {(t0.a, t0.e)}')
        row, col = (t.f - t0.f) / t0.e, (t.c - t0.c) / t0.a
        if abs(row - round(row)) > 1e-6 or abs(col - round(col)) > 1e-6:
            raise ValueError(f'HUC {s.huc}: origin {(t.c, t.f)} is off the shared lattice')
        offsets[s.huc] = (round(row), round(col))
    r0 = min(r for r, _ in offsets.values())
    c0 = min(c for _, c in offsets.values())
    r1 = max(offsets[s.huc][0] + s.shape[0] for s in sheds)
    c1 = max(offsets[s.huc][1] + s.shape[1] for s in sheds)
    transform = t0 @ rasterio.Affine.translation(c0, r0)
    return Grid(
        transform, (r1 - r0, c1 - c0), {h: (r - r0, c - c0) for h, (r, c) in offsets.items()}
    )


def template(grid: Grid, sheds: list[Watershed], run: str, distance: str) -> xr.Dataset:
    """All-missing dataset on `grid`, lazily, so writing it stores only metadata and coordinates.
    Its encoding packs fdp, fdp_se and contribution as contributions.tif does."""
    drivers = sheds[0].meta['drivers']
    if any(s.meta['drivers'] != drivers for s in sheds):
        raise ValueError('watersheds disagree on driver order')
    if any((s.meta['scale'], s.meta['nodata']) != (STEP, NODATA) for s in sheds):
        raise ValueError(f'contributions.tif scale or nodata is not {STEP} or {NODATA}')
    t, (ny, nx) = grid.transform, grid.shape

    def empty(dims: tuple[str, ...], attrs: dict) -> tuple:
        shape = tuple({'driver': len(drivers), 'y': ny, 'x': nx}[d] for d in dims)
        return dims, dask.array.full(shape, np.nan, dtype='float32'), attrs

    ds = xr.Dataset(
        {
            FDP: empty(('y', 'x'), {'long_name': 'probability of flood damage'}),
            SE: empty(
                ('y', 'x'),
                {'long_name': 'standard error of fdp', 'method': 'infinitesimal jackknife'},
            ),
            CONTRIBUTION: empty(('driver', 'y', 'x'), {'long_name': 'contribution to fdp'}),
        },
        coords={
            'x': t.c + t.a * (np.arange(nx) + 0.5),
            'y': t.f + t.e * (np.arange(ny) + 0.5),
            'driver': ('driver', np.arange(len(drivers), dtype='int8'), _driver_flags(drivers)),
        },
        attrs={
            'title': 'CarbonPlan refit of the USGS Flood Damage Probability model',
            'source': SOURCE,
            'run': run,
            'distance': distance,
            'hucs': [s.huc for s in sheds],
            'drivers': drivers,
            'baselines': {s.huc: s.meta['baseline'] for s in sheds},
            'attribution': sheds[0].meta['method'],
        },
    )
    for name in (FDP, SE, CONTRIBUTION):
        ds[name].encoding = PACKED
    return attach_geozarr_metadata(ds.proj.assign_crs(spatial_ref=CRS).proj.write_crs_info())


def point_template(ds: xr.Dataset) -> xr.Dataset:
    """Group "point" for `ds` (`template`), all missing and lazy like it."""
    layers = [FDP, SE, *(f'{CONTRIBUTION}_{d}' for d in ds.attrs['drivers'])]
    n, ny, nx = len(layers), ds.sizes['y'], ds.sizes['x']
    shard = (n, POINT_SHARD, POINT_SHARD)
    data = dask.array.full((n, ny, nx), np.nan, dtype='float32', chunks=shard)
    point = xr.Dataset(
        {VALUES: (('layer', 'y', 'x'), data, {'long_name': 'every layer, for reading one cell'})},
        coords={
            'x': ds.x.variable,
            'y': ds.y.variable,
            'layer': (
                'layer',
                np.arange(n, dtype='int8'),
                {'flag_values': list(range(n)), 'flag_meanings': ' '.join(layers)},
            ),
        },
    )
    point[VALUES].encoding = PACKED | {
        'chunks': (n, POINT_CHUNK, POINT_CHUNK),
        'shards': shard,
    }
    return attach_geozarr_metadata(point.proj.assign_crs(spatial_ref=CRS).proj.write_crs_info())


def write_point(base: zarr.Group, point: zarr.Group) -> None:
    """Fill `point`'s values from `base`'s packed layers, whole shards at a time."""
    out, step = point[VALUES], 4 * POINT_SHARD
    _, ny, nx = out.shape
    for r in range(0, ny, step):
        for c in range(0, nx, step):
            win = (slice(r, r + step), slice(c, c + step))
            block = np.concatenate(
                [base[FDP][win][None], base[SE][win][None], base[CONTRIBUTION][:, *win]]
            )
            if (block != NODATA).any():
                out[:, *win] = block


def _driver_flags(drivers: list[str]) -> dict:
    return {'flag_values': list(range(len(drivers))), 'flag_meanings': ' '.join(drivers)}


def write_dominant(group: zarr.Group) -> None:
    """(Re)write `group`'s dominant_driver: per cell, the index of the driver with the largest
    `contribution`, or NO_DRIVER where a contribution is missing. A pyramid level's averaged
    contributions give that level's own dominant driver."""
    contrib, like = group[CONTRIBUTION], group[FDP]
    drivers = group['driver'].attrs['flag_meanings'].split()
    out = group.create_array(
        DOMINANT,
        shape=like.shape,
        dtype='uint8',
        chunks=like.chunks,
        shards=like.shards,
        fill_value=NO_DRIVER,
        dimension_names=('y', 'x'),
        attributes={
            'long_name': 'driver with the largest contribution',
            '_FillValue': NO_DRIVER,
            **_driver_flags(drivers),
        },
        overwrite=True,
    )
    (ny, nx), (bh, bw) = like.shape, like.shards or like.chunks
    for r in range(0, ny, bh):
        for c in range(0, nx, bw):
            block = contrib[:, r : r + bh, c : c + bw]
            valid = (block != contrib.fill_value).all(axis=0)
            if valid.any():
                out[r : r + bh, c : c + bw] = np.where(valid, block.argmax(axis=0), NO_DRIVER)


def _read_band(cfg: StorageConfig, s: Watershed, file: str, band: int) -> np.ma.MaskedArray:
    """One band of a watershed's raster, masked at nodata. Errors unless it is on fdp.tif's grid."""
    with rasterio.open(s.uri(cfg, file)) as src:
        if (src.crs, src.transform, src.shape) != (CRS, s.transform, s.shape):
            raise ValueError(f'HUC {s.huc}: {file} is not on the grid of its fdp.tif')
        return src.read(band, masked=True)


def _pack(values: np.ma.MaskedArray, name: str, huc: str) -> np.ma.MaskedArray:
    """fdp or fdp_se in int16 steps of STEP."""
    values = np.ma.masked_invalid(values)
    steps = np.ma.round(values.astype('float64') / STEP)
    top = np.iinfo('int16').max
    # se_fdp.R floors the calibrated variance at 0 and has no upper bound
    if steps.count() and (steps.min() < 0 or steps.max() > top):
        raise ValueError(f'HUC {huc}: {name} outside [0, {top * STEP:g}]')
    return np.ma.array(steps.filled(NODATA).astype('int16'), mask=np.ma.getmaskarray(steps))


def _is_set(values: np.ndarray, fill) -> np.ndarray:
    return np.isfinite(values) if np.isnan(fill) else values != fill


def _sample_cells(valid: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cells = np.flatnonzero(valid)
    pick = rng.choice(cells, size=min(CHECK_CELLS, cells.size), replace=False)
    return np.unravel_index(np.sort(pick), valid.shape)


def _merge(
    arr: zarr.Array, key: tuple, offset: tuple[int, int], band: np.ma.MaskedArray, huc: str
) -> None:
    """Write the unmasked cells of `band` into `arr[*key, window]`, keeping what other watersheds
    wrote."""
    (r, c), (h, w) = offset, band.shape
    win = (*key, slice(r, r + h), slice(c, c + w))
    valid = ~np.ma.getmaskarray(band)
    current = arr[win]
    if _is_set(current[valid], arr.fill_value).any():
        raise ValueError(f'HUC {huc}: {arr.basename} overlaps a watershed already written')
    current[valid] = band.data[valid]
    arr[win] = current


def check(
    group: zarr.Group, point: zarr.Group, sheds: list[Watershed], cells: dict[str, np.ndarray]
) -> list[str]:
    """Per watershed, at `cells` (rows, cols): every layer is set, baseline + contributions = fdp,
    dominant_driver is the largest contribution and `point` holds the same values."""
    contrib = group[CONTRIBUTION]
    n = contrib.shape[0]
    fails = []
    for s in sheds:
        rows, cols = cells[s.huc]
        p = group[FDP].vindex[rows, cols]
        se = group[SE].vindex[rows, cols]
        raw = contrib.vindex[np.arange(n)[:, None], rows[None], cols[None]]
        dominant = group[DOMINANT].vindex[rows, cols]
        values = point[VALUES].vindex[np.arange(n + 2)[:, None], rows[None], cols[None]]
        if (values != np.concatenate([p[None], se[None], raw])).any():
            fails.append(f'HUC {s.huc}: point values differ from the layers')
        if (se == NODATA).any() or (raw == NODATA).any():
            fails.append(f'HUC {s.huc}: fdp_se or a contribution missing where fdp is set')
            continue
        residual = np.abs(
            p * STEP - s.meta['baseline'] - raw.sum(axis=0, dtype='float64') * STEP
        ).max()
        # each contribution and fdp are rounded to their steps
        if residual > (n + 0.5) * STEP + 1e-6:
            fails.append(f'HUC {s.huc}: baseline + contributions off fdp by {residual:.1e}')
        if (dominant != raw.argmax(axis=0)).any():
            fails.append(f'HUC {s.huc}: dominant_driver is not the largest contribution')
    return fails


def score(
    out: OutputPaths,
    run: str = RUN,
    distance: Literal['planar', 'lonlat'] = DISTANCE,
    hucs: Annotated[tuple[Literal[*HUCS], ...], MinLen(1)] = HUCS,
) -> str:
    """Write the refit of `hucs` (all by default) from model run `run` to group "0" of `out.store`
    and return the snapshot id. Refuses an existing store (the stage's `--overwrite` deletes it
    first). Checks the result before committing."""
    cfg = StorageConfig()
    sheds = [read_watershed(cfg, huc, run, distance) for huc in sorted(hucs)]
    grid = union_grid(sheds)
    ds = template(grid, sheds, run, distance)

    repo = icechunk.Repository.create(cfg.icechunk_storage(out.store))
    session = repo.writable_session('main')
    ds.to_zarr(
        session.store,
        group='0',
        mode='w',
        compute=False,
        consolidated=False,
        encoding={k: v | ds[k].encoding for k, v in recommend_encoding(ds).items()},
    )
    point_template(ds).to_zarr(
        session.store, group=POINT, mode='w', compute=False, consolidated=False
    )
    group = zarr.open_group(session.store, path='0', mode='r+')
    point = zarr.open_group(session.store, path=POINT, mode='r+')

    rng = np.random.default_rng(0)
    cells = {}
    layers = [(FDP, (), 'fdp.tif', 1), (SE, (), 'se.tif', 1)]
    layers += [(CONTRIBUTION, (i,), 'contributions.tif', i + 1) for i in range(ds.sizes['driver'])]
    for name, key, file, band in layers:
        print(f'{name}{list(key)}: {len(sheds)} watersheds')
        for s in sheds:
            values = _read_band(cfg, s, file, band)
            if name in (FDP, SE):
                values = _pack(values, name, s.huc)
            if name == FDP:
                r, c = _sample_cells(~np.ma.getmaskarray(values), rng)
                cells[s.huc] = (r + grid.offsets[s.huc][0], c + grid.offsets[s.huc][1])
            _merge(group[name], key, grid.offsets[s.huc], values, s.huc)
            del values
    print(DOMINANT)
    write_dominant(group)
    print(POINT)
    write_point(group, point)

    if fails := check(group, point, sheds, cells):
        raise ValueError('refit check failed:\n  ' + '\n  '.join(fails))
    return session.commit(f'FDP refit {run} {distance}, HUCs {",".join(s.huc for s in sheds)}')
