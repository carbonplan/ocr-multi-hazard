"""Flood `score` stage: CarbonPlan's refit of the USGS Flood Damage Probability model.

`input-data/tensor/fdp/fdp_train.py` fits one random forest per HUC-2 watershed (in R) and uploads
each watershed's forest (`rf.rds`), probability (`fdp.tif`), standard error (`se.tif`), per-driver
attributions (`contributions.tif`, `contributions.json`) and fit summary (`summary.json`) under
`RUNS/<run>/<distance>/huc<NN>/`. `score` mosaics
RUN's watersheds into group "0" of the flood store, on their shared EPSG:5070 100 m grid cropped
to their union:

    fdp              (y, x)          int16  probability of flood damage
    fdp_se           (y, x)          int16  standard error of fdp (infinitesimal jackknife)
    contribution     (driver, y, x)  int16  change in fdp credited to each driver; per cell,
                                            the watershed's baseline + the sum over drivers = fdp
    dominant_driver  (y, x)          uint8  driver with the largest contribution

The first three are packed in steps of STEP (scale_factor), with fill NODATA. The `driver`
coordinate is the driver's index; its flag_meanings name them in order. These layers are chunked
for map tiles, and the pyramid stage adds coarser levels of them.

Group "0" attributes carry the driver order, the run, and each watershed's baseline, forest (the
ETag of its rf.rds) and the R and R package versions that fit it. Score fills
one row of shards of one layer at a time from every watershed it crosses, so each shard is written
once and CONUS fits a small VM.
"""

import json
from contextlib import ExitStack
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
from rasterio.windows import Window
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
CHECK_CELLS = 1000  # per watershed, read back from the store for the sum identity
# order of summary.json's `packages` where it is a list rather than named
SUMMARY_PACKAGES = ('raster', 'terra', 'sp', 'dismo', 'ranger', 'tuneRanger', 'mlr')


@dataclass(frozen=True)
class Watershed:
    huc: str
    prefix: str
    transform: rasterio.Affine
    shape: tuple[int, int]
    meta: dict  # contributions.json: drivers, baseline, scale, nodata, method
    software: dict  # R and R package versions, from summary.json
    model: str  # ETag of rf.rds

    def uri(self, cfg: StorageConfig, name: str) -> str:
        return cfg.uri(f'{self.prefix}/{name}')


def read_watershed(
    cfg: StorageConfig, huc: str, run: str = RUN, distance: str = DISTANCE
) -> Watershed:
    prefix = f'{RUNS}/{run}/{distance}/huc{huc}'
    store = cfg.object_store()
    meta = json.loads(bytes(obstore.get(store, f'{prefix}/contributions.json').bytes()))
    summary = json.loads(bytes(obstore.get(store, f'{prefix}/summary.json').bytes()))
    packages = summary['packages']
    if isinstance(packages, list):
        packages = dict(zip(SUMMARY_PACKAGES, packages, strict=True))
    software = {'r': summary['r'], 'packages': packages}
    model = obstore.head(store, f'{prefix}/rf.rds')['e_tag'].strip('"')
    with rasterio.open(cfg.uri(f'{prefix}/fdp.tif')) as src:
        if src.crs != CRS:
            raise ValueError(f'HUC {huc}: fdp.tif is in {src.crs}, not {CRS}')
        shed = Watershed(huc, prefix, src.transform, src.shape, meta, software, model)
    for file, bands in (('se.tif', 1), ('contributions.tif', len(meta['drivers']))):
        with rasterio.open(shed.uri(cfg, file)) as src:
            if (src.crs, src.transform, src.shape) != (CRS, shed.transform, shed.shape):
                raise ValueError(f'HUC {huc}: {file} is not on the grid of its fdp.tif')
            if src.count != bands:
                raise ValueError(f'HUC {huc}: {file} has {src.count} bands, not {bands}')
    return shed


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
            'software': {s.huc: s.software for s in sheds},
            'models': {s.huc: s.model for s in sheds},
            'attribution': sheds[0].meta['method'],
        },
    )
    for name in (FDP, SE, CONTRIBUTION):
        ds[name].encoding = PACKED
    return attach_geozarr_metadata(ds.proj.assign_crs(spatial_ref=CRS).proj.write_crs_info())


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


def _fail(fails: list[str], message: str) -> None:
    if message not in fails:
        fails.append(message)


def _pack(values: np.ma.MaskedArray, name: str, huc: str, fails: list[str]) -> np.ma.MaskedArray:
    """fdp or fdp_se in int16 steps of STEP, clipped to [0, int16 max]."""
    values = np.ma.masked_invalid(values)
    steps = np.ma.round(values.astype('float64') / STEP)
    top = np.iinfo('int16').max
    # se_fdp.R floors the calibrated variance at 0 and has no upper bound
    if steps.count() and (steps.min() < 0 or steps.max() > top):
        _fail(fails, f'HUC {huc}: {name} outside [0, {top * STEP:g}]')
        steps = np.ma.clip(steps, 0, top)
    return np.ma.array(steps.filled(NODATA).astype('int16'), mask=np.ma.getmaskarray(steps))


def _write_layer(
    cfg: StorageConfig,
    group: zarr.Group,
    grid: Grid,
    sheds: list[Watershed],
    layer: tuple[str, tuple, str, int],
    fails: list[str],
    sample: dict[str, list] | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Fill one layer of `group` a row of shards at a time, each row from the watersheds it
    crosses; a cell two watersheds set keeps the first one's value. Problems go to `fails`. With
    `sample`, adds to it, per watershed and row, the count of cells set and up to CHECK_CELLS of
    them in random order."""
    name, key, file, band = layer
    arr = group[name]
    step = (arr.shards or arr.chunks)[-2]
    ny, nx = grid.shape
    with ExitStack() as stack:
        sources = {s.huc: stack.enter_context(rasterio.open(s.uri(cfg, file))) for s in sheds}
        for r0 in range(0, ny, step):
            r1 = min(r0 + step, ny)
            strip = np.full((r1 - r0, nx), arr.fill_value, arr.dtype)
            for s in sheds:
                (row, col), (h, w) = grid.offsets[s.huc], s.shape
                lo, hi = max(r0, row), min(r1, row + h)
                if lo >= hi:
                    continue
                values = sources[s.huc].read(
                    band, window=Window(0, lo - row, w, hi - lo), masked=True
                )
                if name in (FDP, SE):
                    values = _pack(values, name, s.huc, fails)
                valid = ~np.ma.getmaskarray(values)
                target = strip[lo - r0 : hi - r0, col : col + w]
                taken = target != arr.fill_value
                if (valid & taken).any():
                    _fail(fails, f'HUC {s.huc}: {name} overlaps a watershed already written')
                    valid &= ~taken
                target[valid] = values.data[valid]
                if sample is not None:
                    rows, cols = np.nonzero(valid)
                    pick = rng.choice(rows.size, size=min(CHECK_CELLS, rows.size), replace=False)
                    sample[s.huc].append((rows.size, rows[pick] + lo, cols[pick] + col))
            arr[(*key, slice(r0, r1), slice(None))] = strip


def _sample_cells(
    sample: dict[str, list], rng: np.random.Generator
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Up to CHECK_CELLS of each watershed's set cells, drawn uniformly from all of them: each
    row's share is drawn in proportion to its count, from its cells in random order."""
    cells = {}
    for huc, parts in sample.items():
        if not parts:
            cells[huc] = (np.array([], 'int64'), np.array([], 'int64'))
            continue
        counts = np.array([n for n, _, _ in parts], dtype='int64')
        quota = rng.multivariate_hypergeometric(counts, min(CHECK_CELLS, int(counts.sum())))
        rows = np.concatenate([r[:q] for (_, r, _), q in zip(parts, quota)])
        cols = np.concatenate([c[:q] for (_, _, c), q in zip(parts, quota)])
        cells[huc] = (rows.astype('int64'), cols.astype('int64'))
    return cells


def check(group: zarr.Group, sheds: list[Watershed], cells: dict[str, np.ndarray]) -> list[str]:
    """Per watershed, at `cells` (rows, cols), where fdp is set: fdp_se and every contribution are
    set, and where the contributions are, baseline + contributions = fdp and dominant_driver is
    the largest contribution."""
    contrib = group[CONTRIBUTION]
    n = contrib.shape[0]
    fails = []
    for s in sheds:
        rows, cols = cells[s.huc]
        if not rows.size:
            fails.append(f'HUC {s.huc}: no fdp cells of its own to check')
            continue
        p = group[FDP].vindex[rows, cols]
        se = group[SE].vindex[rows, cols]
        raw = contrib.vindex[np.arange(n)[:, None], rows[None], cols[None]]
        dominant = group[DOMINANT].vindex[rows, cols]
        if missing := int((se == NODATA).sum()):
            fails.append(f'HUC {s.huc}: fdp_se missing at {missing} of {rows.size} cells checked')
        full = (raw != NODATA).all(axis=0)
        if missing := int((~full).sum()):
            fails.append(f'HUC {s.huc}: a contribution missing at {missing} of {rows.size} cells')
        if not full.any():
            continue
        p, raw, dominant = p[full], raw[:, full], dominant[full]
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
    first). Checks the inputs' grids before writing and the result before committing, and
    raises one ValueError listing every problem found."""
    cfg = StorageConfig()
    sheds = [read_watershed(cfg, huc, run, distance) for huc in sorted(hucs)]
    grid = union_grid(sheds)
    ds = template(grid, sheds, run, distance)

    repo = icechunk.Repository.create(cfg.icechunk_storage(out.store))
    session = repo.writable_session('main')
    encoding = {k: v | ds[k].encoding for k, v in recommend_encoding(ds).items()}
    # one driver per shard, so each contribution shard is written once
    for k in ('chunks', 'shards'):
        encoding[CONTRIBUTION][k] = (1, *encoding[CONTRIBUTION][k][1:])
    for k, enc in encoding.items():  # xarray requires dask chunks aligned to the shards
        ds[k] = ds[k].chunk(dict(zip(ds[k].dims, enc['shards'])))
    ds.to_zarr(
        session.store, group='0', mode='w', compute=False, consolidated=False, encoding=encoding
    )
    group = zarr.open_group(session.store, path='0', mode='r+')

    rng = np.random.default_rng(0)
    fails = []
    sample = {s.huc: [] for s in sheds}
    layers = [(FDP, (), 'fdp.tif', 1), (SE, (), 'se.tif', 1)]
    layers += [(CONTRIBUTION, (i,), 'contributions.tif', i + 1) for i in range(ds.sizes['driver'])]
    for layer in layers:
        name, key = layer[:2]
        print(f'{name}{list(key)}: {len(sheds)} watersheds')
        _write_layer(cfg, group, grid, sheds, layer, fails, sample if name == FDP else None, rng)
    cells = _sample_cells(sample, rng)
    print(DOMINANT)
    write_dominant(group)

    if fails := fails + check(group, sheds, cells):
        raise ValueError('refit check failed:\n  ' + '\n  '.join(fails))
    return session.commit(f'FDP refit {run} {distance}, HUCs {",".join(s.huc for s in sheds)}')
