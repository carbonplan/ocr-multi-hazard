"""Flood `score` stage: CarbonPlan's refit of the USGS Flood Damage Probability model.

`input-data/tensor/fdp/fdp_train.py` fits one random forest per HUC-2 watershed (in R) and uploads
each watershed's probability (`fdp.tif`), standard error (`se.tif`) and per-driver attributions
(`contributions.tif`, `contributions.json`) under `RUNS/<run>/<distance>/huc<NN>/`. `score` mosaics
RUN's watersheds into group "0" of the flood store, on their shared EPSG:5070 100 m grid cropped
to their union:

    fdp                   probability of flood damage
    fdp_se                standard error of fdp (infinitesimal jackknife)
    contribution_<driver> change in fdp credited to the driver; per cell, the
                          watershed's baseline + the sum over drivers = fdp

NaN is no data. Group attributes carry the driver order, each watershed's baseline and the run.
One band of one watershed is in memory at a time, so CONUS fits a small VM.
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

RUNS = 'ocr-explore/FDP/runs'
RUN = 'noaa-200612-202512'
DISTANCE = 'planar'
HUCS = tuple(f'{i:02d}' for i in range(1, 19))
CRS = 'EPSG:5070'
SOURCE = 'https://doi.org/10.5066/P954TTQN'

FDP = 'fdp'
SE = 'fdp_se'
CHECK_CELLS = 1000  # per watershed, read back from the store for the sum identity


def contribution(driver: str) -> str:
    return f'contribution_{driver}'


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
    """All-NaN dataset on `grid`, lazily, so writing it stores only metadata and coordinates."""
    drivers = sheds[0].meta['drivers']
    if any(s.meta['drivers'] != drivers for s in sheds):
        raise ValueError('watersheds disagree on driver order')
    t, (ny, nx) = grid.transform, grid.shape

    def empty(attrs: dict) -> tuple:
        return ('y', 'x'), dask.array.full((ny, nx), np.nan, dtype='float32'), attrs

    data = {
        FDP: empty({'long_name': 'probability of flood damage'}),
        SE: empty({'long_name': 'standard error of fdp', 'method': 'infinitesimal jackknife'}),
    }
    for d in drivers:
        data[contribution(d)] = empty({'long_name': f'contribution of {d} to fdp'})
    ds = xr.Dataset(
        data,
        coords={'x': t.c + t.a * (np.arange(nx) + 0.5), 'y': t.f + t.e * (np.arange(ny) + 0.5)},
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
    return attach_geozarr_metadata(ds.proj.assign_crs(spatial_ref=CRS).proj.write_crs_info())


def _read_band(cfg: StorageConfig, s: Watershed, file: str, band: int, scale: float) -> np.ndarray:
    """One band of a watershed's raster, NaN for nodata. Errors unless it is on fdp.tif's grid."""
    with rasterio.open(s.uri(cfg, file)) as src:
        if (src.crs, src.transform, src.shape) != (CRS, s.transform, s.shape):
            raise ValueError(f'HUC {s.huc}: {file} is not on the grid of its fdp.tif')
        values = src.read(band, masked=True).astype('float32')
    return (values * np.float32(scale)).filled(np.nan)


def _sample_cells(values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    valid = np.flatnonzero(np.isfinite(values))
    pick = rng.choice(valid, size=min(CHECK_CELLS, valid.size), replace=False)
    return np.unravel_index(np.sort(pick), values.shape)


def _merge(arr: zarr.Array, offset: tuple[int, int], values: np.ndarray, huc: str) -> None:
    """Write the valid cells of `values` into `arr`, keeping what other watersheds wrote."""
    (r, c), (h, w) = offset, values.shape
    win = (slice(r, r + h), slice(c, c + w))
    valid = np.isfinite(values)
    current = arr[win]
    if np.isfinite(current[valid]).any():
        raise ValueError(f'HUC {huc}: {arr.basename} overlaps a watershed already written')
    current[valid] = values[valid]
    arr[win] = current


def check(group: zarr.Group, sheds: list[Watershed], cells: dict[str, np.ndarray]) -> list[str]:
    """Per watershed, at `cells` (rows, cols): every layer is set and baseline + contributions = fdp."""
    drivers = sheds[0].meta['drivers']
    fails = []
    for s in sheds:
        rows, cols = cells[s.huc]
        p = group[FDP].vindex[rows, cols]
        se = group[SE].vindex[rows, cols]
        c = np.stack([group[contribution(d)].vindex[rows, cols] for d in drivers])
        if not (np.isfinite(se).all() and np.isfinite(c).all()):
            fails.append(f'HUC {s.huc}: fdp_se or a contribution missing where fdp is set')
            continue
        # se_fdp.R floors the calibrated variance at 0 and has no upper bound
        if se.min() < 0:
            fails.append(f'HUC {s.huc}: negative fdp_se')
        residual = np.abs(p - s.meta['baseline'] - c.sum(axis=0, dtype='float64')).max()
        # contributions.tif stores each driver in steps of `scale`
        if residual > len(drivers) * s.meta['scale'] + 1e-6:
            fails.append(f'HUC {s.huc}: baseline + contributions off fdp by {residual:.1e}')
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
        encoding=recommend_encoding(ds),
    )
    group = zarr.open_group(session.store, path='0', mode='r+')

    rng = np.random.default_rng(0)
    cells = {}
    layers = [(FDP, 'fdp.tif', 1, False), (SE, 'se.tif', 1, False)]
    layers += [
        (contribution(d), 'contributions.tif', i, True)
        for i, d in enumerate(sheds[0].meta['drivers'], 1)
    ]
    for name, file, band, scaled in layers:
        print(f'{name}: {len(sheds)} watersheds')
        for s in sheds:
            values = _read_band(cfg, s, file, band, s.meta['scale'] if scaled else 1.0)
            if name == FDP:
                r, c = _sample_cells(values, rng)
                cells[s.huc] = (r + grid.offsets[s.huc][0], c + grid.offsets[s.huc][1])
            _merge(group[name], grid.offsets[s.huc], values, s.huc)
            del values

    if fails := check(group, sheds, cells):
        raise ValueError('refit check failed:\n  ' + '\n  '.join(fails))
    return session.commit(f'FDP refit {run} {distance}, HUCs {",".join(s.huc for s in sheds)}')
