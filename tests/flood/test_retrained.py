import json
from pathlib import Path

import icechunk
import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk

from ocr.config import OutputPaths, Risk, StorageConfig
from ocr.risks.flood.etl.flood import STORE_PREFIX as USGS_STORE
from ocr.risks.flood.retrained import (
    CONTRIBUTION,
    DOMINANT,
    FDP,
    NO_DRIVER,
    SE,
    dominant,
    mosaic,
    read_grid,
    read_watershed,
    run_prefix,
    verify,
    write_store,
)

DRIVERS = ['floodplain', 'hand', 'roads']
NODATA = -32768
SCALE = 1e-4
BASELINE = 0.5
X0, Y0, RES = -2_000_000.0, 2_000_000.0, 100.0
SHAPE = (64, 96)
# (row, col, height, width) of each watershed on the grid; they overlap
SHEDS = {'03': (0, 0, 40, 50), '05': (20, 40, 44, 56)}


def _write_tif(path: Path, data: np.ndarray, row: int, col: int, nodata) -> None:
    transform = rasterio.Affine(RES, 0, X0 + col * RES, 0, -RES, Y0 - row * RES)
    bands = data[None] if data.ndim == 2 else data
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=bands.shape[2],
        height=bands.shape[1],
        count=bands.shape[0],
        dtype=bands.dtype,
        crs='EPSG:5070',
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(bands)


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> StorageConfig:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    cfg = StorageConfig()

    x = X0 + RES * (np.arange(SHAPE[1]) + 0.5)
    y = Y0 - RES * (np.arange(SHAPE[0]) + 0.5)
    grid = xr.Dataset({'fdp': (('y', 'x'), np.zeros(SHAPE, 'float32'))}, coords={'x': x, 'y': y})
    session = icechunk.Repository.create(cfg.icechunk_storage(USGS_STORE)).writable_session('main')
    to_icechunk(grid, session)
    session.commit('grid')

    rng = np.random.default_rng(0)
    for huc, (row, col, h, w) in SHEDS.items():
        out = tmp_path / run_prefix('test-run', 'planar', huc)
        out.mkdir(parents=True)
        contribution = rng.integers(-2000, 2000, size=(len(DRIVERS), h, w)).astype('int16')
        fdp = (BASELINE + contribution.sum(axis=0) * SCALE).astype('float32')
        fdp[:3, :3] = np.nan
        contribution[:, :3, :3] = NODATA
        se = np.where(np.isfinite(fdp), 0.05, np.nan).astype('float32')
        _write_tif(out / 'fdp.tif', fdp, row, col, np.nan)
        _write_tif(out / 'contributions.tif', contribution, row, col, NODATA)
        _write_tif(out / 'se.tif', se, row, col, np.nan)
        meta = {
            'drivers': DRIVERS,
            'baseline': BASELINE,
            'scale': SCALE,
            'nodata': NODATA,
            'method': 'saabas',
        }
        (out / 'contributions.json').write_text(json.dumps(meta))
    return cfg


def _build(cfg: StorageConfig, out: OutputPaths, overwrite: bool = False) -> xr.Dataset:
    grid = read_grid(cfg)
    sheds = {huc: read_watershed(cfg, run_prefix('test-run', 'planar', huc), grid) for huc in SHEDS}
    ds = mosaic(sheds, grid, 'test-run', 'planar')
    write_store(cfg, ds, out, overwrite, levels=3)
    return ds


def _open(cfg: StorageConfig, out: OutputPaths) -> zarr.Group:
    repo = icechunk.Repository.open(cfg.icechunk_storage(out.store))
    return zarr.open_group(repo.readonly_session('main').store, mode='r')


def test_mosaic_crops_to_watersheds(cfg: StorageConfig) -> None:
    grid = read_grid(cfg)
    sheds = {huc: read_watershed(cfg, run_prefix('test-run', 'planar', huc), grid) for huc in SHEDS}
    ds = mosaic(sheds, grid, 'test-run', 'planar')
    assert dict(ds.sizes) == {'y': 64, 'x': 96, 'driver': 3}
    assert ds[CONTRIBUTION].dtype == np.float32
    assert np.isnan(ds[CONTRIBUTION].values[:, 0, 0]).all()
    assert np.isnan(ds[FDP].values[50, 0])  # outside both watersheds
    identity = BASELINE + ds[CONTRIBUTION].sum('driver', skipna=False)
    np.testing.assert_allclose(identity.values, ds[FDP].values, atol=1e-6)


def test_store_roundtrip(cfg: StorageConfig) -> None:
    out = OutputPaths(Risk.FLOOD, 'v0.0.1')
    ds = _build(cfg, out)
    root = _open(cfg, out)
    assert root.attrs['drivers'] == DRIVERS and root.attrs['hucs'] == sorted(SHEDS)
    level0 = root['0']
    contribution = level0[CONTRIBUTION]
    assert contribution.dtype == np.float16 and np.isnan(contribution.fill_value)
    np.testing.assert_array_equal(level0[FDP][:], ds[FDP].values)
    np.testing.assert_array_equal(contribution[:], ds[CONTRIBUTION].values.astype('float16'))
    np.testing.assert_array_equal(level0[DOMINANT][:], dominant(ds[CONTRIBUTION].values))
    assert level0[DOMINANT][0, 0] == NO_DRIVER
    assert SE in level0 and sorted(root.group_keys()) == ['0', '1', '2']
    assert verify(cfg, out) == []


def test_overwrite_guard(cfg: StorageConfig) -> None:
    out = OutputPaths(Risk.FLOOD, 'v0.0.1')
    _build(cfg, out)
    with pytest.raises(FileExistsError):
        _build(cfg, out)
    _build(cfg, out, overwrite=True)


def test_verify_catches_a_bad_dominant_driver(cfg: StorageConfig) -> None:
    out = OutputPaths(Risk.FLOOD, 'v0.0.1')
    _build(cfg, out)
    repo = icechunk.Repository.open(cfg.icechunk_storage(out.store))
    session = repo.writable_session('main')
    arr = zarr.open_group(session.store, mode='r+')['0'][DOMINANT]
    values = arr[:]
    values[10, 10] = (values[10, 10] + 1) % len(DRIVERS)
    arr[:] = values
    session.commit('corrupt')
    assert any('dominant_driver' in f for f in verify(cfg, out))
