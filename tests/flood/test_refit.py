import json
from pathlib import Path

import icechunk
import numpy as np
import pytest
import rasterio
import xarray as xr

from ocr import stages
from ocr.config import OutputPaths, Risk, StorageConfig
from ocr.risks.flood.refit import FDP, RUNS, SE, contribution, score

DRIVERS = ['floodplain', 'hand', 'roads']
NODATA = -32768
SCALE = 1e-4
X0, Y0, RES = -2_000_000.0, 2_000_000.0, 100.0
# (row, col, height, width) of each watershed's window; windows overlap, cells don't
SHEDS = {'03': (0, 0, 40, 50), '05': (20, 40, 44, 56)}
BASELINES = {'03': 0.4, '05': 0.6}
OUT = OutputPaths(Risk.FLOOD, 'v0.0.1')


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


def _write_run(root: Path, sheds=SHEDS, baselines=BASELINES, disjoint: bool = True) -> None:
    rng = np.random.default_rng(0)
    for huc, (row, col, h, w) in sheds.items():
        out = root / RUNS / 'test-run' / 'planar' / f'huc{huc}'
        out.mkdir(parents=True)
        c = rng.integers(-2000, 2000, size=(len(DRIVERS), h, w)).astype('int16')
        fdp = (baselines[huc] + c.sum(axis=0, dtype='float64') * SCALE).astype('float32')
        outside = np.zeros((h, w), bool)
        outside[:3, :3] = True
        if disjoint and huc == '05':  # 03's window covers 05's top-left corner
            outside[: 40 - row, : 50 - col] = True
        fdp[outside] = np.nan
        c[:, outside] = NODATA
        se = np.where(outside, np.nan, 0.05).astype('float32')
        _write_tif(out / 'fdp.tif', fdp, row, col, np.nan)
        _write_tif(out / 'contributions.tif', c, row, col, NODATA)
        _write_tif(out / 'se.tif', se, row, col, np.nan)
        meta = {
            'drivers': DRIVERS,
            'baseline': baselines[huc],
            'scale': SCALE,
            'nodata': NODATA,
            'method': 'saabas',
        }
        (out / 'contributions.json').write_text(json.dumps(meta))


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('OCR_STORAGE_LOCAL_ROOT', str(tmp_path))
    return tmp_path


def _score() -> None:
    # Through the stage, as `ocr stage flood score` runs it.
    stages.score(stages.RISKS[Risk.FLOOD], OUT, False, run='test-run', hucs=tuple(SHEDS))


def test_score_reads_only_the_hucs_given(root: Path) -> None:
    _write_run(root)
    score(OUT, run='test-run', hucs=('05',))
    ds = _open()
    assert ds.attrs['hucs'] == ['05'] and dict(ds.sizes) == {'y': 44, 'x': 56}


def _open() -> xr.Dataset:
    repo = icechunk.Repository.open(StorageConfig().icechunk_storage(OUT.store))
    session = repo.readonly_session('main')
    return xr.open_zarr(session.store, group='0', consolidated=False, chunks=None)


def test_score_mosaics_watersheds(root: Path) -> None:
    _write_run(root)
    _score()
    ds = _open()
    assert dict(ds.sizes) == {'y': 64, 'x': 96}
    assert set(ds.data_vars) == {FDP, SE, *map(contribution, DRIVERS)}
    assert all(ds[v].dtype == np.float32 for v in ds.data_vars)
    assert ds.odc.crs == 'EPSG:5070'
    assert ds.attrs['drivers'] == DRIVERS and ds.attrs['baselines'] == BASELINES
    assert ds.attrs['run'] == 'test-run' and ds.attrs['hucs'] == ['03', '05']
    np.testing.assert_allclose(ds.x[:2], [X0 + 50, X0 + 150])
    np.testing.assert_allclose(ds.y[:2], [Y0 - 50, Y0 - 150])

    fdp = ds[FDP].values
    assert np.isnan(fdp[0, 0]) and np.isnan(fdp[63, 0])  # nodata, and outside both windows
    contrib = sum(ds[contribution(d)].values.astype('float64') for d in DRIVERS)
    in_03 = np.zeros(fdp.shape, bool)
    in_03[:40, :50] = True
    baseline = np.where(in_03, 0.4, 0.6)
    np.testing.assert_allclose(fdp, baseline + contrib, atol=len(DRIVERS) * SCALE)
    np.testing.assert_array_equal(np.isfinite(ds[SE].values), np.isfinite(fdp))


def test_score_refuses_existing_store(root: Path) -> None:
    _write_run(root)
    _score()
    with pytest.raises(icechunk.IcechunkError):
        _score()


def test_score_rejects_overlapping_watersheds(root: Path) -> None:
    _write_run(root, disjoint=False)
    with pytest.raises(ValueError, match='HUC 05: fdp overlaps'):
        _score()


def test_score_checks_the_sum_identity(root: Path) -> None:
    _write_run(root)
    meta = root / RUNS / 'test-run/planar/huc05/contributions.json'
    meta.write_text(json.dumps(json.loads(meta.read_text()) | {'baseline': 0.5}))
    with pytest.raises(ValueError, match='HUC 05: baseline \\+ contributions off fdp'):
        _score()
    # nothing committed
    repo = icechunk.Repository.open(StorageConfig().icechunk_storage(OUT.store))
    assert len(list(repo.ancestry(branch='main'))) == 1


def test_off_lattice_watershed_fails(root: Path) -> None:
    _write_run(root)
    path = root / RUNS / 'test-run/planar/huc05/fdp.tif'
    with rasterio.open(path, 'r+') as dst:
        t = dst.transform
        dst.transform = rasterio.Affine(t.a, 0, t.c + 30, 0, t.e, t.f)
    with pytest.raises(ValueError, match='off the shared lattice'):
        _score()


def test_shifted_ancillary_raster_fails(root: Path) -> None:
    _write_run(root)
    path = root / RUNS / 'test-run/planar/huc05/se.tif'
    with rasterio.open(path, 'r+') as dst:
        t = dst.transform
        dst.transform = rasterio.Affine(t.a, 0, t.c + RES, 0, t.e, t.f)
    with pytest.raises(ValueError, match='HUC 05: se.tif is not on the grid of its fdp.tif'):
        _score()


def test_pyramid_stage_on_refit_store(root: Path) -> None:
    _write_run(root)
    _score()
    stages.pyramid(stages.RISKS[Risk.FLOOD], OUT, overwrite=False, levels=3)
    repo = icechunk.Repository.open(StorageConfig().icechunk_storage(OUT.store))
    session = repo.readonly_session('main')
    level1 = xr.open_zarr(session.store, group='1', consolidated=False, chunks=None)
    assert dict(level1.sizes) == {'y': 32, 'x': 48}
    assert set(level1.data_vars) == {FDP, SE, *map(contribution, DRIVERS)}
