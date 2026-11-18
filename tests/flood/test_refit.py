import json
from pathlib import Path

import dask
import icechunk
import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr

from ocr import stages
from ocr.config import OutputPaths, Risk, StorageConfig
from ocr.risks.flood import refit
from ocr.risks.flood.refit import CONTRIBUTION, DOMINANT, FDP, RUNS, SE, score

DRIVERS = ['floodplain', 'hand', 'roads']
NODATA = -32768
SCALE = 1e-4
X0, Y0, RES = -2_000_000.0, 2_000_000.0, 100.0
# (row, col, height, width) of each watershed's window; windows overlap, cells don't
SHEDS = {'03': (0, 0, 40, 50), '05': (20, 40, 44, 56)}
BASELINES = {'03': 0.4, '05': 0.6}
OUT = OutputPaths(Risk.FLOOD, 'v0.0.1')
R_VERSION = 'R version 4.5.3 (2026-03-11)'
PACKAGES = ('raster', 'terra', 'sp', 'dismo', 'ranger', 'tuneRanger', 'mlr')
VERSIONS = ('3.6.32', '1.9.50', '2.2.3', '1.3.16', '0.18.0', '0.8.1', '2.19.4')


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
        c = rng.integers(-1000, 1000, size=(len(DRIVERS), h, w)).astype('int16')
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
        packages = list(VERSIONS) if huc == '03' else dict(zip(PACKAGES, VERSIONS))
        summary = {'r': R_VERSION, 'packages': packages}
        (out / 'summary.json').write_text(json.dumps(summary))


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
    assert ds.attrs['hucs'] == ['05'] and dict(ds.sizes) == {'driver': 3, 'y': 44, 'x': 56}


def _open(group: str = '0') -> xr.Dataset:
    repo = icechunk.Repository.open(StorageConfig().icechunk_storage(OUT.store))
    session = repo.readonly_session('main')
    return xr.open_zarr(session.store, group=group, consolidated=False, chunks=None)


def test_score_mosaics_watersheds(root: Path) -> None:
    _write_run(root)
    _score()
    ds = _open()
    assert dict(ds.sizes) == {'driver': 3, 'y': 64, 'x': 96}
    assert set(ds.data_vars) == {FDP, SE, CONTRIBUTION, DOMINANT}
    assert ds.driver.attrs['flag_meanings'] == ' '.join(DRIVERS)
    assert ds[CONTRIBUTION].dims == ('driver', 'y', 'x')
    assert all(ds[v].encoding['dtype'] == 'int16' for v in (FDP, SE, CONTRIBUTION))
    assert ds[DOMINANT].encoding['dtype'] == 'uint8'
    assert ds.odc.crs == 'EPSG:5070'
    assert ds.attrs['drivers'] == DRIVERS and ds.attrs['baselines'] == BASELINES
    assert ds.attrs['run'] == 'test-run' and ds.attrs['hucs'] == ['03', '05']
    # 03's summary.json lists the versions unnamed
    software = {'r': R_VERSION, 'packages': dict(zip(PACKAGES, VERSIONS))}
    assert ds.attrs['software'] == {'03': software, '05': software}
    np.testing.assert_allclose(ds.x[:2], [X0 + 50, X0 + 150])
    np.testing.assert_allclose(ds.y[:2], [Y0 - 50, Y0 - 150])

    fdp = ds[FDP].values
    # nodata, and outside both windows
    for v in ds.data_vars:
        assert ds[v][..., 0, 0].isnull().all() and ds[v][..., 63, 0].isnull().all()
    contrib = ds[CONTRIBUTION].sum('driver', skipna=False).values
    in_03 = np.zeros(fdp.shape, bool)
    in_03[:40, :50] = True
    baseline = np.where(in_03, 0.4, 0.6)
    np.testing.assert_allclose(fdp, baseline + contrib, atol=(len(DRIVERS) + 0.5) * SCALE)
    np.testing.assert_array_equal(np.isfinite(ds[SE].values), np.isfinite(fdp))
    dominant = ds[CONTRIBUTION].fillna(0).argmax('driver').where(np.isfinite(fdp))
    np.testing.assert_array_equal(ds[DOMINANT].values, dominant.values)


def _small_shards(ds: xr.Dataset) -> dict:
    # 64 x 96 cells in 12 x 20 shards, so both ends of each axis are partial
    lead = {k: (1,) * (v.ndim - 2) for k, v in ds.data_vars.items()}
    return {k: {'chunks': (*o, 6, 10), 'shards': (*o, 12, 20)} for k, o in lead.items()}


def test_score_spans_many_shards(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(refit, 'recommend_encoding', _small_shards)
    _write_run(root)
    with dask.config.set({'array.chunk-size': '4KiB'}):  # template chunks unaligned to shards
        _score()
    ds = _open()
    assert ds[CONTRIBUTION].encoding['shards'] == (1, 12, 20)
    assert ds[DOMINANT].encoding['shards'] == (12, 20)
    dominant = ds[CONTRIBUTION].fillna(0).argmax('driver').where(ds[FDP].notnull())
    np.testing.assert_array_equal(ds[DOMINANT].values, dominant.values)


def test_score_refuses_existing_store(root: Path) -> None:
    _write_run(root)
    _score()
    with pytest.raises(icechunk.IcechunkError):
        _score()


def test_score_rejects_overlapping_watersheds(root: Path) -> None:
    _write_run(root, disjoint=False)
    with pytest.raises(ValueError, match='HUC 05: fdp overlaps'):
        _score()


def test_score_reports_every_problem_before_committing(root: Path) -> None:
    _write_run(root, disjoint=False)
    path = root / RUNS / 'test-run/planar/huc03/se.tif'
    with rasterio.open(path, 'r+') as dst:
        dst.write(np.full((1, dst.height, dst.width), 5.0, 'float32'))
    with pytest.raises(ValueError) as err:
        _score()
    assert 'HUC 03: fdp_se outside' in str(err.value)
    assert 'HUC 05: fdp overlaps' in str(err.value)
    repo = icechunk.Repository.open(StorageConfig().icechunk_storage(OUT.store))
    assert len(list(repo.ancestry(branch='main'))) == 1


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
    for name in ('fdp.tif', 'se.tif', 'contributions.tif'):
        with rasterio.open(root / RUNS / 'test-run/planar/huc05' / name, 'r+') as dst:
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
    stages.pyramid(stages.RISKS[Risk.FLOOD], OUT, overwrite=True, levels=3)
    repo = icechunk.Repository.open(StorageConfig().icechunk_storage(OUT.store))
    session = repo.readonly_session('main')
    level1 = xr.open_zarr(session.store, group='1', consolidated=False, chunks=None)
    assert dict(level1.sizes) == {'driver': 3, 'y': 32, 'x': 48}
    assert set(level1.data_vars) == {FDP, SE, CONTRIBUTION, DOMINANT}
    assert zarr.open_group(session.store, path='1', mode='r')[CONTRIBUTION].dtype == 'int16'
    # means of the valid cells under each level-1 cell; nodata never averaged in
    level0 = _open()
    for v in (FDP, SE, CONTRIBUTION):
        expected = level0[v].coarsen(y=2, x=2).mean()
        np.testing.assert_allclose(level1[v].values, expected.values, atol=SCALE)
    # from level 1's own averaged contributions
    valid = level1[CONTRIBUTION].notnull().all('driver')
    dominant = level1[CONTRIBUTION].fillna(0).argmax('driver').where(valid)
    np.testing.assert_array_equal(level1[DOMINANT].values, dominant.values)


def test_pyramid_levels_stay_packed(root: Path) -> None:
    _write_run(root)
    _score()
    stages.pyramid(stages.RISKS[Risk.FLOOD], OUT, overwrite=True, levels=3)
    repo = icechunk.Repository.open(StorageConfig().icechunk_storage(OUT.store))
    session = repo.readonly_session('main')
    for level in ('1', '2'):
        group = zarr.open_group(session.store, path=level, mode='r')
        for v in (FDP, SE, CONTRIBUTION):
            assert group[v].dtype == 'int16' and group[v].fill_value == NODATA
            assert group[v].attrs['scale_factor'] == SCALE
            assert group[v].attrs['_FillValue'] == NODATA
