"""Input data ETL for USGS Flood Damage Probability (FDP).

Converts the CONUS 100 m FDP GeoTIFF into group "0" of the flood output store (`OutputPaths.store`),
GeoZarr-tagged on its native EPSG:5070 grid (no reprojection; joins should move points into 5070,
not the raster out). The flood `score` stage writes the store from CarbonPlan's refit instead
(`ocr.risks.flood.refit`); this ETL writes the published USGS surface, e.g. for a comparison version.

Raw tif provenance (manual, one-time):
1. Download Output_CONUS_FDP_100m.zip from SOURCE (ScienceBase only serves expiring presigned links).
   SOURCE: https://www.sciencebase.gov/catalog/item/6170694ed34ea36449a67ef7
2. Unzip Output_CONUS_FDP_100m/CONUS_FDP_100m.tif (striped GeoTIFF, 100 m, EPSG:5070).
3. Upload it to RAW_KEY.

Run, then `ocr run` without score: pixi run -e flood flood-etl --version v2.0.0 [--prefix scratch/<you>] [--overwrite].
On Coiled: coiled batch run --vm-type r8g.xlarge --disk-size 50 --arm --tag Project=OCR -- python -m ocr.risks.flood.etl.flood --version v2.0.0
"""

from typing import Annotated

import icechunk
import rioxarray
import typer
import xarray as xr
import xproj  # noqa: F401
import zarr
from icechunk.xarray import to_icechunk
from topozarr import attach_geozarr_metadata, recommend_encoding

from ocr.config import OUTPUT_PREFIX, OutputPaths, Risk, StorageConfig

# TODO: move RAW_KEY into a shared input-path config once a second risk has raw inputs.
SOURCE = 'https://www.sciencebase.gov/catalog/item/6170694ed34ea36449a67ef7'
RAW_KEY = 'input/flood/raw/usgs-fdp/CONUS_FDP_100m.tif'


def read_tif(cfg: StorageConfig) -> xr.Dataset:
    """Read the FDP tif `fdp` variable in EPSG:5070"""
    da = rioxarray.open_rasterio(cfg.uri(RAW_KEY), masked=True, chunks=None)
    da = da.squeeze('band', drop=True).astype('float32').drop_vars('spatial_ref')

    # Removes "GDAL STATISTICS_*"" which serialize to invalid JSON in zarr.json.
    da.attrs = {k: v for k, v in da.attrs.items() if not k.startswith('STATISTICS_')}
    ds = da.to_dataset(name='fdp')
    ds.attrs = {
        'title': 'USGS Flood Damage Probability (FDP), CONUS 100 m',
        'source': SOURCE,
        'doi': 'https://doi.org/10.5066/P954TTQN',
    }
    # write_crs_info stores CF crs_wkt on spatial_ref, so the CRS survives a reopen (odc reads it).
    return ds.proj.assign_crs(spatial_ref='EPSG:5070').proj.write_crs_info()


def write_icechunk(cfg: StorageConfig, ds: xr.Dataset, prefix: str, overwrite: bool) -> str:
    """Write ds to group "0" of the repo at `prefix` and return the snapshot id. Refuses an existing
    repo unless overwrite, which replaces the whole hierarchy (drops pyramid levels) in a new commit
    on main. buildings/regions outputs are not deleted; rerun them after."""
    storage = cfg.icechunk_storage(prefix)
    if icechunk.Repository.exists(storage) and not overwrite:
        raise FileExistsError(f'{cfg.uri(prefix)} exists; pass --overwrite')
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session('main')
    zarr.open_group(session.store, mode='w')  # drop stale pyramid levels and root multiscales attrs
    to_icechunk(ds, session, group='0', mode='w', encoding=recommend_encoding(ds))
    return session.commit('USGS FDP 100m EPSG:5070')


def run(out: OutputPaths, overwrite: bool = False) -> None:
    cfg = StorageConfig()
    ds = read_tif(cfg)
    # Adds GeoZarr metadata
    ds = attach_geozarr_metadata(ds)
    snapshot = write_icechunk(cfg, ds, out.store, overwrite)
    print(f'wrote {cfg.uri(out.store)} : {snapshot}')


def main(
    version: Annotated[str, typer.Option()], prefix: str = OUTPUT_PREFIX, overwrite: bool = False
) -> None:
    run(OutputPaths(Risk.FLOOD, version, prefix), overwrite)


if __name__ == '__main__':
    typer.run(main)
