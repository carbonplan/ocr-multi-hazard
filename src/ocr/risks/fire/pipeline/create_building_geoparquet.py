from ocr.lod import write_buildings_lod
from ocr.risks.fire.config import OCRConfig
from ocr.risks.fire.console import console
from ocr.risks.fire.utils import get_temp_dir

# Buildings the coarsest zooms sample; zero-risk ones join, biggest first, from z6.
SHOWN = 'rps_2011 > 0 AND rps_2047 > 0'
# Columns map tiles carry for styling, at half precision: the map colours by
# bins, and selecting a building reads its full-precision values by row.
TILE_COLUMNS = [('rps_2011', 'halffloat'), ('rps_2047', 'halffloat')]


def create_building_geoparquet(config: OCRConfig) -> None:
    """Write every processed building to one GeoParquet file with map overviews,
    for analysis and for the web map."""
    src = f'{config.vector.region_geoparquet_uri}/*.parquet'
    dest = config.vector.building_geoparquet_uri
    write_buildings_lod(
        src,
        dest,
        shown=SHOWN,
        carry=TILE_COLUMNS,
        metadata=config.vector.metadata_dict,
        temp_dir=get_temp_dir(),
        log=console.log if config.debug else (lambda *_: None),
    )
    if config.debug:
        console.log(f'buildings written to {dest}')
