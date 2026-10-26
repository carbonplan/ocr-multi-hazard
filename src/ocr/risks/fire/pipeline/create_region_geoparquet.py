from ocr.regions import write_region_files
from ocr.risks.fire.config import OCRConfig
from ocr.risks.fire.console import console
from ocr.risks.fire.utils import get_temp_dir

# Values the web map shows for a region, read in every lookup.
SHOWN = ['rps_2011_median', 'rps_2047_median', 'risk_score_2011_hist', 'risk_score_2047_hist']


def create_region_geoparquet(config: OCRConfig) -> None:
    """Rewrite the region summary statistics as GeoParquet laid out for both
    analysis and the web map's region lookups."""
    vector = config.vector
    sources = {
        'nation': vector.nation_summary_stats_uri,
        'states': vector.states_summary_stats_uri,
        'counties': vector.counties_summary_stats_uri,
        'tracts': vector.tracts_summary_stats_uri,
        'blocks': vector.block_summary_stats_uri,
    }
    write_region_files(
        sources,
        vector.regions_uri,
        shown=SHOWN,
        metadata=vector.metadata_dict,
        temp_dir=get_temp_dir(),
        log=console.log if config.debug else (lambda *_: None),
    )
    if config.debug:
        console.log(f'region files written to {vector.regions_uri}')
