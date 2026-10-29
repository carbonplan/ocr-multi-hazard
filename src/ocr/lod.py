"""Building GeoParquet for analysis and the map in one file.

The output is GeoParquet with every source column and the original geometry,
plus geoparquet-lod overviews: per-zoom simplified copies, Hilbert order and a
spatial index, which a browser reads as map tiles without PMTiles.
"""

import tempfile
from collections.abc import Mapping
from pathlib import Path

from upath import UPath

from ocr.risks.fire.utils import copy_or_upload, get_temp_dir

# Buildings kept in the busiest tile at each zoom, up to and including the
# key's zoom. One fraction is kept everywhere at a zoom, so relative density
# reads the same at every scale.
DEFAULT_TILE_BUDGETS: Mapping[int, int] = {
    5: 70_000,
    7: 50_000,
    10: 70_000,
    11: 120_000,
    12: 80_000,
    13: 120_000,
}
# Below this zoom the map draws only a uniform sample of the buildings a `shown`
# predicate selects, as a dot layer would; from it, every building is also
# thinned the way tippecanoe thins polygons (smallest dropped first, tiny ones
# folded into squares), so the biggest show first as the map zooms in.
POLYGONS_FROM_ZOOM = 6
# Estimated bytes of that polygon thinning in the busiest tile at each zoom.
POLYGON_TILE_BYTES = 850_000
# Every building is drawn from this zoom.
EVERY_FROM_ZOOM = 13
# Full detail from this zoom; the map overzooms it past.
COMPLETE_ZOOM = 14
# A building under a pixel draws as a square this many pixels wide.
PLACEHOLDER_PIXELS = 0.25


def display_thinning(
    shown: str,
    tile_budgets: Mapping[int, int] = DEFAULT_TILE_BUDGETS,
    polygons_from: int = POLYGONS_FROM_ZOOM,
    every_from: int = EVERY_FROM_ZOOM,
):
    """Rows each zoom draws: a uniform sample of rows matching the SQL predicate
    `shown`, sized so the busiest tile at the zoom holds its budget; from
    `polygons_from` also every row thinned smallest first; from `every_from`
    every row."""
    from geoparquet_lod import chain, combine, density_budget, drop_smallest, only, tiny_polygons

    limits = sorted(tile_budgets.items())

    def budget(zoom: int) -> int:
        return next((n for top, n in limits if zoom <= top), limits[-1][1])

    sample = only(shown, density_budget(budget))
    with_polygons = combine(sample, chain(drop_smallest(POLYGON_TILE_BYTES), tiny_polygons()))

    def thin(rows, zoom, ctx):
        if zoom >= every_from:
            return rows
        return (with_polygons if zoom >= polygons_from else sample)(rows, zoom, ctx)

    return thin


def write_buildings_lod(
    src: str,
    dest: UPath,
    *,
    shown: str,
    carry: list[str | tuple[str, str]],
    metadata: Mapping[str, str] | None = None,
    tile_budgets: Mapping[int, int] = DEFAULT_TILE_BUDGETS,
    temp_dir: Path | None = None,
    memory_limit: str | None = None,
    workers: int | None = None,
    log=print,
) -> None:
    """Write the buildings in `src` (a GeoParquet path or glob, local or S3) to
    `dest` with overviews. `shown` is a SQL predicate on the source columns for
    the buildings coarse zooms draw; every building is drawn from
    EVERY_FROM_ZOOM. `carry` names the columns tiles carry for styling, each a
    name or a (name, Arrow type) pair such as ('rps_2011', 'halffloat') to carry
    it at that precision; `metadata` is added to the file's key-value metadata."""
    from geoparquet_lod import write

    # the writer reads `src` where it is; only its output is local
    with tempfile.TemporaryDirectory(dir=get_temp_dir() if temp_dir is None else temp_dir) as tmp:
        out = Path(tmp) / 'buildings.parquet'
        write(
            str(src),
            out,
            thinning=display_thinning(shown, tile_budgets),
            carry=carry,
            zooms=list(range(COMPLETE_ZOOM + 1)),
            complete_from=COMPLETE_ZOOM,
            placeholder_pixels=PLACEHOLDER_PIXELS,
            metadata=metadata,
            memory_limit=memory_limit,
            workers=workers,
            temp_dir=tmp,
            log=log,
        )
        log(f'uploading {out} to {dest}')
        copy_or_upload(UPath(out), UPath(dest))
