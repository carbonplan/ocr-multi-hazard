"""Region GeoParquet for analysis and the map in one dataset.

Each region level keeps every source column and the original geometry; the
rewrite adds a bbox covering and lays rows out for range-request lookups:

    nation.parquet, states.parquet
    counties/state_fips=SS/data.parquet
    tracts/state_fips=SS/county_fips=CCC/data.parquet
    blocks/state_fips=SS/county_fips=CCC/data.parquet

Rows are in GEOID order with a row cap per page, so every column breaks pages
at the same rows, and GEOID and the bbox carry statistics and page indexes. A
reader finds a region by GEOID from the page statistics, or by point from the
bbox ones, and reads one page. Partitions keep each file's footer and page
indexes small, so opening a file costs one small read.
"""

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from upath import UPath

from ocr.risks.fire.utils import get_temp_dir

LEVELS = ('nation', 'states', 'counties', 'tracts', 'blocks')
PAGE_ROWS = {'nation': 1, 'states': 1, 'counties': 1, 'tracts': 8, 'blocks': 32}
PARTITIONED = {'tracts', 'blocks'}
# Columns a lookup shows come first, so its reads sit close together.
FIRST = ['GEOID', 'STUSPS', 'NAME', 'building_count']
BBOX = ('xmin', 'ymin', 'xmax', 'ymax')


def _prepare(regions: pa.Table, shown: list[str]) -> pa.Table:
    import shapely

    geoms = shapely.from_wkb(regions.column('geometry').to_numpy(zero_copy_only=False))
    bounds = shapely.bounds(geoms)
    bbox = pa.StructArray.from_arrays([pa.array(bounds[:, i]) for i in range(4)], list(BBOX))
    first = [c for c in [*FIRST, *shown] if c in regions.column_names]
    rest = [c for c in regions.column_names if c not in first and c not in ('geometry', 'bbox')]
    out = regions.select(first + rest).append_column('bbox', bbox)
    return out.append_column('geometry', regions['geometry']).sort_by('GEOID')


def _geo(source_geo: dict, regions: pa.Table) -> dict:
    column = dict(source_geo['columns']['geometry'])
    column['covering'] = {'bbox': {k: ['bbox', k] for k in BBOX}}
    column['bbox'] = [
        pc.min(pc.struct_field(regions['bbox'], 'xmin')).as_py(),
        pc.min(pc.struct_field(regions['bbox'], 'ymin')).as_py(),
        pc.max(pc.struct_field(regions['bbox'], 'xmax')).as_py(),
        pc.max(pc.struct_field(regions['bbox'], 'ymax')).as_py(),
    ]
    return {'version': '1.1.0', 'primary_column': 'geometry', 'columns': {'geometry': column}}


def _write(parts: list[pa.Table], path: Path, rows_per_page: int, source_geo: dict, metadata: dict):
    """One row group per part."""
    whole = pa.concat_tables(parts)
    schema = whole.schema.with_metadata({**metadata, 'geo': json.dumps(_geo(source_geo, whole))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(
        path,
        schema,
        compression='zstd',
        compression_level=9,
        use_dictionary=False,
        write_page_index=True,
        write_statistics=['GEOID', *(f'bbox.{k}' for k in BBOX)],
        max_rows_per_page=rows_per_page,
        write_batch_size=min(rows_per_page, 1024),
        data_page_size=1 << 20,
    ) as writer:
        for part in parts:
            writer.write_table(part.cast(schema), row_group_size=max(part.num_rows, 1))


def _split(regions: pa.Table, width: int) -> list[pa.Table]:
    """`regions` (in GEOID order) split where the first `width` GEOID characters change."""
    key = pc.utf8_slice_codeunits(regions['GEOID'], 0, width).to_numpy(zero_copy_only=False)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    return [regions.slice(s, e - s) for s, e in zip(starts, ends, strict=True)]


def _write_counties_of(regions: pa.Table, root: Path, level: str, source_geo: dict, metadata: dict):
    for county in _split(regions, 5):
        fips = county['GEOID'][0].as_py()[:5]
        path = root / level / f'state_fips={fips[:2]}' / f'county_fips={fips[2:]}' / 'data.parquet'
        _write([county], path, PAGE_ROWS[level], source_geo, metadata)


def write_region_files(
    sources: Mapping[str, UPath],
    dest: UPath,
    *,
    shown: list[str] | None = None,
    metadata: Mapping[str, str] | None = None,
    temp_dir: Path | None = None,
    log=print,
) -> None:
    """Rewrite each level's GeoParquet in `sources` (keyed by LEVELS) into
    `dest`. `shown` names the columns a lookup reads, placed right after the
    identifying ones; `metadata` is added to every file's key-value metadata."""
    shown = shown or []
    metadata = dict(metadata or {})
    with tempfile.TemporaryDirectory(dir=get_temp_dir() if temp_dir is None else temp_dir) as tmp:
        root = Path(tmp)
        for level in LEVELS:
            src = UPath(sources[level])
            dataset = pads.dataset(src.path, filesystem=src.fs, format='parquet')
            source_geo = json.loads(dataset.schema.metadata[b'geo'])
            if level == 'blocks':
                # one state at a time, so the national table is never in memory
                geoid = dataset.to_table(columns=['GEOID'])['GEOID']
                states = sorted(set(pc.utf8_slice_codeunits(geoid, 0, 2).to_pylist()))
                del geoid
                for state in states:
                    part = dataset.to_table(
                        filter=(pads.field('GEOID') >= state)
                        & (pads.field('GEOID') < state + '\uffff')
                    )
                    _write_counties_of(_prepare(part, shown), root, level, source_geo, metadata)
                log(f'{level}: {len(states)} states')
                continue
            regions = _prepare(dataset.to_table(), shown)
            if level in PARTITIONED:
                _write_counties_of(regions, root, level, source_geo, metadata)
            elif level == 'counties':
                for state in _split(regions, 2):
                    state_fips = state['GEOID'][0].as_py()[:2]
                    path = root / level / f'state_fips={state_fips}' / 'data.parquet'
                    _write([state], path, PAGE_ROWS[level], source_geo, metadata)
            else:
                _write([regions], root / f'{level}.parquet', PAGE_ROWS[level], source_geo, metadata)
            log(f'{level}: {regions.num_rows:,} rows')
        dest = UPath(dest)
        log(f'uploading region files to {dest}')
        # one put for the whole tree; on S3 it sends the per-county files concurrently
        dest.fs.put(f'{root}/', dest.path, recursive=True)
