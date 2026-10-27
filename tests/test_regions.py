import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from upath import UPath

from ocr.regions import LEVELS, write_region_files

GEO = {
    'version': '1.0.0',
    'primary_column': 'geometry',
    'columns': {'geometry': {'encoding': 'WKB', 'geometry_types': ['Polygon']}},
}


def _level(geoids: list[str]) -> pa.Table:
    rng = np.random.default_rng(len(geoids))
    boxes = [shapely.box(-100 + i, 30, -99.5 + i, 30.5) for i in range(len(geoids))]
    t = pa.table(
        {
            'GEOID': pa.array(geoids),
            'building_count': pa.array(rng.integers(1, 1000, len(geoids))),
            'rps_2011_mean': pa.array(rng.random(len(geoids))),
            'rps_2011_median': pa.array(rng.random(len(geoids)).astype(np.float32)),
            'risk_score_2011_hist': pa.array(
                [rng.random(11).tolist() for _ in geoids], pa.list_(pa.float64())
            ),
            'geometry': pa.array(shapely.to_wkb(boxes), pa.binary()),
        }
    )
    return t.replace_schema_metadata({'geo': json.dumps(GEO)})


@pytest.fixture
def sources(tmp_path):
    geoids = {
        'nation': ['1'],
        'states': ['08', '06'],
        'counties': ['08013', '06001', '06037'],
        'tracts': ['08013012204', '06037101110', '06001400100', '06037101122'],
        'blocks': [f'0603710111{i:04d}' for i in range(70)] + ['080130122041000'],
    }
    paths = {}
    for level in LEVELS:
        # shuffled, so the rewrite has to order rows itself
        t = _level(geoids[level])
        t = t.take(np.random.default_rng(1).permutation(t.num_rows))
        path = tmp_path / 'src' / f'{level}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, path)
        paths[level] = UPath(path)
    return paths


def test_region_files_keep_every_row_and_column(sources, tmp_path):
    dest = UPath(tmp_path / 'regions')
    write_region_files(
        sources,
        dest,
        shown=['rps_2011_median'],
        metadata={'license_name': 'ODBL'},
        log=lambda *_: None,
    )

    for level in ('nation', 'states'):
        out = pq.read_table(tmp_path / 'regions' / f'{level}.parquet')
        src = pq.read_table(sources[level].path)
        assert sorted(out['GEOID'].to_pylist()) == sorted(src['GEOID'].to_pylist())
        assert set(src.column_names) <= set(out.column_names)
        assert out['GEOID'].to_pylist() == sorted(out['GEOID'].to_pylist())

    for level in ('counties', 'tracts', 'blocks'):
        files = sorted((tmp_path / 'regions' / level).rglob('data.parquet'))
        out = pa.concat_tables(pq.read_table(f) for f in files)
        src = pq.read_table(sources[level].path)
        assert sorted(out['GEOID'].to_pylist()) == sorted(src['GEOID'].to_pylist())
        assert set(src.column_names) <= set(out.column_names)
        for f in files:
            parts = {p.split('=')[0]: p.split('=')[1] for p in f.parent.parts if '=' in p}
            prefix = parts['state_fips'] + parts.get('county_fips', '')
            geoids = pq.read_table(f, columns=['GEOID'])['GEOID'].to_pylist()
            assert all(g.startswith(prefix) for g in geoids)
            assert geoids == sorted(geoids)


def test_region_files_are_laid_out_for_lookups(sources, tmp_path):
    write_region_files(
        sources, UPath(tmp_path / 'regions'), shown=['rps_2011_median'], log=lambda *_: None
    )

    counties = pq.ParquetFile(tmp_path / 'regions' / 'counties' / 'state_fips=06' / 'data.parquet')
    # a file per state, one county per page
    assert counties.metadata.num_rows == 2
    names = counties.schema_arrow.names
    assert names[:3] == ['GEOID', 'building_count', 'rps_2011_median']
    assert names[-2:] == ['bbox', 'geometry']
    geo = json.loads(counties.schema_arrow.metadata[b'geo'])
    assert geo['version'] == '1.1.0'
    assert geo['columns']['geometry']['covering']['bbox']['xmin'] == ['bbox', 'xmin']
    chunk = counties.metadata.row_group(0).column(0)
    assert chunk.path_in_schema == 'GEOID'
    assert chunk.statistics is not None and chunk.statistics.has_min_max

    blocks = pq.ParquetFile(
        tmp_path / 'regions' / 'blocks' / 'state_fips=06' / 'county_fips=037' / 'data.parquet'
    )
    assert blocks.metadata.num_rows == 70
    # every column breaks pages at the same rows: 32 per page
    index = blocks.metadata.row_group(0).column(0)
    assert index.has_offset_index
