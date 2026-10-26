import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from upath import UPath

from ocr.lod import COMPLETE_ZOOM, EVERY_FROM_ZOOM, POLYGONS_FROM_ZOOM, write_buildings_lod

GEO = {
    'version': '1.1.0',
    'primary_column': 'geometry',
    'columns': {'geometry': {'encoding': 'WKB', 'geometry_types': ['Polygon']}},
}


@pytest.fixture
def buildings(tmp_path):
    rng = np.random.default_rng(3)
    n = 3000
    x = -105.4 + rng.random(n) * 0.3
    y = 39.9 + rng.random(n) * 0.2
    boxes = shapely.box(x, y, x + 1e-4, y + 1e-4)
    risk = rng.random(n).astype(np.float32)
    risk[: n // 3] = 0
    t = pa.table(
        {
            'GEOID': pa.array([f'08013{i:010d}' for i in range(n)]),
            'rps_2011': pa.array(risk),
            'rps_2047': pa.array(risk * 1.5),
            'bp_2011': pa.array(rng.random(n).astype(np.float32)),
            'geometry': pa.array(shapely.to_wkb(boxes), pa.binary()),
        }
    ).replace_schema_metadata({'geo': json.dumps(GEO)})
    path = tmp_path / 'region' / 'y1_x1.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(t, path)
    return path


def test_buildings_lod_keeps_every_building(buildings, tmp_path):
    from geoparquet_lod import validate

    dest = UPath(tmp_path / 'out' / 'buildings.parquet')
    write_buildings_lod(
        str(buildings.parent / '*.parquet'),
        dest,
        shown='rps_2011 > 0 AND rps_2047 > 0',
        carry=['rps_2011', 'rps_2047'],
        metadata={'license_name': 'ODBL'},
        workers=1,
        log=lambda *_: None,
    )
    validate(str(dest))

    out = pq.ParquetFile(dest.path)
    src = pq.read_table(buildings)
    assert out.metadata.num_rows == src.num_rows
    assert set(src.column_names) <= set(out.schema_arrow.names)
    assert out.schema_arrow.metadata[b'license_name'] == b'ODBL'

    lod = json.loads(out.schema_arrow.metadata[b'geo'])['lod_columns']
    assert max(lod['zooms']) == COMPLETE_ZOOM
    # the coarsest zooms draw only risky buildings. At z0 one
    # tile covers everything, so a row is non-null only where it is drawn.
    t = pq.read_table(dest.path, columns=['rps_2011', 'rps_2047', 'lod'])
    coarse = next(name for name, c in lod['lods'].items() if c['zooms'] == [0])
    shown = t['lod'].combine_chunks().field(coarse).is_valid().to_numpy(zero_copy_only=False)
    risky = (t['rps_2011'].to_numpy() > 0) & (t['rps_2047'].to_numpy() > 0)
    assert shown.any()
    assert not (shown & ~risky).any()
    # zero-risk buildings join from POLYGONS_FROM_ZOOM, and all are drawn from EVERY_FROM_ZOOM
    lods = t['lod'].combine_chunks()

    def drawn(zoom):
        name = next(n for n, c in lod['lods'].items() if zoom in c['zooms'])
        return lods.field(name).is_valid().to_numpy(zero_copy_only=False)

    assert not any((drawn(z) & ~risky).any() for z in range(POLYGONS_FROM_ZOOM))
    assert any((drawn(z) & ~risky).any() for z in range(POLYGONS_FROM_ZOOM, EVERY_FROM_ZOOM))
    assert drawn(EVERY_FROM_ZOOM).all()
