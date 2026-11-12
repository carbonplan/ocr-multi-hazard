import json
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import duckdb
import geoparquet_lod
import pytest

from ocr.duck import duckdb_connect
from ocr.historic import mtbs


def ring(x0, y0, size=1.0):
    x1, y1 = x0 + size, y0 + size
    return f'{x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}'


def square(x0, y0, size=1.0):
    return f'POLYGON(({ring(x0, y0, size)}))'


def densified_square(x0, y0, size=0.001, step=1e-5, jitter=3e-7):
    n = int(size / step)
    bottom = ','.join(
        f'{x0 + i * step:.7f} {y0 + (jitter if i % 2 else 0):.8f}' for i in range(1, n)
    )
    bottom = f'{x0} {y0},{bottom}'
    return (
        f'POLYGON(({bottom},{x0 + size} {y0},{x0 + size} {y0 + size},{x0} {y0 + size},{x0} {y0}))'
    )


ROWS = {
    'normal': ('Wildfire', '2020-08-15', square(-110, 40)),
    'bowtie': ('Wildfire', '2019-06-01', 'POLYGON((-100 40,-99 41,-99 40,-100 41,-100 40))'),
    'dup_parts': (
        'Prescribed Fire',
        '2001-03-09',
        f'MULTIPOLYGON((({ring(-90, 35)})),(({ring(-90, 35)})))',
    ),
    'spike': (
        'Wildfire',
        '1999-12-31',
        'POLYGON((-100 30,-99 30,-98 30,-99 30,-99 31,-100 31,-100 30))',
    ),
    'densified': ('Wildland Fire Use', '2010-07-04', densified_square(-120, 45)),
    'straddles_edge': ('Wildfire', '2015-09-09', square(-66.5, 30)),
    'alaska': ('Wildfire', '2004-07-01', square(-150, 64)),
    'hawaii': ('Wildfire', '2012-08-01', square(-156, 20)),
}


@pytest.fixture(scope='module')
def con():
    with duckdb_connect() as con:
        values = ', '.join(
            f"('{name}', 'fire {name}', '{kind}', '{ig}', 12.5, 'Initial', ST_GeomFromText('{wkt}'))"
            for name, (kind, ig, wkt) in ROWS.items()
        )
        con.sql(
            'CREATE TABLE src AS SELECT * FROM (VALUES '
            f'{values}) t(event_id, incid_name, incid_type, ig_date, burnbndac, asmnt_type, geom)'
        )
        yield con


def test_transform_cleans_geometry_and_keeps_conus_only(con, subtests):
    rel = mtbs.perimeters(con, 'src')
    assert rel.columns == [
        'event_id',
        'incid_name',
        'incid_type',
        'ig_date',
        'year',
        'burnbndac',
        'asmnt_type',
        'geometry',
        'bbox',
    ]
    out = {
        r[0]: r
        for r in con.sql(
            """
            SELECT event_id, ig_date, year, incid_type, ST_IsValid(geometry), ST_GeometryType(geometry),
                   ST_Area(geometry), ST_NPoints(geometry), ST_XMin(geometry), ST_YMin(geometry),
                   bbox.xmin, bbox.ymax
            FROM rel
            """
        ).fetchall()
    }
    assert set(out) == {
        'normal',
        'bowtie',
        'dup_parts',
        'spike',
        'densified',
        'straddles_edge',
    }
    assert str(rel.types[rel.columns.index('ig_date')]) == 'VARCHAR'

    with subtests.test(case='all_valid_polygons'):
        for name, r in out.items():
            assert r[4] and r[5] in ('POLYGON', 'MULTIPOLYGON'), name

    with subtests.test(case='normal_keeps_lon_lat_order_and_dates'):
        _, ig_date, year, kind, *_, xmin, ymin, bbox_xmin, bbox_ymax = out['normal']
        assert (ig_date, year, kind) == ('2020-08-15', 2020, 'Wildfire')
        assert xmin == pytest.approx(-110, abs=1e-4)
        assert ymin == pytest.approx(40, abs=1e-4)
        assert bbox_xmin == pytest.approx(xmin)
        assert bbox_ymax == pytest.approx(41, abs=1e-4)

    with subtests.test(case='bowtie_becomes_two_triangles'):
        assert out['bowtie'][6] == pytest.approx(0.5, rel=1e-3)

    with subtests.test(case='dup_parts_area_is_one_part'):
        assert out['dup_parts'][6] == pytest.approx(1.0, rel=1e-3)

    with subtests.test(case='non_wildfire_types_kept'):
        assert out['dup_parts'][3] == 'Prescribed Fire'
        assert out['densified'][3] == 'Wildland Fire Use'

    with subtests.test(case='densified_vertices_stripped'):
        assert out['densified'][7] == 5

    with subtests.test(case='old_dates_keep_year'):
        assert (out['spike'][1], out['spike'][2]) == ('1999-12-31', 1999)


def test_transform_raises_on_malformed_ig_date(con, subtests):
    cases = {
        'text': ('unknown', 'invalid date field format'),
        'impossible_month': ('2020-13-45', 'out of range'),
    }
    for case, (bad, message) in cases.items():
        con.sql(
            f"CREATE OR REPLACE TABLE bad AS SELECT 'x' AS event_id, 'n' AS incid_name, 't' AS incid_type, "
            f"'{bad}' AS ig_date, 1.0 AS burnbndac, 'a' AS asmnt_type, ST_GeomFromText('{square(-110, 40)}') AS geom"
        )
        with subtests.test(case=case):
            with pytest.raises(duckdb.ConversionException, match=message):
                mtbs.perimeters(con, 'bad').fetchall()


def test_output_key_dated_from_last_modified_else_today(subtests):
    today = date(2026, 11, 4)
    cases = {
        'header': (
            'Tue, 03 Mar 2026 15:04:05 GMT',
            'input/historic/mtbs/20260303/mtbs_perims.parquet',
        ),
        'offset_rolls_to_utc': (
            'Tue, 03 Mar 2026 23:30:00 -0500',
            'input/historic/mtbs/20260304/mtbs_perims.parquet',
        ),
        'missing': (None, 'input/historic/mtbs/20261104/mtbs_perims.parquet'),
    }
    for case, (header, expected) in cases.items():
        with subtests.test(case=case):
            assert mtbs.key_for(mtbs.source_date(header, today=today)) == expected


def test_ocr_version_marked_unknown_on_scm_fallback(monkeypatch, subtests):
    versions = {
        'scm_hash': ('1.1.0.post31+ga48e41f94.d20261105', '1.1.0.post31+ga48e41f94.d20261105'),
        'scm_fallback': ('999', 'unknown (999)'),
    }
    for case, (version, expected) in versions.items():
        monkeypatch.setattr(mtbs.importlib_metadata, 'version', lambda name, v=version: v)
        with subtests.test(case=case):
            assert mtbs.ocr_version() == expected


def test_metadata_records_provenance():
    meta = mtbs.source_metadata(
        'https://example.org/x.zip',
        last_modified='Tue, 03 Mar 2026 15:04:05 GMT',
        sha256='abc',
        retrieved=date(2026, 11, 4),
        raw_mirror='input/historic/mtbs/20260303/raw/abcdef0123456789/mtbs_perimeter_data.zip',
    )
    assert meta == {
        'mtbs_url': 'https://example.org/x.zip',
        'mtbs_last_modified': 'Tue, 03 Mar 2026 15:04:05 GMT',
        'mtbs_sha256': 'abc',
        'mtbs_raw_mirror': 'input/historic/mtbs/20260303/raw/abcdef0123456789/mtbs_perimeter_data.zip',
        'mtbs_retrieved': '2026-11-04',
        'mtbs_attribution': 'MTBS (USGS/USFS)',
        'mtbs_license': 'Public domain (US Government work)',
    }
    missing = mtbs.source_metadata(
        'u', last_modified=None, sha256='abc', retrieved=date(2026, 1, 1), raw_mirror='r'
    )
    assert missing['mtbs_last_modified'] == ''


@pytest.fixture
def shapefile_zip(con, tmp_path):
    shp_dir = tmp_path / 'shp'
    shp_dir.mkdir()
    con.sql(
        f"""
        COPY (SELECT * FROM src) TO '{shp_dir}/mtbs_perims_DD.shp'
        (FORMAT GDAL, DRIVER 'ESRI Shapefile')
        """
    )
    zip_path = tmp_path / 'mtbs_perimeter_data.zip'
    with zipfile.ZipFile(zip_path, 'w') as z:
        for f in shp_dir.iterdir():
            z.write(f, f.name)
    return zip_path


def test_write_local_builds_geoparquet_and_overviews_from_zipped_shapefile(shapefile_zip, tmp_path):
    out_dir = tmp_path / 'out'
    meta = {'mtbs_sha256': 'deadbeef'}
    lod = mtbs.write_local(shapefile_zip, out_dir, metadata=meta)

    with duckdb_connect() as con:
        n_lod, ids = con.execute(
            'SELECT count(*), list(DISTINCT event_id ORDER BY event_id) FROM read_parquet($p)',
            {'p': str(lod)},
        ).fetchone()
        kv = dict(
            con.execute(
                'SELECT decode(key), decode(value) FROM parquet_kv_metadata($p)', {'p': str(lod)}
            ).fetchall()
        )
        types = con.execute(
            'SELECT DISTINCT ST_GeometryType(geometry) FROM read_parquet($p)', {'p': str(lod)}
        ).fetchall()

    assert ids == sorted(['normal', 'bowtie', 'dup_parts', 'spike', 'densified', 'straddles_edge'])
    assert n_lod == 6
    lod_meta = json.loads(kv['geo'])[geoparquet_lod.LOD_KEY]
    assert {t for (t,) in types} <= {'POLYGON', 'MULTIPOLYGON'}
    assert kv['mtbs_sha256'] == 'deadbeef'
    assert {k: v for k, v in kv.items() if k.startswith('mtbs_') and k != 'mtbs_sha256'} == {
        'mtbs_ocr_version': mtbs.ocr_version(),
        'mtbs_simplify_tolerance': '1e-06',
        'mtbs_complete_from': str(mtbs.COMPLETE_FROM),
    }
    assert [a['name'] for a in lod_meta['attributes']] == ['year']
    assert lod_meta['thinning']['name'].startswith('drop_smallest(')
    assert lod_meta['zooms'][-1] == 11


SHA = 'abcdef0123456789ffffffff'
RAW = 'input/historic/mtbs/20260303/raw/abcdef0123456789/mtbs_perimeter_data.zip'
OUT = 'input/historic/mtbs/20260303/mtbs_perims.parquet'
BUILD = 'build'


def test_raw_key_is_content_addressed_per_date(subtests):
    day = date(2026, 3, 3)
    assert mtbs.raw_key(day, SHA) == RAW
    with subtests.test(case='other_bytes_same_date_get_another_key'):
        assert mtbs.raw_key(day, 'ffffffffffffffff00') != mtbs.raw_key(day, SHA)
    with subtests.test(case='other_date_same_bytes_get_another_key'):
        assert mtbs.raw_key(date(2026, 3, 4), SHA) != RAW


@pytest.mark.parametrize(
    ('case', 'stored', 'built_sha', 'overwrite', 'build_fails', 'events', 'message'),
    [
        ('absent', set(), None, False, False, [BUILD, RAW, OUT], 'wrote'),
        ('output_present_mirror_missing_sha_matches', {OUT}, SHA, False, False, [RAW], 'skip'),
        (
            'output_present_mirror_missing_sha_differs',
            {OUT},
            'old',
            False,
            False,
            [],
            'upstream file changed since',
        ),
        ('output_and_mirror_present', {OUT, RAW}, SHA, False, False, [], 'skip'),
        ('mirror_present_output_absent', {RAW}, None, False, False, [BUILD, OUT], 'wrote'),
        ('overwrite_keeps_existing_mirror', {OUT, RAW}, SHA, True, False, [BUILD, OUT], 'wrote'),
        ('overwrite_uploads_missing_mirror', {OUT}, 'old', True, False, [BUILD, RAW, OUT], 'wrote'),
        ('build_failure_uploads_nothing', {OUT}, 'old', True, True, [BUILD], ''),
    ],
)
def test_main_mirrors_zip_after_build_and_never_overwrites_mirror(
    monkeypatch, capsys, case, stored, built_sha, overwrite, build_fails, events, message
):
    written_meta = {}

    def download(url, dest):
        dest.write_bytes(b'x' * 10)
        return SHA, 'Tue, 03 Mar 2026 15:04:05 GMT'

    def head(store, key):
        if key not in stored:
            raise FileNotFoundError(key)
        return {'size': 1}

    def write_local(zip_path, out_dir, *, metadata, temp_dir):
        log.append(BUILD)
        if build_fails:
            raise RuntimeError('boom')
        written_meta.update(metadata)
        return Path(out_dir) / 'mtbs_perims.parquet'

    log = []
    monkeypatch.setattr(mtbs, 'download', download)
    monkeypatch.setattr(mtbs, 'write_local', write_local)
    monkeypatch.setattr(mtbs, 'built_from_sha256', lambda key: built_sha)
    monkeypatch.setattr(mtbs.obstore, 'head', head)
    monkeypatch.setattr(mtbs.obstore, 'put', lambda store, key, path: log.append(key))
    monkeypatch.setattr(mtbs, 'StorageConfig', lambda: SimpleNamespace(object_store=lambda: None))

    if build_fails:
        with pytest.raises(RuntimeError, match='boom'):
            mtbs.main(overwrite=overwrite)
    else:
        mtbs.main(overwrite=overwrite)
        out = capsys.readouterr().out
        assert message in out, case
        assert ('wrote' in out) == (OUT in events), case
    assert log == events, case
    if OUT in events:
        assert written_meta['mtbs_raw_mirror'] == RAW, case
