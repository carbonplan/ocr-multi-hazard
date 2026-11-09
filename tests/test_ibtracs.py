import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from ocr.duck import duckdb_connect
from ocr.historic import ibtracs

# (sid, basin, wind kt, sshs, R34 quadrants NE/SE/SW/NW in nmi, segment start lon/lat, hours after
# 2005-08-29 12:00)
ROWS = [
    ('observed', 'NA', 25, -1, None, (-30, 15), 0),
    ('observed', 'NA', 100, 3, (50, 100, 80, 60), (-90, 27), 3),
    # far from CONUS, and 9 h from the storm's radius
    ('observed', 'NA', 40, 0, None, (-30, 16), 12),
    ('observed_peer', 'NA', 105, 3, (120, 120, 120, 120), (-91, 27), 0),
    # a gap 3 h after one radius and 4 h before another
    ('gap', 'NA', 110, 3, (60, 60, 60, 60), (-89, 28), 0),
    ('gap', 'NA', 112, 3, None, (-88.5, 28), 3),
    ('gap', 'NA', 105, 3, (100, 100, 100, 100), (-88, 28), 7),
    ('depression', 'NA', 30, -1, None, (-90, 29), 0),
    ('offshore_reaches', 'NA', 60, 0, (150, 150, 150, 150), (-64, 35), 0),
    ('offshore_short', 'NA', 60, 0, (30, 30, 30, 30), (-60, 30), 0),
    # 0.97 and 1.1 R34 east of CONUS at 45N, where a degree of longitude is 78.85 km.
    ('just_inside', 'NA', 60, 0, (100, 100, 100, 100), (-66 + 0.97 * 185.2 / 78.85, 45), 0),
    ('just_outside', 'NA', 60, 0, (100, 100, 100, 100), (-66 + 1.1 * 185.2 / 78.85, 45), 0),
    ('west_pacific', 'WP', 100, 3, (100, 100, 100, 100), (140, 20), 0),
    ('before_radii', 'NA', 100, 3, (150, 150, 150, 150), (-90, 28), 0),
]
SEASONS = {'before_radii': ibtracs.FIRST_SEASON - 1}
STORMS_REACHING_CONUS = ['gap', 'just_inside', 'observed', 'observed_peer', 'offshore_reaches']


def source_sql(rows):
    values = []
    for sid, basin, wind, sshs, r34, (x, y), hours in rows:
        quadrants = ', '.join(str(q) for q in r34) if r34 else 'NULL, NULL, NULL, NULL'
        time = datetime(2005, 8, 29, 12) + timedelta(hours=hours)
        values.append(
            f"('{sid}', 'NAME {sid}', {SEASONS.get(sid, 2005)}, '{basin}', '{time:%Y-%m-%d %H:%M:%S}', "
            f"'TS', {wind}, {sshs}, "
            f"{quadrants}, ST_GeomFromText('LINESTRING({x} {y}, {x + 0.5} {y})'))"
        )
    return (
        f'SELECT * FROM (VALUES {", ".join(values)}) t(SID, NAME, SEASON, BASIN, ISO_TIME, NATURE, '
        'USA_WIND, USA_SSHS, USA_R34_NE, USA_R34_SE, USA_R34_SW, USA_R34_NW, geom)'
    )


@pytest.fixture(scope='module')
def con():
    with duckdb_connect() as con:
        con.sql(f'CREATE TABLE src AS {source_sql(ROWS)}')
        yield con


def test_tracks_keep_every_segment_of_storms_whose_winds_reach_conus(con, subtests):
    rel = ibtracs.tracks(con, 'src')
    assert rel.columns == [
        'sid',
        'name',
        'season',
        'basin',
        'iso_time',
        'nature',
        'usa_wind',
        'usa_sshs',
        'r34_nmi',
        'r34_source',
        'geometry',
        'bbox',
    ]
    rows = con.sql(
        'SELECT sid, usa_wind, r34_nmi, r34_source, ST_GeometryType(geometry), bbox.xmin '
        'FROM rel ORDER BY sid, usa_wind'
    ).fetchall()
    out = {(r[0], r[1]): r for r in rows}
    assert sorted({r[0] for r in rows}) == STORMS_REACHING_CONUS

    with subtests.test(case='centerlines'):
        assert {r[4] for r in rows} == {'LINESTRING'}

    with subtests.test(case='weak_segments_kept_without_r34'):
        assert out[('observed', 25)][2:] == (None, None, 'LINESTRING', -30)

    with subtests.test(case='observed_takes_largest_quadrant'):
        assert out[('observed', 100)][2:4] == (100, 'observed')

    with subtests.test(case='missing_radii_take_the_storms_nearest_in_time'):
        assert out[('gap', 112)][2:4] == (60, 'nearest')

    with subtests.test(case='missing_radii_far_from_conus_and_from_a_radius_stay_empty'):
        assert out[('observed', 40)][2:4] == (None, None)


@pytest.mark.parametrize(
    'rows',
    [
        pytest.param(
            [
                ('lonely', 'NA', 140, 5, None, (-90, 27), 0),
                ('neighbour', 'NA', 140, 5, (100, 100, 100, 100), (-90, 27), 0),
            ],
            id='only_another_storm_has_a_radius',
        ),
        pytest.param(
            [
                ('lonely', 'NA', 60, 0, (80, 80, 80, 80), (-90, 27), 0),
                ('lonely', 'NA', 60, 0, None, (-89.5, 27), 7),
            ],
            id='its_radius_is_too_far_in_time',
        ),
        # 16 degrees east of CONUS at 45N is ~680 nmi, within the largest radius in the source.
        pytest.param(
            [
                ('lonely', 'NA', 60, 0, None, (-50, 45), 0),
                ('large', 'NA', 60, 0, (720, 720, 720, 720), (-30, 20), 0),
            ],
            id='beyond_15_degrees_but_within_the_largest_radius',
        ),
    ],
)
def test_tracks_raise_on_a_gap_near_conus_with_no_radius_to_take(rows):
    with duckdb_connect() as con:
        con.sql(f'CREATE TABLE src AS {source_sql(rows)}')
        with pytest.raises(ValueError, match=r"\('lonely', '2005-08-29"):
            ibtracs.tracks(con, 'src')


def test_output_key_dated_from_last_modified_else_today(subtests):
    today = date(2026, 11, 4)
    cases = {
        'header': ('Tue, 03 Mar 2026 15:04:05 GMT', '20260303'),
        'offset_rolls_to_utc': ('Tue, 03 Mar 2026 23:30:00 -0500', '20260304'),
        'missing': (None, '20261104'),
    }
    for case, (header, day) in cases.items():
        with subtests.test(case=case):
            assert (
                ibtracs.key_for(ibtracs.source_date(header, today=today))
                == f'input/historic/ibtracs/{day}/ibtracs_tracks.parquet'
            )


@pytest.fixture
def shapefile_zip(con, tmp_path):
    shp_dir = tmp_path / 'shp'
    shp_dir.mkdir()
    con.sql(
        f"""
        COPY (SELECT * FROM src) TO '{shp_dir}/{ibtracs.SHAPEFILE}'
        (FORMAT GDAL, DRIVER 'ESRI Shapefile')
        """
    )
    zip_path = tmp_path / ibtracs.ZIP_NAME
    with zipfile.ZipFile(zip_path, 'w') as z:
        for f in shp_dir.iterdir():
            z.write(f, f.name)
    return zip_path


def test_write_local_builds_geoparquet_sorted_by_storm_from_zipped_shapefile(
    shapefile_zip, tmp_path
):
    out = ibtracs.write_local(shapefile_zip, tmp_path / 'out', metadata={'ibtracs_sha256': 'beef'})

    with duckdb_connect() as con:
        sids, sized = con.execute(
            'SELECT list(sid), count(r34_nmi) FROM read_parquet($p)', {'p': str(out)}
        ).fetchall()[0]
        kv = dict(
            con.execute(
                'SELECT decode(key), decode(value) FROM parquet_kv_metadata($p)', {'p': str(out)}
            ).fetchall()
        )

    assert sids == sorted(sids) and sorted(set(sids)) == STORMS_REACHING_CONUS
    assert (len(sids), sized) == (9, 7)
    assert kv['ibtracs_max_r34_nmi'] == '150.0'
    assert kv['ibtracs_sha256'] == 'beef'
    assert kv['ibtracs_ocr_version']
    assert 'lod_columns' not in kv['geo']
    meta = pq.ParquetFile(out).metadata
    assert meta.num_row_groups == 1
    assert meta.row_group(0).column(0).compression == 'ZSTD'


SHA = 'abcdef0123456789ffffffff'
RAW = f'input/historic/ibtracs/20260303/raw/abcdef0123456789/{ibtracs.ZIP_NAME}'
OUT = 'input/historic/ibtracs/20260303/ibtracs_tracks.parquet'
BUILD = 'build'


@pytest.mark.parametrize(
    ('case', 'stored', 'built_sha', 'overwrite', 'build_fails', 'events', 'message'),
    [
        ('absent', set(), None, False, False, [BUILD, RAW, OUT], 'wrote'),
        ('output_present_mirror_missing_sha_matches', {OUT}, SHA, False, False, [RAW], 'skip'),
        ('output_present_sha_differs', {OUT}, 'old', False, False, [], 'upstream file changed'),
        ('output_and_mirror_present', {OUT, RAW}, SHA, False, False, [], 'skip'),
        ('overwrite_keeps_existing_mirror', {OUT, RAW}, SHA, True, False, [BUILD, OUT], 'wrote'),
        ('build_failure_uploads_nothing', {OUT}, 'old', True, True, [BUILD], ''),
    ],
)
def test_main_mirrors_zip_after_build_and_never_overwrites_mirror(
    monkeypatch, capsys, case, stored, built_sha, overwrite, build_fails, events, message
):
    log = []

    def download(url, dest):
        dest.write_bytes(b'x')
        return SHA, 'Tue, 03 Mar 2026 15:04:05 GMT'

    def head(store, key):
        if key not in stored:
            raise FileNotFoundError(key)

    def write_local(zip_path, out_dir, *, metadata):
        log.append(BUILD)
        if build_fails:
            raise RuntimeError('boom')
        assert metadata['ibtracs_raw_mirror'] == RAW
        return Path(out_dir) / 'ibtracs_tracks.parquet'

    monkeypatch.setattr(ibtracs, 'download', download)
    monkeypatch.setattr(ibtracs, 'write_local', write_local)
    monkeypatch.setattr(ibtracs, 'built_from_sha256', lambda key: built_sha)
    monkeypatch.setattr(ibtracs.obstore, 'head', head)
    monkeypatch.setattr(ibtracs.obstore, 'put', lambda store, key, path: log.append(key))
    monkeypatch.setattr(
        ibtracs, 'StorageConfig', lambda: SimpleNamespace(object_store=lambda: None)
    )

    if build_fails:
        with pytest.raises(RuntimeError, match='boom'):
            ibtracs.main(overwrite=overwrite)
    else:
        ibtracs.main(overwrite=overwrite)
        assert message in capsys.readouterr().out, case
    assert log == events, case
