import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import duckdb
import geoparquet_lod
import pytest

from ocr.duck import duckdb_connect
from ocr.historic import usfd

HEADER = 'ID,DATE_BEGIN,DATE_END,LON,LAT,STATE,LOCATION,FATALITY,DAMAGE,CAUSE,SOURCE_DB,SOURCE_ID,DESCRIPTION,year'
# One CSV line per case; the line number is the expected id.
ROWS = {
    'fedb_14_digits': '7,20020130013000.0,20020201000000.0,-72.5,41.7,CONNECTICUT,ST. LOUIS COUNTY,,,Heavy rain,FEDB,01184000,,',
    'noaa_12_digits': '7,199610211400,199610211400,-96.35,35.85,OKLAHOMA,BRYAN,2.0,1500.0,,NOAA Storm report,5570362,"a bridge, washed out",1996.0',
    'noaa_nat': '8,NaT,199610211400,-96.0,35.0,OKLAHOMA,BRYAN,,,,NOAA Storm report,5570363,,1996.0',
    'dfo_8_digits': '9,19850222.0,19850301.0,-90.0,30.0,,,,,,DFO,1,,1985.0',
    'emdat_year_only': '10,2005,,-91.0,31.0,,,,,,EM-DAT,2005-0001,,2005.0',
    'year_column_wrong': '11,20050601120000.0,20050602000000.0,-80.0,35.0,,,,,,FEDB,02000000,,1903.0',
    'alaska': '12,20100101000000.0,,-150.0,64.0,ALASKA,,,,,mPing,,,2010.0',
    'north_of_conus': '13,20100101000000.0,,-100.0,50.3,,,,,,GFM,,,2010.0',
    'no_coords': '14,199601241700,199601241700,,,INDIANA,FAYETTE,,,,NOAA Storm report,5536059,,1996.0',
    'after_dropped_rows': '15,20140501000000.0,,-100.0,40.0,,,,,,mPing,,,2014.0',
}


def write_csv(path, rows):
    path.write_text('\n'.join([HEADER, *rows]) + '\n')
    return path


@pytest.fixture
def csv_path(tmp_path):
    return write_csv(tmp_path / 'USFD_v1.1.csv', ROWS.values())


def test_points_match_old_pmtiles_fields_with_year_from_date(csv_path):
    with duckdb_connect() as con:
        rel = usfd.points(con, csv_path)
        assert rel.columns == [
            'id',
            'source',
            'source_id',
            'begin_date',
            'end_date',
            'year',
            'state',
            'location',
            'damage',
            'fatalities',
            'cause',
            'geometry',
            'bbox',
        ]
        rows = con.sql(
            'SELECT * EXCLUDE (geometry, bbox), ST_X(geometry), ST_Y(geometry), bbox.xmin FROM rel ORDER BY id'
        ).fetchall()
    d = date
    assert rows == [
        (0, 'FEDB', '01184000', d(2002, 1, 30), d(2002, 2, 1), 2002, 'Connecticut', 'St. Louis County', None, None, 'Heavy rain', -72.5, 41.7, -72.5),
        (1, 'NOAA Storm report', '5570362', d(1996, 10, 21), d(1996, 10, 21), 1996, 'Oklahoma', 'Bryan', 1500.0, 2, None, -96.35, 35.85, -96.35),
        (2, 'NOAA Storm report', '5570363', None, d(1996, 10, 21), 1996, 'Oklahoma', 'Bryan', None, None, None, -96.0, 35.0, -96.0),
        (3, 'DFO', '1', d(1985, 2, 22), d(1985, 3, 1), 1985, None, None, None, None, None, -90.0, 30.0, -90.0),
        (4, 'EM-DAT', '2005-0001', None, None, 2005, None, None, None, None, None, -91.0, 31.0, -91.0),
        (5, 'FEDB', '02000000', d(2005, 6, 1), d(2005, 6, 2), 2005, None, None, None, None, None, -80.0, 35.0, -80.0),
        (9, 'mPing', None, d(2014, 5, 1), None, 2014, None, None, None, None, None, -100.0, 40.0, -100.0),
    ]  # fmt: skip


def test_points_raise_on_digits_that_are_no_date(tmp_path):
    bad = ROWS['dfo_8_digits'].replace('19850222.0', '19851345.0')
    with duckdb_connect() as con:
        with pytest.raises(duckdb.InvalidInputException, match='Could not parse string "19851345"'):
            usfd.points(con, write_csv(tmp_path / 'bad.csv', [bad])).fetchall()


def test_write_local_builds_geoparquet_and_overviews_with_provenance(csv_path, tmp_path):
    meta = usfd.source_metadata(sha256='abc', retrieved=date(2026, 11, 12))
    out = usfd.write_local(csv_path, tmp_path / 'out', metadata=meta)

    with duckdb_connect() as con:
        ids = con.execute(
            'SELECT list(id ORDER BY id) FROM read_parquet($p)', {'p': str(out)}
        ).fetchone()[0]
        kv = dict(
            con.execute(
                'SELECT decode(key), decode(value) FROM parquet_kv_metadata($p)', {'p': str(out)}
            ).fetchall()
        )
    lod_meta = json.loads(kv['geo'])[geoparquet_lod.LOD_KEY]
    assert ids == [0, 1, 2, 3, 4, 5, 9]
    assert {k: v for k, v in kv.items() if k.startswith('usfd_')} == {
        'usfd_url': 'https://zenodo.org/records/7545697/files/USFD_v1.1.csv',
        'usfd_doi': '10.5281/zenodo.7545697',
        'usfd_version': 'v1.1',
        'usfd_md5': 'f91b3447c900c39fd0a5bf44dc8edf11',
        'usfd_sha256': 'abc',
        'usfd_raw_mirror': 'input/historic/usfd/v1.1/raw/USFD_v1.1.csv',
        'usfd_retrieved': '2026-11-12',
        'usfd_attribution': 'United States Flood Database v1.1, Li et al. 2021',
        'usfd_license': 'CC-BY-4.0',
        'usfd_ocr_version': usfd.ocr_version(),
        'usfd_complete_from': '10',
    }
    assert [a['name'] for a in lod_meta['attributes']] == ['source', 'year']
    assert lod_meta['thinning']['name'].startswith('density_budget(')
    assert lod_meta['zooms'][-1] == 10


OUT, RAW, BUILD, DOWNLOAD = usfd.KEY, usfd.RAW_KEY, 'build', 'download'


@pytest.mark.parametrize(
    ('case', 'stored', 'overwrite', 'md5', 'build_fails', 'events', 'message'),
    [
        ('absent', set(), False, usfd.MD5, False, [DOWNLOAD, BUILD, RAW, OUT], 'wrote'),
        ('output_and_mirror_present', {OUT, RAW}, False, usfd.MD5, False, [], 'skip'),
        ('output_present_mirror_missing', {OUT}, False, usfd.MD5, False, [DOWNLOAD, RAW], 'skip'),
        ('mirror_present_output_absent', {RAW}, False, usfd.MD5, False, [DOWNLOAD, BUILD, OUT], 'wrote'),
        ('overwrite_keeps_existing_mirror', {OUT, RAW}, True, usfd.MD5, False, [DOWNLOAD, BUILD, OUT], 'wrote'),
        ('md5_mismatch_uploads_nothing', set(), False, 'other', False, [DOWNLOAD], 'zenodo publishes'),
        ('build_failure_uploads_nothing', set(), False, usfd.MD5, True, [DOWNLOAD, BUILD], 'boom'),
    ],
)  # fmt: skip
def test_main_builds_once_and_mirrors_csv_after_build(
    monkeypatch, capsys, case, stored, overwrite, md5, build_fails, events, message
):
    log = []

    def download(url, dest):
        log.append(DOWNLOAD)
        dest.write_bytes(b'x')
        return md5, 'sha'

    def head(store, key):
        if key not in stored:
            raise FileNotFoundError(key)
        return {'size': 1}

    def write_local(csv_path, out_dir, *, metadata, temp_dir):
        log.append(BUILD)
        if build_fails:
            raise RuntimeError('boom')
        assert metadata['usfd_sha256'] == 'sha', case
        return Path(out_dir) / 'usfd_floods.parquet'

    monkeypatch.setattr(usfd, 'download', download)
    monkeypatch.setattr(usfd, 'write_local', write_local)
    monkeypatch.setattr(usfd.obstore, 'head', head)
    monkeypatch.setattr(usfd.obstore, 'put', lambda store, key, path: log.append(key))
    monkeypatch.setattr(usfd, 'StorageConfig', lambda: SimpleNamespace(object_store=lambda: None))

    if OUT in events or message == 'skip':
        usfd.main(overwrite=overwrite)
        assert message in capsys.readouterr().out, case
    else:
        with pytest.raises((ValueError, RuntimeError), match=message):
            usfd.main(overwrite=overwrite)
    assert log == events, case
