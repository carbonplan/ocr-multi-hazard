import json
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pytest

from ocr.risks.hurricane import batch, store


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """store against a local directory standing in for S3, with a fresh cache."""
    fs = fsspec.filesystem('file', auto_mkdir=True)
    monkeypatch.setattr(store, 'S3_ROOT', str(tmp_path / 'root'))
    monkeypatch.setattr(store, 'CACHE', tmp_path / 'cache')
    monkeypatch.setattr(store, 's3', lambda: fs)
    monkeypatch.setattr(store, '_uploaded', {})
    return fs


def remote(key):
    return f'{store.S3_ROOT}/{key}'


def test_path_survives_concurrent_downloads(bucket):
    bucket.pipe(remote('forcing/era5/2000/A_200001.nc'), b'x' * 100_000)
    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(lambda _: store.path('forcing/era5/2000/A_200001.nc'), range(16)))
    assert {p.read_bytes() for p in paths} == {b'x' * 100_000}
    assert not list(paths[0].parent.glob('*.part'))


def test_push_pattern_and_fragment(bucket):
    runs = store.CACHE / 'runs/era5/TCGI_CRH_PI'
    runs.mkdir(parents=True)
    (runs / '2000_r000.nc').write_bytes(b'a')
    (runs / '2001_r000.nc').write_bytes(b'b')
    files = store.push('runs/era5/TCGI_CRH_PI', fragment='run-2000', pattern='2000_*')
    assert list(files) == ['runs/era5/TCGI_CRH_PI/2000_r000.nc']
    assert not bucket.exists(remote('runs/era5/TCGI_CRH_PI/2001_r000.nc'))
    assert bucket.exists(remote('manifests/run-2000.json'))


def test_merge_keeps_newest_entry_and_consumes_fragments(bucket):
    old = {'k': {'sha256': 'old', 'recorded': '2026-01-01T00:00:00+00:00'}}
    new = {'k': {'sha256': 'new', 'recorded': '2026-02-01T00:00:00+00:00'}}
    bucket.pipe(remote('manifest.json'), json.dumps({'files': new}).encode())
    bucket.pipe(remote('manifests/a.json'), json.dumps({'files': old}).encode())
    assert store.merge_manifest() == 1
    merged = json.loads(bucket.cat(remote('manifest.json')))['files']
    assert merged['k']['sha256'] == 'new'
    assert not bucket.glob(remote('manifests/*.json'))


@pytest.mark.parametrize(
    ('task', 'bad'),
    [
        ({'state': 'done', 'exit_code': 0}, False),
        ({'state': 'done', 'exit_code': 137}, True),
        ({'state': 'error', 'exit_code': None}, True),
        ({'state': 'running', 'exit_code': None}, True),
    ],
)
def test_failed_tasks(task, bad):
    assert batch.failed(task) is bad


def test_push_records_but_skips_files_already_uploaded(bucket, monkeypatch):
    raw = store.CACHE / 'inputs/era5/2000'
    raw.mkdir(parents=True)
    (raw / 'sl_025.nc').write_bytes(b'a')
    store.upload('inputs/era5/2000/sl_025.nc')
    puts = []
    monkeypatch.setattr(bucket, 'put', lambda *a, **k: puts.append(a))
    files = store.push('inputs/era5/2000', fragment='raw-2000')
    assert list(files) == ['inputs/era5/2000/sl_025.nc']
    assert puts == []
    assert bucket.cat(remote('inputs/era5/2000/sl_025.nc')) == b'a'
