import json
import os
import time
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
    monkeypatch.setattr(store, '_checked', set())
    monkeypatch.setattr(store, 'OFFLINE', False)
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


def age(file, seconds):
    t = time.time() - seconds
    os.utime(file, (t, t))


def test_path_replaces_a_cached_copy_that_s3_has_overwritten(bucket):
    key = 'forcing/era5/2000/I_MLR_2000.nc'
    local = store.CACHE / key
    local.parent.mkdir(parents=True)
    local.write_bytes(b'old')
    age(local, 3600)
    bucket.pipe(remote(key), b'new')
    assert store.path(key).read_bytes() == b'new'


def test_path_keeps_a_cached_copy_that_is_current_or_only_local(bucket):
    bucket.pipe(remote('a.nc'), b'same')
    (store.CACHE / 'a.nc').parent.mkdir(parents=True)
    (store.CACHE / 'a.nc').write_bytes(b'same')
    (store.CACHE / 'b.nc').write_bytes(b'local only')
    assert store.path('a.nc').read_bytes() == b'same'
    assert store.path('b.nc').read_bytes() == b'local only'


def test_push_refuses_to_overwrite_a_newer_s3_copy(bucket):
    key = 'forcing/era5/2017/PI_2017.nc'
    local = store.CACHE / key
    local.parent.mkdir(parents=True)
    local.write_bytes(b'built on the laptop')
    age(local, 3600)
    bucket.pipe(remote(key), b'rebuilt on Coiled')
    with pytest.raises(RuntimeError, match='older than their S3 copies'):
        store.push('forcing/era5/2017')
    assert bucket.cat(remote(key)) == b'rebuilt on Coiled'
    store.push('forcing/era5/2017', force=True)
    assert bucket.cat(remote(key)) == b'built on the laptop'


def test_push_after_upload_does_not_see_a_stale_copy(bucket, monkeypatch):
    key = 'inputs/era5/2000/sl_025.nc'
    (store.CACHE / key).parent.mkdir(parents=True)
    (store.CACHE / key).write_bytes(b'a')
    store.upload(key)
    monkeypatch.setattr(store, '_uploaded', {})
    store.push('inputs/era5/2000')
    assert bucket.cat(remote(key)) == b'a'


def test_merge_keeps_a_fragment_rewritten_during_the_merge(bucket, monkeypatch):
    frag = remote('manifests/run-2000.json')
    entry = {'sha256': 'x', 'recorded': '2026-01-01T00:00:00+00:00'}
    bucket.pipe(frag, json.dumps({'files': {'a': entry}}).encode())
    cat = bucket.cat

    def cat_then_rewrite(path, *a, **k):
        out = cat(path, *a, **k)
        if path == frag:
            time.sleep(0.01)
            bucket.pipe(frag, json.dumps({'files': {'b': entry}}).encode())
        return out

    monkeypatch.setattr(bucket, 'cat', cat_then_rewrite)
    store.merge_manifest()
    assert bucket.exists(frag)
    monkeypatch.setattr(bucket, 'cat', cat)
    assert store.merge_manifest() == 2
    assert not bucket.exists(frag)
