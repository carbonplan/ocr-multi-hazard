import json
import threading
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
    assert len(bucket.glob(remote('manifests/run-2000-*.json'))) == 1


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


def no_puts(bucket, monkeypatch):
    puts = []
    monkeypatch.setattr(bucket, 'put', lambda *a, **k: puts.append(a))
    return puts


def test_push_records_but_skips_files_already_uploaded(bucket, monkeypatch):
    raw = store.CACHE / 'inputs/era5/2000'
    raw.mkdir(parents=True)
    (raw / 'sl_025.nc').write_bytes(b'a')
    store.upload('inputs/era5/2000/sl_025.nc')
    puts = no_puts(bucket, monkeypatch)
    files = store.push('inputs/era5/2000', fragment='raw-2000')
    assert list(files) == ['inputs/era5/2000/sl_025.nc']
    assert puts == []
    assert bucket.cat(remote('inputs/era5/2000/sl_025.nc')) == b'a'


def test_push_skips_files_downloaded_on_a_fresh_machine(bucket, monkeypatch):
    bucket.pipe(remote('inputs/era5/2000/sl_025.nc'), b'a')
    store.path('inputs/era5/2000/sl_025.nc')
    puts = no_puts(bucket, monkeypatch)
    store.push('inputs/era5/2000')
    assert puts == []


def test_path_replaces_a_cached_copy_of_another_version(bucket):
    key = 'forcing/era5/2000/I_MLR_2000.nc'
    local = store.CACHE / key
    local.parent.mkdir(parents=True)
    local.write_bytes(b'old')
    bucket.pipe(remote(key), b'new')
    assert store.path(key).read_bytes() == b'new'


def test_path_checks_once_per_process_and_keeps_a_current_copy(bucket, monkeypatch):
    bucket.pipe(remote('a.nc'), b'same')
    store.path('a.nc')
    monkeypatch.setattr(bucket, 'get', lambda *a, **k: pytest.fail('downloaded again'))
    store._checked.clear()
    assert store.path('a.nc').read_bytes() == b'same'


def test_path_uses_a_file_only_in_the_cache(bucket):
    (store.CACHE / 'b.nc').parent.mkdir(parents=True)
    (store.CACHE / 'b.nc').write_bytes(b'local only')
    assert store.path('b.nc').read_bytes() == b'local only'


def test_push_refuses_to_overwrite_a_version_the_cache_has_not_seen(bucket):
    key = 'forcing/era5/2017/PI_2017.nc'
    bucket.pipe(remote(key), b'A')
    store.path(key)  # the laptop's copy
    bucket.pipe(remote(key), b'B')  # rebuilt on Coiled, same size
    with pytest.raises(RuntimeError, match='has not seen'):
        store.push('forcing/era5/2017')
    assert bucket.cat(remote(key)) == b'B'
    store.push('forcing/era5/2017', replace=True)
    assert bucket.cat(remote(key)) == b'A'


def test_push_uploads_a_file_changed_since_it_was_synced(bucket):
    key = 'forcing/era5/2017/PI_2017.nc'
    bucket.pipe(remote(key), b'A')
    store.path(key).write_bytes(b'rebuilt')
    store.push('forcing/era5/2017')
    assert bucket.cat(remote(key)) == b'rebuilt'


def test_push_writes_a_new_fragment_each_time(bucket):
    (store.CACHE / 'x').mkdir(parents=True)
    (store.CACHE / 'x' / 'a.nc').write_bytes(b'a')
    store.push('x', fragment='job')
    store.push('x', fragment='job')
    assert len(bucket.glob(remote('manifests/job-*.json'))) == 2


def test_merge_leaves_a_fragment_written_during_the_merge(bucket, monkeypatch):
    entry = {'sha256': 'x', 'recorded': '2026-01-01T00:00:00+00:00'}
    first, late = remote('manifests/a.json'), remote('manifests/b.json')
    bucket.pipe(first, json.dumps({'files': {'a': entry}}).encode())
    cat = bucket.cat

    def cat_then_another_job_writes(path, *a, **k):
        out = cat(path, *a, **k)
        if path == first:
            bucket.pipe(late, json.dumps({'files': {'b': entry}}).encode())
        return out

    monkeypatch.setattr(bucket, 'cat', cat_then_another_job_writes)
    assert store.merge_manifest() == 1
    assert not bucket.exists(first) and bucket.exists(late)
    monkeypatch.setattr(bucket, 'cat', cat)
    assert store.merge_manifest() == 2


def test_concurrent_requests_for_a_key_download_it_once(bucket, monkeypatch):
    bucket.pipe(remote('a.nc'), b'x' * 1000)
    get, calls = bucket.get, []

    def slow_get(*a, **k):
        calls.append(1)
        time.sleep(0.05)
        return get(*a, **k)

    monkeypatch.setattr(bucket, 'get', slow_get)
    threads = [threading.Thread(target=store.path, args=('a.nc',)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def stored_sha(bucket, monkeypatch, sha_of):
    """Give the local stand-in S3 object metadata: sha_of(path) is what each object stores."""
    monkeypatch.setattr(
        bucket, 'metadata', lambda path, refresh=False: {'sha256': sha_of(path)}, raising=False
    )


def test_upload_does_not_record_a_version_another_writer_replaced(bucket, monkeypatch):
    key = 'runs/era5/x/2000_r000.nc'
    (store.CACHE / key).parent.mkdir(parents=True)
    (store.CACHE / key).write_bytes(b'ours')
    stored_sha(bucket, monkeypatch, lambda path: 'theirs')
    store.upload(key)
    assert store._synced(key) is None


def test_a_cached_copy_with_the_stored_checksum_is_adopted_without_downloading(bucket, monkeypatch):
    key = 'inputs/era5/2000/big.nc'
    bucket.pipe(remote(key), b'same bytes')
    (store.CACHE / key).parent.mkdir(parents=True)
    (store.CACHE / key).write_bytes(b'same bytes')
    stored_sha(bucket, monkeypatch, lambda path: store._hash(store.CACHE / key))
    monkeypatch.setattr(bucket, 'get', lambda *a, **k: pytest.fail('downloaded'))
    assert store.path(key).read_bytes() == b'same bytes'
    puts = no_puts(bucket, monkeypatch)
    store.push('inputs/era5/2000')
    assert puts == []


def test_push_of_keys_leaves_other_cached_files_alone(bucket):
    out = store.CACHE / 'validation/chaz/landfall'
    out.mkdir(parents=True)
    (out / 'summary_TCGI_CRH_PI.csv').write_bytes(b'crh')
    (out / 'summary_TCGI_SD_PI.csv').write_bytes(b'stale sd')
    bucket.pipe(remote('validation/chaz/landfall/summary_TCGI_SD_PI.csv'), b'newer sd')
    store.push(keys=['validation/chaz/landfall/summary_TCGI_CRH_PI.csv'], replace=True)
    assert bucket.cat(remote('validation/chaz/landfall/summary_TCGI_SD_PI.csv')) == b'newer sd'
    assert bucket.cat(remote('validation/chaz/landfall/summary_TCGI_CRH_PI.csv')) == b'crh'


def test_merge_keeps_the_entry_for_the_version_s3_holds(bucket):
    key = 'forcing/era5/2000/PI_2000.nc'
    (store.CACHE / key).parent.mkdir(parents=True)
    (store.CACHE / key).write_bytes(b'new')
    store.push('forcing/era5/2000', fragment='later-upload')
    held = json.loads(bucket.cat(bucket.glob(remote('manifests/later-upload-*.json'))[0]))
    wrong_clock = {'sha256': 'old', 'recorded': '2099-01-01T00:00:00+00:00', 'version': 'gone'}
    bucket.pipe(
        remote('manifests/fast-clock.json'), json.dumps({'files': {key: wrong_clock}}).encode()
    )
    store.merge_manifest()
    merged = json.loads(bucket.cat(remote('manifest.json')))['files'][key]
    assert merged['sha256'] == held['files'][key]['sha256'] == store._hash(store.CACHE / key)
