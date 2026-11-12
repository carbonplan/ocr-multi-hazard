"""Where the hurricane pipeline's CHAZ files live: S3 keys under S3_ROOT, cached locally.

Every file has a key under S3_ROOT (s3://carbonplan-ocr/input/hurricane/chaz) and is cached in
CACHE under the same key. Beside the cache, CACHE/.sync/{key}.json records which S3 version
(ETag) and content (sha256) each cached file had when it was last downloaded or uploaded:

  - `path` downloads a key on first use and, once per process, again when S3 holds a version the
    cached copy did not come from
  - `push` uploads cached files that S3 lacks or that changed locally since they were synced,
    and refuses to overwrite a version on S3 the cache has not seen, unless the caller is the
    source of those files (`replace`). It records each file's size, sha256 and source in
    S3_ROOT/manifest.json, or, from a parallel job, in a fragment that `merge_manifest` folds in.

A file that changes on S3 while a job is reading it is not handled. CHAZ_OFFLINE=1 reads the
cache without checking it against S3.

    inputs/era5/{year}/           ERA5 from the CDS, with each request beside its file
    inputs/zenodo-11631930/       CHAZ tutorial data (CC BY 4.0)
    forcing/era5/{year}/          CHAZ forcing built from ERA5 (forcing.py)
    runs/era5/{TCGI}/             CHAZ tracks (chaz.py)
    validation/                   files the review notebooks read
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import xarray as xr

S3_ROOT = os.environ.get('CHAZ_S3_ROOT', 's3://carbonplan-ocr/input/hurricane/chaz')
CACHE = Path(os.environ.get('CHAZ_CACHE', Path.home() / '.cache' / 'ocr' / 'chaz'))
OFFLINE = os.environ.get('CHAZ_OFFLINE', '') not in ('', '0')
ERA5 = 'inputs/era5'
TUTORIAL = 'inputs/zenodo-11631930'
FORCING = 'forcing/era5'
RUNS = 'runs/era5'
VALIDATION = 'validation/tcgi'
MANIFEST = 'manifest.json'
SYNC = '.sync'
PUBLISHED_TRACKS = os.environ.get(
    'CHAZ_PUBLISHED_TRACKS', 's3://carbonplan-ocr/ocr-explore/CHAZ/tracks/ERA5'
)  # the ten ERA5 track files Meiler shared, Sept 2026
FRAGMENTS = 'manifests'

SOURCES = {
    ERA5: 'ERA5 from the CDS (forcing.py fetch; the request is beside each file)',
    TUTORIAL: 'CHAZ tutorial data, Zenodo 11631930 (CC BY 4.0)',
    FORCING: 'CHAZ forcing from ERA5 (forcing.py build)',
    RUNS: 'CHAZ tracks (chaz.py)',
    f'{VALIDATION}/era5-201708-6h': 'ERA5 6-hourly T, August 2017, 2-degree grid (tcgi.py fetch-validation)',
    f'{VALIDATION}/compare_2017.nc': 'derived: tcgi.py compare',
    f'{VALIDATION}/seeding_1981_2005': 'derived: tcgi.py seeding',
    'validation/forcing/uv_201701': 'ERA5 hourly and 6-hourly 250/850 hPa winds, January 2017, '
    '2-degree grid (forcing.py fetch-validation)',
    f'{VALIDATION}/chaz_port_2017_storms.csv': 'our CHAZ tutorial port on ERA5 2017, 40 track '
    'realizations at ratio 1.13: seeds and storms kept (Shane98c/chaz-local chaz_downscale.py)',
}


def s3():
    import s3fs

    return s3fs.S3FileSystem()


def _part(local: Path) -> Path:
    """A temporary file beside `local`, unique to this process, for writing then replacing."""
    local.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=local.parent, prefix=local.name + '.', suffix='.part')
    os.close(fd)
    return Path(name)


def _version(info: dict) -> str:
    """An object's version: its S3 ETag, or mtime and size on a filesystem without one."""
    etag = info.get('ETag')
    return etag.strip('"') if etag else f'{info.get("mtime")}:{info["size"]}'


def _info(fs, key: str) -> dict | None:
    try:
        return fs.info(f'{S3_ROOT}/{key}', refresh=True)
    except FileNotFoundError:
        return None


def _synced(key: str) -> dict | None:
    f = CACHE / SYNC / f'{key}.json'
    return json.loads(f.read_text()) if f.exists() else None


def _record(key: str, info: dict, sha: str) -> None:
    f = CACHE / SYNC / f'{key}.json'
    part = _part(f)
    part.write_text(json.dumps({'version': _version(info), 'sha256': sha}))
    os.replace(part, f)


def _same_content(local: Path, info: dict) -> bool:
    """Whether a file matches an S3 object whose ETag is its MD5 (a single-part upload)."""
    etag = (info.get('ETag') or '').strip('"')
    return bool(etag) and '-' not in etag and _hash(local, 'md5') == etag


_checked: set[str] = set()


def path(key: str) -> Path:
    """Local path of a key under S3_ROOT, downloaded into CACHE on first use. Once per process a
    cached copy is checked against S3 and downloaded again unless it came from the version S3
    holds. A file only in the cache is used as it is. Safe when several processes ask for the
    same key at once: each downloads to its own temporary file and moves it into place."""
    local = CACHE / key
    if local.exists() and (OFFLINE or key in _checked):
        return local
    fs = s3()
    info = _info(fs, key)
    if info is None:
        if not local.exists():
            raise FileNotFoundError(f'{S3_ROOT}/{key}')
    elif local.exists() and (_synced(key) or {}).get('version') == _version(info):
        pass
    elif local.exists() and _same_content(local, info):
        _record(key, info, _hash(local))
    else:
        _download(fs, key, info)
    _checked.add(key)
    return local


def _download(fs, key: str, info: dict) -> None:
    local = CACHE / key
    for _ in range(3):
        part = _part(local)
        try:
            fs.get(f'{S3_ROOT}/{key}', str(part))
            after = _info(fs, key)
            if after is None:
                raise FileNotFoundError(f'{S3_ROOT}/{key}')
            if _version(after) == _version(info):
                os.replace(part, local)
                _record(key, info, _hash(local))
                return
            info = after
        finally:
            part.unlink(missing_ok=True)
    raise RuntimeError(f'{key} kept changing on S3 while it downloaded')


def on_s3(key: str) -> bool:
    return s3().exists(f'{S3_ROOT}/{key}')


def open_era5(file: Path) -> xr.Dataset:
    """A CDS netCDF with the dimension names the rest of the pipeline uses."""
    ds = xr.open_dataset(file)
    names = {'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time', 'pressure_level': 'level'}
    ds = ds.rename({k: v for k, v in names.items() if k in ds.dims or k in ds.coords})
    return ds.drop_vars([v for v in ('number', 'expver') if v in ds.coords], errors='ignore')


def tutorial(name: str) -> xr.Dataset:
    return xr.open_dataset(path(f'{TUTORIAL}/{name}'), decode_times=False)


ZENODO_RECORD = 11631930


def fetch_tutorial() -> list[str]:
    """Every file of the CHAZ tutorial's Zenodo record into the cache, checked against Zenodo's
    md5s. These are the reference files the review notebooks compare with, and the intensity
    model's regressions and best-track predictors."""
    import requests

    files = requests.get(f'https://zenodo.org/api/records/{ZENODO_RECORD}', timeout=60).json()[
        'files'
    ]
    got = []
    for f in files:
        dest = CACHE / TUTORIAL / f['key']
        algo, digest = f['checksum'].split(':', 1)
        if not (dest.exists() and _hash(dest, algo) == digest):
            part = _part(dest)
            try:
                with (
                    requests.get(f['links']['self'], stream=True, timeout=600) as r,
                    open(part, 'wb') as out,
                ):
                    r.raise_for_status()
                    for block in r.iter_content(1 << 20):
                        out.write(block)
                if (got_digest := _hash(part, algo)) != digest:
                    raise ValueError(f'{f["key"]}: {algo} {got_digest} != {digest}')
                os.replace(part, dest)
            finally:
                part.unlink(missing_ok=True)
        got.append(f'{TUTORIAL}/{f["key"]}')
    return got


def _hash(file: Path, algo: str = 'sha256') -> str:
    h = hashlib.new(algo)
    with open(file, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _entry(local: Path, key: str) -> dict:
    source = next((v for k, v in SOURCES.items() if key.startswith(k)), '')
    return {
        'bytes': local.stat().st_size,
        'sha256': _hash(local),
        'source': source,
        'recorded': datetime.now(UTC).isoformat(),
    }


def upload(key: str) -> None:
    """Upload one cached file now, so a killed job's retry finds it on S3. `push` still records
    it in the manifest and does not upload it again."""
    _put(s3(), key)


def _put(fs, key: str) -> None:
    local = CACHE / key
    fs.put(str(local), f'{S3_ROOT}/{key}')
    _record(key, _info(fs, key), _hash(local))


def _remote_manifest(fs) -> dict:
    key = f'{S3_ROOT}/{MANIFEST}'
    return json.loads(fs.cat(key))['files'] if fs.exists(key) else {}


def push(
    prefix: str = '', fragment: str | None = None, pattern: str = '*', replace: bool = False
) -> dict:
    """Upload cached files under `prefix` matching `pattern` that S3 lacks or that changed since
    they were synced. Their entries go into manifest.json, or, from a parallel job, into a new
    fragment manifests/<fragment>-<time>-<id>.json for `merge_manifest`. Writing manifest.json
    directly is a read-modify-write: do it from one place at a time, and let parallel jobs write
    fragments.

    A cached file whose S3 object is a version the cache has not seen (and has other content)
    may be a stale copy, so nothing is uploaded while any exists. A job that makes the files
    passes `replace` to overwrite whatever S3 holds."""
    fs = s3()
    files, todo, unseen = {}, [], []

    def keep(p: Path) -> bool:
        if not p.is_file() or p.name.endswith('.part') or p.name == MANIFEST:
            return False
        return p.relative_to(CACHE).parts[0] != SYNC

    for local in sorted(filter(keep, (CACHE / prefix).rglob(pattern))):
        key = str(local.relative_to(CACHE))
        files[key] = _entry(local, key)
        sha = files[key]['sha256']
        info = _info(fs, key)
        synced = _synced(key)
        if info is None:
            todo.append(key)
        elif synced is not None and synced['version'] == _version(info):
            if synced['sha256'] != sha:
                todo.append(key)
        elif _same_content(local, info):
            _record(key, info, sha)
        else:
            if not replace:
                unseen.append(key)
            todo.append(key)
    if unseen:
        raise RuntimeError(
            f'S3 holds versions of {len(unseen)} cached files that the cache has not seen; '
            f'nothing was uploaded. Remove them from {CACHE} to fetch them again, or replace '
            f'what S3 holds. First: {unseen[:5]}'
        )
    for key in todo:
        _put(fs, key)
        print('pushed', key, flush=True)
    if fragment:
        stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S')
        name = f'{fragment}-{stamp}-{uuid.uuid4().hex[:8]}.json'
        with fs.open(f'{S3_ROOT}/{FRAGMENTS}/{name}', 'w') as f:
            json.dump({'files': files}, f, indent=1)
    else:
        remote = _remote_manifest(fs)
        with fs.open(f'{S3_ROOT}/{MANIFEST}', 'w') as f:
            json.dump({'files': {**remote, **files}}, f, indent=1, sort_keys=True)
    return files


def merge_manifest() -> int:
    """Fold the parallel jobs' manifest fragments into manifest.json, keeping each file's most
    recently recorded entry, then delete the fragments it read. Fragments are never rewritten, so
    one written during the merge waits for the next. Run it from one place at a time. Returns the
    entry count."""
    fs = s3()
    files = _remote_manifest(fs)
    fs.invalidate_cache(f'{S3_ROOT}/{FRAGMENTS}')
    frags = sorted(fs.glob(f'{S3_ROOT}/{FRAGMENTS}/*.json'))
    for frag in frags:
        for key, entry in json.loads(fs.cat(frag))['files'].items():
            if entry.get('recorded', '') >= files.get(key, {}).get('recorded', ''):
                files[key] = entry
    with fs.open(f'{S3_ROOT}/{MANIFEST}', 'w') as f:
        json.dump({'files': files}, f, indent=1, sort_keys=True)
    if frags:
        fs.rm(frags)
    return len(files)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    pu = sub.add_parser('push', help='upload the whole cache where it differs from S3')
    pu.add_argument(
        '--replace', action='store_true', help='overwrite versions on S3 the cache has not seen'
    )
    sub.add_parser('manifest', help="fold the parallel jobs' fragments into manifest.json")
    sub.add_parser('fetch-tutorial', help="the CHAZ tutorial's Zenodo record, then push it")
    a = p.parse_args()
    if a.cmd == 'push':
        push(replace=a.replace)
    elif a.cmd == 'fetch-tutorial':
        fetch_tutorial()
        push(TUTORIAL, fragment='tutorial', replace=True)
    else:
        print(merge_manifest(), 'files in manifest')


if __name__ == '__main__':
    main()
