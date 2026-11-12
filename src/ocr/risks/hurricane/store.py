"""Where the hurricane pipeline's CHAZ files live: S3 keys under S3_ROOT, cached locally.

Every file has a key under S3_ROOT (s3://carbonplan-ocr/input/hurricane/chaz) and is cached in
CACHE under the same key; `path` downloads a key on first use and, once per process, replaces a
cached copy that S3 has since overwritten. Jobs upload what they write with `push`, which records
each file's size, sha256 and source in S3_ROOT/manifest.json (or, from a parallel Coiled job, in a
fragment that `merge_manifest` folds in), and refuses to overwrite a newer copy on S3.
CHAZ_OFFLINE=1 reads the cache without checking it against S3.

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


def _mtime(info: dict) -> float:
    """Modification time of a remote object, from S3's LastModified or a local filesystem."""
    t = info.get('LastModified', info.get('mtime'))
    return t.timestamp() if hasattr(t, 'timestamp') else float(t)


def _version(info: dict):
    return info.get('ETag') or info.get('mtime')


_checked: set[str] = set()


def path(key: str) -> Path:
    """Local path of a key under S3_ROOT, downloaded into CACHE on first use. A cached copy is
    checked once per process and downloaded again when the S3 object differs in size or is newer.
    Safe when several processes ask for the same key at once: each downloads to its own temporary
    file and moves it into place."""
    local = CACHE / key
    if local.exists() and (OFFLINE or key in _checked):
        return local
    fs = s3()
    stale = not local.exists()
    if not stale:
        try:
            info = fs.info(f'{S3_ROOT}/{key}')
        except FileNotFoundError:
            info = None
        stale = info is not None and (
            info['size'] != local.stat().st_size or _mtime(info) > local.stat().st_mtime
        )
    if stale:
        part = _part(local)
        try:
            fs.get(f'{S3_ROOT}/{key}', str(part))
            os.replace(part, local)
        finally:
            part.unlink(missing_ok=True)
    _checked.add(key)
    return local


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


_uploaded: dict[str, str] = {}


def upload(key: str) -> None:
    """Upload one cached file now, so a killed job's retry finds it on S3. `push` still records
    it in the manifest and does not upload it again."""
    local = CACHE / key
    s3().put(str(local), f'{S3_ROOT}/{key}')
    os.utime(local)  # not older than its S3 copy, so `push` sees it as current
    _uploaded[key] = _hash(local)


def _remote_manifest(fs) -> dict:
    key = f'{S3_ROOT}/{MANIFEST}'
    return json.loads(fs.cat(key))['files'] if fs.exists(key) else {}


def push(
    prefix: str = '', fragment: str | None = None, pattern: str = '*', force: bool = False
) -> dict:
    """Upload cached files under `prefix` matching `pattern` whose checksum differs from the
    manifest (or that S3 lacks). Their entries go into manifest.json, or, from a parallel job,
    into manifests/<fragment>.json for `merge_manifest`. Writing manifest.json directly is a
    read-modify-write: do it from one place at a time, and let parallel jobs write fragments.

    A local file older than its S3 object is a stale copy, so nothing is uploaded while any
    exists unless `force`."""
    fs = s3()
    remote = _remote_manifest(fs)
    files, todo, stale = {}, [], []

    def keep(p: Path) -> bool:
        return p.is_file() and not p.name.endswith('.part') and p.name != MANIFEST

    for local in sorted(filter(keep, (CACHE / prefix).rglob(pattern))):
        key = str(local.relative_to(CACHE))
        files[key] = _entry(local, key)
        sha = files[key]['sha256']
        if _uploaded.get(key) == sha:
            continue
        try:
            info = fs.info(f'{S3_ROOT}/{key}')
        except FileNotFoundError:
            todo.append((local, key))
            continue
        if remote.get(key, {}).get('sha256') == sha and info['size'] == local.stat().st_size:
            continue
        if _mtime(info) > local.stat().st_mtime and not force:
            stale.append(key)
        todo.append((local, key))
    if stale:
        raise RuntimeError(
            f'{len(stale)} cached files are older than their S3 copies; nothing was uploaded. '
            f'Delete them from {CACHE} or pass force. First: {stale[:5]}'
        )
    for local, key in todo:
        fs.put(str(local), f'{S3_ROOT}/{key}')
        os.utime(local)
        print('pushed', key, flush=True)
    if fragment:
        with fs.open(f'{S3_ROOT}/{FRAGMENTS}/{fragment}.json', 'w') as f:
            json.dump({'files': files}, f, indent=1)
    else:
        with fs.open(f'{S3_ROOT}/{MANIFEST}', 'w') as f:
            json.dump({'files': {**remote, **files}}, f, indent=1, sort_keys=True)
    return files


def merge_manifest() -> int:
    """Fold the parallel jobs' manifest fragments into manifest.json, keeping each file's most
    recently recorded entry, then delete the merged fragments so none can later override a newer
    upload. A fragment rewritten during the merge is kept for the next one. Run it from one place
    at a time. Returns the entry count."""
    fs = s3()
    files = _remote_manifest(fs)
    seen = {}
    for frag in sorted(fs.glob(f'{S3_ROOT}/{FRAGMENTS}/*.json')):
        seen[frag] = _version(fs.info(frag))
        for key, entry in json.loads(fs.cat(frag))['files'].items():
            if entry.get('recorded', '') >= files.get(key, {}).get('recorded', ''):
                files[key] = entry
    with fs.open(f'{S3_ROOT}/{MANIFEST}', 'w') as f:
        json.dump({'files': files}, f, indent=1, sort_keys=True)
    done = [frag for frag, v in seen.items() if _version(fs.info(frag)) == v]
    if done:
        fs.rm(done)
    return len(files)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    pu = sub.add_parser('push', help='upload the whole cache where it differs from S3')
    pu.add_argument('--force', action='store_true', help='upload cached files older than S3')
    sub.add_parser('manifest', help="fold the parallel jobs' fragments into manifest.json")
    sub.add_parser('fetch-tutorial', help="the CHAZ tutorial's Zenodo record, then push it")
    a = p.parse_args()
    if a.cmd == 'push':
        push(force=a.force)
    elif a.cmd == 'fetch-tutorial':
        fetch_tutorial()
        push(TUTORIAL, fragment='tutorial')
    else:
        print(merge_manifest(), 'files in manifest')


if __name__ == '__main__':
    main()
