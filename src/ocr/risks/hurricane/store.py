"""Where the hurricane pipeline's CHAZ files live: S3 keys under S3_ROOT, cached locally.

Every file has a key under S3_ROOT (s3://carbonplan-ocr/input/hurricane/chaz) and is cached in
CACHE under the same key; `path` downloads a key on first use. Jobs upload what they write with
`push`, which records each file's size, sha256 and source in S3_ROOT/manifest.json (or, from a
parallel Coiled job, in a fragment that `merge_manifest` folds in).

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
from pathlib import Path

import xarray as xr

S3_ROOT = os.environ.get('CHAZ_S3_ROOT', 's3://carbonplan-ocr/input/hurricane/chaz')
CACHE = Path(os.environ.get('CHAZ_CACHE', Path.home() / '.cache' / 'ocr' / 'chaz'))
ERA5 = 'inputs/era5'
TUTORIAL = 'inputs/zenodo-11631930'
FORCING = 'forcing/era5'
RUNS = 'runs/era5'
VALIDATION = 'validation/tcgi'
MANIFEST = 'manifest.json'
FRAGMENTS = 'manifests'

SOURCES = {
    ERA5: 'ERA5 from the CDS (forcing.py fetch; the request is beside each file)',
    TUTORIAL: 'CHAZ tutorial data, Zenodo 11631930 (CC BY 4.0)',
    FORCING: 'CHAZ forcing from ERA5 (forcing.py build)',
    RUNS: 'CHAZ tracks (chaz.py)',
    f'{VALIDATION}/era5-201708-6h': 'ERA5 6-hourly T, August 2017, 2-degree grid (tcgi.py fetch-validation)',
    f'{VALIDATION}/compare_2017.nc': 'derived: tcgi.py compare',
    f'{VALIDATION}/seeding_1981_2005': 'derived: tcgi.py seeding',
    f'{VALIDATION}/chaz_port_2017_storms.csv': 'our CHAZ tutorial port on ERA5 2017, 40 track '
    'realizations at ratio 1.13: seeds and storms kept (Shane98c/chaz-local chaz_downscale.py)',
}


def s3():
    import s3fs

    return s3fs.S3FileSystem()


def path(key: str) -> Path:
    """Local path of a key under S3_ROOT, downloaded into CACHE on first use."""
    local = CACHE / key
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + '.part')
        s3().get(f'{S3_ROOT}/{key}', str(part))
        part.rename(local)
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


def _entry(local: Path, key: str) -> dict:
    h = hashlib.sha256()
    with open(local, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    source = next((v for k, v in SOURCES.items() if key.startswith(k)), '')
    return {'bytes': local.stat().st_size, 'sha256': h.hexdigest(), 'source': source}


def _remote_manifest(fs) -> dict:
    key = f'{S3_ROOT}/{MANIFEST}'
    return json.loads(fs.cat(key))['files'] if fs.exists(key) else {}


def push(prefix: str = '', fragment: str | None = None) -> dict:
    """Upload cached files under `prefix` whose checksum differs from the manifest (or that S3
    lacks). Their entries are merged into manifest.json, or, from a parallel job, written to
    manifests/<fragment>.json for `merge_manifest`."""
    fs = s3()
    remote = _remote_manifest(fs)
    files = {}

    def keep(p: Path) -> bool:
        return p.is_file() and not p.name.endswith('.part') and p.name != MANIFEST

    for local in sorted(filter(keep, (CACHE / prefix).rglob('*'))):
        key = str(local.relative_to(CACHE))
        files[key] = _entry(local, key)
        stale = remote.get(key, {}).get('sha256') != files[key]['sha256']
        if stale or not fs.exists(f'{S3_ROOT}/{key}'):
            fs.put(str(local), f'{S3_ROOT}/{key}')
            print('pushed', key, flush=True)
    if fragment:
        with fs.open(f'{S3_ROOT}/{FRAGMENTS}/{fragment}.json', 'w') as f:
            json.dump({'files': files}, f, indent=1)
    else:
        with fs.open(f'{S3_ROOT}/{MANIFEST}', 'w') as f:
            json.dump({'files': {**remote, **files}}, f, indent=1, sort_keys=True)
    return files


def merge_manifest() -> int:
    """Fold the parallel jobs' manifest fragments into manifest.json; returns the entry count."""
    fs = s3()
    files = _remote_manifest(fs)
    for frag in sorted(fs.glob(f'{S3_ROOT}/{FRAGMENTS}/*.json')):
        files.update(json.loads(fs.cat(frag))['files'])
    with fs.open(f'{S3_ROOT}/{MANIFEST}', 'w') as f:
        json.dump({'files': files}, f, indent=1, sort_keys=True)
    return len(files)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('push', help='upload the whole cache where it differs from S3')
    sub.add_parser('manifest', help="fold the parallel jobs' fragments into manifest.json")
    a = p.parse_args()
    if a.cmd == 'push':
        push()
    else:
        print(merge_manifest(), 'files in manifest')


if __name__ == '__main__':
    main()
