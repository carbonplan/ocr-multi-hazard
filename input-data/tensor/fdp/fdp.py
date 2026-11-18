"""Keys, download helpers and the Coiled launcher shared by the FDP refit scripts (fdp_inputs,
fdp_points, fdp_train).

Source: Collins et al. (2022), "Predicting flood damage probability across the
conterminous United States", USGS ScienceBase doi:10.5066/P954TTQN.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import zipfile
from datetime import timedelta
from functools import cache
from pathlib import Path

import obstore
import requests
from obstore.auth.boto3 import Boto3CredentialProvider
from obstore.store import S3Store
from upath import UPath

ROOT = UPath('s3://carbonplan-ocr')
REFIT = (
    ROOT / 'input/flood/fdp-refit'
)  # training inputs and model runs, read by ocr.risks.flood.refit
# The published surface; fdp_train compares each refit with it.
PUBLISHED_ZIP = 'Output_CONUS_FDP_100m.zip'
PUBLISHED = ROOT / 'input/flood/raw/usgs-fdp/CONUS_FDP_100m.tif'
PROJ4 = (
    '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 '
    '+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'
)
SB_ITEM = '6170694ed34ea36449a67ef7'
VM_TYPE = 'FDP_VM_TYPE'  # set by `launch` on its VMs
POLL_SECONDS = 60
CACHE = Path(os.environ.get('FDP_CACHE', Path.home() / '.cache/ocr/fdp'))


def sciencebase_url(name: str) -> str:
    return f'https://www.sciencebase.gov/catalog/file/get/{SB_ITEM}?name={name}'


def download(url: str, dest: Path) -> None:
    """Stream a ScienceBase file to `dest`, unless it is already there.

    The endpoint ignores Range headers and streams the whole file, so there is
    no way to pull just the tif member: the zip lands on disk in full. It
    arrives as `<dest>.part` and is renamed when complete.
    """
    if dest.exists():
        print(f'  downloaded: {dest}')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    print(f'  downloading {dest.name}')
    t0 = time.time()
    # A captured log has no carriage return, so tick every ~320 MB there instead
    # of redrawing one line every chunk.
    tty = sys.stdout.isatty()
    with requests.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        got = 0
        with part.open('wb') as fh:
            for i, chunk in enumerate(r.iter_content(chunk_size=32 * 1024 * 1024)):
                fh.write(chunk)
                got += len(chunk)
                if tty:
                    print(f'    {got / 1e9:5.2f} GB', end='\r', flush=True)
                elif i % 10 == 0:
                    print(f'    {got / 1e9:5.2f} GB', flush=True)
    part.rename(dest)
    dt = time.time() - t0
    print(f'    {got / 1e9:.2f} GB in {dt:.0f}s ({got / 1e6 / (dt + 1e-9):.1f} MB/s)')


@cache
def store() -> S3Store:
    """ROOT's bucket; boto3's credential chain covers SSO profiles on laptops. obstore asks for
    credentials again every 5 min, and botocore refreshes any within 10 min of expiring, so
    obstore gets fresh ones before the old ones expire."""
    provider = Boto3CredentialProvider(ttl=timedelta(minutes=5))
    return S3Store(ROOT.drive, region='us-west-2', credential_provider=provider)


def key(path: UPath) -> str:
    return str(path.relative_to(ROOT))


def exists(path: UPath) -> bool:
    try:
        obstore.head(store(), key(path))
    except FileNotFoundError:
        return False
    return True


def read_bytes(path: UPath) -> bytes:
    return bytes(obstore.get(store(), key(path)).bytes())


def get_file(path: UPath, dest: Path) -> None:
    """Stream `path` to `dest`, by way of `<dest>.part`."""
    part = dest.with_suffix(dest.suffix + '.part')
    with part.open('wb') as fh:
        for chunk in obstore.get(store(), key(path)).stream():
            fh.write(chunk)
    part.rename(dest)


def put_file(local: Path, path: UPath) -> None:
    obstore.put(store(), key(path), local)


def put_bytes(path: UPath, data: bytes) -> None:
    obstore.put(store(), key(path), data)


def delete(path: UPath) -> None:
    obstore.delete(store(), key(path))


def has_objects(path: UPath) -> bool:
    """Whether any object is under `path`/."""
    return any(batch for batch in obstore.list(store(), prefix=f'{key(path)}/', chunk_size=1))


def delete_prefix(path: UPath) -> None:
    """Delete every object under `path`/."""
    for batch in obstore.list(store(), prefix=f'{key(path)}/'):
        obstore.delete(store(), [obj['path'] for obj in batch])


def extract_tif(zip_path: Path, dest: Path) -> None:
    """Extract the one GeoTIFF in `zip_path` to `dest`."""
    with zipfile.ZipFile(zip_path) as zf:
        tifs = [
            n
            for n in zf.namelist()
            if n.lower().endswith('.tif') and not Path(n).name.startswith('._')
        ]
        if len(tifs) != 1:
            sys.exit(f'expected exactly one .tif in {zip_path.name}, found {tifs}')
        with zf.open(tifs[0]) as src, dest.open('wb') as out:
            shutil.copyfileobj(src, out, length=32 * 1024 * 1024)


def launch(
    name: str,
    script: list[str],
    *,
    vm_type: str,
    disk_size: int = 200,
    spot: bool = False,
    map_over: list[str] | None = None,
    conda_extras: list[str] | tuple[str, ...] = (),
) -> None:
    """Run `python -m <module> <args...>` (`script`) on a Coiled VM (x86, us-west-2, next to the
    bucket) and wait for every task, exiting non-zero if any fails; with `map_over`, one VM per
    value, which the module reads from $COILED_BATCH_TASK_INPUT. Package sync ships these scripts
    as they are now, installed in the `fdp` environment as the fdp-refit package. The VM uses the
    Coiled workspace's AWS access. Each job's cluster is named `name` plus its start time, so a
    retry never shares a name with a cluster still running."""
    import coiled

    job = coiled.batch.run(
        command=['python', '-m', *script],
        name=f'{name}-{time.strftime("%Y%m%d-%H%M%S")}',
        region='us-west-2',
        vm_type=[vm_type],
        # a one-task job runs its task on the scheduler VM
        scheduler_vm_type=[vm_type] if len(map_over or [None]) == 1 else None,
        disk_size=str(disk_size),
        tag={'Project': 'OCR'},
        spot_policy='spot_with_fallback' if spot else 'on-demand',
        map_over_values=map_over,
        package_sync_conda_extras=list(conda_extras) or None,
        env={VM_TYPE: vm_type},
    )
    cluster = job['cluster_id']
    logs = f'coiled batch logs {cluster}'
    print(f'{name}: cluster {cluster}; logs: {logs}', flush=True)
    try:
        while (failed := job_failures(coiled.batch.status(cluster))) is None:
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        sys.exit(
            f'stopped waiting; the job keeps running (stop it with `coiled cluster stop {cluster}`)'
        )
    if failed:
        sys.exit(f'{name}: {failed} failed; see {logs}')
    print(f'{name}: done', flush=True)


def job_failures(jobs: list[dict]) -> str | None:
    """None while any task can still run; once every task has finished or the cluster has
    stopped, '' if every task succeeded, else how many did not."""
    tasks = [t for j in jobs for t in j['tasks']]
    finished = bool(tasks) and all(t['exit_code'] or t['state'] in ('done', 'error') for t in tasks)
    stopped = bool(jobs) and jobs[0]['cluster_state'] in ('error', 'stopped')
    if not (finished or stopped):
        return None
    bad = sum(1 for t in tasks if t['exit_code'] or t['state'] != 'done')
    if not tasks:
        return 'the cluster stopped before any task'
    return f'{bad} of {len(tasks)} tasks' if bad else ''
