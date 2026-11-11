"""Keys, download helpers and the Coiled launcher shared by the FDP refit scripts (fdp_inputs,
fdp_points, fdp_train).

Source: Collins et al. (2022), "Predicting flood damage probability across the
conterminous United States", USGS ScienceBase doi:10.5066/P954TTQN.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests
import s3fs

BUCKET = 'carbonplan-ocr'
REFIT = 'input/flood/fdp-refit'  # training inputs and model runs, read by ocr.risks.flood.refit
# The published surface, as ocr.risks.flood.etl.flood reads it; fdp_train compares each refit with it.
PUBLISHED_ZIP = 'Output_CONUS_FDP_100m.zip'
PUBLISHED_KEY = f'{BUCKET}/input/flood/raw/usgs-fdp/CONUS_FDP_100m.tif'
PROJ4 = (
    '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 '
    '+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'
)
SB_ITEM = '6170694ed34ea36449a67ef7'
CACHE = Path(os.environ.get('FDP_CACHE', Path.home() / '.cache/ocr/fdp'))
HERE = Path(__file__).resolve().parent
# What a Coiled VM fetches to run any of the scripts (`launch`).
CODE = (
    'fdp.py',
    'fdp_inputs.py',
    'fdp_points.py',
    'fdp_train.py',
    'train_fdp.R',
    'contrib_fdp.R',
    'fdp_contrib.cpp',
    'se_fdp.R',
)


def sciencebase_url(name: str) -> str:
    return f'https://www.sciencebase.gov/catalog/file/get/{SB_ITEM}?name={name}'


def download(url: str, dest: Path) -> None:
    """Stream a ScienceBase file to `dest`.

    The endpoint ignores Range headers and streams the whole file, so there is
    no way to pull just the tif member: the zip lands on disk in full.
    """
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


def stage_code() -> str:
    """Copy the scripts to S3 under a hash of their contents; returns the prefix."""
    digest = hashlib.sha256(b''.join((HERE / name).read_bytes() for name in CODE)).hexdigest()
    prefix = f'{BUCKET}/{REFIT}/code/{digest[:12]}'
    s3 = s3fs.S3FileSystem()
    for name in CODE:
        s3.put_file(str(HERE / name), f'{prefix}/{name}')
    return prefix


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
    """Run `python <script...>` on a Coiled VM (x86, us-west-2, next to the bucket), from the
    scripts as they are now (`stage_code`); with `map_over`, one VM per value, which the script
    reads from $COILED_BATCH_TASK_INPUT. The VM uses the Coiled workspace's AWS access."""
    cmd = [
        'coiled',
        'batch',
        'run',
        '--name',
        name,
        '--region',
        'us-west-2',
        '--vm-type',
        vm_type,
        '--disk-size',
        str(disk_size),
        '--tag',
        'Project=OCR',
        '--spot-policy',
        'spot_with_fallback' if spot else 'on-demand',
    ]
    if map_over:
        cmd += ['--map-over-values', ','.join(map_over)]
    for pkg in conda_extras:
        cmd += ['--package-sync-conda-extras', pkg]
    cmd += ['--', 'bash', str(HERE / 'fdp_task.sh'), stage_code(), *script]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def vm_type() -> str | None:
    """This machine's EC2 instance type, or None off EC2 (IMDSv2)."""
    imds = 'http://169.254.169.254/latest'
    try:
        token = requests.put(
            f'{imds}/api/token', headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'}, timeout=1
        ).text
        r = requests.get(
            f'{imds}/meta-data/instance-type',
            headers={'X-aws-ec2-metadata-token': token},
            timeout=1,
        )
        return r.text if r.ok else None
    except requests.RequestException:
        return None
