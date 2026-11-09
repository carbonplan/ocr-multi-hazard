"""Keys and download helpers shared by the FDP refit scripts (fdp_inputs, fdp_points, fdp_train).

Source: Collins et al. (2022), "Predicting flood damage probability across the
conterminous United States", USGS ScienceBase doi:10.5066/P954TTQN.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

BUCKET = 'carbonplan-ocr'
REFIT = 'input/flood/fdp-refit'  # training inputs and model runs, read by ocr.risks.flood.refit
# The published surface as a zarr pyramid; fdp_train compares each refit with level 0.
PUBLISHED = 'ocr-explore/FDP/processed/flood_damage_probability'
VARIABLE = 'fdp'
PROJ4 = (
    '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 '
    '+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'
)
SB_ITEM = '6170694ed34ea36449a67ef7'
CACHE = Path(os.environ.get('FDP_CACHE', Path.home() / '.cache/ocr/fdp'))


def store_uri() -> str:
    return f's3://{BUCKET}/{PUBLISHED}'


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
