"""Raw CHAZ netCDF transfer into input/hurricane/raw/chaz/ (bytes unchanged; encoding happens in chaz.py).

Dataset: "Global coastal wind hazard maps from the CHAZ tropical cyclone model",
doi:10.5061/dryad.qfttdz0vz (CC0). Ported from carbonplan/ocr branch Shane98c/chaz (fetch_chaz.py).

copy-raw: server-side copy of the files already fetched to ocr-explore/CHAZ/ (complete except the
exceedance_intensity raster: 3/108 per-GCM files).

fetch: stream members of a Dryad zip into RAW_PREFIX. Dryad's download endpoint 302s to a presigned,
range-readable S3 URL (valid 24 h), so the multi-GB zip is read with range requests, never downloaded.
A fresh URL is resolved per member. Members already present at the same size are skipped (resumable).
    export DRYAD_TOKEN=<token>  # Dryad "My account" > API account; copy a token there (valid 10 h)
    pixi run hurricane-fetch fetch exceedance_intensity --only '*raster.nc*' --dry-run
On Coiled: coiled batch run --vm-type m8g.large --arm --tag Project=OCR --secret-env DRYAD_TOKEN=$DRYAD_TOKEN -- python -m ocr.risks.hurricane.etl.fetch fetch exceedance_intensity --only '*raster.nc*'
"""

import os
import zipfile
from fnmatch import fnmatch
from typing import Annotated

import obstore
import requests
import typer
from obspec_utils.readers import BufferedStoreReader
from obstore.store import HTTPStore

from ocr.config import INPUT_PREFIX, StorageConfig
from ocr.risks.hurricane.config import METRICS, Metric

RAW_PREFIX = f'{INPUT_PREFIX}/hurricane/raw/chaz'
EXPLORE_PREFIX = 'ocr-explore/CHAZ'

# Dryad file ids for dataset version 408938 (doi:10.5061/dryad.qfttdz0vz).
FILES: dict[Metric, int] = {
    'exceedance_intensity': 4471224,
    'return_periods': 4471225,
}  # 44.7 / 13.8 GB
DOWNLOAD = 'https://datadryad.org/api/v2/files/{id}/download'

app = typer.Typer(no_args_is_help=True)


def _sizes(store, prefix: str) -> dict[str, int]:
    return {m['path']: m['size'] for m in obstore.list(store, prefix).collect()}


def _keep(path: str) -> bool:
    """netCDFs only: skips csv/ and macOS zip cruft (__MACOSX/._*)."""
    return path.endswith('.nc') and not {'csv', '__MACOSX'} & set(path.split('/'))


def _raw(explore_key: str) -> str:
    return f'{RAW_PREFIX}/{explore_key.removeprefix(f"{EXPLORE_PREFIX}/")}'


@app.command()
def copy_raw(dry_run: bool = False) -> None:
    """Copy ocr-explore/CHAZ/{metric}/{nc,raster.nc}/**.nc to RAW_PREFIX (csv/ skipped)."""
    store = StorageConfig().object_store()
    existing = _sizes(store, f'{RAW_PREFIX}/')
    src = {
        k: n
        for m in METRICS
        for k, n in _sizes(store, f'{EXPLORE_PREFIX}/{m}/').items()
        if _keep(k)
    }
    todo = {k: n for k, n in src.items() if existing.get(_raw(k)) != n}
    for k, n in todo.items():
        print(f'  {n / 1e6:8.1f} MB  {_raw(k)}')
    print(f'{"would copy" if dry_run else "copying"}={len(todo)} skipped={len(src) - len(todo)}')
    if dry_run:
        return
    for k in todo:
        obstore.copy(store, k, _raw(k))


def _presigned_url(file_id: int) -> str:
    """Resolve the Dryad download to its presigned, range-readable S3 URL."""
    token = os.environ.get('DRYAD_TOKEN')
    if not token:
        raise RuntimeError('set DRYAD_TOKEN')
    r = requests.get(
        DOWNLOAD.format(id=file_id),
        headers={'Authorization': f'Bearer {token}'},
        allow_redirects=False,
        timeout=60,
    )
    if r.is_redirect and 'Location' in r.headers:
        return r.headers['Location']
    r.raise_for_status()
    raise RuntimeError(f'expected a redirect to presigned storage, got {r.status_code}')


class _PresignedStore:
    """GET-only view of one presigned URL: the signature rejects HEAD, so size comes from a ranged GET."""

    def __init__(self, url: str) -> None:
        self._store = HTTPStore.from_url(url)

    def get(self, path: str, **kw):
        return obstore.get(self._store, path, **kw)

    def get_range(self, path: str, *, start: int, length: int):
        return obstore.get_range(self._store, path, start=start, length=length)

    def head(self, path: str):
        return obstore.get(self._store, path, options={'range': (0, 1)}).meta


def _open_zip(file_id: int) -> zipfile.ZipFile:
    reader = BufferedStoreReader(_PresignedStore(_presigned_url(file_id)), '', buffer_size=8 << 20)
    return zipfile.ZipFile(reader)


@app.command()
def fetch(
    name: Metric,
    only: Annotated[str, typer.Option(help="glob on member path, e.g. '*raster.nc*'")] = '*',
    dry_run: bool = False,
) -> None:
    """Stream members of one Dryad zip into RAW_PREFIX/<name>/."""
    file_id = FILES[name]
    store = StorageConfig().object_store()
    existing = _sizes(store, f'{RAW_PREFIX}/{name}/')
    with _open_zip(file_id) as zf:
        members = [m for m in zf.infolist() if _keep(m.filename) and fnmatch(m.filename, only)]
    dst = {
        m.filename: f'{RAW_PREFIX}/{name}/{m.filename.lstrip("/").removeprefix(f"{name}/")}'
        for m in members
    }
    todo = [m for m in members if existing.get(dst[m.filename]) != m.file_size]
    gb = sum(m.file_size for m in todo) / 1e9
    print(
        f'{len(todo)} files, {gb:.2f} GB -> {RAW_PREFIX}/{name}/ (skipped={len(members) - len(todo)})'
    )
    for i, m in enumerate(todo, 1):
        print(f'  [{i}/{len(todo)}] {dst[m.filename]} ({m.file_size / 1e6:.1f} MB)')
        if dry_run:
            continue
        # Fresh presigned URL per member: one URL can expire during a multi-GB stream.
        with _open_zip(file_id) as zf, zf.open(m) as src:
            obstore.put(store, dst[m.filename], src, chunk_size=32 << 20)
    copied = 0 if dry_run else len(todo)
    print(
        f'done: copied={copied} skipped={len(members) - len(todo)}'
        + (' (dry run)' if dry_run else '')
    )


if __name__ == '__main__':
    app()
