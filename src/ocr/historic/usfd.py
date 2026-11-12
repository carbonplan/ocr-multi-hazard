"""United States Flood Database v1.1 (Li et al. 2021) flood reports, CONUS, as GeoParquet with overviews.

one point per report, from seven merged databases (FEDB, NOAA storm events, GFM, mPing, DFO, CyberFlood,
EM-DAT, IFI). rows without coordinates or outside CONUS are dropped. `id` is the row's 0-based line in the
CSV: the source's own ID repeats across databases, and the zenodo file never changes,
so the line number is stable. dates keep the first 8 digits of DATE_BEGIN/DATE_END (YYYYMMDD); 41 rows have
no parseable start. `year` comes from the start date, else the year column, which is wrong for ~200 rows that
have a date. noaa reports sharing a place and start date are kept as published

writes input/historic/usfd/v1.1/usfd_floods.parquet. zenodo versions are immutable, so the key carries the
version, not a date. the CSV, checked against zenodo's md5, is mirrored beside it at raw/USFD_v1.1.csv

cite Li et al. (2021), doi:10.5194/essd-13-3755-2021; data doi:10.5281/zenodo.7545697 (CC BY 4.0)

run: pixi run historic-usfd [--overwrite]
"""

import hashlib
import tempfile
import urllib.request
from collections.abc import Mapping
from datetime import UTC, date, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import duckdb
import geoparquet_lod
import obstore
import pyarrow.compute as pc
import typer

from ocr.buildings import CONUS
from ocr.config import StorageConfig
from ocr.duck import duckdb_connect
from ocr.geoparquet import bbox_sql, write_relation
from ocr.risks.fire.utils import get_temp_dir

VERSION = 'v1.1'
CSV_NAME = 'USFD_v1.1.csv'
URL = f'https://zenodo.org/records/7545697/files/{CSV_NAME}'
MD5 = 'f91b3447c900c39fd0a5bf44dc8edf11'
DOI = '10.5281/zenodo.7545697'
ATTRIBUTION = 'United States Flood Database v1.1, Li et al. 2021'
LICENSE = 'CC-BY-4.0'
KEY = f'input/historic/usfd/{VERSION}/usfd_floods.parquet'
RAW_KEY = f'input/historic/usfd/{VERSION}/raw/{CSV_NAME}'
ZOOMS = list(range(11))
COMPLETE_FROM = 10
THINNING = geoparquet_lod.density_budget()
CARRY = ['source', 'year']
SCM_FALLBACK_VERSION = '999'
# Every column the build reads, typed: inferred types turn source ids' leading zeros and dates' shapes into numbers.
CSV_TYPES = {
    'DATE_BEGIN': 'VARCHAR',
    'DATE_END': 'VARCHAR',
    'LON': 'DOUBLE',
    'LAT': 'DOUBLE',
    'STATE': 'VARCHAR',
    'LOCATION': 'VARCHAR',
    'FATALITY': 'VARCHAR',
    'DAMAGE': 'DOUBLE',
    'CAUSE': 'VARCHAR',
    'SOURCE_DB': 'VARCHAR',
    'SOURCE_ID': 'VARCHAR',
    'year': 'DOUBLE',
}


def source_metadata(*, sha256: str, retrieved: date) -> dict[str, str]:
    return {
        'usfd_url': URL,
        'usfd_doi': DOI,
        'usfd_version': VERSION,
        'usfd_md5': MD5,
        'usfd_sha256': sha256,
        'usfd_raw_mirror': RAW_KEY,
        'usfd_retrieved': retrieved.isoformat(),
        'usfd_attribution': ATTRIBUTION,
        'usfd_license': LICENSE,
    }


def ocr_version() -> str:
    """The ocr version, whose commit also pins geoparquet-lod in pyproject.toml."""
    version = importlib_metadata.version('ocr')
    return f'unknown ({version})' if version == SCM_FALLBACK_VERSION else version


def processing_metadata() -> dict[str, str]:
    """What shaped the output that geoparquet-lod doesn't already record (it writes zooms, carried fields and thinning)."""
    return {'usfd_ocr_version': ocr_version(), 'usfd_complete_from': str(COMPLETE_FROM)}


def date_sql(column: str) -> str:
    """SQL for the date in the first 8 digits of `column` ('20020130013000.0', '199610211400', '20060407.0'),
    NULL when it has fewer. Raises on digits that are no date."""
    return (
        f"CASE WHEN regexp_full_match(split_part({column}, '.', 1), '[0-9]{{8,}}') "
        f"THEN CAST(strptime(left({column}, 8), '%Y%m%d') AS DATE) END"
    )


def points(con: duckdb.DuckDBPyConnection, csv_path: str | Path) -> duckdb.DuckDBPyRelation:
    """The CONUS flood reports in the USFD CSV at `csv_path`."""
    # DuckDB has no title case. Arrow's matches Python's str.title, the old PMTiles', on every STATE/LOCATION.
    con.create_function(
        'title', lambda text: pc.utf8_title(text), ['VARCHAR'], 'VARCHAR', type='arrow'
    )
    # A table, so rowid is the CSV line: insertion order is kept and nothing is filtered before it.
    con.execute(
        'CREATE OR REPLACE TEMP TABLE usfd_csv AS SELECT * FROM read_csv($p, header=true, types=$t)',
        {'p': str(csv_path), 't': CSV_TYPES},
    )
    return con.sql(
        f"""
        SELECT *, {bbox_sql('geometry')} AS bbox FROM (
            SELECT rowid AS id,
                   SOURCE_DB AS source,
                   SOURCE_ID AS source_id,
                   {date_sql('DATE_BEGIN')} AS begin_date,
                   {date_sql('DATE_END')} AS end_date,
                   coalesce(year({date_sql('DATE_BEGIN')}), CAST(year AS INTEGER)) AS year,
                   title(STATE) AS state,
                   title(LOCATION) AS location,
                   DAMAGE AS damage,
                   CAST(CAST(FATALITY AS DOUBLE) AS INTEGER) AS fatalities,
                   CAUSE AS cause,
                   ST_Point(LON, LAT) AS geometry
            FROM usfd_csv
            WHERE LON BETWEEN {CONUS.left} AND {CONUS.right} AND LAT BETWEEN {CONUS.bottom} AND {CONUS.top}
        )
        """
    )


def write_local(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
    temp_dir: str | Path | None = None,
) -> Path:
    """Write the CONUS flood reports from the USFD CSV as GeoParquet with overviews to out_dir/usfd_floods.parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'usfd_floods.parquet'
    meta = processing_metadata() | dict(metadata or {})
    with tempfile.TemporaryDirectory(dir=temp_dir) as tmp, duckdb_connect() as con:
        # geoparquet-lod reads its source several times, so it needs a file, not a stream.
        staged = Path(tmp) / 'floods.parquet'
        write_relation(points(con, csv_path), meta, staged)
        geoparquet_lod.write(
            str(staged),
            out,
            zooms=ZOOMS,
            thinning=THINNING,
            carry=CARRY,
            complete_from=COMPLETE_FROM,
            metadata=meta,
            temp_dir=tmp,
        )
    return out


def download(url: str, dest: Path) -> tuple[str, str]:
    """Stream `url` to `dest`; returns its md5 and sha256."""
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with urllib.request.urlopen(url, timeout=300) as resp, dest.open('wb') as f:
        while chunk := resp.read(1 << 20):
            md5.update(chunk)
            sha256.update(chunk)
            f.write(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def exists(key: str) -> bool:
    try:
        obstore.head(StorageConfig().object_store(), key)
    except FileNotFoundError:
        return False
    return True


def main(overwrite: bool = typer.Option(False, '--overwrite')) -> None:
    build, mirrored = overwrite or not exists(KEY), exists(RAW_KEY)
    if not build and mirrored:
        typer.echo(f'skip: {KEY} exists; pass --overwrite')
        return
    with tempfile.TemporaryDirectory(dir=get_temp_dir()) as tmp:
        csv_path = Path(tmp) / CSV_NAME
        md5, sha256 = download(URL, csv_path)
        if md5 != MD5:
            raise ValueError(f'{URL}: md5 {md5}, zenodo publishes {MD5}')
        if build:
            meta = source_metadata(sha256=sha256, retrieved=datetime.now(UTC).date())
            out = write_local(csv_path, Path(tmp) / 'out', metadata=meta, temp_dir=tmp)
        if not mirrored:
            obstore.put(StorageConfig().object_store(), RAW_KEY, csv_path)
            typer.echo(f'mirrored {RAW_KEY}')
        if build:
            obstore.put(StorageConfig().object_store(), KEY, out)
            typer.echo(f'wrote {KEY}')
        else:
            typer.echo(f'skip: {KEY} exists; pass --overwrite')


if __name__ == '__main__':
    typer.run(main)
