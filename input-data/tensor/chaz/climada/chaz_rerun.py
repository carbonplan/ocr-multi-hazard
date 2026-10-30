"""Rerun the published CHAZ ERA5 hazard maps on current CLIMADA, globally.

The published maps (Meiler et al. 2026) carry Holland winds from a CLIMADA
that double-counted translational velocity (fixed in 4.1.0). This module runs
the authors' own pipeline (github.com/simonameiler/CHAZ-hazard-maps, pinned
at MAPS_COMMIT) on the ERA5 tracks under the CLIMADA of the `climada` pixi
environment:

  wind      the call sequence of CHAZ-pre-processing/compute_era5_windfields.py
            (CLIMADA's CHAZ reader, 0.5 h resampling, Holland 2008 on the
            global 300 arcsec cells), one track file and intensity member per
            task so it spreads over Coiled VMs
  concat    CHAZ-pre-processing/concat_freq_corr_era5.py as published: joins
            the chunks, splits six basins, sets one frequency per basin
  maps      main/compute_exceedance_intensity_era5_parallel.py and
            main/compute_return_periods_era5_parallel.py as published, per tile
  combine   main/combine_tiles.py as published

The published scripts are fetched at the pinned commit and run with their
hard-coded cluster paths replaced by environment variables; PATCHES lists
every replaced string. What is ours:

  - winds at padding nodes are blanked before CLIMADA reads a track file. The
    raw files carry a few finite winds where time is the fill value and the
    position is NaN, which leaves the track extent NaN and selects no cells.
  - storms are limited to 1981-2019 with the reader's year_range, the period
    of the published catalogue (400 member runs x 39 years); the track files
    start in 1951.
  - every task uses the full global cell set. The published script keeps the
    cells within 5 degrees of its file's tracks; winds reach 300 km, so the
    nonzeros are the same and the chunks share one set of columns.
  - the cells are the published ERA5 points with distance to coast, land flag
    and region id from the CLIMADA data API's 150 arcsec centroids.

Everything lands under s3://carbonplan-ocr/ocr-explore/CHAZ/rerun/ERA5/:

  centroids                      centroids_global_0300as.hdf5
  wind      --task 3:17          hazard/present/TC_global_0300as_CHAZ_ERA5_2ens003_H08_17.hdf5
  batch     --files 0-9          submits the wind tasks to Coiled
  submit    concat               hazard/present/TC_{basin,global}_0300as_CHAZ_ERA5_freq-corr.hdf5
  submit    maps:<tile> ...      maps/tiles/TC_<tile>_0300as_CHAZ_ERA5_*.nc
  submit    combine              maps/TC_global_0300as_CHAZ_ERA5_*.nc

Run in the `climada` pixi environment (pixi run -e climada python ...).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chaz_events as ce

log = logging.getLogger('chaz_rerun')

RERUN = f'{ce.PREFIX}/rerun/ERA5'
CENTROIDS_KEY = f'{RERUN}/centroids_global_0300as.hdf5'
PRESENT = f'{RERUN}/hazard/present'
MAPS = f'{RERUN}/maps'
YEAR_RANGE = (1981, 2019)
TRACK_BATCH = 25  # tracks per from_tracks call; about 70 MB each on the global cells

MAPS_REPO = 'https://raw.githubusercontent.com/simonameiler/CHAZ-hazard-maps'
MAPS_COMMIT = '16378779bb826bf4608cd6eb82d97c68c6ee1714'
_ENV_HAZ = 'Path(os.environ["CHAZ_HAZ_DIR"])'
_ENV_MAPS = 'Path(os.environ["CHAZ_MAPS_DIR"])'
_UTILS = (
    'sys.path.append("/cluster/project/climate/meilers/scripts/columbia_haz_maps")',
    'import os\nsys.path.append(os.environ["CHAZ_MAPS_REPO"])',
)
_MAP_PATCHES = [
    _UTILS,
    ('Path("/cluster/work/climate/meilers/climada/data/hazard/")', _ENV_HAZ),
    ('Path("/cluster/work/climate/meilers/climada/data/hazard/future/CHAZ/maps")', _ENV_MAPS),
]
PATCHES = {
    'CHAZ-pre-processing/concat_freq_corr_era5.py': [
        ('SYSTEM_DIR / "hazard" / "present"', _ENV_HAZ)
    ],
    'main/compute_exceedance_intensity_era5_parallel.py': _MAP_PATCHES,
    'main/compute_return_periods_era5_parallel.py': _MAP_PATCHES,
    'main/combine_tiles.py': [
        _UTILS,
        (
            'input_dir = "/cluster/work/climate/meilers/climada/data/hazard/future/CHAZ/maps"',
            'input_dir = os.environ["CHAZ_MAPS_DIR"]',
        ),
        (
            'variable = "exceedance_intensity"  # or "return_periods"',
            'variable = os.environ["CHAZ_VARIABLE"]',
        ),
    ],
    'main/hazard_map_utils.py': [],
}


def published(work: Path) -> Path:
    """The published scripts at the pinned commit, paths patched, under work/repo."""
    repo = work / 'repo'
    for rel, patches in PATCHES.items():
        src = urllib.request.urlopen(f'{MAPS_REPO}/{MAPS_COMMIT}/{rel}').read().decode()
        for old, new in patches:
            if src.count(old) != 1:
                raise RuntimeError(f'{rel}: expected one occurrence of {old!r}')
            src = src.replace(old, new)
        out = repo / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(src)
    return repo


def run_published(repo: Path, rel: str, *argv: str, **env: str) -> None:
    cmd = [sys.executable, str(repo / rel), *argv]
    log.info('running %s %s', rel, ' '.join(argv))
    subprocess.run(cmd, check=True, env={**os.environ, 'CHAZ_MAPS_REPO': str(repo), **env})


# ---------------------------------------------------------------------------
# centroids


def cmd_centroids(args):
    from climada.hazard import Centroids
    from climada.util.api_client import Client
    from scipy.spatial import cKDTree

    s3 = ce._s3()
    if not args.force and s3.exists(f'{ce.BUCKET}/{CENTROIDS_KEY}'):
        print(f'exists: s3://{ce.BUCKET}/{CENTROIDS_KEY} (use --force)')
        return
    pts = xr.open_dataset(ce._fetch(s3, ce.DRYAD_LEVELS))
    lat, lon = pts['lat'].values, pts['lon'].values
    cent = Centroids(lat=lat, lon=lon)
    g = (
        Client()
        .get_centroids(res_arcsec_land=150, res_arcsec_ocean=1800, extent=(-180, 180, -60, 60))
        .gdf
    )
    d, j = cKDTree(np.c_[g.geometry.x.values, g.geometry.y.values]).query(np.c_[lon, lat])
    if d.max() * 3600 > 1.0:
        raise RuntimeError(
            f'Dryad cells are not on the API lattice (max offset {d.max() * 3600:.1f} arcsec)'
        )
    for col in ('dist_coast', 'on_land', 'region_id'):
        cent.gdf[col] = g[col].values[j]
    print(f'{cent.size:,} global centroids, {int(cent.gdf["on_land"].sum()):,} on land')
    local = ce._local(CENTROIDS_KEY)
    cent.write_hdf5(str(local))
    ce._put(s3, local, CENTROIDS_KEY)


# ---------------------------------------------------------------------------
# wind: one track file and intensity member


def wind_name(i: int, m: int) -> str:
    return f'TC_global_0300as_CHAZ_ERA5_2ens00{i}_H08_{m}.hdf5'


_CENT = None


def _init_worker(centroids_path: str):
    global _CENT
    from climada.hazard import Centroids

    logging.getLogger('climada').setLevel(logging.WARNING)
    _CENT = Centroids.from_hdf5(centroids_path)


def _winds(data):
    from climada.hazard import TCTracks, TropCyclone

    parts = []
    for k in range(0, len(data), TRACK_BATCH):
        chunk = TCTracks(data=data[k : k + TRACK_BATCH])
        chunk.equal_timestep(time_step_h=ce.TIME_STEP_H)
        parts.append(TropCyclone.from_tracks(chunk, centroids=_CENT, model=ce.WIND_MODEL))
    return TropCyclone.concat(parts)


def compute_wind(i: int, m: int, s3, force: bool, workers: int):
    import multiprocessing as mp
    from importlib.metadata import version

    from climada.hazard import TCTracks, TropCyclone

    key = f'{PRESENT}/{wind_name(i, m)}'
    if not force and s3.exists(f'{ce.BUCKET}/{key}'):
        print(f'exists: s3://{ce.BUCKET}/{key}')
        return
    t0 = time.time()
    cent_path = ce._fetch(s3, CENTROIDS_KEY)
    raw = xr.open_dataset(
        ce._fetch(s3, f'{ce.TRACKS}/{ce.track_file(i)}'), engine='netcdf4', decode_times=False
    )
    fill = raw['time'].values <= ce.TIME_FILL
    one = raw.isel(ensembleNum=[m])
    one['Mwspd'] = one['Mwspd'].where(~fill[None])
    clean = ce.CACHE / 'rerun' / f'm{m:02d}' / ce.track_file(i)
    clean.parent.mkdir(parents=True, exist_ok=True)
    one.to_netcdf(clean)
    tracks = TCTracks.from_simulations_chaz(clean, year_range=YEAR_RANGE)
    log.info(
        'climada %s; file %d member %d: %d tracks read in %.0fs',
        version('climada'),
        i,
        m,
        tracks.size,
        time.time() - t0,
    )
    t1 = time.time()
    parts = [tracks.data[k::workers] for k in range(workers)]
    hazards = []
    with mp.get_context('spawn').Pool(
        workers, initializer=_init_worker, initargs=(str(cent_path),)
    ) as pool:
        for haz in pool.imap_unordered(_winds, [p for p in parts if p]):
            hazards.append(haz)
            log.info('part %d/%d done, %.0fs', len(hazards), len(parts), time.time() - t1)
    haz = TropCyclone.concat(hazards)
    dt = time.time() - t1
    print(
        f'file {i} member {m}: {tracks.size:,} tracks, {haz.intensity.getnnz():,} nonzeros, '
        f'{dt:.0f}s ({dt * workers / max(tracks.size, 1):.2f} core-s/track), '
        f'max {haz.intensity.max():.1f} m/s'
    )
    local = ce._local(key)
    haz.write_hdf5(str(local))
    ce._put(s3, local, key)


def cmd_wind(args):
    task = args.task or os.environ.get('COILED_BATCH_TASK_INPUT')
    i, _, m = task.partition(':')
    compute_wind(int(i), int(m), ce._s3(), args.force, args.workers or os.cpu_count())


# ---------------------------------------------------------------------------
# concat, maps, combine: the published scripts on one VM each


def cmd_concat(args):
    s3 = ce._s3()
    work = Path(args.work)
    haz = work / 'present'
    haz.mkdir(parents=True, exist_ok=True)
    keys = s3.glob(f'{ce.BUCKET}/{PRESENT}/TC_global_0300as_CHAZ_ERA5_2ens00*_H08_*.hdf5')
    log.info('fetching %d wind chunks', len(keys))
    s3.get(keys, str(haz) + '/')
    run_published(
        published(work), 'CHAZ-pre-processing/concat_freq_corr_era5.py', CHAZ_HAZ_DIR=str(haz)
    )
    for out in sorted(haz.glob('TC_*_0300as_CHAZ_ERA5*.hdf5')):
        if '_H08_' not in out.name:
            ce._put(s3, out, f'{PRESENT}/{out.name}')


def cmd_maps(args):
    s3 = ce._s3()
    work = Path(args.work)
    haz, maps = work / 'present', work / 'maps'
    haz.mkdir(parents=True, exist_ok=True)
    name = 'TC_global_0300as_CHAZ_ERA5_freq-corr.hdf5'
    if not (haz / name).exists():
        s3.get(f'{ce.BUCKET}/{PRESENT}/{name}', str(haz / name))
    repo = published(work)
    bounds = [x for x in args.tile.split('_')]
    argv = sum(
        ([f'--{k}', v] for k, v in zip(('lon_min', 'lon_max', 'lat_min', 'lat_max'), bounds)), []
    )
    for rel in (
        'main/compute_exceedance_intensity_era5_parallel.py',
        'main/compute_return_periods_era5_parallel.py',
    ):
        run_published(repo, rel, *argv, CHAZ_HAZ_DIR=str(haz), CHAZ_MAPS_DIR=str(maps))
    for out in sorted(maps.glob('TC_*')):
        ce._put(s3, out, f'{MAPS}/tiles/{out.name}')


def cmd_tiles(args):
    """Tiles that hold cells, as `submit` task names."""
    from climada.hazard import Centroids

    cent = Centroids.from_hdf5(str(ce._fetch(ce._s3(), CENTROIDS_KEY)))
    step = args.size
    counts = {}
    for lo, la in zip(np.floor(cent.lon / step) * step, np.floor(cent.lat / step) * step):
        counts[(int(lo), int(la))] = counts.get((int(lo), int(la)), 0) + 1
    for (lo, la), n in sorted(counts.items()):
        print(
            f'maps:{lo}_{lo + step}_{la}_{la + step}' + (f'  # {n:,} cells' if args.counts else '')
        )


def cmd_combine(args):
    s3 = ce._s3()
    work = Path(args.work)
    maps = work / 'maps'
    maps.mkdir(parents=True, exist_ok=True)
    keys = [k for k in s3.glob(f'{ce.BUCKET}/{MAPS}/tiles/TC_*.nc') if '_raster' not in k]
    s3.get(keys, str(maps) + '/')
    repo = published(work)
    for variable in ('exceedance_intensity', 'return_periods'):
        run_published(
            repo, 'main/combine_tiles.py', CHAZ_MAPS_DIR=str(maps), CHAZ_VARIABLE=variable
        )
    for out in sorted(maps.glob('TC_global_*')):
        ce._put(s3, out, f'{MAPS}/{out.name}')


# ---------------------------------------------------------------------------
# Coiled


def _submit(tasks, command, log_prefix, args, max_workers):
    import coiled

    s3 = ce._s3()
    here = Path(__file__).resolve()
    files = (here, here.parent / 'chaz_events.py')
    for p in files:
        ce._put(s3, p, f'{RERUN}/code/{p.name}')
    fetch = '\n'.join(
        f"s3fs.S3FileSystem().get('{ce.BUCKET}/{RERUN}/code/{p.name}', '{p.name}')" for p in files
    )
    script = f"""#!/bin/bash
set -uo pipefail
exec > >(tee task.log) 2>&1
echo "task $COILED_BATCH_TASK_INPUT on $(hostname) $(date -u +%FT%TZ)"
python - <<'PY'
import s3fs
{fetch}
PY
cat > upload_log.py <<'PY'
import os, s3fs
name = '{log_prefix}' + os.environ.get('COILED_BATCH_TASK_INPUT', 'task').replace(':', '_')
s3fs.S3FileSystem().put('task.log', '{ce.BUCKET}/{RERUN}/logs/' + name + '.log')
PY
export CHAZ_EVENTS_DIR=$PWD/cache
{command} &
pid=$!
while kill -0 $pid 2>/dev/null; do sleep 60; python upload_log.py || true; done
wait $pid
code=$?
echo "exit $code $(date -u +%FT%TZ)"
python upload_log.py
exit $code
"""
    if args.dry_run:
        print(f'{len(tasks)} tasks, e.g. {tasks[0]} .. {tasks[-1]}')
        print(script)
        return
    result = coiled.batch.run(
        script,
        command_as_script=True,
        name=args.name,
        software=args.software,
        region='us-west-2',
        vm_type=[args.vm_type],
        scheduler_vm_type=[args.vm_type],
        arm=True,
        disk_size=args.disk_size,
        spot_policy=args.spot_policy,
        max_workers=max_workers,
        forward_aws_credentials=True,
        max_retries=args.max_retries,
        tag={'Project': 'OCR'},
        map_over_values=tasks,
    )
    print(
        f'{len(tasks)} tasks submitted: cluster {result.get("cluster_id")}, job {result.get("job_id")}'
    )


def cmd_batch(args):
    s3 = ce._s3()
    done = {Path(k).name for k in s3.glob(f'{ce.BUCKET}/{PRESENT}/*_H08_*.hdf5')}
    tasks = [
        f'{i}:{m}'
        for i in ce.parse_range(args.files)
        for m in ce.parse_range(args.members)
        if args.force or wind_name(i, m) not in done
    ]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print('nothing to do')
        return
    force = ' --force' if args.force else ''
    _submit(tasks, f'python chaz_rerun.py wind{force}', 'wind_', args, args.max_workers)


def cmd_submit(args):
    command = (
        'case "$COILED_BATCH_TASK_INPUT" in\n'
        '  maps:*) python chaz_rerun.py maps --tile "${COILED_BATCH_TASK_INPUT#maps:}" --work $PWD/work ;;\n'
        '  *) python chaz_rerun.py $COILED_BATCH_TASK_INPUT --work $PWD/work ;;\n'
        'esac'
    )
    _submit(args.tasks, command, '', args, len(args.tasks))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    logging.getLogger('climada').setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)

    q = sub.add_parser('centroids', help='global cells with distance to coast')
    q.add_argument('--force', action='store_true')
    q.set_defaults(func=cmd_centroids)

    q = sub.add_parser('wind', help='wind fields for one file and member')
    q.add_argument('--task', help="'<file>:<member>'; defaults to $COILED_BATCH_TASK_INPUT")
    q.add_argument('--workers', type=int, default=0, help='processes; 0 = all cores')
    q.add_argument('--force', action='store_true')
    q.set_defaults(func=cmd_wind)

    for name, func in (('concat', cmd_concat), ('maps', cmd_maps), ('combine', cmd_combine)):
        q = sub.add_parser(name, help=f'the published {name} step, locally')
        q.add_argument('--work', default=str(ce.CACHE / 'rerun' / 'work'))
        if name == 'maps':
            q.add_argument('--tile', required=True, help='lonmin_lonmax_latmin_latmax')
        q.set_defaults(func=func)

    q = sub.add_parser('tiles', help='list the map tiles that hold cells')
    q.add_argument('--size', type=int, default=15, help='tile edge in degrees')
    q.add_argument('--counts', action='store_true')
    q.set_defaults(func=cmd_tiles)

    def coiled_args(q, vm_type, disk_size, name):
        q.add_argument('--software', default='ocr-climada')
        q.add_argument('--vm-type', default=vm_type)
        q.add_argument('--disk-size', type=int, default=disk_size, help='GiB')
        q.add_argument('--name', default=name)
        # a reclaimed spot VM fails every task handed to it while it shuts down
        q.add_argument('--spot-policy', default='on-demand')
        q.add_argument('--max-retries', type=int, default=2)
        q.add_argument('--dry-run', action='store_true')

    q = sub.add_parser('batch', help='submit wind tasks to Coiled')
    q.add_argument('--files', default='0-9')
    q.add_argument('--members', default=f'0-{ce.N_MEMBERS - 1}')
    q.add_argument('--max-workers', type=int, default=100, help='concurrent VMs')
    q.add_argument('--limit', type=int, default=0, help='submit only the first N pending tasks')
    q.add_argument('--force', action='store_true')
    coiled_args(q, 'r8g.xlarge', 60, 'chaz-rerun-wind')
    q.set_defaults(func=cmd_batch)

    q = sub.add_parser('submit', help='run concat, maps:<tile> or combine on Coiled, one VM each')
    q.add_argument('tasks', nargs='+')
    coiled_args(q, 'r8g.8xlarge', 300, 'chaz-rerun')
    q.set_defaults(func=cmd_submit)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
