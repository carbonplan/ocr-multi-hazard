# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "rasterio",
#   "s3fs>=2024.0.0",
# ]
# ///
"""Train and predict the FDP random forest for HUC-2 watersheds.

`task` runs one watershed on the current machine: it window-reads that
watershed's predictors from the staged COGs, masks them to its HUC raster
cells (cached under $FDP_CACHE/predictors/), runs train_fdp.R, compares the
prediction with the published surface, and uploads everything to

  input/flood/fdp-refit/runs/<run>/<distance>[-quick]/huc<NN>/

`run` names a points set built by fdp_points.py; its absence seed also seeds
the folds and tuning in train_fdp.R. `--distance lonlat` keeps the
released script's spatial-sorting distances; `planar` corrects them (see
train_fdp.R). `--quick` shrinks tuning and the forest for a smoke test.
`--contributions` adds per-cell attributions of the prediction to each
predictor (contrib_fdp.R); `--se` adds per-cell standard errors (se_fdp.R).

`se` adds standard errors to a watershed already trained and uploaded, from
its saved forest and prediction, without retraining.

`launch` submits one `task` per watershed as a Coiled batch job (`fdp.launch`).
dismo has no linux-aarch64 build, so the VMs are x86.

The R packages are not in any pixi environment: locally, point `--rscript` at
an R with them installed (see the README); on Coiled they are added to the
synced environment as conda extras.

Usage:
  python fdp_train.py task --run original --huc 03 [--rscript PATH] [--quick] [--contributions] [--se]
  python fdp_train.py se --run original --huc 03 [--rscript PATH]
  python fdp_train.py launch --run original --hucs 03 [--vm-type m7i.4xlarge] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import s3fs
from fdp import CACHE, PUBLISHED_KEY, launch, vm_type
from fdp_inputs import HUC_KEY, HUCS, PREDICTORS, predictor_key, read_manifest
from fdp_points import RUNS, SEED, run_key
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
R_SCRIPT = 'train_fdp.R'
CONTRIB_SCRIPT = 'contrib_fdp.R'
SE_SCRIPT = 'se_fdp.R'
SE_FILES = ('se.tif', 'se.json', 'se-log.txt')
R_PACKAGES = [
    'compilers',
    'r-rcpp',
    'r-base',
    'r-raster',
    'r-terra',
    'r-sp',
    'r-dismo',
    'r-ranger',
    'r-tuneranger',
    'r-mlr',
    'r-data.table',
    'r-jsonlite',
    'r-snow',
]
QUICK = {'iters': 3, 'trees': 300}


def model_prefix(run: str, distance: str, quick: bool, huc: str) -> str:
    return f'{RUNS}/{run}/{distance}{"-quick" if quick else ""}/huc{huc}'


def r_script(name: str = R_SCRIPT) -> Path:
    for path in (HERE / name, Path.cwd() / name):
        if path.exists():
            return path
    sys.exit(f'{name} not found next to {__file__} or in {Path.cwd()}')


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------


def prepare_predictors(huc: str, manifest: dict) -> tuple[Path, np.ndarray]:
    """Write the watershed's predictors, cells outside it set to nodata.

    The files are cached per watershed and reused while the staged inputs they
    were cut from are unchanged. Returns their directory and the watershed's
    cell mask over its window.
    """
    grid = manifest['grid']
    offsets = {name: manifest['predictors'][name]['offset'] for name in PREDICTORS}
    window = manifest['huc2']['windows'][huc]
    r0, c0, h, w = window
    win = Window(c0, r0, w, h)
    with rasterio.open(f's3://{HUC_KEY}') as src:
        outside = src.read(1, window=win) != int(huc)

    dest = CACHE / 'predictors' / f'huc{huc}'
    stamp = {
        'window': window,
        'grid': grid,
        'huc2': manifest['huc2']['source'],
        'predictors': {
            name: {k: manifest['predictors'][name][k] for k in ('zip_sha256', 'offset')}
            for name in PREDICTORS
        },
    }
    if (dest / 'inputs.json').exists() and json.loads((dest / 'inputs.json').read_text()) == stamp:
        print(f'  cached: {dest}', flush=True)
        return dest, ~outside
    if dest.exists():
        shutil.rmtree(dest)
    partial = dest.with_name(dest.name + '.partial')
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    transform = rasterio.windows.transform(win, rasterio.Affine(*grid['transform']))
    for name in PREDICTORS:
        t0 = time.time()
        dr, dc = offsets[name]
        with rasterio.open(f's3://{predictor_key(name)}') as src:
            if src.nodata is None:
                sys.exit(f'{name} has no nodata value')
            data = src.read(1, window=Window(c0 + dc, r0 + dr, w, h))
            data[outside] = src.nodata
            profile = src.profile | {
                'driver': 'GTiff',
                'height': h,
                'width': w,
                'transform': transform,
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'compress': 'deflate',
                'BIGTIFF': 'IF_SAFER',
            }
        with rasterio.open(partial / f'{name}.tif', 'w', **profile) as out:
            out.write(data, 1)
        print(f'  {name}: {h}x{w} in {time.time() - t0:.0f}s', flush=True)
    (partial / 'inputs.json').write_text(json.dumps(stamp, indent=2))
    partial.rename(dest)
    return dest, ~outside


class Monitor(threading.Thread):
    """Poll peak memory of a process group and peak disk use of a directory."""

    def __init__(self, pgid: int, path: Path, every: float = 10.0):
        super().__init__(daemon=True)
        self.pgid, self.path, self.every = pgid, path, every
        self.peak_rss_gb = 0.0
        self.peak_disk_gb = 0.0
        self.done = threading.Event()

    def rss(self) -> float:
        out = subprocess.run(
            ['ps', '-A', '-o', 'pgid=,rss='], capture_output=True, text=True
        ).stdout
        kb = sum(
            int(r) for g, r in (line.split() for line in out.splitlines()) if int(g) == self.pgid
        )
        return kb / 1e6

    def disk(self) -> float:
        return sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file()) / 1e9

    def run(self) -> None:
        while not self.done.wait(self.every):
            self.peak_rss_gb = max(self.peak_rss_gb, self.rss())
            self.peak_disk_gb = max(self.peak_disk_gb, self.disk())


def run_r(cmd: list[str], log: Path, work: Path) -> dict:
    """Run an R script, teeing its output to `log`; returns time and peaks."""
    t0 = time.time()
    with log.open('w') as fh:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
        )
        monitor = Monitor(proc.pid, work)
        monitor.start()
        for line in proc.stdout:
            if 'codetools' in line:
                continue
            fh.write(line)
            print(line, end='', flush=True)
        code = proc.wait()
        monitor.done.set()
    if code:
        sys.exit(f'{Path(cmd[1]).name} exited {code}; see {log}')
    return {
        'seconds': round(time.time() - t0),
        'peak_rss_gb': round(monitor.peak_rss_gb, 1),
        'peak_disk_gb': round(monitor.peak_disk_gb, 1),
    }


def agreement(a: np.ndarray, b: np.ndarray, sample: int, seed: int) -> dict:
    d = np.abs(a - b)
    rng = np.random.default_rng(seed)
    pick = rng.choice(a.size, size=min(sample, a.size), replace=False)
    spearman = pd.Series(a[pick]).rank().corr(pd.Series(b[pick]).rank())
    top_a, top_b = a >= np.quantile(a, 0.8), b >= np.quantile(b, 0.8)
    return {
        'cells': int(a.size),
        'mean': {'ours': float(a.mean()), 'published': float(b.mean())},
        'pearson': float(np.corrcoef(a, b)[0, 1]),
        'spearman_sample': float(spearman),
        'abs_diff': {
            'mean': float(d.mean()),
            'p50': float(np.median(d)),
            'p90': float(np.quantile(d, 0.9)),
            'share_gt_0.1': float((d > 0.1).mean()),
            'share_gt_0.25': float((d > 0.25).mean()),
        },
        'top20_overlap': float((top_a & top_b).sum() / top_b.sum()),
    }


def compare(
    prediction: Path,
    inside: np.ndarray,
    floodplain: Path,
    sample: int = 2_000_000,
    seed: int = 0,
) -> dict:
    """Agreement with the published surface over the watershed's cells.

    The 100-yr floodplain has no pseudo-absences, so the model scores it near
    1 throughout; `outside_floodplain` measures agreement where the surface
    actually ranks cells.
    """
    with rasterio.open(prediction) as src:
        ours = src.read(1, masked=True)
        t = src.transform
    with rasterio.open(floodplain) as src:
        in_floodplain = src.read(1) == 1
    with rasterio.open(published_tif()) as src:
        p = src.transform
        col, row = (t.c - p.c) / p.a, (t.f - p.f) / p.e
        if (p.a, p.e) != (t.a, t.e) or max(abs(col - round(col)), abs(row - round(row))) > 1e-3:
            sys.exit('the published surface is off the predictor lattice')
        win = Window(round(col), round(row), ours.shape[1], ours.shape[0])
        published = src.read(1, window=win, masked=True).astype('float64').filled(np.nan)

    mine, theirs = ~ours.mask & inside, np.isfinite(published) & inside
    both = mine & theirs
    a = ours.data[both].astype('float64')
    b = published[both].astype('float64')
    fp = in_floodplain[both]
    return {
        'cells_huc': int(inside.sum()),
        'cells_both': int(both.sum()),
        'cells_ours_only': int((mine & ~theirs).sum()),
        'cells_published_only': int((theirs & ~mine).sum()),
        **agreement(a, b, sample, seed),
        'floodplain_share': float(fp.mean()),
        'outside_floodplain': agreement(a[~fp], b[~fp], sample, seed),
    }


def published_tif() -> Path:
    """The published surface (`fdp_inputs.py stage` puts it in S3), cached locally."""
    tif = CACHE / Path(PUBLISHED_KEY).name
    if not tif.exists():
        tif.parent.mkdir(parents=True, exist_ok=True)
        part = tif.with_suffix('.part')
        s3fs.S3FileSystem().get_file(PUBLISHED_KEY, str(part))
        part.rename(tif)
    return tif


def upload(local: Path, prefix: str) -> None:
    if '/runs/' not in prefix or not prefix.rsplit('/', 1)[-1].startswith('huc'):
        raise ValueError(f'refusing to write to {prefix!r}')
    s3 = s3fs.S3FileSystem()
    if s3.exists(prefix):
        s3.rm(prefix, recursive=True)
    for f in sorted(local.iterdir()):
        s3.put_file(str(f), f'{prefix}/{f.name}')
    print(f'uploaded {len(list(local.iterdir()))} files -> s3://{prefix}')


def run_se(args, out: Path, predictors: Path, work: Path, threads: int) -> dict:
    print('estimating standard errors', flush=True)
    cmd = [
        args.rscript,
        str(r_script(SE_SCRIPT)),
        '--model',
        str(out / 'rf.rds'),
        '--predictors',
        str(predictors),
        '--prediction',
        str(out / 'fdp.tif'),
        '--out',
        str(out),
        '--threads',
        str(threads),
    ]
    return run_r(cmd, out / 'se-log.txt', work)


def cmd_task(args) -> None:
    huc = args.huc or os.environ.get('COILED_BATCH_TASK_INPUT')
    if huc not in HUCS:
        sys.exit(f'--huc must be one of {HUCS}, got {huc!r}')
    prefix = model_prefix(args.run, args.distance, args.quick, huc)
    threads = args.threads or os.cpu_count()
    settings = QUICK if args.quick else {'iters': 70, 'trees': 1000}
    print(f'HUC {huc} -> s3://{prefix} ({threads} threads, {settings})', flush=True)

    work = Path(tempfile.mkdtemp(prefix=f'fdp-huc{huc}-', dir=args.work))
    out, tmp = work / 'out', work / 'tmp'
    out.mkdir()
    tmp.mkdir()

    t0 = time.time()
    manifest = read_manifest()
    predictors, inside = prepare_predictors(huc, manifest)
    inputs = {'window': manifest['huc2']['windows'][huc], 'cells': int(inside.sum())}
    s3 = s3fs.S3FileSystem()
    s3.get_file(run_key(args.run, 'points.csv'), str(work / 'points.csv'))
    with s3.open(run_key(args.run, 'points.json')) as fh:
        seed = json.load(fh).get('absence_sampling', {}).get('seed', SEED)
    prepare_seconds = round(time.time() - t0)

    cmd = [
        args.rscript,
        str(r_script()),
        '--huc',
        huc,
        '--predictors',
        str(predictors),
        '--points',
        str(work / 'points.csv'),
        '--out',
        str(out),
        '--tmp',
        str(tmp),
        '--threads',
        str(threads),
        '--iters',
        str(settings['iters']),
        '--trees',
        str(settings['trees']),
        '--distance',
        args.distance,
        '--seed',
        str(seed),
    ]
    r = run_r(cmd, out / 'log.txt', work)
    contributions = None
    if args.contributions:
        print('attributing the prediction to its predictors', flush=True)
        contrib_cmd = [
            args.rscript,
            str(r_script(CONTRIB_SCRIPT)),
            '--model',
            str(out / 'rf.rds'),
            '--train',
            str(out / 'train.rds'),
            '--predictors',
            str(predictors),
            '--prediction',
            str(out / 'fdp.tif'),
            '--out',
            str(out),
            '--threads',
            str(threads),
        ]
        contributions = run_r(contrib_cmd, out / 'contrib-log.txt', work)
    se = run_se(args, out, predictors, work, threads) if args.se else None

    print('comparing with the published surface', flush=True)
    stats = compare(out / 'fdp.tif', inside, predictors / 'floodplain.tif')
    (out / 'compare.json').write_text(json.dumps(stats, indent=2))
    (out / 'run.json').write_text(
        json.dumps(
            {
                'huc': huc,
                'run': args.run,
                'distance': args.distance,
                'quick': args.quick,
                'settings': settings,
                'seed': seed,
                'threads': threads,
                'inputs': inputs,
                'prepare_seconds': prepare_seconds,
                'r': r,
                'contributions': contributions,
                'se': se,
                'host': {
                    'platform': platform.platform(),
                    'cpus': os.cpu_count(),
                    'vm_type': vm_type(),
                },
            },
            indent=2,
        )
    )
    upload(out, prefix)
    if not args.keep:
        shutil.rmtree(work)


def cmd_se(args) -> None:
    """Standard errors for an uploaded watershed, added to its prefix."""
    huc = args.huc or os.environ.get('COILED_BATCH_TASK_INPUT')
    if huc not in HUCS:
        sys.exit(f'--huc must be one of {HUCS}, got {huc!r}')
    prefix = model_prefix(args.run, args.distance, args.quick, huc)
    s3 = s3fs.S3FileSystem()
    if not s3.exists(f'{prefix}/rf.rds'):
        sys.exit(f's3://{prefix} has no trained forest; run `task` first')
    threads = args.threads or os.cpu_count()
    print(f'HUC {huc} standard errors -> s3://{prefix} ({threads} threads)', flush=True)

    work = Path(tempfile.mkdtemp(prefix=f'fdp-se-huc{huc}-', dir=args.work))
    out = work / 'out'
    out.mkdir()
    predictors, _ = prepare_predictors(huc, read_manifest())
    for name in ('rf.rds', 'fdp.tif', 'run.json'):
        s3.get_file(f'{prefix}/{name}', str(out / name))
    se = run_se(args, out, predictors, work, threads)

    record = json.loads((out / 'run.json').read_text())
    record['se'] = se | {'host': {'platform': platform.platform(), 'cpus': os.cpu_count()}}
    (out / 'run.json').write_text(json.dumps(record, indent=2))
    for name in (*SE_FILES, 'run.json'):
        s3.put_file(str(out / name), f'{prefix}/{name}')
    print(f'uploaded {", ".join(SE_FILES)} -> s3://{prefix}')
    if not args.keep:
        shutil.rmtree(work)


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


def cmd_launch(args) -> None:
    task = ['fdp_train.py', 'task', '--run', args.run, '--distance', args.distance]
    task += [f'--{flag}' for flag in ('quick', 'contributions', 'se') if getattr(args, flag)]
    launch(
        f'fdp-{args.run}-{args.distance}{"-quick" if args.quick else ""}',
        task,
        vm_type=args.vm_type,
        disk_size=args.disk_size,
        spot=args.spot,
        map_over=args.hucs or HUCS,
        conda_extras=R_PACKAGES,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--run', required=True, help='points set from fdp_points.py')
    common.add_argument('--distance', choices=['lonlat', 'planar'], default='lonlat')
    common.add_argument('--quick', action='store_true')

    outputs = argparse.ArgumentParser(add_help=False)
    outputs.add_argument(
        '--contributions', action='store_true', help='also write per-cell predictor attributions'
    )
    outputs.add_argument('--se', action='store_true', help='also write per-cell standard errors')

    local = argparse.ArgumentParser(add_help=False)
    local.add_argument('--huc', help='defaults to $COILED_BATCH_TASK_INPUT')
    local.add_argument('--rscript', default='Rscript')
    local.add_argument('--threads', type=int)
    local.add_argument('--work', help='parent directory for the working files')
    local.add_argument('--keep', action='store_true', help='keep the working directory')

    sub.add_parser(
        'task', parents=[common, outputs, local], help='train and predict one watershed here'
    )
    sub.add_parser(
        'se', parents=[common, local], help='add standard errors to an uploaded watershed'
    )

    la = sub.add_parser(
        'launch', parents=[common, outputs], help='submit one task per watershed to Coiled'
    )
    la.add_argument('--hucs', nargs='+', choices=HUCS)
    la.add_argument('--vm-type', default='m7i.4xlarge')
    la.add_argument('--disk-size', type=int, default=200)
    la.add_argument('--spot', action='store_true')

    args = p.parse_args()
    {'task': cmd_task, 'se': cmd_se, 'launch': cmd_launch}[args.cmd](args)


if __name__ == '__main__':
    main()
