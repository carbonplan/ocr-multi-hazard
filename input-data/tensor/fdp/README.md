# USGS Flood Damage Probability (CONUS)

- Data: https://doi.org/10.5066/P954TTQN (USGS ScienceBase, public domain)
- Paper: Collins et al. (2022), _Environmental Research Letters_ 17, 034006.
  https://doi.org/10.1088/1748-9326/ac4f0f

The scripts in this directory stage the inputs of the USGS flood damage probability (FDP)
model, rebuild its training points from NOAA reports, and refit its 18 random forests, one
per HUC-2 watershed (the two-digit USGS hydrologic regions). They are run by hand, outside
the pipeline; the flood pipeline's `score` stage then reads the refit. The
[flood methods overview](../../../docs/methods/flood-risk/overview.md) describes the model
and how the refit differs from the published one, and
[Evaluation](../../../docs/methods/flood-risk/evaluation.md) checks it.

The refit's inputs and model runs go under `input/flood/fdp-refit/`. The published raster
(100 m, 31327 × 48357, EPSG:5070) goes under `input/flood/raw/usgs-fdp/`, where training
compares each refit with it.

## Retraining

`train_fdp.R` ports the released R script (its header lists what differs). Three steps, each
writing under `input/flood/fdp-refit/`. On Coiled, one pixi task runs all three in order for
the served run, `noaa-200612-202512` (the run `runs/flood/v2.0.0.toml` gives `score`). It waits
for each job and stops if one fails. Rerunning it skips work already done, and training never
replaces a watershed's outputs without `--overwrite`; to refit, run step 3 below with
`--overwrite`. Ctrl-C stops the task from waiting; the Coiled job keeps running.

```bash
pixi run -e fdp fdp-refit

# then the flood pipeline: score, pyramid, buildings, regions
ocr run runs/flood/v2.0.0.toml
```

The steps, as the tasks run them (`fdp-inputs`, `fdp-points`, `fdp-train` in `pyproject.toml`):

```bash
# 1. predictors, released points, the HUC-2 raster and the published surface (~18 GB of downloads)
pixi run -e fdp python input-data/tensor/fdp/fdp_inputs.py stage --coiled

# 2. training points for a run -> runs/<run>/points.{csv,json}
pixi run -e fdp python input-data/tensor/fdp/fdp_points.py build --runs noaa-200612-202512 --coiled

# 3. every watershed of that run, one VM each, with the layers `score` reads
pixi run -e fdp python input-data/tensor/fdp/fdp_train.py launch --run noaa-200612-202512 \
  --contributions --se --vm-type r7i.4xlarge
```

Without `--coiled`, steps 1 and 2 run locally; `fdp_train.py task --run <run> --huc 03` trains
one watershed locally. The `fdp` environment installs these scripts as the `fdp-refit` package
(`pyproject.toml` here), so Coiled package sync ships the local copies, uncommitted edits
included; the jobs (`fdp.launch`) use the Coiled workspace's AWS access, and launching needs no
local AWS credentials.

Runs:

- `original`: the released points, to check that the port reproduces the published
  surface.
- `noaa-200612-202512`: every Flood/Flash Flood report from Dec 2006 through Dec 2025 at
  its begin coordinates, with pseudo-absences resampled by the paper's rules. The served
  run.
- `noaa-200612-202005`: the same construction over the released points' window. Against
  `original` it isolates the effect of rebuilding the points; against
  `noaa-200612-202512`, the effect of the added years.

Training measures the spatial-sorting distances in metres; `--distance lonlat`
reproduces the released script, which reads them as degrees (see the overview). `--quick` shrinks tuning
and the forest for a smoke test.

Points are reproducible across machines up to one PROJ quirk: reports on the 45th parallel
sit at the edge of Canada's NAD83(CSRS) shift grid, where PROJ's inverse shift lands
slightly differently on x86 and arm64. Rebuilding `noaa-200612-202512` (built on an arm64
Mac) on Coiled moves 51 of its 103,331 presences by up to 0.25 m, each within its cell.

### Environment and resources

The `fdp` pixi environment pins R and the model's packages (raster, terra, sp, ranger, mlr) to
the versions the served run recorded in each watershed's `summary.json` (R 4.5.3, which 17 of
the 18 watersheds ran; HUC 03 was trained locally on R 4.4.3), and the other R packages to
versions that solve with them. dismo and tuneRanger have no osx-arm64 builds on conda-forge
(nor do tuneRanger's smoof and mlrMBO), so the environment cannot pin them: locally install
dismo 1.3-16 and tuneRanger 0.8.1 from CRAN (`remotes::install_version`) into the environment's
R, and on Coiled `launch` adds them as conda extras at whatever version conda-forge has, which
is the served run's today. `task` and `se` check those two and the five above against
`fdp_train.R_PACKAGES` before fitting and stop on any other version, so a newer conda-forge
build stops training until the pins are updated. `--rscript` points training at another R.
Coiled runs one x86 VM per watershed, because dismo has no Linux ARM build either. The paper
reports an average of 5.55 h per watershed on 8 cores with ~23 GB RAM; each task's `run.json`
records its actual time, peak memory, disk and VM type. Peak memory sums the forked R workers,
so it overstates what the VM needs; HUC 06 reported 74 GB and ran on `r7i.4xlarge` (128 GB).

A task's cut of the predictors is cached per watershed under `~/.cache/ocr/fdp/predictors/`
(1.6 GB for HUC 03) and reused while the staged inputs are unchanged, so repeat runs of a
watershed skip the ~8 min of reads.

### Task outputs

Each task uploads `fdp.tif` (the watershed's probabilities), `summary.json` (CV AUCs,
thresholds, tuned parameters, OOB Brier, stage timings), `compare.json` (agreement with the
published surface over the same cells, and separately outside the 100-yr floodplain),
`importance.csv`, the final ranger model, CV predictions and the R log.

`task --contributions` also runs `contrib_fdp.R`, compiled from `fdp_contrib.cpp`, which
writes `contributions.tif` (int16 bands, x 1e-4) and `contributions.json` (driver order,
baseline, checks); it adds about the time of the prediction (~20 min for HUC 03).
`task --se` also runs `se_fdp.R`, which writes `se.tif` and `se.json`. `se` adds the
standard errors to a watershed that is already trained and uploaded, from its saved forest
and prediction:

```bash
pixi run -e fdp python input-data/tensor/fdp/fdp_train.py se --run noaa-200612-202512 --huc 03
```

HUC 03 (67.6M cells) took 68 min locally on 18 workers for its standard errors, including a
165 s calibration fit.

## Served store

The flood pipeline's `score` stage (`ocr.risks.flood.refit`) mosaics the watersheds of the
run given in `runs/flood/v2.0.0.toml` into `output/flood/<version>/store.icechunk`, in the
layout documented in [the data schema](../../../docs/reference/data-schema.md#flood-risk-variables).
It fills one row of shards of one layer at a time, so it needs no large VM, and checks the sum
identity, dominant driver and point group at sampled cells of each watershed before
committing. `pyramid`, `buildings` and `regions` then run as for any risk:

```bash
ocr run runs/flood/v2.0.0.toml
```

To run `score` locally against S3: obstore uses boto3's credential chain, which covers SSO
profiles, but rasterio reads the watershed rasters through GDAL, which does not, so export
credentials first with `eval "$(aws configure export-credentials --format env)"`.

## Outputs

```
input/flood/raw/usgs-fdp/CONUS_FDP_100m.tif
input/flood/fdp-refit/inputs/predictors/{floodplain,distwater,...}.tif
input/flood/fdp-refit/inputs/response/{presence,absence}.csv
input/flood/fdp-refit/inputs/huc2_100m.tif
input/flood/fdp-refit/inputs/manifest.json
input/flood/fdp-refit/runs/<run>/points.{csv,json}
input/flood/fdp-refit/runs/<run>/<distance>[-quick]/huc<NN>/
```
