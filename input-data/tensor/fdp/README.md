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
compares each refit with it and the flood ETL reads it.

## Retraining

`train_fdp.R` ports the released R script (its header lists what differs). Three steps,
each resumable and each writing under `input/flood/fdp-refit/`. On Coiled, from start to
finish (each step needs the previous one's output):

```bash
# 1. predictors, released points, the HUC-2 raster and the published surface (~18 GB of downloads)
pixi run -e fdp python input-data/tensor/fdp/fdp_inputs.py stage --coiled

# 2. training points for a run -> runs/<run>/points.{csv,json}
pixi run -e fdp python input-data/tensor/fdp/fdp_points.py build --runs noaa-200612-202512 --coiled

# 3. every watershed of that run, one VM each, with the layers `score` reads
pixi run -e fdp python input-data/tensor/fdp/fdp_train.py launch --run noaa-200612-202512 \
  --contributions --se --vm-type r7i.4xlarge

# then the flood pipeline: score, pyramid, buildings, regions
ocr run runs/flood/v2.0.0.toml
```

Without `--coiled`, steps 1 and 2 run locally; `fdp_train.py task --run <run> --huc 03`
trains one watershed locally. Coiled jobs copy these scripts to
`input/flood/fdp-refit/code/<hash>/` and run them from there (`fdp.launch`), with the
Coiled workspace's AWS access; launching needs local AWS credentials to copy the scripts
there.

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

Locally, the `fdp-r` environment needs four CRAN packages that conda-forge does not build
for macOS. Compile them once:

```bash
pixi run -e fdp-r fdp-r-cran
```

On Coiled, `launch` adds the R packages to the synced `fdp` environment as conda extras,
one x86 VM per watershed (dismo has no ARM build). The paper reports an average of 5.55 h
per watershed on 8 cores with ~23 GB RAM; each task's `run.json` records its actual time,
peak memory, disk and VM type. Peak memory sums the forked R workers, so it overstates what
the VM needs; HUC 06 reported 74 GB and ran on `r7i.4xlarge` (128 GB).

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
pixi run -e fdp-r python input-data/tensor/fdp/fdp_train.py se --run noaa-200612-202512 --huc 03
```

HUC 03 (67.6M cells) took 68 min locally on 18 workers for its standard errors, including a
165 s calibration fit.

## Served store

The flood pipeline's `score` stage (`ocr.risks.flood.refit`) mosaics the watersheds of the
run given in `runs/flood/v2.0.0.toml` into `output/flood/<version>/store.icechunk`, in the
layout documented in [the data schema](../../../docs/reference/data-schema.md#flood-risk-variables).
It reads one band of one watershed at a time, so it needs no large VM, and checks the sum
identity, dominant driver and point group at sampled cells of each watershed before
committing. `pyramid`, `buildings` and `regions` then run as for any risk:

```bash
ocr run runs/flood/v2.0.0.toml
```

obstore reads AWS credentials from the environment only, so locally export them first with
`eval "$(aws configure export-credentials --format env)"`.

## Temporal validation

[Temporal validation](../../../docs/methods/flood-risk/temporal-validation.ipynb) tests the
published map against NOAA reports from June 2020 through December 2025.
`validation-2026-10-15/` holds its run record, archived results and input hashes, and
`validate_fdp.py` reruns it.

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
