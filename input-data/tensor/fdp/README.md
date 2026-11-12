# USGS Flood Damage Probability (CONUS)

- Data: https://doi.org/10.5066/P954TTQN (USGS ScienceBase, public domain)
- Paper: Collins et al. (2022), _Environmental Research Letters_ 17, 034006.
  https://doi.org/10.1088/1748-9326/ac4f0f

Random-forest-predicted probability of flood damage, 0..1, on a 100 m
31327 × 48357 CONUS grid in NAD83 / Conus Albers (EPSG:5070).

The scripts in this directory are run by hand, outside the pipeline. The refit's training inputs
and model runs go under `input/flood/fdp-refit/`, where the flood pipeline's
`score` stage reads them (see [Served store](#served-store)). The published
raster goes under `input/flood/raw/usgs-fdp/`, where training compares each
refit with it and the flood ETL reads it.

[The validation and rerun specification](RERUN_SPEC.md) lays out the temporal
validation and the retraining with newer NOAA reports;
[Retraining](#retraining) below runs the refit.

The executed [local validation notebook](validation-2026-10-15/validation.ipynb)
compares the published raster with June 2020–December 2025 NOAA reports. Its
[run record](validation-2026-10-15/README.md) lists the results, limitations and
input hashes, and how to rerun the validation script locally.

## Retraining

Refits the paper's 18 random forests, one per HUC-2 watershed (the two-digit
USGS hydrologic regions), with `train_fdp.R`, a port of the released R script
(see its header for what differs). Three steps, each resumable and each writing
under `input/flood/fdp-refit/`. On Coiled, from start to finish (each step needs
the previous one's output):

```bash
# 1. predictors, released points, the HUC-2 raster and the published surface (~18 GB of downloads)
pixi run -e fdp python input-data/tensor/fdp/fdp_inputs.py stage --coiled

# 2. training points for a run -> runs/<run>/points.{csv,json}
pixi run -e fdp python input-data/tensor/fdp/fdp_points.py build --runs noaa-200612-202512 --coiled

# 3. every watershed of that run, one VM each, with the layers `score` reads
pixi run -e fdp python input-data/tensor/fdp/fdp_train.py launch --run noaa-200612-202512 \
  --distance planar --contributions --se --vm-type r7i.4xlarge

# then the flood pipeline: score, pyramid, buildings, regions
ocr run runs/flood/v2.0.0.toml
```

Without `--coiled`, steps 1 and 2 run locally; `fdp_train.py task --run <run> --huc 03`
trains one watershed locally. Coiled jobs copy these scripts to
`input/flood/fdp-refit/code/<hash>/` and run them from there (`fdp.launch`), with
the Coiled workspace's AWS access; launching needs local AWS credentials to copy
the scripts there.

Points are reproducible across machines up to one PROJ quirk: reports on the
45th parallel sit at the edge of Canada's NAD83(CSRS) shift grid, where PROJ's
inverse shift lands slightly differently on x86 and arm64. Rebuilding
`noaa-200612-202512` (built on an arm64 Mac) on Coiled moves 51 of its 103,331
presences by up to 0.25 m, each within its cell; every other value matches.

Runs:

- `original`: the released points, to check that the port reproduces the
  published surface.
- `noaa-200612-202512`: every Flood/Flash Flood report from Dec 2006 through
  Dec 2025 at its begin coordinates, with pseudo-absences resampled by the
  paper's rules. `points.json` records counts, the NOAA files and their hashes.
- `noaa-200612-202005`: the same construction over the released points'
  window. Against `original` it isolates the effect of rebuilding the points;
  against `noaa-200612-202512`, the effect of the added years.

`--distance planar` corrects the spatial-sorting distances that the released
script computes as if Albers metres were degrees (see
[the rerun specification](RERUN_SPEC.md#data-readiness)); the default keeps the
original behaviour. `--quick` shrinks tuning and the forest for a smoke test.

Locally, the `fdp-r` environment needs four CRAN packages that conda-forge does
not build for macOS. Compile them once:

```bash
pixi run -e fdp-r fdp-r-cran
```

On Coiled, `launch` adds the R packages to the synced `fdp` environment as conda
extras, one x86 VM per watershed (dismo has no ARM build). The paper reports an
average of 5.55 h per watershed on 8 cores with ~23 GB RAM; each task's
`run.json` records its actual time, peak memory, disk and VM type. Peak memory
sums the forked R workers, so it overstates what the VM needs; HUC 06 reported
74 GB and ran on `r7i.4xlarge` (128 GB).

A task's cut of the predictors is cached per watershed under
`~/.cache/ocr/fdp/predictors/` (1.6 GB for HUC 03) and reused while the staged
inputs are unchanged, so repeat runs of a watershed skip the ~8 min of reads.

Each task uploads `fdp.tif` (the watershed's probabilities), `summary.json`
(CV AUCs, thresholds, tuned parameters, OOB Brier, stage timings),
`compare.json` (agreement with the published surface over the same cells, and
separately outside the 100-yr floodplain), `importance.csv`, the final ranger
model, CV predictions and the R log.

The 100-yr floodplain has no pseudo-absences, so the model scores it near 1
throughout: in HUC 03 every floodplain cell (20% of the watershed) falls in the
top fifth of the surface. Agreement over the whole watershed is therefore
inflated by a block both surfaces get trivially right, and the
outside-floodplain figures are the informative ones.

### Driver attributions

`task --contributions` also runs `contrib_fdp.R`, which splits every cell's
probability into one contribution per predictor along its tree paths (Saabas
attribution, compiled from `fdp_contrib.cpp`): per cell, baseline +
contributions = probability. It writes `contributions.tif` (int16 bands,
x 1e-4) and `contributions.json` (driver order, baseline, checks). Two checks
fail the task: the internal node values must reproduce the leaf values ranger
stored, and the reconstructed probabilities must match `fdp.tif` (to 3e-8 on
HUC 06). It adds about the time of the prediction (~20 min for HUC 03).

The attributions explain the model, not causation: correlated terrain
predictors share credit arbitrarily, and inside the floodplain nearly all of
the probability is credited to the floodplain, for the reason above.

### Standard errors

`task --se` also runs `se_fdp.R`, which writes each cell's standard error as
`se.tif` with a record in `se.json`. `se` adds them to a watershed that is
already trained and uploaded, from its saved forest and prediction:

```bash
pixi run -e fdp-r python input-data/tensor/fdp/fdp_train.py se --run noaa-200612-202512 --distance planar --huc 03
```

The method is the paper's (supplement 1.1.2): ranger's infinitesimal jackknife
with its empirical Bayes calibration. The paper mapped it for four cities only;
here it covers every cell. Three things differ from calling
`predict(type = "se")`:

- The jackknife variance is computed from the B × B Gram matrix of in-bag
  counts, so a cell costs one per-tree prediction plus 1e6 multiply-adds,
  whatever the training size. It matches ranger's uncalibrated variance to
  3.6e-14.
- ranger fits the calibration prior to whatever batch it is given, so the same
  cell gets different values in different batches. Here the prior and the Monte
  Carlo noise scale are fitted once per forest on a seeded sample of 100k cells
  in the watershed, and every cell uses them. Both match ranger's to 1.2e-7 on
  the same cells.
- ranger's `calibrateEB` passes `q =` to `quantile()`, which ignores it, so it
  interpolates the calibrated variance linearly between the batch's quartiles
  and its maximum. Across the top quartile that overstates the standard error.
  `se_fdp.R` evaluates the exact posterior mean instead
  ([imbs-hl/ranger#792](https://github.com/imbs-hl/ranger/issues/792), open).

Before calibration ~40% of cells have a negative variance, so the calibration
does real work. The paper's maps carry the interpolation artefact. On the port
of the released points around Charleston, stock ranger run on the paper's
panel as one batch puts 22% of cells at or above 0.05, against 6% here, with
the same median (0.038). Its map has the speckle seen in the paper's
Figure 8a.

HUC 03 (`noaa-200612-202512`, planar; 67.6M cells) took 68 min locally on 18
workers, including a 165 s calibration fit. Its standard errors have median
0.047, p90 0.060 and p99 0.091, and barely vary with the probability (median
0.047–0.048 in every 0.2 band). The calibration shrinks each cell toward a
floor near 0.04, so spatial structure is weak.

The jackknife measures only the forest's resampling noise around one training
set. It leaves out the absence draw and the choice of best CV fold, which
together move the map about as much as six extra years of reports.

### Served store

The flood pipeline's `score` stage (`ocr.risks.flood.refit`) mosaics the
watersheds of the run it pins (`RUN`, `DISTANCE`) into
`output/flood/<version>/store.icechunk`, in the layout documented in
[the data schema](../../../docs/reference/data-schema.md#flood-risk-variables).
It reads one band of one watershed at a time, so it needs no large VM, and
checks the sum identity, dominant driver and point group at sampled cells of
each watershed before committing. `pyramid`, `buildings` and `regions` then run
as for any risk:

```bash
ocr run runs/flood/v2.0.0.toml
```

obstore reads AWS credentials from the environment only, so locally export
them first with `eval "$(aws configure export-credentials --format env)"`.

The ocr-web branch `Shane98c/fdp-retrained` reads an earlier HUC 03 mosaic at
`ocr-explore/FDP/processed/retrained/noaa-200612-202512-planar/`.

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

## Caveats

- **Probability of damage, not depth or loss.** The response variable is NOAA
  Storm Events flood-damage reports (presence) against sampled absence points,
  so the layer ranks where damage has been reported and is likely, not how much.
  It carries no depth, frequency, or return-period information.
- **Trained on reported damage.** Presence points inherit the reporting biases
  of the NOAA record: populated, well-observed places are better represented.
- **Historical.** The predictors (2016 land cover, road density, floodplain,
  terrain) describe recent conditions; there is no climate scenario dimension.
- **Continuous vs. binary.** The published product also ships a binary
  presence/absence raster, `Output_CONUS_binary_pres_abs_100m`. Its cut is not
  one number: `Code_FDP_RandomForest_code.R` fits a separate error-minimizer
  threshold per HUC-2 watershed, the probability where that watershed's false
  positive and false negative rates cross, and the release ships the code but
  not the resulting values. No client-side threshold reproduces it, so the
  flood store carries only the continuous surface.
