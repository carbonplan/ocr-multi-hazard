# Flood risk methods overview

The flood layers are CarbonPlan's refit of the USGS flood damage probability (FDP) model of
[Collins et al. 2022](../../reference/data-sources.md#collins-et-al-2022). FDP is a
random-forest score, from 0 to 1, of how likely a 100 m cell is to see a reported flood
damage event, on the CONUS Albers grid (EPSG:5070). The refit trains the published model
again on NOAA reports through December 2025 and adds a standard error and a per-predictor
attribution to every cell. How well it does is checked in [Evaluation](evaluation.md).

## Summary

1. **Training points.** Presences are NOAA Storm Events reports of Flood or Flash Flood,
   December 2006–December 2025, one per 100 m cell at the report's begin coordinates.
   Pseudo-absences are sampled in equal number, matched to the presences by HUC-2 watershed
   and land-cover class, outside the combined FEMA/EPA 100-year floodplain and off presence
   cells ([Training points](training-points.ipynb)).
2. **One forest per watershed.** Each of the 18 HUC-2 watersheds (the two-digit USGS
   hydrologic regions) gets a ranger probability forest of 1,000 trees, tuned with
   `tuneRanger` (70 iterations, Brier score) across four cross-validation folds
   ([Refit against the published map](refit.ipynb)).
3. **Every cell.** Each forest predicts `fdp` for its watershed's cells, splits it into one
   contribution per predictor, and gives it a standard error.
4. **The store.** The `score` stage mosaics the watersheds into `store.icechunk`; `pyramid`,
   `buildings` and `regions` then run as for every risk ([data schema](../../reference/data-schema.md)).

## Limits of the score

The forests train on equal numbers of presences and pseudo-absences, and the
pseudo-absences are kept out of the floodplain, so `fdp` ranks cells by how much they
resemble places with reported damage. It is not a calibrated annual probability, a depth,
or a loss. Each watershed's baseline, the mean prediction its attribution starts from, sits
near 0.5 for the same reason.

Presences are reports, so they carry the reporting biases of the NOAA record: populated,
well-observed places are better represented. The 100-year floodplain has no
pseudo-absences, so every forest scores it near 1. The predictors (2016 land cover and
road density, precipitation averaged over 2006–2019, floodplain, terrain) are held at the
published vintages; there is no climate scenario.

The published release also ships a binary presence/absence raster. The released code cuts
each watershed at its own error-minimizing threshold, and the thresholds are not released,
so the store carries only the continuous score.

## Predictors

The 11 predictors of the published model, unchanged: floodplain, distance to stream,
elevation, height above nearest drainage, land cover, profile curvature, hydric soils,
topographic wetness index, channelization, road density, and mean annual maximum three-day
precipitation. Their names in the store are listed in the [data schema](../../reference/data-schema.md).

## Differences from the published model

`train_fdp.R` ports the released script `Code_FDP_RandomForest_code.R` and keeps its
modelling steps, including its fold handling: the final model trains on the best CV fold's
training split. The refit differs in:

- **Training points** rebuilt from NOAA through December 2025. Over the released window
  they cover 99.6% of the released presence cells.
- **Distances for spatial sorting.** The released script calls dismo's `ssb()` and
  `pwdSample()` with their default `lonlat = TRUE` on Albers metre coordinates, so the
  distances it uses to keep test points away from training points are haversine values of
  metres read as degrees, unrelated to true separation. The refit measures true distances
  (`--distance lonlat` reproduces the released script). Correcting them changes the
  cross-validation folds, and through the choice of best fold it can change the map. On the
  released points it left the map of HUC 03 and HUC 06 unchanged and lowered their CV AUC by
  0.06–0.07, so the released script's AUCs, like the paper's 0.750, are measured on test
  points close to their training points.
- **Watersheds.** The released code expects a HUC-2 raster that is not in the release; the
  refit rasterizes the USGS Watershed Boundary Dataset's HUC-2 regions onto the predictor
  grid.
- **Seeds.** Point sampling, folds and tuning are seeded, so a run can be repeated.

## Driver attributions

`contribution` splits every cell's `fdp` into one term per predictor along the forest's tree
paths (Saabas attribution): per cell, the watershed's baseline plus the contributions equals
`fdp`. `dominant_driver` is the predictor with the largest contribution. Two checks fail a
watershed's run: the node values the attribution derives must reproduce the leaf values
ranger stored, and the reconstructed probabilities must match the forest's prediction.

The attributions explain the model, not causation. Correlated terrain predictors share
credit arbitrarily, and inside the floodplain nearly all of the score is credited to the
floodplain, for the reason above.

## Standard errors

`fdp_se` uses the method of Collins et al. (supplement 1.1.2): ranger's infinitesimal
jackknife with its empirical Bayes calibration. The paper mapped it for four cities; the
refit computes it for every cell, and differs from calling `predict(type = "se")` in three
ways:

- The jackknife variance is computed from the B × B Gram matrix of in-bag counts, so a cell
  costs one per-tree prediction plus 1e6 multiply-adds, whatever the training size. It
  matches ranger's uncalibrated variance to 3.6e-14.
- ranger fits the calibration prior to whatever batch of cells it is given, so the same
  cell gets different values in different batches. Here the prior and the Monte Carlo noise
  scale are fitted once per forest on a seeded sample of 100,000 cells in the watershed, and
  every cell uses them. Both match ranger's to 1.2e-7 on the same cells.
- ranger's `calibrateEB` passes `q =` to `quantile()`, which ignores it, so it interpolates
  the calibrated variance linearly between the batch's quartiles and its maximum, which
  overstates the standard error across the top quartile. The refit evaluates the exact
  posterior mean ([imbs-hl/ranger#792](https://github.com/imbs-hl/ranger/issues/792)).

On the released points around Charleston, stock ranger run on the paper's panel as one
batch puts 22% of cells at or above 0.05, against 6% with the exact posterior mean, with
the same median (0.038); its map has the speckle of the paper's Figure 8a.

The jackknife measures only the forest's resampling noise around one training set. It
leaves out the pseudo-absence draw and the choice of best fold, which in HUC 03 move the map
outside the floodplain nearly as much as six more years of reports ([Refit](refit.ipynb#3-what-moves-the-map)).
Its level depends on each watershed's tuned settings, so a standard error compares cells
within a watershed, not across watersheds ([Standard errors](standard-errors.ipynb)).
