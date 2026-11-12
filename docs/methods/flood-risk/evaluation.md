# Evaluation

These notebooks check the flood refit step by step, from its training points to the
buildings it reaches, and test the published map against reports made after it. Each one
reads its inputs from `s3://carbonplan-ocr/input/flood/` and runs with AWS credentials for
that bucket.

## [Training points](training-points.ipynb)

How NOAA reports become presence points and how pseudo-absences are matched to them, the
check that the rebuilt points reproduce the released ones over the released window, and
how many presences each watershed gains from the added years.

## [Refit against the published map](refit.ipynb)

Whether the port of the released R script reproduces the published surface, the served
run's cross-validation and agreement with the published surface in every watershed, and, in
one watershed, how much the map moves with the random draws compared with six more years of
reports.

## [Standard errors](standard-errors.ipynb)

The level and spread of `fdp_se` across watersheds, what sets its level, and how it varies
with `fdp` within one watershed.

## [Drivers and coverage](drivers-and-coverage.ipynb)

The checks on the per-predictor attribution, predictor importance by watershed, and, at
building locations, which buildings have no `fdp` and which predictor dominates.

## [Temporal validation](temporal-validation.ipynb)

The published map, frozen, against NOAA reports from June 2020 through December 2025, after
its training window: how well it ranks later report locations above background, by
watershed, year and event type. Its run record, with the archived results and input hashes,
is in `input-data/tensor/fdp/validation-2026-10-15/`.

```{toctree}
:hidden:
:maxdepth: 1

training-points
refit
standard-errors
drivers-and-coverage
temporal-validation
```
