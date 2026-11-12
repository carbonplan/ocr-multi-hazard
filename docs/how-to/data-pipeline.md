# Data pipeline

Each risk runs as an ordered list of stages: `score`, `pyramid`, `buildings`, `regions`. A risk without a `score` has its store written by its own ETL, run first. Storage is the hand-off: each stage reads what an earlier one wrote. `pyramid` and `buildings` each read only the risk store (`store.icechunk`); `regions` reads the buildings file. A stage runs in process, or as one Coiled batch job with `--coiled`.

Fire still uses the older `ocr-deploy` CLI (`ocr-deploy --help`) until its refactor.

## Run a deployment

A deployment is one committed TOML file under `runs/<risk>/`:

```bash
ocr run runs/flood/v2.0.0.toml --dry-run      # print the plan, check inputs, run nothing
ocr run runs/flood/v2.0.0.toml                # or: pixi run flood-all
```

```toml
risk = "flood"
version = "v2.0.0"
stages = ["pyramid", "buildings", "regions"]
coiled = true

[kwargs.buildings]          # keyword args of the stage's utils, checked before any job starts
point_strategy = "ST_Centroid"

[job.buildings]             # optional coiled.batch.run overrides, over `ocr.cli.JOBS`
vm_type = ["r7g.8xlarge"]
```

## Stage kwargs

`[kwargs.<stage>]` in a run file, or `--kwargs '{"<stage>": {...}}'`, sets keyword args of the stage's utils (`STAGE_KWARGS` in `ocr/stages.py`): score the risk's own (`RiskSpec.score`); pyramid `write_multiscales`, `create_pyramid`; buildings `write_buildings`, `write_buildings_lod`; regions `write_regions`. Names and types are checked before any job starts; a bad name fails and the error lists the valid ones. `--kwargs` on `ocr run` merges key by key over the file's `[kwargs.<stage>]`.

| kwarg | default | meaning |
| --- | --- | --- |
| `score.run` (flood) | `noaa-200612-202512` | FDP refit model run to mosaic |
| `score.hucs` (flood) | all 18 | HUC-2 watersheds, e.g. `["03"]` (a subset, see below) |
| `pyramid.levels` | `"auto"` | number of levels, or `"auto"` |
| `pyramid.branch` | `"main"` | icechunk branch to write |
| `buildings.bbox` | CONUS | lon/lat subset (see below) |
| `buildings.point_strategy` | `"ST_Centroid"` | or `"ST_PointOnSurface"` |
| `buildings.memory_limit`, `.workers` | DuckDB's | DuckDB limits for the LOD step |
| `regions.levels` | all | e.g. `["states", "counties"]` |

`[job.<stage>]` takes any `JobConfig` field (`ocr/cli.py`), e.g. `vm_type`, `disk_size`, `env`, `tag`, `spot_policy`, `job_timeout`, over the defaults in `ocr.cli.JOBS`.

## Rerun some stages

Rerun with the same file's settings, using `--stages` and `--kwargs`:

```bash
ocr run runs/flood/v2.0.0.toml --stages regions --kwargs '{"regions": {"levels": ["counties"]}}'
```

Or name everything on the command line:

```bash
ocr stage --version v2.0.0 --coiled flood pyramid
```

`ocr run --help` and `ocr stage --help` list more examples.

- Stages always run in pipeline order, whatever order you give them.
- A failed stage stops the run; later stages don't start. Rerun from it with `--stages`.
- A stage you skip must already have its input in storage, or the run fails before it starts. So must TIGER (`pixi run regions-etl`) for `buildings` and `regions`.
- A stage the risk has no code for also fails before any job starts.
- `--overwrite` deletes a stage's old output first, where that applies (`score`, `pyramid`, `regions`). `buildings` and `regions` replace their files on every run; pass `--overwrite` to `regions` after a smaller run (`bbox`, fewer `levels`) so no old files stay.
- Coiled jobs print their job and cluster id and a `coiled batch logs <cluster id>` command when submitted.
- `--vm TYPE` sets the Coiled VM type for every stage in the run.

## Storage and environment

| variable | meaning |
| --- | --- |
| `OCR_STORAGE_BUCKET`, `OCR_STORAGE_REGION` | S3 bucket and region. Forwarded to Coiled jobs. |
| `OCR_STORAGE_LOCAL_ROOT` | Write under a local directory instead of S3. Not allowed with `--coiled`. |

A run file can set `prefix = "..."`, the output prefix (default `output`; inputs stay under `input/`). The prefix has no env var: it comes from the run file or `ocr stage --prefix`, and Coiled jobs get it as `--prefix`.

Scratch runs: use a prerelease version (`v2.0.0-scratch`, `v0.0.0-nrh`), or a scratch `prefix`. The version must be semver with a leading `v`.

Local smoke run: `OCR_STORAGE_LOCAL_ROOT=~/ocr-data pixi run flood-local` (`runs/flood/local.toml`).

Subset: `hucs` on flood's `score` builds the store from only those watersheds, cropped to them. `bbox` (lon/lat xmin, ymin, xmax, ymax) on `buildings` keeps only buildings inside it, and `regions` then covers only those, with partial stats for regions cut by the box; pair it with `hucs` so buildings outside the watersheds aren't sampled as empty. So a subset never replaces production outputs, `hucs` and `bbox` need a scratch `prefix` or a prerelease version (the run fails before it starts otherwise):

```bash
ocr stage --version v2.0.0-dev --kwargs '{"buildings": {"bbox": [-122.5, 37.7, -122.3, 37.8]}}' flood buildings
```

One watershed end to end on Coiled (South Atlantic-Gulf, buildings around Charleston):

```bash
ocr stage --version v0.0.0-fdp-test --prefix scratch/<you> --coiled \
  --kwargs '{"score": {"hucs": ["03"]}, "buildings": {"bbox": [-80.2, 32.6, -79.8, 33.0]}}' \
  flood score pyramid buildings regions
```

## How it works

`ocr run` and `ocr stage` (`ocr/cli.py`) run a risk's stages one at a time, in `STAGES` order (`ocr/stages.py`). There is no graph and no scheduler, only a for-loop in `execute()`. Paths are under `<prefix>/<risk>/v<version>/` (prefix default `output`) (`OutputPaths.store`, `.buildings`, `.regions`).

| stage | util | reads | writes |
| --- | --- | --- | --- |
| score | `RiskSpec.score` | risk inputs (not checked) | `store.icechunk` group "0" |
| pyramid | `write_multiscales` | `store.icechunk` | `store.icechunk` groups 1..N |
| buildings | `write_buildings` | `store.icechunk` | `vector/buildings.geoparquet` |
| regions | `write_regions` | `vector/buildings.geoparquet` | `vector/regions/` |

The store is written by `score`, or by the risk's own ETL when it has no score. Flood's score (`ocr.risks.flood.refit`) mosaics the per-watershed outputs of the FDP refit (`input-data/tensor/fdp/`) into `fdp`, `fdp_se` and one `contribution_<driver>` per predictor; `buildings` samples all of them. Before anything runs, `preflight()` does the checks listed under [Rerun some stages](#rerun-some-stages), so a bad stage name, kwarg or subset never boots a VM.

## Change the pipeline

New stage: a function `(spec, out, overwrite, **kw)` in `STAGES` at its place in run order, plus its row in `preflight` if it reads another stage's output. New risk: `Risk` (`ocr/config.py`), a `RISKS` entry and `runs/<risk>/vX.Y.Z.toml`.

`test_run_files_dry_run` dry-runs every `runs/**/*.toml`. After a change, run:

```bash
pixi run -e dev pytest tests/test_cli.py tests/test_stages.py tests/flood/test_flood_etl.py
```
