# CHAZ event set: testing the EAD tail against the full ERA5 storm catalogue

The served CHAZ stores are six return levels per cell, and `chaz_damage.py`
integrates them into an expected annual damage that extrapolates past the
1000-year band (walkthrough §5). The scripts here rebuild the ERA5 event set
for CONUS in CLIMADA so the damage–frequency curve is exact per cell, and
the extrapolation convention can be scored where it is applied, re-scored at
the scenario stores' sample size, and stressed under warming-like shifts.
Part 5 of the walkthrough reads the results below. Section 5 reproduces the
NA2 impact function the damage stores apply and refits it on corrected winds.
Section 6 runs the CHAZ model, CMIP6 winds and the map step on a laptop.

The hazard is built the way the published maps were (Meiler et al. 2026,
[CHAZ-hazard-maps](https://github.com/simonameiler/CHAZ-hazard-maps)):
CLIMADA's CHAZ reader, half-hour resampling, Holland (2008) winds with
CLIMADA's defaults, the published 300 arcsec land cells, one frequency per
basin (IBTrACS annual count over simulated storms with tropical-storm winds
in the basin), and empirical exceedance ranking with constant extrapolation.
It does not reproduce her numbers, and is not meant to: the published maps
carry winds from a CLIMADA version that double-counted translational
velocity (see `check` below), so our levels sit 10–15% under the Dryad
points by construction. A small residual sample difference is discussed
there too.

## Environment

CLIMADA pins Python below 3.13, so it lives in its own pixi environment:

```bash
pixi install -e climada
pixi run -e climada python input-data/tensor/chaz/climada/chaz_events.py --help
```

The Coiled software environment `ocr-climada` is that environment exported
for `linux-aarch64`:

```bash
pixi workspace export conda-environment -e climada -p linux-aarch64 \
  | pixi run -e climada python .github/scripts/export_pixi_to_conda.py \
      --stdin --output environment-climada.yaml --coiled-name ocr-climada --no-include-local-code
```

`chaz_tail_test.py` needs only numpy and `chaz_damage.py`, and runs in the
default environment.

Simona Meiler's own files behind the published ERA5 maps sit under
`s3://carbonplan-ocr/ocr-explore/CHAZ/published/`: the frequency-corrected
global hazard (`TC_global_0300as_CHAZ_ERA5_freq-corr.hdf5`, 33 GiB, the
event set the maps were ranked from), the exceedance impact output
(`TC_global_0300as_CHAZ_ERA5_corrected_exceedance_impact.nc`), and the
LitPop 2020 exposure (`litpop_0300as_2020_global.hdf5`). Section 4 covers
what is in the hazard file and how it is cropped to CONUS.

## 1. Hazard (`chaz_events.py`)

Everything lands under `s3://carbonplan-ocr/ocr-explore/CHAZ/hazard/ERA5/`
and is cached locally in `~/.cache/ocr/chaz-events` (`CHAZ_EVENTS_DIR`).

```bash
E="pixi run -e climada python input-data/tensor/chaz/climada/chaz_events.py"
$E centroids                       # the Dryad ERA5 points in the CONUS box, 127,602 cells
$E events --files 0-9              # per-file event tables and CONUS track subsets
$E hazard --file 0 --members 0-7   # one chunk locally (about 8,000 tracks, ~3 min on 8 cores)
$E batch --files 0-9               # the remaining chunks on Coiled, 4 members per task
$E levels --files 0-9 --years 1981-2019 --tag era5_10f
$E check --tag era5_10f            # ratio table and figure against the Dryad points
```

- `events` scans each track file once and records, per (member, storm):
  year, peak wind, and whether it carries tropical-storm winds within 5° of
  the CONUS box and in which basin. Only reaching storms go into the subset
  file CLIMADA reads. The writer blanks winds past a storm's end, since the
  raw fill dates otherwise survive as a spurious node and break resampling.
- `hazard` is one Coiled task: read a file's subset for a run of members,
  crop each track to the 5° buffer plus two nodes, resample, compute Holland
  winds on the CONUS cells across our own process pool, and write a CLIMADA
  HDF5. Tracks go through `TropCyclone.from_tracks` fifty at a time to keep
  memory near 3 GB. About 0.19 core-seconds per track; the ten-file set is
  roughly 370,000 tracks, or 20 CPU-hours. Distance to coast comes from the
  data API's 150 arcsec centroids, since the NASA raster CLIMADA expects is
  no longer hosted.
- `batch` submits chunks through `coiled.batch.run` on ARM spot VMs
  (m8g.xlarge, us-west-2). Each task pulls the script from our bucket,
  uploads its log to `logs/` every minute, and skips chunks already on S3.
  The 96 tasks for ten files ran in 29 minutes on 25 VMs.
- `levels` joins the chunks (deduplicating overlapping events), applies the
  year window, sets frequencies from the basin counts over the members held,
  and writes `sets/<tag>.hdf5` plus the six return levels and the 33 and 50
  m/s return periods to `levels/<tag>.nc`. Because the weight is an observed
  count over a simulated count, only the storm sample matters, which lets
  the tail test subsample freely.
- `check` compares our levels to the published ERA5 points as a median
  ratio per band, binned by the published 100-year wind. On the ten-file
  1981–2019 set (206,770 events) our winds sit a uniform 10–15% below the
  published maps in every band, and the 33 and 50 m/s return periods are
  2–4 times longer.

That gap is CLIMADA, not the tracks. Before CLIMADA 4.1.0 the Holland wind
models added the storm's translational velocity twice (fixed in PR #833,
January 2024); on 150 of our tracks, 3.3.2 and 4.0.1 give winds 12–16%
higher than 4.1.0 through 6.1.0. Simona Meiler confirmed (September 2026)
that the ERA5 fields were built on ~3.3.0-dev and the CMIP6 fields on
~4.0.0-dev, and the 2025 frequency correction did not touch winds. Every
published map therefore carries the double count. The tail test is internal
to our set and unaffected. A smaller residual remains: the published weights
imply ~18% more storms per basin than the shared files hold, worth a
question to the authors but worth only a percent or two in return levels.

## 2. Tail test (`chaz_tail_test.py`)

```bash
T="pixi run python input-data/tensor/chaz/climada/chaz_tail_test.py"
$T --tag era5_10f check                 # ranking reimplementation vs the CLIMADA levels
$T --tag era5_10f decompose             # truth vs reconstruction by T range, per cell
$T --tag era5_10f subsample --draws 50  # at the CMIP6 stores' design, 10 files x 8 members x 20 yr
$T --tag era5_10f perturb               # frequency x0.7/x1.3, wind x1.05/x1.10
```

Truth is the frequency-weighted sum of the impact function over events at
unit exposure, with the NA2 `v_half` of the chosen calibration (default
`TDR1.0`, the product's). Reconstruction is `chaz_damage.ead_from_levels`
on the six ranked levels in three tail variants: the shipped slope
extension, a flat tail, and truncation at rp_1000. Both sides are split into
T < 10, 10–1000 and > 1000 years.

`subsample` re-levels random subsets of (file, member) pairs with the same
ranking and weight rules, so each draw is a self-contained event set like
the ones behind the scenario stores. `perturb` rescales weights and winds in
place; scaling wind is a first-order stand-in for a hotter track set.

Outputs go to `hazard/ERA5/tail/`, named `<tag>_[<calibration>_]<experiment>`
with no calibration tag for TDR1.0 (the `CAL_TAG` convention of
`chaz_damage.py`); the parquets also carry a `calibration` column.

## 3. Results on the ten-file ERA5 set (`era5_10f`)

206,770 events reaching CONUS, 1981–2019, NA2 `v_half` 89.2 m/s (TDR1.0),
rarest resolved return period 18,344 years. Ratios are reconstruction over
truth; "weighted" is LitPop 2020 value over US cells.

|                          | truth tail share (T > 1000) | slope (shipped) | flat  | truncate |
| ------------------------ | --------------------------- | --------------- | ----- | -------- |
| unit exposure, all cells | 22%                         | 1.088           | 0.907 | 0.780    |
| LitPop-weighted, US      | 24%                         | 1.058           | 0.907 | 0.767    |
| rp_100 wind 26–30 m/s    | 57%                         | 1.126           | 0.690 | 0.441    |
| rp_100 wind 40–45 m/s    | 15%                         | 1.013           | 0.956 | 0.850    |
| rp_100 wind 50–55 m/s    | 7%                          | 1.012           | 0.994 | 0.940    |

Truth sits between the two conventions everywhere, closer to the slope
extension where damage lives. Within the bands (10–1000 yr) the
reconstruction is 1% low everywhere; the whole disagreement is the tail.
Warming-like rescaling (frequency ×0.7 and ×1.3, wind ×1.05 and ×1.10)
keeps the weighted slope ratio between 1.03 and 1.07 and the flat ratio
between 0.87 and 0.95, so the convention transfers to shifted distributions
at this sample size.

**At the scenario stores' sample size it does not.** Fifty draws at the
CMIP6 design (10 files × 8 members × 20 years, 1,600 simulated years, rarest
resolved return period ~2,000 years), each re-weighted and re-ranked:

| ratio to the draw's own truth  | median | IQR         |
| ------------------------------ | ------ | ----------- |
| slope (shipped), unit exposure | 1.248  | 1.227–1.260 |
| slope, LitPop-weighted         | 1.184  | 1.165–1.200 |
| flat, unit exposure            | 1.000  | 0.995–1.005 |
| flat, LitPop-weighted          | 0.996  | 0.989–1.001 |
| truncate, unit exposure        | 0.847  | 0.844–0.850 |

With 1,600 simulated years the 1000-year band rests on the top one or two
events and CLIMADA's constant extrapolation holds the curve flat beyond
them, so the true tail share is small (12%) while the slope extension keeps
climbing from a segment set by the rarest events.

**Verdict.** The slope extension is 6% high on ERA5 (18,000 simulated
years) and 18–25% high on the CMIP6 stores the product ships, where a flat
tail is unbiased. A flat tail past rp_1000 is the better single convention:
worst case 9% low on the ERA5 reference against 22% high on every scenario
store. Changing `ead_convention` in `chaz_damage.py` means rebuilding the
damage, v2 and buildings products, and `verify` will flag stores built under
the old convention.

Caveats: truth is our own event set, the draws use ERA5 storms at a GCM
sample size rather than GCM storms, and the tail contribution scales with
the impact function's curvature, so the ratios hold for NA2 at this `v_half`
and would tighten with a flatter function.

## 4. The published event set (`chaz_published.py`)

```bash
P="pixi run -e climada python input-data/tensor/chaz/climada/chaz_published.py"
$P submit                 # crop + match on Coiled, one VM each (~7 min crop, ~3 h Dryad check)
$P tailset                # set, levels and events files for the tail test, tag `published`
T="pixi run python input-data/tensor/chaz/climada/chaz_tail_test.py"
$T --tag published check  # ranking reimplementation vs the Dryad points
$T --tag published decompose
$T --tag published subsample --draws 50
```

The hazard file is one catalogue of 1,395,323 storms (10 track files × 40
members, 1981–2019, every track with wind) stored six times, once per
basin. Each copy carries that basin's constant frequency, one number per
basin, and winds only on that basin's cells; for CONUS the split is at
100°W, East Pacific storms west of it and North Atlantic storms east of
it. Dividing the IBTrACS annual counts by her frequencies gives the storm
counts she corrected to: 207,708 in the North Atlantic (4.8% above our
count over ten files) and 319,096 in the East Pacific (20% above). The
sample size matches ours; the remainder is how "in the basin" is decided.
The file is 35 GB of unchunked, uncompressed HDF5 (two CSR matrices, the
`fraction` one all ones), so `crop` runs on a VM in-region: it keeps the
128,242 columns that are CONUS cells and the 178,255 events with wind
there, and checks CLIMADA's ranking on the crop against the Dryad points.
The published values are reproduced to 4e-6 m/s in every band, so the
Dryad points are the crop's own six levels and serve as the levels file
for the tail test. Ranking 178,255 events on 128,242 cells took CLIMADA
2.8 hours; the tail test's numpy ranking does the same in minutes.

The tracks are not the ones she shared. `match` splits her catalogue into
its 400 member runs (the year sequence restarts at each member, and the
storm-date overlap between consecutive runs drops at every 40th, so the
order is file, member, storm) and tests each against all 40 shared track
files two ways: by the event dates CLIMADA's reader would assign (the first
node with a valid wind in that member, storms selected by the year of node
0), and, independently of dates, by the multiset of the 40 per-member storm
counts of each file against each of her ten file groups. No run matches
any member of any file, no file's member counts match any of her groups,
and the best date overlap is 36%, the chance level. Her February 2023
catalogue is a different sample from the June 2024 files, so a per-storm
comparison of her pre-4.1 winds with current CLIMADA's needs her original
tracks.

### Results on the published set

The same tests as section 3, with her catalogue as truth and the Dryad
points as the six levels, so the ratios score the ERA5 store exactly as
served. 178,254 events with CONUS wind, rarest resolved return period
19,232 years (median over cells).

|                          | truth tail share (T > 1000) | slope (shipped) | flat  | truncate |
| ------------------------ | --------------------------- | --------------- | ----- | -------- |
| unit exposure, all cells | 18%                         | 1.049           | 0.930 | 0.823    |
| LitPop-weighted, US      | 17%                         | 1.024           | 0.942 | 0.837    |
| rp_100 wind 26–30 m/s    | 58%                         | 1.128           | 0.682 | 0.426    |
| rp_100 wind 40–45 m/s    | 16%                         | 1.012           | 0.949 | 0.838    |
| rp_100 wind 50–55 m/s    | 8%                          | 1.010           | 0.990 | 0.931    |

Within the bands the reconstruction is under 1% low everywhere. The
bracket is tighter than on our rebuild: where damage lives the slope
extension is 2% high and a flat tail 6% low. Rescaling frequency ×0.7/×1.3
and wind ×1.05/×1.10 keeps the weighted slope ratio at 1.02–1.03 and the
flat ratio at 0.92–0.98.

Fifty draws at the CMIP6 design (10 files × 8 members × 20 years), each
re-weighted by the fraction of simulated years drawn and re-ranked:

| ratio to the draw's own truth  | median | IQR         |
| ------------------------------ | ------ | ----------- |
| slope (shipped), unit exposure | 1.137  | 1.130–1.142 |
| slope, LitPop-weighted         | 1.088  | 1.084–1.094 |
| flat, unit exposure            | 0.988  | 0.983–0.993 |
| flat, LitPop-weighted          | 0.996  | 0.992–0.999 |
| truncate, unit exposure        | 0.871  | 0.863–0.876 |

The true tail share drops to 11% at that sample size and the slope
extension overshoots it, by less than on our rebuild (9% weighted against
18%) since her winds sit higher on the impact function, but in the same
direction. The verdict of section 3 stands on the served data: a flat tail
past rp_1000 is 2–6% low on the ERA5 store and unbiased at the scenario
stores' sample size, while the slope extension is 2% high on ERA5 and
9–14% high at the scenario sample size.

## 5. The NA2 impact function (`eberenz_na2.py`)

The damage stores apply the NA2 `v_half` of Eberenz et al. (2021), 89.2 m/s
(TDR), fit on CLIMADA 1.4.1 winds. Winds rebuilt on CLIMADA 4.1.0 or later
need a refit, so `eberenz_na2.py` reproduces the published fit and repeats it
under other CLIMADA versions and track conventions.

```bash
E="pixi run -e climada python input-data/tensor/chaz/climada/eberenz_na2.py"
$E prep                                # events, paper damages, exposure, IBTrACS
$E wind --radius est --records 6h      # winds on CLIMADA 6.1
uv run --no-project --python 3.10 --with climada==4.0.1 --with 'numpy<2' --with 'pandas<2.2' \
  --with 'xarray<2024.3' --with 'scipy<1.13' --with 'shapely<2.1' --with 'geopandas<1' \
  --with 'pyproj<3.7' --with 'dask<2024.3' --with pyarrow \
  python input-data/tensor/chaz/climada/eberenz_na2.py wind --radius p141 --records 6h
F="pixi run python input-data/tensor/chaz/climada/eberenz_na2.py"
$F fit 4.0.1_p141_6h 6.1.0_est_6h      # fits, Table A2 statistics, per-event ratios
$F ead 80.3 70.2                       # CONUS EAD of the corrected-wind rebuild
```

Wind files are tagged `<CLIMADA version>_<radius>_<records>[_<pressure>][_penv1010][_dt<hours>]`
and land in `~/.cache/ocr/eberenz-na2`. The 43 NA2 events (40 USA, 3
Canada, 1980–2017) and their damages normalized to 2014 come from CLIMADA's
`tc_impf_cal_v01_EDR.csv`, the paper's simulated damage per event at the
default `v_half` of 74.7 from climada_papers. Tracks are IBTrACS v04r01, US
agency, with CLIMADA filling missing values (Iwa 1982 needs it; Luis 1995 was
renumbered after the paper). Exposure is CLIMADA's LitPop (2018, 150 arcsec)
summed to 300 arcsec and rescaled to the paper's 2014 produced capital.
`v_half` is swept from 25.8 to 325.7 m/s in 0.1 steps through the
piecewise-linear impact function of `chaz_damage._mdr`, and events without
simulated damage are dropped, as in the paper. CLIMADA 4.0.1, the last
release before the wind fix, stands in for 1.4.1.

### Reproduction on pre-fix winds

Two conventions of the original setup are in the code, not the paper.
CLIMADA 1.4.1 overwrote the radius of maximum wind at every record from
central pressure (15 nm at 950 hPa and below, 25 at 980, 50 at 1020, beside a
`# TODO: always extrapolate???!!!`); the paper and Geiger et al. (2018)
describe an observed radius with pressure as the fallback. And the
calibration read pre-processed tracks (`read_processed_ibtracs_csv`, not
published) whose time step Geiger et al. give as "usually 6 h". IBTrACS v04
adds 3-hourly and landfall records: Andrew's put 922 hPa over Homestead,
where the 6-hourly track interpolates to 944 hPa.

| CLIMADA 4.0.1, radius / records | TDR at 74.7 / 86.0 / 89.2 | `v_half` TDR / RMSF | Andrew ÷ paper |
| ------------------------------- | ------------------------- | ------------------- | -------------- |
| paper (Table A2)                | 2.11 / 1.16 / 1.00        | 89.2 / 86.0         |                |
| observed / all records          | 3.11 / 1.77 / 1.54        | 99.6 / 90.2         | 1.14           |
| 1.4.1 rule / all records        | 2.24 / 1.28 / 1.11        | 91.7 / 89.4         | 1.55           |
| 1.4.1 rule / 6-hourly           | 2.10 / 1.19 / 1.03        | 89.9 / 86.6         | 1.04           |

The remaining 0.7 m/s is inside the paper's own spread: the per-event file
in climada_papers sums to a ratio of 2.03 at 74.7 where Table A2 has 2.11.
The match holds in the total more than per storm (24 of 43 within 25% of the
paper's damage, Gloria at 1.8×), and the TDR fit only sees the total. A flat
1010 hPa environmental pressure moves the fit away (TDR 92.0, RMSF 15
against 8.7), so the processed tracks kept the agency's outer-isobar pressure.

### Refit and the CHAZ conventions

`v_half` absorbs the biases of the hazard it is fit with, so the refit should
represent the 43 storms the way the CHAZ hazard represents its own. CLIMADA's
CHAZ reader hands the wind model 6-hourly records with no radius (CLIMADA
estimates one from pressure), environmental pressure 1010 hPa, and central
pressure from `_estimate_pressure`, a regression on wind, latitude and
longitude. It passes CHAZ's 0–360 longitudes, while the regression was fit on
IBTrACS's −180–180: over the US agency's 1980–2019 records the regression is
2.1 hPa too deep in the western hemisphere with −180–180 longitudes and 14.8
hPa too deep with 0–360. Every Atlantic and East Pacific CHAZ storm reaches
the wind model 12.7 hPa deeper than the regression intends. The reader does
this in every version checked (3.0.0, 3.2.0, 4.0.1, 6.1.0), so it is in the
published maps and in our rebuild.

All with 6-hourly records:

| radius, central pressure, environmental pressure    | 4.0.1 TDR / RMSF | 6.1 TDR / RMSF | 6.1 events |
| --------------------------------------------------- | ---------------- | -------------- | ---------- |
| 1.4.1 rule, observed, observed                      | 89.9 / 86.6      | 68.0 / 60.3    | 40         |
| observed, observed, observed                        | 98.2 / 87.7      | 73.8 / 61.3    | 40         |
| estimated, observed, observed                       | 92.8 / 88.9      | 70.2 / 62.3    | 40         |
| estimated, from wind with 0–360, 1010 (CHAZ reader) | 102.0 / 123.3    | 79.1 / 85.9    | 43         |
| same at 0.5 h steps (ERA5 maps, our rebuild)        | 104.5 / 126.0    | 80.3 / 87.5    | 43         |
| estimated, from wind with −180–180, 1010            | 83.7 / 82.2      | 64.4 / 56.3    | 40         |

Scaling the published value by the reproduction's version shift (89.2 × 68.0
/ 89.9 = 67.5) lands within 0.5 m/s of the direct 6.1 fit under Eberenz's
conventions. Pressure from wind discards each storm's own pressure, and on
4.0.1 the per-storm fit degrades: 5 of 43 events within 25% of the paper's
damage (20 with observed pressure), Sandy halving and Irene dropping 40%
while Alicia, Ike and Katrina more than double, and the TDR and RMSF optima
21 m/s apart. Both criteria move the same way.

### Damage

`ead` scores the corrected-wind ERA5 rebuild of section 3 (CLIMADA 6.1, 0.5
h steps) against the served store (published levels × 89.2, 35.0 $bn/yr
over US cells, LitPop 2020), with the rebuild's per-event frequencies
rescaled to the published ones.

| `v_half` on the rebuild                    | national | TX–AL | FL   | GA–Carolinas | VA–ME |
| ------------------------------------------ | -------- | ----- | ---- | ------------ | ----- |
| 89.2 (shipped)                             | 0.38     | 0.37  | 0.41 | 0.27         | 0.29  |
| 80.3 (CHAZ reader conventions)             | 0.58     | 0.57  | 0.62 | 0.42         | 0.44  |
| 70.2 (estimated radius, observed pressure) | 1.01     | 1.00  | 1.09 | 0.74         | 0.78  |
| 68.0 (Eberenz's conventions)               | 1.15     | 1.14  | 1.24 | 0.85         | 0.90  |

The served store is not a consistent pairing either: its winds come through
the same reader, and the matching pre-fix fit, 104.5, takes it to 18.9
$bn/yr (×0.54). Paired consistently on both sides, the corrected rebuild at
80.3 sits 7% above that. In every pairing Georgia–Carolinas and
Virginia–Maine fall 25–30% relative to Florida and the Gulf. The two
catalogues are different storm samples, so cell-level ratios are noisy
(0.21–1.03, 5–95%, at 80.3 among the cells carrying 90% of the loss).

Not settled: whether a rerun keeps the reader's longitudes and fits with
them (79–80 m/s) or corrects them and fits with −180–180 (64.4 m/s at 1 h
steps), and how far the offset moves CHAZ winds directly, which a per-track
comparison of H08 peak winds with CHAZ's own `Mwspd` would show.

## 6. Running CHAZ ourselves (`chaz_downscale.py`, `chaz_tropicona.py`, `chaz_maptime.py`)

These scripts test, on a laptop, the three stages of a from-scratch rerun:
the CHAZ model itself, the Holland winds on CMIP6 tracks with both CLIMADA
fixes, and the published map step. Everything they write stays under
`~/.cache/ocr/chaz-local/`, and nothing goes to S3. The inputs are public:
the CHAZ tutorial's ERA5 2017 forcing and regressions (Zenodo 11631930,
CC BY 4.0) and 25 CMIP6 CHAZ track files from Chia-Ying Lee's TROPICONA
tutorial (Zenodo 12533388, CC BY 4.0). `fetch_chaz_local.sh` downloads
both records and checks their md5 sums.

```bash
L="pixi run -e climada python input-data/tensor/chaz/climada"
$L/chaz_downscale.py run --year 2017 --realizations 1-40 --ratio 1.13 --workers 12
$L/chaz_downscale.py compare --year 2017
$L/chaz_tropicona.py run --files all --workers 12     # --pressure raw for the reader's pressure
$L/chaz_tropicona.py compare --file CESM2_Global_2100_2ens000_SD
$L/chaz_tropicona.py summary
$L/chaz_maptime.py run --tile=-90,-75,25,40 --sample 200
```

### The CHAZ model on ERA5 2017

`chaz_downscale.py` ports the CHAZ downscaling tutorial notebook to a script.
It changes the numpy 2 spellings, turns the synchronous dask calls into
loops, and draws new genesis seeds per track realization, as `CHAZ.py` does.
The seeding ratio rescales genesis-index seeds to the observed storm count,
and computing it needs 1981–2005 genesis index, which the Zenodo record
lacks. 1.13 matches the published files' 2017 trajectory count (140 against
124 seeds at ratio 1).

| 2017, per track realization (mean over 40 members) | ours, 40 realizations | published, 10 files |
| -------------------------------------------------- | --------------------- | ------------------- |
| storms with wind                                   | 88.5 ± 3.4            | 92.8 ± 4.7          |
| hurricanes / major                                 | 40.0 / 17.7           | 42.0 / 18.8         |
| lifetime peak, median / 90th percentile (kt)       | 60.5 / 115.3          | 60.4 / 114.7        |
| North Atlantic / East Pacific / West Pacific       | 12.1 / 14.7 / 26.6    | 11.8 / 18.2 / 25.6  |

The intensity distribution matches. Counts run about 5% low, and the East
and South Pacific are thinner than in the published files, which may be the
seeding ratio or differences in CHAZ version.

One realization-year with 40 intensity members took 79 s per process with
12 running together: 38 s for tracks, 10 s for predictors and 30 s for
intensity. Alone it takes about 50 s. The Colab timing in the notebook
(317 CPU-s) is about four times slower.

### CMIP6 winds with both fixes

`chaz_tropicona.py` runs the CMIP6 wind step of the published maps on the
25 TROPICONA files, on CONUS cells. It uses her members (0, 5, …, 35), her
base (1995–2014) and end-of-century (2081–2100) windows, and her 1 h step,
on CLIMADA 6.1. It also re-estimates central pressure from wind with
longitudes wrapped to −180..180. That adds exactly 12.73 hPa at every
western-hemisphere node and changes nothing else. A 38 kt storm goes from
988 hPa under the reader to 1000 hPa.

The longitude bug matters more on CONUS cells than on lifetime peaks. On
CESM2 SD (ens000), the reader's pressure raises each track's peak CONUS wind
by a median 25% (5–95%: 8–51%), and 22% among tracks at 33 m/s or more.
Twice as many tracks reach 33 m/s (1,018 against 507). The offset is fixed
in hPa, so it weighs most on weak and overland storms. On ERA5 lifetime
peaks the effect is 14%.

Storms per member-year with CONUS winds of 33 m/s or more, before frequency
correction (`summary`):

| model        | CRH base | CRH 2081–2100 | SD base | SD 2081–2100 |
| ------------ | -------- | ------------- | ------- | ------------ |
| CESM2        | 2.96     | 1.82          | 2.73    | 0.55         |
| CNRM-CM6-1   | 3.71     | 5.99          | 3.54    | 2.84         |
| EC-Earth3    | 3.02     | 4.91          | 2.58    | 2.53         |
| IPSL-CM6A-LR | 3.23     | 4.33          | 3.24    | 1.68         |
| MIROC6       | 2.38     | 2.82          | 2.45    | 1.46         |
| MRI-ESM2-0   | 3.86     | 3.68          | 3.14    | 1.86         |

CRH mostly increases and SD decreases, the two diverging genesis scenarios
of Lee et al. (2020). The files don't state their scenario. Two of ten
realizations per model leave these as samples, not hazard levels.

The 105,254 cropped tracks took 0.21 core-s each at 1 h steps, with 12
workers. That compares with 0.29 on Coiled for the ERA5 CONUS set at 0.5 h.

### Both fixes with a matched damage curve

`chaz_events.py` takes `--pressure wrapped` on `hazard`, `batch` and `levels`,
and `--years` restricts a hazard run to the analysis window. The rebuild
`era5_10f_wrapped` is the `era5_10f` event set (206,770 events, files 0–9,
1981–2019) with pressure re-estimated from wrapped longitudes. The pressure
fix halves the cells above 17.5 m/s per event, from 500 to 244. `levels
--workers` ranks blocks of cells in parallel and drops the events with no
wind in each block. On a test tile it matches CLIMADA's serial call exactly.

The matched curve is fit with `eberenz_na2.py wind --radius est --records 6h
--pressure wind180 --penv 1010 --time-step 0.5`, the conventions of CHAZ
winds once both fixes are in. TDR gives 65.4 and RMSF gives 51.3. The fit is
weak: 5 of 43 events land within 25%, and Sandy, Irene, Hugo and Irma
simulate almost no damage, because pressure estimated from wind can't
represent large storms with modest peak winds.

`eberenz_na2.py ead --tag era5_10f_wrapped` scores the result against the
served store (35.0 $bn/yr):

| pairing                                        | national | TX–AL | FL   | GA–Carolinas | VA–ME |
| ---------------------------------------------- | -------- | ----- | ---- | ------------ | ----- |
| both fixes × 89.2 (shipped curve)              | 0.15     | 0.14  | 0.16 | 0.09         | 0.09  |
| both fixes × 65.4 (matched, TDR)               | 0.53     | 0.50  | 0.58 | 0.32         | 0.34  |
| both fixes × 70.2 (observed-pressure fit)      | 0.39     | 0.37  | 0.43 | 0.24         | 0.25  |
| both fixes × 51.3 (matched, RMSF)              | 1.51     | 1.45  | 1.67 | 0.98         | 1.05  |
| reader's pressure × 104.5 (matched, section 5) | 0.54     |       |      |              |       |

Each wind convention paired with its own curve gives 18–19 $bn/yr: 18.4 with
both fixes and 18.9 with the reader's pressure. The calibration storms are not
the benchmark for that number, since they cover only 40 storms in 2014
dollars. The observed-storm test below is.

### Observed storms through the same pipeline (`chaz_observed.py`)

The 654 IBTrACS storms of 1981–2019 that come within reach of CONUS get
Holland winds on the same cells, at 0.5 h on CLIMADA 6.1, under three
conventions:

- `track`: observed pressure and radius, every record. The most realistic.
- `chaz`: CHAZ conventions with both fixes.
- `chaz360`: CHAZ conventions with the reader's longitudes.

`compare` counts storms per year with peak wind at or above a threshold on US
land cells (LitPop value > 0). Observed rates carry a 90% Poisson interval.

| storms/yr ≥ 33 m/s on US land  | US               | TX–AL | FL   | GA–Car | VA–ME |
| ------------------------------ | ---------------- | ----- | ---- | ------ | ----- |
| observed, `track`              | 1.18 (0.91–1.51) | 0.49  | 0.49 | 0.56   | 0.18  |
| observed, `chaz`               | 1.36 (1.07–1.71) | 0.64  | 0.51 | 0.49   | 0.10  |
| observed, `chaz360`            | 2.33 (1.95–2.78) | 1.03  | 1.03 | 0.92   | 0.26  |
| CHAZ published (served winds)  | 2.09             | 1.05  | 0.82 | 0.63   | 0.30  |
| CHAZ `era5_10f` (#833 fixed)   | 1.35             | 0.67  | 0.52 | 0.38   | 0.13  |
| CHAZ `era5_10f_wrapped` (both) | 0.65             | 0.32  | 0.25 | 0.17   | 0.06  |

Under identical wind construction, CHAZ gives about half the observed rate:
0.65 against 1.36 with both fixes, and 1.35 against 2.33 with the reader's
longitudes. At 17.5 m/s the ratio is 0.73–0.83, and at 43 m/s it is
0.41–0.50. Part of this is the frequency correction. It targets 10.8 North
Atlantic and 14.5 East Pacific storms a year, while IBTrACS 1981–2019 has
12.9 and 17.6 storms with 34 kt winds in each basin, about 20% more. The rest
is CHAZ's ERA5 landfall climatology. The two CLIMADA bugs inflate winds by
enough to offset it. The served winds land 1.8× above the realistic observed
rate at 33 m/s, and the #833-only rebuild sits closest to it.

`ead` scores damage from the events themselves, with each wind convention's
own TDR fit. It uses the same `_mdr` and LitPop 2020 values throughout. The
`track` fit has to match the observed winds' 0.5 h step. Refit at 0.5 h on all
43 calibration storms, it is 77.6 (75.4 at 1 h, which inflates the benchmark
by 13%).

| $bn/yr                         | US   | TX–AL | FL   | GA–Car | VA–ME |
| ------------------------------ | ---- | ----- | ---- | ------ | ----- |
| observed `track` × 77.6        | 32.4 |       |      |        | 8.0   |
| observed `chaz` × 65.4         | 31.7 | 7.2   | 21.4 | 1.8    | 1.1   |
| served (published × 89.2)      | 34.1 | 10.2  | 13.8 | 3.0    | 4.8   |
| published × 104.5 (matched)    | 18.4 |       |      |        |       |
| CHAZ `era5_10f` × 80.3         | 20.5 | 6.1   | 9.0  | 1.3    | 2.1   |
| CHAZ `era5_10f_wrapped` × 65.4 | 18.2 | 5.2   | 8.3  | 1.0    | 1.6   |

The observed benchmark is 32.4 $bn/yr, with a 90% interval over resampled
years of 15.9–51.2. It is driven by Sandy, Andrew and Wilma. Every CHAZ set
paired with its own curve gives 18–20 $bn/yr, about 0.56 of the benchmark,
which is in line with the rate deficit. The served store lands near the
benchmark only because 89.2 is not the curve for its winds; matched, it gives
18.4. So the served hazard runs 1.8× high at 33 m/s, while its damage matches
history because of the curve mismatch.

Two caveats from a review of this comparison (Codex, 2026-10-30):

- `chaz_events` gives an event whose track touches both basin outlines the
  sum of both weights over its whole footprint. The published file instead
  applies each basin's weight on its own side of 100°W. Splitting the
  footprint the same way lowers the wrapped 33 m/s rate from 0.65 to 0.62 and
  its EAD from 18.2 to 17.4.
- The published set is a different track realization with different basin
  denominators. The published-to-rebuild step therefore mixes the #833 fix
  with a change of sample. Only the `era5_10f` → `era5_10f_wrapped` step
  isolates a bug (the longitude one).

The CHAZ wind conventions can't reproduce Sandy. Its matched-curve loss falls
from 262 to 6 $bn, mostly because pressure is inferred from wind and the
environmental pressure is set to 1010 hPa. That collapses the observed
storms' Virginia–Maine damage (8.0 → 1.1). It doesn't show that no synthetic
catalogue can carry the region's loss: the wrapped CHAZ set gives 1.6 there.

A deficit of CHAZ landfalls in the Gulf and the eastern US was already
reported by Lee et al. (2018, §5.2), who matched observed 35 kt return
periods with a regional frequency adjustment. The published maps apply only
the basin-wide correction.

### The published frequency rule on corrected winds (`chaz_freqnorm.py`)

The published correction divides each basin's observed count by N, the number
of storms with any wind (at least 17.5 m/s) on a cell inside the basin. That
cell set includes about 22,000 coarse ocean cells. `chaz_freqnorm.py count`
counts N her way on global CLIMADA 6.1 winds for the same tracks: four member
runs of file 0, 1981–2019. It uses the Oct 13 rerun chunks for the reader's
pressure and `chaz_rerun.py wind --pressure wrapped` for the fix
(1.21 core-s per track on the laptop).

| storms per member-year counted    | NA    | EP    | NI   | SI    | SP    | WP    |
| --------------------------------- | ----- | ----- | ---- | ----- | ----- | ----- |
| published file (pre-#833 winds)   | 13.31 | 20.45 |      |       |       |       |
| footprint, 6.1, reader's pressure | 12.96 | 19.21 | 3.73 | 17.24 | 15.74 | 28.91 |
| footprint, 6.1, wrapped pressure  | 10.76 | 16.48 | 3.73 | 17.24 | 14.86 | 28.91 |
| along-track count (`chaz_events`) | 12.65 | 17.00 |      |       |       |       |

With the fix, 17% fewer North Atlantic storms leave a footprint, so the
published rule would raise each one's weight by 1.205. The East Pacific
weight rises by 1.166 and the South Pacific by 1.060 (only its east side
crosses 180°), and the others don't change. The published file used a
different track sample, so its counts compare only loosely: #833 had already
trimmed N by 3–6%.

Reweighting `era5_10f_wrapped` by this rule (NA × 1.176 and EP × 1.032 over
the along-track weights) gives, on US land:

| storms/yr; $bn/yr                     | ≥ 17.5 | ≥ 33 | ≥ 43 | EAD × 65.4 |
| ------------------------------------- | ------ | ---- | ---- | ---------- |
| as rebuilt                            | 2.58   | 0.65 | 0.22 | 18.2       |
| published rule on corrected winds     | 2.98   | 0.76 | 0.26 | 21.0       |
| plus IBTrACS 1981–2019 ≥ 34 kt counts | 3.56   | 0.91 | 0.31 | 25.1       |
| observed storms, `chaz` conventions   | 3.54   | 1.36 | 0.54 | 31.7       |

The normalization rule and storm-count targets close the gap at
tropical-storm strength (3.56 against 3.54). The rest of the deficit is in
intensity: CHAZ storms reaching the US are too weak, giving 0.67 of the
observed rate at 33 m/s and 0.57 at 43 m/s. The 12.9 and 17.6 targets use a
34 kt any-node definition, and Bloemendaal's constants count genesis at about
40 kt, so the last row is a sensitivity, not a correction.

### What else was tuned or validated against the bugged winds

Ranked by evidence and likely effect (Codex, 2026-10-30):

1. **Regional damage curves and their uncertainty ranges.** Every Eberenz
   curve (NA1, NA2, NI, OC, SI, WP1–4, global) was fit on winds with the
   double count, and CLIMADA ships them unchanged. NA2's per-event IQR goes
   from 74–115 m/s on 1.4.1 to about 42–82 under the corrected CHAZ
   conventions. So the uncertainty samples in Meiler's pipeline, and the
   bounds in `chaz_damage.py`, are off too.
2. **Meiler's frequency denominator depends on the winds** (measured
   above: about +20% North Atlantic weight once the pressure is fixed).
   `freq_corr_era5.py` and `CHAZ_wind_concat_freq.py` divide the observed
   count by the number of storms with a nonzero footprint (at least 17.5 m/s
   on some cell) in the basin. Fixing the winds shrinks that count and raises
   each storm's weight, a partial automatic compensation. `chaz_events`
   counts along-track `Mwspd` instead, so the rebuilds above leave out that
   compensation.
3. **CHAZ's landfall deficit**, known since 2018, which the inflated winds hid.
4. **Pressure from wind, pressure-derived radius and 1010 hPa.** These
   regressions are fit to IBTrACS, not to CLIMADA winds, but they lose large
   storms. Fixing the longitude also widens the estimated radius by 18–38%,
   so the fix is not a uniform wind multiplier.
5. **Validation that shared the errors**: Meiler et al. 2022 (CLIMADA 3.1.2
   on both the synthetic and the historical sets), the ERA5-against-GCM map
   checks, and our tail convention. The slope tail's ratio to event EAD moves
   from 1.024 on the published winds to 1.058 after the #833 fix alone.

Not compensations: CHAZ's intensity regression, genesis, survival ratio and
beta drift, which are fit to best tracks, and the Holland-B, surface-factor
and translational-attenuation formulas. Two confounds remain. CLIMADA 6.1
clips Holland B to [0.81, 2.025] where 4.0.1 and 4.1.0 used [1, 2.5], so the
4.0.1 → 6.1 refit shift is not all #833. And the pressure regression is fit
on |lat| but evaluated on signed latitude, about 2.2 hPa at 20°S.

### The map step

The published map scripts select a tile from the global hazard and call
`local_exceedance_intensity` and `local_return_period`. For every cell those
densify the cell's column over all event rows, and a tile keeps every row.
`chaz_maptime.py` times that loop body on tiles of the CONUS crop padded with
empty rows to her row counts. The replica agrees with a real CLIMADA call to
within 10%.

| rows per cell                     | 15° SE tile, ms/cell | 5° S Florida tile, ms/cell |
| --------------------------------- | -------------------- | -------------------------- |
| events with wind in the tile only | 21                   | 6                          |
| 1,395,323 (one catalogue)         | 41                   | 31                         |
| 8,371,938 (the six basin copies)  | 175                  | 168                        |

That is about 20 ms per cell per million rows, plus a term that grows with
the tile's nonzeros. On the published ERA5 file, either metric works out to
about 45 serial hours over the 929,389 global cells. The numpy ranking
(`chaz_tail_test.Ranked`) matches CLIMADA to 2e-9 m/s and ranks the whole
15° tile in 57 s, against about 6 minutes for CLIMADA without the empty rows.
