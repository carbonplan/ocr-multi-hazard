# Hurricane risk methods overview

## Summary

The aim is global tropical-cyclone wind hazard maps for the recent climate and for future climate
scenarios. We follow the approach of [Meiler et al. (2026)](https://doi.org/10.1038/s41597-025-06452-0):
synthetic storms from CHAZ, the Columbia statistical-dynamical tropical-cyclone hazard model, turned
into wind fields with CLIMADA. CHAZ makes many synthetic storm tracks with wind histories from the
large-scale environment. We implement each step ourselves so that every step can be rerun and
checked.

1. **Forcing** (ERA5 1981–2019 done). Build the environmental fields CHAZ reads, from ERA5 and from
   CMIP6 climate models.
2. **Genesis** (ERA5 1981–2019 done). Compute the tropical-cyclone genesis index (TCGI), which sets where
   and how often CHAZ seeds storms.
3. **Tracks and intensity** (ERA5 1981–2019 runs done, with the public code's settings). Run CHAZ.
4. **Climate models** (one model so far). Repeat steps 1 to 3 with CMIP6 fields, bias-correcting the
   fields TCGI is computed from toward ERA5.
5. **Wind fields and hazard maps** (not started). Turn each synthetic track into a wind field with
   CLIMADA, then into return-period wind maps.
6. **Damage** (not started). Apply a damage curve to the hazard maps.

Each step is a module of `ocr.risks.hurricane`, run as `python -m ocr.risks.hurricane.<module>`;
`--help` prints the module's description and its commands.

## Approach

### 1. Forcing

CHAZ reads monthly 250 and 850 hPa winds, mid-level relative humidity, potential intensity (PI) and
the daily covariance of the winds, all on a 2° grid. The CHAZ authors publish a pre-processing
tutorial that builds these files from ERA5, along with its output for 2017
([Zenodo 11631930](https://doi.org/10.5281/zenodo.11631930)). `ocr.risks.hurricane.forcing` ports the
tutorial and builds the files for 1981–2019, with PI from tcpyPI
([Gilford 2021](https://doi.org/10.5194/gmd-14-2351-2021)). For 2017 our files match the tutorial's to
within rounding, except PI at one coastal cell where the land masks differ, which matters little to
CHAZ. See [forcing.ipynb](forcing.ipynb).

### 2. Genesis index

TCGI is a Poisson regression of observed genesis on vorticity, a moisture variable, PI and shear
([Tippett et al. 2011](https://doi.org/10.1175/2010JCLI3811.1);
[Camargo et al. 2014](https://doi.org/10.1175/JCLI-D-13-00505.1)). There are two versions, with column
relative humidity (CRH) or saturation deficit (SD) as the moisture variable. They agree in the
recent climate and diverge in the future ([Lee et al. 2020](https://doi.org/10.1175/JCLI-D-19-0452.1)).
We use the ERA5 coefficients of
[Fosu et al. (2024)](https://doi.org/10.63024/dpva-2pa1).
The ERA5 runs use TCGI-CRH; the climate-model runs are made once with each. `ocr.risks.hurricane.tcgi` computes
TCGI, with the saturation column water vapour computed as Yi Xia's TCGI code
([yixia23/TCGI](https://github.com/yixia23/TCGI); [Xia et al. 2026](https://doi.org/10.1175/JCLI-D-25-0317.1))
did until March 2026, converting kelvin to Celsius by subtracting 273.3 rather than 273.15. Only
with that offset do we reproduce the tutorial's TCGI: our 2017 TCGI then matches it in every basin
(each basin's ratio rounds to 1.000).

CHAZ seeds storm precursors each month in proportion to TCGI. One ratio per run scales TCGI to
seeds. As in the public code, it is set so that the seeds over 1981–2005, times a survival rate of
0.78, come to about the number of storms in the tutorial's observed-track file in those years
(see the seeding-count caveat below). A climate-model run takes the ratio of its
own historical run, so its future run keeps it. An optional regional calibration
(`chaz --genesis calibrated`, built by `ocr.risks.hurricane.calibrate`; off by default) scales TCGI
by a map fitted to where observed storms first reach 34 kt over 1981–2019. The current map was
fitted on an earlier run and needs refitting; no run in the notebooks uses it. See [genesis-index.ipynb](genesis-index.ipynb).

### 3. Tracks and intensity (CHAZ)

CHAZ ([Lee et al. 2018](https://doi.org/10.1002/2017MS001186)) moves each seeded storm with the
beta-and-advection model ([Emanuel et al. 2006](https://doi.org/10.1175/BAMS-87-3-299)): the
large-scale steering wind, plus a synthetic sub-monthly wind (the steering noise), plus a beta
drift. Along each track CHAZ runs a set of intensity
members, 40 in our runs. Each member's wind follows an autoregressive linear regression over 12-hour steps on the
storm's own wind and its change over the previous 12 hours, and on the environment along the track:
PI, shear, humidity, and a land term that weakens storms over land. A random error is added at each
step ([Lee et al. 2015](https://doi.org/10.1175/MWR-D-14-00171.1),
[2016](https://doi.org/10.1175/JCLI-D-15-0909.1)). Lee et al. (2016) fit the regression on
ERA-Interim along the observed tracks of 1981–1999, with the land term a 0.5° land mask averaged
over 300 km, and draw the random error from the regression's errors on observed steps that start at
a similar wind. The nine ocean coefficients in the tutorial data differ from their Table 2 by at
most 0.044 ([intensity.ipynb](intensity.ipynb), section 3); the near-land coefficients differ more.
The tutorial data also hold the observed tracks of 1981–2012.

`ocr.risks.hurricane.chaz` ports the public CHAZ code ([cl3225/CHAZ](https://github.com/cl3225/CHAZ))
and runs it as batch jobs on Coiled cloud machines. Its defaults follow the
public code and the tutorial's files. Departures from them are command-line options of `chaz`; the
main ones:

- `--steering diagonal` draws the steering noise of each wind component independently, as the
  tutorial's notebook code effectively does; the default (`lower`) uses the full covariance factor
  the tutorial's forcing files hold. The CHAZ team's ERA5 tracks are consistent with the diagonal
  version.
- `--land fine` reads the land term at the storm's position on the tutorial's 1/12° land mask, and
  `--land separate` reads the 300 km land mean the regression was fit on. The default (`shared`)
  reads the 2° mask cell the storm is in, as the public code does.
- `--intensity refit` uses intensity regressions refit on our ERA5 environment along the tutorial's
  observed tracks of 1981–2012, the way the regression script in the public CHAZ code does it
  (`CHAZvBeta/tools/regression4.py`). `python -m ocr.risks.hurricane.refit fit` makes them. The
  default (`tutorial`) uses the coefficients shipped with the public code.
- `--intensity refit-pi` also draws the random error from observed steps with similar PI, and
  `--intensity refit-basin` from steps with similar PI in the same basin.
- With a refit intensity option, `--fit 1981-1999` uses regressions fit on those years only
  (`refit fit --years 1981-1999`), so that the storms of 2000–2019 can test them.

The notebooks count US landfalls by the first-crossing rule of
[Emanuel (2025)](https://doi.org/10.1088/2752-5295/add9ba), which
[John et al. (2026)](https://doi.org/10.1088/2752-5295/ae7202) also use when they test CHAZ: each
storm counts once, at its first crossing from sea onto the Atlantic or Gulf coast, at the wind
interpolated to the time of the crossing. Our version is adapted. It counts the first time a
track's centre is on US land, not crossings of coastal gates, and it puts a Poisson band on coast
totals, which Emanuel judges instead by the share of basin storms that cross the coast
([landfall.ipynb](landfall.ipynb) lists the differences). The per-country landfalls in
intensity.ipynb use the same rule.

landfall.ipynb also tests how much the result depends on how the wind is read near landfall. Neither
record gives the wind at the moment of landfall (IBTrACS has a wind every 6 hours, CHAZ every 12),
and Emanuel (2025) notes that interpolating to the crossing can affect observed and synthetic winds
differently. The alternative readings, and HURDAT's landfall records, are in landfall.ipynb.

[tracks.ipynb](tracks.ipynb) compares our ERA5 run with the defaults against IBTrACS and against an
ERA5 track set the CHAZ team shared with us, which is not public, and tests `--steering diagonal`
and `--land separate`. We don't know the settings behind the CHAZ team's set (Questions below).
[intensity.ipynb](intensity.ipynb) compares the land and intensity options with IBTrACS around the
globe.

Our ERA5 runs use the public code's settings, which are what `chaz` uses with no options. Their
intensity model follows Lee et al. (2016), the model Meiler et al. (2026) cite for the hazard maps
we compare against, and the climate-model runs keep it, as Lee et al. (2020) and Fosu et al. (2024)
do. intensity.ipynb tests the options against it with the measures of Lee et al. (2018), who built
CHAZ: where each observed count ranks among a run's 400 realizations, and their t statistic, on 39
counts of landfalls and time at strength around the globe, over 2000–2019 with the refits fit on
1981–1999. By the rule we set before that test, refit-basin, whose error sampling is our addition,
does not replace plain refit (`--intensity refit`). The mean |t|, added afterwards, favours the
refits (0.638 with the public settings, 0.506 with plain refit, 0.364 with refit-basin), but plain
refit's median |t| is worse (0.517 against 0.419), and Lee et al. (2016) warned that basin tuning
could add errors under climate change. We keep the public settings and carry refit-basin as a
tested option.

[north-atlantic.ipynb](north-atlantic.ipynb) applies the skill tests of Emanuel (2025) to the
North Atlantic metrics [John et al. (2026)](https://doi.org/10.1088/2752-5295/ae7202) use to score
CHAZ, and checks our code by partly reproducing their scores for RAFT, a synthetic set whose tracks
are public. The public settings and refit-basin score alike there: close to observed for the basin,
lower at landfall. The metrics scale the storm count away, so they serve as a check.

### 4. Climate models

For a CMIP6 model, the four TCGI input fields (vorticity, shear, PI and the moisture variable) are
bias-corrected toward ERA5 before computing TCGI: the model's monthly climatology for 1981–2010 is
replaced by ERA5's. The fields for tracks and intensity, including the PI the intensity model reads,
are used as the model gives them, with the intensity predictors normalized for the model (Fosu et
al. 2024 describe normalizing by each model's tropical mean). Ours shifts and scales the shipped
normalization by how the predictors differ between the model and ERA5 along the observed tracks
([cmip.ipynb](cmip.ipynb)). [Camargo et al. (2025)](https://doi.org/10.1175/JCLI-D-24-0629.1)
compute a model's monthly climatology from its ensemble mean; our bias correction uses the one
member we run. `ocr.risks.hurricane.cmip` and
`ocr.risks.hurricane.meanstd` build the forcing and normalization. So far CHAZ has run on one
model, UKESM1-0-LL r1i1p1f2, for 1995–2014 (historical) and 2081–2100 (SSP5-8.5), with both TCGI
versions. [cmip.ipynb](cmip.ipynb) reviews that run. The refit intensity options run on ERA5
only so far.

The maps need SSP3-7.0 for 2041–2060 and 2081–2100 from the six models Meiler et al. use. The copy of
CMIP6 on AWS (see Datasets) has the inputs for the historical and SSP5-8.5 runs of all six models,
but for SSP3-7.0 none of the six has the daily 250 and 850 hPa winds there, so those have to come
from another CMIP6 archive.

### 5. Wind fields and hazard maps (not started)

The plan follows [Meiler et al. (2026)](https://doi.org/10.1038/s41597-025-06452-0). CLIMADA's
parametric wind model ([Aznar-Siguan and Bresch 2019](https://doi.org/10.5194/gmd-12-3085-2019))
turns each synthetic track into a wind field, a per-basin frequency correction matches observed
storm counts, and the result is return-period wind maps. [Lee et al. (2023)](https://doi.org/10.1029/2023EF003539)
adjust frequency differently: by year and basin, and with a factor at each of a set of line gates
for regional risk. Meiler et al.'s published maps
([data](https://doi.org/10.5061/dryad.qfttdz0vz)) are the comparison target.
[Meiler et al. (2025)](https://doi.org/10.1126/sciadv.adn4607) attribute the uncertainty in future
tropical-cyclone risk estimates to its sources.

### 6. Damage (not started)

The damage curve is not yet chosen.

## Status and open questions

Where the ERA5 run with the public code's settings stands against IBTrACS
([intensity.ipynb](intensity.ipynb), [landfall.ipynb](landfall.ipynb)):

- On the US Atlantic and Gulf coasts, by the first-crossing rule and without the scaling to the
  observed North Atlantic storm count, the run makes 0.88 of the observed landfalls at 34 kt and
  above, 0.59 at hurricane strength (64 kt) and 0.66 at major-hurricane strength (96 kt). The
  hurricane total lies outside the range of Poisson noise around the run's expected count (31.4
  against 53 observed), however the wind near landfall is read; the major total lies inside it.
  In the coastal regions with enough landfalls to test, they are distributed as observed
  (Emanuel's K′ of 1). By the test of Lee
  et al. (2018) the run makes too few landfalls on the Atlantic coast. The shortfall is larger in
  1981–1999 than in 2000–2019 ([landfall.ipynb](landfall.ipynb)).
- At hurricane and major strength the run spends too little time in the Philippine Sea, the North
  Atlantic tropics and the Caribbean: at 96 kt, 0.27, 0.56 and 0.35 of the observed time.
- It spends far too long at hurricane strength at high latitudes: 13.2 times the observed time
  south of 30°S and 7.4 times north of 35°N in the western Pacific.
- Our runs make more storms in 2000–2019 than in 1981–1999. Storms reaching 34 kt rise by a factor
  of 1.145 from 1981–1997 to 2000–2019 with the public settings and 1.245 with refit-basin, while
  IBTrACS changes by 0.991. Our
  ERA5 TCGI steps up around 1998–2000: its 2000–2019 mean is 1.233 times its 1981–1997 mean. PI
  carries most of the step, with a smaller part from humidity
  ([genesis-index.ipynb](genesis-index.ipynb), section 5).

In UKESM1-0-LL, US Atlantic and Gulf hurricane landfalls in 2081–2100 are 0.64 of those in
1995–2014 with TCGI-CRH and 0.31 with TCGI-SD ([cmip.ipynb](cmip.ipynb)). The UKESM runs use the
public code's settings, as the ERA5 runs do.

An ERA5 run of 1981–2019, with 10 sets of tracks and 40 intensity members per track, costs about $0.30–0.40
on Coiled.

Caveats:

- **The refit's training record.** The refit regressions learn from the tutorial's observed-track
  file and are tested against IBTrACS (section 3).
- **The counts behind the test.** Lee et al.'s measures are published, but the 39 counts we apply
  them to are our choice, as Emanuel (2025) says of his own metrics.
- **The seeding count.** The seeding ratio aims at 98.1 storms a year, which is every storm in the
  tutorial's track file over 1981–2005. Of these, 87.1 a year reach 34 kt; IBTrACS has 89.2 a
  year reaching 34 kt (section 2). Emanuel (2025) calibrates his model to 85 storms a year
  worldwide and compares only storms whose peak exceeds 40 kt.
- **Intensity outside the North Atlantic.** Observed winds outside the North Atlantic and eastern
  Pacific are mostly satellite estimates, from JTWC or the regional agencies. Emanuel (2025) limits his
  tests of lifetime-maximum intensity and 6-hour intensity change to the North Atlantic for that
  reason. The counts of time at strength elsewhere in intensity.ipynb's test, and the high-latitude
  ratios above, rest on those estimates.

Open:

- **US hurricane landfalls.** By every reading of the wind near landfall, the run's US hurricane
  landfalls fall short of observed beyond sampling noise; by Lee et al.'s test the Atlantic
  coast's shortfall is in frequency ([landfall.ipynb](landfall.ipynb)). The shortfall is larger in
  1981–1999, the years when our run also makes fewer storms, so the storm count should be
  understood before any correction. Weighting refit-basin's intensity members by the zone and wind
  class of their first landfall, fitted on one half of 1981–2019, does not consistently improve US
  hurricane landfalls on the other half ([landfall-weights.ipynb](landfall-weights.ipynb)).
- **Refit-basin.** It has the lowest mean |t| on the held-out years but failed the rule set before
  that test, and its error sampling is our addition; Lee et al. (2016)'s warning about basin tuning
  is a reason for caution (section 3). Test it again only with a test declared in advance and run on
  new evidence, such as the coastal winds of step 5, or once the CHAZ team answers the questions
  below; on a climate model it would also need its own normalization.
- **Coastal winds in the maps.** CLIMADA builds each wind field from the track's intensity
  interpolated between records, the same reading as the first-crossing rule, so the maps' coastal
  winds may carry the landfall shortfall. At step 5, compare CLIMADA's coastal winds from our CHAZ
  run with CLIMADA run the same way on IBTrACS (Meiler et al.'s maps, also from CHAZ, cannot test
  this). If they fall short, the fix depends on the cause: holding the wind at its last offshore
  value until the crossing, or the land term (`--land separate`, which reads the 300 km land mean
  Lee et al. (2016) fit on).
- **Storm counts.** Whether the late-1990s rise in ERA5's PI, which carries the step in TCGI, is a
  change in the tropical oceans or in ERA5, and how much the step moves the tests that fit on one
  half of 1981–2019 and score on the other.
- **Genesis calibration.** Refit the regional map on the public settings (section 2).
- **SSP3-7.0 winds.** Another source for the daily winds (section 4).
- **Questions for the CHAZ team:**
  - What produced the ERA5 set they shared with us: the code version, the intensity coefficients
    and normalization file, the land term, the steering covariance and the seeding ratio. Their
    tracks cross the US coast as often as ours, but more of their intensity members have lost
    their wind before the crossing ([tracks.ipynb](tracks.ipynb)). Meiler et al. (2026) describe the
    genesis index fitted to ERA5 and cite Lee et al. (2016) for intensity, so we expect the shipped
    intensity model.
  - Whether the shipped coefficient files (`result_w.nc` over water, `result_l.nc` near land) are
    the fit of Lee et al. (2016), and why they differ from their Table 2: slightly over water (PI
    terms 0.326, 0.211 and −0.454 against 0.31, 0.18 and −0.41), more near land.
  - What sample built the shipped `coefficient_meanstd.nc`. Its PI-minus-wind predictor has a mean
    of 30 kt and a standard deviation of 70 kt; along the tutorial's observed tracks, where a refit
    gives back the shipped coefficients, it has 85 and 44. Does the file include near-land or
    weak-PI points, or tropical means, as Fosu et al. (2024) describe for climate models?
  - Which land term production runs use: Lee et al. (2016) fit on a 0.5° land mask averaged over
    300 km, while the public code reads the 2° cell the storm is in.
  - Whether the random error is drawn by starting wind only, as Lee et al. (2016) describe.
    `--intensity refit-pi` also conditions it on PI, and `refit-basin` on PI and basin.
  - How they handle intensity between the 12-hourly nodes near landfall when they build wind
    fields.
  - How each climate model's normalization file is built, and whether the genesis bias correction
    uses the ensemble-mean climatology, as Camargo et al. (2025) define the bias, or one member's.
  - Whether the 273.3 offset (section 2) is what the TCGI coefficients were fitted with.
  - Whether the seeding ratio should count only the storms in the track file that reach 34 kt.
  - How the per-track frequency parameter that John et al. (2026) describe is computed, and
    whether they can share it. It adjusts tracks to observed landfall statistics and is applied to
    return-period analysis only; the ERA5 files they shared with us do not have it.

## References

### CHAZ

- Lee, C.-Y., Tippett, M. K., Sobel, A. H., and Camargo, S. J. (2018). An environmentally forced
  tropical cyclone hazard model. _Journal of Advances in Modeling Earth Systems_, 10.
  [doi:10.1002/2017MS001186](https://doi.org/10.1002/2017MS001186)
- Lee, C.-Y., Tippett, M. K., Camargo, S. J., and Sobel, A. H. (2015). Probabilistic multiple
  linear regression modeling for tropical cyclone intensity. _Monthly Weather Review_, 143.
  [doi:10.1175/MWR-D-14-00171.1](https://doi.org/10.1175/MWR-D-14-00171.1)
- Lee, C.-Y., Tippett, M. K., Sobel, A. H., and Camargo, S. J. (2016). Autoregressive modeling for
  tropical cyclone intensity climatology. _Journal of Climate_, 29, 7815–7830.
  [doi:10.1175/JCLI-D-15-0909.1](https://doi.org/10.1175/JCLI-D-15-0909.1)
- Emanuel, K., Ravela, S., Vivant, E., and Risi, C. (2006). A statistical deterministic approach to
  hurricane risk assessment. _Bulletin of the American Meteorological Society_, 87.
  [doi:10.1175/BAMS-87-3-299](https://doi.org/10.1175/BAMS-87-3-299)
- Lee, C.-Y., Camargo, S. J., Sobel, A. H., and Tippett, M. K. (2020). Statistical–dynamical
  downscaling projections of tropical cyclone activity in a warming climate: two diverging genesis
  scenarios. _Journal of Climate_, 33, 4815–4834.
  [doi:10.1175/JCLI-D-19-0452.1](https://doi.org/10.1175/JCLI-D-19-0452.1)

Applications of CHAZ, read for how its output is corrected and evaluated:

- Sobel, A. H., et al. (2019). Tropical cyclone hazard to Mumbai in the recent historical climate.
  _Monthly Weather Review_, 147.
  [doi:10.1175/MWR-D-18-0419.1](https://doi.org/10.1175/MWR-D-18-0419.1)
- Lee, C.-Y., Sobel, A. H., Camargo, S. J., Tippett, M. K., and Yang, Q. (2022). New York State
  hurricane hazard: history and future projections. _Journal of Applied Meteorology and
  Climatology_, 61. [doi:10.1175/JAMC-D-21-0173.1](https://doi.org/10.1175/JAMC-D-21-0173.1)
- Lee, C.-Y., Sobel, A. H., Tippett, M. K., et al. (2023). Climate change signal in Atlantic
  tropical cyclones today and near future. _Earth's Future_, 11, e2023EF003539.
  [doi:10.1029/2023EF003539](https://doi.org/10.1029/2023EF003539)
- John, E. B., et al. (2026). Characterizing uncertainty in synthetic tropical cyclone hazard
  models for U.S. energy infrastructure resilience. _Environmental Research: Climate_, 5, 035015.
  [doi:10.1088/2752-5295/ae7202](https://doi.org/10.1088/2752-5295/ae7202)

### Genesis index and climate models

- Tippett, M. K., Camargo, S. J., and Sobel, A. H. (2011). A Poisson regression index for tropical
  cyclone genesis and the role of large-scale vorticity in genesis. _Journal of Climate_, 24,
  2335–2357. [doi:10.1175/2010JCLI3811.1](https://doi.org/10.1175/2010JCLI3811.1)
- Camargo, S. J., Tippett, M. K., Sobel, A. H., Vecchi, G. A., and Zhao, M. (2014). Testing the
  performance of tropical cyclone genesis indices in future climates using the HiRAM model.
  _Journal of Climate_, 27, 9171–9196.
  [doi:10.1175/JCLI-D-13-00505.1](https://doi.org/10.1175/JCLI-D-13-00505.1)
- Fosu, B., et al. (2024). Assessing future tropical cyclone risk using downscaled CMIP6
  projections. _Journal of Catastrophe Risk and Resilience_, 2.
  [doi:10.63024/dpva-2pa1](https://doi.org/10.63024/dpva-2pa1)
- Camargo, S. J., et al. (2025). Tropical cyclones and associated environmental fields in CMIP6
  models. _Journal of Climate_.
  [doi:10.1175/JCLI-D-24-0629.1](https://doi.org/10.1175/JCLI-D-24-0629.1)
- Xia, Y., et al. (2026). Dependence of global tropical cyclones on the tropical Pacific mean state
  in the HighResMIP models. _Journal of Climate_, 39, 1099–1118.
  [doi:10.1175/JCLI-D-25-0317.1](https://doi.org/10.1175/JCLI-D-25-0317.1)
- Gilford, D. M. (2021). pyPI (v1.3): tropical cyclone potential intensity calculations in Python.
  _Geoscientific Model Development_, 14.
  [doi:10.5194/gmd-14-2351-2021](https://doi.org/10.5194/gmd-14-2351-2021)
- Eyring, V., et al. (2016). Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6)
  experimental design and organization. _Geoscientific Model Development_, 9, 1937–1958.
  [doi:10.5194/gmd-9-1937-2016](https://doi.org/10.5194/gmd-9-1937-2016)

### Hazard maps and risk

- Emanuel, K. (2025). A framework for testing tropical cyclone hazard models. _Environmental
  Research: Climate_, 4, 025011.
  [doi:10.1088/2752-5295/add9ba](https://doi.org/10.1088/2752-5295/add9ba)
- Meiler, S., et al. (2026). Global coastal wind hazard maps from the CHAZ tropical cyclone model.
  _Scientific Data_. [doi:10.1038/s41597-025-06452-0](https://doi.org/10.1038/s41597-025-06452-0)
- Meiler, S., et al. (2025). Navigating and attributing uncertainty in future tropical cyclone risk
  estimates. _Science Advances_, 11, eadn4607.
  [doi:10.1126/sciadv.adn4607](https://doi.org/10.1126/sciadv.adn4607)
- Aznar-Siguan, G., and Bresch, D. N. (2019). CLIMADA v1: a global weather and climate risk
  assessment platform. _Geoscientific Model Development_, 12, 3085–3097.
  [doi:10.5194/gmd-12-3085-2019](https://doi.org/10.5194/gmd-12-3085-2019)

### Data

- Hersbach, H., et al. (2020). The ERA5 global reanalysis. _Quarterly Journal of the Royal
  Meteorological Society_, 146, 1999–2049. [doi:10.1002/qj.3803](https://doi.org/10.1002/qj.3803)
- Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J., and Neumann, C. J. (2010). The
  International Best Track Archive for Climate Stewardship (IBTrACS). _Bulletin of the American
  Meteorological Society_, 91, 363–376.
  [doi:10.1175/2009BAMS2755.1](https://doi.org/10.1175/2009BAMS2755.1)

## Datasets

| Dataset | Used for | Source |
| --- | --- | --- |
| ERA5 | forcing, TCGI, PI | Copernicus Climate Data Store: [hourly on pressure levels](https://doi.org/10.24381/cds.bd0915c6), [monthly on pressure levels](https://doi.org/10.24381/cds.6860a573), [monthly on single levels](https://doi.org/10.24381/cds.f17050d7) |
| IBTrACS v04r01 | the regional genesis calibration and validation: main tracks at the 6-hourly synoptic times, with the US agencies' 1-minute maximum wind (`USA_WIND`), or where that is missing, the WMO agency's wind converted to a 1-minute wind as CLIMADA converts it. `USA_WIND` outside the North Atlantic and eastern Pacific comes mostly from JTWC, which often stops following a storm around extratropical transition. Some Southern Hemisphere storms have no `USA_WIND`. HURDAT's landfall records (`USA_RECORD` 'L') are the observed reference in landfall.ipynb | [NOAA NCEI](https://doi.org/10.25921/82ty-9e16) |
| CHAZ code | the model we port | [github.com/cl3225/CHAZ](https://github.com/cl3225/CHAZ) |
| TCGI code | how the saturation column water vapour is computed | [github.com/yixia23/TCGI](https://github.com/yixia23/TCGI) |
| CHAZ tutorial data | reference forcing and TCGI for 2017, the land masks, the intensity coefficients, and the observed tracks with their environment | [Zenodo 11631930](https://doi.org/10.5281/zenodo.11631930) (CC BY 4.0) |
| CMIP6 | climate-model forcing: daily 250 and 850 hPa winds, monthly winds, temperature, humidity, column water vapour, sea-level pressure and surface temperature; historical and SSP5-8.5 so far | [CMIP6 on AWS](https://registry.opendata.aws/cmip6/) (`s3://cmip6-pds`) |
| RAFT synthetic tracks | reproducing John et al.'s scores in north-atlantic.ipynb | [Zenodo 18879988](https://doi.org/10.5281/zenodo.18879988) |
| CHAZ wind hazard maps | comparison target | [Dryad](https://doi.org/10.5061/dryad.qfttdz0vz); code: [Zenodo 16058513](https://doi.org/10.5281/zenodo.16058513) |

Our inputs and outputs are in `s3://carbonplan-ocr/input/hurricane/chaz/`, which needs AWS
credentials for the bucket. `manifest.json` there lists each file's source and checksum.

## Code and checks

| Module (`ocr.risks.hurricane`) | Step |
| --- | --- |
| `forcing` | ERA5 forcing for CHAZ |
| `tcgi` | genesis index and seeding ratio |
| `calibrate` | optional regional genesis calibration |
| `chaz` | the CHAZ port and its options, run on Coiled |
| `refit` | intensity regressions refit on the environment of a forcing data set (ERA5 so far) |
| `cmip`, `meanstd` | CMIP6 forcing, bias correction, intensity normalization |
| `landfall`, `intensity`, `tracks` | checks against IBTrACS in tracks.ipynb and landfall.ipynb: landfall, intensity changes over water, steering noise |
| `validate` | each run against IBTrACS around the globe, by year and realization (intensity.ipynb) |
| `skill` | the measures of Lee et al. (2018) and the skill tests of Emanuel (2025) |
| `report` | the North Atlantic metrics John et al. (2026) apply to CHAZ, for IBTrACS, RAFT and the CHAZ runs (north-atlantic.ipynb) |
| `reweight` | intensity members weighted by their first landfall (landfall-weights.ipynb) |
| `runaways` | intensity members whose wind grows without bound |
| `store`, `batch` | S3 cache with its manifest, Coiled jobs |

| Notebook | Checks |
| --- | --- |
| [forcing.ipynb](forcing.ipynb) | our ERA5 forcing against the tutorial's |
| [genesis-index.ipynb](genesis-index.ipynb) | our TCGI against the tutorial's, and the seeding ratio |
| [tracks.ipynb](tracks.ipynb) | our ERA5 CHAZ run against IBTrACS and the CHAZ team's tracks |
| [landfall.ipynb](landfall.ipynb) | five ways of counting US landfalls, against HURDAT's landfall records |
| [intensity.ipynb](intensity.ipynb) | the land and intensity options against IBTrACS around the globe, with the measures of Lee et al. (2018) |
| [north-atlantic.ipynb](north-atlantic.ipynb) | Emanuel's tests on John et al.'s North Atlantic metrics, and a reproduction of their RAFT scores |
| [landfall-weights.ipynb](landfall-weights.ipynb) | weighting intensity members by first landfall, tested on held-out years |
| [cmip.ipynb](cmip.ipynb) | CHAZ on one CMIP6 model, UKESM1-0-LL |
