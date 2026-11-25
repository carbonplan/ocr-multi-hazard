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
TCGI. It computes the saturation column water vapour, which both moisture variables depend on,
as Yi Xia's TCGI code ([yixia23/TCGI](https://github.com/yixia23/TCGI);
[Xia et al. 2026](https://doi.org/10.1175/JCLI-D-25-0317.1)) did from 2024 to March 2026. That
version converted kelvin to Celsius by subtracting 273.3 rather than 273.15. We keep the 273.3,
because only with it do we reproduce the tutorial's values, and the TCGI coefficients were
presumably fit with it. Our 2017 TCGI then matches the tutorial's in every basin: each basin's
ratio rounds to 1.000.

CHAZ seeds storm precursors each month in proportion to TCGI. One ratio per run scales TCGI to
seeds. As in the public code, it is set so that the seeds over 1981–2005, times a survival rate of
0.78, come to about the number of storms in the tutorial's observed-track file in those years, 98.1 a year.
That count includes storms that never reach 34 kt. Of those storms, 87.1 a year reach 34 kt, and
IBTrACS has 89.2 a year reaching 34 kt over the 1981–2005 seasons ([genesis-index.ipynb](genesis-index.ipynb), section 4).
A climate-model run takes the ratio of its
own historical run, so its future run keeps it. An optional regional calibration
(`chaz --genesis calibrated`, built by `ocr.risks.hurricane.calibrate`; off by default) scales TCGI
by a map fitted to where observed storms first reach 34 kt over 1981–2019. The current map was
fitted on a run with other land and intensity settings than those of section 3 and needs refitting
on them; none of the runs in the notebooks uses it. See [genesis-index.ipynb](genesis-index.ipynb).

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

- `--steering diagonal` zeroes the off-diagonal entries of the steering noise's covariance factor,
  so the noise in each wind component is drawn independently of the others. The default
  (`lower`) uses the full lower Cholesky factor, which is what the tutorial's forcing files hold.
  The tutorial's notebook code, as written, transposes that factor, which leaves CHAZ reading
  only its diagonal. The CHAZ team's ERA5 tracks are consistent with the diagonal version.
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

landfall.ipynb also tests how much the result depends on how the wind is read near landfall.
Neither record gives the wind at landfall: IBTrACS's track has a wind every 6 hours, and CHAZ predicts one
every 12 hours. Interpolated to the crossing time, the wind takes part of the next record's, which
is often over land, and Emanuel (2025) notes that this can affect observed and synthetic winds
differently. Read instead at the last wind over water on the track's 12-hourly steps before each storm's strongest crossing,
the public-settings run's hurricane counts rise more than the observed ones but stay short of them.
On the observed
side that reading finds 16 storms at 96 kt or more, against 10 by the first-crossing rule and 17 in
HURDAT's landfall records, which count each storm at its strongest landfall.

[tracks.ipynb](tracks.ipynb) compares our ERA5 run with the defaults against IBTrACS and against an
ERA5 track set the CHAZ team shared with us, which is not public, and tests `--steering diagonal`
and `--land separate`. We don't know the settings behind the CHAZ team's set (Questions below), so it
is a point of comparison we have not checked. [intensity.ipynb](intensity.ipynb) compares the land
and intensity options with IBTrACS around the globe and tests the refits on years they were not
fit on.

Our ERA5 runs use the public code's settings, which are what `chaz` uses with no options. Their
intensity model follows Lee et al. (2016), the model Meiler et al. (2026) cite for the hazard maps
we compare against, and the climate-model runs keep it, as Lee et al. (2020) and Fosu et al. (2024)
do. intensity.ipynb tests the options against it with two measures of Lee et al. (2018), who built
CHAZ. The first is where each observed count ranks among a run's 400 realizations of the same years,
a realization being one track realization with one of its 40 intensity members. The second is
their t statistic: the simulated mean minus the observed mean, over the pooled year-to-year spread.
The measures are applied to 39 counts: US landfalls at 34, 64 and 96 kt, time at hurricane and
major strength in 14 regions, and hurricane landfalls in eight countries.

The test on years the fits did not see uses 2000–2019, with the regressions fit on 1981–1999. Before
it we set a rule for one choice: refit-basin, whose error sampling is our addition, would replace
plain refit (`--intensity refit`) only if more of the counts ranked between the 5th and 95th
percentiles of its realizations and the lead held in 95% of 2,000 resamples of the years.
Refit-basin has 0.564 of the counts in that range and plain refit 0.359 (0.385 with `--land fine`,
0.333 with the public settings), but the lead does not hold in 95% of the resamples, so
refit-basin does not replace plain refit. The share counts a near miss and a large miss alike.
The mean |t|, which we added after seeing that result, keeps the size of each miss: it is 0.638
with the public settings, 0.506 with plain refit and 0.364 with refit-basin, and each lead holds
in nearly all resamples. Plain refit's median |t| is worse than the public settings', though
(0.517 against 0.419), averaging |t| over our counts is our own summary, and Lee et al. (2016)
found basin-specific regressions no better than a global one and warned that basin tuning could add
errors under climate change. We keep the public settings and carry refit-basin as a tested option.
Reading the shipped coefficients with a normalization built from our ERA5 along the observed tracks
lowers the high-latitude excess of the public settings but also their US hurricane landfalls
(intensity.ipynb, section 6).

[north-atlantic.ipynb](north-atlantic.ipynb) applies the skill tests of Emanuel (2025) to the
North Atlantic metrics that [John et al. (2026)](https://doi.org/10.1088/2752-5295/ae7202) use to
score CHAZ: intensity histograms of all storms and of US landfalling storms, 6-hour intensity
changes, the annual cycle, and the share of basin storms that reach the coast. To check our code it
scores RAFT, the one synthetic set in their Table 1 whose tracks are public. The annual cycle
matches their scores; the basin intensity histogram matches on D′ and r² but not on K′; the 6-hour
change and the landfall scores match only in part. Their coastal gates and code are not published,
so we cannot say which readings differ. On these metrics the public settings and refit-basin score
alike: close to observed for the basin as a whole, lower at landfall (K′ 0.58 on the wind at
landfall). The metrics scale the storm count away, so we use them as a check and compare setups in
intensity.ipynb.

### 4. Climate models

For a CMIP6 model, the four TCGI input fields (vorticity, shear, PI and the moisture variable) are
bias-corrected toward ERA5 before computing TCGI: the model's monthly climatology for 1981–2010 is
replaced by ERA5's. The fields for tracks and intensity, including the PI the intensity model reads,
are used as the model gives them, with the intensity predictors normalized for the model; Fosu et al. (2024) describe this as
normalizing by each model's tropical mean. Ours maps the shipped normalization to the model: the
public code's `getMeanStd.py`, ported, evaluates the predictors along the observed tracks in the
model and in ERA5, and the shipped means and standard deviations are shifted and scaled by the
difference ([cmip.ipynb](cmip.ipynb)). [Camargo et al. (2025)](https://doi.org/10.1175/JCLI-D-24-0629.1)
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
adjust frequency differently. Each year's basin count is rescaled using the mean and standard
deviation of the model's historical run and of the observed record (their equation 1), and the
same adjustment is applied to the future run. For regional risk, they apply a multiplicative factor at each
of their line gates (one crosses the open-ocean main development region) so that the rate of
storms above 40 kt crossing each gate matches the observed rate. Meiler et al.'s published maps
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
  above, 0.59 at hurricane strength (64 kt) and 0.66 at major-hurricane strength (96 kt); with the
  scaling, 0.89, 0.60 and 0.67. Over 1981–2019 the hurricane total lies above the 5th–95th
  percentile band of Poisson noise around the run's expected count (31.4 against 53 observed); the
  major total lies inside it. Normalized to the observed total, its landfalls lie within sampling noise of the
  observed ones in every coastal region with more than 5 observed landfalls (Emanuel's K′ of 1;
  four regions at 34 kt, three at 64 kt). The hurricane
  shortfall holds however the wind is read: at each storm's strongest crossing and last offshore
  wind, and with IBTrACS's winds thinned to CHAZ's 12-hour form, the observed totals stay above the
  band. By the test of Lee et al. (2018), the run makes too few landfalls on the Atlantic coast, and
  its strong Gulf landfalls are low but within the spread of its realizations once the rate is
  matched. The hurricane shortfall is larger in 1981–1999 (13.4 expected against 27 observed) than
  in 2000–2019 (18.0 against 26).
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
- **The CHAZ team's set.** We don't know its settings, so we have not checked it as a comparison.
  Meiler et al. (2026), who shared it, describe the genesis index fitted to ERA5 and cite Lee et
  al. (2016) for the intensity model; which coefficient files produced their tracks is not
  confirmed.
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
  winds may carry the shortfall above. How closely they do depends on CLIMADA's time step and on
  how much the hours before landfall set the peak wind on the coast. At step 5, compare CLIMADA's
  coastal winds from our CHAZ run with CLIMADA run the same way on IBTrACS. Matching Meiler et
  al.'s maps would not test this, since they also come from CHAZ through CLIMADA. A fix, if one is
  needed, depends on the cause: for the 12-hourly interpolation, the wind could be held at its last
  offshore value until the crossing; for CHAZ's weakening over land, the land term
  (`--land separate`, tested in [tracks.ipynb](tracks.ipynb)) is the option to change. It reads
  the 300 km land mean that Lee et al. (2016) fit the regression on.
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
| Natural Earth 1:10m land and countries | landfall statistics in the checks | [naturalearthdata.com](https://www.naturalearthdata.com/) |

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
