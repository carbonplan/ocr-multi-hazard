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
3. **Tracks and intensity** (ERA5 1981–2019 runs done, configuration chosen). Run CHAZ.
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
[2016](https://doi.org/10.1175/JCLI-D-15-0909.1)). The authors fit the regression on observed best
tracks. The tutorial data hold their coefficients and the observed tracks of 1981–2012, and the
random error is drawn from the regression's errors on observed steps that start at a similar wind.

`ocr.risks.hurricane.chaz` ports the public CHAZ code ([cl3225/CHAZ](https://github.com/cl3225/CHAZ))
and runs it as batch jobs on [Coiled](https://www.coiled.io/) cloud machines. Its defaults follow the
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
([landfall.ipynb](landfall.ipynb) lists the differences). The per-country landfalls in the score
of section 3 use the same rule.

landfall.ipynb also tests how much the result depends on how the wind is read near landfall.
Neither record gives the wind at landfall: IBTrACS's track has a wind every 6 hours, and CHAZ predicts one
every 12 hours. Interpolated to the crossing time, the wind takes part of the next record's, which
is often over land, and Emanuel (2025) notes that this can affect observed and synthetic winds
differently. Read instead at the last wind over water on the track's 12-hourly steps before each storm's strongest crossing,
refit-basin's counts come closer to observed. On the observed side that reading finds 16 storms
at 96 kt or more, against 10 by the first-crossing rule and 17 in HURDAT's landfall records, which
count each storm at its strongest landfall.

[tracks.ipynb](tracks.ipynb) compares our ERA5 run with the defaults against IBTrACS and against an
ERA5 track set the CHAZ team shared with us, which is not public, and tests `--steering diagonal`
and `--land separate`. We don't know the settings behind the CHAZ team's set (Questions below), so it
is a point of comparison we have not checked. [intensity.ipynb](intensity.ipynb) compares the land and intensity options
with IBTrACS around the globe, scores them, and tests the refit on years it was not fit on.

The configuration we chose for our ERA5 runs is `--land fine --intensity refit-basin`; with no
options, `chaz` uses the public code's land and intensity settings. intensity.ipynb scores each set of storms by its mean absolute log ratio to IBTrACS over 41
metrics, leaving out counts IBTrACS has 5 or fewer of in the period (lower is better). Over
2000–2019 this configuration scores 0.323, the public code's settings 0.454 and the CHAZ team's
tracks 0.402. With the regressions fitted on 1981–1999 only, it scores 0.317 on 2013–2019, years
the fit did not see, against 0.465 for `--land fine` alone. The ranking is the same with
landfalls read at the strongest crossing and last offshore wind. The held-out
test splits 1981–2019 by year, and our run makes more storms in 2000–2019 than in 1981–1999 while
the observed count stays level ([landfall-weights.ipynb](landfall-weights.ipynb), section 4). The
test does not account for that. The refits also learn from the tutorial's observed-track file and
are scored against IBTrACS as we read it, and where the two records differ, the difference may
affect the refits' scores.

### 4. Climate models

For a CMIP6 model, the four TCGI input fields (vorticity, shear, PI and the moisture variable) are
bias-corrected toward ERA5 before computing TCGI: the model's monthly climatology for 1981–2010 is
replaced by ERA5's. The fields for tracks and intensity, including the PI the intensity model reads,
are used as the model gives them, with the intensity predictors normalized for the model ([Fosu et al. 2024](https://doi.org/10.63024/dpva-2pa1);
[Camargo et al. 2025](https://doi.org/10.1175/JCLI-D-24-0629.1)). We build that normalization as the public code's `getMeanStd.py`
does, from the model's environment at the positions and dates of the observed tracks. `ocr.risks.hurricane.cmip` and
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

Where the ERA5 run with the chosen configuration stands against IBTrACS
([intensity.ipynb](intensity.ipynb)):

- On the US Atlantic and Gulf coasts, by the first-crossing rule and without the scaling to the
  observed North Atlantic storm count, the run makes 1.00 of the observed landfalls at 34 kt and
  above, 0.67 at hurricane strength (64 kt) and 0.53 at major-hurricane strength (96 kt); with the
  scaling, 0.96, 0.64 and 0.50. The hurricane and major totals over 1981–2019 lie above the
  5th–95th percentile band of Poisson noise around the run's expected counts. Normalized to the
  observed total, its landfalls over four coastal regions lie within sampling noise of the observed
  ones (Emanuel's K′ of 1). The size of the shortfall depends on how the wind is read: read at each storm's strongest crossing and last offshore wind, the counts
  lie within sampling noise, and with IBTrACS's winds thinned to CHAZ's 12-hour form the major
  counts do too, but the hurricane counts do not. By the test of Lee et al. (2018), storms come
  ashore too weak on the Gulf coast and too rarely on the Atlantic coast
  ([landfall.ipynb](landfall.ipynb)). Split by period, unscaled hurricane landfalls are 0.80 of
  observed over 2000–2019 and 0.54 over 1981–1999
  ([landfall-weights.ipynb](landfall-weights.ipynb)).
- Too many storms reach 34 kt in the South Pacific (1.93 times observed), and the time storms spend
  at major strength there is 2.3 times observed.
- Storms spend too long at hurricane strength at high latitudes: 5.3 times the observed time south of
  30°S and 3.4 times north of 35°N in the western Pacific.
- Our run makes more storms in 2000–2019 than in 1981–1999: 103.89 a year reaching 34 kt worldwide,
  against 84.81, per intensity member. IBTrACS has 88.60 and 88.58. The count steps up around
  1998–2000 ([landfall-weights.ipynb](landfall-weights.ipynb), section 4), as our ERA5 TCGI
  does: its 2000–2019 mean is 1.233 times its 1981–1997 mean. PI carries most of the step, with a
  smaller part from humidity ([genesis-index.ipynb](genesis-index.ipynb), section 5).

In UKESM1-0-LL, US Atlantic and Gulf hurricane landfalls in 2081–2100 are 0.64 of those in
1995–2014 with TCGI-CRH and 0.31 with TCGI-SD ([cmip.ipynb](cmip.ipynb)). The UKESM runs use the public
code's intensity model, not the configuration of section 3.

An ERA5 run of 1981–2019, with 10 sets of tracks and 40 intensity members per track, costs about $0.30–0.40
on Coiled.

Caveats:

- **The refit's training record.** The refit regressions learn from the tutorial's observed-track
  file and are scored against IBTrACS (section 3).
- **The seeding count.** The seeding ratio aims at 98.1 storms a year, which is every storm in the
  tutorial's track file over 1981–2005. Of these, 87.1 a year reach 34 kt; IBTrACS has 89.2 a
  year reaching 34 kt (section 2). Emanuel (2025) calibrates his model to 85 storms a year
  worldwide and compares only storms whose peak exceeds 40 kt.
- **The CHAZ team's set.** It is a comparison we cannot check, since we don't know its settings.
- **Intensity outside the North Atlantic.** Observed winds outside the North Atlantic and eastern
  Pacific are mostly satellite estimates, from JTWC or the regional agencies. Emanuel (2025) limits his
  tests of lifetime-maximum intensity and 6-hour intensity change to the North Atlantic for that
  reason. The score's intensity metrics elsewhere,
  and the high-latitude ratios above, rest on those estimates.

Open:

- **Landfall intensity.** By the first-crossing rule, US hurricane and major landfalls fall short
  of observed beyond sampling noise. With IBTrACS's winds thinned to CHAZ's 12-hour form the major
  counts come within it but the hurricane counts do not, and Lee et al.'s test places the Gulf
  shortfall in intensity and the Atlantic one in frequency ([landfall.ipynb](landfall.ipynb)). The shortfall is larger in
  1981–1999, the years when our run also makes fewer storms, so the storm count should be
  understood before any correction of landfall intensity. Weighting each intensity member by the
  zone and wind class of its first landfall, fitted on one half of 1981–2019, does not consistently
  improve US hurricane landfalls on the other half ([landfall-weights.ipynb](landfall-weights.ipynb)).
- **Coastal winds in the maps.** CLIMADA builds each wind field from the track's intensity
  interpolated between records, the same reading as the first-crossing rule, so the maps' coastal
  winds may carry the shortfall above. How closely they do depends on CLIMADA's time step and on
  how much the hours before landfall set the peak wind on the coast. At stage 5, compare CLIMADA's
  coastal winds from our CHAZ run with CLIMADA run the same way on IBTrACS. Matching Meiler et
  al.'s maps would not test this, since they also come from CHAZ through CLIMADA. A fix, if one is
  needed, depends on the cause: for the 12-hourly interpolation, the wind could be held at its last
  offshore value until the crossing; for CHAZ's weakening over land, the land term
  (`--land separate`, tested in [tracks.ipynb](tracks.ipynb)) is the option to change.
- **Storm counts.** Whether the late-1990s rise in ERA5's PI, which carries the step in TCGI, is a
  change in the tropical oceans or in ERA5, and how much the step moves the tests that fit on one
  half of 1981–2019 and score on the other.
- **Genesis calibration.** Refit the regional map on the chosen configuration (section 2).
- **Refit intensity on climate models.** Each model needs its own normalization of the refit
  regressions.
- **SSP3-7.0 winds.** Another source for the daily winds (section 4).
- **Questions for the CHAZ team:**
  - Which intensity normalization they run. The shipped coefficients come back when the regression
    is fitted on the tutorial's observed-track file, but there the PI-minus-wind predictor has a
    mean of 85 kt and a standard deviation of 44 kt, against 30 and 70 in the shipped normalization
    file.
  - The intensity settings behind their shared ERA5 set, and how they handle intensity between the
    12-hourly nodes near landfall when they build wind fields.
  - How each climate model's normalization file is built.
  - Whether the genesis bias correction uses one ensemble member's climatology or the ensemble mean's.
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
| `validate` | each run against IBTrACS around the globe, year by year, and its score (intensity.ipynb) |
| `reweight` | intensity members weighted by their first landfall (landfall-weights.ipynb) |
| `runaways` | intensity members whose wind grows without bound |
| `store`, `batch` | S3 cache with its manifest, Coiled jobs |

| Notebook | Checks |
| --- | --- |
| [forcing.ipynb](forcing.ipynb) | our ERA5 forcing against the tutorial's |
| [genesis-index.ipynb](genesis-index.ipynb) | our TCGI against the tutorial's, and the seeding ratio |
| [tracks.ipynb](tracks.ipynb) | our ERA5 CHAZ run against IBTrACS and the CHAZ team's tracks |
| [landfall.ipynb](landfall.ipynb) | five ways of counting US landfalls, against HURDAT's landfall records |
| [intensity.ipynb](intensity.ipynb) | the land and intensity options against IBTrACS around the globe |
| [landfall-weights.ipynb](landfall-weights.ipynb) | weighting intensity members by first landfall, tested on held-out years |
| [cmip.ipynb](cmip.ipynb) | CHAZ on one CMIP6 model, UKESM1-0-LL |
