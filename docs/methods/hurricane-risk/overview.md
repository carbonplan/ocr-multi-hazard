# Hurricane risk methods overview

## Summary

The aim is global tropical-cyclone wind hazard maps for the recent climate and for future
scenarios, following the approach of [Meiler et al. (2026)](https://doi.org/10.1038/s41597-025-06452-0), rebuilt so that every
step can be rerun and checked:

1. **Forcing.** Build the environmental fields CHAZ reads from ERA5, and from CMIP6 models for the
   future.
2. **Genesis.** Compute the tropical-cyclone genesis index (TCGI) that sets where and how often
   storms are seeded.
3. **Tracks and intensity.** Run CHAZ, the Columbia statistical-dynamical tropical-cyclone hazard
   model, to make many synthetic storm tracks with wind histories.
4. **Climate models.** Repeat steps 1 to 3 with CMIP6 fields, bias-correcting the genesis fields
   toward ERA5.
5. **Wind fields and hazard maps** (_not started_). Turn each synthetic track into a wind field with
   CLIMADA, then into return-period wind maps.
6. **Damage** (_not started_). Apply a damage curve to the hazard maps.

## Approach

### 1. Forcing

CHAZ reads monthly 250 and 850 hPa winds, mid-level relative humidity, potential intensity (PI),
and the daily covariance of the winds, all on a 2° grid. `ocr.risks.hurricane.forcing` ports the
CHAZ pre-processing tutorial to ERA5, with PI from tcpyPI ([Gilford 2021](https://doi.org/10.5194/gmd-14-2351-2021)). Our 2017
forcing reproduces the tutorial's published files.
See [forcing.ipynb](forcing.ipynb).

### 2. Genesis index

TCGI is a Poisson regression of observed genesis on vorticity, a moisture variable, PI and shear
([Tippett et al. 2011](https://doi.org/10.1175/2010JCLI3811.1); [Camargo et al. 2014](https://doi.org/10.1175/JCLI-D-13-00505.1)). There are two
versions, with column relative humidity (CRH) or saturation deficit (SD) as the moisture variable.
They agree in the recent climate and diverge in the future ([Lee et al. 2020](https://doi.org/10.1175/JCLI-D-19-0452.1)). We use the
ERA5 coefficients of [Fosu et al. (2024)](https://journalofcrr.com/wp-content/uploads/2024/09/JCRR-Fosu-et-al-02-01.pdf). `ocr.risks.hurricane.tcgi` reproduces the
tutorial's 2017 TCGI to about 1%. See [genesis-index.ipynb](genesis-index.ipynb).

### 3. Tracks and intensity (CHAZ)

CHAZ ([Lee et al. 2018](https://doi.org/10.1002/2017MS001186)) seeds storms at the TCGI rate and moves them with the
beta-and-advection model ([Emanuel et al. 2006](https://doi.org/10.1175/BAMS-87-3-299)): the large-scale steering wind plus
a synthetic sub-monthly part, plus a beta drift. Intensity follows an autoregressive linear
regression on the environment along the track, with a random error each step
([Lee et al. 2015](https://doi.org/10.1175/MWR-D-14-00171.1), [2016](https://doi.org/10.1175/JCLI-D-15-0909.1)). The regression and its errors are fit on observed
best tracks. `ocr.risks.hurricane.chaz` ports the public CHAZ code
([cl3225/CHAZ](https://github.com/cl3225/CHAZ)) and runs it on Coiled.
[tracks.ipynb](tracks.ipynb) compares our ERA5 run with IBTrACS and with the ERA5 track set the CHAZ
team shared with us.

Changes from the public code are options of `chaz`, each documented with its evidence:

- the land term read on the 1/12° land mask at the storm's position (`--land fine`)
- the intensity regressions refit on our ERA5 environment by the authors' method (`--intensity refit`)
- the random error drawn by potential intensity and basin as well as wind
  (`--intensity refit-pi`, `refit-basin`)

### 4. Climate models

For a CMIP6 model, the genesis fields are bias-corrected toward the ERA5 1981–2010 climatology
before computing TCGI. The track and intensity fields are used as the model gives them, with the
intensity predictors normalized by the model's own climatology ([Fosu et al. 2024](https://journalofcrr.com/wp-content/uploads/2024/09/JCRR-Fosu-et-al-02-01.pdf);
[Camargo et al. 2025](https://doi.org/10.1175/JCLI-D-24-0629.1)). `ocr.risks.hurricane.cmip` builds the forcing for one model so
far, UKESM1-0-LL.

### 5. Wind fields and hazard maps (_not started_)

The plan follows [Meiler et al. (2026)](https://doi.org/10.1038/s41597-025-06452-0). CLIMADA's parametric wind model
([Aznar-Siguan and Bresch 2019](https://doi.org/10.5194/gmd-12-3085-2019)) turns each synthetic track into a wind field, a
per-basin frequency correction matches observed storm counts, and the result is return-period wind
maps. Their published maps ([data](https://doi.org/10.5281/zenodo.16058513)) are the comparison target.
[Meiler et al. (2025)](https://doi.org/10.1126/sciadv.adn4607) attributes uncertainty in future tropical-cyclone risk across
these modelling choices.

### 6. Damage (_not started_)

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
- Sobel, A. H., et al. (2019). Tropical cyclone hazard to Mumbai in the recent historical climate.
  _Monthly Weather Review_, 147.
  [doi:10.1175/MWR-D-18-0419.1](https://doi.org/10.1175/MWR-D-18-0419.1)
- Lee, C.-Y., Sobel, A. H., Camargo, S. J., Tippett, M. K., and Yang, Q. (2022). New York State
  hurricane hazard: history and future projections. _Journal of Applied Meteorology and
  Climatology_, 61. [doi:10.1175/JAMC-D-21-0173.1](https://doi.org/10.1175/JAMC-D-21-0173.1)
- Lee, C.-Y., Sobel, A. H., Tippett, M. K., et al. (2023). Climate change signal in Atlantic
  tropical cyclones today and near future. _Earth's Future_, 11, e2023EF003539.
  [doi:10.1029/2023EF003539](https://doi.org/10.1029/2023EF003539)

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
  [PDF](https://journalofcrr.com/wp-content/uploads/2024/09/JCRR-Fosu-et-al-02-01.pdf)
- Camargo, S. J., et al. (2025). Tropical cyclones and associated environmental fields in CMIP6
  models. _Journal of Climate_.
  [doi:10.1175/JCLI-D-24-0629.1](https://doi.org/10.1175/JCLI-D-24-0629.1)
- Gilford, D. M. (2021). pyPI (v1.3): tropical cyclone potential intensity calculations in Python.
  _Geoscientific Model Development_, 14.
  [doi:10.5194/gmd-14-2351-2021](https://doi.org/10.5194/gmd-14-2351-2021)
- Eyring, V., et al. (2016). Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6)
  experimental design and organization. _Geoscientific Model Development_, 9, 1937–1958.
  [doi:10.5194/gmd-9-1937-2016](https://doi.org/10.5194/gmd-9-1937-2016)

### Hazard maps and risk

- Meiler, S., et al. (2026). Global coastal wind hazard maps
  from the CHAZ tropical cyclone model. _Scientific Data_.
  [doi:10.1038/s41597-025-06452-0](https://doi.org/10.1038/s41597-025-06452-0)
- Meiler, S., et al. (2025). Navigating and attributing uncertainty in future tropical cyclone risk
  estimates. _Science Advances_, 11, eadn4607.
  [doi:10.1126/sciadv.adn4607](https://doi.org/10.1126/sciadv.adn4607)
- Aznar-Siguan, G., and Bresch, D. N. (2019). CLIMADA v1: a global weather and climate risk
  assessment platform. _Geoscientific Model Development_, 12, 3085–3097.
  [doi:10.5194/gmd-12-3085-2019](https://doi.org/10.5194/gmd-12-3085-2019)
- John, E. B., et al. (2026). Characterizing uncertainty in synthetic tropical cyclone hazard
  models for U.S. energy infrastructure resilience.

## Datasets

| Dataset | Used for | Source |
| --- | --- | --- |
| ERA5 ([Hersbach et al. 2020](https://doi.org/10.1002/qj.3803)) | forcing, TCGI, PI | Copernicus Climate Data Store: [hourly on pressure levels](https://doi.org/10.24381/cds.bd0915c6), [monthly on pressure levels](https://doi.org/10.24381/cds.6860a573), [monthly on single levels](https://doi.org/10.24381/cds.f17050d7) |
| IBTrACS v04r01 ([Knapp et al. 2010](https://doi.org/10.1175/2009BAMS2755.1)) | validation; USA_WIND at synoptic times | [NOAA NCEI](https://doi.org/10.25921/82ty-9e16) |
| CHAZ code | the model we port | [github.com/cl3225/CHAZ](https://github.com/cl3225/CHAZ) |
| CHAZ tutorial data | reference forcing and TCGI for 2017, intensity coefficients, observed tracks with their environment | [Zenodo 11631930](https://doi.org/10.5281/zenodo.11631930) (CC BY 4.0) |
| CMIP6 | future forcing | [CMIP6 on AWS](https://registry.opendata.aws/cmip6/) (`s3://cmip6-pds`) |
| CHAZ wind hazard maps ([Meiler et al. 2026](https://doi.org/10.1038/s41597-025-06452-0)) | comparison target | [Zenodo 16058513](https://doi.org/10.5281/zenodo.16058513), [Dryad](https://doi.org/10.5061/dryad.qfttdz0vz) |
| Natural Earth 1:10m land and countries | land masks, landfall by country | [naturalearthdata.com](https://www.naturalearthdata.com/) |

Our inputs and outputs are in `s3://carbonplan-ocr/input/hurricane/chaz/`. `manifest.json` there
lists each file's source and checksum.

## Code and checks

| Module (`ocr.risks.hurricane`) | Step |
| --- | --- |
| `forcing` | ERA5 forcing for CHAZ |
| `tcgi` | genesis index and seeding rate |
| `chaz` | the CHAZ port and its options, run on Coiled |
| `cmip`, `meanstd` | CMIP6 forcing, bias correction, intensity normalization |
| `refit` | intensity regressions refit on a forcing's environment |
| `calibrate` | regional genesis calibration |
| `landfall`, `intensity`, `tracks`, `runaways` | diagnostics against IBTrACS |
| `store`, `batch` | S3 cache with manifest, Coiled jobs |

| Notebook | Checks |
| --- | --- |
| [forcing.ipynb](forcing.ipynb) | our ERA5 forcing against the tutorial's |
| [genesis-index.ipynb](genesis-index.ipynb) | our TCGI against the tutorial's, and the seeding rule |
| [tracks.ipynb](tracks.ipynb) | our ERA5 CHAZ run against IBTrACS and the CHAZ team's tracks |
