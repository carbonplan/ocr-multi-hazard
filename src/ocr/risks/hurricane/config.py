"""CHAZ store layout shared by the ETL and later hurricane stages."""

from typing import Literal, get_args

STORE = 'hurricane/chaz.icechunk'  # under the input prefix
Metric = Literal['return_periods', 'exceedance_intensity']
METRICS: tuple[Metric, ...] = get_args(Metric)
Rep = Literal['points', 'raster']

GCMS = ['CESM2', 'CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR', 'MIROC6', 'UKESM1-0-LL']
SCENARIOS = ['ssp245', 'ssp370', 'ssp585']
PERIODS = ['base', 'fut1', 'fut2']
VARIANTS = ['CRH', 'SD']
COORDS = {'scenario': SCENARIOS, 'period': PERIODS, 'variant': VARIANTS, 'gcm': GCMS}
DIMS = (*COORDS, 'lat', 'lon')

BANDS: dict[Metric, list[str]] = {
    'return_periods': ['thr_33', 'thr_50'],
    'exceedance_intensity': ['rp_10', 'rp_25', 'rp_50', 'rp_100', 'rp_250', 'rp_1000'],
}
# As in the raw files. Region writes drop attrs, so the template holds them.
BAND_ATTRS: dict[Metric, dict[str, dict[str, str]]] = {
    'return_periods': {
        b: {'long_name': f'Return period for intensity ≥ {b[4:]} m/s', 'units': 'years'}
        for b in BANDS['return_periods']
    },
    'exceedance_intensity': {
        b: {'long_name': f'Exceedance intensity for RP={b[3:]} years', 'units': 'm/s'}
        for b in BANDS['exceedance_intensity']
    },
}


def group_name(metric: Metric, rep: Rep, era5: bool = False) -> str:
    return (metric if rep == 'points' else f'{metric}_raster') + ('_era5' if era5 else '')
