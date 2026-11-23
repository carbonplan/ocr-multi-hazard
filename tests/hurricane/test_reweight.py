import numpy as np
import pandas as pd
import pytest
from test_validate import Land, grid

from ocr.risks.hurricane import reweight, validate


def westward(winds):
    """One storm moving west along 30N from 70W onto US land at 80W, one member per wind."""
    lon = np.linspace(-70.0, -90.0, 6)[:, None]
    lat = np.full_like(lon, 30.0)
    wind = np.array([[w] * 6 for w in winds])[:, :, None]
    return lat, lon, wind


def test_first_landfall_classes_each_member_at_its_wind_and_skips_weak_ones():
    lf = reweight.first_landfall(*westward([100.0, 70.0, 40.0, 20.0]), grid())
    assert lf['member'].tolist() == [0, 1, 2]
    assert lf['cls'].tolist() == ['>=96', '64-96', '34-64']
    assert set(lf['zone']) == {'USA FL'} and set(lf['basin']) == {'NA'}


def test_basin_of_landfall_points():
    lat = np.array([30.0, 20.0, 15.0, 22.0, -20.0, -20.0, -20.0, 20.0])
    lon = np.array([-85.0, -105.0, 120.0, 90.0, 45.0, 120.0, 170.0, 10.0])
    got = reweight.basin(lat, lon).tolist()
    assert got == ['NA', 'EP', 'WP', 'NI', 'SI', 'AU', 'SP', 'other']


def test_unit_weights_give_the_unweighted_counts():
    lat, lon, wind = westward([100.0, 70.0, 40.0, 20.0])
    plain = validate.counts(lat, lon, wind, grid(), Land())
    ones = validate.counts(lat, lon, wind, grid(), Land(), weight=np.ones((4, 1)))
    assert ones == pytest.approx(plain)


def test_group_counts_weighted_by_group_equal_counts_weighted_by_member():
    lat, lon, wind = westward([100.0, 70.0, 40.0, 20.0])
    lf, table = reweight.grouped_counts(lat, lon, wind, grid(), Land())
    assert table['zone'].tolist()[0] == reweight.NONE and len(table) == 4
    by_class = {'>=96': 3.0, '64-96': 2.0, '34-64': 0.5, reweight.NONE: 1.0}
    member_w = np.array([[3.0], [2.0], [0.5], [1.0]])
    direct = validate.counts(lat, lon, wind, grid(), Land(), weight=member_w)
    w = pd.DataFrame(
        {'weight': [by_class[c] for c in table['cls']]},
        index=pd.MultiIndex.from_arrays([table['zone'], table['cls']], names=['key', 'cls']),
    )
    annual = reweight.weighted_annual(table.assign(year=2000), w, 'zone', pd.Series({2000: 4}))
    got = annual.loc[2000]
    assert got['members'] == 4
    for k, v in direct.items():
        if k != 'members':
            assert got[k] == pytest.approx(v), k
    assert direct['landfall USA >= 64'] == pytest.approx(5.0)


def test_weights_are_shrunk_rate_ratios_over_the_fit_years():
    rec = pd.DataFrame(
        {
            'year': [2000] * 8 + [2001] * 4 + [2010] * 40,
            'zone': ['MEX'] * 52,
            'basin': ['NA'] * 52,
            'cls': ['64-96'] * 52,
        }
    )
    obs = pd.DataFrame({'year': [2000, 2001, 2001], 'zone': 'MEX', 'basin': 'NA', 'cls': '64-96'})
    members = pd.Series(4, index=range(2000, 2011))
    obs_members = pd.Series(1, index=range(2000, 2011))
    w = reweight.weights(rec, obs.iloc[:1], members, obs_members, [2000, 2001], 'zone', k=2.0)
    row = w.loc[('MEX', '64-96')]
    # 12 member landfalls over 8 member-years is 3 over the 2 observed years
    assert row['n_obs'] == 1 and row['chaz_expected'] == pytest.approx(3.0)
    assert row['weight'] == pytest.approx((1 + 2) / (3 + 2))
    w0 = reweight.weights(rec, obs.iloc[:1], members, obs_members, [2000, 2001], 'basin', k=0.0)
    assert w0.loc[('NA', '64-96'), 'weight'] == pytest.approx(1 / 3)


def test_other_land_splits_by_basin_only_in_that_scheme():
    df = pd.DataFrame(
        {'zone': ['other land', 'MEX', reweight.NONE], 'basin': ['WP', 'EP', reweight.NONE]}
    )
    assert reweight.key(df, 'zone, other land by basin').tolist() == ['other land WP', 'MEX', '']
    assert reweight.key(df, 'basin').tolist() == ['WP', 'EP', '']
