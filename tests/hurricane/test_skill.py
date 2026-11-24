import numpy as np
import pandas as pd
import pytest

from ocr.risks.hurricane import skill


def test_t_statistic_pools_observed_and_simulated_spread():
    observed = pd.Series([1.0, 3.0])  # mean 2, variance 2
    simulated = pd.Series([3.0, 5.0, 3.0, 5.0])  # mean 4, variance 4/3
    pooled = np.sqrt((1 * 2 + 3 * 4 / 3) / 4)
    assert skill.t_statistic(simulated, observed) == pytest.approx(2 / pooled)


def test_rank_counts_ties_as_half():
    assert skill.rank([1, 2, 3, 4], 3) == pytest.approx(62.5)
    assert skill.rank([1, 2, 3, 4], 10) == 100 and skill.rank([1, 2, 3, 4], 0) == 0


def test_lee_ranks_period_totals_of_each_realization():
    index = pd.MultiIndex.from_product([[2000, 2001], [0, 1, 2]], names=['year', 'realization'])
    realizations = pd.DataFrame({'a': [1, 2, 3, 1, 2, 3]}, index=index)
    observed = pd.DataFrame({'a': [2, 2]}, index=[2000, 2001])
    got = skill.lee(realizations, observed, ['a'], [2000, 2001]).loc['a']
    assert got['rank'] == pytest.approx(50)  # totals 2, 4, 6 against 4
    assert got['observed'] == 2 and got['simulated'] == 2 and got['t'] == 0


def test_perfect_counts_score_one_on_every_test():
    y = np.array([10.0, 20.0, 40.0, 30.0])
    assert skill.emanuel(y, y) == pytest.approx({"D'": 1, "K'": 1, 'r2': 1, "chi2'": 1})


def test_k_skill_is_one_inside_the_poisson_range_and_falls_outside_it():
    syn = np.array([100.0])
    assert skill.k_skill([110.0], syn) == 1  # inside 84-117
    upper = skill.poisson.ppf(0.95, 100)
    assert skill.k_skill([150.0], syn) == pytest.approx(1 - (150 - upper) / 100)


def test_tests_leave_out_bins_with_five_or_fewer_events():
    obs, syn = np.array([3.0, 20.0, 40.0]), np.array([30.0, 20.0, 40.0])
    assert skill.chi2_skill(obs, syn) == 1 and skill.k_skill(obs, syn) == 1
    assert np.isnan(skill.deviance_skill([1.0, 2.0], [10.0, 10.0]))


def test_deviance_skill_normalizes_by_the_mean_synthetic_count():
    obs, syn = np.array([10.0, 30.0]), np.array([12.0, 28.0])

    def deviance(o, s):
        return np.sum(2 * (o * np.log(o / s) - (o - s)))

    want = 1 - 0.4 * deviance(obs, syn) / deviance(obs, np.full(2, 20.0))
    assert skill.deviance_skill(obs, syn) == pytest.approx(want)


def test_ratio_skill_compares_shares_of_basin_storms():
    assert skill.ratio_skill(30, 100, 10, 50) == pytest.approx(1.5)


def test_band_bootstrap_ranks_match_lee_on_the_full_sample():
    index = pd.MultiIndex.from_product([[2000, 2001], range(20)], names=['year', 'realization'])
    rng = np.random.default_rng(1)
    realizations = pd.DataFrame({'a': rng.poisson(5, 40), 'b': rng.poisson(2, 40)}, index=index)
    observed = pd.DataFrame({'a': [5, 6], 'b': [9, 9]}, index=[2000, 2001])
    sim, obs = skill._arrays(realizations, observed, ['a', 'b'], [2000, 2001])
    ranks = skill._ranks(sim, obs, np.array([0, 1]))
    want = skill.lee(realizations, observed, ['a', 'b'], [2000, 2001])['rank'].to_numpy()
    assert ranks == pytest.approx(want)
    assert skill.in_band(ranks) == 0.5  # b lies above every realization
    boot = skill.band_bootstrap(
        {'run': realizations, 'same': realizations}, observed, ['a', 'b'], [2000, 2001], n=20
    )
    assert boot.shape == (20, 2) and (boot['run'] == boot['same']).all()
