"""Published skill measures for synthetic tropical-cyclone sets against observations.

Lee et al. (2018, section 2.4), the CHAZ paper's own measures, for a set of realizations of a
period:

  - `t_statistic`: the simulated mean minus the observed mean over a standard deviation that pools
    the observed years and the simulated (realization, year) samples, their equations (1)-(2)
  - `rank`: the percentage of realizations whose total over the period lies below the observed
    total, ties counting half; across variables, these ranks form their rank histogram

Emanuel (2025, section 4), for a metric made of counts y_obs(i) and y_syn(i) over bins i:

  - `deviance_skill` D' = max(1 - 0.4 D / D_mean, 0), equations (6)-(8)
  - `k_skill` K' = max(1 - K, 0), equations (2)-(5), (9)-(10)
  - `chi2_skill` chi2' = max(1 - 0.5 chi2, 0), equations (11)-(12)
  - `r2`, the coefficient of determination (section 4c)
  - `ratio_skill` Rk' = Rk_syn / Rk_obs, equation (13)

D, K and chi2 sum only over the bins with y_obs > 5 and y_syn > 5, and K and chi2 are divided
by the number of those bins, as Emanuel does; he does not say whether r2 leaves bins out, and
`r2` uses them all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

MIN_COUNT = 5  # Emanuel's tests keep bins where both counts exceed this


def t_statistic(simulated: pd.Series, observed: pd.Series) -> float:
    """Lee et al. (2018) equations (1)-(2). `simulated` holds one value per (realization, year)
    sample and `observed` one per year."""
    sim, obs = np.asarray(simulated, float), np.asarray(observed, float)
    n_o, n_m = obs.size, sim.size
    pooled = np.sqrt(((n_o - 1) * obs.var(ddof=1) + (n_m - 1) * sim.var(ddof=1)) / (n_o + n_m - 2))
    return float((sim.mean() - obs.mean()) / pooled) if pooled > 0 else np.nan


def rank(simulated_totals, observed_total: float) -> float:
    """Rank of the observed total among the realizations' totals, in percent: the share of
    realizations below it, ties counting half."""
    sims = np.asarray(simulated_totals, float)
    below = (sims < observed_total).sum() + 0.5 * (sims == observed_total).sum()
    return float(100 * below / sims.size)


def lee(realizations: pd.DataFrame, observed: pd.DataFrame, variables, years) -> pd.DataFrame:
    """Lee et al. (2018)'s measures of each of `variables` over `years`: `realizations` is
    indexed by (year, realization), `observed` by year. Also the observed and simulated means
    per year and the 5th and 95th percentiles of the realizations' means."""
    years, variables = list(years), list(variables)
    sim, obs = _arrays(realizations, observed, variables, years)
    totals = sim.sum(axis=1)
    rows = {}
    for k, v in enumerate(variables):
        per_year = totals[:, k] / len(years)
        rows[v] = {
            'observed': obs[:, k].mean(),
            'simulated': sim[:, :, k].mean(),
            '5%': np.quantile(per_year, 0.05),
            '95%': np.quantile(per_year, 0.95),
            't': t_statistic(sim[:, :, k].ravel(), obs[:, k]),
            'rank': rank(totals[:, k], obs[:, k].sum()),
        }
    return pd.DataFrame(rows).T


def _arrays(realizations: pd.DataFrame, observed: pd.DataFrame, variables, years):
    """(realization, year, variable) and (year, variable) arrays over `years`, which may repeat;
    a year missing from either table is an error."""
    years = list(years)
    sim = realizations[list(variables)].unstack('year')
    if sim.isna().any().any():
        raise ValueError('a realization is missing a year')
    sim = np.stack([sim[v][years].to_numpy(float) for v in variables], axis=-1)
    return sim, observed.loc[years, list(variables)].to_numpy(float)


def _ranks(sim: np.ndarray, obs: np.ndarray, sample) -> np.ndarray:
    """rank() of each variable over the years in `sample` (indices, repeats allowed)."""
    totals, observed = sim[:, sample].sum(axis=1), obs[sample].sum(axis=0)
    below = (totals < observed).mean(axis=0) + 0.5 * (totals == observed).mean(axis=0)
    return 100 * below


def in_band(ranks, low: float = 5, high: float = 95) -> float:
    """Share of variables whose observed total lies within the realizations' `low`-`high`
    percentile range."""
    ranks = np.asarray(ranks, float)
    return float(((ranks >= low) & (ranks <= high)).mean())


def _t_values(sim: np.ndarray, obs: np.ndarray, sample, scale=None) -> np.ndarray:
    """t_statistic() of each variable over the years in `sample`; with `scale`, the difference of
    means is divided by it instead of the pooled standard deviation."""
    s, o = sim[:, sample].reshape(-1, sim.shape[2]), obs[sample]
    if scale is None:
        n_o, n_m = o.shape[0], s.shape[0]
        scale = np.sqrt(
            ((n_o - 1) * o.var(axis=0, ddof=1) + (n_m - 1) * s.var(axis=0, ddof=1))
            / (n_o + n_m - 2)
        )
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(scale > 0, (s.mean(axis=0) - o.mean(axis=0)) / scale, np.nan)


def mean_abs_t(t) -> float:
    return float(np.nanmean(np.abs(np.asarray(t, float))))


def _summary(statistic: str, sim, obs, sample, scale=None) -> float:
    if statistic == 'in band':
        return in_band(_ranks(sim, obs, sample))
    if statistic == 'mean |t|':
        return mean_abs_t(_t_values(sim, obs, sample, scale))
    raise ValueError(f'unknown statistic {statistic!r}')


def bootstrap(
    sources: dict[str, pd.DataFrame],
    observed: pd.DataFrame,
    variables,
    years,
    n: int = 2000,
    seed: int = 0,
    statistic: str = 'in band',
    scale: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`statistic` ('in band', or 'mean |t|' over the variables) of each source over `n`
    resamples of whole years, the same years for every source. With `scale` (realizations of
    one source), every source's t is divided by that source's simulated standard deviation of each
    variable over the resampled years, a common denominator."""
    years = list(years)
    arrays = {k: _arrays(r, observed, variables, years) for k, r in sources.items()}
    common = None if scale is None else _arrays(scale, observed, variables, years)[0]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        sample = rng.integers(0, len(years), len(years))
        denominator = (
            None
            if common is None
            else common[:, sample].reshape(-1, common.shape[2]).std(axis=0, ddof=1)
        )
        rows.append(
            {
                k: _summary(statistic, sim, obs, sample, denominator)
                for k, (sim, obs) in arrays.items()
            }
        )
    return pd.DataFrame(rows)


def _kept(y_obs, y_syn):
    y_obs, y_syn = np.asarray(y_obs, float), np.asarray(y_syn, float)
    keep = (y_obs > MIN_COUNT) & (y_syn > MIN_COUNT)
    return y_obs, y_syn, keep


def _deviance(y_obs, y_syn) -> float:
    return float(np.sum(2 * (y_obs * np.log(y_obs / y_syn) - (y_obs - y_syn))))


def deviance_skill(y_obs, y_syn) -> float:
    """D' (Emanuel 2025, equations 6-8): the Poisson deviance over the kept bins, normalized by
    the deviance from the mean synthetic count."""
    y_obs, y_syn, keep = _kept(y_obs, y_syn)
    if not keep.any():
        return np.nan
    o, s = y_obs[keep], y_syn[keep]
    d_mean = _deviance(o, np.full_like(s, s.mean()))
    return max(1 - 0.4 * _deviance(o, s) / d_mean, 0.0) if d_mean > 0 else np.nan


def k_skill(y_obs, y_syn) -> float:
    """K' (Emanuel 2025, equations 2-5, 9-10): how far each kept observed count lies outside the
    5th-95th percentile range of a Poisson distribution with the synthetic count as its mean,
    over the synthetic count, averaged over the kept bins."""
    y_obs, y_syn, keep = _kept(y_obs, y_syn)
    if not keep.any():
        return np.nan
    o, s = y_obs[keep], y_syn[keep]
    upper, lower = poisson.ppf(0.95, s), poisson.ppf(0.05, s)
    outside = np.where(o >= s, np.maximum(o - upper, 0), np.maximum(lower - o, 0))
    return max(1 - float(np.mean(outside / s)), 0.0)


def chi2_skill(y_obs, y_syn) -> float:
    """chi2' (Emanuel 2025, equations 11-12): the mean squared difference over the square of the
    synthetic count, over the kept bins."""
    y_obs, y_syn, keep = _kept(y_obs, y_syn)
    if not keep.any():
        return np.nan
    o, s = y_obs[keep], y_syn[keep]
    return max(1 - 0.5 * float(np.mean((o - s) ** 2 / s**2)), 0.0)


def r2(y_obs, y_syn) -> float:
    """The coefficient of determination (Emanuel 2025, section 4c): the squared correlation,
    over every bin."""
    y_obs, y_syn = np.asarray(y_obs, float), np.asarray(y_syn, float)
    return float(np.corrcoef(y_obs, y_syn)[0, 1] ** 2)


def ratio_skill(crossings_syn, basin_syn, crossings_obs, basin_obs) -> float:
    """Rk' (Emanuel 2025, equations 1 and 13): the synthetic share of basin storms that cross a
    coastline over the observed share, neither normalized."""
    return (crossings_syn / basin_syn) / (crossings_obs / basin_obs)


def emanuel(y_obs, y_syn) -> dict:
    """D', K', r2 and chi2' of one metric."""
    return {
        "D'": deviance_skill(y_obs, y_syn),
        "K'": k_skill(y_obs, y_syn),
        'r2': r2(y_obs, y_syn),
        "chi2'": chi2_skill(y_obs, y_syn),
    }
