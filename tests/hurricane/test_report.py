import numpy as np
import pandas as pd
import pytest
from test_validate import Land

from ocr.risks.hurricane import report


def test_chaz_form_keeps_every_other_node_and_averages_between():
    wind = np.array([[[40.0], [60.0], [50.0], [80.0], [70.0]]])  # member, node, storm
    assert report.chaz_form(wind)[0, :, 0].tolist() == [40, 45, 50, 60, 70]
    assert np.isnan(report.chaz_form(wind, 1)[0, [0, 4], 0]).all()
    assert report.chaz_form(wind, 1)[0, 1:4, 0].tolist() == [60, 70, 80]


def test_month_of_days_since_1950():
    days = np.array([0.0, 31.0, 365.0 + 59.25, np.nan])
    assert report._month(days).tolist() == [1, 2, 3, 0]


def test_double_sided_accumulates_towards_the_peak_from_each_end():
    hist = np.array([1.0, 2.0, 10.0, 3.0, 1.0])
    assert report.double_sided(hist, 2).tolist() == [1, 3, 14, 4, 1]


def test_counts_a_north_atlantic_storm_and_its_us_landfall():
    lon = np.linspace(-70.0, -90.0, 6)[:, None]  # west along 30N, 6-hourly, onto land at -80
    lat = np.full_like(lon, 30.0)
    month = np.full(lat.shape, 9)
    wind = np.array([[45.0, 60.0, 70.0, 80.0, 50.0, 30.0], [30.0] * 6])[:, :, None]
    c = report.counts(lat, lon, wind, month, Land())
    assert c['na'] == 1 and c['month 9'] == 1 and c['lmi >= 80'] == 1 and c['lmi >= 85'] == 0
    assert c['us'] == 1 and c['lf lmi >= 80'] == 1
    # changes of +15, +10, +10, -30 and -20 kt over 6 hours
    assert c['dv 10'] == 2 and c['dv -30'] == 1 and sum(c[f'dv {v}'] for v in report.DV) == 5
    # the track reaches land at 80W, 15 h in, at 75 kt, after 65 kt 6 hours earlier
    assert c['lf wind >= 75'] == 1 and c['lf wind >= 80'] == 0 and c['lf dv 10'] == 1


def test_card_scores_a_copy_of_the_observations_as_perfect():
    counts = {f'lmi >= {v}': max(100 - v, 0) for v in report.LMI}
    counts |= {f'lf lmi >= {v}': max(60 - v, 0) for v in report.LMI}
    counts |= {f'lf wind >= {v}': max(50 - v, 0) for v in report.LMI}
    counts |= {f'dv {c}': 100 - abs(c) for c in report.DV}
    counts |= {f'lf dv {c}': 50 - abs(c) // 2 for c in report.DV}
    counts |= {f'month {m}': 10 * m for m in range(1, 13)}
    counts |= {'na': 60, 'us': 20}
    obs = pd.DataFrame([{'members': 1, **counts}], index=[2000])
    syn = obs * 40
    syn['members'] = 40
    card = report.card({'IBTrACS': obs, 'run': syn}, [2000])
    assert card['run'].dropna().to_numpy() == pytest.approx(1.0)
