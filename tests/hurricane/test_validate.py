import numpy as np
import pandas as pd
import pytest

from ocr.risks.hurricane import landfall, validate

USA = validate.COUNTRIES.index('USA') + 2


def grid():
    """1-degree raster: the US from 100W to 80W between 25N and 45N, sea elsewhere."""
    g = np.zeros((180, 360), dtype='int16')
    g[90 - 45 : 90 - 25, 180 - 100 : 180 - 80] = USA
    return g


class Land:
    """landfall.USLand over the same US block."""

    def __call__(self, lat, lon):
        lon = landfall.wrap(lon)
        return np.isfinite(lat) & (lon >= -100) & (lon < -80) & (lat >= 25) & (lat < 45)

    def water(self, lat, lon):
        return np.isfinite(lat) & ~self(lat, lon)

    def anywhere(self, lat, lon):
        return self(lat, lon)

    def inside(self, lat, lon):
        return np.isfinite(lat)

    def atlantic_km(self, lat, lon):
        return np.where(np.isfinite(lat), np.where(self(lat, lon), 0.0, 1000.0), np.nan)

    def lee(self, lat, lon):
        return self.anywhere(lat, lon), self(lat, lon)


def test_lookup_reads_the_raster_cell_and_marks_missing_points():
    got = validate.lookup(grid(), np.array([30.5, 30.5, np.nan]), np.array([-90.5, -70.5, 0.0]))
    assert got.tolist() == [USA, 0, -1]


def test_counts_a_us_landfall_and_time_over_sea_per_member():
    lon = np.linspace(-70.0, -90.0, 6)[:, None]  # west along 30N, 6-hourly
    lat = np.full_like(lon, 30.0)
    wind = np.array([[70.0] * 6, [40.0] * 6])[:, :, None]
    c = validate.counts(lat, lon, wind, grid(), Land())
    assert c['members'] == 2
    assert c['landfall USA >= 34'] == 2 and c['landfall USA >= 64'] == 1
    assert c['Atl+Gulf landfalls >= 64 kt'] == 1 and c['FL landfalls >= 64 kt'] == 1
    assert c['NA TS'] == 2 and c['NA peak >= 64'] == 1
    assert c['TS formed global'] == 2
    # nodes at -70, -74, -78 are over sea in the W Atlantic box (to 82W); -82 is US land
    assert c['NA time >= 64: W Atlantic'] == 3
    assert c['Pacific coast landfalls >= 34 kt'] == 0


def table(na, landfalls, members, years=(2000, 2001)):
    rows = {
        'members': members,
        'NA TS': na,
        'NA peak >= 64': na / 2,
        'NA peak >= 96': na / 4,
        'Atl+Gulf landfalls >= 64 kt': landfalls,
        'landfall PHL >= 64': landfalls,
    }
    return pd.DataFrame({k: [v] * len(years) for k, v in rows.items()}, index=list(years))


def test_metrics_scale_north_atlantic_counts_to_the_observed_storm_count():
    tables = {'IBTrACS': table(10, 2, 1), 'run': table(800, 80, 40)}
    m = validate.metrics(tables, [2000, 2001])
    assert m.loc['NA TS storms/yr', 'run'] == pytest.approx(20)
    # twice the observed storm rate: Atlantic landfalls halved, country landfalls left raw
    assert m.loc['Atl+Gulf landfalls >= 64 kt', 'run'] == pytest.approx(1.0)
    assert m.loc['landfall PHL >= 64', 'run'] == pytest.approx(2.0)
    assert m.loc['NA share peak >= 64 kt', 'run'] == pytest.approx(0.5)


def test_score_is_the_mean_absolute_log_ratio_with_a_floor(monkeypatch):
    monkeypatch.setattr(validate, 'SCORED', ['a', 'b'])
    t = pd.DataFrame({'IBTrACS': [1.0, 2.0], 'run': [np.e, 0.0]}, index=['a', 'b'])
    assert validate.score(t)['run'] == pytest.approx((1 + abs(np.log(validate.FLOOR))) / 2)


def test_bootstrap_resamples_the_same_years_for_every_source(monkeypatch):
    monkeypatch.setattr(validate, 'SCORED', ['Atl+Gulf landfalls >= 64 kt'])
    obs = table(10, 2, 1, years=(2000, 2001, 2002))
    run = obs * 40
    b = validate.bootstrap({'IBTrACS': obs, 'run': run, 'same': run}, [2000, 2001, 2002], n=5)
    assert b.shape == (5, 2)
    assert (b['run'] == b['same']).all()


def test_a_run_file_counts_only_its_own_year(monkeypatch):
    monkeypatch.setattr(validate, '_grid', grid())
    monkeypatch.setattr(validate, '_land', Land())
    lon = np.linspace(-70.0, -90.0, 6)[:, None]
    lat = np.full_like(lon, 30.0)
    wind = np.full((40, 6, 1), 70.0)
    rows = validate._by_year(lat, lon, wind, np.array([2000]), [2000])
    assert [r['year'] for r in rows] == [2000] and rows[0]['members'] == 40
    empty = validate._by_year(lat[:, :0], lon[:, :0], wind[..., :0], np.array([], int), [2001])
    assert empty == [{'year': 2001, 'members': 40}]


def test_score_leaves_out_what_was_not_observed(monkeypatch):
    monkeypatch.setattr(validate, 'SCORED', ['a', 'b'])
    t = pd.DataFrame({'IBTrACS': [1.0, 0.0], 'run': [np.e, 5.0]}, index=['a', 'b'])
    assert validate.score(t)['run'] == pytest.approx(1.0)


def test_bootstrap_survives_resamples_without_a_rare_event(monkeypatch):
    monkeypatch.setattr(validate, 'SCORED', ['Atl+Gulf landfalls >= 64 kt', 'landfall PHL >= 64'])
    obs = table(10, 2, 1, years=(2000, 2001))
    obs.loc[2001, ['Atl+Gulf landfalls >= 64 kt', 'landfall PHL >= 64']] = [0, 1]
    obs.loc[2000, 'landfall PHL >= 64'] = 1
    run = table(400, 40, 40, years=(2000, 2001))
    b = validate.bootstrap({'IBTrACS': obs, 'run': run}, [2000, 2001], n=50)
    assert b['run'].notna().all()
    obs.loc[2000, 'Atl+Gulf landfalls >= 64 kt'] = 0
    with pytest.raises(ValueError, match='no observed events'):
        validate.bootstrap({'IBTrACS': obs, 'run': run}, [2000, 2001], n=5)


def test_country_landfalls_count_the_last_independent_offshore_wind():
    lon = np.linspace(-70.0, -90.0, 6)[:, None]
    lat = np.full_like(lon, 30.0)
    # interpolated to the crossing hour (16 h) the wind would be 90 kt
    wind = np.array([[50.0, 60.0, 110.0, 80.0, 40.0, 30.0]])[:, :, None]
    c = validate.counts(lat, lon, wind, grid(), Land())
    assert c['landfall USA >= 96'] == 1 and c['Atl+Gulf landfalls >= 96 kt'] == 1


def test_counts_with_no_country_landfall():
    lat = np.full((4, 1), -40.0)
    lon = np.linspace(0.0, 6.0, 4)[:, None]
    wind = np.full((2, 4, 1), 70.0)
    c = validate.counts(lat, lon, wind, grid(), Land())
    assert c['landfall USA >= 34'] == 0 and c['TS formed global'] == 2
