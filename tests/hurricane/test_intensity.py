import numpy as np
import pytest

from ocr.risks.hurricane import intensity, landfall


def ocean_except(*cells):
    ocean = np.ones((90, 180), bool)
    for lat, lon in cells:
        ocean[int((lat + 89) // 2), int(lon // 2)] = False
    return ocean


def test_model_cell_matches_the_models_nearest_cell_search():
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-60, 60, 500), rng.uniform(0, 360, 500)
    xx, yy = np.meshgrid(intensity.GRID_LON, intensity.GRID_LAT)
    londis = 2 * np.pi * 6371.0 * np.cos(np.deg2rad(yy)) / 360
    row, col = intensity.model_cell(lat, lon)
    for k in range(lat.size):
        d = np.hypot(londis * (xx - lon[k]), 110 * (yy - lat[k]))
        assert np.unravel_index(np.argmin(d), d.shape) == (row[k], col[k])


def test_a_point_that_rounds_to_ocean_but_the_model_reads_as_land():
    ocean = ocean_except((31, 266))
    assert not intensity.over_water(ocean, np.array([29.999]), np.array([265.1]))[0]


def test_water_steps_start_at_even_nodes_and_skip_steps_ending_on_land():
    lat = np.array([[20.0], [22.0], [24.0], [26.0], [28.0], [29.0], [31.0]])
    lon = np.full_like(lat, -60.0)
    wind = np.array([[[40.0], [50.0], [60.0], [55.0], [50.0], [45.0], [40.0]]])
    steps, members = intensity.tables(lat, lon, wind, ocean_except((27, 300)), 'test', False)
    # even nodes start steps: 0 -> 2 is over water; 2 -> 4 ends on land at 28N (the model reads
    # the 27N cell) and 4 -> 6 starts there; the odd node 1 starts none
    assert steps['v0'].tolist() == [40.0]
    assert steps['dv'].tolist() == [20.0]
    assert steps['h_since_peak'].tolist() == [-12]
    all_water = intensity.tables(lat, lon, wind, np.ones((90, 180), bool), 'test', False)[0]
    assert all_water['v0'].tolist() == [40.0, 60.0, 50.0]
    m = members.iloc[0]
    assert (m['peak'], m['hours_with_wind'], m['peak_to_end_h']) == (60.0, 36, 24)
    assert m['ends_over_water']


def test_members_count_only_when_they_reach_34_kt_inside_the_north_atlantic():
    lat = np.array([[20.0], [20.0], [20.0]])
    lon = np.array([[-40.0], [-30.0], [-20.0]])
    inside = np.array([[[30.0], [33.0], [40.0]]])
    outside = np.array([[[40.0], [33.0], [30.0]]])
    pacific_lon = np.array([[-120.0], [-121.0], [-122.0]])
    ocean = np.ones((90, 180), bool)
    assert len(intensity.tables(lat, lon, inside, ocean, 't', True)[1]) == 1
    assert len(intensity.tables(lat, pacific_lon, outside, ocean, 't', True)[1]) == 0


def test_observed_tracks_sit_on_a_regular_6_hourly_grid():
    rows = [
        ('SID', 'SEASON', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND'),
        ('', 'Year', '', 'deg', 'deg', 'kts'),
        ('A', '2000', '2000-08-01 00:00:00', '20', '-60', '40'),
        ('A', '2000', '2000-08-01 12:00:00', '21', '-61', '50'),
        ('A', '2000', '2000-08-01 18:00:00', '22', '-62', '55'),
    ]
    csv = '\n'.join(','.join(r) for r in rows).encode()
    lat, lon, wind, season = landfall.observed_arrays(csv)
    assert lat[:, 0].tolist()[:1] == [20.0] and np.isnan(lat[1, 0])
    assert wind[0, :, 0][~np.isnan(wind[0, :, 0])].tolist() == [40.0, 50.0, 55.0]
    steps, _ = intensity.tables(lat, lon, wind, np.ones((90, 180), bool), 't', every_node=True)
    assert steps['dv'].tolist() == pytest.approx(
        [10.0]
    )  # 00 -> 12 h; 12 -> 18 h is not a 12-hour step
    assert season.tolist() == [2000]
