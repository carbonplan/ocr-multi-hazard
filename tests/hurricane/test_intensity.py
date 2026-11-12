import numpy as np
import pytest

from ocr.risks.hurricane import intensity


def test_water_steps_start_at_even_nodes_and_skip_land():
    ocean = np.ones((90, 180), bool)
    ocean[(30 + 89) // 2 + 1, :] = False  # land along 31N
    lat = np.array([[20.0], [22.0], [24.0], [26.0], [28.0], [31.0], [33.0]])
    lon = np.full_like(lat, -60.0)
    wind = np.array([[[40.0], [50.0], [60.0], [55.0], [50.0], [40.0], [np.nan]]])
    steps, members = intensity.tables(lat, lon, wind, ocean, 'test', every_node=False)
    assert steps['v0'].tolist() == [40.0, 60.0]  # nodes 0 and 2; the step from node 4 ends on land
    assert steps['dv'].tolist() == [20.0, -10.0]
    assert steps['h_since_peak'].tolist() == [-12, 0]
    m = members.iloc[0]
    assert (m['peak'], m['hours_with_wind'], m['peak_to_end_h']) == (60.0, 30, 18)
    assert not m['ends_over_water']


def test_every_node_steps_for_observed_tracks():
    ocean = np.ones((90, 180), bool)
    lat = np.array([[20.0], [21.0], [22.0], [23.0]])
    lon = np.full_like(lat, -60.0)
    wind = np.array([[[35.0], [45.0], [55.0], [50.0]]])
    steps, _ = intensity.tables(lat, lon, wind, ocean, 'test', every_node=True)
    assert steps['dv'].tolist() == pytest.approx([20.0, 5.0])
