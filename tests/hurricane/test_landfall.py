import numpy as np
import pytest

from ocr.risks.hurricane import landfall


class Land:
    """US land west of 80W between 25N and 45N; other land west of 80W from 22N to 25N."""

    def __call__(self, lat, lon):
        lon = landfall.wrap(lon)
        return np.isfinite(lat) & (lon < -80) & (lat >= 25) & (lat < 45)

    def water(self, lat, lon):
        lon = landfall.wrap(lon)
        return np.isfinite(lat) & ~((lon < -80) & (lat >= 22) & (lat < 45))

    def anywhere(self, lat, lon):
        return ~self.water(lat, lon) & np.isfinite(lat)

    def inside(self, lat, lon):
        return np.isfinite(lat)


def test_unwrap_crosses_the_date_line():
    lon = np.array([[178.0], [-179.0], [-176.0], [np.nan]])
    got = landfall.unwrap(lon)
    assert got[:3, 0] == pytest.approx([178.0, 181.0, 184.0])
    assert np.isnan(got[3, 0])


def test_hourly_is_linear_between_nodes_and_missing_next_to_a_gap():
    x = np.array([0.0, 6.0, np.nan])
    got = landfall.hourly(x)
    assert got[:7] == pytest.approx(np.arange(7.0))
    assert np.isnan(got[7:]).all()


def test_crossing_needs_water_the_hour_before():
    # from the Gulf onto US land; then from other land (Mexico) onto US land
    lat = np.array([[30.0, 20.0], [30.0, 24.0], [30.0, 26.0]])
    lon = np.array([[-79.5, -85.0], [-80.5, -85.0], [-81.5, -85.0]])
    cross = landfall.crossings(lat, lon, Land())
    assert cross[:, 0].tolist() == [False, True, False]
    assert not cross[:, 1].any()


def storm(wind):
    """One storm heading west along 30N from 70W to 90W, 6-hourly; wind (member, node)."""
    lon = np.linspace(-70.0, -90.0, 6)[:, None]
    lat = np.full_like(lon, 30.0)
    return lat, lon, np.asarray(wind, float)[:, :, None]


def test_storm_rows_landfall_wind_region_and_end_of_wind():
    lat, lon, wind = storm([[70] * 6, [70, 70, np.nan, np.nan, np.nan, np.nan]])
    rows = landfall.storm_rows(lat, lon, wind, np.array([2000]), Land(), 'test', 'u0')
    a, b = rows.iloc[0], rows.iloc[1]
    assert a['crosses'] and b['crosses'] and a['genesis_na']
    assert a['lf_wind'] == pytest.approx(70) and a['lf_region'] == 'FL'
    assert np.isnan(b['lf_wind']) and b['lf_region'] is None
    assert a['wind_at_end'] and not b['wind_at_end']


def test_landfall_share_splits_into_crossing_and_wind_at_crossing():
    lat, lon, wind = storm([[70] * 6, [40] * 6, [70, 70, np.nan, np.nan, np.nan, np.nan]])
    lat2 = np.concatenate([lat, np.full_like(lat, 20.0)], axis=1)
    lon2 = np.concatenate([lon, np.full_like(lon, -60.0)], axis=1)
    wind2 = np.concatenate([wind, np.full_like(wind, 50.0)], axis=2)
    rows = landfall.storm_rows(lat2, lon2, wind2, np.array([2000, 2000]), Land(), 'test', 'u0')
    s = landfall.summarize(rows, member_years=1, scale=1.0)
    assert s['NA TS storms crossing the US coast, share'] == pytest.approx(0.5)
    assert s['crossings at >= 34 kt, share'] == pytest.approx(2 / 3)
    assert s['landfall share of NA TS storms'] == pytest.approx(
        s['NA TS storms crossing the US coast, share'] * s['crossings at >= 34 kt, share']
    )
    assert s['tracks from NA genesis crossing the US coast, share'] == pytest.approx(0.5)


def test_end_rules_tell_track_end_from_the_35_kt_stop():
    nan = np.nan
    wind = np.array(
        [
            [[30], [50], [34], [50], [33], [30], [31], [nan], [nan], [nan]],  # fell to 35 kt
            [[30], [50], [60], [55], [50], [45], [40], [38], [36], [nan]],  # runs to the stop
            [[30], [50], [60], [55], [12], [nan], [nan], [nan], [nan], [nan]],  # other
        ]
    ).astype(float)
    alive = np.isfinite(wind)
    w_end = wind.shape[1] - 1 - np.argmax(alive[:, ::-1], axis=1)
    got = landfall.end_rules(wind, w_end, np.array([[8]]), observed=False)
    assert got[:, 0].tolist() == ['fell to 35 kt', 'track end', 'other']


def test_wind_around_landfall_and_end_after_landfall():
    lat, lon, wind = storm([[70] * 6, [70, 70, 70, 70, np.nan, np.nan]])
    rows = landfall.storm_rows(lat, lon, wind, np.array([2000]), Land(), 'test', 'u0')
    a, b = rows.iloc[0], rows.iloc[1]
    assert a['lf_w+12'] == pytest.approx(70) and np.isnan(a['lf_w+48'])
    assert b['end_h_after_lf'] == pytest.approx(2)  # last wind at 18 h, landfall at 16 h
    assert b['end_before_track_h'] == 12 and b['end_on_land'] == 1.0
