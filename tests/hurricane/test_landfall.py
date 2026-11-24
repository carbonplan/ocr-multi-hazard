import numpy as np
import pytest

from ocr.risks.hurricane import landfall


class Land:
    """US land from 115W to 80W between 25N and 45N; other land there from 22N to 25N."""

    def __call__(self, lat, lon):
        lon = landfall.wrap(lon)
        return np.isfinite(lat) & (lon < -80) & (lon > -115) & (lat >= 25) & (lat < 45)

    def water(self, lat, lon):
        lon = landfall.wrap(lon)
        return np.isfinite(lat) & ~((lon < -80) & (lon > -115) & (lat >= 22) & (lat < 45))

    def anywhere(self, lat, lon):
        return ~self.water(lat, lon) & np.isfinite(lat)

    def inside(self, lat, lon):
        return np.isfinite(lat)

    def atlantic_km(self, lat, lon):
        return np.where(np.isfinite(lat), np.where(self(lat, lon), 0.0, 1000.0), np.nan)

    def lee(self, lat, lon):
        return self.anywhere(lat, lon), self(lat, lon)


def test_unwrap_crosses_the_date_line():
    lon = np.array([[178.0], [-179.0], [-176.0], [np.nan]])
    got = landfall.unwrap(lon)
    assert got[:3, 0] == pytest.approx([178.0, 181.0, 184.0])
    assert np.isnan(got[3, 0])


def test_unwrap_keeps_positions_after_a_missing_node():
    lon = np.array([[-70.0, 178.0], [np.nan, np.nan], [-80.0, -179.0], [-81.0, -178.0]])
    got = landfall.unwrap(lon)
    assert got[[0, 2, 3], 0] == pytest.approx([-70.0, -80.0, -81.0])
    assert got[[0, 2, 3], 1] == pytest.approx([178.0, 181.0, 182.0])
    assert np.isnan(got[1]).all()


def test_region_puts_southern_california_on_the_pacific_coast():
    got = landfall.region(np.array([32.7, 29.0]), np.array([-117.2, -95.0]))
    assert got.tolist() == ['Pacific', 'TX-AL']


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


def test_pacific_landfalls_are_counted_apart_from_us_landfalls():
    lat, lon, wind = storm([[70] * 6])
    pacific_lon = np.linspace(-130.0, -110.0, 6)[:, None]
    lat2 = np.concatenate([lat, np.full_like(lat, 35.0)], axis=1)
    lon2 = np.concatenate([lon, pacific_lon], axis=1)
    wind2 = np.concatenate([wind, wind], axis=2)
    rows = landfall.storm_rows(lat2, lon2, wind2, np.array([2000, 2000]), Land(), 'test', 'u0')
    assert rows['lf_region'].tolist() == ['FL', 'Pacific']
    s = landfall.summarize(rows, member_years=1, scale=1.0)
    assert s['US landfalls/yr (>= 34 kt)'] == pytest.approx(1)
    assert s['US Pacific landfalls/yr (>= 34 kt)'] == pytest.approx(1)
    assert s['US Pacific landfalls/yr >= 64 kt'] == pytest.approx(1)


def chaz_member(evens, n):
    """One member as get_stochastic stores it: predictions at even nodes, odd nodes between two
    of them interpolated; NaN beyond."""
    w = np.full(n, np.nan)
    w[0 : 2 * len(evens) : 2] = evens
    for k in range(1, 2 * len(evens) - 1, 2):
        w[k] = 0.5 * (w[k - 1] + w[k + 1])
    return w


@pytest.mark.parametrize(
    ('evens', 'rule'),
    [
        ([30, 50, 60, 34, 33, 30], 'fell to 35 kt'),
        (
            [25, 30, 30, 60],
            'other',
        ),  # the next prediction fell below 10 kt; 60 is not a peak before
        ([30, 50, 60, 55, 12], 'other'),
        ([30, 50, 60, 55, 50, 45, 40, 38], 'track end'),
    ],
)
def test_end_rules_follow_get_stochastic(evens, rule):
    n, last = 20, 14  # track to node 14; the model reaches node 14
    wind = chaz_member(evens, n)[None, :, None]
    w_end = np.array([[2 * (len(evens) - 1)]])
    reachable = landfall.reach(np.array([last]), n)[None]
    assert landfall.end_rules(wind, w_end, reachable, observed=False)[0, 0] == rule


@pytest.mark.parametrize(
    ('last', 'n', 'expected'),
    [(10, 15, 10), (11, 15, 10), (12, 13, 10), (20, 13, 10), (9, 30, 8)],
)
def test_reach_of_the_intensity_model(last, n, expected):
    # it runs to min(last, n - 5) + 1 in even steps and stores 12 h (2 nodes) ahead, on the track
    assert landfall.reach(np.array([last]), n)[0] == expected


def test_wind_around_landfall_and_end_after_landfall():
    lat, lon, wind = storm([[70] * 6, [70, 70, 70, 70, np.nan, np.nan]])
    rows = landfall.storm_rows(lat, lon, wind, np.array([2000]), Land(), 'test', 'u0')
    a, b = rows.iloc[0], rows.iloc[1]
    assert a['lf_w+12'] == pytest.approx(70) and np.isnan(a['lf_w+48'])
    assert b['end_h_after_lf'] == pytest.approx(2)  # last wind at 18 h, landfall at 16 h
    assert b['end_before_track_h'] == 12 and b['end_on_land'] == 1.0


def test_landfall_wind_is_interpolated_to_the_crossing_hour_and_offshore_wind_is_kept():
    # the track reaches land (west of 80W) at 15 h 10 min, between nodes 2 (12 h, 110 kt,
    # offshore) and 3 (18 h, 90 kt, on land)
    lat, lon, wind = storm([[50, 60, 110, 90, 40, 30]])
    r = landfall.storm_rows(lat, lon, wind, np.array([2000]), Land(), 'test', 'u0').iloc[0]
    assert r['lf_wind'] == pytest.approx(110 - 20 * (15 + 1 / 6 - 12) / 6)
    assert r['lf_offshore_wind'] == pytest.approx(110)


def test_landfall_wind_skips_the_interpolated_odd_node():
    # nodes 6 h apart along 30N from 66W; the crossing at 22 h follows node 3, which CHAZ
    # interpolates, so the wind of node 2 counts
    lon = np.linspace(-66.0, -90.0, 7)[:, None]
    lat = np.full_like(lon, 30.0)
    wind = np.array([[50.0, 60.0, 100.0, 80.0, 60.0, 40.0, 30.0]])[:, :, None]
    rows = landfall.storm_rows(lat, lon, wind, np.array([2000]), Land(), 'test', 'u0')
    assert rows.iloc[0]['lf_offshore_wind'] == pytest.approx(100)


class Coast:
    """US land west of 80W and north of 25N; other land on an island from 77.5W to 76.5W."""

    def __call__(self, lat, lon):
        lon = landfall.wrap(lon)
        return np.isfinite(lat) & (lon < -80) & (lat >= 25)

    def water(self, lat, lon):
        lon = landfall.wrap(lon)
        return np.isfinite(lat) & ~self(lat, lon) & ~((lon > -77.5) & (lon < -76.5))

    def anywhere(self, lat, lon):
        return ~self.water(lat, lon) & np.isfinite(lat)

    def inside(self, lat, lon):
        return np.isfinite(lat)

    def atlantic_km(self, lat, lon):
        return np.where(np.isfinite(lat), np.where(self(lat, lon), 0.0, 1000.0), np.nan)

    def lee(self, lat, lon):
        return self.anywhere(lat, lon), self(lat, lon)


def test_first_crossing_counts_and_the_offshore_wind_takes_the_strongest():
    # ashore at 9 h (approach wind 60 kt at node 0), back over water south of 25N, ashore again
    # at 29 h (110 kt at node 4)
    lat = np.array([26.0, 26.0, 26.0, 20.0, 20.0, 26.0, 30.0])[:, None]
    lon = np.array([-76.0, -79.0, -82.0, -85.0, -88.0, -90.0, -90.0])[:, None]
    wind = np.array([[60.0, 60.0, 50.0, 50.0, 110.0, 90.0, 80.0]])[:, :, None]
    rows = landfall.storm_rows(lat, lon, wind, np.array([2000]), Coast(), 'test', 'u0')
    r = rows.iloc[0]
    # first crossing at 8 h 10 min, between nodes 1 (6 h, 60 kt) and 2 (12 h, 50 kt)
    assert r['lf_wind'] == pytest.approx(60 - 10 * (8 + 1 / 6 - 6) / 6) and r['lf_region'] == 'FL'
    assert np.isnan(r['lf_first_offshore_wind'])  # the approach to it crosses the island
    assert r['lf_offshore_wind'] == pytest.approx(110) and r['lf_offshore_region'] == 'TX-AL'
    assert r['lf_crossings'] == 2


def test_no_landfall_wind_when_the_approach_crosses_other_land():
    # node 0 sits on the island, so no offshore node lies within 12 h of the crossing at 10 h
    lat = np.full((3, 1), 30.0)
    lon = np.array([-77.0, -79.0, -83.0])[:, None]
    wind = np.array([[100.0, 100.0, 100.0]])[:, :, None]
    r = landfall.storm_rows(lat, lon, wind, np.array([2000]), Coast(), 'test', 'u0').iloc[0]
    assert r['crosses'] and np.isnan(r['lf_offshore_wind']) and r['lf_wind'] == pytest.approx(100)


def test_wind_around_landfall_is_read_from_the_first_crossing_hour():
    lat = np.array([26.0, 26.0, 26.0, 20.0, 20.0, 26.0, 30.0])[:, None]
    lon = np.array([-76.0, -79.0, -82.0, -85.0, -88.0, -90.0, -90.0])[:, None]
    wind = np.array([[60.0, 60.0, 50.0, 50.0, 110.0, 90.0, 80.0]])[:, :, None]
    r = landfall.storm_rows(lat, lon, wind, np.array([2000]), Coast(), 'test', 'u0').iloc[0]
    assert r['lf_w+0'] == pytest.approx(60 - 10 * 3 / 6)  # 9 h, the first hour on land
    assert r['lf_w+12'] == pytest.approx(80.0)  # 21 h, halfway from node 3 to node 4


def test_near_coast_peak_reads_even_nodes_near_us_land():
    lat, lon, wind = storm([[50, 60, 110, 120, 100, 30]])
    r = landfall.storm_rows(lat, lon, wind, np.array([2000]), Land(), 'test', 'u0').iloc[0]
    assert r['near_peak'] == pytest.approx(100)  # nodes 3-5 are on land; node 3 is odd


def test_lee_landfalls_interpolate_to_15_minutes():
    # the centre is on land from 15 h 15 min, 13/24 of the way from node 2 to node 3
    lat, lon, wind = storm([[50, 60, 110, 90, 40, 30], [50, 60, 70, 60, 40, 30]])
    rows = landfall.storm_rows(lat, lon, wind, np.array([2000]), Land(), 'test', 'u0')
    assert rows['lee_96'].tolist() == [1, 0] and rows['lee_64'].tolist() == [1, 1]


HEADER = (
    'SID,SEASON,NAME,ISO_TIME,LAT,LON,USA_WIND,WMO_WIND,WMO_AGENCY,TRACK_TYPE\n'
    ',Year,,,deg,deg,kts,kts,,\n'
)


def test_observed_records_fill_missing_usa_wind_from_the_agency_as_climada_does():
    csv = (
        HEADER
        + 'A,2000,X,2000-01-01 00:00:00,-35,170,70, ,wellington,main\n'
        + 'A,2000,X,2000-01-01 06:00:00,-36,171, ,88,wellington,main\n'
        + 'A,2000,X,2000-01-01 07:30:00,-36,171, ,88,wellington,main\n'
        + 'B,2000,Y,2000-01-01 00:00:00,30,140, ,83.3,tokyo,main\n'
        + 'C,2000,Z,2000-01-01 00:00:00,30,140,50, ,,spur-other\n'
    ).encode()
    r = landfall.observed_records(csv, (2000, 2000))
    assert r['SID'].tolist() == ['A', 'A', 'B']
    assert r['wind'].tolist() == pytest.approx([70, 100, 100])
    assert r['wind_source'].tolist() == ['usa', 'wellington', 'tokyo']


def test_lee_landfalls_need_both_100_km_and_6_hours_between_them():
    # ashore at 1 h 45 min, back over water, ashore again at 7 h 15 min: 5.5 h later, 500 km away
    lat = np.array([30.0, 24.0, 30.0])[:, None]
    lon = np.array([-79.0, -83.0, -87.0])[:, None]
    wind = np.full((1, 3, 1), 100.0)
    rows = landfall.storm_rows(lat, lon, wind, np.array([2000]), Coast(), 'test', 'u0')
    assert rows.iloc[0]['lee_34'] == 1


def test_lee_records_write_to_parquet(tmp_path):
    lat, lon, wind = storm([[50, 60, 110, 90, 40, 30]])
    out = []
    landfall.storm_rows(lat, lon, wind, np.array([2000]), Land(), 'test', 'u0', out)
    records = out[0]
    assert records['region'].tolist() == ['FL']
    records.to_parquet(tmp_path / 'lee.parquet')
