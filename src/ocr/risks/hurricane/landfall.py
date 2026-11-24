"""US landfall statistics of CHAZ tracks against IBTrACS, in track space with no wind model.

The same rules for every source:

  - IBTrACS: main tracks, synoptic 6-hourly records, seasons 1981-2019, at the 1-minute wind of
    `observed_records` (USA_WIND, or the WMO agency's wind where USA_WIND is missing). CHAZ and
    IBTrACS tracks are interpolated to hourly, longitudes unwrapped across the date line
  - one US land mask: Natural Earth 1:10m at 1/24 degree, a cell US land wherever the US
    touches it, so narrow land such as the Florida Keys counts
  - a crossing: an hour on US land right after an hour over water
  - US landfall (the lf_ columns), as Emanuel (2025) counts landfalls: each (storm, member) once,
    at its first crossing, at the wind interpolated to that hour. The US rows count the Atlantic
    and Gulf coasts; landfalls on the Pacific coast, which IBTrACS has none of over these years,
    have rows of their own
  - the crossing time is found to 10 minutes along the line between the two hourly positions, and
    lf_wind is interpolated to it
  - lf_first_offshore_wind is the first crossing's approach wind (below)
  - the lf_offshore_ columns count each (storm, member) at its strongest crossing at the approach
    wind (`approach`): the member's wind at the last even node before the crossing hour, at most
    12 h earlier, with every hour from that node to the crossing over water and the member still
    carrying wind at the crossing. CHAZ predicts intensity 12-hourly at even nodes and fills odd
    nodes with the mean of their neighbours; IBTrACS is read the same way, at every other
    synoptic record from a track's first
  - a (storm, member) counts only where it carries wind, a landfall only at 34 kt or more
  - two other measures sit beside them: `near_coast_peak`, the strongest wind within NEAR_KM of
    US land east of 105W, and `lee_records`, every independent US landfall by the rule of Lee et
    al. (2018, section 2.2)
  - CHAZ rates are also given scaled so its North Atlantic tropical-storm count matches the
    observed one, which leaves where storms go and how strong they are
  - the landfall share of North Atlantic storms splits exactly into the share whose track crosses
    the US coast at all, whatever its wind, and the share of those crossings made at 34 kt or more
    (undefined when no track crosses). A CHAZ track runs on after its intensity member ends, so
    the crossing share counts tracks, not storms that still exist

  run    tables for IBTrACS, the published ERA5 tracks and our runs, under validation/chaz/landfall
  rows   rows and per-file statistics of one run of any forcing source over a span of years
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from ocr.risks.hurricane import store

YEARS = (1981, 2019)
TS, HU, MH = 34.0, 64.0, 96.0  # kt
BOX = (-130.0, 15.0, -60.0, 52.0)  # lon0, lat0, lon1, lat1 of the land raster
RES = 1 / 24
ATLANTIC = ('TX-AL', 'FL', 'GA-Car', 'VA-ME')
REGIONS = (*ATLANTIC, 'Pacific', 'other')
TIME_FILL = -54786.0  # 1800-01-01 in days since 1950: CHAZ's marker for no data
PUBLISHED = store.PUBLISHED_TRACKS
OUT = 'validation/chaz/landfall'
APPROACH_H = 12  # hours between even nodes, and the most an approach wind may precede a crossing
AROUND = (-24, -12, -6, 0, 6, 12, 24, 36, 48)  # hours from US landfall with a wind column each
END_RULES = ('track end', 'fell to 35 kt', 'other', 'observed')
NEAR_KM = 300.0
LEE_RES = 0.5  # degrees: the land mask of Lee et al. (2018)
LEE_STEPS = 24  # 15-minute steps per 6-hourly node
LEE_KM, LEE_H = 100.0, 6.0  # a landfall closer than either to the last one counted is not new
EARTH_KM = 6371.0
# CLIMADA's IBTRACS_AGENCY_1MIN_WIND_FACTOR: an agency's wind is scale * 1-minute wind + shift
ONE_MINUTE = {
    'tokyo': (0.60, 23.3),
    'newdelhi': (1.0, 0.0),
    'reunion': (0.88, 0.0),
    'bom': (0.88, 0.0),
    'nadi': (0.88, 0.0),
    'wellington': (0.88, 0.0),
    'cma': (0.871, 0.0),
    'hko': (0.9, 0.0),
}


def wrap(lon):
    return (lon + 180) % 360 - 180


class USLand:
    """US land and any land on a 1/24 degree raster, looked up by cell."""

    def __init__(self):
        import cartopy.io.shapereader as shpreader
        import shapely
        from rasterio import features
        from rasterio.transform import from_origin

        lon0, lat0, lon1, lat1 = BOX
        self.lons = np.arange(lon0, lon1, RES) + RES / 2
        self.lats = np.arange(lat0, lat1, RES) + RES / 2
        gx, gy = np.meshgrid(self.lons, self.lats)
        countries = shpreader.Reader(
            shpreader.natural_earth('10m', 'cultural', 'admin_0_countries')
        )
        us = next(r.geometry for r in countries.records() if r.attributes['ADM0_A3'] == 'USA')
        land = shapely.union_all(
            list(shpreader.Reader(shpreader.natural_earth('10m', 'physical', 'land')).geometries())
        )
        land = shapely.clip_by_rect(land, lon0, lat0, lon1, lat1)
        shapely.prepare(land)
        self.mask = features.rasterize(
            [(us, 1)],
            out_shape=(self.lats.size, self.lons.size),
            transform=from_origin(lon0, lat0 + self.lats.size * RES, RES, RES),
            fill=0,
            all_touched=True,
            dtype='uint8',
        )[::-1].astype(bool)
        self.land = self.mask | shapely.contains_xy(land, gx, gy)
        i, j = np.nonzero(self.mask & (gx >= -105))
        self._atlantic = cKDTree(xyz(self.lats[i], self.lons[j]))
        lee_lon = np.arange(lon0, lon1, LEE_RES) + LEE_RES / 2
        lee_lat = np.arange(lat0, lat1, LEE_RES) + LEE_RES / 2
        lx, ly = np.meshgrid(lee_lon, lee_lat)
        shapely.prepare(us)
        self.lee_us = shapely.contains_xy(us, lx, ly)
        self.lee_land = self.lee_us | shapely.contains_xy(land, lx, ly)

    def water(self, lat, lon):
        return self._lookup(~self.land, lat, lon, outside=True)

    def anywhere(self, lat, lon):
        """Any land, not only US."""
        return self._lookup(self.land, lat, lon, outside=False)

    def inside(self, lat, lon):
        lon = wrap(lon)
        return (
            np.isfinite(lat)
            & (lat >= BOX[1]) & (lat < BOX[3])
            & (lon >= BOX[0]) & (lon < BOX[2])
        )  # fmt: skip

    def __call__(self, lat, lon):
        return self._lookup(self.mask, lat, lon, outside=False)

    def atlantic_km(self, lat, lon):
        """Great-circle distance (km) to the nearest US land cell east of 105W; NaN where
        missing."""
        ok = np.isfinite(lat) & np.isfinite(lon)
        out = np.full(np.shape(lat), np.nan)
        chord, _ = self._atlantic.query(xyz(np.asarray(lat)[ok], np.asarray(lon)[ok]))
        out[ok] = 2 * EARTH_KM * np.arcsin(np.minimum(chord / 2, 1))
        return out

    def lee(self, lat, lon):
        """(any land, US land) on the LEE_RES mask, by the cell's centre."""
        return (
            self._lookup(self.lee_land, lat, lon, outside=False, res=LEE_RES),
            self._lookup(self.lee_us, lat, lon, outside=False, res=LEE_RES),
        )

    def _lookup(self, grid, lat, lon, outside: bool, res: float = RES):
        lon = wrap(lon)
        ok = np.isfinite(lat) & np.isfinite(lon)
        i = np.floor((np.where(ok, lat, 0) - BOX[1]) / res).astype(int)
        j = np.floor((np.where(ok, lon, 0) - BOX[0]) / res).astype(int)
        inside = ok & (i >= 0) & (i < grid.shape[0]) & (j >= 0) & (j < grid.shape[1])
        out = np.full(np.shape(lat), outside) & ok
        out[inside] = grid[i[inside], j[inside]]
        return out


def xyz(lat, lon) -> np.ndarray:
    """Points on the unit sphere, (n, 3)."""
    la, lo = np.radians(np.ravel(lat)), np.radians(np.ravel(lon))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def in_north_atlantic(lat, lon) -> np.ndarray:
    """Atlantic side of the Central American divide, 5-60N."""
    lon = wrap(lon)
    atl = (lat > 5) & (lat < 60) & (lon > -100) & (lon < 0)
    pacific = ((lon < -84) & (lat < 9)) | ((lon < -90) & (lat < 15)) | ((lon < -98) & (lat < 18))
    return atl & ~pacific


def region(lat, lon) -> np.ndarray:
    """Coastal regions of the landfall point."""
    lon = wrap(lon)
    out = np.full(np.shape(lat), 'other', dtype=object)
    out[(lon >= -82) & (lat >= 36.6)] = 'VA-ME'
    out[(lon >= -87.6) & (lat >= 31) & (lat < 36.6)] = 'GA-Car'
    out[(lon >= -87.6) & (lat < 31)] = 'FL'
    out[(lon < -87.6) & (lat < 33)] = 'TX-AL'
    out[lon < -105] = 'Pacific'
    return out


def unwrap(lon: np.ndarray) -> np.ndarray:
    """Longitudes along axis 0 (node, storm) made continuous across the date line and 0/360;
    each node follows the last finite one before it, across missing nodes."""
    filled = pd.DataFrame(lon).ffill().bfill().to_numpy()
    step = (np.diff(filled, axis=0) + 180) % 360 - 180
    out = filled.copy()
    out[1:] = filled[:1] + np.cumsum(np.nan_to_num(step), axis=0)
    return np.where(np.isfinite(lon), out, np.nan)


def hourly(x: np.ndarray, steps: int = 6) -> np.ndarray:
    """6-hourly nodes along axis 0 to `steps` steps a node (hourly by default), linearly; NaN
    next to a missing node."""
    n = x.shape[0]
    h = np.arange((n - 1) * steps + 1)
    i0 = np.minimum(h // steps, n - 2)
    f = ((h - steps * i0) / steps).reshape((-1,) + (1,) * (x.ndim - 1))
    out = x[i0] * (1 - f) + x[i0 + 1] * f
    out = np.where(f == 0, x[i0], out)
    return np.where(f == 1, x[i0 + 1], out)


CROSSING_STEPS = 6  # sub-steps an hour when timing a crossing: 10 minutes


def crossing_time(lat_h, lon_h, h, s, on_land) -> np.ndarray:
    """Hours from the first node at which each crossing (an hour h on land after hour h - 1 over
    water, storm s) reaches land, to 1/CROSSING_STEPS of an hour along the straight line between
    the two hourly positions. on_land(lat, lon) -> bool."""
    f = np.arange(1, CROSSING_STEPS + 1) / CROSSING_STEPS
    lat0, lat1 = lat_h[h - 1, s][:, None], lat_h[h, s][:, None]
    lon0 = lon_h[h - 1, s][:, None]
    dlon = (lon_h[h, s][:, None] - lon0 + 180) % 360 - 180
    land = on_land(lat0 + (lat1 - lat0) * f, wrap(lon0 + dlon * f))
    land[:, -1] = True
    return h - 1 + f[np.argmax(land, axis=1)]


def wind_at(wind: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(member, crossing) wind interpolated linearly between 6-hourly nodes to t hours from the
    first node of storm s; NaN next to a missing node."""
    i0 = np.floor(t / 6).astype(int)
    f = t / 6 - i0
    w0 = wind[:, i0, s]
    w1 = wind[:, np.minimum(i0 + 1, wind.shape[1] - 1), s]
    return np.where(f == 0, w0, w0 + (w1 - w0) * f)


def crossings(lat_h, lon_h, land: USLand) -> np.ndarray:
    """(hour, storm) flags: an hour on US land right after an hour over water."""
    on = land(lat_h, lon_h)
    off = land.water(lat_h, lon_h)
    out = np.zeros(on.shape, bool)
    out[1:] = on[1:] & off[:-1]
    return out


def approach(wind: np.ndarray, entries: np.ndarray, offshore: np.ndarray):
    """Approach wind of every member at each entry onto land: the hours and storms of the
    entries, and a (member, entry) array of the member's wind at the last even node before the
    entry hour, NaN where an hour from that node to the entry is not offshore or the member
    carries no wind at the entry hour. wind (member, node, storm) 6-hourly; entries and offshore
    (hour, storm) hourly."""
    h, s = np.nonzero(entries)
    node = 2 * ((h - 1) // APPROACH_H)
    hours = 6 * node[:, None] + np.arange(APPROACH_H)
    between = hours < h[:, None]
    clear = np.where(between, offshore[np.minimum(hours, offshore.shape[0] - 1), s[:, None]], True)
    out = wind[:, node, s].astype(float)
    alive = np.isfinite(wind[:, h // 6, s]) & np.isfinite(wind[:, -(-h // 6), s])
    out[:, ~clear.all(axis=1)] = np.nan
    out[~alive] = np.nan
    return h, s, out


def strongest(storm: np.ndarray, wind: np.ndarray, select: np.ndarray, n_s: int) -> np.ndarray:
    """(member, storm) index of each member's strongest selected entry, the first of equals, or
    the storm's first selected entry where none has a wind; -1 where a storm has none selected.
    storm and select (entry,), in time order within a storm; wind (member, entry)."""
    out = np.full((wind.shape[0], n_s), -1)
    for st in np.unique(storm[select]):
        k = np.nonzero(select & (storm == st))[0]
        v = wind[:, k]
        best = np.argmax(np.where(np.isfinite(v), v, -np.inf), axis=1)
        out[:, st] = k[np.where(np.isfinite(v).any(axis=1), best, 0)]
    return out


def near_coast_peak(lat, lon, wind, land: USLand) -> np.ndarray:
    """(member, storm) strongest wind at even nodes within NEAR_KM of US Atlantic and Gulf land
    (land.atlantic_km), NaN where none is that close."""
    near = land.atlantic_km(lat[::2], lon[::2]) <= NEAR_KM
    w = np.where(near[None], wind[:, ::2], np.nan)
    with np.errstate(all='ignore'):
        out = np.nanmax(np.where(np.isfinite(w), w, -np.inf), axis=1)
    return np.where(np.isfinite(out), out, np.nan)


def lee_records(lat, lon, wind, land: USLand) -> pd.DataFrame:
    """Every independent US Atlantic and Gulf landfall of each member by the rule of Lee et al.
    (2018, section 2.2): tracks and winds interpolated to 15 minutes, land from a LEE_RES mask
    (Natural Earth land by the cell's centre, standing in for their 0.5-degree topography), a
    landfall where the centre moves from ocean to land, landfalls on any land less than LEE_KM or
    LEE_H from the last one counted not counted again, and the wind the interpolated wind there.
    One row per (member, landfall) with wind at TS or more."""
    lat_q, lon_q = hourly(lat, LEE_STEPS), wrap(hourly(unwrap(lon), LEE_STEPS))
    on, us = land.lee(lat_q, lon_q)
    ocean = np.isfinite(lat_q) & ~on
    entry = np.zeros(on.shape, bool)
    entry[1:] = on[1:] & ocean[:-1]
    rows = []
    for s in np.nonzero(entry.any(axis=0))[0]:
        last = None
        for q in np.nonzero(entry[:, s])[0]:
            p = xyz(lat_q[q, s], lon_q[q, s])[0]
            if last is not None:
                km = 2 * EARTH_KM * np.arcsin(min(np.linalg.norm(p - last[1]) / 2, 1))
                if (q - last[0]) / LEE_STEPS * 6 < LEE_H or km < LEE_KM:
                    continue
            last = (q, p)
            r = str(region(lat_q[q, s], lon_q[q, s]))
            if not us[q, s] or r == 'Pacific':
                continue
            w = wind_at(wind, np.array([q * 6 / LEE_STEPS]), np.array([s]))[:, 0]
            with np.errstate(invalid='ignore'):
                keep = w >= TS
            for m in np.nonzero(keep)[0]:
                rows.append((m, s, q * 6 / LEE_STEPS, w[m], lat_q[q, s], lon_q[q, s], r))
    columns = ['member', 'storm', 'hour', 'wind', 'lat', 'lon', 'region']
    return pd.DataFrame(rows, columns=columns).astype(
        {'member': int, 'storm': int, 'hour': float, 'wind': float, 'lat': float, 'lon': float}
    )


def storm_rows(
    lat, lon, wind, year, land: USLand, source: str, unit: str, lee_out: list | None = None
) -> pd.DataFrame:
    """One row per (member, storm): North Atlantic TS flag, lifetime peak, whether the track
    crosses the US coast, how the member's wind ends, and US landfall wind, region, 24 h approach
    peak, decay and wind at AROUND hours from landfall. lat, lon (node, storm); wind (member,
    node, storm) in kt. With `lee_out`, the lee_records of these storms, with source, unit and
    year, are appended to it.

    How a CHAZ member ends (`end_rule`): its wind reaches the last node the intensity model can
    reach on its track ('track end'); the model stopped it after two 12-hourly steps at or below
    35 kt following a peak above 35 kt ('fell to 35 kt'); or neither: a step predicted below
    10 kt, or missing predictors ('other'). Observed storms are 'observed'."""
    lon = wrap(lon)
    n_m, n_n, n_s = wind.shape
    na = ((wind >= TS) & in_north_atlantic(lat, lon)[None] & np.isfinite(lat)[None]).any(axis=1)
    track = np.isfinite(lat)
    first = np.argmax(track, axis=0)
    last = n_n - 1 - np.argmax(track[::-1], axis=0)
    s_idx = np.arange(n_s)
    genesis_na = track.any(axis=0) & in_north_atlantic(lat[first, s_idx], lon[first, s_idx])
    wind_at_end = np.isfinite(wind[:, last, s_idx])
    alive = np.isfinite(wind)
    w_end = np.where(alive.any(axis=1), n_n - 1 - np.argmax(alive[:, ::-1], axis=1), -1)
    ends = end_rules(wind, w_end, reach(last, n_n)[None], observed=source == 'IBTrACS')
    m_idx = np.arange(n_m)[:, None]
    safe = np.maximum(w_end, 0)
    end_lat, end_lon = lat[safe, s_idx[None]], lon[safe, s_idx[None]]
    with np.errstate(all='ignore'):
        peak = np.nanmax(np.where(np.isfinite(wind), wind, -np.inf), axis=1)
    lat_h, lon_h = hourly(lat), wrap(hourly(unwrap(lon)))
    cross = crossings(lat_h, lon_h, land)
    h_e, s_e, a_e = approach(wind, cross, land.water(lat_h, lon_h))
    atlantic = region(lat_h[h_e, s_e], lon_h[h_e, s_e]) != 'Pacific'
    has_atlantic = np.zeros(n_s, bool)
    has_atlantic[s_e[atlantic]] = True
    pick = strongest(s_e, a_e, atlantic | ~has_atlantic[s_e], n_s)
    first = strongest(s_e, np.full_like(a_e, np.nan), atlantic | ~has_atlantic[s_e], n_s)
    lee = lee_records(lat, lon, wind, land)
    lee_counts = {t: np.zeros((n_m, n_s)) for t in (TS, HU, MH)}
    for t in lee_counts:
        hit = lee[lee['wind'] >= t]
        np.add.at(lee_counts[t], (hit['member'].values, hit['storm'].values), 1)
    if lee_out is not None:
        lee_out.append(
            lee.assign(source=source, unit=unit, year=np.asarray(year)[lee['storm'].values])
        )
    df = pd.DataFrame(
        {
            'source': source,
            'unit': unit,
            'member': np.repeat(np.arange(n_m), n_s),
            'storm': np.tile(np.arange(n_s), n_m),
            'year': np.tile(year, n_m),
            'na_ts': na.ravel(),
            'peak': peak.ravel(),
            'genesis_na': np.tile(genesis_na, n_m),
            'crosses': np.tile(cross.any(axis=0), n_m),
            'wind_at_end': wind_at_end.ravel(),
            'end_rule': ends.ravel(),
            'end_wind': np.where(w_end >= 0, wind[m_idx, safe, s_idx[None]], np.nan).ravel(),
            'end_before_track_h': (6 * (last[None] - w_end)).ravel(),
            'end_on_land': np.where(
                land.inside(end_lat, end_lon), land.anywhere(end_lat, end_lon), np.nan
            ).ravel(),
            'lf_crossings': np.tile(np.bincount(s_e, minlength=n_s), n_m),
            'near_peak': near_coast_peak(lat, lon, wind, land).ravel(),
            'lee_34': lee_counts[TS].ravel(),
            'lee_64': lee_counts[HU].ravel(),
            'lee_96': lee_counts[MH].ravel(),
        }
    )
    m_i, s_i = np.nonzero(first >= 0)
    h = h_e[first[m_i, s_i]]
    storms = np.nonzero(first[0] >= 0)[0]
    t_first = crossing_time(lat_h, lon_h, h_e[first[0, storms]], storms, land)
    w_first = wind_at(wind, t_first, storms)  # member, storm with a crossing
    e = pick[m_i, s_i]
    h_off = h_e[e]
    cols = np.unique(s_i)
    w_h = hourly(np.moveaxis(wind[:, :, cols], 0, 1))  # hour, member, storm
    c_i = np.searchsorted(cols, s_i)

    def hours_from_landfall(dh: int) -> np.ndarray:
        t = h + dh
        ok = (t >= 0) & (t < w_h.shape[0])
        out = np.full(t.shape, np.nan)
        out[ok] = w_h[t[ok], m_i[ok], c_i[ok]]
        return out

    with np.errstate(invalid='ignore'):
        pre = np.fmax.reduce([hours_from_landfall(dh) for dh in range(-24, 1)], axis=0)
    lf_lat, lf_lon = lat_h[h, s_i], lon_h[h, s_i]
    off_lat, off_lon = lat_h[h_off, s_i], lon_h[h_off, s_i]
    values = {
        'lf_wind': w_first[m_i, np.searchsorted(storms, s_i)],
        'lf_pre24': pre,
        'lf_post12': hours_from_landfall(12),
        'lf_post24': hours_from_landfall(24),
        'lf_lat': lf_lat,
        'lf_lon': lf_lon,
        **{f'lf_w{dh:+d}': hours_from_landfall(dh) for dh in AROUND},
        'end_h_after_lf': np.where(w_end[m_i, s_i] >= 0, 6 * w_end[m_i, s_i] - h, np.nan),
        'lf_offshore_wind': a_e[m_i, e],
        'lf_first_offshore_wind': a_e[m_i, first[m_i, s_i]],
        'lf_offshore_lat': off_lat,
        'lf_offshore_lon': off_lon,
    }
    idx = m_i * n_s + s_i
    for k, v in values.items():
        col = np.full(n_m * n_s, np.nan)
        col[idx] = v
        df[k] = col
    for name, crossing_wind, la, lo in (
        ('lf_region', values['lf_wind'], lf_lat, lf_lon),
        ('lf_offshore_region', values['lf_offshore_wind'], off_lat, off_lon),
    ):
        col = np.full(n_m * n_s, None, dtype=object)
        with np.errstate(invalid='ignore'):
            landed = crossing_wind >= TS
        col[idx[landed]] = region(la[landed], lo[landed])
        df[name] = pd.Series(col, index=df.index, dtype=object)
    return df[np.isfinite(df['peak']) & (df['peak'] > 0)]


def reach(last, n: int):
    """The last node CHAZ's intensity model can give a wind on a track whose last node is `last`,
    in arrays of `n` nodes (get_stochastic): it predicts 12 h ahead from even nodes up to
    min(last, n - 5) + 1, and only to nodes on the track."""
    it2 = np.minimum(last, n - 5)
    final_it = it2 + 1 - (it2 + 1) % 2
    return np.minimum(np.minimum(final_it + 2, last - last % 2), n - 1)


def end_rules(wind, w_end, reachable, observed: bool) -> np.ndarray:
    """(member, storm) labels of how each member's wind ends; see storm_rows. wind (member,
    node, storm); w_end, its last node with wind (-1 for none); reachable, from `reach`.

    get_stochastic stops after storing the prediction at w_end when the two values before it
    are at or below 35 kt and the peak before it is above 35 kt; the prediction itself does not
    count toward that peak."""
    out = np.full(w_end.shape, 'observed' if observed else 'other', dtype=object)
    if observed:
        return out
    m, s = np.indices(w_end.shape)
    before = np.arange(wind.shape[1])[None, :, None] <= (w_end - 2)[:, None, :]
    with np.errstate(invalid='ignore'):
        peak = np.max(np.where(before & np.isfinite(wind), wind, -np.inf), axis=1)
    w2 = np.where(w_end >= 2, wind[m, np.maximum(w_end - 2, 0), s], np.nan)
    w4 = np.where(w_end >= 4, wind[m, np.maximum(w_end - 4, 0), s], np.nan)
    out[(peak > 35) & (w2 <= 35) & (w4 <= 35)] = 'fell to 35 kt'
    out[(w_end >= 0) & (w_end >= reachable)] = 'track end'
    return out


def chaz_arrays(ds: xr.Dataset, years: tuple[int, int] = YEARS):
    """lat, lon (node, storm), wind (member, node, storm) and year of the storms of a CHAZ track
    file in `years`, with CHAZ's padding nodes as NaN."""
    year = ds['year'].values
    sel = np.nonzero((year >= years[0]) & (year <= years[1]))[0]
    lat = ds['latitude'].values[:, sel]
    lon = ds['longitude'].values[:, sel]
    wind = ds['Mwspd'].transpose('ensembleNum', 'lifelength', 'stormID').values[:, :, sel]
    fill = ds['time'].values[:, sel] <= TIME_FILL
    lat = np.where(fill, np.nan, lat)
    lon = np.where(fill, np.nan, lon)
    wind = np.where(fill[None] | ~np.isfinite(lat)[None], np.nan, wind)
    return lat, lon, wind, year[sel]


def chaz_rows(
    ds: xr.Dataset,
    land: USLand,
    source: str,
    unit: str,
    years: tuple[int, int] = YEARS,
    lee_out: list | None = None,
) -> pd.DataFrame:
    """storm_rows for the storms of a CHAZ track file in `years`."""
    return storm_rows(*chaz_arrays(ds, years), land, source, unit, lee_out)


def observed_records(csv: bytes, seasons: tuple[int, int] = YEARS) -> pd.DataFrame:
    """IBTrACS main-track synoptic records of `seasons` with `wind` the 1-minute wind: USA_WIND,
    or where it is missing the WMO agency's wind converted to 1-minute as CLIMADA does
    (ONE_MINUTE). USA_WIND stops where JTWC stops, at extratropical transition, while the
    regional agencies follow the storm on; their records count whatever the storm's nature, as
    HURDAT's do in the North Atlantic."""
    bt = pd.read_csv(
        io.BytesIO(csv),
        skiprows=[1],
        low_memory=False,
        usecols=[
            'SID', 'SEASON', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND',
            'WMO_AGENCY', 'TRACK_TYPE',
        ],
    )  # fmt: skip
    t = pd.to_datetime(bt['ISO_TIME'])
    keep = (
        (t.dt.minute == 0)
        & (t.dt.hour % 6 == 0)
        & bt['SEASON'].between(*seasons)
        & (bt['TRACK_TYPE'] == 'main')
    )
    bt = bt[keep].assign(time=t[keep])
    usa = pd.to_numeric(bt['USA_WIND'], errors='coerce')
    agency = bt['WMO_AGENCY'].astype(str).str.strip()
    scale = agency.map({k: v[0] for k, v in ONE_MINUTE.items()})
    shift = agency.map({k: v[1] for k, v in ONE_MINUTE.items()})
    converted = (pd.to_numeric(bt['WMO_WIND'], errors='coerce') - shift) / scale
    return bt.assign(
        wind=usa.fillna(converted),
        wind_source=np.where(usa.notna(), 'usa', np.where(converted.notna(), agency, '')),
    )


def observed_arrays(csv: bytes, seasons: tuple[int, int] = YEARS):
    """chaz_arrays for the IBTrACS since-1980 CSV: the tracks of observed_records, by season,
    with one 'member' of their wind. Each track is laid on a regular 6-hourly grid from its first
    record to its last, so a missing record is a NaN node, not a shorter step."""
    bt = observed_records(csv, seasons)
    groups = []
    for _, g in bt.groupby('SID', sort=True):
        g = g.drop_duplicates('time').set_index('time').sort_index()
        grid = pd.date_range(g.index[0], g.index[-1], freq='6h')
        g = g.reindex(grid)
        g['SEASON'] = g['SEASON'].ffill()
        groups.append(g)
    n, life = len(groups), max(len(g) for g in groups)
    lat = np.full((life, n), np.nan)
    lon = np.full((life, n), np.nan)
    wind = np.full((1, life, n), np.nan)
    season = np.zeros(n, int)
    for k, g in enumerate(groups):
        lat[: len(g), k], lon[: len(g), k], wind[0, : len(g), k] = g['LAT'], g['LON'], g['wind']
        season[k] = g['SEASON'].iloc[0]
    return lat, lon, wind, season


def landfall_records(csv: bytes, land: USLand, seasons: tuple[int, int] = YEARS) -> pd.DataFrame:
    """HURDAT's landfall records (USA_RECORD 'L') on main tracks within 25 km of US Atlantic and
    Gulf land, with the index of their storm in observed_arrays: the best-track wind at each
    landfall, for checking our rule's observed landfalls."""
    bt = pd.read_csv(
        io.BytesIO(csv),
        skiprows=[1],
        low_memory=False,
        usecols=['SID', 'SEASON', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'USA_RECORD',
                 'TRACK_TYPE'],
    )  # fmt: skip
    bt = bt[
        (bt['USA_RECORD'].astype(str).str.strip() == 'L')
        & (bt['TRACK_TYPE'] == 'main')
        & bt['SEASON'].between(*seasons)
    ]
    lat, lon = bt['LAT'].astype(float).values, bt['LON'].astype(float).values
    near = (land.atlantic_km(lat, lon) <= 25) & (region(lat, lon) != 'Pacific')
    bt = bt[near].assign(
        wind=pd.to_numeric(bt.loc[near, 'USA_WIND'], errors='coerce'),
        region=region(lat[near], lon[near]),
    )
    sids = np.array(sorted(observed_records(csv, seasons)['SID'].unique()))
    return bt.assign(storm=np.searchsorted(sids, bt['SID'].values))[
        ['storm', 'SID', 'SEASON', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'wind', 'region']
    ].reset_index(drop=True)


def observed(land: USLand, csv: bytes, lee_out: list | None = None) -> pd.DataFrame:
    """storm_rows for IBTrACS (observed_arrays)."""
    return storm_rows(*observed_arrays(csv), land, 'IBTrACS', 'ibtracs', lee_out)


def ibtracs_csv(fs) -> bytes:
    """The latest IBTrACS since-1980 CSV kept under inputs/."""
    key = sorted(
        fs.glob(f'{store.S3_ROOT}/inputs/ibtracs-v04r01/*/ibtracs.since1980.list.v04r01.csv')
    )[-1]
    print('IBTrACS', key.rsplit('/', 2)[-2], flush=True)
    return fs.cat(key)


def atlantic_crossing(df: pd.DataFrame) -> pd.Series:
    """Rows whose track crosses the Atlantic or Gulf coast, whatever the wind there; lf_lat and
    lf_lon are then the first such crossing."""
    pacific = region(df['lf_lat'].values, df['lf_lon'].values) == 'Pacific'
    return df['crosses'] & ~pacific


def summarize(df: pd.DataFrame, member_years: float, scale: float) -> dict:
    pacific = df['crosses'] & ~atlantic_crossing(df)
    df = df.assign(crosses=df['crosses'] & ~pacific)
    pac = df[pacific & (df['lf_wind'] >= TS)]
    df.loc[pacific, 'lf_region'] = None
    na = df[df['na_ts']]
    lf = df[df['lf_region'].notna()]
    big = lf[lf['lf_wind'] >= HU]
    crossing = na[na['crosses']]
    tracks = df.drop_duplicates(['unit', 'storm'])
    tracks = tracks[tracks['genesis_na']]
    out = {
        'NA TS storms/yr': len(na) / member_years * scale,
        'tracks from NA genesis crossing the US coast, share': tracks['crosses'].mean(),
        'NA TS storms crossing the US coast, share': na['crosses'].mean(),
        'crossings at >= 34 kt, share': (crossing['lf_wind'] >= TS).mean(),
        'crossings at >= 64 kt, share': (crossing['lf_wind'] >= HU).mean(),
        'NA TS storms with wind at their last track node, share': na['wind_at_end'].mean(),
        'NA share peak >= 64 kt': (na['peak'] >= HU).mean(),
        'NA share peak >= 96 kt': (na['peak'] >= MH).mean(),
        'US landfalls/yr (>= 34 kt)': len(lf) / member_years * scale,
        'US landfalls/yr >= 64 kt': len(big) / member_years * scale,
        'US landfalls/yr >= 96 kt': (lf['lf_wind'] >= MH).sum() / member_years * scale,
        'landfall share of NA TS storms': len(lf[lf['na_ts']]) / max(len(na), 1),
        'share of landfalls >= 64 kt': (lf['lf_wind'] >= HU).mean(),
        'share of landfalls >= 96 kt': (lf['lf_wind'] >= MH).mean(),
        'landfall wind median (kt)': lf['lf_wind'].median(),
        'peak of landfallers median (kt)': lf['peak'].median(),
        'landfall / 24 h approach peak, median': (lf['lf_wind'] / lf['lf_pre24']).median(),
        '12 h after / at landfall (>= 64 kt), median': (big['lf_post12'] / big['lf_w+0']).median(),
    }
    for r in ATLANTIC:
        sub = lf[lf['lf_region'] == r]
        out[f'{r} landfalls/yr >= 64 kt'] = (sub['lf_wind'] >= HU).sum() / member_years * scale
    out['US Pacific landfalls/yr (>= 34 kt)'] = len(pac) / member_years * scale
    out['US Pacific landfalls/yr >= 64 kt'] = (pac['lf_wind'] >= HU).sum() / member_years * scale
    return out


def compare(obs: pd.DataFrame, runs: dict[str, tuple[pd.DataFrame, int]]) -> pd.DataFrame:
    """Observed against each CHAZ set, raw and scaled to the observed NA storm count.
    runs: name -> (rows, member count)."""
    years = YEARS[1] - YEARS[0] + 1
    o = summarize(obs, years, 1.0)
    cols = {'IBTrACS': o}
    for name, (rows, members) in runs.items():
        raw = summarize(rows, members * years, 1.0)
        k = o['NA TS storms/yr'] / raw['NA TS storms/yr']
        cols[f'{name} (raw)'] = raw
        cols[f'{name} (x{k:.2f})'] = summarize(rows, members * years, k)
        cols[f'{name} / IBTrACS'] = {
            key: v / o[key] if o[key] else np.nan for key, v in cols[f'{name} (x{k:.2f})'].items()
        }
    return pd.DataFrame(cols)


def run(run_name: str = 'TCGI_CRH_PI') -> pd.DataFrame:
    fs = store.s3()
    land = USLand()
    lee = {'ibtracs': [], 'published': [], f'ours_{run_name}': []}
    obs = observed(land, ibtracs_csv(fs), lee['ibtracs'])

    pub = []
    for i in range(10):
        with fs.open(f'{PUBLISHED}/global_2019_2ens{i:03d}_pre.nc', 'rb') as f:
            ds = xr.open_dataset(f, engine='scipy', decode_times=False).load()
        pub.append(chaz_rows(ds, land, 'published', f'ens{i:03d}', lee_out=lee['published']))
        print('published file', i, flush=True)
    pub = pd.concat(pub, ignore_index=True)

    ours = []
    for r in range(10):
        for year in range(YEARS[0], YEARS[1] + 1):
            key = f'{store.RUNS}/{run_name}/{year}_r{r:03d}.nc'
            with xr.open_dataset(store.path(key), decode_times=False) as ds:
                part = []
                rows = chaz_rows(ds.load(), land, 'ours', f'r{r:03d}', lee_out=part)
            rows['storm'] = rows['storm'] + year * 10_000
            ours.append(rows)
            lee[f'ours_{run_name}'] += [x.assign(storm=x['storm'] + year * 10_000) for x in part]
        print('our realization', r, flush=True)
    ours = pd.concat(ours, ignore_index=True)

    members = {
        name: int(rows.groupby('unit')['member'].nunique().sum())
        for name, rows in (('published', pub), ('ours', ours))
    }
    tab = compare(obs, {'published': (pub, members['published']), 'ours': (ours, members['ours'])})
    local = store.CACHE / OUT
    local.mkdir(parents=True, exist_ok=True)
    written = [f'summary_{run_name}.csv']
    tab.to_csv(local / written[0])
    for name, rows in (('ibtracs', obs), ('published', pub), (f'ours_{run_name}', ours)):
        written.append(f'rows_{name}.parquet')
        rows.to_parquet(local / written[-1])
        written.append(f'lee_{name}.parquet')
        pd.concat(lee[name], ignore_index=True).to_parquet(local / written[-1])
    store.push(
        fragment=f'landfall-{run_name}', replace=True, keys=[f'{OUT}/{name}' for name in written]
    )
    return tab


def rows_key(source: str, run_name: str, years: tuple[int, int]) -> str:
    return f'{OUT}/rows_{source.replace("/", "-")}_{run_name}_{years[0]}-{years[1]}'


def source_rows(
    source: str, run_name: str, years: tuple[int, int], land: USLand, realizations: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """storm_rows for the run runs/{source}/{run_name} over `years`, and one row per track file:
    storms kept, global storms reaching 34 kt per member, members dropped above 300 kt, the
    seeding ratio and the normalization the intensity model read."""
    rows, files = [], []
    for r in range(realizations):
        for year in range(years[0], years[1] + 1):
            key = f'runs/{source}/{run_name}/{year}_r{r:03d}.nc'
            with xr.open_dataset(store.path(key), decode_times=False) as ds:
                ds = ds.load()
            part = chaz_rows(ds, land, run_name, f'r{r:03d}', years)
            part['storm'] = part['storm'] + year * 10_000
            rows.append(part)
            n, m = ds.sizes['stormID'], ds.sizes['ensembleNum']
            files.append(
                {
                    'year': year,
                    'realization': r,
                    'storms': n,
                    'members': n * m,
                    'ts_storms_per_member': float((ds['Mwspd'].max('lifelength') >= TS).sum()) / m,
                    'dropped_over_300kt': float(
                        ds.attrs.get('members_dropped_over_300kt', np.nan)
                    ),  # runs made before chaz.py recorded it have none
                    'seeding_ratio': float(ds.attrs.get('seeding_ratio', np.nan)),
                    'normalization': ds.attrs.get('predictor_normalization', ''),
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(files)


def save_rows(source: str, run_name: str, years: tuple[int, int]) -> None:
    rows, files = source_rows(source, run_name, years, USLand())
    key = rows_key(source, run_name, years)
    local = store.CACHE / key
    local.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(f'{local}.parquet')
    files.to_csv(f'{local}_files.csv', index=False)
    store.push(
        fragment=f'landfall-rows-{source.replace("/", "-")}-{run_name}',
        replace=True,
        keys=[f'{key}.parquet', f'{key}_files.csv'],
    )
    print(key, len(rows), 'rows', flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--run', default='TCGI_CRH_PI', help='run under runs/era5 (chaz.run_name)')
    w = sub.add_parser('rows')
    w.add_argument('--source', default='era5', help="'era5' or 'cmip6/{model}/{member}/{exp}'")
    w.add_argument('--run', default='TCGI_CRH_PI', help='chaz.run_name')
    w.add_argument('--years', required=True, help='first-last, e.g. 1995-2014')
    a = p.parse_args()
    if a.cmd == 'rows':
        y0, y1 = (int(x) for x in a.years.split('-'))
        save_rows(a.source, a.run, (y0, y1))
        return
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 20)
    print(run(a.run).round(3).to_string())


if __name__ == '__main__':
    main()
