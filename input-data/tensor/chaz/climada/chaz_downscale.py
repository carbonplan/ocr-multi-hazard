"""Run the CHAZ model (genesis, tracks, intensity) for one forcing year, locally.

A port of the downscaling tutorial notebook in github.com/cl3225/CHAZ
(Turtorial_chaz_downscaling_colab.ipynb, MIT), reading the CC BY 4.0 inputs of
its Zenodo record (doi:10.5281/zenodo.11631930): ERA5 forcing for 2017, the
intensity-model regressions, and the best-track predictors. The model code is
the notebook's, with three kinds of change:

  - numpy 2 spellings (np.float64, np.nan) and local paths for pooch
  - dask map_blocks with the synchronous scheduler became plain loops, which
    is what that scheduler ran
  - each track realization draws its own genesis seeds, as CHAZvBeta/CHAZ.py
    does; the notebook seeds once and reuses them. Each realization gets a
    fixed random seed, so runs are repeatable.

The seeding ratio defaults to the tutorial's 1. The model's own ratio rescales
TCGI seeds to the observed 1981-2005 storm count over a survival rate of 0.78,
and needs TCGI for those years, which the Zenodo record doesn't hold. Storms
that run past 31 December stop there, as in the notebook.

  run      --year 2017 --realizations 1-40 --workers 12
           out/{year}_r{NNN}.nc in the layout of the published track files,
           plus timing.csv
  compare  --year 2017
           the realizations against the year in the published ERA5 track
           files (chaz_events cache): counts, basins, lifetime peak winds

Inputs are read from ~/.cache/ocr/chaz-local/zenodo (fetch_chaz_local.sh
downloads and checks them). Run in the `climada` pixi environment.
"""

from __future__ import annotations

import argparse
import calendar
import copy
import gc
import os
import random
import sys
import time
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
from netCDF4 import date2num
from numpy import nanmean
from scipy import stats
from scipy.io import netcdf_file

ROOT = Path(os.environ.get('CHAZ_LOCAL_DIR', Path.home() / '.cache' / 'ocr' / 'chaz-local'))
INPUTS = ROOT / 'zenodo'
OUT = ROOT / 'out'
PUBLISHED_TRACKS = Path.home() / '.cache' / 'ocr' / 'chaz-events' / 'tracks' / 'ERA5'


def get_landmask():
    f = netcdf_file(INPUTS / 'landmask.nc', mmap=False)
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    landmask = f.variables['landmask'][:, :]
    f.close()
    return lon, lat, landmask


class gv:
    Model = 'ERA5'
    ENS = 'r1i1p1f1'
    TCGIinput = 'TCGI_CRH_PI'  # or 'TCGI_SD_PI'
    CHAZ_Int_ENS = 40
    uBeta = -1.5
    vBeta = 2.0
    survivalrate = 0.78
    obpath = 'bt_global_predictors.nc'
    llon = llat = lldmask = ldmask = None


def _load_landmask():
    if gv.ldmask is None:
        gv.llon, gv.llat, lldmask = get_landmask()
        gv.ldmask = lldmask[-12::-24, ::24]
        gv.lldmask = lldmask[::-1, :]


yearTC = None

# ---------------------------------------------------------------------------
# genesis


def TCgiSeeding(gxlon, gxlat, gi, climInitLon, climInitLat, climInitDate, ratio, iy):
    for im in range(12):
        xk, pk = np.arange(gi[im, :, :].size), gi[im, :, :].ravel()
        dummy = xk * pk
        xk, pk = xk[dummy == dummy], pk[dummy == dummy]
        custm = stats.rv_discrete(name='custm', values=(xk, pk / pk.sum()))
        r = custm.rvs(size=np.int_(np.rint(pk.sum() * ratio)))
        iix = gxlon.ravel()[r.ravel()]
        iiy = gxlat.ravel()[r.ravel()]
        iday = np.random.choice(
            np.arange(calendar.monthrange(iy, im + 1)[1]),
            size=np.int_(np.rint(pk.sum() * ratio)),
            replace=True,
        )
        if iday.size > 0:
            for id in range(iday.size):
                x1 = np.arange(iix[id] - 1, iix[id] + 1.01, 0.01)
                y1 = np.arange(iiy[id] - 1, iiy[id] + 1.01, 0.01)
                xx = np.random.choice(x1, 1)
                yy = np.random.choice(y1, 1)
                climInitDate.append(datetime(iy, im + 1, iday[id] + 1, 0, 0))
                climInitLon.append(xx)
                climInitLat.append(yy)
    return climInitLon, climInitLat, climInitDate


def getSeeding(iy, ratio):
    climInitLon, climInitLat, climInitDate = [], [], []
    ds = xr.open_dataset(INPUTS / f'{gv.TCGIinput}_{iy:04d}.nc')
    gi = ds.TCGI.values
    gxlon, gxlat = np.meshgrid(ds['lon'], ds['lat'])
    climInitLon, climInitLat, climInitDate = TCgiSeeding(
        gxlon, gxlat, gi, climInitLon, climInitLat, climInitDate, ratio, iy
    )
    return (
        np.array(climInitDate).ravel(),
        np.array(climInitLon).ravel(),
        np.array(climInitLat).ravel(),
    )


# ---------------------------------------------------------------------------
# tracks: beta-advection model


def func_first(x):
    return None if x.first_valid_index() is None else x.first_valid_index()


def func_last(x):
    return None if x.last_valid_index() is None else x.last_valid_index()


def removeland(iS):
    b = np.int_(iS.mean(keepdims=True))
    iT3 = -1
    if b < fstldmask.shape[1]:
        a = np.argwhere(fstldmask[:, b] == 3).ravel()
        if a.size:
            if a.size > 3:
                iT3 = a[0] + 2
                fstlon[iT3:, b] = np.nan
                fstlat[iT3:, b] = np.nan
    return iT3


def calF(nday):
    """Fourier series with random phases (Emanuel et al. 2006), hourly."""
    dt = 1.0 * 60 * 60
    T = np.float64(15)
    N = 15
    nt = np.arange(0, nday * 60 * 60 * 24, dt)
    F = np.zeros([nt.shape[0], 4])
    for iff in range(0, 4, 1):
        X = [random.uniform(0, 1) for iN in range(N)]
        for itt in range(nt.shape[0]):
            F[itt, iff] = np.sqrt(
                2.0 / np.sum([iN ** (-3.0) for iN in range(1, N + 1, 1)])
            ) * np.sum(
                [
                    (iN ** (-3.0 / 2.0)) * np.sin(2.0 * np.pi * (iN * itt / (24.0 * T) + X[iN - 1]))
                    for iN in range(1, N + 1, 1)
                ]
            )
    return F


def fillinNaN(var, neighbors):
    for ii in range(var.shape[0]):
        a = var[ii, :, :]
        while np.any(a.mask):
            a_copy = a.copy()
            for hor_shift, vert_shift in neighbors:
                if not np.any(a.mask):
                    break
                a_shifted = np.roll(
                    np.roll(a_copy, shift=hor_shift, axis=1), shift=vert_shift, axis=0
                )
                idx = a.mask & ~a_shifted.mask
                a[idx] = a_shifted[idx]
        var[ii, :, :] = a
    return var


def date_interpolation(dateInput, fileInput):
    """Monthly fields interpolated linearly to the date; December holds."""
    if dateInput.day >= 15:
        if dateInput.month < 12:
            date0 = datetime(dateInput.year, dateInput.month, 15, 0, 0)
            date1 = datetime(dateInput.year, dateInput.month + 1, 15, 0, 0)
            dfdays = (dateInput - date0).days
            xp = [0, (date1 - date0).days]
            ratio = np.interp(dfdays, xp, [0, 1])
            fileOutput = (
                fileInput[dateInput.month - 1, :, :] * (1 - ratio)
                + fileInput[dateInput.month, :, :] * ratio
            )
        else:
            fileOutput = fileInput[dateInput.month - 1, :, :]
    else:
        if dateInput.month == 1:
            fileOutput = fileInput[dateInput.month - 1, :, :]
        else:
            date0 = datetime(dateInput.year, dateInput.month - 1, 15, 0, 0)
            date1 = datetime(dateInput.year, dateInput.month, 15, 0, 0)
            dfdays = (dateInput - date0).days
            xp = [0, (date1 - date0).days]
            ratio = np.interp(dfdays, xp, [0, 1])
            fileOutput = (
                fileInput[dateInput.month - 2, :, :] * (1 - ratio)
                + fileInput[dateInput.month - 1, :, :] * ratio
            )
    return fileOutput


def getTrackPrediction(u250, v250, u850, v850, dt, fstLon, fstLat, fstDate):
    earth_rate = 7.2921150e-5
    r0 = 6371000
    lat0 = np.arange(-90, 100, 10)
    phi0 = lat0 / 180.0 * np.pi
    beta0 = 2.0 * earth_rate * np.cos(phi0) / r0
    beta0 = beta0 / beta0[10]
    ratio = np.interp(fstLat, np.arange(-90, 100, 10), beta0)
    uBeta = gv.uBeta * ratio
    vBeta = gv.vBeta * ratio
    alpha = 0.8
    uTrack = alpha * u850 + (1.0 - alpha) * u250 + uBeta
    vTrack = alpha * v850 + (1.0 - alpha) * v250 + vBeta * np.sign(np.sin(fstLat * np.pi / 180.0))
    dx = uTrack * dt
    dy = vTrack * dt
    lon2, lat2 = getLonLatfromDistance(fstLon, fstLat, dx, dy)
    fstDate += timedelta(seconds=dt)
    return lon2, lat2, fstDate


def getLonLatfromDistance(lonInit, latInit, dx, dy):
    er = 6371000
    londis = 2 * np.pi * er * np.cos(latInit / 180 * np.pi) / 360.0
    lon2 = lonInit + dx / londis
    latdis = 2 * np.pi * er / 360.0
    lat2 = latInit + dy / latdis
    return lon2, lat2


def bam(iS):
    neighbors = (
        (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1),
        (1, -1), (-1, -1), (0, 2), (0, -2), (2, 0), (-2, 0),
    )  # fmt: skip
    missing_value = 1e20
    dt = 1.0 * 60 * 60
    T = np.float64(15)
    F = calF(15)
    nt = np.arange(0, T * 60 * 60 * 24, dt)
    b = np.int_(iS.mean(keepdims=True)[0])
    fstDate = climInitDate[b]
    fstLon = climInitLon[b]
    fstLat = climInitLat[b]
    fstlon[0, b] = fstLon
    fstlat[0, b] = fstLat
    fstldmask[0, b] = 0
    endhours = fstlon.shape[0] - 1
    endDate = climInitDate[b] + timedelta(hours=endhours)
    count, year0, month0, day0 = 1, 0, 0, 0

    dsA = xr.open_dataset(INPUTS / f'A_{fstDate.year:04d}{fstDate.month:02d}.nc')
    A_matrix = dsA.variables['A'].load()
    dsA.close()
    month0 = fstDate.month

    while fstDate < endDate and fstDate.year == yearTC:
        if fstDate.year != year0:
            fileName = INPUTS / f'I_MLR_{fstDate.year:04d}.nc'
            try:
                ds = xr.open_dataset(fileName)
                u250m = ds['u250'].values
                u850m = ds['u850'].values
                v250m = ds['v250'].values
                v850m = ds['v850'].values
                for data in [u250m, u850m, v250m, v850m]:
                    data[np.isnan(data)] = missing_value
                    data = np.ma.masked_values(data, missing_value)
                    data = fillinNaN(data, neighbors)
                xlong = ds['lon'][:]
                xlat = ds['lat'][:]
                xxlong, xxlat = np.meshgrid(xlong, xlat)
                ds.close()
                year0 = fstDate.year
            except Exception:
                print(f'no {fileName}')
                break
        if fstDate.day != day0:
            u250m2d = date_interpolation(fstDate, u250m)
            v250m2d = date_interpolation(fstDate, v250m)
            u850m2d = date_interpolation(fstDate, u850m)
            v850m2d = date_interpolation(fstDate, v850m)
            day0 = fstDate.day

        if fstDate.month != month0:
            dsA = xr.open_dataset(INPUTS / f'A_{fstDate.year:04d}{fstDate.month:02d}.nc')
            A_matrix = dsA.variables['A'].load()
            dsA.close()
            month0 = fstDate.month
        A = A_matrix[:, fstDate.day - 1, :, :]
        day0 = fstDate.day

        distance = np.sqrt((fstLon - xxlong) ** 2 + (fstLat - xxlat) ** 2)
        iy, ix = np.unravel_index(np.argmin(distance), distance.shape)
        iy1, ix1 = np.max([iy - 2, 0]), np.max([ix - 2, 0])
        iy2, ix2 = np.min([iy + 2, distance.shape[0]]), np.min([ix + 2, distance.shape[1]])

        iit = np.mod(count, nt.shape[0])
        box = (slice(iy1, iy2 + 1), slice(ix1, ix2 + 1))
        u250 = u250m2d[box] + A[0][box] * F[iit, 0]
        v250 = v250m2d[box] + A[1][box] * F[iit, 0] + A[2][box] * F[iit, 1]
        u850 = u850m2d[box] + A[3][box] * F[iit, 0] + A[4][box] * F[iit, 1] + A[5][box] * F[iit, 2]
        v850 = (
            v850m2d[box]
            + A[6][box] * F[iit, 0]
            + A[7][box] * F[iit, 1]
            + A[8][box] * F[iit, 2]
            + A[9][box] * F[iit, 3]
        )
        u250 = np.nanmean(u250)
        u850 = np.nanmean(u850)
        v250 = np.nanmean(v250)
        v850 = np.nanmean(v850)

        fstLon, fstLat, fstDate = getTrackPrediction(
            u250, v250, u850, v850, dt, fstLon, fstLat, fstDate
        )
        if (fstLon < 0.0) or (fstLon > 360) or (fstLat < -60) or (fstLat > 60):
            break
        fstlon[count, b] = fstLon
        fstlat[count, b] = fstLat
        fstldmask[count, b] = np.rint(np.nanmean(gv.ldmask[iy1 : iy2 + 1, ix1 : ix2 + 1]))
        count += 1
    return b


def getBam(cDate, cLon, cLat):
    global climInitDate, climInitLon, climInitLat, fstlon, fstlat, fstldmask
    climInitDate, climInitLon, climInitLat = cDate, cLon, cLat
    nnt = np.int_(31)  # longest track, days
    nS = climInitLon.shape[0]
    fstlon = np.zeros([nnt * 24 + 1, nS]) * np.nan
    fstlat = np.zeros([nnt * 24 + 1, nS]) * np.nan
    fstldmask = np.zeros(fstlat.shape)
    niS = np.arange(0, nS, 1).astype(dtype=np.int32)
    for iS in niS:
        bam(np.array([iS], dtype=np.int32))
    fstlon = fstlon[::6, :]
    fstlat = fstlat[::6, :]
    fstldmask = fstldmask[::6, :]
    for iS in niS:
        removeland(np.array([iS], dtype=np.int32))
    fsttime = np.empty(fstlon.shape, dtype=object)
    fsttime[0, :] = climInitDate
    dummy = pd.DataFrame(fstlon)
    iT1 = np.int16(dummy.apply(func_first, axis=0)) + 1
    iT2 = np.int16(dummy.apply(func_last, axis=0))
    for iS in niS:
        fsttime[iT1[iS] : iT2[iS] + 1, iS] = [
            climInitDate[iS] + timedelta(hours=6 * iit) for iit in range(iT1[iS], iT2[iS] + 1, 1)
        ]
    return {'lon': fstlon, 'lat': fstlat, 'Time': fsttime, 'ldmask': fstldmask}


# ---------------------------------------------------------------------------
# predictors along the tracks


class fst2bt:
    """Track dictionary to the best-track layout the intensity model reads."""

    def __init__(self, data):
        nan = np.float64('nan')
        self.StormId = np.arange(0, data['lon'].shape[1], 1)
        self.StormYear = []
        self.StormInitMonth = []
        for iS in self.StormId:
            if data['Time'][0, iS] is not None:
                self.StormYear.append(data['Time'][0, iS].year)
                self.StormInitMonth.append(data['Time'][0, iS].month)
            else:
                self.StormYear.append(1800)
                self.StormInitMonth.append(1)
        self.StormYear = np.array(self.StormYear)
        self.StormInitMonth = np.array(self.StormInitMonth)
        for iS in range(data['Time'].shape[1]):
            data['Time'][:, iS] = np.array(
                [datetime(1800, 1, 1, 0) if v is None else v for v in data['Time'][:, iS]]
            )
        data['Time'][data['lon'] != data['lon']] = datetime(1800, 1, 1, 0)
        data['lon'][data['Time'] == datetime(1800, 1, 1, 0)] = nan
        self.StormLon = data['lon']
        self.StormLat = data['lat']
        self.Time = data['Time']
        shape = data['lon'].shape
        for name in (
            'PIwspd', 'PIslp', 'PIwspdMean', 'dPIwspd', 'PIslpMean', 'UShearMean', 'UShear',
            'VShearMean', 'VShear', 'div200Mean', 'div200', 'T200Mean', 'T200',
            'rh500_300Mean', 'rh500_300', 'rhMean', 'rh', 'T100Mean', 'T100',
            'dThetaEMean', 'dThetaE', 'dThetaEs', 'dThetaEsMean',
        ):  # fmt: skip
            setattr(self, name, np.empty(shape, dtype=float) * nan)
        self.landmask = data['ldmask']
        self.landmaskMean = data['ldmask']
        self.trSpeed = np.zeros(shape) * nan
        self.trDir = np.zeros(shape) * nan
        self.dVdt = np.zeros(shape)


def getSpeedDir(iiS):
    iS = np.int_(iiS.mean(keepdims=True)[0])
    if (bt.StormLat[:, iS] == bt.StormLat[:, iS]).any():
        it1 = np.argwhere(bt.StormLat[:, iS] == bt.StormLat[:, iS])[0, 0]
        it2 = np.argwhere(bt.StormLat[:, iS] == bt.StormLat[:, iS])[-1, 0]
        if it2 - it1 >= 2:
            bt.trDir[it1:it2, iS], bt.trSpeed[it1:it2, iS] = getStormTranslation(
                bt.StormLon[it1:it2, iS], bt.StormLat[it1:it2, iS], bt.Time[it1:it2, iS]
            )
    return iS


def getStormTranslation(lon, lat, time):
    er = 6371.0
    timeInt, lonInt, latInt = [], [], []
    for iN in range(time.shape[0] - 1):
        timeInt.append(time[iN])
        lonInt.append(lon[iN])
        latInt.append(lat[iN])
        delt = 6
        inv = 1.0 / np.float64(delt)
        for iM in range(1, delt, 1):
            timeInt.append(time[iN] + timedelta(hours=iM))
            lonInt.append((1.0 - iM * inv) * lon[iN] + iM * inv * lon[iN + 1])
            latInt.append((1.0 - iM * inv) * lat[iN] + iM * inv * lat[iN + 1])
    speed = np.zeros(lon.shape[0], dtype=float) + float('nan')
    sdir = np.zeros(lon.shape[0], dtype=float) + float('nan')
    count = 0
    for it in time:
        nup = argminDatetime(it + timedelta(hours=3), timeInt)
        ndn = argminDatetime(it - timedelta(hours=3), timeInt)
        londis = 2 * np.pi * er * np.cos(latInt[nup] / 180 * np.pi) / 360
        dlon = lonInt[nup] - lonInt[ndn]
        dlat = latInt[nup] - latInt[ndn]
        dx = londis * (lonInt[nup] - lonInt[ndn])
        dy = 110 * (latInt[nup] - latInt[ndn])
        distance = np.sqrt(dx * dx + dy * dy)
        sdir[count] = np.arctan2(dlat, dlon)
        speed[count] = distance * 1000.0 / (nup - ndn + 1) / 60 / 60
        count += 1
    return sdir, speed


def argminDatetime(time0, time):
    time = np.array(time)
    n0 = 0
    delt = np.abs(time[-1] - time[0])
    for n in range(time.size):
        if np.abs(time0 - time[n]) < delt:
            delt = np.abs(time0 - time[n])
            n0 = n
    return n0


def get_predictors(iiS):
    iS = np.int_(iiS.mean(keepdims=True)[0])
    if bt.StormYear[iS] == yearTC:
        ds = xr.open_dataset(INPUTS / f'I_MLR_{bt.StormYear[iS]:04d}.nc')
        xlong = ds['lon'].values
        xlat = ds['lat'].values
        PIVmax = ds['PI'].values * 1.94384449  # m/s -> kt
        u = ds['u250'].values - ds['u850'].values
        v = ds['v250'].values - ds['v850'].values
        meanrh = ds['hur'].values
        xxlong, xxlat = np.meshgrid(xlong, xlat)
        ds.close()
        for it in range(0, bt.StormLon[:, iS].shape[0], 1):
            if bt.Time[it, iS] != datetime(1800, 1, 1, 0):
                er = 6371.0
                londis = 2 * np.pi * er * np.cos(xxlat / 180 * np.pi) / 360
                dx = londis * (xxlong - bt.StormLon[it, iS])
                dy = 110 * (xxlat - bt.StormLat[it, iS])
                distance = np.sqrt(dx * dx + dy * dy)
                (j0, i0) = np.unravel_index(np.argmin(distance), distance.shape)

                var, radius1, radius2 = date_interpolation(bt.Time[it, iS], PIVmax), 0, 500
                ring = (distance <= radius2) & (distance >= radius1) & (var == var)
                bt.PIwspdMean[it, iS] = np.mean(var[ring])
                bt.PIwspd[it, iS] = var[j0, i0]

                var, radius1, radius2 = date_interpolation(bt.Time[it, iS], u), 200, 800
                ring = (distance <= radius2) & (distance >= radius1) & (var == var)
                bt.UShearMean[it, iS] = np.mean(var[ring])
                bt.UShear[it, iS] = var[j0, i0]

                var, radius1, radius2 = date_interpolation(bt.Time[it, iS], v), 200, 800
                ring = (distance <= radius2) & (distance >= radius1) & (var == var)
                bt.VShearMean[it, iS] = np.mean(var[ring])
                bt.VShear[it, iS] = var[j0, i0]

                var, radius1, radius2 = date_interpolation(bt.Time[it, iS], meanrh), 200, 800
                ring = (distance <= radius2) & (distance >= radius1) & (var == var)
                bt.rhMean[it, iS] = np.mean(var[ring])
                bt.rh[it, iS] = var[j0, i0]

                var, radius1, radius2 = copy.copy(gv.ldmask), 0, 300
                var[var == 0] = -1.0
                var[var == 3] = 0.0
                ring = (distance <= radius2) & (distance >= radius1) & (var == var)
                bt.landmaskMean[it, iS] = np.mean(var[ring])
                bt.landmask[it, iS] = var[j0, i0]
    return iS


def calPredictors(fst):
    global bt
    bt = fst2bt(fst)
    nS = fst['lon'].shape[1]
    for iS in range(nS):
        getSpeedDir(np.array([iS], dtype=np.int32))
    gc.collect()
    for iS in range(nS):
        get_predictors(np.array([iS], dtype=np.int32))
    gc.collect()


# ---------------------------------------------------------------------------
# intensity: deterministic and stochastic autoregressive model


def get_E1(bt):
    NS = np.argwhere((bt['StormYear'][:] >= 1981) & (bt['StormYear'][:] <= 2012))[:, 0]
    data = bt['errors'][:, NS]
    data2 = bt['StormMwspd'][:, NS]
    E1 = E0 = v0E = None
    for it in range(0, data.shape[1], 1):
        data1 = data[:, it]
        data4 = data2[:, it]
        data4 = data4[data1 == data1]
        data1 = data1[data1 == data1]
        if E1 is None:
            E1 = data1[0 : data1.shape[0]]
            E0 = data1[0 : data1.shape[0]]
            v0E = data4[0 : data4.shape[0]]
        else:
            E1 = np.hstack([E1, data1[0 : data1.shape[0]]])
            E0 = np.hstack([E0, data1[0 : data1.shape[0]]])
            v0E = np.hstack([v0E, data4[0 : data4.shape[0]]])
    E2 = E0 * E1 * v0E
    E0 = E0[E2 == E2]
    E1 = E1[E2 == E2]
    v0E = v0E[E2 == E2]
    c1 = v0E.min()
    c2 = v0E.max()
    for iC in np.arange(v0E.min(), v0E.max(), 10):
        if v0E[v0E < iC].shape[0] > 50:
            c1 = iC
            break
    for iC in np.arange(v0E.max(), v0E.min(), -10):
        if v0E[v0E > iC].shape[0] > 50:
            c2 = iC
            break
    cat1 = np.arange(c1, c2 + 10, 10)
    return E0, v0E, cat1


def findError(errort, v0E, v0, cat1):
    if v0 == v0:
        if v0 <= cat1.min():
            j0 = 0
            error1 = errort[v0E < cat1[j0 + 1]]
        elif v0 >= cat1.max():
            j0 = len(cat1) - 1
            error1 = errort[v0E >= cat1[j0]]
        else:
            [j0, j1] = find_range(cat1, v0)
            error1 = errort[((v0E >= cat1[j0]) & (v0E < cat1[j0 + 1]))]
        error = random.choice(error1)
    else:
        error = 0
    return error


def find_range(array, a):
    if (np.diff(array) < 0).any():
        start = bisect_right(array[::-1], a)
        end = array.shape[0] - start
        start = end
    else:
        start = bisect_right(array, a)
        end = bisect_left(array, a)
    return (start - 1, end)


def getPrediction_v0input_result(
    bt, meanX, meanY, stdX, stdY, it, index, fstperiod, result, predictors, TimeDepends, v0, dvdt
):
    v1 = [v0]
    h1 = [bt.Time[it, index]]
    for ih in fstperiod:
        h1.append(bt.Time[it, index] + timedelta(hours=ih))
        lT = int(ih / 6)
        dy = result[0]
        count2 = 0
        for var in predictors:
            if var in 'SHRD':
                rvar = nanmean(
                    np.sqrt(
                        bt.UShearMean[it : it + lT + 1, index] ** 2
                        + bt.VShearMean[it : it + lT + 1, index] ** 2
                    )
                )
            elif var in 'dPIwspd':
                rvar = nanmean(bt.PIwspdMean[it : it + lT + 1, index]) - v0
            elif var in TimeDepends:
                rvar = nanmean(getattr(bt, var)[it : it + lT + 1, index])
            elif var in 'StormMwspd':
                rvar = v0
            elif var in 'dVdt':
                rvar = dvdt
            elif var in 'landmaskMean':
                rvar = getattr(bt, var)[it + lT + 1, index] - getattr(bt, var)[it, index]
            elif var in 'StormMwspd2':
                rvar = (getattr(bt, 'StormMwspd')[it, index]) ** 2
            elif var in 'StormMwspd3':
                rvar = (getattr(bt, 'StormMwspd')[it, index]) ** 3
            elif var in 'MPI':
                rvar = nanmean(bt.PIwspdMean[it : it + lT + 1, index])
            elif var in 'MPI2':
                rvar = nanmean(bt.PIwspdMean[it : it + lT + 1, index]) ** 2
            elif var in 'dPIwspd2':
                rvar = (nanmean(bt.PIwspdMean[it : it + lT + 1, index]) - v0) ** 2
            elif var in 'dPIwspd3':
                rvar = (nanmean(bt.PIwspdMean[it : it + lT + 1, index]) - v0) ** 3
            elif 'dVdt2' in var:
                rvar = dvdt**2
            elif 'dVdt3' in var:
                rvar = dvdt**3
            else:
                rvar = getattr(bt, var)[it, index]
            rvar = (rvar - meanX[count2 + 1]) / stdX[count2 + 1]
            dy = dy + rvar * result[count2 + 1]
            count2 = count2 + 1
        dy = (dy * stdY) + meanY
        v1.append(v0 + dy)
    return h1, v1


PREDICTORS_W = [
    'StormMwspd',
    'dVdt',
    'trSpeed',
    'dPIwspd',
    'SHRD',
    'rhMean',
    'dPIwspd2',
    'dPIwspd3',
    'dVdt2',
]
PREDICTORS_L = PREDICTORS_W + ['landmaskMean']
TIME_DEPENDS = ['dThetaEsMean', 'T200Mean', 'rhMean', 'rh500_300Mean', 'div200Mean']


def _regression(it, lT, iS):
    if (bt.landmaskMean[it + lT, iS] <= -0.5) & (bt.landmaskMean[it, iS] <= -0.5):
        return PREDICTORS_W, result_w, meanX_w, meanY_w, stdX_w, stdY_w
    return PREDICTORS_L, result_l, meanX_l, meanY_l, stdX_l, stdY_l


def get_stochastic(iiS):
    iS = np.int_(iiS.mean(keepdims=True))
    dummy = bt.StormLon[:, iS][bt.StormLon[:, iS] == bt.StormLon[:, iS]]
    if (dummy.any()) and (np.abs(bt.StormLat[0, iS]) >= 5.0):
        it1 = 0
        it2 = np.min(
            [np.argwhere(bt.StormLon[:, iS] == dummy[-1])[-1, 0], bt.StormLon.shape[0] - 5]
        )
        if (
            (bt.StormLat[0, iS] >= 0)
            and (bt.StormLon[0, iS] >= 120)
            and (bt.StormLon[0, iS] <= 180)
        ):
            bt.stochastic[0, iS, iNN] = np.max([20, random.choice(intV[intV == intV])])  # kt
        else:
            bt.stochastic[0, iS, iNN] = np.max([25, random.choice(intV[intV == intV])])  # kt
        ih = 12
        lT = np.int_(ih / 6)
        v0 = bt.stochastic[0, iS, iNN]
        dvdt = 0.0
        for it in range(it1, it2 + 2, 2):
            if (
                (it + lT < bt.StormLon.shape[0])
                and (bt.StormLon[it, iS] == bt.StormLon[it, iS])
                and (bt.landmaskMean[it, iS] == bt.landmaskMean[it, iS])
                and (bt.landmaskMean[it + lT, iS] == bt.landmaskMean[it + lT, iS])
                and not np.isnan(v0)
            ):
                predictors, result, meanX, meanY, stdX, stdY = _regression(it, lT, iS)
                h1, v1 = getPrediction_v0input_result(
                    bt,
                    meanX,
                    meanY,
                    stdX,
                    stdY,
                    it,
                    iS,
                    [ih],
                    result,
                    predictors,
                    TIME_DEPENDS,
                    v0,
                    dvdt,
                )
                error = findError(E0, v0E, v0, cat1)
                bt.error[it, iS, iNN] = error
                v1 = v1[1] - error
                if v1 < 10:
                    break
                bt.stochastic[it + lT, iS, iNN] = v1
                if (
                    (bt.stochastic[it1 : it + lT : lT, iS, iNN].max() > 35)
                    and (bt.stochastic[it, iS, iNN] <= 35)
                    and (bt.stochastic[it - lT, iS, iNN] <= 35)
                ):
                    break
                dvdt = v1 - v0
                v0 = v1
        for iit in range(it1 + 1, np.min([it2 + 1, bt.StormLon.shape[0] - 2]), 2):
            a = bt.stochastic[iit - 1, iS, iNN] * bt.stochastic[iit + 1, iS, iNN]
            if a == a:
                bt.stochastic[iit, iS, iNN] = 0.5 * (
                    bt.stochastic[iit - 1, iS, iNN] + bt.stochastic[iit + 1, iS, iNN]
                )
    return iS


def calIntensity(n_members: int):
    """Stochastic intensity members along the tracks in the global `bt`. The
    notebook's deterministic pass fills `determin`, which nothing downstream
    reads, so it is skipped."""
    global result_w, meanY_w, stdX_w, meanX_w, stdY_w
    global result_l, meanY_l, stdX_l, meanX_l, stdY_l
    global E0, v0E, cat1, intV, iNN

    bt2 = nc.Dataset(INPUTS / gv.obpath, 'r')
    with nc.Dataset(INPUTS / 'result_w.nc', 'r') as f:
        result_w = ma.getdata(f['params'][:])
    with nc.Dataset(INPUTS / 'result_l.nc', 'r') as f:
        result_l = ma.getdata(f['params'][:])
    NS = np.argwhere((bt2['StormYear'][:] >= 1981) & (bt2['StormYear'][:] <= 2012))[:, 0]
    intV = bt2['StormMwspd'][:][0, NS]
    E0, v0E, cat1 = get_E1(bt2)
    bt2.close()

    with nc.Dataset(INPUTS / 'coefficient_meanstd.nc', 'r') as f:
        meanX_w = ma.getdata(f['meanX_w'][:])
        stdX_w = ma.getdata(f['stdX_w'][:])
        meanY_w = ma.getdata(f['meanY_w'][:])
        stdY_w = ma.getdata(f['stdY_w'][:])
        meanX_l = ma.getdata(f['meanX_l'][:])
        stdX_l = ma.getdata(f['stdX_l'][:])
        meanY_l = ma.getdata(f['meanY_l'][:])
        stdY_l = ma.getdata(f['stdY_l'][:])

    shape = bt.StormLon.shape
    bt.stochastic = np.zeros([shape[0], shape[1], n_members]) * np.nan
    bt.error = np.zeros([shape[0], shape[1], n_members]) * np.nan
    for iNN in range(n_members):
        for iS in range(shape[1]):
            get_stochastic(np.array([iS], dtype=np.int32))
        gc.collect()
    return bt


# ---------------------------------------------------------------------------
# clean-up and output, as in the notebook


def getbasinMap():
    xbin, ybin = np.arange(0, 365, 5), np.arange(-90, 95, 5)
    xcenter = 0.5 * (xbin[0:-1] + xbin[1:])
    ycenter = 0.5 * (ybin[0:-1] + ybin[1:])
    basinMap = np.zeros([xcenter.shape[0], ycenter.shape[0]])
    lonc, latc = np.meshgrid(xcenter, ycenter)
    regions = [
        (1, (lonc >= 235) & (latc >= 0) & (latc <= 45)),
        (2, (lonc >= 180) & (lonc < 235) & (latc >= 0) & (latc <= 45)),
        (3, (lonc >= 100) & (lonc < 180) & (latc >= 0) & (latc <= 45)),
        (4, (lonc < 100) & (latc >= 0) & (latc <= 45)),
        (5, (lonc < 90) & (latc < 0) & (latc >= -45)),
        (6, (lonc >= 90) & (lonc < 160) & (latc < 0) & (latc >= -45)),
        (7, (lonc >= 160) & (lonc < 240) & (latc < 0) & (latc >= -45)),
    ]
    for code, m in regions:
        n = np.argwhere(m)
        basinMap[n[:, 1], n[:, 0]] = code
    a = np.arange(xcenter.shape[0] * ycenter.shape[0]).reshape(xcenter.shape[0], ycenter.shape[0])
    # cells of the notebook's hand-drawn East Pacific / Atlantic boundary
    ep_cells = [
        1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1746, 1747, 1748, 1749,
        1750, 1751, 1752, 1753, 1754, 1755, 1782, 1783, 1784, 1785, 1786, 1787, 1818, 1819,
        1820, 1821, 1822, 1823, 1854, 1855, 1856, 1857, 1890, 1891, 1892, 1893, 1926, 1927,
        1928, 1962, 1963, 1964, 1998, 1999,
    ]  # fmt: skip
    basinMap[np.isin(a, ep_cells)] = 2
    return xbin, ybin, basinMap


BASIN_NAMES = {0: 'none', 1: 'NA', 2: 'EP', 3: 'WP', 4: 'NI', 5: 'SI', 6: 'AU', 7: 'SP'}


def defineBasin(lon0_obs, lat0_obs, basinlon, basinlat, basinMap):
    basin = np.zeros(lon0_obs.shape)
    lat0_obs = np.clip(lat0_obs, -89.9, 89.9)
    notNaN_arg = np.where(~np.isnan(lon0_obs * lat0_obs))
    x = np.floor(lon0_obs[notNaN_arg] / np.diff(basinlon)[0])
    y = np.floor((lat0_obs[notNaN_arg] - basinlat[0]) / np.diff(basinlat)[0])
    x[x == basinMap.shape[0]] = basinMap.shape[0] - 1
    y[y == basinMap.shape[1]] = basinMap.shape[1] - 1
    basin[notNaN_arg] = basinMap[np.int_(x), np.int_(y)]
    return basin


def to_dataset(bt1) -> xr.Dataset:
    nan = np.float64('nan')
    xbin, ybin, basinMap = getbasinMap()
    basin = defineBasin(bt1.StormLon[0, :], bt1.StormLat[0, :], xbin, ybin, basinMap)

    # storms that never developed
    bt1.stochastic[bt1.stochastic == 0] = nan
    max5 = np.nanmax(bt1.stochastic[0:21, :, :], axis=0)
    bt1.stochastic[:, *np.nonzero(max5 < 35)] = nan
    # storms that became unstable
    maxall = np.nanmax(bt1.stochastic, axis=0)
    bt1.stochastic[:, *np.nonzero(maxall > 300)] = nan
    # storms seeded within 2 degrees of the equator, or outside every basin
    iS1 = np.argwhere(np.abs(bt1.StormLat[0, :]) < 2)[:, 0]
    bt1.stochastic[:, iS1, :] = nan
    bt1.StormLon[0, iS1] = nan
    bt1.StormLon[0, np.argwhere(basin == 0)] = nan
    # storms with no intensity record
    maxall = np.nanmax(bt1.stochastic, axis=0)
    bt1.stochastic[:, *np.nonzero(maxall <= 0)] = nan
    arg = np.argwhere(bt1.StormLon[0, :] == bt1.StormLon[0, :])[:, 0]

    newwspd = bt1.stochastic[:, arg]
    newdatenum = bt1.Time[:, arg]
    dims2 = ['lifelength', 'stormID']
    ds = xr.Dataset(
        {
            'longitude': (dims2, bt1.StormLon[:, arg], {'units': 'degrees east'}),
            'latitude': (dims2, bt1.StormLat[:, arg], {'units': 'degrees north'}),
            'Mwspd': (['ensembleNum', *dims2], np.rollaxis(newwspd, 2, 0), {'units': 'kt'}),
            'year': (['stormID'], bt1.StormYear[arg], {'units': 'year'}),
            'time': (
                dims2,
                date2num(newdatenum, units='days since 1950-01-01 00:00', calendar='standard'),
                {'units': 'days since 1950-01-01 00:00'},
            ),
        }
    )
    return ds


# ---------------------------------------------------------------------------
# driver


def run_one(task):
    """One track realization with its intensity members; returns stage times."""
    global yearTC
    year, ichaz, ratio, n_members, tcgi = task
    out = OUT / f'{year}_r{ichaz:03d}.nc'
    if out.exists():
        return None
    yearTC = year
    gv.TCGIinput = tcgi
    seed = year * 1000 + ichaz
    random.seed(seed)
    np.random.seed(seed)
    _load_landmask()
    t = {'year': year, 'realization': ichaz, 'seed': seed}
    t0 = time.perf_counter()
    cDate, cLon, cLat = getSeeding(year, ratio)
    t['seeds'] = cLon.size
    t['s_genesis'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    fst = getBam(cDate, cLon, cLat)
    t['s_tracks'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    calPredictors(fst)
    t['s_predictors'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    calIntensity(n_members)
    t['s_intensity'] = time.perf_counter() - t0
    ds = to_dataset(bt)
    t['storms_kept'] = ds.sizes['stormID']
    tmp = out.with_suffix('.nc.part')
    ds.to_netcdf(tmp)
    tmp.rename(out)
    return t


def cmd_run(args):
    import multiprocessing as mp

    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [
        (args.year, r, args.ratio, args.members, args.tcgi) for r in parse_range(args.realizations)
    ]
    timing_path = OUT / 'timing.csv'
    t_start = time.perf_counter()
    rows = []
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:
        for t in pool.imap_unordered(run_one, tasks):
            if t is None:
                continue
            t['s_total'] = t['s_genesis'] + t['s_tracks'] + t['s_predictors'] + t['s_intensity']
            rows.append(t)
            print(
                f'{t["year"]} r{t["realization"]:03d}: {t["seeds"]} seeds, {t["storms_kept"]} kept; '
                f'tracks {t["s_tracks"]:.0f}s, predictors {t["s_predictors"]:.0f}s, '
                f'intensity {t["s_intensity"]:.0f}s ({time.perf_counter() - t_start:.0f}s elapsed)',
                flush=True,
            )
            pd.DataFrame([t]).to_csv(
                timing_path, mode='a', header=not timing_path.exists(), index=False
            )
    if rows:
        df = pd.DataFrame(rows)
        wall = time.perf_counter() - t_start
        print(
            f'{len(df)} realizations in {wall:.0f}s on {args.workers} workers; '
            f'per realization-year {df["s_total"].mean():.0f}s '
            f'(tracks {df["s_tracks"].mean():.0f}, predictors {df["s_predictors"].mean():.0f}, '
            f'intensity {df["s_intensity"].mean():.0f})'
        )


# ---------------------------------------------------------------------------
# compare with the published ERA5 track files


def year_stats(path: Path, year: int, source: str, unit: str) -> list[dict]:
    """Per intensity member: storms with wind, by genesis basin, and lifetime
    peak winds, for storms whose first node falls in the year."""
    ds = xr.open_dataset(path, decode_times=False)
    yr = ds['year'].values
    sel = np.nonzero(yr == year)[0]
    lon0 = ds['longitude'].values[0, sel]
    lat0 = ds['latitude'].values[0, sel]
    w = ds['Mwspd'].values[:, :, sel]  # member, node, storm; kt
    w = np.where(np.isfinite(ds['longitude'].values[:, sel])[None], w, np.nan)
    xbin, ybin, basinMap = getbasinMap()
    basin = defineBasin(lon0 % 360, lat0.copy(), xbin, ybin, basinMap)
    peak = np.nanmax(np.where(np.isfinite(w), w, -np.inf), axis=1)  # member, storm
    rows = []
    for m in range(w.shape[0]):
        alive = np.isfinite(peak[m]) & (peak[m] > 0)
        p = peak[m][alive]
        row = {
            'source': source,
            unit: path.stem,
            'member': m,
            'trajectories': sel.size,
            'storms': int(alive.sum()),
            'hurricanes': int((p >= 64).sum()),
            'major': int((p >= 96).sum()),
            'peak_median_kt': float(np.median(p)) if p.size else np.nan,
            'peak_p90_kt': float(np.percentile(p, 90)) if p.size else np.nan,
        }
        for code, name in BASIN_NAMES.items():
            if code:
                row[name] = int((alive & (basin == code)).sum())
        rows.append(row)
    return rows


def cmd_compare(args):
    ours = sorted(OUT.glob(f'{args.year}_r*.nc'))
    theirs = sorted(PUBLISHED_TRACKS.glob('global_2019_2ens*_pre.nc'))
    rows = []
    for p in ours:
        rows += year_stats(p, args.year, 'ours', 'file')
    for p in theirs:
        rows += year_stats(p, args.year, 'published', 'file')
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'compare_{args.year}.csv', index=False)
    cols = ['trajectories', 'storms', 'hurricanes', 'major', 'peak_median_kt', 'peak_p90_kt'] + [
        n for c, n in BASIN_NAMES.items() if c
    ]
    per_file = df.groupby(['source', 'file'])[cols].mean()
    summary = per_file.groupby('source').agg(['mean', 'std']).T.unstack()
    pd.set_option('display.width', 160)
    print(f'{args.year}: {len(ours)} of our realizations, {len(theirs)} published files')
    print('per realization (file), averaged over its 40 members; mean and spread across files')
    print(summary.round(1).to_string())


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-')
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('run', help='downscale one year, one process per track realization')
    q.add_argument('--year', type=int, default=2017)
    q.add_argument('--realizations', default='1-40')
    q.add_argument('--members', type=int, default=40, help='intensity members per realization')
    q.add_argument('--ratio', type=float, default=1.0, help='seeding ratio')
    q.add_argument('--tcgi', default='TCGI_CRH_PI', choices=['TCGI_CRH_PI', 'TCGI_SD_PI'])
    q.add_argument('--workers', type=int, default=max((os.cpu_count() or 2) - 6, 1))
    q.set_defaults(func=cmd_run)
    q = sub.add_parser('compare', help='our realizations against the published track files')
    q.add_argument('--year', type=int, default=2017)
    q.set_defaults(func=cmd_compare)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
