"""
My solar functions

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import os
import time
import datetime as dt
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

from pvlib.tools import cosd, sind


# !!ALSO: Make sure to check all of your punctuation like semicolons for neatness.

def getCvsrCSM(how='all', time_it='False'):
    """
    Calculates CVSR CSM for irradiance and power output as well as all other
    relevant variables.

    Parameters
    ----------
    how     : [string] 'all' to get the entire DataFrame back
            :          'pow_only' to get only the modeled clear sky power output
    time_it : [bool] weather or not to time the process

    Returns
    -------
    data    : [DataFrame] if how == 'all'
    or
    CSM_Pow : [array of floats] if how == 'pow_only'
    """

    # Upload CVSR data
    data = getCvsrData('cvsr_ground_2015.csv', time_it=time_it)

    # Get the Ineichen CSM, zenith angle, airmass, and extraterrestrial irradiance
    (data['GHI'], data['DNI'], data['DHI'],
     data['zenith'], data['AM'], data['IE']) = getIneichenCsm(data.index, lat=35.33,
                                                              lon=-119.91, Lz=120.,
                                                              Z=610., Turb=2.5,
                                                              time_it=time_it)
    # Get the azimuthal
    data['azimuth'] = getAzimuthAngle(data.index, lat=35.33, lon=-119.91,
                                      Z=610, time_it=time_it)

    # Remove elevated zenith angles
    data = removeHighZenith(data, 85, time_it=time_it)

    # Determine the optimal rotation angle for CVSR panels
    data['R'] = getCvsrPanelRotation(data.zenith, data.azimuth, time_it=time_it)

    # Determine the panel surface's azimuthal angle at the optimal rotation angle
    data['gamma'] = getCvsrPanelAzimuth(data.R, time_it=time_it)

    # Determine the sun's angle of incidence on a panel at the optimal roation angle
    data['AOI'] = getAOI(data.zenith, data.azimuth, data.R, data.gamma, time_it=time_it)

    # Model the panel POA diffuse irradiance using the Perez 1999 model
    (data['POAd'], data['F1'], data['F2']) = getPerez1999PoaDiff(data.DNI, data.DHI,
                                                                 data.zenith, data.AOI,
                                                                 data.R, data.AM,
                                                                 data.IE, time_it=time_it)

    # Model the POA global irradiance using the closure equation
    data['POA'] = getPoaGlobal(data.DNI, data.AOI, data.POAd, time_it=time_it)

    # Interpolate the data when rotation angle is greater than 65 degrees
    data = interpR65(data, time_it=time_it)

    # Convert irradiance to power
    data['CSM_Pow'] = poa2pow(data.index, data.POA, time_it=time_it)

    # Return the dataframe to the user
    if how == 'all':
        out = data
    elif how == 'pow_only':
        out = data.CSM_Pow
    return out


def getCvsrData(fname='cvsr_ground_2015.csv', time_it=False):
    """
    Reads in the CVSR data from csv file stored in current working directory

    Parameters
    ----------
    fname   : [string] filename
    time_it : [bool] weather or not to time the process

    Returns
    -------
    data    : [DataFrame] DataFrame of CVSR data    
    """

    # Start a timer
    tic = time.clock();

    # Load the data
    data = pd.read_csv('cvsr_ground_2015.csv', delimiter=',', parse_dates=0,
                       index_col=0, header=0,
                       usecols=['timestamp', 'station1_poa1', 'station1_poa2',
                                'station2_poa1', 'station3_poa1', 'power_output', 'station1_temperature',
                                'station2_temperature',
                                'station3_temperature', 'station1_wind_speed', 'station2_wind_speed',
                                'station3_wind_speed'])

    # Corrects for UTC time:
    # This is necessary because the data from CVSR was (supposedly) logged in UTC,
    # however, we noticed that the data had the 1 hour shift of DST.
    idx = (data.index > '2015-03-08') & (data.index < '2015-11-01')
    data[idx].index = data[idx].index - pd.Timedelta(hours=1)

    # Change the dataframe index to PST
    data.index = data.index - pd.Timedelta(hours=8)

    # Stop the timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        # Show the user
        print('Importing data took ' + "%.2f" % (toc - tic) + ' s.')

    # Return the CVSR DataFrame
    return data


def getIneichenCsm(times=pd.date_range('1/1/2015', periods=24 * 60, freq='min'),
                   lat=35.33, lon=-119.91, Lz=120., Z=610., Turb='mean', time_it=False):
    """
    Note: Make sure that you have the 'LinkeTurbidity' folder in your current 
    working directory
        
    Clear sky GHI, DNI, and DHI from Ineichen model.

    Parameters
    ----------
    times   : [DatetimeIndex] local time
    lat     : [float] Latitude  (South is negative) in degrees
    lon     : [float] Longitude (West is negative)  in degrees
    Lz      : [float] Local time meridian. 
            :     In the United States 
            :     Lz = 120 for the Pacific  time zone 
            :     Lz = 105 for the Mountain time zone 
            :     Lz = 90  for the Central  time zone
            :     Lz = 75  for the Eastern  time zone
    Z       : [float] Altitude in meters
    Turb    : [string] 'mean'  to use average annual average Linke turbidity
            :          'month' to use monthly averages of Linke turbidity
    time_it : [bool] weather or not to time the process

    Returns
    -------
    GHI    : [array of floats] Global  Horizontal Irradiance in W/m2
    DNI    : [array of floats] Direct  Normal     Irradiance in W/m2
    DHI    : [array of floats] Diffuse Horizontal Irradiance in W/m2
    zenith : [array of floats] solar zenith angle in degrees 
    am     : [array of floats] Kasten air mass, dimensionless
    IE     : [array of floats] Extraterrestrial   Irradiance in W/m2
    
    """

    # Start a timer
    tic = time.clock()

    # Day of the year
    DoY = times.dayofyear

    # Time of Day in hours
    ToD = times.hour + times.minute / 60. + times.second / 3600.

    # Longitude
    lon = - lon

    # dr = inverse relative distance Earth-Sun 
    # (correction for eccentricity of Earth's orbit around the sun) 
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * DoY)
    b = 2 * np.pi / 364 * (DoY - 81)

    # delta = Declination of the sun above the celestial equator (radians)
    delta = 0.409 * np.sin(2 * np.pi / 365 * DoY - 1.39)
    Sc = 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)

    # ASCE
    omega = np.pi / 12 * (ToD + 1. / 15 * (Lz - lon) + Sc - 12)
    phi = np.pi * lat / 180

    del ToD  # Clean up

    # theta = solar altitude 
    theta = np.arcsin(np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega)) * 180 / np.pi;

    # zenith = zenith angle
    zenith = 90. - theta

    # Solar altitude in radians
    h = theta / 180 * np.pi

    # Solar constant
    I0 = 1367.7
    IE = I0 * dr
    d1 = times.month

    if type(Turb) == str:
        # Set up months
        Months = ('January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December')

        # Map location
        La = np.linspace(90, -90, 2160)
        Lo = np.linspace(-180, 180, 4320)
        I1 = np.argmin(abs(lat - La)) + 1
        I2 = np.argmin(abs(-lon - Lo)) + 1
        Tl = np.zeros(12)

        # Get the turbidity for EVERY month from Worldwide Maps
        for m in range(12):
            fname = os.getcwd() + '/LinkeTurbidity/' + Months[int(m)] + '.tif'
            Lt = imread(fname)
            L1 = Lt[I1 - 1, I2 - 1]
            L2 = float(L1)
            Tl[int(m)] = L2 / 20.

        # Decide how to use the monthly average turbidity
        TL = np.zeros(len(h))
        if Turb == 'mean':
            # Use annual average turbidity
            TL[:] = np.mean(Tl)
        else:
            # Use monthly turbidity from maps
            for j in range(len(h)):
                TL[j] = Tl[times[j].month - 1]
    elif type(Turb) == float:
        TL = np.zeros(len(h))
        TL[:] = Turb

    # Constants
    a = 0.15
    b = 3.885
    c = 1.253
    h1 = np.array(h.values)
    #print(h1)
    h1[h1 < 0] = 0

    # Kasten air mass
    am = (np.sin(h1) + a * (h1 * np.pi / 180 + b) ** (-c)) ** (-1)

    # altitude corrections
    fh1 = np.e ** (-Z / 8000)
    fh2 = np.e ** (-Z / 1250)

    # Coefficients found by Ineichen
    cg1 = (0.0000509 * Z + 0.868)
    cg2 = (0.0000392 * Z + 0.0387)
    bn = 0.664 + 0.163 / fh1

    # Calculate CSM
    GHI = np.zeros(len(am))
    DNI = np.zeros(len(am))
    for i in range(len(am)):
        if np.e ** (0.01 * am[i] ** (1.8)) < 5:
            GHI[i] = cg1 * IE[i] * np.cos(np.pi / 2. - h[i]) * np.e ** (
                        -cg2 * am[i] * (fh1 + fh2 * (TL[i] - 1))) * np.e ** (0.01 * am[i] ** (1.8))
        else:
            GHI[i] = cg1 * IE[i] * np.cos(np.pi / 2. - h[i]) * np.e ** (-cg2 * am[i] * (fh1 + fh2 * (TL[i] - 1))) * 0.5;
        DNI[i] = bn * IE[i] * np.e ** (-0.09 * am[i] * (TL[i] - 1))
    DHI = GHI - DNI * np.cos(np.deg2rad(zenith.values))
    # remove negative
    GHI[GHI < 0.01] = 0
    DNI[DNI < 0.01] = 0
    DHI[DHI < 0.01] = 0

    # Stop the timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        # Show the user
        print('Ineichen model took ' + "%.2f" % (toc - tic) + ' s.')

    # Return result
    return GHI, DNI, DHI, zenith, am, IE


def getAzimuthAngle(times=pd.date_range('1/1/2015', periods=24 * 60, freq='min'),
                    lat=35.33, lon=-119.91, Z=610., time_it=False):
    """
    Calculates the solar azimuth angle

    Parameters
    ----------
    times   : [DatetimeIndex] local time
    lat     : [float] Latitude  (South is negative) in degrees
    lon     : [float] Longitude (West is negative)  in degrees
    Z       : [float] Altitude in meters
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    az      : [array] Solar azimuth angle in degrees
    """

    # Start the timer
    tic = time.clock()

    # Shift the local time to UTC (No Daylight Savings)
    UTCIndex = times + pd.Timedelta('8 hours')  # Convert to UTC

    # julian date number (number of days since noon on January 1, 4713 B.C.)
    jd = UTCIndex.to_julian_date()

    # offset (2451543.5)
    d_offset = pd.Timestamp('1999-12-31 00:00:00').to_julian_date()
    d = jd.values - d_offset

    # Keplerian elements for the sun (geocentric)
    w = 282.9404 + 4.70935E-5 * d  # longitude of perihelion [degrees]
    a = 1.0  # mean distance [AU]
    e = 0.016709 - 1.151E-9 * d  # eccentricity [-]
    M = np.mod(356.0470 + 0.9856002585 * d, 360.0)  # mean anomaly [degrees]
    L = w + M  # Sun's mean longitude [degrees]
    oblecl = 23.4393 - 3.563E-7 * d  # Sun's obliquity of the eliptic [degrees]

    # Auxiliary angle [degrees]
    E = M + (180.0 / np.pi) * e * np.sin(np.deg2rad(M)) * (1.0 + e * np.cos(np.deg2rad(M)))

    # Rectangular coordinates in the plane of the ecliptic (x-axis toward perihelion)
    x = np.cos(np.deg2rad(E)) - e
    y = np.sin(np.deg2rad(E)) * np.sqrt(1 - (e ** 2))

    # Distance (r) and true anomaly (v)
    r = np.sqrt((x ** 2) + (y ** 2))
    v = np.rad2deg(np.arctan2(y, x))

    # Longitude of the sun
    lon_sun = v + w

    # Ecliptic rectangular coordinates
    xeclip = r * np.cos(np.deg2rad(lon_sun))
    yeclip = r * np.sin(np.deg2rad(lon_sun))
    zeclip = 0.0

    # Rotate coordinates to equatorial rectangular coordinates
    xequat = xeclip
    yequat = yeclip * np.cos(np.deg2rad(oblecl)) + zeclip * np.sin(np.deg2rad(oblecl))
    zequat = yeclip * np.sin(np.deg2rad(23.4406)) + zeclip * np.cos(np.deg2rad(oblecl))

    # Convert equatorial rectangular coordinates to right-ascension (RA) and declination
    r = np.sqrt(xequat ** 2 + yequat ** 2 + zequat ** 2) - (Z / 149598000.0)
    RA = np.rad2deg(np.arctan2(yequat, xequat))
    delta = np.rad2deg(np.arcsin(zequat / r))

    # Calculate local siderial time
    uth = UTCIndex.hour + (UTCIndex.minute / 60.0) + (UTCIndex.second / 3600.0)
    gmst0 = np.mod(L + 180.0, 360.0) / 15.0
    sidtime = gmst0 + uth + (lon / 15.0)

    # Replace RA with hour-angle (HA)
    HA = sidtime * 15.0 - RA

    # Convert to rectangular coordinates
    x = np.cos(np.deg2rad(HA)) * np.cos(np.deg2rad(delta))
    y = np.sin(np.deg2rad(HA)) * np.cos(np.deg2rad(delta))
    z = np.sin(np.deg2rad(delta))

    # Rotate along an axis going East-West
    xhor = x * np.cos(np.deg2rad(90.0 - lat)) - z * np.sin(np.deg2rad(90.0 - lat))
    yhor = y

    # Stop the timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        # Show the user
        print('Azimuth calculations took ' + "%.2f" % (toc - tic) + ' s.')

    # Find azimuthal angle
    return np.rad2deg(np.arctan2(yhor, xhor)) + 180.0


def removeHighZenith(data, Z_thresh=75., time_it=False):
    """
    Removes zenith angles greater than the given threshold

    Parameters
    ----------
    data     : [DataFrame] DataFrame of CVSR data to be cleaned up
    Z_thresh : [float or int] upper bound on zenith angles
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    data      : [DataFrame] DataFrame of CVSR that has been cleaned up
    """
    # Start the timer
    tic = time.clock()

    # Filter based on the zenith threshold
    data.loc[data.zenith > Z_thresh,
             ['GHI', 'DNI', 'DHI', 'zenith', 'AM', 'IE', 'azimuth', 'station1_poa1',
              'station1_poa2', 'station2_poa1', 'station3_poa1', 'power_output']] = np.NaN

    # Stop the timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        # Show the user
        print('Filter elevated zenith angles took ' + "%.2f" % (toc - tic) + ' s.')

    # Return new DataFrame
    return data


def getCvsrPanelRotation(zenith, azimuth, time_it=False):
    """
    Calculates CVSR panel rotation for given solar position

    Parameters
    ----------
    zenith  : [array] Solar zenith  angles in degrees
    azimuth : [array] Solar azimuth angles in degrees
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    R      : [array] Panel rotation angle in degrees
    """

    # Start a timer
    tic = time.clock()

    # Convert to radians to simplify calculations
    zenith = np.deg2rad(zenith)
    azimuth = np.deg2rad(azimuth)

    # Calculate optimal rotation angle [in radians]
    R = np.arctan(np.tan(zenith) * np.sin(azimuth))

    # Convert back to degrees
    R = np.rad2deg(R)

    # Stop the timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        print('Determining optimal panel rotation took ' + "%.0f" % ((toc - tic) * 1000) +
              ' ms.')

    # Return the result
    return R


def getCvsrPanelAzimuth(R, time_it=False):
    """
    Calculates CVSR panel azimuth in degrees at a given rotation angle in degrees

    Parameters
    ----------
    R       : [array] Panel rotation angle in degrees
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    gamma   : [array] Panel azimuth angle in degrees at a given rotation in degrees
    """
    # Start a timer
    tic = time.clock()

    # Convert to radians
    R = np.deg2rad(R)

    # Calculate panel surface azimuth [in radians]
    gamma = np.arcsin(np.sin(R) / np.sin(abs(R)))

    # Convert back to degrees
    gamma = np.rad2deg(gamma)

    # Stop the timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        print('Determining panel surface azimuth ' + "%.2f" % (toc - tic) + ' s.');

    # Return the result
    return gamma


def getAOI(zenith, azimuth, R, gamma, time_it=False):
    """
    Calculates CVSR panel angle of incidence in degrees at a given rotation angle 
    in degrees

    Parameters
    ----------
    zenith  : [array] Solar zenith   angles in degrees
    azimuth : [array] Solar azimuth  angles in degrees
    R       : [array] Panel rotation angles in degrees
    gamma   : [array] Panel azimuth  angles in degrees
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    AOI     : [array] Panel incidence angle in degrees at a given rotation in degrees
    """
    # Start a timer
    tic = time.clock()

    # Convert to radians
    zenith = np.deg2rad(zenith)
    azimuth = np.deg2rad(azimuth)
    R = np.deg2rad(R)
    gamma = np.deg2rad(gamma)

    # Calculate the sun's AOI on the panel [in radians]
    AOI = np.arccos(np.cos(abs(R)) * np.cos(zenith) + np.sin(abs(R)) * np.sin(zenith) *
                    np.cos(azimuth - gamma))

    # Convert back to degrees
    AOI = np.rad2deg(AOI)

    # Stop the timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        print('Determining the angle of incidence took ' + "%.2f" % (toc - tic) + ' s.');

    # Return the result
    return AOI


def get1999PoaDiff(DNI, DHI, zenith, AOI, R, AM, IE, time_it=False):
    """
    Calculates plane of array diffuse irradiance using the 1999 Perez model

    Parameters
    ----------
    DNI     : [array of floats] Direct  Normal     Irradiance in W/m2
    DHI     : [array of floats] Diffuse Horizontal Irradiance in W/m2
    zenith  : [array of floats] solar zenith angle in degrees 
    AOI     : [array] Panel incidence angle in degrees at a given rotation in degrees
    R       : [array] Panel rotation angles in degrees
    AM      : [array of floats] Kasten air mass, dimensionless
    IE      : [array of floats] Extraterrestrial   Irradiance in W/m2
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    POAd    : [array] Plane of array diffuse irradiance using the 1999 Perez model
    F1      : [array] Perez's coefficients of circumsolar    anisotropy
    F2      : [array] Perez's coefficients of horizon/zenith anisotropy
    """
    # Start the timer
    tic = time.clock()

    # Convert to radians
    zenith = np.deg2rad(zenith)
    AOI = np.deg2rad(AOI)
    R = np.deg2rad(R)

    # Calculate the sky's clearness
    eps = ((DHI + DNI) / DHI + 1.041 * zenith ** 3) / (1 + 1.041 * zenith)
    print(eps)
    # Calculate the sky's brightness
    Delta = DHI * AM / IE

    # Input the Fmatrix
    Fmat = pd.DataFrame(index=range(1, 9), columns=['F11', 'F12', 'F13', 'F21', 'F22', 'F23'],
                        data=[[-0.008, 0.588, -0.062, -0.060, 0.072, -0.022],
                              [0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
                              [0.330, 0.487, -0.221, 0.055, -0.064, -0.026],
                              [0.568, 0.187, -0.295, 0.109, -0.152, -0.014],
                              [0.873, -0.392, -0.362, 0.226, -0.462, 0.001],
                              [1.132, -1.237, -0.412, 0.288, -0.823, 0.056],
                              [1.060, -1.600, -0.359, 0.264, -1.127, 0.131],
                              [0.678, -0.327, -0.250, 0.156, -1.377, 0.251]])

    # Do the epsBin mapping
    epsBin = np.copy(eps)
    print(epsBin)
    epsBin[(epsBin >= 1.000) * (epsBin < 1.065)] = 1
    print(epsBin)
    epsBin[(epsBin >= 1.065) * (epsBin < 1.230)] = 2
    print(epsBin)
    epsBin[(epsBin >= 1.230) * (epsBin < 1.500)] = 3
    print(epsBin)
    epsBin[(epsBin >= 1.500) * (epsBin < 1.950)] = 4
    print(epsBin)
    epsBin[(epsBin >= 1.950) * (epsBin < 2.800)] = 5
    print(epsBin)
    epsBin[(epsBin >= 2.800) * (epsBin < 4.500)] = 6
    print(epsBin)
    epsBin[(epsBin >= 4.500) * (epsBin < 6.200)] = 7
    print(epsBin)
    epsBin[epsBin >= 6.200] = 8
    print(epsBin)

    F1 = np.zeros(len(eps))
    F1.fill(np.NaN)
    F2 = np.copy(F1)
    # Calculate F1 and F2
    for i in range(1, 9):
        idx = epsBin == i
        F1[idx] = Fmat.F11[i] + Fmat.F12[i] * Delta[idx] + Fmat.F13[i] * (zenith[idx])
        F2[idx] = Fmat.F21[i] + Fmat.F22[i] * Delta[idx] + Fmat.F23[i] * (zenith[idx])

    # calculate a and b
    a = np.cos(AOI)
    a[a <= 0] = 0
    b = np.cos(zenith)
    b[b <= 0.087] = 0.087
    print(F1)
    print(F2)
    # Now employ the model
    POAd = DHI * ((1 - F1) * (1 + np.cos(abs(R))) / 2. + F1 * a / b + F2 * np.sin(abs(R)))

    # Stop the timer
    toc = time.clock()

    # If the user waned to time it,
    if time_it == True:
        print('The Perez 1999 model for POA irradiance took ' + "%.2f" % (toc - tic) +
              ' s to run.')

    # Return the result
    return POAd, F1, F2


def getPoaGlobal(DNI, AOI, POAd, time_it=False):
    """
    Calculates plane of array global irradiance using the closure equation

    Parameters
    ----------
    DNI     : [array of floats] Direct  Normal     Irradiance in W/m2
    AOI     : [array] Panel incidence angle in degrees at a given rotation in degrees
    POAd    : [array] Plane of array diffuse irradiance using the 1999 Perez model
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    POA    : [array] Plane of array global irradiance using the closure equation
    """
    # Start timer
    tic = time.clock()

    # Conver to radians
    AOI = np.deg2rad(AOI)

    # Use the closure equation
    POA = DNI * np.cos(AOI) + POAd

    # Stop timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        print('Calculating POA global from closure equation took ' + "%.2f" % (toc - tic) +
              ' s.')

    # Return the result
    return POA


def interpR65(data, time_it=False):
    """
    Interpolates from global irradiance at sunrise/sunset to POA at |R| = 65

    Parameters
    ----------
    data    : [DataFrame] DataFrame of CVSR data
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    data    : [DataFrame] DataFrame of CVSR data
    """
    # Start a timer
    tic = time.clock()

    # Only keep the year 2015
    data = data[data.index.year == 2015]

    # Get rid of the 31st, it is incomplete
    data = data[data.index.date < dt.date(2015, 12, 31)]

    # Find rotation angles greater then 65 degrees
    idx = abs(data.R.values) >= 65

    # for local maxima
    locMax = argrelextrema(data.POA[idx].values, np.greater)[0]

    # for local minima
    locMin = argrelextrema(data.POA[idx].values, np.less)[0]
    locMin = np.append(0, locMin)
    locMin = np.append(locMin, sum(idx))

    x = np.append(locMax - 1, locMin - 1)
    y = np.append(data.POA[idx].values[locMax - 1], data.GHI[idx].values[locMin - 1])
    x, y = zip(*sorted(zip(x, y)))
    f = interp1d(x, y)
    xnew = range(min(x), max(x))
    ynew = f(xnew)

    # Now I need to map them back
    data.POA[idx] = ynew

    # Stop the timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        # Show the user
        print('Interpolating for elevated rotation angeles took ' + "%.2f" % (toc - tic) +
              ' s.')

    # Return the result to the user
    return data


def poa2pow(times, POA, time_it=False):
    """
    Converts the modeled clear sky POA irradiance to power output

    Parameters
    ----------
    times   : [DatetimeIndex] local time
    POA     : [array] Plane of array global irradiance using the closure equation
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    CSM_Pow : [array] Modeled clear sky power output
    """
    # Start a timer
    tic = time.clock()

    clrDays = getCvsrClearDays()
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];

    DoY = pd.date_range('1/1/2015', periods=0, freq='D')
    for m in months:
        for d in clrDays[m][1]:
            DoY = DoY.append(pd.date_range(m + ' ' + str(d) + ' 2015',
                                           periods=1, freq='D'))

    DoY = DoY.dayofyear

    scale = np.array([300, 290, 280, 270, 265, 262, 262, 245, 245, 250, 250, 250, 250,
                      250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 245, 245, 245,
                      245, 240, 260, 250, 250, 250, 250, 250, 250, 270, 280, 275, 275,
                      280, 290, 290, 280, 280, 280, 280, 285, 285, 300, 300])

    coef = np.polyfit(DoY, scale, 4)
    DoY = times.dayofyear
    power_scale = (coef[0] * DoY ** 4 + coef[1] * DoY ** 3 + coef[2] * DoY ** 2 +
                   coef[3] * DoY ** 1 + coef[4])
    CSM_Pow = POA * power_scale
    CSM_Pow[CSM_Pow > 250000] = 250000

    # Stop the timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        # Show the user how long it took
        print('Converting POA irradiance to power took ' + "%.2f" % (toc - tic) + ' s.')

    return CSM_Pow


def plotDay(data, month=1, day=1, which='POA'):
    """
    Plot a single given day

    Parameters
    ----------
    data    : [DataFrame] DataFrame of CVSR data
    month   : [int] Month of the year to plot
    day     : [int] Day of the month to plot
    which   : [string] signifies what should be plotted

    Returns
    -------
    idx     : [DatetimeIndex] times for the day plotted
    """
    # Find the indices corresponding to the day in question
    idx = np.bitwise_and(data.index.month == month, data.index.day == day)

    # Create the figure
    plt.figure(figsize=(12, 8))

    # Decide what to plot
    if 'poa' in which.lower():
        plt.plot(data.index[idx].time, data.POA[idx])
    if 'p11' in which.lower():
        plt.plot(data.index[idx].time, data.station1_poa1[idx])
    if 'p12' in which.lower():
        plt.plot(data.index[idx].time, data.station1_poa2[idx])
    if 'p21' in which.lower():
        plt.plot(data.index[idx].time, data.station2_poa1[idx])
    if 'p31' in which.lower():
        plt.plot(data.index[idx].time, data.station3_poa1[idx])
    if 'pow' in which.lower():
        plt.plot(data.index[idx].time, data.power_output[idx])
    if 'ghi' in which.lower():
        plt.plot(data.index[idx].time, data.GHI[idx])
    if 'dni' in which.lower():
        plt.plot(data.index[idx].time, data.DNI[idx])
    if 'dhi' in which.lower():
        plt.plot(data.index[idx].time, data.DHI[idx])
    if 'zen' in which.lower():
        plt.plot(data.index[idx].time, data.zenith[idx])
    if 'am' in which.lower():
        plt.plot(data.index[idx].time, data.AM[idx])
    if 'ie' in which.lower():
        plt.plot(data.index[idx].time, data.IE[idx])
    if 'az' in which.lower():
        plt.plot(data.index[idx].time, data.azimuth[idx])
    if 'rot' in which.lower():
        plt.plot(data.index[idx].time, data.R[idx])
    if 'gam' in which.lower():
        plt.plot(data.index[idx].time, data.gamma[idx])
    if 'aoi' in which.lower():
        plt.plot(data.index[idx].time, data.AOI[idx])
    if 'dpi' in which.lower():
        plt.plot(data.index[idx].time, data.POAd[idx])
    if 'f1' in which.lower():
        plt.plot(data.index[idx].time, data.F1[idx])
    if 'f2' in which.lower():
        plt.plot(data.index[idx].time, data.F2[idx])
    if 'csp' in which.lower():
        plt.plot(data.index[idx].time, data.CSM_Pow[idx])

    # Adjust the limits
    plt.xlim([min(data.index[idx].time), max(data.index[idx].time)])
    plt.grid()

    # Place time ticks in a place that makes sense
    plt.xticks([dt.time(3), dt.time(6), dt.time(9), dt.time(12),
                dt.time(15), dt.time(18), dt.time(21)])

    # Return x-axis to the user
    return idx


def plotMonthClr(data, listClearSky, month='jan', which='POA'):
    """
    Plot all clear days in a given month

    Parameters
    ----------
    data         : [DataFrame] DataFrame of CVSR data
    listClearSky : [dict] Dictionary of clear days (see getCvsrClearDays function)
    month        : [string] Three letter abbreviation of the month
    which        : [string] signifies what should be plotted

    Returns
    -------
    Null
    """
    # set the month
    month2plot = month

    # Create the figure
    plt.figure(figsize=(12, 8))

    for d in listClearSky[month2plot][1]:
        # Find the indices corresponding to the day in question
        idx = np.bitwise_and(data.index.month == listClearSky[month2plot][0],
                             data.index.day == d)
        # Decide what to plot
        if 'poa' in which.lower():
            plt.plot(data.index[idx].time, data.POA[idx])
        if 'p11' in which.lower():
            plt.plot(data.index[idx].time, data.station1_poa1[idx])
        if 'p12' in which.lower():
            plt.plot(data.index[idx].time, data.station1_poa2[idx])
        if 'p21' in which.lower():
            plt.plot(data.index[idx].time, data.station2_poa1[idx])
        if 'p31' in which.lower():
            plt.plot(data.index[idx].time, data.station3_poa1[idx])
        if 'pow' in which.lower():
            plt.plot(data.index[idx].time, data.power_output[idx])
        if 'p21' in which.lower():
            plt.plot(data.index[idx].time, data.station2_poa1[idx])
        if 'ghi' in which.lower():
            plt.plot(data.index[idx].time, data.GHI[idx])
        if 'dni' in which.lower():
            plt.plot(data.index[idx].time, data.DNI[idx])
        if 'dhi' in which.lower():
            plt.plot(data.index[idx].time, data.DHI[idx])
        if 'zen' in which.lower():
            plt.plot(data.index[idx].time, data.zenith[idx])
        if 'am' in which.lower():
            plt.plot(data.index[idx].time, data.AM[idx])
        if 'ie' in which.lower():
            plt.plot(data.index[idx].time, data.IE[idx])
        if 'az' in which.lower():
            plt.plot(data.index[idx].time, data.azimuth[idx])
        if 'rot' in which.lower():
            plt.plot(data.index[idx].time, data.R[idx])
        if 'gam' in which.lower():
            plt.plot(data.index[idx].time, data.gamma[idx])
        if 'aoi' in which.lower():
            plt.plot(data.index[idx].time, data.AOI[idx])
        if 'dpi' in which.lower():
            plt.plot(data.index[idx].time, data.POAd[idx])
        if 'f1' in which.lower():
            plt.plot(data.index[idx].time, data.F1[idx])
        if 'f2' in which.lower():
            plt.plot(data.index[idx].time, data.F2[idx])
        if 'csp' in which.lower():
            plt.plot(data.index[idx].time, data.CSM_Pow[idx])

    # Adjust the limits
    plt.xlim([min(data.index[idx].time), max(data.index[idx].time)])
    plt.grid()

    # Place time ticks in a place that makes sense
    plt.xticks([dt.time(3), dt.time(6), dt.time(9), dt.time(12),
                dt.time(15), dt.time(18), dt.time(21)])


def getCvsrClearDays():
    """
    Constructs the dictionary of clear days for CVSR 2015

    Parameters
    ----------
    null

    Returns
    -------
    list    : [dict] Dictionary of clear days
    """
    return {'jan': [1, [1, 24]], 'feb': [2, [12]], 'mar': [3, [7, 19, 26, 29]],
            'apr': [4, [12, 15]], 'may': [5, [2]], 'jun': [5, [4]],
            'jul': [7, [5, 15, 16, 17, 26, 27, 28]], 'aug': [8, [7, 8, 10, 11, 14, 22, 24]],
            'sep': [9, [3, 7, 8, 17, 19, 20, 22]], 'oct': [10, [9, 10, 11, 21, 30, 31]],
            'nov': [11, [1, 7, 11, 14, 20, 21, 22, 23]], 'dec': [12, [17, 18, 26, 30]]}


def getAnnualRemundTurb(lat=35.33, lon=-119.91, time_it=False):
    """
    Get the annual average Linke turbidity from worldwide maps

    Parameters
    ----------
    lat     : [float] Latitude  (South is negative) in degrees
    lon     : [float] Longitude (West is negative)  in degrees
    time_it : [bool] Weather or not to time the process

    Returns
    -------
    TL      : [float] Average annual Linke turbidity from worldwide maps
    """
    # Start timer
    tic = time.clock()

    # Set up months
    Months = ('January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December')

    # Map location
    La = np.linspace(90, -90, 2160)
    Lo = np.linspace(-180, 180, 4320)
    I1 = np.argmin(abs(lat - La)) + 1
    I2 = np.argmin(abs(lon - Lo)) + 1
    Tl = np.zeros(12)

    # Get the turbidity for EVERY month from Worldwide Maps
    for m in range(12):
        fname = os.getcwd() + '/LinkeTurbidity/' + Months[int(m)] + '.tif'
        Lt = imread(fname)
        L1 = Lt[I1 - 1, I2 - 1]
        L2 = float(L1)
        Tl[int(m)] = L2 / 20.

    # Calculate annual average TL
    TL = np.mean(Tl)

    # Stop timer
    toc = time.clock()

    # If the user wanted to time it,
    if time_it == True:
        # Show the user
        print('It took ' + "%.2f" % (toc - tic) + ' s to look up monthly Linke ' +
              'turbidities and find thier annual average.')

    # Return the result to the user
    return TL


# function from PV lib to calculate panel tilt angle
def singleaxis(apparent_zenith, apparent_azimuth,
               axis_tilt=0, axis_azimuth=0, max_angle=90,
               backtrack=True, gcr=2.0/7.0):
    """
    Determine the rotation angle of a single axis tracker using the
    equations in [1] when given a particular sun zenith and azimuth
    angle. backtracking may be specified, and if so, a ground coverage
    ratio is required.

    Rotation angle is determined in a panel-oriented coordinate system.
    The tracker azimuth axis_azimuth defines the positive y-axis; the
    positive x-axis is 90 degress clockwise from the y-axis and parallel
    to the earth surface, and the positive z-axis is normal and oriented
    towards the sun. Rotation angle tracker_theta indicates tracker
    position relative to horizontal: tracker_theta = 0 is horizontal,
    and positive tracker_theta is a clockwise rotation around the y axis
    in the x, y, z coordinate system. For example, if tracker azimuth
    axis_azimuth is 180 (oriented south), tracker_theta = 30 is a
    rotation of 30 degrees towards the west, and tracker_theta = -90 is
    a rotation to the vertical plane facing east.

    Parameters
    ----------
    apparent_zenith : float, 1d array, or Series
        Solar apparent zenith angles in decimal degrees.

    apparent_azimuth : float, 1d array, or Series
        Solar apparent azimuth angles in decimal degrees.

    axis_tilt : float, default 0
        The tilt of the axis of rotation (i.e, the y-axis defined by
        axis_azimuth) with respect to horizontal, in decimal degrees.

    axis_azimuth : float, default 0
        A value denoting the compass direction along which the axis of
        rotation lies. Measured in decimal degrees East of North.

    max_angle : float, default 90
        A value denoting the maximum rotation angle, in decimal degrees,
        of the one-axis tracker from its horizontal position (horizontal
        if axis_tilt = 0). A max_angle of 90 degrees allows the tracker
        to rotate to a vertical position to point the panel towards a
        horizon. max_angle of 180 degrees allows for full rotation.

    backtrack : bool, default True
        Controls whether the tracker has the capability to "backtrack"
        to avoid row-to-row shading. False denotes no backtrack
        capability. True denotes backtrack capability.

    gcr : float, default 2.0/7.0
        A value denoting the ground coverage ratio of a tracker system
        which utilizes backtracking; i.e. the ratio between the PV array
        surface area to total ground area. A tracker system with modules
        2 meters wide, centered on the tracking axis, with 6 meters
        between the tracking axes has a gcr of 2/6=0.333. If gcr is not
        provided, a gcr of 2/7 is default. gcr must be <=1.

    Returns
    -------
    dict or DataFrame with the following columns:

    * tracker_theta: The rotation angle of the tracker.
        tracker_theta = 0 is horizontal, and positive rotation angles are
        clockwise.
    * aoi: The angle-of-incidence of direct irradiance onto the
        rotated panel surface.
    * surface_tilt: The angle between the panel surface and the earth
        surface, accounting for panel rotation.
    * surface_azimuth: The azimuth of the rotated panel, determined by
        projecting the vector normal to the panel's surface to the earth's
        surface.

    References
    ----------
    [1] Lorenzo, E et al., 2011, "Tracking and back-tracking", Prog. in
    Photovoltaics: Research and Applications, v. 19, pp. 747-753.
    """

    # MATLAB to Python conversion by
    # Will Holmgren (@wholmgren), U. Arizona. March, 2015.

    if isinstance(apparent_zenith, pd.Series):
        index = apparent_zenith.index
    else:
        index = None

    # convert scalars to arrays
    apparent_azimuth = np.atleast_1d(apparent_azimuth)
    apparent_zenith = np.atleast_1d(apparent_zenith)

    if apparent_azimuth.ndim > 1 or apparent_zenith.ndim > 1:
        raise ValueError('Input dimensions must not exceed 1')

    # Calculate sun position x, y, z using coordinate system as in [1], Eq 2.

    # Positive y axis is oriented parallel to earth surface along tracking axis
    # (for the purpose of illustration, assume y is oriented to the south);
    # positive x axis is orthogonal, 90 deg clockwise from y-axis, and parallel
    # to the earth's surface (if y axis is south, x axis is west);
    # positive z axis is normal to x, y axes, pointed upward.

    # Equations in [1] assume solar azimuth is relative to reference vector
    # pointed south, with clockwise positive.
    # Here, the input solar azimuth is degrees East of North,
    # i.e., relative to a reference vector pointed
    # north with clockwise positive.
    # Rotate sun azimuth to coordinate system as in [1]
    # to calculate sun position.

    az = apparent_azimuth - 180
    apparent_elevation = 90 - apparent_zenith
    x = cosd(apparent_elevation) * sind(az)
    y = cosd(apparent_elevation) * cosd(az)
    z = sind(apparent_elevation)

    # translate array azimuth from compass bearing to [1] coord system
    # wholmgren: strange to see axis_azimuth calculated differently from az,
    # (not that it matters, or at least it shouldn't...).
    axis_azimuth_south = axis_azimuth - 180

    # translate input array tilt angle axis_tilt to [1] coordinate system.

    # In [1] coordinates, axis_tilt is a rotation about the x-axis.
    # For a system with array azimuth (y-axis) oriented south,
    # the x-axis is oriented west, and a positive axis_tilt is a
    # counterclockwise rotation, i.e, lifting the north edge of the panel.
    # Thus, in [1] coordinate system, in the northern hemisphere a positive
    # axis_tilt indicates a rotation toward the equator,
    # whereas in the southern hemisphere rotation toward the equator is
    # indicated by axis_tilt<0.  Here, the input axis_tilt is
    # always positive and is a rotation toward the equator.

    # Calculate sun position (xp, yp, zp) in panel-oriented coordinate system:
    # positive y-axis is oriented along tracking axis at panel tilt;
    # positive x-axis is orthogonal, clockwise, parallel to earth surface;
    # positive z-axis is normal to x-y axes, pointed upward.
    # Calculate sun position (xp,yp,zp) in panel coordinates using [1] Eq 11
    # note that equation for yp (y' in Eq. 11 of Lorenzo et al 2011) is
    # corrected, after conversation with paper's authors.

    xp = x*cosd(axis_azimuth_south) - y*sind(axis_azimuth_south)
    yp = (x*cosd(axis_tilt)*sind(axis_azimuth_south) +
          y*cosd(axis_tilt)*cosd(axis_azimuth_south) -
          z*sind(axis_tilt))
    zp = (x*sind(axis_tilt)*sind(axis_azimuth_south) +
          y*sind(axis_tilt)*cosd(axis_azimuth_south) +
          z*cosd(axis_tilt))

    # The ideal tracking angle wid is the rotation to place the sun position
    # vector (xp, yp, zp) in the (y, z) plane; i.e., normal to the panel and
    # containing the axis of rotation.  wid = 0 indicates that the panel is
    # horizontal.  Here, our convention is that a clockwise rotation is
    # positive, to view rotation angles in the same frame of reference as
    # azimuth.  For example, for a system with tracking axis oriented south,
    # a rotation toward the east is negative, and a rotation to the west is
    # positive.

    # Use arctan2 and avoid the tmp corrections.

    # angle from x-y plane to projection of sun vector onto x-z plane
#     tmp = np.degrees(np.arctan(zp/xp))

    # Obtain wid by translating tmp to convention for rotation angles.
    # Have to account for which quadrant of the x-z plane in which the sun
    # vector lies.  Complete solution here but probably not necessary to
    # consider QIII and QIV.
#     wid = pd.Series(index=times)
#     wid[(xp>=0) & (zp>=0)] =  90 - tmp[(xp>=0) & (zp>=0)]  # QI
#     wid[(xp<0)  & (zp>=0)] = -90 - tmp[(xp<0)  & (zp>=0)]  # QII
#     wid[(xp<0)  & (zp<0)]  = -90 - tmp[(xp<0)  & (zp<0)]   # QIII
#     wid[(xp>=0) & (zp<0)]  =  90 - tmp[(xp>=0) & (zp<0)]   # QIV

    # Calculate angle from x-y plane to projection of sun vector onto x-z plane
    # and then obtain wid by translating tmp to convention for rotation angles.
    wid = 90 - np.degrees(np.arctan2(zp, xp))

    # filter for sun above panel horizon
    zen_gt_90 = apparent_zenith > 90
    wid[zen_gt_90] = np.nan

    # Account for backtracking; modified from [1] to account for rotation
    # angle convention being used here.
    if backtrack:
        axes_distance = 1/gcr
        temp = np.minimum(axes_distance*cosd(wid), 1)

        # backtrack angle
        # (always positive b/c acosd returns values between 0 and 180)
        wc = np.degrees(np.arccos(temp))

        # Eq 4 applied when wid in QIV (wid < 0 evalulates True), QI
        tracker_theta = np.where(wid < 0, wid + wc, wid - wc)
    else:
        tracker_theta = wid

    tracker_theta[tracker_theta > max_angle] = max_angle
    tracker_theta[tracker_theta < -max_angle] = -max_angle

    # calculate panel normal vector in panel-oriented x, y, z coordinates.
    # y-axis is axis of tracker rotation.  tracker_theta is a compass angle
    # (clockwise is positive) rather than a trigonometric angle.
    # the *0 is a trick to preserve NaN values.
    panel_norm = np.array([sind(tracker_theta),
                           tracker_theta*0,
                           cosd(tracker_theta)])

    # sun position in vector format in panel-oriented x, y, z coordinates
    sun_vec = np.array([xp, yp, zp])

    # calculate angle-of-incidence on panel
    aoi = np.degrees(np.arccos(np.abs(np.sum(sun_vec*panel_norm, axis=0))))

    # calculate panel tilt and azimuth
    # in a coordinate system where the panel tilt is the
    # angle from horizontal, and the panel azimuth is
    # the compass angle (clockwise from north) to the projection
    # of the panel's normal to the earth's surface.
    # These outputs are provided for convenience and comparison
    # with other PV software which use these angle conventions.

    # project normal vector to earth surface.
    # First rotate about x-axis by angle -axis_tilt so that y-axis is
    # also parallel to earth surface, then project.

    # Calculate standard rotation matrix
    rot_x = np.array([[1, 0, 0],
                      [0, cosd(-axis_tilt), -sind(-axis_tilt)],
                      [0, sind(-axis_tilt), cosd(-axis_tilt)]])

    # panel_norm_earth contains the normal vector
    # expressed in earth-surface coordinates
    # (z normal to surface, y aligned with tracker axis parallel to earth)
    panel_norm_earth = np.dot(rot_x, panel_norm).T

    # projection to plane tangent to earth surface,
    # in earth surface coordinates
    projected_normal = np.array([panel_norm_earth[:, 0],
                                 panel_norm_earth[:, 1],
                                 panel_norm_earth[:, 2]*0]).T

    # calculate vector magnitudes
    projected_normal_mag = np.sqrt(np.nansum(projected_normal**2, axis=1))

    # renormalize the projected vector
    # avoid creating nan values.
    non_zeros = projected_normal_mag != 0
    projected_normal[non_zeros] = (projected_normal[non_zeros].T /
                                   projected_normal_mag[non_zeros]).T

    # calculation of surface_azimuth
    # 1. Find the angle.
#     surface_azimuth = pd.Series(
#         np.degrees(np.arctan(projected_normal[:,1]/projected_normal[:,0])),
#                                 index=times)
    surface_azimuth = \
        np.degrees(np.arctan2(projected_normal[:, 1], projected_normal[:, 0]))

    # 2. Clean up atan when x-coord or y-coord is zero
#     surface_azimuth[(projected_normal[:,0]==0) & (projected_normal[:,1]>0)] =  90
#     surface_azimuth[(projected_normal[:,0]==0) & (projected_normal[:,1]<0)] =  -90
#     surface_azimuth[(projected_normal[:,1]==0) & (projected_normal[:,0]>0)] =  0
#     surface_azimuth[(projected_normal[:,1]==0) & (projected_normal[:,0]<0)] = 180

    # 3. Correct atan for QII and QIII
#     surface_azimuth[(projected_normal[:,0]<0) & (projected_normal[:,1]>0)] += 180 # QII
#     surface_azimuth[(projected_normal[:,0]<0) & (projected_normal[:,1]<0)] += 180 # QIII

    # 4. Skip to below

    # at this point surface_azimuth contains angles between -90 and +270,
    # where 0 is along the positive x-axis,
    # the y-axis is in the direction of the tracker azimuth,
    # and positive angles are rotations from the positive x axis towards
    # the positive y-axis.
    # Adjust to compass angles
    # (clockwise rotation from 0 along the positive y-axis)
#    surface_azimuth[surface_azimuth<=90] = 90 - surface_azimuth[surface_azimuth<=90]
#    surface_azimuth[surface_azimuth>90] = 450 - surface_azimuth[surface_azimuth>90]

    # finally rotate to align y-axis with true north
    # PVLIB_MATLAB has this latitude correction,
    # but I don't think it's latitude dependent if you always
    # specify axis_azimuth with respect to North.
#     if latitude > 0 or True:
#         surface_azimuth = surface_azimuth - axis_azimuth
#     else:
#         surface_azimuth = surface_azimuth - axis_azimuth - 180
#     surface_azimuth[surface_azimuth<0] = 360 + surface_azimuth[surface_azimuth<0]

    # the commented code above is mostly part of PVLIB_MATLAB.
    # My (wholmgren) take is that it can be done more simply.
    # Say that we're pointing along the postive x axis (likely west).
    # We just need to rotate 90 degrees to get from the x axis
    # to the y axis (likely south),
    # and then add the axis_azimuth to get back to North.
    # Anything left over is the azimuth that we want,
    # and we can map it into the [0,360) domain.

    # 4. Rotate 0 reference from panel's x axis to it's y axis and
    #    then back to North.
    surface_azimuth = 90 - surface_azimuth + axis_azimuth

    # 5. Map azimuth into [0,360) domain.
    surface_azimuth[surface_azimuth < 0] += 360
    surface_azimuth[surface_azimuth >= 360] -= 360

    # Calculate surface_tilt
    dotproduct = (panel_norm_earth * projected_normal).sum(axis=1)
    surface_tilt = 90 - np.degrees(np.arccos(dotproduct))

    # Bundle DataFrame for return values and filter for sun below horizon.
    #out = {'tracker_theta': tracker_theta, 'aoi': aoi,
           #'surface_azimuth': surface_azimuth, 'surface_tilt': surface_tilt}
    #if index is not None:
        #out = pd.DataFrame(out, index=index)
        #out = out[['tracker_theta', 'aoi', 'surface_azimuth', 'surface_tilt']]
        #out[zen_gt_90] = np.nan
    #else:
        #out = {k: np.where(zen_gt_90, np.nan, v) for k, v in out.items()}

    return surface_tilt,surface_azimuth
