"""Pre-process GOES-R ABI products/ SURFRAD data."""

import pandas as pd
import glob
import numpy as np
import pvlib
import os
from scipy import interpolate
from multiprocessing import Pool
from pandas.core.indexes.base import Index


def read_site(file):
    n_pixels = 11 * 11
    names = ["ts_created", "ts_start", "ts_end", "channel"]
    names += ["x{}".format(i) for i in range(n_pixels)]
    names += ["kappa0"]

    df = pd.read_csv(file, names=names)
    df = df.drop("kappa0", axis=1)

    df["ts_created"] = [x[:10] + ' ' + x[11:19] for x in df['ts_created'].values]

    return df


def merge_site(site, channel):
    filenames = glob.glob('data/goes/GOES/cropped_c{}/*{}*.csv'.format(channel, site))
    df = pd.concat([read_site(filename) for filename in filenames])
    df["ts_created"] = pd.to_datetime(df["ts_created"])
    df = df.sort_values(by="ts_created")
    n_pixels = 11 * 11
    df = df.set_index("ts_created")
    col = "x{}".format((n_pixels - 1) // 2)
    df = df[["channel", col]]
    df = df.rename(columns={col: "C{}".format(channel.zfill(2))})

    # convert radiance from [mW/(m^2 sr cm^-1)] to [W/(m^2 sr cm^-1)]
    df["C{}".format(channel.zfill(2))] /= 1e3

    # round up to the nearest 5-minute timestamp
    df = df.resample("5min", label="right").mean()

    df = df.reset_index()


    return df

def process_site(site, channels):
    df = merge_site(site, channels[0])
    for channel in channels[1:]:
        df_ = merge_site(site, channel)
        df = pd.merge(left=df, right=df_, how='outer', on='ts_created')


    # drop NAN values
    df = df.dropna()

    df = df.set_index("ts_created")

    df.to_hdf("data/goes/{}_radiance.h5".format(site), "df", mode="w")
    print(site, df.shape, df.index[0], df.index[-1])


def preprocess_dat(filename):

    f = open(filename, 'r')

    # save processed data as CSV
    filename_base = filename.split('/')[-1].split('.dat')[0]
    new_filename = 'F:/GOES16/RAD_SURFRAD/{}.csv'.format(filename_base)
    f_new = open(new_filename, 'w')

    # deal with non-standard delimiters in original file
    for line in f:
        raw = list(filter(None, line.strip('\n').split(' ')))
        if len(raw) == 48:
            new_line = ','.join([s for s in raw])
            f_new.write(new_line + '\n')

    f.close()
    f_new.close()


def preprocess_csv(filename):
    '''Pre-process CSV files.'''

    # column names
    names = [
        'year', 'jday', 'month', 'day', 'hour', 'min', 'dt',
        'zenith',
        'ghi', 'qc_ghi',            # downwelling = GHI [W/m^2]
        'uw_solar', 'qc_uwsolar',   # upwelling
        'dni', 'qc_dni',            # DNI [W/m^2]
        'dhi', 'qc_dhi',            # DHI [W/m^2]
        'dw_ir', 'qc_dwir',
        'dw_casetemp', 'qc_dwcasetemp',
        'dw_dometemp', 'qc_dwdometemp',
        'uw_ir', 'qc_uwir',
        'uw_casetemp', 'qc_uwcasetemp',
        'uw_dometemp', 'qc_uwdometemp',
        'uvb', 'qc_uvb',
        'par', 'qc_par',
        'netsolar', 'qc_netsolar',
        'netir', 'qc_netir',
        'totalnet', 'qc_totalnet',
        'temp', 'qc_temp',
        'rh', 'qc_rh',
        'windspd', 'qc_windspd',
        'winddir', 'qc_winddir',
        'pressure', 'qc_pressure'
    ]

    df = pd.read_csv(filename, delimiter=',', names=names)

    # generate timestamps from time columns
    ts_str = []
    for index, row in df.iterrows():
        # yyyy-mm-dd HH:MM:SS (e.g. 2018-04-01 17:23:30)
        ts_str.append('{:04d}-{:02d}-{:02d} {:02d}:{:02d}:00'.format(
            int(row['year']),
            int(row['month']),
            int(row['day']),
            int(row['hour']),
            int(row['min']))
        )

    df.insert(df.shape[1], 'timestamp', pd.DatetimeIndex(ts_str))
    df = df.set_index('timestamp')

    # return DataFrame
    return df


def preprocess_ground(site, year):
    # pre-process .dat file
    # filenames = glob.glob("F:/GOES16/RAD_SURFRAD/{}/{}/*.dat".format(site.lower(), str(year)))
    # pool = Pool()
    # pool.map(preprocess_dat, filenames)
    # pool.close()

    # combine CSV files
    filenames = glob.glob("./SURFRAD/{}*{}*.csv".format(str(year), site.lower()))
    df = pd.concat([preprocess_csv(filename) for filename in filenames])

    df = df.sort_index()

    # remove "bad" data
    df.loc[df['qc_ghi'] != 0, 'ghi'] = np.nan
    df.loc[df['qc_dni'] != 0, 'dni'] = np.nan
    df.loc[df['qc_dhi'] != 0, 'dhi'] = np.nan
    df.loc[df['qc_dwir'] != 0, 'dw_ir'] = np.nan
    df.loc[df['qc_uwir'] != 0, 'uw_ir'] = np.nan
    df.loc[df['qc_temp'] != 0, 'temp'] = np.nan
    df.loc[df['qc_rh'] != 0, 'rh'] = np.nan

    # only keep essential columns
    # - ghi: GHI (global) [W/m^2]
    # - dni: DNI (direct) [W/m^2]
    # - dhi: DHI (diffuse) [W/m^2
    # - zenith: zenith angle [deg]
    # - uw_ir: upwelling IR [W/m^2]
    # - dw_ir: downwelling IR [W/m^2]
    # - temp: air temperature at 10m [deg C]
    # - rh: relative humidity (RH) [%]
    #
    df = df[['ghi', 'dni', 'dhi', 'zenith', 'dw_ir', 'uw_ir', 'temp', 'rh']]


    # ambient temperature (T_a [K])
    df.insert(df.shape[1], 'T_a', df['temp'] + 273.15)

    # estimate surface temperature (T_s [K]) from upwelling IR (using the
    # Stefan-Boltzmann law):
    #
    #   IR = sigma T^4  ==>  T = (IR/sigma)^(1/4)
    #
    sigma = 5.670373e-8   # [W/(m^2 K^4)]
    IR = df['uw_ir']
    T_s = (IR / sigma / 1.0) ** 0.25   # emissivity=1.0 (blackbody)
    df.insert(df.shape[1], 'T_s', T_s)

    # export results
    df.to_hdf('./SURFRAD/{}_ground.h5'.format(site), 'df')
    print("{}: {}".format(site, df.shape))



def preprocess_clearsky_periods(year, site, lat, lon, alt):
    """
    Find the clearsky periods.

    Parameters
    ----------
    site : str
        SURFRAD station name.
    lat, lon, alt : float
        The latitude, longitude, and altitude of the site.

    """

    # the load the pre-processed ground data
    files = glob.glob('./SURFRAD/{}*{}*.csv'.format(str(year), site.lower()))
    df = pd.read_csv(files[0])


    # resample
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df = df.resample("5min", closed="right", label="right").mean()
    df = df.dropna(how="any")


    # split into day/night using solar angle
    zenith_threshold = 85  # zenith threshold [deg] for day/night split
    df_night = df[df["zen"] > zenith_threshold]
    #df_night.to_hdf("data/surfrad/{}_night.h5".format(site), "df", mode="w")
    print("Night:", df_night.shape, df_night.index[0], df_night.index[-1])
    df = df[df["zen"] <= zenith_threshold]

    # add clearsky irradiance
    loc = pvlib.location.Location(lat, lon, altitude=alt)
    cs = loc.get_clearsky(df.index)
    df.insert(df.shape[1], 'ghi_clear', cs["ghi"])
    df.insert(df.shape[1], 'dni_clear', cs["dni"])

    # GHI and DNI threshold values: G, M, L, sig, S
    #thresholds = [75, 75, 10, 0.005, 8]      # Reno's 2016 paper
    thresholds = [100, 100, 50, 0.01, 10.0]  # Rich's paper

    # rolling window
    i_window = 30  # window size [mins]
    criteria_sum = np.zeros(df.shape[0])
    for i in range(i_window, df.shape[0], i_window):

        # backwards selection: (i_start, i_end]
        i_start = i - i_window + 1
        i_end = i + 1
        ghi = df.iloc[i_start:i_end]['dw_solar'].values
        ghi_clear = df.iloc[i_start:i_end]['ghi_clear'].values

        # clearsky criteria
        criteria = np.zeros(5)

        # average irradiance
        criteria[0] = (np.abs(np.nanmean(ghi) - np.nanmean(ghi_clear)) < thresholds[0])

        # max irradiance
        criteria[1] = np.abs(np.max(ghi) - np.max(ghi_clear)) < thresholds[1]

        # irradiance increment
        diff_ghi = np.diff(ghi)
        diff_ghi_clear = np.diff(ghi_clear)
        criteria[2] = (
            np.abs(np.sum(np.sqrt(diff_ghi ** 2)) - np.sum(np.sqrt(diff_ghi_clear ** 2)))
            < thresholds[2]
        )

        # std of irradiance increment
        criteria[3] = np.abs(
            np.std(diff_ghi) / np.mean(ghi)
            - np.std(diff_ghi_clear) / np.mean(ghi_clear)
        ) < thresholds[3]

        # max irradiance increment
        criteria[4] = np.max(np.abs(diff_ghi - diff_ghi_clear)) < thresholds[4]

        # sum of the criteria (5=all criteria met)
        criteria_sum[i_start:i_end] = criteria.sum()

    # if all criteria are met, then the sky is clear
    df.insert(df.shape[1], 'clearsky', criteria_sum == 5)
    df.reset_index(inplace=True)
    # export data
    df.to_hdf(f"./SURFRAD/preprocessed/{site}_day.h5", "df", mode="w")
    df[df['clearsky'] == True].to_hdf('./SURFRAD/preprocessed/{}_clear.h5'.format(site), 'df', mode="w",)
    df[df['clearsky'] == False].to_hdf('./SURFRAD/preprocessed/{}_cloudy.h5'.format(site), 'df', mode="w")
    print(site, df.shape, df.index[0], df.index[-1])
    n_clear = df[df["clearsky"] == True].shape[0]
    n_cloudy = df[df["clearsky"] == False].shape[0]
    n_day = df.shape[0]
    print("clear = {:>6,d} ({:>3.1%}), cloudy = {:>12,d} ({:>3.1%})".format(n_clear, n_clear / n_day, n_cloudy, n_cloudy / n_day))
    #print("Clear: ", df[df['clearsky'] == True].shape)
    #print("Cloudy:", df[df['clearsky'] == False].shape)
    print(df.columns)



def eppley_dlw(filename):
    """DLW radiances that would be seen by Eppley PIR (4-50 um)."""

    # wavenumber resolution [cm^-1]
    dnu = 0.10

    df = pd.read_hdf(filename, "df")
    T = df["T"].values[0]
    T_delta = df["T_delta"].values[0]
    RH = df["RH"].values[0]
    COD = df["COD"].values[0]
    kap_top = df["kap_top"].values[0]
    kap_bottom = df["kap_bottom"].values[0]
    kap = "{}-{}".format(kap_bottom, kap_top)
    AOD = df["AOD"].values[0]

    # downwelling at the surface or layer i [W/(m^2 cm^-1)]
    radiance = df["dw_surface"].values
    nu = df.index.values  # wavenumber [cm^-1]

    # remove NaNs
    idx = ~np.isnan(radiance)
    radiance = radiance[idx]
    nu = nu[idx]

    # integrate downwelling radiance:
    # - PIR: 4 to 50 um (200 to 2500 cm^-1)
    #nu_min, nu_max = 200.0, 2500.00  # spectral range: 4-50 um
    nu_min, nu_max = 200.0, 3333.33  # spectral range: 3-50 um (SURFRAD)
    #nu_min, nu_max = 200.0, 2857.14  # spectral range: 3.5-50 um (ARM SGP C1)
    idx = (nu >= nu_min) & (nu <= nu_max)
    dw_pir = radiance[idx]
    dwir = np.trapz(np.sort(dw_pir), dx=dnu)
    results = [{
        'RH': RH, 'T': T, "T_delta": T_delta,
        "COD": COD, "AOD": AOD,
        "kap": kap, "kap_top": kap_top, "kap_bottom": kap_bottom,
        'dwir': dwir,
    }]

    return results


def partial_pressure(T, RH):
    """
    Partial pressure of water vapor (P_w).

    Parameters
    ----------
    T : float or array_like
        Temperature [K].
    RH : float or array_like
        Relative humidity [%].

    Returns
    -------
    P_s : array_like
        Saturated water vapor pressure [Pa].
    P_w : array_like
        Partial pressure of water vapor [Pa].

    References
    ----------
    Mengying's papers

    """

    # calculate saturated water vapor pressure (P_s [Pa]) using
    # August-Roch-Magnus (ARM) expression (following Eq. 5 in Li et al. 2018)
    P_s = 610.94 * np.exp((17.625 * (T - 273.15)) / (T - 30.11))
    P_w = (RH / 100.0) * P_s
    return P_s, P_w


def combine_datasets(site, sky="clear"):

    # GOES-R ABI spectral radiance [W/(m^2 sr cm^-1)]
    goes = pd.read_hdf('data/goes/{}_radiance.h5'.format(site), 'df')


    # SURFRAD ground data
    surfrad = pd.read_hdf('data/surfrad/{}_{}.h5'.format(site, sky), 'df')
    surfrad = surfrad.resample('5min', closed='right', label='right').mean()
    surfrad = surfrad.dropna(how="any")

    # merge datasets
    df = pd.merge(goes, surfrad, left_index=True, right_index=True, how='inner')


    # convert units
    df['temp'] += 273.15  # temperature [K]

    # partial pressure (P_w [Pa])
    P_s, P_w = partial_pressure(df['temp'], df['rh'])
    df.insert(df.shape[1], 'P_w', P_w)

    # rename variables
    df = df.rename(columns={'temp': 'T', 'rh': 'RH'})

    ## remove night values
    #df = df[df['zenith'] <= 85]

    #print(df.dropna(how="any").shape)

    # export
    df = df.sort_index()
    df.to_hdf('data/{}_processed_{}.h5'.format(site, sky), 'df', mode="w")
    print(site, sky, df.shape[0], df.index[0], df.index[-1])


def channel_radiance_satellite(site, sky="clear"):
    """
    Compute radiance [W/m^2] for each channel.
    """

    # calculate radiance [W/m^2] for each channel
    df = pd.read_hdf('data/{}_processed_{}.h5'.format(site, sky), 'df')
    df = df.dropna(how='any')

    # iterate through the channels
    channels = ['C{:02d}'.format(i) for i in range(7, 16 + 1)]

    # assume T_surface = T_ambient
    T = df["T"].values


    for channel in channels:
        #print(channel)

        # load ABI calibration data
        channel_number = int(channel[-2:])
        filename = 'abi_calibration/GOES-R_ABI_PFM_SRF_CWG_ch{}.txt'.format(channel_number)
        data = np.genfromtxt(filename, delimiter=' ' * 6, skip_header=2)
        nu = data[:, 1]  # wavenumber [cm^-1]
        rel_srf = data[:, 2]  # relative SRF [-]

        # reverse the order (to be increasing in wavenumber)
        nu = nu[::-1]
        rel_srf = rel_srf[::-1]
        #print("Wavenumber [cm^-1]: {:.1f}, {:.1f}".format(nu.min(), nu.max()))

        #==================================================
        # radiance per channel
        #==================================================

        # get band equivalent width = Eqw = \int_{nu_1}^{nu_2} R_{nu} dnu
        Eqw = np.trapz(rel_srf, x=nu)

        # get the channel radiance [W/(m^2 sr)]
        spectral_rad = df[channel].values   # spectral radiance [W/(m^2 sr cm^-1)]
        rad = spectral_rad * Eqw * np.pi           # radiance [W/m^2]
        df.insert(df.shape[1], '{}_satellite'.format(channel), rad)

        #==================================================
        # blackbody radiation via Planck's distribution
        #==================================================

        # compute Planck distribution per channel
        h = 6.6261e-34      # Planck's constant [J s]
        kB = 1.3806485e-23  # Boltzmann constant [J / K]
        c = 299792458       # speed of light [m / s]
        C1 = 2 * h * c ** 2
        C2 = h * c / kB

        # convert nu (m,) and T (n,) to (m, n) arrays
        # - enables vectorized computaiton
        m = len(nu)
        n = len(T)
        nu_matrix = np.tile(nu, (n, 1)).T  # (m, n): repeat along columns
        T_matrix = np.tile(T, (m, 1))      # (m, n): repeat along rows
        nu_matrix *= 100.0   # convert from [cm^-1] to [m^-1]

        # blackbody emission
        Eb_nu = C1 * (nu_matrix ** 3) / (np.exp(C2 * nu_matrix / T_matrix) - 1)
        Eb_nu *= 100  # convert to [W/(m^2 sr cm^-1)]
        Eb_nu *= np.pi  # solid angle => [W/(m^2 cm^-1)]

        # integrate over wavenumbers => (m,) values (one per T) [W/m^2]
        Eb = np.trapz(Eb_nu, x=nu, axis=0)
        df.insert(df.shape[1], "{}_blackbody".format(channel), Eb)

    # export
    df = df.sort_index()
    #print(df.head())
    df.to_hdf('data/{}_radiance_satellite_{}.h5'.format(site, sky), 'df', mode="w")
    print("{} ({}): {}, {}, {}".format(site, sky, df.shape[0], df.index[0], df.index[-1]))

if __name__ == "__main__":
    dnu = 3 #0.10
    n_layers = 54 #32
    AOD = 0.1243
    year = 2019

    sites = [
        ["BON", 40.05192, -88.37309, 213],
        # ["DRA", 36.62373, -116.01947, 1007],
        # ["FPK", 48.30783, -105.10170, 634],
        # ["GWN", 34.25470, -89.87290, 98],
        # ["PSU", 40.72012, -77.93085, 376],
        # ["SXF", 43.73403, -96.62328, 473],
        # ["TBL", 40.12498, -105.23680, 1689],
    ]

    channels = ['{}'.format(i) for i in range(7, 16 + 1)]

    for site, lat, lon, alt in sites:
        print(site)

        # preprocess GOES data
        #process_site(site, channels)

        # preprocess surfrad data
        # preprocess_ground(site, year=2019)
        preprocess_clearsky_periods(year,site, lat, lon, alt)

        # for sky in ['night']:  # ["day", "clear", "cloudy", 'night']
        #     print(sky)
        #     # combine surfrad data with satellite data
        #     combine_datasets(site, sky=sky)  # satellite in [W/(m^2 sr cm^-1)]
        #     channel_radiance_satellite(site, sky=sky)  # satellite in [W/m^2]



