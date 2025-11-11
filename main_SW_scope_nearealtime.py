"""Post-process model outputs, and compare OLR and DLW."""

import os
import pandas as pd
import numpy as np
from scipy import interpolate
from itertools import chain
from multiprocessing import Pool
import re
import h5py
import time
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from SCOPE_func import *
from LBL_funcs_fullSpectrum import *
from fun_nearealtime_RTM import *
from scipy.optimize import curve_fit

def load_clearsky(site, df, tz='UTC'):
    import pvlib
    lat, lon, alt = 40.05192, -88.37309, 213
    df1 = df.set_index('Time', inplace=False)
    location = pvlib.location.Location(lat, lon, tz, alt)
    clearsky_data = location.get_clearsky(df1.index)#.resample('5min').mean()
    ghi = clearsky_data['ghi']
    dni = clearsky_data['dni']
    return ghi, dni

def nearealtime_process_satellite(figlabel, site, phase, file_dir=None, sky="day", N_bundles = 10000):
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"
    Sat_dir = Sat_preprocess(file_dir, site, figlabel, sky, phase, sat='FY4A')
    if sky=='clearsky':
        sat = pd.read_hdf(
        os.path.join(file_dir + Sat_dir,
                     "{}_radiance_satellite_{}.h5".format(site, timeofday, sky)),
                    'df'
                    )
    else:
        sat = pd.read_hdf(os.path.join(file_dir + Sat_dir,
                     f"{site}_radiance_satellite_{sky}.h5"),'df')

    #sat['COD_pre'] = sat.apply(Rad_to_Flux_sug_COD, axis=1)
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    #sat[channels] = sat.apply(Rad_to_Flux_sug_COD, axis=1)
    sat[channels] = sat.apply(Ref_to_Flux_LUT, axis=1)
    # reflectance to Flux [W/m2]
    return sat


def loss_function(Rc_real, Rc_rtm):
    SE = (Rc_real - Rc_rtm) ** 2 # Square Error
    MSE = SE.mean() # Mean Square Error
    RMSE = np.sqrt(MSE)
    return SE, MSE, RMSE


def compute_gradient_simple(Rc_rtm_df, COD_list, Rc_real, Costs, EDU=True):
    """
    Linear fitting of COD, based on two nearest neighbors of Rc_real.
    """
    n = Rc_rtm_df.shape[1]  # number
    sorted_indices = np.argsort(Costs)  # ascendent order
    nearest_indices = sorted_indices[:2]
    # Sort nearest indices to ensure consistent order
    # nearest_indices.sort()
    Rc1, Rc2 = Rc_rtm_df.iloc[nearest_indices[0]], Rc_rtm_df.iloc[nearest_indices[1]]
    COD1, COD2 = COD_list[nearest_indices[0]], COD_list[nearest_indices[1]]
    #eT = np.exp(-COD2)
    # Compute the gradient using the two nearest points
    ## method 1 : take w as the comprehensive gradient for comprehensive X
    if EDU == True:
        w = (COD2 - COD1) / np.linalg.norm(Rc2 - Rc1)  # np.sum(Rc2 - Rc1)
        w = 1 / np.linalg.norm(Rc2)
        dx = np.linalg.norm(Rc_real.values - Rc_rtm_df.iloc[nearest_indices[0]].values)
    else:
        w = (COD2 - COD1) / (Rc2 - Rc1).sum()
        dx = (Rc_real.values - Rc_rtm_df.iloc[nearest_indices[0]].values).sum(axis=1)

    COD_guess = COD1 + w * dx  # the positive and negative cancelled
    print('point1', COD1, 'point2', COD2, 'COD_guess', float(COD_guess))

    # Extrapolation check
    max_COD = max(COD1, COD2)
    # avoid repeat value
    condition = np.abs(np.array(COD_list) - COD_guess) <= 1
    # Check extrapolation condition
    is_extrapolation = COD_guess > max_COD or (np.sum(condition) >= 1 and len(COD_list) > 2)
    if is_extrapolation:
        print("Extrapolation triggered")
        x = Rc_rtm_df.sum(axis=1).values # distance from the 0 vector
        y = np.array(COD_list)
        # Safe fitting with bounds on w
        try:
            popt, _ = curve_fit(expol_func, x, y, bounds=(1e-8, 10)) # bound for a in expol_func
            a_fit = popt[0]
            x_r = Rc_real.sum(axis=1).values
            COD_guess = expol_func(x_r, a_fit)
            condition = np.abs(np.array(COD_list) - COD_guess) <= 1
            print('Extrap triggered COD = ', COD_guess)
            if COD_guess>10 and np.sum(condition) >= 2: # COD>20, when testing.
                lr = len(COD_list)  # lr is estimated by number of COD_list
                print(f"Exponential fitted step too slow for 2 times, lr = {lr} is used")
                dj_dw = np.sum(Rc_rtm_df.iloc[nearest_indices[0]].values - Rc_real.values) * COD1
                w = w - lr * 0.5 * dj_dw / n  # Adjust w to avoid extrapolation
                # w = w -lr * dj_dw /2*n is the defination of GD.
                dx = np.sum(Rc_real.values - Rc_rtm_df.iloc[nearest_indices[0]].values)
                COD_guess = COD1 + w * dx  # fallback
                print('Condition 2 COD = ', COD_guess)
        except Exception as e:# not exit when testing.
            lr = len(COD_list)  # lr is estimated by number of COD_list
            print(f"Exponential fit failed: {e}, lr = {lr} is used")
            dj_dw = np.sum(Rc_rtm_df.iloc[nearest_indices[0]].values - Rc_real.values) * COD1
            w = w - lr * 0.5 * dj_dw / n  # Adjust w to avoid extrapolation
            # w = w -lr *dj_dw /2*n is the defination of GD.
            dx = np.sum(Rc_real.values - Rc_rtm_df.iloc[nearest_indices[0]].values)
            COD_guess = COD1 + w * dx  # fallback
            print('Exception COD = ',COD_guess)
    return float(COD_guess)

def compute_gradient_eT(Rc_rtm_df, COD_list, Rc_real, Costs, C, EDU=True):
    """
    Linear fitting of COD, based on two nearest neighbors of Rc_real.
    """
    n = Rc_rtm_df.shape[1]  # number
    sorted_indices = np.argsort(Costs)  # ascendent order
    nearest_indices = sorted_indices[:2]
    # Sort nearest indices to ensure consistent order
    # nearest_indices.sort()
    
    Rc0 = Rc_rtm_df.iloc[nearest_indices[0]]
    COD0 = COD_list[nearest_indices[0]]
    #eT = np.exp(-COD2)
    # Compute the gradient using the two nearest points
    ## method 1 : take w as the comprehensive gradient for comprehensive X
    if EDU == True:
        f1 = 1 / (C - np.linalg.norm(Rc0))
        dx = np.linalg.norm(Rc_real.values - Rc_rtm_df.iloc[nearest_indices[0]].values)
    else:
        f1 = 1 / (C - Rc0.sum())
        dx = (Rc_real.values - Rc_rtm_df.iloc[nearest_indices[0]].values).sum(axis=1)
        
    COD_guess = COD0 + f1 * dx  # the positive and negative cancelled
    print('point1', COD0, 'COD_guess', float(COD_guess),'f1',f1,'dx',dx)

    # learning rate
    lr = 1  # lr is estimated by number of COD_list
    dj_dw = gradient_function(COD_list, Rc_rtm_df, C)
    C = C + lr * 0.5 * dj_dw / n  # Adjust w to avoid extrapolation
    return COD_guess, C

def gradient_function(x, y_sat, C):
    """
    Cost = 1/m * sum(  f(x_i) - y_i_sat)^2  )
    f(x_i) = C - exp(-x_i)
    df(x_i)_dC = 1
    dCost_dC = 1/m * sum( f(x_i) - y_i_sat) )

    Parameters
    ----------
    x : 1D COD
    y_sat : 2D pd.DataFrame, 6 channels
    C : float scalar

    Returns
    -------

    """
    diff = 0
    for i in range(len(x)):
        xi = x[i]
        yi_sat = y_sat.iloc[i].values
        y_pred = C-np.exp(-xi)
        diff += np.linalg.norm(yi_sat) - y_pred
    dj_dw = diff/ len(x)  # Average gradient
    return dj_dw


def get_uw_radiance(sat, whoseCOD='COD_pre', file_dir='./GOES_data/'):
    results_list = []
    for i in range(sat.shape[0]):
        Sun_Zen = sat['th0'].iloc[i]
        T_a = sat['Ta'].iloc[i]
        RH = sat['rh'].iloc[i]*100
        local_zen = sat.get('Local_Ze', sat.get('local_Zen')).iloc[i]
        rela_azi = sat['rela_azi'].iloc[i]
        COD = sat[whoseCOD].iloc[i]
        result = nearealtime_LUT(Sun_Zen, local_zen, rela_azi, COD, T_a, RH,
                                 file_dir=file_dir, bandmode='GOES')
        results_list.append(result[0])
    return results_list


def compare_oswr(site, sourcefile, sky="clear", file_dir=None,figlabel=None):
    """
    Compare ref_OSWR of satellite and model.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    file_dir : path, optional
        The directory containing the SURFRAD + satellite data files.

    Returns
    -------
    None
    """

    # not used for current version
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"
    # open Satellite observation data
    file_path = file_dir + 'retrived_results/'+ f'{sourcefile}'
    sat = pd.read_csv(file_path)

    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    sat_rad = sat[channels] # radiance measured by GOES ABI 16, well preprocessed.
    goes_channel = [f'goes_{ch}' for ch in channels]
    rtm_columns = [f'rtm_{ch}' for ch in channels]
    savefile = './GOES_validation/' + f"Result_BON_{sky}_radiance_satellite_water_{figlabel}.csv"
    try:
        sat = pd.read_csv(savefile)
        sat_rad = sat[channels]  # radiance measured by GOES ABI 16, well preprocessed.
        goes_rad = pd.read_csv(savefile)[goes_channel]
        rtm_rad = pd.read_csv(savefile)[rtm_columns]
        print('UW Rad read from existing csv file:', savefile)
    except Exception:
        print(f"Run GOES COD output radiance.")
        if sky != "clear":
            sat[rtm_columns] = get_uw_radiance(sat, 'COD_pre', file_dir=file_dir)
        sat[goes_channel] = get_uw_radiance(sat, 'COD', file_dir=file_dir)
        sat.to_csv(savefile, index=False)

    VAR = 'Rad'
    for CODfromWhom in ['GOES']: #RTM_clear']: #, 'RTM'
        if CODfromWhom == 'GOES':
            goes_rad = pd.read_csv(savefile)[goes_channel]
            goes_rad.columns = [col.replace('goes_', '') for col in goes_rad.columns]
            plot_data(sat_rad, goes_rad, channels, VAR, CODfromWhom, figlabel)
        elif CODfromWhom == 'RTM':
            rtm_rad = pd.read_csv(savefile)[rtm_columns]
            rtm_rad.columns = [col.replace('rtm_', '') for col in rtm_rad.columns]
            plot_data(sat_rad, rtm_rad, channels, VAR, CODfromWhom, figlabel)
        elif CODfromWhom == 'RTM_clear':
            goes_rad.columns = [col.replace('goes_', '') for col in goes_rad.columns]
            plot_data(sat_rad, goes_rad, channels, VAR, CODfromWhom, figlabel)

def compare_dsw(site, sourcefile, sky="clear", file_dir=None, figlabel=None):
    """Compare the ground and modeled DLW.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    file_dir : path, optional
        The directory containing the SURFRAD + satellite data files.

    Returns
    -------
    None

    """
    # not used for current version
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"
    # open Satellite observation data
    file_path = file_dir + 'retrived_results/' + sourcefile
    sat = pd.read_csv(file_path)
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    site_GHI = sat['Site_dsw']  # should not be sat[dw_ir], it includes the surface reflected radiation
    site_DNI = sat['direct_n']

    csvfile = './GOES_validation/' +  f"Result_{site}_{sky}_radiance_satellite_water_{figlabel}.csv"
    rtm_dsw, rtm_dni, rtm_dhi, goes_dsw, goes_dni, goes_dhi = [], [], [], [], [], []
    try:
        df_combined = pd.read_csv(csvfile)
        goes_DNI, goes_GHI = df_combined['goes_dni'], df_combined['goes_dsw']
        if sky != "clear":
            rtm_DNI, rtm_GHI = df_combined['rtm_dni'], df_combined['rtm_dsw']
        print('dsw read from existing csv file:', csvfile)
    except Exception:
        print(f"Run GOES COD output DSW.")
        for i in range(sat.shape[0]):
            Sun_Zen, local_zen, rela_azi = sat['th0'][i], sat['local_Zen'][i], sat['rela_azi'][i]
            COD_goes = sat['COD'][i]  # Assuming COD is a column in sat_rad
            try:
                if sky != "clear":
                    COD_rtm = sat['COD_rtm_6c'][i]
                T_a, RH = sat['temp'][i], sat['rh'][i]
            except Exception:
                if sky != "clear":
                    COD_rtm = sat['COD_pre'][i]
                T_a, RH = sat['Ta'][i], sat['rh'][i]
            if sky != "clear":
                dsw, dni, dhi = get_RTM_dsw(Sun_Zen, COD_rtm, T_a, RH)
                rtm_dsw.append(dsw)
                rtm_dni.append(dni)
                rtm_dhi.append(dhi)

            dsw, dni, dhi = get_RTM_dsw(Sun_Zen, COD_goes, T_a, RH)
            goes_dsw.append(dsw)
            goes_dni.append(dni)
            goes_dhi.append(dhi)

            df_new = pd.DataFrame({
                'rtm_dsw': rtm_dsw,
                'rtm_dni': rtm_dni,
                'rtm_dhi': rtm_dhi,
                'goes_dsw': goes_dsw,
                'goes_dni': goes_dni,
                'goes_dhi': goes_dhi
            })
            df_combined = pd.concat([sat, df_new], axis=1)
            # GET REAL DNI
            cos_t = np.cos(np.deg2rad(df_combined['th0']))
            df_combined['rtm_dni'] = df_combined['rtm_dni'] / cos_t
            df_combined['goes_dni'] = df_combined['goes_dni'] / cos_t
            df_combined.to_csv(csvfile, index=False)
            # Get for plot
            goes_DNI, goes_GHI = df_combined['goes_dni'], df_combined['goes_dsw']
            if sky != "clear":
                rtm_DNI, rtm_GHI = df_combined['rtm_dni'], df_combined['rtm_dsw']
    nsrdb_ghi, nsrdb_dni = load_NSRDB(site, df_combined, sky_type=sky, year='2019')
    print(figlabel)
    for CODfromwho in ['NSRDB']:#['GOES','RTM']: #'GOES','RTM',
        if CODfromwho == 'GOES':
            plot_data_dw(site_GHI, goes_GHI, site_DNI, goes_DNI, CODfromwho, figlabel=figlabel)
        elif CODfromwho == 'RTM':
            plot_data_dw(site_GHI, rtm_GHI, site_DNI, rtm_DNI, CODfromwho, figlabel=figlabel)
        elif CODfromwho == 'RTM_clear':
            # COD = 0, no goes_channel is rtm_channel.
            plot_data_dw(site_GHI, goes_GHI, site_DNI, goes_DNI, CODfromwho, figlabel=figlabel)
        elif CODfromwho == 'NSRDB':
            plot_data_dw(site_GHI, nsrdb_ghi, site_DNI, nsrdb_dni, CODfromwho, figlabel=figlabel)

def compare_clear_dsw(site, sourcefile, sky="clear", file_dir=None, figlabel=None):
    """Compare the ground and modeled DLW.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    file_dir : path, optional
        The directory containing the SURFRAD + satellite data files.

    Returns
    -------
    None

    """
    # not used for current version
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    csvfile = ('./GOES_validation/' + f"Result_{timeofday}_BON_radiance_satellite_{figlabel}_{sky}.csv")
               #f"Result_day_BON_radiance_satellite_a_clearsky.csv")

    rtm_dsw, rtm_dni, rtm_dhi = [], [], []
    try:
        df_combined = pd.read_csv(csvfile)

        df_combined['Time'] = pd.to_datetime(df_combined['Time'])
        # df2 = pd.read_csv('./GOES_validation/' + 'Result_day_BON_radiance_satellite_July13Z70_clearsky.csv')
        # df2['Time'] = pd.to_datetime(df2['Time'])
        # df2['rtm_dni'] = df2['rtm_dni']/ np.cos(np.deg2rad(df2['Site_zen'])) # fill NaN with 0
        # df_combined = df_combined.merge(
        #                 df2[['Time', 'rtm_dni']].rename(columns={'rtm_dni': 'rtm_dni_airm'}),
        #                 on='Time',
        #                 how='left')
        # target_day = pd.to_datetime('2019-07-12').date()
        # df_combined = df_combined[df_combined['Time'].dt.date == target_day]
        try:
            df_combined['rtm_dni_1HG'] = df_combined['rtm_dni_1HG']/np.cos(np.deg2rad(df_combined['Site_zen']))
        except Exception:
            pass
        rtm_GHI, rtm_DNI = df_combined['rtm_dsw'],df_combined['rtm_dni']
        site_GHI, site_DNI = df_combined['Site_dsw'], df_combined['direct_n']
        nsrdb_ghi, nsrdb_dni = load_NSRDB(site, df_combined,sky_type=sky, year='2019')
        pvlib_ghi, pvlib_dni = load_clearsky(site, df_combined)
        print('dsw read from existing csv file:', csvfile)
        plot_day(df_combined, figlabel=None)
    except Exception:
        # open Satellite observation data
        file_path = file_dir + 'GOES16_site_sat_data/' + sourcefile
        sat = pd.read_csv(file_path)
        site_GHI = sat['Site_dsw']  # should not be sat[dw_ir], it includes the surface reflected radiation
        site_DNI = sat['direct_n']
        print(f"Run RTM to get output DSW.")
        from aod_codes import read_aod
        # aod time series from surfrad
        df_aod = read_aod(site)
        # concate previous SCOPE results & aod series
        sat['Time'] = pd.to_datetime(sat['Time'])
        sat = sat.set_index('Time')
        c_index = [index for index in sat.index if index in df_aod.index]
        sat = pd.concat([sat.loc[c_index], df_aod.loc[c_index]], join='inner', axis=1)
        print('# of sat:', sat.shape[0])
        for i in range(sat.shape[0]):
            Sun_Zen, local_zen, rela_azi = sat['Sun_Zen'][i], sat['local_Zen'][i], sat['rela_azi'][i]
            COD_goes = 0  # Assuming COD is a column in sat_rad
            T_a, RH = sat['temp'].iloc[i], sat['rh'].iloc[i]
            AOD = sat['aod'].iloc[i]  # Assuming AOD is a column in sat_rad
            # AOD = 0.1243 #None
            if Sun_Zen>60 or RH == np.nan:
                dsw, dni, dhi = np.nan, np.nan, np.nan
            else:
                if T_a <200:
                    T_a = T_a + 273.15
                dsw, dni, dhi = get_RTM_dsw(Sun_Zen, COD_goes, T_a, RH, AOD)
            rtm_dsw.append(dsw)
            rtm_dni.append(dni)
            rtm_dhi.append(dhi)

            df_new = pd.DataFrame({
                'rtm_dsw': rtm_dsw,
                'rtm_dni': rtm_dni,
                'rtm_dhi': rtm_dhi,
            })
        sat = sat.reset_index()
        sat = sat.rename(columns={"index": "Time"})
        df_combined = pd.concat([sat, df_new], axis=1)
        df_combined['rtm_dni'] = df_combined['rtm_dni']/np.cos(np.deg2rad(df_combined['Site_zen']))
        df_combined.to_csv(csvfile, index=False)
        rtm_DNI, rtm_GHI = df_combined['rtm_dni'], df_combined['rtm_dsw']

    # CODfromwho = 'RTM_clear'
    # # COD = 0, no rtm_channel is rtm_channel.
    # plot_data_dw(site_GHI, rtm_GHI, site_DNI, rtm_DNI, CODfromwho, figlabel=figlabel)
    # CODfromwho = 'NSRDB_clear'
    # plot_data_dw(site_GHI, nsrdb_ghi, site_DNI, nsrdb_dni, CODfromwho, figlabel=figlabel)
    # CODfromwho = 'pvlib_clear'
    # plot_data_dw(site_GHI, pvlib_ghi, site_DNI, pvlib_dni, CODfromwho, figlabel=figlabel)

def load_NSRDB(site, df, sky_type='clear', year = '2019'):
    nsrdb_dir = './data/NSRDB/nsrdb_site_files'
    df_site = pd.read_hdf(
        os.path.join(nsrdb_dir,
                     "nsrdb_{}_{}.h5".format(year,site)),
        'data'
    )
    df['Time'] = pd.to_datetime(df['Time'])
    df_site['Time'] = df_site['Time'].dt.tz_localize(None)

    df_site.set_index('Time', inplace=True)
    df_site = df_site.select_dtypes(include=[np.number]).resample("5min", label="right").mean()
    df_site.reset_index(inplace=True)

    if sky_type == 'clear':
        sky_type = 'clearsky_'
    else:
        sky_type = ''
    df_combined = pd.merge(
        df,
        df_site[['Time', sky_type+'ghi', sky_type+'dni']].rename(columns={sky_type+'ghi': 'nsrdb_ghi', sky_type+'dni': 'nsrdb_dni'}),
        #df_site[['Time', 'ghi', 'dni']].rename(columns={'ghi': 'nsrdb_ghi', 'dni': 'nsrdb_dni'}),
        on='Time',
        how='left'
    )
    return df_combined['nsrdb_ghi'], df_combined['nsrdb_dni']


def plot_NSRDB(site, plotwho, sky="clear", file_dir=None, figlabel=None):
    """
    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    file_dir : path, optional
        The directory containing the SURFRAD + satellite data files.

    Returns
    -------
    None

    """
    # not used for current version
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"
    csvfile = './GOES_data/flux/nsrdb/' + f"Result_{site}_{sky}_radiance_satellite_water_{figlabel}.csv"

    df_combined = pd.read_csv(csvfile)
    df_combined = df_combined[df_combined['Site_dsw'] != -9999.9]
    df_combined = df_combined[df_combined['direct_n'] != -9999.9]
    site_GHI = df_combined['Site_dsw']  # should not be sat[dw_ir], it includes the surface reflected radiation
    site_DNI = df_combined['direct_n']
    if plotwho == 'NSRDB':
        plot_ghi, plot_dni = load_NSRDB(site, df_combined, sky_type=sky, year='2019')
        df_combined['nsrdb_ghi'] = plot_ghi
        df_combined['nsrdb_dni'] = plot_dni
        csvfile = './GOES_data/flux/nsrdb/' + f"Result_{site}_{sky}_radiance_satellite_water_{figlabel}.csv"
        df_combined.to_csv(csvfile, index=False)
    elif plotwho == 'Surrogate':
        plot_ghi = df_combined['y_pred']
        plot_dni = df_combined['y_pred']
    COD = df_combined['COD']
    print(figlabel)
    for CODfromwho in [plotwho]:  # ['GOES','RTM']: #'GOES','RTM',
        plot_data_dw(site_GHI, plot_ghi, site_DNI, plot_dni, CODfromwho, COD, site, figlabel=figlabel)
    return None

def plot_day(df_combined, figlabel=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo

    # Ensure Time is datetime and localized
    df_combined.set_index('Time', inplace=True)
    df_combined.index = pd.to_datetime(df_combined.index)
    df_combined.index = df_combined.index.tz_localize('UTC').tz_convert(ZoneInfo('America/Chicago'))

    #  First figure: absolute values
    # fig, ax = plt.subplots(figsize=(8, 5))
    # df_combined['Site_dsw'].plot(ax=ax, color='C0')
    # df_combined['rtm_dsw'].plot(ax=ax, color='C0', linestyle='', marker='^', markersize=3)
    # try:
    #     df_combined['rtm_dsw_aodc'].plot(ax=ax, color='C0', linestyle='--')
    # except Exception:
    #     pass
    #
    # df_combined['direct_n'].plot(ax=ax, color='C1')
    # df_combined['rtm_dni'].plot(ax=ax, color='C1', linestyle='', marker='^', markersize=3)
    # try:
    #     df_combined['rtm_dni_aodc'].plot(ax=ax, color='C1', linestyle='--')
    # except Exception:
    #     pass
    #
    # df_combined['diffuse'].plot(ax=ax, color='C2')
    # df_combined['rtm_dhi'].plot(ax=ax, color='C2', linestyle='', marker='^', markersize=3)
    # try:
    #     df_combined['rtm_dhi_1HG'].plot(ax=ax, color='C2', linestyle='--')
    # except Exception:
    #     pass
    # #
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Radiation [W/m²]')
    # ax.set_title('Radiation Comparison')
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H', tz=ZoneInfo('America/Chicago')))
    # plt.tight_layout()
    # plt.show()

    # 🎯 Second figure: differences (RTM - Site)
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    # DSW differences
    (df_combined['rtm_dsw'] - df_combined['Site_dsw']).plot(ax=ax2, color='C0', linestyle='-', label='DSW(RTM) - site')
    # (df_combined['rtm_dsw_aodc'] - df_combined['Site_dsw']).plot(ax=ax2, color='C0', linestyle='--',
    #                                                              label='DSW - RTM (aodc)')
    # DNI differences
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    (df_combined['rtm_dni'] - df_combined['direct_n']).plot(ax=ax2, color='C1', linestyle='-', label='DNI(rtm) - site')
    #(df_combined['rtm_dni_airm'] - df_combined['direct_n']).plot(ax=ax2, color='C2', linestyle='-', label='DNI(rtm_am) - site')
    # (df_combined['rtm_dni_1HG'] - df_combined['direct_n']).plot(ax=ax2, color='C2', linestyle='--',
    #                                                             label='DNI(1HG) - site')

    # Optional: DHI differences (commented out)
    # (df_combined['rtm_dhi'] - df_combined['diffuse']).plot(ax=ax2, color='C4', linestyle='-', label='DHI - RTM')
    # (df_combined['rtm_dhi_1HG'] - df_combined['diffuse']).plot(ax=ax2, color='C4', linestyle='--', label='DHI - RTM (1HG)')

    # 📈 Add second y-axis for solar zenith angle
    ax2b = ax2.twinx()
    df_combined['aod'].plot(ax=ax2b, color='gray', linestyle=':', label='AOD')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Difference [W/m²]')
    ax2b.set_ylabel('AOD', color='gray')
    ax2b.tick_params(axis='y', labelcolor='gray')

    # Legend (merge from both axes)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center',ncols=4)#, bbox_to_anchor=(1, 0.5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H', tz=ZoneInfo('America/Chicago')))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #for timeofday in ["day"]:
    file_dir = './'
    spectral = 'SW'
    phase = 'water' #'water' clearsky
    N_bundles = 1000
    figlabel = ['test'] #['COD<10','COD>20','COD>10'] # COD>20  July13
        # preprocess_model_oswr(file_dir=file_dir+testfile, timeofday=timeofday)   # OSW Radiance [W/m^2/sr]
        # preprocess_model_dsw(file_dir=file_dir+testfile, timeofday=timeofday)   # DLW [W/m^2]

    # compare model against measured data
    # file_dir = os.path.join("data")      # directory with SURFRAD+GOES data
    # for sky in ["clear", "cloudy", "day", "night"]:
    # for sky in ["cloudy", "day"]:
    sites = pd.read_csv(file_dir+'FY4A_data/'+"CERN_info.csv", header=0, index_col=False, names=['site', 'lon', 'lat', 'elev'])
    sites=sites.values.tolist()

    # for figlabel in figlabels:
    #     if phase == 'clearsky':
    #         filename = f"GOES_day_BON_radiance_satellite_{figlabel}_clearsky"  #"GOES_day_BON_radiance_satellite_a_clearsky"#
    #         sky = "clearsky"
    #         compare_clear_dsw("BON", f"GOES_day_BON_radiance_satellite_{figlabel}_{sky}.csv",
    #                            sky=sky, file_dir=file_dir, figlabel=figlabel)
    #     else:
    for site, lat, lon, elev in sites:
        print(site)
        for sky in ["day"]: # clearsky,day
            sat = nearealtime_process_satellite(figlabel, site, phase, file_dir=file_dir, sky=sky, N_bundles = N_bundles)
            sat.to_csv(file_dir+'flux/'+f"Result_{site}_{sky}_radiance_satellite_{phase}_{figlabel}.csv", index=False)
            # sites = ["BON", "DRA", "FPK", "GWN", "PSU", "SXF", "TBL"]
            sourcefile = f"Result_BON_{sky}_radiance_satellite_{phase}_{figlabel}.csv"
            #compare_oswr(site, sourcefile, sky=sky, file_dir=file_dir, figlabel=figlabel)
            # compare_dsw(site, sourcefile, sky=sky, file_dir=file_dir, figlabel=figlabel)
            plotwho = 'Surrogate'#'NSRDB'
            plot_NSRDB(site, plotwho, sky=sky, file_dir=file_dir, figlabel=figlabel)