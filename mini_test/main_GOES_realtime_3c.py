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


def nearealtime_process_satellite(data_dir=None, sky="day", N_bundles = 10000):
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"
    Sat_dir = Sat_preprocess(data_dir, sat='GOES16')
    sat = pd.read_hdf(
        os.path.join(data_dir + Sat_dir,
                     "GOES_{}_BON_radiance_satellite_June_COD>10_water_{}.h5".format(sky, timeofday)),
        'df'
    )
    channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    # fake rtm COD>10
    save_path = './GOES_validation/2019_June_water/'
    sat_ref = pd.read_hdf(os.path.join(save_path, f'Rc_rtm_df_2019_June.h5'), key='df')
    rtm_columns = [f'rtm_{ch}' for ch in channels]
    # COD>20
    #sat_ref = pd.read_csv("Rc_rtm_df_COD>10.csv")
    #rtm_columns = [f'rtm_{ch}' for ch in channels]
    for col in rtm_columns:
        sat[col] = 0
    # try:
    #     sat_ref = sat[channels]
    # except KeyError:
    #     sat_ref = sat['Radiance']

    max_iterations = 5
    # test 0.0006 is a little bit large, most converge less than 0.0001
    epsilon = 0.0001  # n * 0,0001 , V_magnitude 1 * 0.01(1%), single SE = 0.0001,
    max_COD, min_COD = 50, 0

    for i in range(sat_ref.shape[0]):
        print("--" * 20)
        print(f'Column {i} of {sat_ref.shape[0]}')
        print(f"{'Iteration':<10} {'COD':<10} {'MSE':<10} {'RMSE':<10} {'Cost':<10}")
        Sun_Zen = sat['Sun_Zen'][i]
        T_a = sat['temp'][i] + 273.15
        RH = sat['rh'][i]
        local_zen = sat.get('Local_Ze', sat.get('local_Zen'))[i]
        rela_azi = sat['rela_azi'][i]
        Rc_real = min_max_nor(sat_ref.iloc[i].to_frame().T)  # sat_ref.iloc[i], sys=sys)

        Rc_rtm_df = pd.DataFrame()
        Rc_rtm_rad_df = pd.DataFrame()
        RMSE_dic, COD_list = [], [10, 20]
        Costs = []

        for COD_guess in COD_list:
            Rc_rtm_radiance=nearealtime_RTM(Sun_Zen, local_zen, rela_azi, COD_guess, T_a, RH,
                            channels, file_dir=data_dir, bandmode='GOES', N_bundles=N_bundles)
            Rc_rtm_rad_df = pd.concat([Rc_rtm_rad_df, Rc_rtm_radiance], ignore_index=True)

            Rc_rtm = min_max_nor(Rc_rtm_radiance)
            Rc_rtm_df = pd.concat([Rc_rtm_df, Rc_rtm], ignore_index=True)
            SE, MSE, RMSE = loss_function(Rc_real[['C01','C02','C03']].values, Rc_rtm[['C01','C02','C03']].values)
            cost = np.sum(SE) / 6  # 2*n_channels=12
            RMSE_dic.append(RMSE)
            Costs.append(cost)

            print(f"{'--':<10} {COD_guess:<10.2f}{MSE:<10.4f} {RMSE:<10.4f} {cost:<10.4f}")
            if cost < epsilon:
                sat.at[i, 'COD_rtm_6c'] = COD_guess
                break

        else:  # for else loop
            COD_guess = compute_gradient_simple(Rc_rtm_df[['C01','C02','C03']], COD_list,
                                                Rc_real[['C01','C02','C03']], Costs)

            for j in range(max_iterations):
                if COD_guess > max_COD or COD_guess < min_COD or j == max_iterations:
                    COD_guess = COD_list[np.argmin(RMSE_dic)]
                    sat.at[i, 'COD_rtm_6c'] = COD_guess
                    print('larer, smaller or max iterations, break')
                    break
                COD_list.append(COD_guess)
                Rc_rtm_radiance = nearealtime_RTM(Sun_Zen, local_zen, rela_azi, COD_guess, T_a, RH,
                                                  channels, file_dir=data_dir, bandmode='GOES', N_bundles=N_bundles)
                Rc_rtm_rad_df = pd.concat([Rc_rtm_rad_df, Rc_rtm_radiance], ignore_index=True)

                Rc_rtm = min_max_nor(Rc_rtm_radiance)
                Rc_rtm_df = pd.concat([Rc_rtm_df, Rc_rtm], ignore_index=True)
                SE, MSE, RMSE = loss_function(Rc_real[['C01','C02','C03']].values, Rc_rtm[['C01','C02','C03']].values)
                cost = np.sum(SE) / 6
                RMSE_dic.append(RMSE)
                Costs.append(cost)
                print(f"Iter {j:<5} {COD_guess:<10.2f}{MSE:<10.4f} {RMSE:<10.4f} {cost:<10.4f}")
                if cost < epsilon:
                    sat.at[i, 'COD_rtm_6c'] = COD_guess
                    break
                COD_guess = compute_gradient_simple(Rc_rtm_df[['C01','C02','C03']], COD_list,
                                                    Rc_real[['C01','C02','C03']], Costs)


        print(f"Final COD: {COD_guess:.4f}")
        for channel in channels:
            rtm_channel = 'rtm_'+channel
            sat.at[i, rtm_channel] = Rc_rtm_rad_df[channel].values[-1]
    return sat

def loss_function(Rc_real, Rc_rtm):
    SE = (Rc_real - Rc_rtm) ** 2 # Square Error
    MSE = SE.mean(axis=1) # Mean Square Error
    RMSE = np.sqrt(MSE) # Root Mean Square Error
    return SE, MSE[0], RMSE[0]

def cost_function(Rc_real, Rc_rtm):
    error = Rc_real - Rc_rtm
    mse = (error ** 2).to_numpy().mean()
    rmse = np.sqrt(mse)
    return (1 / 12) * rmse



def compute_gradient_simple(Rc_rtm_df, COD_list, Rc_real, Costs, lr=6):
    """
    Linear fitting of COD, based on two nearest neighbors of Rc_real.
    """
    n = Rc_rtm_df.shape[1]  # number of channels
    sorted_indices = np.argsort(Costs)  # ascendent order
    nearest_indices = sorted_indices[:2]
    # Sort nearest indices to ensure consistent order
    # nearest_indices.sort()
    Rc1, Rc2 = Rc_rtm_df.iloc[nearest_indices[0]], Rc_rtm_df.iloc[nearest_indices[1]]
    COD1, COD2 = COD_list[nearest_indices[0]], COD_list[nearest_indices[1]]

    # Compute the gradient using the two nearest points
    ## method 1 : take w as the comprehensive gradient for comprehensive X
    w = (COD2 - COD1) / np.linalg.norm(Rc2 - Rc1)  # np.sum(Rc2 - Rc1)
    dx = np.linalg.norm(Rc_real.values - Rc_rtm_df.iloc[nearest_indices[0]].values)
    COD_guess = COD1 + w * dx  # the positive and negative cancelled
    ## method 2 dot product np.dot(w,dx)
    # w = (COD2 - COD1) / (Rc2 - Rc1)
    # dx = Rc_real - Rc_rtm_df.iloc[nearest_indices[0]]
    # COD_guess = COD1 + np.dot(w, dx.T)
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
            if np.sum(condition) >= 2: # not exit, when testing.
                lr = len(COD_list)  # lr is estimated by number of COD_list
                print(f"Exponential fitted step too slow for 2 times, lr = {lr} is used")
                dj_dw = np.sum(Rc_rtm_df.iloc[nearest_indices[0]].values - Rc_real.values) * COD1
                w = w - lr * 0.5 * dj_dw / n  # Adjust w to avoid extrapolation
                # w = w -lr *dj_dw /2*n is the defination of GD.
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

def compare_oswr(site, sourcefile, sky="clear", data_dir=None):
    """
    Compare ref_OSWR of satellite and model.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
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
    file_path = data_dir + 'retrived_results/'+ f'{sourcefile}'
    sat = pd.read_csv(file_path)
    #print(sat.head())
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    sat_rad = sat[channels] # radiance measured by GOES ABI 16, well preprocessed.
    rtm_columns = [f'rtm_{ch}' for ch in channels]
    rtm_rad = sat[rtm_columns]  # radiance simulated by RTM

    goes_rad = pd.DataFrame(columns=channels)
    for i in range(sat_rad.shape[0]):
        Sun_Zen, local_zen, rela_azi = sat['Sun_Zen'][i], sat['local_Zen'][i], sat['rela_azi'][i]
        COD_goes = sat['COD'][i]  # Assuming COD is a column in sat_rad
        T_a, RH = sat['temp'][i], sat['rh'][i]
        N_bundles = 10000
        goes_single_rad = run_GOES_in_RTM(Sun_Zen, local_zen, rela_azi, COD_goes, T_a, RH,
                                          channels, file_dir=data_dir, bandmode='GOES', N_bundles=N_bundles)
        goes_rad = pd.concat([goes_rad, goes_single_rad], ignore_index=True)
    VAR = 'Rad'
    for CODfromWhom in ['GOES','RTM']:
        if CODfromWhom == 'GOES':
            plot_data(sat_rad, goes_rad, channels, VAR, CODfromWhom)
        else:
            rtm_rad.columns = [col.replace('rtm_', '') for col in rtm_rad.columns]
            plot_data(sat_rad, rtm_rad, channels, VAR, CODfromWhom)
    # save goes_rad to after sat
    goes_channel = [f'goes_{ch}' for ch in channels]
    sat[goes_channel] = goes_rad
    sat.to_csv('./GOES_validation/' + f"Result_RTM_{sky}_radiance_satellite_COD>10_water.csv",
               index=False)

def compare_dsw(site, sourcefile, sky="clear", file_dir=None):
    """Compare the ground and modeled DLW.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
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

    rtm_dsw, rtm_dni, rtm_dhi, goes_dsw, goes_dni, goes_dhi = [], [], [], [], [], []
    for i in range(sat.shape[0]):
        Sun_Zen, local_zen, rela_azi = sat['Sun_Zen'][i], sat['local_Zen'][i], sat['rela_azi'][i]
        COD_goes = sat['COD'][i]  # Assuming COD is a column in sat_rad
        COD_rtm = sat['COD_rtm_6c'][i]
        T_a, RH = sat['temp'][i], sat['rh'][i]

        dsw, dni, dhi = get_RTM_dsw(Sun_Zen, COD_goes, T_a, RH)
        rtm_dsw.append(dsw)
        rtm_dni.append(dni)
        rtm_dhi.append(dhi)

        dsw, dni, dhi = get_RTM_dsw(Sun_Zen, COD_rtm, T_a, RH)
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
        df_combined.to_csv('./GOES_validation/' + f"Result_RTM_{sky}_radiance_satellite_COD>10_water.csv",
                    index=False)

    goes_DNI, goes_GHI = df_combined['goes_dni'], df_combined['goes_dsw']
    rtm_DNI, rtm_GHI = df_combined['rtm_dni'], df_combined['rtm_dsw']
    for CODfromwho in ['GOES','RTM']:
        if CODfromwho == 'GOES':
            plot_data_dw(site_GHI, goes_GHI, site_DNI, goes_DNI, CODfromwho)
        else:
            plot_data_dw(site_GHI, rtm_GHI, site_DNI, rtm_DNI, CODfromwho)


if __name__ == "__main__":
    for timeofday in ["day"]:
        file_dir = './GOES_data/'  #'./lut_test_file/'
        spectral = 'SW'
        phase = 'water'
        COD_boundary = 10
        N_bundles = 10000
        #preprocess_model_oswr(file_dir=file_dir+testfile, timeofday=timeofday)   # O SW Radiance [W/m^2/sr]
        #preprocess_model_dsw(file_dir=file_dir+testfile, timeofday=timeofday)   # DLW [W/m^2]

    # compare model against measured data
    # file_dir = os.path.join("data")      # directory with SURFRAD+GOES data
    # for sky in ["clear", "cloudy", "day", "night"]:
    # for sky in ["cloudy", "day"]:
    for sky in ["day"]:
        sat = nearealtime_process_satellite(data_dir=file_dir, sky=sky, N_bundles = N_bundles)
        sat.to_csv(file_dir+'retrived_results/'+f"Result_RTM_{sky}_radiance_satellite_COD>{COD_boundary}_{phase}.csv", index=False)
        # sites = ["BON", "DRA", "FPK", "GWN", "PSU", "SXF", "TBL"]
        sites = ["BON"]
        for site in sites:
            print(sky, site)
            sourcefile = f"Result_RTM_day_radiance_satellite_COD>{COD_boundary}_water.csv"
            compare_oswr(site, sourcefile, sky=sky, data_dir=file_dir)
            compare_dsw(site, sourcefile, sky=sky, file_dir=file_dir)

            # estimate the cloud optical properties
                #postprocess_olr(site, sky='day', file_dir=file_dir)
                #postprocess_dlw(site, sky=sky)