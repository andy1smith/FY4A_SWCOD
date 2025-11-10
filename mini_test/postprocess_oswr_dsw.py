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

def preprocess_model_oswr(data_dir=None, timeofday="day"):
    """
    Processing: OSWR (Outgoing Shortwave Radiance) [W/m2/sr]
    Save model data as reflectance (not correction yet) of OSWR
    for next step: given sat observation, extract model ref_OSWR.
    """
    # dnu = 3
    # n_layers = 54

    Fls = os.listdir(data_dir) # Raidaition W/m2
    Fls = [f for f in Fls if f.endswith('.npy')]
    Fls = [f for f in Fls if f.startswith('Results')]
    print("# files:", len(Fls))

    # parallel process files
    # pool = Pool()
    # pool.starmap(channel_oswr, [(data_dir, f) for f in Fls])
    # pool.close()
    # extract channel result, convert Intensity, *RSF, convert to reflectance
    for f in Fls:
        channel_oswr(data_dir, f)
    ## results = list(chain(*results))
    print('Model OSWR results saved')

    # print("OSWR:", df.shape)
    # print(df.columns)
    # print("kap:", sorted(df["kap"]))
    # print(sorted(df["kap_top"].unique()))
    # print(sorted(df["kap_bottom"].unique()))
    # print("TPW:", sorted(df["TPW"]))
    return None


def channel_oswr(data_dir, filename):
    """
    1. Convert Channel Outgoing Shortwave Radiation [W/m^2] to Intensity [W/m^2/sr].
    2. Multiplation of Spec Response func with Intensity
    3. to reflectance (not correction yet).
    4. save each file' results to h5 file.
    output : dict of results for each channel.
    """
    dnu = 3
    pattern = r'Results_case2_TWP=([0-9.]+)_AOD=([0-9.]+)_COD=([0-9.]+)_kap=\[([0-9,\s]+)\]_th0=([0-9.]+).npy'
    match = re.search(pattern, filename)
    if match:
        TPW, AOD, COD, kap_str, th0 = match.groups()
        TPW, AOD, COD, th0 = map(float, [TPW, AOD, COD, th0])
        kap = list(map(int, kap_str.split(',')))
        kap_top, kap_bottom = kap[-1], kap[0]
    else:
        print("No match:", filename)
    data = np.load('data/computed/F_dw_os_SW.npz')
    nu, F_dw_os = data['nu'], data['f_dw_os']
    del data
    # Mono upwelling calculation
    # toa = np.load(data_dir +'/'+filename,  allow_pickle=True).item().get('F_uw')
    # Mono Intensity calculation
    uw_rxyz_file = f"uwxyzr_COD={COD}_th0={int(th0)}_TWP={TPW}.npy"
    file_path = os.path.join(data_dir, uw_rxyz_file)
    try:
        results = np.load(file_path, allow_pickle=True).item()
    except(Exception):
        COD = np.round(COD,3)
        uw_rxyz_file = f"uwxyzr_COD={COD}_th0={int(th0)}_TWP={TPW}.npy"
        file_path = os.path.join(data_dir, uw_rxyz_file)
        results = np.load(file_path, allow_pickle=True).item()
    uw_rxyz_M = results.get('uw_rxyz_M')

    channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    results = []

    for channel in channels:
        # load calibration data : Spectral Response Func
        channel_number = int(channel[-2:])
        dirpath = './lut_test_file/FY4Aparameter/'
        FY4A_channel_srf = os.path.join(
                    dirpath,
                    'FY4A_AGRI_SRF_ch{}_20180306.txt'.format(channel_number)
                    )
        calibration = np.genfromtxt(FY4A_channel_srf, delimiter='\t', skip_header=1)
        calibration_wl = calibration[:, 0]  # wavelength [nm]
        calibration_nu = 1e7/calibration_wl # cm-1
        calibration_srf = calibration[:, 1]/100  # relative SRF [-]

        # reverse order (so wavenumber is increasing)
        calibration_nu = calibration_nu[::-1]
        calibration_srf = calibration_srf[::-1]

        # interpolate calibration to match model
        srf = np.interp(nu, calibration_nu, calibration_srf)
        F_dw_os_SRF = np.multiply(F_dw_os, srf)
        # integrate spectral radiance over the channel
        # Channal 2D radiance [W/m2/sr]
        nu_idx = np.where((nu >= calibration_nu.min()) & (nu <= calibration_nu.max()))[0]
        OSWR_channel = cal_mono_Intensity(uw_rxyz_M, nu_idx, th0, nu, F_dw_os_SRF,
                                    is_flux=False, Norm=False, dirc='UW')
        F_dw_os_channal = np.trapz(F_dw_os[nu_idx], nu[nu_idx])
        # need sun-earth distance correction, will be done in extract_sta_oswr
        ref_OSWR_channel = OSWR_channel * np.pi / F_dw_os_channal # reflectance
        results.append({'TPW': TPW, "COD": COD,
                        "kap": kap, "kap_top": kap_top, "kap_bottom": kap_bottom,
                        "AOD": AOD, "channel": channel, 'OSWR': OSWR_channel,
                        "ref_OSWR_channel": ref_OSWR_channel,
                        })
    # export by h5py, since pandas does not support nested dataframes for 2D Radiacen matrix
    output_dir = './lut_test_file/preprocess_model/'
    fl = output_dir+filename[:-4] + '_dnu={:.2f}_oswr_{}.h5'.format(dnu, timeofday)
    with h5py.File(fl, 'w') as h5file:
        for result in results:
            channel = result['channel']
            group = h5file.create_group(channel)
            for key, value in result.items():
                if key != 'channel':
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        group.create_dataset(key, data=value)
                    else:
                        group.attrs[key] = value
    return None


def preprocess_model_dsw(data_dir=None, timeofday="day"):
    """
    Process downwelling shortwave model results.

    Output : downwelling irradiance [W/m^2].
    """

    dnu = 3
    n_layers = 54
    Fls = os.listdir(data_dir)
    Fls = [f for f in Fls if f.endswith('.npy')]
    Fls = [f for f in Fls if f.startswith('Results')]
    print("# files:", len(Fls))

    # parallel process files
    pool = Pool()
    results = pool.starmap(model_dsw, [(data_dir, f) for f in Fls])
    pool.close()
    results = list(chain(*results))  # flatten: list(list(dict)) ==> list(dict)
    # results = model_dsw(data_dir, Fls[0])
    # export
    df = pd.DataFrame(results)
    df = df.sort_index()

    df['kap'] = df['kap'].astype(str)
    df['kap_top'] = df['kap_top'].astype(str)
    df['kap_bottom'] = df['kap_bottom'].astype(str)
    output_dir = "./lut_test_file/preprocess_model/"
    df.to_hdf(
        path_or_buf = output_dir + "results_case2_{}layers_dnu={:.2f}_dswir_{}.h5".format(n_layers, dnu, timeofday),
        key="df",
        mode="w")

    print("DSW:", df.shape)
    #print(df.head())
    #print(df.columns)
    # print("kap:", sorted(df["kap"]))
    #print(sorted(df["kap_top"].unique()))
    #print(sorted(df["kap_bottom"].unique()))
    print("TPW:", sorted(df["TPW"]))



def model_dsw(data_dir, filename):
    """DSW radiances that would be seen by Satellite devices [FY4A]."""

    # wavenumber resolution [cm^-1]
    dnu = 3
    pattern = r'TWP=([0-9.]+)_AOD=([0-9.]+)_COD=([0-9.]+)_kap=\[([0-9, ]+)\]_th0=([0-9.]+)'
    match = re.search(pattern, filename)
    if match:
        TPW, AOD, COD, kap_str, th0 = match.groups()
        TPW, AOD, COD, th0 = map(float, [TPW, AOD, COD, th0])
        kap = list(map(int, kap_str.split(', ')))
        kap_top, kap_bottom = kap[-1], kap[0]
    else:
        print("No match found.")
    data = np.load('data/computed/F_dw_os_SW.npz')
    nu = data['nu']
    del data

    F_dw = np.load(data_dir +'/'+filename,  allow_pickle=True).item().get('F_dw')

    # remove invalid data
    idx = ~np.isnan(F_dw)
    F_dw = F_dw[idx]
    nu = nu[idx]

    channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    results = []

    # integrate downwelling radiation:
    lamb_min, lamb_max = 0.25, 4  # spectral range: 3-50 um (SURFRAD)
    nu_min, nu_max = 1/lamb_max*1e4, 1/lamb_min*1e4  # spectral range: 3.5-50 um (ARM SGP C1)
    idx = (nu >= nu_min) & (nu <= nu_max)
    dw_ = F_dw[idx]
    ghi = np.trapz(np.sort(dw_), dx=dnu)
    results = [{
        'TPW': TPW, 'th0': th0, 
        "COD": COD, "AOD": AOD,
        "kap": kap, "kap_top": kap_top, "kap_bottom": kap_bottom,
        'dwir': ghi,
    }]

    return results


def compare_oswr(site, sun_zen, sky="clear", data_dir=None):
    """
    Match ref_OSWR of satellite and model, interpolation, and generate a LUT for one site
    ! Since fix solar zenith (or time), local zenith, but the relative azimuth is not fixed,
    a vector of azimuth is extracted.

    step 1: extract satation, as the grid
    step 2: extract model data, prepare for interpolation (TPW)
    step 3: generate LUT according to each sat data
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
    FY4A_dir=Sat_preprocess(data_dir)
    sat = pd.read_hdf(
        os.path.join(data_dir+FY4A_dir, "{}_radiance_satellite_{}.h5".format(site, sky)),
        'df'
    )
    if sun_zen == 0:
        sat = sat[sat['Sun_Zen'] < 5]
    else:
        targ_sat = sat[(sat['Sun_Zen'] >= (sun_zen-5)) & (sat['Sun_Zen'] < (sun_zen+5))]
        print(targ_sat.index.unique())
        #targ_sat = sat[(sat['Sun_Zen'] == sun_zen)]
        # free unused columns to save memory
        local_zen = targ_sat['Sat_Zen'].unique()[0]
    del sat
    targ_sat.loc[:, 'rela_azi'] = abs(targ_sat['Sun_Azi'] - targ_sat['Sat_Azi'])
    targ_sat.loc[targ_sat['rela_azi'] > 180, 'rela_azi'] = 360 - targ_sat['rela_azi']
    print('target_sat:')
    print('Sun zenith:','mean =', targ_sat['Sun_Zen'].mean(), 'std =', targ_sat['Sun_Zen'].std())
    print('local zenith =', local_zen)
    # Even local zenith is fixed while rela azi it not, so the relative azimuth extracted for prepared.
    print('relative_azi:', 'mean =', targ_sat['rela_azi'].mean(), 'std=', targ_sat['rela_azi'].std())
    targ_sat['tpw'] = targ_sat.apply(lambda row: calculate_tpw(row['T_a'], row['RH']), axis=1)
    # targ_sat['tpw'].describe() 5-120 cm

    # open model data, prepare for interpolation
    model_dir = 'preprocess_model/'
    Fls = os.listdir(data_dir+model_dir)
    Fls = [f for f in Fls if f.endswith('_oswr_{}.h5'.format(timeofday))]
    Fls = [f for f in Fls if f.startswith('Results')]

    modelfiles = []
    for file in Fls:
        match = re.search(r'AOD=([0-9.]+)_COD=([0-9.]+)_kap=\[([0-9, ]+)\]_th0=([0-9.]+)', file)
        if match:
            AOD, COD, kap_str, th0 = match.groups()
            AOD = float(AOD)
            COD = np.round(float(COD),3)
            kap = list(map(int, kap_str.split(', ')))
            kap_str = '-'.join(map(str, [kap[0], kap[-1]]))
            th0 = float(th0)

            # Filter for all COD
            if th0 != sun_zen:
                continue
            if AOD != 0.1243:
                continue
            # ["0-0", "6-8", "6-10", "6-12", "6-14", "6-16", "6-18", "6-20", "6-22", "24-24"]
            if kap_str in ["0-0", "24-24"]:
                continue
            if sky == "clear" and COD == 0.0:
                modelfiles.append(file)
            elif sky == "cloudy" and COD > 0:
                modelfiles.append(file)
            elif sky in ["day", "night"] and COD >= 0:
                modelfiles.append(file)
            else:
                print("No matched model files")

    # prepare model data for interpolation
    # generate LUT according for each sat data
    frames = []
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    for chan in channels:
        TPW_x = []
        Ref_OSWR_vec_x = []
        for i in range(len(modelfiles)):
            with h5py.File(data_dir+model_dir+modelfiles[i], 'r') as h5file:
                if chan in h5file:
                    channal = h5file[chan]
                    ref_OSWR = channal['ref_OSWR_channel'][:]
                    # extract local zenith,
                    local_zen_index = find_bin_indices(targ_sat['Sat_Zen'], 0, 'zenith')
                    ref_OSWR_azimuth = ref_OSWR[list(set(local_zen_index))[0]][:]
                    AOD = channal.attrs['AOD']
                    COD = channal.attrs['COD']
                    kap = channal.attrs['kap']
                    tpw = channal.attrs['TPW']
                    TPW_x.append(tpw)
                    Ref_OSWR_vec_x.append(ref_OSWR_azimuth)
                else:
                    continue
            # extract model OSWR by zenith, then (relative azimuth)
            Ref_OSWR_vec = np.vstack(Ref_OSWR_vec_x)
            model_ref_azimuth = np.zeros((targ_sat['tpw'].shape[0], Ref_OSWR_vec[0].shape[0]))
            for angle_idx in range(Ref_OSWR_vec[0].shape[0]):  # Loop over each angle_bin
                # Create interpolator for the current angle_bin
                f = interp1d(TPW_x, Ref_OSWR_vec[:,angle_idx],
                             kind='linear', bounds_error=False, fill_value='extrapolate')
                # Interpolate to new_TPW_x
                model_ref_azimuth[:, angle_idx] = f(targ_sat['tpw'].values)
            targ_sat['num_ind'] = np.linspace(0, targ_sat.shape[0]-1, targ_sat.shape[0])
            targ_sat['time'] = targ_sat.index.values
            targ_sat['ref_OSWR_model'] = targ_sat.apply(
                lambda row: extract_ref_correct(row['rela_azi'], model_ref_azimuth,
                                                row['num_ind'], row['time'], row['ele'],
                                                site), axis=1)
            # extract_ref_correct(targ_sat['rela_azi'][10], model_ref_azimuth,
            #                                     targ_sat['num_ind'][10], targ_sat.index[10], targ_sat['ele'][10],
            #                                     site)
            if sky == "night":
                frame = sat[["T_a", "RH", "P_w"]]
            else:  # day
                # frame = sat[["T_a", "RH", "P_w", "ghi", "ghi_clear", "dni", "dni_clear", "clearsky"]]
                frame = targ_sat[["tpw", "ref_OSWR_model"]].copy()  # , "RH"
            frame.loc[:, 'COD'] = COD
            frame.loc[:, 'AOD'] = AOD
            frame.loc[:, 'kap'] = kap_str
            kap_bottom, kap_top = kap_str.split("-")
            frame.loc[:, 'kap_top'] = int(kap_top)
            frame.loc[:, 'kap_bottom'] = int(kap_bottom)
            frame.loc[:, 'channel'] = chan
            frame.loc[:, 'ref_OSWR_sat'] = targ_sat[chan]
            frames.append(frame)
    # export results
    df = pd.concat(frames).sort_index()
    df.to_hdf(data_dir+"LUT/"+"results_{}_{}_{}_oswr.h5".format(sun_zen, site, sky), "df", mode="w")
    print(site, df.shape[0], df.index[0], df.index[-1])
    print(df.columns)
    print(sorted(df["kap"].unique()))
    print(sorted(df["kap_top"].unique()))
    print(sorted(df["kap_bottom"].unique()))




def compare_dlw(site, sky="clear", data_dir=None):
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

    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    dnu = 3.0
    n_layers = 54
    model_dir = "preprocess_model/"
    model = pd.read_hdf(data_dir+model_dir+'results_case2_{}layers_dnu={:.2f}_dswir_{}.h5'.format(n_layers, dnu, timeofday), 'df')
    model = model.dropna(how="any")
    ground = pd.read_hdf(
        os.path.join(data_dir, '{}_radiance_satellite_{}.h5'.format(site, sky)),
        'df'
    )

    AOD = 0.1243
    model = model[model["AOD"] == AOD]

    if sky == "clear":
        model = model[(model["COD"] == 0.0) & (model["kap"] == "0-0")]
    elif sky == "cloudy":
        model = model[model["COD"] > 0]
    else:  # all-sky (day or night)
        model = model[model["COD"] >= 0]

    # DWIR = DWIR(T, RH)
    #model = model[~model["kap"].isin(["0-0", "6-8", "6-10", "6-12", "6-14", "6-16", "6-18", "6-20", "6-22", "24-24"])]
    model = model[~model["kap"].isin(["0-0", "24-24"])]
    frames = []
    for [COD, kap], group in model.groupby(["COD", "kap"]):
        #print(site, sky, COD, kap)

        # DWIR = DWIR(T, RH)
        dwir_pred = interpolate.griddata(
            group[["T", "RH"]].values,
            group["dwir"].values,
            ground[["T_a", "RH"]].values,
            method="linear"
        )

        # store as a DataFrame
        cols = ground.columns[~ground.columns.str.match("C[0-9]{2}")]
        frame = ground[cols].copy()
        frame = frame.rename(columns={"dw_ir": "dwir_true"})
        frame.insert(frame.shape[1], "dwir_pred", dwir_pred)
        frame.insert(frame.shape[1], "COD", COD)
        frame.insert(frame.shape[1], "AOD", AOD)
        frame.insert(frame.shape[1], "kap", kap)
        kap_bottom, kap_top = kap.split("-")
        frame.insert(frame.shape[1], "kap_top", int(kap_top))
        frame.insert(frame.shape[1], "kap_bottom", int(kap_bottom))

        frames.append(frame)

    # export results
    df = pd.concat(frames).sort_index()
    df.to_hdf("results_{}_{}_dlw.h5".format(site, sky), "df", mode="w")
    print(site, df.shape[0], df.index[0], df.index[-1])
    print(df.columns)
    print(sorted(df["kap"].unique()))
    print(sorted(df["kap_top"].unique()))
    print(sorted(df["kap_bottom"].unique()))


def postprocess_olr(site, solar_zen, sky="cloudy", data_dir=None):
    """Post-process the OLR to estimate cloud optical properties.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".

    Returns
    -------
    None

    """
    lut_dir = "LUT/"
    df = pd.read_hdf(data_dir+lut_dir+"results_{}_{}_{}_oswr.h5".format(solar_zen, site, sky), "df")
    #df = df[df.index <= "2018-07-01 00:00:00"]
    # df = df[df.index > "2018-07-01 00:00:00"]
    print(df.shape)
    print(sorted(df["COD"].unique()))
    print(sorted(df["kap"].unique()))
    print(sorted(df["kap_top"].unique()))
    print(sorted(df["kap_bottom"].unique()))
    #print(sorted(df["channel"].unique()))

    # limit potential cloud layers (single layer)
    # if sky == "cloudy":
        #print("kap:", sorted(df["kap"].unique()))
        #df = df[df["kap"].isin(["8-8", "10-10", "12-12", "14-14", "16-16", "18-18", "20-20"])]
        #df = df[~df["kap"].isin(["0-0"])]

        ## single-layer
        #df = df[df["kap"].isin(["8-8", "10-10", "12-12", "14-14", "16-16", "18-18", "20-20", "22-22"])]

        # multi-layer
    #     df = df[df["kap"].isin([
    #         "8-8", "10-10", "12-12", "14-14", "16-16", "18-18", "20-20", "22-22",  # single-layer
    #         "6-8", "6-10", "6-12", "6-14", "6-16", "6-18", "6-20", "6-22",
    #         "10-12", "10-14", "10-16", "10-18", "10-20", "10-22",
    #         "14-16", "14-18", "14-20", "14-22",
    #         "18-20", "18-22",
    #     ])]
    # elif sky in ["night", "day"]:
    #     df = df[df["kap"].isin(["0-0", "8-8", "10-10", "12-12", "14-14", "16-16", "18-18", "20-20"])]

    print("{}: kap = ".format(sky), sorted(df["kap"].unique()))

    # No need for reflectance, normalized OLR: OLR* = OLR / Planck
    # df.insert(df.shape[1], "OLR_satellite_star", df["OLR_satellite"] / df["OLR_blackbody"])
    # df.insert(df.shape[1], "OLR_model_star", df["OLR_model"] / df["OLR_blackbody"])

    if sky == "clear":
        df.to_hdf("timeseries_{}_{}_oswr.h5".format(site, sky), "df", mode="w")
        print("{} ({}): OLR".format(site, sky), df.shape[0])
        #print("AOD:", sorted(df["AOD"].unique()))
    else:
        # if cloudy, don't consider clear-sky
        if sky == "cloudy":
            df = df[df["COD"] > 0.0]
        # make a backup of before the estimation step
        df2 = df.copy()
        # error per channel
        #df.insert(df.shape[1], "error", (df["OLR_satellite"] - df["OLR_model"]).abs())
        df.insert(df.shape[1], "error", (df["ref_OSWR_sat"] - df["ref_OSWR_model"]).abs())

        # if all-sky (day or night): repeat COD=0 case
        if sky in ["day", "night"]:
            print("Before: clear kap=", sorted(df[df["COD"] == 0.0]["kap"].unique()))
            # cloud locations
            kaps = sorted(df[df["COD"] > 0]["kap"].unique())
            df_clear = df[(df["COD"] == 0.0) & (df["kap"] == "0-0")]
            for kap in kaps:
                df_clear_copy = df_clear.copy()
                df_clear_copy["kap"] = kap
                df = pd.concat([df, df_clear_copy])

            print("After: clear kap=", sorted(df[df["COD"] == 0.0]["kap"].unique()))

        # remove clear-sky periods from estimation
        if sky == "day" and df_clear.shape[0] > 0:
            # split into clear and cloudy based on LW channels
            df_clear = df.loc[(df["COD"] == 0) & (df["kap"] == "0-0"), :].reset_index()
            df_clear = pd.pivot_table(df_clear, index=["timestamp", "clearsky"], columns="channel", values="error")
            df_clear = df_clear[["C11", "C13", "C14", "C15"]].mean(axis=1)
            df_clear.name = "error"
            df_clear = df_clear.reset_index().set_index("timestamp")
            df_clear.insert(df_clear.shape[1], "clearsky_pred", df_clear["error"] < 0.05)  # threshold
            df_clear = df_clear[df_clear["clearsky_pred"] == True]
            df_clear = pd.DataFrame(index=df_clear.index, data={"COD": 0.0, "kap": "0-0"})
            #print(df_clear.head())

            # remove the clear-sky periods from estimation
            print("Include clear:", df.shape[0])
            df = df[~df.index.isin(df_clear.index)]
            print("Remove clear:", df.shape[0])

        # 1) select CTH (z_T) based on channels 8-10 (fixed COD)
        # print("CTH selection...")
        # cth = df[(df["channel"].isin(["C08", "C09", "C10"])) & (df["COD"] == 0.1)]
        # #print("CTH: kap=", sorted(cth["kap"].unique()))
        # cth = cth.reset_index()
        # cth = cth.groupby(["timestamp", "kap_top"])[["error"]].mean()
        # cth = cth.reset_index()
        # cth = cth.loc[cth.groupby("timestamp")["error"].idxmin(), :]
        #print("CTH: kap=", sorted(cth["kap"].unique()))

        # 2) select COD and CBH (z_B)
        print("COD and CBH selection...")


        cod = df[(df["channel"].isin(["C01", "C02", "C03", "C04", "C05",'C06']))]
        #print("COD: kap=", sorted(cod["kap"].unique()))
        #print("COD: COD=", sorted(cod["COD"].unique()))
        cod = cod.reset_index()
        cod = cod.groupby(["timestamp", "COD"])[["error"]].mean()
        #cod = cod.reset_index().set_index(["timestamp", "kap_top"])

        cod = cod.loc[cod.groupby("timestamp")["error"].idxmin(), :]  # error(COD, CBH)
        cod = cod.reset_index().set_index("timestamp")
        df = cod[["COD"]]
        #print("COD: kap=", sorted(cod["kap"].unique()))
        #print("COD: COD=", sorted(cod["COD"].unique()))

        # add back clear-sky timestamps
        if sky == "day" and df_clear.shape[0]>0:
            df = pd.concat([df, df_clear]).sort_index()

        # add back in other columns
        df2 = df2.reset_index().set_index(["timestamp", "COD"])
        df = df.reset_index().set_index(["timestamp", "COD"])
        df2 = df2.loc[df2.index.isin(df.index), :]
        df2 = df2.reset_index().set_index("timestamp").sort_index()
        #print(df2.head())
        del df

        # export
        df2.to_hdf(data_dir+"/results/"+"timeseries_{}_{}_oswr_multilayer2.h5".format(site, sky), "df", mode="w")
        #df2['error'] = df2['ref_OSWR_sat'] - df2['ref_OSWR_model']
        print("{} ({}): OLR".format(site, sky), df2.shape[0], df2.index[0], df2.index[-1])
        print("COD:", sorted(df2["COD"].unique()))
        print("kap:", sorted(df2["kap"].unique()))
        print("kap_top:", sorted(df2["kap_top"].unique()))
        print("kap_bottom:", sorted(df2["kap_bottom"].unique()))
        #print(df2.head())
        #print(df2.tail())
        del df2

def postprocess_dlw(site, sky="cloudy"):
    """Post-process DLW with the estimated cloud properties.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".

    Returns
    -------
    None

    """

    # DLW [W/m^2]
    dlw = pd.read_hdf("results_{}_{}_dlw.h5".format(site, sky), "df")
    dlw.index.name = "timestamp"
    print(dlw.columns)

    if sky != "clear":

        # use (COD, CBH) from OLR
        olr = pd.read_hdf("timeseries_{}_{}_olr_multilayer2.h5".format(site, sky), "df")
        olr = olr.reset_index().set_index(["timestamp", "kap_top", "kap_bottom", "COD"])
        dlw = dlw.reset_index().set_index(["timestamp", "kap_top", "kap_bottom", "COD"])
        dlw = dlw.loc[dlw.index.isin(olr.index)]
        dlw = dlw.reset_index().set_index("timestamp")

    # export
    dlw = dlw.sort_index()
    dlw.to_hdf(data_dir+"/results/"+"timeseries_{}_{}_dlw_multilayer2.h5".format(site, sky), "df", mode="w")
    print("{} ({}): DLW".format(site, sky), dlw.shape[0], dlw.index[0], dlw.index[-1])
    #print(dlw.shape[0], dlw.resample("5min", closed="right", label="right").mean().dropna(how="any").shape[0])
    print(dlw.columns)
    print("COD:", sorted(dlw["COD"].unique()))
    print("kap:", sorted(dlw["kap"].unique()))
    print("kap_top:", sorted(dlw["kap_top"].unique()))
    print("kap_bottom:", sorted(dlw["kap_bottom"].unique()))
    del dlw
if __name__ == "__main__":
    for timeofday in ["day"]:
        data_dir = './lut_test_file/'
        testfile = 'testfile/'
        spectral = 'SW'
        #preprocess_model_oswr(data_dir=data_dir+testfile, timeofday=timeofday)   # O SW Radiance [W/m^2/sr]
        #preprocess_model_dsw(data_dir=data_dir+testfile, timeofday=timeofday)   # DLW [W/m^2]

    # compare model against measured data
    # data_dir = os.path.join("data")      # directory with SURFRAD+GOES data
    # for sky in ["clear", "cloudy", "day", "night"]:
    # for sky in ["cloudy", "day"]:
    for sky in ["day"]:
        # sites = ["BON", "DRA", "FPK", "GWN", "PSU", "SXF", "TBL"]
        sites = ["AKA"]
        # sites = ["DRA", "FPK", "GWN", "PSU", "SXF", "TBL"]
        # sites = ["DRA", "FPK", "GWN"]
        # sites = ["PSU", "SXF", "TBL"]
        for site in sites:
            for solar_zen in [45, 60, 65]: # zenith for the day time
                # 0, 15, 30,
            # one time one site, one zenith, (local_zei, azimuth)
                print(sky, site, solar_zen)
                # interpolate OLR and DLW to ground conditions (T, RH)
                compare_oswr(site, solar_zen, sky=sky, data_dir=data_dir)
                compare_dlw(site, sky=sky, data_dir=data_dir)

            # estimate the cloud optical properties
                #postprocess_olr(site, solar_zen, sky='day', data_dir=data_dir)
                #postprocess_dlw(site, sky=sky)