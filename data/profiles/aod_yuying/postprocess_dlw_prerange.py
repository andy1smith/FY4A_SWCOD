"""Post-process model outputs, and compare OLR and DLW."""

import os
import glob
import pandas as pd
import numpy as np
from scipy import interpolate
from itertools import chain
from multiprocessing import Pool
from pandas.core.indexes.base import Index
import re


def kap_file(files, kappas):
    filenames = []
    for file in files:
        kap = re.findall(r"kap=(.*?)_AOD", file)
        if kap[0] in kappas:
            filenames.append(file)
    return filenames

def preprocess_model_olr(data_dir=None, timeofday="day",  dnu=0.10, n_layers=32, kappas=None):
    """Process OLR model results."""

    # read all files
    filenames = glob.glob(os.path.join(data_dir, "F_mono_{}layers_rh=*_tdelta=*.h5".format(n_layers)))

    # # read all files with specific kappa
    # files = glob.glob(os.path.join(data_dir, "F_mono_{}layers_rh=*_tdelta=*.h5".format(n_layers)))
    # print("# files:", len(files))
    # filenames = kap_file(files, kappas)

    print("# files:", len(filenames))

    # parallel process files
    pool = Pool()
    results = pool.map(channel_olr, filenames)
    pool.close()
    results = list(chain(*results))  # flatten: list(list(dict)) ==> list(dict)

    # export
    df = pd.DataFrame(results)
    df = df.sort_index()
    df.to_hdf("./results/results_{}layers_{:.2f}_olr_{}.h5".format(n_layers, dnu, timeofday), "df", mode="w")
    print("OLR:", df.shape)
    # print("kap:", sorted(df["kap"].unique()))
    # print("RH:", sorted(df["RH"].unique()))


def preprocess_model_olr_compare(data_dir=None, timeofday="day",  dnu=0.10, n_layers=32, aod=None):
    """Process OLR model results."""

    filenames = glob.glob(
        os.path.join(data_dir, "F_mono_{}layers*.h5".format(n_layers))
    )

    # filenames = glob.glob(os.path.join(data_dir, "F_mono_{}layers_rh=*_tdelta=*AOD={:.4f}.h5".format(n_layers, aod)))

    print("# files:", len(filenames))

    # parallel process files
    pool = Pool()
    results = pool.map(channel_olr, filenames)
    pool.close()
    results = list(chain(*results))  # flatten: list(list(dict)) ==> list(dict)

    # export
    df = pd.DataFrame(results)
    df = df.sort_index()
    df.to_hdf("./results/results_{}layers_{:.2f}_olr_{}_aods.h5".format(n_layers, dnu, timeofday), "df", mode="w")
    print("OLR:", df.shape)
    print("kap:", sorted(df["kap"].unique()))
    print("RH:", sorted(df["RH"].unique()))
    print("COD:", sorted(df["COD"].unique()))


def preprocess_model_dlw_compare(data_dir=None, timeofday="day", dnu=0.10, n_layers=32, aod=None):
    """Process DLW model results."""

    # read all files
    filenames = glob.glob(os.path.join(data_dir, "F_mono_{}layers_rh=0.*0_tdelta=*.h5".format(n_layers)))
    # filenames = glob.glob(os.path.join(data_dir, "F_mono_{}layers_rh=*_tdelta=*AOD={:.4f}.h5".format(n_layers, aod)))


    print("# files:", len(filenames))

    # parallel process files
    pool = Pool()
    results = pool.map(eppley_dlw, filenames)
    pool.close()
    results = list(chain(*results))  # flatten: list(list(dict)) ==> list(dict)

    # export
    df = pd.DataFrame(results)
    df = df.sort_index()
    df.to_hdf("./results/results_{}layers_{:.02f}_dlw_{}_aods.h5".format(n_layers, dnu, timeofday), "df", mode="w")
    print("DLW:", df.shape)
    print("kap:", sorted(df["kap"].unique()))
    print("RH:", sorted(df["RH"].unique()))


def preprocess_model_csolr(data_dir=None, timeofday="day",  dnu=0.10, n_layers=32):
    """Process clear sky OLR model results."""

    filenames = glob.glob(
        os.path.join(data_dir, "F_mono_{}*COD=0.00_kap=0*.h5".format(n_layers))
    )


    print("# files:", len(filenames))

    # parallel process files
    pool = Pool()
    results = pool.map(channel_olr, filenames)
    pool.close()
    results = list(chain(*results))  # flatten: list(list(dict)) ==> list(dict)

    # export
    df = pd.DataFrame(results)
    df = df.sort_index()
    df.to_hdf("./results/results_{}layers_{:.2f}_olr_{}_clearsky.h5".format(n_layers, dnu, timeofday), "df", mode="w")
    print("OLR:", df.shape)
    print("kap:", sorted(df["kap"].unique()))
    print("RH:", sorted(df["RH"].unique()))


def channel_olr(filename):
    """Channel OLR [W/m^2]."""

    df = pd.read_hdf(filename, "df")
    nu = df.index.values  # wavenumber [cm^-1]
    T = df["T"].values[0]
    T_delta = df["T_delta"].values[0]
    RH = df["RH"].values[0]
    COD = df["COD"].values[0]
    kap_top = df["kap_top"].values[0]
    kap_bottom = df["kap_bottom"].values[0]
    kap = df["kap"].values[0]
    AOD = df["AOD"].values[0]
    toa = df['uw_toa'].values

    # remove invalid data
    idx = ~np.isnan(toa)
    toa = toa[idx]
    nu = nu[idx]

    channels = ['C{:02d}'.format(c) for c in range(7, 16 + 1)]
    results = []

    for channel in channels:
        # load ABI calibration data
        channel_number = int(channel[-2:])
        filename = os.path.join(
            'abi_calibration',
            'GOES-R_ABI_PFM_SRF_CWG_ch{}.txt'.format(channel_number)
        )
        calibration = np.genfromtxt(filename, delimiter=' ' * 6, skip_header=2)
        calibration_nu = calibration[:, 1]  # wavenumber [cm^-1]
        calibration_srf = calibration[:, 2]  # relative SRF [-]

        # reverse order (so wavenumber is increasing)
        calibration_nu = calibration_nu[::-1]
        calibration_srf = calibration_srf[::-1]

        # interpolate calibration to match model
        srf = np.interp(nu, calibration_nu, calibration_srf)

        # integrate spectral radiance over the channel
        idx = (nu >= calibration_nu.min()) & (nu <= calibration_nu.max())
        toa_channel = toa[idx]
        srf_channel = srf[idx]
        nu_channel = nu[idx]
        OLR_channel = np.trapz(np.multiply(toa_channel, srf_channel), x=nu_channel)

        results.append({'RH': RH, 'T': T, "T_delta": T_delta, "COD": COD, "kap": kap, "kap_top": kap_top, "kap_bottom": kap_bottom, "AOD": AOD, 'channel': channel, 'OLR': OLR_channel})

    return results


def preprocess_model_dlw(data_dir=None, timeofday="day", dnu=0.10, n_layers=32, kappas=None):
    """Process DLW model results."""

    # read all files
    filenames = glob.glob(os.path.join(data_dir, "F_mono_{}layers_rh=0.*0_tdelta=*.h5".format(n_layers)))

    # # read files with specific kappa
    # files = glob.glob(os.path.join(data_dir, "F_mono_{}layers_rh=*_tdelta=*.h5".format(n_layers)))
    # print("# files:", len(files))
    # filenames = kap_file(files, kappas)

    print("# files:", len(filenames))

    # parallel process files
    pool = Pool()
    results = pool.map(eppley_dlw, filenames)
    pool.close()
    results = list(chain(*results))  # flatten: list(list(dict)) ==> list(dict)

    # export
    df = pd.DataFrame(results)
    df = df.sort_index()

    save_path = './results_predlw/'
    if not os.path.isdir(save_path):
        print("Creating directory:", save_path)
        os.mkdir(save_path)

    df.to_hdf(save_path + "results_{}layers_{:.02f}_dlw_{}.h5".format(n_layers, dnu, timeofday), "df", mode="w")
    print("DLW:", df.shape)
    #print(df.head())
    #print(df.columns)
    # print("kap:", sorted(df["kap"].unique()))
    #print(sorted(df["kap_top"].unique()))
    #print(sorted(df["kap_bottom"].unique()))
    # print("RH:", sorted(df["RH"].unique()))


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
    kap = df["kap"].values[0]
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
    # nu_min, nu_max = 0, 3000  # longwave whole spectral range
    # nu_min, nu_max = 50, 2500  # longwave spectral range
    #nu_min, nu_max = 200.0, 2857.14  # spectral range: 3.5-50 um (ARM SGP C1)

    idx = (nu >= nu_min) & (nu <= nu_max)
    dw_pir = radiance[idx]
    dwir = np.trapz(np.sort(dw_pir), dx=dnu)

    # save downwelling radiance to ~2km altitude
    dwir_layers = ['dwir_layer{}'.format(i) for i in range(2, 10 + 1)]
    for i in range(0, 8 + 1):
        radiance = df["dw_layer{}".format(i+2)].values
        nu = df.index.values  # wavenumber [cm^-1]

        # remove NaNs
        idx = ~np.isnan(radiance)
        radiance = radiance[idx]
        nu = nu[idx]

        idx = (nu >= nu_min) & (nu <= nu_max)
        dw_pir = radiance[idx]
        dwir_layers[i] = np.trapz(np.sort(dw_pir), dx=dnu)


    results = [{
        'RH': RH, 'T': T, "T_delta": T_delta,
        "COD": COD, "AOD": AOD,
        "kap": kap, "kap_top": kap_top, "kap_bottom": kap_bottom,
        'dwir': dwir, 'dwir_layer2': dwir_layers[0], 'dwir_layer3': dwir_layers[1],
        'dwir_layer4': dwir_layers[2], 'dwir_layer5': dwir_layers[3],
        'dwir_layer6': dwir_layers[4], 'dwir_layer7': dwir_layers[5],
        'dwir_layer8': dwir_layers[6], 'dwir_layer9': dwir_layers[7],
        'dwir_layer10': dwir_layers[8],
    }]

    return results


def compare_olr_cs(site, sky="clear", data_dir=None, dnu=0.10, n_layers=32, AOD = 0.1243):
    """Compare satellite and model OLR.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + satellite data files.
    dnu ： Wavenumber resolution
    n_layers : Number of atmospheric layers
    AOD : aerosol optical depths

    Returns
    -------
    None

    """
    print("{}: sky={}".format(site, sky))

    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    # read data
    model = pd.read_hdf('./results/results_{}layers_{:.2f}_olr_{}_clearsky.h5'.format(n_layers, dnu, timeofday), 'df')
    sat = pd.read_hdf(os.path.join(data_dir, "{}_radiance_satellite_{}.h5".format(site, sky)), 'df')
    sat.index.name = "timestamp"

    model = model.dropna(how="any")
    model = model[model["channel"].str.startswith("C")]
    model = model[model["AOD"] == AOD]

    if sky == "clear":
        model = model[(model["COD"] == 0.0) & (model["kap"] == "0")]
    elif sky == "cloudy":
        model = model[model["COD"] > 0]
    else:   # day or night
        model = model[model["COD"] >= 0]

    frames = []
    for [COD, kap, kap_top, kap_bottom, channel], group in model.groupby(['COD', 'kap', 'kap_top', 'kap_bottom', 'channel']):
        print(group.shape, COD, kap, channel)

        # model OLR [W/m^2] (corrected for T and RH)
        olr_model = interpolate.griddata(
            group[["T", "RH"]].values,
            group["OLR"].values,
            sat[["T_a", "RH"]].values,
            method="linear"
        )
        del group

        # satellite OLR [W/m^2]
        olr_sat = sat["{}_satellite".format(channel)]

        # blackbody radiative flux per channel [W/m^2]
        olr_black = sat["{}_blackbody".format(channel)]

        if sky == "night":
            frame = sat[["T_a", "RH", "P_w"]]
        else:
            frame = sat[["T_a", "RH"]]
        frame.insert(frame.shape[1], "T", sat["T_a"].values)
        frame.insert(frame.shape[1], "COD", COD)
        frame.insert(frame.shape[1], "AOD", AOD)
        frame.insert(frame.shape[1], "kap", kap)
        frame.insert(frame.shape[1], "kap_top", int(kap_top))
        frame.insert(frame.shape[1], "kap_bottom", int(kap_bottom))
        frame.insert(frame.shape[1], "channel", channel)
        frame.insert(frame.shape[1], "OLR_model", olr_model)
        frame.insert(frame.shape[1], "OLR_satellite", olr_sat)
        frame.insert(frame.shape[1], "OLR_blackbody", olr_black)
        frames.append(frame)

    # export results
    df = pd.concat(frames).sort_index()
    df.to_hdf("./results/results_{}_{}_olr_clearsky.h5".format(site, sky), "df", mode="w")
    print(site, df.shape[0], df.index[0], df.index[-1])
    print(df.columns)
    print(sorted(df["kap"].unique()))
    print(sorted(df["kap_top"].unique()))
    print(sorted(df["kap_bottom"].unique()))


def compare_olr_compare(site, sky="clear", data_dir=None, dnu=0.10, n_layers=32, AOD = 0.1243):
    """Compare satellite and model OLR.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + satellite data files.
    dnu ： Wavenumber resolution
    n_layers : Number of atmospheric layers
    AOD : aerosol optical depths
    kappas : cloud layers

    Returns
    -------
    None

    """
    print("{}: sky={}".format(site, sky))

    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    # read data
    model = pd.read_hdf('./results/results_{}layers_{:.2f}_olr_{}_compare.h5'.format(n_layers, dnu, timeofday), 'df')
    sat = pd.read_hdf(os.path.join(data_dir, "{}_radiance_satellite_{}.h5".format(site, sky)), 'df')
    sat.index.name = "timestamp"

    model = model.dropna(how="any")
    model = model[model["channel"].str.startswith("C")]
    model = model[model["AOD"] == AOD]

    if sky == "clear":
        model = model[(model["COD"] == 0.0) & (model["kap"] == "0")]
    elif sky == "cloudy":
        model = model[model["COD"] > 0]
    else:   # day or night
        model = model[model["COD"] >= 0]


    frames = []
    for [COD, kap, kap_top, kap_bottom, channel], group in model.groupby(["COD", "kap", "kap_top", "kap_bottom", "channel"]):
        # print(group.shape, COD, kap, channel)

        # model OLR [W/m^2] (corrected for T and RH)
        olr_model = interpolate.griddata(
            group[["T", "RH"]].values,
            group["OLR"].values,
            sat[["T_a", "RH"]].values,
            method="linear"
        )
        del group

        # satellite OLR [W/m^2]
        olr_sat = sat["{}_satellite".format(channel)]

        # blackbody radiative flux per channel [W/m^2]
        olr_black = sat["{}_blackbody".format(channel)]

        if sky == "night":
            frame = sat[["T_a", "RH", "P_w"]]
        else:
            frame = sat[["T_a", "RH"]]
        frame.insert(frame.shape[1], "T", sat["T_a"].values)
        frame.insert(frame.shape[1], "COD", COD)
        frame.insert(frame.shape[1], "AOD", AOD)
        frame.insert(frame.shape[1], "kap", kap)
        frame.insert(frame.shape[1], "kap_top", kap_top)
        frame.insert(frame.shape[1], "kap_bottom", kap_bottom)
        frame.insert(frame.shape[1], "channel", channel)
        frame.insert(frame.shape[1], "OLR_model", olr_model)
        frame.insert(frame.shape[1], "OLR_satellite", olr_sat)
        frame.insert(frame.shape[1], "OLR_blackbody", olr_black)
        frames.append(frame)

    # export results
    df = pd.concat(frames).sort_index()
    df.to_hdf("./results/results_{}_{}_olr_aod.h5".format(site, sky), "df", mode="w")
    print(site, df.shape[0], df.index[0], df.index[-1])
    print(df.columns)
    print(sorted(df["kap"].unique()))
    print(sorted(df["COD"].unique()))
    print(sorted(df["kap_top"].unique()))
    print(sorted(df["kap_bottom"].unique()))


def compare_dlw_compare(site, sky="clear", data_dir=None, dnu=0.10, n_layers=32, AOD = 0.1243):
    """Compare the ground and modeled DLW.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + satellite data files.
    dnu ： Wavenumber resolution
    n_layers : Number of atmospheric layers
    AOD : aerosol optical depths
    kappas : cloud layers

    Returns
    -------
    None

    """
    print("{}: sky={}".format(site, sky))

    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    # read data
    model = pd.read_hdf('./results/results_{}layers_{:.2f}_dlw_{}_compare.h5'.format(n_layers, dnu, timeofday), 'df')
    sat = pd.read_hdf(os.path.join(data_dir, '{}_radiance_satellite_{}.h5'.format(site, sky)), 'df')

    model = model.dropna(how="any")
    model = model[model["AOD"] == AOD]



    if sky == "clear":
        model = model[(model["COD"] == 0.0) & (model["kap"] == "0")]
    elif sky == "cloudy":
        model = model[model["COD"] > 0]
    else:  # all-sky (day or night)
        model = model[model["COD"] >= 0]


    frames = []
    for [COD, kap, kap_top, kap_bottom], group in model.groupby(["COD", "kap", "kap_top", "kap_bottom"]):
        # print(site, sky, COD, kap)

        dwir_pred = interpolate.griddata(
            group[["T", "RH"]].values,
            group["dwir"].values,
            sat[["T_a", "RH"]].values,
            method="linear"
        )

        # store as a DataFrame
        cols = sat.columns[~sat.columns.str.match("C[0-9]{2}")]
        frame = sat[cols].copy()
        frame = frame.rename(columns={"dw_ir": "dwir_true"})
        frame.insert(frame.shape[1], "dwir_pred", dwir_pred)
        frame.insert(frame.shape[1], "COD", COD)
        frame.insert(frame.shape[1], "AOD", AOD)
        frame.insert(frame.shape[1], "kap", kap)
        frame.insert(frame.shape[1], "kap_top", kap_top)
        frame.insert(frame.shape[1], "kap_bottom", kap_bottom)

        frames.append(frame)

    # export results
    df = pd.concat(frames).sort_index()
    df.to_hdf("./results/results_{}_{}_dlw_aod.h5".format(site, sky), "df", mode="w")
    print(site, df.shape[0], df.index[0], df.index[-1])
    print(df.columns)
    print(sorted(df["kap"].unique()))
    print(sorted(df["COD"].unique()))
    print(sorted(df["kap_top"].unique()))
    print(sorted(df["kap_bottom"].unique()))

def compare_olr(site, sky="clear", data_dir=None, dnu=0.10, n_layers=32, AOD = 0.1243, kappas=None, rh=None):
    """Compare satellite and model OLR.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + satellite data files.
    dnu ： Wavenumber resolution
    n_layers : Number of atmospheric layers
    AOD : aerosol optical depths
    kappas : cloud layers

    Returns
    -------
    None

    """
    print("{}: sky={}, kaps={}".format(site, sky, kappas))

    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    # read data
    model = pd.read_hdf('./results/results_{}layers_{:.2f}_olr_{}.h5'.format(n_layers, dnu, timeofday), 'df')
    df_sat = pd.read_hdf(os.path.join(data_dir, "{}_radiance_satellite_{}.h5".format(site, sky)), 'df')
    df_sat.index.name = "timestamp"

    model = model.dropna(how="any")
    model = model[model["channel"].str.startswith("C")]
    model = model[model["AOD"] == AOD]
    model = model[model["kap"].isin(kappas)]

    if sky == "clear":
        model = model[(model["COD"] == 0.0) & (model["kap"] == "0")]
    elif sky == "cloudy":
        model = model[model["COD"] > 0]
    else:   # day or night
        model = model[model["COD"] >= 0]

    # added by YY to avoid memory issue [2023/12/13]
    sats = [df_sat[df_sat['RH'] <= rh], df_sat[df_sat['RH'] > rh]]
    print('rh <= {}:'.format(rh), len(df_sat[df_sat['RH'] <= rh]))
    print('rh > {}'.format(rh), len(df_sat[df_sat['RH'] > rh]))
    rhs = ['low', 'high']

    for i in range(len(sats)):
        sat = sats[i]
        frames = []
        for [COD, kap, kap_top, kap_bottom, channel], group in model.groupby(["COD", "kap", "kap_top", "kap_bottom", "channel"]):
            # print(group.shape, COD, kap, channel)

            # model OLR [W/m^2] (corrected for T and RH)
            olr_model = interpolate.griddata(
                group[["T", "RH"]].values,
                group["OLR"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )
            del group

            # satellite OLR [W/m^2]
            olr_sat = sat["{}_satellite".format(channel)]

            # blackbody radiative flux per channel [W/m^2]
            olr_black = sat["{}_blackbody".format(channel)]

            if sky == "night":
                frame = sat[["T_a", "RH", "P_w"]]
            else:
                frame = sat[["T_a", "RH"]]
            frame.insert(frame.shape[1], "T", sat["T_a"].values)
            frame.insert(frame.shape[1], "COD", COD)
            frame.insert(frame.shape[1], "AOD", AOD)
            frame.insert(frame.shape[1], "kap", kap)
            frame.insert(frame.shape[1], "kap_top", kap_top)
            frame.insert(frame.shape[1], "kap_bottom", kap_bottom)
            frame.insert(frame.shape[1], "channel", channel)
            frame.insert(frame.shape[1], "OLR_model", olr_model)
            frame.insert(frame.shape[1], "OLR_satellite", olr_sat)
            frame.insert(frame.shape[1], "OLR_blackbody", olr_black)
            frames.append(frame)

        # export results
        df = pd.concat(frames).sort_index()
        df.to_hdf("./results/results_{}_{}_olr_{}.h5".format(site, sky, rhs[i]), "df", mode="w")
        print(site, df.shape[0], df.index[0], df.index[-1])
        print(df.columns)
        print(sorted(df["kap"].unique()))
        print(sorted(df["COD"].unique()))
        print(sorted(df["kap_top"].unique()))
        print(sorted(df["kap_bottom"].unique()))


def compare_dlw(site, sky="clear", data_dir=None, dnu=0.10, n_layers=32, AOD = 0.1243, kappas=None, rh=None):
    """Compare the ground and modeled DLW.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + satellite data files.
    dnu ： Wavenumber resolution
    n_layers : Number of atmospheric layers
    AOD : aerosol optical depths
    kappas : cloud layers

    Returns
    -------
    None

    """
    print("{}: sky={}, kaps={}".format(site, sky, kappas))

    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    # read data
    model = pd.read_hdf('./results_predlw/results_{}layers_{:.2f}_dlw_{}.h5'.format(n_layers, dnu, timeofday), 'df')
    df_sat = pd.read_hdf(os.path.join(data_dir, '{}_radiance_satellite_{}.h5'.format(site, sky)), 'df')

    model = model.dropna(how="any")
    model = model[model["AOD"] == AOD]
    model = model[model["kap"].isin(kappas)]

    # COD = 10 ** np.arange(-1.0, 1.0 + 0.2, 0.2)  # 0.5 to 10
    # CODs = np.hstack((np.array([0.0]), COD))
    # model = model[model["COD"].isin(CODs)]

    if sky == "clear":
        model = model[(model["COD"] == 0.0) & (model["kap"] == "0")]
    elif sky == "cloudy":
        model = model[model["COD"] > 0]
    else:  # all-sky (day or night)
        model = model[model["COD"] >= 0]

    # added by YY to avoid memory issue [2023/12/13]
    sats = [df_sat[df_sat['RH'] <= rh], df_sat[df_sat['RH'] > rh]]
    print('rh <= {}:'.format(rh), len(df_sat[df_sat['RH'] <= rh]))
    print('rh > {}'.format(rh), len(df_sat[df_sat['RH'] > rh]))

    rhs = ['low', 'high']
    for i in range(len(sats)):
        sat = sats[i]
        frames = []
        for [COD, kap, kap_top, kap_bottom], group in model.groupby(["COD", "kap", "kap_top", "kap_bottom"]):
            # print(site, sky, COD, kap)

            dwir_pred = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            # downwelling flux up to ~2km
            dwir_pred2 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer2"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred3 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer3"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred4 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer4"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred5 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer5"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred6 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer6"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred7 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer7"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred8 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer8"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred9 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer9"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred10 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer10"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )


            # store as a DataFrame
            cols = sat.columns[~sat.columns.str.match("C[0-9]{2}")]
            frame = sat[cols].copy()
            frame = frame.rename(columns={"dw_ir": "dwir_true"})
            frame.insert(frame.shape[1], "dwir_pred", dwir_pred)
            frame.insert(frame.shape[1], "COD", COD)
            frame.insert(frame.shape[1], "AOD", AOD)
            frame.insert(frame.shape[1], "kap", kap)
            frame.insert(frame.shape[1], "kap_top", kap_top)
            frame.insert(frame.shape[1], "kap_bottom", kap_bottom)
            frame.insert(frame.shape[1], "dwir_pred2", dwir_pred2)
            frame.insert(frame.shape[1], "dwir_pred3", dwir_pred3)
            frame.insert(frame.shape[1], "dwir_pred4", dwir_pred4)
            frame.insert(frame.shape[1], "dwir_pred5", dwir_pred5)
            frame.insert(frame.shape[1], "dwir_pred6", dwir_pred6)
            frame.insert(frame.shape[1], "dwir_pred7", dwir_pred7)
            frame.insert(frame.shape[1], "dwir_pred8", dwir_pred8)
            frame.insert(frame.shape[1], "dwir_pred9", dwir_pred9)
            frame.insert(frame.shape[1], "dwir_pred10", dwir_pred10)

            frames.append(frame)

        # export results
        df = pd.concat(frames).sort_index()
        df.to_hdf("./results_predlw/results_{}_{}_dlw_{}.h5".format(site, sky, rhs[i]), "df", mode="w")
        print(site, df.shape[0], df.index[0], df.index[-1])
        print(df.columns)
        print(sorted(df["kap"].unique()))
        print(sorted(df["COD"].unique()))
        print(sorted(df["kap_top"].unique()))
        print(sorted(df["kap_bottom"].unique()))


def compare_olr_aod(site, sky="clear", data_dir=None, dnu=0.10, n_layers=32, AOD = 0.1243, kappas=None, rh=None):
    """Compare satellite and model OLR.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + satellite data files.
    dnu ： Wavenumber resolution
    n_layers : Number of atmospheric layers
    AOD : aerosol optical depths
    kappas : cloud layers

    Returns
    -------
    None

    """
    print("{}: sky={}, kaps={}".format(site, sky, kappas))

    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    # read data
    model = pd.read_hdf('./results/results_{}layers_{:.2f}_olr_{}_aods.h5'.format(n_layers, dnu, timeofday), 'df')  # clear sky for all sites
    # model = pd.read_hdf('./results/results_{}layers_{:.2f}_olr_{}_compare.h5'.format(n_layers, dnu, timeofday), 'df')  # TBL for all sky conditions
    df_sat = pd.read_hdf(os.path.join(data_dir, "{}_radiance_satellite_{}.h5".format(site, sky)), 'df')
    df_sat.index.name = "timestamp"

    model = model.dropna(how="any")
    model = model[model["channel"].str.startswith("C")]
    model = model[model["AOD"] == AOD]
    model = model[model["kap"].isin(kappas)]

    if sky == "clear":
        model = model[(model["COD"] == 0.0) & (model["kap"] == "0")]
    elif sky == "cloudy":
        model = model[model["COD"] > 0]
    else:   # day or night
        model = model[model["COD"] >= 0]

    # added by YY to avoid memory issue [2023/12/13]
    sats = [df_sat[df_sat['RH'] <= rh], df_sat[df_sat['RH'] > rh]]
    print('rh <= {}:'.format(rh), len(df_sat[df_sat['RH'] <= rh]))
    print('rh > {}'.format(rh), len(df_sat[df_sat['RH'] > rh]))
    rhs = ['low', 'high']

    for i in range(len(sats)):
        sat = sats[i]
        frames = []
        for [COD, kap, kap_top, kap_bottom, channel], group in model.groupby(["COD", "kap", "kap_top", "kap_bottom", "channel"]):
            # print(group.shape, COD, kap, channel)

            # model OLR [W/m^2] (corrected for T and RH)
            olr_model = interpolate.griddata(
                group[["T", "RH"]].values,
                group["OLR"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )
            del group

            # satellite OLR [W/m^2]
            olr_sat = sat["{}_satellite".format(channel)]

            # blackbody radiative flux per channel [W/m^2]
            olr_black = sat["{}_blackbody".format(channel)]

            if sky == "night":
                frame = sat[["T_a", "RH", "P_w"]]
            else:
                frame = sat[["T_a", "RH"]]
            frame.insert(frame.shape[1], "T", sat["T_a"].values)
            frame.insert(frame.shape[1], "COD", COD)
            frame.insert(frame.shape[1], "AOD", AOD)
            frame.insert(frame.shape[1], "kap", kap)
            frame.insert(frame.shape[1], "kap_top", kap_top)
            frame.insert(frame.shape[1], "kap_bottom", kap_bottom)
            frame.insert(frame.shape[1], "channel", channel)
            frame.insert(frame.shape[1], "OLR_model", olr_model)
            frame.insert(frame.shape[1], "OLR_satellite", olr_sat)
            frame.insert(frame.shape[1], "OLR_blackbody", olr_black)
            frames.append(frame)

        # export results
        df = pd.concat(frames).sort_index()
        df.to_hdf("./results/results_{}_{}_olr_{}_aod.h5".format(site, sky, rhs[i]), "df", mode="w")
        print(site, df.shape[0], df.index[0], df.index[-1])
        print(df.columns)
        print(sorted(df["kap"].unique()))
        print(sorted(df["COD"].unique()))
        print(sorted(df["kap_top"].unique()))
        print(sorted(df["kap_bottom"].unique()))


def compare_dlw_aod(site, sky="clear", data_dir=None, dnu=0.10, n_layers=32, AOD = 0.1243, kappas=None, rh=None):
    """Compare the ground and modeled DLW.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + satellite data files.
    dnu ： Wavenumber resolution
    n_layers : Number of atmospheric layers
    AOD : aerosol optical depths
    kappas : cloud layers

    Returns
    -------
    None

    """
    print("{}: sky={}, kaps={}".format(site, sky, kappas))

    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    # read data
    model = pd.read_hdf('./results/results_{}layers_{:.2f}_dlw_{}_aods.h5'.format(n_layers, dnu, timeofday), 'df')  # clear sky for all sites
    # model = pd.read_hdf('./results/results_{}layers_{:.2f}_dlw_{}_compare.h5'.format(n_layers, dnu, timeofday), 'df')  # TBL for all sky conditions
    df_sat = pd.read_hdf(os.path.join(data_dir, '{}_radiance_satellite_{}.h5'.format(site, sky)), 'df')

    model = model.dropna(how="any")
    model = model[model["AOD"] == AOD]
    model = model[model["kap"].isin(kappas)]

    # COD = 10 ** np.arange(-1.0, 1.0 + 0.2, 0.2)  # 0.5 to 10
    # CODs = np.hstack((np.array([0.0]), COD))
    # model = model[model["COD"].isin(CODs)]

    if sky == "clear":
        model = model[(model["COD"] == 0.0) & (model["kap"] == "0")]
    elif sky == "cloudy":
        model = model[model["COD"] > 0]
    else:  # all-sky (day or night)
        model = model[model["COD"] >= 0]

    # added by YY to avoid memory issue [2023/12/13]
    sats = [df_sat[df_sat['RH'] <= rh], df_sat[df_sat['RH'] > rh]]
    print('rh <= {}:'.format(rh), len(df_sat[df_sat['RH'] <= rh]))
    print('rh > {}'.format(rh), len(df_sat[df_sat['RH'] > rh]))

    rhs = ['low', 'high']
    for i in range(len(sats)):
        sat = sats[i]
        frames = []
        for [COD, kap, kap_top, kap_bottom], group in model.groupby(["COD", "kap", "kap_top", "kap_bottom"]):
            # print(site, sky, COD, kap)

            dwir_pred = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            # downwelling flux up to ~2km
            dwir_pred2 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer2"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred3 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer3"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred4 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer4"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred5 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer5"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred6 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer6"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred7 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer7"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred8 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer8"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred9 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer9"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )

            dwir_pred10 = interpolate.griddata(
                group[["T", "RH"]].values,
                group["dwir_layer10"].values,
                sat[["T_a", "RH"]].values,
                method="linear"
            )


            # store as a DataFrame
            cols = sat.columns[~sat.columns.str.match("C[0-9]{2}")]
            frame = sat[cols].copy()
            frame = frame.rename(columns={"dw_ir": "dwir_true"})
            frame.insert(frame.shape[1], "dwir_pred", dwir_pred)
            frame.insert(frame.shape[1], "COD", COD)
            frame.insert(frame.shape[1], "AOD", AOD)
            frame.insert(frame.shape[1], "kap", kap)
            frame.insert(frame.shape[1], "kap_top", kap_top)
            frame.insert(frame.shape[1], "kap_bottom", kap_bottom)
            frame.insert(frame.shape[1], "dwir_pred2", dwir_pred2)
            frame.insert(frame.shape[1], "dwir_pred3", dwir_pred3)
            frame.insert(frame.shape[1], "dwir_pred4", dwir_pred4)
            frame.insert(frame.shape[1], "dwir_pred5", dwir_pred5)
            frame.insert(frame.shape[1], "dwir_pred6", dwir_pred6)
            frame.insert(frame.shape[1], "dwir_pred7", dwir_pred7)
            frame.insert(frame.shape[1], "dwir_pred8", dwir_pred8)
            frame.insert(frame.shape[1], "dwir_pred9", dwir_pred9)
            frame.insert(frame.shape[1], "dwir_pred10", dwir_pred10)

            frames.append(frame)

        # export results
        df = pd.concat(frames).sort_index()
        df.to_hdf("./results/results_{}_{}_dlw_{}_aod.h5".format(site, sky, rhs[i]), "df", mode="w")
        print(site, df.shape[0], df.index[0], df.index[-1])
        print(df.columns)
        print(sorted(df["kap"].unique()))
        print(sorted(df["COD"].unique()))
        print(sorted(df["kap_top"].unique()))
        print(sorted(df["kap_bottom"].unique()))


def postprocess_olr(site, sky="cloudy", rhs=None):
    """Post-process the OLR to estimate cloud optical properties.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    rhs: str
        relative humid range: 'high' or 'low'

    Returns
    -------
    None

    """

    # added by YY to solve the bugs in the latest pandas: [nan] not in index (2023/10/11)
    def _ignore_if_missing(self, *args, **kwargs):
        pass

    setattr(Index, '_raise_if_missing', _ignore_if_missing)

    dfs = []
    for rh in rhs:
        print('the current rh', rh)
        # read data
        df = pd.read_hdf("./results/results_{}_{}_olr_{}.h5".format(site, sky, rh), "df")

        print(sorted(df["COD"].unique()))
        print(sorted(df["kap"].unique()))
        print(sorted(df["kap_top"].unique()))
        print(sorted(df["kap_bottom"].unique()))


        # normalized OLR: OLR* = OLR / Planck
        df.insert(df.shape[1], "OLR_satellite_star", df["OLR_satellite"] / df["OLR_blackbody"])
        df.insert(df.shape[1], "OLR_model_star", df["OLR_model"] / df["OLR_blackbody"])

        if sky == "clear":
            dfs.append(df)
        else:
            # if cloudy, don't consider clear-sky
            if sky == "cloudy":
                df = df[df["COD"] > 0.0]

            # make a backup of before the estimation step
            df2 = df.copy()

            # error per channel
            df.insert(df.shape[1], "error", (df["OLR_satellite_star"] - df["OLR_model_star"]).abs())

            # if all-sky (day or night): repeat COD=0 case
            if sky in ["day", "night"]:
                print("Before: clear kap=", sorted(df[df["COD"] == 0.0]["kap"].unique()))
                # cloud locations
                kaps = sorted(df[df["COD"] > 0]["kap"].unique())
                df_clear = df[(df["COD"] == 0.0) & (df["kap"] == "0")]
                for kap in kaps:
                    df_clear_copy = df_clear.copy()
                    df_clear_copy["kap"] = kap
                    # df = df.append(df_clear_copy)
                    df = pd.concat([df, df_clear_copy])

                print("After: clear kap=", sorted(df[df["COD"] == 0.0]["kap"].unique()))

            # remove clear-sky periods from estimation
            if sky == "day":
                # split into clear and cloudy based on LW channels
                df_clear = df.loc[(df["COD"] == 0) & (df["kap"] == "0"), :].reset_index()
                df_head = df_clear.head()
                df_clear = pd.pivot_table(df_clear, index=["timestamp"], columns="channel", values="error")
                df_clear = df_clear[["C11", "C13", "C14", "C15"]].mean(axis=1)
                df_clear.name = "error"
                df_clear = df_clear.reset_index().set_index("timestamp")
                df_clear.insert(df_clear.shape[1], "clearsky_pred", df_clear["error"] < 0.05)  # threshold：0.05
                df_clear = df_clear[df_clear["clearsky_pred"] == True]
                df_clear = pd.DataFrame(index=df_clear.index, data={"COD": 0.0, "kap": "0"})
                print(df_clear.head())

                # remove the clear-sky periods from estimation
                print("Include clear:", df.shape[0])
                df = df[~df.index.isin(df_clear.index)]
                print("Remove clear:", df.shape[0])

            # select CTH (z_T), COD and CBH (z_B) based on channels
            df3 = df.reset_index()  # channels 7-16
            # df3 = df[~df["channel"].isin(["C07"])].reset_index()  # channels 8-16
            df3 = df3.groupby(["timestamp", "COD", "kap", 'kap_top', 'kap_bottom'])[["error"]].mean()
            df3 = df3.reset_index()
            print('before:', df3.head())
            print('before:', df3.shape)
            df3 = df3.loc[df3.groupby("timestamp")["error"].idxmin(), :]
            print('after:', df3.head())
            print('after:', df3.shape)
            df3 = df3.reset_index().set_index("timestamp")
            df = df3[["COD", 'kap', "kap_top", "kap_bottom"]]
            del df3


            # # 1) select CTH (z_T) based on channels 8-10 (fixed COD)
            # print("CTH selection...")
            # cth = df[(df["channel"].isin(["C08", "C09", "C10"])) & (df["COD"] == 0.1)]
            # cth = cth.reset_index()
            # cth = cth.groupby(["timestamp", "kap_top"])[["error"]].mean()
            # cth = cth.reset_index()
            # print('before:', cth.head())
            # print('before:', cth.shape)
            # cth = cth.loc[cth.groupby("timestamp")["error"].idxmin(), :]
            # print('after:', cth.head())
            # print('after:', cth.shape)
            # #print("CTH: kap=", sorted(cth["kap"].unique()))
            # #
            # # 2) select COD and CBH (z_B)
            # print("COD and CBH selection...")
            # cod = df[(df["channel"].isin(["C07", "C11", "C13", "C14", "C15"]))]
            # # cod = df[(df["channel"].isin(["C11", "C13", "C14", "C15"]))]
            # cod = cod.reset_index()
            # cod = cod.groupby(["timestamp", "COD", "kap_top", "kap_bottom"])[["error"]].mean()
            # cod = cod.reset_index().set_index(["timestamp", "kap_top"])
            # cth = cth.reset_index().set_index(["timestamp", "kap_top"])
            # cod = cod.loc[cod.index.isin(cth.index), :]
            # cod = cod.reset_index()
            # del cth
            #
            # cod = cod.loc[cod.groupby("timestamp")["error"].idxmin(), :]  # error(COD, CBH)
            # cod = cod.reset_index().set_index("timestamp")
            # df = cod[["COD", "kap_top", "kap_bottom"]]
            #print("COD: kap=", sorted(cod["kap"].unique()))
            #print("COD: COD=", sorted(cod["COD"].unique()))

            # add back clear-sky timestamps
            if sky == "day":
                df = pd.concat([df, df_clear]).sort_index()

            # add back in other columns
            df2 = df2.reset_index().set_index(["timestamp", "COD", 'kap', "kap_top", "kap_bottom"])
            df = df.reset_index().set_index(["timestamp", "COD", 'kap', "kap_top", "kap_bottom"])
            df2 = df2.loc[df2.index.isin(df.index), :]
            df2 = df2.reset_index().set_index("timestamp").sort_index()
            #print(df2.head())
            del df


            dfs.append(df2)
            # df2.to_hdf("./results/timeseries_{}_{}_olr_multilayer_{}.h5".format(site, sky, rh), "df", mode="w")
            # print("{} ({}): OLR".format(site, sky), df2.shape[0], df2.index[0], df2.index[-1])
            print("COD:", sorted(df2["COD"].unique()))
            print("kap:", sorted(df2["kap"].unique()))
            print("kap_top:", sorted(df2["kap_top"].unique()))
            print("kap_bottom:", sorted(df2["kap_bottom"].unique()))
            #print(df2.head())
            #print(df2.tail())
            del df2

    # concate & export
    df_olr = pd.concat(dfs).sort_index()
    print(df_olr.head())
    df_olr.to_hdf("./results/timeseries_{}_{}_olr_multilayer.h5".format(site, sky), 'df', mode='w')
    del df_olr


def postprocess_dlw(site, sky="cloudy", rhs=None):
    """Post-process DLW with the estimated cloud properties.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    rhs: str
        relative humid range: 'high' or 'low'

    Returns
    -------
    None

    """

    dfs = []
    for rh in rhs:
        # DLW [W/m^2]
        dlw = pd.read_hdf("./results_predlw/results_{}_{}_dlw_{}.h5".format(site, sky, rh), "df")
        dlw.index.name = "timestamp"
        print(dlw.columns)

        if sky != "clear":
            # use (COD, CBH) from OLR
            olr = pd.read_hdf("./results/timeseries_{}_{}_olr_multilayer.h5".format(site, sky), "df")
            olr = olr.reset_index().set_index(["timestamp", 'kap', "kap_top", "kap_bottom", "COD"])
            dlw = dlw.reset_index().set_index(["timestamp", 'kap', "kap_top", "kap_bottom", "COD"])
            dlw = dlw.loc[dlw.index.isin(olr.index)]
            dlw = dlw.reset_index().set_index("timestamp")

        dlw = dlw.sort_index()
        dfs.append(dlw)
        # dlw.to_hdf("./results/timeseries_{}_{}_dlw_multilayer_{}.h5".format(site, sky, rh), "df", mode="w")
        print("{} ({}): DLW".format(site, sky), dlw.shape[0], dlw.index[0], dlw.index[-1])
        #print(dlw.shape[0], dlw.resample("5min", closed="right", label="right").mean().dropna(how="any").shape[0])
        print(dlw.columns)
        print("COD:", sorted(dlw["COD"].unique()))
        print("kap:", sorted(dlw["kap"].unique()))
        print("kap_top:", sorted(dlw["kap_top"].unique()))
        print("kap_bottom:", sorted(dlw["kap_bottom"].unique()))
        del dlw

    # concate & export
    df_dlw = pd.concat(dfs).sort_index()
    print(df_dlw.head())
    df_dlw.to_hdf("./results_predlw/timeseries_{}_{}_dlw_multilayer.h5".format(site, sky), 'df', mode='w')
    del df_dlw


def postprocess_olr_aod(site, sky="cloudy", rhs=None):
    """Post-process the OLR to estimate cloud optical properties.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    rhs: str
        relative humid range: 'high' or 'low'

    Returns
    -------
    None

    """

    # added by YY to solve the bugs in the latest pandas: [nan] not in index (2023/10/11)
    def _ignore_if_missing(self, *args, **kwargs):
        pass

    setattr(Index, '_raise_if_missing', _ignore_if_missing)

    dfs = []
    for rh in rhs:
        print('the current rh', rh)
        # read data
        df = pd.read_hdf("./results/results_{}_{}_olr_{}_aod.h5".format(site, sky, rh), "df")

        print(sorted(df["COD"].unique()))
        print(sorted(df["kap"].unique()))
        print(sorted(df["kap_top"].unique()))
        print(sorted(df["kap_bottom"].unique()))


        # normalized OLR: OLR* = OLR / Planck
        df.insert(df.shape[1], "OLR_satellite_star", df["OLR_satellite"] / df["OLR_blackbody"])
        df.insert(df.shape[1], "OLR_model_star", df["OLR_model"] / df["OLR_blackbody"])

        if sky == "clear":
            dfs.append(df)
        else:
            # if cloudy, don't consider clear-sky
            if sky == "cloudy":
                df = df[df["COD"] > 0.0]

            # make a backup of before the estimation step
            df2 = df.copy()

            # error per channel
            df.insert(df.shape[1], "error", (df["OLR_satellite_star"] - df["OLR_model_star"]).abs())

            # if all-sky (day or night): repeat COD=0 case
            if sky in ["day", "night"]:
                print("Before: clear kap=", sorted(df[df["COD"] == 0.0]["kap"].unique()))
                # cloud locations
                kaps = sorted(df[df["COD"] > 0]["kap"].unique())
                df_clear = df[(df["COD"] == 0.0) & (df["kap"] == "0")]
                for kap in kaps:
                    df_clear_copy = df_clear.copy()
                    df_clear_copy["kap"] = kap
                    # df = df.append(df_clear_copy)
                    df = pd.concat([df, df_clear_copy])

                print("After: clear kap=", sorted(df[df["COD"] == 0.0]["kap"].unique()))

            # remove clear-sky periods from estimation
            if sky == "day":
                # split into clear and cloudy based on LW channels
                df_clear = df.loc[(df["COD"] == 0) & (df["kap"] == "0"), :].reset_index()
                df_head = df_clear.head()
                df_clear = pd.pivot_table(df_clear, index=["timestamp"], columns="channel", values="error")
                df_clear = df_clear[["C11", "C13", "C14", "C15"]].mean(axis=1)
                df_clear.name = "error"
                df_clear = df_clear.reset_index().set_index("timestamp")
                df_clear.insert(df_clear.shape[1], "clearsky_pred", df_clear["error"] < 0.05)  # threshold：0.05
                df_clear = df_clear[df_clear["clearsky_pred"] == True]
                df_clear = pd.DataFrame(index=df_clear.index, data={"COD": 0.0, "kap": "0"})
                print(df_clear.head())

                # remove the clear-sky periods from estimation
                print("Include clear:", df.shape[0])
                df = df[~df.index.isin(df_clear.index)]
                print("Remove clear:", df.shape[0])

            # select CTH (z_T), COD and CBH (z_B) based on channels
            df3 = df.reset_index()  # channels 7-16
            # df3 = df[~df["channel"].isin(["C07"])].reset_index()  # channels 8-16
            df3 = df3.groupby(["timestamp", "COD", "kap", 'kap_top', 'kap_bottom'])[["error"]].mean()
            df3 = df3.reset_index()
            print('before:', df3.head())
            print('before:', df3.shape)
            df3 = df3.loc[df3.groupby("timestamp")["error"].idxmin(), :]
            print('after:', df3.head())
            print('after:', df3.shape)
            df3 = df3.reset_index().set_index("timestamp")
            df = df3[["COD", 'kap', "kap_top", "kap_bottom"]]
            del df3


            # # 1) select CTH (z_T) based on channels 8-10 (fixed COD)
            # print("CTH selection...")
            # cth = df[(df["channel"].isin(["C08", "C09", "C10"])) & (df["COD"] == 0.1)]
            # cth = cth.reset_index()
            # cth = cth.groupby(["timestamp", "kap_top"])[["error"]].mean()
            # cth = cth.reset_index()
            # print('before:', cth.head())
            # print('before:', cth.shape)
            # cth = cth.loc[cth.groupby("timestamp")["error"].idxmin(), :]
            # print('after:', cth.head())
            # print('after:', cth.shape)
            # #print("CTH: kap=", sorted(cth["kap"].unique()))
            # #
            # # 2) select COD and CBH (z_B)
            # print("COD and CBH selection...")
            # cod = df[(df["channel"].isin(["C07", "C11", "C13", "C14", "C15"]))]
            # # cod = df[(df["channel"].isin(["C11", "C13", "C14", "C15"]))]
            # cod = cod.reset_index()
            # cod = cod.groupby(["timestamp", "COD", "kap_top", "kap_bottom"])[["error"]].mean()
            # cod = cod.reset_index().set_index(["timestamp", "kap_top"])
            # cth = cth.reset_index().set_index(["timestamp", "kap_top"])
            # cod = cod.loc[cod.index.isin(cth.index), :]
            # cod = cod.reset_index()
            # del cth
            #
            # cod = cod.loc[cod.groupby("timestamp")["error"].idxmin(), :]  # error(COD, CBH)
            # cod = cod.reset_index().set_index("timestamp")
            # df = cod[["COD", "kap_top", "kap_bottom"]]
            #print("COD: kap=", sorted(cod["kap"].unique()))
            #print("COD: COD=", sorted(cod["COD"].unique()))

            # add back clear-sky timestamps
            if sky == "day":
                df = pd.concat([df, df_clear]).sort_index()

            # add back in other columns
            df2 = df2.reset_index().set_index(["timestamp", "COD", 'kap', "kap_top", "kap_bottom"])
            df = df.reset_index().set_index(["timestamp", "COD", 'kap', "kap_top", "kap_bottom"])
            df2 = df2.loc[df2.index.isin(df.index), :]
            df2 = df2.reset_index().set_index("timestamp").sort_index()
            #print(df2.head())
            del df


            dfs.append(df2)
            # df2.to_hdf("./results/timeseries_{}_{}_olr_multilayer_{}.h5".format(site, sky, rh), "df", mode="w")
            # print("{} ({}): OLR".format(site, sky), df2.shape[0], df2.index[0], df2.index[-1])
            print("COD:", sorted(df2["COD"].unique()))
            print("kap:", sorted(df2["kap"].unique()))
            print("kap_top:", sorted(df2["kap_top"].unique()))
            print("kap_bottom:", sorted(df2["kap_bottom"].unique()))
            #print(df2.head())
            #print(df2.tail())
            del df2

    # concate & export
    df_olr = pd.concat(dfs).sort_index()
    print(df_olr.head())
    df_olr.to_hdf("./results/timeseries_{}_{}_olr_multilayer_aod.h5".format(site, sky), 'df', mode='w')
    del df_olr


def postprocess_dlw_aod(site, sky="cloudy", rhs=None):
    """Post-process DLW with the estimated cloud properties.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    rhs: str
        relative humid range: 'high' or 'low'

    Returns
    -------
    None

    """

    dfs = []
    for rh in rhs:
        # DLW [W/m^2]
        dlw = pd.read_hdf("./results/results_{}_{}_dlw_{}_aod.h5".format(site, sky, rh), "df")
        dlw.index.name = "timestamp"
        print(dlw.columns)

        if sky != "clear":
            # use (COD, CBH) from OLR
            olr = pd.read_hdf("./results/timeseries_{}_{}_olr_multilayer_aod.h5".format(site, sky), "df")
            olr = olr.reset_index().set_index(["timestamp", 'kap', "kap_top", "kap_bottom", "COD"])
            dlw = dlw.reset_index().set_index(["timestamp", 'kap', "kap_top", "kap_bottom", "COD"])
            dlw = dlw.loc[dlw.index.isin(olr.index)]
            dlw = dlw.reset_index().set_index("timestamp")

        dlw = dlw.sort_index()
        dfs.append(dlw)
        # dlw.to_hdf("./results/timeseries_{}_{}_dlw_multilayer_{}.h5".format(site, sky, rh), "df", mode="w")
        print("{} ({}): DLW".format(site, sky), dlw.shape[0], dlw.index[0], dlw.index[-1])
        #print(dlw.shape[0], dlw.resample("5min", closed="right", label="right").mean().dropna(how="any").shape[0])
        print(dlw.columns)
        print("COD:", sorted(dlw["COD"].unique()))
        print("kap:", sorted(dlw["kap"].unique()))
        print("kap_top:", sorted(dlw["kap_top"].unique()))
        print("kap_bottom:", sorted(dlw["kap_bottom"].unique()))
        del dlw

    # concate & export
    df_dlw = pd.concat(dfs).sort_index()
    print(df_dlw.head())
    df_dlw.to_hdf("./results/timeseries_{}_{}_dlw_multilayer_aod.h5".format(site, sky), 'df', mode='w')
    del df_dlw

def table_dlw(data_dir):
    """Summary statistics table for OLR and DLW."""

    # sites = ["BON", "DRA", "FPK", "GWN", "PSU", "SXF", "TBL"]
    sites = [
        ["BON", 40.05192, -88.37309, 230],
        # ["DRA", 36.62373, -116.01947, 1007],
        ["FPK", 48.30783, -105.10170, 634],
        ["GWN", 34.25470, -89.87290, 98],
        ["PSU", 40.72012, -77.93085, 376],
        ["SXF", 43.73403, -96.62328, 473],
        ["TBL", 40.12498, -105.23680, 1689],
    ]

    detal = 5.6697e-8  # Stefan-Boltzman constant [w/m^2 k^4]
    c3 = 0.150 #0.150  # altitude correction coefficient
    H = 8500  # Scale height [m]

    results = []
    for site, lat, lon, alt in sites:
        # # OLR [W/m^2]
        # df = pd.read_hdf(os.path.join(data_dir, "./results/timeseries_{}_{}_olr_multilayer.h5".format(site, sky)), "df")
        #
        # # handle NaNs (before they impact the statistics)
        # df = df[["channel", "OLR_satellite", "OLR_model", "OLR_blackbody"]].dropna(how="any")
        # print('olr length:', len(df))
        #
        # for channel, group in df.groupby("channel"):
        #     #error = group["OLR_satellite"] - group["OLR_model"]
        #     #error = ((group["OLR_satellite"] - group["OLR_model"]) / group["OLR_blackbody"]).dropna(how="any").values
        #     error = ((group["OLR_model"] - group["OLR_satellite"]) / group["OLR_blackbody"]).dropna(how="any").values
        #     mae = np.mean(np.abs(error))
        #     mbe = np.mean(error)
        #     rmse = np.sqrt(np.mean(error ** 2))
        #     #mape = np.mean(np.abs(error / group["OLR_satellite"])) * 100
        #     # results.append({"site": site, "sky": sky, "channel": channel, "MAE": mae, "MBE": mbe, "RMSE": rmse})

        for sky in ["day", "cloudy", "clear", "night"]:
            # DLW [W/m^2]
            df_dlw = pd.read_hdf(os.path.join(data_dir, "./results_predlw/timeseries_{}_{}_dlw_multilayer.h5".format(site, sky)),
                                 "df")

            df_dlw[df_dlw._get_numeric_data() < 0] = np.nan   # abnormal data if exist

            error_dlw = (df_dlw["dwir_pred"] - df_dlw["dwir_true"]).dropna(how="any").values
            mae_dlw = np.mean(np.abs(error_dlw))
            mbe_dlw = np.mean(error_dlw)
            rmse_dlw = np.sqrt(np.mean(error_dlw ** 2))


            results.append({"site": site, "sky": sky, "channel": "DLW", "MAE": mae_dlw, "MBE": mbe_dlw, "RMSE": rmse_dlw})


    # summary
    df_result = pd.DataFrame(results)
    df_result.to_csv("./results_predlw/results_table_dlw.csv", index=False)
    print(df_result.shape)
    df_result = pd.pivot_table(df_result, index=["site", "sky"], columns="channel", values="RMSE")
    print(df_result.round(2))   # rRMSE [-]
    #print(df.round(3) * 100)   # rRMSE [%]

if __name__ == "__main__":
    dnu = 0.10
    n_layers = 32
    AOD = 0.1243
    AOD_tbl = 0.0694  # 2019 average for TBL station
    single = False

    sites = [
        ["BON", 70, 0.1240],
        ["DRA", 20, 0.0549],
        ["FPK", 60, 0.1179],
        ["GWN", 65, 0.1394],
        ["PSU", 60, 0.1448],
        ["SXF", 75, 0.1063],
        ["TBL", 42, 0.0694],
    ]


    kap = ['0', '6', '11', '15', '11-12-13-14', '11-15-16-17', '16', '19', '21', '16-17', '16-20', '18-22',
           '19-20', '19-22', '16-17-18', '16-21-22', '17-18-19', '17-18-22', '18-19-20', '18-20-22',
           '16-17-18-19', '16-17-18-22', '16-17-18-19-20', '16-17-18-21-22']

    # all stations (AOD=0.1243)
    # for sky in ["day", "night"]:
    #     # pre-process files
    #     LBL_dir = os.path.join("results", "layers_AFGL_midlatitude_summer_{}".format(sky))
    #     # nu_min, nu_max = 200.0, 3333.33  # spectral range: 3-50 um (SURFRAD)
    #     preprocess_model_dlw(data_dir=LBL_dir, timeofday=sky, dnu=dnu, n_layers=n_layers, kappas=kap)   # DLW [W/m^2]

    # for sky in ["day", "cloudy", "clear", "night"]:
    #     for site, RH, aod in sites:  # ["BON", "DRA", "FPK", "GWN", "PSU", "SXF", "TBL"]
    #         print(site)
    #         data_dir = os.path.join("data")  # directory with SURFRAD+GOES data
    #         # interpolate DLW to ground conditions (T, RH),
    #         compare_dlw(site, sky=sky, data_dir=data_dir, dnu=dnu, n_layers=n_layers, AOD=AOD, kappas=kap, rh=RH)
    #         #
    #         # postprocess DLW
    #         postprocess_dlw(site, sky=sky, rhs=['low', 'high'])

    table_dlw(data_dir='.')

