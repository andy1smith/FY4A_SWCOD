"""Post-process model outputs, and compare OLR and DLW."""

import os
import glob
import pandas as pd
import numpy as np
from scipy import interpolate
from itertools import chain
from multiprocessing import Pool


def preprocess_model_olr(data_dir=None, timeofday="day"):
    """Process OLR model results."""

    dnu = 0.10
    n_layers = 32
    filenames = glob.glob(
        os.path.join(data_dir, "F_mono_{}layers_rh=0.*0_tdelta=*.h5".format(n_layers))
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
    df.to_hdf("results_{}layers_{:.2f}_olr_{}.h5".format(n_layers, dnu, timeofday), "df", mode="w")
    print("OLR:", df.shape)
    #print(df.columns)
    print("kap:", sorted(df["kap"].unique()))
    #print(sorted(df["kap_top"].unique()))
    #print(sorted(df["kap_bottom"].unique()))
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
    kap = "{}-{}".format(kap_bottom, kap_top)
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
        results.append({'RH': RH, 'T': T, "T_delta": T_delta, "COD": COD, "kap": kap, "kap_top": kap_top,
                        "kap_bottom": kap_bottom, "AOD": AOD, 'channel': channel, 'OLR': OLR_channel})

    return results


def preprocess_model_dlw(data_dir=None, timeofday="day"):
    """Process DLW model results."""

    dnu = 0.10
    n_layers = 32
    filenames = glob.glob(
        os.path.join(data_dir, "F_mono_{}layers_rh=0.*0_tdelta=*.h5".format(n_layers))
    )
    print("# files:", len(filenames))

    # parallel process files
    pool = Pool()
    results = pool.map(eppley_dlw, filenames)
    pool.close()
    results = list(chain(*results))  # flatten: list(list(dict)) ==> list(dict)

    # export
    df = pd.DataFrame(results)
    df = df.sort_index()
    df.to_hdf("results_{}layers_{:.02f}_dlw_{}.h5".format(n_layers, dnu, timeofday), "df", mode="w")
    print("DLW:", df.shape)
    #print(df.head())
    #print(df.columns)
    print("kap:", sorted(df["kap"].unique()))
    #print(sorted(df["kap_top"].unique()))
    #print(sorted(df["kap_bottom"].unique()))
    print("RH:", sorted(df["RH"].unique()))


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


def compare_olr(site, sky="clear", data_dir=None):
    """Compare satellite and model OLR.

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

    dnu = 0.10
    n_layers = 32
    model = pd.read_hdf('results_{}layers_{:.2f}_olr_{}.h5'.format(n_layers, dnu, timeofday), 'df')
    sat = pd.read_hdf(
        os.path.join(data_dir, "{}_radiance_satellite_{}.h5".format(site, sky)),
        'df'
    )
    sat.index.name = "timestamp"

    model = model.dropna(how="any")
    model = model[model["channel"].str.startswith("C")]
    AOD = 0.1243
    model = model[model["AOD"] == AOD]

    if sky == "clear":
        model = model[(model["COD"] == 0.0) & (model["kap"] == "0-0")]
    elif sky == "cloudy":
        model = model[model["COD"] > 0]
    else:   # day or night
        model = model[model["COD"] >= 0]

    # OLR = OLR(T, RH)
    #model = model[~model["kap"].isin(["0-0", "6-8", "6-10", "6-12", "6-14", "6-16", "6-18", "6-20", "6-22", "24-24"])]
    model = model[~model["kap"].isin(["0-0", "24-24"])]
    frames = []
    for [COD, kap, channel], group in model.groupby(["COD", "kap", "channel"]):
        print(group.shape, COD, kap, channel)
        #print("{}: COD={:>7.1e}, kap={}, {}".format(site, COD, kap, channel))

        # model OLR [W/m^2] (corrected for T and RH)
        olr_model = interpolate.griddata(
            group[["T", "RH"]].values,
            group["OLR"].values,
            sat[["T_a", "RH"]].values,
            method="linear"
        )
        del group

        # satellite OLR [W/m^2] = OLR per solid angle [W/(m^2 sr)] * pi
        olr_sat = sat["{}_satellite".format(channel)] * np.pi

        # blackbody radiative flux per channel [W/m^2]
        olr_black = sat["{}_blackbody".format(channel)]

        if sky == "night":
            frame = sat[["T_a", "RH", "P_w"]]
        else:
            #frame = sat[["T_a", "RH", "P_w", "ghi", "ghi_clear", "dni", "dni_clear", "clearsky"]]
            frame = sat[["T_a", "RH"]]
        frame.insert(frame.shape[1], "T", sat["T_a"].values)
        frame.insert(frame.shape[1], "COD", COD)
        frame.insert(frame.shape[1], "AOD", AOD)
        frame.insert(frame.shape[1], "kap", kap)
        kap_bottom, kap_top = kap.split("-")
        frame.insert(frame.shape[1], "kap_top", int(kap_top))
        frame.insert(frame.shape[1], "kap_bottom", int(kap_bottom))
        frame.insert(frame.shape[1], "channel", channel)
        frame.insert(frame.shape[1], "OLR_model", olr_model)
        frame.insert(frame.shape[1], "OLR_satellite", olr_sat)
        frame.insert(frame.shape[1], "OLR_blackbody", olr_black)
        frames.append(frame)

    # export results
    df = pd.concat(frames).sort_index()
    df.to_hdf("results_{}_{}_olr.h5".format(site, sky), "df", mode="w")
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

    dnu = 0.10
    n_layers = 32
    model = pd.read_hdf('results_{}layers_{:.2f}_dlw_{}.h5'.format(n_layers, dnu, timeofday), 'df')
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


def postprocess_olr(site, sky="cloudy"):
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

    df = pd.read_hdf("results_{}_{}_olr.h5".format(site, sky), "df")
    #df = df[df.index <= "2018-07-01 00:00:00"]
    df = df[df.index > "2018-07-01 00:00:00"]
    #print(df.shape)
    print(sorted(df["COD"].unique()))
    print(sorted(df["kap"].unique()))
    print(sorted(df["kap_top"].unique()))
    print(sorted(df["kap_bottom"].unique()))
    #print(sorted(df["channel"].unique()))

    # limit potential cloud layers (single layer)
    if sky == "cloudy":
        #print("kap:", sorted(df["kap"].unique()))
        #df = df[df["kap"].isin(["8-8", "10-10", "12-12", "14-14", "16-16", "18-18", "20-20"])]
        #df = df[~df["kap"].isin(["0-0"])]

        ## single-layer
        #df = df[df["kap"].isin(["8-8", "10-10", "12-12", "14-14", "16-16", "18-18", "20-20", "22-22"])]

        # multi-layer
        df = df[df["kap"].isin([
            "8-8", "10-10", "12-12", "14-14", "16-16", "18-18", "20-20", "22-22",  # single-layer
            "6-8", "6-10", "6-12", "6-14", "6-16", "6-18", "6-20", "6-22",
            "10-12", "10-14", "10-16", "10-18", "10-20", "10-22",
            "14-16", "14-18", "14-20", "14-22",
            "18-20", "18-22",
        ])]

    elif sky in ["night", "day"]:
        df = df[df["kap"].isin(["0-0", "8-8", "10-10", "12-12", "14-14", "16-16", "18-18", "20-20"])]

    print("{}: kap = ".format(sky), sorted(df["kap"].unique()))

    # normalized OLR: OLR* = OLR / Planck
    df.insert(df.shape[1], "OLR_satellite_star", df["OLR_satellite"] / df["OLR_blackbody"])
    df.insert(df.shape[1], "OLR_model_star", df["OLR_model"] / df["OLR_blackbody"])

    if sky == "clear":
        df.to_hdf("timeseries_{}_{}_olr.h5".format(site, sky), "df", mode="w")
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
        df.insert(df.shape[1], "error", (df["OLR_satellite_star"] - df["OLR_model_star"]).abs())

        # if all-sky (day or night): repeat COD=0 case
        if sky in ["day", "night"]:
            print("Before: clear kap=", sorted(df[df["COD"] == 0.0]["kap"].unique()))
            # cloud locations
            kaps = sorted(df[df["COD"] > 0]["kap"].unique())
            df_clear = df[(df["COD"] == 0.0) & (df["kap"] == "0-0")]
            for kap in kaps:
                df_clear_copy = df_clear.copy()
                df_clear_copy["kap"] = kap
                df = df.append(df_clear_copy)

            print("After: clear kap=", sorted(df[df["COD"] == 0.0]["kap"].unique()))

        # remove clear-sky periods from estimation
        if sky == "day":
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
        print("CTH selection...")
        cth = df[(df["channel"].isin(["C08", "C09", "C10"])) & (df["COD"] == 0.1)]
        #print("CTH: kap=", sorted(cth["kap"].unique()))
        cth = cth.reset_index()
        cth = cth.groupby(["timestamp", "kap_top"])[["error"]].mean()
        cth = cth.reset_index()
        cth = cth.loc[cth.groupby("timestamp")["error"].idxmin(), :]
        #print("CTH: kap=", sorted(cth["kap"].unique()))

        # 2) select COD and CBH (z_B)
        print("COD and CBH selection...")
        cod = df[(df["channel"].isin(["C07", "C11", "C13", "C14", "C15"]))]
        #print("COD: kap=", sorted(cod["kap"].unique()))
        #print("COD: COD=", sorted(cod["COD"].unique()))
        cod = cod.reset_index()
        cod = cod.groupby(["timestamp", "COD", "kap_top", "kap_bottom"])[["error"]].mean()
        cod = cod.reset_index().set_index(["timestamp", "kap_top"])
        cth = cth.reset_index().set_index(["timestamp", "kap_top"])
        cod = cod.loc[cod.index.isin(cth.index), :]
        cod = cod.reset_index()
        del cth

        cod = cod.loc[cod.groupby("timestamp")["error"].idxmin(), :]  # error(COD, CBH)
        cod = cod.reset_index().set_index("timestamp")
        df = cod[["COD", "kap_top", "kap_bottom"]]
        #print("COD: kap=", sorted(cod["kap"].unique()))
        #print("COD: COD=", sorted(cod["COD"].unique()))

        # add back clear-sky timestamps
        if sky == "day":
            df = pd.concat([df, df_clear]).sort_index()

        # add back in other columns
        df2 = df2.reset_index().set_index(["timestamp", "COD", "kap_top", "kap_bottom"])
        df = df.reset_index().set_index(["timestamp", "COD", "kap_top", "kap_bottom"])
        df2 = df2.loc[df2.index.isin(df.index), :]
        df2 = df2.reset_index().set_index("timestamp").sort_index()
        #print(df2.head())
        del df

        # export
        df2.to_hdf("timeseries_{}_{}_olr_multilayer2.h5".format(site, sky), "df", mode="w")
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
    dlw.to_hdf("timeseries_{}_{}_dlw_multilayer2.h5".format(site, sky), "df", mode="w")
    print("{} ({}): DLW".format(site, sky), dlw.shape[0], dlw.index[0], dlw.index[-1])
    #print(dlw.shape[0], dlw.resample("5min", closed="right", label="right").mean().dropna(how="any").shape[0])
    print(dlw.columns)
    print("COD:", sorted(dlw["COD"].unique()))
    print("kap:", sorted(dlw["kap"].unique()))
    print("kap_top:", sorted(dlw["kap_top"].unique()))
    print("kap_bottom:", sorted(dlw["kap_bottom"].unique()))
    del dlw


if __name__ == "__main__":

    # pre-process files
    # for timeofday in ["day", "night"]:
    for timeofday in ["day"]:
       data_dir = os.path.join("results", "layers_AFGL_midlatitude_summer_{}".format(timeofday))
       preprocess_model_olr(data_dir=data_dir, timeofday=timeofday)   # OLR [W/m^2]
       preprocess_model_dlw(data_dir=data_dir, timeofday=timeofday)   # DLW [W/m^2]

    # compare model against measured data
    data_dir = os.path.join("data")      # directory with SURFRAD+GOES data
    #for sky in ["clear", "cloudy", "day", "night"]:
    #for sky in ["cloudy", "day"]:
    for sky in ["cloudy"]:
        #sites = ["BON", "DRA", "FPK", "GWN", "PSU", "SXF", "TBL"]
        sites = ["BON"]
        # sites = ["DRA", "FPK", "GWN", "PSU", "SXF", "TBL"]
        #sites = ["DRA", "FPK", "GWN"]
        #sites = ["PSU", "SXF", "TBL"]
        for site in sites:
            print(sky, site)
            ## interpolate OLR and DLW to ground conditions (T, RH)
            compare_olr(site, sky=sky, data_dir=data_dir)
            compare_dlw(site, sky=sky, data_dir=data_dir)

            # estimate the cloud optical properties
            postprocess_olr(site, sky=sky)
            postprocess_dlw(site, sky=sky)
