
import time

import pandas as pd
from matplotlib.pyplot import fignum_exists

from SCOPE_func import *
import os,sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from LBL_funcs_shortwave import *
import seaborn as sns
from scipy.interpolate import interpn
from matplotlib.colors import LinearSegmentedColormap
from AngDistLUT import *
import math
import sklearn.metrics
import joblib

import warnings
warnings.filterwarnings('ignore')

def fit_TWP_rh0(TPW):
    from scipy import interpolate
    data = np.load('./data/computed/TPW_rh0_lib.npz', allow_pickle=True)
    TPW_v = data['TPW']
    rh0_v = data['rh0_v']
    f = interpolate.interp1d(TPW_v, rh0_v)
    return f(TPW)

def FY4A_calinu(nu, channels, file_dir, dnu = 3, sensor='FY4A'):
    # convert nu to AGRI device nu range. return cm-1.
    nus = set()
    if sensor == 'FY4A' :
        dirpath = file_dir + 'AGRI_calibration/'
    else :
        print('!!! Lack sensor calibration')
    for channel in channels:
        # load ABI calibration data
        channel_number = int(channel[-2:])
        channel_srf = os.path.join(
            dirpath,
            'FY4A_AGRI_SRF_ch{:d}.txt'.format(channel_number)
        )
        calibration = np.loadtxt(channel_srf, delimiter=',', skiprows=1)
        # calibration_wl = calibration[:, 0]  # wavelength [um]
        calibration_nu = calibration[:, 1]  # cm-1
        # calibration_srf = calibration[:, 2] # relative SRF [-]
        # reverse order (so wavenumber is increasing)
        calibration_nu = calibration_nu[::-1]
        # keep the wavenumber within range
        channel_mask = (nu >= calibration_nu.min()) & (nu <= calibration_nu.max())
        nus.update(nu[channel_mask])
        #nus.update(calibration_nu[::dnu])

    nus = np.array(sorted(nus))
    return nus

def get_calibration_srf(channel, file_dir):
    sensor = 'FY4A'
    channel_number = int(channel[-2:])
    dirpath = file_dir + 'AGRI_calibration/'
    channel_srf = os.path.join(dirpath,'FY4A_AGRI_SRF_ch{:d}.txt'.format(channel_number))
    calibration = np.loadtxt(channel_srf, delimiter=',', skiprows=1)
    calibration_nu = calibration[:, 1]
    calibration_srf = calibration[:, 2]
    nu_channel = FY4A_calinu(nu, [channel], file_dir, dnu=3)
    calibration_nu = calibration_nu[::-1]
    calibration_srf = calibration_srf[::-1]
    srf = np.interp(nu_channel, calibration_nu, calibration_srf)
    return srf, nu_channel

def LUT(uw, COD, target_zenith, local_zen, rela_azi, file_dir='./FY4A_data/'):
    '''
    Convert uw to reflectance using LUT

    Parameters
    ----------
    uw
    COD
    target_zenith
    local_zen
    rela_azi
    file_dir

    Returns
    -------

    '''
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    nu0 = np.arange(2500, 35000, 3)  # Wavenumber grid
    nu_channels = FY4A_calinu(nu0, channels, "./FY4A_data/", dnu=3)
    df = pd.DataFrame(columns=channels)
    COD_v = np.concatenate([np.linspace(0, 20, 11), np.linspace(25, 50, 6)])
    COD_ = COD_v[np.argmin(abs(COD - COD_v))]
    fdir = "./FY4A_data/" + 'LUT/'

    for channel in channels:
        # load calibration data : Spectral Response Func
        srf, nu_channel = get_calibration_srf(channel, file_dir)
        theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
        U, S, VT = load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD_)}.h5', channel, target_zenith)
        H_r = reconstruct_hc(U, S, VT)
        nu_idx = np.nonzero(np.isin(nu_channels, nu_channel))[0]  # fixed 1 April.
        # correct uw
        uw_cor = np.multiply(uw[nu_idx], srf)
        uw_channel = np.trapz(uw_cor,nu_channel)
        df.loc[0, channel] = uw_channel/np.pi * H_r[theta_idx, phi_idx] # W/m2/sr radiance
    return df

def Rad_to_Flux_sug_COD(df_row, file_dir='./FY4A_data/'):
    """
    work with Toty for FY4A retrive COD.
    Parameters
    ----------
    df_row
    file_dir

    Returns
    -------

    """
    COD_v = np.concatenate([np.arange(0, 22, 2), np.arange(20, 50+5, 5)])
    fdir = "./data/LUT/"

    local_zen = float(df_row['local_Zen'])
    rela_azi = float(df_row['rela_azi'])
    theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
    target_zenith = float(df_row['th0'])

    # refl
    df_rad = pd.DataFrame([
        {**df_row.to_dict(), 'COD_v': cod} for cod in COD_v
    ])
    df_flux = df_rad.copy()

    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    COD_v = np.concatenate([np.linspace(0, 20, 11), np.linspace(25, 50, 6)])
    #[df_row['COD']]

    #COD_i = df_row['COD']
    # for i, COD in enumerate(COD_i):
    # #for i in range(1):
    #     COD = COD_v[np.argmin(abs(COD_i - COD_v))]
    #     for channel in channels:
    #         U, S, VT = load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD)}.h5', channel, target_zenith)
    #         H_r = reconstruct_hc(U, S, VT)
    #         df_flux.loc[i, channel] = df_rad[channel][i]/H_r[theta_idx, phi_idx] * np.pi  # correct uw_channel
    for i, COD in enumerate(COD_v):
        H_r_series = pd.Series({
            channel: reconstruct_hc(
                *load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD)}.h5', channel, target_zenith))[
                theta_idx, phi_idx]
            for channel in channels
        })
        df_flux.loc[i, channels] = (df_rad.loc[i, channels] / H_r_series) * np.pi


    # gpr_model = joblib.load(r"./data/Surrogate/gpr_model_improved.pkl")
    # scaler_X = joblib.load(r"./data/Surrogate/scaler_X_improved.pkl")
    # scaler_y = joblib.load(r"./data/Surrogate/scaler_y_improved.pkl")
    # F_dw_os_srf_channel = [74.87, 134.24, 33.70, 4.92, 11.08, 3.52]
    #
    # required_columns = ['Ta', 'rh', 'th0', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    # X_test_df = df_flux[required_columns].copy()
    # X_test_df['th0'] = np.cos(np.radians(X_test_df['th0']))
    # X_test_df[channels] = X_test_df[channels].div(F_dw_os_srf_channel)
    #
    # X_test_scaled = scaler_X.transform(X_test_df.values)
    # y_pred_scaled, y_std = gpr_model.predict(X_test_scaled, return_std=True)
    # # print(y_pred_scaled)
    # # print(y_std)
    # # Back to the original COD scale
    # df_flux['COD_pre'] = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    # COD_p= df_flux['COD_v'][np.argmin(abs(df_flux['COD_pre'] - df_flux['COD_v']))]
    # print(COD_p)
    return df_flux[channels].iloc[0] #COD_p #

def Ref_to_Flux_LUT(df_row, file_dir='./FY4A_data/'):
    """
    FY4A : df_row is reflectance
    GOES: df_row is radiance
    Parameters
    ----------
    df_row
    file_dir

    Returns
    -------

    """
    COD_v = np.concatenate([np.arange(0, 22, 2), np.arange(20, 50+5, 5)])
    fdir = "./data/LUT/"

    local_zen = float(df_row['local_Zen'])
    rela_azi = float(df_row['rela_azi'])
    theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
    target_zenith = float(df_row['th0'])

    # refl
    df_rad = pd.DataFrame([
        {**df_row.to_dict(), 'COD_v': cod} for cod in COD_v
    ])
    df_flux = df_rad.copy()

    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    COD_v = np.concatenate([np.linspace(0, 20, 11), np.linspace(25, 50, 6)])
    #[df_row['COD']]

    #COD_i = df_row['COD']
    # for i, COD in enumerate(COD_i):
    # #for i in range(1):
    #     COD = COD_v[np.argmin(abs(COD_i - COD_v))]
    #     for channel in channels:
    #         U, S, VT = load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD)}.h5', channel, target_zenith)
    #         H_r = reconstruct_hc(U, S, VT)
    #         df_flux.loc[i, channel] = df_rad[channel][i]/H_r[theta_idx, phi_idx] * np.pi  # correct uw_channel
    for i, COD in enumerate(COD_v):
        H_r_series = pd.Series({
            channel: reconstruct_hc(
                *load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD)}.h5', channel, target_zenith))[
                theta_idx, phi_idx]
            for channel in channels
        })
        df_flux.loc[i, channels] = (df_rad.loc[i, channels] / H_r_series) * np.pi

    # gpr_model = joblib.load(r"./data/Surrogate/gpr_model_improved.pkl")
    # scaler_X = joblib.load(r"./data/Surrogate/scaler_X_improved.pkl")
    # scaler_y = joblib.load(r"./data/Surrogate/scaler_y_improved.pkl")
    # F_dw_os_srf_channel = [74.87, 134.24, 33.70, 4.92, 11.08, 3.52]
    #
    # required_columns = ['Ta', 'rh', 'th0', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    # X_test_df = df_flux[required_columns].copy()
    # X_test_df['th0'] = np.cos(np.radians(X_test_df['th0']))
    # X_test_df[channels] = X_test_df[channels].div(F_dw_os_srf_channel)
    #
    # X_test_scaled = scaler_X.transform(X_test_df.values)
    # y_pred_scaled, y_std = gpr_model.predict(X_test_scaled, return_std=True)
    # # print(y_pred_scaled)
    # # print(y_std)
    # # Back to the original COD scale
    # df_flux['COD_pre'] = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    # COD_p= df_flux['COD_v'][np.argmin(abs(df_flux['COD_pre'] - df_flux['COD_v']))]
    # print(COD_p)
    return df_flux[channels].iloc[0] #COD_p #

def nearealtime_LUT(sun_zen, local_zen, rela_azi, COD_guess, T_a, RH, file_dir, bandmode):
    # Round values to two decimal places
    N_bundles = 1000
    sun_zen = round(sun_zen)
    local_zen = round(local_zen)
    rela_azi = round(rela_azi)
    COD_guess = round(COD_guess)
    T_a = round(T_a)
    RH = round(RH)
    channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    if sys.platform != 'darwin':
        file_dir = '/mnt/dengnan/'
    flux_file = f"Results_case2_AOD=0.1243_COD={COD_guess}_kap=[10, 11, 12]_th0={sun_zen}_Ta={T_a}_RH={RH}.npy"
    if N_bundles == 1000:
        if bandmode == 'FY4A':
            uw_path = os.path.join(file_dir, 'RTM/channels/TTHG/', flux_file)
        else:
            uw_path = os.path.join(file_dir, 'RTM/fullspectrum/TTHG/', flux_file)

    if not os.path.exists(uw_path):
        print(f"File {flux_file} not found. Running RTM...")
        run_RTM(sun_zen, COD_guess, T_a, RH, file_dir, channels, bandmode, N_bundles)
        #else:
    results = np.load(uw_path, allow_pickle=True).item()
    uw = results.get('F_uw')
    df = LUT(uw, COD_guess, sun_zen, local_zen, rela_azi)
    return df.values.tolist()


def nearealtime_RTM(sun_zen, local_zen, rela_azi, COD_guess, T_a, RH, channels, file_dir, bandmode, N_bundles):
    # Round values to two decimal places
    sun_zen = round(sun_zen)
    local_zen = round(local_zen)
    rela_azi = round(rela_azi)
    COD_guess = round(COD_guess)
    T_a = round(T_a)
    RH = round(RH)
    
    uw_rxyz_file = f"uwxyzr_COD={COD_guess}_th0={sun_zen}_Ta={T_a}_RH={RH}.npy"
    #bandmode = 'FY4A' # FY4A
    # print(bandmode)
    if N_bundles == 1000:
        if bandmode == 'FY4A':
            uw_rxyz_path = os.path.join(file_dir, 'RTM/channels', uw_rxyz_file)
        else:
            uw_rxyz_path = os.path.join(file_dir, 'RTM/fullspectrum', uw_rxyz_file)
    if N_bundles == 10000:
        if sys.platform != 'darwin':
            file_dir='/mnt/dengnan/'
        if bandmode == 'FY4A':
            uw_rxyz_path = os.path.join(file_dir, 'RTM_10000/channels', uw_rxyz_file)
        else:
            uw_rxyz_path = os.path.join(file_dir, 'RTM_10000/fullspectrum', uw_rxyz_file)

    if not os.path.exists(uw_rxyz_path):
        print(f"File {uw_rxyz_file} not found. Running RTM...")
        run_RTM(sun_zen, COD_guess, T_a, RH, file_dir, channels, bandmode, N_bundles)
    results = np.load(uw_rxyz_path, allow_pickle=True).item()
    uw_rxyz_M = results.get('uw_rxyz_M')
    #df_channel_ref1 = RTM_preprocess(uw_rxyz_M, sun_zen, local_zen, rela_azi, channels, file_dir, 'rad', 'full')
    df_channel_ref = RTM_preprocess(uw_rxyz_M, sun_zen, local_zen, rela_azi, channels,
                                      file_dir, 'rad', bandmode, N_bundles)

    return df_channel_ref


def run_FY4A_in_RTM(sun_zen, local_zen, rela_azi, COD_guess, T_a, RH, channels, file_dir, bandmode, N_bundles):
    # Round values to two decimal places
    sun_zen = round(sun_zen)
    local_zen = round(local_zen)
    rela_azi = round(rela_azi)
    COD_guess = round(COD_guess)
    T_a = round(T_a)
    RH = round(RH)

    uw_rxyz_file = f"uwxyzr_COD={COD_guess}_th0={sun_zen}_Ta={T_a}_RH={RH}.npy"
    # bandmode = 'FY4A' # FY4A
    # print(bandmode)
    if N_bundles == 1000:
        if bandmode == 'FY4A':
            uw_rxyz_path = os.path.join(file_dir, 'RTM/channels', uw_rxyz_file)
        else:
            uw_rxyz_path = os.path.join(file_dir, 'RTM/fullspectrum', uw_rxyz_file)
    if N_bundles == 10000:
        if sys.platform != 'darwin':
            file_dir = '/mnt/dengnan/'
        if bandmode == 'FY4A':
            uw_rxyz_path = os.path.join(file_dir, 'RTM_10000/channels', uw_rxyz_file)
        else:
            uw_rxyz_path = os.path.join(file_dir, 'RTM_10000/fullspectrum', uw_rxyz_file)

    if not os.path.exists(uw_rxyz_path):
        print(f"File {uw_rxyz_file} not found. Running RTM...")
        run_RTM(sun_zen, COD_guess, T_a, RH, file_dir, channels, bandmode, N_bundles)
    results = np.load(uw_rxyz_path, allow_pickle=True).item()
    uw_rxyz_M = results.get('uw_rxyz_M')
    # df_channel_ref1 = RTM_preprocess(uw_rxyz_M, sun_zen, local_zen, rela_azi, channels, file_dir, 'rad', 'full')

    df_channel_ref = RTM_preprocess(uw_rxyz_M, sun_zen, local_zen, rela_azi, channels,
                                    file_dir, 'rad', bandmode, N_bundles)

    return df_channel_ref

def RTM_preprocess(uw_rxyz_M, Sun_zen, local_zen, rela_azi, channels, file_dir,
                   outputtype = 'rad', bandmode = 'FY4A', N_bundles = 1000):
    # 1. convert FY4A channel
    # 2. Convert channal radiance to 2D reflectance
    # 3. select [local_zenith, relative_azimuth]
    #from LBL_funcs_utl import *
    data = np.genfromtxt('data/profiles/ASTMG173.csv', delimiter=',', skip_header=2,  # in wavenumber basis
                    names=['wavelength', 'extraterrestrial', '37tilt', 'direct_circum'])
    ref_lam = data['wavelength']  # nm avoid hearder 1
    ref_E = data['extraterrestrial']
    ref_E_nu = -ref_E * ref_lam ** 2 / 1e7  # W/[m2*nm-1] tp W/[m2*cm-1]
    # nu = np.arange(2500, 35000, 3)
    # F_dw_os = -np.interp(-nu, -1e4 / ref_lam, ref_E_nu)
    # from LBL_funcs_utl import plot_3D_AngDist
    # theta_index, phi_index = find_bin_indices(local_zen, rela_azi, 'both')
    # H = plot_3D_AngDist(3, 10, Sun_zen, 0, uw_rxyz_M, nu, F_dw_os, 1000,
    #                     'full', is_flux=False,Norm=False)
    # print(H[theta_index, phi_index])

    # convert to intensity
    #channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    n = len(channels)
    df = pd.DataFrame(data=[[0.0] * n], columns=channels)
    dnu = 3  # spectral resolution 0.1 is enough, 0.01 is too fine, especially for cloudy periods
    nu = np.arange(2500, 35000, dnu)
    if bandmode == 'full':
        # LBL
        nu_input = nu
    else:
        # 6 channels
        channel_6c = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
        nu_input = FY4A_calinu(nu, channel_6c, file_dir, dnu=3)
    for channel in channels:
        channel_number = int(channel[-2:])
        srf,nu_channel=get_calibration_srf(channel, file_dir)
        # Solor TOA and surface albedo
        F_dw_os_channel = -np.interp(-nu_channel, -1e7 / ref_lam, ref_E_nu)  # in wavenumber basis
        # Normalize SRF if necessary
        F_dw_os_SRF = np.multiply(F_dw_os_channel, srf)

        # Integrate spectral radiance over the channel
        # Channal 2D radiance [W/m2/sr]
        nu_idx = np.nonzero(np.isin(nu_input,nu_channel))[0] # fixed 1 April.
        #print(channel)
        result = [uw_rxyz_M[i] for i in nu_idx]
        OSWR_channel = cal_mono_Intensity(result, Sun_zen, nu_input[nu_idx], F_dw_os_SRF,
                                            local_zen, rela_azi, N_bundles=N_bundles,
                                          is_flux=False, Norm=False, dirc='UW')
        if outputtype != 'rad':
            F_dw_os_channal = np.trapz(F_dw_os_channel, nu_channel)
            # need sun-earth distance correction, will be done in extract_sta_oswr
            ref_OSWR_channel = OSWR_channel * np.pi / F_dw_os_channal  # reflectance
            df.loc[0, f"C0{channel_number}"] = ref_OSWR_channel
        else: # Rad
            df.loc[0, f"C0{channel_number}"] = OSWR_channel
    return df

def run_RTM(sun_zen, COD_guess, T_a, RH, file_dir, channels, bandmode, N_bundles=1000, AOD=None):
    ## general inputs
    N_layer = 54 # 54 # the number of atmospheric layers

    dnu = 3 # spectral resolution 0.1 is enough, 0.01 is too fine, especially for cloudy periods
    nu = np.arange(2500,35000,dnu) # spectral grid on wavenumber
    if bandmode == 'FY4A':
        nu = FY4A_calinu(nu, channels, file_dir, dnu=3)
    molecules=['H2O','CO2','O3','N2O','CH4','O2','N2'] # considered atmospheric gases
    #current trace gas surface vmr from http://cdiac.ornl.gov/pns/current_ghg.html, except O3
    vmr0={'H2O':0.03,'CO2':399.5/10**6,'O3':50/10**9,'N2O':328/10**9,
              'CH4':1834/10**9,'O2':2.09/10,'N2':7.81/10}
    model='AFGL midlatitude summer' #profile model, 'AFGL tropical','AFGL midlatitude summer','AFGL midlatitude winter',
    #'AFGL subarctic summer','AFGL subarctic winter','AFGL US standard'
    cld_model = 'default' # cloud model, 'default' or 'caseX'
    period = 'day' # choose 'day' or 'night' for proper temperature profile
    spectral ='SW' # choose 'LW' or 'SW'
    alt = 0 # 22.48/1000 # altitude of location, by default is 0 [km]

    ##inputs for desired atmoshperic and surface conditions
    #surface_v=['case2','PV','CSP'] # name of surface
    surface_v=['case2'] # name of surface
    rh0_v = np.array([RH])/100
    T_surf_v = np.array([T_a]) # K
    if AOD is not None:
        AOD_v = np.array([AOD])
    else:
        AOD_v = np.array([0.1243]) # aerosol optical depth at 479.5 nm
    COD_v = np.array([COD_guess])
    #10 ** np.arange(-1.0,1.6+ 0.2,0.2) # cloud optical depth at 479.5 nm #np.array([0])#
    kap_v = [[10, 11, 12]]
    ##inputs of angles
    th0_v = np.array([sun_zen])
    theta0_v = th0_v / 180 * math.pi  # solar zenith angle in rad
    phi0 = 0 / 180 * math.pi  #solar azimuth angle in rad
    del_angle= 2.5/180*math.pi # DNI acceptance angle, in rad, default is 0.5 degree
    beta_v=np.array([0])/180*math.pi # surface tilt angles in rad
    phi_v=phi0+np.array([0])/180*math.pi # surface azimuth angles in rad
    isTilted=False # whether to compute transposition on inclined surfaces

    ##inputs of finite power plant computation
    x0_v=120.0*np.tan(theta0_v)*np.cos(phi0) # photon starting x location, in km
    y0_v=120.0*np.tan(theta0_v)*np.sin(phi0) # photon starting y location, in km
    R_pp=1 # radius of power plant in km
    is_pp=False # whether to consider power plant of finite size
    #dx_v=np.arange(-5.0,5.2,0.2)# displacement of input photon location
    dx_v=np.array([0.0])# displacement of input photon location
    ## folder directory to store the results
    #file_dir='results_shortwave/project_data/RH/'#SW_cloudTop/'#COD_SWSCOPE/' ##' # create the directory first
    if N_bundles == 1000:
        file_dir = '/mnt/dengnan/'
        if bandmode == '':
            file_dir+='RTM/channels/'
        else:
            file_dir+='RTM/fullspectrum/'
    elif N_bundles == 10000:
        file_dir = '/mnt/dengnan/'
        if bandmode == 'FY4A':
            file_dir+='RTM_10000/channels/FY4A/'
        else:
            file_dir+='RTM_10000/fullspectrum/'
    #file_dir='results_shortwave/sw_scope/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        print(f"Created path: '{file_dir}'")
    
    # compute case by case
    for iSurf in range(0,len(surface_v)):
        inputs_main={'N_layer':N_layer, 'N_bundles':N_bundles, 'nu':nu, 'molecules':molecules,'vmr0':vmr0,
           'model':model,'cld_model':cld_model,'period':period,'spectral':spectral,'surface':surface_v[iSurf],
                     'alt':alt}
        for iT in range(0,len(T_surf_v)):
            for iRH in range(0,len(rh0_v)):
                for iAOD in range(0,len(AOD_v)):
                    for iKAP in range(0, len(kap_v)):
                        for iCOD in range(0,len(COD_v)):
                            properties={'rh0':rh0_v[iRH],'T_surf':T_surf_v[iT],'AOD':AOD_v[iAOD],
                                        'COD':COD_v[iCOD],'kap':kap_v[iKAP]}
                            # print(properties)
                            for iTH in range(0,len(theta0_v)):
                                angles={'theta0':theta0_v[iTH],'phi0':phi0,'del_angle':del_angle,'beta':beta_v,
                                        'phi':phi_v,'isTilted':isTilted}
                                for idx in range(0,len(dx_v)):
                                    finitePP={'x0':-x0_v[iTH]+dx_v[idx],'y0':-y0_v[iTH],'R_pp':R_pp,'is_pp':is_pp}
                                    print ("Start MonteCarlo once.")
                                    start_time = time.time()
                                    out1,out2 = LBL_shortwave(properties,inputs_main,angles,finitePP)
                                    end_time = time.time()
                                    print ("CPU time:", end_time - start_time)
                                    #del out1, out3
                                    if N_bundles == 1000:
                                        fileName1="Results_{}_AOD={}_COD={}_kap={}_th0={}_Ta={}_RH={}".format(
                                            surface_v[iSurf],AOD_v[iAOD],COD_v[iCOD],kap_v[iKAP],th0_v[iTH], T_a, RH)
                                        np.save(file_dir+fileName1,out1)# save results to local directory
                                    else:
                                        fileName2 = "uwxyzr_COD={}_th0={}_Ta={}_RH={}.npy".format(COD_v[iCOD], th0_v[iTH], T_a, RH)
                                        np.save(file_dir + fileName2, out2)  # save results to local directory
                                    del out1, out2
                                    return None


def get_RTM_usw(Sun_Zen, COD, T_a, RH, bandmode='FY4A'):
    file_dir = './FY4A_data/'
    if sys.platform != 'darwin':
        file_dir = '/mnt/dengnan/'
    N_bundles = 1000
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']

    Sun_Zen = round(Sun_Zen)
    COD = round(COD)
    T_a = round(T_a)
    RH = round(RH)

    surface_v = ['case2']  # name of surface
    AOD_v = np.array([0.1243])  # aerosol optical depth at 479.5 nm
    kap_v = [[10, 11, 12]]
    fileName = "Results_{}_AOD={}_COD={}_kap={}_th0={}_Ta={}_RH={}.npy".format(
        surface_v[0], AOD_v[0], COD, kap_v[0], Sun_Zen, T_a, RH)
    path = os.path.join(file_dir, 'RTM/channels', fileName)
    #if not os.path.exists(path):
    print(fileName)
    run_RTM(Sun_Zen, COD, T_a, RH, file_dir, channels, bandmode, N_bundles)
    out = np.load(path, allow_pickle=True).item()
    return out['F_uw']


def get_RTM_dsw(Sun_Zen, COD, T_a, RH, AOD = None):
    if sys.platform != 'darwin':
        file_dir = '/mnt/dengnan/'
    else:
        file_dir = '/Users/dengnan/Documents/git_store/Shortwave_MCRTM/'
    N_bundles = 1000
    bandmode = 'fullspctrum'

    Sun_Zen = round(Sun_Zen)
    COD = round(COD)
    T_a = round(T_a)
    if RH<1:
        RH= RH*100
    RH = round(RH)
    AOD = round(AOD, 4) if AOD is not None else 0.1243  # default AOD at 479.5 nm

    nu = np.arange(2500, 35000, 3)
    surface_v = ['case2']  # name of surface
    kap_v = [[10, 11, 12]]
    fileName = "Results_{}_AOD={}_COD={}_kap={}_th0={}_Ta={}_RH={}.npy".format(
        surface_v[0], AOD, COD, kap_v[0], Sun_Zen, T_a, RH)
    path = os.path.join(file_dir, 'RTM/fullspectrum/cdf/', fileName)
    if not os.path.exists(path):
        print(path)
        run_RTM(Sun_Zen, COD, T_a, RH, file_dir, '', bandmode, N_bundles, AOD)
    out = np.load(path, allow_pickle=True).item()
    dsw = np.trapz(out['F_dw'],nu)
    usw = np.trapz(out['F_uw'],nu)
    F_dni = np.trapz(out['F_dni'],nu)
    F_dhi = np.trapz(out['F_dhi'],nu)
    return dsw, F_dni, F_dhi

def min_max_nor(pd_data):
    if sys.platform != 'darwin':
        sysdir = '/home/dengnan/SW_RTM/'
    else:
        sysdir = '/Users/dengnan/Documents/git_store/Shortwave_MCRTM/'
    filedir = sysdir + 'FY4A_tool/BON_ABI-L2-CODC_cropped_COD_2019/'
    results = pd.read_csv(filedir + 'min_max_values.csv', index_col=0)

    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    pd_new = pd_data.copy()
    for channel in channels:
        max_v = results.loc[channel, 'Max']
        min_v = results.loc[channel, 'Min']
        # min - max normalization
        pd_new[channel] = (pd_data[channel] - min_v) / (max_v - min_v)
    return pd_new


def density_scatter( x , y, ax = None, sort = True, bins = 50, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        # (1e-2, '#440053'),
        # (0.1, '#404388'),
        (0.2, '#2a788e'),
        (0.5, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    #cmap =  white_viridis #'Spectral_r' #plt.cm.jet #white_viridis
    ax.scatter( x, y, c=z, s=10,alpha=0.8, cmap=white_viridis, **kwargs )  # plt.cm.viridis

    #norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    #cbar = plt.colorbar(cm.ScalarMappable(norm = norm,cmap=cmap), ax=ax,cmap = plt.cm.jet)
    #cbar.ax.tick_params(labelsize=font)
    #cbar.ax.set_ylabel('Density',fontsize=font,fontfamily=fontfml)

    return ax



def plot_data(sat_ref, Rc_rtm_df, channels, VAR, CODfromWhom,figlabel=None):
    font = 13
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = fontfml
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fig = plt.figure(figsize=(11, 6))
    gs1 = gridspec.GridSpec(2, 3)
    gs1.update(wspace=0.18, hspace=0.15)

    for idx, ch in enumerate(channels):
        ax = fig.add_subplot(gs1[idx // 3, idx % 3])
        try :
            x = sat_ref[ch].values
        except KeyError :
            x = sat_ref.loc[ch].values
        y = Rc_rtm_df[ch].values
        #model = LinearRegression().fit(x.reshape(-1, 1), y)
        #y_pred = model.predict(x.reshape(-1, 1))
        #r2 = r2_score(x, y)
        #mae = np.mean(np.abs(x - y))
        mbe = np.mean((x - y))
        rmse = np.sqrt(np.mean((x - y) ** 2))
        #rmae = mae/x.shape[0]/np.sum(x)*100
        rmbe = mbe * x.shape[0]/np.sum(x) *100
        rrmse = rmse *  x.shape[0]/np.sum(x)*100
        try:
            R = np.corrcoef(x, y)[0, 1]
        except Exception:
            R =  np.corrcoef(x, y)
        # bias = np.mean(y - x)
        # slope = model.coef_[0]
        # intercept = model.intercept_
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())

        # Use a visually appealing colormap
        palette = sns.color_palette("viridis", as_cmap=True)

        # Scatter plot with gradient color and better marker aesthetics
        sns.scatterplot(
            x=x, y=y, ax=ax,
            hue=x - y, palette=palette, legend=False,
            edgecolor='w', s=30, alpha=0.8
        )
        # Diagonal reference line with softer color
        ax.plot([min_val * 0.9, max_val * 1.1], [min_val * 0.9, max_val * 1.1], color='gray', linestyle='--', linewidth=1.5)
        #ax.plot(x, y_pred, color='blue', linestyle='-', linewidth=1.5, label='Regression')

        if ch == 'C04':
            ax.set_xlim(min_val -0.005, max_val * 1.1)
            ax.set_ylim(min_val -0.005, max_val * 1.1)

        stats_text = (
        f'MBE: {mbe:.2f}\n'
        f'RMSE: {rmse:.2f}\n'
        f'rMBE: {rmbe:.2f}%\n'
        f'rRMSE: {rrmse:.2f}%\n'
        f'R: {R:.2f}'
        #f'n: {len(x)}'
        #f'R² ={float(r2):.3f}\n'
        #f'Bias = {bias:.3f}'
        )
        print(figlabel, CODfromWhom, '\n', stats_text)

        if ch == 'C04':
            ax.text(0.4, 0.42, stats_text, transform=ax.transAxes, fontsize=12-0.5, verticalalignment='top',weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        elif ch == 'C05':
            ax.text(0.5, 0.42, stats_text, transform=ax.transAxes, fontsize=12 - 0.5, verticalalignment='top',
                    weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12-0.5, verticalalignment='top',weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        if idx == 4:
            ax.set_xlabel(f'{CODfromWhom} UW Radidance at BON [W/(m$^2$ sr)]', fontsize=font, family=fontfml)
        if idx in [0,1,2]:
            ax.set_xticklabels([])
        # Apply the y-axis ticks to the x-axis
        ax.set_xticks(ax.get_yticks())
        ax.set_xlim(min_val * 0.9, max_val * 1.1)
        ax.set_ylim(min_val * 0.9, max_val * 1.1)
        fig.supylabel('Measured UW Radidance at BON [W/(m$^2$ sr)]', fontsize=font, family=fontfml,ha='left', x=0.06)

        ax.grid(color='grey', linestyle='--', linewidth=0.5)
        ax.set_title(f'{ch}', fontsize=font, family=fontfml,pad=2)
        # ax.legend(loc='lower right', fontsize=10)

    figname = './FY4A_validation/' + f'{VAR}_{CODfromWhom}_BON_water_{figlabel}.png'
    fig.savefig(figname, dpi=600, bbox_inches='tight')
    #plt.show()

def expol_func(x, a):
    return a * x**3


def plot_data_dw(site_GHI, GHI, site_DNI, DNI, CODfromWhom, COD, site, figlabel=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    font = 13
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = fontfml
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fig = plt.figure(figsize=(10, 4))
    #gs1 = gridspec.GridSpec(1, 2)
    gs1 = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[0.8, 1], wspace=0.15
    )
    #gs1.update(wspace=0.15, hspace=0.1)
    x_ = [GHI,DNI]
    y_ = [site_GHI,site_DNI]
    for idx in range(2):
        ax = fig.add_subplot(gs1[0, idx % 2])
        x = x_[idx].values
        y = y_[idx].values
        #model = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
        #y_pred = model.predict(x.reshape(-1, 1))
        # r2 = r2_score(x, y)
        # mae = np.mean(np.abs(x - y))
        mbe = np.mean((x - y))
        mse = sklearn.metrics.mean_squared_error(y, x)
        rmse = np.sqrt(mse)
        # rmae = mae/x.shape[0]/np.sum(x)*100
        rmbe = mbe * x.shape[0] / np.sum(x) * 100
        rrmse = rmse * x.shape[0] / np.sum(x) * 100
        R = np.corrcoef(x, y)[0, 1]
        # bias = np.mean(y - x)
        # slope = model.coef_[0]
        # intercept = model.intercept_
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())

        # Use a visually appealing colormap
        palette = sns.color_palette("viridis", as_cmap=True)

        # Scatter plot with gradient color and better marker aesthetics
        sns.scatterplot(
            x=x, y=y, ax=ax,
            hue=x - y, palette=palette, legend=False,
            vmin=0,  # Set the minimum for the color mapping
            vmax=50,
            edgecolor='w', s=30, alpha=0.8
        )
        norm = plt.Normalize(COD.min(), COD.max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        # Diagonal reference line with softer color
        ax.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=1.5)
        #ax.plot(x, y_pred, color='blue', linestyle='-', linewidth=1.5, label='Regression')
        ax.set_xlim(min_val, max_val*1.1)
        ax.set_ylim(min_val, max_val*1.1)
        ax.set_xticks(ax.get_yticks())

        stats_text = (
            # f'R² = {float(r2):.3f}\n'
            # f'Bias = {bias:.3f}'
            f'MBE: {mbe:.2f}\n'
            f'RMSE: {float(rmse):.2f}\n'
            f'rMBE: {rmbe:.2f}%\n'
            f'rRMSE: {rrmse:.2f}%\n'
            f'R ={R:.2f}'
        )
        # print(CODfromWhom, '\n', stats_text)
        save_metric_txt(site, idx, mbe, rmse, rmbe, rrmse, R, file_dir='./FY4A_data/flux/')
        if idx == 0:
            if CODfromWhom == 'FY4A':
                ax.text(0.6, 0.3, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                        weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            else:
                ax.text(0.03, 0.98, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('COD from FY4A', rotation=270, labelpad=15)
            ticks_to_show = [0, 10, 20, 30, 40, 50]
            cbar.set_ticks(ticks_to_show)
            ax.text(0.6, 0.3, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax.set_xlabel(f'{CODfromWhom} [W/(m$^2$)]', fontsize=font, family=fontfml)
        if idx == 0:
            ax.set_title('GHI', fontsize=font,family=fontfml)
            ax.set_ylabel(r'Measured DW [W/(m$^2$)]', fontsize=font, family=fontfml)
        else:
            if CODfromWhom == 'AODc':
                ax.set_title('AOD correction', fontsize=font, family=fontfml)
            else:
                ax.set_title('DNI', fontsize=font, family=fontfml)
        ax.grid(color='grey', linestyle='--', linewidth=0.5)
        # ax.legend(loc='lower right', fontsize=10)
    #figname = './FY4A_validation/' + f'dsw_{CODfromWhom}_BON_water_{figlabel}.png'
    figname = './FY4A_data/flux/' + f'dsw_{CODfromWhom}_{site}_water_{figlabel}.png'
    plt.tight_layout()
    plt.show()
    fig.savefig(figname, dpi=600, bbox_inches='tight')
    ##plt.show()

def save_metric_txt(site, idx, mbe, rmse, rmbe, rrmse, R, file_dir='./FY4A_data/flux/'):
    headers = ['Site', 'MBE', 'RMSE', 'rMBE(%)', 'rRMSE(%)', 'R']
    values = [site, mbe, rmse, rmbe, rrmse, R]
    if idx == 0:
        type = 'GHI'
    else:
        type = 'DNI'
    # 2. Format the numeric values into strings with the desired precision
    formatted_values = [
        f'{v:.2f}' if isinstance(v, (int, float)) else str(v)
        for v in values
    ]

    # 3. Write the headers and values to the file
    filename = file_dir + f'stats_{type}.txt'
    file_exists = os.path.exists(filename)
    with open(filename, 'a') as f:
        if not file_exists:
            header_line = '\t'.join(headers)
            f.write(header_line + '\n')

        value_line = '\t'.join(formatted_values)
        f.write(value_line + '\n')  # Added a newline to ensure next entry is on a new line

        
        
