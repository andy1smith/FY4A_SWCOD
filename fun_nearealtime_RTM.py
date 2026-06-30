
import time

import numpy as np
import pandas as pd
from matplotlib.pyplot import fignum_exists

from SCOPE_func import *
import os,sys
import platform
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

def cal_surface_albedo(sat, surface='MODIS'):
    # calculate white or black albedo from MODIS p1,p2,p3, for each soalr zenith
    black_col = [c for c in sat.columns if c.startswith('BSA_C')]
    white_col = [c for c in sat.columns if c.startswith('WSA_C')]
    df_albedo = sat[black_col + white_col].copy()
    if surface == 'BRDF':  # b, w, p0,p1,p2
        p0_cols = [c for c in sat.columns if c.startswith('Abdo_') and c.endswith('_p0')]
        p1_cols = [c for c in sat.columns if c.startswith('Abdo_') and c.endswith('_p1')]
        p2_cols = [c for c in sat.columns if c.startswith('Abdo_') and c.endswith('_p2')]
        df_albedo = sat[black_col + white_col + p0_cols + p1_cols + p2_cols].copy()
    return df_albedo

def FY4A_calinu(nu, channels, file_dir, dnu = 3, sensor='FY4A'):
    # convert nu to AGRI device nu range. return cm-1.
    nus = set()
    if sensor == 'FY4A' :
        dirpath = './' + 'FY4A_data/AGRI_calibration/'
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
    nu = np.arange(2500, 35000, 3)
    sensor = 'FY4A'
    channel_number = int(channel[-2:])
    dirpath = './' + 'FY4A_data/AGRI_calibration/'
    channel_srf = os.path.join(dirpath,'FY4A_AGRI_SRF_ch{:d}.txt'.format(channel_number))
    calibration = np.loadtxt(channel_srf, delimiter=',', skiprows=1)
    calibration_nu = calibration[:, 1]
    calibration_srf = calibration[:, 2]
    nu_channel = FY4A_calinu(nu, [channel],file_dir, dnu=3)
    calibration_nu = calibration_nu[::-1]
    calibration_srf = calibration_srf[::-1]
    srf = np.interp(nu_channel, calibration_nu, calibration_srf)
    return srf, nu_channel


def replace_sat_band_albedo(
    nu, albedo_spectral,       # original spectral albedo (same size as nu)
    white_albedo,black_albedo,channels,# ['C01','C02','C03','C05','C06']
    file_dir, brdf_p1,brdf_p2,brdf_p3, sensor='FY4A_AGRI',
):
    """
    Replace spectral surface albedo inside and outside FY4A AGRI bandpasses
    using physical boundary band division scaling for gaps.
    """

    nu = np.asarray(nu)
    albedo_spectral = np.asarray(albedo_spectral)
    wsa_albedo_new = albedo_spectral.copy()
    bsa_albedo_new = albedo_spectral.copy()
    in_channel = np.zeros(len(wsa_albedo_new), dtype=np.uint8)
    p1 = np.zeros(len(wsa_albedo_new), dtype=np.uint8)
    p2 = np.zeros(len(wsa_albedo_new), dtype=np.uint8)
    p3 = np.zeros(len(wsa_albedo_new), dtype=np.uint8)
    if sensor == 'FY4A_AGRI':
        dirpath = os.path.join(file_dir, 'AGRI_calibration')
    else:
        raise ValueError(f'Unsupported sensor calibration: {sensor}')

    ratio_w_dict = {}
    ratio_b_dict = {}
    band_masks = {}
    active_channels = []
    missing_channels = []

    ii = 0
    for band_wsa, band_bsa, channel in zip(white_albedo, black_albedo, channels):
        ch_num = int(channel[-2:])
        srf_file = os.path.join(
            dirpath,
            f'FY4A_AGRI_SRF_ch{ch_num:d}.txt'
        )
        calibration = np.loadtxt(srf_file, delimiter=',', skiprows=1)
        calibration_nu = calibration[:, 1]  # cm-1
        calibration_nu = calibration_nu[::-1]
        band_mask = (nu >= calibration_nu.min()) & (nu <= calibration_nu.max())
        band_masks[channel] = band_mask

        if not np.any(band_mask):
            missing_channels.append(channel)
            ii += 1
            continue

        case2_mean = np.mean(albedo_spectral[band_mask])
        if np.isfinite(case2_mean) and case2_mean > 0:
            ratio_w_dict[channel] = band_wsa / case2_mean
            ratio_b_dict[channel] = band_bsa / case2_mean
        else:
            ratio_w_dict[channel] = 1.0
            ratio_b_dict[channel] = 1.0
        active_channels.append(channel)
        ii += 1

    if not active_channels:
        raise ValueError(
            f"No requested FY4A AGRI channels overlap the supplied nu grid. "
            f"Requested channels={channels}."
        )
    if missing_channels:
        warnings.warn(
            "replace_sat_band_albedo skipped channels with no nu overlap: "
            + ", ".join(missing_channels),
            RuntimeWarning,
        )

    wl = 1e4 / nu
    for i, w in enumerate(wl):
        ratio_w = 1.0
        ratio_b = 1.0

        if w < 0.442:
            pass
        elif 0.442 <= w < 0.550:
            ratio_w, ratio_b = ratio_w_dict.get('C01', 1.0), ratio_b_dict.get('C01', 1.0)
        elif 0.550 <= w < 0.730:
            ratio_w, ratio_b = ratio_w_dict.get('C02', 1.0), ratio_b_dict.get('C02', 1.0)
        elif 0.730 <= w < 1.400:
            ratio_w, ratio_b = ratio_w_dict.get('C03', 1.0), ratio_b_dict.get('C03', 1.0)
        elif 1.400 <= w < 1.900:
            ratio_w, ratio_b = ratio_w_dict.get('C05', 1.0), ratio_b_dict.get('C05', 1.0)
        else:
            ratio_w, ratio_b = ratio_w_dict.get('C06', 1.0), ratio_b_dict.get('C06', 1.0)

        wsa_albedo_new[i] = albedo_spectral[i] * ratio_w
        bsa_albedo_new[i] = albedo_spectral[i] * ratio_b

    ii = 0
    for band_wsa, band_bsa, channel in zip(white_albedo, black_albedo, channels):
        ch_num = int(channel[-2:])
        band_mask = band_masks[channel]
        if not np.any(band_mask):
            ii += 1
            continue

        wsa_albedo_new[band_mask] = band_wsa
        bsa_albedo_new[band_mask] = band_bsa
        in_channel[band_mask] = ch_num # if ch_num = C02, C03, set it to 0 (due to BRDF works bad for C02,C03)
        if channel in ['C02', 'C03']:
            in_channel[band_mask] = 0
        p1[band_mask] = brdf_p1[ii]
        p2[band_mask] = brdf_p2[ii]
        p3[band_mask] = brdf_p3[ii]
        ii+=1
    return wsa_albedo_new, bsa_albedo_new, in_channel, p1,p2,p3


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
    df = pd.DataFrame(columns=channels)
    COD_v = np.concatenate([np.linspace(0, 20, 11), np.linspace(25, 50, 6)])
    COD_ = COD_v[np.argmin(abs(COD - COD_v))]
    fdir = "./FY4A_tool/" + 'FY4A_ADMLUT/'
    F_dw_os_srf_channel = [100.56360014402173,293.8703639771758,146.06104052297425,
                           12.06884597258561,13.936208329862962,18.20438461023419]
    #mu0 = np.cos(np.deg2rad(target_zenith))
    Fuw = np.trapz(uw, nu0)
    for i, channel in enumerate(channels):
        # load calibration data : Spectral Response Func
        srf, nu_channel = get_calibration_srf(channel, file_dir)
        # theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
        # U, S, VT = load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD_)}.h5', channel, target_zenith)
        # H_r = reconstruct_hc(U, S, VT)
        nu_idx = np.nonzero(np.isin(nu0, nu_channel))[0]
        uw_cor = np.multiply(uw[nu_idx], srf)
        uw_channel = np.trapz(uw_cor, nu_channel)#/F_dw_os_srf_channel[i]
        
        rho_band = uw_channel/(F_dw_os_srf_channel[i]) # f / f_dw
        #L_band = (mu0 / np.pi* rho_band* (np.trapz(E0_lam * srf, wl) / np.trapz(srf, wl)))

        df.loc[0, channel] = rho_band #* H_r[theta_idx, phi_idx] # W/m2/sr radiance    #/np.pi #

    return df, Fuw

def LUT_wl(Flux_nu, COD, target_zenith, local_zen, rela_azi, file_dir='./FY4A_data/'):
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
    nu = np.arange(2500, 35000, 3)  # Wavenumber grid
    df = pd.DataFrame(columns=channels)
    COD_v = np.concatenate([np.linspace(0, 20, 11), np.linspace(25, 50, 6)])
    COD_ = COD_v[np.argmin(abs(COD - COD_v))]
    fdir = "./FY4A_tool/" + 'FY4A_ADMLUT/'
    F_dw_os_srf_channel = [100.56360014402173,293.8703639771758,146.06104052297425,
                           12.06884597258561,13.936208329862962,18.20438461023419]
    wl = 1e7 / nu[::-1]  # cm-1 to nm
    Flux_lam = Flux_nu * nu ** 2 / 1e7  # cm -> nm 1e7
    Flux_lam = Flux_lam[::-1]

    for i, channel in enumerate(channels):
        sensor = 'FY4A'
        channel_number = int(channel[-2:])
        dirpath = file_dir + 'AGRI_calibration/'
        channel_srf = os.path.join(dirpath, 'FY4A_AGRI_SRF_ch{:d}.txt'.format(channel_number))
        calibration = np.loadtxt(channel_srf, delimiter=',', skiprows=1)
        #calibration_nu = calibration[:, 1]
        calibration_wl = calibration[:, 0]  # wavelength [nm]
        calibration_srf = calibration[:, 2]
        # keep the wavenumber within range
        channel_mask = (wl >= calibration_wl.min()) & (wl <= calibration_wl.max())
        channel_wl = wl[channel_mask]
        srf = np.interp(channel_wl, calibration_wl, calibration_srf)
        idx = np.nonzero(np.isin(wl, channel_wl))  # fixed 1 April.
        uw_cor = np.multiply(Flux_lam[idx], srf)
        uw_channel = np.trapz(uw_cor, channel_wl)#/F_dw_os_srf_channel[i]
        # theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
        # U, S, VT = load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD_)}.h5', channel, target_zenith)
        # H_r = reconstruct_hc(U, S, VT)
        df.loc[0, channel] = uw_channel/np.pi #* H_r[theta_idx, phi_idx] # W/m2/sr radiance    #/np.pi #
    return df

def get_uwrxyz_Rfactor(uw_rxyz_path, Sun_zen, local_zen, rela_azi,
                       surface, N_bundles = 1000):
    '''


    Parameters
    ----------
    uw_rxyz_path
    Sun_zen
    local_zen
    rela_azi

    outputtype
    bandmode
    N_bundles

    Returns
    -------

    '''
    results = np.load(uw_rxyz_path, allow_pickle=True).item()
    uw_rxyz_M = results.get('uw_rxyz_M')

    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    nu0 = np.arange(2500, 35000, 3)  # Wavenumber grid
    R_theta = pd.DataFrame([[1] * len(channels)], columns=channels)
    R_phi = pd.DataFrame([[1] * len(channels)], columns=channels)
    R_2D = pd.DataFrame([[1] * len(channels)], columns=channels)

    data = np.genfromtxt('./data/profiles/ASTMG173.csv', delimiter=',', skip_header=2,  # in wavenumber basis
                         names=['wavelength', 'extraterrestrial', '37tilt', 'direct_circum'])
    ref_lam = data['wavelength'] / 1e3  # nm -> um
    ref_E = data['extraterrestrial'] * 1e3 # W/m2/nm -> W/m2/um
    ref_E_nu = -ref_E * ref_lam ** 2 / 1e4

    for i, channel in enumerate(channels):
        channel_number = int(channel[-2:])
        nu_channel = FY4A_calinu(nu0, [channel], file_dir='./GOES_data/', dnu=3)
        nu_idx = np.nonzero(np.isin(nu0, nu_channel))[0] # fixed 1 April.
        F_dw_os_channel = -np.interp(-nu_channel, -1e4 / ref_lam, ref_E_nu)
        Mrxyz= [uw_rxyz_M[i] for i in nu_idx]
        # if channel in ['C01', 'C02','C06']:
        #     R_2D.loc[0, f"C0{channel_number}"] = anti_iso_factor_2d(Sun_zen, Mrxyz, local_zen, rela_azi, nu_channel,
        #                                                          F_dw_os_channel, N_bundles)
        if surface == 'BRDF':
            if channel in ['C06']: # only channel6 improved.
                r_theta = anti_iso_factor_1d(Sun_zen, Mrxyz, local_zen, rela_azi, nu_channel, F_dw_os_channel,
                                            N_bundles, along_side='theta')
                R_theta.loc[0, f"C0{channel_number}"] = r_theta
        else: # MODIS
            if channel in ['C02']:
                r_theta = anti_iso_factor_1d(Sun_zen, Mrxyz, local_zen, rela_azi, nu_channel, F_dw_os_channel,
                                            N_bundles, along_side='theta')
                R_theta.loc[0, f"C0{channel_number}"] = r_theta
        if channel in ['C01','C02']: # only works for C01,C02
            r_phi = anti_iso_factor_1d(Sun_zen, Mrxyz, local_zen, rela_azi, nu_channel, F_dw_os_channel,
                                     N_bundles, along_side='phi')
            R_phi.loc[0, f"C0{channel_number}"] = r_phi

    # print(f't_sz={Sun_zen}, t_sat={local_zen}, rela_azi={rela_azi}')
    # print('R_2D', R_2D)
    # print('R_theta:', R_theta)
    # print('R_phi', R_phi)
    df = pd.DataFrame(R_2D.values * R_theta.values * R_phi.values, columns=channels)
    return df

def anti_iso_factor_1d(theta0, Mrxyz, local_zen, rela_azi, nu, F_dw_os, N_bundles, along_side='theta'):
    '''
    Calculates the azimuthally-averaged anti-isotropic factor.
    By integrating photons into 1D zenith rings first, it prevents
    Monte Carlo noise from blowing up during solid angle division.
    '''
    if along_side == 'theta':
        d_th = 5 # 2-> 5
        bins_theta = np.arange(0.0, 90.0 + d_th, d_th)
    else: # phi
        d_th = 10
        bins_theta = np.arange(0, 180.0 + d_th, d_th)
        # d_phi = 5
        # bins_phi = np.arange(-180.0, 180.0 + d_phi, d_phi)
    # 1. Collect all valid theta values across all wavelengths (nu)
    H = 0
    for k in range(len(nu)):
        uw_rxyz = np.array(Mrxyz[k])
        if len(uw_rxyz) == 0:
            continue
        theta_v,phi_v = theta_phi_scope(uw_rxyz[:, 0], uw_rxyz[:, 1], uw_rxyz[:, 2])
        if along_side != 'theta':
            theta_v = phi_v
        # Filter NaNs and append
        valid_theta = theta_v[~np.isnan(theta_v)]

        # 2. Create a 1D Histogram (Summing all phi naturally)
        if along_side != 'theta':
            v_deg = np.rad2deg(valid_theta)
            v_deg = v_deg % 360
            v_deg[v_deg > 180] = 360 - v_deg[v_deg > 180]
            H_theta, theta_ = np.histogram(v_deg, bins=bins_theta)
        else:
            H_theta, theta_ = np.histogram(np.rad2deg(valid_theta), bins=bins_theta)

        ratio = F_dw_os[k]*3 * np.cos(theta0) / 1000
        H += H_theta * ratio  # Flux W/m2
    ths = np.deg2rad(theta_.T + d_th / 2)  # rad dw # division 2 for the 2sintcost
    # Rad is L(theta,phi) = Histogram / (2*pi *0.5 * sin(2*theta) * d_theta_rad)
    if along_side == 'theta':
        L = H / (np.sin(2 * ths[:-1]) * np.deg2rad(d_th)) # /0.5*2pi  W/m2/sr
    else:
        L = H / np.deg2rad(d_th) * np.pi
    # L_ios = sum(H_theta)/pi
    L_iso = np.sum(H) # N_uw_tot: W/m2  /pi
    if L_iso == 0:
        return np.nan  # Safety check if no photons made it to TOA

    R_theta = L / L_iso # N_bin/N_uw_tot
    # 3. Geometry Setup
    theta_centers = 0.5 * (bins_theta[:-1] + bins_theta[1:])

    # check Normalization.
    theta = np.deg2rad(theta_centers)
    dth = np.deg2rad(d_th)

    # 5. Interpolate at the exact satellite viewing angle
    if along_side == 'theta':
        # normalization check
        check = np.sum(R_theta * np.cos(theta) * np.sin(theta) * dth)
        if abs(check - 0.5) > 0.05:
            print("R_theta is not normalized, ={}".format(check))
        R = np.interp(local_zen, theta_centers, R_theta.T)
    else:
        check = np.sum(R_theta * dth)
        if abs(check - np.pi) > 0.05:
            print("R_phi is not normalized, ={}".format(check))

        rela_azi = rela_azi % 360
        if np.isscalar(rela_azi):
            if rela_azi > 180:
                rela_azi = 360 - rela_azi
        else:
            rela_azi[rela_azi > 180] = 360 - rela_azi[rela_azi > 180]

        R = np.interp(rela_azi, theta_centers, R_theta.T)
    return R

def ChConvert_nu(uw, COD, target_zenith, local_zen, rela_azi, file_dir='./GOES_data/'):
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    nu0 = np.arange(2500, 35000, 3)  # Wavenumber grid
    nu_channels = FY4A_calinu(nu0, channels, "./FY4A_data/", dnu=3)
    df = pd.DataFrame(columns=channels)
    COD_v = np.concatenate([np.linspace(0, 20, 11), np.linspace(25, 50, 6)])
    COD_ = COD_v[np.argmin(abs(COD - COD_v))]
    fdir = "./FY4Adata/" + 'LUT/'
    F_dw_os_ch = [100.56360014402173,293.8703639771758,146.06104052297425,
                       12.06884597258561,13.936208329862962,18.20438461023419]

    for i, channel in enumerate(channels):
        # load calibration data : Spectral Response Func
        channel_number = int(channel[-2:])
        dirpath = file_dir + 'AGRI_calibration/'
        channel_srf = os.path.join(dirpath, 'FY4A_AGRI_SRF_ch{:d}.txt'.format(channel_number))
        calibration = np.loadtxt(channel_srf, delimiter=',', skiprows=1)
        calibration_nu = calibration[:, 1]
        calibration_srf = calibration[:, 2]
        nu_channel = FY4A_calinu(nu0, [channel], file_dir, dnu=3)
        calibration_nu = calibration_nu[::-1]
        calibration_srf = calibration_srf[::-1]
        srf = np.interp(nu_channel, calibration_nu, calibration_srf)
        #eqw = np.trapz(srf, nu_channel)
        # theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
        # U, S, VT = load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD_)}.h5', channel, target_zenith)
        # H_r = reconstruct_hc(U, S, VT)
        nu_idx = np.nonzero(np.isin(nu_channels, nu_channel))[0]  # fixed 1 April.
        # correct uw
        uw_cor = np.multiply(uw[nu_idx], srf)
        uw_channel = np.trapz(uw_cor,nu_channel) # flux
        # flux / Fdwos
        df.loc[0, channel] =  uw_channel/(F_dw_os_ch[i])#* H_r[theta_idx, phi_idx]
    return df

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
    df_ref = pd.DataFrame([
        {**df_row.to_dict(), 'COD_v': cod} for cod in COD_v
    ])
    df_flux = df_ref.copy()

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
        df_flux.loc[i, channels] = (df_ref.loc[i, channels] / H_r_series) * np.pi


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

def nearealtime_LUT(sun_zen, local_zen, rela_azi, COD_guess, T_s, RH, file_dir, bandmode,
                    df_albedo=None, surface='case2', meth='HG', AOD=0.1243):
    # Round values to two decimal places
    N_bundles = 1000
    sun_zen = round(sun_zen)
    local_zen = round(local_zen)
    rela_azi = round(rela_azi)
    COD_guess = round(COD_guess)
    T_s = round(T_s) # K
    RH = round(RH) # %
    rh = RH/100
    rh = round(rh, 2)
    AOD = round(AOD, 2) if AOD is not None else 0.1243
    channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    if sys.platform != 'darwin':
        file_dir = '/mnt/dengnan/'
    else:
        file_dir = './'
    flux_file = f"Results_{surface}_AOD={AOD:.2f}_COD={COD_guess}_kap=[10, 11, 12]_th0={sun_zen}_Ts={T_s}_RH={RH}.npy"
    if N_bundles == 1000:
        if bandmode == 'FY4A':
            uw_path = os.path.join(file_dir, f'RTM/channels/{meth}/', flux_file)
        else:
            uw_path = os.path.join(file_dir, f'RTM/fullspectrum/{meth}/', flux_file)

    if not os.path.exists(uw_path):
        print(f"File {flux_file} not found. Running RTM...")
        # Use provided albedo, or fall back to zeros for case2
        _albedo = df_albedo if df_albedo is not None else np.zeros(10)
        run_RTM(sun_zen, COD_guess, T_s, rh, _albedo, surface, file_dir, channels, bandmode,
                meth=meth, N_bundles=N_bundles, AOD=AOD)
    results = np.load(uw_path, allow_pickle=True).item()
    uw = results.get('F_uw') # flux in nu
    df_uw = ChConvert_nu(uw, COD_guess, sun_zen, local_zen, rela_azi)
    #df_uw_channels = df_uw.mul(df_R, axis=1)
    return df_uw


def nearealtime_RTM(sun_zen, local_zen, rela_azi, COD_guess, T_s, RH, channels, file_dir, bandmode, N_bundles):
    # Round values to two decimal places
    sun_zen = round(sun_zen)
    local_zen = round(local_zen)
    rela_azi = round(rela_azi)
    COD_guess = round(COD_guess)
    T_s = round(T_s)
    rh = round(RH)/100
    
    uw_rxyz_file = f"uwxyzr_COD={COD_guess}_th0={sun_zen}_Ts={T_s}_RH={rh*100}.npy"
    #bandmode = 'GOES' # GOES
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
        run_RTM(sun_zen, COD_guess, T_s, rh, file_dir, channels, bandmode, N_bundles)
    results = np.load(uw_rxyz_path, allow_pickle=True).item()
    uw_rxyz_M = results.get('uw_rxyz_M')
    #df_channel_ref1 = RTM_preprocess(uw_rxyz_M, sun_zen, local_zen, rela_azi, channels, file_dir, 'rad', 'full')
    df_channel_ref = RTM_preprocess(uw_rxyz_M, sun_zen, local_zen, rela_azi, channels,
                                      file_dir, 'rad', bandmode, N_bundles)

    return df_channel_ref


def run_FY4A_in_RTM(sun_zen, local_zen, rela_azi, COD_guess, T_s, RH, channels, file_dir, bandmode, N_bundles):
    # Round values to two decimal places
    sun_zen = round(sun_zen)
    local_zen = round(local_zen)
    rela_azi = round(rela_azi)
    COD_guess = round(COD_guess)
    T_s = round(T_s)
    rh = round(RH)/100

    uw_rxyz_file = f"uwxyzr_COD={COD_guess}_th0={sun_zen}_Ts={T_s}_RH={RH*100}.npy"
    # bandmode = 'GOES' # GOES
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
        run_RTM(sun_zen, COD_guess, T_s, rh, file_dir, channels, bandmode, N_bundles)
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

def run_RTM(sun_zen, COD_guess, T_s, rh, df_albedo, surface, file_dir, channels,
            bandmode, meth='HG',N_bundles=1000, AOD=None, Save_rxyz=False,
            theta_trunc_cld=None, escape_alpha=None, escape_probability_mode=None):
    Ph_cdf_cld = False
    Ph_cdf_aer = False
    if meth == 'dM':
        deltaM = True
    elif meth == 'dMcdf':
        deltaM = True
        Ph_cdf_aer = True
    elif meth == 'HG':
        deltaM = False
    elif meth == 'TTHG':
        deltaM = False
        print('TTHG, set it on in scatter')
    else:
        print('no method set, default = HG')
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
    # Define a mapping
    SURFACE_TYPES = {'Lambert': 0, 'CSP': 1, 'BRDF': 2, 'MODIS': 3, 'Case2': 0}
    white_albedo = [0, 0, 0, 0, 0]
    black_albedo = [0, 0, 0, 0, 0]
    BRDF_param = [0] * 15
    if surface == 'MODIS':# or surface == 'Case2':
        if df_albedo is not None:
            black_albedo = df_albedo[:5].tolist()
            white_albedo = df_albedo[5:10].tolist()
        else:
            black_albedo = [0.2, 0.2, 0.2, 0.2, 0.2]
            white_albedo = [0.2, 0.2, 0.2, 0.2, 0.2]
        surface_id = SURFACE_TYPES.get(surface, 0)
    if surface == 'BRDF':  # b, w, p0,p1,p2
        if df_albedo is not None:
            black_albedo = df_albedo[:5].tolist()
            white_albedo = df_albedo[5:10].tolist()
            BRDF_param = df_albedo[10:].tolist()
        else:
            black_albedo = [0.2, 0.2, 0.2, 0.2, 0.2]
            white_albedo = [0.2, 0.2, 0.2, 0.2, 0.2]
            BRDF_param = [0] * 15
        surface_id = SURFACE_TYPES.get(surface, 0)
    if surface == 'Case2':
        surface_id = SURFACE_TYPES.get(surface, 0)
    surface_v=[surface] # name of surface
    surface_id_v=[surface_id]

    rh0_v = np.array([rh]) #0-1
    T_surf_v = np.array([T_s]) # K
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
    # update 5, based on surfrad DNI device.
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
        if sys.platform != 'darwin':
            machine_name = platform.node()
            if machine_name == 'user-Super-Server':
                file_dir = '/tmp_rtm_file/'
            if machine_name == 'user-MS-7D30':
                file_dir = '/mnt/dengnan/'
        else:
            file_dir = './'
        if bandmode == 'FY4A':
            file_dir+=f'RTM/channels/{meth}/'
        else:
            if surface == 'MODIS':
                file_dir += f'RTM/fullspectrum/MODIS/{meth}/'
            else:
                file_dir+=f'RTM/fullspectrum/{meth}/'
    elif N_bundles == 10000:
        if sys.platform != 'darwin':
            machine_name = platform.node()
            if machine_name == 'user-Super-Server':
                file_dir = '/home/dengnan/data/'
                if bandmode == 'FY4A':
                    file_dir+='RTM_10000/channels/'
                else:
                    print('dir error')
            if machine_name == 'user-MS-7D30':
                file_dir = '/mnt/dengnan/'
        if bandmode == 'FY4A':
            file_dir+='RTM_10000/channels/FY4A/'
        else:
            file_dir = './'
            if bandmode == 'FY4A':
                file_dir+=f'RTM/RTM_10000/channels/{meth}/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        print(f"Created path: '{file_dir}'")
    
    # compute case by case
    for iSurf in range(0,len(surface_v)):
        inputs_main={'N_layer':N_layer, 'N_bundles':N_bundles, 'nu':nu, 'molecules':molecules,'vmr0':vmr0,
           'model':model,'cld_model':cld_model,'period':period,'spectral':spectral,'surface_id':surface_id_v[iSurf],
                     'white_albedo':white_albedo, 'black_albedo':black_albedo,'BRDF_param':BRDF_param,
                     'alt':alt, 'Ph_cdf_cld':Ph_cdf_cld,'Ph_cdf_aer':Ph_cdf_aer,'deltaM':deltaM
                     }
        if theta_trunc_cld is not None:
            inputs_main['theta_trunc_cld'] = theta_trunc_cld
        if escape_alpha is not None:
            inputs_main['escape_alpha'] = escape_alpha
        if escape_probability_mode is not None:
            inputs_main['escape_probability_mode'] = escape_probability_mode
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
                                    finitePP={'x0':-x0_v[iTH]+dx_v[idx],'y0':-y0_v[iTH],'R_pp':R_pp,'is_pp':is_pp,
                                              'th0':theta0_v[iTH], 'phi0':phi0, 'del_angle':del_angle}
                                    print ("Start MonteCarlo once.")
                                    start_time = time.time()
                                    out1,out2 = LBL_shortwave(properties,inputs_main,angles,finitePP)
                                    end_time = time.time()
                                    print ("CPU time:", end_time - start_time)
                                    #del out1, out3
                                    if N_bundles == 1000:
                                        fileName1="Results_{}_AOD={:.2f}_COD={}_kap={}_th0={}_Ts={}_RH={}".format(
                                            surface_v[iSurf],AOD_v[iAOD],COD_v[iCOD],kap_v[iKAP],th0_v[iTH], T_surf_v[iT], int(rh0_v[iRH]*100))
                                        np.save(file_dir+fileName1,out1)# save results to local directory
                                    if Save_rxyz:
                                        fileName2 = "uwxyzr_{}_AOD={:.2f}_COD={}_th0={}_Ts={}_RH={}.npy".format(
                                            surface_v[iSurf],AOD_v[iAOD],COD_v[iCOD], th0_v[iTH], T_s, int(rh0_v[iRH]*100))
                                        np.save(file_dir + fileName2, out2)  # save results to local directory
                                        print(file_dir + fileName2)
                                    del out1, out2
                                    return None


def get_RTM_usw(Sun_Zen, COD, T_s, RH, bandmode='FY4A'):
    file_dir = './FY4A_data/'
    if sys.platform != 'darwin':
        file_dir = '/mnt/dengnan/'
    N_bundles = 1000
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']

    Sun_Zen = round(Sun_Zen)
    COD = round(COD)
    T_s = round(T_s)
    rh = round(RH)/100

    surface_v = ['case2']  # name of surface
    AOD_v = np.array([0.1243])  # aerosol optical depth at 479.5 nm
    kap_v = [[10, 11, 12]]
    fileName = "Results_{}_AOD={:.2f}_COD={}_kap={}_th0={}_Ts={}_RH={}.npy".format(
        surface_v[0], AOD_v[0], COD, kap_v[0], Sun_Zen, T_s, RH*100)
    path = os.path.join(file_dir, 'RTM/channels', fileName)
    #if not os.path.exists(path):
    print(fileName)
    run_RTM(Sun_Zen, COD, T_s, rh, file_dir, channels, bandmode, N_bundles)
    out = np.load(path, allow_pickle=True).item()
    return out['F_uw']


def get_rtm_output(Sun_Zen, local_zen, rela_azi, COD, T_s, RH, df_albedo,
                    surface, meth='HG', AOD = None):
    if sys.platform != 'darwin':
        machine_name = platform.node()
        if machine_name == 'user-Super-Server':
            file_dir = '/tmp_rtm_file/'
        if machine_name == 'user-MS-7D30':
            file_dir = '/mnt/dengnan/'
    else:
        file_dir = '/Users/dengnan/Documents/git_store/FY4A_SWCOD/'
    N_bundles = 1000
    bandmode = 'fullspctrum'

    Sun_Zen = round(Sun_Zen)
    COD = round(COD)
    if T_s < 200:
        T_s = T_s + 273.15  # to K
    T_s = round(T_s)
    if RH>1:
        rh= round(RH)/100
    AOD = round(AOD, 2) if AOD is not None else 0.1243  # default AOD at 479.5 nm

    nu = np.arange(2500, 35000, 3)
    surface_v = [surface]  # name of surface
    kap_v = [[10, 11, 12]]

    fileName1 = "Results_{}_AOD={:.2f}_COD={}_kap={}_th0={}_Ts={}_RH={}.npy".format(
        surface_v[0], AOD, COD, kap_v[0], Sun_Zen, T_s, int(rh*100))
    fileName2 = "uwxyzr_{}_AOD={:.2f}_COD={}_th0={}_Ts={}_RH={}.npy".format(
        surface_v[0], AOD, COD, Sun_Zen, T_s, int(rh*100))

    path1 = os.path.join(file_dir, f'RTM/fullspectrum/{meth}/', fileName1)
    path2 = os.path.join(file_dir, f'RTM/fullspectrum/{meth}/', fileName2)
    if not os.path.exists(path1):
        print(path1)
        run_RTM(Sun_Zen, COD, T_s, rh, df_albedo, surface, file_dir, '', bandmode,
                meth, N_bundles, AOD, Save_rxyz=True)
    # 1D output
    out1 = np.load(path1, allow_pickle=True).item()
    dsw = np.trapz(out1['F_dw'],nu)
    usw = out1['F_uw']
    uw_srf = np.trapz(out1['F_uw_srf'],nu)
    F_dni = np.trapz(out1['F_dni'],nu)
    F_dhi = np.trapz(out1['F_dhi'],nu)

    # 2D output, get the corrected radiance.
    df_R = get_uwrxyz_Rfactor(path2, Sun_Zen, local_zen, rela_azi,surface,
                                N_bundles=N_bundles)
    return dsw, F_dni, F_dhi, usw, uw_srf, df_R

def get_rtm_output_cld(Sun_Zen, local_zen, rela_azi, COD, T_s, RH, df_albedo, surface, meth='HG', AOD = None, nu_grid_mode='solarspectrum'):
    if sys.platform != 'darwin':
        machine_name = platform.node()
        if machine_name == 'user-Super-Server':
            file_dir = '/tmp_rtm_file/'
        if machine_name == 'user-MS-7D30':
            file_dir = '/mnt/dengnan/'
    else:
        file_dir = '/Users/dengnan/Documents/git_store/Shortwave_MCRTM/'
    N_bundles = 1000
    bandmode = 'fullspctrum'

    Sun_Zen = round(Sun_Zen)
    COD = round(COD)
    if T_s < 200:
        T_s = T_s + 273.15  # to K
    T_s = round(T_s)
    if RH>1:
        rh= RH/100
    rh = round(rh,2)
    AOD = round(AOD, 2) if AOD is not None else 0.1243  # default AOD at 479.5 nm

    nu = np.arange(2500, 35000, 3)
    surface_v = [surface]  # name of surface
    kap_v = [[10, 11, 12]]

    if nu_grid_mode == 'GOES1000':
        fileName1 = "Results_{}_AOD={:.2f}_COD={}_kap={}_th0={}_Ts={}_RH={}_GOES1000.npy".format(
            surface_v[0], AOD, COD, kap_v[0], Sun_Zen, T_s, rh*100)
        fileName2 = "uwxyzr_{}_AOD={:.2f}_COD={}_th0={}_Ts={}_RH={}_GOES1000.npy".format(
            surface_v[0], AOD, COD, Sun_Zen, T_s, rh*100)
    else:
        fileName1 = "Results_{}_AOD={:.2f}_COD={}_kap={}_th0={}_Ts={}_RH={}.npy".format(
            surface_v[0], AOD, COD, kap_v[0], Sun_Zen, T_s, int(rh*100))
        fileName2 = "uwxyzr_{}_AOD={:.2f}_COD={}_th0={}_Ts={}_RH={}.npy".format(surface_v[0], AOD, COD, Sun_Zen, T_s, int(rh*100))

    path1 = os.path.join(file_dir, f'RTM/fullspectrum/{meth}/', fileName1)
    path2 = os.path.join(file_dir, f'RTM/fullspectrum/{meth}/', fileName2)
    if surface == 'MODIS':
        print('surface',surface)
        path1 = os.path.join(file_dir, f'RTM/fullspectrum/MODIS/{meth}/', fileName1)
        path2 = os.path.join(file_dir, f'RTM/fullspectrum/MODIS/{meth}/', fileName2)
    if not os.path.exists(path1):
        print(path1)
        run_RTM(Sun_Zen, COD, T_s, rh, df_albedo, surface, file_dir, '', bandmode, meth, N_bundles, AOD, nu_grid_mode)
    # 1D output
    out1 = np.load(path1, allow_pickle=True).item()
    dsw = np.trapz(out1['F_dw'],nu)
    F_dni = np.trapz(out1['F_dni'],nu)
    F_dhi = np.trapz(out1['F_dhi'],nu)

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

def plot_data(sat_ref, Rc_rtm_df, channels, VAR, CODfromWhom, site_zen,surface, meth='HG',figlabel=None, cbarname='Site_zen'):
    '''

    Parameters
    ----------
    sat_ref : W/m2/sr/um
    Rc_rtm_df : W/m2/sr/um
    channels 0--6
    VAR
    CODfromWhom
    site_zen
    meth
    figlabel

    Returns
    -------

    '''

    font = 13
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = fontfml
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fig = plt.figure(figsize=(12, 6))
    gs1 = gridspec.GridSpec(2, 3)
    gs1.update(wspace=0.18, hspace=0.22, right=0.9)

    # To this:
    zen_values = np.array(site_zen, dtype=float)
    
    norm = plt.Normalize(zen_values.min(), zen_values.max())   # zen_values.min(), zen_values.max()
    cmap = "viridis"
    # Create the ScalarMappable for the Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for idx, ch in enumerate(channels):
        ax = fig.add_subplot(gs1[idx // 3, idx % 3])
        try :
            x = sat_ref[ch].values
        except KeyError :
            x = sat_ref.loc[ch].values
        y = Rc_rtm_df[ch].values

        # Extract stats
        mbe = np.mean((y - x))
        rmse = np.sqrt(np.mean((y - x) ** 2))
        #rmae = mae/x.shape[0]/np.sum(x)*100
        rmbe = mbe * x.shape[0]/np.sum(x) *100
        rrmse = rmse *  x.shape[0]/np.sum(x)*100
        try:
            R = np.corrcoef(x, y)[0, 1]
        except Exception:
            R =  np.corrcoef(x, y)

        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())

        sns.scatterplot(
            x=x, y=y, ax=ax,
            hue=zen_values,       # Color by Zenith
            hue_norm=norm,        # Force consistent color scaling
            palette=cmap,
            legend=False,         # Disable individual legends
            edgecolor='w', s=30, alpha=0.8
        )

        ax.plot([min_val * 0.9, max_val * 1.1], [min_val * 0.9, max_val * 1.1],
                color='gray', linestyle='--', linewidth=1.5)

        # Axis Limits Specifics
        if ch == 'C04':
            ax.set_xlim(0, max_val * 1.1)
            ax.set_ylim(0, max_val * 1.1)
        else:
            # Default limits logic
            ax.set_xlim(min_val * 0.9, max_val * 1.1)
            ax.set_ylim(min_val * 0.9, max_val * 1.1)

        stats_text = (
        #f'n: {len(x)}\n'
        # f'MBE: {mbe:.2f}\n'
        # f'RMSE: {rmse:.2f}\n'
        f'rMBE: {rmbe:.2f}%\n'
        f'rRMSE: {rrmse:.2f}%\n'
        f'R: {R:.2f}'
        #f'R² ={float(r2):.3f}\n'
        #f'Bias = {bias:.3f}'
        )
        print(figlabel, CODfromWhom, '\n', stats_text)

        # if ch in ['C05','C06']:
        #     text_x, text_y = 0.54, 0.42
        # else:
        text_x, text_y = 0.02, 0.98

        ax.text(text_x, text_y, stats_text, transform=ax.transAxes, fontsize=12-0.5,
                    verticalalignment='top', weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        if idx == 4:
            #ax.set_xlabel(f'Measured UW Radidance at {site} [W/(m$^2$ sr)]', fontsize=font, family=fontfml)
            ax.set_xlabel('FY4A/AGRI Reflectance Factor', fontsize=font, family=fontfml)


        # ax.set_xticks(ax.get_yticks()) # Ensure square ticks if desired
        ax.grid(color='grey', linestyle='--', linewidth=0.5)
        ax.set_title(f'{ch}', fontsize=font, family=fontfml,pad=2)

    fig.text(0.13, 0.91, f'n: {len(sat_ref)}',
             fontsize=12-0.5, weight='bold', ha='left', va='top')
    # --- 4. Global Y-Label ---
    fig.supylabel(#f'{CODfromWhom} UW Radiance [W/(m$^2$ um)]',
                f'{CODfromWhom} Reflectance factor',
                  fontsize=font, family=fontfml,
                  ha='center',  # 'center' alignment is usually easier to control than 'left'
                  va='center',
                  x=0.07)

    # --- 5. Add Global Colorbar ---
    # Create a new axes for the colorbar on the right side of the figure
    # [left, bottom, width, height] in figure coordinate fractions
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    cbar = fig.colorbar(sm, cax=cax)
    if cbarname=='Site_zen':
        cbar.set_label('Solar Zenith Angle [°]', rotation=270, labelpad=20, fontsize=font, family=fontfml)
    else:
        cbar.set_label('Relative Azimuth Angle [°]', rotation=270, labelpad=20, fontsize=font, family=fontfml)
        #cbar.set_label('month', rotation=270, labelpad=20, fontsize=font, family=fontfml)
    figname = './FY4A_validation/' + f'{VAR}_{CODfromWhom}_{figlabel}_{meth}_{surface}_{cbarname}.png'
    fig.savefig(figname, dpi=600, bbox_inches='tight')
    plt.show()

def expol_func(x, a):
    return a * x**3


def plot_data_dw_clear(site_GHI, GHI, CODfromWhom, site_zen, site, figlabel=None, meth='HG'):
    font = 13
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = fontfml
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fig = plt.figure(figsize=(6, 5))  # Slightly wider to accommodate colorbar
    gs1 = gridspec.GridSpec(
        1, 2, figure=fig,width_ratios=[1, 0.03],  wspace=0.2,
        bottom=0.2,
    )
    y_ = [GHI]
    x_ = [site_GHI]
    # Prepare color mapping for Zenith (0 to 90 degrees)
    zen_values = site_zen if isinstance(site_zen, (np.ndarray, list)) else site_zen.values
    palette = sns.color_palette("viridis", as_cmap=True)
    norm = plt.Normalize(10, 65)  # Zenith usually goes up to 90
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    global_min = min(x_[0].min(), y_[0].min()) - 50
    global_max = max(x_[0].max(), y_[0].max()) + 50
    for idx in range(1):
        # Assign plot to specific grid column
        ax = fig.add_subplot(gs1[0,idx])

        x = x_[idx].values
        y = y_[idx].values
        # Calculate Metrics
        mbe = np.mean((x - y))
        mse = sklearn.metrics.mean_squared_error(y, x)
        rmse = np.sqrt(mse)
        rmbe = mbe * x.shape[0] / np.sum(x) * 100
        rrmse = rmse * x.shape[0] / np.sum(x) * 100
        R = np.corrcoef(x, y)[0, 1]

        # SOLUTION FOR COLOR: hue = zen_values
        sns.scatterplot(
            x=x, y=y, ax=ax,
            hue=zen_values, palette=palette, legend=False,
            hue_norm=norm,  # Ensures 0 is purple, 90 is yellow
            edgecolor='w', s=30, alpha=0.8
        )

        # Diagonal reference line
        ax.plot([global_min, global_max], [global_min, global_max], color='gray', linestyle='--', linewidth=1.5)
        ax.set_xlim([global_min, global_max])
        ax.set_ylim([global_min, global_max])
        #ax.set_xticks(ax.get_yticks())

        stats_text = (
            f'MBE: {mbe:.2f}\n'
            f'RMSE: {float(rmse):.2f}\n'
            f'rMBE: {rmbe:.2f}%\n'
            f'rRMSE: {rrmse:.2f}%\n'
            f'R ={R:.2f}'
        )

        ax.text(0.03, 0.98, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                    weight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax.set_ylabel(f'{CODfromWhom} [W/(m$^2$)]', fontsize=font, family=fontfml)

        ax.set_title('GHI', fontsize=font, family=fontfml)
        ax.set_xlabel(r'Measured GHI [W/(m$^2$)]', fontsize=font, family=fontfml)
        ax.grid(color='grey', linestyle='--', linewidth=0.5)

    # ADD COLORBAR IN THE 3RD COLUMN
    cax = fig.add_subplot(gs1[0,1])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Solar Zenith Angle [°]', rotation=270, labelpad=15)
    plt.tight_layout()
    figname = './FY4A_validation/' + f'dsw_{site}_{CODfromWhom}_{figlabel}_{meth}.png'
    fig.savefig(figname, dpi=600, bbox_inches='tight')
    #plt.tight_layout() # Careful with tight_layout when using explicit GridSpec ratios
    plt.show()

def plot_data_dw(site_GHI, GHI, CODfromWhom, COD, site, figlabel=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    font = 13
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = fontfml
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fig = plt.figure(figsize=(6, 4))
    #gs1 = gridspec.GridSpec(1, 2)
    gs1 = gridspec.GridSpec(
        1, 1, figure=fig, width_ratios=[0.8, 1], wspace=0.15
    )
    #gs1.update(wspace=0.15, hspace=0.1)
    x_ = [GHI]
    y_ = [site_GHI]
    for idx in range(1):
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


