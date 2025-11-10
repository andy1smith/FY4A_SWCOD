"""
Common functions for both shortwave (SW) and longwave (LW) radiation transfer in the atmosphere.

Author: Mengying Li

Modified: David P. Larson during SCOPE project.
"""
import numpy as np
import os
import math
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.special import jv, yv
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.special import legendre
import deltaM

__all__ = [
    "set_pressure",
    "goes_calinu",
    "set_temperature_default",
    "set_temperature", # include temperature shift and inversion
    "set_height",
    "set_ndensity",
    "set_vmr",
    "saturation_pressure",
    "total_precipitable_water",
    "getMixKappa",
    "absorptionContinuum_MTCKD_H2O",
    "absorptionContinuum_MTCKD_CO2",
    "absorptionContinuum_MTCKD_O3",
    "absorptionContinuum_MTCKD_O2",
    "radiation_field",
    "Planck",
    "Planck_lam",
    "Mie",
    "Mie_ab",
    "MieS1S2",
    "MiePiTau",
    "AutoMie_ab",
    "LowFrequencyMie_ab",
    "Miescat_ab",
    "aerosol_monoLam",
    "aerosol",
    "cloud_efficiency",
    "cloud",
    "rayleigh_kappa_s",
    "surface_albedo",
    "airMass",
    "set_vmr_circ",
    "aerosol_circ",
    "cloud_height",
    "tthg_fit_loss",
    "tthg_phase",
    "hg_phase",
    "TTHG_fitting",
    "phaseFunction",
    "deltaM_phasefunc",
]

def goes_calinu(nu, channels, file_dir, dnu = 3):
    nus = set()
    dirpath = file_dir+'GOES-R_ABI_FM2_SRF_CWG/'
    for channel in channels:
        # load ABI calibration data
        channel_number = int(channel[-2:])
        channel_srf = os.path.join(
            dirpath,
            'GOES-R_ABI_FM2_SRF_CWG_ch{}.txt'.format(channel_number)
        )
        calibration = np.genfromtxt(channel_srf, skip_header=2)
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

def set_pressure(N_layer):
    """
    Set the pressure of the N layers.

    Parameters
    ----------
    N_layer : int
        The number of atmosphere layers.

    Returns
    -------
    p : (N + 2,) array_like
        Presure [Pa] at each node point.
    pa : (N + 1,) array_like
        Average pressure [Pa] of each layer.

    """

    n = np.arange(0.5, N_layer + 1, 0.5)
    sig = (2 * N_layer - 2 * n + 1) / 2 / N_layer
    pa_temp = sig ** 2 * (3 - 2 * sig)

    # initialize, has a layer inside the Earth
    p = np.zeros(N_layer + 2)
    pa = np.zeros(N_layer + 1)
    for i in range(N_layer, 0, -1):
        p[i] = pa_temp[2 * i - 2]
        pa[i] = pa_temp[2 * i - 1]

    # ground layers
    p[0] = p[1]
    pa[0] = p[0]

    # convert to [Pa]
    p *= 1.013e5
    pa *= 1.013e5
    return p, pa


def set_temperature_default(model, p, pa):
    """
    Set temperature of the N layers based on AFGL profile.

    With N atmosphere layers, there are N + 2 node points.

    Parameters
    ----------
    model : str
        Profile model (e.g. 'tropical').
    p : (N + 2,) array_like
        Presure [Pa] at each node point.
    pa : (N + 1,) array_like
        Average pressure [Pa] of each layer.

    Returns
    -------
    t : (N + 2,) array_like
        Temperature [K] at each node point.
    ta : (N + 1,) array_like
        Average temperature [K] of each layer.

    References
    ----------
    [1] G. P. Anderson, "AFGL atmospheric consituent profiles (0-120 km)",
        1986.

    """

    # data from AFGL Atmospheric Constituent Profiles (1986)
    data = np.genfromtxt("data/profiles/{}.csv".format(model), delimiter=",")
    ref_p = data[:, 1]
    ref_t = data[:, 2]
    ref_p = np.asarray(ref_p) * 1e2  # convert unit to Pa
    ref_t = np.asarray(ref_t)
    t = np.interp(-p, -ref_p, ref_t)
    ta = np.zeros((len(pa)))

    # pressure averaged temperature
    for i in range(1, len(pa)):
        ta[i] = (t[i] * (p[i] - pa[i]) + t[i + 1] * (pa[i] - p[i + 1])) / (
            p[i] - p[i + 1]
        )
    ta[0] = t[0]

    return t, ta

def set_temperature(model, p, pa, T_surf, period):
    """
    Set temperature of the N layers including temperature shift according to surface temperature and period 'day' or 'night'.

    With N atmosphere layers, there are N + 2 node points.

    Parameters
    ----------
    model : str
        Profile model (e.g. 'tropical').
    p : (N + 2,) array_like
        Presure [Pa] at each node point.
    pa : (N + 1,) array_like
        Average pressure [Pa] of each layer.
    T_surf: float
        Measured surface temperature [K], set to NaN if using standard AFGL profile.
    period: str
        'day' or 'night', night profile includes temperature inversion.
    Returns
    -------
    t : (N + 2,) array_like
        Temperature [K] at each node point.
    ta : (N + 1,) array_like
        Average temperature [K] of each layer.

    References
    ----------
    [1] G. P. Anderson, "AFGL atmospheric consituent profiles (0-120 km)",
        1986.

    """

    # data from AFGL Atmospheric Constituent Profiles (1986)
    t, ta = set_temperature_default(model, p, pa)
    z, za = set_height(model, p, pa)
    t0, ta0 = t.copy(), ta.copy()

    # linear in troposphere [ 0 - 14 km] during daytime
    if (np.isnan(T_surf)):
        T_delta = 0
    else:
        T_delta = T_surf - t[0]
    idx = np.argmin(abs(z - 14000))+1 # location of top of troposphere
    z_trop, za_trop = z[1:idx], za[:idx]
    t_trop, ta_trop = t0[1:idx], ta0[:idx]
    z_linear_trop, za_linear_trop = z_trop.copy(), za_trop.copy()
    t_linear_trop = np.interp(z_linear_trop, (z_trop[0], z_trop[-1]), (t_trop[0] + T_delta, t_trop[-1]))
    ta_linear_trop = np.interp(za_linear_trop, (za_trop[0], za_trop[-1]), (ta_trop[0] + T_delta, ta_trop[-1]))
    t[0] = t_linear_trop[0]
    t[1:idx] = t_linear_trop
    ta[:idx] = ta_linear_trop

    # add temperature inversion at ~1 km during nighttime
    if (period=='night'):
        T_night_offset = 6   # temperature inversion offset [K]
        t_linear = np.interp(z_linear_trop, (z_trop[0], z_trop[-1]), (t_trop[0] + T_delta + 2 * T_night_offset, t_trop[-1]))
        ta_linear = np.interp(za_linear_trop, (za_trop[0], za_trop[-1]), (ta_trop[0] + T_delta + 2 * T_night_offset, ta_trop[-1]))
        t[0] = t_linear[0]
        t[1:idx] = t_linear
        ta[:idx] = ta_linear
 
        idx2 = np.argmin(abs(z - 1000)) # location of inversion layer
        z_inv, za_inv = z[1:idx2], za[:idx2]
        t_inv, ta_inv = t[1:idx2], ta[:idx2]
        t_linear = np.interp(z_inv, (z_inv[0], z_inv[-1]), (t_inv[0]-2*T_night_offset, t_inv[-1]))
        ta_linear = np.interp(za_inv, (za_inv[0], za_inv[-1]), (ta_inv[0]-2*T_night_offset, ta_inv[-1]))
        t[0] = t_linear[0]
        t[1:idx2] = t_linear
        ta[:idx2] = ta_linear
    return t, ta

def set_height(model, p, pa):
    """
    Set the height of the N layers.

    Parameters
    ----------
    model : str
        Profile model (e.g. 'tropical').
    p : (N + 2,) array_like
        Pressure [Pa] at each node point.
    pa : (N + 1,) array_like
        Average pressure [Pa] at each layer.

    Returns
    -------
    z : (N + 2,) array_like
        Height [m] at each node point.
    za : (N + 1,) array_like
        Average height [m] of each layer.

    """

    data = np.genfromtxt("data/profiles/{}.csv".format(model), delimiter=",")
    ref_p = data[:, 1]
    ref_z = data[:, 0]

    # convert units
    ref_p = np.asarray(ref_p) * 1e2  # pressure [Pa]
    ref_z = np.asarray(ref_z) * 1e3  # height [m]

    # xv needs to be increasing for np.interp to work properly
    z = np.interp(-p, -ref_p, ref_z)

    # average height
    z_avg = np.zeros((len(pa)))
    for i in range(1, len(pa)):
        # pressure averaged t
        z_avg[i] = (z[i] * (p[i] - pa[i]) + z[i + 1] * (pa[i] - p[i + 1])) / (
            p[i] - p[i + 1]
        )

    z_avg[0] = z[0]
    return z, z_avg


def set_ndensity(model, p, pa):
    """
    Set the number density of the N layers.

    Used for calculating the Rayleigh scattering coefficient.

    Parameters
    ----------
    model : str
        Profile model (e.g. 'tropical').
    p : (N + 2,) array_like
        Pressure [Pa] of each node point.
    pa : (N + 1,) array_like
        Average pressure [Pa] of each layer.

    Returns
    -------
    n : (N + 2,) array_like
        Number density [unit/cm^3] of each node point.
    na : (N + 1,) array_like
        Average number density [unit/cm^3] of each layer.

    """

    data = np.genfromtxt("data/profiles/{}.csv".format(model), delimiter=",")
    #data = np.genfromtxt("data/profiles/{}.csv".format(model), delimiter=",",skip_header=1)
    ref_p = data[:, 1]
    ref_n = data[:, 3]  # [1/cm^3]

    ref_p = np.asarray(ref_p) * 1e2  # convert to [Pa]
    ref_n = np.asarray(ref_n)

    # xv needs to be increasing for np.interp to work properly
    n = np.interp(-p, -ref_p, ref_n)
    na = np.zeros((len(pa)))
    for i in range(1, len(pa)):
        # pressure averaged t
        na[i] = (n[i] * (p[i] - pa[i]) + n[i + 1] * (pa[i] - p[i + 1])) / (
            p[i] - p[i + 1]
        )
    na[0] = n[0]
    return n, na


def set_vmr(model, molecules, vmr0, z):
    """
    Set volumetric mixing ratio (vmr) of atmosphere gases for N layers.

    Parameters
    ----------
    model : str
        Profile model (e.g. 'tropical').
    molecules : (M,) array_like
        Names of the M molecules.
    vmr0 : (M,) dict
        Surface vmr of molecules (for scaling purposes).
    z : (N + 2,) array_like
        Height [m] of each node point.

    Returns
    -------
    vmr : (N + 2, M) array_like
        Volumetric mixing ratio (vmr) [ppmv] at each node point.
    densities : (N + 1, M) array_like
        Average density of each gas in each layer (accumulated weight/volume
        [g/cm^3]).

    """

    vmr = np.zeros((len(molecules), len(z)))
    densities = np.zeros((len(molecules), len(z) - 1))
    data = np.genfromtxt("data/profiles/{}.csv".format(model), delimiter=",")
    data2 = np.genfromtxt("data/profiles/AFGL_molecule_profiles.csv", delimiter=",")

    # reference height (converted from [km] to [m]
    ref_z = data[:, 0] * 1e3

    for i in range(0, len(molecules)):
        if molecules[i] == "H2O":
            ref_vmr = data[:, 4]
            M = 18  # molecular weight, g/mol
            vmr0_i=vmr0['H2O']
        elif molecules[i] == "CO2":
            ref_vmr = data2[:, 2]
            M = 44
            vmr0_i=vmr0['CO2']
        elif molecules[i] == "O3":
            ref_vmr = data[:, 5]
            M = 48
            vmr0_i=vmr0['O3']
        elif molecules[i] == "N2O":
            ref_vmr = data[:, 6]
            M = 44
            vmr0_i=vmr0['N2O']
        elif molecules[i] == "CH4":
            ref_vmr = data[:, 8]
            M = 16
            vmr0_i=vmr0['CH4']
        elif molecules[i] == "O2":
            ref_vmr = data2[:, 7]
            M = 32
            vmr0_i=vmr0['O2']
        elif molecules[i] == "N2":
            ref_vmr = data2[:, 8]
            M = 28
            vmr0_i=vmr0['N2']
        ref_vmr = np.asarray(ref_vmr) / 1e6  # change from ppv to unit 1
        ref_Ni = data[:, 3] * ref_vmr  # in unit of #molecules/cm^3
        NA = 6.022 * 1e23  # in unit #/mol

        # reference density [g/cm^3] (after scaling)
        ref_rho = ref_Ni / NA * M / ref_vmr[0] * vmr0_i
        vmr[i, :] = np.interp(z, ref_z, ref_vmr) / ref_vmr[0] * vmr0_i
        for j in range(1, len(z) - 1):
            zz = np.linspace(z[j], z[j + 1], 100)  # evenly spaced coordinate
            rrho = np.interp(zz, ref_z, ref_rho)

            # distance averaged density
            densities[i, j] = np.trapz(rrho, zz) / abs(zz[0] - zz[-1])

        densities[i, 0] = densities[i, 1]

    # output arrays so:
    # - rows = node points
    # - cols = molecules
    vmr = np.transpose(vmr)
    densities = np.transpose(densities)
    return vmr, densities


def saturation_pressure(T):
    """
    Saturation pressure of water vapor for given temperature.

    Calculate the saturation pressure of water vapor using the Magus expression (see [1]).

    Parameters
    ----------
    T : (N + 2,) array_like
        Temperature [K] of the node points.

    Returns
    -------
    P_sat : (N + 2,) array_like
        Saturation pressure [Pa] of water vapor at each of the node points.

    References
    ----------
    [1] M. Li, Y. Jiang, and C. F. M. Coimbra (2017) “On the Determination of
        Atmospheric Longwave Irradiance Under All-Sky Conditions,” Solar Energy
        (144), pp. 40-48.

    """
    P_sat = 610.94 * np.exp(17.625 * (T - 273.15) / (T - 30.11))
    return P_sat

def total_precipitable_water(densities,pa,ta,p):
    # total precipitable water dewpoint
    # is same to Metpy.precipitable_water(p,dewpoint(pe/100 * units.hPa))
    epsilon = 0.622 # epsilon=Mvapor/Mdry=0.622
    pw = 1000 # kg/m3
    g = 9.8 # m/s2
    N_layer = ta.shape[0]-1
    RH, qs, q, tpw, ps,pe = [np.zeros([N_layer + 1]) for i in range(0, 6)]

    for i in range(1, N_layer + 1): # loop layer by layer
        x_h2o = ((densities[i]) / 18 * 8.314 * ta[i] / pa[i])  # mole fraction
        x_h2o *= 1e6  # unit conversion
        ps[i] = saturation_pressure(ta[i]) # unit [pa]
        RH[i] = pa[i] * x_h2o / ps[i] # [0-1]
        if RH[i] > 1:  # if exceeds 1
            RH[i] = 1
        x_h2o = RH[i] / 100 * ps[i] / pa[i]
        x_h2o /= 1e6
        densities[i] = (x_h2o * pa[i] / ta[i] / 8.314 * 18)
        pe[i]=ps[i]*RH[i]
        q[i] = epsilon*pe[i]/(pa[i]-pe[i]) # kg/kg
    TPW=np.trapz(q,-pa)  # kg/m2 or mm
    return 1/(g)*TPW

def getMixKappa(inputs, densities, pa, ta, z, za, na, AOD, COD, kap, Ph_cdf_cld =True, Ph_cdf_aer=False):
    """
    Absorption/scattering coefficients and asymmetry parameters of gas mixture for N layers.

    Parameters
    ----------
    inputs : (6,) list of variables
        Includes N_layer, model, molecules, nu (wavenumber grid), cld_model, spectral.
    densities : (N+1, M) array_like
        Average density of each gas in each layer (accumulated weight/volume
        [g/cm^3]).
    pa : (N + 1,) array_like
        Average pressure [Pa] of the layers.
    ta : (N + 1,) array_like
        Average temperature [K] of the layers.
    z : (N + 2,) array_like
        Height [m] of the node points.
    za : (N + 1,) array_like
        Average height [m] of the layers.
    na : (N + 1,) array_like
        Average number density [1/cm^3] of the layers.
    AOD : float
        Aerosol optical depth at 497.5 nm.
    COD : float
        Cloud optical depth at 497.5 nm.
    kap : list of int
        Layer indexes that containing clouds. e.g. kap=[5,6,7] clouds are in layers 5,6 and 7.

    Returns
    -------
    [ka_gas_M, ks_gas_M, g_gas_M] : (N_layer+1, N_nu)*3 array_like
        Gas absorption, scattering coefficients [cm-1] and asymmetry parameter. 
    [ka_aer_M, ks_aer_M, g_aer_M] : (N_layer+1, N_nu)*3 array_like
        Aerosol absorption, scattering coefficients [cm-1] and asymmetry parameter. 
    [ka_cld_M, ks_cld_M, g_cld_M] : (N_layer+1, N_nu)*3 array_like
        Cloud absorption, scattering coefficients [cm-1] and asymmetry parameter. 
    [ka_all_M, ks_all_M, g_all_M] : (N_layer+1, N_nu)*3 array_like
        Mixture absorption, scattering coefficients [cm-1] and asymmetry parameter. 

    """
    N_layer = inputs['N_layer']
    model = inputs['model']
    molecules = inputs['molecules']
    nu = inputs['nu']
    cld_model = inputs['cld_model']
    spectral = inputs['spectral']
    #if nu.shape[0] == 10834:
    coeff_M = np.load("data/computed/{}_coeffM_{}layers_{}_dnu={:.1f}cm-1.npy".format(
     spectral, N_layer, model, nu[10]-nu[9]))
    # if nu.shape[0] != 10834:
    #     print('CoeffM,nu=', nu.shape[0])
    #     coeff_M = np.load("data/computed/GOES_{}_coeffM_{}layers_{}_dnu={:.2f}cm-1.npy".format(
    #         spectral, N_layer, model, nu[1]-nu[0]))
        # channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
        # nu0 = np.arange(2500, 35000, 3)
        # idx = np.nonzero(np.isin(nu0, nu))[0]
        # #idx = np.nonzero(np.isin(nu0, goes_calinu(nu, channels, '../GOES_data/', dnu=3)))[0]
        # coeff_M = coeff_M[:, :, idx]

    # Add aerosols and clouds
    cldS = np.zeros(N_layer + 1)
    aerS = np.zeros(N_layer + 1)
    #lam = np.concatenate((np.arange(0.1, 40.1, 0.1), np.arange(40.1, 500, 10)))
    lam = np.arange(0.1, 4.1, 0.1)
    nu_ref = 1e4 / lam # um to cm-1
    lam0 = 0.4975
    if AOD > 0:
        aer_ka = np.load("data/computed/ka_aerosol.npy")
        aer_ks = np.load("data/computed/ks_aerosol.npy")
        aer_g = np.load("data/computed/g_aerosol.npy")
        # TTHG
        aer_f = np.load("data/computed/TTHG/f_aerosol.npy")
        aer_g1 = np.load("data/computed/TTHG/g1_aerosol.npy")
        aer_g2 = np.load("data/computed/TTHG/g2_aerosol.npy")
        # Delta-M
        aer_fdelM = np.load("data/computed/fdelM_aerosol.npy")
        aer_cdf = np.load("data/computed/cdf/cdf_aerosol.npy")
        # add new aerosol vertical profile
        aer_vp = np.genfromtxt("data/profiles/aerosol_profile.csv", delimiter=",")
        aerS = (
            np.interp(za, aer_vp[:, 0], aer_vp[:, 3], left=0, right=0) * AOD
        )  # vertical AOD @ 497.5nm
    if COD > 0:
        cld_ks, cld_ka, cld_g, cld_f, cld_g1, cld_g2, cld_fdelM, cld_cdf = cloud(model, cld_model, z, kap,
                                                                                 Ph_cdf_cld)
        cldS[kap] = COD

    ka_gas_M, ks_gas_M, g_gas_M, ka_aer_M, ks_aer_M, g_aer_M = [
        np.zeros([N_layer + 1, len(nu)]) for i in range(0, 6)
    ]
    f_aer_M, g1_aer_M, g2_aer_M, f_cld_M, g1_cld_M, g2_cld_M = [
        np.zeros([N_layer + 1, len(nu)]) for i in range(0, 6)
    ]
    fdelM_aer_M, fdelM_cld_M =[
        np.zeros([N_layer + 1, len(nu)]) for i in range(0, 2)]
    cdf_aer_M = np.zeros([N_layer + 1, len(nu), 498])
    cdf_cld_M = np.zeros([N_layer + 1, len(nu), 498])
    ka_cld_M, ks_cld_M, g_cld_M, ka_all_M, ks_all_M, g_all_M = [
        np.zeros([N_layer + 1, len(nu)]) for i in range(0, 6)
    ]
    for i in range(1, N_layer + 1): # loop layer by layer
        ka_gas, ks_gas, ka_aer, ks_aer, g_aer, ka_cld, ks_cld, g_cld = [
            np.zeros(len(nu)) for i in range(0, 8)
        ]
        f_cld, g1_cld, g2_cld, ff_cld,fdelM_cld = [np.zeros(len(nu)) for i in range(0, 5)]
        cdf_cld = np.zeros([len(nu),498])
        RH = 0  # defalt no water vapor present
        for j in range(0, len(molecules)):
            ka_gas += coeff_M[i][j] * densities[i, j] # add spectral line absorption coeffcient
            if densities[i, j] > 0:  # only if the gas is present
                # add continuum of water vapor
                if molecules[j] == "H2O":
                    # check relative humidity does not exceed 100%
                    x_h2o = (
                        (densities[i, j]) / 18 * 8.314 * ta[i] / pa[i]
                    )  # mole fraction
                    x_h2o *= 1e6  # unit conversion
                    ps = saturation_pressure(ta[i])  # saturated pressure
                    RH = pa[i] * x_h2o / ps * 100  # ranges from 0 to 100
                    if RH > 100:  # if exceeds 100
                        RH = 100
                    elif (cldS[i]>0): # cloud present in this layer
                        RH = 100.0 # RH=100% for cloud layers
                    x_h2o = RH / 100 * ps / pa[i]
                    x_h2o /= 1e6
                    densities[i, j] = (
                        x_h2o * pa[i] / ta[i] / 8.314 * 18
                    )  # change densities according to RH change
                    # #MTCKD continuum
                    ka_cont = absorptionContinuum_MTCKD_H2O(
                        nu, pa[i], ta[i], densities[i, j]
                    )  # return mass absorption coeff
                    ka_gas += (
                        ka_cont * densities[i, j]
                    )  # return volume absorption coeff
                # add continuum of CO2
                if molecules[j] == "CO2":
                    # #MTCKD continuum
                    ka_cont = absorptionContinuum_MTCKD_CO2(
                        nu, pa[i], ta[i], densities[i, j]
                    )  # return mass absorption coeff
                    ka_gas += (
                        ka_cont * densities[i, j]
                    )  # return volume absorption coeff
                    vmr_co2 = (densities[i, j]) / 44 * 8.314 * ta[i] / pa[i]
                    vmr_co2 *= 1e6  # unit conversion to 1
                # add continuum of O3
                if molecules[j] == "O3":
                    # #MTCKD continuum
                    ka_cont = absorptionContinuum_MTCKD_O3(
                        nu, pa[i], ta[i], densities[i, j]
                    )  # return mass absorption coeff
                    ka_gas += (
                        ka_cont * densities[i, j]
                    )  # return volume absorption coeff
                # add continuum of O2 (for shortwave spectrum)
                if (molecules[j]=='O2'):
                    # #MTCKD continuum
                    for k in range(0,len(molecules)):
                        if (molecules[k]=='H2O'):
                            x_h2o=(densities[i,k])/18.0*8.314*ta[i]/pa[i] # mole fraction
                            x_h2o*=10**6 # unit conversion
                    ka_cont=absorptionContinuum_MTCKD_O2(nu,pa[i],ta[i],densities[i,j],x_h2o) # return mass absorption coeff
                    ka_gas+=ka_cont*densities[i,j]# return volume absorption coeff      
        # Add Rayleigh scattering coefficient (for shortwave spectral)
        if spectral == "SW":
            ks_gas = rayleigh_kappa_s(nu, na[i])
        # add aerosols
        if aerS[i] > 0:
            if RH == 0:
                ka_ref = aer_ka[0, :]
                ks_ref = aer_ks[0, :]
                g_ref = aer_g[0, :]
                f_delM_ref = aer_fdelM[0, :]
                cdf_ref = aer_cdf[0, :, :]

                f_ref = aer_f[0, :]
                g1_ref = aer_g1[0, :]
                g2_ref = aer_g2[0, :]
            else:
                n1 = int(np.floor(RH / 10))
                n2 = int(np.ceil(RH / 10))
                if n1 == n2:
                    ka_ref = aer_ka[n1, :]
                    ks_ref = aer_ks[n1, :]
                    g_ref = aer_g[n1, :]
                    f_delM_ref = aer_fdelM[n1, :]
                    cdf_ref = aer_cdf[n1, :, :]

                    f_ref = aer_f[n1, :]
                    g1_ref = aer_g1[n1, :]
                    g2_ref = aer_g2[n1, :]
                else:
                    ka_ref = aer_ka[n1, :] + (aer_ka[n2, :] - aer_ka[n1, :]) * (
                        RH / 10 - n1) / (n2 - n1)
                    ks_ref = aer_ks[n1, :] + (aer_ks[n2, :] - aer_ks[n1, :]) * (
                        RH / 10 - n1) / (n2 - n1)
                    g_ref = aer_g[n1, :] + (aer_g[n2, :] - aer_g[n1, :]) * (
                        RH / 10 - n1) / (n2 - n1)

                    f_delM_ref = aer_fdelM[n1, :] + (aer_fdelM[n2, :] - aer_fdelM[n1, :]) * (
                        RH / 10 - n1) / (n2 - n1)
                    cdf_ref = aer_cdf[n1, :, :] + (aer_cdf[n2, :, :] - aer_cdf[n1, :, :]) * (
                        RH / 10 - n1) / (n2 - n1)

                    f_ref = aer_f[n1, :] + (aer_f[n2, :] - aer_f[n1, :]) * (
                            RH / 10 - n1) / (n2 - n1)
                    g1_ref = aer_g1[n1, :] + (aer_g1[n2, :] - aer_g1[n1, :]) * (
                            RH / 10 - n1) / (n2 - n1)
                    g2_ref = aer_g2[n1, :] + (aer_g2[n2, :] - aer_g2[n1, :]) * (
                            RH / 10 - n1) / (n2 - n1)
            # scale aerosol according to desired AOD @ 497.5nm
            dz = 1575  # scale height,average of 2010_Yu
            kappa_e_ref = aerS[i] / (dz * 100)  # cm-1,desired extincion coeff
            kappa_e = np.interp(lam0, lam, ks_ref + ka_ref)
            ratio = kappa_e_ref / kappa_e
            ka_aer = (
                np.interp(-nu, -nu_ref, ka_ref, left=0, right=0) * ratio
            )  # correct using aerosol vertical profile
            ks_aer = (
                np.interp(-nu, -nu_ref, ks_ref, left=0, right=0) * ratio
            )  # correct using aerosol vertical profile
            g_aer = np.interp(-nu, -nu_ref, g_ref, left=0, right=0)
            
            fdelM_aer = np.interp(-nu, -nu_ref, f_delM_ref, left=0, right=0)
            data_slice = cdf_ref[:, :]
            f = interp1d(-nu_ref, data_slice, axis=0,kind='linear',
                        bounds_error=False,fill_value=0)
            cdf_aer = f(-nu)

            f_aer = np.interp(-nu, -nu_ref, f_ref, left=0, right=0)
            g1_aer = np.interp(-nu, -nu_ref, g1_ref, left=0, right=0)
            g2_aer = np.interp(-nu, -nu_ref, g2_ref, left=0, right=0)
        # add clouds
        if cldS[i] > 0:
            ka_cld = np.interp(-nu, -nu_ref, cld_ka[i, :], left=0, right=0) * cldS[i]
            ks_cld = np.interp(-nu, -nu_ref, cld_ks[i, :], left=0, right=0) * cldS[i]
            g_cld = np.interp(-nu, -nu_ref, cld_g[i, :], left=0, right=0)

            f_cld = np.interp(-nu, -nu_ref, cld_f[i, :], left=0, right=0)
            g1_cld = np.interp(-nu, -nu_ref, cld_g1[i, :], left=0, right=0)
            g2_cld = np.interp(-nu, -nu_ref, cld_g2[i, :], left=0, right=0)

            fdelM_cld = np.interp(-nu, -nu_ref, cld_fdelM[i, :], left=0, right=0)
            data_slice = cld_cdf[i, :, :]
            f = interp1d(-nu_ref, data_slice, axis=0,kind='linear',
                        bounds_error=False,fill_value=0)
            cdf_cld = f(-nu)
        # combine g of aerosol and cloud
        ks_all = ks_gas + ks_aer + ks_cld
        g_mix = g_aer * ks_aer + g_cld * ks_cld
        if aerS[i] > 0.0 or cldS[i] > 0.0:  # avoid dividing by zero
            g_mix /= ks_all  # a+c / (a+c+g)

        ka_gas_M[i, :] = ka_gas  # 0
        ks_gas_M[i, :] = ks_gas  # reyleigh scattering
        ka_aer_M[i, :] = ka_aer  # based on Rh
        ks_aer_M[i, :] = ks_aer  # too
        g_aer_M[i, :] = g_aer  # too
        if Ph_cdf_aer == True:
            g_aer_M[i, :] = -2.0
        fdelM_aer_M[i, :] = fdelM_aer  # too
        ka_cld_M[i, :] = ka_cld  # based on kappa
        ks_cld_M[i, :] = ks_cld  # calculated from mie, and kappa
        g_cld_M[i, :] = g_cld
        g_all_M[i, :] = g_mix
        fdelM_cld_M[i, :] = fdelM_cld

        f_aer_M[i, :] = f_aer
        g1_aer_M[i, :] = g1_aer
        g2_aer_M[i, :] = g2_aer

        f_cld_M[i, :] = f_cld
        g1_cld_M[i, :] = g1_cld
        g2_cld_M[i, :] = g2_cld

        cdf_cld_M[i, :, :] = cdf_cld
        cdf_aer_M[i, :, :] = cdf_aer  # too

    ka_all_M = ka_gas_M + ka_aer_M + ka_cld_M
    ks_all_M = ks_gas_M + ks_aer_M + ks_cld_M

    return (
    [ka_gas_M, ks_gas_M, g_gas_M],
    [ka_aer_M, ks_aer_M, g_aer_M, f_aer_M, g1_aer_M, g2_aer_M, fdelM_aer_M],
    [ka_cld_M, ks_cld_M, g_cld_M, f_cld_M, g1_cld_M, g2_cld_M, fdelM_cld_M],
    [ka_all_M, ks_all_M, g_all_M],
    {"cdf_aer_M": cdf_aer_M, "cdf_cld_M": cdf_cld_M},
)


def absorptionContinuum_MTCKD_H2O(nu, P, T, density):
    """
    Continuum absorption coefficients of water vapor.

    Parameters
    ----------
    nu : (N_nu,) array_like
        Wavenumber grid [cm^-1].
    P : float
        Total pressure [Pa].
    T : float
        Temperature [K].
    density : float
        Density [g/cm^3] of water vapor.

    Returns
    -------
    coeff_cont : (N_nu,) array_like
        Mass continuum absorption coefficients [cm^2/g] of water vapor.

    """

    data_frgn = np.genfromtxt("data/profiles/frgnContm_H2O.csv", delimiter=",")
    coeff_frgn = data_frgn[:, 1]  # in unit cm2/mole (cm)-1 *10^20
    cf_frgn = data_frgn[:, 2]
    data_self = np.genfromtxt("data/profiles/selfContm_H2O.csv", delimiter=",")
    nu_self = data_self[:, 0]
    coeff_self_296 = data_self[:, 1]  # in unit cm2/mole (cm)-1 *10^20
    coeff_self_260 = data_self[:, 2]
    cf_self = data_self[:, 3]

    # compute R factor
    T0 = 296
    P0 = 1.013e5
    c_h2o = density / 18  # molar density of h2o, in unit of mol/cm3
    c_tot = P / 8.314 / T / 1e6  # molar density of air, in unit of mol/cm3
    h2o_fac = c_h2o / c_tot
    RHOave = (P / P0) * (T0 / T)
    R_self = h2o_fac * RHOave  # consider partial pressure
    R_frgn = (1 - h2o_fac) * RHOave

    # compute self continuum coefficients
    tfac = (T - T0) / (260 - T0)
    coeff_self = (
        coeff_self_296 * (coeff_self_260 / coeff_self_296) ** tfac
    )  # temeprature correction
    coeff_self *= cf_self * R_self
    # compute foreign continuum coefficients
    coeff_frgn *= cf_frgn * R_frgn
    # sum the two
    coeff_tot = (
        (coeff_self + coeff_frgn) * 6.022 * 1e3
    )  # unit cm2/mol (cm)-1 * 6.022*10**23*10**(-20)
    coeff_tot *= c_h2o  # unit of cm2/cm3 (cm)-1
    coeff_tot /= density  # unit of cm2/g (cm)-1
    # interpelate to user defiend grid
    coeff_cont = np.interp(nu, nu_self, coeff_tot, left=0, right=0)
    RADFN = radiation_field(nu, T)
    # mass absorption coeffcient in unit [cm^2/g]
    coeff_cont *= RADFN
    return coeff_cont


def absorptionContinuum_MTCKD_CO2(nu, P, T, density):
    """
    Continuum absorption coeffcients of CO2.

    Parameters
    ----------
    nu : (N_nu,) array_like
        Wavenumber grid [cm^-1].
    P : float
        Total pressure [Pa].
    T : float
        Temperature [K].
    density : float
        Density [g/cm^3] of CO2.

    Returns
    -------
    coeff_cont : (N_nu,) array_like
        Mass absoption coefficient [cm^2/g] of CO2.

    """
    data_frgn = np.genfromtxt("data/profiles/frgnContm_CO2.csv", delimiter=",")
    nu_frgn = data_frgn[:, 0]  # wavenumber in cm-1
    coeff_frgn = data_frgn[:, 1]  # in unit cm2/mole (cm)-1 *10^20
    cfac = data_frgn[:, 2]
    tdep = data_frgn[:, 3]
    trat = T / 246
    coeff_frgn *= cfac * trat ** tdep

    # compute R factor
    T0 = 296
    P0 = 1.013 * 1e5
    c_co2 = density / 44  # molar density of h2o, in unit of mol/cm3
    RHOave = (P / P0) * (T0 / T)

    coeff_frgn *= RHOave  # co2_fac corrected 4/10/2018
    coeff_frgn *= 6.022 * 1e3  # unit cm2/mol (cm)-1 * 6.022*10**23*10**(-20)
    coeff_frgn *= c_co2  # unit of cm2/cm3 (cm)-1
    coeff_frgn /= density  # unit of cm2/g (cm)-1

    # apply radiation field
    coeff_cont = np.interp(nu, nu_frgn, coeff_frgn, left=0, right=0)
    RADFN = radiation_field(nu, T)

    # mass absorption coeffcient [cm^2/g]
    coeff_cont *= RADFN
    return coeff_cont

def absorptionContinuum_MTCKD_O3(nu, P, T, density):
    """
    Continuum absorption coeffcients of O3.

    Parameters
    ----------
    nu : (N_nu,) array_like
        Wavenumber grid [cm^-1].
    P : float
        Total pressure [Pa].
    T : float
        Temperature [K].
    density : float
        Density [g/cm^3] of O3.

    Returns
    -------
    coeff_cont : (N_nu,) array_like
        Mass absoption coefficient [cm^2/g] of O3.
    """
    
    c_o3 = density / 48.0  # molar density of O2, in unit of mol/cm3
    wk3=6.022*1e23*c_o3 # MT_CKD code is mole/cm2, here is mole/cm2 *(cm-1)
    DT=T-273.15
    # Band 1: 9170~24665 cm-1
    wo3=wk3*1e-20
    data_contm = np.genfromtxt('data/profiles/Contm_O3_b1.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    c1=data_contm[:,2]#*nu_contm
    c2=data_contm[:,3]#*nu_contm
    temp=c0+(c1+c2*DT)*DT # temperature dependence
    temp*=wo3
    #plt.plot(nu_contm,temp)
    contm_1=np.interp(nu, nu_contm, temp,left=0,right=0)
    #plt.plot(nu,contm_1)

    # Band 2: 27370~40800 cm-1
    wo3=wk3*1e-20
    data_contm = np.genfromtxt('data/profiles/Contm_O3_b2.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    c1=data_contm[:,2]
    c2=data_contm[:,3]
    temp=c0*wo3
    temp*=1.0+c1*DT+c2*DT**2
    contm_2=np.interp(nu, nu_contm, temp,left=0,right=0)

    # Band 3: 40800~54000 cm-1
    wo3=wk3
    data_contm = np.genfromtxt('data/profiles/Contm_O3_b3.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=c0*wo3
    contm_3=np.interp(nu, nu_contm, temp,left=0,right=0)

    coeff_cont=contm_1+contm_2+contm_3

    # apply radiation field
    RADFN=radiation_field(nu, T)
    # mass absorption coeffcient in unit [cm^2/g]
    coeff_cont *= RADFN/density
    return coeff_cont

# -------------------------------------------------------------------------------------------
def absorptionContinuum_MTCKD_O2(nu, P, T, density, x_h2o):
    """
    Continuum absorption coeffcients of O2, makes a difference only in the shortwave spectrum.

    Parameters
    ----------
    nu : (N_nu,) array_like
        Wavenumber grid [cm^-1].
    P : float
        Total pressure [Pa].
    T : float
        Temperature [K].
    density : float
        Density [g/cm^3] of O2.
    x_h2o : float
        Water vapor fraction.

    Returns
    -------
    coeff_cont : (N_nu,) array_like
        Mass absoption coefficient [cm^2/g] of O2.
    """

    P0=1.013*1e5
    T0=296.0
    xlosmt=2.68675*1e19
    amagat=(P/P0)*(273.0/T)
    rhoave=(P/P0)*(T0/T)

    c_o2 = density / 32.0  # molar density of O2, in unit of mol/cm3
    x_o2=c_o2*8.314*T/P # mole fraction
    x_o2*=1e6 # unit conversion
    x_n2=1.0-x_o2-x_h2o

    wk7= 6.022*1e23*c_o2# MT_CKD unit of [mole/cm2], here is mole/cm3

    # Band 1: 7536~8500 cm-1
    a_o2=1.0/0.446
    a_n2=0.3/0.446
    a_h2o=1.0
    tau_fac=wk7/xlosmt*amagat*(a_o2*x_o2+a_n2*x_n2+a_h2o*x_h2o)
    data_contm = np.genfromtxt('data/profiles/Contm_O2_b1.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=tau_fac*c0
    contm_1=np.interp(nu, nu_contm, temp,left=0,right=0)

    # Band 2: 9100~11000 cm-1
    wo2=wk7*1e-20*rhoave
    adjwo2=x_o2*(1.0/0.209)*wo2
    data_contm = np.genfromtxt('data/profiles/Contm_O2_b2.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=adjwo2*c0
    contm_2=np.interp(nu, nu_contm, temp,left=0,right=0)
    # Band 3: 12990.5 ~ 13223.5 cm-1
    tau_fac=wk7/xlosmt*amagat
    data_contm = np.genfromtxt('data/profiles/Contm_O2_b3.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=tau_fac*c0
    contm_3=np.interp(nu, nu_contm, temp,left=0,right=0)
    # Band 4: 15000 ~ 29870 cm-1
    wo2=wk7*1e-20*amagat
    chio2=x_o2
    adjwo2=chio2*wo2 
    data_contm = np.genfromtxt('data/profiles/Contm_O2_b4.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=adjwo2*c0
    contm_4=np.interp(nu, nu_contm, temp,left=0,right=0)
    # Band 5: 36000~40000 cm-1
    wo2=wk7*1e-20
    data_contm = np.genfromtxt('data/profiles/Contm_O2_b5.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=c0*(1.0+0.83*amagat)*wo2
    contm_5=np.interp(nu, nu_contm, temp,left=0,right=0)

    coeff_cont=contm_1+contm_2+contm_3+contm_4+contm_5

    # apply radiation field
    RADFN=radiation_field(nu, T)
    # mass absorption coeffcient in unit [cm^2/g]
    coeff_cont *= RADFN/density
    return coeff_cont


def radiation_field(nu, T):
    """
    The 'radiation field' for calculation of continuum coefficient.

    Parameters
    ----------
    nu : (N_nu,) array_like
        Wavenumber grid [cm^-1].
    T : float
        Temperature [K].

    Returns
    -------
    RADFN : (N_nu,) array_like
        Radiation field (to scale continnum absorption coefficient).

    """

    # times the 'radiation field' to get rid of (cm)-1 in the denominator
    # see cntnv_progr.f function RADFN(VI,XKT)
    XKT = T / 1.4387752  # 1.4387752 is a constant from phys_consts.f90
    RADFN = np.zeros(len(nu))
    #revised to use array operation instead of for loop, Mengying Li 4/21/2020
    XVIOKT = nu / XKT
    EXPVKT = np.exp(-XVIOKT)
    ind1 = (XVIOKT <= 0.01)
    RADFN[ind1] = 0.5 * XVIOKT[ind1] * nu[ind1]
    ind2 = (XVIOKT > 0.01) & (XVIOKT <= 10)
    RADFN[ind2] = nu[ind2] * (1.0 - EXPVKT[ind2]) / (1.0 + EXPVKT[ind2])
    ind3 = (XVIOKT > 10)
    RADFN[ind3] =nu[ind3]
    return RADFN


def Planck(nu, T):
    """
    Planck's law as a function of wavenumber [cm^-1].

    Planck's law (see equaton on page 453 of [1]).

    Parameters
    ----------
    nu : float or (N_nu) array_like
        Wavenumber [cm^-1].
    T : float or array_like
        Temperature [K].

    Returns
    -------
    Eb : float or (N_nu) array_like
        Blackbody emission intensity density [W/(m^2 sr cm^-1)]

    References
    ----------
    [1] Mill and Coimbra, "Basic Heat and Mass Transfer"

    """

    h = 6.6261e-34  # Planck's constant [J s]
    kB = 1.3806485e-23  # Boltzmann constant [J / K]
    c = 299792458  # speed of light [m / s]
    C1 = 2 * h * c ** 2  # coefficient 1
    C2 = h * c / kB  # coefficient 2
    nu = nu * 100  # convert from [cm^-1] to [m^-1]

    # blackbody emission
    # equivalent to MATLAB dot calculations
    Eb_nu = C1 * nu ** 3 / (np.exp(C2 * nu / T) - 1)

    # convert to [W/(m^2 sr cm^-1)]
    Eb_nu *= 100
    return Eb_nu


def Planck_lam(lam, T):
    """
    Planck's law as a function of wavelength [um].

    Uses equation on page 453 of BHMT.

    Parameters
    ----------
    lam : float or (N_nu) array_like
        Wavelength [um].
    T : float or array_like
        Temperature [K].

    Returns
    -------
    Eb_lam : float or (N_nu) array_like
        Blackbody emission intensity density [W/(m^2 sr um)].

    References
    ----------
    [1] Mill and Coimbra, "Basic Heat and Mass Transfer"

    """

    h = 6.6261 * 1e-34  # Planck's constant, J s
    kB = 1.3806485 * 1e-23  # Boltzmann constant, J/K
    c = 299792458  # speed of light, m/s
    C1 = 2 * h * c ** 2
    C2 = h * c / kB
    lam = lam / 1e6  # change unit to m
    Eb_lam = C1 / lam ** 5 / (np.exp(C2 / lam / T) - 1)
    Eb_lam /= 1e6  # in unit of W/m2 um sr
    return Eb_lam


def Mie(lam, radii, refrac):
    """
    Mie efficiencies for single wavelength interacts with particle of one radii.

    Parameters
    ----------
    lam : float
        Wavelength [um].
    radii : float
        Radii [um] of the particles.
    refrac : float
        Index of refraction (complex number).

    Returns
    -------
    Qext :
        Extinction efficiency.
    Qabs :
        Absoprtion efficiency.
    Qsca :
        Scattering efficiency.
    Qbsca :
        Back scattering efficiency.
    g :
        Assymetry parameter.

    References
    ----------
    [1] "A first course in atmospheric radiation" by Petty.
    [2] "mie.py" provided by "Principles of Planetary Climate" by Pierrehumbert.

    """

    size = 2 * np.pi * radii / lam  # size parameter
    Nlim = round(
        size + 4 * size ** (1 / 3) + 2
    )  # number of terms of infinite series, from Petty's P359
    an, bn = Mie_ab(size, refrac, Nlim + 1)
    # compute efficiencies and g
    Qext = 0
    Qsca = 0
    Qbsca = 0
    g = 0
    sn = an + bn  # an and bn are Mie scattering coefficients
    for n in range(1, int(Nlim + 1)):
        Qext += (2 * n + 1) * sn[n - 1].real
        Qsca += (2 * n + 1) * (abs(an[n - 1]) ** 2 + abs(bn[n - 1]) ** 2)
        Qbsca += (2 * n + 1) * (-1) ** n * (an[n - 1] - bn[n - 1])
        temp1 = an[n - 1] * an[n].conjugate() + bn[n - 1] * bn[n].conjugate()
        temp2 = an[n - 1] * bn[n - 1].conjugate()
        g += n * (n + 2) / (n + 1) * temp1.real + (2 * n + 1) / n / (n + 1) * temp2.real

    Qext *= 2 / size ** 2
    Qsca *= 2 / size ** 2
    Qabs = Qext - Qsca
    Qbsca = abs(Qbsca) ** 2 / size ** 2
    g *= 4 / size ** 2 / Qsca
    return Qext, Qabs, Qsca, Qbsca, g

def MieS1S2(m, wavelength, diameter, mu, nMedium=1.0):
  #  http://pymiescatt.readthedocs.io/en/latest/forward.html#MieS1S2

  m /= nMedium
  wavelength /= nMedium
  x = np.pi * diameter / wavelength
  nmax = np.round(2+x+4*np.power(x,1/3))
  an, bn = AutoMie_ab(m, x)

  # Ensure mu is an array so we can loop through it
  if not isinstance(mu, np.ndarray) and not isinstance(mu, list):
      mu = np.array([mu])
  elif isinstance(mu, list):
      mu = np.array(mu)

  # Initialize S1 and S2 as complex-valued arrays
  S1 = np.zeros(len(mu), dtype=complex)
  S2 = np.zeros(len(mu), dtype=complex)

  n = np.arange(1,nmax+1)
  n2 = (2*n+1)/(n*(n+1))

  # Loop through each angle in mu
  for j, val in enumerate(mu):
      pin, taun = MiePiTau(val, nmax)  # Call the helper with a single angle
      S1[j] = np.sum(n2[0:len(an)] * (an * pin[0:len(an)] + bn * taun[0:len(bn)]))
      S2[j] = np.sum(n2[0:len(an)]*(an*taun[0:len(an)]+bn*pin[0:len(bn)]))

  # If the original input was a single number, return single numbers
  if len(S1) == 1:
      return S1[0], S2[0]
  else:
      return S1, S2

def MiePiTau(mu,nmax):
#  http://pymiescatt.readthedocs.io/en/latest/forward.html#MiePiTau
  p = np.zeros(int(nmax))
  t = np.zeros(int(nmax))
  p[0] = 1
  p[1] = 3*mu
  t[0] = mu
  t[1] = 3.0*np.cos(2*np.arccos(mu))
  for n in range(2,int(nmax)):
    p[n] = ((2*n+1)*(mu*p[n-1])-(n+1)*p[n-2])/n
    t[n] = (n+1)*mu*p[n]-(n+2)*p[n-1]
  return p, t

def AutoMie_ab(m,x):
  # pymiescatt
  if x<0.5:
    return LowFrequencyMie_ab(m,x)
  else:
    return Miescat_ab(m,x)
  
def LowFrequencyMie_ab(m,x):
#  http://pymiescatt.readthedocs.io/en/latest/forward.html#LowFrequencyMie_ab
  # B&H page 131
  m2 = m**2
  LL = (m**2-1)/(m**2+2)
  x3 = x**3
  x5 = x**5
  x6 = x**6

  a1 = (-2j*x3/3)*LL-(2j*x5/5)*LL*(m2-2)/(m2+2)+(4*x6/9)*(LL**2)
  a2 = (-1j*x5/15)*(m2-1)/(2*m2+3)
  b1 = (-1j*x5/45)*(m2-1)
  b2 = 0+0j
  an = np.append(a1,a2)
  bn = np.append(b1,b2)
  return an,bn

def AutoMie_ab(m,x):
  # pymiescatt
  if x<0.5:
    return LowFrequencyMie_ab(m,x)
  else:
    return Miescat_ab(m,x)
  
def Miescat_ab(m,x):
#  http://pymiescatt.readthedocs.io/en/latest/forward.html#Mie_ab
  mx = m*x
  nmax = np.round(2+x+4*(x**(1/3)))
  nmx = np.round(max(nmax,np.abs(mx))+16)
  n = np.arange(1,nmax+1) #
  nu = n + 0.5 #

  sx = np.sqrt(0.5*np.pi*x)

  px = sx*jv(nu,x) #
  p1x = np.append(np.sin(x), px[0:int(nmax)-1]) #

  chx = -sx*yv(nu,x) #
  ch1x = np.append(np.cos(x), chx[0:int(nmax)-1]) #
  
  gsx = px-(0+1j)*chx #
  gs1x = p1x-(0+1j)*ch1x #

  # B&H Equation 4.89
  Dn = np.zeros(int(nmx),dtype=complex)
  for i in range(int(nmx)-1,1,-1):
    Dn[i-1] = (i/mx)-(1/(Dn[i]+i/mx))

  D = Dn[1:int(nmax)+1] # Dn(mx), drop terms beyond nMax
  da = D/m+n/x
  db = m*D+n/x

  an = (da*px-p1x)/(da*gsx-gs1x)
  bn = (db*px-p1x)/(db*gsx-gs1x)

  return an, bn

def phaseFunction(lam, radii, refrac, mu, theta):
    """
    Two term HG parameters calculation.
    f,g1,g2 is the optimised factor got from phase function.

    Parameters
    ----------
    lam : float
        Wavelength [um].
    radii : float
        Radii [um] of the particles.
    refrac : float
        Index of refraction (complex number).

    Returns
    -------
    Ground truth phase function
    phase : (N_theta,) array_like
        Phase function.
    mu : (N_theta,) array_like

    References
    ----------
    From python package pymiescatt.

    """
    # calculate the phase function via S1S
    s1, s2 = MieS1S2(m=refrac, wavelength=lam * 1000,
                     diameter=radii * 2 * 1000, mu=mu)
    epsilon = 1e-30
    I_theta = (np.abs(s1) ** 2 + np.abs(s2) ** 2) / 2
    I4pi = np.trapz(I_theta * np.sin(theta), theta) * 2 * np.pi # 4 pi
    phase = I_theta / I4pi # In of phase along cosu should be 2.
    # But which will be canceled * 0.5 when making cdf. so We dont *2 here.
    return phase

def TTHG_fitting(phase_Mie, mu, radii, g, verbose=False):
    """
    Two term HG parameters calculation.
    f,g1,g2 is the optimised factor got from phase function.

    Parameters
    ----------
    lam : float
        Wavelength [um].
    radii : float
        Radii [um] of the particles.
    refrac : float
        Index of refraction (complex number).

    Returns
    -------
    f: two term HG parameter, probability of first term.
    g1: TTHG, first term asymmetry parameter.
    g2: TTHG, second term asymmetry parameter.

    References
    ----------
    From python package pymiescatt.

    """
    # 1. Create a smarter, physically-motivated initial guess
    # Assume the first term (g1) captures the sharp forward peak and the
    # second term (g2) captures the more isotropic/backward part.
    g1_initial = 0.95 * g  # Start with a g1 slightly less than the total g
    g2_initial = -0.3  # A reasonable guess for the backward-scattering term
    # Calculate the initial f to preserve the overall asymmetry parameter g
    # g = f*g1 + (1-f)*g2  =>  f = (g - g2) / (g1 - g2)
    if np.abs(g1_initial - g2_initial) < 1e-3:  # Avoid division by zero
        f_initial = 0.9
    else:
        f_initial = (g - g2_initial) / (g1_initial - g2_initial)
    # Ensure the initial f is within the bounds
    f_initial = np.clip(f_initial, 1e-3, 1 - 1e-3)
    p0 = [f_initial, g1_initial, g2_initial]

    # fitting for the TTHG
    p_target = phase_Mie
    #p0 = [0.7, 0.8, -0.1] # Initial parameter guess (f, g1, g2)
    bounds = [(1e-3, 1-1e-3), (-0.999, 0.999), (-0.999, 0.999)] # Parameter bounds: f in (0,1), g1/g2 in (-1,1)
    result = minimize(tthg_fit_loss, p0, args=(mu, p_target), bounds=bounds)
    f_fit, g1_fit, g2_fit = result.x

    if result.success:  # Add a check on the final loss value
        return f_fit, g1_fit, g2_fit
    else:
        if verbose:
            # If TTHG fails, a single HG is the best simple alternative.
            # This is represented by f=1, g1=g. g2 can be anything.
            print(f"⚠️ TTHG fit failed or was poor for radii={radii}. Reason: {result.message}")
            print("Falling back to a single Henyey-Greenstein function.")
        return 1.0, g, 0.0
    

# Objective function for fitting
def tthg_fit_loss(params, mu, p_target):
    f, g1, g2 = params
    # Enforce physical limits
    if not (0 < f < 1 and -1 < g1 < 1 and -1 < g2 < 1):
        return 1e9
    p_model = tthg_phase(mu, f, g1, g2)
    log_diff = np.log(p_model + 1e-10) - np.log(p_target + 1e-10)
    return np.sum(log_diff**2) # or log-space for better peak matching

# Two-term HG phase function
def tthg_phase(mu, f, g1, g2):
    return f * hg_phase(mu, g1) + (1 - f) * hg_phase(mu, g2)

# Single HG phase function
def hg_phase(mu, g):
    return (1 - g**2) / (2 * (1 + g**2 - 2 * g * mu)**1.5)

def Mie_ab(size, refrac, Nlim):
    """

    Mie coefficients.

    Parameters
    ----------
    size : float
        Size parameter.
    refrac : float
        Index of refraction (complex number).
    Nlim : int
        Number of modes to approximate the summation of an infinite series.

    Returns
    -------
    an, bn : float
        Mie coefficients.

    References
    ----------
    [1] "MATLAB Functions for Mie Scattering and Absorption."

    """

    n_all = np.arange(
        0, Nlim + 1, 1
    )  # number of modes (start with zero to count for n-1)
    n = n_all[1:]
    z = size * refrac
    m2 = refrac * refrac
    sqx = np.sqrt(0.5 * np.pi / size)
    sqz = np.sqrt(0.5 * np.pi / z)

    bx_all = jv(n_all + 0.5, size) * sqx
    bz_all = jv(n_all + 0.5, z) * sqz
    yx_all = yv(n_all + 0.5, size) * sqx
    hx_all = bx_all + yx_all * 1j

    bx = bx_all[1:]
    bz = bz_all[1:]
    # yx = yx_all[1:]
    hx = hx_all[1:]

    b1x = bx_all[0:-1]
    b1z = bz_all[0:-1]
    # y1x = yx_all[0:-1]
    h1x = hx_all[0:-1]

    ax = size * b1x - n * bx
    az = z * b1z - n * bz
    ahx = size * h1x - n * hx

    an = (m2 * bz * ax - bx * az) / (m2 * bz * ahx - hx * az)
    bn = (bz * ax - bx * az) / (bz * ahx - hx * az)
    return an, bn

def deltaM_phasefunc(original_phase, theta, trunc_order=None, verbose=False):
    # Even numbers, no larger than 64
    if trunc_order is None:
        trunc_order_v = np.arange(32,64,2)#[16,24,36,40,48,64]
        n_streams = 64+2
    else:
        trunc_order_v = [trunc_order]  # [16,24,36,40,48,64]
        n_streams = trunc_order + 2
    tol = 5 # 0.1% tolerance
    mask = theta>np.deg2rad(30) # ignore the very forward peak
    for i in range(len(trunc_order_v)):
        trunc_order = trunc_order_v[i]
        delta_m = deltaM.DeltaMApproximation(trunc_order, n_streams)
        results = delta_m.demonstrate_delta_m(original_phase, theta)
        phase_ = results['truncated_phase']
        rd = np.abs(phase_ - original_phase)[mask] / original_phase[mask]
        is_within_tolerance = np.all(rd < tol)
        max_rd = np.max(rd)
        if trunc_order is not None:
            print(f"For truncation order {trunc_order}")
            print(f"Within tolerance of {tol}%: {is_within_tolerance}")
            print('Max relative difference:', max_rd)
        if is_within_tolerance:
            if max_rd < 3:
                if verbose == True:
                    plt.plot(original_phase, label='Mie phase', color='k', alpha=0.5)
                    plt.plot(phase_, label=f'Trunc order={trunc_order_v[i]}')
                    print('Max relative difference:', max_rd)
                    plt.yscale('log')
                    plt.legend()
                    plt.show()
                break
    # norm
    norm = np.trapz(phase_ * np.sin(theta), theta) * 2 * np.pi
    phase_ /= norm
    return phase_, results['f_trunc'], results['trunc_g']
# allow parallel computing
def aerosol_monoLam(inputs, verbose=False, cdf=True):
    """
    Aerosol monochromatic Mie efficiencies.

    Parameters
    ----------
    inputs : (3,) list of variables
        lam : float, wavelength [um].
        refrac : float, index of refraction (complex number).
        r : (N_r,) array_like, considered particle radius. [um]
        ff: forward fraction of the truncated phase function.

    Returns
    -------
    Qabs : Absoprtion efficiency.
    Qsca : Scattering efficiency.
    g : Assymetry parameter.

    """

    lam = inputs[0]
    refrac = inputs[1]
    r = inputs[2]
    Qsca = np.zeros(len(r))
    Qabs = np.zeros(len(r))
    g = np.zeros(len(r))
    scaled_g = np.zeros(len(r))

    f = np.zeros(len(r))
    g1 = np.zeros(len(r))
    g2 = np.zeros(len(r))
    ff_trunc = np.zeros(len(r))


    angles = np.concatenate([
    np.arange(0, 2, 0.01),
    np.arange(2, 5, 0.05),
    np.arange(5, 10, 0.1),
    np.arange(10, 15, 0.5),
    np.arange(15, 176, 1),
    np.arange(176, 180+0.25, 0.25)])  # same to Yang et al., 2013
    #np.linspace(0, 180, 361)  # degrees
    theta = np.deg2rad(angles)
    mu = np.cos(theta)
    phase_Mie = np.zeros((len(r), len(angles)))  # preallocate
    for j in range(len(r)):
        # simple HG phase function
        Qext, Qabs[j], Qsca[j], Qbsca, g[j] = Mie(lam, r[j], refrac)
        phase_Mie[j,:] = phaseFunction(lam, r[j], refrac, mu, theta) # normalized to 2.
        if cdf == True:
            continue
        # Truncated phase function + delta-M approximation
        X = 2 * np.pi * r[j] / lam  # size parameter
        if X > 10: # Delta-M approximation is valid
            # when truncation order is None, it will automatically choose the smallest truncation order that meets the tolerance requirement
            phase_truncated, ff_trunc[j], scaled_g[j] = deltaM_phasefunc(phase_Mie, theta, trunc_order=None,verbose=False)
            f[j], g1[j], g2[j] = TTHG_fitting(phase_truncated, mu, r[j], scaled_g[j], verbose=False)
            phase_target = phase_truncated
        else:
            # TTHG phase function
            f[j], g1[j], g2[j] = TTHG_fitting(phase_Mie, mu, r[j], g[j])
            phase_target = phase_Mie
        if verbose:
            phase_tthg = tthg_phase(mu, f[j], g1[j], g2[j])
            phase_hg = hg_phase(mu, g[j])
            norm = np.trapz(phase_hg * np.sin(theta), theta) * 2 * np.pi
            phase_hg /= norm
            phase_hg_scaled = hg_phase(mu, scaled_g[j])
            norm = np.trapz(phase_hg_scaled * np.sin(theta), theta) * 2 * np.pi
            phase_hg_scaled /= norm
            mse_hg = log_rel_error(phase_hg_scaled, phase_target)
            mse_tthg = log_rel_error(phase_tthg, phase_target)

            plt.figure(figsize=(7, 5))
            ax = plt.gca()
            ax.semilogy(angles, phase_Mie, label='Mie')
            ax.semilogy(angles, phase_truncated, '-.', label='Delta-M')
            ax.semilogy(angles, phase_tthg, '--', label='TTHG')
            ax.semilogy(angles, phase_hg, '--', label='HG')
            ax.semilogy(angles, phase_hg_scaled, '--', label='Scaled HG')
            ax.set_xlabel(r"Scattering angle [$\degree$]")
            ax.set_ylabel("Phase function (normalized)")
            plt.legend()
            ax.set_title(f"Wavelength ={lam} μm, r = {r[j]} μm, X={np.round(X)}")
            plt.show()
            print('two HG is worse than one HG')
            print(f"log-MSE (HG)    = {mse_hg:.3f}")
            print(f"log-MSE (TTHG)  = {mse_tthg:.3f}")
    return Qsca, Qabs, g, f, g1, g2, ff_trunc, phase_Mie

def log_rel_error(p_fit, p_target):
    return np.sqrt(np.mean((np.log10(p_fit + 1e-30) - np.log10(p_target + 1e-30)) ** 2))


def aerosol(cdf=True):
    """

    Compute and save the absoprtion/scattering coefficients and assymetry parameter for aerosols.

    Parameters
    ----------
    None

    Returns
    -------
    None. Save computed data files to local directory. Each data file contains an array_like (N_rh, N_lam).

    References
    ----------
    [1] "Longwave radiative forcing of Indian Ocean tropospheric aerosol" (2002) by Dan Lubin.
    """
    angles = np.concatenate([
        np.arange(0, 2, 0.01),
        np.arange(2, 5, 0.05),
        np.arange(5, 10, 0.1),
        np.arange(10, 15, 0.5),
        np.arange(15, 176, 1),
        np.arange(176, 180 + 0.25, 0.25)
    ])
    theta = np.deg2rad(angles)
    mu = np.cos(theta)[::-1]
    #lam = np.concatenate((np.arange(0.1, 40.1, 0.1), np.arange(40.1, 500, 10)))
    lam = np.arange(0.1, 4.1, 0.1)
    # lam = np.array([0.4]) # for test only
    data_aer = np.genfromtxt("data/profiles/aerosol_refraction.csv", delimiter=",")
    data_w = np.genfromtxt("data/profiles/water_refraction.csv", delimiter=",")

    real_dry = np.interp(lam, data_aer[:, 0], data_aer[:, 1])
    img_dry = np.interp(lam, data_aer[:, 0], data_aer[:, 2])
    real_w = np.interp(lam, data_w[:, 0], data_w[:, 1])
    img_w = np.interp(lam, data_w[:, 0], data_w[:, 2])

    # -------------modify refraction index and size distribution according to RH-------------
    RH = np.arange(0, 110, 10)  # relative humidity
    rh_fac = np.asarray(
        [1.0, 1.0, 1.0, 1.031, 1.055, 1.090, 1.150, 1.260, 1.554, 1.851, 2.151]
    )  # last number is infered by M.Li
    real_intM = np.zeros([len(RH), len(lam)])
    img_intM = np.zeros([len(RH), len(lam)])

    r = np.concatenate(
        (np.arange(0.001, 0.3, 0.001), np.arange(0.3, 20.05, 0.1))
    )  # in um
    # r = np.array([0.1,10])
    rm0 = np.asarray([0.135, 0.955])
    sigma0 = np.asarray([2.477, 2.051])  # * rh_fac[i]# in um
    n0 = np.asarray([1e4, 1])  # number/um-3
    Nr = np.zeros([len(RH), len(r)])  # number/um-3
    for i in range(0, len(RH)):
        # refraction index modification
        real_intM[i, :] = real_dry * rh_fac[i] ** (-3) + real_w * (
            1 - rh_fac[i] ** (-3)
        )
        img_intM[i, :] = img_dry * rh_fac[i] ** (-3) + img_w * (1 - rh_fac[i] ** (-3))
        # size modification
        rm = rm0 * rh_fac[i]
        sigma = sigma0  # *rh_fac[i] #* rh_fac[i]# in um
        dNr = (
            n0[0]
            / (np.sqrt(2 * np.pi) * np.log(sigma[0]))
            * np.exp(-0.5 * (np.log(r / rm[0]) / np.log(sigma[0])) ** 2)
        )
        dNr += (
            n0[1]
            / (np.sqrt(2 * np.pi) * np.log(sigma[1]))
            * np.exp(-0.5 * (np.log(r / rm[1]) / np.log(sigma[1])) ** 2)
        )
        Nr[i, :] = dNr / r  # same length as r, number/cm-4
    refrac_intM = real_intM + 1j * img_intM

    # create input list of args to parallel computation
    kappa_s = np.zeros([len(RH), len(lam)])
    kappa_a = np.zeros([len(RH), len(lam)])
    g_all = np.zeros([len(RH), len(lam)])
    f_M, g1_M, g2_M = [np.zeros([len(RH), len(lam)]) for i in range(0, 3)]
    cdf_M = np.zeros([len(RH), len(lam), len(angles)])
    fdelM_M = np.zeros([len(RH), len(lam)])
    for i in range(0, len(RH)):
        # args = [lam, refrac_intM[i, :], r, Nr[i, :]]
        list_args = []
        for j in range(0, len(lam)):
            args = [lam[j], refrac_intM[i, j], r, 'aerosol']  # single value of lam,refrac and r
            list_args.append(args)

        # result0 = aerosol_monoLam(list_args[0],True)
        # result1 =  result0 #aerosol_monoLam(list_args[1])
        # results=[result0, result1]
        # run in parallel
        pool = Pool()
        results = list(pool.map(aerosol_monoLam, list_args))
        pool.terminate()

        # re-organize the results from parallel computation
        for j in range(0, len(lam)):
            Qsca = results[j][0]
            Qabs = results[j][1]
            g = results[j][2]
            kappa_s[i, j] = np.trapz(Qsca * Nr[i, :] * np.pi * r ** 2, r)
            kappa_a[i, j] = np.trapz(Qabs * Nr[i, :] * np.pi * r ** 2, r)
            g_all[i, j] = (
                np.trapz(Qsca * g * Nr[i, :] * np.pi * r ** 2, r) / kappa_s[i, j]
            )
            if cdf == True:
                phase_Mie = results[j][7] # lam, r, angle
                cdf_M[i, j, :], fdelM_M[i,j] = Cal_cdf_TrAng(Qsca, Nr[i, :], r, mu, kappa_s[i, j], phase_Mie[:,::-1])
                continue
            # without infer number of particles-- correct in getMixKappa function
            f = results[j][3]
            g1 = results[j][4]
            g2= results[j][5]
            f_M[i,j] = (np.trapz(Qsca * f * Nr[i, :] * np.pi * r ** 2, r) / kappa_s[i, j])
            g1_M[i,j] = (np.trapz(Qsca * g1 * Nr[i, :] * np.pi * r ** 2, r) / kappa_s[i, j])
            g2_M[i,j] = (np.trapz(Qsca * g2 * Nr[i, :] * np.pi * r ** 2, r) / kappa_s[i, j])

            # ff = results[j][6]
            # ff_M[i,j] = (np.trapz(Qsca * ff * Nr[i, :] * np.pi * r ** 2, r) / kappa_s[i, j])

    if cdf == True:
        np.save("data/computed/cdf/cdf_aerosol", cdf_M)
        np.save("data/computed/fdelM_aerosol", fdelM_M)
    else:
        np.save("data/computed/ks_aerosol", kappa_s)
        np.save("data/computed/ka_aerosol", kappa_a)
        np.save("data/computed/g_aerosol", g_all)

        np.save("data/computed/f_aerosol", f_M)
        np.save("data/computed/g1_aerosol", g1_M)
        np.save("data/computed/g2_aerosol", g2_M)


def Cal_cdf_TrAng(Qsca_j, Nr, r, mu, ks_j, phaseFunc_j):
    """
    Calculate the cumulative distribution function (CDF) from a phase function.

    Parameters
    ----------
    phaseFunc_j : ndarray [r : 479, theta: 498]

    Returns
    -------
    cdf : array_like
        Cumulative distribution function values corresponding to the input angles.

    """
    Qsca_slice = Qsca_j.reshape(-1, 1)
    Nr_slice = Nr.reshape(-1, 1)
    r_slice = r.reshape(-1, 1)
    integrand = Qsca_slice * phaseFunc_j * Nr_slice * math.pi * r_slice ** 2
    ph_ = np.trapz(integrand, x=r, axis=0) / ks_j
    # cdf
    ph_ /= np.trapz(ph_, mu)  # the phase func is from 180 to 0. mu: (-1,1)
    cdf_v = calculate_cdf_mu(mu, ph_)
    # angle trunction
    theta_trunc = 1  # degrees
    muc = np.cos(np.deg2rad(theta_trunc))
    f_delM = 1.0 - np.interp(muc, mu, cdf_v)
    return cdf_v, f_delM


def cloud_efficiency(cdf=True):
    """
    Compute and save cloud Mie efficiencies.
    lam : wavelength [um], array_like (N_lam,)
    r : particle radii [um], used to build a size distribution
    Parameters
    ----------
    None.

    Returns
    -------
    None. Save computed data files to local directory. Each data file contains an array_like (N_lam, N_r).

    """
    #lam=np.concatenate((np.arange(0.1,40.1,0.1),np.arange(40.1,500,10)))
    lam = np.arange(0.1, 4.1, 0.1)
    #lam = [0.4,0.45,0.6] # 0.628,
    lam0=0.4975
    data_w = np.genfromtxt('data/profiles/water_refraction.csv', delimiter=',')
    real_w = np.interp(lam, data_w[:, 0], data_w[:, 1])
    img_w = np.interp(lam, data_w[:, 0], data_w[:, 2])
    refrac_w=real_w+1j*img_w
    r = np.arange(0.1,50,0.1)  # in um
    #r = np.array([10,20])
    # calculate the Qsca,Qabs,g for all combination of lam, refrac_w and r
    list_args=[]
    for j in range(0,len(lam)):
        args = [lam[j], refrac_w[j], r] # single value of lam,refrac and r
        list_args.append(args)
    # result0 = aerosol_monoLam(list_args[0],True, cdf=True)
    # result1 = aerosol_monoLam(list_args[1])
    # result2 = aerosol_monoLam(list_args[2])
    # results=[result0, result1, result2]
    pool = Pool()
    results = list(pool.map(aerosol_monoLam, list_args))
    pool.terminate()
    # re-organize the results from parallel computation
    Qsca, Qabs, g_M=[np.zeros([len(lam),len(r)]) for i in range(0,3)]
    f_M, g1_M, g2_M = [np.zeros([len(lam),len(r)]) for i in range(0,3)]
    ff_M = np.zeros([len(lam),len(r)])
    phase_Mie = np.zeros([len(lam), len(r), 498])
    for j in range(0,len(lam)):
        Qsca[j,:]=results[j][0]
        Qabs[j,:]=results[j][1]
        g_M[j,:]=results[j][2]
        if cdf == True:
            phase_Mie[j,:,:] = results[j][7]
            continue
        f_M[j,:]=results[j][3]
        g1_M[j,:]=results[j][4]
        g2_M[j,:]=results[j][5]
        ff_M[j,:]=results[j][6]

    np.save("data/computed/Qsca_clouds", Qsca)
    np.save("data/computed/Qabs_clouds", Qabs)
    np.save("data/computed/gM_clouds", g_M)
    if cdf == True:
        import time
        start_time = time.time()
        np.save("data/computed/phasefunc_clouds", phase_Mie[:,:,::-1]) # save in descending order of angle
        # 3. Record the end time
        end_time = time.time()
        print(f"Saving phase function took {end_time - start_time} seconds.")
    else:
        np.save("data/computed/f_clouds", f_M)
        np.save("data/computed/g1_clouds", g1_M)
        np.save("data/computed/g2_clouds", g2_M)
        np.save("data/computed/ff_clouds", ff_M)

def cloud(model,cld_model,z,kap, Ph_cdf=True):
    """

    Compute the absoprtion/scattering coefficients, and asymetry factor of water clouds.

    Parameters
    ----------
    model: str
        Profile model, for CIRC cases only.
    cld_model: str
        Cloud model, by default re=10 um and sig_e = 0.1.
    z : array (n_layer + 2,)
        Heights of the layers.
    kap : list of int
        Layer indexes that containing clouds. e.g. kap=[5,6,7] clouds are in layers 5,6 and 7.
    ff : forward fraction of the truncated phase function

    Returns
    -------
    ks_cld, ka_cld, g_cld : array (n_layer,n_nu)
        Cloud scattering and absorption coefficients (ks and ka), and asymmetry
        factors (g) for all layers and all considered wavenumbers.

    """
    angles = np.concatenate([
        np.arange(0, 2, 0.01),
        np.arange(2, 5, 0.05),
        np.arange(5, 10, 0.1),
        np.arange(10, 15, 0.5),
        np.arange(15, 176, 1),
        np.arange(176, 180 + 0.25, 0.25)])
    theta = np.deg2rad(angles)
    mu = np.cos(theta)[::-1]
    #lam=np.concatenate((np.arange(0.1,40.1,0.1),np.arange(40.1,500,10)))
    lam = np.arange(0.1, 4.1, 0.1)
    lam0=0.4975
    r = np.arange(0.1,50,0.1)  # in um
    Qsca = np.load("data/computed/Qsca_clouds.npy")
    Qabs = np.load("data/computed/Qabs_clouds.npy")
    g_M = np.load("data/computed/gM_clouds.npy")

    if Ph_cdf == True:
        phaseFunc = np.load("data/computed/cdf/phasefunc_clouds.npy")  # inversd 180 to 0, Nor to 1.
        #angle, phaseFunc, r = read_ic_yang2013("ic.plate_10elements.050.1.cdf", 0.65, 3)
    f_M = np.load("data/computed/TTHG/f_clouds.npy")
    g1_M = np.load("data/computed/TTHG/g1_clouds.npy")
    g2_M = np.load("data/computed/TTHG/g2_clouds.npy")

    ks_cld, ka_cld, g_cld = [np.zeros([len(z) - 1, len(lam)]) for i in range(0, 3)]
    f_cld, g1_cld, g2_cld = [np.zeros([len(z) - 1, len(lam)]) for i in range(0, 3)]
    ff_cld = np.zeros([len(z) - 1, len(lam)])
    fdelM_cld = np.zeros([len(z) - 1, len(lam)])  # from 180 to 0 degree
    cdf_cldz = np.zeros([len(z) - 1, len(lam), len(angles)]) # from 180 to 0 degree
    # default cloud model (re=10, sig_e=0.1)
    if ('default' in cld_model):#cld_model=='default'):
        re = 10 # *****effective radius in um, Barker 2003
        #re=5.4 #*****for continental clouds, Miles 2000
        sig_e = 0.1 # effective variance in um, Barker 2003
        Nr=r**(1/sig_e-3)*np.exp(-r/re/sig_e) # size distribution (gamma)
        ks, ka, g, f_delM=[np.zeros(len(lam)) for i in range(0,4)]
        f, g1, g2 = [np.zeros(len(lam)) for i in range(0, 3)]
        cdf_cld = np.zeros([len(lam), len(angles)])
        for j in range(0,len(lam)):
            ks[j] = np.trapz(Qsca[j,:] * Nr * math.pi * r ** 2, r)
            ka[j] = np.trapz(Qabs[j,:] * Nr * math.pi * r ** 2, r) 
            g[j] = np.trapz(Qsca[j,:]* g_M[j,:]* Nr * math.pi * r ** 2, r) / ks[j]
            if Ph_cdf == True:
                cdf_cld[j, :],f_delM[j] = Cal_cdf_TrAng(Qsca[j, :], Nr, r, mu, ks[j], phaseFunc[j, :, :])
            else:
                # TTHG
                f[j] = np.trapz(Qsca[j, :] * f_M[j, :] * Nr * math.pi * r ** 2, r) / ks[j]
                g1[j] = np.trapz(Qsca[j, :] * g1_M[j, :] * Nr * math.pi * r ** 2, r) / ks[j]
                g2[j] = np.trapz(Qsca[j, :] * g2_M[j, :] * Nr * math.pi * r ** 2, r) / ks[j]
                # The forward fraction of the truncated phase function

        dz_cld=z[kap[-1]+1]-z[kap[0]] # in m
        ke_cld_ref=1.0/(dz_cld*100.) # in cm-1, COD by default = 1.0
        ratio_cld=ke_cld_ref/np.interp(lam0,lam,ka+ks)
        for i in range(len(kap)):
            ks_cld[kap[i],:]=ks*ratio_cld
            ka_cld[kap[i],:]=ka*ratio_cld
            g_cld[kap[i],:]= g
            if Ph_cdf == True:
                cdf_cldz[kap[i],:,:] = cdf_cld
                g_cld[kap[i], :] = np.zeros(len(g)) - 2
                fdelM_cld[kap[i],:] = f_delM
            else:
                f_cld[kap[i], :] = f
                g1_cld[kap[i], :] = g1
                g2_cld[kap[i], :] = g2
    else: # cloud model of CIRC cases
        cld_file="data/CIRC/"+model+"_input&output/cloud_input_"+model+".txt"
        cld_input=np.genfromtxt(cld_file,skip_header=2)# layer number, CF, LWP, IWP,re_liq, re_ice
        reS=cld_input[:,4]
        LWP=cld_input[:,2]
        # create input list of args to parallel computation
        list_args = []
        for i in range(len(kap)):
            if ('re10' in cld_model):
                re=10
                sig_e=0.1
            else:
                re=reS[kap[i]] 
                sig_e=0.014 # spectral diserpation of 0.12 for all CIRC cases
            Nr=r**(1/sig_e-3)*np.exp(-r/re/sig_e) # size distribution (gamma)
            x_frac=np.trapz(Nr *4/3* math.pi * r ** 3, r) # volume fraction of water in air.
            ratio_cld=(LWP[kap[i]]/(z[kap[i]+1]-z[kap[i]])/100)/x_frac # z should in cm
            ks, ka, g, f_delM=[np.zeros(len(lam)) for i in range(0,4)]
            f, g1, g2 = [np.zeros(len(lam)) for i in range(0, 3)]
            cdf_cld = np.zeros([len(lam), len(angles)])
            for j in range(0,len(lam)):
                ks[j] = np.trapz(Qsca[j,:] * Nr * math.pi * r ** 2, r)
                ka[j] = np.trapz(Qabs[j,:] * Nr * math.pi * r ** 2, r) 
                g[j] = np.trapz(Qsca[j,:]* g_M[j,:]* Nr * math.pi * r ** 2, r) / ks[j]
                if Ph_cdf == True:
                    cdf_cld[j, :],f_delM[j] = Cal_cdf_TrAng(Qsca[j, :], Nr, r, mu, ks[j], phaseFunc[j, :, :])
                else:
                # TTHG
                    f[j] = np.trapz(Qsca[j, :] * f_M[j, :] * Nr * math.pi * r ** 2, r) / ks[j]
                    g1[j] = np.trapz(Qsca[j, :] * g1_M[j, :] * Nr * math.pi * r ** 2, r) / ks[j]
                    g2[j] = np.trapz(Qsca[j, :] * g2_M[j, :] * Nr * math.pi * r ** 2, r) / ks[j]

            ks_cld[kap[i],:]=ks*ratio_cld
            ka_cld[kap[i],:]=ka*ratio_cld
            g_cld[kap[i],:]=g
            if Ph_cdf == True:
                print('add set using cdf for cloud layer ', kap[i])
                cdf_cldz[kap[i],:,:] = cdf_cld
                g_cld[kap[i], :] = np.zeros(len(g)) - 2
                fdelM_cld[kap[i],:] = f_delM
            else:
                f_cld[kap[i],:]=f
                g1_cld[kap[i],:]=g1
                g2_cld[kap[i],:]=g2
    # corrected for LWP already
    return ks_cld,ka_cld,g_cld,f_cld,g1_cld,g2_cld,fdelM_cld, cdf_cldz

def calculate_cdf_mu(mu, p):

    # Calculate the cumulative distribution function (CDF) from the phase function
    # theta in radians, p is phase function values at corresponding theta
    # Ensure mu is sorted ascending: [-1 → 1]
    if mu[0] > mu[-1]:
        mu = mu[::-1]
        p = p[::-1]

    # cumulative trapezoidal integral
    cdf = np.concatenate(([0.0], cumulative_trapezoid(p, mu)))
    cdf /= cdf[-1]  # normalize so that max=1
    return cdf


def rayleigh_kappa_s(nu, N):
    """
    Compute Rayleigh scattering coefficient.

    Parameters
    ----------
    nu : (N_nu,) array_like
        Wavenumber [cm-1].
    N : float
        Number density [unit/cm^3] of air molecules.

    Returns
    -------
    kappa_s : (N_nu,) array_like
        Rayleigh scattering coefficient [cm^-1].

    Refernces
    -------
    [1] "Rayleigh-scattering calculations for the terrestrial atmosphere" (1995) by Anthony Bucholtz.

    """
    ns=5791817.0/(238.0185-(nu/1e4)**2)+167909.0/(57.362-(nu/1e4)**2)
    ns/=1e8
    ns+=1.0 # refraction index of standard air, function of lamda
    Ns=2.54743*1e19 # number density of stantdard air, [mole/cm3]
    data = np.genfromtxt('data/profiles/Air_Fk.csv', delimiter=',', skip_header=1)
    Fk = np.interp(-1e4/nu, -data[:, 0], data[:, 1]) #King correction factor from 1995_Bucholtz
    sigma = 24.0 * N / Ns ** 2 * math.pi ** 3 * (ns ** 2 - 1) ** 2  # in [cm^2], cross section
    sigma *= nu ** 4
    sigma /= (ns ** 2 + 2) ** 2  # lamda in [cm] bug fixed 10th April 2024,
    #sigma=24.0*nu**4*N*math.pi**3*(ns**2-1)**2 # in [cm^2], cross section
    #sigma/=Ns**2*(ns**2+2)**2 # lamda in [cm] 
    sigma*=Fk
    kappa_s=sigma #[cm2]*[mole/cm3]=[cm-1]
    return kappa_s





def surface_albedo(nu, surface):
    """
    Get surface albedo for different materials.

    Parameters
    ----------
    nu: (N_nu,) array_like
        spectral grid in wavenumber [cm-1].
    surface: string
        considered surface type, CIRC cases or PV or CSP
    
    Returns
    -------
    rho_s: (N_nu, N_deg) array_like
        spectral surface albedo.
    """
    lam = 1e4 / nu
    if 'case' in surface:
        filename = "./data/CIRC/" + surface + "_input&output/sfcalbedo_input_" + surface + ".txt"
        data = np.genfromtxt(filename, skip_header=6)
        rho_s = np.interp(nu, data[:, 0], data[:, 1])
    if surface == 'PV':
        filename = "data/profiles/Reflectance of PV.txt"
        data = np.genfromtxt(filename, skip_header=0)
        rho_s = np.interp(lam, data[:, 0] / 1e3, data[:, 1] / 1e2)  # data in nm and %
    if (surface == 'CSP'):
        filename = "data/profiles/Reflectance of CSP.txt"
        data = np.genfromtxt(filename, skip_header=1)
        rho_s1 = np.interp(lam, data[:, 0] / 1e3, data[:, 1])  # data in nm and %
        rho_s2 = np.interp(lam, data[:, 0] / 1e3, data[:, 2])  # data in nm and %
        rho_s3 = np.interp(lam, data[:, 0] / 1e3, data[:, 3])  # data in nm and %
        rho_s = np.concatenate((np.vstack(rho_s1), np.vstack(rho_s2), np.vstack(rho_s3)), axis=1)
    # else:
    #     site_name = 'BON'
    #     DOY = int(152)
    #     filename = f"data/albedo/{site_name}_spectral_albedo_DOY{DOY}.nc"
    #     da = xr.open_dataarray(filename)
    #     rho_s = np.interp(lam, da.wavelength.values, da.values)  # in nm and %
    return rho_s


# air mass correction for high zenith angle
def airMass(alt, theta):
    '''
    Air mass correction for high solar zenith angle accounting fo the curvature fo the Earth.
    Parameters
    ----------
    alt: float
        altitude of considered location [km].
    theta: float
        solar zenith angle [rad].
    
    Returns
    -------
    cor_airM: float
        corrected air mass.
    cor_theta: float
        corrected solar zenith angle [rad].
    
    References
    -------
    "Revised optical air mass tables and approximation formula" (1989) by Kasten and Young. 
    '''
    
    th = theta*180.0/math.pi
    airM = 1.0/np.cos(theta) # for high_aenith
    cor_airM = airM
    cor_theta = theta
    if (th > 70):
        cor_airM = np.exp(-0.0001184*alt)
        cor_airM /= np.cos(theta)+0.5057*(96.080-th)**(-1.634)
        cor_theta = np.arccos(1.0/cor_airM)
    return cor_airM,cor_theta


#---------------------functions for CIRC cases-------------------------
def set_vmr_circ(model, molecules, pa, ta):
    """
    Set volumetric mixing ratio for N layers (for CIRC cases).

    Parameters
    ----------
    model : str
        CIRC model (e.g. 'case2').
    molecules : (M,) array_like
        Names of the M molecules.
    pa : (N_layer + 1,) array_like
        Average pressure [Pa] of all layers.
    ta : (N_layer + 1,) array_like
        Average temperature [K] of all layers.

    Returns
    -------
    vmr : (N_layer + 1, M)
        Volume mixing ratio (vmr) of each layer for the M molcules.
    densities :( N_layer + 1, M)
        Average density of each gas in each layer (accumulated weight/volume
        [g/cm^3]).
    """

    vmr = np.zeros((len(molecules), len(pa)))
    densities = np.zeros((len(molecules), len(pa)))
    data = np.genfromtxt(
        "data/CIRC/{}_input&output/layer_input_{}.txt".format(model, model),
        skip_header=3,
    )
    data = np.vstack((data[0, :], data))
    for i in range(0, len(molecules)):
        if molecules[i] == "H2O":
            ref_vmr = data[:, 3]
            M = 18  # molecular weight, g/mol
        elif molecules[i] == "CO2":
            ref_vmr = data[:, 4]
            M = 44
        elif molecules[i] == "O3":
            ref_vmr = data[:, 5]
            M = 48
        elif molecules[i] == "N2O":
            ref_vmr = data[:, 6]
            M = 44
        elif molecules[i] == "CO":  # has CO in CIRC
            ref_vmr = data[:, 7]
            M = 28
        elif molecules[i] == "CH4":
            ref_vmr = data[:, 8]
            M = 16
        elif molecules[i] == "O2":
            ref_vmr = data[:, 9]
            M = 32
        elif molecules[i] == "N2":
            ref_vmr = data[:, 9] / 21 * 78  # no N2 in CIRC
            M = 28
        for j in range(1, len(pa)):
            Ru = 8.314  # J/mol K
            density = pa[j] * ref_vmr[j] * M / Ru / ta[j]  # g/m3
            densities[i, j] = density / 1e6  # g/cm3

        densities[i, 0] = densities[i, 1]
        vmr[i, :] = ref_vmr

    densities = np.transpose(densities)
    vmr = np.transpose(vmr)
    return vmr, densities


def aerosol_circ(model, lam, z):
    """
    Aerosol properties for CIRC cases.

    Parameters
    ----------
    model : str
        CIRC model (e.g. 'case2').
    lam : (N_lam,) array_like
        Wavelengths [um].
    z : (N_layer+2,) array_like
        Height of the node points.

    Returns
    -------
    ka_aer_M : (N_layer+2, N_lam) array_like
        Spectral absorption coefficients of each layer.
    ks_aer_M : (N_layer+2, N_lam) array_like
        Spectral scattering coefficients of each layer.
    g_aer_M : (N_layer+2, N_lam) array_like
        Spectral assymetry factors of each layer.
    """

    filename = "data/CIRC/{}_input&output/aerosol_input_{}.txt".format(model, model)
    A = np.genfromtxt(filename, skip_header=3, max_rows=1)
    data = np.genfromtxt(filename, skip_header=5)
    N_layer = data.shape[0] - 1
    ka_aer_M, ks_aer_M, g_aer_M = [
        np.zeros([N_layer + 1, len(lam)]) for i in range(0, 3)
    ]
    for i in range(1, N_layer + 1):
        dz = (z[i + 1] - z[i]) * 100  # in cm
        if data[i, 1] > 0:
            tau = data[i, 1] * lam ** (-A)
            ke_aer = tau / dz  # extinction coeff
            ks_aer_M[i, :] = ke_aer * data[i, 2]  # SSA is not a function of lam
            ka_aer_M[i, :] = ke_aer - ks_aer_M[i, :]
            g_aer_M[i, :] = data[i, 3]  # g is not a function of lam
    return [ka_aer_M, ks_aer_M, g_aer_M]


def cloud_height(N_layer,model,kap,th0_v):
    # N_layer = 54
    # model = 'AFGL midlatitude summer'
    # kap = [[8, 9, 10, 11, 12, 13]]
    theta0_v = th0_v / 180 * math.pi  # solar zenith angle in rad

    p, pa = set_pressure(N_layer)
    z, za = set_height(model, p, pa)
    dz_cld = z[kap[-1] + 1] - z[kap[0]]  # in m

    print(f"cloud bottom = {z[kap[0]]} m")
    print(f"cloud top = {z[kap[-1] + 1]} m")
    print(f"cloud depth = {dz_cld} m")
    return None

def compare_old_new_coeffM():
    import matplotlib.pyplot as plt
    import numpy as np
    Qabs_clouds = np.load("data/computed/Qabs_clouds.npy")
    Qsca_clouds = np.load("data/computed/Qsca_clouds.npy")
    gM_cld = np.load("data/computed/gM_clouds.npy")
    ka_aer = np.load("data/computed/ka_aerosol.npy")
    ks_aer = np.load("data/computed/ks_aerosol.npy")
    g_aer = np.load("data/computed/g_aerosol.npy")

    aer_ka_li = np.load("data/computed/original_version/ka_aerosol.npy")
    aer_ks_li = np.load("data/computed/original_version/ks_aerosol.npy")
    aer_g_li = np.load("data/computed/original_version/g_aerosol.npy")
    Qabs_clouds_li = np.load("data/computed/original_version/Qabs_clouds.npy")
    Qsca_clouds_li = np.load("data/computed/original_version/Qsca_clouds.npy")
    gM_cld_li = np.load("data/computed/original_version/gM_clouds.npy")

    plot_compare(aer_ka_li, ka_aer, 'ka_aer')
    plot_compare(aer_ks_li, ks_aer, 'ks_aer')
    plot_compare(aer_g_li, g_aer, 'g_aer')
    # plot_compare(Qabs_clouds_li, Qabs_clouds, 'Qabs_clouds')
    # plot_compare(Qsca_clouds_li, Qsca_clouds, 'Qsca_clouds')
    # plot_compare(gM_cld_li, gM_cld, 'gM_clouds')

def plot_compare(var_li, var, var_name):
    #lam = np.arange(0.1, 4.1, 0.1)
    lam_li = np.concatenate((np.arange(0.1, 40.1, 0.1), np.arange(40.1, 500, 10)))
    lam = lam_li
    if 'aer' in var_name:
        for i in range(11):
            plt.plot(lam_li[:],var_li[i][:], color=f'C0')
            plt.plot(lam, var[i], color=f'C3')
    else:
        plt.plot(lam_li[:], var_li[:], color=f'C1')
        plt.plot(lam, var, color=f'C3')
    plt.title(var_name)
    plt.yscale('log')
    plt.show()
#
if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # nu = np.arange(2500,35000,3)
    # N = 2.54743e19 # number density of air molecules [mole/cm3]
    # kappa_s=rayleigh_kappa_s(nu, N)
    # plt.plot(nu, kappa_s)
    # plt.show()
    # compare_old_new_coeffM()
    cloud_efficiency()
    # aerosol()