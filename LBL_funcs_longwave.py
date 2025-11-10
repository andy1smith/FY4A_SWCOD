"""
Longwave (LW) specific functions for the LBL model, two-flux integration.

Author: Mengying Li
"""

import numpy as np
import math
import copy
from scipy.special import expn  # exponential integral E_n
from LBL_funcs_fullSpectrum import *

__all__ = ["LBL_longwave", "monoFlux_expInt_scatter", "monoFluxN", "modifiedFs"]

# main program of the longwave model, parallel for computational speed up
def LBL_longwave(args):
    """
    Monochromatic flux density and broadband flux of N+2 layer boudaries.

    Parameters
    ----------
    args['properties'] : (5,) dict
        rh0: float, surface relative humidity, in range of 0 to 1 [unit].
        T_surf : float, surface temerature [K].
        AOD: float, surface aerosol optical depth at 497.5 nm [unit].
        COD: float, cloud optical depth at 495.5 nm [unit].
        kap: list of int, layer indexes that containing clouds. e.g. kap=[5,6,7] clouds are in layers 5,6 and 7.
    args['inputs'] : (8,) dict
        N_layer: int, the number of atmospheric layers.
        nu: (N_nu,) array_like, wavenumber grid [cm^-1].
        molecules: (M,) array_like, names of the M molecules.
        vmr0: (M,) dict, surface volumetric mixing ratio of the considered molecules. 
        model: string, profile model (e.g. 'tropical').
        cld_model: string, cloud model, by default re=10 um and sig_e = 0.1.
        period: string, 'day' or 'night', night profile includes temperature inversion.
        spectral: string, modeled spectral, either 'LW' or 'SW'.  
    args['alpha_s'] : (N_nu,) array-like
        Spectral surface absorptance.

    Returns
    -------
    outputs: (4,) list
        F_dw_mono, F_uw_mono : (N_layer + 2, N_nu) array_like
        Spectral downwelling, and upwelling flux density [W/(m^2 cm^-1)] at all layer boundaries.
        F_dw, F_uw : (N_layer+2,) array_like
        Broadband downwelling, and upwelling flux density [W/(m^2 cm^-1)] at all layer boundaries.

    """
    properties=args['properties']
    rh0=properties['rh0']
    T_surf=properties['T_surf']
    AOD=properties['AOD']
    COD=properties['COD']
    kap=properties['kap']

    inputs=args['inputs']
    N_layer=inputs['N_layer']
    nu=inputs['nu']
    molecules=inputs['molecules']
    vmr0=inputs['vmr0']
    model=inputs['model']
    cld_model=inputs['cld_model']
    period=inputs['period']
    spectral=inputs['spectral']

    p,pa=set_pressure(N_layer)
    z,za=set_height(model,p,pa)
    t,ta=set_temperature(model, p, pa, T_surf, period)
    n,na=set_ndensity(model,p,pa)
    
    F_dw_os=Planck(nu,5778)*math.pi*2.16*1e-5 #intensity,longwave from the Sun @ 5778 K
    alpha_s=args['alpha_s']
    ps=saturation_pressure(t)
    if (vmr0['H2O'] != 0):
        vmr0['H2O']= rh0 * ps[1] / p[1] # for water vapor, dependent on local humidity


    vmr,densities=set_vmr(model,molecules,vmr0,z)
    gas_coeff,aer_coeff,cld_coeff,mix_coeff=getMixKappa(inputs,densities,pa,ta,z,za,na,AOD,COD,kap)# get mixture absorption coefficients
    # optional, save the coefficient
    #fileName="Coeff_{}layers_{}_RH={}_Tsurf={}_AOD={}_COD={}_kap={}.npy".format(
    #        N_layer, model,rh0,T_surf,AOD,COD,kap)
    #np.save('results_longwave/grid/'+fileName,mix_coeff)  
    del gas_coeff,aer_coeff,cld_coeff # free up the memory

    F_dw_mono,F_uw_mono=monoFlux_expInt_scatter(N_layer,nu,mix_coeff[0],
                                                           mix_coeff[1],mix_coeff[2],ta,z,alpha_s,F_dw_os)

    # integrate flux layer by layer
    F_dw, F_uw =  [np.zeros((N_layer+2,1)) for i in range(0,2)]
    F_dw_mono[np.isnan(F_dw_mono)]=0.0 # remove NaN values
    F_uw_mono[np.isnan(F_uw_mono)]=0.0
    for j in range(0,N_layer+2): # range in atmosphere layers
        F_dw[j]=np.trapz(F_dw_mono[j,:],nu) # change unit of nu from cm-1 to m-1
        F_uw[j]=np.trapz(F_uw_mono[j,:],nu)
    print ("Finish LBL_longwave once.")
    
    outputs={'F_dw_mono': F_dw_mono,'F_uw_mono': F_uw_mono, 'F_dw': F_dw, 'F_uw': F_uw}
    return outputs 

def monoFlux_expInt_scatter(
    N_layer, nu, ka_M, ks_M, g_M, ta, z, alpha_s, F_dw_os):
    """
    Monochromatic flux density of N Layers.

    Parameters
    ----------
    N_layer : int
        Number of atmosphere layers.
    nu : (N_nu,) array_like
        Wavenumbers [cm^-1].
    ka_M, ks_M, g_M : (N_layer + 1, N_nu) array_like
        Spectral absorption/scattering coefficient and asymetry factor of all layers.
    ta : (N_layer + 1,) array_like 
        Average temperature [K] of each layer (including the ground surface).
    z : (N_layer + 2,) array_like
        Height [m] of the layer boundaries.
    alpha_s : (N_nu,) array_like
        Spectral absorptance of the surface, by default is 1.
    F_dw_os : (N_nu,) array_like
        Spectral extraterrestrial longwave intensity [W/(m^2 sr)].

    Returns
    -------
    F_dw_mono, F_uw_mono : (N_layer + 2, N_nu) array_like
        Spectral downwelling, and upwelling flux density [W/(m^2 cm^-1)] at all layer boundaries.

    """
    # calculate and save variables as numpy 2D array
    abs_coeff, sca_coeff, ext_coeff, g_asy = [np.zeros((N_layer + 1, len(nu))) for i in range(0,4)]  # volumetric
    albedo, A_star, Ib = [np.zeros((N_layer + 2, len(nu))) for i in range(0,3)]  # volume
    A_star += 1  # A*=1 for surfaces, boundary

    for i in range(1, N_layer + 1): # compute layer-by-layer
        abs_coeff[i, :] = ka_M[i, :]
        sca_coeff[i, :] = ks_M[i, :]
        g_asy[i, :] = g_M[i, :]
        ext_coeff[i, :] = sca_coeff[i, :] + abs_coeff[i, :]
        albedo[i, :] = sca_coeff[i, :] / ext_coeff[i, :]

        # delta_M scaling for anisotropic scattering
        ext_coeff[i, :] *= 1 - albedo[i, :] * g_asy[i, :]
        albedo[i, :] *= (1 - g_asy[i, :]) / (1 - albedo[i, :] * g_asy[i, :])

        # use scaled extinction coefficient (based on z [m])
        A_star[i, :] = 4 * ext_coeff[i, :] * (z[i + 1] - z[i]) * 1e2
        Ib[i, :] = Planck(nu, ta[i]) * np.pi

    Ib[0, :] = Planck(nu, ta[0]) * np.pi * alpha_s
    Ib[N_layer + 1, :] = F_dw_os * np.pi  # outer space irradiance
    albedo[0, :] = 1 - alpha_s  # surface albedo

    # calculate normal optical depth and Delta optical depth
    # inside Earth has od=-inf, outer space has od=+inf
    #
    # (N_layer + 3, N_nu):
    #   - N_layer + 1 optical depth boundary
    #   - 1 with -inf
    #   - 1 with +inf
    #   - ==> N_layer + 3 total
    #
    od = np.zeros((N_layer + 3, len(nu)))
    od[0, :] = np.zeros(len(nu)) - 1e39  # -inf od
    od[N_layer + 2, :] = np.zeros(len(nu)) + 1e39  # +inf od
    D_od = np.zeros((N_layer + 1, len(nu)))
    for i in range(1, N_layer + 1):  # i=[1....,18]
        D_od[i, :] = ext_coeff[i, :] * (z[i + 1] - z[i]) * 1e2  # unit of 1
        od[i + 1, :] = od[i, :] + D_od[i, :]

    # compute exp3 of optDepth[i,:]-optDepth[j,:] to save computation cost (for i>j)
    exp3 = []  # size (N_layer+3)*(N_layer+3)
    for i in range(N_layer + 3):
        expi = []
        for j in range(0, N_layer + 3):
            expj = expn(3, abs(od[i, :] - od[j, :]))
            expi.append(expj)
        exp3.append(expi)
    # print ("Finish calculating exp3.")

    # compute transfer factors
    # initialize size of (N_layer+2)*(N_layer+2), each with len(nu) vector
    Fs = [[np.zeros(len(nu)) for i in range(N_layer + 2)] for i in range(N_layer + 2)]
    # compute inter transfer factors
    for i in range(N_layer + 2):
        for j in range(N_layer + 2):
            if i != j:
                Fs[i][j] = (
                    2 * exp3[j][i + 1]
                    + 2 * exp3[j + 1][i]
                    - 2 * exp3[j][i]
                    - 2 * exp3[j + 1][i + 1]
                )
                Fs[i][j] /= A_star[i, :]
            else:
                Fs[i][j] = 1 - 0.5 / (od[i + 1, :] - od[i, :]) * (
                    1 - 2 * exp3[i][i + 1]
                )
                if i == 0 or i == N_layer + 1:
                    Fs[i][j] = np.zeros(len(nu))  # for Earth and outer space

    # use modFs to calculate Js and Gs
    modFs = modifiedFs(Fs, albedo)
    del Fs
    # save modFs, optional
    #np.save('results_longwave/modFs/modFs_18layers_RH=0.25_AOD=0.0',modFs)

    Js = np.zeros((N_layer + 2, len(nu)))
    Gs = np.zeros((N_layer + 2, len(nu)))
    for i in range(N_layer + 2):
        for j in range(N_layer + 2):
            Gs[i, :] += modFs[i][j] * Ib[j, :]
        Js[i, :] = (1 - albedo[i, :]) * Ib[i, :] + albedo[i, :] * Gs[i, :]
    del modFs

    # calculate downwelling flux from Js and re-calculate Fs
    F_dw_mono = np.zeros((N_layer + 2, len(nu)))
    for n in range(1, N_layer + 2):  # n = 1,2... N_layer
        fs = np.zeros((N_layer + 2, len(nu)))
        for j in range(n, N_layer + 2):  # j =  n,...N_layer
            fs[j, :] = 2 * exp3[j][n] - 2 * exp3[j + 1][n]
            F_dw_mono[n, :] += fs[j, :] * Js[j, :]

    # calculate upwelling flux from Js and re-calculate Fs
    F_uw_mono = np.zeros((N_layer + 2, len(nu)))
    F_uw_mono[1, :] = Js[0, :]  # surface flux
    for n in range(2, N_layer + 2):
        fs = np.zeros((N_layer + 2, len(nu)))
        for j in range(0, n):  # j=1,... n-1
            fs[j, :] = 2 * exp3[j + 1][n] - 2 * exp3[j][n]
            F_uw_mono[n, :] += fs[j, :] * Js[j, :]

    return F_dw_mono, F_uw_mono


def monoFluxN(args):
    """
    Monochromatic radiosity of the N layers, use matrix inversion to solve, slower than using the plating algorithm.

    Parameters
    ----------
    args : list
        albedo : (N_layer + 2,) array_like
            Single scattering albedo.
        Ib : (N_layer + 2,) array_like
            Planck's emissive flux [W/(m^2 cm^-1)]
        Fs: (N_layer + 2, N_layer + 2) array_like
            The transfer factor matrix.

    Returns
    -------
    J_v: array_like (N_layer,)
        Monochromatic radiosity of the N layers.
    """

    # unpack inputs
    albedo = args[0]
    eps = 1 - albedo
    Ib = args[1]
    Fs = args[2]  # F matrix (N+1)*(N+1)

    # construct matrix A and b
    N = len(Ib)
    mtx_A = np.zeros([2 * N, 2 * N])
    mtx_A[0:N, N:] = Fs
    mtx_A[N:, 0:N] = np.diag(albedo, 0)
    b_v = np.zeros([2 * N, 1])
    b_v[N:, 0] = eps * Ib

    # compute radiosity and irradiance vector
    mtx_I = np.identity(2 * N)  # identity matrix
    y_v = np.linalg.solve(mtx_I - mtx_A, b_v)  # solve (mtx_I-mtx_A)y_v=b_v
    J_v = np.hstack(y_v[N:])  # radiosity,horizontal vector
    return J_v


def modifiedFs(Fs, rho):
    """

    Use the plating algorithm to compute modified transfer factors for
    scattering medium.

    Parameters
    ---------
    Fs : (N_layer + 2, N_layer + 2) array_like
        Transfer factor of non-scattering medium.
    rho : (N_layer + 1,) array_like
        Single scattering albedo of the layers.

    Returns
    -------
    modFs : (N_layer + 2, N_layer + 2) array_like
        Modified transfer factors.
    """

    N_layer = len(Fs) - 2
    modFs = copy.deepcopy(Fs)
    newFs = copy.deepcopy(Fs)
    del Fs

    for k in range(1, N_layer + 1):  # plating only gas layers: 1,2,...N_layer
        rho_k = rho[k, :]  # single albedo of k-th gas layer
        eps_k = 1 - rho_k
        D = 1 - rho_k * modFs[k][k]
        for i in range(0, N_layer + 2):
            for j in range(0, N_layer + 2):
                if i != k:
                    if j != k:
                        newFs[i][j] = (
                            modFs[i][j] + rho_k * modFs[i][k] * modFs[k][j] / D
                        )
                    elif j == k:
                        newFs[i][j] = modFs[i][j] * eps_k / D
                elif i == k:
                    if j != k:
                        newFs[i][j] = modFs[i][j] * eps_k / D
                    elif j == k:
                        newFs[i][j] = modFs[i][j] * eps_k * eps_k / D
        modFs = copy.deepcopy(newFs)
    return modFs
