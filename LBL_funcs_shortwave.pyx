"""
    Functions for Monte Carlo shortwave radiative transfer.
    Coded in Cython to improve computational speed.
    
    Author: Mengying Li
"""


# import all necessary libraries
import numpy as np
import math
cimport numpy as np
import random
from multiprocessing import Pool

from LBL_funcs_fullSpectrum import *
from LBL_funcs_inclined import *
from matplotlib import pyplot as plt

from libc.math cimport *
from libc.stdlib cimport rand, RAND_MAX

__all__ = [
    "plot_MCscatter",
    "cartesian_to_spherical",
    "LBL_shortwave",
    "MonteCarlo_mono",
    "MonteCarlo_photon",
    "MonteCarlo_photon_curr",
    "MonteCarlo_ground",
    "MonteCarlo_scatter",
]
def plot_MCscatter(a):
    #twin
    fig = plt.figure(figsize=(6,4))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=0.2, hspace=0.4)
    ax1 = fig.add_subplot(gs1[0])

    for i in range(8000):
        rxyz = np.array([0,0,-1])
        rx,ry,rz=MonteCarlo_scatter(rxyz, 0.5,0.1,0.1,0.1, cdf_cld, mu_)
        if rz<=0:
            continue
        theta,phi=cartesian_to_spherical(rx, ry, rz)
        del rx,ry,rz
        ax1.scatter(90-np.rad2deg(theta),np.rad2deg(phi),color='C0',s=0.5)
    ax1.set_xlim(0,90)
    plt.show()
    return None

def cartesian_to_spherical(rx, ry, rz):
    """
    Convert a 3D Cartesian coordinate to spherical coordinates (theta, phi).

    Parameters:
        rx (float): x-component of the vector
        ry (float): y-component of the vector
        rz (float): z-component of the vector

    Returns:
        theta (float): Angle from the positive z-axis (0 to pi/2)
        phi (float): Azimuthal angle in the xy-plane from the x-axis (0 to 2pi)
    """
    # Compute the magnitude of the vector
    r = np.sqrt(rx**2 + ry**2 + rz**2)

    # Calculate theta and phi
    theta = np.arccos(rz / r)
    phi = np.arctan2(ry, rx)

    # Adjust phi to be in the range (-pi, pi]
    phi[phi >= np.pi] -= 2 * np.pi
    phi[phi < -np.pi] += 2*np.pi

    return theta, phi

cpdef LBL_shortwave(properties,inputs_main,angles,finitePP):
    """
    Monochromatic flux density and broadband flux of N+2 layer boudaries.

    Parameters
    ----------
    properties : (5,) dict
        rh0: float, surface relative humidity, in range of 0 to 1 [unit].
        T_surf : float, surface temerature [K].
        AOD: float, surface aerosol optical depth at 497.5 nm [unit].
        COD: float, cloud optical depth at 495.5 nm [unit].
        kap: list of int, layer indexes that containing clouds. e.g. kap=[5,6,7] clouds are in layers 5,6 and 7.
    inputs_main : (11,) dict
        N_layer: int, the number of atmospheric layers.
        N_bundle: int, the number of photon bundles per wavelength.
        nu: (N_nu,) array_like, wavenumber grid [cm^-1].
        molecules: (M,) array_like, names of the M molecules.
        vmr0: (M,) dict, surface volumetric mixing ratio of the considered molecules. 
        model: string, profile model (e.g. 'tropical').
        cld_model: string, cloud model, by default re=10 um and sig_e = 0.1.
        period: string, 'day' or 'night', night profile includes temperature inversion.
        spectral: string, modeled spectral, either 'LW' or 'SW'. 
        surface: string, the name of ground surface, e.g. PV, CSP, case 2 (CIRC cases). 
        alt: float, the altitude of considered location. [km]
    angles : (6,) dict
        Spectral surface absorptance.
        theta0, phi0: float
            solar zenith and azimuth angles [rad].
        del_angle: float
            receptance angle of DNI, default is 0.5 degree.
        beta, phi: array_like
            tilt and azimuth angles of the inclined surface [rad].
        isTilted: bool
            indicator of whether to do transpostion computation.
    finitePP: (4,) dict
        (x0, y0): float, photon starting coordinates [km].
        R_pp: float, power plant radius [km].
        is_pp: bool, indicator of whether to consider finite power plant.
    Returns
    -------
    out2: dictionary of mono-flux densities [W m-2 cm]
        F_dw/F_uw/F_gas: (N_layer+2, N_nu) array_like
            donwelling/upwelling/absorbed mono-flux density at all layer boundaries or within layers.
        F_dni/F_dhi: (N_nu,) array_like
            surface direct and diffuse mono-flux density.
        F_ghi_2D: (N_x,N_y) array_like
            surface GHI distribution over an (x,y) grid, for finite power plant approach.
        F_inclined/F_dcs: dictionary of (N_beta,N_phi,N_nu) arrays containing
            mono-flux density on inclined surfaces with orientation of (beta,phi).
    """
    # unpack inputs
    cdef float rh0,T_surf, AOD, COD, alt
    cdef int N_layer, N_bundles
    cdef bint deltaM = True
    rh0=properties['rh0']
    T_surf=properties['T_surf']
    AOD=properties['AOD']
    COD=properties['COD']
    kap=properties['kap']
    # !!!!!!! temporaly added for test !!!!!!!!!
    # surf_albedo = properties['surf_albedo']

    N_layer=inputs_main['N_layer']
    N_bundles=inputs_main['N_bundles']
    nu=inputs_main['nu']
    molecules=inputs_main['molecules']
    vmr0=inputs_main['vmr0']
    model=inputs_main['model']
    cld_model=inputs_main['cld_model']
    period=inputs_main['period']
    spectral=inputs_main['spectral']
    surface=inputs_main['surface']
    alt=inputs_main['alt']
    # compute required optical properties

    p, pa = set_pressure(N_layer)
    z, za = set_height(model, p, pa)
    t, ta = set_temperature(model, p, pa, T_surf, period)
    n, na = set_ndensity(model, p, pa)
    ps=saturation_pressure(t)
    if (vmr0['H2O'] != 0):
        vmr0['H2O']= rh0 * ps[1] / p[1] # for water vapor, dependent on local humidity
    vmr, densities = set_vmr(model, molecules, vmr0, z)
    # TPW = total_precipitable_water(densities[:,0],pa,ta,p[1:])
    coeff_gas, coeff_aer, coeff_cld, coeff_all, cdf_dict = getMixKappa(inputs_main, densities, pa, ta, z, za, na,
                                                                     AOD, COD, kap)
    cdf_aer = cdf_dict["cdf_aer_M"]
    cdf_cld = cdf_dict["cdf_cld_M"]

    ke_M=coeff_all[0]+coeff_all[1]
    rho_mix_M=coeff_all[1]/ke_M
    rho_mix_M[np.isnan(rho_mix_M)]=0
    sca_gas_M=coeff_gas[1]/coeff_all[1]
    sca_gas_M[np.isnan(sca_gas_M)]=0
    sca_aer_M=(coeff_gas[1]+coeff_aer[1])/coeff_all[1]
    sca_aer_M[np.isnan(sca_aer_M)]=0
    g_aer_M=coeff_aer[2]
    g_cld_M=coeff_cld[2]
    # two-term-HG
    f_aer_M = coeff_aer[3]
    g1_aer_M = coeff_aer[4]
    g2_aer_M = coeff_aer[5]
    f_cld_M = coeff_cld[3]
    g1_cld_M = coeff_cld[4]
    g2_cld_M = coeff_cld[5]

    # Delta-M scaling
    fdelM_aer = coeff_aer[6]
    fdelM_cld = coeff_cld[6]

    # if deltaM == True:
    #    print('Delta-M scaling turned on.')
    #    ke_M *= (1 - rho_mix_M * fdelM_aer)
    #    ke_M *= (1 - rho_mix_M * fdelM_cld)  # tested: the logic is correct for COD=0
    #    rho_mix_M *= (1 - fdelM_aer) / (1 - rho_mix_M * fdelM_aer)
    #    rho_mix_M *= (1 - fdelM_cld) / (1 - rho_mix_M * fdelM_cld)

    # Solor TOA and surface albedo
    data = np.genfromtxt('./data/profiles/ASTMG173.csv', delimiter=',', skip_header=2,  # in wavenumber basis
                         names=['wavelength', 'extraterrestrial', '37tilt', 'direct_circum'])
    ref_lam = data['wavelength']  # nm avoid hearder 1
    ref_E = data['extraterrestrial']
    ref_E_nu = -ref_E * ref_lam ** 2 / 1e7  # W/[m2*nm-1] tp W/[m2*cm-1]
    F_dw_os = -np.interp(-nu, -1e7 / ref_lam, ref_E_nu)  # W/[m2*cm-1] to W/cm-1

    alpha_s=  1.0-surface_albedo(nu,surface) # surface albedo #np.zeros(nu.shape[0]) + 1 - surf_albedo
    alpha_s_g=  1.0-surface_albedo(nu,'case2') # default ground albedo, hard-coded 'case2'.
    # corrected zenith angle for th>70 deg
    cdef float theta0,phi0
    theta0=angles['theta0']
    phi0=angles['phi0']
    cor_airM, cor_theta0=airMass(alt,theta0) #*******
    angles_cor=angles
    angles_cor['theta0']= cor_theta0
    # Nan update 2025/7/17
    # ke_M = ke_M * cor_airM # correct total extinction coefficient by air mass
    #****** cor_theta0
    angles_ = np.concatenate([
        np.arange(0, 2, 0.01),
        np.arange(2, 5, 0.05),
        np.arange(5, 10, 0.1),
        np.arange(10, 15, 0.5),
        np.arange(15, 176, 1),
        np.arange(176, 180 + 0.25, 0.25)
    ])
    theta = np.deg2rad(angles_)
    mu = np.cos(theta)[::-1]

    z_V=z*100.0 # change unit to cm, so z_V*ke has unit of 1
    z_V=np.hstack(z_V)
    cdef int N_lam=len(nu)
    # prepare inputs for parallel computing
    list_args = []
    for k in range(0, (int)(N_lam)):
        if (surface=='CSP'):
            temp=alpha_s[k,:]
        else:
            temp=alpha_s[k]
        # inputs is a python dict object
        inputs={'nu':nu[k],'N_layer':N_layer,'z_V':z_V,'surface':surface,'alpha_s':temp,'alpha_s_g':alpha_s_g[k],
        'ke':ke_M[:,k],'rho_mix':rho_mix_M[:,k],'sca_gas':sca_gas_M[:,k],'sca_aer':sca_aer_M[:,k],
        'g_aer':g_aer_M[:,k],'g_c':g_cld_M[:,k],
        'f_aer':f_aer_M[:,k],'g1_aer':g1_aer_M[:,k],'g2_aer':g2_aer_M[:,k],
        'f_cld': f_cld_M[:, k], 'g1_cld': g1_cld_M[:, k], 'g2_cld': g2_cld_M[:, k],
        'cdf_cld':cdf_cld[:,k], 'cdf_aer':cdf_aer[:,k], 'mu':mu,
                }
        args = [N_bundles,inputs,angles_cor,F_dw_os[k],finitePP]
        list_args.append(args)
    # iterate line-by-line and bundle-by-bundle, parallel
    pool=Pool()
    results = list(pool.map(MonteCarlo_mono, list_args))
    pool.terminate()

    # process results to output dni, ghi, dhi, and irradiance on inclined surfaces
    n_uw_M,n_dw_M,n_gas_M= [np.zeros((N_layer + 2,N_lam)) for i in range(0, 3)] # total across boundaries
    uw_rxyz_M, dw_rxyz_M, uw_xyz_M, dw_xyz_M = [[] for i in range(0,4)] # for transposition and finite power plant cases
    dw_rx,dw_ry,dw_rz,uw_rx,uw_ry,uw_rz=[np.zeros((N_bundles*2,N_lam))*np.nan for i in range(0,6)]
    dw_x,dw_y,dw_z,uw_x,uw_y,uw_z=[np.zeros((N_bundles*2,N_lam))*np.nan for i in range(0,6)] # track photon location
    # grid for 2D GHI distribution
    bins_x=np.hstack((np.arange(-100.0,-10.0,2.0),np.arange(-10.0,10.1,0.1),np.arange(11.0,101.0,2.0)))
    bins_y=np.hstack((np.arange(-100.0,-10.0,2.0),np.arange(-10.0,10.1,0.1),np.arange(11.0,101.0,2.0)))
    F_ghi_2D=np.zeros((len(bins_x)-1,len(bins_y)-1))

    for k in range(0, (int)(N_lam)):  # line-by-line
        output = results[k]
        n_uw_M[:, k] = output['n_uw']
        n_dw_M[:, k] = output['n_dw']
        n_gas_M[:, k] = output['n_gas']

        uw_rxyz = output['uw_rxyz']
        dw_rxyz = output['dw_rxyz']
        uw_xyz = output['uw_xyz']
        dw_xyz = output['dw_xyz']
        N_uw = len(uw_rxyz)
        N_dw = len(dw_rxyz)

        uw_rx[0:N_uw,k]=np.array([x[0] for x in uw_rxyz])
        uw_ry[0:N_uw,k]=np.array([x[1] for x in uw_rxyz])
        uw_rz[0:N_uw,k]=np.array([x[2] for x in uw_rxyz])
        dw_rx[0:N_dw,k]=np.array([x[0] for x in dw_rxyz])
        dw_ry[0:N_dw,k]=np.array([x[1] for x in dw_rxyz])
        dw_rz[0:N_dw,k]=np.array([x[2] for x in dw_rxyz])

        # dw_x[0:N_dw,k]=np.array([x[0] for x in dw_xyz])
        # dw_y[0:N_dw,k]=np.array([x[1] for x in dw_xyz])
        # dw_z[0:N_dw,k]=np.array([x[2] for x in dw_xyz])

        # # save F_ghi in terms distance from power plant center (0,0,0)
        # dw_x_V=dw_x[0:N_dw,k]
        # dw_y_V=dw_y[0:N_dw,k]
        # ind_nan=np.isnan(dw_x_V)
        # dw_x_V = dw_x_V[~ind_nan]*1.0e-5# unit of km
        # dw_y_V = dw_y_V[~ind_nan]*1.0e-5# unit of km
        # H, xedges, yedges = np.histogram2d(dw_x_V, dw_y_V, bins=(bins_x, bins_y))
        # F_ghi_2D+=H*F_dw_os[k]*cos(theta0)/(N_bundles*1.0)
        # ****** uncomment if uw_rxyz, dw_rxyz, dw_xyz need to be saved to local files******
        #uw_rxyz_M.append(uw_rxyz)
        #uw_xyz_M.append(uw_xyz)
        #dw_rxyz_M.append(dw_rxyz)
        #dw_xyz_M.append(dw_xyz)
    # np.save("shortwave_results/TOC/uw_rxyz_AOD=0.05_COD=10.0",uw_rxyz_M) # change the directory
    # np.save("shortwave_results/Inclined/dw_rxyz_AOD=0.05_COD=10.0",dw_rxyz_M) # change the directory
    # np.save("shortwave_results/Albedo/dw_xyz_"+surface,dw_xyz_M) # change the directory

    ratio = F_dw_os * np.cos(theta0) / (N_bundles * 1.0)
    # get transposition results
    out = MCtransposition(uw_rx, uw_ry, uw_rz, dw_rx, dw_ry, dw_rz, angles_cor, ratio)
    out1 = {'F_dw': n_dw_M[1,:] * ratio, 'F_uw': n_uw_M[-1,:] * ratio,
            'F_dni':out['F_dni'],'F_dhi':out['F_dhi']}
    out2 = {'uw_rxyz_M':uw_rxyz_M}# #'uw_xyz_M':uw_xyz_M}
    #out3 = {'uw_rxyz_M':uw_rxyz_M, 'uw_xyz_M':uw_xyz_M}
    return out1, out2 #, out3 , TPW
    # out3={'ke_M':ke_M,'rho_mix_M':rho_mix_M, 'coeff_gas': coeff_gas,
    #       'coeff_all':coeff_all, 'coeff_cld':coeff_cld,
    #       'coeff_aer':coeff_aer, 'sca_gas_M':sca_gas_M, 'sca_aero_M':sca_aer_M}
    # return out3 # return results


cpdef MonteCarlo_mono(args):
    """
    Monte Carlo simulation of single wavelength/wavenumber.

    Parameters
    ----------
    args : (5,) list
        N_bundles: int, number of photon bundles.
        inputs: dict, layer optical properties, remain unchanged during the computation.
        angles_cor: dict of angles.
        F_dw_os: float, mono-extraterretial solar intensity [W m-2 cm].
        finitePP: dict, inputs for finite power plant computation. 

    Returns
    -------
    outputs: dict
        recursively changed statistics of photon bundles.
    """
    cdef int N_bundles = args[0]
    cdef float F_dw_os,theta0,phi0
    inputs = args[1]
    z_V=inputs['z_V']
    angles = args[2]
    F_dw_os=args[3]
    finitePP=args[4]
    theta0 = angles['theta0']
    phi0 = angles['phi0']

    cdef int n=0,i,currN0
    cdef bint isAlive0
    cdef float rx0,ry0,rz0
    cdef int N_layer=inputs['N_layer']
    cdef float PI=3.1415927
    n_uw, n_dw, n_gas = [np.zeros((N_layer + 2),dtype=int) for i in range(0, 3)]
    uw_rxyz, dw_rxyz, uw_xyz, dw_xyz=[[] for i in range(0,4)]

    outputs = {'n_uw': n_uw, 'n_dw': n_dw,'n_gas': n_gas,
    'uw_rxyz': uw_rxyz, 'dw_rxyz':dw_rxyz,'uw_xyz': uw_xyz, 'dw_xyz':dw_xyz, 'N_sca':0} # record direction only on surface, to save time
    isAlive0 = True
    rx0 = sin(theta0) * cos(phi0)
    ry0= sin(theta0) * sin(phi0)
    rz0 = -cos(theta0) # modified 2/4/19
    rxyz0=[rx0,ry0,rz0]
    currN0 = inputs['N_layer']
    for n in range(0, N_bundles):  # bundle-by-bundle
        xyz0=[finitePP['x0']*1e5,finitePP['y0']*1e5,z_V[-1]] # from TOA, unit of cm, track photon location
        outputs['N_sca']= 0 # reset number of scattering events
        isAlive,currN,rxyz,xyz,outputs = MonteCarlo_photon(isAlive0,currN0,rxyz0,xyz0,outputs,inputs,finitePP)
    return outputs


cpdef MonteCarlo_photon(bint isAlive,int currN, rxyz, xyz, outputs, inputs,finitePP):
    """
    Monte Carlo simulation of one photon traveling within ALL layers of atmosphere.
    
    Parameters
    ----------
    isAlive,currN,rxyz, xyz: boolean,int,list,list
        recursively changed variables, the status of one photon: 
        whether it is alive, current layer, current traveling direction rxyz, current location xyz
    outputs: recursively changed statistics of photon bundles.
    inputs, finitePP: unchanged inputs for layer optical properties and finite power plant calculation.
    
    Returns
    -------
    isAlive,currN,rxyz, xyz: photon status variables. 
    outputs: dict
        recursively changed statistics of photon bundles.
    """
    cdef float tau,L,s0,z1,zupper,zlower # define variable type to improve speed
    z_V=inputs['z_V']
    ke=inputs['ke']
    cdef int N_layer=inputs['N_layer']
    tau= -log(rand()/(RAND_MAX*1.0))  # sampled optical path,random is open range [0,1)
    while (isAlive):
    #while (isAlive and outputs['N_sca'] < 1e6): # control the number of scattering events to prevent infinite loops
        L = tau / ke[currN]  # positive
        # if photon travel downwards
        if (rxyz[2]<0):
            zlower = z_V[currN] # lower bound,11/27
            s0 = (xyz[2] - zlower) / fabs(rxyz[2])  # remaining distance to lower layer
            z1 = xyz[2] + L * rxyz[2] # rz<0, z0 decrease
            if (z1 >= zlower):  # photon advances to z1 and interacts within the layer
                xyz[2] = z1 # update photon location
                xyz[0]+= L * rxyz[0]
                xyz[1]+=L * rxyz[1]
                isAlive,currN,rxyz,xyz,outputs = MonteCarlo_photon_curr(isAlive,currN,rxyz,xyz,outputs,inputs,finitePP)
            else:  # photon moves to lower boundary
                outputs['n_dw'][currN] += 1
                tau = (L - s0) * ke[currN] # remaining tau
                currN -= 1  # move down to the lower layer
                xyz[2]=zlower # advances to the boundary, update photon location
                xyz[0]+=s0 *rxyz[0]
                xyz[1]+= s0 *rxyz[1]
                if (currN < 1): # reach the ground, different surfaces, 07/11/18
                    isAlive,currN,rxyz,xyz,outputs=MonteCarlo_ground(isAlive,currN,rxyz,xyz,outputs,inputs,finitePP)
        # photon travel upwards
        elif (rxyz[2]>0):
            zupper = z_V[currN+1]  # upper bound, 11/27
            s0 = (zupper - xyz[2]) / fabs(rxyz[2])  # remaining distance to upper layer, positive
            z1 = xyz[2] + L * rxyz[2] # new location
            if (z1 <= zupper):  # photon interacts with current layer
                xyz[2]=z1# update photon location
                xyz[0]+=L* rxyz[0]
                xyz[1]+= L * rxyz[1]
                isAlive,currN,rxyz,xyz,outputs = MonteCarlo_photon_curr(isAlive,currN,rxyz,xyz,outputs,inputs,finitePP)
            else:  # photon moves to upper boundary
                outputs['n_uw'][currN+1] += 1
                tau = (L - s0) * ke[currN]  # remaining optical depth to the next layer
                currN += 1  # move up by one layer
                xyz[2]=zupper # advance to boundary, update photon location
                xyz[0]+= s0 * rxyz[0]
                xyz[1]+= s0 * rxyz[1]
                if (currN > N_layer): # escape to outer space, reset parameters
                    outputs['uw_rxyz'].append(rxyz.copy())
                    outputs['uw_xyz'].append(xyz.copy())
                    isAlive = False
        else: # rz==0, interacts in current layer
            isAlive,currN,rxyz,xyz,outputs = MonteCarlo_photon_curr(isAlive,currN,rxyz,xyz,outputs,inputs,finitePP)
    return isAlive,currN,rxyz,xyz, outputs


cpdef MonteCarlo_photon_curr(bint isAlive,int currN,rxyz,xyz,outputs,inputs,finitePP):
    """
    Monte Carlo simulation of a photon interacts with current layer.
  
    Parameters & Returns
    ----------
    Same as function MonteCarlo_photon.
    """
    cdef float g_aer=inputs['g_aer'][currN]
    cdef float f_aer=inputs['f_aer'][currN]
    cdef float g1_aer = inputs['g1_aer'][currN]
    cdef float g2_aer = inputs['g2_aer'][currN]

    cdef float g_c=inputs['g_c'][currN]
    cdef float f_cld = inputs['f_cld'][currN]
    cdef float g1_cld = inputs['g1_cld'][currN]
    cdef float g2_cld = inputs['g2_cld'][currN]
    cdef float f,g1,g2

    cdef float rho_mix=inputs['rho_mix'][currN]
    cdef float sca_gas=inputs['sca_gas'][currN]
    cdef float sca_aer=inputs['sca_aer'][currN]
    cdef float xsi,g
    cdef double[:] cdf, mu

    cdf_cld = inputs['cdf_cld'][currN]
    cdf_aer = inputs['cdf_aer'][currN]
    mu = inputs['mu']
    if (rand()/(RAND_MAX*1.0) > rho_mix): # absorbed
        #print('Photon is absorbed.')
        outputs['n_gas'][currN] += 1
        isAlive = False  # photon is not alive when being absorbed
    else: # scattered
        #print('Photon is scattered.')
        xsi=rand()/(RAND_MAX*1.0)
        if (xsi < sca_gas): # scattered by gas molecules
            g=0.0
            f,g1,g2,cdf = 0.0, 0.0, 0.0, np.zeros(3)
        elif (xsi<sca_aer):
            g= g_aer
            f=f_aer
            g1=g1_aer
            g2=g2_aer
            cdf=cdf_aer
        else: # cloud
            g = g_c
            f = f_cld
            g1=g1_cld
            g2=g2_cld
            cdf = cdf_cld
        rxyz = MonteCarlo_scatter(rxyz,g,f,g1,g2,cdf,mu) # change travel direction
        outputs['N_sca']+=1 # track the number of scattering events
        isAlive,currN,rxyz,xyz,outputs = MonteCarlo_photon(isAlive,currN,rxyz,xyz,outputs,inputs,finitePP)
    return isAlive,currN,rxyz,xyz,outputs


cpdef MonteCarlo_ground(bint isAlive,int currN,rxyz,xyz,outputs,inputs,finitePP):
    """
    Monte Carlo simulation of a photon interacts with ground.
   
    Parameters & Returns
    ----------
    Same as function MonteCarlo_photon.
    """
    cdef float rx=rxyz[0]
    cdef float ry=rxyz[1]
    cdef float rz=rxyz[2]
    cdef float rho_s,rho_s_g,sinT, xsi2,phi
    cdef bint in_pp
    surface=inputs['surface']
    alpha_s=inputs['alpha_s']

    outputs['dw_xyz'].append(xyz.copy())
    outputs['dw_rxyz'].append(rxyz.copy())

    if (surface=='CSP'): # calculate surface albedo of CSP based on rz
        angle_deg=np.array([15.0,45.0,60.0]) # in degree from raw data
        rz_temp=np.cos((180.0-angle_deg)/180.0*math.pi)
        rho_s=1.0-np.interp(rz,rz_temp,alpha_s)
    else: # caluclate surface albedo of non-CSP
        rho_s=(1.0-alpha_s) # pre-computed surface albedo
    rho_s_g=1.0-inputs['alpha_s_g'] # outside power plant field

    in_pp=((xyz[0]*1e-5)**2.0+(xyz[1]*1e-5)**2.0 <= finitePP['R_pp']**2.0)# photon in power plant field, in km
    if ((not in_pp) and finitePP['is_pp']):
        rho_s=rho_s_g
    # absorbed by surface
    if (rand()/(RAND_MAX*1.0)>rho_s): # absorbed by ground
        isAlive=False
        outputs['n_gas'][0]+=1 # ground is layer 0
    else: #scatterd by ground
        outputs['n_uw'][1] += 1
        if (in_pp and surface=='CSP'): # specular reflection
            rxyz=[rx,ry,rz*(-1)] # rz positive, going up
        else: # diffuse reflection
            rz=np.sqrt(rand()/(RAND_MAX*1.0)) # sampling rule from 0 to 1, 07/09
            sinT=np.sqrt(1.-rz*rz)
            xsi2=rand()/(RAND_MAX*1.0) # rz in range 0 to 1, moving up
            phi=2.0*math.pi*xsi2
            rxyz=[sinT*cos(phi),sinT*sin(phi),rz]
        currN+=1 # move up to 1st gas layer
        # outputs['uw_rxyz'].append(rxyz.copy())
        # outputs['uw_xyz'].append(xyz.copy())
    return isAlive,currN,rxyz,xyz,outputs


cpdef MonteCarlo_scatter(rxyz,float g,float f,float g1,float g2, cdf_, mu_):
    """
    Monte Carlo simulation of a scattering event of one photon bundle.

    Parameters
    ----------
    rxyz: list [rx,ry,rz]
        Photon travel direction prior to the scattering event.
    g: float
        scattering asymmetry factor.
    
    Returns
    -------
    rxyz2: list [rx2,ry2,rz2]
        Photon travel direction after the scattering event.

    References:
    [1] "Monte Carlo Methods for Radiation Transport" (2017) by Oleg N. Vassiliev, Page 42-43.
    """
    cdef float xsi,ksi0,mu,phi,sinT,sinP,cosP,xx,rx2,ry2,rz2
    cdef float rx=rxyz[0],ry=rxyz[1],rz=rxyz[2]
    cdef float mu1,mu2
    # compute scattering zenith angle
    if (g == 0.0): # Rayleigh scattering
        xsi=rand()*4.0/(RAND_MAX*1.0)-2.0
        mu=cbrt(xsi+sqrt(xsi*xsi+1.0))+cbrt(xsi-sqrt(xsi*xsi+1.0))
        #mu=(xsi+2.0)/2.0-1.0 # isotropic scattering
    elif(g==1.0): # added on 2/5/2019
        mu=1.0
    elif (g == -2.0):  # user defined phase function
        #print('pdf scattering')
        #print(cdf_)
        xsi = rand()/(RAND_MAX*1.0)
        mu = np.interp(xsi, cdf_, mu_) # speed fine, if for vector it is best.
    else:
        # For aerosols and clouds,
        # 1. Henyey–Greenstein (H–G) scattering phase function
        xsi = rand() / (RAND_MAX * 1.0)  # angle
        mu= 1.0+g*g-((1.0-g*g)/(1.0-g+2.0*g*xsi))**2 # modified on 2/5/2019
        mu/=2.0*g
        # 2. Two-term H–G
        # xsi=rand()/(RAND_MAX*1.0) # angle
        # ksi0=rand()/(RAND_MAX*1.0) # f
        # if (ksi0 < f):
        #     if (g1 == 0.0):
        #         mu=xsi*2.0-1.0
        #     else:
        #         mu = 1.0+g1*g1-((1.0-g1*g1)/(1.0-g1+2.0*g1*xsi))**2
        #         mu = mu / (2.0 * g1)
        # else:
        #     if (g2 == 0.0):
        #         mu=xsi*2.0-1.0
        #     else:
        #         mu = 1.0 + g2 * g2 - ((1.0 - g2 * g2) / (1.0 - g2 + 2.0 * g2 * xsi)) ** 2
        #         mu = mu / (2.0 * g2)
        # 3. isotropic scattering
        # mu=xsi*2.0-1.0
    # compute photon traveling direction change
    Phi=2.0*math.pi*rand()/(RAND_MAX*1.0) # scattering azimuth angle, [0,2pi]
    sinT = sqrt(1.0-mu*mu)#sin(Theta) # in the range [0,1]
    sinP = sin(Phi) # in the range [-1,1]
    cosP = cos(Phi) # in the range [-1,1]
    if (abs(rz)==1.0): # modified 2/4/19
        rx2=sinT*cosP
        ry2=sinT*sinP
        rz2=rz*mu # positive or negative
    else:
        xx=sqrt(1.0 - rz*rz)
        rx2 = rx * mu - sinT / xx * (rx * rz * cosP + ry * sinP)
        ry2 = ry * mu - sinT / xx * (ry * rz * cosP - rx * sinP)
        rz2 = rz * mu + xx * sinT * cosP #  11/27-- method 1, MC book
    return [rx2,ry2,rz2]