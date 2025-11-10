'''This is the LBL model written by Mengying Li.
   Include common functions for both LW and SW. Main program is in the iPython notebook.'''

# import all necessary libraries
import scipy.integrate as integrate
import numpy as np
from multiprocessing import Pool
from scipy.special import expn, jv, yv
import math
import copy
from random import *
from scipy.optimize import fsolve

# # Start of function definitions
# -------------------------------------------------------------------------------
# set pressure of each layer
def setPres(N_layer):
    # LAST MODIFIED: Mengying Li 11/16/2017
    # INPUTS:
    # N_layer: number of atmosphere layers
    # OUTPUT:
    # p: pressure at each node point (Pa)
    # pa：average pressure of each layer (Pa)
    dsig = 1 / N_layer
    n=np.arange(0.5,N_layer+1,0.5)
    sig = (2*N_layer-2*n+1)/2/N_layer 
    pa_temp = sig ** 2 * (3 - 2 * sig) 
    # initialize, has a layer inside the Earth
    p = np.zeros((N_layer + 2)) 
    pa= np.zeros((N_layer + 1)) 
    for i in range(N_layer,0,-1):
        p[i]=pa_temp[2*i-2]
        pa[i]=pa_temp[2*i-1]
    p[0]=p[1] # for Ground layer
    pa[0]=p[0] # for Ground layer
    p = p * 1.013 * 10 ** 5  # change unit to Pa
    pa = pa * 1.013 * 10 ** 5  # change unit to Pa
    return p, pa
# ---------------------------------------------------------------------------------
def setTemp(model, p, pa):
    # LAST MODIFIED: Mengying Li 11/16/2017
    # set temperature of each node point and average of each layer
    # INPUTS:
    # model: profile model, e.g. 'tropical'
    # p/pa: pressure at each node point/average pressure of each layer (Pa)
    # OUTPUT:
    # t/ta：temperauture at each node point/average temperature of each layer (K)
    # data from AFGL Atmospheric Constituent Profiles (1986)
    file = "Profile data/" + model + ".csv"
    data = np.genfromtxt(file, delimiter=',')
    ref_p = data[:, 1]
    ref_t = data[:, 2]
    ref_p = np.asarray(ref_p) * 10 ** 2  # convert unit to Pa
    ref_t = np.asarray(ref_t)
    t = np.interp(-p, -ref_p, ref_t)  # xv needs to be increasing for np.interp to work properly
    ta = np.zeros((len(pa)))
    for i in range(1, len(pa)):
        ta[i] = (t[i] * (p[i] - pa[i]) + t[i+1] * (pa[i] - p[i+1])) /(
            p[i] - p[i+1])  # pressure averaged t, modified by M.Li 11/11/2017
    ta[0] = t[0]
    return t, ta

# --------------------------------------------------------------------------------------
def setHeight(model, p, pa):
    # LAST MODIFIED: Mengying Li 11/16/2017
    # set height of each node point and average height of each layer
    # INPUTS:
    # model: profile model, e.g. 'tropical'
    # p/pa: pressure at each node point/average pressure of each layer (Pa)
    # OUTPUT:
    # z/za：height of each node point/average height of each layer (m)

    file = "Profile data/" + model + ".csv"
    data = np.genfromtxt(file, delimiter=',')
    ref_p = data[:, 1]
    ref_z = data[:, 0]
    ref_p = np.asarray(ref_p) * 10 ** 2  # convert unit to Pa
    ref_z = np.asarray(ref_z) * 10 ** 3  # convert unit to m
    z = np.interp(-p, -ref_p, ref_z)  # xv needs to be increasing for np.interp to work properly
    za = np.zeros((len(pa)))
    for i in range(1, len(pa)):
        za[i] = (z[i] * (p[i] - pa[i]) + z[i+1] * (pa[i] - p[i+1])) / (
            p[i] - p[i+1])  # pressure averaged t, modified by M.Li 11/11/2017
    za[0]=z[0]
    return z, za
# --------------------------------------------------------------------------------------
# for calculating Rayleigh scattering coefficient
def setNdensity(model, p, pa):
    # LAST MODIFIED: Mengying Li 11/16/2017
    # set number density of each node point and average number density of each layer
    # INPUTS:
    # model: profile model, e.g. 'tropical'
    # p/pa: pressure at each node point/average pressure of each layer (Pa)
    # OUTPUT:
    # n/na：number density of each node point/average number dnesity of each layer (unit/cm3)
    file = "Profile data/" + model + ".csv"
    data = np.genfromtxt(file, delimiter=',')
    ref_p = data[:, 1]
    ref_n = data[:, 3] # unit of 1/cm3
    ref_p = np.asarray(ref_p) * 10 ** 2  # convert unit to Pa
    ref_n = np.asarray(ref_n)
    n = np.interp(-p, -ref_p, ref_n)  # xv needs to be increasing for np.interp to work properly
    na = np.zeros((len(pa)))
    for i in range(1, len(pa)):
        na[i] = (n[i] * (p[i] - pa[i]) + n[i+1] * (pa[i] - p[i+1])) / (
            p[i] - p[i+1])  # pressure averaged t, modified by M.Li 11/11/2017
    na[0]=n[0]
    return n, na # in unit of /cm3

# ------------------------------------------------------------------------------------------
# set volumetric mixing ratio of atmosphere gases of each node point and average of each layer
def setvmr(model, molecules, vmr0, z):
    # LAST MODIFIED: Mengying Li 11/16/2017 add N2
    # 05/17/2017 use accumulative weight in each layer
    # INPUTS:
    # model: profile model, e.g. 'tropical'
    # molecules: molecule names
    # vmr0: surface vmr of molecules, used for scaling
    # z: height of each node point (m)
    # OUTPUT:
    # vmr：volumetric mixing ratio of each node point (ppmv) (N_layer+2)
    # densities: average density of each gas in each layer: accumulated weight/volume (g/cm^3) (N_layer+1 including ground)

    vmr = np.zeros((len(molecules), len(z)))
    densities = np.zeros((len(molecules), len(z)-1))
    file = "Profile data/" + model + ".csv"
    data = np.genfromtxt(file, delimiter=',')
    data2 = np.genfromtxt('Profile data/AFGL molecule profiles.csv', delimiter=',')
    ref_z = data[:, 0]  # in unit of km
    ref_z = np.asarray(ref_z) * 10 ** 3  # change unit to m

    for i in range(0, len(molecules)):
        if (molecules[i] == 'H2O'):
            ref_vmr = data[:, 4]
            M = 18.0  # molecular weight, g/mol
        elif (molecules[i] == 'CO2'):
            ref_vmr = data2[:, 2]
            M = 44.0
        elif (molecules[i] == 'O3'):
            ref_vmr = data[:, 5]
            M = 48.0
        elif (molecules[i] == 'N2O'):
            ref_vmr = data[:, 6]
            M = 44.0
        elif (molecules[i] == 'CH4'):
            ref_vmr = data[:, 8]
            M = 16.0
        elif (molecules[i] == 'O2'):
            ref_vmr = data2[:, 7]
            M = 32.0
        elif (molecules[i] == 'N2'):
            ref_vmr=data2[:, 8]
            M = 28.0
        ref_vmr = np.asarray(ref_vmr) / 10 ** 6  # change from ppv to unit 1
        ref_Ni = data[:, 3] * ref_vmr  # in unit of #molecules/cm^3
        NA = 6.022 * 1e23  # in unit #/mol
        ref_rho = ref_Ni / NA * M / ref_vmr[0] * vmr0[i]  # in unit of g/cm^3, after scaling
        vmr[i, :] = np.interp(z, ref_z, ref_vmr) / ref_vmr[0] * vmr0[i]
        for j in range(1, len(z)-1):
            zz = np.linspace(z[j], z[j+1], 100)  # evenly spaced coordinate
            rrho = np.interp(zz, ref_z, ref_rho)
            densities[i, j] = integrate.trapz(rrho, zz) / abs(zz[0] - zz[-1])  # distance averaged rho
        densities[i,0]=densities[i,1]
    vmr = np.transpose(vmr)
    densities = np.transpose(densities)
    return vmr, densities
# ------------------------------------------------------------------------------------------
# set volumetric mixing ratio of atmosphere gases of each node point and average of each layer-- for CIRC cases
def setvmr_CIRC(model,molecules, pa,ta):
    # Modified by: Mengying Li 03/07/2018
    vmr = np.zeros((len(molecules), len(pa)))
    densities = np.zeros((len(molecules),len(pa)))
    file="Profile data/CIRC/"+model+"_input&output/layer_input_"+model+".txt" 
    data=np.genfromtxt(file,skip_header=3)
    data=np.vstack((data[0,:],data))
    for i in range(0, len(molecules)):
        if (molecules[i] == 'H2O'):
            ref_vmr = data[:, 3]
            M = 18.0  # molecular weight, g/mol
        elif (molecules[i] == 'CO2'):
            ref_vmr = data[:, 4]
            M = 44.0
        elif (molecules[i] == 'O3'):
            ref_vmr = data[:, 5]
            M = 48.0
        elif (molecules[i] == 'N2O'):
            ref_vmr = data[:, 6]
            M = 44.0
        elif (molecules[i] == 'CO'): # has CO in CIRC
            ref_vmr = data[:, 7]
            M = 28.0
        elif (molecules[i] == 'CH4'):
            ref_vmr = data[:, 8]
            M = 16.0
        elif (molecules[i] == 'O2'):
            ref_vmr = data[:, 9]
            M = 32.0
        elif (molecules[i] == 'N2'):
            ref_vmr=data[:, 9]/21.0*78.0 # no N2 in CIRC
            M = 28
        for j in range(1, len(pa)):
            Ru=8.314 # J/mol K
            density=pa[j]*ref_vmr[j]*M/Ru/ta[j] # g/m3
            densities[i,j]=density/10**6 # g/cm3
        densities[i,0]=densities[i,1]
        vmr[i,:]=ref_vmr
    densities = np.transpose(densities)
    vmr=np.transpose(vmr)
    return vmr,densities
# ---------------------------------------------------------------------------------------
# calculate the saturation pressure of water vapor using Magus expression: Tsky paper 1 
def psat(t):
    # LAST MODIFIED: Mengying Li 03/17/2017
    psat = 610.94 * np.exp(17.625 * (t - 273.15) / (t - 30.11))
    return psat

def find_dir(spectral):
    if (spectral=='LW'):
        fileTitle='Longwave/'
        fileEnd='_0.01cm-1'
    elif (spectral=='SW'):	
        fileTitle='Shortwave/'
        fileEnd='_3cm-1'
    return fileTitle,fileEnd
#-------------------------------------------------------------------------------------------------
# Get absortion coefficients of gas mixture of all layers
def getMixKappa(inputs,densities,pa,ta,z,za,na,AOD,COD,kap):
# LAST MODIFIED: Mengying Li 03/08/2018, includes aerosols and clouds and accommodates for SW Monte Carlo.
#INPUTS:
#inputs: N_layer,model,molecules,nu
#densities: partial density of each moledule in each layer
#pa: pressure, Pa (vector)
#ta: temperature, K (vector)
#AOD: aerosol optical depth @ 497.5 nm.
#COD: cloud optical depth @ 497.5 nm.
#kap: layers containing clouds. e.g. kap=[5,6,7] clouds are in layers 5,6 and 7.
#OUTPUT: 
#volumetric absorption/scattering coefficients (cm-1) and asymmetry factor.
#WRITTEN BY: Mengying Li 04/21/2017
    N_layer=inputs[0]
    model=inputs[1]
    molecules=inputs[2]
    nu=inputs[3]
    cld_model=inputs[4]
    spectral=inputs[5]

    fileTitle,fileEnd=find_dir(spectral)
    dnu=nu[10]-nu[9]
    fileName=fileTitle+"results/coeffM_"+str(N_layer)+"layers"+"_"+model+"_"+str(dnu)+"cm-1"+".npy"
    coeff_M=np.load(fileName)
    # Add aerosols and clouds
    cldS=np.zeros(N_layer+1)
    aerS=np.zeros(N_layer+1)
    lam=np.concatenate((np.arange(0.1,40.1,0.1),np.arange(40.1,500,10)))
    nu_ref=10**4/lam
    lam0=0.4975
    if (AOD>0):
        aer_ka=np.load(fileTitle+"results/ka_aerosol.npy")
        aer_ks=np.load(fileTitle+"results/ks_aerosol.npy")
        aer_g=np.load(fileTitle+"results/g_aerosol.npy")
        # add new aerosol vertical profile
        aer_vp=np.genfromtxt("Profile data/aerosol profile.csv", delimiter=',')
        aerS=np.interp(za, aer_vp[:,0], aer_vp[:,3],left=0,right=0)*AOD # vertical AOD @ 497.5nm
    if (COD>0):
        cld_ka=np.load(fileTitle+"results/ka_cloud_"+cld_model+".npy")
        cld_ks=np.load(fileTitle+"results/ks_cloud_"+cld_model+".npy")
        cld_g=np.load(fileTitle+"results/g_cloud_"+cld_model+".npy")
        cldS[kap]=COD 
        
    ka_gas_M,ks_gas_M,g_gas_M,ka_aer_M,ks_aer_M,g_aer_M=[np.zeros([N_layer+1,len(nu)]) for i in range(0,6)]
    ka_cld_M,ks_cld_M,g_cld_M,ka_all_M,ks_all_M,g_all_M=[np.zeros([N_layer+1,len(nu)]) for i in range(0,6)]
    for i in range(1,N_layer+1):
        ka_gas,ks_gas,ka_aer,ks_aer,g_aer,ka_cld,ks_cld,g_cld=[np.zeros(len(nu)) for i in range(0,8)]
        RH=0 # defalt no water vapor present
        for j in range(0,len(molecules)):
            ka_gas += coeff_M[i][j] * densities[i, j]
            if densities[i,j]>0: # only if the gas is present
                # add continuum of water vapor
                if (molecules[j]=='H2O'):
                    # check relative humidity does not exceed 100%
                    x_h2o=(densities[i,j])/18.0*8.314*ta[i]/pa[i] # mole fraction
                    x_h2o*=10**6 # unit conversion
                    ps=psat(ta[i]) # saturated pressure
                    RH=pa[i]*x_h2o/ps*100 # ranges from 0 to 100
                    if (RH>100): # if exceeds 100
                        RH=100
                    elif (cldS[i]>0): # cloud present in this layer
                        RH=100 # RH=100% for cloud layers
                    x_h2o=RH/100*ps/pa[i]
                    x_h2o/=10**6
                    densities[i,j]=x_h2o*pa[i]/ta[i]/8.314*18 # change densities according to RH change
                    # #MTCKD continuum
                    ka_cont=absorptionContinuum_MTCKD(nu,pa[i],ta[i],densities[i,j]) # return mass absorption coeff
                    ka_gas+=ka_cont*densities[i,j]# return volume absorption coeff                    
                # add continuum of CO2
                if (molecules[j]=='CO2'):
                    # #MTCKD continuum
                    ka_cont=absorptionContinuum_MTCKD_CO2(nu,pa[i],ta[i],densities[i,j]) # return mass absorption coeff
                    ka_gas+=ka_cont*densities[i,j]# return volume absorption coeff
                    vmr_co2=(densities[i, j]) / 44.0 * 8.314 * ta[i] / pa[i]
                    vmr_co2 *= 10 ** 6  # unit conversion to 1
                # add continuum of O3
                if (molecules[j]=='O3'):
                    # #MTCKD continuum
                    ka_cont=absorptionContinuum_MTCKD_O3(nu,pa[i],ta[i],densities[i,j]) # return mass absorption coeff
                    ka_gas+=ka_cont*densities[i,j]# return volume absorption coeff
                if (molecules[j]=='O2'):
                    # #MTCKD continuum
                    for k in range(0,len(molecules)):
                        if (molecules[k]=='H2O'):
                            x_h2o=(densities[i,k])/18.0*8.314*ta[i]/pa[i] # mole fraction
                            x_h2o*=10**6 # unit conversion
                    ka_cont=absorptionContinuum_MTCKD_O2(nu,pa[i],ta[i],densities[i,j],x_h2o) # return mass absorption coeff
                    ka_gas+=ka_cont*densities[i,j]# return volume absorption coeff
        # Add Rayleigh scattering coefficient
        if (spectral=="SW"):
            ks_gas, temp = rayleigh_kappa_s(nu, ta[i], pa[i], na[i],z[i+1]-z[i], vmr_co2, x_h2o)
            #ks_gas*=1.5
            #ks_gas*=z[i+1]-z[i] # cummulated number of molecules, added on 3/28/2018
        # add aerosols
        if (aerS[i]>0): 
            if (RH==0):
                ka_ref=aer_ka[0,:]
                ks_ref=aer_ks[0,:]
                g_ref=aer_g[0,:]
            else:
                n1=math.floor(RH/10)
                n2=math.ceil(RH/10)
                if n1==n2:
                    ka_ref=aer_ka[n1,:]
                    ks_ref=aer_ks[n1,:]
                    g_ref=aer_g[n1,:]
                else:
                    ka_ref=aer_ka[n1,:]+(aer_ka[n2,:]-aer_ka[n1,:])*(RH/10-n1)/(n2-n1)
                    ks_ref=aer_ks[n1,:]+(aer_ks[n2,:]-aer_ks[n1,:])*(RH/10-n1)/(n2-n1)
                    g_ref=aer_g[n1,:]+(aer_g[n2,:]-aer_g[n1,:])*(RH/10-n1)/(n2-n1)
            #scale aerosol according to desired AOD @ 500nm
            dz = 1575  # scale height,average of 2010_Yu
            kappa_e_ref = aerS[i] / (dz * 100)  # cm-1,desired extincion coeff
            kappa_e=np.interp(lam0,lam,ks_ref+ka_ref)
            ratio=kappa_e_ref/kappa_e
            ka_aer=np.interp(-nu,-nu_ref,ka_ref,left=0,right=0)*ratio # correct using aerosol vertical profile
            ks_aer=np.interp(-nu,-nu_ref,ks_ref,left=0,right=0)*ratio # correct using aerosol vertical profile
            g_aer=np.interp(-nu,-nu_ref,g_ref,left=0,right=0)
        # add clouds
        if (cldS[i]>0): 
            ka_cld=np.interp(-nu,-nu_ref,cld_ka[i,:],left=0,right=0)*cldS[i]
            ks_cld=np.interp(-nu,-nu_ref,cld_ks[i,:],left=0,right=0)*cldS[i]
            g_cld=np.interp(-nu,-nu_ref,cld_g[i,:],left=0,right=0)
        # combine g of aerosol and cloud
        ks_all=ks_gas+ks_aer+ks_cld
        g_mix=g_aer*ks_aer+g_cld*ks_cld
        if (aerS[i]>0 or cldS[i]>0): # avoid dividing by zero
            g_mix/=ks_all
    
        ka_gas_M[i,:]=ka_gas
        ks_gas_M[i,:]=ks_gas
        ka_aer_M[i,:]=ka_aer
        ks_aer_M[i,:]=ks_aer
        g_aer_M[i,:]=g_aer
        ka_cld_M[i,:]=ka_cld
        ks_cld_M[i,:]=ks_cld
        g_cld_M[i,:]=g_cld
        g_all_M[i,:]=g_mix
    ka_all_M=ka_gas_M+ka_aer_M+ka_cld_M
    ks_all_M=ks_gas_M+ks_aer_M+ks_cld_M
    return [ka_gas_M,ks_gas_M,g_gas_M],[ka_aer_M,ks_aer_M,g_aer_M],[ka_cld_M,ks_cld_M,g_cld_M],[ka_all_M,ks_all_M,g_all_M]

# -------------------------------------------------------------------------------------------
def absorptionContinuum_MTCKD(nu, P, T, density):
    # LAST MODIFIED: Mengying Li 05/18/2017 data from contnm.f90 and processed in matlab
    # calculate the continuum absorption coefficients of water vapor
    # nu: wavenumber considered (cm-1)
    # P: total pressure (Pa)
    # T: temperature (K)
    # density: density of water vapor (g/cm3)
    # OUTPUT:
    # coeff_cont：mass continuum absorption coefficients of water vapor (cm^2/g)

    data_frgn = np.genfromtxt('Profile data/frgnContm.csv', delimiter=',')
    coeff_frgn = data_frgn[:, 1]  # in unit cm2/mole (cm)-1 *10^20
    cf_frgn = data_frgn[:, 2]
    data_self = np.genfromtxt('Profile data/selfContm.csv', delimiter=',')
    nu_self = data_self[:, 0]
    coeff_self_296 = data_self[:, 1]  # in unit cm2/mole (cm)-1 *10^20
    coeff_self_260 = data_self[:, 2]
    cf_self = data_self[:, 3]

    # compute R factor
    T0 = 296.0
    P0 = 1.013 * 1e5
    c_h2o = density / 18.0  # molar density of h2o, in unit of mol/cm3
    wk1 = 6.022 * 1e23 * c_h2o * 1e-20
    c_tot = P / 8.314 / T / 10 ** 6  # molar density of air, in unit of mol/cm3
    h2o_fac = c_h2o / c_tot
    #print (h2o_fac)
    RHOave = (P / P0) * (T0 / T)
    R_self = h2o_fac * RHOave  # consider partial pressure
    R_frgn = (1.0 - h2o_fac) * RHOave
    R_frgn_aj=h2o_fac * RHOave

    # compute self continuum coefficients
    tfac = (T - T0) / (260.0 - T0)
    coeff_self = coeff_self_296 * (coeff_self_260 / coeff_self_296) ** tfac  # temeprature correction
    coeff_self *= cf_self * R_self * wk1

    # compute foreign continuum coefficients
    #FH2O=coeff_frgn*cf_frgn
    #c_f=FH2O*wk1
    #cfh2o=1e-20*FH2O
    coeff_frgn *= cf_frgn * R_frgn*wk1
    #coeff_frgn_aj=coeff_frgn/R_frgn*R_frgn_aj

    #coeff_tot=coeff_self-coeff_frgn_aj
    coeff_tot=coeff_self+coeff_frgn

    # sum the two
    #coeff_tot = (coeff_self + coeff_frgn) * 6.022 * 1e3  # unit cm2/mol (cm)-1 * 6.022*10**23*10**(-20)
    #coeff_tot *= c_h2o  # unit of cm2/cm3 (cm)-1
    coeff_tot /= density  # unit of cm2/g (cm)-1
    # interpelate to user defiend grid
    coeff_cont = np.interp(nu, nu_self, coeff_tot,left=0,right=0)
    RADFN=RadiationField(nu, T)
    # mass absorption coeffcient in unit [cm^2/g]
    coeff_cont *= RADFN
    return coeff_cont
# -------------------------------------------------------------------------------------------
def absorptionContinuum_MTCKD_CO2(nu, P, T, density):
    # LAST MODIFIED: Mengying Li 05/18/2017 data from contnm.f90 and processed in Matlab
    # calculate the continuum absorption coefficients of CO2
    # nu: wavenumber considered (cm-1)
    # P: total pressure (Pa)
    # T: temperature (K)
    # density: density of CO2 (g/cm3)
    # OUTPUT:
    # coeff_cont：mass continuum absorption coefficients of CO2 (cm^2/g)
    data_frgn = np.genfromtxt('Profile data/frgnContm_CO2.csv', delimiter=',')
    nu_frgn=data_frgn[:, 0] # wavenumber in cm-1
    coeff_frgn = data_frgn[:, 1]  # in unit cm2/mole (cm)-1 *10^20
    cfac=data_frgn[:,2]
    tdep=data_frgn[:,3]
    trat=T/246.0
    coeff_frgn*=cfac*trat**tdep
    # compute R factor
    T0 = 296.0
    P0 = 1.013 * 1e5
    c_co2 = density / 44  # molar density of h2o, in unit of mol/cm3
    #c_tot = P / 8.314 / T / 10 ** 6  # molar density of air, in unit of mol/cm3
    #co2_fac = c_co2 / c_tot
    RHOave = (P / P0) * (T0 / T)
    coeff_frgn*=RHOave#*co2_fac corrected 4/10/2018

    # unit coversion from MT_CKD to this model
    coeff_frgn *=  6.022 * 1e3  # unit cm2/mol (cm)-1 * 6.022*10**23*10**(-20)
    coeff_frgn *= c_co2  # unit of cm2/cm3 (cm)-1
    coeff_frgn /= density  # unit of cm2/g (cm)-1
    # apply radiation field
    coeff_cont = np.interp(nu, nu_frgn, coeff_frgn,left=0,right=0)
    RADFN=RadiationField(nu, T)
    # mass absorption coeffcient in unit [cm^2/g]
    coeff_cont *= RADFN
    return coeff_cont
# -------------------------------------------------------------------------------------------
def absorptionContinuum_MTCKD_O3(nu, P, T, density):
    # LAST MODIFIED: Mengying Li 04/09/2018 data from contnm.f90 and processed in Matlab
    # calculate the continuum absorption coefficients of O3
    # nu: wavenumber considered (cm-1)
    # P: total pressure (Pa)
    # T: temperature (K)
    # density: density of O3 (g/cm3)
    # OUTPUT:
    # coeff_cont：mass continuum absorption coefficients of O3 (cm^2/g)
    
    c_o3 = density / 48.0  # molar density of O2, in unit of mol/cm3
    wk3=6.022*1e23*c_o3 # MT_CKD code is mole/cm2, here is mole/cm2 *(cm-1)
    DT=T-273.15
    # Band 1: 9170~24665 cm-1
    wo3=wk3*1e-20
    data_contm = np.genfromtxt('Profile data/Contm_O3_b1.csv', delimiter=',')
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
    data_contm = np.genfromtxt('Profile data/Contm_O3_b2.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    c1=data_contm[:,2]
    c2=data_contm[:,3]
    temp=c0*wo3
    temp*=1.0+c1*DT+c2*DT**2
    contm_2=np.interp(nu, nu_contm, temp,left=0,right=0)

    # Band 3: 40800~54000 cm-1
    wo3=wk3
    data_contm = np.genfromtxt('Profile data/Contm_O3_b3.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=c0*wo3
    contm_3=np.interp(nu, nu_contm, temp,left=0,right=0)

    coeff_cont=contm_1+contm_2+contm_3

    # apply radiation field
    RADFN=RadiationField(nu, T)
    # mass absorption coeffcient in unit [cm^2/g]
    coeff_cont *= RADFN/density
    return coeff_cont

# -------------------------------------------------------------------------------------------
def absorptionContinuum_MTCKD_O2(nu, P, T, density, x_h2o):
    # LAST MODIFIED: Mengying Li 07/12/2018 data from contnm.f90 and processed in Matlab
    # calculate the continuum absorption coefficients of O2
    # nu: wavenumber considered (cm-1)
    # P: total pressure (Pa)
    # T: temperature (K)
    # density: density of O2 (g/cm3)
    # OUTPUT:
    # coeff_cont：mass continuum absorption coefficients of O2 (cm^2/g)
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
    #print (wk7)
    #wk7=6.022*1e23*c_o2/density # unit of [mole/cm2]/[g/cm3]

    # Band 1: 7536~8500 cm-1
    a_o2=1.0/0.446
    a_n2=0.3/0.446
    a_h2o=1.0
    tau_fac=wk7/xlosmt*amagat*(a_o2*x_o2+a_n2*x_n2+a_h2o*x_h2o)
    data_contm = np.genfromtxt('Profile data/Contm_O2_b1.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=tau_fac*c0
    contm_1=np.interp(nu, nu_contm, temp,left=0,right=0)

    # Band 2: 9100~11000 cm-1
    wo2=wk7*1e-20*rhoave
    adjwo2=x_o2*(1.0/0.209)*wo2
    data_contm = np.genfromtxt('Profile data/Contm_O2_b2.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=adjwo2*c0
    contm_2=np.interp(nu, nu_contm, temp,left=0,right=0)
    # Band 3: 12990.5 ~ 13223.5 cm-1
    tau_fac=wk7/xlosmt*amagat
    data_contm = np.genfromtxt('Profile data/Contm_O2_b3.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=tau_fac*c0
    contm_3=np.interp(nu, nu_contm, temp,left=0,right=0)
    # Band 4: 15000 ~ 29870 cm-1
    wo2=wk7*1e-20*amagat
    chio2=x_o2
    adjwo2=chio2*wo2 
    data_contm = np.genfromtxt('Profile data/Contm_O2_b4.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=adjwo2*c0
    contm_4=np.interp(nu, nu_contm, temp,left=0,right=0)
    # Band 5: 36000~40000 cm-1
    wo2=wk7*1e-20
    data_contm = np.genfromtxt('Profile data/Contm_O2_b5.csv', delimiter=',')
    nu_contm=data_contm[:, 0] # wavenumber in cm-1
    c0=data_contm[:,1] # continuum coefficient processed in Matlab
    temp=c0*(1.0+0.83*amagat)*wo2
    contm_5=np.interp(nu, nu_contm, temp,left=0,right=0)

    coeff_cont=contm_1+contm_2+contm_3+contm_4+contm_5

    # apply radiation field
    RADFN=RadiationField(nu, T)
    # mass absorption coeffcient in unit [cm^2/g]
    coeff_cont *= RADFN/density
    return coeff_cont

# -------------------------------------------------------------------------------------------
# 'radiation field' in calculation of continuum coefficient
def RadiationField(nu, T):
    # times the 'radiation field' to get rid of (cm)-1 in the denominator
    # see cntnv_progr.f function RADFN(VI,XKT)
    XKT = T / 1.4387752  # 1.4387752 is a constant from phys_consts.f90
    RADFN = np.zeros(len(nu))  # initialize to zero
    for i in range(0, len(nu)):
        XVIOKT = nu[i] / XKT
        if (XVIOKT <= 0.01):
            RADFN[i] = 0.5 * XVIOKT * nu[i]
        elif (XVIOKT <= 10):
            EXPVKT = np.exp(-XVIOKT)
            RADFN[i] = nu[i] * (1.0 - EXPVKT) / (1.0 + EXPVKT)
        else:
            RADFN[i] = nu[i]
    return RADFN

# -------------------------------------------------------------------------------------------
# LBL or k-distribution method of calculating integration of y over x, when y varies a lot with x (e.g. kappa v.s. nu)
def kdIntegrate(x, y, method, partition):
    # LAST MODIFIED: Mengying Li 02/28/2017
    # INPUTS:
    # x: x-axis variable
    # y: a function of x
    # method='LBL' for LBL integration, method ='k-distribution' for k-distribution integration
    # partition: the # of partitions in k-distribution method
    # OUTPUTS:
    # integral: int_{x_min}^{x_max} y dx
    # WRITTEN BY: Mengying Li 02/14/2017
    where_are_NaNs = np.isnan(y)  # replace NaN with zero
    y[where_are_NaNs] = 0
    delta = 10 ** (-19) + (-min(y))  # avoid y goes below or equal to zero
    ka = y + delta
    if (method == 'LBL'):
        integral = integrate.trapz(ka, x)
    elif (method == 'k-distribution'):
        [hist, X] = np.histogram(ka, partition)
        bin_edge = np.delete(X, -1)
        pk = hist / len(ka)  # probability of ka
        gk = np.cumsum(pk)  # accumulated probability of ka
        integral = (max(x) - min(x)) * integrate.trapz(bin_edge, gk)
    integral -= delta * (max(x) - min(x))
    return integral

# -------------------------------------------------------------------------------
# Planck's law in unit of wavenumber
def Planck(nu, T):
    # LAST MODIFIED: Mengying Li 03/01/2017
    # INPUTS:
    # nu: wavenumber, cm-1 (vector or scalar)
    # T: temperature, K (vector or scalar)
    # OUTPUT:
    # Eb: blackbody emission intensity density (vector or scalar), W/(m**2 sr cm-1)
    # WRITTEN BY: Mengying Li 03/01/2017 using Eq on P453 of Mills&Coimbra's BHMT book
    h = 6.6261 * 10 ** (-34)  # Planck's constant, J s
    kB = 1.3806485 * 10 ** (-23)  # Boltzmann constant, J/K
    c = 299792458  # speed of light, m/s
    C1 = 2 * h * c ** 2.0
    C2 = h * c / kB
    nu = nu * 100  # change unit to m-1
    Eb_nu = C1 * nu ** 3.0 / (np.exp(C2 * nu / T) - 1)  # equivalent to MATLAB dot calculations
    Eb_nu *= 100  # in unit of W/m2 cm-1 sr
    return Eb_nu

# -------------------------------------------------------------------------------
def Mie(lam, radii, refrac):
    # LAST MODIFIED: Mengying Li 06/27/2017
    # INPUTS:
    # nu: wavelength, um (scalar)
    # radii: radii of particles, um (scalar)
    # refrac: index of rafraction (complex number)
    # OUTPUT:
    # Qext,Qabs,Qsca: extinction/absorption/scattering efficiency
    # g: assmmetry parameter (optional)
    # WRITTEN BY: Mengying Li 06/20/2017 according to "A first course in atmospheric
    #            radiation" and "mie.py" provided by "Principles of Planetary Climate"
    size = 2 * math.pi * radii /lam  # size parameter
    Nlim = round(size + 4 * size ** (1 / 3) + 2)  # number of terms of infinite series, from Petty's P359
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

# ---------------------------------------------------------------------------------
def Mie_ab(size, refrac, Nlim):
    # LAST MODIFIED: Mengying Li 06/22/2017
    # INPUTS:
    # size: size parameter (scalar)
    # refrac: index of rafraction (complex number)
    # OUTPUT:
    # an,bn: Mie scattering coefficients
    # WRITTEN BY: Mengying Li 06/22/2017 according to "MATLAB Functions for Mie
    #            Scattering and Absorption"
    n_all = np.arange(0, Nlim + 1, 1)  # number of modes (start with zero to count for n-1)
    n = n_all[1:]
    # nu = n+0.5 # to be used in Bessel function
    z = size * refrac
    m2 = refrac * refrac
    sqx = np.sqrt(0.5 * math.pi / size)
    sqz = np.sqrt(0.5 * math.pi / z)

    bx_all = jv(n_all + 0.5, size) * sqx
    bz_all = jv(n_all + 0.5, z) * sqz
    yx_all = yv(n_all + 0.5, size) * sqx
    hx_all = bx_all + yx_all * 1j

    bx = bx_all[1:]
    bz = bz_all[1:]
    #yx = yx_all[1:]
    hx = hx_all[1:]

    b1x = bx_all[0:-1]
    b1z = bz_all[0:-1]
    #y1x = yx_all[0:-1]
    h1x = hx_all[0:-1]

    ax = size * b1x - n * bx
    az = z * b1z - n * bz
    ahx = size * h1x - n * hx

    an = (m2 * bz * ax - bx * az) / (m2 * bz * ahx - hx * az)
    bn = (bz * ax - bx * az) / (bz * ahx - hx * az)
    return an, bn

# ----------------------------------------------------------------------------------
# allow parallel computing
def aerosol_monoLam(inputs):
# Last modified: Mengying Li 03/08/2018
    lam = inputs[0]
    refrac = inputs[1]
    r = inputs[2]
    Qsca = np.zeros(len(r))
    Qabs = np.zeros(len(r))
    g = np.zeros(len(r))
    #for i in range(0, len(lam)):
    for j in range(0, len(r)):
        Qext, Qabs[j], Qsca[j], Qbsca, g[j] = Mie(lam, r[j], refrac)
    return Qsca,Qabs,g

# ----------------------------------------------------------------------------------
# compute absorption/scattering coefficents and single albedo, according to Lubin 2002
def aerosol(spectral):
    lam=np.concatenate((np.arange(0.1,40.1,0.1),np.arange(40.1,500,10)))
    data_aer = np.genfromtxt('Profile data/aerosol refraction.csv', delimiter=',')
    data_w = np.genfromtxt('Profile data/water refraction.csv', delimiter=',')

    real_dry = np.interp(lam, data_aer[:, 0], data_aer[:, 1])
    img_dry = np.interp(lam, data_aer[:, 0], data_aer[:, 2])
    real_w = np.interp(lam, data_w[:, 0], data_w[:, 1])
    img_w = np.interp(lam, data_w[:, 0], data_w[:, 2])

    # -------------modify refraction index and size distribution according to RH-------------
    RH = np.arange(0, 110, 10)  # relative humidity
    rh_fac = np.asarray(
       [1.0, 1.0, 1.0, 1.031, 1.055, 1.090, 1.150, 1.260, 1.554, 1.851, 2.151])  # last number is infered by M.Li
    real_intM = np.zeros([len(RH), len(lam)])
    img_intM = np.zeros([len(RH), len(lam)])
    
    r = np.concatenate((np.arange(0.001,0.3,0.001),np.arange(0.3,20.05,0.1)))  # in um
    rm0 = np.asarray([0.135, 0.955]) 
    sigma0 = np.asarray([2.477, 2.051]) #* rh_fac[i]# in um
    n0 = np.asarray([10 ** 4, 1])  # number/um-3
    Nr = np.zeros([len(RH), len(r)]) # number/um-3
    for i in range(0, len(RH)):
        # refraction index modification
        real_intM[i, :] = real_dry * rh_fac[i] ** (-3) + real_w * (1 - rh_fac[i] ** (-3))
        img_intM[i, :] = img_dry * rh_fac[i] ** (-3) + img_w * (1 - rh_fac[i] ** (-3))
        # size modification
        rm = rm0 * rh_fac[i]
        sigma = sigma0#*rh_fac[i] #* rh_fac[i]# in um
        dNr = n0[0] / (np.sqrt(2 * math.pi) * np.log(sigma[0])) * np.exp(
            -0.5 * (np.log(r / rm[0]) / np.log(sigma[0])) ** 2)
        dNr += n0[1] / (np.sqrt(2 * math.pi) * np.log(sigma[1])) * np.exp(
            -0.5 * (np.log(r / rm[1]) / np.log(sigma[1])) ** 2)
        Nr[i, :] = dNr / r  # same length as r, number/cm-4
    refrac_intM = real_intM + 1j * img_intM 

    # create input list of args to parallel computation
    kappa_s = np.zeros([len(RH), len(lam)])
    kappa_a = np.zeros([len(RH), len(lam)])
    g_all = np.zeros([len(RH), len(lam)])
    for i in range(0, len(RH)):
        #args = [lam, refrac_intM[i, :], r, Nr[i, :]]
        list_args=[]
        for j in range(0,len(lam)):
            args = [lam[j], refrac_intM[i, j], r] # single value of lam,refrac and r
            list_args.append(args)
        pool = Pool(11)  # workers
        results = list(pool.map(aerosol_monoLam, list_args))
        pool.terminate()
        # re-organize the results from parallel computation
        for j in range(0,len(lam)):
            Qsca=results[j][0]
            Qabs=results[j][1]
            g=results[j][2]
            kappa_s[i, j] = integrate.trapz(Qsca * Nr[i,:] * math.pi * r ** 2, r)
            kappa_a[i, j] = integrate.trapz(Qabs * Nr[i,:] * math.pi * r ** 2, r) 
            g_all[i, j] = integrate.trapz(Qsca * g * Nr[i,:] * math.pi * r ** 2, r) / kappa_s[i, j]
	# without infer number of particles-- correct in getMixKappa function
    fileTitle,fileEnd=find_dir(spectral)
    np.save(fileTitle+"results/ks_aerosol", kappa_s)
    np.save(fileTitle+"results/ka_aerosol", kappa_a)
    np.save(fileTitle+"results/g_aerosol", g_all)

# ----------------------------------------------------------------------------------
# compute absorption/scattering coefficents and single albedo, assymetry factor of clouds
def cloud(model,cld_model,z,kap,spectral):
# Last modified: Mengying Li 2/9/2018
    lam=np.concatenate((np.arange(0.1,40.1,0.1),np.arange(40.1,500,10)))
    lam0=0.4975
    data_w = np.genfromtxt('Profile data/water refraction.csv', delimiter=',')
    real_w = np.interp(lam, data_w[:, 0], data_w[:, 1])
    img_w = np.interp(lam, data_w[:, 0], data_w[:, 2])
    refrac_w=real_w+1j*img_w
    r = np.arange(0.1,50,0.1)  # in um
    ks_cld,ka_cld,g_cld=[np.zeros([len(z)-1,len(lam)]) for i in range(0,3)]
    # calculate the Qsca,Qabs,g for all combination of lam, refrac_w and r
    list_args=[]
    for j in range(0,len(lam)):
        args = [lam[j], refrac_w[j], r] # single value of lam,refrac and r
        list_args.append(args)
    pool = Pool(11)  # workers
    results = list(pool.map(aerosol_monoLam, list_args))
    pool.terminate()
    # re-organize the results from parallel computation
    Qsca,Qabs,g_M=[np.zeros([len(lam),len(r)]) for i in range(0,3)]
    for j in range(0,len(lam)):
        Qsca[j,:]=results[j][0]
        Qabs[j,:]=results[j][1]
        g_M[j,:]=results[j][2]
    print ("Finish Mie calculation.")
    # default cloud model (re=10, sig_e=0.1)
    if ('default' in cld_model):#cld_model=='default'):
        re = 10 # effective radius in um, Barker 2003
        sig_e = 0.1 # effective variance in um, Barker 2003
        Nr=r**(1/sig_e-3)*np.exp(-r/re/sig_e) # size distribution (gamma)
        ks,ka,g=[np.zeros(len(lam)) for i in range(0,3)]
        for j in range(0,len(lam)):
            ks[j] = integrate.trapz(Qsca[j,:] * Nr * math.pi * r ** 2, r)
            ka[j] = integrate.trapz(Qabs[j,:] * Nr * math.pi * r ** 2, r) 
            g[j] = integrate.trapz(Qsca[j,:]* g_M[j,:]* Nr * math.pi * r ** 2, r) / ks[j]
        dz_cld=z[kap[-1]+1]-z[kap[0]] # in m
        ke_cld_ref=1.0/(dz_cld*100) # in cm-1, COD by default = 1.0
        ratio_cld=ke_cld_ref/np.interp(lam0,lam,ka+ks)
        for i in range(len(kap)):
            ks_cld[kap[i],:]=ks*ratio_cld
            ka_cld[kap[i],:]=ka*ratio_cld
            g_cld[kap[i],:]=g
    else: # cloud model of CIRC cases
        cld_file="Profile data/CIRC/"+model+"_input&output/cloud_input_"+model+".txt"
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
            x_frac=integrate.trapz(Nr *4/3* math.pi * r ** 3, r) # volume fraction of water in air.
            ratio_cld=(LWP[kap[i]]/(z[kap[i]+1]-z[kap[i]])/100)/x_frac # z should in cm
            ks,ka,g=[np.zeros(len(lam)) for i in range(0,3)]
            for j in range(0,len(lam)):
                ks[j] = integrate.trapz(Qsca[j,:] * Nr * math.pi * r ** 2, r)
                ka[j] = integrate.trapz(Qabs[j,:] * Nr * math.pi * r ** 2, r) 
                g[j] = integrate.trapz(Qsca[j,:]* g_M[j,:]* Nr * math.pi * r ** 2, r) / ks[j]
            ks_cld[kap[i],:]=ks*ratio_cld
            ka_cld[kap[i],:]=ka*ratio_cld
            g_cld[kap[i],:]=g
    # corrected for LWP already
    fileTitle,fileEnd=find_dir(spectral)
    np.save(fileTitle+"results/ks_cloud_"+cld_model, ks_cld)
    np.save(fileTitle+"results/ka_cloud_"+cld_model, ka_cld)
    np.save(fileTitle+"results/g_cloud_"+cld_model, g_cld)

# ----------------------------------------------------------------------------------
# Ciddar's formula of refractive index of air
def rayleigh_kappa_s(nu, T, P, N, dz, vmr_co2, vmr_h2o):
    # LAST MODIFIED: Mengying Li 11/09/2017
    # INPUTS:
    # nu: wavenumber, in cm-1   # T: temperature, in K
    # P: pressure, in Pa       # N: number density of air molecules, in moles/cm3
    # vmr_co2/vmr_h2o: volume mixing ratio of co2 and h2o, unit of 1 e.g.400/10**6
    # constants for water vapor content
    # OUTPUTS:
    # kappa_s: scattering coefficient, cm-1
    # m_r: refractive index of air.
    lam=1e4/nu
    vmr_co2*=10**6 # in unit of ppm
    w0, w1, w2, w3 = 295.235, 2.6422, -0.03238, 0.004028
    # constants for temerature scale and CO2 content
    k0, k1, k2, k3 = 238.0185, 5792105, 57.362, 167917
    # start of calculation
    S = 1 / lam ** 2  # lam in um
    # refractivity of standard air
    n_as = 10 ** (-8) * (k1 / (k0 - S) + k3 / (k2 - S)) + 1
    # refractivity with variable CO2 content
    n_axs = (n_as - 1) * (1 + 5.34 * 10 ** (-7) * (vmr_co2 - 450)) + 1
    # refractivity of water vapor at 20 C and 1333 Pa
    n_ws = 1.022 * 10 ** (-8) * (w0 + w1 * S + w2 * S ** 2 + w3 * S ** 3) + 1
    Ma = 10 ** (-3) * (28.9635 + 12.011 * 10 ** (-6) * (vmr_co2 - 400))  # kg/mol
    Mw = 0.018015  # molar mass of water vapor, kg/mol
    R = 8.314510  # gas constant J/mol K
    rho_a=(1 - vmr_h2o) * P*Ma/(Z_air(T, P, vmr_h2o)*R*T)#*(1-vmr_h2o*(1-Mw/Ma))
    rho_w=vmr_h2o * P*Ma/(Z_air(T, P, vmr_h2o)*R*T)#*(1-vmr_h2o*(1-Mw/Ma))
    rho_axs=101325*Ma/(Z_air(15+273.15, 101325, 0)*R*(15+273.15))
    rho_ws=1333*Ma/(Z_air(20+273.15, 1333, 1)*R*(20+273.15))
    # prop method of calculating the refractive index
    m_r = 1 + (rho_a / rho_axs) * (n_axs - 1) + (rho_w / rho_ws) * (n_ws - 1)

    # find the scattering coefficient with the refractive index
    # according to Penndorf 1995 with some modifications
    #N = N * 1e-12  # number density mole/um3
    ##### method 1 calculating kappa_s
    #Ns=2.54743*10**19*10**(-12) # standard number density
    #kappa_s = 8 * math.pi ** 3 / 3 * (m_r ** 2 - 1) ** 2 / lam ** 4 / N  # in unit of um-1
    #rho_n = 0.035  # depolarization factor
    #kappa_s *= (6 + 3 * rho_n) / (6 - 7 * rho_n)
    ##### method 2 calculating kappa_s
    #kappa_s=32*math.pi**3*(m_r-1)**2/3/lam**4/N # P73 Thomas & Stamnes book
    #kappa_s *= 10 ** 4  # in unit of cm-1
    ###### method 3 calculating kappa_s, from code lblrtm.f
    #conv_cm2mol = 1e-20 / (2.68675 * 1e-1 * 1e5)
    #xnu = nu / 1e4
    #wtot = N  # wtot is the total number of molecules in the layer, mole/cm3
    #print (conv_cm2mol,wtot,wtot*conv_cm2mol)
    #ks_4 = (xnu ** 4 / (9.38076 * 1e2 - 10.8426 * xnu ** 2)) * wtot * conv_cm2mol  # km-1
    #kappa_s=ks_4*(dz*1e2)*1e-5 # convert unit from km-1 to cm-1
    ###### method 4 calculating kappa_s based on 1995_Bucholtz
    ns=5791817.0/(238.0185-(nu/1e4)**2)+167909.0/(57.362-(nu/1e4)**2)
    ns/=1e8
    ns+=1.0 # refraction index of standard air, function of lamda
    Ns=2.54743*1e19 # number density of stantdard air, [mole/cm3]
    data = np.genfromtxt('Profile data/Air Fk.csv', delimiter=',')
    Fk = np.interp(-1e4/nu, -data[:, 0], data[:, 1]) #King correction factor from 1995_Bucholtz
    sigma=24.0*math.pi**3*(ns**2-1)**2 # in [cm^2], cross section
    sigma/=(1.0/nu)**4*Ns**2*(ns**2+2)**2 # lamda in [cm]
    sigma*=Fk
    kappa_s=sigma*N #[cm2]*[mole/cm3]=[cm-1]
    return kappa_s, m_r
#--------------------------------------------------------------------------------------
# calculate density of moist air (Eq.4 of Ciddar paper)
def Z_air(T, P, vmr_h2o):
    # Inputs: temperature T in K, pressure P in Pa
    #        vmr_co2/vmr_h2o: volume fraction of co2 and h2o in unit of 1
    # Outputs: compressibility of air
    # constants to calculate compressibility
    a0, a1, a2 = 1.58123 * 10 ** (-6), -2.9331 * 10 ** (-8), 1.1043 * 10 ** (-10)
    b0, b1 = 5.707 * 10 ** (-6), -2.051 * 10 ** (-8)
    c0, c1 = 1.9898 * 10 ** (-4), -2.376 * 10 ** (-6)
    d = 1.83 * 10 ** (-11)
    e = -0.765 * 10 ** (-8)
    # calculation
    t = T - 273.15
    # molar mass of dry air with consideration of Co2 content
    Z = 1 - (P / T) * (a0 + a1 * t + a2 * t ** 2 + (b0 + b1 * t) * vmr_h2o
                       + (c0 + c1 * t) * vmr_h2o ** 2) + (P / T) ** 2 * (d + e * vmr_h2o ** 2)
    return Z

# -------------------------------------------------------------------------------
# Planck's law in unit of wavelength
def Planck_lam(lam, T):
    # LAST MODIFIED: Mengying 11/11/2017
    # INPUTS:
    # lam: wavelength, um (vector or scalar)
    # T: temperature, K (vector or scalar)
    # OUTPUT:
    # Eb: blackbody emission intensity density (vector or scalar), W/(m**2 sr um)
    # WRITTEN BY: Mengying Li 11/11/2017 using Eq on P453 of Mills&Coimbra's BHMT book
    h = 6.6261 * 10 ** (-34)  # Planck's constant, J s
    kB = 1.3806485 * 10 ** (-23)  # Boltzmann constant, J/K
    c = 299792458  # speed of light, m/s
    C1 = 2 * h * c ** 2
    C2 = h * c / kB
    lam = lam /10**6  # change unit to m
    Eb_lam = C1 /lam**5 / (np.exp(C2 /lam/ T) - 1)
    Eb_lam /= 10**6  # in unit of W/m2 um sr
    return Eb_lam


def surface_albedo(nu, surface):
    """
    Get surface albedo for different materials.
    LAST MODIFIED: Mengying Li 06/28/2018

    Parameters
    ----------
    nu: (N_nu,) array_like
        spectral grid in wavenumber [cm-1].
    surface: string
        considered surface type, CIRC cases or PV or CSP
    Returns
    -------
    rho_s: (N_nu,N_deg) array_like
        spectral surface albedo.
    """
    lam = 1e4 / nu
    if ('case' in surface):
        file = "results/CIRC/" + surface + "_input&output/sfcalbedo_input_" + surface + ".txt"
        data = np.genfromtxt(file, skip_header=6)
        rho_s = np.interp(nu, data[:, 0], data[:, 1])
    if (surface == 'PV'):
        file = "Profile data/Reflectance of PV.txt"
        data = np.genfromtxt(file, skip_header=0)
        rho_s = np.interp(lam, data[:, 0] / 1e3, data[:, 1] / 1e2)  # data in nm and %
    if (surface == 'CSP'):
        file = "Profile data/Reflectance of CSP.txt"
        data = np.genfromtxt(file, skip_header=1)
        rho_s1 = np.interp(lam, data[:, 0] / 1e3, data[:, 1])  # data in nm and %
        rho_s2 = np.interp(lam, data[:, 0] / 1e3, data[:, 2])  # data in nm and %
        rho_s3 = np.interp(lam, data[:, 0] / 1e3, data[:, 3])  # data in nm and %
        rho_s = np.concatenate((np.vstack(rho_s1), np.vstack(rho_s2), np.vstack(rho_s3)), axis=1)
    return rho_s


# get aerosol properties for CIRC cases
def aerosol_CIRC(model, nu, z):
    file = "results/CIRC/" + model + "_input&output/aerosol_input_" + model + ".txt"
    A = np.genfromtxt(file, skip_header=3, max_rows=1)
    data = np.genfromtxt(file, skip_header=5)
    N_layer = data.shape[0] - 1
    ka_aer_M, ks_aer_M, g_aer_M = [np.zeros([N_layer + 1, len(nu)]) for i in range(0, 3)]
    lam = 1e4 / nu
    for i in range(1, N_layer + 1):
        dz = (z[i + 1] - z[i]) * 100  # in cm
        if (data[i, 1] > 0):
            tau = data[i, 1] * lam ** (-A)
            ke_aer = tau / dz  # extinction coeff
            ks_aer_M[i, :] = ke_aer * data[i, 2]  # SSA is not a function of lam
            ka_aer_M[i, :] = ke_aer - ks_aer_M[i, :]
            g_aer_M[i, :] = data[i, 3]  # g is not a function of lam
    return [ka_aer_M, ks_aer_M, g_aer_M]