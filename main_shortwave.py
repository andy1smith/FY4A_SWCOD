import numpy as np
import math
import time
from multiprocessing import Pool

from LBL_funcs_fullSpectrum import *
from LBL_funcs_shortwave import *

import warnings
warnings.filterwarnings('ignore')

#================ User defined Variables ==============
## general inputs
N_layer= 54 # the number of atmospheric layers
N_bundles=10000 # the number of photon bundles per wavenumber
dnu = 3# spectral resolution 0.1 is enough, 0.01 is too fine, especially for cloudy periods
nu=np.arange(2500,35000,dnu) # spectral grid on wavenumber
molecules=['H2O','CO2','O3','N2O','CH4','O2','N2'] # considered atmospheric gases
#current trace gas surface vmr from http://cdiac.ornl.gov/pns/current_ghg.html, except O3
vmr0={'H2O':0.03,'CO2':399.5/10**6,'O3':50/10**9,'N2O':328/10**9,
          'CH4':1834/10**9,'O2':2.09/10,'N2':7.81/10}
model='AFGL midlatitude summer' #profile model, 'AFGL tropical','AFGL midlatitude summer','AFGL midlatitude winter',
#'AFGL subarctic summer','AFGL subarctic winter','AFGL US standard'
cld_model = 'default' # cloud model, 'default' or 'caseX'
period = 'day' # choose 'day' or 'night' for proper temperature profile
spectral='SW' # choose 'LW' or 'SW'
alt = 0 # altitude of location, by default is 0

##inputs for desired atmoshperic and surface conditions
#surface_v=['case2','PV','CSP'] # name of surface
surface_v=['case2'] # name of surface
#[0.999996361794019, 1.000108723300104, 1.0003925027330018, 0.9996338661641606, 1.0009310269137879]
#rh0_v=np.array([0.9464999999999949, 0.7184999999999925, 0.5504999999999944, 0.42500000000000004, 0.3315])
rh0_v=np.array([70])/100 # surface relative humidity
T_surf_v=np.array([295])#np.arange(294.2,295,5) # surface temperature
AOD_v=np.array([0.0]) # aerosol optical depth at 479.5 nm
COD_v=np.array([0,10]) # cloud optical depth at 479.5 nm
#CODs=np.array([0,0.1,0.3,0.5,0.7,1.0,3.0,5.0,10.0,50.0]) # cloud optical depth at 479.5 nm
kap_v=[[8,9,10]]
#kap_v=[[10],[8,9,10],[6,7,8,9,10],[4,5,6,7,8,9,10],
#      [22],[19,20,21,22],[16,17,18,19,20,21,22],[13,14,15,16,17,18,19,20,21,22],
#      [10,11,12,13,14,15,16,17,18,19,20,21,22],[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
#      [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]] # cloud residancy layer

##inputs of angles
#th0_v = np.array([45,60,75])
np.concatenate(np.arange(22,32,2),np.array([35,40,45,50]))
th0_v = np.array([0])
theta0_v = th0_v / 180 * math.pi  # solar zenith angle in rad
phi0 = 0 / 180 * math.pi  #solar azimuth angle in rad
del_angle= 0.5/180*math.pi # DNI acceptance angle, in rad, default is 0.5 degree
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
file_dir='results_shortwave/sw_scope/' # create the directory first


# compute case by case
for iSurf in range(0,len(surface_v)):
    inputs_main={'N_layer':N_layer, 'N_bundles':N_bundles, 'nu':nu, 'molecules':molecules,'vmr0':vmr0,
       'model':model,'cld_model':cld_model,'period':period,'spectral':spectral,'surface':surface_v[iSurf], 
                 'alt':alt}
    for iRH in range(0,len(rh0_v)):
        for iT in range(0,len(T_surf_v)):
            for iAOD in range(0,len(AOD_v)):
                for iCOD in range(0,len(COD_v)):
                    for iKAP in range(0,len(kap_v)):
                        properties={'rh0':rh0_v[iRH],'T_surf':T_surf_v[iT],'AOD':AOD_v[iAOD],
                                    'COD':COD_v[iCOD],'kap':kap_v[iKAP]}
                        for iTH in range(0,len(theta0_v)):
                            angles={'theta0':theta0_v[iTH],'phi0':phi0,'del_angle':del_angle,'beta':beta_v,
                                    'phi':phi_v,'isTilted':isTilted}
                            for idx in range(0,len(dx_v)):
                                finitePP={'x0':-x0_v[iTH]+dx_v[idx],'y0':-y0_v[iTH],'R_pp':R_pp,'is_pp':is_pp}                          
                                print ("Start MonteCarlo once.")
                                start_time = time.time()
                                out1, out2, out3 = LBL_shortwave(properties,inputs_main,angles,finitePP)
                                # out3 = LBL_shortwave(properties,inputs_main,angles,finitePP)
                                # rho.append(out)
                                # Den.append(densities)
                                #out = LBL_shortwave(properties,inputs_main,angles,finitePP)
                                fileName1="Results_{}_RH={}_Tsurf={}_AOD={}_COD={}_kap={}_th0={}".format(
                                    surface_v[iSurf],rh0_v[iRH],T_surf_v[iT],AOD_v[iAOD],COD_v[iCOD],kap_v[iKAP],th0_v[iTH])
                                # fileName3="Coeff_{}_RH={}_Tsurf={}_AOD={}_COD={}".format(
                                #     surface_v[iSurf],rh0_v[iRH],T_surf_v[iT],AOD_v[iAOD],COD_v[iCOD])
                                fileName2="uwxyzr_COD={}_th0={}.npy".format(COD_v[iCOD],th0_v[iTH])
                                fileName3="dwxyzr_COD={}_th0={}.npy".format(COD_v[iCOD],th0_v[iTH])
                                np.save(file_dir+fileName1,out1)# save results to local directory
                                del out1
                                np.save(file_dir+fileName2,out2)# save results to local directory
                                del out2
                                np.save(file_dir+fileName3,out3)
                                del out3
                                
                                #print ("End MonteCarlo once.")
                                end_time = time.time()
                                print ("CPU time:", end_time - start_time)
                                
print('Finish')