import numpy as np
import math
import time
import os
from LBL_funcs_shortwave_aodCases import *

import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    Ph_cdf_cld = False
    Ph_cdf_aer = False
    deltaM = False

    N_layer= 54 # 54 # the number of atmospheric layers
    N_bundles=1000 # the number of photon bundles per wavenumber
    dnu = 3 # spectral resolution 0.1 is enough, 0.01 is too fine, especially for cloudy periods
    nu=np.arange(2500,35000,dnu) # spectral grid on wavenumber
    molecules=['H2O','CO2','O3','N2O','CH4','O2','N2'] # considered atmospheric gases
    #current trace gas surface vmr from http://cdiac.ornl.gov/pns/current_ghg.html, except O3
    vmr0={'H2O':0.03,'CO2':399.5/10**6,'O3':50/10**9,'N2O':328/10**9,
              'CH4':1834/10**9,'O2':2.09/10,'N2':7.81/10}
    model='AFGL midlatitude summer' #profile model, 'AFGL tropical','AFGL midlatitude summer','AFGL midlatitude winter',
    #'AFGL subarctic summer','AFGL subarctic winter','AFGL US standard'
    cld_model = 'default' # cloud model, 'default' or 'caseX'
    period = 'day' # choose 'day' or 'night' for proper temperature profile
    spectral ='SW' # choose 'LW' or 'SW'
    alt = 0 #22.48/1000 # altitude of location, by default is 0 [km]

    ##inputs for desired atmoshperic and surface conditions
    #surface_v=['case2','PV','CSP'] # name of surface
    surface_v=['case2'] # name of surface
    #rh0_v = np.array([15,30,45,60,70])/100
    #rh0_v = np.arange(20,100+20,20,dtype="float")/100.0
    rh0_v = np.arange(10, 100 + 10, 20, dtype="float") / 100.0
    #rh0_v = np.arange(60,120,20)/100  #np.arange(20,120,20)/100 # # surface relative humidity
    #T_surf_v = np.array([294])
    #T_surf_v = np.arange(294.2,295,5) # surface temperature
    T_surf_v = np.arange(-40, 30+5, 5, dtype="float") +273 # KCOD_v = [0]

    #T_surf_v = np.array([302, 304, 306]) # surface temperature in K 294, 296, 298, 300])#
    AOD_v = np.array([0.1243]) # 0.05,0.1,0.2,0.3,0.4,0.6,0.7])
    #np.array([0.05,0.10,0.12,0.14,0.16,0.18,0.20]) #0.05,0.10,0.12,0.14,0.16,0.18,0.20])#
    #0, 0.25, 0.5, 1, 1.5, 2, 4])#np.array([0.1243]) # aerosol optical depth at 479.5 nm
    #COD_v = np.concatenate([[0.5,1],np.arange(2, 32, 2), np.array([35, 40, 45, 50])])  #np.arange(0,20+2,2)
    # yuying cases
    #COD_v = [0] # dl[30,40] hp[(50])
    COD_v = np.array([15,20,30,40,50]) #10 ** np.arange(-1.0, 2.0 + 0.2, 0.2)   # 0.1 to 100
    #10 ** np.arange(-1.0,1.6+ 0.2,0.2) # cloud optical depth at 479.5 nm #np.array([0])#
    kap_v = [[10, 11, 12]]

    ##inputs of angles
    th0_v = np.array([0,15,30,45,60])
    #th0_v = np.array([0,10,15,20,25,30,35,40,45,50,55,60])
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
    #file_dir='results_shortwave/project_data/RH/'#SW_cloudTop/'#COD_SWSCOPE/' ##' # create the directory first
    #file_dir='/mnt/dengnan/AODcases_Ta_rh_t0/'
    file_dir = '/mnt/dengnan/LUTcases/HG/'
    #file_dir='./AODcases_Ta_rh_t0/'
    #file_dir = '/home/dengnan/data/RTM/LUTcases/HG/'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    #file_dir='results_shortwave/sw_scope/'

    # compute case by case
    for iSurf in range(0,len(surface_v)):
        inputs_main={'N_layer':N_layer, 'N_bundles':N_bundles, 'nu':nu, 'molecules':molecules,'vmr0':vmr0,
           'model':model,'cld_model':cld_model,'period':period,'spectral':spectral,'surface':surface_v[iSurf],
                             'alt':alt, 'Ph_cdf_cld':Ph_cdf_cld,'Ph_cdf_aer':Ph_cdf_aer,'deltaM':deltaM
                             }
        for iT in range(0,len(T_surf_v)):
            for iRH in range(0,len(rh0_v)):
                for iAOD in range(0,len(AOD_v)):
                    for iCOD in range(0,len(COD_v)):
                        for iKAP in range(0,len(kap_v)):
                            properties={'rh0':rh0_v[iRH],'T_surf':T_surf_v[iT],'AOD':AOD_v[iAOD],
                                        'COD':COD_v[iCOD],'kap':kap_v[iKAP]}
                            #print(properties)
                            # add cloud top and cloud depth
                            # p, pa = set_pressure(N_layer)
                            # z, za = set_height(model, p, pa)
                            # kap = kap_v[iKAP]
                            # dz_cld=z[kap[-1]+1]-z[kap[0]] # in m
                            # topH_cld=z[kap[-1]+1]
                            # print("ctop: ",topH_cld)
                            for iTH in range(0,len(theta0_v)):
                                angles={'theta0':theta0_v[iTH],'phi0':phi0,'del_angle':del_angle,'beta':beta_v,
                                        'phi':phi_v,'isTilted':isTilted}
                                for idx in range(0,len(dx_v)):
                                    finitePP={'x0':-x0_v[iTH]+dx_v[idx],'y0':-y0_v[iTH],'R_pp':R_pp,'is_pp':is_pp}
                                    fileName1="Results_{}_RH={}_Tsurf={}_AOD={}_COD={}_th0={}".format(
                                            surface_v[iSurf],rh0_v[iRH],T_surf_v[iT],AOD_v[iAOD],COD_v[iCOD],th0_v[iTH])
                                    if not os.path.exists(file_dir+fileName1+'.npy'):
                                        print(f'{fileName1} not exist, run MonteCarlo.')
                                        start_time = time.time()
                                        out1 = LBL_shortwave(properties,inputs_main,angles,finitePP)
                                        # TPW=np.round(TPW,3)
                                        #fileName2="uwxyzr_COD={}_th0={}_TPW={}.npy".format(COD_v[iCOD],th0_v[iTH],TPW)
                                        #fileName3="dwxyzr_COD={}_th0={}_TPW={}.npy".format(COD_v[iCOD],th0_v[iTH],TPW)
                                        np.save(file_dir+fileName1,out1)# save results to local directory
                                        del out1
                                        end_time = time.time()
                                        print ("CPU time:", end_time - start_time)
                                    #exit()
                                    # else:
                                    #     print(f'{fileName1}.npy exist')

