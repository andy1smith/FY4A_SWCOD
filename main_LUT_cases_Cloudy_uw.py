import numpy as np
import math
import time
import os
from LBL_funcs_shortwave import *
import socket
from fun_nearealtime_RTM import *

import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    meth = 'HG'
    hostname = socket.gethostname()
    if hostname == 'user-Super-Server': # Replace with actual hostname
        file_dir = f"/home/dengnan/data/RTM/LUTcases/{meth}/fy4a_channels/"
    elif hostname == 'user-MS-7D30':
        file_dir = f"/mnt/dengnan/LUTcases/{meth}/fy4a_channels/"
    elif hostname == 'h07mgt1': 
        file_dir = f"/puhome/22117689r/projects/FY4A_SWCOD/LUTcases/{meth}/fy4a_channels/"
    elif hostname == 'dengnans-MacBook-Pro.local':
        file_dir = "/Users/dengnan/Documents/git_store/FY4A_SWCOD/RTM/LUTcases/HG/fy4a_channels/"
    else:
        # Fallback or Error
        file_dir = f"/puhome/22117689r/projects/FY4A_SWCOD/LUTcases/{meth}/fy4a_channels/"
        raise ValueError(f"Unknown server: {hostname}. Please set fdir manually.")
    # if not exit, create it
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    Ph_cdf_cld = False
    Ph_cdf_aer = False
    deltaM = False # HG
    if meth == 'dM':
        deltaM = True
    

    N_layer= 54 # 54 # the number of atmospheric layers
    N_bundles=1000 # the number of photon bundles per wavenumber
    bandmode = 'FY4A'
    dnu = 3 # spectral resolution 0.1 is enough, 0.01 is too fine, especially for cloudy periods
    nu=np.arange(2500,35000,dnu) # spectral grid on wavenumber
    if bandmode == 'FY4A':
        channels=['C01','C02','C03','C05','C06']
        nu = FY4A_calinu(nu, channels, './FY4A_data/', dnu=3)

    molecules=['H2O','CO2','O3','N2O','CH4','O2','N2'] # considered atmospheric gases
    #current trace gas surface vmr from http://cdiac.ornl.gov/pns/current_ghg.html, except O3
    vmr0={'H2O':0.03,'CO2':399.5/10**6,'O3':50/10**9,'N2O':328/10**9,
              'CH4':1834/10**9,'O2':2.09/10,'N2':7.81/10}
    model='AFGL midlatitude summer' #profile model, 'AFGL tropical','AFGL midlatitude summer','AFGL midlatitude winter',
    #'AFGL subarctic summer','AFGL subarctic winter','AFGL US standard'
    cld_model = 'default' # COD-controlled cloud model, 'default' or CIRC 'caseX'
    period = 'day' # choose 'day' or 'night' for proper temperature profile
    spectral ='SW' # choose 'LW' or 'SW'
    alt = 0 #22.48/1000 # altitude of location, by default is 0 [km]

    # C01,C02,C05,C06 need to analysis the already have
    df_albedo = []
    ##inputs for desired atmoshperic and surface conditions
    #surface_v=['case2','PV','CSP'] # name of surface
    # Define a mapping
    SURFACE_TYPES = {'Lambert': 0, 'CSP': 1, 'BRDF': 2, 'MODIS': 3, 'Case2': 0}
    white_albedo = [0, 0, 0, 0, 0]
    black_albedo = [0, 0, 0, 0, 0]
    BRDF_param = [0] * 15
    surface = 'MODIS'
    if surface == 'MODIS':# or surface == 'Case2':
        surface_id = SURFACE_TYPES.get(surface, 0)
    surface_v=[surface] # name of surface
    surface_id_v=[surface_id]

    # rh0_v = np.array([0, 20, 40, 60, 80, 100]) / 100.0 # gpt sugg
    # T_surf_v = np.array([270, 275, 280, 285, 290, 295, 300, 305, 315, 320]) # gpt sugg
    #rh0_v = np.arange(10, 100 + 10, 10, dtype="float") / 100.0
    #T_deltas = np.arange(-60, 30 + 5, 5, dtype="float")
    #T_surf_v = T_deltas + 294.2
    rh0_v = np.array([10]) #, 50, 90]) / 100.0
    T_surf_v = np.array([270, 285, 300, 320])


    AOD_v =  np.array([0.1243])
    COD_v = [0.5, 1, 3, 5, 10, 20, 50]
    kap_v = [[10, 11, 12]] 

    # 5 Typical Albedo Sets (C01, C02, C03, C05, C06) from PCA
    albedo_sets = [
        [0.0145, 0.0288, 0.4156, 0.2031, 0.0641], # 05%
        [0.0251, 0.0472, 0.3922, 0.2218, 0.0897], # 25%
        [0.0407, 0.0745, 0.3575, 0.2494, 0.1275], # 50%
        [0.0673, 0.1207, 0.2988, 0.2961, 0.1916], # 75%
        [0.0938, 0.1669, 0.2401, 0.3428, 0.2555]  # 95%
    ]

    ##inputs of angles
    th0_v = np.array([0,15,30,45,60,65]) 
    # th0_v = np.array([0, 20, 40, 60, 75])gpt 
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

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    #file_dir='results_shortwave/sw_scope/'

    # compute case by case
    for iSurf in range(0,len(surface_v)):
        for iALB, alb_set in enumerate(albedo_sets):
            # if iALB in [0,1,2]:
            #     continue
            if surface == 'MODIS':
                # Assuming White-Sky Albedo = Black-Sky Albedo for Lambertian simplification in LUT
                black_albedo = alb_set
                white_albedo = alb_set

            inputs_main={'N_layer':N_layer, 'N_bundles':N_bundles, 'nu':nu, 'molecules':molecules,'vmr0':vmr0,
               'model':model,'cld_model':cld_model,'period':period,'spectral':spectral,'surface_id':surface_id_v[iSurf],
                         'white_albedo':white_albedo, 'black_albedo':black_albedo,'BRDF_param':BRDF_param,
                         'alt':alt, 'Ph_cdf_cld':Ph_cdf_cld,'Ph_cdf_aer':Ph_cdf_aer,'deltaM':deltaM,
                         'escape_alpha':0.0, 'escape_cone_deg':-1.0,
                         'escape_probability_mode':'none', 'scale_deltaM_g':False
                         }
            for iT in range(0,len(T_surf_v)):
                for iRH in range(0,len(rh0_v)):
                    for iAOD in range(0,len(AOD_v)):
                        for iKAP in range(0, len(kap_v)):
                            for iCOD in range(0,len(COD_v)):
                                properties={'rh0':rh0_v[iRH],'T_surf':T_surf_v[iT],'AOD':AOD_v[iAOD],
                                            'COD':COD_v[iCOD],'kap':kap_v[iKAP]}
                                # print(properties)
                                # print the cloud height, base height, top height, thickness of cld.
                                for iTH in range(0,len(theta0_v)):
                                    angles={'theta0':theta0_v[iTH],'phi0':phi0,'del_angle':del_angle,'beta':beta_v,
                                            'phi':phi_v,'isTilted':isTilted}
                                    for idx in range(0,len(dx_v)):
                                        finitePP={'x0':-x0_v[iTH]+dx_v[idx],'y0':-y0_v[iTH],'R_pp':R_pp,'is_pp':is_pp,
                                                  'th0':theta0_v[iTH], 'phi0':phi0, 'del_angle':del_angle}
                                        if N_bundles == 1000:
                                            fileName1="Results_{}_AlbSet{}_AOD={:.2f}_COD={}_kap={}_th0={}_Ts={}_RH={}".format(
                                                surface_v[iSurf],iALB,AOD_v[iAOD],COD_v[iCOD],kap_v[iKAP],th0_v[iTH], T_surf_v[iT], int(rh0_v[iRH]*100))
                                            output_path = os.path.join(file_dir, fileName1 + '.npy')
                                            if os.path.exists(output_path):
                                                print(f'{output_path} exists, continue.')
                                                continue
                                        print ("Start MonteCarlo once.")
                                        start_time = time.time()
                                        out1,out2 = LBL_shortwave(properties,inputs_main,angles,finitePP)
                                        end_time = time.time()
                                        print ("CPU time:", end_time - start_time)
                                        #del out1, out3
                                        if N_bundles == 1000:
                                            np.save(os.path.join(file_dir, fileName1), out1)# save results to local directory
                                        
                                        del out1, out2
