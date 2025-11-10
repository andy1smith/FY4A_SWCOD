import numpy as np
import math
import time
from multiprocessing import Pool

from LBL_funcs_fullSpectrum import *
from fun_nearealtime_RTM import goes_calinu

import warnings
warnings.filterwarnings('ignore')


N_layer= 54 # the number of atmospheric layers
N_bundles=1000 # the number of photon bundles per wavenumber
dnu = 3 # spectral resolution 0.1 is enough, 0.01 is too fine, especially for cloudy periods
nu=np.arange(2500,35000,dnu) # spectral grid on wavenumber

channels = ['C02']#['C{:02d}'.format(c) for c in range(1, 6 + 1)]
nu = goes_calinu(nu, channels, file_dir='./GOES_data/', dnu = 3)

N_layer = 54 # the number of atmospheric layers
N_bundles = 1000 # the number of photon bundles per wavenumber
molecules=["H2O",'CO2','O3','N2O','CH4','O2','N2'] # considered atmospheric gases
#current trace gas surface vmr from http://cdiac.ornl.gov/pns/current_ghg.html, except O3
vmr0={'H2O':0.03,'CO2':399.5/10**6,'O3':50/10**9,'N2O':328/10**9,
          'CH4':1834/10**9,'O2':2.09/10,'N2':7.81/10}  # surface volumetric mixing ratio
model='AFGL midlatitude summer' #profile model, 'AFGL tropical','AFGL midlatitude summer','AFGL midlatitude winter',
#'AFGL subarctic summer','AFGL subarctic winter','AFGL US standard'
cld_model = 'default' # cloud model, 'default' or 'caseX'
period = 'day' # choose 'day' or 'night' for proper temperature profile
spectral='SW' # choose 'LW' or 'SW'
alt = 0 # altitude of location, by default is 0

##inputs for desired atmoshperic and surface conditions
#surface_v=['case2','PV','CSP'] # name of surface
surface_v=['case2'] # name of surface
rh0_v=np.arange(70,75,5)/100 # surface relative humidity
T_surf_v=np.arange(294.2,295,5) # surface temperature
AOD_v=np.array([0.05]) # aerosol optical depth at 479.5 nm
COD_v=np.array([1.0]) # cloud optical depth at 479.5 nm
#CODs=np.array([0,0.1,0.3,0.5,0.7,1.0,3.0,5.0,10.0,50.0]) # cloud optical depth at 479.5 nm
kap_v=[[8,9,10]]
cld_xy = 20 * 1e5 ## km->cm
#kap_v=[[10],[8,9,10],[6,7,8,9,10],[4,5,6,7,8,9,10],
#      [22],[19,20,21,22],[16,17,18,19,20,21,22],[13,14,15,16,17,18,19,20,21,22],
#      [10,11,12,13,14,15,16,17,18,19,20,21,22],[7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
#      [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]] # cloud residancy layer

##inputs of angles
#th0_v = np.array([0,15,30,45,60,75])
th0_v = np.array([30])
theta0_v = th0_v / 180 * math.pi  # solar zenith angle in rad
phi0 = 0 / 180 * math.pi  #solar azimuth angle in rad
del_angle= 0.5/180*math.pi # DNI acceptance angle, in rad, default is 0.5 degree
beta_v=np.array([0,15,30,45,60,75,90])/180*math.pi # surface tilt angles in rad
phi_v=phi0+np.array([0,45,90,135,180])/180*math.pi # surface azimuth angles in rad
isTilted=True # whether to compute transposition on inclined surfaces

##inputs of finite power plant computation
x0_v=120.0*np.tan(theta0_v)*np.cos(phi0) # photon starting x location, in km
y0_v=120.0*np.tan(theta0_v)*np.sin(phi0) # photon starting y location, in km
R_pp=1 # radius of power plant in km
is_pp=True # whether to consider power plant of finite size
#dx_v=np.arange(-5.0,5.2,0.2)# displacement of input photon location
dx_v=np.array([0.0])# displacement of input photon location
## folder directory to store the results
file_dir='results_shortwave/tests/' # crea

# get spectral absorption coefficients of gases of all layers and save to local directory
# takes a couple minutes to run, require parallel computing
# temperature dependence of kappa_a is not strong, so this cell only needed to run once for different T_surf
# need to re-run if N_layer, model, period, molecules or nu changed
from LBL_funcs_getHitran import *
p,pa=set_pressure(N_layer) 
# use a representative surface temperature of 290 K for optical properties
t,ta=set_temperature(model,p,pa, 294, period) 
inputs_kappa={'N_layer':N_layer,'model':model,'molecules':molecules,'nu':nu,
              'pa':pa,'ta':ta,'spectral':spectral}
getKappa_AllLayers(inputs_kappa)