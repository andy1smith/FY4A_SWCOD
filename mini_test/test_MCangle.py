import numpy as np
import math
import time
from multiprocessing import Pool

from LBL_funcs_fullSpectrum import *
from LBL_funcs_shortwave import *
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib import colors
import numpy as np
import os,re
import math
# import plotting packages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D # 3d plot
from matplotlib import cm #color map
from matplotlib.ticker import FormatStrFormatter # set decimals in ticks
from matplotlib import rc,re
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

from LBL_funcs_fullSpectrum import *
from LBL_funcs_plotting import *
from LBL_funcs_inclined import *

import warnings
warnings.filterwarnings('ignore')

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
    phi = np.array(phi)
    # Adjust phi to be in the range (-pi, pi]
    phi[phi >= np.pi] -= 2 * np.pi
    phi[phi < -np.pi] += 2*np.pi
    
    return theta, phi

fig = plt.figure(figsize=(6,4))
gs1 = gridspec.GridSpec(1, 1) 
gs1.update(wspace=0.2, hspace=0.4)
ax1 = fig.add_subplot(gs1[0])

for i in range(8000):
    rxyz = np.array([0,0,-1])
    rx,ry,rz=MonteCarlo_scatter(rxyz, 0.5)
    if rz<=0:
        continue
    theta,phi=cartesian_to_spherical(rx, ry, rz)
    del rx,ry,rz
    ax1.scatter(90-np.rad2deg(theta),np.rad2deg(phi),color='C0',s=0.5)
ax1.set_xlim(0,90)
fig.savefig('test.png', dpi=300, bbox_inches='tight')