"""
Functions to compute shortwave (SW) irradiance on inclined surfaces.

Author: Mengying Li
"""

import numpy as np
import math

__all__ = [
    "theta_phi",
    "incidence_angle",
    "incidence_angle0",
    "perez4",
    "perez4_Rd",
    "perez3_Rd",
    "MCtransposition",
]

def theta_phi(rx,ry,rz):
    """
    Computes incident angles of (rx,ry,rz) on a horizontal surface.

    Parameters
    ----------
    rx, ry, rz : (N_p,) array_like
        The traveling direction of photons (rx,ry,rz).

    Returns
    -------
    theta: (N_p,) array_like
        Incident zenith angle on a horizontal surface [rad].
    phi : (N_p,) array_like
        Incident azimuth angle on a horizontal surface [rad].

    """
    theta=np.arccos(rz) # in [0,pi]
    sin_th=np.sqrt(1.-rz**2.)
    # Bug fixed: when theta = 0, sin_th = nan. Nancy 2024.7.16
    p = np.random.uniform(low=-np.pi, high=np.pi, size=theta.shape[0])
    cosP = rx / sin_th  # in [0,pi]
    cosP = np.clip(cosP, -1, 1) # bug 2 fixed: cosP may be slightly larger than 1, then phi = nan
    phi = np.arccos(cosP)  # in [0,pi]
    phi[rz==1] = p[rz==1]
    ind=(ry*sin_th<0) # in [pi,2*pi]
    if (ind.size!=0):
        phi[ind]=2*math.pi-phi[ind]
    theta[rx ** 2 + ry ** 2 + rz ** 2 == 0] = np.nan
    phi[rx ** 2 + ry ** 2 + rz ** 2 == 0] = np.nan
    return theta, phi


def incidence_angle(rx,ry,rz, beta, phi):
    """
    Computes incident angles of (rx,ry,rz) on a tilted surface with specified tilt angle (beta) and azimuth angle (phi).

    Parameters
    ----------
    rx, ry, rz : (N_p,) array_like
        The traveling direction of photons (rx,ry,rz).
    beta: float
        tilt angle of the inclined surface [rad].
    phi:  float
        azimuh angle of the inclined surface [rad].
    
    Returns
    -------
    cos_theta_i: (N_p,) array_like
        Cosine of the incident angle on the inclined surface [rad].

    References
    -------
    [1] "Solar Engineering Of Thermal Processes" 4th Edition, by Duffie and Beckman.
    """

    cos_th=rz
    sin_th=np.sqrt(1.-cos_th**2.)
    cosP=rx/sin_th # in [0,pi]
    sinP=np.sqrt(1.-cosP**2.) # positive, in [0,pi]
    ind=(ry/sin_th < 0) # in [pi,2*pi]
    sinP[ind]*=(-1)
    cos_diff=np.cos(phi)*cosP+np.sin(phi)*sinP
    cos_theta_i=cos_th*np.cos(beta)+sin_th*np.sin(beta)*cos_diff
    cos_theta_i[rx**2+ry**2+rz**2==0]=np.nan
    return cos_theta_i


def incidence_angle0(theta0, phi0, beta, phi):
    """
    Computes incident angles of (theta0, phi0) on a tilted surface with specified tilt angle (beta) and azimuth angle (phi).

    Parameters
    ----------
    theta0, phi0 : (N_p,) array_like
        The incident zenith and azimuth angles of photons on horizontal surfaces [rad].
    beta: float
        tilt angle of the inclined surface [rad].
    phi:  float
        azimuh angle of the inclined surface [rad].
    
    Returns
    -------
    theta_i: (N_p,) array_like
        Incident angle on the inclined surface [rad].

    References
    -------
    [1] "Solar Engineering Of Thermal Processes" 4th Edition, by Duffie and Beckman.
    """
    cos_theta_i=np.cos(theta0)*np.cos(beta)+np.sin(theta0)*np.sin(beta)*np.cos(phi0-phi)
    theta_i=np.arccos(cos_theta_i) # in [0,pi]
    return theta_i 


def perez4(ghi, dni, dhi, F_os, rho_s, angles):
    '''
    The transpostion model PEREZ4 proposed by Perez that reported in 2016_Yang.
    Parameters
    ----------
    ghi, dni, dhi: (N_nu,) array_like
        Spectral global horizontal irradiance, direct normal irradiance and diffuse horizontal irradiance [W m-2].
    F_os: (N_nu,) array_like
        Spectral extraterrestrial solar irradiance [W m-2].
    rho_s: (N_nu,) array_like
        Spectral reflectance of foregroud.
    angles: (6,), list 
        theta0, phi0: float
            solar zenith and azimuth angles [rad].
        del_angle: float
            receptance angle of DNI, default is 0.5 degree.
        beta, phi: array_like
            tilt and azimuth angles of the inclined surface [rad].
        isTilted: bool
            indicator of whether to do transpostion computation.
    
    Returns
    -------
    Gc: (N_beta,N_phi,N_nu) array_like
        Spectral plane-of-array (POA) irradiance for specified surface beta and phi [W m-2].
    Ic,Dc,Dg: (N_beta,N_phi,N_nu) array_like
        Direct, sky diffuse and ground reflection components of POA [W m-2].
    Dc_iso,Dc_cs,Dc_hz: (N_beta,N_phi,N_nu) array_like
        Isotropic, circumsolar and horizon components of Dc [W m-2].

    References #!!!! start from here.
    -------
    [1] "Solar radiation on inclined surfaces: Corrections and benchmarks" (2016) by Dazhi Yang.
    '''
    
    theta0=angles['theta0']
    phi0=angles['phi0']
    beta=angles['beta']
    phi=angles['phi']
    #initialize outputs as 3D array
    Gc,Ic,Dc,Dg,Dc_iso,Dc_cs,Dc_hz=[np.zeros((len(beta),len(phi),len(ghi))) for i in range(0,7)]

    for i in range(0,len(beta)):
        for j in range (0,len(phi)):
            #incidence angle
            theta_i=incidence_angle0(theta0, phi0, beta[i], phi[j])
            Ic[i,j,:]=dni*np.cos(theta_i)
            Ic[Ic<0]=0 # added by MLi on 2/13/2019
            Rr=(1-np.cos(beta[i]))/2
            Dg[i,j,:]=rho_s*ghi*Rr
            #Dg[Dg<0]=0 # added by MLi on 2/13/2019
            Rd,temp1,temp2,temp3=perez4_Rd(dni,dhi,F_os,theta0,beta[i],theta_i)
            Dc[i,j,:]=dhi*Rd
            Dc_iso[i, j, :] = dhi * temp1
            Dc_cs[i, j, :] = dhi * temp2
            Dc_hz[i, j, :] = dhi * temp3
            #Dc[Dc<0]=0 # added by MLi on 2/13/2019
    Gc=Ic+Dc+Dg
    return Gc,Ic,Dc,Dg,Dc_iso,Dc_cs,Dc_hz


def perez4_Rd(dni,dhi,F_os,theta0,beta,theta_i):
    '''
    Being called in perez4(...) function, computes the diffuse factor.
    Parameters
    ----------
    dni, dhi, F_os: (N_nu,) array_like
        Spectral direct normal irradiance, diffuse horizontal irradiance and extraterrestrial solar irradiance [W m-2].
    theta0, beta: float
        solar zenith angle, surface tilt angle [rad].
    theta_i: float
        beam incident angle [rad].
    
    Returns
    -------
    Rd: (N_nu,) array_like
        Spectral diffuse factor.
    temp1, temp2, temp3: (N_nu) array_like
        Isotropic, circumsolar and horizon components of Rd.

    References
    -------
    [1] "Solar radiation on inclined surfaces: Corrections and benchmarks" (2016) by Dazhi Yang.
    '''
 
    alpha=math.radians(25.) # be default, alpha is 25 degree
    epsilon=(dhi+dni)/dhi # array-like
    Delta=dhi/(F_os*np.cos(theta0))

    fmat = np.array([[-0.008,0.588,-0.062,-0.060,0.072,-0.022],
            [0.130,0.683,-0.151,-0.019,0.066,-0.029],
            [0.330,0.487,-0.221,0.055,-0.064,-0.026],
            [0.568,0.187,-0.295,0.109,-0.152,-0.014],
            [0.873,-0.392,-0.362,0.226,-0.462,0.001],
            [1.133,-1.237,-0.412,0.288,-0.823,0.056],
            [1.060,-1.600,-0.359,0.264,-1.127,0.131],
            [0.678,-0.327,-0.250,0.156,-1.377,0.251]])
    ebin = np.array([[1, 1.065],
            [1.065, 1.23],
            [1.23, 1.5],
            [1.5, 1.95],
            [1.95, 2.8],
            [2.8, 4.5],
            [4.5, 6.2],
            [6.2, 100000]]) # replace infty by 100
    #assign points to the correct epsilon bin
    Fi=np.zeros((len(dhi),6)) # vectorize
    for i in range(0, len(ebin)):
        ind=(epsilon >= ebin[i,0]) & (epsilon < ebin[i,1])
        Fi[ind,:]=np.tile(fmat[i,:],(np.sum(ind),1)) # vertically stack array
    F1=Fi[:,0]+Delta*Fi[:,1]+theta0*Fi[:,2] # array
    F1[F1<0]=0.
    F2=Fi[:,3]+Delta * Fi[:, 4] + theta0 * Fi[:,5] # array
    #F2[F2<0]=0. # added by MLi on 2/14/19
    #compute a' and c' 
    if (theta0 > math.pi/2. - alpha):
        psih = (math.pi/2. - theta0 + alpha)/(2.*alpha)
    else:
        psih = 1.
    psic = (math.pi/2.-theta_i+alpha)/2./alpha
    if (theta_i < math.pi/2.-alpha):
        chic = psih * np.cos(theta_i)
    elif ((theta_i > math.pi/2. - alpha) and (theta_i < math.pi/2.+alpha)):#2/12/19
        chic = psih*psic*np.sin(psic*alpha)
    else:
        chic = 0
    if (theta0 < math.pi/2. - alpha):
        chih = np.cos(theta0)
    else:
        chih = psih * np.sin(psih * alpha) 
    a = 2 * (1 - np.cos(alpha))*chic
    c = 2 * (1 - np.cos(alpha))*chih    
    #a = np.max(2 * (1 - np.cos(alpha))*chic,0)
    #c = np.max(2 * (1 - np.cos(alpha))*chih,0)
    temp1=(1-F1)*0.5*(1+np.cos(beta))
    temp2=F1*(a/c)
    temp3=F2*np.sin(beta)
    Rd = temp1+temp2+temp3
    return Rd, temp1,temp2,temp3


def perez3_Rd(dni,dhi,F_os,theta0,beta,theta_i):
    '''
    Being called in perez4(...) function, computes the diffuse factor.
    Parameters
    ----------
    dni, dhi, F_os: (N_nu,) array_like
        Spectral direct normal irradiance, diffuse horizontal irradiance and extraterrestrial solar irradiance [W m-2].
    theta0, beta: float
        solar zenith angle, surface tilt angle [rad].
    theta_i: float
        beam incident angle [rad].
    
    Returns
    -------
    Rd: (N_nu,) array_like
        Spectral diffuse factor.

    References
    -------
    [1] "Solar radiation on inclined surfaces: Corrections and benchmarks" (2016) by Dazhi Yang.
    '''
    
    alpha=math.radians(25) # be default, alpha is 25 degree
    epsilon=((dhi+dni)/dhi+1.041*theta0**3)/(1+1.041*theta0)
    Delta=dhi/(F_os*np.cos(theta0))

    fmat = np.array([[-0.008,0.588,-0.062,-0.060,0.072,-0.022],
            [0.130,0.683,-0.151,-0.019,0.066,-0.029],
            [0.330,0.487,-0.221,0.055,-0.064,-0.026],
            [0.568,0.187,-0.295,0.109,-0.152,-0.014],
            [0.873,-0.392,-0.362,0.226,-0.462,0.001],
            [1.133,-1.237,-0.412,0.288,-0.823,0.056],
            [1.060,-1.600,-0.359,0.264,-1.127,0.131],
            [0.678,-0.327,-0.250,0.156,-1.377,0.251]])
    ebin = np.array([[1, 1.065],
            [1.065, 1.23],
            [1.23, 1.5],
            [1.5, 1.95],
            [1.95, 2.8],
            [2.8, 4.5],
            [4.5, 6.2],
            [6.2, 100000]]) # replace infty by 100
    #assign points to the correct epsilon bin
    Fi=np.zeros((len(dhi),6)) # vectorize
    for i in range(0, len(ebin)):
        ind=(epsilon >= ebin[i,0]) & (epsilon < ebin[i,1])
        Fi[ind,:]=np.tile(fmat[i,:],(np.sum(ind),1)) # vertically stack array
    F1=Fi[:,0]+Delta*Fi[:,1]+theta0*Fi[:,2] # array
    F1[F1<0]=0
    F2=Fi[:,3]+Delta * Fi[:, 4] + theta0 * Fi[:,5] # array
    #compute a' and c'  
    a = np.maximum(0,np.cos(theta_i))
    c = np.maximum(np.cos(85/180*math.pi),np.cos(theta0))
    Rd = (1-F1)*0.5*(1+np.cos(beta)) + F1*(a/c) + F2*np.sin(beta)
    return Rd


    #----------------------------------------------------------------------------------------------------------
# transposition model for the Monte Carlo method--all wavenumbers
def MCtransposition(uw_rx,uw_ry,uw_rz,dw_rx,dw_ry,dw_rz, angles, ratio):
    """
    Monte Carlo transpostion of mono-flux on inclined surfaces.
    
    Parameters
    ----------
    uw_rx,uw_ry,uw_rz,dw_rx,dw_ry,dw_rz: (N_bundles,N_lam) array_like
        upwelling/downwelling photon attacking direction (at the horizontal surface).
    angles: (6,), list 
        theta0, phi0: float
            solar zenith and azimuth angles [rad].
        del_angle: float
            receptance angle of DNI, default is 0.5 degree.
        beta, phi: array_like
            tilt and azimuth angles of the inclined surface [rad].
        isTilted: bool
            indicator of whether to do transpostion computation.
    ratio: (N_lam,) array_like 
        energy carried by each photon [W m-2 cm].
    
    Returns
    -------
    F_dni, F_dhi: (N_lam,) array_like 
        direct and diffuse mono-flux at horizontal surfaces [W m-2 cm]. 
    F_gc,F_ic,F_dc,F_dg: (N_beta,N_phi,N_lam) array_like 
        Mono-flux of in-plane global, in-plane direct, in-plane diffuse, ground reflection [W m-2 cm].
    F_gc2,F_dc2,F_dg2: (N_beta,N_phi,N_lam) array_like 
        Mono-flux of in-plane global, in-plane diffuse, ground reflection [W m-2 cm], 
        without correcting for photon incident angles.
    """
    
    theta0=angles['theta0']
    phi0=angles['phi0']
    del_angle=angles['del_angle']
    beta=angles['beta']
    phi=angles['phi']
    isTilted=angles['isTilted']

    # part I, calculating r_ghi, r_dni, r_dhi
    # calculating downwelling
    theta_p=np.arccos(-dw_rz)
    ind_dni=(theta_p>=theta0-del_angle) & (theta_p<=theta0+del_angle)
    ind_dhi=(theta_p<theta0-del_angle) | (theta_p>theta0+del_angle)
    # three components of Dc, added on 6/20/2019
    ind_hz=(theta_p>np.deg2rad(80))*ind_dhi # horizon region
    alpha=np.deg2rad(25) # same as Perez model
    ind_cs=((theta_p<theta0-alpha) | (theta_p>theta0+alpha))*ind_dhi #circumsolar region
    ind_iso=ind_dhi*(~ind_hz)*(~ind_cs) # horizon region
    try:
        F_dni=np.nansum(ind_dni,axis=0)*ratio # (N_lam,1) array
    except:
        F_dni=ratio*0
    try:
        F_dhi=np.nansum(ind_dhi,axis=0)*ratio # (N_lam,1) array
    except:
        F_dhi=ratio*0
    # output results on tilted surfaces if needed
    if (isTilted):
        F_ic,F_dc,F_dg,F_dc2,F_dg2,F_dc_hz,F_dc_iso,F_dc_cs = [np.zeros((len(beta),len(phi),len(ratio))) for i in range(0,8)]
        for i in range(0,len(beta)):
            for j in range (0,len(phi)):
                theta_i=incidence_angle0(theta0,phi0,beta[i],phi[j])
                cos_thetaI_dw=incidence_angle(dw_rx,dw_ry,-dw_rz, beta[i], phi[j]) # speed up by call this func only once
                ind_incident=(cos_thetaI_dw>0)
                if (theta_i<math.pi/2):
                    F_ic[i,j,:]=np.nansum(cos_thetaI_dw*ind_dni>0,axis=0)*ratio*np.cos(theta_i)/np.cos(theta0)
                temp=ind_incident*ind_dhi*cos_thetaI_dw/dw_rz*(-1)
                F_dc[i,j,:]=np.nansum(temp,axis=0)*ratio
                
                cos_thetaI_uw=incidence_angle(uw_rx,uw_ry,-uw_rz, beta[i], phi[j])
                temp=(cos_thetaI_uw>0)*cos_thetaI_uw/uw_rz
                F_dg[i,j,:]=np.nansum(temp,axis=0)*ratio

                # three components of F_dc, sky diffused, added on 06/20/2019
                temp=ind_incident*ind_hz*cos_thetaI_dw/dw_rz*(-1)
                F_dc_hz[i,j,:]=np.nansum(temp,axis=0)*ratio
                
                temp=ind_incident*ind_iso*cos_thetaI_dw/dw_rz*(-1)
                F_dc_iso[i,j,:]=np.nansum(temp,axis=0)*ratio
         
                temp=ind_incident*ind_cs*cos_thetaI_dw/dw_rz*(-1)
                F_dc_cs[i,j,:]=np.nansum(temp,axis=0)*ratio

        F_gc=F_ic+F_dc+F_dg
    try:
        output = {'F_dni':F_dni,'F_dhi': F_dhi, 'F_inclined':[F_gc,F_ic,F_dc,F_dg],
                  'F_dcs':[F_dc_iso,F_dc_cs,F_dc_hz]} # isotropic, circumsolar, horizon region
    except:
        output = {'F_dni':F_dni,'F_dhi': F_dhi, 'F_inclined':[],'F_dcs':[]}
    return output

