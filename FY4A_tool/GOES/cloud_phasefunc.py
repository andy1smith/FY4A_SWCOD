
import numpy as np
import math
import random
import importlib.resources
import numpy as np
import matplotlib.pyplot as plt
import miepython as mie
from miepython import monte_carlo

def test_mie():
    """    Test Mie scattering for a sphere with a refractive index of 1.5 - 1j and a diameter of 100 nm.
    """

    m = 1.5 - 1j  # refractive index of sphere
    d = 100  # nm diameter of sphere
    lambda0 = 314.15  # nm wavelength in vacuum

    qext, qsca, qback, g = mie.efficiencies(m, d, lambda0)

    print("The extinction efficiency  is %.3f" % qext)
    print("The scattering efficiency  is %.3f" % qsca)
    print("The backscatter efficiency is %.3f" % qback)
    print("The scattering anisotropy  is %.3f" % g)
    return g

def cloud(nu):
    """
    Compute the asymetry factor of top of cloud layer.

    Parameters
    ----------
    model: str
        Profile model, for CIRC cases only.
    cld_model: str
        Cloud model, by default re=10 um and sig_e = 0.1.

    Returns
    -------
    ks_cld, ka_cld, g_cld : array (n_layer,n_nu)
        Cloud scattering asymmetry
        factors (g) for Top of cloud layer and all considered wavenumbers.

    """
    lam=np.concatenate((np.arange(0.1,40.1,0.1),np.arange(40.1,500,10)))
    lam0=0.4975
    nu_ref = 1e4 / lam
    r = np.arange(0.1,50,0.1)  # in um
    Qsca = np.load("../data/computed/Qsca_clouds.npy")
    g_M = np.load("../data/computed/gM_clouds.npy")

    ks_cld, g_cld = [np.zeros([0, len(lam)]) for i in range(0, 2)]
    # default cloud model (re=10, sig_e=0.1)
    #cld_model=='default'):
    re = 10 # *****effective radius in um, Barker 2003
    #re=5.4 #*****for continental clouds, Miles 2000
    sig_e = 0.1 # effective variance in um, Barker 2003
    Nr=r**(1/sig_e-3)*np.exp(-r/re/sig_e) # size distribution (gamma)
    ks,g=[np.zeros(len(lam)) for i in range(0,2)]
    for j in range(0,len(lam)):
        ks[j] = np.trapz(Qsca[j,:] * Nr * math.pi * r ** 2, r)
        g[j] = np.trapz(Qsca[j,:]* g_M[j,:]* Nr * math.pi * r ** 2, r) / ks[j]
    g_cld = np.interp(-nu, -nu_ref, g, left=0, right=0)
    return np.mean(g_cld)

def scattering(rxyz, g):
    # g < 0.0: backscattering
    # g = 0.0: isotropic scattering
    rx = rxyz[0]
    ry = rxyz[1]
    rz = rxyz[2]
    # For aerosols and clouds, the Henyey–Greenstein (H–G) scattering phase function
    xsi= random.random() #rand()/(RAND_MAX*1.0)
    mu= 1.0+g*g-((1.0-g*g)/(1.0-g+2.0*g*xsi))**2 # modified on 2/5/2019
    mu/=2.0*g
    #mu=xsi*2.0-1.0 # isotropic scattering
    # compute photon traveling direction change
    #Phi=2.0*math.pi*rand()/(RAND_MAX*1.0) # scattering azimuth angle, [0,2pi]
    Phi = 2.0 * math.pi * random.random()
    np.sinT = np.sqrt(1.0-mu*mu)#np.sin(Theta) # in the range [0,1]
    np.sinP = np.sin(Phi) # in the range [-1,1]
    np.cosP = np.cos(Phi) # in the range [-1,1]
    if (abs(rz)==1.0): # modified 2/4/19
        rx2=np.sinT*np.cosP
        ry2=np.sinT*np.sinP
        rz2=rz*mu # positive or negative
    else:
        xx=np.sqrt(1.0 - rz*rz)
        rx2 = rx * mu - np.sinT / xx * (rx * rz * np.cosP + ry * np.sinP)
        ry2 = ry * mu - np.sinT / xx * (ry * rz * np.cosP - rx * np.sinP)
        rz2 = rz * mu + xx * np.sinT * np.cosP #  11/27-- method 1, MC book
    return rx2, ry2, rz2


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
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

    # Calculate theta and phi
    theta = np.arccos(rz / r)
    phi = np.arctan2(ry, rx)

    # Adjust phi to be in the range (-pi, pi]
    try:
        phi[phi >= np.pi] -= 2 * np.pi
        phi[phi < -np.pi] += 2 * np.pi
    except TypeError:
        if phi >= np.pi:
            phi -= 2 * np.pi
        elif phi < -np.pi:
            phi += 2 * np.pi

    return theta, phi

def Mie_backscattering(nu, theta0, phi0):
    """
    Compute the backscattering phase function for Mie scattering.

    Parameters:
        theta0 (float): Angle from the positive z-axis (0 to pi/2)
        phi0 (float): Azimuthal angle in the xy-plane from the x-axis (0 to 2pi)

    Returns:
        rx0, ry0, rz0 : float
            Direction conp.sines of the scattering direction
    """
    # Convert spherical coordinates to Cartesian coordinates
    rx0 = np.sin(theta0) * np.cos(phi0)
    ry0= np.sin(theta0) * np.sin(phi0)
    rz0 = -np.cos(theta0) # modified 2/4/19
    rxyz0 = [rx0, ry0, rz0]
    g  = cloud(nu)
    # HG tend to underestimate the backscattering phase function for large particles.
    rx2, ry2, rz2 = scattering(rxyz0,g)
    theta, phi = cartesian_to_spherical(rx2, ry2, rz2)
    return theta, phi

def cdf_miepython(nu, rxyz):
    rx = rxyz[0]
    ry = rxyz[1]
    rz = rxyz[2]

    # Solor TOA and surface albedo
    data = np.genfromtxt('../data/profiles/SolarTOA.csv', delimiter=',')
    ref_lam = data[:, 0]  # in unit of um
    ref_E = data[:, 1]  # in unit of W/m2 um
    ref_E_nu = -ref_E * ref_lam ** 2 / 1e4
    F_dw_os = -np.interp(-nu, -1e4 / ref_lam, ref_E_nu)  # in wavenumber basis

    lam = np.concatenate((np.arange(0.1, 40.1, 0.1), np.arange(40.1, 500, 10)))
    nu_ref = 1e4 / lam
    lam0 = 0.4975
    data_w = np.genfromtxt('../data/profiles/water_refraction.csv', delimiter=',')
    real_w = np.interp(lam, data_w[:, 0], data_w[:, 1])
    img_w = np.interp(lam, data_w[:, 0], data_w[:, 2])
    # refrac_w = real_w + 1j * img_w
    rw = np.interp(-nu, -nu_ref, real_w, left=0, right=0)
    iw = np.interp(-nu, -nu_ref, img_w, left=0, right=0)

    m = np.mean(rw) + 1j * np.mean(iw)
    x = 10  # np.linspace(0.1, 50, 446)  # in um
    # num = 1000
    mu = np.linspace(-1, 0, 1000)
    pdf = mie.i_unpolarized(m, x, mu)
    # mu, cdf_values = monte_carlo.mu_with_uniform_cdf(m, x, num)

    # Calculate phase function density
    # mu = np.linspace(-1, 0, 1000)  # cos(theta) from backward (-1) to forward (1)
    # pdf = np.zeros((len(mu), len(nu)))
    # for i in range(len(nu)):
    #     m = rw[i] + 1j * iw[i]
    #     pdf[:,i] = mie.i_unpolarized(m, x, mu)

    # Compute total PDF
    # Multiply the PDF by the respective incident light for each wavenumber
    # weighted_pdf = pdf * F_dw_os[np.newaxis, :] / np.trapz(F_dw_os,nu) # Shape becomes (1000, 10834) after broadcasting
    #
    # # Integrate over the spectral range
    # Total_pdf = np.trapz(weighted_pdf, x=nu, axis=1)  # Integrate along the nu dimension
    # np.savez('theory_pdf_z=0.npz', mu=mu, theory_pdf=Total_pdf)
    plt.figure(figsize=(10, 6))
    plt.plot(mu, pdf, label='PDF', color='blue')
    #plt.plot(mu, Total_pdf, label='PDF', color='blue')
    plt.xlabel('mu')
    #plt.ylabel('CDF')
    plt.ylabel('PDF')
    plt.title('Visible, Zenith = 0, Azimuth = 0')
    plt.grid()
    plt.legend()
    # # plt.savefig('cdf_mu.png')
    plt.show()

    Phi = 2.0 * math.pi * random.random()
    np.sinT = np.sqrt(1.0 - mu * mu)  # np.sin(Theta) # in the range [0,1]
    np.sinP = np.sin(Phi)  # in the range [-1,1]
    np.cosP = np.cos(Phi)  # in the range [-1,1]
    if (abs(rz) == 1.0):  # modified 2/4/19
        rx2 = np.sinT * np.cosP
        ry2 = np.sinT * np.sinP
        rz2 = rz * mu  # positive or negative
    else:
        xx = np.sqrt(1.0 - rz * rz)
        rx2 = rx * mu - np.sinT / xx * (rx * rz * np.cosP + ry * np.sinP)
        ry2 = ry * mu - np.sinT / xx * (ry * rz * np.cosP - rx * np.sinP)
        rz2 = rz * mu + xx * np.sinT * np.cosP  # 11/27-- method 1, MC book
    theta, phi = cartesian_to_spherical(rx2, ry2, rz2)
    return theta, phi, cdf_values


if __name__ == '__main__':
    #mie_g = test_mie()

    theta0 = 30 * np.pi / 180  # Convert degrees to radians
    phi0 = 0
    lamb = np.arange(0.4,0.7+0.1,0.1) # 0.7 um
    nu = 1e4 / lamb[::-1]
    #dnu = 3  # spectral resolution 0.1 is enough, 0.01 is too fine, especially for cloudy periods
    #nu = np.arange(2500, 35000, dnu)
    # method 1
    #theta, phi=Mie_backscattering(nu,0.5, 0.5)

    # method 2
    rx0 = np.sin(theta0) * np.cos(phi0)
    ry0 = np.sin(theta0) * np.sin(phi0)
    rz0 = -np.cos(theta0)  # modified 2/4/19
    rxyz0 = [rx0, ry0, rz0]
    theta, phi, cdf_values = cdf_miepython(nu, rxyz0)

    # Convert theta and phi from radians to degree
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)
    # Plot the 2D figure with x=theta, y=phi, z=cdf_values
    # plt.figure(figsize=(10, 6))
    # plt.scatter(theta, phi, c=cdf_values, cmap='viridis')
    # plt.colorbar(label='CDF Values')
    # plt.xlabel('Theta (degrees)')
    # plt.ylabel('Phi (degrees)')
    # plt.title('2D Plot of Theta, Phi, and CDF Values')
    # plt.grid()
    # plt.show()

    print('theta',np.rad2deg(theta), 'phi', np.rad2deg(phi))