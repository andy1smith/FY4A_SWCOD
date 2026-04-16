import pandas as pd
import numpy as np
import os
import calendar

def get_solar_zenith(doy, latitude, ndoy=365):
    """
    Calculate solar zenith angle (1) at local
    noon for input day of the year and latitude.
    """
    term1 = (doy + 10) * (360.0 / ndoy)
    declination = np.cos(np.radians(term1)) * -23.45
    # Zenith = 90 - Altitude
    # Altitude = 90 - Latitude + Declination
    # Therefore: Zenith = 90 - (90 - Latitude + Declination) = Latitude - Declination
    zenith = latitude - declination
    return zenith


def black(par1, par2, par3, sza_deg):
    """
    Calculate Black-Sky Albedo (Directional-Hemispherical Reflectance).
    """

    # 1. Convert SZA to Radians (Crucial for the polynomial to work)
    # The coefficients are derived for theta in radians.
    sza_rad = np.deg2rad(sza_deg)

    # 2. Pre-calculate powers of SZA
    sza2 = sza_rad ** 2
    sza3 = sza_rad ** 3

    # Apply MODIS scale factor
    f_iso = par1 #* 0.001
    f_vol = par2 #* 0.001
    f_geo = par3 #* 0.001

    # 3. Define the polynomial coefficients (Schaaf et al. 2002)
    # Format: (constant, term for x^2, term for x^3)
    iso_c = (1.000000, 0.000000, 0.000000)
    vol_c = (-0.007574, -0.070987, 0.307588)
    geo_c = (-1.284909, -0.166314, 0.041840)

    # 4. Calculate the Kernel Integrals (g_iso, g_vol, g_geo)
    # g_iso is just 1.0
    g_iso = iso_c[0] + (iso_c[1] * sza2) + (iso_c[2] * sza3)
    g_vol = vol_c[0] + (vol_c[1] * sza2) + (vol_c[2] * sza3)
    g_geo = geo_c[0] + (geo_c[1] * sza2) + (geo_c[2] * sza3)

    # 5. Calculate Albedo
    # before or after np.trapz(albedo*srf_lam, lam)/np.trapz(srf_lam, lam), is same.
    albedo = (f_iso * g_iso + f_vol * g_vol + par3 * f_geo)

    return np.clip(albedo,0,1)


def white(par1, par2, par3):
    """
    Calculate White-Sky Albedo (Bi-hemispherical Reflectance)
    Using standard MODIS kernel integrals.
    """
    # 1. Define White-Sky Kernel Integrals (Constants)
    # These are fixed integrals of the kernels over the hemisphere.
    g_iso_wsa = 1.0
    g_vol_wsa = 0.189184
    g_geo_wsa = -1.377622

    # 2. Calculate Albedo
    # Apply the 0.001 scale factor here
    wsa = (par1 * g_iso_wsa + par2 * g_vol_wsa + par3 * g_geo_wsa)  # * 0.001

    return wsa


def blue(wsa, D, bsa):
    """
    Calculate Blue-Sky Albedo.
    """
    albedo = wsa * (1 - D) + bsa * D
    return albedo




def ross_thick(theta_i, theta_v, rel_phi):
    """
    Computes the RossThick volumetric scattering kernel.

    Parameters:
    theta_i : Illumination zenith angle [radians]
    theta_v : Viewing zenith angle [radians]
    rel_phi : Relative azimuth angle [radians]

    Reference: MCD43 ATBD, Eq. 37
    """
    # Calculate Phase Angle (xi)
    cos_xi = (np.cos(theta_i) * np.cos(theta_v) +
              np.sin(theta_i) * np.sin(theta_v) * np.cos(rel_phi))  # eq(43)

    # Clip to avoid numerical errors slightly outside [-1, 1]
    cos_xi = np.clip(cos_xi, -1.0, 1.0)
    xi = np.arccos(cos_xi)

    # Kernel Formula
    k_vol = ((np.pi / 2 - xi) * cos_xi + np.sin(xi)) / \
            (np.cos(theta_i) + np.cos(theta_v)) - (np.pi / 4)

    return k_vol


def li_sparse(theta_i, theta_v, rel_phi):
    """
    LiSparse-Reciprocal Geometric Kernel (Standard MODIS)
    Constants: h/b = 2, b/r = 1

    reference: MCD43_ATBD equation (39-44)
    """
    h_b = 2.0  # Crown relative height
    b_r = 1.0  # Crown shape factor

    # tan_ti = np.tan(b_r * theta_i)
    # tan_tv = np.tan(b_r * theta_v)
    tan_ti = b_r * np.tan(theta_i)  # Correct
    tan_tv = b_r * np.tan(theta_v)  # Correct

    ti = np.arctan(tan_ti)
    tv = np.arctan(tan_tv)

    # Secants (1/cos)
    sec_ti = 1.0 / np.cos(ti)
    sec_tv = 1.0 / np.cos(tv)

    # 3. Calculate Phase Angle cos(xi) again
    cos_xi = np.cos(ti) * np.cos(tv) + np.sin(ti) * np.sin(tv) * np.cos(rel_phi)
    cos_xi = np.clip(cos_xi, -1.0, 1.0)  # Safety clip

    # D_sq = tan^2(ti) + tan^2(tv) - 2*tan(ti)*tan(tv)*cos(phi)
    D_sq = tan_ti ** 2 + tan_tv ** 2 - 2 * tan_ti * tan_tv * np.cos(rel_phi)
    D_sq = np.maximum(D_sq, 0)  # Ensure non-negative
    D = np.sqrt(D_sq)

    # 5. Calculate Overlap Term (t)
    # Cost_t = h/b * sqrt(D^2 + (tan_ti*tan_tv*sin_phi)^2) / (sec_ti + sec_tv)
    # This is the tricky part often implemented wrong!

    temp = (tan_ti * tan_tv * np.sin(rel_phi)) ** 2
    cost_t = h_b * np.sqrt(D_sq + temp) / (sec_ti + sec_tv)
    cost_t = np.clip(cost_t, -1.0, 1.0)  # Clip is crucial here
    t = np.arccos(cost_t)

    # Overlap O
    O = (1 / np.pi) * (t - np.sin(t) * cost_t) * (sec_ti + sec_tv)

    # 6. Final Kernel: LiSparse-Reciprocal
    # K = O - sec_ti - sec_tv + 0.5 * (1 + cos_xi) * sec_tv
    k_geo = O - sec_ti - sec_tv + 0.5 * (1 + cos_xi) * sec_ti * sec_tv

    return k_geo


def build_brdf_pdf(theta_i_rad, par1, par2, par3):
    """
    Builds the PDF for sampling reflected photons.
    theta_i_rad: Incident Zenith (Radians)
    """

    # Grid Setup (Degrees for range, convert to Rads)
    d_th = 5
    d_phi = 5
    # Note: Theta view goes 0 to 90
    bins_theta_deg = np.arange(d_th / 2, 90, d_th)
    bins_phi_deg = np.arange(-180 + d_phi / 2, 180, d_phi)

    theta_bin_rad = np.deg2rad(bins_theta_deg)
    phi_bin_rad = np.deg2rad(bins_phi_deg)  # This acts as Relative Azimuth

    pdf = np.zeros((len(theta_bin_rad), len(phi_bin_rad)))

    # Vectorized calculation (Much faster than loops)
    # Create 2D meshgrids for vector operations
    TV, PH = np.meshgrid(theta_bin_rad, phi_bin_rad, indexing='ij')

    # Calculate Kernels
    k_vol = ross_thick(theta_i_rad, TV, PH)
    k_geo = li_sparse(theta_i_rad, TV, PH)

    # Reconstruct BRDF
    brdf = par1 + (par2 * k_vol) + (par3 * k_geo)

    # Physics check: BRDF cannot be negative (physically),
    # though the model can produce negatives mathematically.
    brdf = np.maximum(brdf, 0.0)

    # Calculate PDF Weight
    # Weight = BRDF * cos(theta_v) * sin(theta_v)
    # cos(theta_v): Lambert's Law (Projected area)
    # sin(theta_v): Spherical Jacobian (Solid angle factor)
    pdf = brdf * np.cos(TV) * np.sin(TV) * np.deg2rad(d_th) * np.deg2rad(d_phi)

    # Normalize to create a valid Probability Mass Function (PMF)
    # Note: For Monte Carlo, we usually treat this as a Discrete PDF
    pdf_sum = pdf.sum()

    if pdf_sum > 0:
        pdf /= pdf_sum
    else:
        # Fallback for total absorption or error (Uniform distribution)
        pdf[:] = 1.0 / pdf.size

    return pdf

def sample_from_pdf(pdf):
    cdf = np.cumsum(pdf.ravel())
    u = np.random.rand()
    idx = np.searchsorted(cdf, u)

    d_th = 2
    d_phi = 5
    # Note: Theta view goes 0 to 90
    bins_theta_deg = np.arange(d_th / 2, 90, d_th)
    bins_phi_deg = np.arange(-180 + d_phi / 2, 180, d_phi)

    theta_bin_rad = np.deg2rad(bins_theta_deg)
    phi_bin_rad = np.deg2rad(bins_phi_deg)  # This acts as Relative Azimuth

    i, j = np.unravel_index(idx, pdf.shape)
    return theta_bin_rad[i], phi_bin_rad[j]


from scipy.special import rel_entr


def calculate_metrics(h_model, h_lambert):
    """
    h_model: Normalized 2D histogram of your Monte Carlo BRDF
    h_lambert: Normalized 2D histogram of Ideal Lambertian
    """
    epsilon = 1e-10  # Small number to prevent division by zero
    # 1. RMSE (Global Deviation)
    # We flatten the 2D arrays to 1D for easy calculation
    rmse = np.sqrt(np.mean((h_model.flatten() - h_lambert.flatten()) ** 2))

    # 1. Robust ANIX (Using 98th and 2nd percentiles)
    # Filter for bins that actually received photons in the model to reduce noise
    valid_mask = h_lambert > epsilon
    ratio_map = np.divide(h_model, h_lambert, out=np.ones_like(h_model), where=valid_mask)
    # Filter for bins that actually received photons in the model to reduce noise
    valid_ratios = ratio_map[h_model > epsilon]

    if len(valid_ratios) > 0:
        # Use percentiles on the RATIO, not the raw counts
        p98 = np.percentile(valid_ratios, 98)
        p02 = np.percentile(valid_ratios, 2)
        robust_anix = p98 / max(p02, epsilon)
    else:
        robust_anix = 1.0

    # 3. KL Divergence (Information Difference)
    # We add a tiny epsilon to avoid log(0) errors
    # It handles zeros and infinities correctly automatically
    P = h_model.flatten() + epsilon
    Q = h_lambert.flatten() + epsilon

    # Normalize again just to be safe (KL requires sum=1)
    P /= np.sum(P)
    Q /= np.sum(Q)

    kl_elementwise = rel_entr(P, Q)
    kl_div = np.sum(kl_elementwise)

    return rmse, robust_anix, kl_div

