import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skyfield.api import load, Topos
import pvlib

from LBL_funcs_inclined import *
from LBL_funcs_fullSpectrum import *


def calculate_relative_azimuth_angle(point_azimuth, sun_azimuth, input_type='deg'):
    """
    Calculate the relative azimuth angle between a point and the sun.

    Parameters:
    -----------
    point_azimuth : np.array
        Azimuth angle of the point
    sun_azimuth : float
        Azimuth angle of the sun
    input_type : str, optional
        Type of input angles.
        'deg' (default): inputs are in degrees
        'rad': inputs are in radians

    Returns:
    --------
    relative_azimuth : np.array
        Relative azimuth angle between the point and sun
        Range: 0 to 180 degrees (or equivalent radians)
    """
    # Convert inputs to degrees if they are in radians
    if input_type == 'rad':
        point_azimuth = np.rad2deg(point_azimuth)
        sun_azimuth = np.rad2deg(sun_azimuth)

    # Calculate the absolute difference between point and sun azimuth
    relative_azimuth = np.abs(point_azimuth - sun_azimuth)

    # Ensure the result is between 0 and 180 degrees
    relative_azimuth = np.minimum(relative_azimuth, 360 - relative_azimuth)

    # Convert back to radians if input was in radians
    if input_type == 'rad':
        relative_azimuth = np.deg2rad(relative_azimuth)

    return relative_azimuth

def extract_ref_correct(rela_azi, ref_oswr_model, index, time, ele, site, vector=True):
    # Convert the 'time' column to datetime
    time = pd.to_datetime(time, format='%Y/%m/%d %H:%M')
    # Convert to pandas Series to access dt accessor
    time_series = pd.Series(time)
    # Extract day, month, and year
    day = time_series.dt.day
    month = time_series.dt.month
    year = time_series.dt.year

    filename = './lut_test_file/'+'site_lonlat.csv'
    data = pd.read_csv(filename, index_col=0).loc[site]
    lat, lon = data['latitude'], data['longitude']


    j = find_bin_indices(0, rela_azi,'azimuth')
    # dataset = f['theta_phi_loc']
    if vector:  # extract local zenith vector
        model = ref_oswr_model[int(index)]
        element = model[j]  # Example: extract the element at row 10, column 10
    else: # extract realative azimuth bin, a value
        element = ref_oswr_model[j]
    #print(f"Extracted element: {element}")
    d = sun_earth_dis(day, month, year, lat, lon, ele)
    ref = earth_sun_dist_norm(element, d)
    return ref

def sun_earth_dis(day,month,year,latitude,longitude, elevation):
    #load = Loader('skyfield-data')  # folder where DE files are downloaded, avoiding repetitive downloads.
    ts = load.timescale()
    planets = load('./data/other/de421.bsp')  # Loading the planetary ephemeris data

    # Define observer's location on Earth
    observer = Topos(latitude, longitude, elevation)
    # Choose a specific time
    # Define the date of interest
    t = ts.utc(year, month, day)

    # Use Earth as a reference point for basic celestial observations
    earth = planets['earth']
    location = earth + observer
    astrometric = location.at(t[0]).observe(planets['sun'])
    apparent = astrometric.apparent()

    # Use helper functions for RA and DEC
    ra, dec, distance = apparent.radec()

    #print("Sun's Right Ascension:", ra)
    #print("Sun's Declination:", dec)
    #print("Distance to the Sun (in AU):", distance.au)
    return distance.au


def calculate_tpw(T_surf, rh0, period='day'):
    """
    Clausius-cla method of Total Precipitable Water (TPW),
    the profiles is consistent with RTM model atmosphere.

    Parameters:
    T_surf (float): Surface temperature in K
    rh0 (float): Relative humidity [0-1]

    Returns:
    float: Total Precipitable Water in kg/m²
    """
    # SW RTM model atmosphere profile
    molecules = ['H2O']
    vmr0 = {'H2O': 0.03}
    model = 'AFGL midlatitude summer'
    N_layer = 54

    p, pa = set_pressure(N_layer)
    z, za = set_height(model, p, pa)
    t, ta = set_temperature(model, p, pa, T_surf, period)
    ps = saturation_pressure(t)
    if vmr0['H2O'] != 0:
        vmr0['H2O'] = rh0 * ps[1] / p[1]  # for water vapor, dependent on local humidity

    vmr, densities = set_vmr(model, molecules, vmr0, z)
    TPW = total_precipitable_water(densities[:, 0], pa, ta, p[1:])
    # next: cut down parameters, set calculation, and load from np.
    return TPW


def interpolate_with_neighbors(H, theta_idx, phi_idx):
    i, j = int(theta_idx), int(phi_idx)

    # Extract the neighborhood indices
    neighborhood_indices = [
        (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
        (i, j - 1), (i,j), (i, j + 1),
        (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)
    ]
    # Accumulate the weighted contributions from neighboring points
    total_weight = 0
    total_value = 0

    for ni, nj in neighborhood_indices:
        # Check if indices are within bounds
        if (0 <= ni < H.shape[0]) and (0 <= nj < H.shape[1]):
            # Calculate the distance and determine the weight (inverse distance weighting)
            distance = np.sqrt((theta_idx - ni) ** 2 + (phi_idx - nj) ** 2)
            weight = 1.0 / (distance + 1e-5)  # +1e-5 to avoid division by zero

            # Accumulate weighted values
            total_value += H[ni, nj] * weight
            total_weight += weight

    # Calculate the weighted average
    if total_weight > 0:
        interpolated_value = total_value / total_weight
    else:
        # Fallback if something goes wrong, should not happen with valid inputs
        interpolated_value = H[i, j]

    return interpolated_value


def cal_mono_Intensity(rxyz_M, theta0, nu, F_dw_os, local_zen, rela_azi, N_bundles=1000,
                       is_flux=False, Norm=False, dirc='UW', bin_scale=1):  # Z_csky
    """
    bins_theta: local zenith angle
    bin_phi: relative difference between the angle of solar azimuth and local zimuth
    solid anlge (bin_theta,bin_phi) determined to the intensity at an angle of (satellite local solar).
    rxyz_M : the vector of each photon fall on the ground. Each band saved many photons, 
            we sum the number of photon, multiple with ratio to get  the flux.
    F_dw_os: the downwelling flux at TOA on each nu
    """
    theta0 = theta0 / 180 * math.pi
    phi0 = 0 / 180 * math.pi

    # bin_scale = 1
    d_th = 2 * bin_scale
    d_phi = 5 * bin_scale
    # GOES solar zenith 45, local zenith angle 45, relative azimuth difference angle 45
    bins_theta = np.arange(0, 91, d_th)
    # symmetric, so we change (-180, 180) to (0, 180)
    bins_phi = np.arange(-180, 181, d_phi)
    #bins_phi = np.hstack((np.arange(0, 170 + 5, 5), np.arange(170, 180 + 1, 1)))
    fw_rx, fw_ry, fw_rz, uw_rx, uw_ry, uw_rz = [np.zeros((N_bundles + 10, len(nu))) * np.nan for _ in range(6)]
    H = np.zeros((len(bins_theta) - 1, len(bins_phi) - 1))
    for i in range(len(rxyz_M)):
        fw_rxyz = rxyz_M[i]
        N_dw = len(fw_rxyz)
        fw_rx[0:N_dw, i] = np.array([x[0] for x in fw_rxyz])
        fw_ry[0:N_dw, i] = np.array([x[1] for x in fw_rxyz])
        fw_rz[0:N_dw, i] = np.array([x[2] for x in fw_rxyz])
        if dirc == 'UW':
            theta_v, phi_v = theta_phi(fw_rx[:, i], fw_ry[:, i], fw_rz[:, i])
        else:  # DW
            theta_v, phi_v = theta_phi(fw_rx[:, i], fw_ry[:, i], -fw_rz[:, i])
        ind = np.isnan(phi_v)
        theta_v = theta_v[~ind]
        phi_v = phi_v[~ind] - phi0
        phi_v[phi_v > math.pi] -= 2 * math.pi
        H_i, xedges, yedges = np.histogram2d(np.rad2deg(theta_v), np.rad2deg(phi_v), bins=(bins_theta, bins_phi))
        if Norm:
            H += H_i * np.cos(theta0) * 1 / N_bundles  # *F_dw_os[k]*3/N_bundles # 3 is dnu
        else:
            H += H_i * np.cos(theta0) * F_dw_os[i] * 3 / N_bundles
    theta_, phi_ = np.meshgrid(xedges[0:-1], yedges[0:-1])
    if not is_flux: # if the output should be radiance.
        ths = np.deg2rad(theta_.T + d_th / 2)  # rad dw # division 2 for the 2sintcost
        #print(np.sum(H))
        H /= 0.5 * np.sin(2 * ths)
    H /= np.deg2rad(d_th) * np.deg2rad(d_phi)  # per solid angle, in the direction of beam
    # ghi2d_show(H, logscale=False)
    theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
    H_t_p = intepolation_H(H, bins_phi, bins_theta, theta_idx, phi_idx, d_phi)
    # H_t_p = 0.5 * H_theta[theta_idx] / np.pi # isotripic
    #print(np.sum(H))
    #print(H[theta_idx, phi_idx])
    return H_t_p

def intepolation_H(H, bins_phi, bins_theta, theta_idx, phi_idx, d_phi=5):
    """
    Interpolates the weight at a specific azimuth angle (phi0) based on the histogram of intensities (H).
    """
    # Normalize H so that it sums to 1 over all phi
    # H is a 2D array where each row corresponds to a theta bin and each column corresponds to a phi bin
    # We are interested in the row corresponding to theta_idx
    # Assume:
    # H[theta_idx] is a 1D array of intensities over phi bins
    # phi = array of bin centers, same length as H[theta_idx]
    # phi0 = target azimuth angle (in degrees)
    # Step 1: Compute bin centers
    phi_centers = (bins_phi[:-1] + bins_phi[1:]) / 2  # shape: (n_phi,)
    theta_centers = (bins_theta[:-1] + bins_theta[1:]) / 2

    # Step 2: Total intensity at each theta
    H_theta = np.sum(H, axis=1) * np.deg2rad(d_phi)  # shape: (n_theta,)
    H_t = H[theta_idx, :]  # 1D intensity slice at specific theta

    # Step 3: Normalize azimuthal shape at that theta
    if H_theta[theta_idx] > 0:
        H_phi_normalized = H_t / H_theta[theta_idx]
    else:
        H_phi_normalized = np.zeros_like(H_t)

    # Step 4: Interpolation using cos(phi)
    cos_phi_centers = np.cos(np.deg2rad(phi_centers))
    cos_phi0 = np.cos(np.deg2rad(phi_centers[phi_idx]))  # or any target φ

    phi_weight = np.interp(cos_phi0, cos_phi_centers, H_phi_normalized)

    # Step 5: Multiply by total intensity at theta
    H_interp = H_theta[theta_idx] * phi_weight
    return H_interp

def calculate_relative_azimuth(sun_azimuth, satellite_azimuth):
    relative_azimuth = abs(sun_azimuth - satellite_azimuth)
    if relative_azimuth > 180:
        relative_azimuth = 360 - relative_azimuth
    return relative_azimuth


def find_bin_indices(thetai, phij, param='zenith'):
    # Define the bin edges
    d_th = 2
    d_phi = 5
    bins_theta = np.arange(0, 91, d_th) #46
    bins_phi = np.arange(-180, 181, d_phi) #73
    #bins_phi = np.hstack((np.arange(0, 170 + d_phi, d_phi), np.arange(170, 180 + 1, 1)))

    # Adjust phij to be in the range [0, 180]
    if phij > 180:
        phij = 360 - phij
    # Find the bin indices
    if param == 'zenith':
        i = np.digitize(thetai, bins_theta) - 1
        return i
    elif param == 'azimuth':
        j = np.digitize(phij, bins_phi) - 1
        return j

    if param == 'both':
        i = np.digitize(thetai, bins_theta) - 1
        j = np.digitize(phij, bins_phi) - 1
        return i, j

def earth_sun_dist_norm(ref, d2=0.3):
    # https://edc.occ-data.org/goes16/python/#radiance-to-reflectance
    # Define some constants needed for the conversion. From the pdf linked above
    # Apply the formula to convert radiance to reflectance
    ref = ref * d2

    # Make sure all data is in the valid data range
    ref = np.maximum(ref, 0.0)
    ref = np.minimum(ref, 1.0)
    return ref

def Sat_preprocess(data_dir, site, figlabel, sky, phase, sat='FY4A',timeofday='day'):
    if sat == 'FY4A':
        FY4A_dir= 'FY4A_data/'
        filename = 'AKA_radiance_satellite'
    elif sat == 'GOES17':
        FY4A_dir= 'GOES17_data/'
        filename = 'GOES17_day_radiance_satellite'
    elif sat == 'GOES16':
        FY4A_dir= 'GOES16_site_sat_data/'
        filename = f'GOES_{timeofday}_{site}_radiance_satellite_{phase}_{figlabel}'
    else:
        print("Invalid satellite name")
        return None
    df = pd.read_csv(data_dir + FY4A_dir + filename + '.csv')
    try:
        df = df.drop(columns=['C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14',
                                      'C08_rad', 'C09_rad', 'C10_rad', 'C11_rad',
                                      'C12_rad', 'C13_rad', 'C14_rad'])
    except KeyError:
        pass
    df = df[df['Sun_Zen'] <= 65]
    df['rela_azi'] = calculate_relative_azimuth_angle(df['Sat_Azi'], df['Sun_Azi'], input_type='deg')
    df['RH'] = df['RH'] / 100
    df.rename(columns={'RH': 'rh', 'Sun_Zen': 'th0','Sat_Zen':'local_Zen'}, inplace=True)
    # if phase == 'water':
    #     filtered_df = df[df['COD'] <= 50]
    try:
        df =df.set_index('time')
        df.index.name = "timestamp"
    except Exception:
        pass

    # Save the filtered DataFrame to an HDF5 file
    if sky =='clearsky':
        hdf5_file_path = data_dir + FY4A_dir + filename +'.h5'
    else:
        hdf5_file_path = data_dir + FY4A_dir + filename +'_day'+'.h5'
    df.to_hdf(hdf5_file_path, key='df', mode='w')

    print(f"Filtered DataFrame saved to {hdf5_file_path}")
    return FY4A_dir


def ghi2d_show(F_ghi_2d, logscale=True):
    font = 15
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    d_th = 2
    d_phi = 5
    bins_theta = np.arange(0, 91, d_th)
    bins_phi = np.arange(0, 181, d_phi)

    # Create a new figure with defined size
    fig = plt.figure(figsize=(5, 4))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=-0.1, hspace=0.0)

    # Apply logarithmic scaling if needed
    if logscale:
        Z = np.log10(F_ghi_2d.T + 1.0)
    else:
        Z = F_ghi_2d.T

    # Create the subplot
    ax1 = fig.add_subplot(gs1[0, 0])

    # The extent parameter defines the bounding box in data coordinates
    extent = [bins_theta[0], bins_theta[-1], bins_phi[0], bins_phi[-1]]

    # Display the image with the appropriate bin edges
    im = ax1.imshow(Z, cmap='Spectral_r', origin="lower", extent=extent)

    # Set x and y labels
    ax1.set_xlabel('Theta (degrees)')
    ax1.set_ylabel('Phi (degrees)')

    # Set ticks for x-axis based on the bins
    ax1.set_xticks(np.arange(0, 90 + 30, 30))  # Set ticks at each bin edge
    ax1.set_yticks(np.arange(-180, 180 + 30, 30))  # Set ticks at each bin edge

    # Add a color bar
    cbar1 = plt.colorbar(im)
    if logscale:
        cbar1.set_label('log$_{10}$ of intensity [W m$^{-2}$ sr$^{-1}$]', rotation=90,
                        labelpad=0, fontsize=font, family=fontfml)  # ****
    else:
        cbar1.set_label('Intensity [W m$^{-2}$ sr$^{-1}$]', rotation=90,
                        labelpad=0, fontsize=font, family=fontfml)
    cbar1.ax.tick_params(labelsize=font, labelcolor='black')  # ****
    # Show the plot
    plt.show()
    return None