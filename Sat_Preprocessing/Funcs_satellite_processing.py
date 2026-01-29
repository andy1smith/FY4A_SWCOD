from skyfield.api import load, wgs84
from datetime import datetime
import pytz
import glob
import pandas as pd
import pvlib

def preprocess_clearsky_periods(year, site, lat, lon, alt):
    """
    Find the clearsky periods.

    Parameters
    ----------
    site : str
        SURFRAD station name.
    lat, lon, alt : float
        The latitude, longitude, and altitude of the site.

    """

    # the load the pre-processed ground data
    files = glob.glob('./SURFRAD/{}*{}*.csv'.format(str(year), site.lower()))
    df = pd.read_csv(files[0])


    # resample
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df = df.resample("5min", closed="right", label="right").mean()
    df = df.dropna(how="any")


    # split into day/night using solar angle
    zenith_threshold = 85  # zenith threshold [deg] for day/night split
    df_night = df[df["zen"] > zenith_threshold]
    #df_night.to_hdf("data/surfrad/{}_night.h5".format(site), "df", mode="w")
    print("Night:", df_night.shape, df_night.index[0], df_night.index[-1])
    df = df[df["zen"] <= zenith_threshold]

    # add clearsky irradiance
    loc = pvlib.location.Location(lat, lon, altitude=alt)
    cs = loc.get_clearsky(df.index)
    df.insert(df.shape[1], 'ghi_clear', cs["ghi"])
    df.insert(df.shape[1], 'dni_clear', cs["dni"])

    # GHI and DNI threshold values: G, M, L, sig, S
    #thresholds = [75, 75, 10, 0.005, 8]      # Reno's 2016 paper
    thresholds = [100, 100, 50, 0.01, 10.0]  # Rich's paper

    # rolling window
    i_window = 30  # window size [mins]
    criteria_sum = np.zeros(df.shape[0])
    for i in range(i_window, df.shape[0], i_window):

        # backwards selection: (i_start, i_end]
        i_start = i - i_window + 1
        i_end = i + 1
        ghi = df.iloc[i_start:i_end]['dw_solar'].values
        ghi_clear = df.iloc[i_start:i_end]['ghi_clear'].values

        # clearsky criteria
        criteria = np.zeros(5)

        # average irradiance
        criteria[0] = (np.abs(np.nanmean(ghi) - np.nanmean(ghi_clear)) < thresholds[0])

        # max irradiance
        criteria[1] = np.abs(np.max(ghi) - np.max(ghi_clear)) < thresholds[1]

        # irradiance increment
        diff_ghi = np.diff(ghi)
        diff_ghi_clear = np.diff(ghi_clear)
        criteria[2] = (
            np.abs(np.sum(np.sqrt(diff_ghi ** 2)) - np.sum(np.sqrt(diff_ghi_clear ** 2)))
            < thresholds[2]
        )

        # std of irradiance increment
        criteria[3] = np.abs(
            np.std(diff_ghi) / np.mean(ghi)
            - np.std(diff_ghi_clear) / np.mean(ghi_clear)
        ) < thresholds[3]

        # max irradiance increment
        criteria[4] = np.max(np.abs(diff_ghi - diff_ghi_clear)) < thresholds[4]

        # sum of the criteria (5=all criteria met)
        criteria_sum[i_start:i_end] = criteria.sum()

    # if all criteria are met, then the sky is clear
    df.insert(df.shape[1], 'clearsky', criteria_sum == 5)
    df.reset_index(inplace=True)
    # export data
    df.to_hdf(f"./SURFRAD/preprocessed/{site}_day.h5", "df", mode="w")
    df[df['clearsky'] == True].to_hdf('./SURFRAD/preprocessed/{}_clear.h5'.format(site), 'df', mode="w",)
    df[df['clearsky'] == False].to_hdf('./SURFRAD/preprocessed/{}_cloudy.h5'.format(site), 'df', mode="w")
    print(site, df.shape, df.index[0], df.index[-1])
    n_clear = df[df["clearsky"] == True].shape[0]
    n_cloudy = df[df["clearsky"] == False].shape[0]
    n_day = df.shape[0]
    print("clear = {:>6,d} ({:>3.1%}), cloudy = {:>12,d} ({:>3.1%})".format(n_clear, n_clear / n_day, n_cloudy, n_cloudy / n_day))
    #print("Clear: ", df[df['clearsky'] == True].shape)
    #print("Cloudy:", df[df['clearsky'] == False].shape)
    print(df.columns)


def shadow_matching(time, lon, lat, half_crop,lon_int,lat_int,lon_s,lat_e):
    theta_z, phi_az = satellite_initial_guess_angle(time, lon, lat)
    lat_c, lon_c = shadow_matching_Hz(lat, lon, theta_z, phi_az)
    Shadcor_lon_idx = int((lon_c - lon_s) / lon_int)
    Shadcor_lat_idx = int((lat_e - lat_c) / lat_int)

    lon_start_idx = max(0, Shadcor_lon_idx - half_crop)
    lon_end_idx = min(1750, Shadcor_lon_idx + half_crop)  # lon, 1750 pixel
    lat_start_idx = max(0, Shadcor_lat_idx - half_crop)
    lat_end_idx = min(1000, Shadcor_lat_idx + half_crop)  # lat, 1000 pixel
    return lon_start_idx,lon_end_idx,lat_start_idx,lat_end_idx

def satellite_initial_guess_angle(utc_time, lon, lat, input_type='deg'):
    """
    Calculate the zenith angle of the sun at a given time and location
    utc_time: string 'YYYY-MM-DD HH:MM:SS' or datetime
    lon, lat: degrees
    """

    # 1. Convert string to datetime
    if isinstance(utc_time, str):
        utc_time = datetime.strptime(utc_time, '%Y-%m-%d %H:%M:%S')

    # 2. Make timezone-aware (naive → assume UTC)
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    else:
        utc_time = utc_time.astimezone(pytz.utc)

    # 3. Skyfield time conversion (Skyfield needs a list)
    ts = load.timescale()
    t = ts.from_datetimes([utc_time])    # <-- FIX

    # Load Ephemeris
    eph = load('../data/other/de421.bsp')
    sun = eph['sun']
    earth = eph['earth']

    # Observer location
    observer = earth + wgs84.latlon(lat, lon)

    # Sun position
    astrometric = observer.at(t).observe(sun)
    alt, az, distance = astrometric.apparent().altaz()

    # Extract scalar values (alt and az are arrays)
    alt_deg = alt.degrees[0]
    az_deg = az.degrees[0]

    # Zenith angle
    zenith = 90.0 - alt_deg

    return zenith, az_deg


def shadow_matching_Hz(lat_s, lon_s, theta_z, phi_az, cth_km=7):
    """
    Calculate Cloud Location using given station coord, create by gpt
    geometric formulas (Small Angle Approximation).

    Args:
        lat_s, lon_s: Station coordinates (Degrees)
        theta_z: Solar Zenith Angle (Degrees)
        phi_az:  Solar Azimuth Angle (Degrees, 0=N, 90=E, 180=S)
        cth_km:  Cloud Top Height (km)

    Returns:
        lat_c, lon_c: Cloud coordinates (Degrees)
    """
    # Constants
    R_earth_km = 6371.0

    # 1. Convert inputs to Radians for numpy
    # Note: Lat/Lon are only converted when used inside trig functions
    rad_z = np.radians(theta_z)
    rad_az = np.radians(phi_az)
    rad_lat_s = np.radians(lat_s)

    # 2. Calculate the Horizontal Distance (Shadow Length)
    # d = H * tan(theta)
    dist_h = cth_km * np.tan(rad_z)

    # 3. Calculate Displacements (in Radians on the sphere)
    # Delta Lat = (d * cos(az)) / R
    delta_lat_rad = (dist_h * np.cos(rad_az)) / R_earth_km

    # Delta Lon = (d * sin(az)) / (R * cos(station_lat))  <-- CRITICAL CORRECTION
    delta_lon_rad = (dist_h * np.sin(rad_az)) / (R_earth_km * np.cos(rad_lat_s))

    # 4. Convert Displacements to Degrees
    delta_lat_deg = np.degrees(delta_lat_rad)
    delta_lon_deg = np.degrees(delta_lon_rad)

    # 5. Apply shift
    # Since we go FROM Station (Shadow) TO Cloud (Source), 
    # we move TOWARDS the Sun. The sign is effectively positive along the Azimuth vector.
    lat_c = lat_s + delta_lat_deg
    lon_c = lon_s + delta_lon_deg

    return lat_c, lon_c

from pyresample import geometry, kd_tree
import matplotlib.pyplot as plt
import numpy as np


def plot_parallax_comparison(xr_original, xr_corrected, site_name="Site"):
    """
    Plots Original vs Parallax Corrected COD.
    """
    # Get Data
    cod_orig = xr_original['Retrieved_COD'].values
    cod_corr = xr_corrected['COD_Corrected'].values

    # Get Center Coordinates (Your Site)
    # Assuming 11x11 grid, center is (5, 5)
    cy, cx = cod_orig.shape[0] // 2, cod_orig.shape[1] // 2

    # Setup Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Common settings
    vmin = 0
    vmax = np.nanmax(cod_orig)
    cmap = 'Blues'

    # --- Plot 1: Original (Apparent Position) ---
    im1 = axes[0].imshow(cod_orig, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("1. Original Retrieved COD\n(Where satellite SEES it)")
    axes[0].scatter(cx, cy, c='red', marker='x', s=100, label='Site')
    plt.colorbar(im1, ax=axes[0], label='COD')

    # --- Plot 2: Corrected (True Position) ---
    im2 = axes[1].imshow(cod_corr, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("2. Parallax Corrected COD\n(Where cloud ACTUALLY is)")
    axes[1].scatter(cx, cy, c='red', marker='x', s=100, label='Site')
    plt.colorbar(im2, ax=axes[1], label='COD')

    # --- Plot 3: Difference (The Shift) ---
    # Shows how much values changed at each pixel
    diff = cod_corr - cod_orig
    limit = np.nanmax(np.abs(diff))
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-limit, vmax=limit)
    axes[2].set_title(f"3. Difference (Shift)\nRed = Cloud Moved IN\nBlue = Cloud Moved OUT")
    axes[2].scatter(cx, cy, c='black', marker='x', s=100)
    plt.colorbar(im3, ax=axes[2], label='Delta COD')

    plt.suptitle(f"Parallax Correction Check - {site_name}", fontsize=14)
    plt.tight_layout()
    plt.show()


def generate_latlon_grid(xr_sat, center_lat, center_lon, resolution_deg=0.04):
    """
    Converts x,y indices back to Lat/Lon maps based on the center coordinate.

    Args:
        xr_sat: The cropped DataArray/Dataset (shape e.g. 11x11)
        center_lat: The latitude of the middle pixel
        center_lon: The longitude of the middle pixel
        resolution_deg: The pixel size in degrees.
                        FY-4A 4km resolution is approx 0.04 degrees.

    Returns:
        xr_sat_with_coords: The input dataset with 'lat' and 'lon' coordinates added.
    """

    # 1. Get Shapes
    # We assume 'y' is rows (latitude), 'x' is columns (longitude)
    ny = xr_sat.sizes['y']
    nx = xr_sat.sizes['x']

    # 2. Find the index of the center pixel
    # If shape is 11, indices are 0..10, center is 5.
    cy = ny // 2
    cx = nx // 2

    # 3. Create Grid of Indices
    # shape (ny, nx)
    y_indices, x_indices = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    # 4. Calculate Offsets from Center
    # How many pixels away from center?
    dy = y_indices - cy
    dx = x_indices - cx

    # 5. Apply Inverse Logic
    # Forward: lat_idx = (lat_e - lat) / res  -> Lat decreases as Index increases (Top to Bottom)
    # Inverse: lat = center_lat - (dy * res)
    lat_grid = center_lat - (dy * resolution_deg)

    # Forward: lon_idx = (lon - lon_s) / res  -> Lon increases as Index increases (Left to Right)
    # Inverse: lon = center_lon + (dx * resolution_deg)
    lon_grid = center_lon + (dx * resolution_deg)

    # 6. Assign to Dataset
    xr_sat_out = xr_sat.copy()

    # Assign as Coordinates so Parallax function finds them automatically
    xr_sat_out = xr_sat_out.assign_coords({
        'lat': (('y', 'x'), lat_grid),
        'lon': (('y', 'x'), lon_grid)
    })

    return xr_sat_out

def apply_parallax_correction(xr_sat, cloud_height_km=7.0):
    """
    Corrects the Lat/Lon coordinates of the COD data based on cloud height.
    Resamples the shifted data back to the original grid.

    Args:
        xr_sat: Dataset containing 'COD', 'Local_Zen', 'Sat_Azi', 'lat', 'lon'
        cloud_height_km: Scalar (e.g. 6.0) or 2D array of Cloud Top Heights (km)

    Returns:
        xr_sat_corrected: Dataset with 'COD_corrected' on the original grid.
    """

    # --- 1. Get Inputs ---
    # Convert angles to radians
    vza_rad = np.deg2rad(xr_sat['Local_Zen'].values)  # View Zenith
    vaa_rad = np.deg2rad(xr_sat['Sat_Azi'].values)  # View Azimuth (0=North, 90=East)


    lat_old = xr_sat['lat'].values
    lon_old = xr_sat['lon'].values

    # If lat/lon are 1D axes, mesh them into 2D grids
    if lat_old.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_old, lat_old)
    else:
        lon_grid, lat_grid = lon_old, lat_old

    # --- 2. Calculate Displacement ---
    # Distance to shift (in km)
    # d = H * tan(theta)
    dist_km = cloud_height_km * np.tan(vza_rad)

    # Break into Lat/Lon components (km)
    # NOTE: We want to move the cloud TOWARDS the satellite.
    # If Sat_Azi is the direction looking AT the satellite, we move in that direction.
    # If Sat_Azi is the direction FROM the satellite (standard), we move opposite.
    # *Standard Satellite Azimuth* usually points from Pixel -> North -> East.
    # To correct parallax, we move the coordinate *towards* the satellite.
    # This effectively means calculating the "True Ground Location" of the pixel.

    # Standard meteorological convention: Azimuth is direction *from* pixel *to* satellite?
    # Usually: 0 is North. 180 is South.
    # Corrected Lat = Old Lat - displacement_North
    # We use approximation: 1 deg lat = 111 km. 1 deg lon = 111 * cos(lat) km.

    # Calculate shifts in Degrees
    # We subtract because the cloud appears "pushed" away.
    # We want the coordinate where the cloud *actually* is.

    delta_lat_km = -dist_km * np.cos(vaa_rad)
    delta_lon_km = -dist_km * np.sin(vaa_rad)

    # Convert km to degrees
    R_earth = 6371.0  # km
    deg_per_km_lat = 1 / 111.32
    # Longitude degrees depend on latitude
    deg_per_km_lon = 1 / (111.32 * np.cos(np.deg2rad(lat_grid)))

    lat_new = lat_grid + (delta_lat_km * deg_per_km_lat)
    lon_new = lon_grid + (delta_lon_km * deg_per_km_lon)

    # --- 3. Resample back to Original Grid ---
    # We now have the "Real" locations of the data points (lat_new, lon_new).
    # But these are irregular (swath-like). We need to map them back to the
    # regular 11x11 grid of the original image.

    # A. Define the Target Grid (Your original regular grid)
    target_def = geometry.GridDefinition(lons=lon_grid, lats=lat_grid)

    # B. Define the Source Grid (The shifted irregular coordinates)
    source_def = geometry.SwathDefinition(lons=lon_new, lats=lat_new)

    # C. Resample
    # We use Nearest Neighbor or Bilinear.
    # For Parallax, we are essentially "moving pixels", so KDTree (Nearest) is robust.
    # radius_of_influence: 4km pixel -> search within ~6km
    print("Resampling Parallax Corrected Grid...")

    # Get the COD data to move
    cod_data = xr_sat['Retrieved_COD'].values  # or 'Retrieved_COD'

    # Run Resampling
    # fill_value=np.nan ensures pixels shifted "out of frame" become empty
    cod_corrected = kd_tree.resample_nearest(
        source_def,
        cod_data,
        target_def,
        radius_of_influence=6000,  # 6km radius for 4km pixels
        fill_value=np.nan
    )

    # --- 4. Package Result ---
    xr_sat_corrected = xr_sat.copy()
    xr_sat_corrected['COD_Corrected'] = (('y', 'x'), cod_corrected)

    return xr_sat_corrected


# ==========================================
# How to use it
# ==========================================

# 1. Define Height (Scalar or Map)
# If you don't have CTH, assume a mean height for deep convection (e.g., 7km)
# cloud_height = 7.0
#
# # 2. Run
# xr_parallax = apply_parallax_correction(xr_sat, cloud_height)
#
# # 3. Visualization Check
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# xr_parallax['COD'].plot(cmap='jet')
# plt.title("Original COD")
#
# plt.subplot(1, 2, 2)
# xr_parallax['COD_Corrected'].plot(cmap='jet')
# plt.title(f"Parallax Corrected (H={cloud_height}km)")
#
# plt.tight_layout()
# plt.show()