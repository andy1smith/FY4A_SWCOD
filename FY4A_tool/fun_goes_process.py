import datetime
import numpy as np
from timezonefinder import TimezoneFinder
import pytz
from skyfield.api import load, Topos
import numpy as np
import matplotlib.pyplot as plt
import os
import netCDF4 as nc



def convert_to_utc(t,lat,lon):
    # J2000 epoch mid-point in seconds since 2000-01-01 12:00:00
    j2000_epoch = t

    # Convert to UTC time
    j2000_start = datetime.datetime(2000, 1, 1, 12, 0, 0)
    utc_time = j2000_start + datetime.timedelta(seconds=j2000_epoch)
    #print("UTC Time:", utc_time)
    #print(t)

    # Convert UTC time to Eastern Time
    # tf = TimezoneFinder()
    # timezone_name = tf.timezone_at(lat=lat, lng=lon)
    # region = pytz.timezone(timezone_name)
    # local_time = utc_time.astimezone(region)

    return utc_time#local_time



def calculate_geodetic_coordinates(y, x, r_eq, r_pol, H, lon_0):
    """
    Calculate geodetic latitude and longitude from fixed grid coordinates
    
    Parameters:
    y: N/S Grid Fixed Scanning Angle (in radians)
    x: E/W Grid Fixed Scanning Angle (in radians)
    r_eq: Equatorial radius (semi-major axis) in meters
    r_pol: Polar radius (semi-minor axis) in meters
    H: Perspective point height + semi-major axis in meters
    lon_0: Longitude of projection origin (in radians)
    
    Returns:
    Tuple of (geodetic latitude, geodetic longitude) in radians
    """
    # Calculate intermediate values
    a = np.sin(x)**2 + np.cos(x)**2 * (np.cos(y)**2 + (r_eq**2 / r_pol**2) * np.sin(y)**2)
    b = -2 * H * np.cos(x) * np.cos(y)
    c = (H+r_eq)*(H-r_eq)
    # print(a, b, c)
    # Calculate satellite distance from the point
    rs = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    
    # Calculate S_x, S_y, S_z components
    sx = rs * np.cos(x) * np.cos(y)
    sy = -rs * np.sin(x)
    sz = rs * np.cos(x) * np.sin(y)
    # print(rs, sx, sy, sz)
    # Calculate geodetic latitude
    lat = np.arctan((r_pol**2 / r_eq**2) * sz / np.sqrt((H - sx)**2 + sy**2))
    
    # Calculate geodetic longitude
    lon = lon_0 - np.arctan2(sy, H - sx)
    
    return lat, lon



def sun_zenith_angle(utc_time, lons, lats, input_type = 'deg'):
    """
    Calculate the zenith angle of the sun at a given time and location
    longs degrees, lats degrees
    """
    # Load timescale and ephemeris data
    ts = load.timescale()
    eph = load('../data/other/de421.bsp')  # planetary ephemeris data

    # Sun object
    sun = eph['sun']

    # Convert single datetime object to Skyfield time
    time = ts.utc(utc_time.year, utc_time.month, utc_time.day, utc_time.hour, utc_time.minute, utc_time.second)

    zenith_angles = []
    azimuth_angles = []

    for lon, lat in zip(lons, lats):
        # Define the position on Earth
        observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
        
        # Calculate the position of the sun
        observer_at_time = (eph['earth'] + observer).at(time)
        astrometric = observer_at_time.observe(sun)
        alt, az, distance = astrometric.apparent().altaz()

        # Calculate zenith angle
        zenith_angle = 90 - alt.degrees
        azimuth_angles.append(az.degrees)
        zenith_angles.append(zenith_angle)

    return zenith_angles,azimuth_angles



def calculate_local_zenith_angle(satellite_lon, satellite_lat, point_lon, point_lat, input_type='deg',
                                 H=42164160, r_eq=6378137):
    """
    Calculate the local zenith angle for a given point observed by a satellite.
    
    Parameters:
    -----------
    satellite_lon : float
        Longitude of the satellite sub-point (in degrees)
    satellite_lat : float
        Latitude of the satellite sub-point (in degrees)
    point_lon : float
        Longitude of the observation point (in degrees)
    point_lat : float
        Latitude of the observation point (in degrees)
    H : float, optional
        Satellite height from the center of the Earth (in meters)
    r_eq : float, optional
        Earth's semi-major axis (in meters)
    
    Returns:
    --------
    local_zenith_angle_deg : float
        Local zenith angle in degrees
    local_zenith_angle_rad : float
        Local zenith angle in radians
    """
    # Convert all inputs to radians
    if input_type == 'deg':
        sat_lon_rad = np.deg2rad(satellite_lon)
        sat_lat_rad = np.deg2rad(satellite_lat)
        point_lon_rad = np.deg2rad(point_lon)
        point_lat_rad = np.deg2rad(point_lat)
    else:
        sat_lon_rad = satellite_lon
        sat_lat_rad = satellite_lat
        point_lon_rad = point_lon
        point_lat_rad = point_lat
    
    # Calculate β (beta) - angular distance between satellite sub-point and observation point
    beta = np.arccos(
        np.cos(point_lat_rad - sat_lat_rad) * 
        np.cos(point_lon_rad - sat_lon_rad)
    )
    
    # Calculate the denominator term
    denominator = np.sqrt(
        H**2 + r_eq**2 - 2 * H * r_eq * np.cos(beta)
    )
    
    # Calculate local zenith angle calculation term
    zenith_calc_term = (H * np.sin(beta)) / denominator
    
    # Bound the term to valid arcsin input
    zenith_calc_term = np.clip(zenith_calc_term, -1, 1)
    
    # Calculate local zenith angle in degrees
    local_zenith_angle_deg = np.rad2deg(np.arcsin(zenith_calc_term))
    
    # Calculate local zenith angle in radians
    local_zenith_angle_rad = np.deg2rad(local_zenith_angle_deg)
    
    # Check visibility condition
    visibility_limit = np.arccos(r_eq / (H + r_eq))
    is_visible = beta < visibility_limit
    
    if input_type == 'deg':
        return local_zenith_angle_deg
    else:
        return local_zenith_angle_rad
    
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

def calculate_local_azimuth(point_lon, point_lat, input_type='deg', ref_lat=90, ref_lon=0):
    """
    Calculate the azimuth angle from a reference point to another point.
    
    Parameters:
    -----------
    ref_lat, ref_lon : North Pole
    point_lat : np.array
        Latitude of the target points (in degrees)
    point_lon : np.array
        Longitude of the target points (in degrees)
    
    Returns:
    --------
    azimuth : np.array
        Azimuth angle from reference point to target points (in degrees)
        0 degrees is North, increasing clockwise
    """
    # Convert to radians
    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)
    
    if input_type == 'deg':
        point_lat_rad = np.deg2rad(point_lat)
        point_lon_rad = np.deg2rad(point_lon)
    else:  # radians
        point_lat_rad = point_lat
        point_lon_rad = point_lon
    
    # Calculate difference in longitudes
    delta_lon = point_lon_rad - ref_lon_rad

    # Use arctan2 for correct quadrant handling
    azimuth_rad = np.arctan2(
        np.sin(delta_lon),
        np.cos(ref_lat_rad) * np.tan(point_lat_rad) - np.sin(ref_lat_rad) * np.cos(delta_lon)
    )
    
    # Convert to degrees and normalize to 0-360 range
    azimuth_deg = np.rad2deg(azimuth_rad)
    azimuth_deg = (azimuth_deg + 360) % 360
    
    return azimuth_deg, azimuth_rad

def Conver_ref(Rad,ch_name,CMIP_file_path):
    #CMIP_file_path = '../GOES_data/goes16/ABI-L2-CMIPF/2021160/'
    #Fls = os.listdir(CMIP_file_path)
    #head = 'ABI-L2-CMIPF_2021_160_20_OR_ABI-L2-CMIPF-M6'
    #CMIP_flie = [f for f in Fls if f.startswith(head + ch_name)][0]
    CMIP_dataset = nc.Dataset(CMIP_file_path, mode='r')
    kappa0 = CMIP_dataset.variables['kappa0'][:]
    return Rad*kappa0

def goes_lam(pd_data, file_dir):
    # [W/m2/sr/um] -> [W/m2/sr]
    channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    dirpath = file_dir + 'GOES-R_ABI_FM2_SRF_CWG/'
    Rad = []
    for channel in channels:
        channel_number = int(channel[-2:])
        channel_srf = os.path.join(
            dirpath,
            'GOES-R_ABI_FM2_SRF_CWG_ch{}.txt'.format(channel_number)
        )
        calibration = np.genfromtxt(channel_srf, skip_header=2)
        # calibration_nu = calibration[:, 1]  # cm-1
        calibration_wl = calibration[:, 0]  # wavelength [um]
        calibration_srf = calibration[:, 2] # relative SRF [-]
        # radiance per channel
        # get band equivalent width = Eqw = \int_{nu_1}^{nu_2} R_{nu} dnu
        Eqw = np.trapz(calibration_srf, x=calibration_wl)

        # get the channel radiance [W/(m^2 sr)]
        spectral_rad = pd_data[channel_number-1]  # spectral radiance [W/(m^2 sr um^-1)]
        rad = spectral_rad * Eqw  # radiance [W/m^2 sr]
        Rad.append(rad)
    return Rad