import numpy as np
def shadow_matching(lat_s, lon_s, theta_z, phi_az, cth_km=7):
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