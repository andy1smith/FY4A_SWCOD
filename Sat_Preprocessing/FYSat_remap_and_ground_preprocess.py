import h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool
import pvlib
from pvlib import clearsky
from pvlib.solarposition import get_solarposition
from skyfield.api import load, Topos
import os,re
from itertools import islice
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from Funcs_satellite_processing import *
from clearsky_model.clearsky_filter import daytype_filter

from pysolar.solar import get_geocentric_sun_declination, get_hour_angle

def solar_hour_angle(df,lat,lon,alt):
    """
    Calculates the Solar Declination and Hour Angle for a single UTC timestamp.
    """
    # Calculate Solar Declination Angle (δ) for the specific date
    solar_pos = pvlib.solarposition.get_solarposition(
        time=df.index,
        latitude=lat,
        longitude=lon,
        altitude=alt,
        # Use the 'nrel_numpy' method for fast, high-precision results
        method='nrel_numpy'
    )
    eot_minutes = solar_pos['equation_of_time'] # minutes
    utc_hour_decimal = df.index.hour + df.index.minute / 60.0 + df.index.second / 3600.0

    # local solar time
    lst_hours = utc_hour_decimal + (lon / 15.0) + (eot_minutes / 60.0)
    return lst_hours

def process_site(args):
    """

    Parameters
    ----------
    coords : station coordinates
    time: UTC time
    theta_z, phi_az:  degree
    Returns
    -------

    """
    data_dir = "/Volumes/HP P900/"
    filtered_files, site_name, coords, lon_s, lon_e, lat_s, lat_e, lon_int, lat_int, pixel, save_path, sky = args
    half_crop = pixel // 2
    # location & index for site
    central_lon_idx = int((coords['longitude'] - lon_s) / lon_int)
    central_lat_idx = int((coords['latitude'] - lat_s) / lat_int)
    #central_lat_idx = int((lat_e - coords['latitude']) / lat_int)
    # crop according to pixel size
    lon_start_idx = max(0, central_lon_idx - half_crop)
    lon_end_idx = min(1750, central_lon_idx + half_crop)  # lon, 1750 pixel
    lat_start_idx = max(0, central_lat_idx - half_crop)
    lat_end_idx = min(1000, central_lat_idx + half_crop)  # lat, 1000 pixel

    # process data
    channels = ['SunZenith', 'SunAzimuth', 'SatelliteAzimuth', 'SatelliteZenith',
                'SunGlintAngle','elevation' ]+ ['Channel{:02d}'.format(i + 1) for i in range(7)]
    channel_data = {channel: [] for channel in channels}
    timestamps = []

    # filtered_files is a list of all timestamp
    for file_path in filtered_files:
        #lon_start_idx,lon_end_idx,lat_start_idx,lat_end_idx = lon_start_idx0,lon_end_idx0,lat_start_idx0,lat_end_idx0
        with h5py.File(data_dir + 'FY_L1_2021/' + file_path, 'r') as f:

            # check to ensure all channels exist
            if not all(channel in f for channel in channels):
                logging.warning(f'Missing one or more channels in file {file_path}')
                continue

            timestamp = os.path.basename(file_path).split('_')[3]
            time = datetime.strptime(timestamp, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            timestamps.append(time)
            if sky == 'cloudy':
                lon_start_idx, lon_end_idx, lat_start_idx, lat_end_idx = shadow_matching(time, coords['longitude'],
                                                                                         coords['latitude'],
                                                                                         half_crop, lon_int, lat_int,
                                                                                         lon_s, lat_e)
            # calibrate & crop
            for channel in channels:
                df_channel = f[channel][:].astype(float)
                # invalid values (-9999)
                df_channel[df_channel == -9999] = np.nan

                if 'scale_factor' in f[channel].attrs and 'add_offset' in f[channel].attrs:
                    scale_factor = f[channel].attrs['scale_factor']
                    add_offset = f[channel].attrs['add_offset']
                else:
                    if channel in ['Channel{:02d}'.format(i + 1) for i in range(7)]:
                        scale_factor, add_offset = 0.0001, 0
                    # elif channel in ['Channel{:02d}'.format(i + 1) for i in range(7, 14)]:
                    #     scale_factor, add_offset = 0.01, 273
                    elif channel in ['SatelliteAzimuth', 'SatelliteZenith', 'SunAzimuth', 'SunGlintAngle', 'SunZenith', 'elevation']:
                        scale_factor, add_offset = 0.02, 0

                channel_cali = df_channel * scale_factor + add_offset  # calibrated data
                channel_crop = channel_cali[lat_start_idx:lat_end_idx+1, lon_start_idx:lon_end_idx+1].flatten()
                channel_data[channel].append(channel_crop)

    # save to csv
    for channel, data in channel_data.items():
        if data:
            df = pd.DataFrame(data, columns=range(pixel * pixel), index=pd.Index(timestamps, name='time'))
            csv_path = os.path.join(save_path, '{}_{}.csv'.format(site_name, channel))
            df.to_csv(csv_path)
            logging.info(f'Saved: {csv_path}')


def sun_zenith_angle(times, lon, lat):
    """
    Vectorized calculation of sun zenith and azimuth angles.

    Parameters:
    times : pandas.DatetimeIndex or series of datetime objects (assumed UTC)
    lon   : float, longitude in degrees
    lat   : float, latitude in degrees
    """
    # 1. Load data (Load once to save I/O time)
    # Note: Ideally, load 'eph' outside the function if calling this multiple times
    ts = load.timescale()
    eph = load('../data/other/de421.bsp')
    sun = eph['sun']
    earth = eph['earth']

    # 2. Convert pandas DatetimeIndex to Skyfield Time object (Vectorized)
    # ts.from_datetimes handles the whole array at once
    t = ts.from_datetimes(times)

    # 3. Define observer
    observer = earth + Topos(latitude_degrees=lat, longitude_degrees=lon)

    # 4. Calculate position (Vectorized)
    # .observe() and .altaz() will return arrays of length 4000
    astrometric = observer.at(t).observe(sun)
    alt, az, distance = astrometric.apparent().altaz()

    # 5. Calculate Zenith (90 - Altitude)
    # ele.degrees is a numpy array, so we can do direct subtraction
    zenith_angle = 90.0 - alt.degrees
    azimuth_angle = az.degrees

    return zenith_angle, azimuth_angle

def clearsky_filter(data_dir, site, lat, lon, ele):
    data = pd.read_csv(data_dir + 'CERN_instGHI_2021_UTC.csv')
    if site not in data.columns:
        print(f"Site {site} not found in data columns.")
        return None
     # prepare dataframe
    df = data[[site]]
    df = df.rename(columns={site: 'ghi'})
    df['Time'] = pd.date_range(start='2021-01-01', end='2022-01-01', freq='h')[:-1]
    df.set_index('Time', inplace=True)
    df = df[df['ghi']>0]
    df['ghi'] = df['ghi'].replace(0, np.nan)
    df = df.dropna(how="any")
    if df.empty:
        return None

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    #df['Sun_Zen'], df['Sun_Azi'] = sun_zenith_angle(df.index, lon, lat)
    solpos = get_solarposition(
        time=df.index,
        latitude=lat,
        longitude=lon,
        altitude=ele,  # meters
        #temperature=15  # Example: 15°C (optional, affects refraction)
    )
    # Extract the values
    df['Sun_Zen'] = solpos['zenith']  # True geometric zenith
    df['Sun_Zen_App'] = solpos['apparent_zenith']  # Refraction-corrected zenith
    df['Sun_Azi'] = solpos['azimuth']
    # split into day/night using solar angle
    zenith_threshold = 85  # zenith threshold [deg] for day/night split
    # df_night = df[df["Sun_Zen"] > zenith_threshold]
    # df_night.to_hdf("data/surfrad/{}_night.h5".format(site), "df", mode="w")
    # print("Night:", df_night.shape, df_night.index[0], df_night.index[-1])
    df = df[df["Sun_Zen"] <= zenith_threshold]

    # add clearsky irradiance
    loc = pvlib.location.Location(lat, lon, altitude=ele)
    tl = pvlib.clearsky.lookup_linke_turbidity(df.index, lat, lon)
    cs = loc.get_clearsky(df.index, model='ineichen', linke_turbidity=tl)
    df['ghi_clear'] = cs['ghi']
    df['LST'] = solar_hour_angle(df.copy(), lat, lon, ele) # local solar time in hours
    clear_day, cloudy_day, whole_clearday = daytype_filter(df.copy(), lon)

    clear_day.reset_index(inplace=True)
    cloudy_day.reset_index(inplace=True)
    whole_clearday.reset_index(inplace=True)
    # save
    clear_day.to_hdf('./Ground/preprocessed_GHI/{}_clear.h5'.format(site), key='df', mode="w", )
    cloudy_day.to_hdf('./Ground/preprocessed_GHI/{}_cloudy.h5'.format(site), key='df', mode="w")
    whole_clearday.to_hdf('./Ground/preprocessed_GHI/{}_consistent_clear_days.h5'.format(site), key='df', mode="w")
    return None


def extract_region(pixel, sites, lon_s, lon_e, lat_s, lat_e, lon_int, lat_int, sky, filtered_files):
    """extract data in parallel by DOY

    parameter:
    pixel: int
        crop size (e.g. 11: crop image in 11*11 pixels)
    sites: dict-like
        stations information (e.g. {'AKA': {'longitude': 80.82883333, 'latitude': 40.61683333}}'
    lon_s, lon_e: float
        longitude start / end for original image
    lat_s, lat_e: float
        latitude start / end for original image
    lon_int, lat_int: float
        longitude / latitude resolution in degree"""
    site_name, coords = sites
    # save path
    scenarios = []
    # for site_name, coords in sites:#.items():
    save_path = f'./cropped_FY2021_{sky}/{site_name}'
    os.makedirs(save_path, exist_ok=True)
    scenarios.append([filtered_files, site_name, coords, lon_s, lon_e, lat_s, lat_e, lon_int,
                      lat_int, pixel, save_path, sky])

    # process data in parallel
    pool = Pool()
    pool.map(process_site, scenarios)
    pool.close()
    # process_site(scenarios[0])

def extract_fy4a_date(filename):
    match = re.search(r'(\d{14})_(\d{14})', filename)
    if match:
        start_str, _ = match.groups()
        start_dt = datetime.strptime(start_str, '%Y%m%d%H%M%S')
        return start_dt
    return None

def preprocess_ground(df, data_dir):
    # site by site preprocess clear/cloudy sky periods
    #df = df[df['site'] == 'FQA']
    for row in df.itertuples():
        print(row.site)
        clearsky_filter(data_dir, row.site, row.latitude, row.longitude,  # assuming the CSV header is 'longitude' based on 'latitude'
                                             row.elve)
    print('All ground stations preprocessed!')

if __name__ == '__main__':
    data_dir = "/Volumes/HP P900/"#"../FY4A_data/"
    # Setup basic configuration for logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # location information
    # 44 sites to 14 sites for testing
    df = pd.read_csv("../FY4A_data/"+'CERN_info.csv')
    df = df[
        (df['latitude'] >= 0) &
        (df['latitude'] <= 60)# &
        #(df['elve'] <= 500)  # you can adjust or remove altitude filter as needed
        ]
    # Groud meansurement data export the cloudy day / clear sky periods based on pvlib
    ground_preprocess = True
    #df = df [df['site']=='BJC']
    if ground_preprocess:
        preprocess_ground(df.copy(), data_dir)
    else:
        print('Skip ground clear, cloudy classification!')

    sites = df.set_index('site')[['longitude', 'latitude']].to_dict(orient='index')
    months = [0,1,2,3,4,5,6,7,8,9,10,11,12]  # June, July, August
    ground_dir = './Ground/preprocessed/'
    sky = 'clear'  # 'clear' or 'cloudy'
    # for i in range(17,len(sites)):
    for idx, (site_name, coords) in enumerate(sites.items()):
        #site = dict(islice(sites.items(), i))
        #site_name = list(site.keys())[i-1]
        # if site_name != 'FQA':
        #     continue # ONLY PROCESS agricutural sites
        print("Processing site:", site_name)
        try:
            ground_path = ground_dir + '{}_{}.h5'.format(site_name,sky)
            df_ground = pd.read_hdf(ground_path, key='df')
        except FileNotFoundError:
            print(f"File not found for site {site_name}, skipping.")
            continue
        df_ground['Time'] = pd.to_datetime(df_ground['Time'])
        df_ground = df_ground[df_ground['Time'].dt.month.isin(months)]
        df_ground['Time'] = df_ground['Time'].dt.tz_localize(None)

        filename_list = [f for f in os.listdir(data_dir + "FY_L1_2021/") if f.endswith('.hdf5') and 'FY_L1_china_' in f]
        df_sat = pd.DataFrame(filename_list, columns=['filename'])
        df_sat['utc_dt'] = df_sat['filename'].apply(extract_fy4a_date)
        df_sat = df_sat.dropna(subset=['utc_dt'])
        #df_sat = df_sat[df_sat['utc_dt'].dt.minute.isin([0, 45])]
        #df_sat['Time_Rounded'] = df_sat['utc_dt'].dt.round('1h')
        matched_df = pd.merge(
            df_sat,
            df_ground,
            left_on='utc_dt',
            right_on='Time',
            how='inner' # strict, only keep exact matches
        )

        filtered_files = matched_df['filename'].tolist()

        print(f"Total number of files in daytime for Month {months}:",len(filtered_files))
        # latitude & longtitude ranges

        with h5py.File(data_dir + 'FY_L1_2021/' + filtered_files[0], 'r') as f:
            Lat, Lon = f['lat_4000'][:], f['lon_4000'][:]
            lon_s, lon_e = Lon[0], Lon[-1]
            lat_s, lat_e = Lat[0], Lat[-1] # 4 km resolution

            lon_interval = (lon_e - lon_s) / len(Lon)  # 1750  pixel for longitude
            lat_interval = (lat_e - lat_s) / len(Lat)  # 1000 pixel for latitude
            pixel = 11  # in 11*11 image size
            # estimate by the cloud height, assume 4km-9km, the parallax shift is around 1-4 pixels
        # crop central data
        extract_region(pixel, (site_name, coords), lon_s, lon_e, lat_s, lat_e, lon_interval, lat_interval, sky, filtered_files)
