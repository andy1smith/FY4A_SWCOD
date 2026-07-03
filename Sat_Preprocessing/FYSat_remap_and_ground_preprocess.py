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


ANGLE_CHANNELS = [
    'SatelliteAzimuth', 'SatelliteZenith', 'SunAzimuth',
    'SunGlintAngle', 'SunZenith', 'elevation'
]
FY4A_EXTRACT_CHANNELS = ANGLE_CHANNELS + ['Channel{:02d}'.format(i + 1) for i in range(7)]
BAD_FY4A_FILES = set()


def get_fy4a_scale_offset(dataset, channel):
    if 'scale_factor' in dataset.attrs and 'add_offset' in dataset.attrs:
        scale_factor = dataset.attrs['scale_factor']
        add_offset = dataset.attrs['add_offset']
    elif channel in ['Channel{:02d}'.format(i + 1) for i in range(7)]:
        scale_factor, add_offset = 0.0001, 0
    elif channel in ANGLE_CHANNELS:
        scale_factor, add_offset = 0.02, 0
    else:
        scale_factor, add_offset = 1, 0
    return scale_factor, add_offset


def calibrate_fy4a_values(values, dataset, channel):
    values = np.asarray(values, dtype=float)
    values = np.where(values == -9999, np.nan, values)
    scale_factor, add_offset = get_fy4a_scale_offset(dataset, channel)
    return values * scale_factor + add_offset

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
    # FY4A lat_4000 is treated as stored south-to-north here. Keep the
    # signed interval from Lat[0] to Lat[-1], unlike the GOES north-to-south
    # convention.
    central_lat_idx = int((coords['latitude'] - lat_s) / lat_int)
    # crop according to pixel size
    base_lon_start_idx = max(0, central_lon_idx - half_crop)
    base_lon_end_idx = min(1750, central_lon_idx + half_crop)  # lon, 1750 pixel
    base_lat_start_idx = max(0, central_lat_idx - half_crop)
    base_lat_end_idx = min(1000, central_lat_idx + half_crop)  # lat, 1000 pixel

    # process data
    channels = FY4A_EXTRACT_CHANNELS
    channel_data = {channel: [] for channel in channels}
    timestamps = []

    # site time matching
    mid_latitude = 35.0  # FY4A scan from north to south.
    if coords['latitude'] - mid_latitude > 0:
        nominal_time_id = 3  # site closer to start time
    else:
        nominal_time_id = 4  # site closer to end time
    # filtered_files can be either filenames or (filename, matched_ground_time)
    # pairs. The pair form keeps the CSV timestamps aligned to hourly ground GHI.
    for file_record in filtered_files:
        if isinstance(file_record, (tuple, list)):
            file_path, matched_time = file_record
            time = pd.to_datetime(matched_time).strftime('%Y-%m-%d %H:%M:%S')
        else:
            file_path = file_record
            timestamp = os.path.basename(file_path).split('_')[nominal_time_id]
            time = datetime.strptime(timestamp, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
        try:
            with h5py.File(data_dir + 'FY_L1_2021/' + file_path, 'r') as f:

                # check to ensure all channels exist
                if not all(channel in f for channel in channels):
                    logging.warning(f'Missing one or more channels in file {file_path}')
                    continue

                lon_start_idx, lon_end_idx = base_lon_start_idx, base_lon_end_idx
                lat_start_idx, lat_end_idx = base_lat_start_idx, base_lat_end_idx
                if sky == 'cloudy':
                    n_lat, n_lon = f['SunZenith'].shape
                    sun_zen = calibrate_fy4a_values(
                        f['SunZenith'][central_lat_idx, central_lon_idx],
                        f['SunZenith'],
                        'SunZenith'
                    ).item()
                    sun_az = calibrate_fy4a_values(
                        f['SunAzimuth'][central_lat_idx, central_lon_idx],
                        f['SunAzimuth'],
                        'SunAzimuth'
                    ).item()
                    sat_zen = calibrate_fy4a_values(
                        f['SatelliteZenith'][central_lat_idx, central_lon_idx],
                        f['SatelliteZenith'],
                        'SatelliteZenith'
                    ).item()
                    sat_az = calibrate_fy4a_values(
                        f['SatelliteAzimuth'][central_lat_idx, central_lon_idx],
                        f['SatelliteAzimuth'],
                        'SatelliteAzimuth'
                    ).item()

                    if np.all(np.isfinite([sun_zen, sun_az, sat_zen, sat_az])):
                        lon_start_idx, lon_end_idx, lat_start_idx, lat_end_idx = shadow_parallax_matching(
                            coords['longitude'], coords['latitude'],
                            sun_zen, sun_az, sat_zen, sat_az,
                            half_crop, lon_int, lat_int, lon_s, lat_s,
                            cth_km=2.0, n_lon=n_lon, n_lat=n_lat
                        )
                    else:
                        logging.warning(
                            f'Invalid FY4A geometry angles for {site_name} at {time}; '
                            'using the uncorrected site-centered crop.'
                        )

                file_channel_data = {}
                for channel in channels:
                    channel_slice = f[channel][lat_start_idx:lat_end_idx+1, lon_start_idx:lon_end_idx+1]
                    channel_crop = calibrate_fy4a_values(channel_slice, f[channel], channel).flatten()
                    file_channel_data[channel] = channel_crop

                timestamps.append(time)
                for channel, channel_crop in file_channel_data.items():
                    channel_data[channel].append(channel_crop)
        except OSError as exc:
            logging.warning(f'Could not read FY4A file {file_path} for {site_name} at {time}; skipping. {exc}')
            continue

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

def extract_fy4a_time(filename, lat_idx=1, total_rows=20):
    match = re.search(r'(\d{14})_(\d{14})', filename)
    if not match:
        return None

    start_str, end_str = match.groups()
    start_dt = datetime.strptime(start_str, '%Y%m%d%H%M%S')
    end_dt = datetime.strptime(end_str, '%Y%m%d%H%M%S')

    return pd.Series([start_dt, end_dt])

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
        (df['latitude'] <= 60) #&
        #(df['elve'] <= 500)  # you can adjust or remove altitude filter as needed
    ]
    # Groud meansurement data export the cloudy day / clear sky periods based on pvlib
    ground_preprocess = False
    #df = df [df['site']=='BJC']
    if ground_preprocess:
        preprocess_ground(df.copy(), data_dir)
    else:
        print('Skip ground clear, cloudy classification!')

    sites = df.set_index('site')[['longitude', 'latitude']].to_dict(orient='index')
    months = [4,5,6,7,8,9,10] #[0,1,2,3,4,5,6,7,8,9,10,11,12]  # June, July, August
    ground_dir = './Ground/preprocessed_GHI/'
    sky = 'cloudy'  # 'clear' or 'cloudy'
    filename_list = [
        f for f in os.listdir(data_dir + "FY_L1_2021/")
        if f.endswith('.hdf5') and 'FY_L1_china_' in f and f not in BAD_FY4A_FILES
    ]
    df_sat_all = pd.DataFrame(filename_list, columns=['filename'])
    df_sat_all[['utc_start','utc_end']] = df_sat_all['filename'].apply(extract_fy4a_time)
    df_sat_all = df_sat_all.dropna(subset=['utc_start'])
    # for i in range(17,len(sites)):
    for idx, (site_name, coords) in enumerate(sites.items()):
        mid_latitude= 35.0 # FY4A scan from north to south.
        if coords['latitude']-mid_latitude > 0:
            nominal_time = 'utc_start' # site closer to start time
        else:
            nominal_time = 'utc_end' # site closer to end time
        #site = dict(islice(sites.items(), i))
        #site_name = list(site.keys())[i-1]
        # if site_name != 'FQA':
        #     continue # ONLY PROCESS agricutural sites
        print("Processing site:", site_name)
        save_path = f'./cropped_FY2021_{sky}/{site_name}'
        expected_outputs = [
            os.path.join(save_path, '{}_{}.csv'.format(site_name, channel))
            for channel in FY4A_EXTRACT_CHANNELS
        ]
        if all(os.path.exists(path) for path in expected_outputs):
            print(f"All cloudy extraction files already exist for site {site_name}, skipping.")
            continue

        try:
            ground_path = ground_dir + '{}_{}.h5'.format(site_name,sky)
            df_ground = pd.read_hdf(ground_path, key='df')
        except FileNotFoundError:
            print(f"File not found for site {site_name}, skipping.")
            continue
        df_ground['Time'] = pd.to_datetime(df_ground['Time'])
        df_ground = df_ground[df_ground['Time'].dt.month.isin(months)]
        df_ground['Time'] = df_ground['Time'].dt.tz_localize(None)

        df_sat = df_sat_all.copy()
        if nominal_time == 'utc_end':
            # FY4A end timestamps are often 01:59:59 rather than 02:00:00.
            # Only map scan ends that are effectively at the next hourly GHI time.
            end_hour = df_sat['utc_end'].dt.ceil('1h')
            seconds_to_hour = (end_hour - df_sat['utc_end']).dt.total_seconds()
            df_sat['match_time'] = end_hour.where(seconds_to_hour <= 2)
        else:
            df_sat['match_time'] = df_sat['utc_start']
        df_sat = df_sat.dropna(subset=['match_time'])
        matched_df = pd.merge(
            df_sat,
            df_ground,
            left_on='match_time',
            right_on='Time',
            how='inner' # strict, only keep exact matches
        )

        filtered_files = list(zip(matched_df['filename'], matched_df['Time']))

        print(f"Total number of files in daytime for Month {months}:",len(filtered_files))
        if len(filtered_files) == 0:
            print(f"No matching satellite files found for site {site_name} in the specified months.")
            continue
        # latitude & longtitude ranges

        first_file = filtered_files[0][0] if isinstance(filtered_files[0], (tuple, list)) else filtered_files[0]
        with h5py.File(data_dir + 'FY_L1_2021/' + first_file, 'r') as f:
            Lat, Lon = f['lat_4000'][:], f['lon_4000'][:]
            lon_s, lon_e = Lon[0], Lon[-1] # from 70E to 140E, west to east
            lat_s, lat_e = Lat[0], Lat[-1] # from south to north   4 km resolution

            lon_interval = (lon_e - lon_s) / len(Lon)  # 1750  pixel for longitude
            lat_interval = (lat_e - lat_s) / len(Lat)  # 1000 pixel for latitude
            pixel = 11  # in 11*11 image size
            # estimate by the cloud height, assume 4km-9km, the parallax shift is around 1-4 pixels
        # crop central data
        extract_region(pixel, (site_name, coords), lon_s, lon_e, lat_s, lat_e,
                       lon_interval, lat_interval, sky, filtered_files)
