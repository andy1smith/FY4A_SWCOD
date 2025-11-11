import h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool
import glob
import os
from itertools import islice
from datetime import datetime
import matplotlib.pyplot as plt
import logging


def process_site(args):
    file_paths, site_name, coords, lon_s, lon_e, lat_s, lat_e, lon_int, lat_int, pixel, save_path = args

    # location & index for site
    central_lon_idx = int((coords['longitude'] - lon_s) / lon_int)
    central_lat_idx = int((lat_e - coords['latitude']) / lat_int)

    # crop according to pixel size
    half_crop = pixel // 2
    lon_start_idx = max(0, central_lon_idx - half_crop)
    lon_end_idx = min(1750, central_lon_idx + half_crop)  # lon, 1750 pixel
    lat_start_idx = max(0, central_lat_idx - half_crop)
    lat_end_idx = min(1000, central_lat_idx + half_crop)  # lat, 1000 pixel


    # process data
    channels = ['Channel{:02d}'.format(i + 1) for i in range(14)] + ['SatelliteAzimuth', 'SatelliteZenith',
                'SunAzimuth', 'SunGlintAngle', 'SunZenith', 'elevation']
    channel_data = {channel: [] for channel in channels}
    timestamps = []


    for file_path in file_paths[:2]:
        with h5py.File(file_path, 'r') as f:

            # check to ensure all channels exist
            if not all(channel in f for channel in channels):
                logging.warning(f'Missing one or more channels in file {file_path}')
                continue

            timestamp = os.path.basename(file_path).split('_')[3]
            time = datetime.strptime(timestamp, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            timestamps.append(time)

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
                    elif channel in ['Channel{:02d}'.format(i + 1) for i in range(7, 14)]:
                        scale_factor, add_offset = 0.01, 273
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


def extract_region(pixel, sites, lon_s, lon_e, lat_s, lat_e, lon_int, lat_int):
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


    file_paths = glob.glob('G:/FY_L1_2021/FY_L1_china_2021*.hdf5')
    # file_paths = glob.glob('F:/FY_L1_2021/FY_L1_china_20211231233000_20211231233417.hdf5')


    # save path
    scenarios = []
    for site_name, coords in sites.items():
        save_path = f'F:/cropped_FY2021/{site_name}'
        os.makedirs(save_path, exist_ok=True)

        scenarios.append([file_paths, site_name, coords, lon_s, lon_e, lat_s, lat_e, lon_int,
                      lat_int, pixel, save_path])

    # process data in parallel
    pool = Pool()
    pool.map(process_site, scenarios)
    pool.close()



if __name__ == '__main__':
    # Setup basic configuration for logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # location information
    df = pd.read_csv('G:/CERN_info.csv')
    sites = df.set_index('site')[['longitude', 'latitude']].to_dict(orient='index')
    site = dict(islice(sites.items(), 1))  # slice the first site
    # site = dict(list(sites.items())[-2:])  # slice the last two sites

    # latitude & longtitude ranges
    with h5py.File('G:/FY_L1_2021/FY_L1_china_20211231233000_20211231233417.hdf5', 'r') as f:
        Lat, Lon = f['lat_4000'][:], f['lon_4000'][:]
        lon_s, lon_e = Lon[0], Lon[-1]
        lat_s, lat_e = Lat[0], Lat[-1]
        lon_interval = (lon_e - lon_s) / 1750  # 1750  pixel for longitude
        lat_interval = (lat_e - lat_s) / 1000  # 1000 pixel for latitude

    pixel = 11  # in 11*11 image size


    # crop central data
    extract_region(pixel, site, lon_s, lon_e, lat_s, lat_e, lon_interval, lat_interval)
