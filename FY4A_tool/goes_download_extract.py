#import GOES
import os, sys
import glob
from multiprocessing import Pool
from netCDF4 import Dataset

from itertools import product
from datetime import datetime
import numpy as np
import pandas as pd
from Site_prepare_whole import read_csv
from fun_goes_process import *


def is_water_phase(filename, name, extent, resolution, x1, y1, x2, y2):
    phase_filename = filename.replace('ABI-L2-CODC', 'ABI-L2-ACTPC')
    # find the phase file according to the COD
    # Step 1: Extract the part before 'c2019'
    index = phase_filename.find('c2019')
    start_part = phase_filename[:index]
    # Step 2: Extract the directory path
    dir_path = os.path.dirname(phase_filename) + '/'
    # Step 3: Search for .nc files that start with start_part
    matching_files = glob.glob(os.path.join(dir_path, '*.nc'))
    matching_files_with_start = [f for f in matching_files if f.startswith(start_part)]
    if len(matching_files_with_start) == 0:
        print('No matching phase files found for:', filename)
        return False
    elif len(matching_files_with_start) > 1:
        print('Multiple matching phase files found for:', filename)
        return False
    grid = remap(matching_files_with_start[0], 'Phase', extent, resolution, x1, y1, x2, y2)
    data = grid
    #print(grid)
    #data = grid.ReadAsArray()
    median_value = np.median(data.flatten())
    if median_value == 0 or median_value == 1:
        #print('All pixels are in warm or super cold water phase!')
        return True
    else:
        #print('no water phase')
        return False

def get_Rad_file(filename,channel,hour,phase_filter, COD_filter):
    filename = filename.replace(f'ABI-L2-CODC/{hour}/', f'ABI-L1b-RadC/{hour}/{channel}/')
    filename = filename.replace('ABI-L2-CODC', 'ABI-L1b-RadC')
    phase_filename = filename.replace('-M6_', f'-M6{channel}_')
    # find the phase file according to the COD
    # Step 1: Extract the part before 'c2019'
    index = phase_filename.find('e2019')
    start_part = phase_filename[:index]
    # Step 2: Extract the directory path
    dir_path = os.path.dirname(phase_filename) + '/'
    # Step 3: Search for .nc files that start with start_part
    matching_files = glob.glob(os.path.join(dir_path, '*.nc'))
    matching_files_with_start = [f for f in matching_files if f.startswith(start_part)]
    if len(matching_files_with_start) == 0:
        print('No matching phase files found for:', filename)
        return False
    elif len(matching_files_with_start) > 1:
        print('Multiple matching phase files found for:', filename)
        return False
    return matching_files_with_start[0]

def match_Rad_file(filename,channel):
    rad_file = filename.replace('C01', channel)
    # find the phase file according to the COD
    # Step 1: Extract the part before 'c2019'
    index = rad_file.find('e2019')
    start_part = rad_file[:index]
    # Step 2: Extract the directory path
    dir_path = os.path.dirname(rad_file) + '/'
    # Step 3: Search for .nc files that start with start_part
    matching_files = glob.glob(os.path.join(dir_path, '*.nc'))
    matching_files_with_start = [f for f in matching_files if f.startswith(start_part)]
    if len(matching_files_with_start) == 0:
        print('No matching phase files found for:', filename)
        return False
    elif len(matching_files_with_start) > 1:
        print('Multiple matching phase files found for:', filename)
        return False
    return matching_files_with_start[0]

def get_CMIP_file(rad_file):
    CMIPath = rad_file.replace('ABI-L1b-RadC/20/C02/', 'ABI-L2-CMIPC/20/')
    CMIPath = CMIPath.replace('ABI-L1b-RadC', 'ABI-L2-CMIPC')

    # find the phase file according to the COD
    # Step 1: Extract the part before 'c2019'
    index = CMIPath.find('c2019')
    start_part = CMIPath[:index]
    # Step 2: Extract the directory path
    dir_path = os.path.dirname(CMIPath) + '/'
    # Step 3: Search for .nc files that start with start_part
    matching_files = glob.glob(os.path.join(dir_path, '*.nc'))
    matching_files_with_start = [f for f in matching_files if f.startswith(start_part)]
    if len(matching_files_with_start) == 0:
        print('No matching phase files found for:', rad_file)
        return False
    elif len(matching_files_with_start) > 1:
        print('Multiple matching phase files found for:', rad_file)
        return False
    return matching_files_with_start[0]

def get_angles(path, lons, lats):
    # validate to solar zenith with the site's measurement, difference within 1.5 degree.
    # L1b Rad data
    nc = Dataset(path, mode='r')

    t_ = nc.variables['t'][:]
    utc_time = convert_to_utc(t_.item(), lats[0], lons[0])

    lon_0 = nc.variables['goes_imager_projection'].longitude_of_projection_origin
    lat_0 = nc.variables['goes_imager_projection'].latitude_of_projection_origin
    # important notice, official guide mark all grids in (y,x)

    sun_zenith_angles, sun_azimuth_angles = sun_zenith_angle(utc_time, lons, lats)

    local_zenith_angles = calculate_local_zenith_angle(lon_0, lat_0,
                                                       lons, lats, input_type='deg')

    local_azimuth_angles, _ = calculate_local_azimuth(lons,lats, input_type='deg')

    relative_azimuth_angles = calculate_relative_azimuth_angle(local_azimuth_angles,
                                                               sun_azimuth_angles,
                                                               input_type='deg')

    nc.close()
    return sun_zenith_angles, local_zenith_angles, relative_azimuth_angles

def extract_region(filename, dir, date, bucket, product_name, dlat=0.1, dlon=0.1,
                   outputype = 'Rad', COD_filter=True,phase_filter=True):
    '''
    Extract image region centered around site at (lat, lon).
    '''

    # choose the image resolution [km/pixel]
    resolution = 2
    YEAR, month, day, _ = date
    if product_name == 'COD':
        hour = filename.split('/')[-2]  # e.g. '19'
    else:
        hour = filename.split('/')[-3]
    # hour = filename.split('/')[-3]

    nc = Dataset(filename, 'r')
    H = nc.variables['goes_imager_projection'].perspective_point_height
    x1 = nc.variables['x_image_bounds'][0] * H
    x2 = nc.variables['x_image_bounds'][1] * H
    y1 = nc.variables['y_image_bounds'][1] * H
    y2 = nc.variables['y_image_bounds'][0] * H
    #print(x1,x2,y1,y2)

    if product_name == 'COD': # daytime
        v = nc.variables['nighttime_cloud_pixels'][:]
        if v != 0:
            print('nighttime_cloud_pixels=', v)
            v = nc.variables['daytime_cloud_pixels'][:]
            print('daytime_cloud_pixels=', v)
            return None
        # else:
        #     print('All pixels are in daytime!')

    # Get the timestamp info
    ts_created = nc.date_created
    ts_start = nc.time_coverage_start
    ts_end = nc.time_coverage_end
    ts_str = '{},{},{}'.format(ts_created, ts_start, ts_end)
    dataset_name = nc.dataset_name

    # output directory for cropped files
    dir = './'
    output_dir = dir+f"{bucket}_cropped_{product_name}_{YEAR}/"
    if not os.path.isdir(output_dir):
        print("Creating directory:", output_dir)
        os.mkdir(output_dir)

    # SURFRAD & SGP stations
    sites = [
       {'name': 'PSU', 'lat': 40.72012, 'lon': -77.93085},
       {'name': 'BON', 'lat': 40.05192, 'lon': -88.37309},
        {'name': 'GWN', 'lat': 34.2547, 'lon': -89.8729},
        {'name': 'SXF', 'lat': 43.73403, 'lon': -96.62328},
        {'name': 'TBL', 'lat': 40.12498, 'lon': -105.23680},
       {'name': 'DRA', 'lat': 36.62373, 'lon': -116.01947},
        {'name': 'FPK', 'lat': 48.30783, 'lon': -105.10170},
       {'name': 'SGP', 'lat': 36.607322, 'lon': -97.487643},
    ]

    # iterate through sites
    for site in sites:
        # Choose the visualization extent (min lon, min lat, max lon, max lat)
        # all pixels
        # extent = [min_lon, min_lat, max_lon, max_lat]  # all pixels
        # square area (centered at given lat-lon)
        lon_min, lon_max = site['lon'] - dlon, site['lon'] + dlon
        lat_min, lat_max = site['lat'] - dlat, site['lat'] + dlat
        extent = [lon_min, lat_min, lon_max, lat_max]

        phase = 'whole'
        if phase_filter:
            phase = 'water'
        COD_boundary = 0
        if COD_filter:
            COD_boundary = 10
        # open file to save data
        # if data[:-1] in ['ABI-L1b-Rad', 'ABI-L2-CMIP']:
        if product_name == 'Rad':
            channel = dataset_name.split('-')[-1].split('_')[0][-3:]
            allchannel_inafile = True
            if allchannel_inafile == True:
                f = open(
                    os.path.join(
                        output_dir,
                        f'{int(YEAR)}{int(month)}{int(day)}_{site["name"]}_{product_name}.csv'), 'a'  # append pre-existing files
                )
            else:
                f = open(
                    os.path.join(
                        output_dir,
                        f'{int(YEAR)}_{site["name"]}_{channel}_{product_name}.csv'),'a'    # append pre-existing files
                )
        else:
            f = open(
                os.path.join(output_dir,
                    f'{int(YEAR)}_{site["name"]}_{product_name}_{phase}_June.csv'),'a'  # append pre-existing files
            )
            # print(output_dir,
            #         f'{int(YEAR)}_{site["name"]}_{product_name}_{phase}.csv')

        # call the reprojection function
        # 1. extract cloud phase = water phase
        if phase_filter:
            water_phase = is_water_phase(filename, 'Phase', extent, resolution, x1, y1, x2, y2)
            if not water_phase:
                # print("Not water phase, return None")
                return None
            # print('Water phase detected!')
        # 2. extract cloud COD != 0
        grid = remap(filename, product_name, extent, resolution, x1, y1, x2, y2)
        data = grid
        #data = grid.ReadAsArray()
        median_value = np.median(data.flatten())
        if COD_filter:
            if median_value != 0:
                COD = median_value
                print(COD)
            else:
                f.close()
                nc.close()
                return None
            # if 0 < median_value and median_value < COD_boundary:
            #     # print('All pixels are in COD > 10!')
            #     COD = median_value
            # else:
            #     f.close()
            #     nc.close()
            #     return None
        # 3. read Radiance data
        if COD_filter:
            channels = ['C0{}'.format(i) for i in range(1, 6 + 1)]
            center_rad = []
            for channel in channels:
                rad_file = get_Rad_file(filename,channel,hour,phase_filter, COD_filter)
                if channel == 'C01':
                    sun_zen, local_zen, rela_azi = get_angles(filename, [site['lon']], [site['lat']])
                if rad_file == False:
                    f.close()
                    nc.close()
                    return None
                grid = remap(rad_file, 'Rad', extent, resolution, x1, y1, x2, y2)
                data = grid #grid.ReadAsArray()
                median_value = np.median(data.flatten()) # 'BON'
                center_rad.append(median_value)
            if outputype == 'reflectance':
                CMIPath = get_CMIP_file(rad_file)
                ref_Rad = Conver_ref(median_value, 'C02', CMIPath)
                # format data as stringwhat s
                data_str = ','.join(['{:.6f}'.format(i) for i in ref_Rad.reshape((-1))])
            else:
                Rad = goes_lam(center_rad, "../GOES_data/")
                data_str = ','.join(['{:.6f}'.format(i) for i in Rad])
            # save to file
            if f is not None:
                if f.tell() == 0:
                    f.write('ts_created,ts_start,ts_end,C01,C02,C03,C04,C05,C06,COD,Sun_Zen,local_Zen,rela_azi\n')
                f.write('{},{},{},{},{},{}\n'.format(ts_str, data_str, COD, sun_zen[0], local_zen[0], rela_azi[0]))
                    # else:
                    #     f.write('{},{}\n'.format(ts_str, data_str))
                # cleanup: csv
            print('Extracted data for site:', site['name'])
            f.close()
        else:
            # save to file
            if f is not None:
                if f.tell() == 0:
                    f.write('ts_created,ts_start,ts_end,Radiance\n')
                f.write('{},{}\n'.format(ts_str, str(median_value)))
            f.close()
        print('Extracted data for site:', site['name'])
    nc.close()
    return None

def extract_region_Rad(filename, dir, date, bucket, product_name, dlat=0.1, dlon=0.1,
                   ):
    '''
    Extract image region centered around site at (lat, lon).
    '''
    # choose the image resolution [km/pixel]
    resolution = 2
    YEAR, month, day, _ = date
    # hour = filename.split('/')[-2]  # e.g. '19'

    nc = Dataset(filename, 'r')
    H = nc.variables['goes_imager_projection'].perspective_point_height
    x1 = nc.variables['x_image_bounds'][0] * H
    x2 = nc.variables['x_image_bounds'][1] * H
    y1 = nc.variables['y_image_bounds'][1] * H
    y2 = nc.variables['y_image_bounds'][0] * H

    # Get the timestamp info
    ts_created = nc.date_created
    ts_start = nc.time_coverage_start
    ts_end = nc.time_coverage_end
    ts_str = '{},{},{}'.format(ts_created, ts_start, ts_end)

    # output directory for cropped files
    dir = './'
    output_dir = dir+f"{bucket}_cropped_{product_name}_{YEAR}/"
    if not os.path.isdir(output_dir):
        print("Creating directory:", output_dir)
        os.mkdir(output_dir)

    # SURFRAD & SGP stations
    sites = [
        {'name': 'PSU', 'lat': 40.72012, 'lon': -77.93085},
       {'name': 'BON', 'lat': 40.05192, 'lon': -88.37309},
        {'name': 'GWN', 'lat': 34.2547, 'lon': -89.8729},
        {'name': 'SXF', 'lat': 43.73403, 'lon': -96.62328},
        {'name': 'TBL', 'lat': 40.12498, 'lon': -105.23680},
         {'name': 'DRA', 'lat': 36.62373, 'lon': -116.01947},
        {'name': 'FPK', 'lat': 48.30783, 'lon': -105.10170},
       {'name': 'SGP', 'lat': 36.607322, 'lon': -97.487643},
    ]
    channels = ['C0{}'.format(i) for i in range(1, 6 + 1)]
    # iterate through sites
    for site in sites:
        lon_min, lon_max = site['lon'] - dlon, site['lon'] + dlon
        lat_min, lat_max = site['lat'] - dlat, site['lat'] + dlat
        extent = [lon_min, lat_min, lon_max, lat_max]

        if product_name == 'Rad':
            f = open(
                os.path.join(output_dir,
                    f'{int(YEAR)}{month}{day[0]}_{site["name"]}_{product_name}.csv'), 'a') # append pre-existing files

        center_rad = []
        for channel in channels:
            if channel == 'C01':
                rad_file = filename
                sun_zen, local_zen, rela_azi = get_angles(filename, [site['lon']], [site['lat']])
            else:
                rad_file = match_Rad_file(filename, channel)
            if rad_file == False:
                f.close()
                nc.close()
                return None
            grid = remap(rad_file, 'Rad', extent, resolution, x1, y1, x2, y2)
            data = grid
            #data = grid.ReadAsArray()
            median_value = np.median(data.flatten())
            center_rad.append(median_value)
        Rad = goes_lam(center_rad, "../GOES_data/")
        data_str = ','.join(['{:.6f}'.format(i) for i in Rad])
        # save to file
        if f is not None:
            if f.tell() == 0:
                f.write('ts_created,ts_start,ts_end,C01,C02,C03,C04,C05,C06,Sun_Zen,local_Zen,rela_azi\n')
            f.write('{},{},{},{},{}\n'.format(ts_str, data_str, sun_zen[0], local_zen[0], rela_azi[0]))
        f.close()
    nc.close()
    return None

def goespy_download(date, bucket, product, spectral):
    '''
    Download data from GOES satellite using goespy package.
    '''
    if spectral == 'SW':
        channels = ['C01','C02', 'C03','C04', 'C05', 'C06']
    elif spectral == 'LW':
        channels = ['C{:02d}'.format(i) for i in range(7, 16 + 1)]
    elif spectral == '':
        channels = ''
    else:
        channels = spectral # ['C{:02d}'.format(i) for i in range(1, 16 + 1)]

    year, month, day, hour = date

    save_path = "/home/dengnan/data/goes/"
    #'/Volumes/DN1T_SSD/data/'
    if not os.path.isdir(save_path):
        print('Creating directory:', save_path)
        os.mkdir(save_path)
    ABI_Downloader(save_path, bucket, year, month, day, hour, product, channels)

def extractRad_via_COD_phase(date, bucket, product, name, dir='../GOES_data/',
                             COD_filter=True, phase_filter = True):
    year, months, days, hours = date

    # days =[str(day).zfill(2) for day in range(1, 31)] # ['09']#
    # GOES-16 Cloud Optical Depth (COD) data
    if sys.platform == 'darwin':
        dir0 = '/Volumes/DN1T_SSD/data/'
    else:
        dir0 = '/home/dengnan/data/goes/'
        #'/mnt/dengnan/data/'

    goes_COD_files = []
    for month in months:
        for day in days:
            for hour in hours:
                if COD_filter:
                    goes_COD_file = glob.glob('{dir}{bucket}/{year}/{month}/{day}/{product}/{hour}/*'.format(
                                        dir=dir0,
                                        bucket=bucket[5:],
                                        year=year,
                                        month=month,
                                        day=day,
                                        product=product,
                                        hour=hour,)  # note that there's no channel directory
                                        )
                else:
                # if COD_filter:
                    goes_COD_file = glob.glob('{dir}{bucket}/{year}/{month}/{day}/{product}/{hour}/C01/*'.format(
                    dir=dir0,
                    bucket=bucket[5:],
                    year=year,
                    month=month,
                    day=day,
                    product=product,
                    hour=hour, )  # note that there's no channel directory
                )
                # goes_COD_files.append(goes_COD_file)
                goes_COD_files.extend(goes_COD_file)
    print('number of fiels:',len(goes_COD_files))
    dlat, dlon = 0.1, 0.1  # (11, 11) pixels, crop size [degrees]
    for filename in goes_COD_files:
        # try:
        if product == 'ABI-L2-CODC':
            extract_region(filename, dir, date, product, name, dlat=dlat, dlon=dlon,
                           outputype = 'Rad', COD_filter=COD_filter, phase_filter=phase_filter)
        #     if product == 'ABI-L1b-RadC':
        #         extract_region_Rad(filename, dir, date, product, name, dlat=dlat, dlon=dlon)
        # except:
        #     print("{}: ... FAILED".format(filename))

def print_dayhour(year, month, day, site='BON'):
    '''
    Print the date and time in a readable format.
    '''
    file_path = './SURFRAD/'

    filenames = glob.glob(file_path + '{}*{}*.csv'.format(str(year), site.lower()))
    df = pd.read_csv(filenames[0], header=0)
    df.columns = ['Time', 'Site_zen', 'Site_dsw', 'direct_n', 'diffuse', 'dw_ir',
                  'temp', 'rh', 'windspd', 'pressure']
    df = df[['Time', 'Site_zen', 'Site_dsw', 'direct_n', 'diffuse', 'dw_ir',
             'temp', 'rh']]
    df['Time'] = pd.to_datetime(df['Time'])


    df_target = df[df['Time'].dt.date == datetime.date(int(year), int(month), int(day))]
    dayhour = df_target[df_target['Site_dsw'] > 0]['Time'].dt.hour.unique()
    print(f'For {site} on {year} {month} {day}')
    print('The day hours are:', dayhour)
    return dayhour


def print_clearsky(site, tz='UTC'):
    import pvlib
    file_path = './SURFRAD/'
    lat, lon, alt = 40.05192, -88.37309, 213
    filenames = glob.glob(file_path + '{}*{}*.csv'.format(str(year), site.lower()))
    df = pd.read_csv(filenames[0], header=0)
    df.columns = ['Time', 'Site_zen', 'Site_dsw', 'direct_n', 'diffuse', 'dw_ir',
                  'temp', 'rh', 'windspd', 'pressure']
    df = df[['Time', 'Site_zen', 'Site_dsw', 'direct_n', 'diffuse', 'dw_ir',
             'temp', 'rh']]
    df['Time'] = pd.to_datetime(df['Time'])
    # day=df[df['Time'].dt.date == datetime.date(2019, 7, 13)]
    # plt.plot(day['Time'].dt.hour, day['Site_dsw'])
    # plt.xticks(np.arange(0,24+6,6))  # UTC
    # plt.tickslabel(np.arange(0,24,6))
    # plt.show()


    # 1. Prepare your DataFrame (ensure 'Time' is datetime):
    df = df.set_index('Time')
    # 2. Set your location (example: latitude, longitude, altitude)
    location = pvlib.location.Location(lat, lon, tz, alt)
    # 3. Generate modeled clear sky irradiance for your timestamps
    clearsky = location.get_clearsky(df.index)
    clearsky_dsw = clearsky['ghi'].resample('5min').mean()  # Use 'ghi' for horizontal irradiance

    # 4. Compute clearness index
    df = df.resample('5min').mean()  # Resample to 5-minute intervals
    df['clearness_index'] = df['Site_dsw'] / clearsky_dsw

    # 5. Flag clear sky periods (threshold can be 0.9 or 0.95)
    df['is_clear'] = df['clearness_index'] > 0.9

    # 6. Aggregate by day to find clear sky days (e.g., >90% of daylight points are clear)
    df['date'] = df.index.date
    clear_day_fraction = df.groupby('date')['is_clear'].mean()
    clear_sky_days = clear_day_fraction[clear_day_fraction > 0.6].index
    print(clear_sky_days)

    start = datetime.datetime(2019, 7, 13, 6, 0)
    end = datetime.datetime(2019, 7, 14, 6, 0)
    c1 = df[(df.index >= start) & (df.index < end)]

    pv_c1 = location.get_clearsky(c1.index)
    pv_c1.plot()
    c1['Site_dsw'].plot(legend='Site DSW', color='red')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #product, product_name, spectral = 'ABI-L1b-RadC', 'Rad', 'SW' # radiance data
    # # data, name = 'ABI-L2-ACHAC', 'HT'  # Cloud Top Height
    product, product_name, spectral = 'ABI-L2-CODC', 'COD', ''  # Cloud Optical Depth # 2km
    # product, name, spectral = 'ABI-L2-CMIPC', 'CMIP', ''  # Cloud Optical Depth # 2km
    #product, name, spectral = 'ABI-L2-ACTPC', 'Phase', ''  # Cloud Top Phase # 2km
    # product, name = 'ABI-L2-CMIPF', 'Cloud_Microphysics'  # Cloud Microphysics
    # # data, name = 'ABI-L2-CTPC', 'PRES'  # Cloud Top Pressure

    # # download data
    year = '2019'
    months = ['06'] #['03','04','05','07','08','09']
    days = [str(day).zfill(2) for day in range(1, 31)]  # 11,12,24,25,
    #hours = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    bucket = 'noaa-goes16'

    #print_clearsky('BON')
    hours = print_dayhour(year, months[0], days[0], site='BON')  # print the day hours for BON site
    hours = [f"{int(h):02}" for h in hours]
    #hours = ['18']
    
    extract = True #True
    download = False #False

    file_dir = '../GOES_data/'

    if download == True:
        from goespy.Downloader import ABI_Downloader
        for month in months:
            for day in days:
                for hour in hours:
                    date = [year, month, day, hour]
                    goespy_download(date, bucket, product, spectral)

    # crop data, extract region
    if extract == True:
        from remap import remap
        date = [year, months, days, hours]
        extractRad_via_COD_phase(date, bucket, product, product_name, file_dir,
                                     COD_filter=True,phase_filter=True)
    print('\nData extracted done')


