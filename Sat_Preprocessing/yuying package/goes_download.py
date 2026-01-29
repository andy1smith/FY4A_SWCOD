import GOES
import os
import glob
from multiprocessing import Pool
from netCDF4 import Dataset
from remap import remap
from itertools import product

def extract_region(path, YEAR, DOY, data, name, dlat=0.1, dlon=0.1):
    '''
    Extract image region centered around site at (lat, lon).
    '''

    # choose the image resolution [km/pixel]
    resolution = 2

    # Calculate the image extent required for the reprojection
    nc = Dataset(path, 'r')
    H = nc.variables['goes_imager_projection'].perspective_point_height
    x1 = nc.variables['x_image_bounds'][0] * H
    x2 = nc.variables['x_image_bounds'][1] * H
    y1 = nc.variables['y_image_bounds'][1] * H
    y2 = nc.variables['y_image_bounds'][0] * H
    #print x1, x2, x2 - x1
    #print y1, y2, y2 - y1

    # Get the timestamp info
    ts_created = nc.date_created
    ts_start = nc.time_coverage_start
    ts_end = nc.time_coverage_end
    ts_str = '{},{},{}'.format(ts_created, ts_start, ts_end)
    dataset_name = nc.dataset_name


    # output directory for cropped files
    output_dir = "./data/SGP/cropped_{}_{}/".format(name, YEAR)
    if not os.path.isdir(output_dir):
        print("Creating directory:", output_dir)
        os.mkdir(output_dir)

    # SURFRAD & SGP stations
    sites = [
        # {'name': 'PSU', 'lat': 40.72012, 'lon': -77.93085},
        # {'name': 'BON', 'lat': 40.05192, 'lon': -88.37309},
        # {'name': 'GWN', 'lat': 34.2547, 'lon': -89.8729},
        # {'name': 'SXF', 'lat': 43.73403, 'lon': -96.62328},
        # {'name': 'TBL', 'lat': 40.12498, 'lon': -105.23680},
        # {'name': 'DRA', 'lat': 36.62373, 'lon': -116.01947},
        # {'name': 'FPK', 'lat': 48.30783, 'lon': -105.10170},
        {'name': 'SGP', 'lat': 36.607322, 'lon': -97.487643},
    ]

    # iterate through sites
    for site in sites:

        # Choose the visualization extent (min lon, min lat, max lon, max lat)
        # all pixels
        #extent = [min_lon, min_lat, max_lon, max_lat]  # all pixels

        # square area (centered at given lat-lon)
        lon_min, lon_max = site['lon'] - dlon, site['lon'] + dlon
        lat_min, lat_max = site['lat'] - dlat, site['lat'] + dlat
        extent = [lon_min, lat_min, lon_max, lat_max]

        # open file to save data
        # if data[:-1] in ['ABI-L1b-Rad', 'ABI-L2-CMIP']:
        if name == 'Rad':
            channel = dataset_name.split('-')[-1].split('_')[0][-3:]
            f = open(
                os.path.join(
                    output_dir,
                    '{:04d}_{:03d}_{}_{}_{}.csv'.format(YEAR, DOY, site['name'], channel, name)
                ),
                'a'    # append pre-existing files
            )
        else:
            f = open(
                os.path.join(
                    output_dir,
                    '{:04d}_{:03d}_{}_{}.csv'.format(YEAR, DOY, site['name'], name)
                ),
                'a'  # append pre-existing files
            )

        # call the reprojection funcion
        grid = remap(path, name, extent, resolution, x1, y1, x2, y2)
        data = grid.ReadAsArray()

        # format data as string
        data_str = ','.join(['{:.6f}'.format(i) for i in data.reshape((-1))])

        # save to file
        if f is not None:
            if name == 'Rad':
                f.write('{},{},{}\n'.format(ts_str, channel, data_str))
            else:
                f.write('{},{}\n'.format(ts_str, data_str))

        # cleanup: csv
        f.close()

    # cleanup: netCDF
    nc.close()
    nc = None


def process_parallel(DOY):
    """Process data in parallel (by DOY).

    Parameters
    ----------
    DOY : int
        Day of year (e.g. 10 = 10th day of the year).

    """

    YEAR = 2020
    # data, name = 'ABI-L1b-RadC', 'Rad'  # radiance data
    # data, name = 'ABI-L2-ACHAC', 'HT'  # Cloud Top Height
    data, name = 'ABI-L2-CODC', 'COD'  # Cloud Optical Depth
    # data, name = 'ABI-L2-CTPC', 'PRES'  # Cloud Top Pressure YEAR = 2023


    # crop size [degrees]
    dlat, dlon = 0.1, 0.1  # (11, 11) pixels
    #dlat, dlon = 0.5, 0.5  # (55, 55) pixels
    #dlat, dlon = 1.0, 1.0  # (111, 111) pixels
    #dlat, dlon = 2.0, 2.0  # (222, 222) pixels
    #dlat, dlon = 4.0, 4.0  # (445, 445) pixels
    #dlat, dlon = 8.0, 8.0  # (890, 890) pixels

    # list of all filenames
    filenames = glob.glob('./data/SGP/{}_{}/*{:04d}{:03d}*.nc'.format(data, YEAR, YEAR, DOY))
    print("{:04d}, {:03d} ({} files): start ...".format(YEAR, DOY, len(filenames)))

    # extract regions and save to .csv files
    for filename in filenames:
        try:
            extract_region(filename, YEAR, DOY, data, name,  dlat=dlat, dlon=dlon)
        except:
            print("{}: ... FAILED".format(filename))

    print("{:04d}, {:03d} ({} files): DONE".format(YEAR, DOY, len(filenames)))


# download data
def download(data, YEAR):
    save_path = './data/SGP/{}_{}/'.format(data, YEAR)
    if not os.path.isdir(save_path):
        print('Creating directory:', save_path)
        os.mkdir(save_path)

    GOES.download('goes16', data,
                          DateTimeIni='{}0101-000000'.format(YEAR), DateTimeFin='{}0102-000000'.format(YEAR+1),
                          channel=channels, rename_fmt='%Y%j%H%M', path_out=save_path)


if __name__ == '__main__':
    channels = ['{:02d}'.format(i) for i in range(7, 16 + 1)]

    # download data
    # YEAR = 2020
    # data, name = 'ABI-L1b-RadC', 'Rad'  # radiance data
    # data, name = 'ABI-L2-ACHAC', 'HT'  # Cloud Top Height
    # data, name = 'ABI-L2-CODC', 'COD'  # Cloud Optical Depth
    # data, name = 'ABI-L2-CTPC', 'PRES'  # Cloud Top Pressure
    # download(data=data, YEAR=YEAR)

    # crop data
    DOYs = range(1, 1 + 1)  # day of year (DOY) (e.g. 2023/001 = 1st day of 2023)
    pool = Pool()
    pool.map(process_parallel, DOYs)
    pool.close()



