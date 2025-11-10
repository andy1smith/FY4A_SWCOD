'''
Test code for working with the GOES netCDF files using gdal. The main thing to
figure out is how to grab portions of the image corresponding to the latitude
and longitude of the target site (e.g. given the lat-lon of the site and
desired image size, load the data, crop out the selected region, and export in
a usable format).
'''

import os
import glob
from multiprocessing import Pool
from netCDF4 import Dataset
from remap import remap


def extract_region(path, YEAR, DOY, dlat=0.1, dlon=0.1):
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
    channel = dataset_name.split('-')[-1].split('_')[0][-3:]

    # output directory for cropped files
    output_dir = "cropped"
    if not os.path.isdir(output_dir):
        print("Creating directory:", output_dir)

    # SURFRAD stations
    sites = [
        {'name': 'PSU', 'lat': 40.72012, 'lon': -77.93085},
        {'name': 'BON', 'lat': 40.05192, 'lon': -88.37309},
        {'name': 'GWN', 'lat': 34.2547, 'lon': -89.8729},
        {'name': 'SXF', 'lat': 43.73403, 'lon': -96.62328},
        {'name': 'TBL', 'lat': 40.12498, 'lon': -105.23680},
        {'name': 'DRA', 'lat': 36.62373, 'lon': -116.01947},
        {'name': 'FPK', 'lat': 48.30783, 'lon': -105.10170},
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
        f = open(
            os.path.join(
                output_dir,
                '{:04d}_{:03d}_{}_RadC.csv'.format(YEAR, DOY, site['name'], channel)
            ),
            'a'    # append pre-existing files
        )

        # call the reprojection funcion
        grid = remap(path, extent, resolution, x1, y1, x2, y2)
        data = grid.ReadAsArray()

        # format data as string
        data_str = ','.join(['{:.6f}'.format(i) for i in data.reshape((-1))])

        # save to file
        if f is not None:
            f.write('{},{},{}\n'.format(ts_str, channel, data_str))

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

    # crop size [degrees]
    dlat, dlon = 0.1, 0.1  # (11, 11) pixels
    #dlat, dlon = 0.5, 0.5  # (55, 55) pixels
    #dlat, dlon = 1.0, 1.0  # (111, 111) pixels
    #dlat, dlon = 2.0, 2.0  # (222, 222) pixels
    #dlat, dlon = 4.0, 4.0  # (445, 445) pixels
    #dlat, dlon = 8.0, 8.0  # (890, 890) pixels

    # list of all filenames
    YEAR = 2021
    #filenames = glob.glob(os.path.join('ABI-L1b-RadC', '{:04d}'.format(YEAR), '{:03d}'.format(DOY), "*", "*.nc"))
    filenames = glob.glob(os.path.join('ABI-L2-CODF', '{:04d}'.format(YEAR), '{:03d}'.format(DOY), "*", "*.nc"))
    print("{:04d}, {:03d} ({} files): start ...".format(YEAR, DOY, len(filenames)))

    # extract regions and save to .csv files
    for filename in filenames:
        try:
            extract_region(filename, YEAR, DOY, dlat=dlat, dlon=dlon)
        except:
            print("{}: ... FAILED".format(filename))

    print("{:04d}, {:03d} ({} files): DONE".format(YEAR, DOY, len(filenames)))


if __name__ == '__main__':

    # day of year (DOY)
    # - one file per year+DOY (e.g. 2018/001 = 1st day of 2018)
    DOYs = range(1, 10)

    # extract images in parallel
    pool = Pool()
    pool.map(process_parallel, DOYs)
    pool.close()
