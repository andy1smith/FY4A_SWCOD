
import netCDF4 as nc
import os
import numpy as np
import pandas as pd

from fun_goes_process import *

ind1,ind2,ind3,ind4 = 829,832, 1713,1716
file_path = './GOES_data/GOES-17/ABI-L1b-RadF/202116020/'
Fls = [f for f in os.listdir(file_path) if f.endswith('.nc')]

dataset = nc.Dataset(file_path+Fls[0], mode='r')
Rad_data = dataset.variables['Rad']
ds_f = Rad_data.shape[0] // 2701 # 4km grid=2701
nrows, ncols = Rad_data.shape[0], Rad_data.shape[1]
# downsample to the grid 4km
if ds_f > 1:
    print(ds_f)
    # Method 1: No average between grids
    #rad = Rad_data[::ds_f, ::ds_f]
    # Method 2: Average between grids
    rad = np.mean(
        Rad_data[:].reshape(nrows // ds_f, ds_f,
                        ncols // ds_f, ds_f),
                        axis=(1, 3)
    )
    y = dataset.variables['y'][::ds_f]
    x = dataset.variables['x'][::ds_f]
else:
    rad = Rad_data[:]
    y = dataset.variables['y']
    x = dataset.variables['x']
# extract the radiance data
rad_chx = rad[ind1:ind2, ind3:ind4].flatten()
ch_name = Fls[0][43:46]
# convert the time variable 't_' to UTC time
t_ = dataset.variables['t'][:]
utc_time = convert_to_utc(t_.item())

# Calculate original indices from downsampled indices
# Generate a (x, y) n*n grid
yo, xo = np.meshgrid(y[ind1:ind2], x[ind3:ind4])

# Calculate the original row and column indices
yo = yo.flatten()
xo = xo.flatten()

# official constant
r_eq = 6378137  # semi-major axis in meters
r_pol = 6356752.31414   # semi-minor axis in meters
H = 42164160  # perspective point height + semi-major axis
# r_eq = dataset.variables['goes_imager_projection'].semi_major_axis  # m
# r_pol = dataset.variables['goes_imager_projection'].semi_minor_axis # m
# h0 = dataset.variables['goes_imager_projection'].perspective_point_height # m
# H = h0 + r_eq
lon_0 = dataset.variables['goes_imager_projection'].longitude_of_projection_origin # degree
lat_0 = dataset.variables['goes_imager_projection'].latitude_of_projection_origin # degree
# print(r_eq, r_pol, H, lon_0)

# important notice, official guide mark all grids in (y,x)
lats,lons=calculate_geodetic_coordinates(yo, xo, r_eq, r_pol, H, np.deg2rad(lon_0))

sun_zenith_angles, sun_azimuth_angles = sun_zenith_angle(utc_time, np.rad2deg(lons), np.rad2deg(lats))

local_zenith_angles = calculate_local_zenith_angle(lon_0, lat_0,
                                np.rad2deg(lons), np.rad2deg(lats),input_type='deg')

local_azimuth_angles,_ = calculate_local_azimuth(np.rad2deg(lons), 
                                                 np.rad2deg(lats),input_type='deg')

relative_azimuth_angles = calculate_relative_azimuth_angle(local_azimuth_angles, 
                                                           sun_azimuth_angles,
                                                 input_type='deg')
ref_chx = Conver_ref(rad_chx,ch_name)

utc_time_list = [utc_time]*len(lats)
data = {'time': utc_time_list, #'Latitude': np.rad2deg(lats), 'Longitude': np.rad2deg(lons),
        ch_name: ref_chx,
        'Sun_Zen': sun_zenith_angles, 'Sun_Azi': sun_azimuth_angles,
        'Local_Zen': local_zenith_angles, 'Sat_Azi': local_azimuth_angles,
        'rela_azi': relative_azimuth_angles}
df = pd.DataFrame(data)
dataset.close()
del utc_time_list

for i in range(1,len(Fls)):
    dataset = nc.Dataset(file_path + Fls[i], mode='r')
    Rad_data = dataset.variables['Rad']
    ds_f = Rad_data.shape[0] // 2701  # 4km grid=2701
    nrows, ncols = Rad_data.shape[0], Rad_data.shape[1]
    # downsample to the grid 4km
    if ds_f > 1:
        print(ds_f)
        # Method 1: No average between grids
        #rad = Rad_data[::ds_f, ::ds_f]
        # Method 2: Average between grids
        rad = np.mean(
            Rad_data[:].reshape(nrows // ds_f, ds_f,
                            ncols // ds_f, ds_f),
                            axis=(1, 3)
        )
    else:
        rad = Rad_data[:]
    # extract the radiance data
    rad_chx = rad[ind1:ind2, ind3:ind4].flatten()
    ch_name = Fls[i][43:46]
    ref_chx = Conver_ref(rad_chx, ch_name)
    df[ch_name]=ref_chx
    dataset.close()
df['TPW'] = [25.1428]*df.shape[0]
df = df[['time','C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'Sun_Zen', 'Sun_Azi', 'Local_Zen', 'Sat_Azi', 'rela_azi','TPW']]
sky = 'day'
df.to_csv("GOES_{}_radiance_satellite.csv".format(sky), index=False)
