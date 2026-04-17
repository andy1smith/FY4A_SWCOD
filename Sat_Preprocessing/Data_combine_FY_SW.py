import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
from mcd43a1_albedo import black,white


def read_channel(site, channel, idx, phase='clear'):
    df = pd.read_csv('./cropped_FY2021_clears/{}/{}_{}.csv'.format(site, site, channel))
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time").set_index("time")
    data = df.iloc[idx]
    return data

def read_satellite_1D(site):
    # extract center pixel
    channels = ['Channel{:02d}'.format(i+1) for i in range(6)]+\
               ['SatelliteAzimuth', 'SatelliteZenith', 'SunAzimuth', 'SunGlintAngle', 'SunZenith', 'elevation']
    names = ["C{:02d}".format(i+1) for i in range(6)] + \
            ['Sat_Azi', 'Sat_Zen', 'Sun_Azi', 'Sun_Gli', 'Sun_Zen', 'ele']

    dfs = pd.DataFrame()
    grid_size = 11 * 11
    center_idx = (grid_size - 1) // 2  #

    # Pre-calculate the 9 indices for the 3x3 block
    indices_3x3 = []
    for r in range(center_idx - 1, center_idx + 2):
        for c in range(center_idx - 1, center_idx + 2):
            indices_3x3.append(str(r * grid_size + c))

    for channel, name in zip(channels, names):
        file_path = f'./cropped_FY2021_clear/{site}/{site}_{channel}.csv'
        df_raw = pd.read_csv(file_path)
        df_raw["time"] = pd.to_datetime(df_raw["time"])
        df_raw = df_raw.sort_values(by="time").set_index("time")

        # df_3x3 = df_raw[indices_3x3].astype(float)
        # spatial_mean = df_3x3.mean(axis=1)
        # spatial_std = df_3x3.std(axis=1)
        # spatial_cv = spatial_std / (spatial_mean + 1e-6)

        # col = center_idx
        col = str(center_idx)
        df = df_raw[[col]].rename(columns={col: name})

        if dfs.empty:
            dfs = df.copy()
        else:
            dfs = pd.concat([dfs, df], axis=1, join='inner')

    # round up to the nearest 1-hour timestamp
    #dfs = dfs.resample("1h", label="right").mean()
    return dfs

def Planck(nu, T):
    """
    Planck's law as a function of wavenumber [cm^-1].

    Planck's law (see equaton on page 453 of [1]).

    Parameters
    ----------
    nu : float or array_like
        Wavenumber [cm^-1].
    T : float or array_like
        Temperature [K].

    Returns
    -------
    Eb :
        Blackbody emission intensity density [W/(m^2 sr cm^-1)]

    References
    ----------
    [1] Mill and Coimbra, "Basic Heat and Mass Transfer"

    """

    h = 6.6261e-34  # Planck's constant [J s]
    kB = 1.3806485e-23  # Boltzmann constant [J / K]
    c = 299792458  # speed of light [m / s]
    C1 = 2 * h * c ** 2  # coefficient 1
    C2 = h * c / kB  # coefficient 2
    nu = nu * 100  # convert from [cm^-1] to [m^-1]

    # blackbody emission
    # equivalent to MATLAB dot calculations
    Eb_nu = C1 * nu ** 3 / (np.exp(C2 * nu / T) - 1)

    # convert to [W/(m^2 sr cm^-1)]
    Eb_nu *= 100
    return Eb_nu


def save_npy(idx):
    channels = []
    for channel in ['Channel{:02d}'.format(i + 1) for i in range(14)] + ['SunZenith']:
        df = read_channel(site=site, channel=channel, idx=idx)
        channels.append(df)
    all_channels = np.stack(channels)
    np.save('./Exp_data/FY4A.npy', all_channels)


def plt_FY4A():
    channels = np.load('./Exp_data/FY4A.npy')

    df_channels = []
    df_radiances = []

    # convert reflectance to radiance (SW)
    data = np.genfromtxt('./Exp_data/SolarTOA.csv', delimiter=',')
    ref_lam = data[:, 0]  # in unit of um
    ref_E = data[:, 1]  # in unit of W/(m2 um)
    ref_E_nu = -ref_E * ref_lam ** 2 / 1e4  # in unit of W/(m^2 cm^-1)
    lamda_sw = [0.47, 0.65, 0.825, 1.379, 1.61, 2.25]  # center wavelength for SW

    for i, lam in zip([i for i in range(6)], lamda_sw):
        # reflectance
        df_ref = channels[i]
        df_channels.append(df_ref)

        # solar zenith angle
        df_soz = channels[-1]

        # Radiance = Reflectance*cos(SZA)*F0/pai*d2
        F_dw_os = np.interp(lam, ref_lam, ref_E)  # W/m2 um in wavelength basis
        # F_dw_os = -np.interp(-1e4 / lam, -1e4 / ref_lam, ref_E_nu)  # W/(m^2 cm^-1) in wavenumber basis

        # # plot
        # plt.figure(figsize=(10, 6), dpi=300)
        # plt.scatter(ref_lam[200:1400], ref_E[200:1400], s=2)
        # plt.scatter(lam, F_dw_os, color='red', zorder=5)
        # plt.tight_layout()
        # plt.show()


        df_rad = df_ref * np.cos(np.radians(
            df_soz)) * F_dw_os / np.pi  # W/(m^2 sr um) in wavenumber basis, W/(m^2 sr cm^-1) in wavenumber basis
        df_radiances.append(df_rad)

    # convert brightness temperature to radiance (LW)
    lamda_lw = [3.75, 3.75, 6.25, 6.95, 7.42, 8.55, 10.8, 12.0]  # center wavelength for LW
    for i, lam in zip([i for i in range(6, 14+1)], lamda_lw):
        # brightness temperature [K]
        df_T = channels[i]
        df_channels.append(df_T)

        df_rad = Planck(1e4 / lam, df_T)  # W/(m^2 sr cm^-1)
        df_radiances.append(df_rad * 1e3)  # mW/(m^2 sr cm^-1)

    # plot
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    plt.figure(figsize=(8, 8), dpi=300)
    for i in range(0, 14):
        plt.subplot(4, 4, i + 1)
        depthmap = df_channels[i].reshape(11, 11)
        plt.imshow(depthmap)
        plt.colorbar(orientation="horizontal")
        plt.title(i + 1)
    plt.suptitle('FY-4A (SW: reflectance, LW: brightness temperature [K])', fontproperties='Times New Roman', x=0.5,
                 y=0.985, size=10, weight='bold')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 8), dpi=300)
    for i in range(0, 14):
        plt.subplot(4, 4, i + 1)
        depthmap = df_radiances[i].reshape(11, 11)
        plt.imshow(depthmap)
        plt.colorbar(orientation="horizontal")
        plt.title(i + 1)
    plt.suptitle('FY-4A (SW: W/(m^2 sr um), LW: mW/(m^2 sr cm^-1))', fontproperties='Times New Roman', x=0.5,
                 y=0.985, size=10, weight='bold')
    plt.tight_layout()
    plt.show()




def read_satellite_2Dmap(site):
    # extract whole 2D map time series for all channels

    channels = ['SunZenith', 'SunAzimuth', 'SatelliteAzimuth', 'SatelliteZenith', 'SunGlintAngle', 'elevation'] +\
    ['Channel{:02d}'.format(i+1) for i in range(6)]

    names =  ['Sun_Zen', 'Sun_Azi', 'Sat_Azi', 'Sat_Zen','Sun_Gli', 'ele'] +\
            ["C{:02d}".format(i+1) for i in range(6)]

    # Prepare storage for shapes and arrays
    var_data = {}
    times = None
    n_pixels = 121

    for channel, name in zip(channels, names):
        df = pd.read_csv(f'./cropped_FY2021_cloudy/{site}/{site}_{channel}.csv')
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        if times is None:
            times = df["time"].values

        # Assume columns "0", "1", ..., "120"
        pixel_array = df[[str(i) for i in range(n_pixels)]].values
        # Reshape to (time, y, x) with 11x11 pixels
        var_data[name] = pixel_array.reshape(-1, 11, 11)

    coords = {
        "time": times,
        "y": np.arange(11),
        "x": np.arange(11)
    }

    # Assemble xarray Dataset
    ds = xr.Dataset(
        {name: (("time", "y", "x"), arr) for name, arr in var_data.items()},
        coords=coords
    )
    return ds


def read_measures(site):
    # read NoAA RH & T measurement
    #df = pd.read_excel('./Ground/Station/{}2021.xls'.format(site), skiprows=6, usecols=[0, 1, 5])
    df = pd.read_csv(f'./Ground/CERN_preprocessed/{site}2021.csv')

    df = df.rename(columns={df.columns[0]: 'time', df.columns[1]: 'T_s', df.columns[2]: 'RH'})
    df['time'] = pd.to_datetime(df['time'])
    #df['time'] = df['time'].dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)  # convert local time to UTC time
    df['T_s'] = df['T_s'] + 273.15  # convert Celsius to kelvin

    # round up to the nearest 1-hour timestamp
    df = df.sort_values(by="time").set_index("time").sort_index()
    df = df.resample("1h", label="right").mean()
    df = df.interpolate(method="time")
    return df


def read_ghi(site):
    data = pd.read_csv('./CERN/CERN_instGHI_2021_UTC.csv')
    df = data[[site]]
    df = df.rename(columns={site: 'ghi'})
    df['time'] =  pd.date_range(start='2021-01-01', end='2022-01-01', freq='h')[:-1]  # generate hourly timestamps for 2021
    df = df.set_index('time').sort_index()

    return df

def modis_albedo_load(site, df_combined):
    """
    Load MODIS mcd43a1 BRDF and albedo product, only load
    p1,p2,p3 for calculating while-, black-, blue-sky albedo.

    Parameters
    ----------
    site
    year
    df_combined
    phase

    Returns
    -------

    """
    file_dir = './mcd43a1_albedo/data/'
    filename = 'CERN2021-MCD43A1-061-results.csv'
    mcd43_df = pd.read_csv(os.path.join(file_dir, filename))
    xsf_df = mcd43_df[mcd43_df['Category'] == site].copy()

    # Quality control
    for iband in range(1, 8):
        # Construct the column names
        qa_col = f'MCD43A1_061_BRDF_Albedo_Band_Mandatory_Quality_Band{iband}'

        # The three parameter columns for this band
        param_cols = [
            f'MCD43A1_061_BRDF_Albedo_Parameters_Band{iband}_0',
            f'MCD43A1_061_BRDF_Albedo_Parameters_Band{iband}_1',
            f'MCD43A1_061_BRDF_Albedo_Parameters_Band{iband}_2'
        ]

        quality_mask = xsf_df[qa_col] > 1  # Standard threshold for "bad" data

        # Set the parameters to NaN for the rows that failed
        for col in param_cols:
            if col in xsf_df.columns:
                xsf_df.loc[quality_mask, col] = np.nan
    cols_to_check = [f'MCD43A1_061_BRDF_Albedo_Parameters_Band{b}_{i}'
                     for b in range(1, 8) for i in range(3)]
    xsf_df = xsf_df.dropna(subset=cols_to_check)
    xsf_df['Date'] = pd.to_datetime(xsf_df['Date'])

    # create a pandas to save df_combined + mcd43a1
    df_combined.reset_index(inplace=True)
    df_filter = df_combined.copy()
    # df_filter['D_portion'] = (
    #         df_filter['diffuse'] / df_filter['Site_dsw']
    # ).clip(0.0, 1.0)
    df_filter['Time'] = pd.to_datetime(df_filter['Time'])

    # 1. Prepare the Join Key in your high-frequency dataframe
    # dt.normalize() converts "2023-01-01 12:30:00" -> "2023-01-01 00:00:00" for matching.
    df_filter['Join_Date'] = df_filter['Time'].dt.normalize()
    # 2. Build the exact list of columns to keep
    # Start ONLY with the merge key
    cols_to_fetch = ['Date']
    rename_map = {}

    FY4A_channels_map = {
        'C01': 3,  # Blue ~0.47 µm → MODIS Band 3 (0.459–0.479)
        'C02': 1,  # Red  ~0.64 µm → MODIS Band 1 (0.620–0.670)
        'C03': 2,  # NIR  ~0.86 µm → MODIS Band 2 (0.841–0.876)
        'C04': 5,  # SWIR ~1.38 µm → MODIS Band 5 (1.24 µm)
        'C05': 6,  # SWIR ~1.6 µm → MODIS Band 6
        'C06': 7  # SWIR ~2.2 µm → MODIS Band 7
    }

    for ch, iband in FY4A_channels_map.items():
        # Source Names (The messy names in xsf_df)
        src_p0 = f'MCD43A1_061_BRDF_Albedo_Parameters_Band{iband}_0'
        src_p1 = f'MCD43A1_061_BRDF_Albedo_Parameters_Band{iband}_1'
        src_p2 = f'MCD43A1_061_BRDF_Albedo_Parameters_Band{iband}_2'

        # Target Names (Your clean names)
        tgt_p0 = f'Abdo_{ch}_p0'
        tgt_p1 = f'Abdo_{ch}_p1'
        tgt_p2 = f'Abdo_{ch}_p2'

        # Add ONLY these specific columns to our fetch list
        cols_to_fetch.extend([src_p0, src_p1, src_p2])
        # Store how to rename them
        rename_map[src_p0] = tgt_p0
        rename_map[src_p1] = tgt_p1
        rename_map[src_p2] = tgt_p2
    # 3. Create a clean subset (The Guardrail)
    # This line ensures the other ~70 columns in xsf_df are completely ignored.
    # We use intersection with xsf_df.columns to avoid errors if a specific band is missing.
    valid_cols = [c for c in cols_to_fetch if c in xsf_df.columns]
    xsf_subset = xsf_df[valid_cols].rename(columns=rename_map)

    # 4. Merge
    # how='left': Keep all df_filter rows/times
    df_final = pd.merge(
        df_filter,
        xsf_subset,
        left_on='Join_Date',
        right_on='Date',
        how='left'
    )

    # 5. Cleanup
    # Drop the helper columns used for joining
    df_final = df_final.dropna()
    df_final = df_final.drop(columns=['Join_Date', 'Date'])

    print(f"Original: {df_combined.shape[0]}")
    print(f"Final: {df_final.shape[0]}")  # -1 for 'Date'

    for ch,iband in FY4A_channels_map.items():
        # Target Names (Your clean names)
        tgt_p0 = f'Abdo_{ch}_p0'
        tgt_p1 = f'Abdo_{ch}_p1'
        tgt_p2 = f'Abdo_{ch}_p2'

        df_final[f'BSA_{ch}'] = black(df_final[tgt_p0], df_final[tgt_p1], df_final[tgt_p2],df_final['Sun_Zen'])
        df_final[f'WSA_{ch}'] = white(df_final[tgt_p0], df_final[tgt_p1], df_final[tgt_p2])
        #df_final[f'Albedo_{ch}'] = blue(df_final[f'WSA_{ch}'], df_final['D_portion'], df_final[f'BSA_{ch}'])
    return df_final
def sample_subset(df_combined):
    # 1. Ensure Time is datetime to extract Month
    # df_combined.index.name = 'Time'
    # df_combined = df_combined.reset_index()
    #df_combined['Time'] = pd.to_datetime(df_combined['Time'])

    # 2. Create "Bins" for your continuous variables
    # We divide the data into 4 or 5 chunks for each variable.
    # pd.cut creates bins based on values (uniform spacing).
    df_combined['month_bin'] = df_combined['Time'].dt.month
    df_combined['Ta_bin'] = pd.cut(df_combined['T_s'], bins=4)  # 5 Temperature zones
    df_combined['RH_bin'] = pd.cut(df_combined['RH'], bins=4)  # 5 Humidity zones
    df_combined['RAA_bin'] = pd.cut(df_combined['RAZ'], bins=4)  # 5 Solar angles
    df_combined['Zen_bin'] = pd.cut(df_combined['Sun_Zen'], bins=4)  # 5 Solar angles

    # 3. Stratified Sampling
    # Group by all bins and take 1 random sample from each valid combination.
    # This forces the model to pick points that are distinct from each other.
    sampled_df = df_combined.groupby(['month_bin', 'Ta_bin', 'RH_bin', 'RAA_bin', 'Zen_bin']).apply(
        lambda x: x.sample(1, random_state=42)
    ).reset_index(drop=True)
    sampled_df.drop(columns=['month_bin', 'Ta_bin', 'RH_bin', 'RAA_bin', 'Zen_bin'], inplace=True)
    # 4. Cleanup (remove the temporary bin columns)
    #sampled_df = sampled_df.drop(columns=['month_bin', 'Ta_bin', 'RH_bin', 'RAA_bin', 'Zen_bin'])
    print(f"Original size: {len(df_combined)}")
    print(f"Sampled size:  {len(sampled_df)}")
    return sampled_df


if __name__ == "__main__":
    sites = [ 'CSA', 'DHL', 'FKD', 'FQA', 'HLA', 'JZB', 'LCA', 'NMD', 'SJM', 'THL', 'YCA']
    #'BJC',
    # df = pd.read_csv('../FY4A_data/' + 'CERN_info.csv')
    # sites = df['site'].tolist()
    sky = 'clear'
    #sites = ['BJC']
    for site in sites:
        # load CERN ghi data [W/m2]
        try:
            if sky == 'clear':
                ground_dir = './Ground/preprocessed_GHI/'
                ground_path = ground_dir + '{}_{}.h5'.format(site, sky)
                df_ground = pd.read_hdf(ground_path, key='df')
                df_ground['Time'] = pd.to_datetime(df_ground['Time'])
                df_ground.set_index('Time', inplace=True)
            else:
                df_ghi = read_ghi(site)
        except FileNotFoundError:
            print(f"File not found for site {site}, skipping.")
            continue

        # load NoAA RH & T measurement
        df_mea = read_measures(site)
        if sky == 'clear':
            df_ground.index = df_ground.index.tz_localize(None)
            df1d = pd.merge(
                df_ground,
                df_mea,
                left_index=True,
                right_index=True,
                how='left'  # 'left' keeps keys from the first dataframe (df_ground)
            )
        else:
            df1d = df_mea.join(df_ghi, how='inner')
        # FY4A channels reflectance SW
        extract2D = False
        if extract2D:
            xr_sat = read_satellite_2Dmap(site)
            df1d = df1d.reindex(xr_sat.time.values)
            xr_all = xr_sat.assign(
                RH=(('time',), df1d['RH'].values), # %
                T_s=(('time',), df1d['T_s'].values), # K
                GHI=(('time',), df1d['ghi'].values)
            )
            xr_all.to_netcdf('../FY4A_data/{}_SW_ref_satellite.nc'.format(site))
        else:
            # extract center pixel
            df_sat = read_satellite_1D(site)
            # df_Sat match with ground df1d
            dfs = pd.merge(
                df1d,
                df_sat,
                left_index=True,
                right_index=True,
                how = 'inner',
                #how='left'  # 'left' keeps keys from the first dataframe (df_ground)
            )
            dfs = dfs.dropna()
            zenith_diff = (dfs['Sun_Zen_x'] - dfs['Sun_Zen_y']).abs()
            dfs_clean = dfs[zenith_diff <= 3].copy()
            n_dropped = len(dfs) - len(dfs_clean)
            d = dfs[['Sun_Zen_x', 'Sun_Zen_y']]
            print(f"Dropped {n_dropped} rows where Sun_Zen difference was > 3 degrees.")
            dfs_clean = dfs_clean[['ghi', 'Sun_Zen_x', 'Sun_Azi_x', 'ghi_clear', 'T_s', 'RH', 'C01', 'C02',
                       'C03', 'C04', 'C05', 'C06', 'Sat_Azi', 'Sat_Zen', 'Sun_Azi_y',
                       'Sun_Gli', 'ele']]
            dfs_clean = dfs_clean.rename(columns={
                'Sun_Zen_x': 'Sun_Zen',
                'Sun_Azi_x': 'Sun_Azi',
                'Sun_Azi_y': 'Sun_Azi_sat' })
            data = dfs_clean.reset_index()
            if 'index' in data.columns:
                data = data.rename(columns={'index': 'Time'})
            data = data.sort_values(by='Time')

            #data.to_csv('../FY4A_data/{}_radiance_satellite_clear_noalbedo.csv'.format(site), index=False)
            # match with ground albedo
            df_final = modis_albedo_load(site, data)

            rel_az = np.abs(df_final['Sun_Azi'] - df_final['Sat_Azi'])
            df_final['RAZ'] = np.minimum(rel_az, 360 - rel_az)
            df_final = df_final[df_final['Sun_Zen'] <= 65] # day filter
            df_final = df_final.sort_values(by='Time')
            df_final.drop(columns=['Sun_Azi', 'Sat_Azi', 'Sun_Azi_sat'], inplace=True)
            filename = '../FY4A_data/site_sat_data/{}_radiance_satellite_{}.csv'.format(site, sky)
            df_final.to_csv(filename, index=False)
            print('successfully saved {}'.format(site))

            df_final = pd.read_csv(filename)
            df_final['Time'] = pd.to_datetime(df_final['Time'])
            df_sample = sample_subset(df_final)
            filename = "../FY4A_data/site_sat_data/{}_radiance_satellite_{}_sample.csv".format(site, sky)
            df_sample.to_csv(filename, index=False)
            print('Data sampled data saved to:', filename)









