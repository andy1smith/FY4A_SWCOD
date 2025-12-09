import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def read_channel(site, channel, idx):
    df = pd.read_csv('F:/cropped_FY2021/{}/{}_{}.csv'.format(site, site, channel))
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
    for channel, name in zip(channels, names):
        df = pd.read_csv('./cropped_FY2021/{}/{}_{}.csv'.format(site, site, channel))
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by="time").set_index("time")
        n_pixels = 11 * 11
        col = "{}".format((n_pixels - 1) // 2)
        df = df[[col]].rename(columns={col: name})

        if dfs.empty:
            dfs = df.copy()
        else:
            dfs = pd.concat([dfs, df], axis=1, join='inner')

    # round up to the nearest 1-hour timestamp
    dfs = dfs.resample("1h", label="right").mean()
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
        df = pd.read_csv(f'./cropped_FY2021/{site}/{site}_{channel}.csv')
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
    df = pd.read_excel('./Ground/Station/{}2021.xls'.format(site), skiprows=6, usecols=[0, 1, 5])
    df = df.rename(columns={df.columns[0]: 'time', df.columns[1]: 'T_a', df.columns[2]: 'RH'})
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)  # convert local time to UTC time
    df['T_a'] = df['T_a'] + 273.15  # convert Celsius to kelvin

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



if __name__ == "__main__":
    sites = ['BJC','CSA', 'DHL', 'FKD', 'FQA', 'HLA', 'JZB', 'LCA', 'NMD', 'SJM', 'THL', 'YCA']
    sky = 'clear'
    for site in sites[:1]:
        # read ghi data [W/m2]
        if sky == 'clear':
            ground_dir = './Ground/preprocessed/'
            ground_path = ground_dir + '{}_{}.h5'.format(site, sky)
            df_ground = pd.read_hdf(ground_path, key='df')
            df_ground['Time'] = pd.to_datetime(df_ground['Time'])
            df_ground.set_index('Time', inplace=True)
        else:
            df_ghi = read_ghi(site)

        # RH & T measurement
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
                RH=(('time',), df1d['RH'].values),
                T_a=(('time',), df1d['T_a'].values),
                GHI=(('time',), df1d['ghi'].values)
            )
            xr_all.to_netcdf('../FY4A_data/{}_SW_ref_satellite.nc'.format(site))
        else:
            df_sat = read_satellite_1D(site)
            dfs = pd.merge(
                df1d,
                df_sat,
                left_index=True,
                right_index=True,
                how='left'  # 'left' keeps keys from the first dataframe (df_ground)
            )
            dfs = dfs.dropna()
            zenith_diff = (dfs['Sun_Zen_x'] - dfs['Sun_Zen_y']).abs()
            dfs_clean = dfs[zenith_diff <= 3].copy()
            n_dropped = len(dfs) - len(dfs_clean)
            print(f"Dropped {n_dropped} rows where Sun_Zen difference was > 3 degrees.")
            dfs_clean = dfs_clean[['ghi', 'Sun_Zen_x', 'Sun_Azi_x', 'ghi_clear', 'T_a', 'RH', 'C01', 'C02',
                       'C03', 'C04', 'C05', 'C06', 'Sat_Azi', 'Sat_Zen', 'Sun_Azi_y',
                       'Sun_Gli', 'ele']]
            dfs_clean = dfs_clean.rename(columns={
                'Sun_Zen_x': 'Sun_Zen',
                'Sun_Azi_x': 'Sun_Azi',
                'Sun_Azi_y': 'Sun_Azi_sat' })
            data = dfs_clean.reset_index().sort_values(by='Time')
            data.to_csv('../FY4A_data/{}_radiance_satellite_clear.csv'.format(site), index=False)
            print('successfully saved {}'.format(site))









