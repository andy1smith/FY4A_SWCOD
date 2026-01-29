import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def plt_F0():
    data = np.genfromtxt('./Exp_data/SolarTOA.csv', delimiter=',')

    dnu = 3  # spectral resolution 0.1 is enough, 0.01 is too fine, especially for cloudy periods
    nu = np.arange(2500, 35000, dnu)
    ref_lam = data[:, 0]  # in unit of um
    ref_E = data[:, 1]  # in unit of W/m2 um
    ref_E_nu = -ref_E * ref_lam ** 2 / 1e4  # in unit of W/(m^2 cm^-1)

    idx = 1400
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(ref_lam[200:idx], ref_E[200:idx], s=2)
    # x = np.array([0.47, 0.65, 0.825, 1.379, 1.61, 2.25])  # FY4A center wavelength
    x = np.array([0.47, 0.51, 0.64, 0.86, 1.6, 2.3])  # H8 center wavelength
    y = np.interp(x, ref_lam, ref_E)
    plt.scatter(x, y, color='red', zorder=5)
    plt.tight_layout()
    plt.show()

    F_dw_os = -np.interp(-nu, -1e4 / ref_lam, ref_E_nu)  # in wavenumber basis
    plt.plot(nu, F_dw_os)
    plt.show()


def read_channel(site, channel, idx):
    df = pd.read_csv('F:/cropped_FY2021/{}/{}_{}.csv'.format(site, site, channel))
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by="time").set_index("time")
    data = df.iloc[idx]

    return data


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


def read_satellite(site):
    # extract center pixel
    channels = ['Channel{:02d}'.format(i+1) for i in range(6)] + ['Channel{:02d}'.format(i+1) for i in
                                                                    range(7, 14)] + \
               ['SatelliteAzimuth', 'SatelliteZenith', 'SunAzimuth', 'SunGlintAngle', 'SunZenith', 'elevation']
    names = ["C{:02d}".format(i+1) for i in range(6)] + ["C{:02d}".format(i+1) for i in range(7, 14)] + \
            ['Sat_Azi', 'Sat_Zen', 'Sun_Azi', 'Sun_Gli', 'Sun_Zen', 'ele']

    dfs = pd.DataFrame()
    for channel, name in zip(channels, names):
        df = pd.read_csv('F:/cropped_FY2021/{}/{}_{}.csv'.format(site, site, channel))
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by="time").set_index("time")
        n_pixels = 11 * 11
        col = "{}".format((n_pixels - 1) // 2)
        df = df[[col]].rename(columns={col: name})

        if dfs.empty:
            dfs = df.copy()
        else:
            dfs = pd.concat([dfs, df], axis=1, join='inner')

    # convert brightness temperature to radiance (LW)
    lamda_lw = [3.75, 3.75, 6.25, 6.95, 7.42, 8.55, 10.8, 12.0]  # center wavelength for LW
    for col, lam in zip(["C{:02d}".format(i+1) for i in range(7, 14)], lamda_lw):
        # brightness temperature [K]
        df_T = dfs[col]
        df_rad = Planck(1e4 / lam, df_T)  # W/(m^2 sr cm^-1)
        dfs["{}_rad".format(col)] = df_rad

    # round up to the nearest 1-hour timestamp
    dfs = dfs.resample("1H", label="right").mean()

    return dfs


def read_measures(site):
    df = pd.read_excel('./Exp_data/{}2021.xls'.format(site), skiprows=6, usecols=[0, 1, 5])
    df = df.rename(columns={df.columns[0]: 'time', df.columns[1]: 'T_a', df.columns[2]: 'RH'})
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC').dt.tz_localize(None)  # convert local time to UTC time
    df['T_a'] = df['T_a'] + 273.15  # convert Celsius to kelvin

    # round up to the nearest 1-hour timestamp
    df = df.sort_values(by="time").set_index("time")
    df = df.resample("1H", label="right").mean()

    return df


def read_ghi(site):
    data = pd.read_csv('./Exp_data/CERN_instGHI_2021_UTC.csv')
    df = data[[site]]
    df = df.rename(columns={site: 'ghi'})
    df['time'] = pd.date_range(start='2021-01-01', end='2022-01-01', freq='H', closed='left')
    df = df.set_index('time')

    return df

def cal_black(site):
    df = pd.read_csv('./Exp_data/{}_radiance_satellite.csv'.format(site))

    channels = ['C{:02d}'.format(i) for i in range(8, 14 + 1)]
    T = df["T_a"].values
    for channel in channels:
        # load AGRI calibration data
        filename = os.path.join('agri_calibration', 'agri_srf.xlsx')
        calibration = pd.read_excel(filename, sheet_name=channel)

        wv = calibration['wvl [um]']  # wavelength [um]
        nu = 1e4 / wv  # wavenumber [cm^-1]
        rel_srf = calibration['SRF [%]'] / 100  # relative SRF [-]

        # reverse order (so wavenumber is increasing)
        nu = nu[::-1]
        rel_srf = rel_srf[::-1]

        # get band equivalent width = Eqw = \int_{nu_1}^{nu_2} R_{nu} dnu
        Eqw = np.trapz(rel_srf, x=nu)

        # get the channel radiance [W/(m^2 sr)]
        spectral_rad = df['{}_rad'.format(channel)].values  # spectral radiance [W/(m^2 sr cm^-1)]
        rad = spectral_rad * Eqw * np.pi  # radiance [W/m^2]
        df.insert(df.shape[1], '{}_satellite'.format(channel), rad)

        # ==================================================
        # blackbody radiation via Planck's distribution
        # ==================================================

        # compute Planck distribution per channel
        h = 6.6261e-34  # Planck's constant [J s]
        kB = 1.3806485e-23  # Boltzmann constant [J / K]
        c = 299792458  # speed of light [m / s]
        C1 = 2 * h * c ** 2
        C2 = h * c / kB

        # convert nu (m,) and T (n,) to (m, n) arrays
        # - enables vectorized computaiton
        m = len(nu)
        n = len(T)
        nu_matrix = np.tile(nu, (n, 1)).T  # (m, n): repeat along columns
        T_matrix = np.tile(T, (m, 1))  # (m, n): repeat along rows
        nu_matrix *= 100.0  # convert from [cm^-1] to [m^-1]

        # blackbody emission
        Eb_nu = C1 * (nu_matrix ** 3) / (np.exp(C2 * nu_matrix / T_matrix) - 1)
        Eb_nu *= 100  # convert to [W/(m^2 sr cm^-1)]
        Eb_nu *= np.pi  # solid angle => [W/(m^2 cm^-1)]

        # integrate over wavenumbers => (m,) values (one per T) [W/m^2]
        Eb = np.trapz(Eb_nu, x=nu, axis=0)
        df.insert(df.shape[1], "{}_blackbody".format(channel), Eb)

        print('Measured chanel:{}, min:{}, max:{}'.
              format(channel, df['{}_satellite'.format(channel)].min(),
                     df['{}_satellite'.format(channel)].max()))  # W/(m^2)

    # save
    df.to_csv('./Exp_data/{}_radiance_satellite_all.csv'.format(site), index=False)


if __name__ == "__main__":
    sites = ['AKA']

    for site in sites:
        # FY4A channels [W/(m^2 sr cm^-1)]
        df_sat = read_satellite(site)

        # RH & T measurement
        df_mea = read_measures(site)

        # read ghi data [W/m2]
        df_ghi = read_ghi(site)

        # concate & save
        dfs = pd.concat([df_sat, df_mea, df_ghi], axis=1, join='inner')
        data = dfs.reset_index().sort_values(by='time')
        data = data.dropna()
        data.to_csv('./Exp_data/{}_radiance_satellite.csv'.format(site), index=False)


        # convert channel unit to W/m2, calculate blackbody
        cal_black(site)



