"""Post-process model outputs, and compare OLR and DLW."""

import os
import pandas as pd
import numpy as np
from scipy import interpolate
from itertools import chain
from multiprocessing import Pool
import re
import h5py
import time
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from SCOPE_func import *
from LBL_funcs_fullSpectrum import *
from fun_nearealtime_RTM import *
from scipy.optimize import curve_fit

def load_clearsky(site, df, tz='UTC'):
    import pvlib
    lat, lon, alt = 40.05192, -88.37309, 213
    df1 = df.set_index('Time', inplace=False)
    location = pvlib.location.Location(lat, lon, tz, alt)
    clearsky_data = location.get_clearsky(df1.index)#.resample('5min').mean()
    ghi = clearsky_data['ghi']
    dni = clearsky_data['dni']
    return ghi, dni


def compare_clear_dsw(site, sourcefile, sky="clear", file_dir=None, figlabel=None):
    """Compare the ground and modeled DLW.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + surfradellite data files.

    Returns
    -------
    None

    """
    # not used for current version
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"

    csvfile = ('./GOES_validation/' + f"Result_{timeofday}_BON_radiance_surfradellite_{figlabel}_{sky}.csv")
    rtm_dsw, rtm_dni, rtm_dhi = [], [], []
    try:
        df_combined = pd.read_csv(csvfile)
        df_combined['Time'] = pd.to_datetime(df_combined['Time'])
        rtm_GHI, rtm_DNI = df_combined['rtm_dsw'],df_combined['rtm_dni']
        site_GHI, site_DNI = df_combined['Site_dsw'], df_combined['direct_n']
        nsrdb_ghi, nsrdb_dni = load_NSRDB(site, df_combined)
        pvlib_ghi, pvlib_dni = load_clearsky(site, df_combined)
        print('dsw read from existing csv file:', csvfile)
        plot_day(df_combined, figlabel=None)
    except Exception:
        # open surfradellite observation data
        file_path = sourcefile
        surfrad = pd.read_hdf(file_path, key="df")
        surfrad.rename(columns={"dw_solar": "Site_dsw"}, inplace=True)
        site_GHI = surfrad['Site_dsw']  # should not be surfrad[dw_ir], it includes the surface reflected radiation
        site_DNI = surfrad['direct_n']
        print(f"Run RTM to get output DSW.")
        from aod_codes import read_aod
        # aod time series from surfrad
        df_aod = read_aod(site)
        # concate previous SCOPE results & aod series
        c_index = [index for index in surfrad.index if index in df_aod.index]
        surfrad = pd.concat([surfrad.loc[c_index], df_aod.loc[c_index]], join='inner', axis=1)
        print('# of surfrad:', surfrad.shape[0])
        for i in range(surfrad.shape[0]):
            Sun_Zen= surfrad['zen'][i]
            COD_goes = 0  # Assuming COD is a column in surfrad_rad
            T_a, RH = surfrad['temp'].iloc[i], surfrad['rh'].iloc[i]
            AOD = surfrad['aod'].iloc[i]  # Assuming AOD is a column in surfrad_rad
            AOD = 0.1243 #None
            dsw, dni, dhi = get_RTM_dsw(Sun_Zen, COD_goes, T_a, RH, AOD)
            rtm_dsw.append(dsw)
            rtm_dni.append(dni)
            rtm_dhi.append(dhi)

            df_new = pd.DataFrame({
                'rtm_dsw': rtm_dsw,
                'rtm_dni': rtm_dni,
                'rtm_dhi': rtm_dhi,
            })
        surfrad = surfrad.reset_index()
        surfrad = surfrad.rename(columns={"index": "Time"})
        df_combined = pd.concat([surfrad, df_new], axis=1)
        df_combined['rtm_dni'] = df_combined['rtm_dni']/np.cos(np.deg2rad(df_combined['Site_zen']))
        df_combined.to_csv(csvfile, index=False)
        rtm_DNI, rtm_GHI = df_combined['rtm_dni'], df_combined['rtm_dsw']

    CODfromwho = 'RTM_clear'
    # COD = 0, no rtm_channel is rtm_channel.
    plot_data_dw(site_GHI, rtm_GHI, site_DNI, rtm_DNI, CODfromwho, figlabel=figlabel)
    CODfromwho = 'NSRDB_clear'
    plot_data_dw(site_GHI, nsrdb_ghi, site_DNI, nsrdb_dni, CODfromwho, figlabel=figlabel)
    CODfromwho = 'pvlib_clear'
    plot_data_dw(site_GHI, pvlib_ghi, site_DNI, pvlib_dni, CODfromwho, figlabel=figlabel)

def load_NSRDB(site, df, year = '2019'):
    nsrdb_dir = './data/NSRDB/nsrdb_site_files'
    df_site = pd.read_hdf(
        os.path.join(nsrdb_dir,
                     "nsrdb_{}_{}.h5".format(year,site)),
        'data'
    )
    df['Time'] = pd.to_datetime(df['Time'])
    df_site['Time'] = df_site['Time'].dt.tz_localize(None)

    df_site.set_index('Time', inplace=True)
    df_site = df_site.select_dtypes(include=[np.number]).resample("5min", label="right").mean()
    df_site.reset_index(inplace=True)

    df_combined = pd.merge(
        df,
        df_site[['Time', 'clearsky_ghi', 'clearsky_dni']].rename(columns={'clearsky_ghi': 'nsrdb_ghi', 'clearsky_dni': 'nsrdb_dni'}),
        #df_site[['Time', 'ghi', 'dni']].rename(columns={'ghi': 'nsrdb_ghi', 'dni': 'nsrdb_dni'}),
        on='Time',
        how='left'
    )
    return df_combined['nsrdb_ghi'], df_combined['nsrdb_dni']


def plot_day(df_combined, figlabel=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo

    # Ensure Time is datetime and localized
    df_combined.set_index('Time', inplace=True)
    df_combined.index = pd.to_datetime(df_combined.index)
    df_combined.index = df_combined.index.tz_localize('UTC').tz_convert(ZoneInfo('America/Chicago'))

    #  First figure: absolute values
    # fig, ax = plt.subplots(figsize=(8, 5))
    # df_combined['Site_dsw'].plot(ax=ax, color='C0')
    # df_combined['rtm_dsw'].plot(ax=ax, color='C0', linestyle='', marker='^', markersize=3)
    # try:
    #     df_combined['rtm_dsw_aodc'].plot(ax=ax, color='C0', linestyle='--')
    # except Exception:
    #     pass
    #
    # df_combined['direct_n'].plot(ax=ax, color='C1')
    # df_combined['rtm_dni'].plot(ax=ax, color='C1', linestyle='', marker='^', markersize=3)
    # try:
    #     df_combined['rtm_dni_aodc'].plot(ax=ax, color='C1', linestyle='--')
    # except Exception:
    #     pass
    #
    # df_combined['diffuse'].plot(ax=ax, color='C2')
    # df_combined['rtm_dhi'].plot(ax=ax, color='C2', linestyle='', marker='^', markersize=3)
    # try:
    #     df_combined['rtm_dhi_1HG'].plot(ax=ax, color='C2', linestyle='--')
    # except Exception:
    #     pass
    # #
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Radiation [W/m²]')
    # ax.set_title('Radiation Comparison')
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H', tz=ZoneInfo('America/Chicago')))
    # plt.tight_layout()
    # plt.show()

    # 🎯 Second figure: differences (RTM - Site)
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    # DSW differences
    (df_combined['rtm_dsw'] - df_combined['Site_dsw']).plot(ax=ax2, color='C0', linestyle='-', label='DSW(RTM) - site')
    # (df_combined['rtm_dsw_aodc'] - df_combined['Site_dsw']).plot(ax=ax2, color='C0', linestyle='--',
    #                                                              label='DSW - RTM (aodc)')
    # DNI differences
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    (df_combined['rtm_dni'] - df_combined['direct_n']).plot(ax=ax2, color='C1', linestyle='-', label='DNI(rtm) - site')
    #(df_combined['rtm_dni_airm'] - df_combined['direct_n']).plot(ax=ax2, color='C2', linestyle='-', label='DNI(rtm_am) - site')
    (df_combined['rtm_dni_1HG'] - df_combined['direct_n']).plot(ax=ax2, color='C2', linestyle='--',
                                                                label='DNI(1HG) - site')

    # Optional: DHI differences (commented out)
    (df_combined['rtm_dhi'] - df_combined['diffuse']).plot(ax=ax2, color='C4', linestyle='-', label='DHI - RTM')
    (df_combined['rtm_dhi_1HG'] - df_combined['diffuse']).plot(ax=ax2, color='C4', linestyle='--', label='DHI - RTM (1HG)')

    # 📈 Add second y-axis for solar zenith angle
    ax2b = ax2.twinx()
    df_combined['aod'].plot(ax=ax2b, color='gray', linestyle=':', label='AOD')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Difference [W/m²]')
    ax2b.set_ylabel('AOD', color='gray')
    ax2b.tick_params(axis='y', labelcolor='gray')

    # Legend (merge from both axes)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center',ncols=4)#, bbox_to_anchor=(1, 0.5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H', tz=ZoneInfo('America/Chicago')))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sites = [
        # ["BON", 40.05192, -88.37309, 213], # aod = 0.2
        #["DRA", 36.62373, -116.01947, 1007], # aod = 0.2
        # ["FPK", 48.30783, -105.10170, 634],  # aod =0.78
        # ["GWN", 34.25470, -89.87290, 98], # aod = 0.55
        # ["PSU", 40.72012, -77.93085, 376], # aod = 0.85
       #  ["SXF", 43.73403, -96.62328, 473], # 1.55
        ["TBL", 40.12498, -105.23680, 1689], # 0.958
    ]
    year = '2019'
    for timeofday in ["day"]:
        data_dir = './GOES_tool/SURFRAD/preprocessed/'  #'./lut_test_file/'
        spectral = 'SW'
        phase = 'clearsky' #'water' clearsky
        N_bundles = 1000

    for sky in ["clear"]: #"cloudy", "day", "night"
        for site_name, lat, lon, elev in sites:
            file_path = os.path.join(data_dir, f"{site_name}_clear.h5")
            if os.path.exists(file_path):
                compare_clear_dsw(site_name, file_path,
                                  sky=sky, file_dir='./GOES_data/', figlabel=f"{site_name}_{year}")
                print(f"Loaded {site_name} with {len(df)} rows.")
        else:
            print(f"⚠️ File not found for site {site_name}: {file_path}")


