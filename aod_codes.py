import pandas as pd
import numpy as np
from scipy import interpolate
from tqdm import tqdm
import sys, os

from SCOPE_func import calculate_tpw
from fun_nearealtime_RTM import plot_data_dw


def read_aod(site):
    df = pd.read_csv('./AOD_correction/AOD_SURFRAD/{}_2019.csv'.format(site.lower()))
    df.columns = ['time', 'aod']
    print(site, 'mean of AOD', round(df['aod'].mean(), 4))

    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[~df.index.duplicated(keep='first')]  # Remove any duplicates
    df = df.reindex(pd.date_range(start='2019-01-04', end='2019-12-31 23:59:00', freq='T'))
    df = df.interpolate(method='linear')
    df = df.resample("5min", closed="right", label="right").mean()

    # df = df.ffill().bfill()   # backward fill after forward fill
    # df = df.fillna(aod)  # fill mean value
    df = df.dropna()  # drop NAN value

    return df

def evaluate_error(pre, true):
    error_dlw = (pre - true).dropna(how="any").values
    mae_dlw = np.mean(np.abs(error_dlw))
    mbe_dlw = np.mean(error_dlw)
    rmse_dlw = np.sqrt(np.mean(error_dlw ** 2))
    return mae_dlw, mbe_dlw, rmse_dlw



def correction3(dsw, aod):
    dswc = []
    dsw['TPW'] = dsw.apply(lambda row: calculate_tpw(row['temp']+273.15, row['rh']/100), axis=1)
    th0_values = aod['th0'].unique()
    for idx in tqdm(range(len(dsw))):
        sun_zen = dsw['Sun_Zen'].iloc[idx]
        # Find nearest th0
        nearest_th0 = th0_values[np.abs(th0_values - sun_zen).argmin()]
        # Subset aod for this th0
        aod_subset = aod[aod['th0'] == nearest_th0]
        # Interpolate over TPW and AOD
        point = dsw[['TPW', 'aod']].iloc[idx].values
        dsw_cor = interpolate.griddata(
            aod_subset[['TPW', 'AOD']].values,
            aod_subset['dsw'].values,
            point,
            method='linear'
        )
        dswc.append(dsw_cor[0] if dsw_cor is not None else np.nan)
    return dswc

def correction2(dsw, aod):
    dswc = []
    dsw['TPW'] = dsw.apply(lambda row: calculate_tpw(row['temp']+273.15, row['rh']/100), axis=1) # dsw in zenith angle # dsw in zenith angle
    for idx in tqdm(range(len(dsw))):
        dsw_cor = interpolate.griddata(
            aod[["TPW", "th0", "AOD"]].values,
            aod["dsw"].values, # zenith angle = 0
            dsw[["TPW", "Sun_Zen", "aod"]].iloc[idx].values,  # AOD time series
            method="linear"
        )
        dsw_z = dsw_cor[0]#*np.cos(np.deg2rad(dsw['Sun_Zen'].iloc[idx])) # cannot do this, since AOD scattering is larger in clearsky.
        #print(dsw_z)
        dswc.append(dsw_z)
    return dswc

def correction(dsw, aod):
    dswc = []
    dsw['Ta'] = dsw['temp'] + 273.15  # Convert temperature to Kelvin
    dsw['rh'] = dsw['rh'] / 100  # Convert relative humidity to fraction
    #dsw['TPW'] = dsw.apply(lambda row: calculate_tpw(row['temp']+273.15, row['rh']/100), axis=1) # dsw in zenith angle # dsw in zenith angle
    for idx in tqdm(range(len(dsw))):
        dsw_cor = interpolate.griddata(
            aod[["rh", "Ta", "AOD"]].values,
            aod["dsw"].values, # zenith angle = 0
            dsw[["rh", "Ta", "aod"]].iloc[idx].values,  # AOD time series
            method="linear",
        )
        dsw_z = dsw_cor[0]*np.cos(np.deg2rad(dsw['Sun_Zen'].iloc[idx])) # cannot do this, since AOD scattering is larger in clearsky.
        #print(dsw_z)
        dswc.append(dsw_z)
    return dswc

def correction4(dsw, aod):
    dswc = []
    dsw['Ta'] = dsw['temp'] + 273.15  # Convert temperature to Kelvin
    dsw['rh'] = dsw['rh'] / 100  # Convert relative humidity to fraction
    #dsw['TPW'] = dsw.apply(lambda row: calculate_tpw(row['temp']+273.15, row['rh']/100), axis=1) # dsw in zenith angle # dsw in zenith angle
    for idx in tqdm(range(len(dsw))):
        dsw_cor = interpolate.griddata(
            aod[["rh", "Ta", "th0", "AOD"]].values,
            aod["dsw"].values, # zenith angle = 0
            dsw[["rh", "Ta", "Sun_Zen", "aod"]].iloc[idx].values,  # AOD time series
            method="linear",
        )
        dsw_z = dsw_cor[0] # *np.cos(np.deg2rad(dsw['Sun_Zen'].iloc[idx])) # cannot do this, since AOD scattering is larger in clearsky.
        #print(dsw_z)
        dswc.append(dsw_z)
    return dswc

def correction_all(dsw, aod):
    dsw['Ta'] = dsw['temp'] + 273.15
    dsw['rh'] = dsw['rh'] / 100

    dswc, dnic, dhic = [], [], []

    input_cols = ["rh", "Ta", "th0", "AOD"]
    target_cols = ["dsw", "dni", "dhi"]

    for idx in tqdm(range(len(dsw))):
        #point = dsw.loc[idx, ["rh", "Ta", "Sun_Zen", "aod"]].values
        point = dsw.iloc[idx][["rh", "Ta", "Sun_Zen", "aod"]]

        dsw_val = interpolate.griddata(aod[input_cols], aod["dsw"].values, point.values, method="linear")
        dni_val = interpolate.griddata(aod[input_cols].values, aod["dni"].values, point, method="linear")
        dhi_val = interpolate.griddata(aod[input_cols].values, aod["dhi"].values, point, method="linear")

        dswc.append(dsw_val[0] if dsw_val is not None else np.nan)
        dnic.append(dni_val[0] if dni_val is not None else np.nan)
        dhic.append(dhi_val[0] if dhi_val is not None else np.nan)

    return dswc, dnic, dhic

def aod_correction_clear(data_dir='./', site='BON', sky='day'):
    results = []
    timeofday = 'night' if sky == 'night' else 'day'
    # SCOPE results under clear sky condition
    # figlabel = "COD=0"
    # csvfile = './GOES_validation/2019_June_COD=0_albedo/' + f"Result_BON_{sky}_radiance_satellite_water_{figlabel}.csv"
    # df_combined = pd.read_csv(csvfile)
    # dsw = df_combined
    # dsw['Time'] = pd.to_datetime(dsw['Time'])
    # dsw = dsw.set_index('Time')

    dsw= pd.read_hdf(data_dir+f"{site}_{sky}.h5", "df")
    dsw['Time'] = pd.to_datetime(dsw['Time'])
    dsw.set_index('Time', inplace=True)
    dsw.rename(columns={'zen': 'Sun_Zen'}, inplace=True)
    dsw[dsw._get_numeric_data() < 0] = np.nan  # abnormal data if exist

    # aod time series from surfrad
    df_aod = read_aod(site)

    # concate previous SCOPE results & aod series
    c_index = [index for index in dsw.index if index in df_aod.index]
    df_dsw = pd.concat([dsw.loc[c_index], df_aod.loc[c_index]], join='inner', axis=1)

    # LBL results for different AOD under clear sky condition
    dsw_aod = pd.read_hdf("AOD_correction/results_54layers_3_dsw_{}_aods_ta_rh_th0.h5".format(timeofday), "data")
    print(sorted(dsw_aod['AOD'].unique()))
    #df_dsw['dsw_aodc'] = correction4(df_dsw, dsw_aod)
    df_dsw['dsw_aodc'],df_dsw['dni_aodc'],df_dsw['dhi_aodc'] = correction_all(df_dsw, dsw_aod)
    df_dsw = df_dsw.dropna()

    # save corrected dsw
    try:
        df_dsw['dsw_true'] = dsw['Site_dsw'].copy()
        df_dsw['dsw_pred'] = dsw['goes_dsw'].copy()
        # DNI
        df_dsw['dni_true'] = df_dsw['direct_n'].copy()
        df_dsw['dni_pred'] = df_dsw['goes_dni'].copy()
    except KeyError:
        df_dsw['dsw_true'] = df_dsw['dw_solar'].copy()
        df_dsw['dsw_pred'] = df_dsw['ghi_clear'].copy()
        # DNI
        df_dsw['dni_true'] = df_dsw['direct_n'].copy()
        df_dsw['dni_pred'] = df_dsw['dni_clear'].copy()

    df_dsws = df_dsw[['dsw_true', "dsw_pred", "dsw_aodc",
                      'dni_true', 'dni_pred', 'dni_aodc']]

    df_dsws.to_csv('AOD_correction/results/aodcor_{}_{}.csv'.format(site, sky))

def print_metric(pred, true, aodc):
    mae_dsw, mbe_dsw, rmse_dsw = evaluate_error(pred, true)
    mae_aod, mbe_aod, rmse_aod = evaluate_error(aodc, true)
    print(f"{'':10s} {'MAE':>10s} {'MBE':>10s} {'RMSE':>10s}")
    print(f"{'predict':10s} {mae_dsw:10.2f} {mbe_dsw:10.2f} {rmse_dsw:10.2f}")
    print(f"{'corrected':10s} {mae_aod:10.2f} {mbe_aod:10.2f} {rmse_aod:10.2f}")

def result_display(data_dir='', site='BON', sky='day'):
    df_dsws = pd.read_csv('AOD_correction/results/aodcor_{}_{}.csv'.format(site, sky))
    print('-' * 30, 'GHI', '-' * 30)
    print_metric(df_dsws['dsw_pred'], df_dsws['dsw_true'], df_dsws["dsw_aodc"])
    print('-' * 30, 'DNI', '-' * 30)
    print_metric(df_dsws['dni_pred'], df_dsws['dni_true'], df_dsws['dni_aodc'])
    print('-' * 30, 'DHI', '-' * 30)
    #print_metric(df_dsws['dhi_pred'], df_dsws['dhi_true'], df_dsws['dfi_aodc'])
    # plot_data_dw(df_dsws['dsw_true'], df_dsws['dsw_pred'], df_dsws['dsw_true'], df_dsws["dsw_aodc"], 'AODc',
    #              figlabel='AODc')
    plot_data_dw(df_dsws['dni_true'], df_dsws['dni_pred'], df_dsws['dni_true'], df_dsws["dni_aodc"], 'AODc',
                 figlabel='dni_AODc')

if __name__ == "__main__":
    sites = [
        ["BON", 40.05192, -88.37309, 230, 0.1240],
        # ["DRA", 36.62373, -116.01947, 1007, 0.0549],
        # ["FPK", 48.30783, -105.10170, 634, 0.1179],
        # ["GWN", 34.25470, -89.87290, 98, 0.1394],
        # ["PSU", 40.72012, -77.93085, 376, 0.1448],
        # ["SXF", 43.73403, -96.62328, 473, 0.1063],
        # ["TBL", 40.12498, -105.23680, 1689, 0.0694],
    ]
    data_dir = './GOES_tool/SURFRAD/preprocessed/'

    # for sky in ["clear", "cloudy", "day", "night"]:
    for sky in ["clear"]:
        print(sky)
        #df_result = pd.DataFrame()
        for site, lat, lon, alt, aod in sites:
            print(site)
            #result = aod_correction_clear(data_dir=data_dir, site=site, sky=sky)
            result_display(data_dir='./AOD_correction/results', site=site, sky=sky)
            #df_result = pd.concat((df_result, pd.DataFrame(result)))
            #plot_aod_correction(site=site, day='clear')
        #print(df_result)
