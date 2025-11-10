"""Summary statistics."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from tqdm import tqdm


def evaluate_error(pre, true):
    error_dlw = (pre - true).dropna(how="any").values
    mae_dlw = np.mean(np.abs(error_dlw))
    mbe_dlw = np.mean(error_dlw)
    rmse_dlw = np.sqrt(np.mean(error_dlw ** 2))
    return mae_dlw, mbe_dlw, rmse_dlw


def table_olr_dlw(data_dir, sky="clear"):
    """Summary statistics table for OLR and DLW."""

    # sites = ["BON", "DRA", "FPK", "GWN", "PSU", "SXF", "TBL"]
    # sites = ["BON", ]
    sites = [
        ["BON", 40.05192, -88.37309, 230],
        ["DRA", 36.62373, -116.01947, 1007],
        ["FPK", 48.30783, -105.10170, 634],
        ["GWN", 34.25470, -89.87290, 98],
        ["PSU", 40.72012, -77.93085, 376],
        ["SXF", 43.73403, -96.62328, 473],
        ["TBL", 40.12498, -105.23680, 1689],
    ]


    results = []
    results_aod = []
    results_altm = []
    results_altp = []
    results_acm = []
    results_acp = []
    # for site in sites:
    for site, lat, lon, alt in sites:
        # DLW [W/m^2]
        plt.rc('font', family='Times New Roman')
        dlw = pd.read_hdf(os.path.join(data_dir, "./results/timeseries_{}_{}_dlw_multilayer_cor.h5".format(site, sky)), "df")

        # evaluate
        mae_dlw, mbe_dlw, rmse_dlw = evaluate_error(dlw["dwir_pred"], dlw['dwir_true'])
        mae_aod, mbe_aod, rmse_aod = evaluate_error(dlw["dwir_aodc"], dlw['dwir_true'])
        mae_altm, mbe_altm, rmse_altm = evaluate_error(dlw["dwir_altcm"], dlw['dwir_true'])
        mae_altp, mbe_altp, rmse_altp = evaluate_error(dlw["dwir_altcp"], dlw['dwir_true'])
        mae_acm, mbe_acm, rmse_acm = evaluate_error(dlw["dwir_acm"], dlw['dwir_true'])
        mae_acp, mbe_acp, rmse_acp = evaluate_error(dlw["dwir_acp"], dlw['dwir_true'])


        results.append({"site": site, "sky": sky, "channel": "DLW", "MAE": mae_dlw, "MBE": mbe_dlw, "RMSE": rmse_dlw})
        results_aod.append({"site": site, "sky": sky, "channel": "DLW", "MAE": mae_aod, "MBE": mbe_aod, "RMSE": rmse_aod})
        results_altm.append({"site": site, "sky": sky, "channel": "DLW", "MAE": mae_altm, "MBE": mbe_altm, "RMSE": rmse_altm})
        results_altp.append({"site": site, "sky": sky, "channel": "DLW", "MAE": mae_altp, "MBE": mbe_altp, "RMSE": rmse_altp})
        results_acm.append({"site": site, "sky": sky, "channel": "DLW", "MAE": mae_acm, "MBE": mbe_acm, "RMSE": rmse_acm})
        results_acp.append({"site": site, "sky": sky, "channel": "DLW", "MAE": mae_acp, "MBE": mbe_acp, "RMSE": rmse_acp})


    # summary
    df_result = pd.DataFrame(results)
    df_result_aod = pd.DataFrame(results_aod)
    df_result_altm = pd.DataFrame(results_altm)
    df_result_altp = pd.DataFrame(results_altp)
    df_result_acm = pd.DataFrame(results_acm)
    df_result_acp = pd.DataFrame(results_acp)


    # df_result.to_csv("./results/results_table_{}_olr_dlw.csv".format(sky), index=False)
    print(sky, df_result.shape)
    df_result = pd.pivot_table(df_result, index=["site", "sky"], columns="channel", values="RMSE")
    print(df_result.round(2))   # rRMSE [-]
    #print(df.round(3) * 100)   # rRMSE [%]

def calculate_pressures(df):
    df['P_s'] = 610.94 * np.exp((17.625 * (df['T'] - 273.15)) / (df['T'] - 30.11))
    df['P_w'] = (df['RH'] / 100.0) * df['P_s']
    return df

def perform_regression(df):
    model = LinearRegression()
    x = df['P_w'].values.reshape(-1, 1)
    y = df['delta_dlw']
    regr = model.fit(x, y)
    return regr

def result_correction(data_dir='.', site='BON', sky='day', alt=None, aod=None):
    results = []
    for sky in ['clear']:
        timeofday = 'night' if sky == 'night' else 'day'

        df_dlw = pd.read_hdf(os.path.join(data_dir, "./results/timeseries_{}_{}_dlw_multilayer.h5".format(site, sky)), "df")
        df_dlw[df_dlw._get_numeric_data() < 0] = np.nan  # abnormal data if exist

        dlw_aod = pd.read_hdf("./results/results_32layers_0.10_dlw_{}_aods.h5".format(timeofday), "df")

        # calculate pressure
        dlw_sta = calculate_pressures(dlw_aod.loc[dlw_aod['AOD'] == aod].copy())
        dlw_ori = calculate_pressures(dlw_aod.loc[dlw_aod['AOD'] == 0.1243].copy())

        # LBL delat dlw  [dlw(sta)-dlw(0.1243)]
        dlw_sta = dlw_sta.rename(columns={'dwir': 'dwir_sta'}).set_index(['P_w', 'RH', 'T'])
        dlw_ori = dlw_ori.rename(columns={'dwir': 'dwir_ori'}).set_index(['P_w', 'RH', 'T'])
        dfs = pd.concat([dlw_sta[['dwir_sta']], dlw_ori[['dwir_ori']]], axis=1, join='inner').reset_index()
        dfs['delta_dlw'] = dfs['dwir_sta'] - dfs['dwir_ori']

        # linear regression
        regr = perform_regression(dfs)

        df_dlw['delta'] = regr.predict(df_dlw['P_w'].values.reshape(-1, 1))
        df_dlw['dwir_aodc'] = df_dlw['dwir_pred']+df_dlw['delta']


        # evaluate
        mae_dlw, mbe_dlw, rmse_dlw = evaluate_error(df_dlw["dwir_pred"], df_dlw['dwir_true'])
        mae_aod, mbe_aod, rmse_aod = evaluate_error(df_dlw["dwir_aodc"], df_dlw['dwir_true'])

        results.append({"site": site, "sky": sky, "channel": "DLW", "MAE": mae_dlw, "MBE": mbe_dlw, "RMSE": rmse_dlw})
        results.append(
            {"site": site, "sky": sky, "channel": "DLW_aod", "MAE": mae_aod, "MBE": mbe_aod, "RMSE": rmse_aod})
    results = pd.DataFrame(results)
    return results


def read_cod(site):
    df = pd.read_csv('./data/AOD_SURFRAD/{}_2019.csv'.format(site.lower()))
    df.columns = ['time', 'aod']
    print(site, df['aod'].mean())

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


def correction(dlw, aod):
    dwirc = []
    for idx in tqdm(range(len(dlw))):
        dwir_cor = interpolate.griddata(
            aod[["T", "RH", "AOD"]].values,
            aod["dwir"].values,
            dlw[["T", "RH", "aod"]].iloc[idx].values,  # AOD time series
            method="linear"
        )
        print(dwir_cor[0])
        dwirc.append(dwir_cor[0])
    return dwirc


def aod_correction(data_dir='.', site='BON', sky='day'):
    results = []

    timeofday = 'night' if sky == 'night' else 'day'

    # SCOPE results under clear sky condition
    dlw = pd.read_hdf(os.path.join(data_dir, "./results_scope1.0/timeseries_{}_{}_dlw_multilayer.h5".format(site, sky)), "df")
    dlw[dlw._get_numeric_data() < 0] = np.nan  # abnormal data if exist

    # aod time series from surfrad
    df_aod = read_cod(site)

    # concate previous SCOPE results & aod series
    c_index = [index for index in dlw.index if index in df_aod.index]
    df_dlw = pd.concat([dlw.loc[c_index], df_aod.loc[c_index]], join='inner', axis=1)

    # LBL results for different AOD under clear sky condition
    dlw_aod = pd.read_hdf("./results/results_32layers_0.10_dlw_{}_aods.h5".format(timeofday), "df")
    print(sorted(dlw_aod['AOD'].unique()))

    if sky == 'clear':
        df_dlw['dwir_aodc'] = correction(df_dlw, dlw_aod)
    else:
        df_clear = df_dlw.loc[(df_dlw["COD"] == 0) & (df_dlw["kap"] == "0"), :]
        df_clear['dwir_aodc'] = correction(df_clear, dlw_aod)

        df_cloudy = df_dlw[~df_dlw.index.isin(df_clear.index)]
        df_cloudy['dwir_cor'] = df_cloudy["dwir_pred"]

        # merge
        df_dlw = pd.concat([df_clear, df_cloudy]).sort_index()

    df_dlw = df_dlw.dropna()

    # save corrected dlw
    df_dlws = df_dlw[['dwir_true', "dwir_pred", "dwir_aodc"]]
    df_dlws.to_csv('./results/aodcor_{}_{}.csv'.format(site, sky))

    # evaluate
    mae_dlw, mbe_dlw, rmse_dlw = evaluate_error(df_dlw["dwir_pred"], df_dlw['dwir_true'])
    mae_aod, mbe_aod, rmse_aod = evaluate_error(df_dlw["dwir_aodc"], df_dlw['dwir_true'])

    results.append({"site": site, "sky": sky, "channel": "DLW", "MAE": mae_dlw, "MBE": mbe_dlw, "RMSE": rmse_dlw})
    results.append({"site": site, "sky": sky, "channel": "DLW_aod", "MAE": mae_aod, "MBE": mbe_aod, "RMSE": rmse_aod})
    results = pd.DataFrame(results)
    print(results)
    return results


if __name__ == "__main__":
    sites = [
        ["BON", 40.05192, -88.37309, 230, 0.1240],
        ["DRA", 36.62373, -116.01947, 1007, 0.0549],
        ["FPK", 48.30783, -105.10170, 634, 0.1179],
        ["GWN", 34.25470, -89.87290, 98, 0.1394],
        ["PSU", 40.72012, -77.93085, 376, 0.1448],
        ["SXF", 43.73403, -96.62328, 473, 0.1063],
        ["TBL", 40.12498, -105.23680, 1689, 0.0694],
    ]

    # for sky in ["clear", "cloudy", "day", "night"]:
    for sky in ["clear"]:
        print(sky)
        df_result = pd.DataFrame()
        for site, lat, lon, alt, aod in sites:
            result = aod_correction(data_dir='.', site=site, sky=sky)
            df_result = pd.concat((df_result, pd.DataFrame(result)))
        print(df_result)
        # df_result.to_csv('./results/results_{}_aodcor.csv'.format(sky), index=False)




