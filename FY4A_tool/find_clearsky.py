
import glob
import pandas as pd
import numpy as np
import pvlib


def preprocess_clearsky_periods(year, site, lat, lon, alt):
    """
    Find the clearsky periods.

    Parameters
    ----------
    site : str
        SURFRAD station name.
    lat, lon, alt : float
        The latitude, longitude, and altitude of the site.

    """

    # the load the pre-processed ground data
    files = glob.glob('./SURFRAD/{}*{}*.csv'.format(str(year), site.lower()))
    df = pd.read_csv(files[0])


    # resample
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df = df.resample("5min", closed="right", label="right").mean()
    df = df.dropna(how="any")


    # split into day/night using solar angle
    zenith_threshold = 85  # zenith threshold [deg] for day/night split
    df_night = df[df["zen"] > zenith_threshold]
    #df_night.to_hdf("data/surfrad/{}_night.h5".format(site), "df", mode="w")
    print("Night:", df_night.shape, df_night.index[0], df_night.index[-1])
    df = df[df["zen"] <= zenith_threshold]

    # add clearsky irradiance
    loc = pvlib.location.Location(lat, lon, altitude=alt)
    cs = loc.get_clearsky(df.index)
    df.insert(df.shape[1], 'ghi_clear', cs["ghi"])
    df.insert(df.shape[1], 'dni_clear', cs["dni"])

    # GHI and DNI threshold values: G, M, L, sig, S
    #thresholds = [75, 75, 10, 0.005, 8]      # Reno's 2016 paper
    thresholds = [100, 100, 50, 0.01, 10.0]  # Rich's paper

    # rolling window
    i_window = 30  # window size [mins]
    criteria_sum = np.zeros(df.shape[0])
    for i in range(i_window, df.shape[0], i_window):

        # backwards selection: (i_start, i_end]
        i_start = i - i_window + 1
        i_end = i + 1
        ghi = df.iloc[i_start:i_end]['dw_solar'].values
        ghi_clear = df.iloc[i_start:i_end]['ghi_clear'].values

        # clearsky criteria
        criteria = np.zeros(5)

        # average irradiance
        criteria[0] = (np.abs(np.nanmean(ghi) - np.nanmean(ghi_clear)) < thresholds[0])

        # max irradiance
        criteria[1] = np.abs(np.max(ghi) - np.max(ghi_clear)) < thresholds[1]

        # irradiance increment
        diff_ghi = np.diff(ghi)
        diff_ghi_clear = np.diff(ghi_clear)
        criteria[2] = (
            np.abs(np.sum(np.sqrt(diff_ghi ** 2)) - np.sum(np.sqrt(diff_ghi_clear ** 2)))
            < thresholds[2]
        )

        # std of irradiance increment
        criteria[3] = np.abs(
            np.std(diff_ghi) / np.mean(ghi)
            - np.std(diff_ghi_clear) / np.mean(ghi_clear)
        ) < thresholds[3]

        # max irradiance increment
        criteria[4] = np.max(np.abs(diff_ghi - diff_ghi_clear)) < thresholds[4]

        # sum of the criteria (5=all criteria met)
        criteria_sum[i_start:i_end] = criteria.sum()

    # if all criteria are met, then the sky is clear
    df.insert(df.shape[1], 'clearsky', criteria_sum == 5)
    df.reset_index(inplace=True)
    # export data
    df.to_hdf(f"./SURFRAD/preprocessed/{site}_day.h5", "df", mode="w")
    df[df['clearsky'] == True].to_hdf('./SURFRAD/preprocessed/{}_clear.h5'.format(site), 'df', mode="w",)
    df[df['clearsky'] == False].to_hdf('./SURFRAD/preprocessed/{}_cloudy.h5'.format(site), 'df', mode="w")
    print(site, df.shape, df.index[0], df.index[-1])
    n_clear = df[df["clearsky"] == True].shape[0]
    n_cloudy = df[df["clearsky"] == False].shape[0]
    n_day = df.shape[0]
    print("clear = {:>6,d} ({:>3.1%}), cloudy = {:>12,d} ({:>3.1%})".format(n_clear, n_clear / n_day, n_cloudy, n_cloudy / n_day))
    #print("Clear: ", df[df['clearsky'] == True].shape)
    #print("Cloudy:", df[df['clearsky'] == False].shape)
    print(df.columns)