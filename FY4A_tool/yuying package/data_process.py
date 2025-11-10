import pandas as pd
import glob
import numpy as np
import pvlib
import os
from scipy import interpolate
from multiprocessing import Pool
from pandas.core.indexes.base import Index


def read_site(file):
    n_pixels = 11 * 11
    names = ["ts_created", "ts_start", "ts_end", "channel"]
    names += ["x{}".format(i) for i in range(n_pixels)]
    names += ["kappa0"]

    df = pd.read_csv(file, names=names)
    df = df.drop("kappa0", axis=1)

    df["ts_created"] = [x[:10] + ' ' + x[11:19] for x in df['ts_created'].values]

    return df


def merge_site(site, channel):
    filenames = glob.glob('data/goes/GOES/cropped_c{}/*{}*.csv'.format(channel, site))
    df = pd.concat([read_site(filename) for filename in filenames])
    df["ts_created"] = pd.to_datetime(df["ts_created"])
    df = df.sort_values(by="ts_created")
    n_pixels = 11 * 11
    df = df.set_index("ts_created")
    col = "x{}".format((n_pixels - 1) // 2)
    df = df[["channel", col]]
    df = df.rename(columns={col: "C{}".format(channel.zfill(2))})

    # convert radiance from [mW/(m^2 sr cm^-1)] to [W/(m^2 sr cm^-1)]
    df["C{}".format(channel.zfill(2))] /= 1e3

    # round up to the nearest 5-minute timestamp
    df = df.resample("5min", label="right").mean()

    df = df.reset_index()

    return df


def process_site(site, channels):
    df = merge_site(site, channels[0])
    for channel in channels[1:]:
        df_ = merge_site(site, channel)
        df = pd.merge(left=df, right=df_, how='outer', on='ts_created')


    # drop NAN values
    df = df.dropna()

    df = df.set_index("ts_created")

    df.to_hdf("data/goes/{}_radiance.h5".format(site), "df", mode="w")
    print(site, df.shape, df.index[0], df.index[-1])



if __name__ == "__main__":
    dnu = 0.10
    n_layers = 32
    AOD = 0.1243

    sites = [
        ["BON", 40.05192, -88.37309, 213],
        ["DRA", 36.62373, -116.01947, 1007],
        ["FPK", 48.30783, -105.10170, 634],
        ["GWN", 34.25470, -89.87290, 98],
        ["PSU", 40.72012, -77.93085, 376],
        ["SXF", 43.73403, -96.62328, 473],
        ["TBL", 40.12498, -105.23680, 1689],
    ]

    channels = ['{}'.format(i) for i in range(7, 16 + 1)]

    for site, lat, lon, alt in sites:
        print(site)

        # preprocess GOES data
        process_site(site, channels)