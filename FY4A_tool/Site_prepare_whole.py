
import netCDF4 as nc
import os
import numpy as np
import pandas as pd
import glob
from fun_goes_process import *
import pvlib


def read_site(file):
    n_pixels = 1 #11 * 11
    names = ["ts_created", "ts_start", "ts_end", "C01","C02","C03","C04,"
    ,"product","COD"]
    df = pd.read_csv(file)
    df["ts_created"] = [x[:10] + ' ' + x[11:19] for x in df['ts_created'].values]
    return df

def site_timesample(site, year, phase):
    if phase == 'clearsky':
        file_path = './ABI-L1b-RadC_cropped_Rad_2019/'
        #filenames = [file_path+'2019_a_clearsky_day_BON_Rad.csv']
        file_path = './SURFRAD/preprocessed/'
        filenames = glob.glob(file_path + '{}*{}_Rad_{}.csv'.format(str(year), site, phase))
    else:
        file_path = './ABI-L2-CODC_cropped_COD_2019/'
        filenames = glob.glob(file_path + '{}_{}_COD_{}_June.csv'.format(str(year), site, phase))
    df = pd.concat([read_site(filename) for filename in filenames])
    df["ts_created"] = pd.to_datetime(df["ts_created"])
    df = df.sort_values(by="ts_created")
    df.set_index("ts_created", inplace=True)
    # df = df.rename(columns={'Rad': "{}".format(channel.zfill(2))})
    df.drop(columns=["ts_start", "ts_end"], inplace=True)
    # round up to the nearest 5-minute timestamp
    # df = df.resample("5min", label="right").mean()
    # df = df.reset_index()
    df = df.select_dtypes(include=[np.number]).resample("5min", label="right").mean()
    #df = df.groupby('channel').resample('5min', label='right').mean()
    df = df.reset_index()
    df = df.sort_values(by="ts_created")
    #df.set_index("ts_created", inplace=True)
    return df


def process_site(site, year, hour, phase):
    df = merge_site(site, channels[0])
    for channel in channels[1:]:
        try:
            df_ = site_timesample(site, channel)
        except FileNotFoundError:
            print("File not found for site: {}, channel: {}".format(site, channel))
            continue
        df = pd.merge(left=df, right=df_, how='outer', on='ts_created')
        df_ = merge_site(site, channel)
        df = pd.merge(left=df, right=df_, how='outer', on='ts_created')
    # drop NAN values
    df.to_hdf("data/goes/{}_radiance.h5".format(site), "df", mode="w")
    print(site, df.shape, df.index[0], df.index[-1])
    return df

def make_a_whole_clearskyday(df):
    # 1. Extract date part only (drop time)
    df['Date'] = df['Time'].dt.date
    counts = df['Date'].value_counts()
    most_common_day = counts.idxmax()
    df_most = df[df['Date'] == most_common_day]

    # Step 2: Filter July 1 to July 30
    mask_july = (df['Time'].dt.date >= pd.to_datetime('2019-07-01').date()) & \
                (df['Time'].dt.date <= pd.to_datetime('2019-10-30').date())
    df_july = df[mask_july].copy()
    target_day = pd.to_datetime('2019-07-12').date()

    collected_rows = []
    for hour in [14, 15, 16, 17]:
        best_match = None
        min_distance = float('inf')
        # Find closest day with the desired hour
        for date, group in df_july.groupby(df_july['Time'].dt.date):
            if hour in group['Time'].dt.hour.values:
                dist = abs((date - target_day).days)
                if dist < min_distance:
                    min_distance = dist
                    best_match = date

        if best_match:
            # Collect all rows from that date for the target hour
            match_rows = df_july[
                (df_july['Time'].dt.date == best_match) &
                (df_july['Time'].dt.hour == hour)
                ]
            collected_rows.append(match_rows)

    # Combine into a single DataFrame
    df_fill = pd.concat(collected_rows).sort_values('Time')
    df = pd.concat([df_most, df_fill]).sort_values('Time')
    decimal_hour = df['Time'].dt.hour + df['Time'].dt.minute / 60 + df['Time'].dt.second / 3600
    plt.scatter(decimal_hour, df['Site_dsw'], label='Site DSW')
    plt.xlabel('Hour of Day')
    plt.ylabel('Site DSW')
    plt.legend()
    plt.show()
    for day in df['Time'].dt.date.unique():
        hours = df[df['Time'].dt.date == day]['Time'].dt.hour.unique()
        print(f"{day}: {len(hours)} unique hours → {sorted(hours)}")
    return df

def read_csv(file, df_site, phase):
    # read the csv file
    if phase == 'clearsky':
        df = pd.read_hdf(file, 'df')
        df = df.reset_index()
        df = df.rename(columns={"index": "Time"})
        df.columns = ['Time', 'Site_zen', 'Site_dsw', 'direct_n', 'diffuse', 'dw_ir',
                      'temp', 'rh', 'windspd', 'pressure', 'ghi_clear','dni_clear','clearsky']
        df = df[['Time', 'Site_zen', 'Site_dsw', 'direct_n', 'diffuse', 'dw_ir',
                 'temp', 'rh']]
        df = make_a_whole_clearskyday(df)

    else:
        df = pd.read_csv(file, header=0)
        # rename the columns
        df.columns = ['Time', 'Site_zen', 'Site_dsw', 'direct_n', 'diffuse', 'dw_ir',
                      'temp', 'rh', 'windspd', 'pressure']
        df = df[['Time', 'Site_zen', 'Site_dsw', 'direct_n', 'diffuse', 'dw_ir',
                 'temp', 'rh']]
        df['Time'] = pd.to_datetime(df['Time'])
    # Get the date from df_site (assuming it's a single row)
    df_site['Time'] = pd.to_datetime(df_site['ts_created'])
    # date for filtering
    #date_to_filter = df_site['Time'][0].date()
    #df_filtered = df[df['Time'].dt.date == date_to_filter]
    # Hour for filtering
    # min_hour = df_site['Time'].dt.hour.min()
    # max_hour = df_site['Time'].dt.hour.max()
    hours =  df_site['Time'].dt.hour.unique()
    df_filtered = df[df['Time'].dt.hour.isin(hours)]
    # If you want to merge on time instead (if there are matching times)
    # df_combined = pd.merge(df_site, df_filtered, on='Time', how='left')
    df_filtered.set_index("Time", inplace=True)
    #df_filtered = df_filtered.resample("5min", label="right").mean()
    df_filtered = df_filtered.reset_index()

    return df_filtered

def match_ground(site, df_site, phase,  lat, lon, alt):
    year = 2019
    if phase == 'clearsky':
        file_path = './SURFRAD/preprocessed/'
        filenames = glob.glob(file_path + '{}_{}.h5'.format(site, phase))
    else:
        file_path = './SURFRAD/'
        filenames = glob.glob(file_path + f'{year}_{site.lower()}_solar_data.csv')
        # filenames = f'{int(YEAR)}_{site["name"]}_{product_name}_{phase}_hour={hour}.

    df = pd.concat([read_csv(filename, df_site, phase) for filename in filenames])

    # convert the timestamp to datetime format
    df_site['Time'] = pd.to_datetime(df_site['ts_created'])
    df['Time'] = pd.to_datetime(df['Time'])

    df_filtered = df[df['Time'].isin(df_site['Time'])]
    # extract clear sky
    # if phase == 'clearsky':
    #     start = datetime.datetime(2019, 7, 13, 9, 0)
    #     end = datetime.datetime(2019, 7, 14, 6, 0)
    #     df_filtered = df_filtered[(df_filtered['Time'] >= start) & (df_filtered['Time'] < end)]
    #     df_filtered.set_index('Time', inplace=True)
        # df_clearsky = extract_clearsky_periods(df_filtered,  lat, lon, alt)
        # # import pvlib
        # lat, lon, alt = 40.05192, -88.37309, 213
        # location = pvlib.location.Location(lat, lon, 'UTC', alt)
        # clearsky = location.get_clearsky(df_filtered.index)
        # clearsky_dsw = clearsky['ghi'].resample('5min').mean()
        # df_filtered = df_filtered[df_filtered['Site_dsw']> clearsky_dsw]
        # df_filtered['Site_dsw'].plot(label='BON')
        # clearsky_dsw.plot(label='pvlib clearsky')
        # df_clearsky['Site_dsw'].plot(label='clearsky periods', color='red')
        # plt.tight_layout()
        # plt.legend()
        # plt.show()
        # df_combined = pd.merge(df_filtered, df_site, on='Time', how='left')
    #else:
    df_combined = pd.merge(df_filtered, df_site,   on='Time', how='left')
    df_combined.drop(columns=['ts_created'], inplace=True)

    df_combined.set_index('Time', inplace=True)
    df_combined = df_combined.sort_values(by=['Time'])
    return df_combined


if __name__ == "__main__":
    sites = [["BON", 40.05192, -88.37309, 213],
             ["DRA", 36.62373, -116.01947, 1007],
            ["FPK", 48.30783, -105.10170, 634],
            ["GWN", 34.25470, -89.87290, 98],
            ["PSU", 40.72012, -77.93085, 376],
            ["SXF", 43.73403, -96.62328, 473],
            ["TBL", 40.12498, -105.23680, 1689],
             ]
    #channels = ['C02']

    year= '2019'
    hour = None
    phase = 'water' #'water'
    channels = ['C0{}'.format(i) for i in range(1, 6 + 1)]
    for site, lat, lon, alt in sites:
        print(site)
        try:
            df = site_timesample(site, year, phase)
        except Exception:
            print("File not found for site: {}, phase: {}".format(site, phase))
            continue
        df = df.dropna()
        df_site = match_ground(site, df, phase, lat, lon, alt)

        sky = 'day'
        filename = "../GOES_data/GOES16_site_sat_data/GOES_{}_{}_radiance_satellite_water_2019June.csv".format(sky,site)
        df_site.to_csv(filename)
        print('Finish matching ground data for site:', site)
        print('Data saved to:', filename)

