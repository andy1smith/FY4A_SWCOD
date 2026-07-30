import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
from pathlib import Path
from mcd43a1_albedo import black,white


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
BJC_AOD_PATH = REPO_ROOT / 'AOD_correction' / 'AERONET_china' / '2021_BJC_CAMS.csv'
CERN_GHI_PATHS = [
    BASE_DIR / 'CERN' / 'CERN_instGHI_2021_UTC.csv',
    Path('/Volumes/HP P900/CERN_instGHI_2021_UTC.csv'),
]
CARSNET_AOD_PATHS = [
    REPO_ROOT / 'AOD_correction' / 'CARSNET_data' / 'cern_to_carsnet_aod_match_excluding_BJC_497p5nm_alpha1p3.csv',
    REPO_ROOT / 'AOD_correction' / 'CARSNET_data' / 'annual_site_summary' / 'cern_to_carsnet_aod_match_excluding_BJC_497p5nm_alpha1p3.csv',
]
MCCLEAR_DIR = REPO_ROOT / 'FY4A_data' / 'McClear_clearsky'
CLEAR_SKY_QC_SOURCE = os.environ.get('FY4A_CLEAR_SKY_QC_SOURCE', 'pvlib').lower()
CLOUDY_OUTPUT_DIR = Path(os.environ.get('FY4A_CLOUDY_OUTPUT_DIR', REPO_ROOT / 'FY4A_data'))
CLOUDY_GROUND_SOURCE = os.environ.get('FY4A_CLOUDY_GROUND_SOURCE', 'cloudy').lower()

_BJC_AOD = None
_CARSNET_AOD = None

MAX_SZA_DIFF_DEG = 1.0

def cloudy_time_qc_mask(xr_sat, df_site):
    sat_sza_median = xr_sat['Sun_Zen'].median(dim=('y', 'x'), skipna=True).to_pandas()
    ground_sza = df_site['Sun_Zen'].astype(float)
    sza_ok = (sat_sza_median - ground_sza).abs() <= MAX_SZA_DIFF_DEG # make sure sat is in the same time with ground

    return sza_ok.fillna(False)


def cloudy_sky_sources():
    if CLOUDY_GROUND_SOURCE == 'cloudy':
        return ['cloudy']
    raise ValueError(
        f"Unsupported FY4A_CLOUDY_GROUND_SOURCE={CLOUDY_GROUND_SOURCE!r}; "
        "use 'cloudy'. Cloudy processing must come only from *_cloudy.h5."
    )


def read_ground_for_sky(site, sky):
    ground_dir = './Ground/preprocessed_GHI/'
    if sky != 'cloudy':
        ground_path = ground_dir + '{}_{}.h5'.format(site, sky)
        df_ground = pd.read_hdf(ground_path, key='df')
        df_ground['Time'] = pd.to_datetime(df_ground['Time'])
        df_ground.set_index('Time', inplace=True)
        return df_ground

    ground_path = ground_dir + '{}_cloudy.h5'.format(site)
    df_ground = pd.read_hdf(ground_path, key='df')
    df_ground['Time'] = pd.to_datetime(df_ground['Time'])
    df_ground['cloudy_ground_source'] = 'cloudy'
    df_ground = df_ground.sort_values('Time')
    df_ground = df_ground.drop_duplicates(subset='Time', keep='first')
    df_ground.set_index('Time', inplace=True)
    return df_ground


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




def read_satellite_2Dmap(site, source_skies=None):
    # extract whole 2D map time series for all channels
    if source_skies is None:
        source_skies = ['cloudy']

    channels = ['SunZenith', 'SunAzimuth', 'SatelliteAzimuth', 'SatelliteZenith', 'SunGlintAngle', 'elevation'] +\
    ['Channel{:02d}'.format(i+1) for i in range(6)]

    names =  ['Sun_Zen', 'Sun_Azi', 'Sat_Azi', 'Sat_Zen','Sun_Gli', 'ele'] +\
            ["C{:02d}".format(i+1) for i in range(6)]

    # Prepare storage for shapes and arrays
    var_data = {}
    times = None
    n_pixels = 121

    for channel, name in zip(channels, names):
        channel_frames = []
        for source_sky in source_skies:
            csv_path = f'./cropped_FY2021_{source_sky}/{site}/{site}_{channel}.csv'
            if not os.path.exists(csv_path):
                continue
            df_source = pd.read_csv(csv_path)
            df_source["time"] = pd.to_datetime(df_source["time"])
            channel_frames.append(df_source)
        if not channel_frames:
            raise FileNotFoundError(
                f"No cropped FY4A {channel} CSV found for {site} in sources {source_skies}."
            )
        df = pd.concat(channel_frames, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset="time", keep="first")
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
    ghi_path = next((path for path in CERN_GHI_PATHS if path.exists()), None)
    if ghi_path is None:
        searched = ', '.join(str(path) for path in CERN_GHI_PATHS)
        raise FileNotFoundError(f"No CERN GHI file found. Searched: {searched}")
    data = pd.read_csv(ghi_path)
    if site not in data.columns:
        raise KeyError(f"Site {site} not found in {ghi_path}")
    df = data[[site]]
    df = df.rename(columns={site: 'ghi'})
    df['time'] =  pd.date_range(start='2021-01-01', end='2022-01-01', freq='h')[:-1]  # generate hourly timestamps for 2021
    df = df.set_index('time').sort_index()

    return df


def read_mcclear_clearsky(site):
    paths = sorted(MCCLEAR_DIR.glob(f'{site}_mcclear_*_hourly.csv'))
    if not paths:
        raise FileNotFoundError(
            f"No McClear cache found for {site} under {MCCLEAR_DIR}. "
            "Run Sat_Preprocessing/download_mcclear_clearsky.py first."
        )
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if 'Time' not in df.columns:
            raise ValueError(f"{path} is missing Time column")
        if 'ghi_clear_mcclear' not in df.columns:
            if 'ghi_clear' in df.columns:
                df = df.rename(columns={'ghi_clear': 'ghi_clear_mcclear'})
            else:
                raise ValueError(f"{path} is missing ghi_clear_mcclear column")
        keep_cols = ['Time', 'ghi_clear_mcclear']
        for col in ['dni_clear_mcclear', 'dhi_clear_mcclear', 'bhi_clear_mcclear', 'ghi_extra_mcclear']:
            if col in df.columns:
                keep_cols.append(col)
        df = df[keep_cols].copy()
        df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize(None)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['Time', 'ghi_clear_mcclear'])
    df = df.drop_duplicates(subset='Time', keep='first').sort_values('Time')
    return df.set_index('Time')


def add_mcclear_to_site(site, df_combined):
    df = df_combined.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    if 'ghi_clear' in df.columns and 'ghi_clear_pvlib' not in df.columns:
        df['ghi_clear_pvlib'] = df['ghi_clear']
    for col in [
        'ghi_clear_mcclear',
        'dni_clear_mcclear',
        'dhi_clear_mcclear',
        'bhi_clear_mcclear',
        'ghi_extra_mcclear',
        'clear_index_mcclear',
    ]:
        if col in df.columns:
            df = df.drop(columns=col)

    before = len(df)
    mcclear = read_mcclear_clearsky(site)
    df = df.sort_values('Time').set_index('Time')
    df = df.join(mcclear, how='left')
    matched = int(df['ghi_clear_mcclear'].notna().sum())
    if matched == 0:
        raise ValueError(f"McClear cache for {site} did not match any cloudy ground/FY4A times.")

    df['ghi_clear'] = df['ghi_clear_mcclear']
    df['clear_index_mcclear'] = df['ghi'] / df['ghi_clear_mcclear'].replace(0, np.nan)
    df['clear_sky_qc_source'] = 'mcclear'
    df = df.reset_index()
    print(f"{site} McClear matched {matched}/{before} rows.")
    return df


def read_bjc_aod():
    global _BJC_AOD
    if _BJC_AOD is None:
        df = pd.read_csv(BJC_AOD_PATH)
        df = df.rename(columns={'time': 'Time', 'AOD_500nm': 'aod'})
        required_cols = {'Time', 'aod'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{BJC_AOD_PATH} is missing required columns: {sorted(missing)}")
        df['Time'] = pd.to_datetime(df['Time'])
        df = df[['Time', 'aod']].dropna()
        df = df.set_index('Time')
        df = df[~df.index.duplicated(keep='first')]
        _BJC_AOD = df.sort_index()
    return _BJC_AOD.copy()


def read_carsnet_aod():
    global _CARSNET_AOD
    if _CARSNET_AOD is None:
        aod_path = next((path for path in CARSNET_AOD_PATHS if path.exists()), None)
        if aod_path is None:
            searched = ', '.join(str(path) for path in CARSNET_AOD_PATHS)
            raise FileNotFoundError(f"No CARSNET AOD match file found. Searched: {searched}")
        df = pd.read_csv(aod_path)
        required_cols = {'cern_site', 'suggested_AOD_fixed_497p5nm'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{aod_path} is missing required columns: {sorted(missing)}")
        df = df[['cern_site', 'suggested_AOD_fixed_497p5nm']].dropna()
        df = df.drop_duplicates(subset='cern_site', keep='first')
        _CARSNET_AOD = df.set_index('cern_site')['suggested_AOD_fixed_497p5nm']
    return _CARSNET_AOD.copy()


def add_aod_to_site(site, df_combined, drop_missing=True, missing_fill_value=None):
    df = df_combined.copy()
    df['Time'] = pd.to_datetime(df['Time'])
    if site == 'BJC':
        before = len(df)
        df_aod = read_bjc_aod()
        df = df.sort_values('Time').set_index('Time')
        df = pd.merge_asof(
            df,
            df_aod,
            left_index=True,
            right_index=True,
            direction='nearest',
            tolerance=pd.Timedelta('3min')
        )
        matched = int(df['aod'].notna().sum())
        if drop_missing:
            df = df.dropna(subset=['aod'])
        elif missing_fill_value is not None:
            df['aod'] = df['aod'].fillna(float(missing_fill_value))
        df = df.reset_index()
        filled = len(df) - matched if not drop_missing and missing_fill_value is not None else 0
        fill_msg = f"; filled {filled} missing AOD with {missing_fill_value:g}" if filled else ""
        print(f"BJC AOD matched {matched}/{before} rows within 3 minutes{fill_msg}.")
        return df

    aod_lookup = read_carsnet_aod()
    if site not in aod_lookup.index:
        print(f"No CARSNET fixed AOD found for {site}; filling aod with NaN.")
        df['aod'] = np.nan
        return df
    df['aod'] = float(aod_lookup.loc[site])
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
    filenames = [
        'CERN2021-MCD43A1-061-results.csv',
        'CERN34-MCD43A1-061-results.csv'
    ]
    mcd43_dfs = []
    for filename in filenames:
        file_path = os.path.join(file_dir, filename)
        if os.path.exists(file_path):
            mcd43_dfs.append(pd.read_csv(file_path))
    if not mcd43_dfs:
        raise FileNotFoundError(f"No MCD43A1 albedo files found in {file_dir}: {filenames}")
    mcd43_df = pd.concat(mcd43_dfs, ignore_index=True)
    xsf_df = mcd43_df[mcd43_df['Category'] == site].copy()

    FY4A_channels_map = {
        'C01': 3,  # Blue ~0.47 µm → MODIS Band 3 (0.459–0.479)
        'C02': 1,  # Red  ~0.64 µm → MODIS Band 1 (0.620–0.670)
        'C03': 2,  # NIR  ~0.86 µm → MODIS Band 2 (0.841–0.876)
        'C04': 5,  # SWIR ~1.38 µm → MODIS Band 5 (1.24 µm)
        'C05': 6,  # SWIR ~1.6 µm → MODIS Band 6
        'C06': 7  # SWIR ~2.2 µm → MODIS Band 7
    }

    def add_empty_albedo_columns(df):
        df = df.copy()
        for ch in FY4A_channels_map:
            for suffix in ['p0', 'p1', 'p2']:
                df[f'Abdo_{ch}_{suffix}'] = np.nan
            df[f'BSA_{ch}'] = np.nan
            df[f'WSA_{ch}'] = np.nan
        return df

    if 'Time' in df_combined.columns:
        df_filter = df_combined.copy()
    else:
        df_filter = df_combined.reset_index().copy()
        if 'index' in df_filter.columns:
            df_filter = df_filter.rename(columns={'index': 'Time'})
    df_filter['Time'] = pd.to_datetime(df_filter['Time'])

    if xsf_df.empty:
        print(f"No MODIS albedo rows found for {site}; saving satellite/ground rows with NaN albedo.")
        return add_empty_albedo_columns(df_filter)

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

    if xsf_df.empty:
        print(f"No valid MODIS albedo rows found for {site} after QA; saving satellite/ground rows with NaN albedo.")
        return add_empty_albedo_columns(df_filter)

    # create a pandas to save df_combined + mcd43a1
    # df_filter['D_portion'] = (
    #         df_filter['diffuse'] / df_filter['Site_dsw']
    # ).clip(0.0, 1.0)

    # 1. Prepare the Join Key in your high-frequency dataframe
    # dt.normalize() converts "2023-01-01 12:30:00" -> "2023-01-01 00:00:00" for matching.
    df_filter['Join_Date'] = df_filter['Time'].dt.normalize()
    # 2. Build the exact list of columns to keep
    # Start ONLY with the merge key
    cols_to_fetch = ['Date']
    rename_map = {}

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
    required_albedo_cols = [col for col in rename_map.values() if col in df_final.columns]
    df_final = df_final.dropna(subset=required_albedo_cols)
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
    # sites = ['CSA', 'DHL', 'FKD', 'FQA', 'HLA', 'JZB', 'LCA', 'NMD', 'SJM', 'THL', 'YCA',
    # 'AKA', 'ALF', 'ASA', 'BJC', 'BJF', 'BNF', 'CBF', 'CLD', 'CSA',
    #  'CWA', 'DHF', 'DHL', 'DTL', 'DYB', 'ESD', 'FKD', 'FQA', 'GGF',
    #  'GGS', 'HBG', 'HJA', 'HLA', 'HSF', 'HTF', 'JZB', 'LCA', 'LSA',
    #  'LZD', 'MXF', 'NMD', 'NMG', 'PDF', 'QYA', 'QYF', 'SJM', 'SNF',
    #  'SPD', 'SYA', 'SYB', 'THL', 'TYA', 'YCA', 'YGA', 'YTA']
    df = pd.read_csv('../FY4A_data/' + 'CERN_info.csv')
    sites = df['site'].tolist()
    sky = 'cloudy'

    for site in sites:
        # load CERN ghi data [W/m2]
        try:
            if sky in ['clear','cloudy']:
                df_ground = read_ground_for_sky(site, sky)
            else:
                df_ghi = read_ghi(site)
        except (FileNotFoundError, KeyError) as exc:
            print(f"Input not found for site {site}, skipping. {exc}")
            continue

        # load NoAA RH & T measurement
        df_mea = read_measures(site)
        if sky in ['clear','cloudy']:
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
        # Clear-sky validation keeps the center-pixel CSV workflow.
        # Cloudy cases keep the full 11x11 FY4A map in NetCDF.
        extract2D = sky == 'cloudy'
        if extract2D:
            try:
                xr_sat = read_satellite_2Dmap(site, cloudy_sky_sources())
            except FileNotFoundError as exc:
                print(f"Cloudy FY4A 2D extraction not found for site {site}, skipping. {exc}")
                continue

            data = df1d.reset_index()
            if 'index' in data.columns:
                data = data.rename(columns={'index': 'Time'})
            data = data.sort_values(by='Time')

            # Match the clear branch: add AOD, MODIS BRDF/albedo, and keep only daylit cases.
            data = add_aod_to_site(site, data, drop_missing=False, missing_fill_value=0.125)
            if CLEAR_SKY_QC_SOURCE == 'mcclear':
                data = add_mcclear_to_site(site, data)
            elif CLEAR_SKY_QC_SOURCE != 'pvlib':
                raise ValueError(
                    f"Unsupported FY4A_CLEAR_SKY_QC_SOURCE={CLEAR_SKY_QC_SOURCE!r}; "
                    "use 'pvlib' or 'mcclear'."
                )
            df_final = modis_albedo_load(site, data)
            df_final = df_final[df_final['Sun_Zen'] <= 65].copy()
            df_final = df_final.sort_values(by='Time')

            df_final['Time'] = pd.to_datetime(df_final['Time'])
            df_final = df_final.set_index('Time')
            common_times = pd.DatetimeIndex(xr_sat.time.values).intersection(df_final.index)
            if len(common_times) == 0:
                print(f"No common cloudy satellite/ground/albedo times for site {site}, skipping.")
                continue

            xr_sat = xr_sat.sel(time=common_times.values)
            df_final = df_final.loc[common_times]

            qc_mask = cloudy_time_qc_mask(xr_sat, df_final)
            n_before_qc = len(df_final)
            n_dropped_qc = int((~qc_mask).sum())
            if n_dropped_qc:
                print(
                    f"Dropped {n_dropped_qc}/{n_before_qc} cloudy rows for {site}: "
                    f"|FY4A median Sun_Zen - ground Sun_Zen| > {MAX_SZA_DIFF_DEG:g} deg."
                )
            df_final = df_final.loc[qc_mask]
            xr_sat = xr_sat.sel(time=df_final.index.values)
            if len(df_final) == 0:
                print(f"No cloudy rows remain after QC for site {site}, skipping.")
                continue

            rel_az = np.abs(xr_sat['Sun_Azi'] - xr_sat['Sat_Azi'])
            raz = xr.where(rel_az <= 180, rel_az, 360 - rel_az)

            assign_vars = {
                'RAZ': raz,
                'RH': (('time',), df_final['RH'].values), # %
                'T_s': (('time',), df_final['T_s'].values), # K
                'GHI': (('time',), df_final['ghi'].values),
                'GHI_clear': (('time',), df_final['ghi_clear'].values),
                'Sun_Zen_ground': (('time',), df_final['Sun_Zen'].values),
                'Sun_Azi_ground': (('time',), df_final['Sun_Azi'].values),
                'Sun_Zen_App': (('time',), df_final['Sun_Zen_App'].values),
                'aod': (('time',), df_final['aod'].values),
            }
            if 'cloudy_ground_source' in df_final.columns:
                assign_vars['ground_source_is_clear_like'] = (
                    ('time',),
                    (df_final['cloudy_ground_source'].values == 'clear').astype(np.int8),
                )
            for col in [
                'ghi_clear_pvlib',
                'ghi_clear_mcclear',
                'dni_clear_mcclear',
                'dhi_clear_mcclear',
                'bhi_clear_mcclear',
                'ghi_extra_mcclear',
                'clear_index_mcclear',
            ]:
                if col in df_final.columns:
                    assign_vars[col] = (('time',), df_final[col].values)
            for col in df_final.columns:
                if col.startswith(('Abdo_', 'BSA_', 'WSA_')):
                    assign_vars[col] = (('time',), df_final[col].values)

            xr_all = xr_sat.assign(assign_vars)
            xr_all.attrs['clear_sky_qc_source'] = CLEAR_SKY_QC_SOURCE
            xr_all.attrs['cloudy_ground_source'] = CLOUDY_GROUND_SOURCE
            xr_all.attrs['cloudy_qc'] = (
                f"|median FY4A Sun_Zen - ground Sun_Zen| <= {MAX_SZA_DIFF_DEG:g} deg; "
                "no extraction-stage GHI_clear > 300 W/m2 filter; "
                "no extraction-stage clear-index >= 0.15 filter"
            )
            CLOUDY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            filename = CLOUDY_OUTPUT_DIR / '{}_SW_ref_satellite_{}.nc'.format(site, sky)
            xr_all.to_netcdf(filename)
            print('successfully saved {}'.format(filename))
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

            # data.to_csv('../FY4A_data/{}_radiance_satellite_clear_noalbedo.csv'.format(site), index=False)
            # match with AOD
            data = add_aod_to_site(site, data)

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
            #df_final['Time'] = pd.to_datetime(df_final['Time'])
            # df_sample = sample_subset(df_final)
            # filename = "../FY4A_data/site_sat_data/{}_radiance_satellite_{}_sample.csv".format(site, sky)
            # df_sample.to_csv(filename, index=False)
            # print('Data sampled data saved to:', filename)









