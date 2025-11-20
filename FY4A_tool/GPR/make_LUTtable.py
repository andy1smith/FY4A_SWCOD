import os,sys
import pandas as pd
import math
import numpy as np

import re
from tqdm import tqdm

#from fun_nearealtime_RTM import FY4A_calinu, get_calibration_srf

# Get the current directory
# os.getcwd()
# Get the parent directory (which is the 'main dir')
#os.path.dirname(current_dir)

code_dir=os.path.dirname(os.path.dirname(os.getcwd()))
# 1. Add the main directory to the import path
sys.path.append(code_dir)
# 2. Change the current working directory to the main directory
os.chdir(code_dir)
from fun_nearealtime_RTM import FY4A_calinu, get_calibration_srf

def fy4a_calibration_uw(uw):
    file_dir='./FY4A_data/'
    channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    nu0 = np.arange(2500, 35000, 3)  # Wavenumber grid
    nu_channels = FY4A_calinu(nu0, channels, "./FY4A_data/", dnu=3)
    df = pd.DataFrame(columns=channels)
    F_dw_os_srf_channel = [100.56360014402173,293.8703639771758,146.06104052297425,
                           12.06884597258561,13.936208329862962,18.20438461023419]

    for i, channel in enumerate(channels):
        srf, nu_channel = get_calibration_srf(channel, file_dir)
        nu_idx = np.nonzero(np.isin(nu0, nu_channel))[0]  # fixed 1 April.
        
        # correct uw
        uw_cor = np.multiply(uw[nu_idx], srf)
        uw_channel = np.trapz(uw_cor,nu_channel)
        # normalize uw_channel
        df.loc[0, channel] = uw_channel #/ F_dw_os_srf_channel[i]
    return df


if __name__ == "__main__":

    #code_dir=os.path.dirname(os.path.dirname(os.getcwd()))
    # 1. Add the main directory to the import path
    #sys.path.append(code_dir)
    # 2. Change the current working directory to the main directory
    #os.chdir(code_dir)
    
    fdir = "/home/dengnan/data/RTM/LUTcases/HG/" 
    #"/mnt/dengnan/LUTcases/HG/" 
    Fls = os.listdir(fdir)
    #targetregex = re.compile(r"Results_case2_COD=(\d+\.?\d*)_Tsurf=300_AOD=0\.0_COD=0\.0_th0=")
    Fls = [f for f in Fls if f.startswith('Result')]
    #Fls = [f for f in Fls if 'COD=20' in f and 'th0=30' in f]
    Fls = np.sort(Fls)
    print(len(Fls))


    # Step 1: Prepare your columns and lists
    channels = ['C{:02d}'.format(c) for c in range(1, 7)]
    Tsurf, RH, COD, th0 = [], [], [], [] # AOD,[]

    # We'll build a list of dicts, each one a row
    data_rows = []
    nu0 = np.arange(2500, 35000, 3)

    for fl in tqdm(Fls, desc="Processing Radiative Transfer Files"):
        # Load data
        results = dict(np.load(fdir+fl, allow_pickle=True)[0].items())
        #try:
        meta = {}    
        match = re.search(r'Tsurf=([\d.]+)', fl)
        if match: meta['Ta'] = float(match.group(1))

        match = re.search(r'RH=([\d.]+)', fl)
        if match: meta['rh'] = float(match.group(1))

        match = re.search(r'_COD=([\d.]+)', fl)
        if match: meta['COD'] = float(match.group(1))

        match = re.search(r'_th0=([\d.]+)', fl)
        if match: meta['th0'] = float(match.group(1))

        Fdw = results.get('F_dw')  # shape: (len(nu0), )
        meta['dsw'] = np.trapz(Fdw,nu0)
        DNI = results.get('F_dni')
        DHI = results.get('F_dhi')
        meta['dni'] = np.trapz(DNI, nu0)
        meta['dhi'] = np.trapz(DHI, nu0)

        uw = results.get('F_uw')
        df_uw_6channel = fy4a_calibration_uw(uw)
        # Save calibrated 6-channel values into meta
        for ch in df_uw_6channel.columns:
            meta[ch] = df_uw_6channel[ch].values[0]
        
        data_rows.append(meta)
    # Step 2: Convert to pandas DataFrame
    df = pd.DataFrame(data_rows)
    print("\nProcessing Complete. DataFrame created.")

    df['dni'] = df['dni']/np.cos(np.deg2rad(df['th0']))
    df.to_hdf('./SWRTM_case2_54layers_dnu=3_AOD=0.1243_7100.h5', key='data', mode='w')
    print("DataFrame saved to HDF5 file.")
