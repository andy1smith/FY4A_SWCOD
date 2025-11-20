import os,sys
import pandas as pd
import math
import numpy as np
import socket
from multiprocessing import Pool, cpu_count
import re
from tqdm import tqdm

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

def process_one_file(fl):
    # Load data
    try:
        # Use 'fdir' from global scope or pass it in
        data = np.load(fdir + fl, allow_pickle=True)
        results = dict(data[0].items())
    except Exception as e:
        # Good practice: return None or an empty dict if load fails
        print(f"Error loading {fl}: {e}")
        return None

    meta = {}
    
    # Regex Parsing
    match = re.search(r'Tsurf=([\d.]+)', fl)
    if match: meta['Ta'] = float(match.group(1))

    match = re.search(r'RH=([\d.]+)', fl)
    if match: meta['rh'] = float(match.group(1))

    match = re.search(r'_COD=([\d.]+)', fl)
    if match: meta['COD'] = float(match.group(1))

    match = re.search(r'_th0=([\d.]+)', fl)
    if match: meta['th0'] = float(match.group(1))

    # Calculations
    # Note: 'nu0' is accessed from global scope
    Fdw = results.get('F_dw')
    if Fdw is not None:
        meta['dsw'] = np.trapz(Fdw, nu0)
    
    DNI = results.get('F_dni')
    if DNI is not None:
        meta['dni'] = np.trapz(DNI, nu0)
        
    DHI = results.get('F_dhi')
    if DHI is not None:
        meta['dhi'] = np.trapz(DHI, nu0)

    uw = results.get('F_uw')
    if uw is not None:
        # Note: 'fy4a_calibration_uw' is accessed from global scope
        df_uw_6channel = fy4a_calibration_uw(uw)
        for ch in df_uw_6channel.columns:
            meta[ch] = df_uw_6channel[ch].values[0]
            
    return meta

# --- 2. Main Execution Block ---
if __name__ == '__main__':

    hostname = socket.gethostname()

    if hostname == 'user-Super-Server': # Replace with actual hostname
        fdir = "/home/dengnan/data/RTM/LUTcases/HG/"
    elif hostname == 'user-MS-7D30':
        fdir = "/mnt/dengnan/LUTcases/HG/"
    elif hostname == 'h07mgt1': 
        fdir = "/puhome/22117689r/projects/Shortwave_MCRTM/LUTcases"
    else:
        # Fallback or Error
        raise ValueError(f"Unknown server: {hostname}. Please set fdir manually.")
    
    Fls = os.listdir(fdir)
    #targetregex = re.compile(r"Results_case2_COD=(\d+\.?\d*)_Tsurf=300_AOD=0\.0_COD=0\.0_th0=")
    Fls = [f for f in Fls if f.startswith('Result')]
    #Fls = [f for f in Fls if 'COD=20' in f and 'th0=30' in f]
    Fls = np.sort(Fls)
    print(len(Fls))

    # Determine how many cores to use (leave 1 free for system)
    #n_cores = max(1, cpu_count() - 1)
    #print(f"Starting multiprocessing with {n_cores} cores...")

    # Create the Pool
    #with Pool(processes=n_cores) as pool:
    with Pool() as pool:
        result_iter = pool.imap(process_one_file, Fls)
        data_rows = list(tqdm(result_iter, total=len(Fls), desc="Processing Files"))

    # Filter out any 'None' results (failed files)
    data_rows = [x for x in data_rows if x is not None]

    # Step 3: Convert to pandas DataFrame
    df = pd.DataFrame(data_rows)
    print("\nProcessing Complete. DataFrame created.")

    # Final Math
    # Avoid division by zero or errors if th0 is missing
    if 'dni' in df.columns and 'th0' in df.columns:
        df['dni'] = df['dni'] / np.cos(np.deg2rad(df['th0']))
    
    # Save
    output_file = './SWRTM_case2_54layers_dnu=3_AOD=0.1243_1000.h5'
    df.to_hdf(output_file, key='data', mode='w')
    print("DataFrame saved to HDF5 file.")
