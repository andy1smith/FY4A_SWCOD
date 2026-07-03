"""
calculate_and_save_tpw_clear.py
Calculates total precipitable water (TPW) for each clearsky validation CSV file
in GOES_data/GOES16_site_sat_data/clear and saves it back.
Uses exact physics-based integration code from plot_correlation.py.
"""

import os
import glob
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Set up repository root import path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]  # Climb up to Shortwave_MCRTM root from GOES_validation/Analysis/
sys.path.append(str(ROOT_DIR))

from LBL_funcs_fullSpectrum import (
    total_precipitable_water,
    set_pressure,
    saturation_pressure,
    set_temperature,
    set_height,
    set_vmr,
)

def compute_tpw(df):
    N_layer = 54
    model = 'AFGL midlatitude summer'
    period = 'day'
    molecules = ['H2O', 'CO2', 'O3', 'N2O', 'CH4', 'O2', 'N2']
    vmr0 = {
        'H2O': 0.03,
        'CO2': 399.5 / 10**6,
        'O3': 50 / 10**9,
        'N2O': 328 / 10**9,
        'CH4': 1834 / 10**9,
        'O2': 2.09 / 10,
        'N2': 7.81 / 10
    }

    p, pa = set_pressure(N_layer)
    z, za = set_height(model, p, pa)

    T_surf_arr = df['T'].values
    rh_arr = df['RH'].values / 100.0  # Convert 0-100% to 0-1 fraction
    
    tpw = np.zeros(len(df))
    for i in range(len(df)):
        T_surf = T_surf_arr[i]
        rh0 = rh_arr[i]
        t, ta = set_temperature(model, p, pa, T_surf, period)
        ps = saturation_pressure(t)
        vmr0['H2O'] = rh0 * ps[1] / p[1]
        vmr, densities = set_vmr(model, molecules, vmr0, z)
        tpw[i] = total_precipitable_water(densities, pa, ta)
        
    return tpw

def main():
    data_dir = os.path.join(ROOT_DIR, "GOES_data/GOES16_site_sat_data/clear")
    
    # Locate all clearsky validation files
    all_files = sorted(glob.glob(os.path.join(data_dir, "GOES_day_*_radiance_clearsky_2019_MarOct.csv")))
    
    if not all_files:
        print("No CSV files found in GOES_data/GOES16_site_sat_data/clear")
        return
        
    print(f"Found {len(all_files)} files to process.")
    
    for fpath in all_files:
        print(f"Processing: {os.path.basename(fpath)} ...")
        df = pd.read_csv(fpath)
        
        # Verify required columns exist
        if 'T' not in df.columns or 'RH' not in df.columns:
            print(f"  Warning: 'T' or 'RH' missing in {os.path.basename(fpath)}. Skipping.")
            continue
            
        # Check if 'tpw' is already there
        if 'tpw' in df.columns:
            print(f"  Column 'tpw' already exists in {os.path.basename(fpath)}. Skipping.")
            continue

        # Calculate TPW
        tpw_vals = compute_tpw(df)
        df['tpw'] = tpw_vals
        
        # Save back
        df.to_csv(fpath, index=False)
        print(f"  → Successfully calculated and saved tpw column (length: {len(df)}).")

if __name__ == "__main__":
    main()
