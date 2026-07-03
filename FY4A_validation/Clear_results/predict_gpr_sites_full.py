"""
predict_gpr_sites_full.py

Predicts clear-sky GHI for all FY4A validation sites using the trained
SWRTM dM V6 hybrid surrogate.

The V6 bundle contains multiple target models, including DNI branches. This
script intentionally uses only the direct GHI branch.
"""

import glob
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]
sys.path.append(str(ROOT_DIR))

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH = ROOT_DIR / "Surrogate_GRP_COD" / "AOD_dsw_model_dM" / "SWRTM_AOD_clearsky_dM_GPR_V6.pkl"
DATA_DIR = ROOT_DIR / "FY4A_data" / "site_sat_data"
PRED_DIR = ROOT_DIR / "FY4A_validation" / "Clear_Test" / "Surrogate_modeling_surfrad_dM"
OUT_DIR = BASE_DIR / "dM_full_year"
DEFAULT_AOD = 0.1243

PRED_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading preferred dM V6 surrogate model …")
bundle = joblib.load(MODEL_PATH)
if "models" not in bundle or "feature_modes" not in bundle:
    raise KeyError(
        "Expected V6 hybrid bundle with keys 'models' and 'feature_modes'. "
        f"Found keys: {sorted(bundle.keys())}"
    )

models = bundle["models"]
scalers_X = bundle.get("scalers_X", {})
feature_modes = bundle["feature_modes"]
metadata = bundle.get("metadata", {})

if "ghi" not in models:
    raise KeyError(f"V6 bundle does not contain a direct GHI model. Found models: {sorted(models)}")

print("  Description:", metadata.get("description"))
print("  GHI feature mode:", feature_modes.get("ghi"))
print("  GHI input columns:", metadata.get("input_columns", {}).get("ghi"))


def compute_tpw(df_raw):
    from LBL_funcs_fullSpectrum import (
        saturation_pressure,
        set_height,
        set_pressure,
        set_temperature,
        set_vmr,
        total_precipitable_water,
    )

    n_layer = 54
    model = "AFGL midlatitude summer"
    period = "day"
    molecules = ["H2O", "CO2", "O3", "N2O", "CH4", "O2", "N2"]
    vmr0 = {
        "H2O": 0.03,
        "CO2": 399.5 / 10**6,
        "O3": 50 / 10**9,
        "N2O": 328 / 10**9,
        "CH4": 1834 / 10**9,
        "O2": 2.09 / 10,
        "N2": 7.81 / 10,
    }

    p, pa = set_pressure(n_layer)
    z, _ = set_height(model, p, pa)
    tpw = np.zeros(len(df_raw))

    for idx, (temperature, humidity) in enumerate(zip(df_raw["T_s"].values, df_raw["RH"].values / 100.0)):
        t, ta = set_temperature(model, p, pa, temperature, period)
        ps = saturation_pressure(t)
        vmr0["H2O"] = humidity * ps[1] / p[1]
        _, densities = set_vmr(model, molecules, vmr0, z)
        tpw[idx] = total_precipitable_water(densities, pa, ta)

    return tpw


def build_feature_matrix(df_raw, feature_mode):
    cos_th0 = np.cos(np.radians(df_raw["Sun_Zen"].values))
    exp_neg_aod = np.exp(-df_raw["aod"].values)
    interaction = cos_th0 * exp_neg_aod
    tpw = df_raw["tpw"].values

    if feature_mode in {"ghi_rich8_v5", "rich8_v6"}:
        return np.column_stack(
            [
                df_raw["T_s"].values,
                df_raw["RH"].values / 100.0,
                df_raw["aod"].values,
                df_raw["Sun_Zen"].values,
                tpw,
                cos_th0,
                exp_neg_aod,
                interaction,
            ]
        )

    if feature_mode in {"dni_gpr4_v4", "dni_gpr4_v6"}:
        return np.column_stack([cos_th0, exp_neg_aod, interaction, tpw**0.5])

    raise ValueError(f"Unknown GHI feature mode: {feature_mode}")


def predict(df_raw):
    """
    Build the V6 feature matrix and return direct GHI predictions only.
    """
    feature_mode = feature_modes["ghi"]
    X = build_feature_matrix(df_raw, feature_mode)
    scaler_X = scalers_X.get("ghi")
    if scaler_X is not None:
        X = scaler_X.transform(X)
    return np.asarray(models["ghi"].predict(X)).ravel()


def apply_filters(df):
    if 'Sun_Zen' in df.columns:
        df = df[df['Sun_Zen']  <= 65].copy()
    if 'C01' in df.columns:
        df = df[df['C01']        < 0.19]
    if 'C06' in df.columns:
        df = df[df['C06']        > 0.05]
    return df


# ── Process Files ──────────────────────────────────────────────────────────
all_predicted = []
pattern = str(DATA_DIR / "*_radiance_satellite_clear.csv")
files = glob.glob(pattern)

for fpath in files:
    site = os.path.basename(fpath).split('_')[0]
    print(f"── Processing {site} ──────────────────────")
    df = pd.read_csv(fpath)
    
    cols_to_keep = ['Time', 'Sun_Zen', 'ghi', 
                    'C01', 'C02', 'C05', 'C06', 'T_s', 'RH', 'rtm_dsw', 'rtm_dni', 'tpw']
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df = apply_filters(df)
    
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    
    df['aod'] = DEFAULT_AOD
    df.reset_index(inplace=True)
    
    df['Site'] = site
    df['Month'] = df['Time'].dt.month
    
    if df.empty:
        print(f"  Skipping {site} because it is empty after filtering.")
        continue

    required_cols = {"Sun_Zen", "T_s", "RH", "aod"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        print(f"  Skipping {site} because required columns are missing: {missing_cols}")
        continue

    if "tpw" not in df.columns or df["tpw"].isna().any():
        print(f"  Computing TPW for {site} ...")
        df["tpw"] = compute_tpw(df)
        
    try:
        preds = predict(df)
    except ValueError as e:
        print(f"  Skipping {site} due to value error: {e}")
        continue
        
    df['gpr_ghi'] = preds
        
    out_csv = PRED_DIR / f"gpr_predicted_dM_{site}.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Generated {len(df)} predictions and saved to {os.path.basename(out_csv)}")
    
    all_predicted.append(df)

if all_predicted:
    print("\nPredictions completed and saved successfully.")
else:
    print("No data processed.")
