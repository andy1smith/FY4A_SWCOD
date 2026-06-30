"""
predict_gpr_sites_full_withAOD.py

Re-predicts clear-sky GHI for all FY4A validation sites using the trained
SWRTM dM V6 hybrid surrogate and site-specific AOD values.

The V6 bundle contains multiple target models, including DNI branches. This
script intentionally uses only the direct GHI branch. It reads and overwrites
the `gpr_predicted_dM_*.csv` files in this `withAOD` folder.
"""

import glob
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[3]
sys.path.append(str(ROOT_DIR))

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH = ROOT_DIR / "Surrogate_GRP_COD" / "AOD_dsw_model_dM" / "SWRTM_AOD_clearsky_dM_GPR_V6.pkl"
PRED_DIR = BASE_DIR
MANIFEST_PATH = BASE_DIR / "withAOD_manifest.csv"

PRED_DIR.mkdir(parents=True, exist_ok=True)


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

if MANIFEST_PATH.exists():
    manifest = pd.read_csv(MANIFEST_PATH)
    AOD_BY_SITE = {
        str(row["site"]): float(row["new_aod"])
        for _, row in manifest.iterrows()
        if pd.notna(row.get("site")) and pd.notna(row.get("new_aod"))
    }
else:
    AOD_BY_SITE = {}


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


# ── Process Files ──────────────────────────────────────────────────────────
all_predicted = []
pattern = str(PRED_DIR / "gpr_predicted_dM_*.csv")
files = glob.glob(pattern)

for fpath in files:
    site = os.path.basename(fpath).replace("gpr_predicted_dM_", "").replace(".csv", "")
    print(f"── Processing {site} ──────────────────────")
    df = pd.read_csv(fpath)

    df['Time'] = pd.to_datetime(df['Time'])
    df['Site'] = site
    df['Month'] = df['Time'].dt.month

    if df.empty:
        print(f"  Skipping {site} because it is empty.")
        continue

    if site == "BJC":
        if "aod" not in df.columns or not df["aod"].notna().any():
            raise KeyError("BJC requires row-level AOD values from AOD_correction/AERONET_china/2021_BJC_CAMS.csv")
        df["aod"] = pd.to_numeric(df["aod"], errors="coerce")
        if df["aod"].isna().any():
            df["aod"] = df["aod"].fillna(float(df["aod"].median()))
        aod_label = f"{df['aod'].min():.6f}-{df['aod'].max():.6f}"
    elif site in AOD_BY_SITE:
        site_aod = AOD_BY_SITE[site]
        df["aod"] = site_aod
        aod_label = f"{site_aod:.6f}"
    elif "aod" in df.columns and df["aod"].notna().any():
        site_aod = float(pd.to_numeric(df["aod"], errors="coerce").dropna().iloc[0])
        df["aod"] = site_aod
        aod_label = f"{site_aod:.6f}"
    else:
        raise KeyError(f"No site-specific AOD available for {site}")

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
        
    out_csv = Path(fpath)
    df.to_csv(out_csv, index=False)
    print(f"  Used AOD={aod_label}; generated {len(df)} GHI predictions and overwrote {os.path.basename(out_csv)}")
    
    all_predicted.append(df)

if all_predicted:
    print("\nPredictions completed and saved successfully.")
else:
    print("No data processed.")
