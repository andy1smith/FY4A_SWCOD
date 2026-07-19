"""
Preprocess cloudy dM theta-truncation 3 g²-escape downwelling LUT outputs.

This script is designed for files written by ``main_LUT_cases_Cloudy_dw.py``:

    Results_MODIS_AlbSet0_AOD=0.12_COD=5_kap=[10, 11, 12]_th0=30_Ts=300_RH=50_meth=dM_theta_trunc_cld=3_escape_alpha=1.0_escape=g2.npy

It parses the LUT parameters from each filename, computes ``cos_th0`` and TPW,
integrates full-spectrum ``F_dw``, ``F_dni``, and ``F_dhi`` with ``np.trapz``,
and writes one flat CSV LUT.

Run from the repository root, for example:

    python Surrogate_GRP_COD/Cloudy_dM3_escape_g2/preprocess_surrogate_cloudy_dw.py \
        --dir /path/to/LUTcases/dM/LUT_cloud_dw_g2escape \
        --out Surrogate_GRP_COD/Cloudy_dM3_escape_g2/cloudy_dw_dM3_escape_g2_LUT.csv
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
sys.path.insert(0, str(REPO_ROOT))

from LBL_funcs_fullSpectrum import (  # noqa: E402
    saturation_pressure,
    set_height,
    set_pressure,
    set_temperature,
    set_vmr,
    total_precipitable_water,
)


FULL_NU = np.arange(2500, 35000, 3)
ALBEDO_SETS = {
    0: [0.0145, 0.0288, 0.4156, 0.2031, 0.0641],
    1: [0.0251, 0.0472, 0.3922, 0.2218, 0.0897],
    2: [0.0407, 0.0745, 0.3575, 0.2494, 0.1275],
    3: [0.0673, 0.1207, 0.2988, 0.2961, 0.1916],
    4: [0.0938, 0.1669, 0.2401, 0.3428, 0.2555],
}
ALBEDO_COLUMNS = ["alb_C01", "alb_C02", "alb_C03", "alb_C05", "alb_C06"]

MOLECULES = ["H2O", "CO2", "O3", "N2O", "CH4", "O2", "N2"]
VMR0_BASE = {
    "H2O": 0.03,
    "CO2": 399.5 / 10**6,
    "O3": 50 / 10**9,
    "N2O": 328 / 10**9,
    "CH4": 1834 / 10**9,
    "O2": 2.09 / 10,
    "N2": 7.81 / 10,
}


def default_data_dir() -> str:
    hostname = socket.gethostname()
    defaults = {
        "user-Super-Server": "/home/dengnan/data/RTM/LUTcases/dM/g2escape/",
        "user-MS-7D30": "/mnt/dengnan/LUTcases/dM/g2escape/",
        "h07mgt1": "/puhome/22117689r/projects/Shortwave_MCRTM/LUTcases/dM/g2escape/",
        "dengnans-MacBook-Pro.local": str(REPO_ROOT / "RTM/LUTcases/dM/g2escape/"),
    }
    return defaults.get(hostname, defaults["h07mgt1"])


def parse_filename(filename: str) -> dict:
    stem = Path(filename).stem
    meta: dict[str, object] = {}
    patterns = {
        "AlbSet": (r"_AlbSet(\d+)", int),
        "AOD": (r"_AOD=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", float),
        "COD": (r"_COD=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", float),
        "th0": (r"_th0=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", float),
        "Ts": (r"_Ts=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", float),
        "RH": (r"_RH=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", float),
        "theta_trunc_cld": (r"_theta_trunc_cld=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", float),
        "escape_alpha": (r"_escape_alpha=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", float),
        "escape": (r"_escape=([A-Za-z0-9_+\\-\\.]+)$", str),
    }
    for key, (pattern, caster) in patterns.items():
        match = re.search(pattern, stem)
        if match:
            meta[key] = caster(match.group(1))

    if "th0" in meta:
        meta["cos_th0"] = float(np.cos(np.deg2rad(float(meta["th0"]))))
    if "RH" in meta:
        rh_value = float(meta["RH"])
        meta["rh"] = rh_value / 100.0 if rh_value > 1.0 else rh_value

    alb_set = meta.get("AlbSet")
    if isinstance(alb_set, int) and alb_set in ALBEDO_SETS:
        for column, value in zip(ALBEDO_COLUMNS, ALBEDO_SETS[alb_set]):
            meta[column] = value
    return meta


@lru_cache(maxsize=None)
def compute_tpw_cached(ts: float, rh: float) -> float:
    n_layer = 54
    model = "AFGL midlatitude summer"
    period = "day"
    p, pa = set_pressure(n_layer)
    z, _ = set_height(model, p, pa)
    t, ta = set_temperature(model, p, pa, ts, period)
    ps = saturation_pressure(t)
    vmr0 = dict(VMR0_BASE)
    vmr0["H2O"] = rh * ps[1] / p[1]
    _, densities = set_vmr(model, MOLECULES, vmr0, z)
    return float(total_precipitable_water(densities, pa, ta))


def spectral_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        arr = arr[-1, :]
    return arr


def trapz_full_spectrum(values: np.ndarray) -> float:
    arr = spectral_1d(values)
    if arr.shape[-1] != len(FULL_NU):
        raise ValueError(f"Expected full-spectrum length {len(FULL_NU)}, got {arr.shape[-1]}")
    return float(np.trapz(arr, FULL_NU))


def add_flux_outputs(row: dict, result: dict) -> None:
    flux_map = {
        "GHI": "F_dw",
        "DNI": "F_dni",
        "DHI": "F_dhi",
    }
    for output_name, result_key in flux_map.items():
        if result_key not in result:
            raise KeyError(f"Missing {result_key}")
        row[output_name] = trapz_full_spectrum(result[result_key])


def process_cases(data_dir: str | Path) -> pd.DataFrame:
    data_path = Path(data_dir)
    files = sorted(path for path in data_path.iterdir() if path.name.startswith("Results_") and path.suffix == ".npy")
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for path in files:
        row = parse_filename(path.name)
        try:
            result = np.load(path, allow_pickle=True).item()
            if "Ts" in row and "rh" in row:
                row["tpw"] = compute_tpw_cached(round(float(row["Ts"]), 6), round(float(row["rh"]), 6))
            add_flux_outputs(row, result)
            rows.append(row)
        except Exception as exc:
            skipped.append((path.name, str(exc)))

    if skipped:
        print(f"Skipped {len(skipped)} files:")
        for name, reason in skipped[:20]:
            print(f"  {name}: {reason}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    preferred = [
        "AlbSet",
        "AOD",
        "COD",
        "th0",
        "cos_th0",
        "Ts",
        "RH",
        "rh",
        "tpw",
        "theta_trunc_cld",
        "escape_alpha",
        "escape",
        "GHI",
        "DNI",
        "DHI",
        *ALBEDO_COLUMNS,
    ]
    ordered = [column for column in preferred if column in df.columns]
    ordered += [column for column in df.columns if column not in ordered]
    sort_columns = [column for column in ["AlbSet", "AOD", "COD", "th0", "Ts", "RH"] if column in df.columns]
    return df[ordered].sort_values(sort_columns).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a full-spectrum cloudy DW dM g2-escape LUT CSV.")
    parser.add_argument("--dir", default=default_data_dir(), help="Directory containing Results_*.npy files.")
    parser.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "cloudy_dw_dM3_escape_g2_LUT.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(f"Input directory not found: {args.dir}")

    print(f"Processing cloudy DW LUT files in: {args.dir}")
    df = process_cases(args.dir)
    if df.empty:
        print("No matching Results_*.npy files were processed.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Processed {len(df)} cases.")
    print(f"Saved: {out_path}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())


if __name__ == "__main__":
    main()
