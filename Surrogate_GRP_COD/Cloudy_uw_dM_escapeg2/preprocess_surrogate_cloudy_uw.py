"""
Preprocess cloudy dM g2-escape FY4A AGRI upwelling LUT outputs for COD surrogate training.

This script is designed for the files written by ``main_LUT_cases_Cloudy_uw.py``:

    Results_MODIS_AlbSet0_AOD=0.12_COD=5_kap=[10, 11, 12]_th0=30_Ts=310_RH=60.npy

It extracts all filename variables from that driver, computes TPW from ``Ts`` and
``rh``, integrates ``F_uw`` into SRF-weighted FY4A channel radiance and
normalized channel quantities, integrates ``F_uw_nosrf`` directly over each FY4A
channel, and writes one flat CSV for correlation analysis and COD surrogate
training.

Run on the server from the repository root, for example:

    python Surrogate_GRP_COD/Cloudy_uw_dM_escapeg2/preprocess_surrogate_cloudy_uw.py \
        --dir /path/to/LUTcases/dM_g2_escape/fy4a_channels \
        --out Surrogate_GRP_COD/Cloudy_uw_dM_escapeg2/preprocessed_cloudy_uw_dM_escapeg2.csv
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
sys.path.insert(0, str(REPO_ROOT))

from LBL_funcs_fullSpectrum import (  # noqa: E402
    saturation_pressure,
    set_height,
    set_pressure,
    set_temperature,
    set_vmr,
    total_precipitable_water,
)


CHANNELS = ["C01", "C02", "C03", "C05", "C06"]
# FY4A AGRI TOA downwelling channel integrals [W m-2], computed from
# data/profiles/ASTMG173.csv and FY4A_data/AGRI_calibration/FY4A_AGRI_SRF_ch*.txt.
F_DW_OS_CH = {
    "C01": 100.56360014402173,
    "C02": 293.8703639771758,
    "C03": 146.06104052297425,
    "C05": 13.936208329862962,
    "C06": 18.20438461023419,
}
ALBEDO_SETS = {
    0: [0.013735, 0.023839, 0.126622, 0.081469, 0.041332],
    1: [0.032169, 0.051783, 0.239896, 0.170084, 0.079692],
    2: [0.045682, 0.078521, 0.305128, 0.209253, 0.112486],
    3: [0.065650, 0.112386, 0.354232, 0.259652, 0.180991],
    4: [0.100162, 0.201559, 0.454122, 0.368143, 0.289324],
}

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
        "user-Super-Server": "/home/dengnan/data/RTM/LUTcases/dM_g2_escape/fy4a_channels/",
        "user-MS-7D30": "/mnt/dengnan/LUTcases/dM_g2_escape/fy4a_channels/",
        "h07mgt1": "/puhome/22117689r/projects/FY4A_SWCOD/LUTcases/dM_g2_escape/fy4a_channels/",
        "dengnans-MacBook-Pro.local": str(REPO_ROOT / "RTM/LUTcases/dM_g2_escape/fy4a_channels/"),
    }
    return defaults.get(hostname, defaults["h07mgt1"])


def default_fy4a_data_dir() -> str:
    return str(REPO_ROOT / "FY4A_data")


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
    }
    for key, (pattern, caster) in patterns.items():
        match = re.search(pattern, stem)
        if match:
            meta[key] = caster(match.group(1))

    if "RH" in meta:
        rh_value = float(meta["RH"])
        meta["rh"] = rh_value / 100.0 if rh_value > 1.0 else rh_value

    alb_set = meta.get("AlbSet")
    if isinstance(alb_set, int) and alb_set in ALBEDO_SETS:
        for channel, value in zip(CHANNELS, ALBEDO_SETS[alb_set]):
            meta[f"alb_{channel}"] = value

    return meta


def channel_calibration_path(file_dir: str | Path, channel: str, sensor: str = "FY4A_AGRI") -> Path:
    channel_number = int(channel[-2:])
    file_dir = Path(file_dir)
    if sensor in {"FY4A", "FY4A_AGRI"}:
        channel_srf = file_dir / "AGRI_calibration" / "FY4A_AGRI_SRF_ch{:d}.txt".format(channel_number)
    else:
        raise ValueError(f"Unsupported sensor: {sensor}")
    return Path(channel_srf)


def load_channel_calibration(file_dir: str | Path, channel: str, sensor: str = "FY4A_AGRI") -> tuple[np.ndarray, np.ndarray]:
    calibration = np.loadtxt(channel_calibration_path(file_dir, channel, sensor), delimiter=",", skiprows=1)
    calibration_nu = calibration[:, 1]
    calibration_srf = calibration[:, 2]
    return calibration_nu, calibration_srf


def channel_nu_grid(channel: str, file_dir: str | Path, dnu: int = 3, sensor: str = "FY4A_AGRI") -> np.ndarray:
    base_nu = np.arange(2500, 35000, dnu)
    calibration_nu, _ = load_channel_calibration(file_dir, channel, sensor)
    channel_mask = (base_nu >= calibration_nu.min()) & (base_nu <= calibration_nu.max())
    return base_nu[channel_mask]


def load_channel_grid(file_dir: str | Path, channels: list[str], dnu: int = 3, sensor: str = "FY4A_AGRI") -> np.ndarray:
    base_nu = np.arange(2500, 35000, dnu)
    union_values: set[float] = set()

    for channel in channels:
        calibration_nu, _ = load_channel_calibration(file_dir, channel, sensor)
        mask = (base_nu >= calibration_nu.min()) & (base_nu <= calibration_nu.max())
        union_values.update(base_nu[mask].tolist())

    return np.array(sorted(union_values))


def infer_nu_grid(n_points: int, fy4a_nu: np.ndarray) -> tuple[np.ndarray, str]:
    full_nu = np.arange(2500, 35000, 3)
    if n_points == len(fy4a_nu):
        return fy4a_nu, "FY4A"
    if n_points == len(full_nu):
        return full_nu, "full"
    raise ValueError(
        f"Cannot infer spectral grid for {n_points} points. Expected {len(fy4a_nu)} "
        f"FY4A points or {len(full_nu)} full-spectrum points."
    )


def spectral_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        arr = arr[-1, :]
    return arr


def trapz_channel_plain(
    spectral: np.ndarray,
    rtm_nu: np.ndarray,
    channel: str,
    file_dir: str | Path,
    sensor: str = "FY4A_AGRI",
    dnu: int = 3,
) -> float:
    arr = spectral_1d(spectral)
    nu_channel = channel_nu_grid(channel, file_dir, dnu=dnu, sensor=sensor)
    nu_idx = np.nonzero(np.isin(rtm_nu, nu_channel))[0]
    if len(nu_idx) < 2:
        return np.nan
    channel_nu = rtm_nu[nu_idx]
    return float(np.trapz(arr[nu_idx], channel_nu))


def trapz_channel_srf(
    spectral: np.ndarray,
    rtm_nu: np.ndarray,
    channel: str,
    file_dir: str | Path,
    sensor: str = "FY4A_AGRI",
    dnu: int = 3,
) -> float:
    arr = spectral_1d(spectral)
    calibration_nu, calibration_srf = load_channel_calibration(file_dir, channel, sensor)
    nu_channel = channel_nu_grid(channel, file_dir, dnu=dnu, sensor=sensor)
    calibration_nu = calibration_nu[::-1]
    calibration_srf = calibration_srf[::-1]
    srf = np.interp(nu_channel, calibration_nu, calibration_srf)
    nu_idx = np.nonzero(np.isin(rtm_nu, nu_channel))[0]
    if len(nu_idx) < 2:
        return np.nan
    channel_nu = rtm_nu[nu_idx]
    channel_srf = np.interp(channel_nu, nu_channel, srf)
    return float(np.trapz(arr[nu_idx] * channel_srf, channel_nu))


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


def add_integrated_outputs(
    row: dict,
    result: dict,
    nu: np.ndarray,
    file_dir: str | Path,
    sensor: str = "FY4A_AGRI",
) -> None:
    for channel in CHANNELS:
        if "F_uw" in result:
            uw_channel = trapz_channel_srf(result["F_uw"], nu, channel, file_dir, sensor=sensor)
            row[f"{channel}_rad"] = uw_channel
            # do not add R_factor, do never add mu0
            row[channel] = uw_channel / F_DW_OS_CH[channel] # real reflectance factor
        if "F_uw_nosrf" in result:
            row[f"{channel}_nosrf"] = trapz_channel_plain(result["F_uw_nosrf"], nu, channel, file_dir, sensor=sensor)


def process_cases(
    data_dir: str | Path,
    fy4a_data_dir: str | Path = default_fy4a_data_dir(),
    sensor: str = "FY4A_AGRI",
) -> pd.DataFrame:
    data_path = Path(data_dir)
    files = sorted(path for path in data_path.iterdir() if path.name.startswith("Results_") and path.suffix == ".npy")
    fy4a_nu = load_channel_grid(fy4a_data_dir, CHANNELS, sensor=sensor)

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for path in files:
        row = parse_filename(path.name)
        try:
            result = np.load(path, allow_pickle=True).item()
            first_array = next(np.asarray(result[key]) for key in ("F_uw", "F_uw_nosrf", "F_uw_srf") if key in result)
            n_points = first_array.shape[-1]
            nu, _ = infer_nu_grid(n_points, fy4a_nu)
            if "Ts" in row and "rh" in row:
                row["tpw"] = compute_tpw_cached(round(float(row["Ts"]), 6), round(float(row["rh"]), 6))
            if "th0" in row:
                row["cos_th0"] = float(np.cos(np.deg2rad(float(row["th0"]))))
            add_integrated_outputs(row, result, nu, fy4a_data_dir, sensor=sensor)
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
        *[f"alb_{channel}" for channel in CHANNELS],
        *[f"{channel}_rad" for channel in CHANNELS],
        *CHANNELS,
        *[f"{channel}_nosrf" for channel in CHANNELS],
    ]
    ordered = [col for col in preferred if col in df.columns]
    ordered += [col for col in df.columns if col not in ordered]
    return df[ordered].sort_values([col for col in ["COD", "th0", "Ts", "RH", "AlbSet"] if col in df.columns])


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess cloudy dM g2-escape FY4A upwelling RTM outputs.")
    parser.add_argument("--dir", default=default_data_dir(), help="Directory containing Results_*.npy files.")
    parser.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "preprocessed_cloudy_uw_dM_escapeg2.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--fy4a-data-dir",
        "--goes-data-dir",
        dest="fy4a_data_dir",
        default=default_fy4a_data_dir(),
        help="Directory containing FY4A AGRI calibration spectral response files.",
    )
    parser.add_argument("--sensor", default="FY4A_AGRI", help="Satellite sensor name.")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(f"Input directory not found: {args.dir}")

    print(f"Processing cloudy LUT files in: {args.dir}")
    df = process_cases(args.dir, fy4a_data_dir=args.fy4a_data_dir, sensor=args.sensor)
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
