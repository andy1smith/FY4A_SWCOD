"""
Retrieve cloudy FY4A COD maps from the HG upwelling LUT surrogate.

Inputs:
  - FY4A_data/*_SW_ref_satellite_cloudy.nc
  - Surrogate_GRP_COD/Cloudy_uw_HG_FY4A/SWRTM_cloudy_uw_channel_HG_interp_V1.pkl
  - Sat_Preprocessing/mcd43a1_albedo/data/*MCD43A1*.csv

Outputs:
  - FY4A_validation/Cloudy_results/<SITE>_cloudy_COD_uw_LUT.nc
  - FY4A_validation/Cloudy_results/cloudy_cod_manifest.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SAT_PREPROCESSING_DIR = REPO_ROOT / "Sat_Preprocessing"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SAT_PREPROCESSING_DIR))

from LBL_funcs_fullSpectrum import (  # noqa: E402
    saturation_pressure,
    set_height,
    set_pressure,
    set_temperature,
    set_vmr,
    total_precipitable_water,
)
from mcd43a1_albedo import white  # noqa: E402


CHANNELS = ["C01", "C02", "C03", "C05", "C06"]
FY4A_TO_MODIS_BAND = {
    "C01": 3,
    "C02": 1,
    "C03": 2,
    "C05": 6,
    "C06": 7,
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


@lru_cache(maxsize=1)
def _atmosphere_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_layer = 54
    model = "AFGL midlatitude summer"
    p, pa = set_pressure(n_layer)
    z, _ = set_height(model, p, pa)
    return p, pa, z


@lru_cache(maxsize=20000)
def compute_tpw_one(t_s: float, rh_percent: float) -> float:
    if not np.isfinite(t_s) or not np.isfinite(rh_percent):
        return np.nan
    p, pa, z = _atmosphere_grid()
    model = "AFGL midlatitude summer"
    period = "day"
    t, ta = set_temperature(model, p, pa, float(t_s), period)
    ps = saturation_pressure(t)
    vmr0 = dict(VMR0_BASE)
    vmr0["H2O"] = np.clip(float(rh_percent) / 100.0, 0.0, 1.0) * ps[1] / p[1]
    _, densities = set_vmr(model, MOLECULES, vmr0, z)
    return float(total_precipitable_water(densities, pa, ta))


def compute_tpw_series(t_s: xr.DataArray, rh: xr.DataArray) -> np.ndarray:
    t_vals = np.asarray(t_s.values, dtype=float)
    rh_vals = np.asarray(rh.values, dtype=float)
    return np.array(
        [compute_tpw_one(round(t, 2), round(r, 2)) for t, r in zip(t_vals, rh_vals)],
        dtype=float,
    )


def load_mcd43() -> pd.DataFrame:
    files = [
        SAT_PREPROCESSING_DIR / "mcd43a1_albedo" / "data" / "CERN2021-MCD43A1-061-results.csv",
        SAT_PREPROCESSING_DIR / "mcd43a1_albedo" / "data" / "CERN34-MCD43A1-061-results.csv",
    ]
    existing = [path for path in files if path.exists()]
    if not existing:
        raise FileNotFoundError("No MCD43A1 files found under Sat_Preprocessing/mcd43a1_albedo/data")
    return pd.concat([pd.read_csv(path) for path in existing], ignore_index=True)


def default_albedo(bundle: dict, channel: str) -> float:
    axis = np.asarray(bundle["models"][channel]["axes"][f"alb_{channel}"], dtype=float)
    return float(np.nanmedian(axis))


def site_wsa_albedo(site: str, times: np.ndarray, mcd43: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    out = pd.DataFrame({"Time": pd.to_datetime(times)})
    out["Date"] = out["Time"].dt.normalize()
    site_df = mcd43[mcd43["Category"] == site].copy()

    for channel in CHANNELS:
        out[f"alb_{channel}"] = default_albedo(bundle, channel)

    if site_df.empty:
        return out.drop(columns=["Date"])

    for iband in range(1, 8):
        qa_col = f"MCD43A1_061_BRDF_Albedo_Band_Mandatory_Quality_Band{iband}"
        param_cols = [
            f"MCD43A1_061_BRDF_Albedo_Parameters_Band{iband}_{idx}"
            for idx in range(3)
        ]
        if qa_col in site_df.columns:
            bad = site_df[qa_col] > 1
            site_df.loc[bad, param_cols] = np.nan

    site_df["Date"] = pd.to_datetime(site_df["Date"]).dt.normalize()
    keep = ["Date"]
    rename = {}
    for channel, iband in FY4A_TO_MODIS_BAND.items():
        src = [
            f"MCD43A1_061_BRDF_Albedo_Parameters_Band{iband}_{idx}"
            for idx in range(3)
        ]
        tgt = [f"{channel}_p{idx}" for idx in range(3)]
        keep.extend(src)
        rename.update(dict(zip(src, tgt)))

    site_df = site_df[[col for col in keep if col in site_df.columns]].rename(columns=rename)
    merged = out.merge(site_df, on="Date", how="left")

    for channel in CHANNELS:
        pcols = [f"{channel}_p{idx}" for idx in range(3)]
        valid = merged[pcols].notna().all(axis=1)
        if valid.any():
            values = white(merged.loc[valid, pcols[0]], merged.loc[valid, pcols[1]], merged.loc[valid, pcols[2]])
            merged.loc[valid, f"alb_{channel}"] = np.asarray(values, dtype=float)

    return merged[["Time", *[f"alb_{channel}" for channel in CHANNELS]]]


def build_cod_grid(bundle: dict, n_cod: int) -> np.ndarray:
    bounds = bundle["models"][CHANNELS[0]]["bounds"]["log1p_COD"]
    log_grid = np.linspace(bounds[0], bounds[1], n_cod)
    return np.expm1(log_grid)


def make_interpolator(spec: dict) -> RegularGridInterpolator:
    axes = tuple(np.asarray(spec["axes"][feature], dtype=float) for feature in spec["grid_features"])
    return RegularGridInterpolator(
        axes,
        np.asarray(spec["values"], dtype=float),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def retrieve_channel_cod(
    ds: xr.Dataset,
    bundle: dict,
    channel: str,
    cod_grid: np.ndarray,
    tpw: np.ndarray,
    albedo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spec = bundle["models"][channel]
    interp = make_interpolator(spec)
    obs = np.asarray(ds[channel].values, dtype=float)
    cos_th0 = np.cos(np.deg2rad(np.asarray(ds["Sun_Zen"].values, dtype=float)))
    time_len, y_len, x_len = obs.shape
    n_points = time_len * y_len * x_len

    obs_flat = obs.reshape(n_points)
    cos_flat = cos_th0.reshape(n_points)
    tpw_flat = np.repeat(tpw, y_len * x_len)
    alb_flat = np.repeat(albedo, y_len * x_len)

    valid = (
        np.isfinite(obs_flat)
        & np.isfinite(cos_flat)
        & np.isfinite(tpw_flat)
        & np.isfinite(alb_flat)
        & (obs_flat > 0.0)
    )
    cod = np.full(n_points, np.nan, dtype=np.float32)
    error = np.full(n_points, np.nan, dtype=np.float32)
    if not valid.any():
        return cod.reshape(obs.shape), error.reshape(obs.shape)

    valid_idx = np.nonzero(valid)[0]
    log_cod = np.log1p(cod_grid)
    pred = np.empty((len(cod_grid), len(valid_idx)), dtype=np.float32)

    clipped_tpw = np.clip(tpw_flat[valid_idx], *spec["bounds"]["tpw"])
    clipped_cos = np.clip(cos_flat[valid_idx], *spec["bounds"]["cos_th0"])
    alb_key = f"alb_{channel}"
    clipped_alb = np.clip(alb_flat[valid_idx], *spec["bounds"][alb_key])

    for cod_idx, log_value in enumerate(log_cod):
        points = np.column_stack(
            [
                np.full(len(valid_idx), log_value, dtype=float),
                clipped_tpw,
                clipped_cos,
                clipped_alb,
            ]
        )
        pred[cod_idx, :] = interp(points).astype(np.float32)

    diff = np.abs(pred - obs_flat[valid_idx][None, :])
    best = np.nanargmin(diff, axis=0)
    cod[valid_idx] = cod_grid[best].astype(np.float32)
    error[valid_idx] = diff[best, np.arange(len(valid_idx))].astype(np.float32)
    return cod.reshape(obs.shape), error.reshape(obs.shape)


def retrieve_site(
    path: Path,
    bundle: dict,
    mcd43: pd.DataFrame,
    out_dir: Path,
    n_cod: int,
) -> dict:
    site = path.name.split("_SW_ref_satellite_cloudy.nc")[0]
    ds = xr.open_dataset(path).load()
    cod_grid = build_cod_grid(bundle, n_cod).astype(np.float32)
    tpw = compute_tpw_series(ds["T_s"], ds["RH"])
    albedo_df = site_wsa_albedo(site, ds["time"].values, mcd43, bundle)

    data_vars = {
        "tpw": (("time",), tpw.astype(np.float32)),
    }
    for channel in CHANNELS:
        data_vars[f"alb_{channel}"] = (
            ("time",),
            albedo_df[f"alb_{channel}"].to_numpy(dtype=np.float32),
        )

    cod_stack = []
    for channel in CHANNELS:
        cod, error = retrieve_channel_cod(
            ds=ds,
            bundle=bundle,
            channel=channel,
            cod_grid=cod_grid,
            tpw=tpw,
            albedo=albedo_df[f"alb_{channel}"].to_numpy(dtype=float),
        )
        data_vars[f"COD_{channel}"] = (("time", "y", "x"), cod)
        data_vars[f"reflectance_abs_error_{channel}"] = (("time", "y", "x"), error)
        cod_stack.append(cod)

    cod_arr = np.stack(cod_stack, axis=0)
    valid_count = np.sum(np.isfinite(cod_arr), axis=0)
    cod_mean = np.full(cod_arr.shape[1:], np.nan, dtype=np.float32)
    np.divide(
        np.nansum(cod_arr, axis=0),
        valid_count,
        out=cod_mean,
        where=valid_count > 0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cod_median = np.nanmedian(cod_arr, axis=0).astype(np.float32)
        cod_std = np.nanstd(cod_arr, axis=0).astype(np.float32)
    data_vars["COD_mean"] = (("time", "y", "x"), cod_mean)
    data_vars["COD_median"] = (("time", "y", "x"), cod_median)
    data_vars["COD_std"] = (("time", "y", "x"), cod_std)
    data_vars["valid_channel_count"] = (
        ("time", "y", "x"),
        valid_count.astype(np.int16),
    )

    out = xr.Dataset(
        data_vars,
        coords={
            "time": ds["time"].values,
            "y": ds["y"].values,
            "x": ds["x"].values,
            "COD_grid": cod_grid,
        },
        attrs={
            "site": site,
            "source_file": str(path),
            "model_file": str(args_model_path(bundle)),
            "channels": ",".join(CHANNELS),
            "retrieval": "nearest absolute reflectance match on UW HG forward LUT",
            "tpw_source": "computed from T_s and RH using LBL_funcs_fullSpectrum",
            "albedo_source": "MCD43A1 daily WSA; missing/failed QA values fall back to model-grid median",
        },
    )

    out_path = out_dir / f"{site}_cloudy_COD_uw_LUT.nc"
    encoding = {name: {"zlib": True, "complevel": 4} for name in out.data_vars}
    try:
        out.to_netcdf(out_path, encoding=encoding)
    except ValueError:
        out.to_netcdf(out_path)

    finite = np.isfinite(out["COD_mean"].values)
    return {
        "site": site,
        "input_file": str(path),
        "output_file": str(out_path),
        "n_time": int(ds.sizes["time"]),
        "n_valid_pixels": int(finite.sum()),
        "cod_mean": float(np.nanmean(out["COD_mean"].values)),
        "cod_median": float(np.nanmedian(out["COD_median"].values)),
    }


def args_model_path(bundle: dict) -> str:
    return str(bundle.get("_model_path", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict cloudy FY4A COD per 11x11 pixel using the UW HG LUT.")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "FY4A_data"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR))
    parser.add_argument(
        "--model",
        default=str(
            REPO_ROOT
            / "Surrogate_GRP_COD"
            / "Cloudy_uw_HG_FY4A"
            / "SWRTM_cloudy_uw_channel_HG_interp_V1.pkl"
        ),
    )
    parser.add_argument("--n-cod", type=int, default=80, help="Number of COD grid points used in inversion.")
    parser.add_argument("--site", action="append", help="Optional site code. Repeat to process multiple sites.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    bundle = joblib.load(model_path)
    bundle["_model_path"] = str(model_path)
    mcd43 = load_mcd43()

    files = sorted(data_dir.glob("*_SW_ref_satellite_cloudy.nc"))
    if args.site:
        wanted = set(args.site)
        files = [path for path in files if path.name.split("_SW_ref_satellite_cloudy.nc")[0] in wanted]
    if not files:
        raise FileNotFoundError(f"No cloudy FY4A NetCDF files found in {data_dir}")

    rows = []
    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {path.name}")
        row = retrieve_site(path, bundle, mcd43, out_dir, args.n_cod)
        rows.append(row)
        print(f"  saved {Path(row['output_file']).name}; valid pixels={row['n_valid_pixels']}")

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "cloudy_cod_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
