"""Retrieve FY4A cloudy COD maps with the FY4A UW surrogate plus ADM LUT.

This is the FY4A counterpart of the GOES
``main_SW_scope_nearealtime_GradientDescent.py`` surrogate path: predict a
cloudy upwelling channel response for candidate COD values, apply an ADM
angular factor for each pixel, then pick the COD that minimizes the weighted
multi-channel reflectance residual.
"""

from __future__ import annotations

import argparse
import re
import warnings
from functools import lru_cache
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, interp1d

from FY4A_data.ADM_cloud.AngDistLUT import reconstruct_hc
from LBL_funcs_fullSpectrum import (
    saturation_pressure,
    set_height,
    set_pressure,
    set_temperature,
    set_vmr,
    total_precipitable_water,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "FY4A_data" / "Cloudy_site_sat_data"
DEFAULT_OUT_DIR = REPO_ROOT / "FY4A_validation" / "Cloudy_results"
DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "Surrogate_GRP_COD"
    / "Cloudy_uw_HG_FY4A"
    / "SWRTM_cloudy_uw_channel_HG_interp_V1.pkl"
)
DEFAULT_METRICS_PATH = (
    REPO_ROOT
    / "Surrogate_GRP_COD"
    / "Cloudy_uw_HG_FY4A"
    / "metrics_channel_HG_V1.csv"
)
DEFAULT_ADM_LUT_DIR = REPO_ROOT / "FY4A_data" / "ADM_LUT"

SURROGATE_CHANNELS = ["C01", "C02", "C03", "C05", "C06"]
RETRIEVAL_CHANNELS = ["C01", "C02", "C05", "C06"]
PHASE_ICE = 1
MAX_SZA_DIFF_DEG = 1.0
MIN_VALID_GHI_CLEAR = 300.0
MIN_VALID_CLEARNESS = 0.15
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
    p, pa = set_pressure(n_layer)
    z, _ = set_height("AFGL midlatitude summer", p, pa)
    return p, pa, z


@lru_cache(maxsize=20000)
def compute_tpw_one(t_s: float, rh_percent: float) -> float:
    if not np.isfinite(t_s) or not np.isfinite(rh_percent):
        return np.nan
    p, pa, z = _atmosphere_grid()
    t, ta = set_temperature("AFGL midlatitude summer", p, pa, float(t_s), "day")
    ps = saturation_pressure(t)
    vmr0 = dict(VMR0_BASE)
    vmr0["H2O"] = np.clip(float(rh_percent) / 100.0, 0.0, 1.0) * ps[1] / p[1]
    _, densities = set_vmr("AFGL midlatitude summer", MOLECULES, vmr0, z)
    return float(total_precipitable_water(densities, pa, ta))


def compute_tpw_series(t_s: xr.DataArray, rh: xr.DataArray) -> np.ndarray:
    t_vals = np.asarray(t_s.values, dtype=float)
    rh_vals = np.asarray(rh.values, dtype=float)
    return np.array(
        [compute_tpw_one(round(t, 2), round(r, 2)) for t, r in zip(t_vals, rh_vals)],
        dtype=float,
    )


def apply_cloudy_time_qc(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    mask = np.ones(ds.sizes["time"], dtype=bool)
    stats = {
        "n_time_before_qc": int(ds.sizes["time"]),
        "n_sza_qc_removed": 0,
        "n_low_ghi_qc_removed": 0,
        "n_time_after_qc": int(ds.sizes["time"]),
    }

    if "Sun_Zen_ground" in ds and "Sun_Zen" in ds:
        sat_sza = ds["Sun_Zen"].median(dim=("y", "x"), skipna=True).values.astype(float)
        ground_sza = ds["Sun_Zen_ground"].values.astype(float)
        sza_ok = np.isfinite(sat_sza) & np.isfinite(ground_sza) & (np.abs(sat_sza - ground_sza) <= MAX_SZA_DIFF_DEG)
        stats["n_sza_qc_removed"] = int((~sza_ok).sum())
        mask &= sza_ok

    if "GHI" in ds:
        ghi = ds["GHI"].values.astype(float)
        if "GHI_clear" in ds:
            ghi_clear = ds["GHI_clear"].values.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                clearness = ghi / ghi_clear
            ghi_ok = (
                np.isfinite(ghi)
                & np.isfinite(ghi_clear)
                & np.isfinite(clearness)
                & (ghi_clear > MIN_VALID_GHI_CLEAR)
                & (clearness >= MIN_VALID_CLEARNESS)
            )
        else:
            ghi_ok = np.isfinite(ghi)
        stats["n_low_ghi_qc_removed"] = int((~ghi_ok).sum())
        mask &= ghi_ok

    filtered = ds.isel(time=mask)
    stats["n_time_after_qc"] = int(filtered.sizes["time"])
    return filtered, stats


def make_interpolator(spec: dict) -> RegularGridInterpolator:
    axes = tuple(np.asarray(spec["axes"][feature], dtype=float) for feature in spec["grid_features"])
    return RegularGridInterpolator(
        axes,
        np.asarray(spec["values"], dtype=float),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def build_cod_grid(bundle: dict, n_cod: int) -> np.ndarray:
    bounds = bundle["models"][SURROGATE_CHANNELS[0]]["bounds"]["log1p_COD"]
    log_grid = np.linspace(bounds[0], bounds[1], int(n_cod))
    return np.expm1(log_grid).astype(np.float32)


def load_channel_weights(metrics_path: Path, channels: list[str]) -> pd.Series:
    if not metrics_path.exists():
        return pd.Series(1.0, index=channels, dtype=float)

    metrics = pd.read_csv(metrics_path, index_col=0)
    if "RMSE" not in metrics.columns:
        return pd.Series(1.0, index=channels, dtype=float)

    rmse = metrics.reindex(channels)["RMSE"].astype(float)
    if rmse.isna().any() or not np.isfinite(rmse).all():
        return pd.Series(1.0, index=channels, dtype=float)
    weights = 1.0 / np.square(rmse.clip(lower=1e-8))
    return weights * (len(weights) / weights.sum())


def channel_albedo(ds: xr.Dataset, channel: str, bundle: dict) -> np.ndarray:
    for name in (f"WSA_{channel}", f"BSA_{channel}", f"alb_{channel}"):
        if name in ds:
            return np.asarray(ds[name].values, dtype=float)
    axis = np.asarray(bundle["models"][channel]["axes"][f"alb_{channel}"], dtype=float)
    return np.full(ds.sizes["time"], float(np.nanmedian(axis)), dtype=float)


def classify_ice_phase(ds: xr.Dataset) -> np.ndarray:
    """Return FY4A phase codes on (time, y, x); code 1 is ice cloud."""
    from Sat_Preprocessing.phasefilter.fy4a_cloud_phase_filter import classify_phase

    time_len = ds.sizes["time"]
    y_len = ds.sizes["y"]
    x_len = ds.sizes["x"]
    n_pixels = y_len * x_len
    arrays = {
        "C01": np.asarray(ds["C01"].values, dtype=float).reshape(time_len, n_pixels),
        "C05": np.asarray(ds["C05"].values, dtype=float).reshape(time_len, n_pixels),
        "C06": np.asarray(ds["C06"].values, dtype=float).reshape(time_len, n_pixels),
        "SunZenith": np.asarray(ds["Sun_Zen"].values, dtype=float).reshape(time_len, n_pixels),
    }
    products = classify_phase(
        arrays=arrays,
        times=pd.to_datetime(ds["time"].values),
        pixel_columns=[str(idx) for idx in range(n_pixels)],
    )
    return products.phase_code.reshape(time_len, y_len, x_len)


@lru_cache(maxsize=1)
def available_adm_cod_values(lut_dir: str) -> tuple[float, ...]:
    paths = sorted(Path(lut_dir).glob("angular_dist_lut_COD=*.h5"))
    cod_values = []
    for path in paths:
        match = re.search(r"COD=([-+]?\d*\.?\d+)", path.stem)
        if match:
            cod_values.append(float(match.group(1)))
    if not cod_values:
        raise FileNotFoundError(
            f"No angular_dist_lut_COD=*.h5 files found in {lut_dir}. "
            "Generate or copy the FY4A ADM files into FY4A_data/ADM_LUT first."
        )
    return tuple(sorted(set(cod_values)))


@lru_cache(maxsize=2048)
def load_adm_svd(lut_path: str, channel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(lut_path, "r") as f:
        channel_group = f[channel]
        return (
            f["solar_zeniths"][:],
            channel_group["U"][:],
            channel_group["S"][:],
            channel_group["VT"][:],
        )


@lru_cache(maxsize=2048)
def adm_matrix(lut_dir: str, channel: str, cod: float, target_zenith: float) -> np.ndarray:
    lut_path = Path(lut_dir) / f"angular_dist_lut_COD={int(cod)}.h5"
    solar_zeniths, U, S, VT = load_adm_svd(str(lut_path), channel)
    interp_kind = "quadratic" if target_zenith > 30.0 and len(solar_zeniths) >= 3 else "linear"
    U_interp = interp1d(
        solar_zeniths,
        U,
        kind=interp_kind,
        axis=0,
        fill_value="extrapolate",
    )(float(target_zenith))
    VT_interp = interp1d(
        solar_zeniths,
        VT,
        kind="linear",
        axis=0,
        fill_value="extrapolate",
    )(float(target_zenith))
    S_interp = np.array(
        [
            interp1d(solar_zeniths, S[:, r][:, 0], kind="linear", fill_value="extrapolate")(float(target_zenith))
            for r in range(S.shape[1])
        ],
        dtype=float,
    )
    return reconstruct_hc(U_interp, S_interp, VT_interp)


def adm_factor_for_pixels(
    lut_dir: Path,
    channel: str,
    cod_grid: np.ndarray,
    target_zenith: float,
    local_zen_flat: np.ndarray,
    rela_azi_flat: np.ndarray,
) -> np.ndarray:
    """Return ADM factors with shape ``(n_cod, n_pixels)``."""
    available_cod = np.asarray(available_adm_cod_values(str(lut_dir)), dtype=float)
    rela_azi = np.asarray(rela_azi_flat, dtype=float).copy()
    rela_azi[rela_azi > 180.0] = 360.0 - rela_azi[rela_azi > 180.0]

    available_factors = []
    for cod in available_cod:
        H_r = adm_matrix(str(lut_dir), channel, float(cod), round(float(target_zenith), 4))
        theta_edges = np.linspace(0.0, 90.0, H_r.shape[0] + 1)
        phi_edges = np.linspace(0.0, 180.0, H_r.shape[1] + 1)
        theta_idx = np.clip(np.digitize(local_zen_flat, theta_edges) - 1, 0, H_r.shape[0] - 1)
        phi_idx = np.clip(np.digitize(rela_azi, phi_edges) - 1, 0, H_r.shape[1] - 1)
        available_factors.append(H_r[theta_idx.astype(int), phi_idx.astype(int)])

    available_factors_arr = np.asarray(available_factors, dtype=np.float32)
    interpolator = interp1d(
        available_cod,
        available_factors_arr,
        axis=0,
        bounds_error=False,
        fill_value=(available_factors_arr[0], available_factors_arr[-1]),
        assume_sorted=True,
    )
    return interpolator(np.asarray(cod_grid, dtype=float)).astype(np.float32)


def adm_factor_for_valid_pixels_by_time(
    lut_dir: Path,
    channel: str,
    cod_grid: np.ndarray,
    sun_zen_values: np.ndarray,
    local_zen_flat: np.ndarray,
    rela_azi_flat: np.ndarray,
    valid_idx: np.ndarray,
    pixels_per_time: int,
) -> np.ndarray:
    """Return ADM factors for valid pixels, using one SZA per 11x11 scene."""
    factors = np.empty((len(cod_grid), len(valid_idx)), dtype=np.float32)
    time_idx = valid_idx // pixels_per_time
    for t_idx in np.unique(time_idx):
        out_pos = np.nonzero(time_idx == t_idx)[0]
        pix_idx = valid_idx[out_pos]
        target_zenith = float(np.round(np.nanmedian(sun_zen_values[int(t_idx)])))
        factors[:, out_pos] = adm_factor_for_pixels(
            lut_dir,
            channel,
            cod_grid,
            target_zenith,
            local_zen_flat[pix_idx],
            rela_azi_flat[pix_idx],
        )
    return factors


def predict_surrogate_channel(
    bundle: dict,
    channel: str,
    cod_grid: np.ndarray,
    tpw_flat: np.ndarray,
    cos_th0_flat: np.ndarray,
    alb_flat: np.ndarray,
    valid_idx: np.ndarray,
) -> np.ndarray:
    spec = bundle["models"][channel]
    interp = make_interpolator(spec)
    pred = np.empty((len(cod_grid), len(valid_idx)), dtype=np.float32)

    clipped_tpw = np.clip(tpw_flat[valid_idx], *spec["bounds"]["tpw"])
    clipped_cos = np.clip(cos_th0_flat[valid_idx], *spec["bounds"]["cos_th0"])
    alb_key = f"alb_{channel}"
    clipped_alb = np.clip(alb_flat[valid_idx], *spec["bounds"][alb_key])

    for cod_idx, cod in enumerate(cod_grid):
        log_cod = np.log1p(float(cod))
        points = np.column_stack(
            [
                np.full(len(valid_idx), log_cod, dtype=float),
                clipped_tpw,
                clipped_cos,
                clipped_alb,
            ]
        )
        pred[cod_idx, :] = interp(points).astype(np.float32)
    return pred


def retrieve_site_cod(
    path: Path,
    bundle: dict,
    adm_lut_dir: Path,
    out_dir: Path,
    cod_grid: np.ndarray,
    channel_weights: pd.Series,
    channels: list[str] | None = None,
    remove_ice: bool = False,
) -> dict:
    if channels is None:
        channels = RETRIEVAL_CHANNELS

    site = path.name.split("_SW_ref_satellite_cloudy.nc")[0]
    ds = xr.open_dataset(path).load()
    ds, qc_stats = apply_cloudy_time_qc(ds)
    if ds.sizes["time"] == 0:
        raise ValueError(f"No cloudy rows remain after QC for site {site}: {path}")
    time_len, y_len, x_len = ds[channels[0]].shape
    n_points = time_len * y_len * x_len
    phase_code = classify_ice_phase(ds) if remove_ice else None

    tpw = compute_tpw_series(ds["T_s"], ds["RH"])
    tpw_flat = np.repeat(tpw, y_len * x_len)
    cos_th0_flat = np.cos(np.deg2rad(np.asarray(ds["Sun_Zen"].values, dtype=float))).reshape(n_points)
    local_zen_flat = np.asarray(ds["Sat_Zen"].values, dtype=float).reshape(n_points)
    rela_azi_flat = np.asarray(ds["RAZ"].values, dtype=float).reshape(n_points)

    valid = np.isfinite(tpw_flat) & np.isfinite(cos_th0_flat) & np.isfinite(local_zen_flat) & np.isfinite(rela_azi_flat)
    for channel in channels:
        valid &= np.isfinite(np.asarray(ds[channel].values, dtype=float).reshape(n_points))
    valid &= np.any(
        np.column_stack([np.asarray(ds[channel].values, dtype=float).reshape(n_points) > 0.0 for channel in channels]),
        axis=1,
    )
    if phase_code is not None:
        valid &= (phase_code.reshape(n_points) != PHASE_ICE)

    cod = np.full(n_points, np.nan, dtype=np.float32)
    wrmse = np.full(n_points, np.nan, dtype=np.float32)
    channel_cod = {channel: np.full(n_points, np.nan, dtype=np.float32) for channel in channels}
    channel_error = {channel: np.full(n_points, np.nan, dtype=np.float32) for channel in channels}

    if valid.any():
        valid_idx = np.nonzero(valid)[0]
        weighted_sse = np.zeros((len(cod_grid), len(valid_idx)), dtype=np.float32)
        weight_sum = 0.0

        sun_zen_values = np.asarray(ds["Sun_Zen"].values, dtype=float)
        pixels_per_time = y_len * x_len
        for channel in channels:
            obs = np.asarray(ds[channel].values, dtype=float).reshape(n_points)[valid_idx]
            alb = channel_albedo(ds, channel, bundle)
            alb_flat = np.repeat(alb, y_len * x_len)
            pred = predict_surrogate_channel(
                bundle,
                channel,
                cod_grid,
                tpw_flat,
                cos_th0_flat,
                alb_flat,
                valid_idx,
            )
            adm = adm_factor_for_valid_pixels_by_time(
                adm_lut_dir,
                channel,
                cod_grid,
                sun_zen_values,
                local_zen_flat,
                rela_azi_flat,
                valid_idx,
                pixels_per_time,
            )
            corrected = pred * adm
            residual = corrected - obs[None, :]
            abs_error = np.abs(residual)
            best_ch = np.nanargmin(abs_error, axis=0)
            channel_cod[channel][valid_idx] = cod_grid[best_ch].astype(np.float32)
            channel_error[channel][valid_idx] = abs_error[best_ch, np.arange(len(valid_idx))].astype(np.float32)

            weight = float(channel_weights.get(channel, 1.0))
            weighted_sse += weight * np.square(residual).astype(np.float32)
            weight_sum += weight

        best = np.nanargmin(weighted_sse, axis=0)
        cod[valid_idx] = cod_grid[best].astype(np.float32)
        wrmse[valid_idx] = np.sqrt(weighted_sse[best, np.arange(len(valid_idx))] / weight_sum).astype(np.float32)

    data_vars = {
        "Retrieved_COD": (("time", "y", "x"), cod.reshape(time_len, y_len, x_len)),
        "WRMSE_sug_adm": (("time", "y", "x"), wrmse.reshape(time_len, y_len, x_len)),
        "tpw": (("time",), tpw.astype(np.float32)),
    }
    if phase_code is not None:
        data_vars["ice_phase_code"] = (("time", "y", "x"), phase_code.astype(np.int16))
    for channel in channels:
        data_vars[f"COD_{channel}"] = (("time", "y", "x"), channel_cod[channel].reshape(time_len, y_len, x_len))
        data_vars[f"reflectance_abs_error_{channel}"] = (
            ("time", "y", "x"),
            channel_error[channel].reshape(time_len, y_len, x_len),
        )
        data_vars[f"w_{channel}"] = ((), np.float32(channel_weights.get(channel, 1.0)))

    out = xr.Dataset(
        data_vars,
        coords={
            "time": ds["time"].values,
            "y": ds["y"].values,
            "x": ds["x"].values,
            "COD_grid": cod_grid.astype(np.float32),
        },
        attrs={
            "site": site,
            "source_file": str(path),
            "model_file": str(bundle.get("_model_path", "")),
            "adm_lut_dir": str(adm_lut_dir),
            "channels": ",".join(channels),
            "retrieval": "FY4A cloudy UW surrogate plus FY4A ADM angular correction; weighted residual minimization",
            "excluded_channels": "C03 vegetation-sensitive; C04 water-vapor absorption",
            "tpw_source": "computed from T_s and RH using LBL_funcs_fullSpectrum",
            "albedo_source": "time-only WSA_Cxx from FY4A cloudy NetCDF; fallback to BSA_Cxx/model median",
            "phase_filter": "ice cloud removed with FY4A C01/C05/C06 phase filter" if remove_ice else "none",
            "time_qc": (
                f"kept |median FY4A Sun_Zen - Sun_Zen_ground| <= {MAX_SZA_DIFF_DEG:g} deg; "
                f"kept PVLib clear-sky GHI > {MIN_VALID_GHI_CLEAR:g} W/m2 and "
                f"clear-sky index GHI/GHI_clear >= {MIN_VALID_CLEARNESS:g}"
            ),
            **qc_stats,
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{site}_cloudy_COD_uw_ADM.nc"
    encoding = {name: {"zlib": True, "complevel": 4} for name in out.data_vars}
    try:
        out.to_netcdf(out_path, encoding=encoding)
    except ValueError:
        out.to_netcdf(out_path)

    finite = np.isfinite(out["Retrieved_COD"].values)
    return {
        "site": site,
        "input_file": str(path),
        "output_file": str(out_path),
        "n_time": int(time_len),
        **qc_stats,
        "n_valid_pixels": int(finite.sum()),
        "cod_mean": float(np.nanmean(out["Retrieved_COD"].values)),
        "cod_median": float(np.nanmedian(out["Retrieved_COD"].values)),
        "wrmse_mean": float(np.nanmean(out["WRMSE_sug_adm"].values)),
    }


def retrieve_all_sites(
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    adm_lut_dir: Path = DEFAULT_ADM_LUT_DIR,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    n_cod: int = 80,
    sites: list[str] | None = None,
    channels: list[str] | None = None,
    remove_ice: bool = False,
) -> pd.DataFrame:
    if not adm_lut_dir.exists():
        raise FileNotFoundError(f"ADM LUT directory does not exist: {adm_lut_dir}")

    bundle = joblib.load(model_path)
    bundle["_model_path"] = str(model_path)
    cod_grid = build_cod_grid(bundle, n_cod)
    retrieval_channels = channels if channels is not None else RETRIEVAL_CHANNELS
    channel_weights = load_channel_weights(metrics_path, retrieval_channels)

    files = sorted(data_dir.glob("*_SW_ref_satellite_cloudy.nc"))
    if sites:
        wanted = {site.upper() for site in sites}
        files = [path for path in files if path.name.split("_SW_ref_satellite_cloudy.nc")[0].upper() in wanted]
    if not files:
        raise FileNotFoundError(f"No cloudy FY4A NetCDF files found in {data_dir}")

    rows = []
    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {path.name}")
        row = retrieve_site_cod(
            path,
            bundle,
            adm_lut_dir,
            out_dir,
            cod_grid,
            channel_weights,
            channels=retrieval_channels,
            remove_ice=remove_ice,
        )
        rows.append(row)
        print(f"  saved {Path(row['output_file']).name}; valid pixels={row['n_valid_pixels']}")

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "cloudy_cod_adm_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve FY4A cloudy COD with surrogate + FY4A ADM LUT.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory with *_SW_ref_satellite_cloudy.nc files.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for COD NetCDF files.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Cloudy UW surrogate pickle.")
    parser.add_argument("--adm-lut-dir", default=str(DEFAULT_ADM_LUT_DIR), help="Directory containing FY4A angular_dist_lut_COD=*.h5 files.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Per-channel metrics CSV for residual weights.")
    parser.add_argument("--n-cod", type=int, default=80, help="Number of log-spaced COD candidates between surrogate bounds.")
    parser.add_argument("--site", action="append", help="Optional site code. Repeat to process multiple sites.")
    parser.add_argument(
        "--channels",
        default=",".join(RETRIEVAL_CHANNELS),
        help="Comma-separated FY4A channels to use for COD retrieval, e.g. C01,C02,C05.",
    )
    parser.add_argument("--remove-ice", action="store_true", help="Remove pixels classified as ice cloud before COD retrieval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        retrieve_all_sites(
            data_dir=Path(args.data_dir),
            out_dir=Path(args.out_dir),
            model_path=Path(args.model),
            adm_lut_dir=Path(args.adm_lut_dir),
            metrics_path=Path(args.metrics),
            n_cod=args.n_cod,
            sites=args.site,
            channels=[channel.strip() for channel in args.channels.split(",") if channel.strip()],
            remove_ice=args.remove_ice,
        )


if __name__ == "__main__":
    main()
