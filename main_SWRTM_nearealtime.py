"""Post-process model outputs, and compare OLR and DLW."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

from SWRTM_Predictor import *
from fun_nearealtime_RTM import *
from main_SW_scope_nearealtime import (
    DEFAULT_ADM_LUT_DIR,
    DEFAULT_METRICS_PATH,
    adm_factor_for_pixels,
    available_adm_cod_values,
    load_adm_svd,
    load_channel_weights,
)
from Sat_Preprocessing.Funcs_satellite_processing import *


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CLOUDY_DATA_DIR = REPO_ROOT / "FY4A_data" / "Cloudy_site_sat_data"
DEFAULT_PHYSICAL_OUT_DIR = REPO_ROOT / "FY4A_validation" / "Cloudy_results" / "physical_RTM"
PHYSICAL_RETRIEVAL_CHANNELS = ["C01", "C02", "C05", "C06"]
PHYSICAL_ALBEDO_CHANNELS = ["C01", "C02", "C03", "C05", "C06"]

def nearealtime_COD_retrival(figlabel, site, phase, file_dir=None, sky="day", N_bundles = 10000):
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"
    xr_sat = Sat_preprocess(file_dir, site, figlabel, sky, phase, sat='FY4A')
    return xr_sat

def SW_RTM_retrival(xr_sat, file_dir, site):
    """
    Main driver function for Shortwave RTM Retrieval.
    1. Normalizes Satellite Data.
    2. Loads GPR Model (Once).
    3. Loops through time to:
       - Simulate RTM (Forward Model)
       - Retrieve COD (Inversion)
    """
    COD_v = np.concatenate([np.arange(0, 20, 2), np.arange(20, 50 + 5, 5)])
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    lut_dir = os.path.join(file_dir, 'FY4A_tool','FY4A_ADMLUT')
    # preprocess, normalize
    xr_sat['mu0'] = np.cos(np.deg2rad(xr_sat['th0']))
    da_sat = xr_sat[channels].to_array(dim='band')
    xr_sat['obs_ref'] = da_sat

    # --- 3. Load Model (ONCE) ---
    model_predictor = SWRTM_Predictor(file_dir+'FY4A_tool/GPR/')

    # --- 4. Main Processing Loop ---
    print(f"Starting Retrieval for {len(xr_sat.time)} timestamps...")

    retrieved_cod_list = []

    for t in xr_sat.time[:10]:
        time_tag = pd.to_datetime(t.values).strftime("%B %d, %H UTC+8")
        row = xr_sat.sel(time=t)
        # Forward Model: Predict Reflectance for all COD steps
        # Returns: (y, x, cod_v, band)
        da_sim_ref = predict_reflectance_scene(
            xr_sat_row=row,
            COD_v=COD_v,
            predictor_model=model_predictor,
            channels=channels,
            lut_dir=lut_dir
        )
        # C. Inversion: Interpolate to find COD
        # We compare Simulated (da_sim_ref) vs Observed (row['obs_ref'])
        # Output: (y, x, band) -> A COD map retrieved from EACH channel
        ds_retrieval = retrieve_cod_interp(
            da_sim_ref,
            row['obs_ref'],
            COD_v,
            time_tag
        )
        #xr_cod_map = retrieve_cod_vectorized(da_prediction, xr_sat_measure, COD_v)
        # D. Store results
        retrieved_cod_list.append(ds_retrieval)
    # --- 5. Concatenate and Return ---
    print("Concatenating results...")
    # Combine list of time-steps back into a single Xarray object
    xr_cod_result = xr.concat(retrieved_cod_list, dim='time')
    xr_cod_result['time'] = xr_sat.time[:10]  # Ensure time coords match

    out_dir = 'FY4A_data/CODresults/'
    os.makedirs(out_dir, exist_ok=True)
    xr_sat_subset = xr_sat.isel(time=slice(0, 10))
    xr_cod_result['time'] = xr_sat.time[:10]
    ds_debug = xr.merge([xr_sat_subset, xr_cod_result])
    filename = f'Results_FY4A_COD_{site}.nc'
    save_path = os.path.join(out_dir, filename)
    print(f"Saving debug file to: {save_path}")
    encoding_settings = {key: {'zlib': True, 'complevel': 4} for key in ds_debug.data_vars}
    ds_debug.to_netcdf(save_path, encoding=encoding_settings)
    print("Save complete.")

    for t in xr_sat.time[:10]:
        time_tag = pd.to_datetime(t.values).strftime("%B %d, %H UTC+8")
        row = ds_debug.sel(time=t)
        da_diff = validate_reflectance_scene(row,
                                 predictor_model=model_predictor,
                                 channels=channels,
                                 lut_dir=lut_dir,COD_v_grid = COD_v,
                                             t=time_tag)
    return None

def plot_debug_cod_map(da_result, title="Debug: Retrieved COD"):
    """
    Plots the Retrieved COD map and prints basic statistics for debugging.

    Args:
        da_result: xarray DataArray (y, x)
        title: String for the plot title
    """
    # 1. Print Stats
    # We use .item() to convert numpy/xarray scalars to normal python numbers
    d_min = da_result.min().item()
    d_max = da_result.max().item()
    d_mean = da_result.mean().item()
    n_nans = np.isnan(da_result).sum().item()

    print("=" * 30)
    print(f"STATS: {title}")
    print(f"  Min:  {d_min:.2f}")
    print(f"  Max:  {d_max:.2f}")
    print(f"  Mean: {d_mean:.2f}")
    print(f"  NaNs: {n_nans}")
    print("=" * 30)

    # 2. Plot
    plt.figure(figsize=(7, 6))

    # .squeeze() ensures we don't have a stray time dimension breaking the plot
    da_result.squeeze().plot(
        cmap='Blues',
        vmin=d_min, vmax=d_max,
        cbar_kwargs={'label': 'Optical Depth'}
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def validate_ghi(site, lat, lon, elev, out_dir):
    xr_sat = xr.open_dataset(out_dir + f'Results_FY4A_COD_{site}.nc')
    model_predictor = SWRTM_GHI_Predictor('Sat_Preprocessing/GPR/')
    pixel_res = 0.04  # degrees (~4km at equator)
    ghi_pred, ghi_obs = [], []
    Time = []
    for t in xr_sat.time[:10]:
        time_tag = pd.to_datetime(t.values).strftime("%B %d, %H UTC+8")
        row = xr_sat.sel(time=t)
        row = generate_latlon_grid(
            row,
            center_lat=lat,
            center_lon=lon,
            resolution_deg=pixel_res
        )
        #row_parallax = apply_parallax_correction(row, cloud_height_km= 7.0)
        #plot_parallax_comparison(row, row_parallax)

        ghi_simu = predict_GHI_scene(
            xr_sat_row=row,#_parallax,
            predictor_model=model_predictor, t=time_tag
        )
        [ghi_pred_row, ghi_obs_row] = ghi_simu
        ghi_pred.append(ghi_pred_row), ghi_obs.append(ghi_obs_row)
        Time.append(pd.to_datetime(t.values))
        # create a pandas dataframe to save the results
    df_result = pd.DataFrame({
        'Time': Time,
        'pre_GHI': ghi_pred,
        'obs_GHI': ghi_obs
        })

    return df_result


def _expol_func(x, a):
    return a * x**3


def _time_scalar(ds: xr.Dataset, name: str, time_index: int, default=np.nan) -> float:
    if name not in ds:
        return float(default)
    value = ds[name].isel(time=time_index).values
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _pixel_scalar(ds: xr.Dataset, name: str, flat_index: int, default=np.nan) -> float:
    if name not in ds:
        return float(default)
    value = np.asarray(ds[name].values, dtype=float).reshape(-1)
    if flat_index >= value.size:
        return float(default)
    return float(value[flat_index])


def _albedo_row(ds: xr.Dataset, time_index: int, surface: str) -> np.ndarray:
    black = []
    white = []
    for channel in PHYSICAL_ALBEDO_CHANNELS:
        black.append(_time_scalar(ds, f"BSA_{channel}", time_index, default=0.2))
        white.append(_time_scalar(ds, f"WSA_{channel}", time_index, default=0.2))

    values = black + white
    if surface == "BRDF":
        for suffix in ("p0", "p1", "p2"):
            for channel in PHYSICAL_ALBEDO_CHANNELS:
                values.append(_time_scalar(ds, f"Abdo_{channel}_{suffix}", time_index, default=0.0))

    values = np.asarray(values, dtype=float)
    fallback = 0.0 if surface == "BRDF" else 0.2
    return np.nan_to_num(values, nan=fallback, posinf=fallback, neginf=fallback)


def _weighted_loss(obs: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float, float]:
    residual = pred - obs
    weighted_mse = float(np.sum(weights * np.square(residual)) / np.sum(weights))
    weighted_rmse = float(np.sqrt(weighted_mse))
    return residual, weighted_mse, weighted_rmse


def _compute_gradient_simple_weighted(
    rtm_history: pd.DataFrame,
    cod_history: list[float],
    obs: np.ndarray,
    costs: list[float],
    weights: np.ndarray,
    min_cod: float,
    max_cod: float,
) -> float:
    """GOES gradient-step logic on weighted channel-reflectance space."""
    if len(cod_history) < 2:
        return 20.0

    weighted_scale = np.sqrt(np.asarray(weights, dtype=float))
    rtm_values = rtm_history.to_numpy(dtype=float) * weighted_scale
    obs_values = np.asarray(obs, dtype=float) * weighted_scale
    nearest_indices = np.argsort(costs)[:2]

    rtm0 = rtm_values[nearest_indices[0]]
    rtm1 = rtm_values[nearest_indices[1]]
    cod0 = float(cod_history[nearest_indices[0]])
    cod1 = float(cod_history[nearest_indices[1]])
    denom = np.linalg.norm(rtm1 - rtm0)
    if denom <= 1e-12:
        return cod0

    slope_mag = abs(cod1 - cod0) / denom
    step_sign = 1.0 if np.linalg.norm(obs_values) > np.linalg.norm(rtm0) else -1.0
    cod_guess = cod0 + step_sign * slope_mag * np.linalg.norm(obs_values - rtm0)

    repeated = np.any(np.abs(np.asarray(cod_history, dtype=float) - cod_guess) <= 1.0)
    if cod_guess < min_cod or cod_guess > max_cod or (repeated and len(cod_history) > 2):
        x = np.linalg.norm(rtm_values, axis=1)
        y = np.asarray(cod_history, dtype=float)
        x_obs = np.linalg.norm(obs_values)
        if np.unique(np.round(x, 8)).size >= 2:
            try:
                popt, _ = curve_fit(_expol_func, x, y, bounds=(1e-8, 200))
                cod_guess = float(_expol_func(x_obs, popt[0]))
            except Exception:
                order = np.argsort(x)
                cod_guess = float(np.interp(x_obs, x[order], y[order], left=y[order][0], right=y[order][-1]))
        else:
            cod_guess = cod0

    return float(np.clip(cod_guess, min_cod, max_cod))


def _rtm_cache_key(
    sun_zen: float,
    cod: float,
    t_s: float,
    rh: float,
    albedo_row: np.ndarray,
    surface: str,
    meth: str,
    aod: float,
) -> tuple:
    return (
        round(float(sun_zen)),
        round(float(cod)),
        round(float(t_s)),
        round(float(rh)),
        round(float(aod), 2),
        surface,
        meth,
        tuple(np.round(np.asarray(albedo_row, dtype=float), 5)),
    )


def _albedo_cache_suffix(albedo_row: np.ndarray) -> str:
    arr = np.nan_to_num(np.asarray(albedo_row, dtype=np.float32), nan=-9999.0)
    digest = hashlib.sha1(np.round(arr, 5).tobytes()).hexdigest()[:10]
    return f"alb{digest}"


def _physical_upwelling_base_reflectance(
    sun_zen: float,
    local_zen: float,
    rela_azi: float,
    cod: float,
    t_s: float,
    rh: float,
    albedo_row: np.ndarray,
    surface: str,
    meth: str,
    aod: float,
    rtm_cache: dict,
) -> pd.Series:
    cod_eval = float(round(np.clip(cod, 0, 50)))
    key = _rtm_cache_key(sun_zen, cod_eval, t_s, rh, albedo_row, surface, meth, aod)
    if key not in rtm_cache:
        df_ref = nearealtime_LUT(
            sun_zen,
            local_zen,
            rela_azi,
            cod_eval,
            t_s,
            rh,
            file_dir=str(REPO_ROOT),
            bandmode="FY4A",
            df_albedo=albedo_row,
            surface=surface,
            meth=meth,
            AOD=aod,
            cache_suffix=_albedo_cache_suffix(albedo_row),
        )
        rtm_cache[key] = df_ref.iloc[0].astype(float)
    return rtm_cache[key]


def _physical_pixel_reflectance(
    sun_zen: float,
    local_zen: float,
    rela_azi: float,
    cod: float,
    t_s: float,
    rh: float,
    albedo_row: np.ndarray,
    surface: str,
    meth: str,
    aod: float,
    adm_lut_dir: Path,
    channels: list[str],
    rtm_cache: dict,
) -> pd.Series:
    cod_eval = float(round(np.clip(cod, 0, 50)))
    base = _physical_upwelling_base_reflectance(
        sun_zen,
        local_zen,
        rela_azi,
        cod_eval,
        t_s,
        rh,
        albedo_row,
        surface,
        meth,
        aod,
        rtm_cache,
    )
    values = {}
    for channel in channels:
        adm = adm_factor_for_pixels(
            adm_lut_dir,
            channel,
            np.asarray([cod_eval], dtype=float),
            round(float(sun_zen), 4),
            np.asarray([local_zen], dtype=float),
            np.asarray([rela_azi], dtype=float),
        )
        values[channel] = float(base[channel]) * float(adm[0, 0])
    return pd.Series(values, dtype=float)


def _retrieve_cod_pixel_physical(
    obs: np.ndarray,
    sun_zen: float,
    local_zen: float,
    rela_azi: float,
    t_s: float,
    rh: float,
    albedo_row: np.ndarray,
    surface: str,
    meth: str,
    aod: float,
    adm_lut_dir: Path,
    channels: list[str],
    weights: np.ndarray,
    rtm_cache: dict,
    max_iterations: int = 5,
    epsilon: float = 1e-4,
    min_cod: float = 0.0,
    max_cod: float = 50.0,
) -> dict:
    cod_guess = 10.0
    cod_history: list[float] = []
    costs: list[float] = []
    wrmse_history: list[float] = []
    pred_history = []

    for iteration in range(max_iterations):
        cod_eval = float(round(np.clip(cod_guess, min_cod, max_cod)))
        if cod_eval in cod_history:
            break

        pred = _physical_pixel_reflectance(
            sun_zen,
            local_zen,
            rela_azi,
            cod_eval,
            t_s,
            rh,
            albedo_row,
            surface,
            meth,
            aod,
            adm_lut_dir,
            channels,
            rtm_cache,
        )
        pred_values = pred.reindex(channels).to_numpy(dtype=float)
        residual, cost, wrmse = _weighted_loss(obs, pred_values, weights)

        cod_history.append(cod_eval)
        costs.append(cost)
        wrmse_history.append(wrmse)
        pred_history.append(pred)

        if cost < epsilon:
            break
        if len(cod_history) < 2:
            cod_guess = 20.0
        else:
            cod_guess = _compute_gradient_simple_weighted(
                pd.DataFrame(pred_history)[channels],
                cod_history,
                obs,
                costs,
                weights,
                min_cod,
                max_cod,
            )

    if not costs:
        return {
            "cod": np.nan,
            "wrmse": np.nan,
            "n_iter": 0,
            "pred": pd.Series(np.nan, index=channels),
            "channel_cod": pd.Series(np.nan, index=channels),
            "channel_error": pd.Series(np.nan, index=channels),
        }

    best_idx = int(np.nanargmin(costs))
    best_pred = pred_history[best_idx].reindex(channels).astype(float)
    history_df = pd.DataFrame(pred_history)[channels].astype(float)
    channel_cod = {}
    channel_error = {}
    for channel, obs_value in zip(channels, obs):
        abs_error = np.abs(history_df[channel].to_numpy(dtype=float) - obs_value)
        channel_best_idx = int(np.nanargmin(abs_error))
        channel_cod[channel] = cod_history[channel_best_idx]
        channel_error[channel] = float(abs_error[channel_best_idx])

    return {
        "cod": float(cod_history[best_idx]),
        "wrmse": float(wrmse_history[best_idx]),
        "n_iter": len(costs),
        "pred": best_pred,
        "channel_cod": pd.Series(channel_cod, dtype=float),
        "channel_error": pd.Series(channel_error, dtype=float),
    }


def _downwelling_cache_key(
    sun_zen: float,
    cod: float,
    t_s: float,
    rh: float,
    albedo_row: np.ndarray,
    surface: str,
    meth: str,
    aod: float,
    theta_trunc_cld: float,
    escape_scale: float,
    escape_use_g2: bool,
) -> tuple:
    return (
        round(float(sun_zen)),
        round(float(cod)),
        round(float(t_s)),
        round(float(rh)),
        round(float(aod), 2),
        surface,
        meth,
        float(theta_trunc_cld),
        float(escape_scale),
        bool(escape_use_g2),
        tuple(np.round(np.asarray(albedo_row, dtype=float), 5)),
    )


def _physical_downwelling(
    sun_zen: float,
    local_zen: float,
    rela_azi: float,
    cod: float,
    t_s: float,
    rh: float,
    albedo_row: np.ndarray,
    surface: str,
    meth: str,
    aod: float,
    dw_cache: dict,
    theta_trunc_cld: float = 3,
    escape_scale: float = 0.0,
    escape_use_g2: bool = False,
) -> tuple[float, float, float]:
    cod_eval = float(round(np.clip(cod, 0, 50)))
    key = _downwelling_cache_key(
        sun_zen,
        cod_eval,
        t_s,
        rh,
        albedo_row,
        surface,
        meth,
        aod,
        theta_trunc_cld,
        escape_scale,
        escape_use_g2,
    )
    if key not in dw_cache:
        dw_cache[key] = get_rtm_output_cld(
            sun_zen,
            local_zen,
            rela_azi,
            cod_eval,
            t_s,
            rh,
            albedo_row,
            surface=surface,
            meth=meth,
            AOD=aod,
            theta_trunc_cld=theta_trunc_cld,
            escape_scale=escape_scale,
            escape_use_g2=escape_use_g2,
            cache_suffix=_albedo_cache_suffix(albedo_row),
        )
    return dw_cache[key]


def retrieve_site_physical_cloudy(
    path: Path,
    out_dir: Path = DEFAULT_PHYSICAL_OUT_DIR,
    adm_lut_dir: Path = DEFAULT_ADM_LUT_DIR,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    channels: list[str] | None = None,
    surface: str = "MODIS",
    uw_meth: str = "HG",
    dw_meth: str = "dM",
    aod_default: float = 0.1243,
    max_iterations: int = 5,
    epsilon: float = 1e-4,
    max_times: int | None = None,
    dw_mode: str = "center",
) -> dict:
    """Retrieve cloudy COD and cloudy DW RTM for only the station center pixel."""
    if dw_mode not in {"none", "center"}:
        raise ValueError("center-only physical retrieval supports dw_mode='none' or 'center'")

    channels = channels or PHYSICAL_RETRIEVAL_CHANNELS
    available_adm_cod_values(str(adm_lut_dir))
    channel_weights = load_channel_weights(metrics_path, channels)
    weights = channel_weights.reindex(channels).fillna(1.0).to_numpy(dtype=float)

    site = path.name.split("_SW_ref_satellite_cloudy.nc")[0]
    ds = xr.open_dataset(path).load()
    if max_times is not None:
        ds = ds.isel(time=slice(0, max_times))

    time_len, y_len, x_len = ds[channels[0]].shape
    center_y = y_len // 2
    center_x = x_len // 2

    cod_center = np.full(time_len, np.nan, dtype=np.float32)
    wrmse_center = np.full(time_len, np.nan, dtype=np.float32)
    n_iter_center = np.zeros(time_len, dtype=np.int16)
    pred_ref = {channel: np.full(time_len, np.nan, dtype=np.float32) for channel in channels}
    channel_cod = {channel: np.full(time_len, np.nan, dtype=np.float32) for channel in channels}
    channel_error = {channel: np.full(time_len, np.nan, dtype=np.float32) for channel in channels}

    rtm_cache: dict = {}
    dw_cache: dict = {}
    ghi_center = np.full(time_len, np.nan, dtype=np.float32)
    dni_center = np.full(time_len, np.nan, dtype=np.float32)
    dhi_center = np.full(time_len, np.nan, dtype=np.float32)

    for t_idx in range(time_len):
        print(f"{site}: physical cloudy retrieval {t_idx + 1}/{time_len}")
        row = ds.isel(time=t_idx)
        t_s = _time_scalar(ds, "T_s", t_idx)
        rh = _time_scalar(ds, "RH", t_idx)
        aod = _time_scalar(ds, "aod", t_idx, default=aod_default)
        if not np.isfinite(aod):
            aod = aod_default
        albedo = _albedo_row(ds, t_idx, surface)

        sun_zen = float(np.asarray(row["Sun_Zen"].values, dtype=float)[center_y, center_x])
        local_zen = float(np.asarray(row["Sat_Zen"].values, dtype=float)[center_y, center_x])
        rela_azi = float(np.asarray(row["RAZ"].values, dtype=float)[center_y, center_x])
        obs = np.asarray(
            [float(np.asarray(row[channel].values, dtype=float)[center_y, center_x]) for channel in channels],
            dtype=float,
        )
        valid_center = (
            np.isfinite(t_s)
            and np.isfinite(rh)
            and np.isfinite(sun_zen)
            and np.isfinite(local_zen)
            and np.isfinite(rela_azi)
            and np.isfinite(obs).all()
            and sun_zen <= 65.0
            and np.any(obs > 0.0)
        )

        if not valid_center:
            continue

        result = _retrieve_cod_pixel_physical(
            obs,
            sun_zen,
            local_zen,
            rela_azi,
            t_s,
            rh,
            albedo,
            surface,
            uw_meth,
            aod,
            adm_lut_dir,
            channels,
            weights,
            rtm_cache,
            max_iterations=max_iterations,
            epsilon=epsilon,
        )
        cod_center[t_idx] = result["cod"]
        wrmse_center[t_idx] = result["wrmse"]
        n_iter_center[t_idx] = result["n_iter"]
        for channel in channels:
            pred_ref[channel][t_idx] = result["pred"][channel]
            channel_cod[channel][t_idx] = result["channel_cod"][channel]
            channel_error[channel][t_idx] = result["channel_error"][channel]

        if dw_mode == "center" and np.isfinite(cod_center[t_idx]):
            dsw, dni, dhi = _physical_downwelling(
                sun_zen,
                local_zen,
                rela_azi,
                cod_center[t_idx],
                t_s,
                rh,
                albedo,
                surface,
                dw_meth,
                aod,
                dw_cache,
            )
            ghi_center[t_idx] = dsw
            dni_center[t_idx] = dni
            dhi_center[t_idx] = dhi

    data_vars = {
        "Retrieved_COD": (("time",), cod_center),
        "WRMSE_rtm_adm": (("time",), wrmse_center),
        "n_iter_rtm_cod": (("time",), n_iter_center),
        "GHI_rtm_center": (("time",), ghi_center),
        "DNI_rtm_center": (("time",), dni_center),
        "DHI_rtm_center": (("time",), dhi_center),
        "COD_center": (("time",), cod_center),
    }
    for channel in channels:
        data_vars[f"rtm_reflectance_{channel}"] = (
            ("time",),
            pred_ref[channel],
        )
        data_vars[f"COD_{channel}"] = (
            ("time",),
            channel_cod[channel],
        )
        data_vars[f"reflectance_abs_error_{channel}"] = (
            ("time",),
            channel_error[channel],
        )
        data_vars[f"w_{channel}"] = ((), np.float32(channel_weights.get(channel, 1.0)))

    out = xr.Dataset(
        data_vars,
        coords={"time": ds["time"].values},
        attrs={
            "site": site,
            "source_file": str(path),
            "retrieval": "FY4A physical UW RTM plus FY4A ADM angular correction; GOES-style gradient descent",
            "downwelling": "physical cloudy SW RTM driven by retrieved COD",
            "pixel_mode": "center_only",
            "center_y_index": center_y,
            "center_x_index": center_x,
            "source_y_size": y_len,
            "source_x_size": x_len,
            "channels": ",".join(channels),
            "excluded_channels": "C03 vegetation-sensitive; C04 water-vapor absorption",
            "surface": surface,
            "uw_meth": uw_meth,
            "dw_meth": dw_meth,
            "aod_default": aod_default,
            "adm_lut_dir": str(adm_lut_dir),
            "metrics_path": str(metrics_path),
            "time_qc": "none",
            "phase_filter": "none",
            "dw_mode": dw_mode,
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{site}_cloudy_COD_physical_RTM.nc"
    encoding = {name: {"zlib": True, "complevel": 4} for name in out.data_vars}
    try:
        out.to_netcdf(out_path, encoding=encoding)
    except ValueError:
        out.to_netcdf(out_path)

    center_table = pd.DataFrame(
        {
            "time": pd.to_datetime(ds["time"].values),
            "GHI_ground": ds["GHI"].values if "GHI" in ds else np.nan,
            "GHI_clear": ds["GHI_clear"].values if "GHI_clear" in ds else np.nan,
            "GHI_rtm_center": ghi_center,
            "DNI_rtm_center": dni_center,
            "DHI_rtm_center": dhi_center,
            "COD_center": cod_center,
            "WRMSE_center": wrmse_center,
        }
    )
    center_csv = out_dir / f"{site}_cloudy_physical_RTM_center.csv"
    center_table.to_csv(center_csv, index=False)

    finite_cod = np.isfinite(cod_center)
    return {
        "site": site,
        "input_file": str(path),
        "output_file": str(out_path),
        "center_csv": str(center_csv),
        "n_time": int(time_len),
        "n_valid_pixels": int(finite_cod.sum()),
        "n_valid_center_pixels": int(finite_cod.sum()),
        "cod_mean": float(np.nanmean(cod_center)) if finite_cod.any() else np.nan,
        "cod_median": float(np.nanmedian(cod_center)) if finite_cod.any() else np.nan,
        "wrmse_mean": float(np.nanmean(wrmse_center)) if np.isfinite(wrmse_center).any() else np.nan,
        "rtm_cache_size": len(rtm_cache),
        "dw_cache_size": len(dw_cache),
    }


def retrieve_all_physical_cloudy(
    data_dir: Path = DEFAULT_CLOUDY_DATA_DIR,
    out_dir: Path = DEFAULT_PHYSICAL_OUT_DIR,
    adm_lut_dir: Path = DEFAULT_ADM_LUT_DIR,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    sites: list[str] | None = None,
    channels: list[str] | None = None,
    surface: str = "MODIS",
    uw_meth: str = "HG",
    dw_meth: str = "dM",
    aod_default: float = 0.1243,
    max_iterations: int = 5,
    epsilon: float = 1e-4,
    max_times: int | None = None,
    dw_mode: str = "center",
) -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("*_SW_ref_satellite_cloudy.nc"))
    if sites:
        wanted = {site.upper() for site in sites}
        files = [path for path in files if path.name.split("_SW_ref_satellite_cloudy.nc")[0].upper() in wanted]
    if not files:
        raise FileNotFoundError(f"No FY4A cloudy NetCDF files found in {data_dir}")

    rows = []
    for idx, path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {path.name}")
        rows.append(
            retrieve_site_physical_cloudy(
                path=path,
                out_dir=out_dir,
                adm_lut_dir=adm_lut_dir,
                metrics_path=metrics_path,
                channels=channels,
                surface=surface,
                uw_meth=uw_meth,
                dw_meth=dw_meth,
                aod_default=aod_default,
                max_iterations=max_iterations,
                epsilon=epsilon,
                max_times=max_times,
                dw_mode=dw_mode,
            )
        )

    manifest = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "cloudy_physical_rtm_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")
    return manifest

# =============================================================================
# Helper: The Inversion Function (Sim -> COD)
# =============================================================================
def retrieve_cod_interp(da_sim_ref, da_obs, cod_grid,t):
    """
    1. Calculates Sum of Squared Errors (SSE) across bands.
    2. Finds the grid point with minimum error.
    3. Performs Parabolic Interpolation to find the exact sub-grid COD.
    """

    # --- 1. Calculate Cost Function (SSE) ---
    # da_sim_ref: (y, x, cod, band)
    # da_obs:     (band, y, x) -> Broadcasts to (y, x, 1, band)

    # Calculate difference, square it, and sum over bands
    # Result shape: (y, x, cod)
    is_smooth = map_sooth_identifier(da_obs)
    if is_smooth:
        sse_map = ((da_sim_ref - da_obs) ** 2).sum(dim='band')
    else:
        print("not smooth, pixel detection applied!")
    # --- 2. Find Discrete Minimum ---
    # "Which idx in COD grid (0th, 1st, 2nd...) gave the best fit?"
    min_indices = sse_map.argmin(dim='cod')

    # --- 3. Prepare for Interpolation (Numpy) ---
    # We drop to numpy for complex indexing
    sse_vals = sse_map.values  # (y, x, n_cod)
    idx = min_indices.values  # (y, x)
    grid = np.array(cod_grid)  # (22,)

    ny, nx = idx.shape

    # Handle Edge Cases:
    # If min is at index 0 or last index, we cannot interpolate (no neighbor).
    # We clip indices to be at least 1 and at most len-2 for the math,
    # then fix the values later.
    safe_idx = np.clip(idx, 1, len(grid) - 2)

    # Create coordinate grids for advanced indexing
    # We want values at safe_idx-1 (left), safe_idx (center), safe_idx+1 (right)
    grid_y, grid_x = np.indices((ny, nx))

    # --- 4. Extract (x, y) points for 3 neighbors ---
    # x = COD value, y = Error (SSE)

    # Left Point
    x1 = grid[safe_idx - 1]
    y1 = sse_vals[grid_y, grid_x, safe_idx - 1]

    # Center Point (The discrete minimum)
    x2 = grid[safe_idx]
    y2 = sse_vals[grid_y, grid_x, safe_idx]

    # Right Point
    x3 = grid[safe_idx + 1]
    y3 = sse_vals[grid_y, grid_x, safe_idx + 1]

    # --- 5. Inverse Parabolic Interpolation Formula ---
    # We want the 'x' where the parabola vertex is.
    # Formula for non-uniform x-spacing:
    # numer = (x2^2 - x3^2)*y1 + (x3^2 - x1^2)*y2 + (x1^2 - x2^2)*y3
    # denom = (x2 - x3)*y1 + (x3 - x1)*y2 + (x1 - x2)*y3
    # x_min = 0.5 * numer / denom

    numer = (x2 ** 2 - x3 ** 2) * y1 + (x3 ** 2 - x1 ** 2) * y2 + (x1 ** 2 - x2 ** 2) * y3
    denom = (x2 - x3) * y1 + (x3 - x1) * y2 + (x1 - x2) * y3

    # Avoid division by zero (flat surface)
    denom[denom == 0] = 1e-6

    cod_interpolated = 0.5 * numer / denom

    # --- 6. Safety Checks ---
    # If the indices were at the edges (0 or max), keep the discrete value
    # mask_edges is True where the original index was 0 or max
    #mask_edges = (idx == 0) | (idx == len(grid) - 1)

    # Also, if interpolation went wild (outside range [x1, x3]), revert to grid
    mask_wild = (cod_interpolated < x1) | (cod_interpolated > x3)

    # Apply corrections
    # Where edges or wild, use x2 (the discrete grid value)
    cod_final = np.where(mask_wild, x2, cod_interpolated)
    # 3. CRITICAL: Physical bounds check
    #    The parabola might sometimes predict -0.5. We must clip to 0.
    cod_final = np.clip(cod_final, grid[0], grid[-1])

    # --- 7. Package into Xarray ---
    da_result = xr.DataArray(
        cod_final,
        coords={'y': da_sim_ref.y, 'x': da_sim_ref.x},
        dims=('y', 'x'),
        name='Retrieved_COD'
    )
    plot_debug_cod_map(da_result, title=f"COD, {t}")
    return da_result

def predict_GHI_scene(xr_sat_row, predictor_model, t, show_plot=True):
    """
    1. Predicts GHI using inputs ('Ta', 'rh', 'mu0', 'COD')
    2. Compares with reference xr_sat_row['GHI']
    3. Plots the diff for debug and calculates metrics
    """

    # --- A. Prepare Data & Geometry ---
    # Note: Ensure the variable name matches your previous step ('Retrieved_COD' vs 'Retrived_COD')
    cod_key = 'Retrieved_COD'

    ny, nx = xr_sat_row[cod_key].shape
    n_pixels = ny * nx

    # Flatten inputs
    mu0_flat = xr_sat_row['mu0'].values.flatten()
    ta_val = xr_sat_row['T_a'].values
    rh_val = xr_sat_row['rh'].values  # Ensure key matches (RH vs rh)
    cod_val = xr_sat_row[cod_key].values

    # Handle Scalar vs Map
    ta_flat = np.full(n_pixels, ta_val) if np.ndim(ta_val) == 0 else ta_val.flatten()
    rh_flat = np.full(n_pixels, rh_val) if np.ndim(rh_val) == 0 else rh_val.flatten()

    # FIX: cod_val is already a map, just flatten it. Do not use np.full.
    cod_flat = cod_val.flatten()

    # Stack: [Ta, RH, mu0, COD]
    X_batch = np.column_stack((ta_flat, rh_flat, mu0_flat, cod_flat))

    # --- B. Predict Flux ---
    # Initialize with NaNs
    ghi_pred_flat = np.full((n_pixels,), np.nan)

    # Check valid pixels (masking NaNs in input)
    valid_mask = ~np.isnan(X_batch).any(axis=1)

    if np.any(valid_mask):
        X_valid = X_batch[valid_mask]
        # Predict GHI (Output is 1D array of Flux)
        # Assuming predictor_model.predict returns shape (N,) or (N, 1)
        pred_result = predictor_model.predict(X_valid)

        # Handle shape if it returns (N, 1)
        if pred_result.ndim > 1:
            pred_result = pred_result.flatten()

        ghi_pred_flat[valid_mask] = pred_result

    # --- C. Reshape to Map ---
    ghi_pred_map = ghi_pred_flat.reshape(ny, nx)

    # Wrap in Xarray
    da_ghi_pred = xr.DataArray(
        ghi_pred_map,
        coords={'y': xr_sat_row.y, 'x': xr_sat_row.x},
        dims=('y', 'x'),
        name='Predicted_GHI'
    )

    # --- D. Compare & Metrics ---
    # Get Ground Truth / Reference GHI
    if 'GHI' in xr_sat_row:
        ghi_obs = xr_sat_row['GHI'].values
        # Calculate Diff
        diff_map = ghi_pred_map - ghi_obs

        # Calculate Metrics (ignoring NaNs)
        valid_idx = ~np.isnan(diff_map)
        if np.any(valid_idx):
            rmse = np.sqrt(np.mean(diff_map[valid_idx] ** 2))
            mbe = np.mean(diff_map[valid_idx])  # Mean Bias Error
            #corr = np.corrcoef(ghi_pred_map[valid_idx], ghi_obs[valid_idx])[0, 1]
        else:
            rmse, mbe, corr = np.nan, np.nan, np.nan

        print("=" * 30)
        print(f"GHI Evaluation:")
        print(f"  RMSE: {rmse:.2f} W/m2")
        print(f"  MBE:  {mbe:.2f} W/m2")
        #print(f"  R:    {corr:.4f}")
        print("=" * 30)

        # --- E. Plotting ---
        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(9, 5))

            # 1. Prediction
            im0 = axes[0].imshow(ghi_pred_map, cmap='jet', vmin=0, vmax=1000)
            axes[0].set_title(f"GHI_RTM, th0={xr_sat_row['th0'].mean():.1f}°")
            plt.colorbar(im0, ax=axes[0], label='W/m2',shrink=0.7)

            # 3. Difference
            # Use centered colormap (RdBu) to see +/- bias
            limit = np.nanmax(np.abs(diff_map)) if not np.all(np.isnan(diff_map)) else 100
            im2 = axes[1].imshow(diff_map, cmap='RdBu_r', vmin=-limit, vmax=limit)
            cy, cx = ny // 2, nx // 2
            axes[1].scatter(cx, cy, c='black', marker='x', s=100)
            axes[1].set_title(f"Diff (Pred - Obs)\nRMSE={rmse:.1f}")
            plt.colorbar(im2, ax=axes[1], label='Difference (W/m2)',shrink=0.7)
            plt.title(f"{t}")
            plt.tight_layout()
            plt.show()
    return ghi_pred_map[nx//2,ny//2], ghi_obs[nx//2,ny//2]

def ADM_convert(df_row, local_zen, rela_azi, channels, fdir='./FY4A_tool/FY4A_ADMLUT/'):
    """

    Parameters
    ----------
    df_row
    local_zen
    rela_azi
    channels
    fdir

    Returns
    -------

    """


    COD_v = np.concatenate([np.arange(0, 22, 2), np.arange(20, 50 + 5, 5)])
    theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
    RTM_ref_flux = pd.DataFrame([
        {**df_row.to_dict(), 'COD_v': cod} for cod in COD_v
    ])
    ref_rtm = RTM_ref_flux.copy()
    for i, COD in enumerate(COD_v):
        for channel in channels:
            U, S, VT = load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD)}.h5', channel, target_zenith)
            H_r = reconstruct_hc(U, S, VT)
            ref_rtm.loc[i, channel] = RTM_ref_flux.loc[i, channel]/np.pi * H_r[theta_idx, phi_idx]  # correct uw_channel
    return ref_rtm

def compare_clear_dsw(site, sourcefile, meth='HG', surface = 'MODIS', sky="clear", file_dir=None, figlabel=None):
    """Compare the ground and modeled DLW.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + satellite data files.
    F_uw_srf: is surface upwelling. not spectral response function.

    Returns
    -------
    None

    """
    # not used for current version
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"
    csvfile = ('./FY4A_validation/Clear_Test/' + f"Result_{timeofday}_{site}_radiance_satellite_{sky}_{meth}_{surface}_sample.csv")
               #f"BJC_radiance_satellite_clear.csv"

    rtm_dsw, rtm_dni, rtm_dhi, rtm_uw, rtm_uw_srf = [], [], [], [], []
    uw_channels_list = []
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    rtm_channels = [c + '_rtm' for c in channels]
    try:
        df_combined = pd.read_csv(csvfile)
        df_combined['Time'] = pd.to_datetime(df_combined['Time'])
        df_combined = df_combined.dropna()
        # try:
        #     df_combined['rtm_dni_1HG'] = df_combined['rtm_dni_1HG']/np.cos(np.deg2rad(df_combined['Site_zen']))
        # except Exception:
        #     pass
        rtm_GHI, rtm_DNI = df_combined['rtm_dsw'],df_combined['rtm_dni']
        
        site_GHI = df_combined['ghi']
        print('dsw read from existing csv file:', csvfile)
    except Exception:
        # open Satellite observation data
        file_path = sourcefile
        sat = pd.read_csv(file_path)

        print('surface : ', surface)
        df_albedo = cal_surface_albedo(sat, surface)
        site_GHI = sat['ghi']  # should not be sat[dw_ir], it includes the surface reflected radiation
        print(f"Run RTM to get output DSW.")

        sat['Time'] = pd.to_datetime(sat['Time'])
        sat = sat.set_index('Time')
        sat = sat.sort_index()
        if site=='BJC':
            from aod_codes import read_aod  # add_aod_to_sat
            # aod time series from surfrad
            df_aod = read_aod(site)
            sat = pd.merge_asof(sat, df_aod,
                                left_index=True,
                                right_index=True,
                                direction='nearest',
                                tolerance=pd.Timedelta('3min'))
            sat = sat.dropna(subset=['aod'])
            #c_index = [index for index in sat.index if index in df_aod.index]
            # sat = pd.concat([sat.loc[c_index], df_aod.loc[c_index]], join='inner', axis=1)
        # pre analysis
        sat = sat[sat['T_s'] > 283]
        sat = sat[sat['Sun_Zen'] < 65]
        # Select months 4 through 10 (inclusive)
        # sat_filtered = sat[(sat.index.month >= 4) & (sat.index.month <= 10)].copy()
        # #plot_zen_uw(sat_filtered['Sun_Zen'], sat_filtered, channels, 'Reflectance', 'FY4A', meth='HG', figlabel=figlabel + '_Zen410')
        # plot_zen_uw(sat_filtered['ghi'], sat_filtered, channels, 'GHI', 'FY4A', meth='HG',
        #             figlabel=figlabel + '_GHI410')
        print('# of sat:', sat.shape[0])
        for i in range(sat.shape[0]):
            print(i)
            Sun_Zen, local_zen, rela_azi = sat['Sun_Zen'][i], sat['Sat_Zen'][i], sat['RAZ'][i]
            COD_goes = 0  # Assuming COD is a column in sat_rad
            df_albedo_row = df_albedo.iloc[i].values
            T_s, RH = sat['T_s'].iloc[i], sat['RH'].iloc[i] # %, K
            try:
                AOD = sat['aod'].iloc[i]
            except Exception:
                AOD = None
                dsw, dni, dhi, uw, F_uw, uw_nosrf = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                df_uw_channels = pd.DataFrame([ [np.nan] * 6 ], columns=channels)
            else:
                dsw, dni, dhi, uw, uw_srf, df_R =  get_rtm_output(Sun_Zen, local_zen, rela_azi,
                                                                  COD_goes, T_s, RH, df_albedo_row, surface, meth, AOD)
                df_uw, F_uw = LUT(uw, COD_goes, Sun_Zen, local_zen, rela_azi)
                df_uw_channels = df_uw.mul(df_R, axis=1) # reflectance factor
            rtm_dsw.append(dsw)
            rtm_dni.append(dni)
            rtm_dhi.append(dhi)
            rtm_uw.append(F_uw)
            rtm_uw_srf.append(uw_srf)
            uw_channels_list.append(df_uw_channels)

            df_new = pd.DataFrame({
                'rtm_dsw': rtm_dsw,
                'rtm_dni': rtm_dni,
                'rtm_dhi': rtm_dhi,
                'rtm_uw':rtm_uw,
                'rtm_uw_srf':rtm_uw_srf,
            })
        sat = sat.reset_index()
        sat = sat.rename(columns={"index": "Time"})
        df_uw_all = pd.concat(uw_channels_list, ignore_index=True)
        df_uw_all = df_uw_all.add_suffix('_rtm')
        df_combined = pd.concat([sat, df_new, df_uw_all], axis=1)
        #df_combined['rtm_dni'] = df_combined['rtm_dni']/np.cos(np.deg2rad(df_combined['Sun_Zen']))
        df_combined = df_combined[df_combined['T_s'] > 283]
        df_combined = df_combined.dropna()
        df_combined.to_csv(csvfile, index=False)
        rtm_DNI, rtm_GHI = df_combined['rtm_dni'], df_combined['rtm_dsw']

    df_combined = pd.read_csv(csvfile)
    df_combined = df_combined.dropna()
    df_combined['Time'] = pd.to_datetime(df_combined['Time'])
    df_combined = df_combined.dropna()
    rtm_DNI, rtm_GHI = df_combined['rtm_dni'], df_combined['rtm_dsw']
    CODfromwho = 'RTM_clear'
    # uw
    df_combined['Sun_Zen'] = pd.to_numeric(df_combined['Sun_Zen'], errors='coerce')
    # 2. Drop any rows where Sun_Zen became NaN (this removes the bad rows entirely across all columns)
    df_combined = df_combined.dropna(subset=['Sun_Zen'])
    VAR = 'Ref'+f'_{site}'
    sat_rad = df_combined[channels]
    rtm_rad = df_combined[rtm_channels]#.mul(np.cos(np.deg2rad(df_combined['Sun_Zen'])), axis=0)
    #rtm_rad.columns = [col.replace('_rtm', '') for col in rtm_rad.columns]
    # Force each column name to a string before replacing
    rtm_rad.columns = [str(col).replace('_rtm', '') for col in rtm_rad.columns]
    #plot_data(sat_rad, rtm_rad, df_combined['Sun_Zen'], channels, VAR, CODfromwho, site, meth, figlabel)
    # The corrected function call
    plot_data(sat_rad, rtm_rad, channels, VAR, CODfromwho, df_combined['Sun_Zen'], surface, meth=meth, figlabel=figlabel)
    # dw
    plot_data_dw_clear(site_GHI, rtm_GHI, CODfromwho, df_combined['Sun_Zen'], site)
    # plot_zen_uw(df_combined['Sun_Zen'], rtm_rad/sat_rad, channels, VAR, CODfromwho, meth='HG', figlabel=figlabel + '_Zen')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FY4A SWRTM near-real-time processing with physical cloudy COD/DW retrieval."
    )
    parser.add_argument(
        "--mode",
        choices=["cloudy-physical", "clear-legacy"],
        default="cloudy-physical",
        help="Run the physical cloudy RTM path or the original clear-sky validation path.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_CLOUDY_DATA_DIR), help="Directory with cloudy FY4A NetCDFs.")
    parser.add_argument("--out-dir", default=str(DEFAULT_PHYSICAL_OUT_DIR), help="Output directory.")
    parser.add_argument("--adm-lut-dir", default=str(DEFAULT_ADM_LUT_DIR), help="FY4A ADM LUT directory.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Channel metrics CSV for weights.")
    parser.add_argument("--site", action="append", help="Optional site code. Repeat for multiple sites.")
    parser.add_argument(
        "--channels",
        default=",".join(PHYSICAL_RETRIEVAL_CHANNELS),
        help="Comma-separated retrieval channels. Default excludes C03 and C04.",
    )
    parser.add_argument("--surface", choices=["MODIS", "BRDF", "Case2"], default="MODIS")
    parser.add_argument("--uw-meth", default="HG", help="Upwelling RTM scattering method for COD retrieval.")
    parser.add_argument("--dw-meth", default="dM", help="Downwelling cloudy RTM scattering method for GHI.")
    parser.add_argument("--aod-default", type=float, default=0.1243)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--max-times", type=int, help="Limit time steps for smoke tests.")
    parser.add_argument(
        "--dw-mode",
        choices=["none", "center"],
        default="center",
        help="Run physical DW for no pixels or the station center pixel.",
    )
    parser.add_argument("--clear-file-dir", default="back-up/", help="Root used by the original clear-sky path.")
    parser.add_argument("--clear-site", action="append", default=["BJC"], help="Site for --mode clear-legacy.")
    return parser.parse_args()


def run_legacy_clear(args: argparse.Namespace) -> None:
    file_dir = args.clear_file_dir
    figlabel = "test"
    cern_path = Path(file_dir) / "FY4A_data" / "CERN_info.csv"
    cerns = pd.read_csv(cern_path, header=0, index_col=False, names=["site", "lon", "lat", "elev"])
    wanted = {site.upper() for site in args.clear_site}
    sites = [row for row in cerns.values.tolist() if row[0].upper() in wanted]

    for site, lat, lon, elev in sites:
        print(site)
        filename = f"{site}_radiance_satellite_clear_sample.csv"
        compare_clear_dsw(
            site,
            f"./FY4A_data/site_sat_data/{filename}",
            meth="HG",
            surface="BRDF",
            sky="clearsky",
            file_dir=file_dir,
            figlabel=figlabel,
        )


def main() -> None:
    args = parse_args()
    if args.mode == "clear-legacy":
        run_legacy_clear(args)
        return

    channels = [channel.strip() for channel in args.channels.split(",") if channel.strip()]
    retrieve_all_physical_cloudy(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        adm_lut_dir=Path(args.adm_lut_dir),
        metrics_path=Path(args.metrics),
        sites=args.site,
        channels=channels,
        surface=args.surface,
        uw_meth=args.uw_meth,
        dw_meth=args.dw_meth,
        aod_default=args.aod_default,
        max_iterations=args.max_iterations,
        epsilon=args.epsilon,
        max_times=args.max_times,
        dw_mode=args.dw_mode,
    )


if __name__ == "__main__":
    main()
