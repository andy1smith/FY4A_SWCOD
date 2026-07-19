"""BJC cloudy UW forward validation from retrieved COD.

Use the BJC center 3x3 retrieved COD from the ADM-based COD product, feed that
COD back through the cloudy upwelling surrogate, apply the FY4A ADM angular
factor, and compare the resulting reflectance factor with FY4A measurements.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import joblib
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from main_SW_scope_nearealtime import (  # noqa: E402
    DEFAULT_ADM_LUT_DIR,
    DEFAULT_MODEL_PATH,
    adm_factor_for_pixels,
    channel_albedo,
    make_interpolator,
)


SITE = "BJC"
CHANNELS = ["C01", "C02", "C03", "C05", "C06"]
DEFAULT_COD_PATH = (
    REPO_ROOT
    / "FY4A_validation"
    / "Cloudy_results"
    / "Cloudy_COD_results"
    / f"{SITE}_cloudy_COD_uw_ADM.nc"
)
DEFAULT_SOURCE_PATH = REPO_ROOT / "FY4A_data" / "Cloudy_site_sat_data" / f"{SITE}_SW_ref_satellite_cloudy.nc"
OUT_DIR = SCRIPT_DIR

FONT = 17
FONT_FAMILY = "Times New Roman"


def setup_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    sns.set_style({"font.family": "serif", "font.serif": [FONT_FAMILY, FONT_FAMILY]})
    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["font.serif"] = [FONT_FAMILY]
    mpl.rcParams["font.size"] = FONT
    mpl.rcParams["axes.titlesize"] = FONT
    mpl.rcParams["axes.labelsize"] = FONT - 1
    mpl.rcParams["xtick.labelsize"] = FONT - 1
    mpl.rcParams["ytick.labelsize"] = FONT - 1
    mpl.rcParams["xtick.major.size"] = 10
    mpl.rcParams["ytick.major.size"] = 10
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = FONT_FAMILY
    mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
    mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"
    mpl.rcParams["axes.linewidth"] = 1.2
    mpl.rcParams["legend.fontsize"] = FONT - 1


def calc_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return {"N": int(valid.sum()), "MBE": np.nan, "RMSE": np.nan, "rMBE": np.nan, "rRMSE": np.nan, "R": np.nan}

    obs_valid = obs[valid]
    pred_valid = pred[valid]
    diff = pred_valid - obs_valid
    mean_obs = float(np.mean(obs_valid))
    return {
        "N": int(valid.sum()),
        "MBE": float(np.mean(diff)),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "rMBE": float(np.mean(diff) / mean_obs * 100.0) if mean_obs else np.nan,
        "rRMSE": float(np.sqrt(np.mean(diff**2)) / mean_obs * 100.0) if mean_obs else np.nan,
        "R": float(np.corrcoef(obs_valid, pred_valid)[0, 1]),
    }


def center_3x3_indices(y_len: int, x_len: int) -> tuple[slice, slice]:
    cy, cx = y_len // 2, x_len // 2
    return slice(cy - 1, cy + 2), slice(cx - 1, cx + 2)


def predict_channel_reflectance(
    ds_source: xr.Dataset,
    ds_cod: xr.Dataset,
    bundle: dict,
    channel: str,
    adm_lut_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return center-3x3 observed, surrogate+ADM, raw UW surrogate, and COD means."""
    cod = np.asarray(ds_cod["Retrieved_COD"].values, dtype=float)
    obs = np.asarray(ds_source[channel].values, dtype=float)
    time_len, y_len, x_len = cod.shape
    y_slc, x_slc = center_3x3_indices(y_len, x_len)

    cod_center = cod[:, y_slc, x_slc]
    obs_center = obs[:, y_slc, x_slc]
    sun_zen_center = np.asarray(ds_source["Sun_Zen"].values, dtype=float)[:, y_slc, x_slc]
    sat_zen_center = np.asarray(ds_source["Sat_Zen"].values, dtype=float)[:, y_slc, x_slc]
    raz_center = np.asarray(ds_source["RAZ"].values, dtype=float)[:, y_slc, x_slc]
    tpw = np.asarray(ds_cod["tpw"].values, dtype=float)
    alb_time = channel_albedo(ds_source, channel, bundle)

    spec = bundle["models"][channel]
    interpolator = make_interpolator(spec)
    raw_pred = np.full_like(cod_center, np.nan, dtype=np.float32)
    adm_pred = np.full_like(cod_center, np.nan, dtype=np.float32)

    for t_idx in range(time_len):
        cod_flat = cod_center[t_idx].reshape(-1)
        obs_flat = obs_center[t_idx].reshape(-1)
        sun_flat = sun_zen_center[t_idx].reshape(-1)
        sat_flat = sat_zen_center[t_idx].reshape(-1)
        raz_flat = raz_center[t_idx].reshape(-1)

        valid = (
            np.isfinite(cod_flat)
            & np.isfinite(obs_flat)
            & np.isfinite(sun_flat)
            & np.isfinite(sat_flat)
            & np.isfinite(raz_flat)
            & np.isfinite(tpw[t_idx])
            & np.isfinite(alb_time[t_idx])
            & (obs_flat > 0.0)
        )
        if not valid.any():
            continue

        alb_key = f"alb_{channel}"
        points = np.column_stack(
            [
                np.log1p(np.clip(cod_flat[valid], 0.0, None)),
                np.full(valid.sum(), tpw[t_idx], dtype=float),
                np.cos(np.deg2rad(sun_flat[valid])),
                np.full(valid.sum(), alb_time[t_idx], dtype=float),
            ]
        )
        for col_idx, feature in enumerate(spec["grid_features"]):
            if feature == "tpw":
                bounds_key = "tpw"
            elif feature == "tpw_grid":
                bounds_key = "tpw"
            elif feature == alb_key:
                bounds_key = alb_key
            else:
                bounds_key = feature
            if bounds_key in spec["bounds"]:
                lo, hi = spec["bounds"][bounds_key]
                points[:, col_idx] = np.clip(points[:, col_idx], lo, hi)

        pred_valid = interpolator(points).astype(np.float32)
        adm_matrix = adm_factor_for_pixels(
            adm_lut_dir,
            channel,
            cod_flat[valid],
            float(np.round(np.nanmedian(sun_flat))),
            sat_flat[valid],
            raz_flat[valid],
        )
        adm_valid = np.diag(adm_matrix).astype(np.float32)

        raw_time = raw_pred[t_idx].reshape(-1)
        adm_time = adm_pred[t_idx].reshape(-1)
        raw_time[valid] = pred_valid
        adm_time[valid] = pred_valid * adm_valid

    return (
        np.nanmean(obs_center, axis=(1, 2)).astype(float),
        np.nanmean(adm_pred, axis=(1, 2)).astype(float),
        np.nanmean(raw_pred, axis=(1, 2)).astype(float),
        np.nanmean(cod_center, axis=(1, 2)).astype(float),
    )


def build_validation_table(
    cod_path: Path = DEFAULT_COD_PATH,
    source_path: Path = DEFAULT_SOURCE_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    adm_lut_dir: Path = DEFAULT_ADM_LUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = joblib.load(model_path)
    bundle["_model_path"] = str(model_path)
    ds_cod = xr.open_dataset(cod_path).load()
    ds_source = xr.open_dataset(source_path).load()

    rows: dict[str, np.ndarray] = {
        "site": np.full(ds_cod.sizes["time"], SITE, dtype=object),
        "time": pd.to_datetime(ds_cod["time"].values),
    }

    for channel in CHANNELS:
        obs, pred_adm, pred_raw, cod_mean = predict_channel_reflectance(
            ds_source,
            ds_cod,
            bundle,
            channel,
            adm_lut_dir,
        )
        rows[f"obs_{channel}"] = obs
        rows[f"surr_adm_{channel}"] = pred_adm
        rows[f"surr_uw_raw_{channel}"] = pred_raw
        rows[f"COD_center3x3_{channel}"] = cod_mean

    df = pd.DataFrame(rows)
    metrics_rows = []
    for channel in CHANNELS:
        stat = calc_metrics(df[f"obs_{channel}"].to_numpy(float), df[f"surr_adm_{channel}"].to_numpy(float))
        stat["Channel"] = channel
        metrics_rows.append(stat)
    metrics = pd.DataFrame(metrics_rows)

    ds_cod.close()
    ds_source.close()
    return df, metrics


def plot_validation(df: pd.DataFrame, metrics: pd.DataFrame, out_path: Path) -> None:
    setup_style()
    ch_positions = {
        "C01": (0, 0),
        "C02": (0, 1),
        "C03": (1, 1),
        "C05": (2, 0),
        "C06": (2, 1),
    }

    fig = plt.figure(figsize=(10.5, 14.2))
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1, 1],
        wspace=0.03,
        hspace=0.34,
        bottom=0.12,
        top=0.90,
        left=0.08,
        right=0.95,
    )

    for channel, (row, col) in ch_positions.items():
        ax = fig.add_subplot(gs[row, col])
        obs = df[f"obs_{channel}"].to_numpy(dtype=float)
        pred = df[f"surr_adm_{channel}"].to_numpy(dtype=float)
        valid = np.isfinite(obs) & np.isfinite(pred)
        stat = metrics[metrics["Channel"] == channel].iloc[0]

        ax.scatter(obs[valid], pred[valid], s=30, alpha=0.8, color="#0072B2", edgecolor="white", linewidth=0.4)
        if valid.any():
            min_v = float(min(obs[valid].min(), pred[valid].min()))
            max_v = float(max(obs[valid].max(), pred[valid].max()))
            margin = 0.05 * (max_v - min_v) if max_v > min_v else 0.01
            lo, hi = min_v - margin, max_v + margin
            ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))
            ax.figure.canvas.draw()
            yticks = ax.get_yticks()
            yticks = yticks[(yticks >= lo) & (yticks <= hi)]
            ax.set_xticks(yticks)
            ax.set_yticks(yticks)

        stats_text = (
            f"N = {int(stat['N'])}\n"
            f"MBE: {stat['MBE']:.4f}\n"
            f"RMSE: {stat['RMSE']:.4f}\n"
            f"rMBE: {stat['rMBE']:.1f}%\n"
            f"rRMSE: {stat['rRMSE']:.1f}%\n"
            f"R = {stat['R']:.3f}"
        )
        ax.text(
            0.04,
            0.96,
            stats_text,
            transform=ax.transAxes,
            fontsize=FONT - 3,
            family=FONT_FAMILY,
            verticalalignment="top",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        ax.set_title(channel, fontsize=FONT, family=FONT_FAMILY)
        if row == 2:
            ax.set_xlabel("FY4A/AGRI Reflectance Factor", fontsize=FONT - 1, family=FONT_FAMILY)
        if col == 0:
            ax.set_ylabel("Surrogate + ADM Reflectance Factor", fontsize=FONT - 1, family=FONT_FAMILY)
        ax.tick_params(labelsize=FONT - 1)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(FONT_FAMILY)
        ax.grid(color="grey", linestyle="--", linewidth=0.5)

    ax_cod = fig.add_subplot(gs[1, 0])
    ax_cod.set_box_aspect(1)
    cod_values = df[[f"COD_center3x3_{channel}" for channel in CHANNELS]].mean(axis=1).dropna().to_numpy()
    bins = np.linspace(float(cod_values.min()), float(cod_values.max()), 18) if len(cod_values) else 10
    ax_cod.hist(cod_values, bins=bins, color="0.35", edgecolor="white", linewidth=0.8, alpha=0.85)
    if len(cod_values):
        median_cod = float(np.median(cod_values))
        ax_cod.axvline(median_cod, color="#D55E00", linestyle="--", linewidth=1.8)
        ax_cod.text(
            0.96,
            0.94,
            f"n = {len(cod_values)}\nMedian = {median_cod:.2f}",
            transform=ax_cod.transAxes,
            fontsize=FONT - 2,
            family=FONT_FAMILY,
            verticalalignment="top",
            horizontalalignment="right",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
    ax_cod.set_xlabel("Retrieved COD", fontsize=FONT - 1, family=FONT_FAMILY)
    ax_cod.set_ylabel("Sample Count", fontsize=FONT - 1, family=FONT_FAMILY)
    ax_cod.set_title("COD Distribution", fontsize=FONT, family=FONT_FAMILY)
    ax_cod.grid(color="grey", linestyle="--", linewidth=0.5, axis="y")
    ax_cod.tick_params(labelsize=FONT - 1)
    for label in ax_cod.get_xticklabels() + ax_cod.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)

    fig.text(
        0.1,
        0.965,
        f"{SITE}: n = {len(df)} center-3x3 cloudy samples",
        fontfamily=FONT_FAMILY,
        fontsize=FONT - 5,
        weight="bold",
        ha="left",
        va="top",
    )
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, metrics = build_validation_table()
    data_path = OUT_DIR / "BJC_center3x3_forward_uw_reflectance.csv"
    metrics_path = OUT_DIR / "BJC_center3x3_forward_uw_reflectance_metrics.csv"
    fig_path = OUT_DIR / "BJC_UW_Reflectance_Center3x3_3x2_COD_distribution.png"
    df.to_csv(data_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_validation(df, metrics, fig_path)
    print(f"Saved: {data_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
