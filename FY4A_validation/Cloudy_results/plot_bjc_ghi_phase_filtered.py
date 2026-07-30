"""Plot BJC cloudy GHI after removing ice-cloud phase samples.

The phase mask comes from Sat_Preprocessing/phasefilter. Phase code meanings:
  -1 invalid, 0 thin/unclassified, 1 ice, 2 water.

For the center-pixel comparison, rows with an ice center pixel are removed.
For the center-3x3 comparison, rows with any ice pixel in the center 3x3
window are removed.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PRED_PATH = SCRIPT_DIR / "Cloudy_dw_HG" / "cloudy_ghi_dw_surrogate_predictions.csv"
PHASE_PATH = REPO_ROOT / "Sat_Preprocessing" / "phasefilter" / "outputs" / "BJC" / "BJC_cloud_phase_code.csv"
OUT_DIR = SCRIPT_DIR / "Cloudy_dw_HG"
SITE = "BJC"
ICE_CODE = 1
GRID_SIZE = 11


def center_pixel_column() -> str:
    center = GRID_SIZE // 2
    return str(center * GRID_SIZE + center)


def center_3x3_columns() -> list[str]:
    center = GRID_SIZE // 2
    return [
        str(row * GRID_SIZE + col)
        for row in range(center - 1, center + 2)
        for col in range(center - 1, center + 2)
    ]


def calc_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() == 0:
        return {"N": 0, "MBE": np.nan, "RMSE": np.nan, "R": np.nan}
    diff = pred[valid] - obs[valid]
    corr = np.corrcoef(obs[valid], pred[valid])[0, 1] if valid.sum() > 1 else np.nan
    return {
        "N": int(valid.sum()),
        "MBE": float(np.mean(diff)),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "R": float(corr),
    }


def add_phase_flags(pred: pd.DataFrame, phase: pd.DataFrame) -> pd.DataFrame:
    center_col = center_pixel_column()
    box_cols = center_3x3_columns()
    missing = [col for col in [center_col, *box_cols] if col not in phase.columns]
    if missing:
        raise ValueError(f"Phase file missing expected pixel columns: {missing}")

    pixel_cols = list(dict.fromkeys([center_col, *box_cols]))
    phase_sub = phase[["time", *pixel_cols]].copy()
    phase_sub["phase_center"] = phase_sub[center_col].astype(int)
    phase_sub["phase_center_is_ice"] = phase_sub["phase_center"].eq(ICE_CODE)
    phase_sub["phase_center3x3_ice_count"] = phase_sub[box_cols].eq(ICE_CODE).sum(axis=1)
    phase_sub["phase_center3x3_has_ice"] = phase_sub["phase_center3x3_ice_count"].gt(0)
    phase_sub["phase_center3x3_water_count"] = phase_sub[box_cols].eq(2).sum(axis=1)
    phase_sub["phase_center3x3_thin_count"] = phase_sub[box_cols].eq(0).sum(axis=1)

    keep_cols = [
        "time",
        "phase_center",
        "phase_center_is_ice",
        "phase_center3x3_ice_count",
        "phase_center3x3_has_ice",
        "phase_center3x3_water_count",
        "phase_center3x3_thin_count",
    ]
    return pred.merge(phase_sub[keep_cols], on="time", how="left", validate="one_to_one")


def plot_panel(ax, df: pd.DataFrame, pred_col: str, title: str) -> dict[str, float]:
    stat = calc_metrics(df["GHI_ground"].to_numpy(float), df[pred_col].to_numpy(float))
    obs = df["GHI_ground"].to_numpy(float)
    pred = df[pred_col].to_numpy(float)
    valid = np.isfinite(obs) & np.isfinite(pred)
    ax.scatter(obs[valid], pred[valid], s=18, alpha=0.45, color="#2F6F9F", edgecolors="none")
    if valid.any():
        lim_max = float(np.nanmax([obs[valid].max(), pred[valid].max(), 1000.0]))
    else:
        lim_max = 1000.0
    ax.plot([0, lim_max], [0, lim_max], color="#B23A48", linestyle="--", lw=1.2)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("Measured GHI [W m$^{-2}$]")
    ax.set_ylabel("Predicted GHI [W m$^{-2}$]")
    ax.set_title(title)
    ax.text(
        0.04,
        0.96,
        f"N={stat['N']}\nRMSE={stat['RMSE']:.1f}\nMBE={stat['MBE']:.1f}\nR={stat['R']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )
    ax.grid(True, linestyle=":", alpha=0.45)
    return stat


def main() -> None:
    pred = pd.read_csv(PRED_PATH, parse_dates=["time"])
    pred = pred[pred["site"].eq(SITE)].copy()
    phase = pd.read_csv(PHASE_PATH, parse_dates=["time"])
    merged = add_phase_flags(pred, phase)

    center_filtered = merged[~merged["phase_center_is_ice"].fillna(True)].copy()
    box_filtered = merged[~merged["phase_center3x3_has_ice"].fillna(True)].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_DIR / "BJC_cloudy_ghi_dw_phase_flags.csv", index=False)
    center_filtered.to_csv(OUT_DIR / "BJC_cloudy_ghi_dw_no_ice_center.csv", index=False)
    box_filtered.to_csv(OUT_DIR / "BJC_cloudy_ghi_dw_no_ice_center3x3.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), constrained_layout=True)
    center_stat = plot_panel(
        axes[0],
        center_filtered,
        "GHI_center",
        "BJC GHI, Center Pixel\nIce-phase center removed",
    )
    box_stat = plot_panel(
        axes[1],
        box_filtered,
        "GHI_center_3x3_mean",
        "BJC GHI, Center 3x3 Mean\nAny ice-phase pixel removed",
    )
    fig_path = OUT_DIR / "BJC_GHI_phase_filtered_no_ice.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = pd.DataFrame(
        [
            {**center_stat, "prediction": "GHI_center", "filter": "center_phase_not_ice"},
            {**box_stat, "prediction": "GHI_center_3x3_mean", "filter": "center3x3_no_ice"},
        ]
    )
    metrics.to_csv(OUT_DIR / "BJC_cloudy_ghi_dw_phase_filtered_metrics.csv", index=False)

    print(f"Input BJC rows: {len(pred)}")
    print(f"Center no-ice rows: {len(center_filtered)}")
    print(f"Center 3x3 no-ice rows: {len(box_filtered)}")
    print(f"Saved: {fig_path}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
