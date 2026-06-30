"""
Diagnose questionable GHI validation sites for the dM surrogate workflow.

The key physical check is the clear-sky index:

    kt_clear = measured GHI / pvlib Ineichen clear-sky GHI

Rows written to Ground/preprocessed_GHI/*_clear.h5 should not be persistently
below the Quesada-Ruiz cloudy threshold (kt_clear < 0.8). Sites that are both
low-kt and high-residual in the surrogate outputs are poor candidates for a
clear-sky GHI validation set.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[2]
OUT_DIR = BASE_DIR / "diagnostics"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fy4a_swc_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUND_DIR = ROOT_DIR / "Sat_Preprocessing" / "Ground" / "preprocessed_GHI"
SITE_SAT_DIR = ROOT_DIR / "FY4A_data" / "site_sat_data"

TARGET_SITES = ("YGA", "BJF")


def calc_metrics(obs: pd.Series, pred: pd.Series) -> dict[str, float]:
    obs_arr = pd.to_numeric(obs, errors="coerce").to_numpy(dtype=float)
    pred_arr = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    obs_arr = obs_arr[mask]
    pred_arr = pred_arr[mask]
    if obs_arr.size < 2:
        return {
            "n": int(obs_arr.size),
            "mean_ghi": np.nan,
            "mean_pred": np.nan,
            "mbe": np.nan,
            "rmse": np.nan,
            "rmbe": np.nan,
            "rrmse": np.nan,
            "r": np.nan,
        }

    resid = pred_arr - obs_arr
    mean_obs = float(np.mean(obs_arr))
    mbe = float(np.mean(resid))
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {
        "n": int(obs_arr.size),
        "mean_ghi": mean_obs,
        "mean_pred": float(np.mean(pred_arr)),
        "mbe": mbe,
        "rmse": rmse,
        "rmbe": mbe / mean_obs * 100.0 if mean_obs else np.nan,
        "rrmse": rmse / mean_obs * 100.0 if mean_obs else np.nan,
        "r": float(np.corrcoef(obs_arr, pred_arr)[0, 1]),
    }


def load_ground_clear_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for clear_path in sorted(GROUND_DIR.glob("*_clear.h5")):
        site = clear_path.name.replace("_clear.h5", "")
        df = pd.read_hdf(clear_path, key="df")

        whole_path = GROUND_DIR / f"{site}_consistent_clear_days.h5"
        try:
            whole_n = int(len(pd.read_hdf(whole_path, key="df")))
        except (FileNotFoundError, KeyError, OSError):
            whole_n = np.nan

        if df.empty:
            rows.append(
                {
                    "site": site,
                    "clear_h5_n": 0,
                    "maroct_clear_h5_n": 0,
                    "mean_kt_clear": np.nan,
                    "median_kt_clear": np.nan,
                    "p25_kt_clear": np.nan,
                    "p75_kt_clear": np.nan,
                    "max_kt_clear": np.nan,
                    "pct_clear_rows_kt_lt_0p8": np.nan,
                    "consistent_clear_day_rows": whole_n,
                }
            )
            continue

        df["Time"] = pd.to_datetime(df["Time"])
        df["Month"] = df["Time"].dt.month
        kt = pd.to_numeric(df["ghi"], errors="coerce") / pd.to_numeric(df["ghi_clear"], errors="coerce")
        rows.append(
            {
                "site": site,
                "clear_h5_n": int(len(df)),
                "maroct_clear_h5_n": int(df["Month"].between(3, 10).sum()),
                "mean_kt_clear": float(kt.mean()),
                "median_kt_clear": float(kt.median()),
                "p25_kt_clear": float(kt.quantile(0.25)),
                "p75_kt_clear": float(kt.quantile(0.75)),
                "max_kt_clear": float(kt.max()),
                "pct_clear_rows_kt_lt_0p8": float((kt < 0.8).mean() * 100.0),
                "consistent_clear_day_rows": whole_n,
            }
        )

    return pd.DataFrame(rows)


def load_filter_counts() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for site_path in sorted(SITE_SAT_DIR.glob("*_radiance_satellite_clear.csv")):
        site = site_path.name.split("_")[0]
        try:
            df = pd.read_csv(site_path, usecols=["Time", "ghi", "ghi_clear", "Sun_Zen", "C01", "C06"])
        except ValueError:
            continue

        df["Time"] = pd.to_datetime(df["Time"])
        filtered = df[(df["Sun_Zen"] <= 65) & (df["C01"] < 0.19) & (df["C06"] > 0.05)].copy()
        kt_filtered = filtered["ghi"] / filtered["ghi_clear"] if len(filtered) else pd.Series(dtype=float)
        rows.append(
            {
                "site": site,
                "site_sat_raw_n": int(len(df)),
                "after_surrogate_filters_n": int(len(filtered)),
                "maroct_after_surrogate_filters_n": int(filtered["Time"].dt.month.between(3, 10).sum()),
                "mean_kt_after_surrogate_filters": float(kt_filtered.mean()) if len(filtered) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def load_prediction_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for folder in ("withoutAOD", "withAOD"):
        for pred_path in sorted((BASE_DIR / folder).glob("gpr_predicted_dM_*.csv")):
            site = pred_path.name.replace("gpr_predicted_dM_", "").replace(".csv", "")
            df = pd.read_csv(pred_path)
            df = df[df["Month"].between(3, 10)].copy()
            if df.empty:
                continue

            raw_path = SITE_SAT_DIR / f"{site}_radiance_satellite_clear.csv"
            if raw_path.exists():
                raw = pd.read_csv(raw_path, usecols=["Time", "ghi_clear"])
                raw["Time"] = pd.to_datetime(raw["Time"])
                df["Time"] = pd.to_datetime(df["Time"])
                df = df.merge(raw, on="Time", how="left")

            metrics = calc_metrics(df["ghi"], df["gpr_ghi"])
            kt = df["ghi"] / df["ghi_clear"] if "ghi_clear" in df else pd.Series(dtype=float)
            rows.append(
                {
                    "site": site,
                    "aod_case": folder,
                    "maroct_prediction_n": metrics.pop("n"),
                    "maroct_prediction_mean_kt": float(kt.mean()) if len(kt) else np.nan,
                    "maroct_prediction_median_kt": float(kt.median()) if len(kt) else np.nan,
                    "maroct_prediction_pct_kt_lt_0p8": float((kt < 0.8).mean() * 100.0) if len(kt) else np.nan,
                    **{f"maroct_prediction_{key}": value for key, value in metrics.items()},
                }
            )

    if not rows:
        return pd.DataFrame(columns=["site"])

    wide_rows = []
    for site, site_df in pd.DataFrame(rows).groupby("site", sort=True):
        merged: dict[str, object] = {"site": site}
        for _, row in site_df.iterrows():
            case = row["aod_case"]
            for key, value in row.items():
                if key in {"site", "aod_case"}:
                    continue
                merged[f"{key}_{case}"] = value
        wide_rows.append(merged)

    return pd.DataFrame(wide_rows)


def load_target_rows() -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for site in TARGET_SITES:
        raw = pd.read_csv(SITE_SAT_DIR / f"{site}_radiance_satellite_clear.csv", usecols=["Time", "ghi_clear"])
        raw["Time"] = pd.to_datetime(raw["Time"])
        for folder in ("withoutAOD", "withAOD"):
            path = BASE_DIR / folder / f"gpr_predicted_dM_{site}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.merge(raw, on="Time", how="left")
            df["aod_case"] = folder
            df["kt_clear"] = df["ghi"] / df["ghi_clear"]
            df["gpr_minus_ghi"] = df["gpr_ghi"] - df["ghi"]
            df["gpr_over_ghi"] = df["gpr_ghi"] / df["ghi"]
            records.append(
                df[
                    [
                        "aod_case",
                        "Site",
                        "Time",
                        "Month",
                        "Sun_Zen",
                        "ghi",
                        "ghi_clear",
                        "kt_clear",
                        "gpr_ghi",
                        "gpr_minus_ghi",
                        "gpr_over_ghi",
                        "C01",
                        "C06",
                        "aod",
                    ]
                ]
            )

    return pd.concat(records, ignore_index=True).sort_values(["Site", "aod_case", "Time"])


def add_exclusion_flags(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["exclude_from_clear_validation"] = (
        (summary["pct_clear_rows_kt_lt_0p8"] == 100.0)
        & (summary["mean_kt_clear"] < 0.75)
        & (
            (summary.get("maroct_prediction_rrmse_withoutAOD", pd.Series(np.nan, index=summary.index)) > 20.0)
            | (summary.get("maroct_prediction_rrmse_withAOD", pd.Series(np.nan, index=summary.index)) > 20.0)
        )
    )
    return summary


def save_plots(summary: pd.DataFrame, target_rows: pd.DataFrame) -> None:
    rank = summary.dropna(subset=["mean_kt_clear"]).sort_values("mean_kt_clear")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#d95f02" if site in TARGET_SITES else "#4c78a8" for site in rank["site"]]
    ax.bar(rank["site"], rank["mean_kt_clear"], color=colors, width=0.8)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=1.2, label="kt = 0.8 cloudy threshold")
    ax.set_ylabel("Mean measured GHI / clear-sky GHI")
    ax.set_xlabel("Site")
    ax.set_title("Clear-sky index ranking for Ground/preprocessed_GHI/*_clear.h5")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "GHI_Clear_Index_Rank_All_Sites.png", dpi=250)
    plt.close(fig)

    subset = target_rows[target_rows["aod_case"] == "withoutAOD"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=False, sharey=False)
    for ax, site in zip(axes, TARGET_SITES):
        sdf = subset[subset["Site"] == site]
        ax.scatter(sdf["ghi_clear"], sdf["ghi"], color="#d95f02", s=34, edgecolor="black", linewidth=0.4)
        lim = max(float(sdf["ghi_clear"].max()), float(sdf["ghi"].max())) * 1.05
        ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1.0, label="1:1")
        ax.plot([0, lim], [0, 0.8 * lim], color="#666666", linestyle=":", linewidth=1.0, label="kt = 0.8")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(site)
        ax.set_xlabel("pvlib clear-sky GHI [W m-2]")
        ax.grid(True, linewidth=0.4, alpha=0.35)
    axes[0].set_ylabel("Measured GHI [W m-2]")
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "GHI_YGA_BJF_Clear_Sky_Index.png", dpi=250)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ground = load_ground_clear_summary()
    filters = load_filter_counts()
    pred = load_prediction_metrics()

    summary = ground.merge(filters, on="site", how="left").merge(pred, on="site", how="left")
    summary = add_exclusion_flags(summary)
    summary = summary.sort_values(["exclude_from_clear_validation", "mean_kt_clear"], ascending=[False, True])

    target_rows = load_target_rows()

    summary.to_csv(OUT_DIR / "ghi_site_quality_summary.csv", index=False)
    target_rows.to_csv(OUT_DIR / "ghi_yga_bjf_diagnostic_rows.csv", index=False)
    save_plots(summary, target_rows)

    cols = [
        "site",
        "clear_h5_n",
        "mean_kt_clear",
        "max_kt_clear",
        "pct_clear_rows_kt_lt_0p8",
        "consistent_clear_day_rows",
        "maroct_after_surrogate_filters_n",
        "maroct_prediction_rrmse_withoutAOD",
        "maroct_prediction_rrmse_withAOD",
        "exclude_from_clear_validation",
    ]
    print(summary.loc[summary["site"].isin(TARGET_SITES), cols].to_string(index=False))
    print(f"\nWrote diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
