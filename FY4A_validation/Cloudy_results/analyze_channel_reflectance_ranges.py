"""Compare cloudy-channel reflectance ranges for FY4A and GOES.

FY4A input: site NetCDF files with C01-C06 on (time, y, x).
GOES input: water-cloud CSV files with C01-C06 columns.

Outputs are written under Cloudy_results/channel_reflectance_range_analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from Sat_Preprocessing.phasefilter.fy4a_cloud_phase_filter import (  # noqa: E402
    PHASE_ICE,
    classify_phase,
)

FY4A_DIR = REPO_ROOT / "FY4A_data" / "Cloudy_site_sat_data"
GOES_DIR = Path("/Users/dengnan/Documents/git_store/Shortwave_MCRTM/GOES_data/GOES16_site_sat_data")
OUT_DIR = REPO_ROOT / "FY4A_validation" / "Cloudy_results" / "channel_reflectance_range_analysis"
NO_ICE_OUT_DIR = REPO_ROOT / "FY4A_validation" / "Cloudy_results" / "channel_reflectance_range_analysis_no_ice"
CHANNELS = ["C01", "C02", "C03", "C04", "C05", "C06"]
PERCENTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]
OUT_DIR_ACTIVE = OUT_DIR
FY4A_LABEL_ACTIVE = "FY4A"
REFLECTANCE_LABEL_ACTIVE = "Reflectance"
PHASE_SUMMARY_ACTIVE: list[dict[str, float | int | str]] = []


def site_from_fy4a(path: Path) -> str:
    return path.name.split("_SW_ref_satellite_cloudy.nc")[0]


def site_from_goes(path: Path) -> str:
    return path.name.replace("GOES_day_", "").replace("_radiance_water_2019_MarOct.csv", "")


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def summarize(values: np.ndarray) -> dict[str, float]:
    values = finite_values(values)
    row: dict[str, float] = {"n": int(values.size)}
    if values.size == 0:
        for key in ["mean", "std", "min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"]:
            row[key] = np.nan
        row.update(
            {
                "iqr": np.nan,
                "tukey_low": np.nan,
                "tukey_high": np.nan,
                "outlier_n": 0,
                "outlier_pct": np.nan,
                "negative_n": 0,
                "gt1_n": 0,
                "negative_pct": np.nan,
                "gt1_pct": np.nan,
            }
        )
        return row

    pct = np.nanpercentile(values, PERCENTILES)
    pmap = dict(zip(PERCENTILES, pct))
    q1 = float(pmap[25])
    q3 = float(pmap[75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outlier = (values < low) | (values > high)
    negative = values < 0
    gt1 = values > 1
    row.update(
        {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(pmap[0]),
            "p01": float(pmap[1]),
            "p05": float(pmap[5]),
            "p25": q1,
            "median": float(pmap[50]),
            "p75": q3,
            "p95": float(pmap[95]),
            "p99": float(pmap[99]),
            "max": float(pmap[100]),
            "iqr": float(iqr),
            "tukey_low": float(low),
            "tukey_high": float(high),
            "outlier_n": int(outlier.sum()),
            "outlier_pct": float(outlier.mean() * 100.0),
            "negative_n": int(negative.sum()),
            "gt1_n": int(gt1.sum()),
            "negative_pct": float(negative.mean() * 100.0),
            "gt1_pct": float(gt1.mean() * 100.0),
        }
    )
    return row


def phase_arrays_from_dataset(ds: xr.Dataset) -> tuple[pd.DatetimeIndex, list[str], dict[str, np.ndarray]]:
    time_len = ds.sizes["time"]
    n_pixels = ds.sizes["y"] * ds.sizes["x"]
    pixel_columns = [str(idx) for idx in range(n_pixels)]
    arrays = {
        "C01": np.asarray(ds["C01"].values, dtype=float).reshape(time_len, n_pixels),
        "C05": np.asarray(ds["C05"].values, dtype=float).reshape(time_len, n_pixels),
        "C06": np.asarray(ds["C06"].values, dtype=float).reshape(time_len, n_pixels),
        "SunZenith": np.asarray(ds["Sun_Zen"].values, dtype=float).reshape(time_len, n_pixels),
    }
    return pd.DatetimeIndex(pd.to_datetime(ds["time"].values)), pixel_columns, arrays


def no_ice_mask(ds: xr.Dataset, site: str) -> np.ndarray:
    times, pixel_columns, arrays = phase_arrays_from_dataset(ds)
    products = classify_phase(arrays, times, pixel_columns)
    phase_code = products.phase_code.reshape(ds.sizes["time"], ds.sizes["y"], ds.sizes["x"])
    ice_mask = phase_code == PHASE_ICE
    valid_phase = phase_code >= 0
    PHASE_SUMMARY_ACTIVE.append(
        {
            "site": site,
            "n_pixels": int(phase_code.size),
            "n_phase_valid": int(valid_phase.sum()),
            "n_ice": int(ice_mask.sum()),
            "ice_pct_all_pixels": float(ice_mask.mean() * 100.0),
            "ice_pct_valid_phase_pixels": float(ice_mask.sum() / valid_phase.sum() * 100.0)
            if valid_phase.any()
            else np.nan,
        }
    )
    return ~ice_mask


def collect_fy4a(remove_ice: bool = False) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows = []
    values: dict[tuple[str, str], list[np.ndarray]] = {}
    for path in sorted(FY4A_DIR.glob("*_SW_ref_satellite_cloudy.nc")):
        site = site_from_fy4a(path)
        with xr.open_dataset(path) as ds:
            keep_mask = no_ice_mask(ds, site) if remove_ice else None
            for channel in CHANNELS:
                if channel not in ds:
                    continue
                raw = np.asarray(ds[channel].values, dtype=float)
                if keep_mask is not None:
                    raw = np.where(keep_mask, raw, np.nan)
                arr = finite_values(raw)
                rec = {
                    "satellite": "FY4A",
                    "site": site,
                    "channel": channel,
                    **summarize(arr),
                }
                rows.append(rec)
                values.setdefault(("FY4A", channel), []).append(arr)
    combined = {key: np.concatenate(parts) for key, parts in values.items() if parts}
    return pd.DataFrame(rows), combined


def collect_goes() -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows = []
    values: dict[tuple[str, str], list[np.ndarray]] = {}
    for path in sorted(GOES_DIR.glob("GOES_day_*_radiance_water_2019_MarOct.csv")):
        site = site_from_goes(path)
        df = pd.read_csv(path, usecols=CHANNELS)
        for channel in CHANNELS:
            arr = finite_values(df[channel].to_numpy(dtype=float))
            rec = {
                "satellite": "GOES",
                "site": site,
                "channel": channel,
                **summarize(arr),
            }
            rows.append(rec)
            values.setdefault(("GOES", channel), []).append(arr)
    combined = {key: np.concatenate(parts) for key, parts in values.items() if parts}
    return pd.DataFrame(rows), combined


def aggregate_metrics(values: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    rows = []
    for (satellite, channel), arr in sorted(values.items()):
        rows.append({"satellite": satellite, "site": "ALL", "channel": channel, **summarize(arr)})
    return pd.DataFrame(rows)


def comparison_metrics(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel in CHANNELS:
        fy = agg[(agg["satellite"] == "FY4A") & (agg["channel"] == channel)].iloc[0]
        go = agg[(agg["satellite"] == "GOES") & (agg["channel"] == channel)].iloc[0]
        rows.append(
            {
                "channel": channel,
                "FY4A_n": int(fy["n"]),
                "GOES_n": int(go["n"]),
                "FY4A_median": fy["median"],
                "GOES_median": go["median"],
                "median_diff_FY4A_minus_GOES": fy["median"] - go["median"],
                "median_ratio_FY4A_over_GOES": fy["median"] / go["median"] if go["median"] != 0 else np.nan,
                "FY4A_mean": fy["mean"],
                "GOES_mean": go["mean"],
                "mean_diff_FY4A_minus_GOES": fy["mean"] - go["mean"],
                "FY4A_p05": fy["p05"],
                "GOES_p05": go["p05"],
                "FY4A_p95": fy["p95"],
                "GOES_p95": go["p95"],
                "larger_by_median": "FY4A" if fy["median"] > go["median"] else "GOES",
                "larger_by_mean": "FY4A" if fy["mean"] > go["mean"] else "GOES",
            }
        )
    return pd.DataFrame(rows)


def common_bins(a: np.ndarray, b: np.ndarray, n_bins: int = 80) -> np.ndarray:
    both = np.concatenate([a, b])
    lo, hi = np.nanpercentile(both, [0.2, 99.8])
    lo = min(0.0, float(lo))
    hi = max(float(hi), 0.01)
    return np.linspace(lo, hi, n_bins + 1)


def plot_histograms(values: dict[tuple[str, str], np.ndarray], comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), constrained_layout=True)
    for ax, channel in zip(axes.ravel(), CHANNELS):
        fy = values[("FY4A", channel)]
        go = values[("GOES", channel)]
        bins = common_bins(fy, go)
        ax.hist(fy, bins=bins, density=True, alpha=0.48, color="#2f6f9f", label=FY4A_LABEL_ACTIVE)
        ax.hist(go, bins=bins, density=True, alpha=0.48, color="#c45b3d", label="GOES")
        fy_med = float(comparison.loc[comparison["channel"] == channel, "FY4A_median"].iloc[0])
        go_med = float(comparison.loc[comparison["channel"] == channel, "GOES_median"].iloc[0])
        ax.axvline(fy_med, color="#1d4f73", lw=1.5)
        ax.axvline(go_med, color="#923d28", lw=1.5)
        ax.set_title(f"{channel} density")
        ax.set_xlabel(REFLECTANCE_LABEL_ACTIVE)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.text(
            0.98,
            0.96,
            f"median\n{FY4A_LABEL_ACTIVE} {fy_med:.3f}\nGOES {go_med:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8", "alpha": 0.85},
        )
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Cloudy {REFLECTANCE_LABEL_ACTIVE} Distributions: {FY4A_LABEL_ACTIVE} vs GOES", fontsize=14, fontweight="bold")
    fig.savefig(OUT_DIR_ACTIVE / "fy4a_goes_channel_reflectance_histograms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_site_ranges(site_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2), constrained_layout=True, sharey=False)
    for ax, channel in zip(axes.ravel(), CHANNELS):
        sub = site_metrics[site_metrics["channel"] == channel].copy()
        sub = sub.sort_values(["satellite", "median"]).reset_index(drop=True)
        y = np.arange(len(sub))
        colors = np.where(sub["satellite"].eq("FY4A"), "#2f6f9f", "#c45b3d")
        ax.hlines(y, sub["p05"], sub["p95"], color=colors, alpha=0.55, lw=2.0)
        ax.scatter(sub["median"], y, c=colors, s=18, zorder=3)
        labels = sub["satellite"].replace({"FY4A": FY4A_LABEL_ACTIVE}).str.cat(sub["site"], sep=":")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"{channel}: site p05-p95 and median")
        ax.set_xlabel(REFLECTANCE_LABEL_ACTIVE)
        ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    fig.suptitle(f"Per-Site Cloudy {REFLECTANCE_LABEL_ACTIVE} Ranges", fontsize=14, fontweight="bold")
    fig.savefig(OUT_DIR_ACTIVE / "fy4a_goes_site_channel_ranges.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_box_summary(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    x = np.arange(len(CHANNELS))
    width = 0.34
    for offset, satellite, color in [(-width / 2, "FY4A", "#2f6f9f"), (width / 2, "GOES", "#c45b3d")]:
        sub = agg[agg["satellite"] == satellite].set_index("channel").loc[CHANNELS]
        label = FY4A_LABEL_ACTIVE if satellite == "FY4A" else satellite
        ax.bar(x + offset, sub["median"], width=width, color=color, alpha=0.75, label=f"{label} median")
        ax.errorbar(
            x + offset,
            sub["median"],
            yerr=np.vstack([sub["median"] - sub["p25"], sub["p75"] - sub["median"]]),
            fmt="none",
            ecolor="0.25",
            elinewidth=1.2,
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel(REFLECTANCE_LABEL_ACTIVE)
    ax.set_title("Aggregate Median and IQR by Channel")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR_ACTIVE / "fy4a_goes_channel_median_iqr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_outlier_rates(site_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.8), constrained_layout=True, sharey=False)
    for ax, satellite in zip(axes, ["FY4A", "GOES"]):
        piv = site_metrics[site_metrics["satellite"] == satellite].pivot(index="site", columns="channel", values="outlier_pct")
        piv = piv.loc[sorted(piv.index), CHANNELS]
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="magma", vmin=0)
        label = FY4A_LABEL_ACTIVE if satellite == "FY4A" else satellite
        ax.set_title(f"{label} Tukey outlier rate [%]")
        ax.set_xticks(np.arange(len(CHANNELS)))
        ax.set_xticklabels(CHANNELS)
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=5.5, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(OUT_DIR_ACTIVE / "fy4a_goes_site_outlier_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FY4A and GOES cloudy channel reflectance ranges.")
    parser.add_argument(
        "--remove-fy4a-ice",
        action="store_true",
        help="Classify FY4A cloud phase from C01/C05/C06 and remove pixels classified as ice before comparison.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    global OUT_DIR_ACTIVE, FY4A_LABEL_ACTIVE, REFLECTANCE_LABEL_ACTIVE

    args = parse_args()
    if args.out_dir:
        OUT_DIR_ACTIVE = Path(args.out_dir)
    elif args.remove_fy4a_ice:
        OUT_DIR_ACTIVE = NO_ICE_OUT_DIR
    else:
        OUT_DIR_ACTIVE = OUT_DIR

    if args.remove_fy4a_ice:
        FY4A_LABEL_ACTIVE = "FY4A no ice"
        REFLECTANCE_LABEL_ACTIVE = "Reflectance / FY4A ice pixels removed"

    OUT_DIR_ACTIVE.mkdir(parents=True, exist_ok=True)
    fy_site, fy_values = collect_fy4a(remove_ice=args.remove_fy4a_ice)
    go_site, go_values = collect_goes()
    site_metrics = pd.concat([fy_site, go_site], ignore_index=True)
    values = {**fy_values, **go_values}
    agg = aggregate_metrics(values)
    comparison = comparison_metrics(agg)

    site_metrics.to_csv(OUT_DIR_ACTIVE / "site_channel_reflectance_metrics.csv", index=False)
    agg.to_csv(OUT_DIR_ACTIVE / "aggregate_channel_reflectance_metrics.csv", index=False)
    comparison.to_csv(OUT_DIR_ACTIVE / "fy4a_goes_channel_comparison.csv", index=False)
    if args.remove_fy4a_ice:
        pd.DataFrame(PHASE_SUMMARY_ACTIVE).to_csv(OUT_DIR_ACTIVE / "fy4a_no_ice_phase_filter_summary.csv", index=False)

    plot_histograms(values, comparison)
    plot_site_ranges(site_metrics)
    plot_box_summary(agg)
    plot_outlier_rates(site_metrics)

    print(f"FY4A site files: {fy_site['site'].nunique()}")
    print(f"GOES site files: {go_site['site'].nunique()}")
    print(f"FY4A ice pixels removed: {args.remove_fy4a_ice}")
    print(f"Saved outputs to: {OUT_DIR_ACTIVE}")
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
