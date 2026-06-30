from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


YEARS = ("2023", "2024")
PLOT_FONT_SIZE = 13
PLOT_FONT_FAMILY = "Times New Roman"


def apply_plot_style() -> None:
    '''
    Use Times New Roman if available; 
    otherwise use another Times-like serif; 
    otherwise use a standard serif fallback.
    '''
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE,
            "font.family": "serif",
            "font.serif": [PLOT_FONT_FAMILY, "Times", "DejaVu Serif"],
        }
    )

@dataclass
class YearSiteStats:
    year: str
    site: str
    lat: float
    lon: float
    height: float
    count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    std: float
    p10: float
    p25: float
    p75: float
    p90: float


def tukey_filter(values: pd.Series) -> tuple[pd.Series, float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return clean, np.nan, np.nan
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = clean[(clean >= lower) & (clean <= upper)]
    return filtered, float(lower), float(upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize yearly and multi-year fixed AOD features for CARSNET sites."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "CARSNET_data",
        help="Directory containing yearly CARSNET site CSV folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "CARSNET_data" / "annual_site_summary",
        help="Directory for summary CSV and figures.",
    )
    return parser.parse_args()


def compute_stats(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "minimum": np.nan,
            "maximum": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
        }
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "minimum": float(clean.min()),
        "maximum": float(clean.max()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
    }


def load_year_site_frames(base_dir: Path, years: Iterable[str]) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    yearly_frames: dict[str, dict[str, pd.DataFrame]] = {}
    long_frames: list[pd.DataFrame] = []
    for year in years:
        year_dir = base_dir / year
        site_frames: dict[str, pd.DataFrame] = {}
        for csv_path in sorted(year_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            required = {"Timestamp", "AOD_550nm", "site_name", "lat", "lon", "height"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
            df = df.loc[:, ["Timestamp", "AOD_550nm", "site_name", "lat", "lon", "height"]].copy()
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df["AOD_550nm"] = pd.to_numeric(df["AOD_550nm"], errors="coerce")
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df["height"] = pd.to_numeric(df["height"], errors="coerce")
            df["year"] = year
            df["site"] = csv_path.stem
            site_frames[csv_path.stem] = df
            long_frames.append(df)
        yearly_frames[year] = site_frames
    long_df = pd.concat(long_frames, ignore_index=True)
    return yearly_frames, long_df


def summarize_sites(yearly_frames: dict[str, dict[str, pd.DataFrame]], long_df: pd.DataFrame) -> pd.DataFrame:
    sites = sorted(set(long_df["site"].dropna().unique()))
    records: list[dict[str, object]] = []

    for site in sites:
        per_year_stats: dict[str, dict[str, float]] = {}
        per_year_filtered_stats: dict[str, dict[str, float]] = {}
        per_year_filter_bounds: dict[str, tuple[float, float]] = {}
        metadata_frames: list[pd.DataFrame] = []

        for year in YEARS:
            df = yearly_frames.get(year, {}).get(site)
            if df is None:
                per_year_stats[year] = compute_stats(pd.Series(dtype=float))
                per_year_filtered_stats[year] = compute_stats(pd.Series(dtype=float))
                per_year_filter_bounds[year] = (np.nan, np.nan)
                continue
            metadata_frames.append(df.loc[:, ["site_name", "lat", "lon", "height"]])
            stats = compute_stats(df["AOD_550nm"])
            per_year_stats[year] = stats
            filtered_values, lower, upper = tukey_filter(df["AOD_550nm"])
            per_year_filtered_stats[year] = compute_stats(filtered_values)
            per_year_filter_bounds[year] = (lower, upper)

        site_df = long_df.loc[long_df["site"] == site].copy()
        metadata_frames.append(site_df.loc[:, ["site_name", "lat", "lon", "height"]])
        metadata = pd.concat(metadata_frames, ignore_index=True).dropna()

        combined_stats = compute_stats(site_df["AOD_550nm"])
        filtered_site_values, pooled_lower, pooled_upper = tukey_filter(site_df["AOD_550nm"])
        filtered_combined_stats = compute_stats(filtered_site_values)
        lat = float(metadata["lat"].median()) if not metadata.empty else np.nan
        lon = float(metadata["lon"].median()) if not metadata.empty else np.nan
        height = float(metadata["height"].median()) if not metadata.empty else np.nan
        site_name = str(metadata["site_name"].mode().iat[0]) if not metadata.empty else site

        removed_count = combined_stats["count"] - filtered_combined_stats["count"]
        removed_fraction = (
            float(removed_count / combined_stats["count"]) if combined_stats["count"] else np.nan
        )
        suggested_method = "pooled_median_after_tukey_filter"

        record: dict[str, object] = {
            "site": site_name,
            "site_file": site,
            "lon": lon,
            "lat": lat,
            "height_m": height,
            "n_2023": per_year_stats["2023"]["count"],
            "n_2024": per_year_stats["2024"]["count"],
            "AOD_mean_2023": per_year_stats["2023"]["mean"],
            "AOD_mean_2024": per_year_stats["2024"]["mean"],
            "AOD_median_2023": per_year_stats["2023"]["median"],
            "AOD_median_2024": per_year_stats["2024"]["median"],
            "AOD_std_2023": per_year_stats["2023"]["std"],
            "AOD_std_2024": per_year_stats["2024"]["std"],
            "AOD_min_2023": per_year_stats["2023"]["minimum"],
            "AOD_min_2024": per_year_stats["2024"]["minimum"],
            "AOD_max_2023": per_year_stats["2023"]["maximum"],
            "AOD_max_2024": per_year_stats["2024"]["maximum"],
            "AOD_p10_2023": per_year_stats["2023"]["p10"],
            "AOD_p10_2024": per_year_stats["2024"]["p10"],
            "AOD_p25_2023": per_year_stats["2023"]["p25"],
            "AOD_p25_2024": per_year_stats["2024"]["p25"],
            "AOD_p75_2023": per_year_stats["2023"]["p75"],
            "AOD_p75_2024": per_year_stats["2024"]["p75"],
            "AOD_p90_2023": per_year_stats["2023"]["p90"],
            "AOD_p90_2024": per_year_stats["2024"]["p90"],
            "AOD_filtered_mean_2023": per_year_filtered_stats["2023"]["mean"],
            "AOD_filtered_mean_2024": per_year_filtered_stats["2024"]["mean"],
            "AOD_filtered_median_2023": per_year_filtered_stats["2023"]["median"],
            "AOD_filtered_median_2024": per_year_filtered_stats["2024"]["median"],
            "AOD_filtered_min_2023": per_year_filtered_stats["2023"]["minimum"],
            "AOD_filtered_min_2024": per_year_filtered_stats["2024"]["minimum"],
            "AOD_filtered_max_2023": per_year_filtered_stats["2023"]["maximum"],
            "AOD_filtered_max_2024": per_year_filtered_stats["2024"]["maximum"],
            "AOD_filtered_p25_2023": per_year_filtered_stats["2023"]["p25"],
            "AOD_filtered_p25_2024": per_year_filtered_stats["2024"]["p25"],
            "AOD_filtered_p75_2023": per_year_filtered_stats["2023"]["p75"],
            "AOD_filtered_p75_2024": per_year_filtered_stats["2024"]["p75"],
            "AOD_outlier_lower_2023": per_year_filter_bounds["2023"][0],
            "AOD_outlier_lower_2024": per_year_filter_bounds["2024"][0],
            "AOD_outlier_upper_2023": per_year_filter_bounds["2023"][1],
            "AOD_outlier_upper_2024": per_year_filter_bounds["2024"][1],
            "AOD_mean_all": combined_stats["mean"],
            "AOD_median_all": combined_stats["median"],
            "AOD_std_all": combined_stats["std"],
            "AOD_min_all": combined_stats["minimum"],
            "AOD_max_all": combined_stats["maximum"],
            "AOD_p10_all": combined_stats["p10"],
            "AOD_p25_all": combined_stats["p25"],
            "AOD_p75_all": combined_stats["p75"],
            "AOD_p90_all": combined_stats["p90"],
            "AOD_filtered_mean_all": filtered_combined_stats["mean"],
            "AOD_filtered_median_all": filtered_combined_stats["median"],
            "AOD_filtered_std_all": filtered_combined_stats["std"],
            "AOD_filtered_min_all": filtered_combined_stats["minimum"],
            "AOD_filtered_max_all": filtered_combined_stats["maximum"],
            "AOD_filtered_p10_all": filtered_combined_stats["p10"],
            "AOD_filtered_p25_all": filtered_combined_stats["p25"],
            "AOD_filtered_p75_all": filtered_combined_stats["p75"],
            "AOD_filtered_p90_all": filtered_combined_stats["p90"],
            "AOD_outlier_lower_all": pooled_lower,
            "AOD_outlier_upper_all": pooled_upper,
            "AOD_outlier_removed_count_all": removed_count,
            "AOD_outlier_removed_fraction_all": removed_fraction,
            "suggested_AOD_fixed": filtered_combined_stats["median"],
            "suggested_method": suggested_method,
        }
        records.append(record)

    summary = pd.DataFrame(records).sort_values("suggested_AOD_fixed").reset_index(drop=True)
    return summary


def plot_boxplot(long_df: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    apply_plot_style()
    ordered_sites = summary["site_file"].tolist()
    display_labels = summary["site"].tolist()
    site_to_label = dict(zip(ordered_sites, display_labels))
    data = [
        tukey_filter(long_df.loc[long_df["site"] == site, "AOD_550nm"])[0].to_numpy()
        for site in ordered_sites
    ]
    suggested = summary["suggested_AOD_fixed"].to_numpy()

    fig_height = max(8, len(ordered_sites) * 0.38)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    box = ax.boxplot(
        data,
        vert=False,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        tick_labels=[site_to_label[site] for site in ordered_sites],
    )
    for patch in box["boxes"]:
        patch.set(facecolor="#8ecae6", edgecolor="#1d3557", alpha=0.85)
    for median in box["medians"]:
        median.set(color="#023047", linewidth=1.6)
    for whisker in box["whiskers"]:
        whisker.set(color="#577590", linewidth=1.2)
    for cap in box["caps"]:
        cap.set(color="#577590", linewidth=1.2)

    y_positions = np.arange(1, len(ordered_sites) + 1)
    ax.scatter(
        suggested,
        y_positions,
        color="#d62828",
        s=30,
        zorder=3,
        label="Annual fixed AOD",
    )
    ax.set_xlabel("AOD at 550 nm")
    ax.set_ylabel("Site")
    #ax.set_title("CARSNET site AOD distributions after outlier removal")
    ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
    ax.grid(axis="x", color="#d9d9d9", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(loc="lower right", fontsize=PLOT_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_support_figure(summary: pd.DataFrame, output_path: Path) -> None:
    apply_plot_style()
    ordered = summary.copy().reset_index(drop=True)
    y_positions = np.arange(len(ordered))

    fig_height = max(8, len(ordered) * 0.38)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.hlines(
        y_positions,
        ordered["AOD_filtered_p10_all"],
        ordered["AOD_filtered_p90_all"],
        color="#9aa0a6",
        linewidth=2,
        label="10th-90th percentile",
    )
    ax.hlines(
        y_positions,
        ordered["AOD_filtered_p25_all"],
        ordered["AOD_filtered_p75_all"],
        color="#457b9d",
        linewidth=6,
        label="25th-75th percentile",
    )
    ax.scatter(
        ordered["AOD_filtered_median_all"],
        y_positions,
        color="#1d3557",
        s=26,
        label="Pooled median",
        zorder=3,
    )
    ax.scatter(
        ordered["suggested_AOD_fixed"],
        y_positions,
        color="#d62828",
        s=34,
        marker="D",
        label="Suggested fixed AOD",
        zorder=4,
    )
    ax.scatter(
        ordered["AOD_filtered_mean_2023"],
        y_positions,
        color="#2a9d8f",
        s=20,
        marker="o",
        label="2023 mean",
        zorder=3,
    )
    ax.scatter(
        ordered["AOD_filtered_mean_2024"],
        y_positions,
        color="#f4a261",
        s=20,
        marker="^",
        label="2024 mean",
        zorder=3,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered["site"])
    ax.set_xlabel("AOD at 550 nm")
    ax.set_ylabel("Site")
    ax.set_title("AOD support ranges after outlier removal")
    ax.grid(axis="x", color="#d9d9d9", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    yearly_frames, long_df = load_year_site_frames(args.base_dir, YEARS)
    summary = summarize_sites(yearly_frames, long_df)

    summary_csv = args.output_dir / "carsnet_aod_site_summary_2023_2024.csv"
    summary.to_csv(summary_csv, index=False, float_format="%.6f")

    plot_boxplot(long_df, summary, args.output_dir / "carsnet_aod_site_boxplot_2023_2024.png")
    plot_support_figure(summary, args.output_dir / "carsnet_aod_site_support_ranges_2023_2024.png")

    print(f"Saved summary CSV: {summary_csv}")
    print("Suggested AOD rule: pooled median after site-wise Tukey outlier removal.")


if __name__ == "__main__":
    main()
