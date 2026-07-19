"""Compare cloudy measured GHI statistics between FY4A and GOES projects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FY4A_ROOT = SCRIPT_DIR.parents[2]
GOES_ROOT = Path("/Users/dengnan/Documents/git_store/Shortwave_MCRTM")

FY4A_PREDICTION_CSV = FY4A_ROOT / "FY4A_validation" / "Cloudy_results" / "cloudy_ghi_dw_surrogate_predictions.csv"
GOES_RESULT_DIR = GOES_ROOT / "GOES_validation" / "Analysis_cloudy" / "fullyear_DW_surrogateg2"


def load_fy4a() -> pd.DataFrame:
    df = pd.read_csv(FY4A_PREDICTION_CSV, parse_dates=["time"])
    return pd.DataFrame(
        {
            "project": "FY4A",
            "site": df["site"].astype(str),
            "time": df["time"],
            "GHI_measured": pd.to_numeric(df["GHI_ground"], errors="coerce"),
            "GHI_clear": pd.to_numeric(df["GHI_clear"], errors="coerce"),
        }
    )


def load_goes() -> pd.DataFrame:
    frames = []
    pattern = "Result_*_day_radiance_satellite_water_MarOct_DW_sug.csv"
    for path in sorted(GOES_RESULT_DIR.glob(pattern)):
        df = pd.read_csv(path, parse_dates=["Time"])
        site = path.name.split("_")[1]
        frames.append(
            pd.DataFrame(
                {
                    "project": "GOES",
                    "site": site,
                    "time": df["Time"],
                    "GHI_measured": pd.to_numeric(df["Site_dsw"], errors="coerce"),
                    "GHI_clear": pd.to_numeric(df["ghi_clear"], errors="coerce"),
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"No GOES result CSV files found in {GOES_RESULT_DIR}")
    return pd.concat(frames, ignore_index=True)


def summarize(group: pd.DataFrame) -> pd.Series:
    ghi = group["GHI_measured"].dropna().to_numpy(dtype=float)
    cloudy_index = (
        group["GHI_measured"].astype(float) / group["GHI_clear"].astype(float)
    ).replace([np.inf, -np.inf], np.nan)
    ci = cloudy_index.dropna().to_numpy(dtype=float)

    return pd.Series(
        {
            "n": len(ghi),
            "n_sites": group["site"].nunique(),
            "start_time": group["time"].min(),
            "end_time": group["time"].max(),
            "mean_GHI": np.mean(ghi) if len(ghi) else np.nan,
            "median_GHI": np.median(ghi) if len(ghi) else np.nan,
            "std_GHI": np.std(ghi, ddof=1) if len(ghi) > 1 else np.nan,
            "min_GHI": np.min(ghi) if len(ghi) else np.nan,
            "p05_GHI": np.percentile(ghi, 5) if len(ghi) else np.nan,
            "p25_GHI": np.percentile(ghi, 25) if len(ghi) else np.nan,
            "p75_GHI": np.percentile(ghi, 75) if len(ghi) else np.nan,
            "p95_GHI": np.percentile(ghi, 95) if len(ghi) else np.nan,
            "max_GHI": np.max(ghi) if len(ghi) else np.nan,
            "mean_cloudy_index": np.mean(ci) if len(ci) else np.nan,
            "median_cloudy_index": np.median(ci) if len(ci) else np.nan,
            "p25_cloudy_index": np.percentile(ci, 25) if len(ci) else np.nan,
            "p75_cloudy_index": np.percentile(ci, 75) if len(ci) else np.nan,
        }
    )


def main() -> None:
    observations = pd.concat([load_fy4a(), load_goes()], ignore_index=True)
    observations["month"] = observations["time"].dt.month
    observations["date"] = observations["time"].dt.date
    observations["cloudy_index"] = observations["GHI_measured"] / observations["GHI_clear"]
    observations = observations[np.isfinite(observations["GHI_measured"])].copy()

    stat_columns = ["site", "time", "GHI_measured", "GHI_clear"]
    overall = observations.groupby("project", sort=False)[stat_columns].apply(summarize).reset_index()
    by_site = observations.groupby(["project", "site"], sort=False)[stat_columns].apply(summarize).reset_index()
    by_month = observations.groupby(["project", "month"], sort=False)[stat_columns].apply(summarize).reset_index()

    observations.to_csv(SCRIPT_DIR / "combined_cloudy_GHI_measurements.csv", index=False)
    overall.to_csv(SCRIPT_DIR / "overall_cloudy_GHI_measurement_stats.csv", index=False)
    by_site.to_csv(SCRIPT_DIR / "site_cloudy_GHI_measurement_stats.csv", index=False)
    by_month.to_csv(SCRIPT_DIR / "monthly_cloudy_GHI_measurement_stats.csv", index=False)

    print(f"FY4A source: {FY4A_PREDICTION_CSV}")
    print(f"GOES source directory: {GOES_RESULT_DIR}")
    print(f"Output directory: {SCRIPT_DIR}")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
