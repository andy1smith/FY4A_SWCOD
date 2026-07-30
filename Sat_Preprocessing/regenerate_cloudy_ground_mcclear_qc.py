from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location

from clearsky_model.clearsky_filter import (
    cloudy_day_masks,
    empirical_low_ghi_limit,
    ground_irradiance_qc_mask,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_GHI_PATH = REPO_ROOT / "FY4A_data" / "CERN_instGHI_2021_UTC.csv"
SITE_INFO_PATH = REPO_ROOT / "FY4A_data" / "CERN_info.csv"
MCCLEAR_DIR = REPO_ROOT / "FY4A_data" / "McClear_clearsky"
GROUND_DIR = REPO_ROOT / "Sat_Preprocessing" / "Ground" / "preprocessed_GHI"
STATS_DIR = REPO_ROOT / "FY4A_validation" / "Cloudy_results" / "FY4A_GOES_cloudy_GHI_stats"

SUMMARY_PATH = STATS_DIR / "cern_raw_clear_cloudy_mcclear_qc_summary.csv"
SITE_SUMMARY_PATH = STATS_DIR / "cern_raw_clear_cloudy_mcclear_qc_by_site.csv"
PLOT_PATH = STATS_DIR / "FY4A_GOES_measured_cloudy_GHI_distribution_SZA_le65_timeQC_CIlt08.png"


def read_mcclear(site: str) -> pd.Series:
    paths = sorted(MCCLEAR_DIR.glob(f"{site}_mcclear_*_hourly.csv"))
    if not paths:
        raise FileNotFoundError(f"No McClear cache found for {site} in {MCCLEAR_DIR}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "Time" not in df.columns:
            raise ValueError(f"{path} is missing Time")
        if "ghi_clear_mcclear" not in df.columns:
            if "ghi_clear" in df.columns:
                df = df.rename(columns={"ghi_clear": "ghi_clear_mcclear"})
            else:
                raise ValueError(f"{path} is missing ghi_clear_mcclear")
        df = df[["Time", "ghi_clear_mcclear"]].copy()
        df["Time"] = pd.to_datetime(df["Time"]).dt.tz_localize(None)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Time", "ghi_clear_mcclear"])
    df = df.drop_duplicates("Time", keep="first").set_index("Time").sort_index()
    return df["ghi_clear_mcclear"]


def build_site_frame(site: str, raw_ghi: pd.DataFrame, site_info: pd.Series, times_utc: pd.DatetimeIndex) -> pd.DataFrame:
    lat = float(site_info["latitude"])
    lon = float(site_info["longitude"])
    elev = float(site_info["elve"]) if pd.notna(site_info.get("elve")) else 0.0

    loc = Location(lat, lon, tz="UTC", altitude=elev, name=site)
    solpos = loc.get_solarposition(times_utc)
    tl = pvlib.clearsky.lookup_linke_turbidity(times_utc, lat, lon)
    clear_pvlib = loc.get_clearsky(times_utc, model="ineichen", linke_turbidity=tl)["ghi"]

    df = pd.DataFrame(
        {
            "ghi": pd.to_numeric(raw_ghi[site], errors="coerce").to_numpy(dtype=float),
            "Sun_Zen": solpos["zenith"].to_numpy(dtype=float),
            "Sun_Zen_App": solpos["apparent_zenith"].to_numpy(dtype=float),
            "Sun_Azi": solpos["azimuth"].to_numpy(dtype=float),
            "ghi_clear": clear_pvlib.to_numpy(dtype=float),
        },
        index=times_utc,
    )
    df.index.name = "Time"

    times_naive = times_utc.tz_convert("UTC").tz_localize(None)
    mcclear = read_mcclear(site).reindex(times_naive)
    df["ghi_clear_mcclear"] = mcclear.to_numpy(dtype=float)
    df["clear_index_mcclear"] = df["ghi"] / df["ghi_clear_mcclear"].replace(0, np.nan)
    df["GHI_min_empirical"] = empirical_low_ghi_limit(df["Sun_Zen"])
    return df


def hdf_times(path: Path) -> set[pd.Timestamp]:
    if not path.exists():
        return set()
    df = pd.read_hdf(path, key="df")
    if "Time" not in df.columns:
        return set()
    return set(pd.to_datetime(df["Time"], utc=True))


def backup_cloudy_h5() -> Path:
    backup_dir = GROUND_DIR / "backup_cloudy_h5_before_mcclear_qc_20260722"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(GROUND_DIR.glob("*_cloudy.h5")):
        backup_path = backup_dir / path.name
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
    return backup_dir


def write_cloudy_h5(site: str, df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    out = df.loc[mask, ["ghi", "Sun_Zen", "Sun_Azi", "ghi_clear", "Sun_Zen_App"]].copy()
    out = out.rename(columns={"ghi_clear": "ghi_clear_pvlib"})
    out["ghi_clear"] = df.loc[mask, "ghi_clear_mcclear"]
    out["ghi_clear_mcclear"] = df.loc[mask, "ghi_clear_mcclear"]
    out["clear_index_mcclear"] = df.loc[mask, "clear_index_mcclear"]
    out["GHI_min_empirical"] = df.loc[mask, "GHI_min_empirical"]
    out = out[[
        "ghi",
        "Sun_Zen",
        "Sun_Azi",
        "ghi_clear",
        "Sun_Zen_App",
        "ghi_clear_pvlib",
        "ghi_clear_mcclear",
        "clear_index_mcclear",
        "GHI_min_empirical",
    ]]
    out = out.reset_index()
    out.to_hdf(GROUND_DIR / f"{site}_cloudy.h5", key="df", mode="w")
    return out


def kde_or_hist(ax, values: pd.Series, label: str, color: str) -> None:
    values = pd.Series(values).dropna()
    n_total = len(values)
    values = values[(values >= 0) & (values <= 1300)]
    if values.empty:
        return
    try:
        values.plot(kind="kde", ax=ax, color=color, linewidth=2.2, label=f"{label} (n={n_total:,})")
    except Exception:
        bins = np.linspace(0, 1300, 66)
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2.2, color=color, label=f"{label} (n={n_total:,})")


def plot_distribution(clear_values: list[pd.Series], cloudy_before: list[pd.Series], cloudy_after: list[pd.Series]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    kde_or_hist(ax, pd.concat(clear_values, ignore_index=True), "Clear", "#1f77b4")
    kde_or_hist(ax, pd.concat(cloudy_before, ignore_index=True), "Cloudy before QC", "#ff7f0e")
    kde_or_hist(ax, pd.concat(cloudy_after, ignore_index=True), "Cloudy after QC", "#2ca02c")
    ax.set_xlim(0, 1300)
    ax.set_xlabel("Measured GHI (W m$^{-2}$)")
    ax.set_ylabel("Density")
    ax.set_title("CERN 2021 measured GHI distribution, SZA <= 65")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_cloudy_h5()

    raw_ghi = pd.read_csv(RAW_GHI_PATH)
    site_info = pd.read_csv(SITE_INFO_PATH).drop_duplicates("site").set_index("site")
    times_utc = pd.date_range("2021-01-01 00:00", periods=len(raw_ghi), freq="h", tz="UTC")

    site_rows = []
    clear_plot_values = []
    cloudy_before_plot_values = []
    cloudy_after_plot_values = []

    for site in raw_ghi.columns:
        if site not in site_info.index:
            continue
        print(site, flush=True)
        df = build_site_frame(site, raw_ghi, site_info.loc[site], times_utc)
        daylight = pd.Series(
            np.isfinite(df["ghi"]) & np.isfinite(df["Sun_Zen"]) & (df["Sun_Zen"] <= 85),
            index=df.index,
        )
        work = df.loc[daylight].copy()
        if work.empty:
            write_cloudy_h5(site, df, pd.Series(False, index=df.index))
            site_rows.append(
                {
                    "site": site,
                    "total_day_samples_sza_le85": 0,
                    "clear_h5_current": len(hdf_times(GROUND_DIR / f"{site}_clear.h5")),
                    "cloudy_before_qc_sza_le85": 0,
                    "cloudy_after_qc_sza_le85": 0,
                    "removed_by_qc_sza_le85": 0,
                    "cloudy_before_qc_sza_le65": 0,
                    "cloudy_after_qc_sza_le65": 0,
                    "removed_by_qc_sza_le65": 0,
                }
            )
            continue

        quan85, surfrad_clear, cloudy_candidate, cloudy_qc = cloudy_day_masks(work, float(site_info.loc[site, "longitude"]))
        after_mask_work = cloudy_candidate & cloudy_qc
        surfrad_extra_clear = surfrad_clear & (~quan85)
        full_after_mask = pd.Series(False, index=df.index)
        full_after_mask.loc[work.index] = after_mask_work
        out = write_cloudy_h5(site, df, full_after_mask)

        clear_current_times = hdf_times(GROUND_DIR / f"{site}_clear.h5")
        clear_current = pd.Series(df.index.isin(clear_current_times), index=df.index)
        sza65_work = work["Sun_Zen"] <= 65
        before65 = cloudy_candidate & sza65_work
        after65 = after_mask_work & sza65_work
        surfrad_extra_clear65 = surfrad_extra_clear & sza65_work

        removed_work = cloudy_candidate & ~cloudy_qc
        site_rows.append(
            {
                "site": site,
                "total_day_samples_sza_le85": int(daylight.sum()),
                "clear_h5_current": int(len(clear_current_times)),
                "clear_h5_current_sza_le65": int((clear_current & (df["Sun_Zen"] <= 65)).sum()),
                "surfrad_clear_extra_sza_le85": int(surfrad_extra_clear.sum()),
                "cloudy_before_qc_sza_le85": int(cloudy_candidate.sum()),
                "cloudy_after_qc_sza_le85": int(after_mask_work.sum()),
                "removed_by_qc_sza_le85": int(removed_work.sum()),
                "removed_ghi_lt50_sza_le85": int((cloudy_candidate & (work["ghi"] < 50)).sum()),
                "removed_ci_mcclear_lt003_sza_le85": int((cloudy_candidate & (work["clear_index_mcclear"] < 0.03)).sum()),
                "removed_empirical_ghi_min_sza_le85": int((cloudy_candidate & (work["ghi"] < work["GHI_min_empirical"])).sum()),
                "surfrad_clear_extra_sza_le65": int(surfrad_extra_clear65.sum()),
                "cloudy_before_qc_sza_le65": int(before65.sum()),
                "cloudy_after_qc_sza_le65": int(after65.sum()),
                "removed_by_qc_sza_le65": int((before65 & ~cloudy_qc).sum()),
                "removed_ghi_lt50_sza_le65": int((before65 & (work["ghi"] < 50)).sum()),
                "removed_ci_mcclear_lt003_sza_le65": int((before65 & (work["clear_index_mcclear"] < 0.03)).sum()),
                "removed_empirical_ghi_min_sza_le65": int((before65 & (work["ghi"] < work["GHI_min_empirical"])).sum()),
                "written_cloudy_h5_rows": int(len(out)),
            }
        )

        clear_values = df.loc[clear_current & (df["Sun_Zen"] <= 65), "ghi"]
        if not clear_values.empty:
            clear_plot_values.append(clear_values)
        if before65.any():
            cloudy_before_plot_values.append(work.loc[before65, "ghi"])
        if after65.any():
            cloudy_after_plot_values.append(work.loc[after65, "ghi"])

    site_summary = pd.DataFrame(site_rows)
    site_summary.to_csv(SITE_SUMMARY_PATH, index=False)
    total = site_summary.drop(columns=["site"]).sum(numeric_only=True)
    total["backup_dir"] = str(backup_dir)
    total.to_frame().T.to_csv(SUMMARY_PATH, index=False)
    plot_distribution(clear_plot_values, cloudy_before_plot_values, cloudy_after_plot_values)

    print(f"Backup: {backup_dir}")
    print(f"Saved site summary: {SITE_SUMMARY_PATH}")
    print(f"Saved total summary: {SUMMARY_PATH}")
    print(f"Saved plot: {PLOT_PATH}")
    print(total.to_string())


if __name__ == "__main__":
    main()
