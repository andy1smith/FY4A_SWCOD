"""Monthly land-cover-group metrics for FY4A cloudy GHI and COD retrievals.

This script summarizes the re-extracted SZA/time-QC result folders by the
land-cover grouping used for the satellite cloud retrieval analysis.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = SCRIPT_DIR / "seasonal_land_group_metrics" / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUDY_RESULTS = REPO_ROOT / "FY4A_validation" / "Cloudy_results"
GHI_DIR = CLOUDY_RESULTS / "Cloudy_dw_HG_reextract_SZA1_CIge015"
COD_DIR = CLOUDY_RESULTS / "Cloudy_COD_reextract_SZA1_CIge015"
OUT_DIR = CLOUDY_RESULTS / "seasonal_land_group_metrics"


SITE_INFO = {
    "AKA": ("A", "Agriculture", "Vegetated land"),
    "ASA": ("A", "Agriculture", "Vegetated land"),
    "CSA": ("A", "Agriculture", "Vegetated land"),
    "CWA": ("A", "Agriculture", "Vegetated land"),
    "FQA": ("A", "Agriculture", "Vegetated land"),
    "HJA": ("A", "Agriculture", "Vegetated land"),
    "HLA": ("A", "Agriculture", "Vegetated land"),
    "LCA": ("A", "Agriculture", "Vegetated land"),
    "LSA": ("A", "Agriculture", "Vegetated land"),
    "QYA": ("A", "Agriculture", "Vegetated land"),
    "YCA": ("A", "Agriculture", "Vegetated land"),
    "YGA": ("A", "Agriculture", "Vegetated land"),
    "YTA": ("A", "Agriculture", "Vegetated land"),
    "NMG": ("G", "Grassland", "Vegetated land"),
    "BJF": ("F", "Forest", "Forest / mountain forest"),
    "BNF": ("F", "Forest", "Forest / mountain forest"),
    "CBF": ("F", "Forest", "Forest / mountain forest"),
    "DHF": ("F", "Forest", "Forest / mountain forest"),
    "GGS": ("F", "Forest", "Forest / mountain forest"),
    "HSF": ("F", "Forest", "Forest / mountain forest"),
    "HTF": ("F", "Forest", "Forest / mountain forest"),
    "MXF": ("F", "Forest", "Forest / mountain forest"),
    "PDF": ("F", "Forest", "Forest / mountain forest"),
    "SNF": ("F", "Forest", "Forest / mountain forest"),
    "CLD": ("D", "Desert", "Dry / bare land"),
    "ESD": ("D", "Desert", "Dry / bare land"),
    "FKD": ("D", "Desert", "Dry / bare land"),
    "LZD": ("D", "Desert", "Dry / bare land"),
    "NMD": ("D", "Desert", "Dry / bare land"),
    "SJM": ("M", "Marsh", "Water / wetland"),
    "DHL": ("L", "Lake", "Water / wetland"),
    "THL": ("L", "Lake", "Water / wetland"),
    "JZB": ("B", "Marine / bay", "Water / wetland"),
    "SYB": ("B", "Marine / bay", "Water / wetland"),
    "BJC": ("C", "City", "Urban"),
}

GROUP_ORDER = [
    "Vegetated land",
    "Forest / mountain forest",
    "Dry / bare land",
    "Water / wetland",
    "Urban",
]


def metric_summary(frame: pd.DataFrame, pred_col: str) -> pd.Series:
    valid = frame[["GHI_ground", pred_col]].dropna()
    site_count = frame["site"].nunique() if "site" in frame.columns else 1
    if valid.empty:
        return pd.Series(
            {
                "N": 0,
                "site_count": site_count,
                "GHI_ground_mean": np.nan,
                "GHI_pred_mean": np.nan,
                "MBE": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
                "R": np.nan,
                "rMBE": np.nan,
                "rRMSE": np.nan,
            }
        )

    diff = valid[pred_col] - valid["GHI_ground"]
    ground_mean = valid["GHI_ground"].mean()
    mbe = diff.mean()
    rmse = np.sqrt(np.mean(diff**2))
    r = valid[pred_col].corr(valid["GHI_ground"]) if len(valid) > 1 else np.nan
    return pd.Series(
        {
            "N": len(valid),
            "site_count": site_count,
            "GHI_ground_mean": ground_mean,
            "GHI_pred_mean": valid[pred_col].mean(),
            "MBE": mbe,
            "MAE": diff.abs().mean(),
            "RMSE": rmse,
            "R": r,
            "rMBE": 100.0 * mbe / ground_mean if ground_mean else np.nan,
            "rRMSE": 100.0 * rmse / ground_mean if ground_mean else np.nan,
        }
    )


def add_site_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    site_meta = out["site"].map(SITE_INFO)
    missing = sorted(out.loc[site_meta.isna(), "site"].unique())
    if missing:
        raise ValueError(f"Missing site group classification for: {missing}")

    out["class_code"] = site_meta.map(lambda item: item[0])
    out["class_name"] = site_meta.map(lambda item: item[1])
    out["retrieval_group"] = site_meta.map(lambda item: item[2])
    out["retrieval_group"] = pd.Categorical(
        out["retrieval_group"], categories=GROUP_ORDER, ordered=True
    )
    return out


def summarize_ghi() -> pd.DataFrame:
    center = pd.read_csv(GHI_DIR / "cloudy_ghi_dw_surrogate_predictions.csv")
    center["time"] = pd.to_datetime(center["time"])
    center = add_site_groups(center)
    center["month"] = center["time"].dt.month
    center["month_name"] = center["time"].dt.strftime("%b")

    prediction_cols = ["GHI_center", "GHI_center_3x3_mean"]
    rows = []
    for pred_col in prediction_cols:
        summary = (
            center.groupby(["retrieval_group", "month", "month_name"], observed=True)
            .apply(metric_summary, pred_col=pred_col, include_groups=False)
            .reset_index()
        )
        summary.insert(3, "prediction", pred_col)
        rows.append(summary)

    closest_path = GHI_DIR / "cloudy_ghi_dw_surrogate_closest_pixel_predictions.csv"
    if closest_path.exists():
        closest = pd.read_csv(closest_path)
        closest["time"] = pd.to_datetime(closest["time"])
        closest = add_site_groups(closest)
        closest["month"] = closest["time"].dt.month
        closest["month_name"] = closest["time"].dt.strftime("%b")
        summary = (
            closest.groupby(["retrieval_group", "month", "month_name"], observed=True)
            .apply(metric_summary, pred_col="GHI_retrieved", include_groups=False)
            .reset_index()
        )
        summary.insert(3, "prediction", "GHI_closest_pixel")
        rows.append(summary)

    ghi = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "GHI_ground_mean",
        "GHI_pred_mean",
        "MBE",
        "MAE",
        "RMSE",
        "R",
        "rMBE",
        "rRMSE",
    ]
    ghi[numeric_cols] = ghi[numeric_cols].round(3)
    return ghi


def summarize_ghi_by_site() -> pd.DataFrame:
    center = pd.read_csv(GHI_DIR / "cloudy_ghi_dw_surrogate_predictions.csv")
    center["time"] = pd.to_datetime(center["time"])
    center = add_site_groups(center)
    center["month"] = center["time"].dt.month
    center["month_name"] = center["time"].dt.strftime("%b")

    rows = []
    for pred_col in ["GHI_center", "GHI_center_3x3_mean"]:
        summary = (
            center.groupby(
                [
                    "site",
                    "class_code",
                    "class_name",
                    "retrieval_group",
                    "month",
                    "month_name",
                ],
                observed=True,
            )
            .apply(metric_summary, pred_col=pred_col, include_groups=False)
            .reset_index()
        )
        summary.insert(6, "prediction", pred_col)
        rows.append(summary)

    closest_path = GHI_DIR / "cloudy_ghi_dw_surrogate_closest_pixel_predictions.csv"
    if closest_path.exists():
        closest = pd.read_csv(closest_path)
        closest["time"] = pd.to_datetime(closest["time"])
        closest = add_site_groups(closest)
        closest["month"] = closest["time"].dt.month
        closest["month_name"] = closest["time"].dt.strftime("%b")
        summary = (
            closest.groupby(
                [
                    "site",
                    "class_code",
                    "class_name",
                    "retrieval_group",
                    "month",
                    "month_name",
                ],
                observed=True,
            )
            .apply(metric_summary, pred_col="GHI_retrieved", include_groups=False)
            .reset_index()
        )
        summary.insert(6, "prediction", "GHI_closest_pixel")
        rows.append(summary)

    ghi = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "GHI_ground_mean",
        "GHI_pred_mean",
        "MBE",
        "MAE",
        "RMSE",
        "R",
        "rMBE",
        "rRMSE",
    ]
    ghi[numeric_cols] = ghi[numeric_cols].round(3)
    return ghi


def site_cod_timeseries(path: Path) -> pd.DataFrame:
    site = path.name.split("_cloudy_COD_uw_ADM.nc")[0]
    with xr.open_dataset(path) as ds:
        cod = ds["Retrieved_COD"]
        wrmse = ds["WRMSE_sug_adm"]
        center = cod.sel(y=5, x=5)
        center_wrmse = wrmse.sel(y=5, x=5)
        mean_3x3 = cod.sel(y=slice(4, 6), x=slice(4, 6)).mean(("y", "x"), skipna=True)
        wrmse_3x3 = wrmse.sel(y=slice(4, 6), x=slice(4, 6)).mean(
            ("y", "x"), skipna=True
        )
        mean_11x11 = cod.mean(("y", "x"), skipna=True)
        wrmse_11x11 = wrmse.mean(("y", "x"), skipna=True)
        df = pd.DataFrame(
            {
                "site": site,
                "time": pd.to_datetime(ds["time"].values),
                "COD_center": center.values,
                "COD_3x3_mean": mean_3x3.values,
                "COD_11x11_mean": mean_11x11.values,
                "WRMSE_center": center_wrmse.values,
                "WRMSE_3x3_mean": wrmse_3x3.values,
                "WRMSE_11x11_mean": wrmse_11x11.values,
            }
        )
    return df


def cod_summary(frame: pd.DataFrame, value_col: str, wrmse_col: str) -> pd.Series:
    valid = frame[[value_col, wrmse_col]].dropna()
    site_count = frame["site"].nunique() if "site" in frame.columns else 1
    if valid.empty:
        return pd.Series(
            {
                "N": 0,
                "site_count": site_count,
                "COD_mean": np.nan,
                "COD_median": np.nan,
                "COD_p25": np.nan,
                "COD_p75": np.nan,
                "COD_p90": np.nan,
                "WRMSE_mean": np.nan,
            }
        )

    return pd.Series(
        {
            "N": len(valid),
            "site_count": site_count,
            "COD_mean": valid[value_col].mean(),
            "COD_median": valid[value_col].median(),
            "COD_p25": valid[value_col].quantile(0.25),
            "COD_p75": valid[value_col].quantile(0.75),
            "COD_p90": valid[value_col].quantile(0.90),
            "WRMSE_mean": valid[wrmse_col].mean(),
        }
    )


def summarize_cod() -> tuple[pd.DataFrame, pd.DataFrame]:
    cod_frames = [
        site_cod_timeseries(path)
        for path in sorted(COD_DIR.glob("*_cloudy_COD_uw_ADM.nc"))
    ]
    cod_ts = pd.concat(cod_frames, ignore_index=True)
    cod_ts = add_site_groups(cod_ts)
    cod_ts["month"] = cod_ts["time"].dt.month
    cod_ts["month_name"] = cod_ts["time"].dt.strftime("%b")

    products = [
        ("COD_center", "WRMSE_center", "center"),
        ("COD_3x3_mean", "WRMSE_3x3_mean", "center_3x3_mean"),
        ("COD_11x11_mean", "WRMSE_11x11_mean", "full_11x11_mean"),
    ]
    rows = []
    for value_col, wrmse_col, product in products:
        summary = (
            cod_ts.groupby(["retrieval_group", "month", "month_name"], observed=True)
            .apply(
                cod_summary,
                value_col=value_col,
                wrmse_col=wrmse_col,
                include_groups=False,
            )
            .reset_index()
        )
        summary.insert(3, "cod_product", product)
        rows.append(summary)

    cod = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "COD_mean",
        "COD_median",
        "COD_p25",
        "COD_p75",
        "COD_p90",
        "WRMSE_mean",
    ]
    cod[numeric_cols] = cod[numeric_cols].round(3)
    return cod, cod_ts


def summarize_cod_by_site(cod_ts: pd.DataFrame) -> pd.DataFrame:
    products = [
        ("COD_center", "WRMSE_center", "center"),
        ("COD_3x3_mean", "WRMSE_3x3_mean", "center_3x3_mean"),
        ("COD_11x11_mean", "WRMSE_11x11_mean", "full_11x11_mean"),
    ]
    rows = []
    for value_col, wrmse_col, product in products:
        summary = (
            cod_ts.groupby(
                [
                    "site",
                    "class_code",
                    "class_name",
                    "retrieval_group",
                    "month",
                    "month_name",
                ],
                observed=True,
            )
            .apply(
                cod_summary,
                value_col=value_col,
                wrmse_col=wrmse_col,
                include_groups=False,
            )
            .reset_index()
        )
        summary.insert(6, "cod_product", product)
        rows.append(summary)

    cod = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "COD_mean",
        "COD_median",
        "COD_p25",
        "COD_p75",
        "COD_p90",
        "WRMSE_mean",
    ]
    cod[numeric_cols] = cod[numeric_cols].round(3)
    return cod


def write_site_group_table(ghi: pd.DataFrame, cod_ts: pd.DataFrame) -> pd.DataFrame:
    sites = sorted(set(ghi_sites_from_csv()) | set(cod_ts["site"].unique()))
    rows = []
    for site in sites:
        class_code, class_name, group = SITE_INFO[site]
        rows.append(
            {
                "site": site,
                "class_code": class_code,
                "class_name": class_name,
                "retrieval_group": group,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "site_land_group_classification.csv", index=False)
    return table


def ghi_sites_from_csv() -> list[str]:
    return pd.read_csv(GHI_DIR / "cloudy_ghi_dw_surrogate_predictions.csv", usecols=["site"])[
        "site"
    ].unique().tolist()


def plot_ghi(ghi: pd.DataFrame) -> None:
    target = ghi[ghi["prediction"] == "GHI_center_3x3_mean"].copy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    metrics = [("RMSE", "RMSE (W m-2)"), ("MBE", "MBE (W m-2)"), ("R", "R")]
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        for group in GROUP_ORDER:
            sub = target[target["retrieval_group"] == group]
            if sub.empty:
                continue
            ax.plot(sub["month"], sub[metric], marker="o", label=group)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Month in 2021")
    axes[0].legend(ncols=2, fontsize=9)
    fig.suptitle("Monthly GHI metrics by land-cover group (center 3x3 mean)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "monthly_ghi_metrics_by_group.png", dpi=220)
    plt.close(fig)


def plot_cod(cod: pd.DataFrame) -> None:
    target = cod[cod["cod_product"] == "center_3x3_mean"].copy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "COD_median", "Median COD"),
        (axes[1], "COD_mean", "Mean COD"),
    ]:
        for group in GROUP_ORDER:
            sub = target[target["retrieval_group"] == group]
            if sub.empty:
                continue
            ax.plot(sub["month"], sub[metric], marker="o", label=group)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Month in 2021")
    axes[0].legend(ncols=2, fontsize=9)
    fig.suptitle("Monthly retrieved COD by land-cover group (center 3x3 mean)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "monthly_cod_by_group.png", dpi=220)
    plt.close(fig)


def plot_site_heatmaps(ghi_site: pd.DataFrame, cod_site: pd.DataFrame) -> None:
    ghi_target = ghi_site[ghi_site["prediction"] == "GHI_center_3x3_mean"]
    cod_target = cod_site[cod_site["cod_product"] == "center_3x3_mean"]

    site_order = (
        ghi_target[["site", "retrieval_group"]]
        .drop_duplicates()
        .assign(
            retrieval_group=lambda df: pd.Categorical(
                df["retrieval_group"], categories=GROUP_ORDER, ordered=True
            )
        )
        .sort_values(["retrieval_group", "site"])["site"]
        .tolist()
    )

    for frame, value_col, title, path_name in [
        (
            ghi_target,
            "RMSE",
            "Monthly GHI RMSE by site (center 3x3 mean, N >= 10)",
            "monthly_site_ghi_rmse_heatmap.png",
        ),
        (
            cod_target,
            "COD_median",
            "Monthly median COD by site (center 3x3 mean, N >= 10)",
            "monthly_site_cod_median_heatmap.png",
        ),
    ]:
        table = frame[frame["N"] >= 10].pivot(
            index="site", columns="month", values=value_col
        ).reindex(
            site_order
        )
        fig, ax = plt.subplots(figsize=(9, 9))
        im = ax.imshow(table.values, aspect="auto", interpolation="nearest")
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels(table.index)
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels(table.columns)
        ax.set_xlabel("Month in 2021")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(value_col)
        fig.tight_layout()
        fig.savefig(OUT_DIR / path_name, dpi=220)
        plt.close(fig)


def write_findings(ghi: pd.DataFrame, cod: pd.DataFrame, site_table: pd.DataFrame) -> None:
    main_ghi = ghi[ghi["prediction"] == "GHI_center_3x3_mean"].copy()
    main_cod = cod[cod["cod_product"] == "center_3x3_mean"].copy()

    lines = [
        "# Monthly Seasonal Land-Group Metrics",
        "",
        "Main GHI product: `GHI_center_3x3_mean`.",
        "Main COD product: `center_3x3_mean` retrieved COD.",
        "",
        "## Sites Used",
        "",
        site_table.groupby("retrieval_group", observed=True)["site"]
        .apply(lambda s: ", ".join(s))
        .to_string(),
        "",
        "## GHI Highlights",
        "",
    ]

    for group in GROUP_ORDER:
        sub = main_ghi[main_ghi["retrieval_group"] == group]
        if sub.empty:
            continue
        best = sub.loc[sub["RMSE"].idxmin()]
        worst = sub.loc[sub["RMSE"].idxmax()]
        lines.append(
            f"- {group}: RMSE min {best.RMSE:.1f} W m-2 in {best.month_name}; "
            f"max {worst.RMSE:.1f} W m-2 in {worst.month_name}; "
            f"mean bias range {sub.MBE.min():.1f} to {sub.MBE.max():.1f} W m-2."
        )

    lines.extend(["", "## COD Highlights", ""])
    for group in GROUP_ORDER:
        sub = main_cod[main_cod["retrieval_group"] == group]
        if sub.empty:
            continue
        low = sub.loc[sub["COD_median"].idxmin()]
        high = sub.loc[sub["COD_median"].idxmax()]
        lines.append(
            f"- {group}: median COD min {low.COD_median:.2f} in {low.month_name}; "
            f"max {high.COD_median:.2f} in {high.month_name}; "
            f"monthly site_count range {int(sub.site_count.min())}-{int(sub.site_count.max())}."
        )

    lines.extend(
        [
            "",
            "Sparse groups or months should be interpreted cautiously; use the `N` and "
            "`site_count` columns in the CSV outputs before drawing physical conclusions.",
            "",
        ]
    )
    (OUT_DIR / "monthly_seasonal_findings.md").write_text("\n".join(lines))


def write_site_findings(ghi_site: pd.DataFrame, cod_site: pd.DataFrame) -> None:
    main_ghi = ghi_site[
        (ghi_site["prediction"] == "GHI_center_3x3_mean") & (ghi_site["N"] >= 10)
    ].copy()
    main_cod = cod_site[
        (cod_site["cod_product"] == "center_3x3_mean") & (cod_site["N"] >= 10)
    ].copy()

    site_overall_ghi = (
        main_ghi.groupby(["site", "class_code", "class_name", "retrieval_group"], observed=True)
        .agg(
            N=("N", "sum"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_max=("RMSE", "max"),
            RMSE_min=("RMSE", "min"),
            MBE_mean=("MBE", "mean"),
            R_mean=("R", "mean"),
        )
        .reset_index()
    )
    site_overall_cod = (
        main_cod.groupby(["site"], observed=True)
        .agg(
            COD_median_mean=("COD_median", "mean"),
            COD_median_min=("COD_median", "min"),
            COD_median_max=("COD_median", "max"),
            COD_mean_mean=("COD_mean", "mean"),
        )
        .reset_index()
    )
    site_summary = site_overall_ghi.merge(site_overall_cod, on="site", how="left")
    site_summary = site_summary.round(3)
    site_summary.to_csv(OUT_DIR / "site_overall_summary.csv", index=False)

    highest_rmse = site_summary.sort_values("RMSE_mean", ascending=False).head(5)
    lowest_rmse = site_summary.sort_values("RMSE_mean", ascending=True).head(5)
    positive_bias = site_summary.sort_values("MBE_mean", ascending=False).head(5)
    negative_bias = site_summary.sort_values("MBE_mean", ascending=True).head(5)
    high_cod = site_summary.sort_values("COD_median_mean", ascending=False).head(5)
    low_cod = site_summary.sort_values("COD_median_mean", ascending=True).head(5)

    lines = [
        "# Site-Level Seasonal Findings",
        "",
        "Main GHI product: `GHI_center_3x3_mean`; main COD product: `center_3x3_mean`.",
        "Monthly rows with `N < 10` are excluded from these ranked highlights.",
        "",
        "## Strongest GHI Error Sites",
        "",
    ]
    for row in highest_rmse.itertuples(index=False):
        lines.append(
            f"- {row.site} ({row.class_name}, {row.retrieval_group}): "
            f"mean monthly RMSE {row.RMSE_mean:.1f} W m-2, "
            f"mean MBE {row.MBE_mean:.1f} W m-2, mean R {row.R_mean:.2f}."
        )

    lines.extend(["", "## Best GHI Agreement Sites", ""])
    for row in lowest_rmse.itertuples(index=False):
        lines.append(
            f"- {row.site} ({row.class_name}, {row.retrieval_group}): "
            f"mean monthly RMSE {row.RMSE_mean:.1f} W m-2, "
            f"mean MBE {row.MBE_mean:.1f} W m-2, mean R {row.R_mean:.2f}."
        )

    lines.extend(["", "## Largest Mean Bias", ""])
    for row in positive_bias.itertuples(index=False):
        lines.append(f"- Positive: {row.site}, MBE {row.MBE_mean:.1f} W m-2.")
    for row in negative_bias.itertuples(index=False):
        lines.append(f"- Negative: {row.site}, MBE {row.MBE_mean:.1f} W m-2.")

    lines.extend(["", "## COD Extremes", ""])
    for row in high_cod.itertuples(index=False):
        lines.append(
            f"- High COD: {row.site}, mean monthly median COD {row.COD_median_mean:.2f} "
            f"(range {row.COD_median_min:.2f}-{row.COD_median_max:.2f})."
        )
    for row in low_cod.itertuples(index=False):
        lines.append(
            f"- Low COD: {row.site}, mean monthly median COD {row.COD_median_mean:.2f} "
            f"(range {row.COD_median_min:.2f}-{row.COD_median_max:.2f})."
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Site-level analysis exposes heterogeneity hidden by land-cover means.",
            "- Check `monthly_ghi_metrics_by_site.csv` and `monthly_cod_metrics_by_site.csv` for `N` before using a site-month as evidence.",
            "- Forest group results are strongly affected by the small available set in this reextract run: `BJF`, `CBF`, and `GGS`.",
            "",
        ]
    )
    (OUT_DIR / "site_seasonal_findings.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ghi = summarize_ghi()
    ghi_site = summarize_ghi_by_site()
    cod, cod_ts = summarize_cod()
    cod_site = summarize_cod_by_site(cod_ts)
    site_table = write_site_group_table(ghi, cod_ts)

    ghi.to_csv(OUT_DIR / "monthly_ghi_metrics_by_group.csv", index=False)
    ghi_site.to_csv(OUT_DIR / "monthly_ghi_metrics_by_site.csv", index=False)
    cod.to_csv(OUT_DIR / "monthly_cod_metrics_by_group.csv", index=False)
    cod_site.to_csv(OUT_DIR / "monthly_cod_metrics_by_site.csv", index=False)
    cod_ts.to_csv(OUT_DIR / "cod_site_monthly_input_timeseries.csv", index=False)

    plot_ghi(ghi)
    plot_cod(cod)
    plot_site_heatmaps(ghi_site, cod_site)
    write_findings(ghi, cod, site_table)
    write_site_findings(ghi_site, cod_site)

    print(f"Wrote outputs to {OUT_DIR}")
    print("Primary files:")
    for name in [
        "monthly_ghi_metrics_by_group.csv",
        "monthly_cod_metrics_by_group.csv",
        "monthly_ghi_metrics_by_site.csv",
        "monthly_cod_metrics_by_site.csv",
        "monthly_ghi_metrics_by_group.png",
        "monthly_cod_by_group.png",
        "monthly_site_ghi_rmse_heatmap.png",
        "monthly_site_cod_median_heatmap.png",
        "monthly_seasonal_findings.md",
        "site_seasonal_findings.md",
    ]:
        print(f"- {OUT_DIR / name}")


if __name__ == "__main__":
    main()
