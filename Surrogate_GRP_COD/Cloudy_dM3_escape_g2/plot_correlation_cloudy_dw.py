"""
Plot correlation diagnostics for the cloudy dM f-escape downwelling LUT.

Default input:
    Surrogate_GRP_COD/Cloudy_dw_model_dMescape/cloudy_dw_dMescape_LUT.csv

Outputs:
    correlation_matrix_cloudy_dw_dMescape.png
    param_flux_correlation_cloudy_dw_dMescape.png
    param_flux_correlation_cloudy_dw_dMescape.csv
    albedo_sensitivity_cloudy_dw_dMescape.png
    albedo_sensitivity_cloudy_dw_dMescape.csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "cloudy_dw_dMescape_LUT.csv"
FLUX_COLUMNS = ["GHI", "DNI", "DHI"]
ALBEDO_COLUMNS = ["alb_C01", "alb_C02", "alb_C03", "alb_C05", "alb_C06"]
ALBEDO_SOLAR_WEIGHTS = np.array(
    [
        75.08475711742334,
        133.90308595348597,
        33.64633739793085,
        11.032721590149361,
        3.529350607609547,
    ],
    dtype=float,
)
ALBEDO_SOLAR_WEIGHTS = ALBEDO_SOLAR_WEIGHTS / ALBEDO_SOLAR_WEIGHTS.sum()
COD_TRANSFORMS = [
    "COD",
    "exp_neg_COD",
    "one_minus_exp_neg_COD",
    "log1p_COD",
    "sqrt_COD",
    "inv_1p_COD",
    "exp_neg_sqrt_COD",
]
PARAMETER_COLUMNS = [
    *COD_TRANSFORMS,
    "th0",
    "cos_th0",
    "Ts",
    "RH",
    "rh",
    "tpw",
    "sqrt_tpw",
    "AOD",
    "AlbSet",
    "alb_mean",
    "alb_weighted_solar",
    "alb_vis_mean",
    "alb_nir_mean",
    "alb_nir_minus_vis",
    "alb_ratio_nir_vis",
    "alb_slope_C06_C01",
    "alb_PC1",
    *ALBEDO_COLUMNS,
]


def add_feature_transforms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "COD" in out.columns:
        cod = out["COD"].clip(lower=0).astype(float)
        exp_neg_cod = np.exp(-cod)
        out["exp_neg_COD"] = exp_neg_cod
        out["one_minus_exp_neg_COD"] = 1.0 - exp_neg_cod
        out["log1p_COD"] = np.log1p(cod)
        out["sqrt_COD"] = np.sqrt(cod)
        out["inv_1p_COD"] = 1.0 / (1.0 + cod)
        out["exp_neg_sqrt_COD"] = np.exp(-np.sqrt(cod))
    if "th0" in out.columns and "cos_th0" not in out.columns:
        out["cos_th0"] = np.cos(np.deg2rad(out["th0"].astype(float)))
    if "tpw" in out.columns:
        out["sqrt_tpw"] = np.sqrt(out["tpw"].clip(lower=0).astype(float))
    available_albedo = [col for col in ALBEDO_COLUMNS if col in out.columns]
    if len(available_albedo) == len(ALBEDO_COLUMNS):
        albedo = out[ALBEDO_COLUMNS].astype(float)
        out["alb_mean"] = albedo.mean(axis=1)
        out["alb_weighted_solar"] = albedo.to_numpy() @ ALBEDO_SOLAR_WEIGHTS
        out["alb_vis_mean"] = albedo[["alb_C01", "alb_C02", "alb_C03"]].mean(axis=1)
        out["alb_nir_mean"] = albedo[["alb_C05", "alb_C06"]].mean(axis=1)
        out["alb_nir_minus_vis"] = out["alb_nir_mean"] - out["alb_vis_mean"]
        out["alb_ratio_nir_vis"] = out["alb_nir_mean"] / out["alb_vis_mean"]
        out["alb_slope_C06_C01"] = albedo["alb_C06"] - albedo["alb_C01"]
        standardized = (albedo - albedo.mean(axis=0)) / albedo.std(axis=0, ddof=0)
        _, _, vh = np.linalg.svd(standardized.to_numpy(), full_matrices=False)
        pc1 = standardized.to_numpy() @ vh[0]
        if np.corrcoef(pc1, out["AlbSet"].astype(float))[0, 1] < 0:
            pc1 = -pc1
        out["alb_PC1"] = pc1
    return out


def numeric_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def display_name_map(df: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    names = {}
    for col in columns:
        if df[col].nunique(dropna=True) <= 1:
            names[col] = f"{col} (constant)"
        else:
            names[col] = col
    return names


def correlation_table(df: pd.DataFrame, params: list[str], fluxes: list[str]) -> pd.DataFrame:
    corr = df[params + fluxes].corr(numeric_only=True).loc[params, fluxes]
    return corr


def plot_full_matrix(df: pd.DataFrame, columns: list[str], out_path: Path) -> None:
    corr = df[columns].corr(numeric_only=True)
    names = display_name_map(df, columns)
    corr = corr.rename(index=names, columns=names)
    np.fill_diagonal(corr.values, 1.0)
    corr = corr.fillna(0.0)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(max(10, len(columns) * 0.55), max(8, len(columns) * 0.5)))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    plt.title("Cloudy dM f-escape DW LUT Pearson Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_param_flux_heatmap(df: pd.DataFrame, params: list[str], fluxes: list[str], out_path: Path) -> pd.DataFrame:
    corr = correlation_table(df, params, fluxes)
    names = display_name_map(df, params)
    corr_display = corr.rename(index=names).fillna(0.0)

    plt.figure(figsize=(7.2, max(5.0, 0.38 * len(corr_display))))
    sns.heatmap(
        corr_display,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    plt.title("Parameter Correlation with GHI, DNI, and DHI", fontsize=13, fontweight="bold")
    plt.xlabel("Integrated full-spectrum flux")
    plt.ylabel("Input parameter / transform")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return corr


def save_param_flux_csv(corr: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for param in corr.index:
        row = {"parameter": param}
        for flux in corr.columns:
            row[flux] = corr.loc[param, flux]
            row[f"abs_{flux}"] = abs(corr.loc[param, flux]) if pd.notna(corr.loc[param, flux]) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def paired_albedo_sensitivity(df: pd.DataFrame, fluxes: list[str]) -> pd.DataFrame:
    required = {"AlbSet", "COD", "th0", "Ts", "RH", *fluxes}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for paired albedo sensitivity: {sorted(missing)}")

    rows = []
    group_columns = ["COD", "th0", "Ts", "RH"]
    for _, group in df.groupby(group_columns, dropna=False):
        group = group.sort_values("AlbSet")
        if group["AlbSet"].nunique() < 2:
            continue
        low = group.iloc[0]
        high = group.iloc[-1]
        alb_delta = float(high["alb_weighted_solar"] - low["alb_weighted_solar"])
        albset_delta = float(high["AlbSet"] - low["AlbSet"])
        if alb_delta == 0 or albset_delta == 0:
            continue
        for flux in fluxes:
            flux_delta = float(high[flux] - low[flux])
            rows.append(
                {
                    "flux": flux,
                    "delta_flux": flux_delta,
                    "delta_flux_pct": 100.0 * flux_delta / float(low[flux]) if float(low[flux]) != 0 else np.nan,
                    "slope_per_weighted_albedo": flux_delta / alb_delta,
                    "slope_per_AlbSet": flux_delta / albset_delta,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_albedo_sensitivity(sensitivity: pd.DataFrame) -> pd.DataFrame:
    if sensitivity.empty:
        return sensitivity
    summary = (
        sensitivity.groupby("flux")
        .agg(
            mean_delta_flux=("delta_flux", "mean"),
            median_delta_flux=("delta_flux", "median"),
            mean_delta_flux_pct=("delta_flux_pct", "mean"),
            median_delta_flux_pct=("delta_flux_pct", "median"),
            mean_slope_per_weighted_albedo=("slope_per_weighted_albedo", "mean"),
            median_slope_per_weighted_albedo=("slope_per_weighted_albedo", "median"),
            n_pairs=("delta_flux", "size"),
        )
        .reset_index()
    )
    return summary


def plot_albedo_sensitivity(summary: pd.DataFrame, fluxes: list[str], out_path: Path) -> None:
    if summary.empty:
        print("No paired albedo sensitivity rows available.")
        return
    heatmap_data = summary.set_index("flux")[
        ["mean_delta_flux", "mean_delta_flux_pct", "mean_slope_per_weighted_albedo"]
    ].T.reindex(columns=fluxes)
    labels = {
        "mean_delta_flux": "Mean Δflux\nAlbSet 0→4",
        "mean_delta_flux_pct": "Mean Δflux %\nAlbSet 0→4",
        "mean_slope_per_weighted_albedo": "Mean slope\nW m⁻² / albedo",
    }
    heatmap_data = heatmap_data.rename(index=labels)

    plt.figure(figsize=(7.2, 3.8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Paired Broadband Albedo Sensitivity", fontsize=13, fontweight="bold")
    plt.xlabel("Integrated full-spectrum flux")
    plt.ylabel("AlbSet-paired metric")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cloudy dM f-escape DW LUT correlations.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="DW LUT CSV path.")
    parser.add_argument(
        "--matrix-out",
        default=str(SCRIPT_DIR / "correlation_matrix_cloudy_dw_dMescape.png"),
        help="Output full correlation heatmap path.",
    )
    parser.add_argument(
        "--param-flux-out",
        default=str(SCRIPT_DIR / "param_flux_correlation_cloudy_dw_dMescape.png"),
        help="Output parameter-vs-flux heatmap path.",
    )
    parser.add_argument(
        "--param-flux-csv",
        default=str(SCRIPT_DIR / "param_flux_correlation_cloudy_dw_dMescape.csv"),
        help="Output parameter-vs-flux correlation CSV path.",
    )
    parser.add_argument(
        "--albedo-sensitivity-out",
        default=str(SCRIPT_DIR / "albedo_sensitivity_cloudy_dw_dMescape.png"),
        help="Output paired albedo sensitivity heatmap path.",
    )
    parser.add_argument(
        "--albedo-sensitivity-csv",
        default=str(SCRIPT_DIR / "albedo_sensitivity_cloudy_dw_dMescape.csv"),
        help="Output paired albedo sensitivity summary CSV path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.drop(columns=["filename"], errors="ignore")
    df = add_feature_transforms(df)

    params = numeric_columns(df, PARAMETER_COLUMNS)
    fluxes = numeric_columns(df, FLUX_COLUMNS)
    if not fluxes:
        raise ValueError("CSV must contain numeric GHI, DNI, and/or DHI columns.")
    if not params:
        raise ValueError("No numeric parameter columns found for correlation.")

    matrix_columns = params + fluxes
    print(f"Rows: {len(df)}")
    print(f"Parameter columns: {params}")
    print(f"Flux columns: {fluxes}")
    constant_params = [col for col in params if df[col].nunique(dropna=True) <= 1]
    if constant_params:
        print(f"Constant parameters with undefined Pearson r: {constant_params}")

    plot_full_matrix(df, matrix_columns, Path(args.matrix_out))
    corr = plot_param_flux_heatmap(df, params, fluxes, Path(args.param_flux_out))
    save_param_flux_csv(corr, Path(args.param_flux_csv))
    sensitivity = paired_albedo_sensitivity(df, fluxes)
    sensitivity_summary = summarize_albedo_sensitivity(sensitivity)
    if not sensitivity_summary.empty:
        sensitivity_summary.to_csv(args.albedo_sensitivity_csv, index=False)
        print(f"Saved: {args.albedo_sensitivity_csv}")
    plot_albedo_sensitivity(sensitivity_summary, fluxes, Path(args.albedo_sensitivity_out))


if __name__ == "__main__":
    main()
