"""
Plot correlation diagnostics for the cloudy HG COD surrogate dataset.

Default input:
    Surrogate_GRP_COD/Cloudy_uw_model_HG/preprocessed_cloudy_uw_HG.csv

Outputs:
    correlation_matrix_cloudy_uw_HG.png
    cod_feature_correlation_cloudy_uw_HG.png
    cod_transform_channel_correlation_cloudy_uw_HG.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "preprocessed_cloudy_uw_HG.csv"
CHANNELS = ["C01", "C02", "C03", "C05", "C06"]
COD_TRANSFORMS = [
    "COD",
    "exp_neg_COD",
    "one_minus_exp_neg_COD",
    "log1p_COD",
    "sqrt_COD",
    "inv_1p_COD",
    "exp_neg_sqrt_COD",
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
    return out


def select_matrix_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        *COD_TRANSFORMS,
        "th0",
        "cos_th0",
        "Ts",
        "rh",
        "tpw",
        "sqrt_tpw",
        "AOD",
        "AlbSet",
        *[f"alb_{channel}" for channel in CHANNELS],
        *CHANNELS,
    ]
    return [col for col in candidates if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def display_name_map(df: pd.DataFrame, columns: list[str]) -> dict[str, str]:
    names = {}
    for col in columns:
        if df[col].nunique(dropna=True) <= 1:
            names[col] = f"{col} (constant)"
        else:
            names[col] = col
    return names


def plot_matrix(df: pd.DataFrame, columns: list[str], out_path: Path) -> None:
    corr = df[columns].corr()
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
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Cloudy HG LUT Pearson Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_cod_bar(df: pd.DataFrame, columns: list[str], out_path: Path) -> None:
    target_col = "log1p_COD"
    corr = df[columns].corr(numeric_only=True)[target_col].drop(labels=[target_col], errors="ignore")
    names = display_name_map(df, corr.index.tolist())
    corr = corr.rename(index=names).fillna(0.0)
    corr = corr.reindex(corr.abs().sort_values(ascending=True).index)

    plt.figure(figsize=(7.5, max(4.5, 0.28 * len(corr))))
    colors = ["#b2182b" if value < 0 else "#2166ac" for value in corr.values]
    plt.barh(corr.index, corr.values, color=colors, alpha=0.85)
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel("Pearson r with log1p(COD)")
    plt.title("Feature Correlation with log1p(COD)", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_cod_transform_channel_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    cod_columns = [
        col for col in COD_TRANSFORMS if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    channel_columns = [
        col for col in CHANNELS if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not cod_columns or not channel_columns:
        return

    corr = df[cod_columns + channel_columns].corr(numeric_only=True).loc[cod_columns, channel_columns]
    corr = corr.fillna(0.0)
    plt.figure(figsize=(8.5, max(4.2, 0.45 * len(cod_columns))))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    plt.title("COD Transform Correlation with Normalized GOES Channels", fontsize=13, fontweight="bold")
    plt.xlabel("Normalized channel")
    plt.ylabel("COD form")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cloudy HG COD surrogate correlations.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Preprocessed CSV path.")
    parser.add_argument(
        "--matrix-out",
        default=str(SCRIPT_DIR / "correlation_matrix_cloudy_uw_HG.png"),
        help="Output heatmap path.",
    )
    parser.add_argument(
        "--cod-out",
        default=str(SCRIPT_DIR / "cod_feature_correlation_cloudy_uw_HG.png"),
        help="Output transformed-COD correlation bar plot path.",
    )
    parser.add_argument(
        "--cod-transform-out",
        default=str(SCRIPT_DIR / "cod_transform_channel_correlation_cloudy_uw_HG.png"),
        help="Output COD-transform versus channel correlation heatmap path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = add_feature_transforms(df)
    columns = select_matrix_columns(df)
    if "log1p_COD" not in columns:
        raise ValueError("CSV must contain numeric COD column.")

    print(f"Rows: {len(df)}")
    print(f"Correlation columns: {columns}")
    plot_matrix(df, columns, Path(args.matrix_out))
    plot_cod_bar(df, columns, Path(args.cod_out))
    plot_cod_transform_channel_heatmap(df, Path(args.cod_transform_out))


if __name__ == "__main__":
    main()
