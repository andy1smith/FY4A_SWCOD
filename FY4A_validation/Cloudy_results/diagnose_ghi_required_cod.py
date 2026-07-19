"""Diagnose whether FY4A retrieved COD can explain measured cloudy GHI.

This script inverts the GHI-only cloudy downwelling interpolation surrogate:
for each validation time it keeps the FY4A non-COD features fixed and finds
the COD that would make modeled GHI closest to the measured ground GHI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from predict_ghi_cloudy_dw_from_adm_cod import DEFAULT_MODEL_PATH, apply_albedo_pc1, source_albedo


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COD_DIR = SCRIPT_DIR
DEFAULT_OUT_DIR = SCRIPT_DIR


def make_ghi_interpolator(bundle: dict) -> tuple[RegularGridInterpolator, list[str], dict[str, np.ndarray]]:
    spec = bundle["models"]["GHI"]
    features = list(spec["features"])
    axes = {feature: np.asarray(spec["axes"][feature], dtype=float) for feature in features}
    interpolator = RegularGridInterpolator(
        tuple(axes[feature] for feature in features),
        np.asarray(spec["values"], dtype=float),
        bounds_error=False,
        fill_value=np.nan,
    )
    return interpolator, features, axes


def clipped_non_cod_features(ds_cod: xr.Dataset, ds_source: xr.Dataset, bundle: dict, axes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cod = np.asarray(ds_cod["Retrieved_COD"].values, dtype=float)
    time_len, y_len, x_len = cod.shape
    n_points = time_len * y_len * x_len

    cos_th0 = np.cos(np.deg2rad(np.asarray(ds_source["Sun_Zen"].values, dtype=float))).reshape(n_points)
    tpw = np.asarray(ds_cod["tpw"].values, dtype=float)
    alb_pc1 = apply_albedo_pc1(source_albedo(ds_source, bundle), bundle["albedo_transform"])

    values = {
        "cos_th0": cos_th0,
        "sqrt_tpw": np.sqrt(np.clip(np.repeat(tpw, y_len * x_len), 0.0, None)),
        "alb_PC1_score": np.repeat(alb_pc1, y_len * x_len),
    }
    for feature, raw in list(values.items()):
        axis = axes[feature]
        values[feature] = np.clip(raw, axis[0], axis[-1])
    return values


def predict_for_cod_sweep(
    interpolator: RegularGridInterpolator,
    features: list[str],
    non_cod: dict[str, np.ndarray],
    point_idx: np.ndarray,
    sqrt_cod_grid: np.ndarray,
) -> np.ndarray:
    pred = np.full((len(sqrt_cod_grid), len(point_idx)), np.nan, dtype=float)
    for i, sqrt_cod in enumerate(sqrt_cod_grid):
        cols = []
        for feature in features:
            if feature == "sqrt_COD":
                cols.append(np.full(len(point_idx), sqrt_cod, dtype=float))
            else:
                cols.append(non_cod[feature][point_idx])
        points = np.column_stack(cols)
        valid = np.isfinite(points).all(axis=1)
        if valid.any():
            pred[i, valid] = interpolator(points[valid])
    return pred


def nearest_required_cod(pred_by_cod: np.ndarray, target: np.ndarray, cod_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diff = np.abs(pred_by_cod - target[None, :])
    finite = np.isfinite(diff).any(axis=0) & np.isfinite(target)
    required = np.full(len(target), np.nan, dtype=float)
    matched = np.full(len(target), np.nan, dtype=float)
    at_upper = np.zeros(len(target), dtype=bool)
    if finite.any():
        best = np.nanargmin(diff[:, finite], axis=0)
        required[finite] = cod_grid[best]
        matched[finite] = pred_by_cod[best, np.nonzero(finite)[0]]
        at_upper[finite] = best == len(cod_grid) - 1
    return required, matched, at_upper


def diagnose_site(cod_path: Path, bundle: dict, interpolator: RegularGridInterpolator, features: list[str], axes: dict[str, np.ndarray], cod_grid: np.ndarray) -> pd.DataFrame:
    ds_cod = xr.open_dataset(cod_path).load()
    ds_source = xr.open_dataset(ds_cod.attrs["source_file"]).load()
    site = ds_cod.attrs.get("site", cod_path.name.split("_cloudy_COD_uw_ADM.nc")[0])

    cod = np.asarray(ds_cod["Retrieved_COD"].values, dtype=float)
    time_len, y_len, x_len = cod.shape
    cy, cx = y_len // 2, x_len // 2
    pixels_per_time = y_len * x_len
    center_idx = np.arange(time_len) * pixels_per_time + cy * x_len + cx
    center_3x3_offsets = np.array([(cy + dy) * x_len + (cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=int)
    center_3x3_idx = np.arange(time_len)[:, None] * pixels_per_time + center_3x3_offsets[None, :]

    sqrt_cod_grid = np.sqrt(cod_grid)
    non_cod = clipped_non_cod_features(ds_cod, ds_source, bundle, axes)
    target = np.asarray(ds_source["GHI"].values, dtype=float)
    ghi_clear = np.asarray(ds_source["GHI_clear"].values, dtype=float)

    center_pred_sweep = predict_for_cod_sweep(interpolator, features, non_cod, center_idx, sqrt_cod_grid)
    req_center, matched_center, upper_center = nearest_required_cod(center_pred_sweep, target, cod_grid)

    flat_3x3_idx = center_3x3_idx.reshape(-1)
    pred_3x3_by_pixel = predict_for_cod_sweep(interpolator, features, non_cod, flat_3x3_idx, sqrt_cod_grid)
    pred_3x3 = np.nanmean(pred_3x3_by_pixel.reshape(len(cod_grid), time_len, 9), axis=2)
    req_3x3, matched_3x3, upper_3x3 = nearest_required_cod(pred_3x3, target, cod_grid)

    rows = pd.DataFrame(
        {
            "site": site,
            "time": pd.to_datetime(ds_cod["time"].values),
            "GHI_ground": target,
            "GHI_clear": ghi_clear,
            "ground_clearness": target / ghi_clear,
            "COD_center_retrieved": cod[:, cy, cx],
            "COD_center_3x3_retrieved": np.nanmean(cod[:, cy - 1 : cy + 2, cx - 1 : cx + 2], axis=(1, 2)),
            "COD_center_required_for_GHI": req_center,
            "COD_center_3x3_required_for_GHI": req_3x3,
            "GHI_center_at_required_COD": matched_center,
            "GHI_center_3x3_at_required_COD": matched_3x3,
            "required_center_at_upper_grid": upper_center,
            "required_3x3_at_upper_grid": upper_3x3,
        }
    )

    ds_cod.close()
    ds_source.close()
    return rows


def metrics(df: pd.DataFrame, retrieved_col: str, required_col: str) -> dict[str, float]:
    valid = np.isfinite(df[retrieved_col]) & np.isfinite(df[required_col])
    out: dict[str, float] = {"N": int(valid.sum())}
    if not valid.any():
        return out
    retrieved = df.loc[valid, retrieved_col].to_numpy(dtype=float)
    required = df.loc[valid, required_col].to_numpy(dtype=float)
    out.update(
        {
            "retrieved_mean": float(np.mean(retrieved)),
            "required_mean": float(np.mean(required)),
            "retrieved_median": float(np.median(retrieved)),
            "required_median": float(np.median(required)),
            "required_minus_retrieved_mean": float(np.mean(required - retrieved)),
            "required_over_retrieved_median": float(np.nanmedian(required / np.clip(retrieved, 1e-6, None))),
            "required_at_upper_fraction": float(np.mean(required >= 49.999)),
        }
    )
    return out


def plot_required_vs_retrieved(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True, sharey=True)
    pairs = [
        ("COD_center_retrieved", "COD_center_required_for_GHI", "Center pixel"),
        ("COD_center_3x3_retrieved", "COD_center_3x3_required_for_GHI", "Center 3x3 mean"),
    ]
    for ax, (retrieved_col, required_col, title) in zip(axes, pairs):
        valid = np.isfinite(df[retrieved_col]) & np.isfinite(df[required_col])
        ax.scatter(df.loc[valid, retrieved_col], df.loc[valid, required_col], s=10, alpha=0.22, color="#2F6F9F", edgecolors="none")
        ax.plot([0, 50], [0, 50], color="#B23A48", linestyle="--", lw=1.1)
        ax.set_title(title)
        ax.set_xlabel("ADM retrieved COD")
        ax.grid(True, linestyle=":", alpha=0.45)
    axes[0].set_ylabel("COD required by GHI model")
    axes[0].set_xlim(0, 51)
    axes[0].set_ylim(0, 51)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(cod_dir: Path, out_dir: Path, model_path: Path, sites: list[str] | None = None) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    interpolator, features, axes = make_ghi_interpolator(bundle)
    if "sqrt_COD" not in axes:
        raise ValueError("GHI model does not expose sqrt_COD axis.")

    cod_grid = np.linspace(float(axes["sqrt_COD"][0]) ** 2, float(axes["sqrt_COD"][-1]) ** 2, 500)
    cod_files = sorted(cod_dir.glob("*_cloudy_COD_uw_ADM.nc"))
    if sites:
        wanted = {site.upper() for site in sites}
        cod_files = [path for path in cod_files if path.name.split("_cloudy_COD_uw_ADM.nc")[0].upper() in wanted]
    if not cod_files:
        raise FileNotFoundError(f"No *_cloudy_COD_uw_ADM.nc files found in {cod_dir}")

    rows = []
    for idx, cod_path in enumerate(cod_files, start=1):
        print(f"[{idx}/{len(cod_files)}] {cod_path.name}")
        rows.append(diagnose_site(cod_path, bundle, interpolator, features, axes, cod_grid))
    df = pd.concat(rows, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ghi_required_cod_diagnostic.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    summary = pd.DataFrame(
        [
            {"case": "center", **metrics(df, "COD_center_retrieved", "COD_center_required_for_GHI")},
            {"case": "center_3x3", **metrics(df, "COD_center_3x3_retrieved", "COD_center_3x3_required_for_GHI")},
        ]
    )
    summary_path = out_dir / "ghi_required_cod_diagnostic_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    plot_path = out_dir / "GHI_required_COD_vs_ADM_retrieved_COD.png"
    plot_required_vs_retrieved(df, plot_path)
    print(f"Saved: {plot_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invert FY4A cloudy GHI surrogate to diagnose COD mismatch.")
    parser.add_argument("--cod-dir", default=str(DEFAULT_COD_DIR), help="Directory with *_cloudy_COD_uw_ADM.nc files.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Named cloudy DW GHI interpolation pickle.")
    parser.add_argument("--site", action="append", help="Optional site code. Repeat for multiple sites.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.cod_dir), Path(args.out_dir), Path(args.model), args.site)


if __name__ == "__main__":
    main()
