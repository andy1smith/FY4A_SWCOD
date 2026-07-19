"""Predict FY4A cloudy GHI from retrieved COD maps and plot site validation.

This GHI-only driver uses one explicit downwelling surrogate bundle:

    Surrogate_GRP_COD/Cloudy_dw_HG/
    SWRTM_cloudy_dw_HG_GHI_DNI_PC1_interp_V1.pkl

The alias ``model.pkl`` has the same content in the current checkout, but the
named interpolation bundle is used here to keep provenance unambiguous.
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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_COD_DIR = SCRIPT_DIR / "Cloudy_COD_results"
DEFAULT_OUT_DIR = SCRIPT_DIR / "Cloudy_dw_HG"
DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "Surrogate_GRP_COD"
    / "Cloudy_dw_HG"
    / "SWRTM_cloudy_dw_HG_GHI_DNI_PC1_interp_V1.pkl"
)


def apply_albedo_pc1(albedo_df: pd.DataFrame, transform: dict) -> np.ndarray:
    albedo = albedo_df[transform["columns"]].astype(float).to_numpy()
    mean = np.asarray(transform["mean"], dtype=float)
    scale = np.asarray(transform["scale"], dtype=float)
    pc1_vector = np.asarray(transform["pc1_vector"], dtype=float)
    return ((albedo - mean) / scale) @ pc1_vector


def source_albedo(ds_source: xr.Dataset, bundle: dict) -> pd.DataFrame:
    data = {}
    transform = bundle["albedo_transform"]
    for alb_col in transform["columns"]:
        channel = alb_col.replace("alb_", "")
        for name in (f"WSA_{channel}", f"BSA_{channel}", alb_col):
            if name in ds_source:
                data[alb_col] = np.asarray(ds_source[name].values, dtype=float)
                break
        else:
            idx = transform["columns"].index(alb_col)
            data[alb_col] = np.full(ds_source.sizes["time"], float(transform["mean"][idx]))
    return pd.DataFrame(data)


def build_ghi_features(ds_cod: xr.Dataset, ds_source: xr.Dataset, bundle: dict) -> pd.DataFrame:
    cod = np.asarray(ds_cod["Retrieved_COD"].values, dtype=float)
    time_len, y_len, x_len = cod.shape
    n_points = time_len * y_len * x_len

    cos_th0 = np.cos(np.deg2rad(np.asarray(ds_source["Sun_Zen"].values, dtype=float)))
    tpw = np.asarray(ds_cod["tpw"].values, dtype=float)
    alb_pc1 = apply_albedo_pc1(source_albedo(ds_source, bundle), bundle["albedo_transform"])

    return pd.DataFrame(
        {
            "sqrt_COD": np.sqrt(np.clip(cod.reshape(n_points), 0.0, None)),
            "cos_th0": cos_th0.reshape(n_points),
            "sqrt_tpw": np.sqrt(np.clip(np.repeat(tpw, y_len * x_len), 0.0, None)),
            "alb_PC1_score": np.repeat(alb_pc1, y_len * x_len),
        }
    )


def predict_ghi(bundle: dict, feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    spec = bundle["models"]["GHI"]
    features = list(spec["features"])
    axes = tuple(np.asarray(spec["axes"][feature], dtype=float) for feature in features)
    interpolator = RegularGridInterpolator(
        axes,
        np.asarray(spec["values"], dtype=float),
        bounds_error=False,
        fill_value=np.nan,
    )

    points = []
    clipped_any = np.zeros(len(feature_df), dtype=bool)
    for feature in features:
        raw = feature_df[feature].to_numpy(dtype=float)
        axis = np.asarray(spec["axes"][feature], dtype=float)
        clipped = np.clip(raw, axis[0], axis[-1])
        clipped_any |= np.isfinite(raw) & (raw != clipped)
        points.append(clipped)

    pred = np.full(len(feature_df), np.nan, dtype=np.float32)
    valid = np.isfinite(np.column_stack(points)).all(axis=1)
    if valid.any():
        pred[valid] = interpolator(np.column_stack(points)[valid]).astype(np.float32)
    return pred, clipped_any


def predict_ghi_for_site(cod_path: Path, bundle: dict, out_dir: Path) -> tuple[pd.DataFrame, Path]:
    ds_cod = xr.open_dataset(cod_path).load()
    source_path = Path(ds_cod.attrs["source_file"])
    ds_source = xr.open_dataset(source_path).load()
    site = ds_cod.attrs.get("site", cod_path.name.split("_cloudy_COD_uw_ADM.nc")[0])

    cod = np.asarray(ds_cod["Retrieved_COD"].values, dtype=float)
    time_len, y_len, x_len = cod.shape
    feature_df = build_ghi_features(ds_cod, ds_source, bundle)
    ghi_flat, clipped_any = predict_ghi(bundle, feature_df)
    ghi = ghi_flat.reshape(time_len, y_len, x_len)
    clipped = clipped_any.reshape(time_len, y_len, x_len)

    out = xr.Dataset(
        {
            "GHI_pred": (("time", "y", "x"), ghi),
            "GHI_ground": (("time",), np.asarray(ds_source["GHI"].values, dtype=np.float32)),
            "GHI_clear": (("time",), np.asarray(ds_source["GHI_clear"].values, dtype=np.float32)),
            "Retrieved_COD": (("time", "y", "x"), cod.astype(np.float32)),
            "GHI_feature_clipped": (("time", "y", "x"), clipped),
        },
        coords={"time": ds_cod["time"].values, "y": ds_cod["y"].values, "x": ds_cod["x"].values},
        attrs={
            "site": site,
            "cod_file": str(cod_path),
            "source_file": str(source_path),
            "model_file": str(bundle.get("_model_path", "")),
            "prediction": bundle.get("metadata", {}).get(
                "description",
                "GHI from cloudy downwelling interpolation surrogate using retrieved COD",
            ),
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{site}_cloudy_GHI_dw_surr.nc"
    encoding = {name: {"zlib": True, "complevel": 4} for name in out.data_vars}
    try:
        out.to_netcdf(out_path, encoding=encoding)
    except ValueError:
        out.to_netcdf(out_path)

    cy, cx = y_len // 2, x_len // 2
    center = ghi[:, cy, cx]
    center_3x3 = np.nanmean(ghi[:, cy - 1 : cy + 2, cx - 1 : cx + 2], axis=(1, 2))
    rows = pd.DataFrame(
        {
            "site": site,
            "time": pd.to_datetime(ds_cod["time"].values),
            "GHI_ground": np.asarray(ds_source["GHI"].values, dtype=float),
            "GHI_clear": np.asarray(ds_source["GHI_clear"].values, dtype=float),
            "GHI_center": center.astype(float),
            "GHI_center_3x3_mean": center_3x3.astype(float),
            "COD_center": cod[:, cy, cx].astype(float),
            "COD_center_3x3_mean": np.nanmean(cod[:, cy - 1 : cy + 2, cx - 1 : cx + 2], axis=(1, 2)).astype(float),
            "GHI_center_feature_clipped": clipped[:, cy, cx].astype(bool),
        }
    )
    ds_cod.close()
    ds_source.close()
    return rows, out_path


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if not valid.any():
        return {"N": 0, "MBE": np.nan, "RMSE": np.nan, "R": np.nan}
    diff = pred[valid] - obs[valid]
    r = np.corrcoef(obs[valid], pred[valid])[0, 1] if valid.sum() > 1 else np.nan
    return {
        "N": int(valid.sum()),
        "MBE": float(np.mean(diff)),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "R": float(r),
    }


def plot_scatter(df: pd.DataFrame, pred_col: str, out_path: Path, title: str) -> dict:
    obs = df["GHI_ground"].to_numpy(dtype=float)
    pred = df[pred_col].to_numpy(dtype=float)
    stat = metrics(obs, pred)
    valid = np.isfinite(obs) & np.isfinite(pred)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(obs[valid], pred[valid], s=12, alpha=0.28, color="#2F6F9F", edgecolors="none")
    lim_max = float(np.nanmax([obs[valid].max(), pred[valid].max(), 1000.0])) if valid.any() else 1000.0
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stat


def run(cod_dir: Path, out_dir: Path, model_path: Path, sites: list[str] | None = None) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    if "GHI" not in bundle.get("models", {}):
        raise ValueError(f"Model bundle does not contain a GHI model: {model_path}")
    bundle["_model_path"] = str(model_path)

    cod_files = sorted(cod_dir.glob("*_cloudy_COD_uw_ADM.nc"))
    if sites:
        wanted = {site.upper() for site in sites}
        cod_files = [path for path in cod_files if path.name.split("_cloudy_COD_uw_ADM.nc")[0].upper() in wanted]
    if not cod_files:
        raise FileNotFoundError(f"No *_cloudy_COD_uw_ADM.nc files found in {cod_dir}")

    all_rows = []
    for idx, cod_path in enumerate(cod_files, start=1):
        print(f"[{idx}/{len(cod_files)}] {cod_path.name}")
        rows, out_path = predict_ghi_for_site(cod_path, bundle, out_dir)
        all_rows.append(rows)
        print(f"  saved {out_path.name}; rows={len(rows)}")

    df = pd.concat(all_rows, ignore_index=True)
    csv_path = out_dir / "cloudy_ghi_dw_surrogate_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    summary_rows = []
    for pred_col, out_name, title in [
        ("GHI_center", "GHI_center_pixel_vs_ground.png", "FY4A Cloudy GHI: Center Pixel"),
        ("GHI_center_3x3_mean", "GHI_center_3x3_mean_vs_ground.png", "FY4A Cloudy GHI: Center 3x3 Mean"),
    ]:
        stat = plot_scatter(df, pred_col, out_dir / out_name, title)
        stat["prediction"] = pred_col
        summary_rows.append(stat)
        print(f"Saved: {out_dir / out_name}")

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "cloudy_ghi_dw_surrogate_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict cloudy FY4A GHI from ADM COD maps.")
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
