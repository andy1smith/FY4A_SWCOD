"""
Train cloudy HG GOES-channel surrogate V1.

This is the forward emulator:

    input per channel: log1p(COD), tpw, cos(th0), alb_Cxx
    output per channel: Cxx

Each GOES channel is trained independently. For example:

    C01 model features: log1p_COD, tpw, cos_th0, alb_C01
    C05 model features: log1p_COD, tpw, cos_th0, alb_C05

This preserves the physical channel pairing between surface albedo and the
corresponding upwelling channel instead of compressing all albedos into one
shared PCA coordinate.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from scipy.interpolate import RegularGridInterpolator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "preprocessed_cloudy_uw_HG.csv"
CHANNELS = ["C01", "C02", "C03", "C05", "C06"]
BASE_FEATURES = ["log1p_COD", "tpw", "cos_th0"]
MODEL_NAME = "SWRTM_cloudy_uw_channel_HG_V1.pkl"
INTERP_MODEL_NAME = "SWRTM_cloudy_uw_channel_HG_interp_V1.pkl"
RANDOM_STATE = 42


def channel_features(channel: str) -> list[str]:
    return [*BASE_FEATURES, f"alb_{channel}"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "COD" not in out.columns:
        raise ValueError("CSV must contain COD column.")
    if "cos_th0" not in out.columns:
        if "th0" not in out.columns:
            raise ValueError("CSV must contain either cos_th0 or th0.")
        out["cos_th0"] = np.cos(np.deg2rad(out["th0"].astype(float)))
    out["log1p_COD"] = np.log1p(out["COD"].clip(lower=0).astype(float))
    return out


def build_model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.04,
            l2_regularization=1e-6,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    rel = (y_pred - y_true) / (np.abs(y_true) + 1e-8)
    rrmse = np.sqrt(np.mean(rel**2, axis=0)) * 100.0
    metrics = pd.DataFrame(
        {
            "RMSE": rmse,
            "MAE": mae,
            "rRMSE_percent": rrmse,
            "R2": r2,
        },
        index=CHANNELS,
    )
    metrics.loc["AVG"] = metrics.mean(numeric_only=True)
    print(metrics.round(6).to_string())
    return metrics


def _axis_values(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.array(sorted(df[column].dropna().astype(float).unique()), dtype=float)


def build_interpolator_bundle(df: pd.DataFrame) -> dict:
    """
    Build a continuous forward surrogate directly from the full-factorial LUT.

    The HistGradientBoosting model fits the sampled LUT values very well, but it
    is piecewise-constant in COD. The retrieval problem needs a continuous
    forward curve, so this bundle stores the RTM LUT as regular grids and uses
    linear interpolation in log1p(COD), TPW, solar angle, and channel albedo.
    """
    grid_df = df.copy()
    grid_df["tpw_grid"] = grid_df["tpw"].astype(float).round(10)
    grid_df["log1p_COD"] = np.log1p(grid_df["COD"].clip(lower=0).astype(float))

    models: dict[str, dict[str, object]] = {}
    for channel in CHANNELS:
        alb_feature = f"alb_{channel}"
        features = ["log1p_COD", "tpw_grid", "cos_th0", alb_feature]
        axis_map = {feature: _axis_values(grid_df, feature) for feature in features}
        values = np.full(tuple(len(axis_map[feature]) for feature in features), np.nan, dtype=float)
        index_maps = {
            feature: {float(value): idx for idx, value in enumerate(axis_map[feature])}
            for feature in features
        }

        for _, row in grid_df.iterrows():
            index = tuple(index_maps[feature][float(row[feature])] for feature in features)
            values[index] = float(row[channel])

        if np.isnan(values).any():
            raise ValueError(f"Incomplete regular grid for {channel}; cannot build interpolator.")

        models[channel] = {
            "kind": "regular_grid_interpolator",
            "features": ["log1p_COD", "tpw", "cos_th0", alb_feature],
            "grid_features": features,
            "axes": {feature: axis_map[feature] for feature in features},
            "values": values,
            "bounds": {
                feature.replace("tpw_grid", "tpw"): (
                    float(axis_map[feature][0]),
                    float(axis_map[feature][-1]),
                )
                for feature in features
            },
        }

    return {
        "models": models,
        "feature_map": {
            channel: ["log1p_COD", "tpw", "cos_th0", f"alb_{channel}"]
            for channel in CHANNELS
        },
        "target_columns": CHANNELS,
        "metadata": {
            "description": "Cloudy HG forward surrogate: regular-grid interpolation of the RTM LUT",
            "model_type": "RegularGridInterpolator",
            "input_columns_raw": ["COD", "tpw", "cos_th0", *[f"alb_{channel}" for channel in CHANNELS]],
            "derived_features": {
                "log1p_COD": "np.log1p(COD)",
            },
            "target_columns": CHANNELS,
            "n_samples": len(df),
            "interpolation": "linear on log1p(COD), TPW, cos(th0), and same-channel albedo",
            "clipping": "prediction inputs are clipped to the training-grid bounds",
        },
    }


def evaluate_interpolator_bundle(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    feature_df = df.copy()
    feature_df["log1p_COD"] = np.log1p(feature_df["COD"].clip(lower=0).astype(float))
    predictions = []
    truth = []

    for channel in CHANNELS:
        spec = bundle["models"][channel]
        grid_features = spec["grid_features"]
        axes = tuple(np.asarray(spec["axes"][feature], dtype=float) for feature in grid_features)
        interpolator = RegularGridInterpolator(axes, np.asarray(spec["values"], dtype=float))

        point_columns = []
        for feature in spec["features"]:
            source = "tpw_grid" if feature == "tpw" else feature
            values = feature_df[feature].astype(float).to_numpy()
            axis = np.asarray(spec["axes"][source], dtype=float)
            point_columns.append(np.clip(values, axis[0], axis[-1]))
        predictions.append(interpolator(np.column_stack(point_columns)))
        truth.append(feature_df[channel].astype(float).to_numpy())

    return evaluate(np.column_stack(truth), np.column_stack(predictions))


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(3.4 * len(CHANNELS), 3.4))
    for idx, (ax, channel) in enumerate(zip(axes, CHANNELS)):
        truth = y_true[:, idx]
        pred = y_pred[:, idx]
        vmin = min(float(np.min(truth)), float(np.min(pred)))
        vmax = max(float(np.max(truth)), float(np.max(pred)))
        ax.scatter(truth, pred, s=10, alpha=0.3, color="steelblue")
        ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.2)
        ax.set_title(channel)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, linestyle=":", alpha=0.45)
    fig.suptitle("Cloudy HG Independent Channel Surrogates V1", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def collect_feature_importance(
    models: dict[str, object],
    feature_map: dict[str, list[str]],
    df_eval: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for channel, model in models.items():
        features = feature_map[channel]
        result = permutation_importance(
            model,
            df_eval[features],
            df_eval[channel].values,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="neg_root_mean_squared_error",
        )
        for feature, importance in zip(features, result.importances_mean):
            rows.append({"channel": channel, "feature": feature, "importance": importance})
    return pd.DataFrame(rows)


def plot_feature_importance(importance_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(3.2 * len(CHANNELS), 3.6), sharex=True)
    for ax, channel in zip(axes, CHANNELS):
        sub = importance_df[importance_df["channel"] == channel].sort_values("importance")
        ax.barh(sub["feature"], sub["importance"], color="seagreen", alpha=0.85)
        ax.set_title(channel)
        ax.grid(axis="x", linestyle=":", alpha=0.45)
    fig.suptitle("Per-Channel Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def validate_columns(df: pd.DataFrame) -> None:
    required = set(BASE_FEATURES + CHANNELS + [f"alb_{channel}" for channel in CHANNELS])
    missing = [col for col in sorted(required) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cloudy HG independent GOES-channel surrogates V1.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Preprocessed CSV path.")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR), help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Hold-out fraction.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.csv)
    df = add_features(df)
    validate_columns(df)
    feature_map = {channel: channel_features(channel) for channel in CHANNELS}
    all_used_columns = sorted(set(CHANNELS + [feature for features in feature_map.values() for feature in features]))
    df = df.dropna(subset=all_used_columns).copy()

    print(f"Rows: {len(df)}")
    print("Per-channel features:")
    for channel in CHANNELS:
        print(f"  {channel}: {feature_map[channel]}")

    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=args.test_size, random_state=RANDOM_STATE)
    print(f"Train size: {len(idx_train)} | Test size: {len(idx_test)}")

    models: dict[str, object] = {}
    y_pred_channels = []
    y_test_channels = []

    for channel in CHANNELS:
        print(f"\n=== Training {channel} model ===")
        features = feature_map[channel]
        model = build_model()
        t0 = time.time()
        model.fit(df.iloc[idx_train][features], df.iloc[idx_train][channel].values)
        print(f"  Training time: {time.time() - t0:.1f}s")
        models[channel] = model
        y_pred_channels.append(model.predict(df.iloc[idx_test][features]))
        y_test_channels.append(df.iloc[idx_test][channel].values)

    y_pred = np.column_stack(y_pred_channels)
    y_test = np.column_stack(y_test_channels)
    metrics = evaluate(y_test, y_pred)

    metrics.to_csv(out_dir / "metrics_channel_HG_V1.csv")
    importance_df = collect_feature_importance(models, feature_map, df.iloc[idx_test])
    importance_df.to_csv(out_dir / "feature_importance_channel_HG_V1.csv", index=False)
    plot_scatter(y_test, y_pred, out_dir / "validation_scatter_channels_HG_V1.png")
    plot_feature_importance(importance_df, out_dir / "feature_importance_channels_HG_V1.png")

    print("\nRefitting final per-channel models on full dataset...")
    final_models = {}
    for channel in CHANNELS:
        features = feature_map[channel]
        model = build_model()
        model.fit(df[features], df[channel].values)
        final_models[channel] = model

    bundle = {
        "models": final_models,
        "feature_map": feature_map,
        "target_columns": CHANNELS,
        "metadata": {
            "description": "Cloudy HG forward surrogate: independent channel models using log1p(COD), TPW, cos(th0), and same-channel albedo",
            "model_type": "HistGradientBoostingRegressor",
            "input_columns_raw": ["COD", "tpw", "cos_th0", *[f"alb_{channel}" for channel in CHANNELS]],
            "derived_features": {
                "log1p_COD": "np.log1p(COD)",
            },
            "target_columns": CHANNELS,
            "n_samples": len(df),
            "holdout_metrics": metrics.to_dict(orient="index"),
            "random_state": RANDOM_STATE,
        },
    }
    save_path = out_dir / MODEL_NAME
    joblib.dump(bundle, save_path, compress=3)
    print(f"Saved model: {save_path}")

    print("\nBuilding regular-grid interpolation surrogate...")
    interp_bundle = build_interpolator_bundle(df)
    interp_metrics = evaluate_interpolator_bundle(df, interp_bundle)
    interp_metrics.to_csv(out_dir / "metrics_channel_HG_interp_V1.csv")
    interp_save_path = out_dir / INTERP_MODEL_NAME
    joblib.dump(interp_bundle, interp_save_path, compress=3)
    print(f"Saved interpolation model: {interp_save_path}")


if __name__ == "__main__":
    main()
