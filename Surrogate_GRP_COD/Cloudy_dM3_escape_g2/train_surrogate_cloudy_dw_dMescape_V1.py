"""
Train cloudy dM theta-truncation 3 g²-escape downwelling surrogates for GHI and DNI.

Targets:
    GHI: trained directly
    DNI: trained directly
    DHI: derived as GHI - DNI

Feature forms:
    GHI: sqrt_COD, cos_th0, sqrt_tpw, alb_PC1_score
    DNI: exp_neg_sqrt_COD, cos_th0, sqrt_tpw, alb_PC1_score

The albedo vector is compressed to a continuous PC1 score using the five
GOES-like broadband albedo channels. This matches the LUT design, where the
five AlbSet cases are percentile samples along the main full-year SURFRAD
albedo axis.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "cloudy_dw_dM3_escape_g2_LUT.csv"
RANDOM_STATE = 42
ALBEDO_COLUMNS = ["alb_C01", "alb_C02", "alb_C03", "alb_C05", "alb_C06"]
TARGET_SPECS = {
    "GHI": ["sqrt_COD", "cos_th0", "sqrt_tpw", "alb_PC1_score"],
    "DNI": ["exp_neg_sqrt_COD", "cos_th0", "sqrt_tpw", "alb_PC1_score"],
}
HGB_MODEL_NAME = "SWRTM_cloudy_dw_dM3_escape_g2_GHI_DNI_PC1_HGB_V1.pkl"
INTERP_MODEL_NAME = "SWRTM_cloudy_dw_dM3_escape_g2_GHI_DNI_PC1_interp_V1.pkl"
DEFAULT_MODEL_ALIAS = "model.pkl"
DEFAULT_OUTPUT_TAG = "cloudy_dw_dM3_escape_g2"
DEFAULT_MODEL_PREFIX = "SWRTM_cloudy_dw_dM3_escape_g2_GHI_DNI_PC1"
DEFAULT_TITLE = "Cloudy DW dM g2-escape"
DEFAULT_DESCRIPTION = "Cloudy dM theta-truncation 3 g2-escape DW surrogate"


def fit_albedo_pc1(df: pd.DataFrame) -> dict:
    unique = df[["AlbSet", *ALBEDO_COLUMNS]].drop_duplicates().sort_values("AlbSet")
    albedo = unique[ALBEDO_COLUMNS].astype(float)
    mean = albedo.mean(axis=0).to_numpy()
    scale = albedo.std(axis=0, ddof=0).to_numpy()
    if np.any(scale == 0):
        raise ValueError("At least one albedo channel is constant; cannot compute standardized PC1.")

    standardized = (albedo.to_numpy() - mean) / scale
    _, _, vh = np.linalg.svd(standardized, full_matrices=False)
    pc1_vector = vh[0]
    scores = standardized @ pc1_vector
    if np.corrcoef(scores, unique["AlbSet"].astype(float).to_numpy())[0, 1] < 0:
        pc1_vector = -pc1_vector
        scores = -scores

    return {
        "columns": ALBEDO_COLUMNS,
        "mean": mean,
        "scale": scale,
        "pc1_vector": pc1_vector,
        "explained_variance_ratio": float(np.var(scores, ddof=0) / standardized.var(axis=0, ddof=0).sum()),
        "albset_scores": dict(zip(unique["AlbSet"].astype(int).tolist(), scores.astype(float).tolist())),
    }


def apply_albedo_pc1(df: pd.DataFrame, transform: dict) -> np.ndarray:
    albedo = df[transform["columns"]].astype(float).to_numpy()
    standardized = (albedo - np.asarray(transform["mean"], dtype=float)) / np.asarray(transform["scale"], dtype=float)
    return standardized @ np.asarray(transform["pc1_vector"], dtype=float)


def add_features(df: pd.DataFrame, albedo_transform: dict | None = None) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    if albedo_transform is None:
        albedo_transform = fit_albedo_pc1(out)

    out["sqrt_COD"] = np.sqrt(out["COD"].clip(lower=0).astype(float))
    out["exp_neg_sqrt_COD"] = np.exp(-out["sqrt_COD"])
    if "cos_th0" not in out.columns:
        out["cos_th0"] = np.cos(np.deg2rad(out["th0"].astype(float)))
    out["sqrt_tpw"] = np.sqrt(out["tpw"].clip(lower=0).astype(float))
    out["alb_PC1_score"] = apply_albedo_pc1(out, albedo_transform)
    return out, albedo_transform


def validate_columns(df: pd.DataFrame) -> None:
    required = {"COD", "cos_th0", "tpw", "GHI", "DNI", "AlbSet", *ALBEDO_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_hgb_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=800,
        learning_rate=0.035,
        l2_regularization=1e-6,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    rrmse = np.sqrt(np.mean(((y_pred - y_true) / (np.abs(y_true) + 1e-8)) ** 2, axis=0)) * 100.0
    metrics = pd.DataFrame(
        {"RMSE": rmse, "MAE": mae, "rRMSE_percent": rrmse, "R2": r2},
        index=targets,
    )
    metrics.loc["AVG"] = metrics.mean(numeric_only=True)
    return metrics


def predict_hgb(models: dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    pred = {}
    for target, model in models.items():
        features = TARGET_SPECS[target]
        pred[target] = model.predict(df[features])
    pred_df = pd.DataFrame(pred, index=df.index)
    pred_df["DHI"] = pred_df["GHI"] - pred_df["DNI"]
    return pred_df


def plot_scatter(y_true: pd.DataFrame, y_pred: pd.DataFrame, out_path: Path, title: str) -> None:
    targets = ["GHI", "DNI", "DHI"]
    fig, axes = plt.subplots(1, len(targets), figsize=(10.2, 3.4))
    for ax, target in zip(axes, targets):
        truth = y_true[target].to_numpy(dtype=float)
        pred = y_pred[target].to_numpy(dtype=float)
        vmin = min(float(np.min(truth)), float(np.min(pred)))
        vmax = max(float(np.max(truth)), float(np.max(pred)))
        ax.scatter(truth, pred, s=12, alpha=0.35, color="#2474A6")
        ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.1)
        ax.set_title(target)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, linestyle=":", alpha=0.45)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def collect_importance(models: dict[str, object], df_eval: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, model in models.items():
        features = TARGET_SPECS[target]
        result = permutation_importance(
            model,
            df_eval[features],
            df_eval[target].to_numpy(dtype=float),
            n_repeats=12,
            random_state=RANDOM_STATE,
            scoring="neg_root_mean_squared_error",
        )
        for feature, importance in zip(features, result.importances_mean):
            rows.append({"target": target, "feature": feature, "importance": importance})
    return pd.DataFrame(rows)


def plot_importance(importance_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharex=False)
    for ax, target in zip(axes, ["GHI", "DNI"]):
        sub = importance_df[importance_df["target"] == target].sort_values("importance")
        ax.barh(sub["feature"], sub["importance"], color="#6A994E", alpha=0.85)
        ax.set_title(target)
        ax.grid(axis="x", linestyle=":", alpha=0.45)
    fig.suptitle("Permutation Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _axis_values(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.array(sorted(df[column].dropna().astype(float).unique()), dtype=float)


def build_interpolator_spec(df: pd.DataFrame, target: str, features: list[str]) -> dict:
    axes = {feature: _axis_values(df, feature) for feature in features}
    shape = tuple(len(axes[feature]) for feature in features)
    values = np.full(shape, np.nan, dtype=float)
    index_maps = {
        feature: {float(value): idx for idx, value in enumerate(axes[feature])}
        for feature in features
    }

    for _, row in df.iterrows():
        index = tuple(index_maps[feature][float(row[feature])] for feature in features)
        values[index] = float(row[target])

    if np.isnan(values).any():
        raise ValueError(f"Incomplete regular grid for {target}; cannot build interpolator.")

    return {
        "target": target,
        "features": features,
        "axes": axes,
        "values": values,
        "bounds": {
            feature: (float(axes[feature][0]), float(axes[feature][-1]))
            for feature in features
        },
    }


def build_interpolator_bundle(df: pd.DataFrame, albedo_transform: dict) -> dict:
    specs = {
        target: build_interpolator_spec(df, target, features)
        for target, features in TARGET_SPECS.items()
    }
    return {
        "models": specs,
        "target_columns": ["GHI", "DNI"],
        "derived_targets": {"DHI": "GHI - DNI"},
        "feature_map": TARGET_SPECS,
        "albedo_transform": albedo_transform,
        "metadata": {
            "description": "Cloudy dM theta-truncation 3 g2-escape DW surrogate using regular-grid interpolation.",
            "model_type": "RegularGridInterpolator specs",
            "n_samples": int(len(df)),
            "DHI_convention": "The LUT uses GHI = DNI + DHI, so DHI is derived as GHI - DNI.",
        },
    }


def predict_interpolator(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    pred = {}
    for target, spec in bundle["models"].items():
        features = spec["features"]
        axes = tuple(np.asarray(spec["axes"][feature], dtype=float) for feature in features)
        interpolator = RegularGridInterpolator(axes, np.asarray(spec["values"], dtype=float), bounds_error=False)
        points = []
        for feature in features:
            axis = np.asarray(spec["axes"][feature], dtype=float)
            points.append(np.clip(df[feature].astype(float).to_numpy(), axis[0], axis[-1]))
        pred[target] = interpolator(np.column_stack(points))
    pred_df = pd.DataFrame(pred, index=df.index)
    pred_df["DHI"] = pred_df["GHI"] - pred_df["DNI"]
    return pred_df


def leave_one_albset_interpolation(df: pd.DataFrame, albedo_transform: dict) -> pd.DataFrame:
    rows = []
    albsets = sorted(df["AlbSet"].dropna().astype(int).unique())
    for albset in albsets[1:-1]:
        train_df = df[df["AlbSet"].astype(int) != albset].copy()
        test_df = df[df["AlbSet"].astype(int) == albset].copy()
        bundle = build_interpolator_bundle(train_df, albedo_transform)
        pred = predict_interpolator(bundle, test_df)
        truth = test_df[["GHI", "DNI"]].copy()
        truth["DHI"] = test_df["GHI"] - test_df["DNI"]
        metrics = evaluate_predictions(
            truth[["GHI", "DNI", "DHI"]].to_numpy(dtype=float),
            pred[["GHI", "DNI", "DHI"]].to_numpy(dtype=float),
            ["GHI", "DNI", "DHI"],
        )
        for target, values in metrics.drop(index="AVG").iterrows():
            row = {"held_out_AlbSet": albset, "target": target}
            row.update(values.to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def save_metrics(metrics: pd.DataFrame, out_path: Path) -> None:
    metrics.to_csv(out_path)
    print(metrics.round(6).to_string())
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cloudy dM theta-truncation 3 g2-escape DW GHI/DNI surrogates.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Cloudy DW LUT CSV path.")
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR), help="Output directory.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Random holdout fraction.")
    parser.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG, help="Prefix tag for metrics and figures.")
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX, help="Prefix for saved model pickle names.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Short title used in validation figures.")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION, help="Description stored in model metadata.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.csv)
    validate_columns(df)
    df, albedo_transform = add_features(df)
    used_columns = sorted({"GHI", "DNI", *[feature for features in TARGET_SPECS.values() for feature in features]})
    df = df.dropna(subset=used_columns).copy()

    print(f"Rows: {len(df)}")
    print("Feature map:")
    for target, features in TARGET_SPECS.items():
        print(f"  {target}: {features}")
    print(f"Albedo PC1 scores by AlbSet: {albedo_transform['albset_scores']}")

    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=args.test_size, random_state=RANDOM_STATE)
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    hgb_models = {}
    for target, features in TARGET_SPECS.items():
        model = build_hgb_model()
        model.fit(train_df[features], train_df[target].to_numpy(dtype=float))
        hgb_models[target] = model

    hgb_pred = predict_hgb(hgb_models, test_df)
    hgb_truth = test_df[["GHI", "DNI"]].copy()
    hgb_truth["DHI"] = test_df["GHI"] - test_df["DNI"]
    hgb_metrics = evaluate_predictions(
        hgb_truth[["GHI", "DNI", "DHI"]].to_numpy(dtype=float),
        hgb_pred[["GHI", "DNI", "DHI"]].to_numpy(dtype=float),
        ["GHI", "DNI", "DHI"],
    )
    save_metrics(hgb_metrics, out_dir / f"metrics_{args.output_tag}_HGB_V1.csv")
    plot_scatter(
        hgb_truth[["GHI", "DNI", "DHI"]],
        hgb_pred[["GHI", "DNI", "DHI"]],
        out_dir / f"validation_scatter_{args.output_tag}_HGB_V1.png",
        f"{args.title} HGB V1",
    )
    importance = collect_importance(hgb_models, test_df)
    importance.to_csv(out_dir / f"feature_importance_{args.output_tag}_HGB_V1.csv", index=False)
    plot_importance(importance, out_dir / f"feature_importance_{args.output_tag}_HGB_V1.png")

    hgb_bundle = {
        "models": hgb_models,
        "target_columns": ["GHI", "DNI"],
        "derived_targets": {"DHI": "GHI - DNI"},
        "feature_map": TARGET_SPECS,
        "albedo_transform": albedo_transform,
        "metadata": {
            "description": f"{args.description} trained with HistGradientBoostingRegressor.",
            "model_type": "HistGradientBoostingRegressor",
            "n_samples": int(len(df)),
            "random_state": RANDOM_STATE,
            "DHI_convention": "The LUT uses GHI = DNI + DHI, so DHI is derived as GHI - DNI.",
        },
    }
    hgb_path = out_dir / f"{args.model_prefix}_HGB_V1.pkl"
    joblib.dump(hgb_bundle, hgb_path)
    print(f"Saved: {hgb_path}")

    interp_bundle = build_interpolator_bundle(df, albedo_transform)
    interp_bundle["metadata"]["description"] = f"{args.description} using regular-grid interpolation."
    interp_path = out_dir / f"{args.model_prefix}_interp_V1.pkl"
    joblib.dump(interp_bundle, interp_path)
    print(f"Saved: {interp_path}")
    default_model_path = out_dir / DEFAULT_MODEL_ALIAS
    joblib.dump(interp_bundle, default_model_path)
    print(f"Saved: {default_model_path}")

    interp_pred = predict_interpolator(interp_bundle, df)
    interp_truth = df[["GHI", "DNI"]].copy()
    interp_truth["DHI"] = df["GHI"] - df["DNI"]
    interp_metrics = evaluate_predictions(
        interp_truth[["GHI", "DNI", "DHI"]].to_numpy(dtype=float),
        interp_pred[["GHI", "DNI", "DHI"]].to_numpy(dtype=float),
        ["GHI", "DNI", "DHI"],
    )
    save_metrics(interp_metrics, out_dir / f"metrics_{args.output_tag}_interp_V1.csv")

    loo_albedo = leave_one_albset_interpolation(df, albedo_transform)
    loo_path = out_dir / f"metrics_{args.output_tag}_interp_leave_one_albset_V1.csv"
    loo_albedo.to_csv(loo_path, index=False)
    print(loo_albedo.round(6).to_string(index=False))
    print(f"Saved: {loo_path}")


if __name__ == "__main__":
    main()
