#!/usr/bin/env python3
"""
FY-4A visible/SWIR cloud-top phase classifier.

This reproduces the RGB + Lab + K-means phase step described in
Xu et al. (Remote Sensing, 2023, 15, 126) for the existing cropped FY-4A
11x11 site CSVs. It classifies cloud pixels into ice, water, and
thin/unclassified groups from AGRI channels 01, 05, and 06:

    R = Channel05, 1.6 um reflectance, normalized by 0.4
    G = Channel06, 2.2 um reflectance, normalized by 0.4
    B = Channel01, 0.46 um reflectance, normalized by 1.0

The 10.8 um brightness-temperature step in the paper separates warm water
from supercooled water. The cropped shortwave-only inputs used here do not
contain that channel, so this script only separates water cloud from ice cloud.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


PHASE_INVALID = -1
PHASE_THIN_OR_UNCLASSIFIED = 0
PHASE_ICE = 1
PHASE_WATER = 2

PHASE_NAMES = {
    PHASE_INVALID: "invalid",
    PHASE_THIN_OR_UNCLASSIFIED: "thin_or_unclassified",
    PHASE_ICE: "ice",
    PHASE_WATER: "water",
}


@dataclass(frozen=True)
class PhaseProducts:
    times: pd.DatetimeIndex
    pixel_columns: list[str]
    phase_code: np.ndarray
    cluster_id: np.ndarray
    valid_mask: np.ndarray
    optically_thick_mask: np.ndarray
    cluster_summary: pd.DataFrame


def read_pixel_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"{path} is missing a 'time' column")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")
    pixel_columns = [col for col in df.columns if col.isdigit()]
    if not pixel_columns:
        raise ValueError(f"{path} has no numeric pixel columns")
    pixel_columns = sorted(pixel_columns, key=int)
    return df[pixel_columns].astype(float)


def load_site_arrays(input_root: Path, site: str) -> tuple[pd.DatetimeIndex, list[str], dict[str, np.ndarray]]:
    site_dir = input_root / site
    if not site_dir.exists():
        raise FileNotFoundError(f"Site directory not found: {site_dir}")

    channel_files = {
        "C01": site_dir / f"{site}_Channel01.csv",
        "C02": site_dir / f"{site}_Channel02.csv",
        "C03": site_dir / f"{site}_Channel03.csv",
        "C05": site_dir / f"{site}_Channel05.csv",
        "C06": site_dir / f"{site}_Channel06.csv",
        "SunZenith": site_dir / f"{site}_SunZenith.csv",
    }
    missing = [str(path) for path in channel_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))

    frames = {name: read_pixel_csv(path) for name, path in channel_files.items()}
    common_index = None
    common_columns = None
    for df in frames.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
        common_columns = list(df.columns) if common_columns is None else [
            col for col in common_columns if col in df.columns
        ]
    if common_index is None or common_columns is None or len(common_index) == 0:
        raise ValueError("No common timestamps or pixel columns across required inputs")

    arrays = {
        name: df.loc[common_index, common_columns].to_numpy(dtype=float)
        for name, df in frames.items()
    }
    return pd.DatetimeIndex(common_index), common_columns, arrays


def make_rgb(c01: np.ndarray, c05: np.ndarray, c06: np.ndarray) -> np.ndarray:
    red = np.clip(c05 / 0.4, 0.0, 1.0)
    green = np.clip(c06 / 0.4, 0.0, 1.0)
    blue = np.clip(c01, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to CIE Lab using D65 white."""
    rgb = np.asarray(rgb, dtype=float)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883])

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def assign_phase_labels(cluster_stats: pd.DataFrame) -> dict[int, int]:
    if len(cluster_stats) < 3:
        raise ValueError("Expected three clusters for water, ice, and thin/unclassified groups")

    thin_cluster = int(cluster_stats.sort_values(["mean_C01", "mean_rgb_sum"]).iloc[0]["cluster"])
    remaining = cluster_stats[cluster_stats["cluster"] != thin_cluster].copy()

    # Water clouds have larger 1.6 and 2.2 um reflectance than ice in the
    # paper's RGB table; ice clouds are bluer relative to the SWIR channels.
    remaining["water_score"] = remaining["mean_R"] + remaining["mean_G"] - remaining["mean_B"]
    water_cluster = int(remaining.sort_values("water_score", ascending=False).iloc[0]["cluster"])
    ice_cluster = int(remaining[remaining["cluster"] != water_cluster].iloc[0]["cluster"])

    return {
        thin_cluster: PHASE_THIN_OR_UNCLASSIFIED,
        ice_cluster: PHASE_ICE,
        water_cluster: PHASE_WATER,
    }


def classify_phase(
    arrays: dict[str, np.ndarray],
    times: pd.DatetimeIndex,
    pixel_columns: list[str],
    solar_zenith_max: float = 65.0,
    optical_thickness_reflectance: float = 0.4,
    random_state: int = 0,
) -> PhaseProducts:
    c01 = arrays["C01"]
    c05 = arrays["C05"]
    c06 = arrays["C06"]
    sun_zenith = arrays["SunZenith"]
    rgb = make_rgb(c01, c05, c06)
    lab = rgb_to_lab(rgb)

    finite_mask = (
        np.isfinite(c01)
        & np.isfinite(c05)
        & np.isfinite(c06)
        & np.isfinite(sun_zenith)
        & np.all(np.isfinite(lab), axis=-1)
    )
    valid_mask = finite_mask & (sun_zenith < solar_zenith_max)
    optically_thick_mask = valid_mask & (c01 > optical_thickness_reflectance)

    flat_valid = valid_mask.reshape(-1)
    if flat_valid.sum() < 3:
        raise ValueError("Fewer than three valid pixels after solar-zenith and finite-data filtering")

    features = lab.reshape(-1, 3)[flat_valid, 1:3]
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=random_state)
    valid_clusters = kmeans.fit_predict(features)

    cluster_id = np.full(c01.size, -1, dtype=np.int16)
    cluster_id[flat_valid] = valid_clusters
    cluster_id = cluster_id.reshape(c01.shape)

    flat_rgb = rgb.reshape(-1, 3)
    flat_c01 = c01.reshape(-1)
    flat_c05 = c05.reshape(-1)
    flat_c06 = c06.reshape(-1)
    rows = []
    for cluster in sorted(np.unique(valid_clusters)):
        mask = cluster_id.reshape(-1) == cluster
        phase_pixels = int(mask.sum())
        rows.append(
            {
                "cluster": int(cluster),
                "n_pixels": phase_pixels,
                "mean_R": float(np.nanmean(flat_rgb[mask, 0])),
                "mean_G": float(np.nanmean(flat_rgb[mask, 1])),
                "mean_B": float(np.nanmean(flat_rgb[mask, 2])),
                "mean_rgb_sum": float(np.nanmean(flat_rgb[mask].sum(axis=1))),
                "mean_C01": float(np.nanmean(flat_c01[mask])),
                "mean_C05": float(np.nanmean(flat_c05[mask])),
                "mean_C06": float(np.nanmean(flat_c06[mask])),
                "thick_fraction": float(np.nanmean(optically_thick_mask.reshape(-1)[mask])),
            }
        )
    cluster_summary = pd.DataFrame(rows)
    cluster_to_phase = assign_phase_labels(cluster_summary)
    cluster_summary["phase_code"] = cluster_summary["cluster"].map(cluster_to_phase)
    cluster_summary["phase_name"] = cluster_summary["phase_code"].map(PHASE_NAMES)

    phase_code = np.full(c01.shape, PHASE_INVALID, dtype=np.int16)
    for cluster, phase in cluster_to_phase.items():
        phase_code[valid_mask & (cluster_id == cluster)] = phase

    return PhaseProducts(
        times=times,
        pixel_columns=pixel_columns,
        phase_code=phase_code,
        cluster_id=cluster_id,
        valid_mask=valid_mask,
        optically_thick_mask=optically_thick_mask,
        cluster_summary=cluster_summary.sort_values("cluster").reset_index(drop=True),
    )


def array_to_csv_frame(times: pd.DatetimeIndex, pixel_columns: list[str], values: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(values, columns=pixel_columns)
    df.insert(0, "time", times)
    return df


def write_outputs(site: str, products: PhaseProducts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    array_to_csv_frame(products.times, products.pixel_columns, products.phase_code).to_csv(
        output_dir / f"{site}_cloud_phase_code.csv", index=False
    )
    array_to_csv_frame(products.times, products.pixel_columns, products.cluster_id).to_csv(
        output_dir / f"{site}_cloud_phase_cluster.csv", index=False
    )
    array_to_csv_frame(products.times, products.pixel_columns, products.valid_mask.astype(np.int8)).to_csv(
        output_dir / f"{site}_phase_valid_sza_mask.csv", index=False
    )
    array_to_csv_frame(products.times, products.pixel_columns, products.optically_thick_mask.astype(np.int8)).to_csv(
        output_dir / f"{site}_phase_optically_thick_mask.csv", index=False
    )
    products.cluster_summary.to_csv(output_dir / f"{site}_cloud_phase_cluster_summary.csv", index=False)

    counts = pd.DataFrame({"time": products.times})
    for code, name in PHASE_NAMES.items():
        counts[name] = (products.phase_code == code).sum(axis=1)
    counts["valid"] = products.valid_mask.sum(axis=1)
    counts["optically_thick"] = products.optically_thick_mask.sum(axis=1)
    counts.to_csv(output_dir / f"{site}_cloud_phase_counts_by_time.csv", index=False)


def print_summary(site: str, products: PhaseProducts, output_dir: Path) -> None:
    total = products.phase_code.size
    print(f"Site: {site}")
    print(f"Timestamps: {len(products.times)}")
    print(f"Pixels per timestamp: {len(products.pixel_columns)}")
    print(f"Total pixels: {total}")
    print(f"Valid SZA-filtered pixels: {int(products.valid_mask.sum())}")
    print(f"Optically thick pixels (C01 > threshold): {int(products.optically_thick_mask.sum())}")
    for code, name in PHASE_NAMES.items():
        print(f"{name}: {int((products.phase_code == code).sum())}")
    print("\nCluster summary:")
    print(products.cluster_summary.to_string(index=False))
    print(f"\nWrote outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_preprocessing_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Classify FY-4A water/ice cloud phase from cropped shortwave channel CSVs."
    )
    parser.add_argument("--site", default="BJC", help="Site code to process, e.g. BJC")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_preprocessing_dir / "cropped_FY2021_cloudy",
        help="Directory containing <SITE>/<SITE>_Channel*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
        help="Directory for phase-mask outputs",
    )
    parser.add_argument("--solar-zenith-max", type=float, default=65.0)
    parser.add_argument("--optical-thickness-reflectance", type=float, default=0.4)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, pixel_columns, arrays = load_site_arrays(args.input_root, args.site)
    products = classify_phase(
        arrays,
        times,
        pixel_columns,
        solar_zenith_max=args.solar_zenith_max,
        optical_thickness_reflectance=args.optical_thickness_reflectance,
        random_state=args.random_state,
    )
    output_dir = args.output_dir / args.site
    write_outputs(args.site, products, output_dir)
    print_summary(args.site, products, output_dir)


if __name__ == "__main__":
    main()
