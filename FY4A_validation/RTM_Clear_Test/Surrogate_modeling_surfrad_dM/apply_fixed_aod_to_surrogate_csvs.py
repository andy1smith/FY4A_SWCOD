from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[2]
MATCH_CSV = (
    ROOT_DIR
    / "AOD_correction"
    / "CARSNET_data"
    / "annual_site_summary"
    / "cern_to_carsnet_aod_match_excluding_BJC.csv"
)
BJC_AOD_CSV = ROOT_DIR / "AOD_correction" / "AERONET_china" / "2021_BJC_CAMS.csv"
INPUT_DIR = BASE_DIR / "withoutAOD"
OUTPUT_DIR = BASE_DIR / "withAOD"
BJC_AOD_TOLERANCE = pd.Timedelta("1h")


def build_aod_mapping(match_csv: Path) -> dict[str, dict[str, object]]:
    match_df = pd.read_csv(match_csv)
    return {row["cern_site"]: row.to_dict() for _, row in match_df.iterrows()}


def load_bjc_aod(aod_csv: Path) -> pd.DataFrame:
    aod_df = pd.read_csv(aod_csv)
    if "time" not in aod_df.columns:
        raise ValueError(f"{aod_csv} is missing 'time' column")

    aod_col = "AOD_500nm" if "AOD_500nm" in aod_df.columns else "aod"
    if aod_col not in aod_df.columns:
        raise ValueError(f"{aod_csv} is missing AOD column")

    aod_df = aod_df.rename(columns={"time": "Time", aod_col: "bjc_aod"})[["Time", "bjc_aod"]]
    aod_df["Time"] = pd.to_datetime(aod_df["Time"])
    aod_df["bjc_aod"] = pd.to_numeric(aod_df["bjc_aod"], errors="coerce")
    aod_df = aod_df.dropna(subset=["Time", "bjc_aod"])
    return aod_df.sort_values("Time")


def apply_bjc_aod(df: pd.DataFrame, bjc_aod: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if "Time" not in df.columns:
        raise ValueError("BJC CSV is missing 'Time' column")

    df = df.copy()
    original_index = df.index
    df["Time"] = pd.to_datetime(df["Time"])
    df_sorted = df.sort_values("Time")
    aod_median = float(bjc_aod["bjc_aod"].median())

    merged = pd.merge_asof(
        df_sorted[["Time"]],
        bjc_aod,
        on="Time",
        direction="nearest",
        tolerance=BJC_AOD_TOLERANCE,
    )
    matched_count = int(merged["bjc_aod"].notna().sum())
    fill_count = int(merged["bjc_aod"].isna().sum())
    df_sorted["aod"] = merged["bjc_aod"].fillna(aod_median).values
    df = df_sorted.loc[original_index]

    return df, {
        "matched_count": matched_count,
        "fill_count": fill_count,
        "aod_median": aod_median,
    }


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = build_aod_mapping(MATCH_CSV)
    bjc_aod = load_bjc_aod(BJC_AOD_CSV)
    manifest_rows: list[dict[str, object]] = []

    input_files = sorted(INPUT_DIR.glob("gpr_predicted_dM_*.csv"))
    if not input_files:
        raise FileNotFoundError(f"No input CSVs found in {INPUT_DIR}")

    for csv_path in input_files:
        site = csv_path.stem.replace("gpr_predicted_dM_", "")
        df = pd.read_csv(csv_path)
        if "aod" not in df.columns:
            raise ValueError(f"{csv_path} is missing 'aod' column")

        original_unique_aod = sorted(pd.to_numeric(df["aod"], errors="coerce").dropna().unique().tolist())
        bjc_stats = {"matched_count": len(df), "fill_count": 0, "aod_median": float("nan")}

        if site == "BJC":
            df, bjc_stats = apply_bjc_aod(df, bjc_aod)
            new_aod = bjc_stats["aod_median"]
            matched_carsnet_site = "BJC_existing"
            distance_km = 0.0
            match_quality = "time_series"
            aod_method = "BJC_AERONET_CAMS_nearest_1h_median_fill"
        else:
            if site not in mapping:
                raise KeyError(f"No AOD mapping found for site {site}")
            match_row = mapping[site]
            new_aod = float(match_row["suggested_AOD_fixed"])
            matched_carsnet_site = str(match_row["matched_carsnet_site"])
            distance_km = float(match_row["distance_km"])
            match_quality = str(match_row["match_quality"])
            aod_method = str(match_row["aod_method"])
            df["aod"] = new_aod

        out_path = OUTPUT_DIR / csv_path.name
        df.to_csv(out_path, index=False)

        manifest_rows.append(
            {
                "file_name": csv_path.name,
                "site": site,
                "matched_carsnet_site": matched_carsnet_site,
                "distance_km": distance_km,
                "match_quality": match_quality,
                "aod_method": aod_method,
                "original_aod_unique": ";".join(f"{value:.6f}" for value in original_unique_aod),
                "new_aod": new_aod,
                "new_aod_min": float(df["aod"].min()),
                "new_aod_max": float(df["aod"].max()),
                "aod_match_count": bjc_stats["matched_count"] if site == "BJC" else len(df),
                "aod_fill_count": bjc_stats["fill_count"] if site == "BJC" else 0,
                "row_count": len(df),
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("site")
    manifest.to_csv(OUTPUT_DIR / "withAOD_manifest.csv", index=False, float_format="%.6f")
    print(f"Saved {len(manifest_rows)} site files to {OUTPUT_DIR}")
    print(f"Saved manifest to {OUTPUT_DIR / 'withAOD_manifest.csv'}")


if __name__ == "__main__":
    main()
