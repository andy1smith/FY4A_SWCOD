"""Download hourly CAMS McClear clear-sky irradiance for CERN sites.

The cloudy-site QC can use these cached files instead of PVlib Ineichen.
Access is through the SoDa/CAMS service via pvlib.iotools.get_cams and
requires a SoDa account email.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from pvlib.iotools import get_cams


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_SITE_INFO = REPO_ROOT / "FY4A_data" / "CERN_info.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "FY4A_data" / "McClear_clearsky"


def normalize_site_info(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"elve": "elev", "alt": "elev", "altitude": "elev"})
    required = {"site", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "elev" not in df.columns:
        df["elev"] = pd.NA
    df = df[["site", "latitude", "longitude", "elev"]].dropna(subset=["site", "latitude", "longitude"])
    df["site"] = df["site"].astype(str).str.strip()
    return df.drop_duplicates(subset="site", keep="first")


def standardize_mcclear_frame(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    else:
        df.index = df.index.tz_localize(None)
    df = df.reset_index().rename(columns={"index": "Time"})

    rename = {
        "ghi_clear": "ghi_clear_mcclear",
        "dni_clear": "dni_clear_mcclear",
        "dhi_clear": "dhi_clear_mcclear",
        "bhi_clear": "bhi_clear_mcclear",
        "ghi_extra": "ghi_extra_mcclear",
    }
    df = df.rename(columns={key: value for key, value in rename.items() if key in df.columns})
    keep = ["Time"] + [col for col in rename.values() if col in df.columns]
    if "ghi_clear_mcclear" not in keep:
        raise ValueError(f"McClear response did not include clear-sky GHI. Columns: {list(df.columns)}")
    return df[keep].sort_values("Time")


def download_site(row: pd.Series, out_dir: Path, args: argparse.Namespace) -> dict:
    site = row["site"]
    out_path = out_dir / f"{site}_mcclear_{args.start[:4]}_hourly.csv"
    if out_path.exists() and not args.overwrite:
        return {"site": site, "status": "cached", "path": str(out_path)}

    altitude = None if pd.isna(row.get("elev")) else float(row["elev"])
    data, metadata = get_cams(
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        altitude=altitude,
        start=args.start,
        end=args.end,
        email=args.email,
        identifier="mcclear",
        time_step="1h",
        time_ref="UT",
        integrated=False,
        label="left",
        timeout=args.timeout,
    )
    df = standardize_mcclear_frame(data)
    df.insert(0, "site", site)
    df.to_csv(out_path, index=False)
    return {
        "site": site,
        "status": "downloaded",
        "path": str(out_path),
        "n": len(df),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "elev": altitude,
        "metadata_altitude": metadata.get("altitude"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download hourly CAMS McClear clear-sky GHI for CERN sites.")
    parser.add_argument("--email", default=os.environ.get("SODA_EMAIL"), help="SoDa account email, or set SODA_EMAIL.")
    parser.add_argument("--site-info", default=str(DEFAULT_SITE_INFO), help="CERN_info.csv path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for cached McClear CSV files.")
    parser.add_argument("--start", default="2021-01-01", help="First date to request.")
    parser.add_argument("--end", default="2021-12-31", help="Last date to request.")
    parser.add_argument("--site", action="append", help="Optional site code; repeat to request multiple sites.")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing cached files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.email:
        raise SystemExit("Missing SoDa email. Pass --email or set SODA_EMAIL.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = normalize_site_info(Path(args.site_info))
    if args.site:
        wanted = {site.upper() for site in args.site}
        sites = sites[sites["site"].str.upper().isin(wanted)]
    if sites.empty:
        raise SystemExit("No matching sites found.")

    rows = []
    for _, row in sites.iterrows():
        print(f"{row['site']}: downloading/caching McClear", flush=True)
        rows.append(download_site(row, out_dir, args))

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / f"mcclear_download_manifest_{args.start[:4]}.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
