"""Preprocess one server partition of the cloudy UW dM g2 re=5 LUT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BASE_PREPROCESSOR_DIR = SCRIPT_DIR.parent / "Cloudy_uw_dM_escapeg2"
sys.path.insert(0, str(BASE_PREPROCESSOR_DIR))

from preprocess_surrogate_cloudy_uw import CHANNELS, process_cases  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one cloudy UW dM g2 re=5 LUT partition.")
    parser.add_argument("--dir", required=True, help="Directory containing Results_*.npy files.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--source-server", required=True, help="Source label, for example .83 or .85.")
    parser.add_argument("--fy4a-data-dir", default=str(REPO_ROOT / "FY4A_data"))
    args = parser.parse_args()

    df = process_cases(args.dir, fy4a_data_dir=args.fy4a_data_dir)
    if df.empty:
        raise RuntimeError(f"No valid Results_*.npy files found in {args.dir}")

    df.insert(0, "source_server", args.source_server)
    df["re_um"] = 5.0
    df["method"] = "dM"
    df["escape"] = "g2"

    preferred = [
        "source_server",
        "AlbSet",
        "AOD",
        "COD",
        "th0",
        "cos_th0",
        "Ts",
        "RH",
        "rh",
        "tpw",
        "re_um",
        "method",
        "escape",
        *[f"alb_{channel}" for channel in CHANNELS],
        *[f"{channel}_rad" for channel in CHANNELS],
        *CHANNELS,
        *[f"{channel}_nosrf" for channel in CHANNELS],
    ]
    ordered = [column for column in preferred if column in df.columns]
    ordered.extend(column for column in df.columns if column not in ordered)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[ordered].to_csv(out_path, index=False)
    print(f"Processed {len(df)} cases from {args.source_server}.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
