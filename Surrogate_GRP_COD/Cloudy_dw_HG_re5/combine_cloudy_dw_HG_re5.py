"""Combine and validate cloudy DW HG re=5 LUT partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["AlbSet", "AOD", "COD", "th0", "Ts", "RH"]
EXPECTED_AXIS_SIZES = {
    "AlbSet": 5,
    "AOD": 1,
    "COD": 7,
    "th0": 6,
    "Ts": 4,
    "RH": 3,
}


def require_single_value(df: pd.DataFrame, column: str, expected: object) -> None:
    if column not in df.columns:
        raise ValueError(f"Missing metadata column: {column}")
    values = df[column].dropna().unique()
    if len(values) != 1 or values[0] != expected:
        raise ValueError(f"Expected {column}={expected!r}, found {values.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine cloudy DW HG re=5 LUT CSV partitions.")
    parser.add_argument("--input", action="append", required=True, help="Input partition CSV; repeat as needed.")
    parser.add_argument("--out", required=True, help="Combined CSV output path.")
    parser.add_argument("--expected-rows", type=int, default=2520, help="Required combined row count.")
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.input]
    df = pd.concat(frames, ignore_index=True)

    missing = sorted(set(KEY_COLUMNS).difference(df.columns))
    if missing:
        raise ValueError(f"Missing grid key columns: {missing}")
    if len(df) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, found {len(df)}")

    duplicates = df.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, KEY_COLUMNS].head().to_dict("records")
        raise ValueError(f"Found {int(duplicates.sum())} duplicate grid rows; sample: {sample}")

    axis_sizes = {column: int(df[column].nunique()) for column in KEY_COLUMNS}
    if axis_sizes != EXPECTED_AXIS_SIZES:
        raise ValueError(f"Unexpected grid axis sizes: {axis_sizes}; expected {EXPECTED_AXIS_SIZES}")
    if int(np.prod(list(axis_sizes.values()))) != len(df):
        raise ValueError("Combined LUT is not a complete Cartesian grid.")

    require_single_value(df, "re_um", 5.0)
    require_single_value(df, "method", "HG")
    require_single_value(df, "escape", "none")

    df = df.sort_values(KEY_COLUMNS).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Combined rows: {len(df)}")
    print(f"Grid axis sizes: {axis_sizes}")
    if "source_server" in df.columns:
        print(f"Rows by source: {df['source_server'].value_counts().sort_index().to_dict()}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
