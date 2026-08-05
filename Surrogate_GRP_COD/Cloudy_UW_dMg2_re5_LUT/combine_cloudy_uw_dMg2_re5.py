"""Combine and validate cloudy UW dM g2 re=5 LUT partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ["C01", "C02", "C03", "C05", "C06"]
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
    parser = argparse.ArgumentParser(description="Combine cloudy UW dM g2 re=5 LUT CSV partitions.")
    parser.add_argument("--input", action="append", required=True, help="Input partition CSV; repeat as needed.")
    parser.add_argument("--out", required=True, help="Combined CSV output path.")
    parser.add_argument("--expected-rows", type=int, default=2520)
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(path) for path in args.input], ignore_index=True)
    missing = sorted(set(KEY_COLUMNS + CHANNELS).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
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
    require_single_value(df, "method", "dM")
    require_single_value(df, "escape", "g2")
    if not np.isfinite(df[CHANNELS].to_numpy(dtype=float)).all():
        raise ValueError("One or more surrogate target channels contain non-finite values.")

    if "tpw" not in df.columns:
        raise ValueError("Missing TPW feature column: tpw")
    tpw_spans = df.groupby(["Ts", "RH"])["tpw"].agg(lambda values: float(values.max() - values.min()))
    max_tpw_span = float(tpw_spans.max())
    if max_tpw_span > 1e-9:
        raise ValueError(f"TPW differs across equivalent (Ts, RH) cases by up to {max_tpw_span}")
    df["tpw"] = df.groupby(["Ts", "RH"])["tpw"].transform("mean")

    df = df.sort_values(KEY_COLUMNS).reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Combined rows: {len(df)}")
    print(f"Grid axis sizes: {axis_sizes}")
    if "source_server" in df.columns:
        print(f"Rows by source: {df['source_server'].value_counts().sort_index().to_dict()}")
    print(f"Canonical TPW values: {df['tpw'].nunique()} (maximum source drift {max_tpw_span:.3e})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
