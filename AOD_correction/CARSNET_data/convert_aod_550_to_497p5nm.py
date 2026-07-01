from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ALPHA = 1.3
SOURCE_WAVELENGTH_NM = 550.0
TARGET_WAVELENGTH_NM = 497.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CARSNET-derived AOD from 550 nm to 497.5 nm using Angstrom scaling."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "annual_site_summary"
        / "cern_to_carsnet_aod_match_excluding_BJC.csv",
        help="Input match table with AOD referenced at 550 nm.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "annual_site_summary"
        / "cern_to_carsnet_aod_match_excluding_BJC_497p5nm_alpha1p3.csv",
        help="Output CSV path for converted AOD values.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Angstrom exponent used for spectral conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if "suggested_AOD_fixed" not in df.columns:
        raise ValueError(f"{args.input_csv} is missing required column 'suggested_AOD_fixed'")

    conversion_factor = (TARGET_WAVELENGTH_NM / SOURCE_WAVELENGTH_NM) ** (-args.alpha)
    df["source_wavelength_nm"] = SOURCE_WAVELENGTH_NM
    df["target_wavelength_nm"] = TARGET_WAVELENGTH_NM
    df["angstrom_exponent"] = args.alpha
    df["conversion_factor_550_to_497p5"] = conversion_factor
    df["suggested_AOD_fixed_497p5nm"] = df["suggested_AOD_fixed"] * conversion_factor

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, float_format="%.6f")

    print(f"Saved converted CSV: {args.output_csv}")
    print(f"Applied Angstrom exponent alpha = {args.alpha:.3f}")
    print(f"Conversion factor 550 -> 497.5 nm = {conversion_factor:.6f}")


if __name__ == "__main__":
    main()
