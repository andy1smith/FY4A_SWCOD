"""Train cloudy UW dM g2 re=5 models with the established channel trainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_TRAINER_DIR = SCRIPT_DIR.parent / "Cloudy_uw_dM_escapeg2"
sys.path.insert(0, str(BASE_TRAINER_DIR))

import train_surrogate_COD_HG_V1 as trainer  # noqa: E402


MODEL_NAME = "SWRTM_cloudy_uw_channel_dMg2_re5_V1.pkl"
INTERP_MODEL_NAME = "SWRTM_cloudy_uw_channel_dMg2_re5_interp_V1.pkl"


def update_metadata(path: Path) -> None:
    bundle = joblib.load(path)
    metadata = bundle.setdefault("metadata", {})
    description = str(metadata.get("description", "Cloudy dM g2-escape forward surrogate"))
    metadata["description"] = description.replace("Cloudy dM g2-escape", "Cloudy dM g2-escape re=5")
    metadata["cloud_effective_radius_um"] = 5.0
    metadata["delta_m"] = True
    metadata["escape_probability_mode"] = "g2"
    joblib.dump(bundle, path, compress=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train cloudy UW dM g2 re=5 FY4A channel surrogates.")
    parser.add_argument("--csv", default=str(SCRIPT_DIR / "preprocessed_cloudy_uw_dMg2_re5.csv"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR))
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    trainer.MODEL_NAME = MODEL_NAME
    trainer.INTERP_MODEL_NAME = INTERP_MODEL_NAME
    trainer.OUTPUT_TAG = "channel_dMg2_re5"
    trainer.TITLE = "Cloudy dM g2 re=5"
    sys.argv = [
        str(BASE_TRAINER_DIR / "train_surrogate_COD_HG_V1.py"),
        "--csv",
        args.csv,
        "--out-dir",
        args.out_dir,
        "--test-size",
        str(args.test_size),
    ]
    trainer.main()

    out_dir = Path(args.out_dir)
    update_metadata(out_dir / MODEL_NAME)
    update_metadata(out_dir / INTERP_MODEL_NAME)
    print("Updated model metadata for dM g2 re=5.")


if __name__ == "__main__":
    main()
