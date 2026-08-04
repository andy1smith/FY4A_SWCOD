"""Train cloudy DW HG re=5 surrogates with the established DW trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_TRAINER = SCRIPT_DIR.parent / "Cloudy_dM3_escape_g2" / "train_surrogate_cloudy_dw_dMescape_V1.py"


def main() -> None:
    defaults = [
        "--csv",
        str(SCRIPT_DIR / "cloudy_dw_HG_re5_LUT.csv"),
        "--out-dir",
        str(SCRIPT_DIR),
        "--output-tag",
        "cloudy_dw_HG_re5",
        "--model-prefix",
        "SWRTM_cloudy_dw_HG_re5_GHI_DNI_PC1",
        "--title",
        "Cloudy DW HG re=5",
        "--description",
        "Cloudy HG re=5 DW surrogate",
    ]
    sys.argv = [str(BASE_TRAINER), *defaults, *sys.argv[1:]]
    runpy.run_path(str(BASE_TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
