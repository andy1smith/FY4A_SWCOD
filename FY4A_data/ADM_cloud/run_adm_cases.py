import os
import sys

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fun_nearealtime_RTM import run_RTM  # noqa: E402
from FY4A_data.ADM_cloud.AngDistLUT import (  # noqa: E402
    DEFAULT_CHANNELS,
    DEFAULT_COD_GRID,
    DEFAULT_SOLAR_ZENITHS,
)


def output_file(cod, sun_zen):
    return os.path.join(
        REPO_ROOT,
        "FY4A_data",
        "RTM_10000",
        "channels",
        "FY4A",
        f"uwxyzr_Case2_AOD=0.12_COD={int(cod)}_th0={int(sun_zen)}_Ts=294_RH=60.npy",
    )


def run_adm_simulations():
    print(
        "Starting FY4A ADM simulations. "
        f"Total permutations: {len(DEFAULT_COD_GRID) * len(DEFAULT_SOLAR_ZENITHS)}"
    )

    for sun_zen in DEFAULT_SOLAR_ZENITHS:
        for cod in DEFAULT_COD_GRID:
            cod = int(cod)
            sun_zen = int(sun_zen)
            expected_file = output_file(cod, sun_zen)
            if os.path.exists(expected_file):
                print(f"Skipping existing {expected_file}")
                continue
            print(f"Checking/running COD={cod}, solar zenith={sun_zen}...")
            run_RTM(
                sun_zen=sun_zen,
                COD_guess=cod,
                T_s=294,
                rh=0.6,
                df_albedo=None,
                surface="Case2",
                file_dir=os.path.join(REPO_ROOT, "FY4A_data") + os.sep,
                channels=DEFAULT_CHANNELS,
                bandmode="FY4A",
                meth="HG",
                N_bundles=10000,
                AOD=0.1243,
                Save_rxyz=True,
            )


if __name__ == "__main__":
    run_adm_simulations()
