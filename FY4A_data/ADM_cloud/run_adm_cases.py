import os
import platform
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


AOD = 0.1243
T_SURF = 294
RH_PERCENT = 60
N_BUNDLES = 10000
SURFACE = "Case2"
SKIP_CASES = {(35, sun_zen) for sun_zen in DEFAULT_SOLAR_ZENITHS}


def output_filename(cod, sun_zen):
    return (
        f"uwxyzr_{SURFACE}_AOD={AOD:.2f}_COD={int(cod)}_"
        f"th0={int(sun_zen)}_Ts={T_SURF}_RH={RH_PERCENT}.npy"
    )


def output_dirs():
    dirs = [
        os.path.join(REPO_ROOT, "FY4A_data", "RTM_10000", "channels", "FY4A"),
    ]

    machine_name = platform.node()
    if machine_name == "user-Super-Server":
        dirs.extend(
            [
                os.path.join("/home/dengnan/data", "RTM_10000", "channels", "FY4A"),
                # Current run_RTM path on user-Super-Server appends RTM_10000/channels twice.
                os.path.join(
                    "/home/dengnan/data",
                    "RTM_10000",
                    "channels",
                    "RTM_10000",
                    "channels",
                    "FY4A",
                ),
            ]
        )
    elif machine_name == "user-MS-7D30":
        dirs.append(os.path.join("/mnt/dengnan", "RTM_10000", "channels", "FY4A"))

    unique_dirs = []
    for dirname in dirs:
        if dirname not in unique_dirs:
            unique_dirs.append(dirname)
    return unique_dirs


def output_candidates(cod, sun_zen):
    filename = output_filename(cod, sun_zen)
    return [os.path.join(dirname, filename) for dirname in output_dirs()]


def existing_output(cod, sun_zen):
    for path in output_candidates(cod, sun_zen):
        if os.path.exists(path):
            return path
    return None


def run_adm_simulations():
    total = len(DEFAULT_COD_GRID) * len(DEFAULT_SOLAR_ZENITHS)
    completed = 0
    print(
        f"Starting FY4A ADM simulations. Total permutations: {total}",
        flush=True,
    )
    print("Checking output directories:", flush=True)
    for dirname in output_dirs():
        print(f"  {dirname}", flush=True)

    for sun_zen in DEFAULT_SOLAR_ZENITHS:
        for cod in DEFAULT_COD_GRID:
            cod = int(cod)
            sun_zen = int(sun_zen)
            if (cod, sun_zen) in SKIP_CASES:
                print(
                    f"Skipping requested case COD={cod}, solar zenith={sun_zen}",
                    flush=True,
                )
                continue

            existing = existing_output(cod, sun_zen)
            if existing is not None:
                completed += 1
                print(f"[{completed}/{total}] Skipping existing {existing}", flush=True)
                continue

            print(
                f"[{completed + 1}/{total}] Running COD={cod}, solar zenith={sun_zen}...",
                flush=True,
            )
            run_RTM(
                sun_zen=sun_zen,
                COD_guess=cod,
                T_s=T_SURF,
                rh=RH_PERCENT / 100.0,
                df_albedo=None,
                surface=SURFACE,
                file_dir=os.path.join(REPO_ROOT, "FY4A_data") + os.sep,
                channels=DEFAULT_CHANNELS,
                bandmode="FY4A",
                meth="HG",
                N_bundles=N_BUNDLES,
                AOD=AOD,
                Save_rxyz=True,
            )

            existing = existing_output(cod, sun_zen)
            if existing is None:
                raise RuntimeError(
                    "RTM finished but expected output was not found. Checked: "
                    + ", ".join(output_candidates(cod, sun_zen))
                )
            completed += 1
            print(f"[{completed}/{total}] Finished {existing}", flush=True)

    print(f"All requested ADM RTM cases are available: {completed}/{total}", flush=True)


if __name__ == "__main__":
    run_adm_simulations()
