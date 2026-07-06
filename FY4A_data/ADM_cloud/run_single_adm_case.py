import argparse
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fun_nearealtime_RTM import run_RTM  # noqa: E402
from run_adm_cases import (  # noqa: E402
    AOD,
    DEFAULT_CHANNELS,
    N_BUNDLES,
    RH_PERCENT,
    SURFACE,
    T_SURF,
    existing_output,
    output_candidates,
)


def run_single_case(cod, sun_zen, force=False):
    existing = existing_output(cod, sun_zen)
    if existing is not None and not force:
        print(f"Skipping existing {existing}", flush=True)
        return existing

    print(f"Running COD={cod}, solar zenith={sun_zen}...", flush=True)
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
    print(f"Finished {existing}", flush=True)
    return existing


def parse_args():
    parser = argparse.ArgumentParser(description="Run one FY4A ADM RTM case.")
    parser.add_argument("--cod", type=int, required=True)
    parser.add_argument("--sun-zen", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_single_case(args.cod, args.sun_zen, args.force)
