import argparse
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from FY4A_data.ADM_cloud.AngDistLUT import DEFAULT_SOLAR_ZENITHS  # noqa: E402
from FY4A_data.ADM_cloud.run_single_adm_case import run_single_case  # noqa: E402


COD = 35
DEFAULT_POOL_PROCESSES = 8


def run_cod35_cases(force=False, pool_processes=DEFAULT_POOL_PROCESSES):
    if pool_processes is not None:
        os.environ["MCRTM_POOL_PROCESSES"] = str(pool_processes)
        print(f"Using MCRTM_POOL_PROCESSES={pool_processes}", flush=True)

    total = len(DEFAULT_SOLAR_ZENITHS)
    for idx, sun_zen in enumerate(DEFAULT_SOLAR_ZENITHS, start=1):
        sun_zen = int(sun_zen)
        print(f"[{idx}/{total}] COD={COD}, solar zenith={sun_zen}", flush=True)
        run_single_case(COD, sun_zen, force=force)


def parse_args():
    parser = argparse.ArgumentParser(description="Run local FY4A ADM COD=35 cases only.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pool-processes", type=int, default=DEFAULT_POOL_PROCESSES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cod35_cases(force=args.force, pool_processes=args.pool_processes)
