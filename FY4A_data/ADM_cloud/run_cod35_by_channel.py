import argparse
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
    DEFAULT_SOLAR_ZENITHS,
    FY4A_calinu,
)
from FY4A_data.ADM_cloud.run_adm_cases import (  # noqa: E402
    AOD,
    N_BUNDLES,
    RH_PERCENT,
    SURFACE,
    T_SURF,
    existing_output,
    output_candidates,
    output_dirs,
    output_filename,
)


COD = 35
DEFAULT_POOL_PROCESSES = 4


def channel_filename(cod, sun_zen, channel):
    filename = output_filename(cod, sun_zen)
    stem, ext = os.path.splitext(filename)
    return f"{stem}_{channel}{ext}"


def channel_output_candidates(cod, sun_zen, channel):
    filename = channel_filename(cod, sun_zen, channel)
    return [os.path.join(dirname, filename) for dirname in output_dirs()]


def existing_channel_output(cod, sun_zen, channel):
    for path in channel_output_candidates(cod, sun_zen, channel):
        if os.path.exists(path):
            return path
    return None


def run_channel_case(cod, sun_zen, channel, force=False):
    existing = existing_channel_output(cod, sun_zen, channel)
    if existing is not None and not force:
        print(f"Skipping existing channel file {existing}", flush=True)
        return existing

    print(f"Running COD={cod}, solar zenith={sun_zen}, channel={channel}", flush=True)
    run_RTM(
        sun_zen=sun_zen,
        COD_guess=cod,
        T_s=T_SURF,
        rh=RH_PERCENT / 100.0,
        df_albedo=None,
        surface=SURFACE,
        file_dir=os.path.join(REPO_ROOT, "FY4A_data") + os.sep,
        channels=[channel],
        bandmode="FY4A",
        meth="HG",
        N_bundles=N_BUNDLES,
        AOD=AOD,
        Save_rxyz=True,
    )

    standard = existing_output(cod, sun_zen)
    if standard is None:
        raise RuntimeError(
            "RTM finished but expected output was not found. Checked: "
            + ", ".join(output_candidates(cod, sun_zen))
        )

    channel_path = os.path.join(os.path.dirname(standard), channel_filename(cod, sun_zen, channel))
    os.replace(standard, channel_path)
    print(f"Saved channel file {channel_path}", flush=True)
    return channel_path


def combine_channel_files(cod, sun_zen, channels):
    channel_paths = {}
    for channel in channels:
        path = existing_channel_output(cod, sun_zen, channel)
        if path is None:
            raise FileNotFoundError(
                f"Missing channel file for COD={cod}, solar zenith={sun_zen}, channel={channel}"
            )
        channel_paths[channel] = path

    nu = np.arange(2500, 35000, 3)
    file_dir = os.path.join(REPO_ROOT, "FY4A_data")
    nu_all = FY4A_calinu(nu, channels, file_dir, dnu=3)
    nu_to_idx = {float(nu_value): idx for idx, nu_value in enumerate(nu_all)}
    combined = [None] * len(nu_all)

    for channel in channels:
        path = channel_paths[channel]
        print(f"Loading {path}", flush=True)
        results = np.load(path, allow_pickle=True).item()
        uw_rxyz_M = results.get("uw_rxyz_M")
        if uw_rxyz_M is None:
            raise KeyError(f"{path} does not contain 'uw_rxyz_M'")

        nu_channel = FY4A_calinu(nu, [channel], file_dir, dnu=3)
        if len(uw_rxyz_M) != len(nu_channel):
            raise ValueError(
                f"{path} has {len(uw_rxyz_M)} wavelength entries, expected {len(nu_channel)}"
            )

        for nu_value, rxyz in zip(nu_channel, uw_rxyz_M):
            idx = nu_to_idx[float(nu_value)]
            if combined[idx] is None:
                combined[idx] = rxyz

    missing = sum(1 for item in combined if item is None)
    if missing:
        raise RuntimeError(f"Combined COD={cod}, solar zenith={sun_zen} is missing {missing} wavelengths")

    output_path = os.path.join(os.path.dirname(next(iter(channel_paths.values()))), output_filename(cod, sun_zen))
    np.save(output_path, {"uw_rxyz_M": combined})
    print(f"Saved combined file {output_path}", flush=True)
    return output_path


def run_cod35_by_channel(solar_zeniths, channels, force=False, pool_processes=DEFAULT_POOL_PROCESSES):
    if pool_processes is not None:
        os.environ["MCRTM_POOL_PROCESSES"] = str(pool_processes)
        print(f"Using MCRTM_POOL_PROCESSES={pool_processes}", flush=True)

    for sun_zen in solar_zeniths:
        sun_zen = int(sun_zen)
        combined = existing_output(COD, sun_zen)
        if combined is not None and not force:
            print(f"Skipping existing combined file {combined}", flush=True)
            continue

        for channel in channels:
            run_channel_case(COD, sun_zen, channel, force=force)
        combine_channel_files(COD, sun_zen, channels)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FY4A COD=35 ADM cases channel-by-channel, then combine them."
    )
    parser.add_argument("--solar-zeniths", nargs="+", type=int, default=[int(x) for x in DEFAULT_SOLAR_ZENITHS])
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pool-processes", type=int, default=DEFAULT_POOL_PROCESSES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cod35_by_channel(
        solar_zeniths=args.solar_zeniths,
        channels=args.channels,
        force=args.force,
        pool_processes=args.pool_processes,
    )
