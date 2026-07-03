import argparse
import os
import platform
import sys

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from FY4A_data.ADM_cloud.AngDistLUT import (  # noqa: E402
    DEFAULT_CHANNELS,
    DEFAULT_COD_GRID,
    DEFAULT_SOLAR_ZENITHS,
    FY4A_calinu,
    cal_mono_R,
    get_calibration_srf,
    saveLUT,
)


DEFAULT_AOD = 0.1243
DEFAULT_T_SURF = 294
DEFAULT_RH_PERCENT = 60
DEFAULT_N_BUNDLES = 10000


def get_rtm_results_dir():
    machine_name = platform.node()
    if machine_name == "user-Super-Server":
        return "/home/dengnan/data/RTM_10000/channels/FY4A/"
    if machine_name == "user-MS-7D30":
        return "/mnt/dengnan/RTM_10000/channels/FY4A/"
    return os.path.join(REPO_ROOT, "FY4A_data", "RTM_10000", "channels", "FY4A")


def load_toa_spectrum():
    toa_file = os.path.join(REPO_ROOT, "data", "profiles", "SolarTOA.csv")
    toa_data = np.genfromtxt(toa_file, delimiter=",")
    ref_lam = toa_data[:, 0]
    ref_E = toa_data[:, 1]
    ref_E_nu = -ref_E * ref_lam**2 / 1e4
    return ref_lam, ref_E_nu


def rtm_filename(cod, solar_zenith, surface, aod, t_surf, rh_percent):
    return (
        f"uwxyzr_{surface}_AOD={aod:.2f}_COD={int(cod)}_"
        f"th0={int(solar_zenith)}_Ts={int(t_surf)}_RH={int(rh_percent)}.npy"
    )


def rtm_filename_candidates(cod, solar_zenith, surface, aod, t_surf, rh_percent):
    return [
        rtm_filename(cod, solar_zenith, surface, aod, t_surf, rh_percent),
        (
            f"uwxyzr_{surface}_AOD={aod:g}_COD={int(cod)}_"
            f"th0={int(solar_zenith)}_Ts={int(t_surf)}_RH={int(rh_percent)}.npy"
        ),
        f"uwxyzr_COD={int(cod)}_th0={int(solar_zenith)}_Ts={int(t_surf)}_RH={int(rh_percent)}.npy",
    ]


def find_rtm_file(rtm_dir, cod, solar_zenith, surface, aod, t_surf, rh_percent):
    for filename in rtm_filename_candidates(cod, solar_zenith, surface, aod, t_surf, rh_percent):
        filepath = os.path.join(rtm_dir, filename)
        if os.path.exists(filepath):
            return filepath
    return None


def generate_luts(
    rtm_dir,
    out_dir,
    channels,
    cod_grid,
    solar_zeniths,
    surface,
    aod,
    t_surf,
    rh_percent,
    n_bundles,
    allow_missing=False,
):
    file_dir = os.path.join(REPO_ROOT, "FY4A_data")
    ref_lam, ref_E_nu = load_toa_spectrum()
    dnu = 3
    nu = np.arange(2500, 35000, dnu)
    nu_input = FY4A_calinu(nu, DEFAULT_CHANNELS, file_dir, dnu=dnu)

    for cod in cod_grid:
        cod = int(cod)
        print(f"Processing COD={cod}...")
        Ang_D = []

        for solar_zenith in solar_zeniths:
            solar_zenith = int(solar_zenith)
            filepath = find_rtm_file(rtm_dir, cod, solar_zenith, surface, aod, t_surf, rh_percent)

            if filepath is None:
                expected = ", ".join(
                    rtm_filename_candidates(cod, solar_zenith, surface, aod, t_surf, rh_percent)
                )
                message = f"Missing RTM file in {rtm_dir}; tried: {expected}"
                if not allow_missing:
                    raise FileNotFoundError(message)
                print(f"  Warning: {message}; using zero ADM placeholders for this SZA.")
                Ang_D.extend([np.zeros((18, 18)) for _ in channels])
                continue

            results = np.load(filepath, allow_pickle=True).item()
            uw_rxyz_M = results.get("uw_rxyz_M")
            if uw_rxyz_M is None:
                raise KeyError(f"{filepath} does not contain 'uw_rxyz_M'")

            for channel in channels:
                print(f"  SZA={solar_zenith} {channel}")
                srf, nu_channel = get_calibration_srf(channel, file_dir, nu=nu, dnu=dnu)
                F_dw_os_channel = -np.interp(-nu_channel, -1e4 / ref_lam, ref_E_nu)
                F_dw_os_srf = F_dw_os_channel * srf

                nu_idx = np.nonzero(np.isin(nu_input, nu_channel))[0]
                result_subset = [uw_rxyz_M[i] for i in nu_idx]
                R_c = cal_mono_R(
                    result_subset,
                    solar_zenith,
                    F_dw_os_srf,
                    N_bundles=n_bundles,
                )
                Ang_D.append(R_c)

        saveLUT(
            Ang_D,
            cod,
            dir=out_dir,
            channels=channels,
            solar_zeniths=solar_zeniths,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FY4A AGRI channel ADM LUTs from saved RTM photon directions."
    )
    parser.add_argument("--rtm-dir", default=get_rtm_results_dir())
    parser.add_argument("--out-dir", default=os.path.join(CURRENT_DIR, "LUT"))
    parser.add_argument("--surface", default="Case2")
    parser.add_argument("--aod", type=float, default=DEFAULT_AOD)
    parser.add_argument("--t-surf", type=int, default=DEFAULT_T_SURF)
    parser.add_argument("--rh-percent", type=int, default=DEFAULT_RH_PERCENT)
    parser.add_argument("--n-bundles", type=int, default=DEFAULT_N_BUNDLES)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    parser.add_argument("--cod-grid", nargs="+", type=float, default=DEFAULT_COD_GRID.tolist())
    parser.add_argument("--solar-zeniths", nargs="+", type=float, default=DEFAULT_SOLAR_ZENITHS.tolist())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_luts(
        rtm_dir=args.rtm_dir,
        out_dir=args.out_dir,
        channels=args.channels,
        cod_grid=np.asarray(args.cod_grid),
        solar_zeniths=np.asarray(args.solar_zeniths),
        surface=args.surface,
        aod=args.aod,
        t_surf=args.t_surf,
        rh_percent=args.rh_percent,
        n_bundles=args.n_bundles,
        allow_missing=args.allow_missing,
    )
