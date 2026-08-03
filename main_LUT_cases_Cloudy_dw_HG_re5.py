import argparse
import math
import os
import socket
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import numpy as np

from LBL_funcs_shortwave import LBL_shortwave


def default_output_dir() -> Path:
    hostname = socket.gethostname()
    if hostname == "user-Super-Server":
        return Path("/home/dengnan/data/RTM/LUTcases/HG/cloudy_dw_re5/")
    if hostname == "user-MS-7D30":
        return Path("/mnt/dengnan/LUTcases/HG/cloudy_dw_re5/")
    if hostname == "h07mgt1":
        return Path("/puhome/22117689r/projects/Shortwave_MCRTM/LUTcases/HG/cloudy_dw_re5/")
    return Path(__file__).resolve().parent / "RTM" / "LUTcases" / "HG" / "cloudy_dw_re5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure-HG cloudy downwelling LUT cases with cloud re=5 um.")
    parser.add_argument("--out-dir", default=str(default_output_dir()), help="Directory for Results_*.npy files.")
    parser.add_argument("--n-bundles", type=int, default=1000, help="Photon bundles per wavenumber.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of new cases to run.")
    parser.add_argument("--rh", type=float, nargs="*", default=[10.0, 50.0, 90.0], help="Relative humidity values in percent.")
    parser.add_argument("--ts", type=float, nargs="*", default=[270.0, 285.0, 300.0, 320.0], help="Surface temperatures in K.")
    parser.add_argument("--cod", type=float, nargs="*", default=[0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0], help="COD grid.")
    parser.add_argument("--th0", type=float, nargs="*", default=[0.0, 15.0, 30.0, 45.0, 60.0, 65.0], help="Solar zenith angles in degrees.")
    parser.add_argument("--albset", type=int, nargs="*", default=[0, 1, 2, 3, 4], help="Albedo set indexes to run.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the full case list into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only this zero-based shard index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    ph_cdf_cld = False
    ph_cdf_aer = False
    delta_m = False
    n_layer = 54
    dnu = 3
    nu = np.arange(2500, 35000, dnu)
    molecules = ["H2O", "CO2", "O3", "N2O", "CH4", "O2", "N2"]
    vmr0 = {
        "H2O": 0.03,
        "CO2": 399.5 / 10**6,
        "O3": 50 / 10**9,
        "N2O": 328 / 10**9,
        "CH4": 1834 / 10**9,
        "O2": 2.09 / 10,
        "N2": 7.81 / 10,
    }
    albedo_sets = {
        0: [0.0145, 0.0288, 0.4156, 0.2031, 0.0641],
        1: [0.0251, 0.0472, 0.3922, 0.2218, 0.0897],
        2: [0.0407, 0.0745, 0.3575, 0.2494, 0.1275],
        3: [0.0673, 0.1207, 0.2988, 0.2961, 0.1916],
        4: [0.0938, 0.1669, 0.2401, 0.3428, 0.2555],
    }

    surface = "MODIS"
    surface_id = 3
    phi0 = 0.0
    del_angle = 0.5 / 180 * math.pi
    beta_v = np.array([0.0])
    phi_v = np.array([0.0])
    aod_v = [0.1243]
    kap_v = [[10, 11, 12]]
    total_cases = (
        len(args.albset) * len(args.ts) * len(args.rh)
        * len(aod_v) * len(kap_v) * len(args.cod) * len(args.th0)
    )
    print(f"Running cloudy DW HG re=5 LUT: shard {args.shard_index}/{args.num_shards}; total cases={total_cases}; output={out_dir}")

    run_count = 0
    case_index = 0

    for alb_idx in args.albset:
        alb_set = albedo_sets[int(alb_idx)]
        inputs_main = {
            "N_layer": n_layer,
            "N_bundles": args.n_bundles,
            "nu": nu,
            "molecules": molecules,
            "vmr0": vmr0,
            "model": "AFGL midlatitude summer",
            "cld_model": "default_re5",
            "period": "day",
            "spectral": "SW",
            "surface_id": surface_id,
            "white_albedo": alb_set,
            "black_albedo": alb_set,
            "BRDF_param": [0] * 15,
            "alt": 0,
            "Ph_cdf_cld": ph_cdf_cld,
            "Ph_cdf_aer": ph_cdf_aer,
            "deltaM": delta_m,
            "escape_alpha": 0.0,
            "escape_cone_deg": -1.0,
            "escape_probability_mode": "none",
            "scale_deltaM_g": False,
        }
        for ts in args.ts:
            for rh_percent in args.rh:
                for aod in aod_v:
                    for kap in kap_v:
                        for cod in args.cod:
                            for th0 in args.th0:
                                current_case_index = case_index
                                case_index += 1
                                if current_case_index % args.num_shards != args.shard_index:
                                    continue
                                theta0 = th0 / 180 * math.pi
                                angles = {
                                    "theta0": theta0,
                                    "phi0": phi0,
                                    "del_angle": del_angle,
                                    "beta": beta_v,
                                    "phi": phi_v,
                                    "isTilted": False,
                                }
                                x0 = 120.0 * np.tan(theta0) * np.cos(phi0)
                                y0 = 120.0 * np.tan(theta0) * np.sin(phi0)
                                finite_pp = {
                                    "x0": -x0,
                                    "y0": -y0,
                                    "R_pp": 1,
                                    "is_pp": False,
                                    "th0": theta0,
                                    "phi0": phi0,
                                    "del_angle": del_angle,
                                }
                                file_name = (
                                    f"Results_{surface}_AlbSet{alb_idx}_AOD={aod:.2f}_COD={cod:g}_kap={kap}"
                                    f"_th0={th0:g}_Ts={ts:g}_RH={int(rh_percent)}"
                                    "_re=5_meth=HG_escape=none"
                                )
                                output_path = out_dir / f"{file_name}.npy"
                                if output_path.exists():
                                    print(f"{output_path} exists, continue.")
                                    continue
                                if args.limit is not None and run_count >= args.limit:
                                    print(f"Reached --limit={args.limit}.")
                                    return
                                print(f"Start MonteCarlo: {output_path.name}")
                                start_time = time.time()
                                properties = {"rh0": rh_percent / 100.0, "T_surf": ts, "AOD": aod, "COD": cod, "kap": kap}
                                out1, out2 = LBL_shortwave(properties, inputs_main, angles, finite_pp)
                                np.save(output_path, out1)
                                del out1, out2
                                run_count += 1
                                print("CPU time:", time.time() - start_time)


if __name__ == "__main__":
    main()
