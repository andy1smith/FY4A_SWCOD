import argparse
import math
import os
import socket
import time

import numpy as np

from LBL_funcs_shortwave import *
from fun_nearealtime_RTM import *

import warnings
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Run cloudy downwelling dM/f LUT cases with cloud re=5 um.')
    parser.add_argument('--case-start', type=int, default=0, help='Inclusive zero-based case index to start from.')
    parser.add_argument('--case-stop', type=int, default=None, help='Exclusive zero-based case index to stop at.')
    args = parser.parse_args()

    meth = 'dM'
    theta_trunc_cld = 3
    escape_mode = 'f'
    cld_re_um = 5
    hostname = socket.gethostname()
    if hostname == 'user-Super-Server':
        file_dir = '/home/dengnan/data/RTM/LUTcases/dM/LUT_cloud_dw_re5/'
    elif hostname == 'user-MS-7D30':
        file_dir = '/mnt/dengnan/LUTcases/dM/LUT_cloud_dw_re5/'
    elif hostname == 'h07mgt1':
        file_dir = '/puhome/22117689r/projects/FY4A_SWCOD/LUTcases/dM/LUT_cloud_dw_re5/'
    elif hostname == 'dengnans-MacBook-Pro.local':
        file_dir = '/Users/dengnan/Documents/git_store/FY4A_SWCOD/RTM/LUTcases/dM/LUT_cloud_dw_re5/'
    else:
        raise ValueError(f'Unknown server: {hostname}. Please set file_dir manually.')
    os.makedirs(file_dir, exist_ok=True)

    Ph_cdf_cld = False
    Ph_cdf_aer = False
    deltaM = True

    N_layer = 54
    N_bundles = 1000
    dnu = 3
    nu = np.arange(2500, 35000, dnu)

    molecules = ['H2O', 'CO2', 'O3', 'N2O', 'CH4', 'O2', 'N2']
    vmr0 = {
        'H2O': 0.03,
        'CO2': 399.5 / 10**6,
        'O3': 50 / 10**9,
        'N2O': 328 / 10**9,
        'CH4': 1834 / 10**9,
        'O2': 2.09 / 10,
        'N2': 7.81 / 10,
    }
    model = 'AFGL midlatitude summer'
    cld_model = f'default_re{cld_re_um}'
    period = 'day'
    spectral = 'SW'
    alt = 0

    surface_types = {'Lambert': 0, 'CSP': 1, 'BRDF': 2, 'MODIS': 3, 'Case2': 0}
    surface = 'MODIS'
    surface_v = [surface]
    surface_id_v = [surface_types[surface]]
    brdf_param = [0] * 15

    rh0_v = np.array([10], dtype='float') / 100.0
    T_surf_v = np.array([270, 285, 300, 320])
    AOD_v = np.array([0.1243])
    COD_v = [0.5, 1, 3, 5, 10, 20, 50]
    kap_v = [[10, 11, 12]]

    # Cloud_LUT_dMf downwelling albedo sets used by preprocess_surrogate_cloudy_dw.py.
    albedo_sets = [
        [0.0145, 0.0288, 0.4156, 0.2031, 0.0641],
        [0.0251, 0.0472, 0.3922, 0.2218, 0.0897],
        [0.0407, 0.0745, 0.3575, 0.2494, 0.1275],
        [0.0673, 0.1207, 0.2988, 0.2961, 0.1916],
        [0.0938, 0.1669, 0.2401, 0.3428, 0.2555],
    ]

    th0_v = np.array([0, 15, 30, 45, 60, 65])
    theta0_v = th0_v / 180 * math.pi
    phi0 = 0 / 180 * math.pi
    del_angle = 0.5 / 180 * math.pi
    beta_v = np.array([0]) / 180 * math.pi
    phi_v = phi0 + np.array([0]) / 180 * math.pi
    isTilted = False

    x0_v = 120.0 * np.tan(theta0_v) * np.cos(phi0)
    y0_v = 120.0 * np.tan(theta0_v) * np.sin(phi0)
    R_pp = 1
    is_pp = False
    dx_v = np.array([0.0])

    total_cases = (
        len(surface_v) * len(albedo_sets) * len(T_surf_v) * len(rh0_v)
        * len(AOD_v) * len(kap_v) * len(COD_v) * len(theta0_v) * len(dx_v)
    )
    case_stop = total_cases if args.case_stop is None else args.case_stop
    if args.case_start < 0 or case_stop < args.case_start or case_stop > total_cases:
        raise ValueError(f'Invalid case range [{args.case_start}, {case_stop}) for {total_cases} cases.')
    print(
        f'Running Cloud_LUT_dMf re={cld_re_um}um: case range [{args.case_start}, {case_stop}) '
        f'of {total_cases}; output={file_dir}'
    )

    case_index = 0
    for iSurf in range(len(surface_v)):
        for iALB, alb_set in enumerate(albedo_sets):
            black_albedo = alb_set
            white_albedo = alb_set
            inputs_main = {
                'N_layer': N_layer,
                'N_bundles': N_bundles,
                'nu': nu,
                'molecules': molecules,
                'vmr0': vmr0,
                'model': model,
                'cld_model': cld_model,
                'period': period,
                'spectral': spectral,
                'surface_id': surface_id_v[iSurf],
                'white_albedo': white_albedo,
                'black_albedo': black_albedo,
                'BRDF_param': brdf_param,
                'alt': alt,
                'Ph_cdf_cld': Ph_cdf_cld,
                'Ph_cdf_aer': Ph_cdf_aer,
                'deltaM': deltaM,
                'theta_trunc_cld': theta_trunc_cld,
                'escape_alpha': 1.0,
                'escape_cone_deg': -1.0,
                'escape_probability_mode': escape_mode,
                'scale_deltaM_g': True,
            }
            for iT in range(len(T_surf_v)):
                for iRH in range(len(rh0_v)):
                    for iAOD in range(len(AOD_v)):
                        for iKAP in range(len(kap_v)):
                            for iCOD in range(len(COD_v)):
                                properties = {
                                    'rh0': rh0_v[iRH],
                                    'T_surf': T_surf_v[iT],
                                    'AOD': AOD_v[iAOD],
                                    'COD': COD_v[iCOD],
                                    'kap': kap_v[iKAP],
                                }
                                for iTH in range(len(theta0_v)):
                                    angles = {
                                        'theta0': theta0_v[iTH],
                                        'phi0': phi0,
                                        'del_angle': del_angle,
                                        'beta': beta_v,
                                        'phi': phi_v,
                                        'isTilted': isTilted,
                                    }
                                    for idx in range(len(dx_v)):
                                        finitePP = {
                                            'x0': -x0_v[iTH] + dx_v[idx],
                                            'y0': -y0_v[iTH],
                                            'R_pp': R_pp,
                                            'is_pp': is_pp,
                                            'th0': theta0_v[iTH],
                                            'phi0': phi0,
                                            'del_angle': del_angle,
                                        }
                                        if case_index < args.case_start or case_index >= case_stop:
                                            case_index += 1
                                            continue
                                        fileName1 = (
                                            'Results_{}_AlbSet{}_AOD={:.2f}_COD={}_kap={}_th0={}_Ts={}_RH={}'
                                            '_re={}_meth={}_theta_trunc_cld={}_escape={}'
                                        ).format(
                                            surface_v[iSurf], iALB, AOD_v[iAOD], COD_v[iCOD], kap_v[iKAP],
                                            th0_v[iTH], T_surf_v[iT], int(rh0_v[iRH] * 100), cld_re_um,
                                            meth, theta_trunc_cld, escape_mode,
                                        )
                                        output_path = os.path.join(file_dir, fileName1 + '.npy')
                                        if os.path.exists(output_path):
                                            print(f'{output_path} exists, continue.')
                                            case_index += 1
                                            continue
                                        print(f'Start MonteCarlo once. case_index={case_index}')
                                        start_time = time.time()
                                        out1, out2 = LBL_shortwave(properties, inputs_main, angles, finitePP)
                                        end_time = time.time()
                                        print('CPU time:', end_time - start_time)
                                        np.save(output_path[:-4], out1)
                                        del out1, out2
                                        case_index += 1


if __name__ == '__main__':
    main()
