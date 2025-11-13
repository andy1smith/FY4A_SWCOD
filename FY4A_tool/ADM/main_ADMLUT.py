import os,sys
code_dir=os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(code_dir)
os.chdir(code_dir)
#from scope_sw import * # when send the processing part of code to other, update the func.py.
from fun_nearealtime_RTM import *

def nearealtime_RTM(sun_zen, COD_guess, T_a, RH, channels, file_dir, bandmode, N_bundles):
    # Round values to two decimal places
    sun_zen = round(sun_zen)
    COD_guess = round(COD_guess)
    T_a = round(T_a)
    RH = round(RH)

    uw_rxyz_file = f"uwxyzr_COD={COD_guess}_th0={sun_zen}_Ta={T_a}_RH={RH}.npy"
    # bandmode = 'FY4A' # FY4A
    # print(bandmode)
    if N_bundles == 1000:
        if bandmode == 'FY4A':
            uw_rxyz_path = os.path.join(file_dir, 'RTM/channels', uw_rxyz_file)
        else:
            uw_rxyz_path = os.path.join(file_dir, 'RTM/fullspectrum', uw_rxyz_file)
    if N_bundles == 10000:
        if sys.platform != 'darwin':
            file_dir = './'
        if bandmode == 'FY4A':
            uw_rxyz_path = os.path.join(file_dir, 'RTM_10000/channels', uw_rxyz_file)
        else:
            uw_rxyz_path = os.path.join(file_dir, 'RTM_10000/fullspectrum', uw_rxyz_file)

    if not os.path.exists(uw_rxyz_path):
        print(f"File {uw_rxyz_file} not found. Running RTM...")
        run_RTM(sun_zen, COD_guess, T_a, RH, file_dir, channels, bandmode, N_bundles)

if __name__ == "__main__":
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    COD_v = np.concatenate([np.linspace(0, 20, 11), np.linspace(25, 50, 6)])
    Sun_Zen_v = np.array([45,60])  # Dl[0,15,30],polyuHPC[45,60]
    T_a = 294  # +10
    rh0_v = np.array([60])  # /100
    N_bundles = 10000
    bandmode = "FY4A"
    file_dir = "FY4A_data"
    for Sun_Zen in Sun_Zen_v:
        for iCOD in COD_v:
            COD_guess = iCOD
            for RH in rh0_v:
                nearealtime_RTM(Sun_Zen, COD_guess, T_a, RH,
                                channels, file_dir=file_dir, bandmode=bandmode, N_bundles=N_bundles)
