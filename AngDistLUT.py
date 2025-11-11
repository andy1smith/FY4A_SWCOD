import numpy as np
import h5py  # For efficient storage
import pandas as pd
from scipy.interpolate import interp1d
import math,os
from fun_nearealtime_RTM import get_calibration_srf
from SCOPE_func import find_bin_indices
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ================================
# SAVE LUT FUNCTION
# ================================

def FY4A_calinu(nu, channels, file_dir, dnu = 3, sensor='FY4A'):
    # convert nu to AGRI device nu range. return cm-1.
    nus = set()
    if sensor == 'FY4A' :
        dirpath = file_dir + 'AGRI_calibration/'
    else :
        print('!!! Lack sensor calibration')
    for channel in channels:
        # load ABI calibration data
        channel_number = int(channel[-2:])
        channel_srf = os.path.join(
            dirpath,
            'FY4A_AGRI_SRF_ch{:d}.txt'.format(channel_number)
        )
        calibration = np.loadtxt(channel_srf, delimiter=',', skiprows=1)
        # calibration_wl = calibration[:, 0]  # wavelength [um]
        calibration_nu = calibration[:, 1]  # cm-1
        # calibration_srf = calibration[:, 2] # relative SRF [-]
        # reverse order (so wavenumber is increasing)
        calibration_nu = calibration_nu[::-1]
        # keep the wavenumber within range
        channel_mask = (nu >= calibration_nu.min()) & (nu <= calibration_nu.max())
        nus.update(nu[channel_mask])
        #nus.update(calibration_nu[::dnu])

    nus = np.array(sorted(nus))
    return nus


def read_data(file, channel, Sun_Zen, local_zen, rela_azi, N_bundles=10000,
               iCOD=10, ang=32):

    # fdir = "/mnt/dengnan/RTM_10000/"
    fdir = "/Volumes/DN1T_SSD/data/RTM_10000/"
    filename = '/' + f'uwxyzr_COD={iCOD}_th0={ang}_Ta=299_RH=61.npy'
    results = np.load(file, allow_pickle=True).item()
    uw_rxyz_M = results.get('uw_rxyz_M')
    file_dir = "./FY4A_data/"

    dnu = 3
    nu = np.arange(2500, 35000, dnu)

    channel_6c = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    nu_input = FY4A_calinu(nu, channel_6c, file_dir, dnu=3)

    data = np.genfromtxt('data/profiles/SolarTOA.csv', delimiter=',')
    ref_lam = data[:, 0]
    ref_E = data[:, 1]
    ref_E_nu = -ref_E * ref_lam ** 2 / 1e4

    channel_number = int(channel[-2:])
    dirpath = file_dir + 'FY4A-R_ABI_FM2_SRF_CWG/'
    channel_srf = os.path.join(dirpath, f'FY4A-R_ABI_FM2_SRF_CWG_ch{channel_number}.txt')
    calibration = np.genfromtxt(channel_srf, skip_header=2)
    calibration_nu = calibration[:, 1][::-1]
    calibration_srf = calibration[:, 2][::-1]

    nu_channel = FY4A_calinu(nu, [channel], file_dir, dnu=3)
    F_dw_os_channel = -np.interp(-nu_channel, -1e4 / ref_lam, ref_E_nu)
    srf = np.interp(nu_channel, calibration_nu, calibration_srf)
    F_dw_os_SRF = np.multiply(F_dw_os_channel, srf)

    nu_idx = np.nonzero(np.isin(nu_input, nu_channel))[0]
    result = [uw_rxyz_M[i] for i in nu_idx]


    Rc_rtm, R_c = cal_mono_Intensity(result, Sun_Zen, nu_input[nu_idx], F_dw_os_SRF,
                                     local_zen, rela_azi,
                                     N_bundles=10000, is_flux=False, Norm=True, dirc='UW')
        #is_norm = check_normalization(R_c) / np.pi
    return Rc_rtm, R_c


def save_svd_lut(channels, solar_zeniths, svd_data, filename='svd_lut.h5'):
    """
    Save SVD components to HDF5 file with channel-based grouping

    Args:
        channels: List of channel identifiers
        solar_zeniths: Array of solar zenith angles used in LUT
        svd_data: Dictionary of format:
            {
                channel: {
                    'U': np.array (n_zeniths, n_theta, rank),
                    'S': np.array (n_zeniths, rank),
                    'VT': np.array (n_zeniths, rank, n_phi)
                }
            }
        filename: Output filename
    """
    with h5py.File(filename, 'w') as f:
        # Save solar zenith angles (shared across channels)
        f.create_dataset('solar_zeniths', data=solar_zeniths)

        for channel in channels:
            grp = f.create_group(f'{channel}')
            grp.create_dataset('U', data=svd_data[channel]['U'])
            grp.create_dataset('S', data=svd_data[channel]['S'])
            grp.create_dataset('VT', data=svd_data[channel]['VT'])
    print(f"SVD LUT saved to {filename}")

def cal_mono_Intensity(rxyz_M, theta0, nu, F_dw_os, local_zen, rela_azi, N_bundles=10000,
                       is_flux=False, Norm=False, dirc='UW', bin_scale=1):  # Z_csky
    from LBL_funcs_inclined import theta_phi

    """
    bins_theta: local zenith angle
    bin_phi: relative difference between the angle of solar azimuth and local zimuth
    solid anlge (bin_theta,bin_phi) determined to the intensity at an angle of (satellite local solar).
    rxyz_M : the vector of each photon fall on the ground. Each band saved many photons,
            we sum the number of photon, multiple with ratio to get  the flux.
    F_dw_os: the downwelling flux at TOA on each nu
    """
    theta0 = theta0 / 180 * math.pi
    phi0 = 0 / 180 * math.pi

    # bin_scale = 1
    d_th = 2 * bin_scale
    d_phi = 5 * bin_scale
    # FY4A solar zenith 45, local zenith angle 45, relative azimuth difference angle 45
    bins_theta = np.arange(0, 91, d_th)
    # symmetric, so we change (-180, 180) to (0, 180)
    bins_phi = np.arange(-180, 181, d_phi)
    #bins_phi = np.hstack((np.arange(0, 170 + 5, 5), np.arange(170, 180 + 1, 1)))
    fw_rx, fw_ry, fw_rz, uw_rx, uw_ry, uw_rz = [np.zeros((N_bundles + 10, len(nu))) * np.nan for _ in range(6)]
    H = np.zeros((len(bins_theta) - 1, len(bins_phi) - 1))
    for i in range(len(rxyz_M)):
        fw_rxyz = rxyz_M[i]
        N_dw = len(fw_rxyz)
        fw_rx[0:N_dw, i] = np.array([x[0] for x in fw_rxyz])
        fw_ry[0:N_dw, i] = np.array([x[1] for x in fw_rxyz])
        fw_rz[0:N_dw, i] = np.array([x[2] for x in fw_rxyz])
        if dirc == 'UW':
            theta_v, phi_v = theta_phi(fw_rx[:, i], fw_ry[:, i], fw_rz[:, i])
        else:  # DW
            theta_v, phi_v = theta_phi(fw_rx[:, i], fw_ry[:, i], -fw_rz[:, i])
        ind = np.isnan(phi_v)
        theta_v = theta_v[~ind]
        phi_v = phi_v[~ind] - phi0
        phi_v[phi_v > math.pi] -= 2 * math.pi
        H_i, xedges, yedges = np.histogram2d(np.rad2deg(theta_v), np.rad2deg(phi_v), bins=(bins_theta, bins_phi))
        H += H_i * np.cos(theta0) * F_dw_os[i] * 3 / N_bundles
    theta_, phi_ = np.meshgrid(xedges[0:-1], yedges[0:-1])
    theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
    F = np.sum(H)
    if not is_flux: # approximation of domega
        ths = np.deg2rad(theta_.T + d_th / 2)
        #print(np.sum(H))
        H /= 0.5 * np.sin(2 * ths)  # I=dF/dw, dF = H/cos(ths), dw=sin(ths)d_thd_phi. 0.5*sin(2ths)=cos(ths)sin(ths)
    H /= np.deg2rad(d_th) * np.deg2rad(d_phi)  # per solid angle, in the direction of beam
    R = H*np.pi / F
        # Integrated intensity over phi, for each theta bin
        # H_theta = np.sum(H, axis=1) * np.deg2rad(d_phi)
        # H_theta_6c = np.sum(H, axis=1) * np.deg2rad(d_phi)
    return R[theta_idx, phi_idx],R
# ================================
# LOAD AND INTERPOLATE FUNCTION
# ================================
def load_and_interpolate_whole(filename, channel, target_zenith):
    """
    Load SVD components and interpolate to target zenith angle

    Args:
        filename: HDF5 file path
        channel: Target channel identifier
        target_zenith: Desired solar zenith angle (degrees)

    Returns:
        U_interp: (n_theta, rank) interpolated matrix
        S_interp: (rank,) interpolated singular values
        VT_interp: (rank, n_phi) interpolated matrix
    """
    with h5py.File(filename, 'r') as f:
        # Load solar zeniths
        solar_zeniths = f['solar_zeniths'][:]
        channel_group = f[f'{channel}']

        U = channel_group['U'][:]
        S = channel_group['S'][:]
        VT = channel_group['VT'][:]

    # Create interpolation functions
    U_interp = np.zeros((U.shape[1], U.shape[2]))
    S_interp = np.zeros(S.shape[1])
    VT_interp = np.zeros((VT.shape[1], VT.shape[2]))

    interp_kind = 'linear'
    if target_zenith > 30:
        interp_kind = "quadratic"
    # Interpolate each component
    for r in range(U.shape[2]):  # For each rank component
        # Interpolate U components
        # U is (n_zeniths, n_theta, rank), interpolate between zenith angles
        for theta_idx in range(U.shape[1]):
            interp_fn = interp1d(solar_zeniths, U[:, theta_idx, r],
                                 kind=interp_kind, fill_value="extrapolate")
            U_interp[theta_idx, r] = interp_fn(target_zenith)

        # Interpolate S values
        interp_fn = interp1d(solar_zeniths, S[:, r][:,0], kind='linear', fill_value="extrapolate")
        # Interpolate the r-th singular value
        S_interp[r] = interp_fn(target_zenith)

        # Interpolate VT components
        for phi_idx in range(VT.shape[2]):
            interp_fn = interp1d(solar_zeniths, VT[:, r, phi_idx],
                                 kind='linear', fill_value="extrapolate")
            VT_interp[r, phi_idx] = interp_fn(target_zenith)

    return U_interp, S_interp, VT_interp

def load_and_interpolate(filename, channel, target_zenith, theta_, phi_):
    """
    Load SVD components and interpolate to target zenith angle

    Args:
        filename: HDF5 file path
        channel: Target channel identifier
        target_zenith: Desired solar zenith angle (degrees)

    Returns:
        U_interp: (3, rank) interpolated matrix (for 3 theta indices)
        S_interp: (rank,) interpolated singular values
        VT_interp: (rank, 3) interpolated matrix (for 3 phi indices)
    """
    with h5py.File(filename, 'r') as f:
        solar_zeniths = f['solar_zeniths'][:]
        channel_group = f[f'{channel}']
        U = channel_group['U'][:]
        S = channel_group['S'][:]
        VT = channel_group['VT'][:]

    # Define index ranges, ensuring they are within bounds
    theta_indices = np.clip(np.arange(theta_ - 1, theta_ + 2), 0, U.shape[1] - 1)
    phi_indices = np.clip(np.arange(phi_ - 1, phi_ + 2), 0, VT.shape[2] - 1)
    rank = U.shape[2]

    U_interp = np.zeros((3, rank))
    S_interp = np.zeros(rank)
    VT_interp = np.zeros((rank, 3))

    for r in range(rank):
        # Interpolate U for 3 theta indices
        for i, theta_idx in enumerate(theta_indices):
            interp_fn = interp1d(solar_zeniths, U[:, theta_idx, r], kind='linear', fill_value="extrapolate")
            U_interp[i, r] = interp_fn(target_zenith)

        # Interpolate S values
        interp_fn = interp1d(solar_zeniths, S[:, r][:,0], kind='linear', fill_value="extrapolate")
        S_interp[r] = interp_fn(target_zenith)

        # Interpolate VT for 3 phi indices
        for j, phi_idx in enumerate(phi_indices):
            interp_fn = interp1d(solar_zeniths, VT[:, r, phi_idx], kind='linear', fill_value="extrapolate")
            VT_interp[r, j] = interp_fn(target_zenith)

    return U_interp, S_interp, VT_interp

from scipy.ndimage import gaussian_filter1d

def svd_rank_k_approx(matrix, rank=2, Gau_smooth = True):
    """
    Perform a rank-k approximation of a 2D matrix using SVD.

    Parameters:
        matrix (ndarray): Input 2D array (e.g. R(phi|theta))
        rank (int): Desired rank for approximation

    Returns:
        matrix_approx (ndarray): Rank-k approximated matrix
        U_k (ndarray): Left singular vectors (truncated)
        S_k (ndarray): Singular values (truncated)
        VT_k (ndarray): Right singular vectors (truncated)
    """
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    U_k = U[:, :rank]
    S_k = np.diag(S[:rank])
    VT_k = VT[:rank, :]
    if Gau_smooth:
    # Apply Gaussian smoothing to each singular vector (column of U_k, row of VT_k)
        U_k = np.array([gaussian_filter1d(U_k[:, i], sigma=1) for i in range(rank)]).T  # shape: (n_samples, rank)
        VT_k = np.array([gaussian_filter1d(VT_k[i, :], sigma=1) for i in range(rank)])  # shape: (rank, n_features)

    matrix_approx = U_k @ S_k @ VT_k
    return matrix_approx, U_k, S_k, VT_k

def check_normalization(R_nor, d_theta=2, d_phi=5):
    """
    Check if H_nor is properly normalized.

    Returns:
        integral_value (float): Should be ~1.0 if normalized.
    """
    n_theta, n_phi = R_nor.shape

    # Bin centers in radians
    theta_edges = np.linspace(0, 90, n_theta + 1)
    theta_centers = np.radians(theta_edges[:-1] + d_theta / 2)
    cos_theta = np.cos(theta_centers)
    sin_theta = np.sin(theta_centers)
    sin_theta_2d = sin_theta[:, np.newaxis]
    cos_theta_2d = cos_theta[:, np.newaxis]

    dtheta_rad = np.radians(d_theta)
    dphi_rad = np.radians(d_phi)

    # Compute the integral
    integral_value = np.sum(R_nor * cos_theta_2d* sin_theta_2d) * dtheta_rad * dphi_rad
    return integral_value

def theta_phi(rx,ry,rz):
    """
    Computes incident angles of (rx,ry,rz) on a horizontal surface.

    Parameters
    ----------
    rx, ry, rz : (N_p,) array_like
        The traveling direction of photons (rx,ry,rz).

    Returns
    -------
    theta: (N_p,) array_like
        Incident zenith angle on a horizontal surface [rad].
    phi : (N_p,) array_like
        Incident azimuth angle on a horizontal surface [rad].

    """
    theta=np.arccos(rz) # in [0,pi]
    sin_th=np.sqrt(1.-rz**2.)
    # Bug fixed: when theta = 0, sin_th = nan. Nancy 2024.7.16
    p = np.random.uniform(low=-np.pi, high=np.pi, size=theta.shape[0])
    cosP = rx / sin_th  # in [0,pi]
    cosP = np.clip(cosP, -1, 1) # bug 2 fixed: cosP may be slightly larger than 1, then phi = nan
    phi = np.arccos(cosP)  # in [0,pi]
    phi[rz==1] = p[rz==1]
    ind=(ry*sin_th<0) # in [pi,2*pi]
    if (ind.size!=0):
        phi[ind]=2*math.pi-phi[ind]
    theta[rx ** 2 + ry ** 2 + rz ** 2 == 0] = np.nan
    phi[rx ** 2 + ry ** 2 + rz ** 2 == 0] = np.nan
    return theta, phi


def cal_mono_R(rxyz_M, theta0, nu, F_dw_os, N_bundles=1000,
                       is_flux=False, dirc='UW', bin_scale=1):  # Z_csky
    """
    bins_theta: local zenith angle
    bin_phi: relative difference between the angle of solar azimuth and local zimuth
    solid anlge (bin_theta,bin_phi) determined to the intensity at an angle of (satellite local solar).
    rxyz_M : the vector of each photon fall on the ground. Each band saved many photons, 
            we sum the number of photon, multiple with ratio to get  the flux.
    F_dw_os: the downwelling flux at TOA on each nu
    """
    theta0 = theta0 / 180 * math.pi
    phi0 = 0 / 180 * math.pi

    # bin_scale = 1
    d_th = 2 * bin_scale
    d_phi = 5 * bin_scale
    # FY4A solar zenith 45, local zenith angle 45, relative azimuth difference angle 45
    bins_theta = np.arange(0, 91, d_th)
    # symmetric, so we change (-180, 180) to (0, 180)
    bins_phi = np.arange(-180, 181, d_phi)
    #bins_phi = np.hstack((np.arange(0, 170 + 5, 5), np.arange(170, 180 + 1, 1)))
    fw_rx, fw_ry, fw_rz, uw_rx, uw_ry, uw_rz = [np.zeros((N_bundles + 10, len(nu))) * np.nan for _ in range(6)]
    H = np.zeros((len(bins_theta) - 1, len(bins_phi) - 1))
    for i in range(len(rxyz_M)):
        fw_rxyz = rxyz_M[i]
        N_dw = len(fw_rxyz)
        fw_rx[0:N_dw, i] = np.array([x[0] for x in fw_rxyz])
        fw_ry[0:N_dw, i] = np.array([x[1] for x in fw_rxyz])
        fw_rz[0:N_dw, i] = np.array([x[2] for x in fw_rxyz])
        if dirc == 'UW':
            theta_v, phi_v = theta_phi(fw_rx[:, i], fw_ry[:, i], fw_rz[:, i])
        else:  # DW
            theta_v, phi_v = theta_phi(fw_rx[:, i], fw_ry[:, i], -fw_rz[:, i])
        ind = np.isnan(phi_v)
        theta_v = theta_v[~ind]
        phi_v = phi_v[~ind] - phi0
        phi_v[phi_v > math.pi] -= 2 * math.pi
        H_i, xedges, yedges = np.histogram2d(np.rad2deg(theta_v), np.rad2deg(phi_v), bins=(bins_theta, bins_phi))
        H += H_i * np.cos(theta0) * F_dw_os[i] * 3 / N_bundles
    theta_, phi_ = np.meshgrid(xedges[0:-1], yedges[0:-1])

    F = np.sum(H)
    if not is_flux: # approximation of domega
        ths = np.deg2rad(theta_.T + d_th / 2)
        #print(np.sum(H))
        H /= 0.5 * np.sin(2 * ths)  # I=dF/dw, dF = H/cos(ths), dw=sin(ths)d_thd_phi. 0.5*sin(2ths)=cos(ths)sin(ths)
    H /= np.deg2rad(d_th) * np.deg2rad(d_phi)  # per solid angle, in the direction of beam
    R = H*np.pi / F
    return F,R

def hism_diff(H_c, result, ax):
    # --- Your metrics calculations ---
    #mae = np.mean(np.abs(result - H_c))
    #rmse = np.sqrt(np.mean((result - H_c)**2))
    rmse = np.sqrt(mean_squared_error(H_c, result))
    mae = mean_absolute_error(H_c, result)
    rrmse = rmse / np.mean(H_c) *100
    max_err = np.max(np.abs(result - H_c))
    corr = np.corrcoef(result.flatten(), H_c.flatten())[0, 1]

    # --- Shared color limits for imshow ---
    vmin = min(H_c.min(), result.min())
    vmax = max(H_c.max(), result.max())

    diff = (result - H_c).flatten()
    l = ax.hist(diff, bins=72, color='gray', edgecolor='black')
    #ax.set_title("Histogram of Differences")
    #ax.set_xlabel("Smoothed $R$ - Original $R$")
    #ax.set_ylabel("Count")

    # --- Add metrics as text box on the histogram ---
    metrics_text = (
        f"MAE = {mae:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"rrmse = {rrmse:.2f}%\n"
        f"Max Error = {max_err:.4f}\n"
        f"Corr = {corr:.4f}"
    )
    # Place text in the upper right of the histogram axes
    ax.text(0.45, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=11, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    fig_dir = 'figures/'
    #fig.savefig(fig_dir+'Smooth_R_2D.png', dpi=300, bbox_inches='tight')
    # plt.suptitle("Comparison of Original and Smoothed Angular Distribution R", fontsize=15, y=1.05)
    # plt.tight_layout()
    # plt.show()

# ================================
# RECONSTRUCTION FUNCTION
# ================================
def reconstruct_hc(U, S, VT):
    """Reconstruct H_c from SVD components"""
    return U @ np.diag(S) @ VT




def saveLUT(Ang_D,COD, dir='./FY4A_data/LUT/'):
    svd_data = {}
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']  # Example channel names
    solar_zeniths = np.array([0, 15, 30, 45, 60])  # Your 5 angles

    for c_idx, channel in enumerate(channels):
        channel_data = {'U': [], 'S': [], 'VT': []}

        for i in range(len(solar_zeniths)):
            H_c = Ang_D[i * len(channels) + c_idx]
            _, U_k, S_k, VT_k = svd_rank_k_approx(H_c, rank=3, Gau_smooth=True)

            channel_data['U'].append(U_k)
            channel_data['S'].append(S_k)
            channel_data['VT'].append(VT_k)

        # Convert to arrays
        svd_data[channel] = {
            'U': np.array(channel_data['U']),
            'S': np.array(channel_data['S']),
            'VT': np.array(channel_data['VT'])
        }

    # Save LUT
    save_svd_lut(channels, solar_zeniths, svd_data, dir+f'angular_dist_lut_COD={COD}.h5')
    return None
# ================================
# USAGE EXAMPLE
# ================================
# if __name__ == "__main__":
#     # Prepare your data storage
#     # data = np.load("Ang_D.npz")
#     # Ang_D = [data[f"D{i}"] for i in range(len(data.files))]
#     #
    
#     # Load and use example
#     target_zenith = 30  # Degrees
#     local_zen, rela_azi = 30, 120
#     for channel in channels:
#         #channel = 'C02'  # Example channel
#         theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
#         # file for validation
#         file_ = "/Volumes/DN1T_SSD/data/RTM_10000/fullspectrum/uwxyzr_COD=10_th0=32_Ta=299_RH=61.npy"
#         R_bin, R_origial = read_data(file_, channel, target_zenith, local_zen, rela_azi, N_bundles=10000,
#                   iCOD=10, ang=32)
#         # compare with the original H_c
#         U, S, VT = load_and_interpolate_whole('angular_dist_lut.h5', channel, target_zenith)
#         H_r = reconstruct_hc(U, S, VT)
#         #
#         print(H_r[theta_idx, phi_idx]-R_bin)


if __name__ == "__main__":
    
    channels=['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    COD_v = np.concatenate([np.linspace(0,20,11),np.linspace(25,50,6)])
    Sun_Zen_v = np.array([0,10,15,20,25,30,40,45,60]) # 0,15,30,45,60

    bandmode_v = ['channels']
    N_bundles = 10000
    for iCOD in COD_v:
        iCOD = int(iCOD)
        Ang_D = []
        for ang in Sun_Zen_v:
            for ch_idx, channel in enumerate(channels):
                fdir = "/mnt/dengnan/RTM_10000/"
                #fdir = "/Volumes/DN1T_SSD/data/RTM_10000/"
                filename = '/' + f'uwxyzr_COD={iCOD}_th0={ang}_Ta=294_RH=60.npy'
                file_dir = "./FY4A_data/"

                for bandmode in bandmode_v:
                    file_ = fdir + bandmode + filename
                    results = np.load(file_, allow_pickle=True).item()
                    uw_rxyz_M = results.get('uw_rxyz_M')
                    print(len(uw_rxyz_M))

                    dnu = 3
                    nu = np.arange(2500, 35000, dnu)
                    print(len(nu))
                    if bandmode == 'fullspectrum':
                        nu_input = nu
                    else:
                        channel_6c = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
                        nu_input = FY4A_calinu(nu, channel_6c, file_dir, dnu=3)

                    data = np.genfromtxt('data/profiles/ASTMG173.csv', delimiter=',', skip_header=2,  # in wavenumber basis
                            names=['wavelength', 'extraterrestrial', '37tilt', 'direct_circum'])
                    ref_lam = data['wavelength']  # nm avoid hearder 1
                    ref_E = data['extraterrestrial']
                    ref_E_nu = -ref_E * ref_lam ** 2 / 1e7  # W/[m2*nm-1] tp W/[m2*cm-1]

                    srf, nu_channel = get_calibration_srf(channel, file_dir)

                    F_dw_os_channel = -np.interp(-nu_channel, -1e7 / ref_lam, ref_E_nu)
                    F_dw_os_SRF = np.multiply(F_dw_os_channel, srf)

                    nu_idx = np.nonzero(np.isin(nu_input, nu_channel))[0]
                    
                    result = [uw_rxyz_M[i] for i in nu_idx]

                    if bandmode == 'channels':
                        Rc_rtm, R_c = cal_mono_R(result, ang, nu_input[nu_idx], F_dw_os_SRF,
                                                        N_bundles, is_flux=False, Norm=True, dirc='UW')
                        is_norm = check_normalization(R_c)/np.pi
                        if not np.isclose(is_norm, 1.0, rtol=1e-5):
                            print('is_norm:', is_norm)
                        # vmax=0.4
                        # ghi2d_show(R_c, channel, ang, vmax, logscale=True)
                        # H_theta = np.sum(H_c, axis=1)
                    Ang_D.append(R_c)
        saveLUT(Ang_D,iCOD)