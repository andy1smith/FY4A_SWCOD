import math
import os

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


DEFAULT_CHANNELS = ["C01", "C02", "C03", "C04", "C05", "C06"]
DEFAULT_SOLAR_ZENITHS = np.array([0, 15, 30, 45, 60, 65])
DEFAULT_COD_GRID = np.concatenate([np.linspace(0, 20, 11), np.array([25, 30, 35])])


def FY4A_calinu(nu, channels, file_dir, dnu=3, sensor="FY4A"):
    """Return the model wavenumber grid covered by the requested FY4A AGRI channels."""
    if sensor != "FY4A":
        raise ValueError(f"Unsupported sensor for this ADM module: {sensor}")

    dirpath = os.path.join(file_dir, "AGRI_calibration")
    nus = set()
    for channel in channels:
        channel_number = int(channel[-2:])
        channel_srf = os.path.join(dirpath, f"FY4A_AGRI_SRF_ch{channel_number:d}.txt")
        calibration = np.loadtxt(channel_srf, delimiter=",", skiprows=1)
        calibration_nu = calibration[:, 1][::-1]
        channel_mask = (nu >= calibration_nu.min()) & (nu <= calibration_nu.max())
        nus.update(nu[channel_mask])

    return np.array(sorted(nus))


def get_calibration_srf(channel, file_dir, nu=None, dnu=3):
    """Load and interpolate the FY4A AGRI channel spectral response on the RTM grid."""
    if nu is None:
        nu = np.arange(2500, 35000, dnu)

    channel_number = int(channel[-2:])
    channel_srf = os.path.join(
        file_dir,
        "AGRI_calibration",
        f"FY4A_AGRI_SRF_ch{channel_number:d}.txt",
    )
    calibration = np.loadtxt(channel_srf, delimiter=",", skiprows=1)
    calibration_nu = calibration[:, 1][::-1]
    calibration_srf = calibration[:, 2][::-1]
    nu_channel = FY4A_calinu(nu, [channel], file_dir, dnu=dnu)
    srf = np.interp(nu_channel, calibration_nu, calibration_srf)
    return srf, nu_channel


def theta_phi_scope(rx, ry, rz):
    """
    ADM photon-direction convention used by the updated GOES generator.

    0 degrees relative azimuth is backscattering toward the sun, and 180 degrees
    is forward scattering away from the sun.
    """
    rz = np.clip(rz, -1.0, 1.0)
    theta = np.arccos(rz)
    sin_th = np.sqrt(np.maximum(0.0, 1.0 - rz**2))
    p = np.random.uniform(low=-np.pi, high=np.pi, size=theta.shape[0])

    with np.errstate(divide="ignore", invalid="ignore"):
        cosP = -rx / sin_th
    cosP = np.clip(cosP, -1.0, 1.0)
    phi = np.arccos(cosP)
    phi[rz == 1] = p[rz == 1]

    ind = ((-ry) * sin_th < 0)
    if ind.size != 0:
        phi[ind] = 2 * math.pi - phi[ind]

    zero = rx**2 + ry**2 + rz**2 == 0
    theta[zero] = np.nan
    phi[zero] = np.nan
    return theta, phi


def cal_mono_R(
    rxyz_M,
    theta0_deg,
    nu_or_F_dw_os,
    F_dw_os=None,
    N_bundles=10000,
    is_flux=False,
    dirc="UW",
    bin_scale=1,
):
    """
    Compute normalized angular distribution R(theta, phi) for one FY4A channel.

    The updated ADM grid follows the GOES ADM generator: theta bins are 5 deg
    and relative-azimuth bins are 10 deg over the symmetric 0-180 deg range.
    ``nu_or_F_dw_os`` keeps backward compatibility with the old FY4A call style,
    where the unused wavenumber grid was passed before ``F_dw_os``.
    """
    if F_dw_os is None:
        F_dw_os = nu_or_F_dw_os
    if is_flux:
        raise ValueError("The updated ADM LUT generator expects is_flux=False.")
    if dirc != "UW":
        raise ValueError("The updated FY4A ADM generator currently supports only dirc='UW'.")

    F_dw_os = np.asarray(F_dw_os)
    theta0 = math.radians(theta0_deg)
    d_th = 5 * bin_scale
    d_phi = 10 * bin_scale
    bins_theta = np.arange(0.0, 90.0 + d_th, d_th)
    bins_phi = np.arange(0.0, 180.0 + d_phi, d_phi)
    H = np.zeros((len(bins_theta) - 1, len(bins_phi) - 1))

    for i, fw_rxyz in enumerate(rxyz_M):
        if fw_rxyz is None or len(fw_rxyz) == 0:
            continue

        fw_rxyz = np.asarray(fw_rxyz)
        rx = fw_rxyz[:, 0]
        ry = fw_rxyz[:, 1]
        rz = fw_rxyz[:, 2]

        theta_v, phi_v = theta_phi_scope(rx, ry, rz)
        valid = ~np.isnan(phi_v)
        theta_v = theta_v[valid]
        phi_deg = np.rad2deg(phi_v[valid]) % 360
        phi_deg[phi_deg > 180] = 360 - phi_deg[phi_deg > 180]

        H_i, _, _ = np.histogram2d(
            np.rad2deg(theta_v),
            phi_deg,
            bins=(bins_theta, bins_phi),
        )
        H += H_i * np.cos(theta0) * F_dw_os[i] * 3 / N_bundles

    F_iso = np.sum(H)
    if F_iso <= 0:
        return np.zeros_like(H)

    theta_bin_centers = bins_theta[:-1] + d_th / 2
    ths = np.deg2rad(theta_bin_centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        H /= np.sin(2 * ths)[:, np.newaxis]
    H /= np.deg2rad(d_th) * np.deg2rad(d_phi)

    R = H * np.pi / F_iso
    return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)


def save_svd_lut(channels, solar_zeniths, svd_data, filename="svd_lut.h5"):
    """Save SVD-compressed ADM LUT components to an HDF5 file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, "w") as f:
        f.create_dataset("solar_zeniths", data=solar_zeniths)
        for channel in channels:
            grp = f.create_group(f"{channel}")
            grp.create_dataset("U", data=svd_data[channel]["U"])
            grp.create_dataset("S", data=svd_data[channel]["S"])
            grp.create_dataset("VT", data=svd_data[channel]["VT"])
    print(f"SVD LUT saved to {filename}")


def load_and_interpolate_whole(filename, channel, target_zenith):
    """Load SVD components and interpolate them to a target solar zenith angle."""
    with h5py.File(filename, "r") as f:
        solar_zeniths = f["solar_zeniths"][:]
        channel_group = f[f"{channel}"]
        U = channel_group["U"][:]
        S = channel_group["S"][:]
        VT = channel_group["VT"][:]

    U_interp = np.zeros((U.shape[1], U.shape[2]))
    S_interp = np.zeros(S.shape[1])
    VT_interp = np.zeros((VT.shape[1], VT.shape[2]))

    interp_kind = "quadratic" if target_zenith > 30 and len(solar_zeniths) >= 3 else "linear"
    for r in range(U.shape[2]):
        for theta_idx in range(U.shape[1]):
            interp_fn = interp1d(
                solar_zeniths,
                U[:, theta_idx, r],
                kind=interp_kind,
                fill_value="extrapolate",
            )
            U_interp[theta_idx, r] = interp_fn(target_zenith)

        interp_fn = interp1d(
            solar_zeniths,
            S[:, r][:, 0],
            kind="linear",
            fill_value="extrapolate",
        )
        S_interp[r] = interp_fn(target_zenith)

        for phi_idx in range(VT.shape[2]):
            interp_fn = interp1d(
                solar_zeniths,
                VT[:, r, phi_idx],
                kind="linear",
                fill_value="extrapolate",
            )
            VT_interp[r, phi_idx] = interp_fn(target_zenith)

    return U_interp, S_interp, VT_interp


def load_and_interpolate(filename, channel, target_zenith, theta_, phi_):
    """Load and interpolate a 3-by-3 neighborhood around theta/phi indices."""
    with h5py.File(filename, "r") as f:
        solar_zeniths = f["solar_zeniths"][:]
        channel_group = f[f"{channel}"]
        U = channel_group["U"][:]
        S = channel_group["S"][:]
        VT = channel_group["VT"][:]

    theta_indices = np.clip(np.arange(theta_ - 1, theta_ + 2), 0, U.shape[1] - 1)
    phi_indices = np.clip(np.arange(phi_ - 1, phi_ + 2), 0, VT.shape[2] - 1)
    rank = U.shape[2]

    U_interp = np.zeros((3, rank))
    S_interp = np.zeros(rank)
    VT_interp = np.zeros((rank, 3))

    for r in range(rank):
        for i, theta_idx in enumerate(theta_indices):
            interp_fn = interp1d(
                solar_zeniths,
                U[:, theta_idx, r],
                kind="linear",
                fill_value="extrapolate",
            )
            U_interp[i, r] = interp_fn(target_zenith)

        interp_fn = interp1d(
            solar_zeniths,
            S[:, r][:, 0],
            kind="linear",
            fill_value="extrapolate",
        )
        S_interp[r] = interp_fn(target_zenith)

        for j, phi_idx in enumerate(phi_indices):
            interp_fn = interp1d(
                solar_zeniths,
                VT[:, r, phi_idx],
                kind="linear",
                fill_value="extrapolate",
            )
            VT_interp[r, j] = interp_fn(target_zenith)

    return U_interp, S_interp, VT_interp


def svd_rank_k_approx(matrix, rank=3, Gau_smooth=True):
    """Return a smoothed rank-k SVD approximation and components."""
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    U_k = U[:, :rank]
    S_k = np.diag(S[:rank])
    VT_k = VT[:rank, :]

    if Gau_smooth:
        U_k = np.array([gaussian_filter1d(U_k[:, i], sigma=1) for i in range(rank)]).T
        VT_k = np.array([gaussian_filter1d(VT_k[i, :], sigma=1) for i in range(rank)])

    matrix_approx = U_k @ S_k @ VT_k
    return matrix_approx, U_k, S_k, VT_k


def check_normalization(R_nor, d_theta=5, d_phi=10):
    """Approximate the ADM integral over the symmetric 0-180 deg hemisphere."""
    n_theta, _ = R_nor.shape
    theta_edges = np.linspace(0, 90, n_theta + 1)
    theta_centers = np.radians(theta_edges[:-1] + d_theta / 2)
    cos_theta = np.cos(theta_centers)[:, np.newaxis]
    sin_theta = np.sin(theta_centers)[:, np.newaxis]
    return np.sum(R_nor * cos_theta * sin_theta) * np.radians(d_theta) * np.radians(d_phi)


def reconstruct_hc(U, S, VT):
    """Reconstruct an ADM table from interpolated SVD components."""
    return U @ np.diag(S) @ VT


def saveLUT(
    Ang_D,
    COD,
    dir="./FY4A_data/ADM_cloud/LUT/",
    channels=None,
    solar_zeniths=None,
    rank=3,
    gaussian_smooth=True,
):
    """Compress and save one COD ADM LUT using channel-major SVD groups."""
    if channels is None:
        channels = DEFAULT_CHANNELS
    if solar_zeniths is None:
        solar_zeniths = DEFAULT_SOLAR_ZENITHS
    solar_zeniths = np.asarray(solar_zeniths)

    n_expected = len(channels) * len(solar_zeniths)
    if len(Ang_D) != n_expected:
        raise ValueError(
            f"Expected {n_expected} ADM tables "
            f"({len(solar_zeniths)} solar zeniths * {len(channels)} channels), got {len(Ang_D)}"
        )

    svd_data = {}
    for c_idx, channel in enumerate(channels):
        channel_data = {"U": [], "S": [], "VT": []}
        for i in range(len(solar_zeniths)):
            idx = i * len(channels) + c_idx
            _, U_k, S_k, VT_k = svd_rank_k_approx(
                Ang_D[idx],
                rank=rank,
                Gau_smooth=gaussian_smooth,
            )
            channel_data["U"].append(U_k)
            channel_data["S"].append(S_k)
            channel_data["VT"].append(VT_k)

        svd_data[channel] = {
            "U": np.array(channel_data["U"]),
            "S": np.array(channel_data["S"]),
            "VT": np.array(channel_data["VT"]),
        }

    filename = os.path.join(dir, f"angular_dist_lut_COD={int(COD)}.h5")
    save_svd_lut(channels, solar_zeniths, svd_data, filename)
    return filename
