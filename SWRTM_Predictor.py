import xarray as xr
import os
import numpy as np
import h5py
import joblib
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# =============================================================================
# 1. The Model Loader Class (Solves the 600MB Memory/Loading Issue)
# =============================================================================
class SWRTM_Predictor:
    def __init__(self, model_dir):
        """
        Loads the 600MB .pkl file ONCE.
        Keep this object alive and pass it to processing functions.
        """
        model_path = os.path.join(model_dir, "GUI/SWRTM_PCA_GPR_v1.pkl")
        print(f"Loading GPR Model from {model_path} ...")

        loaded_bundle = joblib.load(model_path)
        self.scaler_X = loaded_bundle['scaler_X']
        self.scaler_y = loaded_bundle['scaler_y']
        self.pca = loaded_bundle['pca']
        self.gpr = loaded_bundle['gpr_model']
        print("Model loaded successfully into memory.")

    def predict(self, X_raw):
        """
        Input: Array of shape (N_pixels, 4) -> [Ta, rh, mu0, COD]
        Output: Array of shape (N_pixels, 6) -> [C01 ... C06]
        """
        # A. Scale Input
        X_new_scaled = self.scaler_X.transform(X_raw)

        # B. Predict PCA components
        y_pred_pca = self.gpr.predict(X_new_scaled)

        # C. Inverse PCA
        y_pred_scaled = self.pca.inverse_transform(y_pred_pca)

        # D. Inverse Scale
        y_pred_final = self.scaler_y.inverse_transform(y_pred_scaled)

        # Clip physical bounds
        return np.clip(y_pred_final, 0.0, 1.0)


# ---------------------------------------------------------
# 1. Helper: Vectorized Binning (Runs ONCE for the whole image)
# ---------------------------------------------------------
def get_geometry_indices_map(theta_array, phi_array):
    """
    Vectorized version of your find_bin_indices logic.
    phi: relative azimuth angles to the sun.

    Input: 1D arrays of Theta and Phi (N_pixels,)
    Output: 1D arrays of indices (N_pixels,)
    """
    # Define bin edges exactly as you did
    d_th = 2
    d_phi = 5
    bins_theta = np.arange(0, 91, d_th)       # Shape: (46,)
    bins_phi = np.arange(-180, 181, d_phi)    # Shape: (73,)

    # 1. Handle the Phi wrapping logic: if phij > 180: phij = 360 - phij
    # We use np.where to do this for the whole array at once
    phi_processed = np.where(phi_array > 180, 360 - phi_array, phi_array)

    # 2. Digitize (Find indices)
    # subtract 1 because digitize is 1-based
    theta_idx = np.digitize(theta_array, bins_theta) - 1
    phi_idx = np.digitize(phi_processed, bins_phi) - 1

    # 3. Safety Clip (Ensure we don't go out of bounds of the LUT)
    # Max index is len(bins) - 2 because of the way digitize works (N bins = N_edges - 1)
    theta_idx = np.clip(theta_idx, 0, len(bins_theta) - 2)
    phi_idx = np.clip(phi_idx, 0, len(bins_phi) - 2)

    return theta_idx, phi_idx


# ---------------------------------------------------------
# 2. Main Processing Function
# ---------------------------------------------------------
def predict_reflectance_scene(xr_sat_row,COD_v,predictor_model,channels,lut_dir):
    """
    1. Predicts Flux (GPR)
    2. Converts to Reflectance (ADM/LUT)
    3. Stores result
    xr_sat_row: xarray Dataset for one scene (y,x, channels dims)

    Args:
        xr_sat_row: xarray Dataset (y, x) or (time, y, x) for ONE timestamp
        COD_v: List/Array of COD values
        predictor_model: Instance of SWRTM_Predictor class
        channels: List of channel names
        lut_dir: Directory where .h5 LUTs are stored
    """

    # --- A. Prepare Data & Geometry (Once per scene) ---
    ny, nx = xr_sat_row['mu0'].shape
    n_pixels = ny * nx

    # Flatten inputs
    mu0_flat = xr_sat_row['mu0'].values.flatten()
    th0_deg_flat = xr_sat_row['th0'].values.flatten()
    ta_val = xr_sat_row['T_a'].values
    rh_val = xr_sat_row['rh'].values

    # Handle Scalar vs Map for Ta/RH
    ta_flat = np.full(n_pixels, ta_val) if np.ndim(ta_val) == 0 else ta_val.flatten()
    rh_flat = np.full(n_pixels, rh_val) if np.ndim(rh_val) == 0 else rh_val.flatten()

    # --- B. Pre-calculate ADM Indices (Vectorized) ---
    # We need 'Local_Zen' and 'rela_azi' for the LUT lookup
    # Ensure these exist in your xr_sat_row
    print("Calculating geometry indices...")
    local_zen_flat = xr_sat_row['Local_Zen'].values.flatten()
    rela_azi_flat = xr_sat_row['rela_azi'].values.flatten()

    # Get the integer indices map for the whole image
    theta_idx_map, phi_idx_map = get_geometry_indices_map(local_zen_flat, rela_azi_flat)

    # --- C. Loop Preparation ---
    n_channels = len(channels)  # 6
    # Output stores REFLECTANCE now, not flux
    output_storage = np.zeros((n_pixels, len(COD_v), n_channels), dtype=np.float32)

    print(f"Starting processing for {len(COD_v)} COD steps...")

    # --- D. Main Loop (COD) ---
    for i, cod_val in enumerate(COD_v):

        # 1. Build GPR Input Matrix [Ta, RH, mu0, COD]
        cod_col = np.full(n_pixels, cod_val)
        X_batch = np.column_stack((ta_flat, rh_flat, mu0_flat, cod_col))

        # 2. Predict Flux (batch_result)
        # Initialize with NaNs
        flux_batch = np.full((n_pixels, n_channels), np.nan)

        # Check valid pixels (masking NaNs in input)
        valid_mask = ~np.isnan(X_batch).any(axis=1)

        if np.any(valid_mask):
            X_valid = X_batch[valid_mask]
            # Predict Flux (Values are likely 0~500 W/m2/um)
            flux_batch[valid_mask] = predictor_model.predict(X_valid)

        # 3. Convert Flux -> Reflectance (Channel by Channel)
        #    Formula: rho = (Flux / pi) * H_r
        # NOTE: ADM LUTs usually depend on Solar Zenith Angle (target_zenith).
        # Since the scene is small (11x11), SZA variation is small.
        # Loading a unique LUT for every pixel is too slow.
        # We use the mean SZA of the valid pixels for the LUT selection
        sza_mean = np.nanmean(th0_deg_flat)
        for j, channel in enumerate(channels):
            # a. Load ADM LUT for this specific COD & Channel
            # (Assuming your load functions work like this)
            lut_filename = os.path.join(lut_dir, f"angular_dist_lut_COD={int(cod_val)}.h5")

            # Load U, S, VT and reconstruct H table (Small 2D array, e.g., 46x73)
            U, S, VT = load_and_interpolate_whole(lut_filename, channel, sza_mean)
            H_table = reconstruct_hc(U, S, VT)

            # b. "Advanced Indexing": Map the H_table to the Pixels
            # This creates a vector of H values corresponding to every pixel's geometry
            H_pixel_factors = H_table[theta_idx_map, phi_idx_map]

            # c. Apply Formula
            # If flux is NaN, result remains NaN
            # Slice the specific channel [:, j]
            current_flux_channel = flux_batch[:, j]

            # Reflectance calculation
            reflectance_channel = (current_flux_channel / np.pi) * H_pixel_factors

            # d. Store directly into output， i = COD
            output_storage[:, i, j] = reflectance_channel

    # --- E. Reshape and Return ---
    output_reshaped = output_storage.reshape(ny, nx, len(COD_v), n_channels)

    da_prediction = xr.DataArray(
        output_reshaped,
        coords={
            'y': xr_sat_row.y,
            'x': xr_sat_row.x,
            'cod': COD_v,
            'band': channels
        },
        dims=('y', 'x', 'cod', 'band'),
        name='Simulated_Reflectance'
    )

    return da_prediction




def retrieve_cod_vectorized(da_sim, da_obs, COD_v):
    """
    Invert the simulation to find COD.
    da_sim: (y, x, cod, band) - Simulated Reflectance
    da_obs: (y, x, band)      - Observed Reflectance
    COD_v: 1D array or list of COD values
    """

    # 1. Align Dimensions
    # Ensure da_obs has a 'cod' dimension of size 1 for broadcasting
    # Shape: (y, x, 1, band)
    obs_expanded = da_obs.expand_dims(dim={'cod': 1}, axis=2)

    # 2. Find indices where Sim <= Obs
    # We assume monotonic increase. We count how many sim values are smaller than obs.
    # This gives us the index 'i' just below the observation.
    # Shape: (y, x, band)
    lower_indices = (da_sim <= obs_expanded).sum(dim='cod') - 1

    # 3. Handle Boundary Cases (Clip indices to be safe)
    max_idx = len(COD_v) - 2
    lower_indices = lower_indices.clip(min=0, max=max_idx)
    upper_indices = lower_indices + 1

    # 4. Extract Values for Interpolation
    # We use 'isel' with the calculated indices to pick values for every pixel

    # Get Sim Reflectance at lower and upper bounds
    # We need to use advanced indexing. Since xarray is tricky with dynamic isel,
    # we drop to numpy for the heavy math.

    sim_vals = da_sim.values  # (y, x, n_cod, band)
    obs_vals = da_obs.values  # (y, x, band)
    idx_low = lower_indices.values  # (y, x, band)
    idx_high = upper_indices.values

    # Helper to pull values from the (y,x,band) specific indices
    # We flatten spatial dims for cleaner numpy integer indexing
    ny, nx, n_cod, n_band = sim_vals.shape

    # Create grid indices for y, x, and band
    # logic: result[y,x,b] = sim[y, x, idx[y,x,b], b]
    grid_y, grid_x, grid_b = np.meshgrid(
        np.arange(ny), np.arange(nx), np.arange(n_band), indexing='ij'
    )

    y0 = sim_vals[grid_y, grid_x, idx_low, grid_b]  # Ref at COD_low
    y1 = sim_vals[grid_y, grid_x, idx_high, grid_b]  # Ref at COD_high

    x0 = COD_v[idx_low]  # COD_low
    x1 = COD_v[idx_high]  # COD_high

    # 5. Perform Linear Interpolation
    # Formula: x = x0 + (y_obs - y0) * (x1 - x0) / (y1 - y0)

    slope = (x1 - x0) / (y1 - y0)

    # Avoid division by zero (if y1 == y0)
    slope = np.where(np.isinf(slope), 0, slope)

    cod_retrieved = x0 + (obs_vals - y0) * slope

    # 6. Wrap back into xarray
    da_cod_map = xr.DataArray(
        cod_retrieved,
        coords={'y': da_obs.y, 'x': da_obs.x, 'band': da_obs.band},
        dims=('y', 'x', 'band'),
        name='Retrieved_COD'
    )

    return da_cod_map


# ================================
# RECONSTRUCTION FUNCTION
# ================================
def reconstruct_hc(U, S, VT):
    """Reconstruct H_c from SVD components"""
    return U @ np.diag(S) @ VT

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


def predict_func(xr_sat, channels, file_dir):
    """
    Gassuian Process Regression to get RTM upwelling radiance.
    """
    model_path = file_dir+ "FY4A_tool/GUI/SWRTM_PCA_GPR_v2.pkl"
    loaded_bundle = joblib.load(model_path)
    # 2. Extract the components back into variables
    scaler_X = loaded_bundle['scaler_X']
    scaler_y = loaded_bundle['scaler_y']
    pca = loaded_bundle['pca']
    gpr = loaded_bundle['gpr_model']
    #print("Model loaded successfully!")

    # 3. Re-define the prediction function using these loaded objects
    def predict_spectrum(new_X_raw):
        """
        Input: Array of shape (N, 4) -> [Ta, rh, th0, COD]
        Output: Array of shape (N, 6) -> [C01 ... C06]
        """
        # A. Scale Input (Using the loaded scaler)
        X_new_scaled = scaler_X.transform(new_X_raw)

        # B. Predict PCA components (Using the loaded GPR)
        y_pred_scaled  = gpr.predict(X_new_scaled)

        # C. Inverse PCA (Reconstruct the 6 channels)
        #y_pred_scaled = pca.inverse_transform(y_pred_pca)

        # D. Inverse Scale (Get back to physical units)
        y_pred_final = scaler_y.inverse_transform(y_pred_scaled)

        y_pred_final = np.clip(y_pred_final, 0.0, 1.0)

        return y_pred_final


    X_test = xr_sat[['Ta', 'rh', 'th0', 'COD']].values
    # y_test = data_test[['C01', 'C02', 'C03', 'C04', 'C05', 'C06']].values
    y_pred = predict_spectrum(X_test)
    return y_pred


class SWRTM_GHI_Predictor:
    def __init__(self, model_dir):
        """
        Loads the 600MB .pkl file ONCE.
        Keep this object alive and pass it to processing functions.
        """
        model_path = os.path.join(model_dir, "GUI/SWRTM_ghi_GPR_v1.pkl")
        print(f"Loading GPR Model from {model_path} ...")

        loaded_bundle = joblib.load(model_path)
        self.scaler_X = loaded_bundle['scaler_X']
        self.scaler_y = loaded_bundle['scaler_y']
        self.gpr = loaded_bundle['gpr_model']
        print("Model loaded successfully into memory.")

    def predict(self, X_raw):
        """
        Input: Array of shape (N_pixels, 4) -> [Ta, rh, mu0, COD]
        Output: Array of shape (N_pixels, 6) -> [C01 ... C06]
        """
        # A. Scale Input
        X_new_scaled = self.scaler_X.transform(X_raw)

        # B. Predict PCA components
        y_pred_scaled = self.gpr.predict(X_new_scaled)

        # D. Inverse Scale
        y_pred_final = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

        # Clip physical bounds
        return y_pred_final


def validate_reflectance_scene(xr_sat_cod, predictor_model, channels, lut_dir, COD_v_grid,t):
    """
    1. Predicts Flux (GPR) using EXACT Retrieved COD.
    2. Converts to Reflectance using NEAREST COD for ADM selection.
    3. Compares with Observation and plots metrics.
    """

    # --- A. Prepare Data & Geometry ---
    print("Preparing validation data...")
    ny, nx = xr_sat_cod['mu0'].shape
    n_pixels = ny * nx

    # Flatten inputs
    mu0_flat = xr_sat_cod['mu0'].values.flatten()
    th0_deg_flat = xr_sat_cod['th0'].values.flatten()
    ta_val = xr_sat_cod['T_a'].values
    rh_val = xr_sat_cod['rh'].values

    # Critical: Get the Retrieved COD
    # Assuming the variable is named 'Retrieved_COD' (check your naming!)
    cod_key = 'Retrieved_COD'
    cod_retrieved_flat = xr_sat_cod[cod_key].values.flatten()

    # Handle Scalar vs Map
    ta_flat = np.full(n_pixels, ta_val) if np.ndim(ta_val) == 0 else ta_val.flatten()
    rh_flat = np.full(n_pixels, rh_val) if np.ndim(rh_val) == 0 else rh_val.flatten()

    # --- B. Pre-calculate ADM Indices (Vectorized) ---
    local_zen_flat = xr_sat_cod['Local_Zen'].values.flatten()
    rela_azi_flat = xr_sat_cod['rela_azi'].values.flatten()
    theta_idx_map, phi_idx_map = get_geometry_indices_map(local_zen_flat, rela_azi_flat)

    # --- C. Predict Flux (GPR) ---
    # We use the EXACT retrieved COD here for maximum precision
    X_batch = np.column_stack((ta_flat, rh_flat, mu0_flat, cod_retrieved_flat))

    # Initialize output array (Pixels, Channels)
    # GPR predicts all 6 channels at once
    flux_pred_flat = np.full((n_pixels, len(channels)), np.nan)

    valid_mask = ~np.isnan(X_batch).any(axis=1)

    if np.any(valid_mask):
        print(f"Predicting Flux for {np.sum(valid_mask)} valid pixels...")
        flux_pred_flat[valid_mask] = predictor_model.predict(X_batch[valid_mask])

    # --- D. Convert to Reflectance (ADM Loop) ---
    # Strategy: Map continuous COD to nearest Discrete Grid COD for LUT selection

    # 1. Find nearest grid point for every pixel
    # shape: (N_pixels, 1) - index in COD_v_grid
    cod_grid_indices = (np.abs(cod_retrieved_flat[:, None] - COD_v_grid)).argmin(axis=1)
    cod_nearest_vals = COD_v_grid[cod_grid_indices]

    # Output storage for Reflectance
    ref_pred_flat = np.full_like(flux_pred_flat, np.nan)

    # Mean SZA for LUT loading (Standard approximation for small scenes)
    sza_mean = np.nanmean(th0_deg_flat)

    print("Applying ADM (Reflectance conversion)...")

    for j, channel in enumerate(channels):
        # We assume flux_pred_flat column order matches 'channels' list
        current_flux = flux_pred_flat[:, j]

        # Optimization: Only loop over the COD values actually present in this image
        unique_cods_in_scene = np.unique(cod_nearest_vals[valid_mask])

        for grid_cod in unique_cods_in_scene:
            # 1. Identify pixels that belong to this COD bin
            # We combine the geometry mask (valid_mask) with the bin mask
            pixel_subset_mask = valid_mask & (cod_nearest_vals == grid_cod)

            if not np.any(pixel_subset_mask):
                continue

            # 2. Load LUT for this specific Grid COD
            lut_filename = os.path.join(lut_dir, f"angular_dist_lut_COD={int(grid_cod)}.h5")
            U, S, VT = load_and_interpolate_whole(lut_filename, channel, sza_mean)
            H_table = reconstruct_hc(U, S, VT)

            # 3. Get H factors for these pixels
            # Advanced indexing using the pre-calculated geometry maps
            subset_theta = theta_idx_map[pixel_subset_mask]
            subset_phi = phi_idx_map[pixel_subset_mask]

            H_factors = H_table[subset_theta, subset_phi]

            # 4. Calculate Reflectance
            # rho = Flux / pi * H
            ref_pred_flat[pixel_subset_mask, j] = (current_flux[pixel_subset_mask] / np.pi) * H_factors

    # --- E. Reshape & Metrics ---
    # Reshape to (y, x, band)
    ref_pred_map = ref_pred_flat.reshape(ny, nx, len(channels))

    # Get Observation
    obs_ref = xr_sat_cod['obs_ref'].values  # shape (band, y, x) usually
    # Transpose obs if needed to match (y, x, band)
    if obs_ref.shape[0] == len(channels):
        obs_ref = np.transpose(obs_ref, (1, 2, 0))

    # Calculate Diff
    diff_map = ref_pred_map - obs_ref

    # --- F. Print Metrics & Plot ---
    print("\nValidation Metrics (RMSE):")
    for j, channel in enumerate(channels):
        diff_ch = diff_map[:, :, j]
        rmse = np.sqrt(np.nanmean(diff_ch ** 2))
        mbe = np.nanmean(diff_ch)
        print(f"  {channel}: RMSE={rmse:.4f}, MBE={mbe:.4f}")

    # Plot C02 (Standard Vis) Diff as Sample
    # Assuming C02 is index 1
    idx_c02 = 1
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(obs_ref[:, :, idx_c02], cmap='Greys_r', vmin=0, vmax=0.5)
    plt.title(f"{t}")
    plt.colorbar(shrink=0.7)

    plt.subplot(1, 3, 2)
    plt.imshow(ref_pred_map[:, :, idx_c02], cmap='Greys_r', vmin=0, vmax=0.5)
    plt.title("RTM simulate (C02)")
    plt.colorbar(shrink=0.7)

    plt.subplot(1, 3, 3)
    limit = np.nanmax(np.abs(diff_map[:, :, idx_c02]))
    plt.imshow(diff_map[:, :, idx_c02], cmap='RdBu_r', vmin=-limit, vmax=limit)
    plt.title("Diff (Sim - Obs)")
    plt.colorbar(shrink=0.7)

    plt.tight_layout()
    plt.show()

    # Wrap diff in xarray to return
    da_diff = xr.DataArray(
        diff_map,
        coords={'y': xr_sat_cod.y, 'x': xr_sat_cod.x, 'band': channels},
        dims=('y', 'x', 'band'),
        name='Validation_Diff'
    )

    return da_diff



def map_sooth_identifier(Sat_obs):
    spatial_std = Sat_obs.std(dim=['x', 'y'])
    spatial_mean = Sat_obs.mean(dim=['x', 'y'])

    # 2. Calculate Coefficient of Variation (CV)
    # CV will be an array of 6 values (one for each band)
    cv = spatial_std / spatial_mean

    # 3. Check which bands are "Rough"
    # Define your threshold (e.g., 10% variability)
    threshold = 0.10

    print("--- Smoothness Check (CV) ---")
    print(cv)

    # You can programmatically filter:
    is_smooth = cv < threshold
    print("\n--- Is each band smooth? ---")
    print(is_smooth)
    return is_smooth
