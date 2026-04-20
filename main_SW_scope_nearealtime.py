"""Post-process model outputs, and compare OLR and DLW."""
import numpy as np

from SWRTM_Predictor import *
from fun_nearealtime_RTM import *
from Sat_Preprocessing.Funcs_satellite_processing import *

def nearealtime_COD_retrival(figlabel, site, phase, file_dir=None, sky="day", N_bundles = 10000):
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"
    xr_sat = Sat_preprocess(file_dir, site, figlabel, sky, phase, sat='FY4A')
    return xr_sat

def SW_RTM_retrival(xr_sat, file_dir, site):
    """
    Main driver function for Shortwave RTM Retrieval.
    1. Normalizes Satellite Data.
    2. Loads GPR Model (Once).
    3. Loops through time to:
       - Simulate RTM (Forward Model)
       - Retrieve COD (Inversion)
    """
    COD_v = np.concatenate([np.arange(0, 20, 2), np.arange(20, 50 + 5, 5)])
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    lut_dir = os.path.join(file_dir, 'FY4A_tool','FY4A_ADMLUT')
    # preprocess, normalize
    xr_sat['mu0'] = np.cos(np.deg2rad(xr_sat['th0']))
    da_sat = xr_sat[channels].to_array(dim='band')
    xr_sat['obs_ref'] = da_sat

    # --- 3. Load Model (ONCE) ---
    model_predictor = SWRTM_Predictor(file_dir+'FY4A_tool/GPR/')

    # --- 4. Main Processing Loop ---
    print(f"Starting Retrieval for {len(xr_sat.time)} timestamps...")

    retrieved_cod_list = []

    for t in xr_sat.time[:10]:
        time_tag = pd.to_datetime(t.values).strftime("%B %d, %H UTC+8")
        row = xr_sat.sel(time=t)
        # Forward Model: Predict Reflectance for all COD steps
        # Returns: (y, x, cod_v, band)
        da_sim_ref = predict_reflectance_scene(
            xr_sat_row=row,
            COD_v=COD_v,
            predictor_model=model_predictor,
            channels=channels,
            lut_dir=lut_dir
        )
        # C. Inversion: Interpolate to find COD
        # We compare Simulated (da_sim_ref) vs Observed (row['obs_ref'])
        # Output: (y, x, band) -> A COD map retrieved from EACH channel
        ds_retrieval = retrieve_cod_interp(
            da_sim_ref,
            row['obs_ref'],
            COD_v,
            time_tag
        )
        #xr_cod_map = retrieve_cod_vectorized(da_prediction, xr_sat_measure, COD_v)
        # D. Store results
        retrieved_cod_list.append(ds_retrieval)
    # --- 5. Concatenate and Return ---
    print("Concatenating results...")
    # Combine list of time-steps back into a single Xarray object
    xr_cod_result = xr.concat(retrieved_cod_list, dim='time')
    xr_cod_result['time'] = xr_sat.time[:10]  # Ensure time coords match

    out_dir = 'FY4A_data/CODresults/'
    os.makedirs(out_dir, exist_ok=True)
    xr_sat_subset = xr_sat.isel(time=slice(0, 10))
    xr_cod_result['time'] = xr_sat.time[:10]
    ds_debug = xr.merge([xr_sat_subset, xr_cod_result])
    filename = f'Results_FY4A_COD_{site}.nc'
    save_path = os.path.join(out_dir, filename)
    print(f"Saving debug file to: {save_path}")
    encoding_settings = {key: {'zlib': True, 'complevel': 4} for key in ds_debug.data_vars}
    ds_debug.to_netcdf(save_path, encoding=encoding_settings)
    print("Save complete.")

    for t in xr_sat.time[:10]:
        time_tag = pd.to_datetime(t.values).strftime("%B %d, %H UTC+8")
        row = ds_debug.sel(time=t)
        da_diff = validate_reflectance_scene(row,
                                 predictor_model=model_predictor,
                                 channels=channels,
                                 lut_dir=lut_dir,COD_v_grid = COD_v,
                                             t=time_tag)
    return None

def plot_debug_cod_map(da_result, title="Debug: Retrieved COD"):
    """
    Plots the Retrieved COD map and prints basic statistics for debugging.

    Args:
        da_result: xarray DataArray (y, x)
        title: String for the plot title
    """
    # 1. Print Stats
    # We use .item() to convert numpy/xarray scalars to normal python numbers
    d_min = da_result.min().item()
    d_max = da_result.max().item()
    d_mean = da_result.mean().item()
    n_nans = np.isnan(da_result).sum().item()

    print("=" * 30)
    print(f"STATS: {title}")
    print(f"  Min:  {d_min:.2f}")
    print(f"  Max:  {d_max:.2f}")
    print(f"  Mean: {d_mean:.2f}")
    print(f"  NaNs: {n_nans}")
    print("=" * 30)

    # 2. Plot
    plt.figure(figsize=(7, 6))

    # .squeeze() ensures we don't have a stray time dimension breaking the plot
    da_result.squeeze().plot(
        cmap='Blues',
        vmin=d_min, vmax=d_max,
        cbar_kwargs={'label': 'Optical Depth'}
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()


def validate_ghi(site, lat, lon, elev, out_dir):
    xr_sat = xr.open_dataset(out_dir + f'Results_FY4A_COD_{site}.nc')
    model_predictor = SWRTM_GHI_Predictor('Sat_Preprocessing/GPR/')
    pixel_res = 0.04  # degrees (~4km at equator)
    ghi_pred, ghi_obs = [], []
    Time = []
    for t in xr_sat.time[:10]:
        time_tag = pd.to_datetime(t.values).strftime("%B %d, %H UTC+8")
        row = xr_sat.sel(time=t)
        row = generate_latlon_grid(
            row,
            center_lat=lat,
            center_lon=lon,
            resolution_deg=pixel_res
        )
        #row_parallax = apply_parallax_correction(row, cloud_height_km= 7.0)
        #plot_parallax_comparison(row, row_parallax)

        ghi_simu = predict_GHI_scene(
            xr_sat_row=row,#_parallax,
            predictor_model=model_predictor, t=time_tag
        )
        [ghi_pred_row, ghi_obs_row] = ghi_simu
        ghi_pred.append(ghi_pred_row), ghi_obs.append(ghi_obs_row)
        Time.append(pd.to_datetime(t.values))
        # create a pandas dataframe to save the results
    df_result = pd.DataFrame({
        'Time': Time,
        'pre_GHI': ghi_pred,
        'obs_GHI': ghi_obs
        })

    return df_result
# =============================================================================
# Helper: The Inversion Function (Sim -> COD)
# =============================================================================
def retrieve_cod_interp(da_sim_ref, da_obs, cod_grid,t):
    """
    1. Calculates Sum of Squared Errors (SSE) across bands.
    2. Finds the grid point with minimum error.
    3. Performs Parabolic Interpolation to find the exact sub-grid COD.
    """

    # --- 1. Calculate Cost Function (SSE) ---
    # da_sim_ref: (y, x, cod, band)
    # da_obs:     (band, y, x) -> Broadcasts to (y, x, 1, band)

    # Calculate difference, square it, and sum over bands
    # Result shape: (y, x, cod)
    is_smooth = map_sooth_identifier(da_obs)
    if is_smooth:
        sse_map = ((da_sim_ref - da_obs) ** 2).sum(dim='band')
    else:
        print("not smooth, pixel detection applied!")
    # --- 2. Find Discrete Minimum ---
    # "Which idx in COD grid (0th, 1st, 2nd...) gave the best fit?"
    min_indices = sse_map.argmin(dim='cod')

    # --- 3. Prepare for Interpolation (Numpy) ---
    # We drop to numpy for complex indexing
    sse_vals = sse_map.values  # (y, x, n_cod)
    idx = min_indices.values  # (y, x)
    grid = np.array(cod_grid)  # (22,)

    ny, nx = idx.shape

    # Handle Edge Cases:
    # If min is at index 0 or last index, we cannot interpolate (no neighbor).
    # We clip indices to be at least 1 and at most len-2 for the math,
    # then fix the values later.
    safe_idx = np.clip(idx, 1, len(grid) - 2)

    # Create coordinate grids for advanced indexing
    # We want values at safe_idx-1 (left), safe_idx (center), safe_idx+1 (right)
    grid_y, grid_x = np.indices((ny, nx))

    # --- 4. Extract (x, y) points for 3 neighbors ---
    # x = COD value, y = Error (SSE)

    # Left Point
    x1 = grid[safe_idx - 1]
    y1 = sse_vals[grid_y, grid_x, safe_idx - 1]

    # Center Point (The discrete minimum)
    x2 = grid[safe_idx]
    y2 = sse_vals[grid_y, grid_x, safe_idx]

    # Right Point
    x3 = grid[safe_idx + 1]
    y3 = sse_vals[grid_y, grid_x, safe_idx + 1]

    # --- 5. Inverse Parabolic Interpolation Formula ---
    # We want the 'x' where the parabola vertex is.
    # Formula for non-uniform x-spacing:
    # numer = (x2^2 - x3^2)*y1 + (x3^2 - x1^2)*y2 + (x1^2 - x2^2)*y3
    # denom = (x2 - x3)*y1 + (x3 - x1)*y2 + (x1 - x2)*y3
    # x_min = 0.5 * numer / denom

    numer = (x2 ** 2 - x3 ** 2) * y1 + (x3 ** 2 - x1 ** 2) * y2 + (x1 ** 2 - x2 ** 2) * y3
    denom = (x2 - x3) * y1 + (x3 - x1) * y2 + (x1 - x2) * y3

    # Avoid division by zero (flat surface)
    denom[denom == 0] = 1e-6

    cod_interpolated = 0.5 * numer / denom

    # --- 6. Safety Checks ---
    # If the indices were at the edges (0 or max), keep the discrete value
    # mask_edges is True where the original index was 0 or max
    #mask_edges = (idx == 0) | (idx == len(grid) - 1)

    # Also, if interpolation went wild (outside range [x1, x3]), revert to grid
    mask_wild = (cod_interpolated < x1) | (cod_interpolated > x3)

    # Apply corrections
    # Where edges or wild, use x2 (the discrete grid value)
    cod_final = np.where(mask_wild, x2, cod_interpolated)
    # 3. CRITICAL: Physical bounds check
    #    The parabola might sometimes predict -0.5. We must clip to 0.
    cod_final = np.clip(cod_final, grid[0], grid[-1])

    # --- 7. Package into Xarray ---
    da_result = xr.DataArray(
        cod_final,
        coords={'y': da_sim_ref.y, 'x': da_sim_ref.x},
        dims=('y', 'x'),
        name='Retrieved_COD'
    )
    plot_debug_cod_map(da_result, title=f"COD, {t}")
    return da_result

def predict_GHI_scene(xr_sat_row, predictor_model, t, show_plot=True):
    """
    1. Predicts GHI using inputs ('Ta', 'rh', 'mu0', 'COD')
    2. Compares with reference xr_sat_row['GHI']
    3. Plots the diff for debug and calculates metrics
    """

    # --- A. Prepare Data & Geometry ---
    # Note: Ensure the variable name matches your previous step ('Retrieved_COD' vs 'Retrived_COD')
    cod_key = 'Retrieved_COD'

    ny, nx = xr_sat_row[cod_key].shape
    n_pixels = ny * nx

    # Flatten inputs
    mu0_flat = xr_sat_row['mu0'].values.flatten()
    ta_val = xr_sat_row['T_a'].values
    rh_val = xr_sat_row['rh'].values  # Ensure key matches (RH vs rh)
    cod_val = xr_sat_row[cod_key].values

    # Handle Scalar vs Map
    ta_flat = np.full(n_pixels, ta_val) if np.ndim(ta_val) == 0 else ta_val.flatten()
    rh_flat = np.full(n_pixels, rh_val) if np.ndim(rh_val) == 0 else rh_val.flatten()

    # FIX: cod_val is already a map, just flatten it. Do not use np.full.
    cod_flat = cod_val.flatten()

    # Stack: [Ta, RH, mu0, COD]
    X_batch = np.column_stack((ta_flat, rh_flat, mu0_flat, cod_flat))

    # --- B. Predict Flux ---
    # Initialize with NaNs
    ghi_pred_flat = np.full((n_pixels,), np.nan)

    # Check valid pixels (masking NaNs in input)
    valid_mask = ~np.isnan(X_batch).any(axis=1)

    if np.any(valid_mask):
        X_valid = X_batch[valid_mask]
        # Predict GHI (Output is 1D array of Flux)
        # Assuming predictor_model.predict returns shape (N,) or (N, 1)
        pred_result = predictor_model.predict(X_valid)

        # Handle shape if it returns (N, 1)
        if pred_result.ndim > 1:
            pred_result = pred_result.flatten()

        ghi_pred_flat[valid_mask] = pred_result

    # --- C. Reshape to Map ---
    ghi_pred_map = ghi_pred_flat.reshape(ny, nx)

    # Wrap in Xarray
    da_ghi_pred = xr.DataArray(
        ghi_pred_map,
        coords={'y': xr_sat_row.y, 'x': xr_sat_row.x},
        dims=('y', 'x'),
        name='Predicted_GHI'
    )

    # --- D. Compare & Metrics ---
    # Get Ground Truth / Reference GHI
    if 'GHI' in xr_sat_row:
        ghi_obs = xr_sat_row['GHI'].values
        # Calculate Diff
        diff_map = ghi_pred_map - ghi_obs

        # Calculate Metrics (ignoring NaNs)
        valid_idx = ~np.isnan(diff_map)
        if np.any(valid_idx):
            rmse = np.sqrt(np.mean(diff_map[valid_idx] ** 2))
            mbe = np.mean(diff_map[valid_idx])  # Mean Bias Error
            #corr = np.corrcoef(ghi_pred_map[valid_idx], ghi_obs[valid_idx])[0, 1]
        else:
            rmse, mbe, corr = np.nan, np.nan, np.nan

        print("=" * 30)
        print(f"GHI Evaluation:")
        print(f"  RMSE: {rmse:.2f} W/m2")
        print(f"  MBE:  {mbe:.2f} W/m2")
        #print(f"  R:    {corr:.4f}")
        print("=" * 30)

        # --- E. Plotting ---
        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(9, 5))

            # 1. Prediction
            im0 = axes[0].imshow(ghi_pred_map, cmap='jet', vmin=0, vmax=1000)
            axes[0].set_title(f"GHI_RTM, th0={xr_sat_row['th0'].mean():.1f}°")
            plt.colorbar(im0, ax=axes[0], label='W/m2',shrink=0.7)

            # 3. Difference
            # Use centered colormap (RdBu) to see +/- bias
            limit = np.nanmax(np.abs(diff_map)) if not np.all(np.isnan(diff_map)) else 100
            im2 = axes[1].imshow(diff_map, cmap='RdBu_r', vmin=-limit, vmax=limit)
            cy, cx = ny // 2, nx // 2
            axes[1].scatter(cx, cy, c='black', marker='x', s=100)
            axes[1].set_title(f"Diff (Pred - Obs)\nRMSE={rmse:.1f}")
            plt.colorbar(im2, ax=axes[1], label='Difference (W/m2)',shrink=0.7)
            plt.title(f"{t}")
            plt.tight_layout()
            plt.show()
    return ghi_pred_map[nx//2,ny//2], ghi_obs[nx//2,ny//2]

def ADM_convert(df_row, local_zen, rela_azi, channels, fdir='./FY4A_tool/FY4A_ADMLUT/'):
    """

    Parameters
    ----------
    df_row
    local_zen
    rela_azi
    channels
    fdir

    Returns
    -------

    """


    COD_v = np.concatenate([np.arange(0, 22, 2), np.arange(20, 50 + 5, 5)])
    theta_idx, phi_idx = find_bin_indices(local_zen, rela_azi, 'both')
    RTM_ref_flux = pd.DataFrame([
        {**df_row.to_dict(), 'COD_v': cod} for cod in COD_v
    ])
    ref_rtm = RTM_ref_flux.copy()
    for i, COD in enumerate(COD_v):
        for channel in channels:
            U, S, VT = load_and_interpolate_whole(fdir + f'angular_dist_lut_COD={int(COD)}.h5', channel, target_zenith)
            H_r = reconstruct_hc(U, S, VT)
            ref_rtm.loc[i, channel] = RTM_ref_flux.loc[i, channel]/np.pi * H_r[theta_idx, phi_idx]  # correct uw_channel
    return ref_rtm

def compare_clear_dsw(site, sourcefile, meth='HG', surface = 'MODIS', sky="clear", file_dir=None, figlabel=None):
    """Compare the ground and modeled DLW.

    Parameters
    ----------
    site : str
        Site name: BON, GWN, etc.
    sky : str
        Sky time: "clear" or "cloudy".
    data_dir : path, optional
        The directory containing the SURFRAD + satellite data files.

    Returns
    -------
    None

    """
    # not used for current version
    if sky == "night":
        timeofday = "night"
    else:
        timeofday = "day"
    csvfile = ('./FY4A_validation/' + f"Result_{timeofday}_{site}_radiance_satellite_{sky}_{meth}_{surface}.csv")
               #f"BJC_radiance_satellite_clear.csv"

    rtm_dsw, rtm_dni, rtm_dhi, rtm_uw, rtm_uw_srf = [], [], [], [], []
    uw_channels_list = []
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    rtm_channels = [c + '_rtm' for c in channels]
    try:
        df_combined = pd.read_csv(csvfile)
        df_combined['Time'] = pd.to_datetime(df_combined['Time'])
        df_combined = df_combined.dropna()
        # try:
        #     df_combined['rtm_dni_1HG'] = df_combined['rtm_dni_1HG']/np.cos(np.deg2rad(df_combined['Site_zen']))
        # except Exception:
        #     pass
        rtm_GHI, rtm_DNI = df_combined['rtm_dsw'],df_combined['rtm_dni']
        
        site_GHI = df_combined['ghi']
        print('dsw read from existing csv file:', csvfile)
    except Exception:
        # open Satellite observation data
        file_path = sourcefile
        sat = pd.read_csv(file_path)

        print('surface : ', surface)
        df_albedo = cal_surface_albedo(sat, surface)
        site_GHI = sat['ghi']  # should not be sat[dw_ir], it includes the surface reflected radiation
        print(f"Run RTM to get output DSW.")

        sat['Time'] = pd.to_datetime(sat['Time'])
        sat = sat.set_index('Time')
        sat = sat.sort_index()
        if site=='BJC':
            from aod_codes import read_aod  # add_aod_to_sat
            # aod time series from surfrad
            df_aod = read_aod(site)
            sat = pd.merge_asof(sat, df_aod,
                                left_index=True,
                                right_index=True,
                                direction='nearest',
                                tolerance=pd.Timedelta('3min'))
            sat = sat.dropna(subset=['aod'])
            #c_index = [index for index in sat.index if index in df_aod.index]
            # sat = pd.concat([sat.loc[c_index], df_aod.loc[c_index]], join='inner', axis=1)
        # pre analysis
        sat = sat[sat['T_s'] > 283]
        sat = sat[sat['Sun_Zen'] < 65]
        # Select months 4 through 10 (inclusive)
        # sat_filtered = sat[(sat.index.month >= 4) & (sat.index.month <= 10)].copy()
        # #plot_zen_uw(sat_filtered['Sun_Zen'], sat_filtered, channels, 'Reflectance', 'FY4A', meth='HG', figlabel=figlabel + '_Zen410')
        # plot_zen_uw(sat_filtered['ghi'], sat_filtered, channels, 'GHI', 'FY4A', meth='HG',
        #             figlabel=figlabel + '_GHI410')
        print('# of sat:', sat.shape[0])
        for i in range(sat.shape[0]):
            print(i)
            Sun_Zen, local_zen, rela_azi = sat['Sun_Zen'][i], sat['Sat_Zen'][i], sat['RAZ'][i]
            COD_goes = 0  # Assuming COD is a column in sat_rad
            df_albedo_row = df_albedo.iloc[i].values
            T_s, RH = sat['T_s'].iloc[i], sat['RH'].iloc[i] # %, K
            try:
                AOD = sat['aod'].iloc[i]
            except Exception:
                AOD = None
            if Sun_Zen>60 or RH == np.nan:
                dsw, dni, dhi, uw, F_uw, uw_srf = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                df_uw_channels = pd.DataFrame([ [np.nan] * 6 ], columns=channels)
            else:
                dsw, dni, dhi, uw, uw_srf, df_R =  get_rtm_output(Sun_Zen, local_zen, rela_azi,
                                                                  COD_goes, T_s, RH, df_albedo_row, surface, meth, AOD)
                df_uw, F_uw = LUT(uw, COD_goes, Sun_Zen, local_zen, rela_azi)
                df_uw_channels = df_uw.mul(df_R, axis=1)
            rtm_dsw.append(dsw)
            rtm_dni.append(dni)
            rtm_dhi.append(dhi)
            rtm_uw.append(F_uw)
            rtm_uw_srf.append(uw_srf)
            uw_channels_list.append(df_uw_channels)

            df_new = pd.DataFrame({
                'rtm_dsw': rtm_dsw,
                'rtm_dni': rtm_dni,
                'rtm_dhi': rtm_dhi,
                'rtm_uw':rtm_uw,
                'rtm_uw_srf':rtm_uw_srf
            })
        sat = sat.reset_index()
        sat = sat.rename(columns={"index": "Time"})
        df_uw_all = pd.concat(uw_channels_list, ignore_index=True)
        df_uw_all = df_uw_all.add_suffix('_rtm')
        df_combined = pd.concat([sat, df_new, df_uw_all], axis=1)
        #df_combined['rtm_dni'] = df_combined['rtm_dni']/np.cos(np.deg2rad(df_combined['Sun_Zen']))
        df_combined = df_combined[df_combined['T_s'] > 283]
        df_combined = df_combined.dropna()
        df_combined.to_csv(csvfile, index=False)
        rtm_DNI, rtm_GHI = df_combined['rtm_dni'], df_combined['rtm_dsw']

    df_combined = pd.read_csv(csvfile)
    df_combined['Time'] = pd.to_datetime(df_combined['Time'])
    df_combined = df_combined.dropna()
    rtm_DNI, rtm_GHI = df_combined['rtm_dni'], df_combined['rtm_dsw']
    CODfromwho = 'RTM_clear'
    # uw
    df_combined['Sun_Zen'] = pd.to_numeric(df_combined['Sun_Zen'], errors='coerce')
    # 2. Drop any rows where Sun_Zen became NaN (this removes the bad rows entirely across all columns)
    df_combined = df_combined.dropna(subset=['Sun_Zen'])
    VAR = 'Reflectance'
    sat_rad = df_combined[channels]
    rtm_rad = df_combined[rtm_channels]#.mul(np.cos(np.deg2rad(df_combined['Sun_Zen'])), axis=0)
    #rtm_rad.columns = [col.replace('_rtm', '') for col in rtm_rad.columns]
    # Force each column name to a string before replacing
    rtm_rad.columns = [str(col).replace('_rtm', '') for col in rtm_rad.columns]
    #plot_data(sat_rad, rtm_rad, df_combined['Sun_Zen'], channels, VAR, CODfromwho, site, meth, figlabel)
    # The corrected function call
    plot_data(sat_rad, rtm_rad, channels, VAR, CODfromwho, df_combined['Sun_Zen'], surface, meth=meth, figlabel=figlabel)
    # dw
    plot_data_dw_clear(site_GHI, rtm_GHI, CODfromwho, df_combined['Sun_Zen'], site)
    # plot_zen_uw(df_combined['Sun_Zen'], rtm_rad/sat_rad, channels, VAR, CODfromwho, meth='HG', figlabel=figlabel + '_Zen')



if __name__ == "__main__":
    #for timeofday in ["day"]:
    file_dir = './'
    spectral = 'SW'
    phase = 'clear' # water
    N_bundles = 1000
    figlabel = 'test' #['COD<10','COD>20','COD>10'] # COD>20  July13

    CERNs = pd.read_csv(file_dir+'FY4A_data/'+"CERN_info.csv", header=0, index_col=False, names=['site', 'lon', 'lat', 'elev'])
    CERNs = CERNs.values.tolist()
    target_names = { 'DHL', 'FKD', 'FQA', 'HLA', 'JZB', 'LCA', 'NMD', 'SJM', 'THL', 'YCA'}
    #arget_names = {'BJC','CSA'}
    sites = [CERN for CERN in CERNs if CERN[0] in target_names]
    out_dir = 'FY4A_data/CODresults/'

    for site, lat, lon, elev in sites:
        for sky in ["clearsky"]:  # clearsky,day
            print(site)
            if sky == 'clearsky':
                filename = f"{site}_radiance_satellite_clear_sample.csv"  # "GOES_day_BON_radiance_satellite_a_clearsky"#
                meth = 'HG'
                compare_clear_dsw(site, './FY4A_data/site_sat_data/'+filename, meth=meth, surface = 'BRDF',
                                  sky=sky, file_dir=file_dir, figlabel=figlabel)
            else:
                xr_sat = nearealtime_COD_retrival(figlabel, site, phase, file_dir=file_dir, sky=sky, N_bundles = N_bundles)
                SW_RTM_retrival(xr_sat, file_dir,site)
                df_results = validate_ghi(site, lat, lon, elev, out_dir)
                # save df_results
                df_results.to_csv('FY4A_data/FinalResults1D/' + f'GHI_validation_FY4A_{site}.csv', index=False)
                # plot_NSRDB(site, plotwho, sky=sky, file_dir=file_dir, figlabel=figlabel)
