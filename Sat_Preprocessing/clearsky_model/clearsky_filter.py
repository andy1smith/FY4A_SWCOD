import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.lib.function_base import quantile


def plot_daily_csd(day_df, day_csd, date_label):
    """
    Plots the GHI, Clear Sky Model, and Detected Clear Sky periods for a single day.

    Parameters:
    - day_df: DataFrame containing 'ghi' and 'ghi_clear' for one day.
    - day_csd: Series or array of CSD flags (0=Clear, 1=Cloud) for that day.
    - date_label: String or Date object to display in the title.
    """
    plt.figure(figsize=(10, 4))

    if day_df.shape[0] == 1:
        plt.scatter(day_df.index, day_df['ghi'], s=15, color='gray', alpha=0.5, label='Measured GHI')
        plt.scatter(day_df.index, day_df['ghi_clear'], s=15, color='blue', alpha=0.3, label='Clear Sky Model')
    else:
        plt.plot(day_df.index, day_df['ghi'], color='gray', alpha=0.5, label='Measured GHI')
        plt.plot(day_df.index, day_df['ghi_clear'], color='blue', linestyle='--', alpha=0.3, label='Clear Sky Model')

    # 3. Plot Detected Clear Periods (Bold Black)

    # Mask out the "Cloudy" (1) parts with NaN so they don't get plotted
    # Ensure day_csd aligns with the dataframe index if it's a Series

    plt.scatter(day_csd.index, day_csd['ghi'], s=25, color='red', label='Clear Detected')

    # Formatting
    plt.title(f'Quesada-Ruiz CSD: {date_label}')
    plt.ylabel('Irradiance [$\mathrm{Wm^{-2}}$]')
    plt.xlabel('Time (UTC)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # Format x-axis to show hours clearly (HH:MM)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout()
    plt.show()

def single_clearday_display(df, csd, lon):
    # Group by date
    local_time = df.index + pd.Timedelta(hours=lon / 15.0)

    # Group by the DATE of the Local Time
    # MATLAB: unique_days = unique(floor(datenum(LST)));
    grouped = df.groupby(local_time.date)

    print(f"Starting loop for {len(grouped)} days...")

    for date, day_df in grouped:
        # Get the corresponding CSD flags for this specific day
        ind_csd = csd.loc[day_df.index]
        day_csd = day_df[ind_csd]
        # Skip empty days
        if day_csd.empty:
            continue

        # Call the subfunction
        plot_daily_csd(day_df, day_csd, date)


def polo2009_csd(df, longitude=116.3333,plot_figure=True):
    """
    Python implementation of Polo et al. (2009) Clear Sky Detection.

    Logic:
    1. Group data by local day.
    2. For each day, calculate the correlation matrix between GHI and GHI_Clear.
    3. If the determinant of the correlation matrix is low (< 0.005),
       it implies a very high correlation (r^2 > 0.995).
    4. Mark the entire day as Clear (0). Default is Cloud (1).

    Parameters:
    - df: DataFrame with index (UTC datetime) and columns ['ghi', 'ghi_clear']
    - longitude: Site longitude (used to adjust UTC to Local Time for correct day grouping)

    Returns:
    - csd: Series (same index as df) where 0=Clear, 1=Cloud
    """

    # 1. Initialize result with 1 (Cloud)
    # MATLAB: csd = ones(size(ghi));
    csd = pd.Series(1, index=df.index, name='CSD_Polo')

    # 2. Define Local Time for Day Grouping
    # We must shift UTC to Local Time so we don't split the solar day in half.
    # UTC 00:00 is 08:00 in China, so grouping by UTC date would split the morning.
    local_time = df.index + pd.Timedelta(hours=longitude / 15.0)

    # Group by the DATE of the Local Time
    # MATLAB: unique_days = unique(floor(datenum(LST)));
    grouped = df.groupby(local_time.date)

    # 3. Iterate through each day
    for date, day_data in grouped:

        # Extract the relevant columns
        ghi = day_data['ghi']
        ghi_cs = day_data['ghi_clear']

        # Filter for valid data (remove NaNs to prevent errors)
        # We only care about times where we expect sun (ghi_clear > 0)
        # to avoid correlation noise at night.
        mask = (ghi.notna()) & (ghi_cs.notna()) & (ghi_cs > 0)

        ghi_valid = ghi[mask]
        ghi_cs_valid = ghi_cs[mask]

        # MATLAB Check 1: Length of valid data
        # MATLAB: if length(ind) > 0.2*24*60 (assuming minute data)
        # ADAPTATION: We check if we have > 20% of the day's potential hours.
        # For hourly data, 20% of 24 hours is ~4.8 hours.
        if len(ghi_valid) < 5:
            continue  # Not enough data points to judge the day

        # MATLAB Check 2: Determinant of Correlation Matrix
        # MATLAB: determinant = det(corrcoef(ghi(ind),ghics(ind)));

        # Calculate Correlation Matrix (returns 2x2 matrix)
        # [[1.0, r],
        #  [r,   1.0]]
        corr_matrix = np.corrcoef(ghi_valid, ghi_cs_valid)

        # Safety check: if standard deviation is 0 (flat line), corr is NaN
        if np.isnan(corr_matrix).any():
            continue

        determinant = np.linalg.det(corr_matrix)

        if determinant < 0.005:
            # Mark the INDICES of this day as Clear (0)
            csd.loc[day_data.index] = 0
        # 4. Plot Figure (Optional)
    if plot_figure:
        plt.figure(figsize=(12, 5), facecolor='w')

        # Plot original GHI
        plt.plot(df.index, df['ghi'], label='GHI', color='gray', alpha=0.6, linewidth=1)

        # Create a copy for plotting "Clear Detected" only
        # MATLAB: CSD(csd==1)=NaN;
        ghi_clear_detected = df['ghi'].copy()
        ghi_clear_detected[csd == 1] = np.nan

        # Plot the "Clear" parts in bold black
        plt.plot(df.index, ghi_clear_detected, linewidth=2, color='k', label='Clear detected')

        plt.legend()
        plt.ylabel('Irradiance [$\mathrm{Wm^{-2}}$]')
        plt.xlabel('Time')
        plt.title('Polo2009')

        # Format the x-axis to look nice (handles datetime automatically)
        plt.gcf().autofmt_xdate()
        plt.show()
    return csd


def quesadaruiz2015_csd(df, plot_figure=False):
    """
    Python implementation of Quesada-Ruiz (2015) Clear Sky Detection.

    Logic:
    1. Calculate Clear Sky Index (kc = GHI / GHI_Clear).
    2. If kc < 0.8, the sky is considered Cloudy (1).
    3. Otherwise, it is Clear (0).

    Parameters:
    - df: DataFrame containing 'ghi' and 'ghi_clear' columns.
    - plot_figure: Boolean, set to True to see the plot.

    Returns:
    - csd: Series (same index as df) where 0=Clear, 1=Cloud.
    """

    ghi = df['ghi']
    ghics = df['ghi_clear']

    # 1. Initialize result with 0 (Clear)
    # MATLAB: csd = zeros(size(ghi));
    csd = np.zeros(len(ghi))

    # 2. Calculate Clearness Index (kc)
    # Handle division by zero (nighttime/zero clear sky) gracefully
    with np.errstate(divide='ignore', invalid='ignore'):
        kc = ghi / ghics

    # 3. Apply Threshold
    # MATLAB: csd(kc<0.8)=1;
    # If the ratio is low (measured < 80% of clear sky), it's cloudy (1).
    # We use numpy boolean indexing.
    # Note: kc < 0.8 returns False for NaNs (night), so night stays 0 (Clear)
    # unless you explicitly mask night. This mimics the provided MATLAB logic exactly.
    cloud_mask = (kc < 0.8)
    csd[cloud_mask] = 1

    # Convert to Pandas Series for easy handling
    csd_series = pd.Series(csd, index=df.index, name='CSD_QuesadaRuiz')

    # 4. Plot Figure (Optional)
    if plot_figure:
        plt.figure(figsize=(12, 5), facecolor='w')

        # Plot original GHI
        plt.plot(df.index, ghi, label='GHI', color='gray', alpha=0.6, linewidth=1)

        # Create a copy for plotting "Clear Detected" only
        # MATLAB: CSD(csd==1)=NaN;
        ghi_clear_detected = ghi.copy()
        ghi_clear_detected[csd_series == 1] = np.nan

        # Plot the "Clear" parts in bold black
        plt.plot(df.index, ghi_clear_detected, linewidth=2, color='k', label='Clear detected')

        plt.legend()
        plt.ylabel('Irradiance [$\mathrm{Wm^{-2}}$]')
        plt.xlabel('Time')
        plt.title('Quesada-Ruiz CSD Example')

        # Format the x-axis to look nice (handles datetime automatically)
        plt.gcf().autofmt_xdate()
        plt.show()

    return csd_series

def quantile85_csd(df,plot_figure=False):
    # Calculate raw index
    df['Kc_raw'] = np.where(df['ghi_clear'] > 1, df['ghi'] / df['ghi_clear'], 0)
    # Filter out low sun angles (Zenith > 85) to avoid dividing by near-zero
    mask_sun_up = df['Sun_Zen'] < 85
    # Find the "Upper Limit" of your actual data (approx representing clear days)
    # We use the 75th percentile (0.75) instead of 0.95 for HOURLY data.
    # Why? In hourly data, cloud enhancement spikes (which are >1.0) are smoothed out,
    # but 0.75 is generally a safe proxy for "typical clear sky" without over-fitting to outliers.
    scaling_factor = df.loc[mask_sun_up, 'Kc_raw'].quantile(0.75)
    print(f"Scaling Factor: {scaling_factor:.3f}")
    # 4. Adjusted Model
    df['ghi_clear_adjusted'] = df['ghi_clear'] * scaling_factor
    # 5. Final Kc
    df['Kc'] = np.where(df['ghi_clear_adjusted'] > 5, df['ghi'] / df['ghi_clear_adjusted'], 0)
    # A. Stability Check (Window = 3 Hours)
    # We check: [Previous Hour, Current Hour, Next Hour]
    # If Kc variance is low, the sky condition is stable.
    # min_periods=3 requires all 3 hours to be present to calculate stability.
    df['Kc_stability'] = df['Kc'].rolling(window=3, center=True, min_periods=3).std()

    # B. Dynamic Thresholds
    # 1. Magnitude: Allow GHI to be 30% higher than model (Kc < 1.3)
    #    This explicitly answers your request to include GHI > GHI_clear.
    mask_magnitude = (df['Kc'] > 0.85) & (df['Kc'] < 1.3)
    # 2. Stability:
    #    For hourly data, Kc changes slightly as sun moves, so 0.15 is a safe "smooth" limit.
    #    (If clouds pass by, this std dev usually jumps to > 0.3)
    #    We assume the first and last valid hours are unstable (fillna with high val)
    mask_stable = df['Kc_stability'].fillna(999) < 0.15
    # 3. Final Decision
    df['is_clear'] = (
            mask_magnitude &
            mask_stable &
            mask_sun_up
    )
    # --- Visualization check ---
    if plot_figure == True:
        subset = df[df['is_clear'] == True]
        plt.figure(figsize=(12, 6))
        plt.plot(subset.index, subset['ghi'], label='Measured GHI', color='grey', alpha=0.5)
        plt.plot(subset.index, subset['ghi_clear_adjusted'], label='Adjusted Clear Sky',linestyle='--', color='blue', alpha=0.7)
        # Highlight clear hours
        plt.scatter(subset[subset['is_clear']].index, subset[subset['is_clear']]['ghi'], color='red', zorder=10, s=10,
                    label='Detected Clear')
        plt.legend()
        plt.title(f"Clear Sky Detection (Hourly) - Scaling Factor: {scaling_factor:.2f}")
        plt.show()
    return df['is_clear']

def daytype_filter(df, lon):
    # 1. clear single-day extraction
    polo2009 = polo2009_csd(df, longitude=lon,plot_figure=False) #, all cloudy days
    whole_clearday = df[polo2009 == 0][['ghi', 'Sun_Zen', 'Sun_Azi', 'ghi_clear', 'Sun_Zen_App']]
    # print(len(polo2009[polo2009==0]))
    # is_clear_bool = (polo2009  == 0)
    # single_clearday_display(df, is_clear_bool, lon)

    # 2. cloudy day extraction
    # - quite strict for cloudy day, loose for clear day
    quesadaruiz = quesadaruiz2015_csd(df, plot_figure=False)
    cloudy_day = df[quesadaruiz == 1][['ghi', 'Sun_Zen', 'Sun_Azi', 'ghi_clear','Sun_Zen_App']]
    #single_clearday_display(df, quesadaruiz, lon)

    # 3. clear timestamp extraction - strict
    quan85 = quantile85_csd(df,plot_figure=False)
    clear_day = df[quan85][['ghi', 'Sun_Zen', 'Sun_Azi', 'ghi_clear','Sun_Zen_App']]
    #single_clearday_display(df, quan85, lon)



    return clear_day, cloudy_day, whole_clearday
