"""
uw_reflectance_all_sites.py
---------------------------
Combined all-site UW reflectance scatter plot (channels C01-C06, excluding C04).
- 2 x 3 grid: C01, C02, C03 in row 1; boxplot (C04 slot), C05, C06 in row 2.
- Scatter points coloured by site name (categorical, Okabe-Ito).
- Stats box (MBE, RMSE, rMBE, rRMSE, R) printed on each scatter panel.
- Single shared categorical colorbar on the right for site names.
- Filters identical to generate_validation_flux_scatter.py (lines 37-44).
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import matplotlib as mpl

# --------------- formatting ---------------
font = 17
fontfml = 'Times New Roman'

# Apply AFTER sns.set_theme so seaborn cannot override them
sns.set_theme(style="ticks", context="paper")
sns.set_style({"font.family": "serif",
               "font.serif": ["Times New Roman", "Times New Roman"]})

# These must come AFTER set_theme / set_style
mpl.rcParams['font.family']          = fontfml
mpl.rcParams['font.serif']           = [fontfml]
mpl.rcParams['font.size']            = font
mpl.rcParams['axes.titlesize']       = font
mpl.rcParams['axes.labelsize']       = font - 1
mpl.rcParams['xtick.labelsize']      = font - 1
mpl.rcParams['ytick.labelsize']      = font - 1
mpl.rcParams['xtick.major.size']     = 10
mpl.rcParams['ytick.major.size']     = 10
mpl.rcParams['mathtext.fontset']     = 'custom'
mpl.rcParams['mathtext.rm']          = fontfml
mpl.rcParams['mathtext.it']          = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf']          = 'Times New Roman:bold'
mpl.rcParams['axes.linewidth']       = 1.2
mpl.rcParams['legend.fontsize']      = font - 1


def calc_metrics(obs, pred):
    valid = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[valid], pred[valid]
    if len(obs) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mbe   = np.mean(pred - obs)
    rmse  = np.sqrt(np.mean((pred - obs) ** 2))
    mean_obs = np.mean(obs)
    rmbe  = mbe  / mean_obs * 100 if mean_obs else np.nan
    rrmse = rmse / mean_obs * 100 if mean_obs else np.nan
    R     = np.corrcoef(obs, pred)[0, 1]
    return mbe, rmse, rmbe, rrmse, R


def main():
    # ---- paths ----
    data_dir   = "../../Clear_Test/"
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(data_dir, "Result_day_*_radiance_satellite_clearsky_HG_BRDF_sample.csv"))
    if not files:
        print(f"No CSV files found in {data_dir}")
        return

    # ---- load & filter ----
    all_data = []
    for f in files:
        site_name = os.path.basename(f).split('_')[2]
        df = pd.read_csv(f)
        
        # Apply filters based on existing columns
        if 'Sun_Zen' in df.columns:
            df = df[df['Sun_Zen'] <= 75]
        elif 'Site_zen' in df.columns:
            df = df[df['Site_zen'] <= 75]
            
        if 'ghi' in df.columns:
            df = df[df['ghi'] > 0]
            
        if 'C01' in df.columns:
            df = df[df['C01'] < 0.8]
        if 'C06' in df.columns:
            df = df[df['C06'] > 0.0]
            
        df['Time']  = pd.to_datetime(df['Time'])
        df['Month'] = df['Time'].dt.month
        df['Site']  = site_name
        all_data.append(df)

    df_all     = pd.concat(all_data, ignore_index=True)
    df_plot    = df_all[(df_all['Month'] >= 3) & (df_all['Month'] <= 10)].copy()

    unique_sites = sorted(df_plot['Site'].unique())
    n_sites      = len(unique_sites)

    # ---- colour palette: Okabe-Ito (colorblind-safe, Nature/Science standard) ----
    # Full 8-colour Okabe-Ito set; first n_sites colours are used
    # okabe_ito = [
    #     '#E69F00',  # orange
    #     '#56B4E9',  # sky blue
    #     '#009E73',  # green
    #     '#F0E442',  # yellow
    #     '#0072B2',  # blue
    #     '#D55E00',  # vermilion
    #     '#CC79A7',  # reddish purple
    #     '#000000',  # black
    # ]
    palette  = sns.color_palette("tab10", n_sites) # okabe_ito[:n_sites]
    cmap_cat = mcolors.ListedColormap(palette)
    bounds   = np.arange(n_sites + 1) - 0.5
    norm_cat = mcolors.BoundaryNorm(bounds, cmap_cat.N)
    sm       = plt.cm.ScalarMappable(cmap=cmap_cat, norm=norm_cat)
    sm.set_array([])

    site_colour = {s: palette[i] for i, s in enumerate(unique_sites)}

    # ---- channel layout:  C01 C02 C03 | BOX C05 C06  ----
    channels_plot = ['C01', 'C02', 'C03', 'C05', 'C06']
    # Grid positions: (row, col) for each channel, (1,0) reserved for boxplot
    ch_positions = {
        'C01': (0, 0),
        'C02': (0, 1),
        'C03': (0, 2),
        # (1,0) -> boxplot (C04 slot)
        'C05': (1, 1),
        'C06': (1, 2),
    }

    # ---- figure ----
    fig = plt.figure(figsize=(15, 9))
    # 2 rows × 3 scatter cols + 1 narrow colorbar col
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.04],
        wspace=0.2, hspace=0.2,
        bottom=0.12, top=0.93, left=0.08, right=0.90
    )

    # ---- scatter panels ----
    for ch, (row, col) in ch_positions.items():
        ax = fig.add_subplot(gs[row, col])

        obs_col = ch
        rtm_col = f"{ch}_rtm"

        valid = ~(df_plot[obs_col].isna() | df_plot[rtm_col].isna())
        x     = df_plot.loc[valid, obs_col].values
        y     = df_plot.loc[valid, rtm_col].values
        sites = df_plot.loc[valid, 'Site'].values

        mbe, rmse, rmbe, rrmse, R = calc_metrics(x, y)

        sns.scatterplot(
            x=x, y=y, ax=ax,
            hue=sites, hue_order=unique_sites,
            palette=palette, legend=False,
            edgecolor='w', s=30, alpha=0.8
        )

        min_v = min(x.min(), y.min())
        max_v = max(x.max(), y.max())
        margin = 0.05 * (max_v - min_v)
        ax.plot([min_v - margin, max_v + margin],
                [min_v - margin, max_v + margin],
                color='gray', linestyle='--', linewidth=1.5)

        ax.set_xlim(min_v - margin, max_v + margin)
        ax.set_ylim(min_v - margin, max_v + margin)

        # --- force x-ticks == y-ticks so grid lines align perfectly ---
        ax.set_aspect('equal', adjustable='box')
        # Cap at 5 nice ticks first, then mirror to both axes
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))
        ax.figure.canvas.draw()   # force locator to compute ticks
        lo, hi = min_v - margin, max_v + margin
        yticks = ax.get_yticks()
        yticks = yticks[(yticks >= lo) & (yticks <= hi)]
        ax.set_xticks(yticks)
        ax.set_yticks(yticks)

        stats_text = (
            f'MBE: {mbe:.4f}\n'
            f'RMSE: {rmse:.4f}\n'
            f'rMBE: {rmbe:.1f}%\n'
            f'rRMSE: {rrmse:.1f}%\n'
            f'R = {R:.3f}'
        )
        ax.text(0.04, 0.96, stats_text,
                transform=ax.transAxes, fontsize=font-2, family=fontfml,
                verticalalignment='top', weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        ax.set_title(ch, fontsize=font, family=fontfml)
        if ch == 'C05':
            ax.set_xlabel('GOES/ABI Reflectance Factor', fontsize=font - 1, family=fontfml)
        if col == 0:
            ax.set_ylabel('RTM Reflectance Factor', fontsize=font - 1, family=fontfml)
        # Enforce Times New Roman on every tick label explicitly
        ax.tick_params(labelsize=font - 1)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(fontfml)
        ax.grid(color='grey', linestyle='--', linewidth=0.5)

    # ---- boxplot panel in the C04 slot (row=1, col=0) ----
    ax_box = fig.add_subplot(gs[1, 0])

    # Build long-form data: residual = (rtm - obs) for each channel & site
    box_records = []
    for ch in channels_plot:
        obs_col = ch
        rtm_col = f"{ch}_rtm"
        valid = ~(df_plot[obs_col].isna() | df_plot[rtm_col].isna())
        sub = df_plot.loc[valid, ['Site', obs_col, rtm_col]].copy()
        sub['residual'] = sub[rtm_col] - sub[obs_col]
        sub['Channel']  = ch
        box_records.append(sub[['Site', 'Channel', 'residual']])

    df_box = pd.concat(box_records, ignore_index=True)

    # One group per site, coloured, boxes for each channel
    positions = np.arange(len(channels_plot))
    width = 0.12
    offsets = np.linspace(-(n_sites - 1) / 2 * width,
                           (n_sites - 1) / 2 * width,
                           n_sites)

    for i, site in enumerate(unique_sites):
        site_data = df_box[df_box['Site'] == site]
        bp_data   = [site_data.loc[site_data['Channel'] == ch, 'residual'].dropna().values
                     for ch in channels_plot]
        bp = ax_box.boxplot(
            bp_data,
            positions=positions + offsets[i],
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(facecolor=mcolors.to_rgba(palette[i], alpha=0.65)),
            whiskerprops=dict(color=palette[i]),
            capprops=dict(color=palette[i]),
        )

    ax_box.axhline(0, color='gray', linestyle='--', linewidth=1.2)
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(channels_plot, fontsize=font - 1, fontfamily=fontfml)
    ax_box.tick_params(labelsize=font - 1)
    for lbl in ax_box.get_xticklabels() + ax_box.get_yticklabels():
        lbl.set_fontfamily(fontfml)
    ax_box.set_xlabel('Channel', fontsize=font - 1, family=fontfml)
    ax_box.set_ylabel('RTM − GOES (Reflectance)', fontsize=font - 1, family=fontfml)
    ax_box.set_title('Residual by Site & Channel', fontsize=font, family=fontfml)
    ax_box.grid(color='grey', linestyle='--', linewidth=0.5, axis='y')

    # ---- shared colorbar ----
    cax = fig.add_subplot(gs[:, 3])
    pos = cax.get_position()
    pos.x0 -= 0.02
    pos.x1 -= 0.02
    cax.set_position(pos)
    cbar = fig.colorbar(sm, cax=cax, ticks=np.arange(n_sites))
    cbar.set_ticklabels(unique_sites)
    cbar.ax.tick_params(labelsize=font - 1)
    # Force Times New Roman on colorbar tick labels
    plt.setp(cbar.ax.get_yticklabels(), fontfamily=fontfml, fontsize=font - 1)
    cbar.set_label('Site', rotation=270, labelpad=15, fontsize=font, family=fontfml)

    # ---- global sample count ----
    n_samples = len(df_plot)
    fig.text(0.08, 0.96, f'n = {n_samples} (Mar–Oct)',fontfamily=fontfml,
             fontsize=font-5, weight='bold', ha='left', va='top')

    # fig.suptitle('UW Reflectance Factor: RTM vs GOES/ABI (All Sites)',
    #              fontsize=font + 2, family=fontfml, y=0.98, fontweight='bold')

    # ---- save ----
    figname = os.path.join(output_dir, "UW_Reflectance_AllSites_Combined.png")
    fig.savefig(figname, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {figname}")


if __name__ == "__main__":
    main()
