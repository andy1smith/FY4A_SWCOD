import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from scipy.stats import pearsonr

# SCI Paper Formatting
font = 15
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
mpl.rcParams['xtick.major.size']     = 4
mpl.rcParams['ytick.major.size']     = 4
mpl.rcParams['mathtext.fontset']     = 'custom'
mpl.rcParams['mathtext.rm']          = fontfml
mpl.rcParams['mathtext.it']          = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf']          = 'Times New Roman:bold'
mpl.rcParams['axes.linewidth']       = 1.2
mpl.rcParams['legend.fontsize']      = font - 1


def main():
    data_dir = "../../Clear_Test/"
    #data_dir = "../../Clear_Test/dM/"
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(data_dir, 
        "Result_day_*_radiance_satellite_clearsky_HG_BRDF_sample.csv"))

    sites = []
    all_data = []
    for f in files:
        site_name = os.path.basename(f).split('_')[1]
        df = pd.read_csv(f)
        df = df[df['Sun_Zen'] <= 65]
        df = df[df['ghi'] <= 250]
        df = df[df['C01'] < 0.19]

        df = df[df['C06'] > 0.05]
        df['Time'] = pd.to_datetime(df['Time'])
        df['Month'] = df['Time'].dt.month
        df['Site'] = site_name
        all_data.append(df)
        sites.append(site_name)

    if not all_data:
        print("No data found!")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    df_mar_oct = df_all[(df_all['Month'] >= 1) & (df_all['Month'] <= 12)].copy()

    # Create figure
    fig = plt.figure(figsize=(10, 5))
    gs1 = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1, 1, 0.03], wspace=0.25, bottom=0.2
    )

    unique_sites = sorted(df_mar_oct['Site'].unique())
    n_sites = len(unique_sites)

    # Okabe-Ito palette (colorblind-safe, Nature/Science standard)
    okabe_ito = [
        '#E69F00',  # orange
        '#56B4E9',  # sky blue
        '#009E73',  # green
        '#F0E442',  # yellow
        '#0072B2',  # blue
        '#D55E00',  # vermilion
        '#CC79A7',  # reddish purple
        '#000000',  # black
    ]
    palette = okabe_ito[:n_sites]
    cmap = mcolors.ListedColormap(palette)
    bounds = np.arange(n_sites + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    vars_to_plot = [
        ('Site_dsw', 'rtm_dsw',    'GHI',              'Measured GHI [W/(m$^2$)]'),
        ('Site_usw', 'rtm_uw_srf', 'Surface Upwelling','Measured UW [W/(m$^2$)]'),
    ]

    for idx, (x_var, y_var, title, xlabel) in enumerate(vars_to_plot):
        ax = fig.add_subplot(gs1[0, idx])

        valid = ~(df_mar_oct[x_var].isna() | df_mar_oct[y_var].isna())
        x = df_mar_oct.loc[valid, x_var].values
        y = df_mar_oct.loc[valid, y_var].values
        sites_categorical = df_mar_oct.loc[valid, 'Site'].values

        mbe   = np.mean((y - x))
        rmse  = np.sqrt(np.mean((y - x) ** 2))
        rmbe  = mbe  / np.mean(x) * 100
        rrmse = rmse / np.mean(x) * 100
        R     = np.corrcoef(x, y)[0, 1]

        sns.scatterplot(
            x=x, y=y, ax=ax,
            hue=sites_categorical, hue_order=unique_sites,
            palette=palette, legend=False,
            edgecolor='w', s=40, alpha=0.8
        )

        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        margin = 0.05 * (max_val - min_val)
        lo, hi = min_val - margin, max_val + margin

        ax.plot([lo, hi], [lo, hi], color='gray', linestyle='--', linewidth=1.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Sync x/y ticks so grid lines align perfectly
        ax.set_aspect('equal', adjustable='box')
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))
        ax.figure.canvas.draw()
        yticks = ax.get_yticks()
        yticks = yticks[(yticks >= lo) & (yticks <= hi)]
        ax.set_xticks(yticks)
        ax.set_yticks(yticks)

        stats_text = (
            f'MBE: {mbe:.2f}\n'
            f'RMSE: {rmse:.2f}\n'
            f'rMBE: {rmbe:.2f}%\n'
            f'rRMSE: {rrmse:.2f}%\n'
            f'R = {R:.2f}'
        )

        ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
                fontsize=font - 2, family=fontfml,
                verticalalignment='top', weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        ax.set_title(title, fontsize=font, family=fontfml)
        ax.set_xlabel(xlabel, fontsize=font - 1, family=fontfml)
        if idx == 0:
            ax.set_ylabel('Model simulation [W/(m$^2$)]', fontsize=font - 1, family=fontfml)

        # Enforce Times New Roman on all tick labels explicitly
        ax.tick_params(labelsize=font - 1)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(fontfml)
        ax.grid(color='grey', linestyle='--', linewidth=0.5)

    # Global text for number of samples
    n_samples = len(df_mar_oct)
    fig.text(0.10, 0.92, f'n = {n_samples}  (Mar–Oct)',
             fontsize=font - 3, weight='bold', ha='left', va='top',
             fontfamily=fontfml)

    # Colorbar
    cax = fig.add_subplot(gs1[0, 3])
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

    # Save figure
    figname = os.path.join(output_dir, "DW_Flux_MarOct_Overall_RTM.png")
    fig.savefig(figname, dpi=600, bbox_inches='tight')
    print(f"Generated and saved {figname}")


if __name__ == "__main__":
    main()
