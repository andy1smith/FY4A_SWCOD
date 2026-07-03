"""
gpr_sites_full_scatterplots.py

Plots a composite GHI validation scatter plot from precomputed FY4A dM V6
surrogate prediction CSV files.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]
PRED_DIR = ROOT_DIR / "FY4A_validation" / "Clear_Test" / "Surrogate_modeling_surfrad_dM"
OUT_DIR = BASE_DIR / "dM_full_year"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────
font = 15
fontfml = 'Times New Roman'
mpl.rcParams['font.size'] = font
mpl.rcParams['font.family'] = fontfml
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = fontfml
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
mpl.rcParams['axes.linewidth'] = 1.2
sns.set_theme(style="ticks", context="paper")
sns.set_style({"font.family": "Times New Roman", "font.serif": ["Times New Roman", "Times New Roman"]})


def plot_validation_scatter(df_all):
    fig = plt.figure(figsize=(7, 6))
    gs1 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.08], wspace=0.15, bottom=0.2)
    
    unique_sites = sorted(df_all['Site'].unique())
    n_sites = len(unique_sites)
    
    palette = sns.color_palette("tab10", n_sites)
    cmap = mcolors.ListedColormap(palette)
    bounds = np.arange(n_sites + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
 
    vars_to_plot = [
        ('ghi', 'gpr_ghi', 'GHI', 'Measured GHI [W/(m$^2$)]', 'GPR GHI [W/(m$^2$)]'),
    ]

    for idx, (x_var, y_var, title, xlabel, ylabel) in enumerate(vars_to_plot):
        ax = fig.add_subplot(gs1[0, idx])
        
        valid = ~(df_all[x_var].isna() | df_all[y_var].isna())
        x = df_all.loc[valid, x_var].values
        y = df_all.loc[valid, y_var].values
        sites_cat = df_all.loc[valid, 'Site'].values

        mbe = np.mean((y - x))
        rmse = np.sqrt(np.mean((y - x)**2))
        rmbe = mbe / np.mean(x) * 100
        rrmse = rmse / np.mean(x) * 100
        R = np.corrcoef(x, y)[0, 1]

        sns.scatterplot(x=x, y=y, ax=ax, hue=sites_cat, hue_order=unique_sites,
                        palette=palette, legend=False, edgecolor='w', s=40, alpha=0.8)

        min_val = min(x.min(), y.min())
        if min_val < 0:
            min_val = 0
        max_val = max(x.max(), y.max())
        ax.plot([min_val * 0.9, max_val * 1.1], [min_val * 0.9, max_val * 1.1],
                color='gray', linestyle='--', linewidth=1.5)
        
        ax.set_xlim(min_val * 0.9, max_val * 1.1)
        ax.set_ylim(min_val * 0.9, max_val * 1.1)
        ax.tick_params(axis="both", which="major", labelsize=13)
        
        tick_spacing = 200
        ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))

        stats_text = f'MBE: {mbe:.2f}\nRMSE: {rmse:.2f}\nrMBE: {rmbe:.2f}%\nrRMSE: {rrmse:.2f}%\nR = {R:.2f}'

        ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        ax.set_title(title, fontsize=font, family=fontfml)
        ax.set_xlabel(xlabel, fontsize=font, family=fontfml)
        ax.set_ylabel('Model prediction [W/(m$^2$)]', fontsize=font, family=fontfml)
        ax.grid(color='grey', linestyle='--', linewidth=0.5)

    n_samples = len(df_all)
    fig.text(0.14, 0.92, f'n: {n_samples} (hourly)', fontsize=12, weight='bold', ha='left', va='top')

    cax = fig.add_subplot(gs1[0, 1])
    pos = cax.get_position() 
    new_left = pos.x0 + 0.02
    cax.set_position([new_left, pos.y0, pos.width, pos.height])

    cbar = fig.colorbar(sm, cax=cax, ticks=np.arange(n_sites))
    cbar.set_ticklabels(unique_sites)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Site Name', rotation=270, labelpad=13, fontsize=font, family=fontfml)

    figname = os.path.join(OUT_DIR, "DW_Flux_MarOct_GPR_Overall_dM.png")
    fig.savefig(figname, dpi=600, bbox_inches='tight')
    print(f"Generated and saved {figname}")


# ── Process Files ──────────────────────────────────────────────────────────
all_predicted = []
pattern = str(PRED_DIR / "gpr_predicted_dM_*.csv")
files = glob.glob(pattern)

for fpath in files:
    site = os.path.basename(fpath).replace("gpr_predicted_dM_", "").replace(".csv", "")
    print(f"\n── Reading {site} from precomputed dM CSV ──────────────────────")
    df = pd.read_csv(fpath)
    
    # Map columns to what the plot expects with robust schema detection
    df['Site'] = site
    if 'dM_gpr_ghi' in df.columns:
        df['gpr_ghi'] = df['dM_gpr_ghi']
    elif 'gpr_ghi' in df.columns:
        df['gpr_ghi'] = df['gpr_ghi']
    else:
        raise KeyError(f"Neither 'dM_gpr_ghi' nor 'gpr_ghi' found in {fpath}")
    
    # Convert Time to datetime to ensure correct Monthly grouping/filtering
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df['Month'] = df['Time'].dt.month

    print(f"  Loaded {len(df)} records.")
    all_predicted.append(df)

if all_predicted:
    df_all = pd.concat(all_predicted, ignore_index=True)
    df_mar_oct = df_all[(df_all['Month'] >= 3) & (df_all['Month'] <= 10)].copy()
    
    print("\n── Generating Plots ──────────────────────────────")
    plot_validation_scatter(df_mar_oct)
else:
    print("No data processed.")
