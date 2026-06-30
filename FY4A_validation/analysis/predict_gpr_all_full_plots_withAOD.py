"""
predict_gpr_sites_full_plots.py

Generates the pro-style overall validation scatter plots (GHI only) and seasonal variance plots
using the precomputed dM GPR predicted CSV files from FY4A sites.
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde, pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]
PRED_DIR = ROOT_DIR / "FY4A_validation" / "Clear_Test" / "Surrogate_modeling_surfrad_dM"
OUT_DIR = BASE_DIR / "dM_full_year"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────
font_size = 12
fontfml = 'Times New Roman'
mpl.rcParams['font.size'] = font_size
mpl.rcParams['font.family'] = fontfml
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = fontfml
mpl.rcParams['axes.linewidth'] = 1.2
sns.set_theme(style="ticks", context="paper")
sns.set_style({"font.family": "Times New Roman", "font.serif": ["Times New Roman", "Times New Roman"]})


def calc_metrics(obs, pred):
    valid = ~(np.isnan(obs) | np.isnan(pred))
    obs = obs[valid]
    pred = pred[valid]
    if len(obs) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    mbe = np.mean(pred - obs)
    rmse = np.sqrt(np.mean((pred - obs)**2))
    mean_obs = np.mean(obs)
    rmbe = (mbe / mean_obs) * 100 if mean_obs != 0 else np.nan
    rrmse = (rmse / mean_obs) * 100 if mean_obs != 0 else np.nan
    try:
        r = pearsonr(obs, pred)[0]
    except:
        r = np.nan
    return mbe, rmse, rmbe, rrmse, r


def build_pro_scatter_axes(ax, obs, pred, xlabel, ylabel, title, color="#c8e3eb", marker_label='(a)'):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]

    # Calculate density
    xy = np.vstack([obs, pred])
    z = gaussian_kde(xy)(xy)
    
    # Normalize density to [0, 1]
    z_norm = (z - z.min()) / (z.max() - z.min()) if z.max() > z.min() else z
    
    # Sort the points by density so that high-density points are plotted on top
    idx = z_norm.argsort()
    obs_sorted = obs[idx]
    pred_sorted = pred[idx]
    z_norm_sorted = z_norm[idx]

    sc = ax.scatter(obs_sorted, pred_sorted, c=z_norm_sorted, s=10, cmap='jet', alpha=0.80, edgecolors='none')
    
    vmin = float(min(np.min(obs), np.min(pred)))
    vmax = float(max(np.max(obs), np.max(pred)))
    pad = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
    vmin -= pad
    vmax += pad
    
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.plot([vmin, vmax], [vmin, vmax], color='k', linestyle='--', linewidth=1.1, alpha=0.85)

    # Fit line
    x_fit = np.linspace(vmin, vmax, 200)
    if obs.size >= 2:
        slope, intercept = np.polyfit(obs, pred, 1)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='k', linewidth=1.6)

    # Marginal Hists
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size=0.6, pad=0.10, sharex=ax)
    ax_right = divider.append_axes("right", size=0.6, pad=0.10, sharey=ax)
    
    bins = 18
    edge_lw, edge_color = 0.6, 'black'
    
    ax_top.hist(obs, bins=bins, range=(vmin, vmax), color=color, alpha=0.65, edgecolor=edge_color, linewidth=edge_lw)
    ax_right.hist(pred, bins=bins, range=(vmin, vmax), orientation='horizontal', color=color, alpha=0.65, edgecolor=edge_color, linewidth=edge_lw)
    
    bin_width = (vmax - vmin) / bins
    if obs.size > 1:
        # Top KDE
        kde_x = gaussian_kde(obs)
        density_x = kde_x(x_fit)
        ax_top.plot(x_fit, density_x * obs.size * bin_width, color=color, linewidth=2.0, alpha=0.95)
        # Right KDE
        kde_y = gaussian_kde(pred)
        density_y = kde_y(x_fit) # Use x_fit as generic linspace
        ax_right.plot(density_y * pred.size * bin_width, x_fit, color=color, linewidth=2.0, alpha=0.95)

    ax_top.axis('off')
    ax_right.axis('off')
    ax.grid(True, color='#d0d0d0', linewidth=0.8, alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=font_size + 2, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=font_size + 2, fontweight='bold')
    ax.tick_params(axis='both', labelsize=font_size)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

    # Calculate metrics
    mbe = np.mean(pred - obs)
    rmse = np.sqrt(np.mean((pred - obs)**2))
    rmbe = (mbe / np.mean(obs)) * 100 if np.mean(obs) != 0 else np.nan
    rrmse = (rmse / np.mean(obs)) * 100 if np.mean(obs) != 0 else np.nan
    try:
        R = pearsonr(obs, pred)[0]
    except:
        R = np.nan
    n_samples = len(obs)
    
    stats_text = f'MBE: {mbe:.2f}\nRMSE: {rmse:.2f}\nrMBE: {rmbe:.2f}%\nrRMSE: {rrmse:.2f}%\nR = {R:.2f}'
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.text(0.04, 0.68, f'n: {n_samples}', transform=ax.transAxes, fontsize=10, weight='bold',
            verticalalignment='top')

    # Add density color bar [0, 1]
    cax = inset_axes(
        ax,
        width="3%",
        height="25%",
        loc="lower left",
        bbox_to_anchor=(0.04, 0.35, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    fig = ax.get_figure()
    cbar = fig.colorbar(sc, cax=cax, ticks=[0, 0.5, 1])
    cbar.set_label('Density', fontsize=font_size - 3, fontweight='bold')
    cbar.ax.tick_params(labelsize=font_size - 3)

    # Inset Boxplot of Residuals
    resid = pred - obs
    ax_res = inset_axes(
        ax,
        width="38%",
        height="32%",
        loc="lower right",
        borderpad=0.0,
        bbox_to_anchor=(-0.05, 0.12, 1.0, 1.0),
        bbox_transform=ax.transAxes,
    )
    rmax = float(np.nanmax(np.abs(resid)))
    rpad = rmax * 0.10 if rmax > 0 else 1.0
    ax_res.set_xlim(-rmax - rpad, rmax + rpad)
    ax_res.axvline(0, color='k', linestyle='--', linewidth=1.0, alpha=0.85)
    
    edge_lw = 0.6
    edge_color = 'black'
    bp = ax_res.boxplot(
        [resid],
        vert=False,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        boxprops=dict(linewidth=edge_lw, color=edge_color),
        medianprops=dict(linewidth=1.2, color='k'),
        whiskerprops=dict(linewidth=edge_lw, color=edge_color),
        capprops=dict(linewidth=edge_lw, color=edge_color),
        flierprops=dict(marker='o', markersize=2.8, markerfacecolor='none', markeredgecolor=edge_color, alpha=0.6),
    )
    
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        
    ax_res.set_yticks([])
    ax_res.set_ylim(0.4, 1.6)
    ax_res.grid(True, axis='x', color='#e0e0e0', linewidth=0.6, alpha=0.7)
    ax_res.spines['left'].set_visible(False)
    ax_res.spines['top'].set_visible(False)
    ax_res.spines['right'].set_visible(False)
    ax_res.tick_params(axis='y', left=False, labelleft=False)
    ax_res.set_xlabel('Residual', fontsize=font_size - 3, fontweight='bold')
    ax_res.tick_params(axis='x', labelsize=font_size - 3)
    ax_res.xaxis.set_major_locator(MaxNLocator(5))


# ── Process Files ──────────────────────────────────────────────────────────
all_predicted = []
pattern = str(PRED_DIR / "gpr_predicted_dM_*.csv")
files = glob.glob(pattern)

for fpath in files:
    site = os.path.basename(fpath).replace("gpr_predicted_dM_", "").replace(".csv", "")
    print(f"Reading precomputed predictions for {site}...")
    df = pd.read_csv(fpath)
    all_predicted.append(df)

if all_predicted:
    df_all = pd.concat(all_predicted, ignore_index=True)
    df_mar_oct = df_all[(df_all['Month'] >= 3) & (df_all['Month'] <= 10)].copy()
    
    print("\n── Generating Pro Scatter Plot ──────────────────")
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    
    build_pro_scatter_axes(
        ax=ax1, 
        obs=df_mar_oct['ghi'], 
        pred=df_mar_oct['gpr_ghi'], 
        xlabel='Measured GHI [W/(m$^2$)]', 
        ylabel='GPR Predicted GHI [W/(m$^2$)]', 
        title='GHI Validation',
        color="#4C72B0",
        marker_label="(a)"
    )
    
    plt.tight_layout()
    fig.suptitle('dM V6 GHI Surrogate Performance on FY4A (Mar-Oct)', y=1.02, fontweight='bold', fontsize=14)
    
    pro_fig_name = os.path.join(OUT_DIR, "DW_Flux_MarOct_GPR_Overall_Pro_dM.png")
    fig.savefig(pro_fig_name, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {pro_fig_name}")


    print("\n── Generating Seasonal Line Plots (Per Site) ────")
    dw_season_list = []
    
    for site in df_all['Site'].unique():
        df_site = df_all[df_all['Site'] == site]
        for month in sorted(df_site['Month'].unique()):
            df_month = df_site[df_site['Month'] == month]
            mbe, rmse, rmbe, rrmse, _ = calc_metrics(df_month['ghi'], df_month['gpr_ghi'])
            dw_season_list.append({
                'Site': site, 'Month': month, 'Variable': 'GHI',
                'MBE': mbe, 'RMSE': rmse, 'rMBE': rmbe, 'rRMSE': rrmse
            })

    df_dw_season = pd.DataFrame(dw_season_list)
    df_dw_melted = df_dw_season.melt(id_vars=['Site', 'Month', 'Variable'],
                                     value_vars=['MBE', 'RMSE', 'rMBE', 'rRMSE'],
                                     var_name='Metric', value_name='Value')

    # Plot all sites in one figure
    palette_sites = sns.color_palette("Set2", len(df_all['Site'].unique()))
    
    g_dw = sns.relplot(data=df_dw_melted, x='Month', y='Value', hue='Site', col='Variable', row='Metric',
                       kind='line', marker='o', height=2.5, aspect=1.8, 
                       facet_kws={'sharey': 'row'}, palette=palette_sites, linewidth=2,
                       markeredgecolor='black', markersize=6)
    
    g_dw.set_axis_labels("Month", "")
    g_dw.set_titles(col_template="{col_name}", row_template="")
    
    metric_names_dw = ['MBE', 'RMSE', 'rMBE', 'rRMSE']
    for i, metric in enumerate(metric_names_dw):
        g_dw.axes[i, 0].set_ylabel(metric, fontweight='bold')

    for ax in g_dw.axes.flat:
        ax.axhline(0, color='gray', linestyle='dotted', linewidth=1)
        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)

    sns.despine(fig=g_dw.fig)
    g_dw.fig.suptitle('Seasonal Variance of GPR Predictions by Site (dM)', y=1.02, fontsize=16, fontweight='bold')
    
    out_name = os.path.join(OUT_DIR, "DW_Flux_Seasonal_Lines_All_Sites_dM.png")
    g_dw.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(g_dw.fig)
    print(f"  -> Saved {out_name}")
else:
    print("No data processed.")
