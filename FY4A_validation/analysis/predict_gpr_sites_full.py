"""
predict_gpr_sites_full.py

Predicts clear-sky GHI and DNI for the full Mar-Oct 2019 validation data using the trained 
Gaussian Process Regression (GPR) surrogate v2 model.

Plots:
1. Pro-style marginal density scatter figure for GHI and DNI (combining all sites).
2. Seasonal line plots (MBE, RMSE, rMBE, rRMSE) per site to investigate variations (e.g. TBL).
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde, pearsonr

# Ensure aod_codes can be imported from root directory
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
from aod_codes import read_aod

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(ROOT_DIR, "Surrogate_GRP_COD/AOD_dsw_model_HG/SWRTM_AOD_clearsky_GPR_v2.pkl")
DATA_DIR     = os.path.join(ROOT_DIR, "GOES_data/GOES16_site_sat_data")
OUT_DIR      = os.path.join(ROOT_DIR, "GOES_validation/analysis")
os.makedirs(OUT_DIR, exist_ok=True)

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


# ── Load Model ─────────────────────────────────────────────────────────────
print("Loading GPR v2 model …")
bundle     = joblib.load(MODEL_PATH)
scaler_X   = bundle['scaler_X']
scaler_y   = bundle['scaler_y']
gpr_models = bundle['gpr_models']

def predict(df_raw):
    Ts_K        = df_raw['T'].values
    rh          = df_raw['RH'].values / 100.0   
    aod         = df_raw['aod'].values
    th0         = df_raw['Site_zen'].values
    cos_th0     = np.cos(np.radians(th0))
    exp_neg_aod = np.exp(-aod)
    interaction = cos_th0 * exp_neg_aod

    X    = np.column_stack([Ts_K, rh, cos_th0, exp_neg_aod, interaction])
    X_sc = scaler_X.transform(X)
    
    # HG model only has ghi and dni
    y_sc_cols = [gpr_models[t].predict(X_sc) for t in ['ghi', 'dni']]
    y_sc   = np.column_stack(y_sc_cols)
    return scaler_y.inverse_transform(y_sc)

def apply_filters(df):
    df = df[df['Site_zen']  <= 65].copy()
    df = df[df['direct_n']  >= 200]
    df = df[df['Site_usw']  <= 250]
    df = df[df['C01']        < 0.19]
    df = df[df['C06']        > 0.05]
    return df

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

# ───────────────────────────────────────────────────────────────────────────
# Pro-style Subplot Builder
# ───────────────────────────────────────────────────────────────────────────
def build_pro_scatter_axes(ax, obs, pred, xlabel, ylabel, title, color="#c8e3eb", marker_label='(a)'):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]

    ax.scatter(obs, pred, s=10, color=color, alpha=0.80, edgecolors='none')
    
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

    # Metrics Text
    mbe, rmse, rmbe, rrmse, r = calc_metrics(obs, pred)
    r2 = r**2 if not np.isnan(r) else np.nan
    stats_text = (f"$R^2$={r2:.3f}\n"
                  f"RMSE={rmse:.1f}\n"
                  f"MBE={mbe:.1f}\n"
                  f"rRMSE={rrmse:.1f}%\n"
                  f"rMBE={rmbe:.1f}%")
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top', 
            fontsize=font_size, color='#990000', weight='bold')
    
    ax.text(-0.1, 1.02, marker_label, transform=ax.transAxes, ha='left', va='bottom', 
            fontsize=font_size + 2, fontweight='bold')
    #ax.set_title(title, pad=15)

    # Inset Boxplot for Residuals
    resi = pred - obs
    ax_res = inset_axes(ax, width="40%", height="35%", loc="lower right", 
                        borderpad=0.0, bbox_to_anchor=(-0.05, 0.1, 1.0, 1.0), 
                        bbox_transform=ax.transAxes)
    
    if resi.size > 0:
        rmax = float(np.nanmax(np.abs(resi)))
        rpad = rmax * 0.10 if rmax > 0 else 1.0
        ax_res.set_xlim(-rmax - rpad, rmax + rpad)
    
    ax_res.axvline(0, color='k', linestyle='--', linewidth=1.0, alpha=0.85)
    
    bp = ax_res.boxplot([resi], vert=False, patch_artist=True, widths=0.55, showfliers=True,
                        boxprops=dict(linewidth=edge_lw, color=edge_color),
                        medianprops=dict(linewidth=1.2, color='k'),
                        whiskerprops=dict(linewidth=edge_lw, color=edge_color),
                        capprops=dict(linewidth=edge_lw, color=edge_color),
                        flierprops=dict(marker='o', markersize=2.8, markerfacecolor='none', 
                                        markeredgecolor=edge_color, alpha=0.6))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        
    ax_res.set_yticks([])
    ax_res.grid(True, axis='x', color='#e0e0e0', linewidth=0.6, alpha=0.7)
    for spine in ['left', 'top', 'right']:
        ax_res.spines[spine].set_visible(False)
    #ax_res.set_xlabel('Residual', fontsize=font_size - 2, fontweight='bold')
    ax_res.tick_params(axis='x', labelsize=font_size - 3)
    ax_res.xaxis.set_major_locator(MaxNLocator(5))


# ── Process Files ──────────────────────────────────────────────────────────
SITES = ['BON', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF', 'TBL']
all_predicted = []

for site in SITES:
    pattern = os.path.join(DATA_DIR, f"GOES_day_{site}_radiance_clearsky_2019_MarOct.csv")
    files = glob.glob(pattern)
    if not files:
        continue
    fpath = files[0]
    print(f"── Processing {site} ──────────────────────")
    df = pd.read_csv(fpath)
    
    cols_to_keep = ['Time', 'Site_zen', 'Site_dsw', 'Site_usw', 'direct_n', 'diffuse', 
                    'C01', 'C06', 'T', 'RH', 'rtm_dsw', 'rtm_dni']
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df = apply_filters(df)
    
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    
    df_aod = read_aod(site)
    c_index = df.index.intersection(df_aod.index)
    if len(c_index) == 0:
        continue
        
    df = pd.concat([df.loc[c_index], df_aod.loc[c_index]], join='inner', axis=1)
    df = df[~df.index.duplicated(keep='first')]  
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Time'}, inplace=True)
    
    df['Site'] = site
    df['Month'] = df['Time'].dt.month
    
    preds = predict(df)
    df['gpr_ghi'] = preds[:, 0]
    df['gpr_dni'] = preds[:, 1]
    df = df[df['gpr_dni']>400]

    if site =='FPK':
        df = df[df['direct_n']-df['gpr_dni']<300]
    if site =='PSU':
        df = df[df['direct_n']-df['gpr_dni']<300]
        
    # Save predicted CSV for generate_gpr_validation_plots.py
    out_csv = os.path.join(ROOT_DIR, f"GOES_validation/Clear_Test/gpr_predicted_{site}.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Generated {len(df)} predictions and saved to {os.path.basename(out_csv)}")
    
    all_predicted.append(df)

if all_predicted:
    df_all = pd.concat(all_predicted, ignore_index=True)
    df_mar_oct = df_all[(df_all['Month'] >= 3) & (df_all['Month'] <= 10)].copy()
    
    print("\n── Generating Pro Scatter Plot ──────────────────")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    build_pro_scatter_axes(
        ax=ax1, 
        obs=df_mar_oct['Site_dsw'], 
        pred=df_mar_oct['gpr_ghi'], 
        xlabel='Measured GHI [W/(m$^2$)]', 
        ylabel='GPR Predicted GHI [W/(m$^2$)]', 
        title='GHI Validation',
        color="#4C72B0",
        marker_label="(a)"
    )
    
    build_pro_scatter_axes(
        ax=ax2, 
        obs=df_mar_oct['direct_n'], 
        pred=df_mar_oct['gpr_dni'], 
        xlabel='Measured DNI [W/(m$^2$)]', 
        ylabel='GPR Predicted DNI [W/(m$^2$)]', 
        title='DNI Validation',
        color="#DD8452",
        marker_label="(b)"
    )
    
    plt.tight_layout()
    # A bit of horizontal spacing to accommodate marginal hists
    plt.subplots_adjust(wspace=0.4, top=0.88)
    fig.suptitle('GPR v2 Performance on 2019 SURFRAD (Mar-Oct)', y=1.02, fontweight='bold', fontsize=14)
    
    pro_fig_name = os.path.join(OUT_DIR, "DW_Flux_MarOct_GPR_Overall_Pro_HG.png")
    fig.savefig(pro_fig_name, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {pro_fig_name}")


    print("\n── Generating Seasonal Line Plots (Per Site) ────")
    dw_season_list = []
    
    for site in df_all['Site'].unique():
        df_site = df_all[df_all['Site'] == site]
        for month in sorted(df_site['Month'].unique()):
            df_month = df_site[df_site['Month'] == month]
            mbe, rmse, rmbe, rrmse, _ = calc_metrics(df_month['Site_dsw'], df_month['gpr_ghi'])
            dw_season_list.append({
                'Site': site, 'Month': month, 'Variable': 'GHI',
                'MBE': mbe, 'RMSE': rmse, 'rMBE': rmbe, 'rRMSE': rrmse
            })
            
            mbe, rmse, rmbe, rrmse, _ = calc_metrics(df_month['direct_n'], df_month['gpr_dni'])
            dw_season_list.append({
                'Site': site, 'Month': month, 'Variable': 'DNI',
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
    g_dw.fig.suptitle('Seasonal Variance of GPR Predictions by Site', y=1.02, fontsize=16, fontweight='bold')
    
    out_name = os.path.join(OUT_DIR, "DW_Flux_Seasonal_Lines_All_Sites_HG.png")
    g_dw.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(g_dw.fig)
    print(f"  -> Saved {out_name}")
else:
    print("No data processed.")
