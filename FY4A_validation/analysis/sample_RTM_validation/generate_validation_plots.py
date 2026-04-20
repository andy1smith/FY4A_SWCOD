import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.stats import pearsonr

# SCI Paper Formatting
font = 13
fontfml = 'Times New Roman'

# Apply AFTER sns.set_theme so seaborn cannot override them
sns.set_theme(style="ticks", context="paper")
sns.set_style({"font.family": "serif", "font.serif": ["Times New Roman", "Times New Roman"]})

# These must come AFTER set_theme / set_style
mpl.rcParams['font.family']          = fontfml
mpl.rcParams['font.serif']           = [fontfml]
mpl.rcParams['font.size']            = font
mpl.rcParams['axes.titlesize']       = font
mpl.rcParams['axes.labelsize']       = font - 1
mpl.rcParams['xtick.labelsize']      = font - 1
mpl.rcParams['ytick.labelsize']      = font - 1
mpl.rcParams['axes.linewidth']       = 1.2
mpl.rcParams['legend.fontsize']      = font - 1
mpl.rcParams['mathtext.fontset']     = 'custom'
mpl.rcParams['mathtext.rm']          = fontfml
mpl.rcParams['mathtext.it']          = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf']          = 'Times New Roman:bold'

data_dir = "../../Clear_Test/"
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(data_dir, "sampled_*_satdata_clearsky_HG_BRDF.csv"))

sites = []
all_data = []
for f in files:
    site_name = os.path.basename(f).split('_')[1]
    df = pd.read_csv(f)
    df = df[df['Site_zen']<=65]
    df = df[df['direct_n']>=200]
    df = df[df['Site_usw']<=250]
    df = df[df['C01']<0.19]
    if site_name =='FPK':
        df = df[df['direct_n']-df['rtm_dni']<300]
    df = df[df['rtm_dni']>400]
    df = df[df['C06'] > 0.05]
    df['Time'] = pd.to_datetime(df['Time'])
    df['Month'] = df['Time'].dt.month
    df['Site'] = site_name
    all_data.append(df)
    sites.append(site_name)

if not all_data:
    print("No data found!")
    exit()

df_all = pd.concat(all_data, ignore_index=True)
df_mar_oct = df_all[(df_all['Month'] >= 3) & (df_all['Month'] <= 10)].copy()

unique_sites = sorted(df_all['Site'].unique())
# Okabe-Ito palette (colorblind-safe, Nature/Science standard)
okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
palette_sites = okabe_ito[:len(unique_sites)]

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

channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
rtm_channels = [f"{ch}_rtm" for ch in channels]

# --- 1. UW Seasonal (Faceted Heatmaps) ---
uw_season_list = []
for site in df_all['Site'].unique():
    df_site = df_all[df_all['Site'] == site]
    for month in sorted(df_site['Month'].unique()):
        df_month = df_site[df_site['Month'] == month]
        for ch, rtm_ch in zip(channels, rtm_channels):
            mbe, rmse, rmbe, rrmse, r_val = calc_metrics(df_month[ch], df_month[rtm_ch])
            uw_season_list.append({
                'Site': site, 'Month': month, 'Channel': ch,
                'rMBE': rmbe, 'rRMSE': rrmse, 'R': r_val
            })

df_uw_season = pd.DataFrame(uw_season_list)

uw_plot_channels = [c for c in channels if c != 'C04']
metrics_uw = ['rMBE', 'rRMSE', 'R']
fig_uw, axes_uw = plt.subplots(len(uw_plot_channels), len(metrics_uw), figsize=(18, 14), sharex=True)

for i, ch in enumerate(uw_plot_channels):
    df_ch = df_uw_season[df_uw_season['Channel'] == ch]
    for j, metric in enumerate(metrics_uw):
        ax = axes_uw[i, j]
        pivot_df = df_ch.pivot(index='Site', columns='Month', values=metric)
        
        if metric == 'rMBE':
            cmap = 'RdBu_r'
            center = 0
            vmax = np.nanmax(np.abs(pivot_df.values))
            vmin = -vmax
        elif metric == 'R':
            cmap = 'YlGnBu'
            center = None
            vmin, vmax = 0, 1
        else:
            cmap = 'YlOrRd'
            center = None
            vmin = 0
            vmax = np.nanpercentile(pivot_df.values, 95)

        sns.heatmap(pivot_df, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='lightgray',
                    annot=True, fmt=".1f", annot_kws={"size": 10, "family": fontfml},
                    cbar_kws={'shrink': 0.8})
        
        ax.set_title(f"{ch} {metric}", fontweight='bold', family=fontfml)
        ax.set_xlabel('Month' if i == len(uw_plot_channels)-1 else '', family=fontfml)
        
        if j == 0:
            ax.set_ylabel(ch, fontweight='bold', family=fontfml)
            ax.set_yticklabels(pivot_df.index, rotation=0, family=fontfml)
        else:
            ax.set_ylabel('')
            ax.set_yticks([])

    # Enforce Times New Roman on every tick label explicitly
    # This loop is inside the channel loop by error? No, it should be inside or after plot loop.
    # Fixed to be after metric loop.

for ax in axes_uw.flat:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(fontfml)

plt.tight_layout()
fig_uw.suptitle("Seasonal Variance of Upwelling Reflectance Factor", y=1.02, fontsize=18, fontweight='bold', family=fontfml)
fig_uw.savefig(os.path.join(output_dir, "UW_Reflectance_Seasonal_Variance.png"), dpi=400, bbox_inches='tight')
plt.close(fig_uw)

# --- 2. UW Mar-Oct Overall (Barplots) ---
uw_overall_list = []
for site in df_mar_oct['Site'].unique():
    df_site = df_mar_oct[df_mar_oct['Site'] == site]
    for ch, rtm_ch in zip(channels, rtm_channels):
        mbe, rmse, rmbe, rrmse, r_val = calc_metrics(df_site[ch], df_site[rtm_ch])
        uw_overall_list.append({
            'Site': site, 'Channel': ch,
            'rMBE': rmbe, 'rRMSE': rrmse, 'R': r_val
        })

df_uw_overall = pd.DataFrame(uw_overall_list)
df_uw_overall_plot = df_uw_overall[df_uw_overall['Channel'] != 'C04']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Using Spectral for channel coloring as in original, which is distinct from site coloring.
palette_ch = sns.color_palette("Spectral", len(uw_plot_channels))

sns.barplot(data=df_uw_overall_plot, x='Site', y='rMBE', hue='Channel', ax=axes[0], palette=palette_ch, edgecolor='black', linewidth=1.2)
axes[0].set_title('March-October rMBE (%)', fontweight='bold', family=fontfml)
axes[0].axhline(0, color='black', linewidth=1.2, linestyle='--')

sns.barplot(data=df_uw_overall_plot, x='Site', y='rRMSE', hue='Channel', ax=axes[1], palette=palette_ch, edgecolor='black', linewidth=1.2)
axes[1].set_title('March-October rRMSE (%)', fontweight='bold', family=fontfml)

sns.barplot(data=df_uw_overall_plot, x='Site', y='R', hue='Channel', ax=axes[2], palette=palette_ch, edgecolor='black', linewidth=1.2)
axes[2].set_title('March-October R', fontweight='bold', family=fontfml)

for ax in axes:
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(fontfml)
    # Legend font
    leg = ax.legend(frameon=True)
    if leg:
        plt.setp(leg.get_texts(), family=fontfml)

sns.despine(fig)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "UW_Reflectance_MarOct_Overall.png"), dpi=400, bbox_inches='tight')
plt.close(fig)

# --- 3. DW Seasonal (Faceted Heatmaps) ---
dw_season_list = []
for site in df_all['Site'].unique():
    df_site = df_all[df_all['Site'] == site]
    for month in sorted(df_site['Month'].unique()):
        df_month = df_site[df_site['Month'] == month]
        mbe, rmse, rmbe, rrmse, _ = calc_metrics(df_month['Site_dsw'], df_month['rtm_dsw'])
        dw_season_list.append({'Site': site, 'Month': month, 'Variable': 'GHI', 'MBE': mbe, 'RMSE': rmse, 'rMBE': rmbe, 'rRMSE': rrmse})
        mbe, rmse, rmbe, rrmse, _ = calc_metrics(df_month['direct_n'], df_month['rtm_dni'])
        dw_season_list.append({'Site': site, 'Month': month, 'Variable': 'DNI', 'MBE': mbe, 'RMSE': rmse, 'rMBE': rmbe, 'rRMSE': rrmse})

df_dw_season = pd.DataFrame(dw_season_list)

dw_vars = ['GHI', 'DNI']
metrics_dw = ['MBE', 'RMSE', 'rMBE', 'rRMSE']
fig_dw, axes_dw = plt.subplots(len(dw_vars), len(metrics_dw), figsize=(20, 8), sharex=True)

for i, var in enumerate(dw_vars):
    df_var = df_dw_season[df_dw_season['Variable'] == var]
    for j, metric in enumerate(metrics_dw):
        ax = axes_dw[i, j]
        pivot_df = df_var.pivot(index='Site', columns='Month', values=metric)
        
        if 'MBE' in metric:
            cmap = 'RdBu_r'
            center = 0
            vmax = np.nanmax(np.abs(pivot_df.values))
            vmin = -vmax
        else:
            cmap = 'YlOrRd'
            center = None
            vmin = 0
            vmax = np.nanpercentile(pivot_df.values, 95)

        sns.heatmap(pivot_df, ax=ax, cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='lightgray',
                    annot=True, fmt=".1f", annot_kws={"size": 11, "family": fontfml},
                    cbar_kws={'shrink': 0.8})
        
        ax.set_title(f"{var} {metric}", fontweight='bold', family=fontfml)
        ax.set_xlabel('Month' if i == len(dw_vars)-1 else '', family=fontfml)
        
        if j == 0:
            ax.set_ylabel(var, fontweight='bold', family=fontfml)
            ax.set_yticklabels(pivot_df.index, rotation=0, family=fontfml)
        else:
            ax.set_ylabel('')
            ax.set_yticks([])

for ax in axes_dw.flat:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(fontfml)

plt.tight_layout()
fig_dw.suptitle("Seasonal Variance of Downwelling Fluxes", y=1.05, fontsize=18, fontweight='bold', family=fontfml)
fig_dw.savefig(os.path.join(output_dir, "DW_Flux_Seasonal_Variance.png"), dpi=400, bbox_inches='tight')
plt.close(fig_dw)

# --- 4. DW Mar-Oct Overall (Barplots) ---
dw_overall_list = []
for site in df_mar_oct['Site'].unique():
    df_site = df_mar_oct[df_mar_oct['Site'] == site]
    mbe, rmse, rmbe, rrmse, _ = calc_metrics(df_site['Site_dsw'], df_site['rtm_dsw'])
    dw_overall_list.append({'Site': site, 'Variable': 'GHI', 'MBE': mbe, 'RMSE': rmse, 'rMBE': rmbe, 'rRMSE': rrmse})
    mbe, rmse, rmbe, rrmse, _ = calc_metrics(df_site['direct_n'], df_site['rtm_dni'])
    dw_overall_list.append({'Site': site, 'Variable': 'DNI', 'MBE': mbe, 'RMSE': rmse, 'rMBE': rmbe, 'rRMSE': rrmse})

df_dw_overall = pd.DataFrame(dw_overall_list)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Use explicit Blue (GHI) and Red (DNI)
palette_dw = ["#1f77b4", "#d62728"]

sns.barplot(data=df_dw_overall, x='Site', y='MBE', hue='Variable', ax=axes[0,0], palette=palette_dw, edgecolor='black', linewidth=1.2)
axes[0,0].set_title('March-October MBE', fontweight='bold', family=fontfml)
axes[0,0].axhline(0, color='black', linewidth=1.2, linestyle='--')

sns.barplot(data=df_dw_overall, x='Site', y='RMSE', hue='Variable', ax=axes[0,1], palette=palette_dw, edgecolor='black', linewidth=1.2)
axes[0,1].set_title('March-October RMSE', fontweight='bold', family=fontfml)

sns.barplot(data=df_dw_overall, x='Site', y='rMBE', hue='Variable', ax=axes[1,0], palette=palette_dw, edgecolor='black', linewidth=1.2)
axes[1,0].set_title('March-October rMBE (%)', fontweight='bold', family=fontfml)
axes[1,0].axhline(0, color='black', linewidth=1.2, linestyle='--')

sns.barplot(data=df_dw_overall, x='Site', y='rRMSE', hue='Variable', ax=axes[1,1], palette=palette_dw, edgecolor='black', linewidth=1.2)
axes[1,1].set_title('March-October rRMSE (%)', fontweight='bold', family=fontfml)

for ax in axes.flat:
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(fontfml)
    leg = ax.legend(frameon=True)
    if leg:
        plt.setp(leg.get_texts(), family=fontfml)

sns.despine(fig)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "DW_Flux_MarOct_Overall.png"), dpi=400, bbox_inches='tight')
plt.close(fig)

df_uw_overall.to_csv(os.path.join(output_dir, "Summary_UW_MarOct.csv"), index=False)
df_dw_overall.to_csv(os.path.join(output_dir, "Summary_DW_MarOct.csv"), index=False)
print("Saved SCI-styled heatmaps and barplots to", output_dir)
