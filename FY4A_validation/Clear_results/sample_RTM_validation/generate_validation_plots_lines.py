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
    if site_name == 'PSU':
        df = df[df['Month']!=5]
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

# --- 1. UW Seasonal (Lines) ---
uw_season_list = []
for site in df_all['Site'].unique():
    df_site = df_all[df_all['Site'] == site]
    for month in sorted(df_site['Month'].unique()):
        df_month = df_site[df_site['Month'] == month]
        for ch, rtm_ch in zip(channels, rtm_channels):
            if ch == 'C04':
                continue
            mbe, rmse, rmbe, rrmse, r_val = calc_metrics(df_month[ch], df_month[rtm_ch])
            uw_season_list.append({
                'Site': site, 'Month': month, 'Channel': ch,
                'rMBE': rmbe, 'rRMSE': rrmse, 'R': r_val
            })

df_uw_season = pd.DataFrame(uw_season_list)
df_uw_melted = df_uw_season.melt(id_vars=['Site', 'Month', 'Channel'],
                                 value_vars=['rMBE', 'rRMSE', 'R'],
                                 var_name='Metric',
                                 value_name='Value')

g_uw = sns.relplot(data=df_uw_melted,
                   x='Month',
                   y='Value',
                   hue='Site',
                   col='Channel',
                   row='Metric',
                   kind='line',
                   marker='o',
                   height=3,
                   aspect=1.2,
                   facet_kws={'sharey': 'row'},
                   palette=palette_sites,
                   linewidth=1.5,
                   markeredgecolor='black',
                   markersize=6)

g_uw.set_axis_labels("Month", "", family=fontfml)
g_uw.set_titles(col_template="{col_name}", row_template="")
metric_names_uw = ['rMBE', 'rRMSE', 'R']
for i, metric in enumerate(metric_names_uw):
    g_uw.axes[i, 0].set_ylabel(metric, fontweight='bold', family=fontfml)

for ax in g_uw.axes.flat:
    # Set titles and labels fonts explicitly just in case
    ax.set_title(ax.get_title(), family=fontfml)
    ax.set_xlabel(ax.get_xlabel(), family=fontfml)
    ax.axhline(0, color='gray', linestyle='dotted', linewidth=1)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(fontfml)

# Update legend font
if g_uw._legend:
    plt.setp(g_uw._legend.get_texts(), family=fontfml)
    g_uw._legend.set_title("Site", prop={'family': fontfml, 'weight': 'bold'})

sns.despine(fig=g_uw.fig)
g_uw.fig.suptitle('Seasonal Variance of Upwelling Reflectance Factor',
                  y=1.02, fontsize=18, fontweight='bold', family=fontfml)
g_uw.savefig(os.path.join(output_dir, "UW_Reflectance_Seasonal_Variance_lines.png"),
             dpi=600, bbox_inches='tight')
plt.close(g_uw.fig)

# --- 2. DW Seasonal (Lines) ---
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
df_dw_melted = df_dw_season.melt(id_vars=['Site', 'Month', 'Variable'],
                                 value_vars=['MBE', 'RMSE', 'rMBE', 'rRMSE'],
                                 var_name='Metric',
                                 value_name='Value')

g_dw = sns.relplot(data=df_dw_melted,
                   x='Month',
                   y='Value',
                   hue='Site',
                   col='Variable',
                   row='Metric',
                   kind='line',
                   marker='o',
                   height=3,
                   aspect=1.5,
                   facet_kws={'sharey': 'row'},
                   palette=palette_sites,
                   linewidth=1.5,
                   markeredgecolor='black',
                   markersize=6)

g_dw.set_axis_labels("Month", "", family=fontfml)
g_dw.set_titles(col_template="{col_name}", row_template="")
metric_names_dw = ['MBE', 'RMSE', 'rMBE', 'rRMSE']
for i, metric in enumerate(metric_names_dw):
    g_dw.axes[i, 0].set_ylabel(metric, fontweight='bold', family=fontfml)

for ax in g_dw.axes.flat:
    # Set titles and labels fonts explicitly just in case
    ax.set_title(ax.get_title(), family=fontfml)
    ax.set_xlabel(ax.get_xlabel(), family=fontfml)
    ax.axhline(0, color='gray', linestyle='dotted', linewidth=1)
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(fontfml)

# Update legend font
if g_dw._legend:
    plt.setp(g_dw._legend.get_texts(), family=fontfml)
    g_dw._legend.set_title("Site", prop={'family': fontfml, 'weight': 'bold'})

sns.despine(fig=g_dw.fig)
g_dw.fig.suptitle('Seasonal Variance of Downwelling Fluxes',
                  y=1.02, fontsize=18, fontweight='bold', family=fontfml)
g_dw.savefig(os.path.join(output_dir, "DW_Flux_Seasonal_Variance_lines.png"),
             dpi=600, bbox_inches='tight')
plt.close(g_dw.fig)

print("Saved updated line-based SCI seasonal variance plots.")
