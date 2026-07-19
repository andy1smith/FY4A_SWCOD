from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


FONT_SIZE = 13
FONT_FAMILY = "Times New Roman"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SHORTWAVE_ROOT = REPO_ROOT.parent / "Shortwave_MCRTM"
FY4A_SITE_DIR = REPO_ROOT / "FY4A_data" / "Cloudy_site_sat_data"
GOES_SITE_DIR = SHORTWAVE_ROOT / "GOES_data" / "GOES16_site_sat_data"
CASE2_PATH = REPO_ROOT / "data" / "CIRC" / "case2_input&output" / "sfcalbedo_input_case2.txt"
FY4A_SRF_DIR = REPO_ROOT / "FY4A_data" / "AGRI_calibration"
GOES_SRF_DIR = SHORTWAVE_ROOT / "GOES_data" / "abi_calibration"

CHANNELS = ["C01", "C02", "C03", "C05", "C06"]
LUT_CHANNELS = CHANNELS
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
QNAMES = ["q05", "q25", "q50", "q75", "q95"]
LUT_ALBEDO_SETS = {
    "q05": [0.013735, 0.023839, 0.126622, 0.081469, 0.041332],
    "q25": [0.032169, 0.051783, 0.239896, 0.170084, 0.079692],
    "q50": [0.045682, 0.078521, 0.305128, 0.209253, 0.112486],
    "q75": [0.065650, 0.112386, 0.354232, 0.259652, 0.180991],
    "q95": [0.100162, 0.201559, 0.454122, 0.368143, 0.289324],
}

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.serif": [FONT_FAMILY],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
    }
)


def quantile_row(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    row: dict[str, float | int] = {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }
    for name, quantile in zip(QNAMES, QUANTILES):
        row[name] = float(np.quantile(values, quantile))
    return row


def load_fy4a_wsa() -> tuple[pd.DataFrame, pd.DataFrame]:
    site_rows = []
    all_values = {channel: [] for channel in CHANNELS}
    for path in sorted(FY4A_SITE_DIR.glob("*_SW_ref_satellite_cloudy.nc")):
        site = path.name.split("_SW_ref_satellite_cloudy.nc")[0]
        ds = xr.open_dataset(path).load()
        for channel in CHANNELS:
            name = f"WSA_{channel}"
            if name not in ds:
                continue
            values = np.asarray(ds[name].values, dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            all_values[channel].append(values)
            site_rows.append({"source": "FY4A", "site": site, "channel": channel, **quantile_row(values)})
        ds.close()

    aggregate_rows = []
    for channel in CHANNELS:
        values = np.concatenate(all_values[channel])
        aggregate_rows.append({"source": "FY4A", "channel": channel, **quantile_row(values)})
    return pd.DataFrame(site_rows), pd.DataFrame(aggregate_rows)


def load_goes_wsa() -> tuple[pd.DataFrame, pd.DataFrame]:
    site_rows = []
    all_values = {channel: [] for channel in CHANNELS}
    for path in sorted(GOES_SITE_DIR.glob("GOES_day_*_radiance_water_2019_MarOct.csv")):
        site = path.name.split("GOES_day_")[1].split("_radiance_water_2019_MarOct.csv")[0]
        usecols = [f"WSA_{channel}" for channel in CHANNELS]
        df = pd.read_csv(path, usecols=usecols)
        for channel in CHANNELS:
            values = pd.to_numeric(df[f"WSA_{channel}"], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            all_values[channel].append(values)
            site_rows.append({"source": "GOES", "site": site, "channel": channel, **quantile_row(values)})

    aggregate_rows = []
    for channel in CHANNELS:
        values = np.concatenate(all_values[channel])
        aggregate_rows.append({"source": "GOES", "channel": channel, **quantile_row(values)})
    return pd.DataFrame(site_rows), pd.DataFrame(aggregate_rows)


def save_lut_table() -> pd.DataFrame:
    rows = []
    for label, values in LUT_ALBEDO_SETS.items():
        row = {"label": label}
        row.update({channel: value for channel, value in zip(LUT_CHANNELS, values)})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(SCRIPT_DIR / "updated_fy4a_wsa_lut_albedo_sets.csv", index=False)
    return df


def plot_aggregate_quantiles(df: pd.DataFrame, title: str, out_path: Path, lut_df: pd.DataFrame | None = None) -> None:
    x = np.arange(len(CHANNELS))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y05 = df.set_index("channel").loc[CHANNELS, "q05"].to_numpy(float)
    y95 = df.set_index("channel").loc[CHANNELS, "q95"].to_numpy(float)
    ax.fill_between(x, y05, y95, color="#9ecae1", alpha=0.35, label="q05-q95")
    colors = {
        "q05": "#1f77b4",
        "q25": "#2ca02c",
        "q50": "#111111",
        "q75": "#ff7f0e",
        "q95": "#d62728",
    }
    for name in QNAMES:
        ax.plot(x, df.set_index("channel").loc[CHANNELS, name], marker="o", lw=1.8, color=colors[name], label=name)
    if lut_df is not None:
        for name in QNAMES:
            ax.scatter(
                x,
                lut_df.set_index("label").loc[name, CHANNELS].to_numpy(float),
                marker="x",
                s=55,
                color=colors[name],
                linewidth=1.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("Surface spectral albedo")
    #ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.legend(ncol=3, fontsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_site_ranges(site_df: pd.DataFrame, title: str, out_path: Path) -> None:
    sites = sorted(site_df["site"].unique())
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(15.8, max(4.8, 0.22 * len(sites))), sharey=True)
    y = np.arange(len(sites))
    for ax, channel in zip(axes, CHANNELS):
        sub = site_df[site_df["channel"] == channel].set_index("site").loc[sites]
        ax.hlines(y, sub["q05"], sub["q95"], color="#6baed6", lw=2.0)
        ax.scatter(sub["q50"], y, color="#111111", s=12, zorder=3)
        ax.scatter(sub["q25"], y, color="#31a354", s=10, zorder=3)
        ax.scatter(sub["q75"], y, color="#fd8d3c", s=10, zorder=3)
        ax.set_title(channel)
        ax.set_xlabel("Surface spectral albedo")
        ax.grid(True, axis="x", linestyle=":", alpha=0.45)
        ax.tick_params(labelsize=FONT_SIZE)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sites, fontsize=FONT_SIZE)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_fy4a_goes_comparison(fy4a_df: pd.DataFrame, goes_df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(CHANNELS))
    width = 0.28
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for offset, df, label, color in [(-width / 2, fy4a_df, "CERN sites", "#3182bd"), (width / 2, goes_df, "SURFRAD sites", "#e6550d")]:
        sub = df.set_index("channel").loc[CHANNELS]
        ax.vlines(x + offset, sub["q05"], sub["q95"], color=color, lw=5, alpha=0.35)
        ax.vlines(x + offset, sub["q25"], sub["q75"], color=color, lw=8, alpha=0.70)
        ax.scatter(x + offset, sub["q50"], color=color, s=35, label=label, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("Surface spectral albedo")
    ax.set_title("CERN vs SURFRAD site WSA quantiles")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_case2_spectrum(out_path: Path) -> None:
    data = np.genfromtxt(CASE2_PATH, skip_header=6)
    wavenumber = data[:, 0]
    albedo = data[:, 1]
    wavelength_um = 1e4 / wavenumber
    order = np.argsort(wavelength_um)
    spectrum = pd.DataFrame({"wavelength_um": wavelength_um[order], "wavenumber_cm-1": wavenumber[order], "albedo": albedo[order]})
    spectrum.to_csv(SCRIPT_DIR / "case2_surface_albedo_spectrum.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot(spectrum["wavelength_um"], spectrum["albedo"], color="#252525", lw=1.8)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Surface spectral albedo")
    ax.set_title("Base Case2 surface albedo spectrum")
    ax.set_xlim(float(spectrum["wavelength_um"].min()), float(spectrum["wavelength_um"].max()))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=FONT_SIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def load_case2_spectrum() -> pd.DataFrame:
    data = np.genfromtxt(CASE2_PATH, skip_header=6)
    wavenumber = data[:, 0]
    albedo = data[:, 1]
    wavelength_nm = 1e7 / wavenumber
    order = np.argsort(wavelength_nm)
    return pd.DataFrame(
        {
            "wavelength_nm": wavelength_nm[order],
            "wavenumber_cm-1": wavenumber[order],
            "albedo": albedo[order],
        }
    )


def load_srf_summary(sensor: str) -> pd.DataFrame:
    rows = []
    for channel in CHANNELS:
        channel_number = int(channel[-2:])
        if sensor == "FY4A":
            path = FY4A_SRF_DIR / f"FY4A_AGRI_SRF_ch{channel_number}.txt"
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            wavelength_nm = data[:, 0]
            srf = data[:, 2]
        elif sensor == "GOES":
            path = GOES_SRF_DIR / f"GOES-R_ABI_PFM_SRF_CWG_ch{channel_number}.txt"
            data = np.loadtxt(path, skiprows=2)
            wavelength_nm = data[:, 0] * 1000.0
            srf = data[:, 2]
        else:
            raise ValueError(f"Unsupported sensor: {sensor}")

        finite = np.isfinite(wavelength_nm) & np.isfinite(srf) & (srf > 0.0)
        wavelength_nm = wavelength_nm[finite]
        srf = srf[finite]
        effective = float(np.sum(wavelength_nm * srf) / np.sum(srf))
        strong = srf >= 0.50 * np.nanmax(srf)
        rows.append(
            {
                "sensor": sensor,
                "channel": channel,
                "lambda_min_nm": float(np.min(wavelength_nm[strong])),
                "lambda_eff_nm": effective,
                "lambda_max_nm": float(np.max(wavelength_nm[strong])),
                "wavenumber_eff_cm-1": float(1e7 / effective),
            }
        )
    return pd.DataFrame(rows)


def plot_combined_spectral_albedo(
    fy4a_df: pd.DataFrame,
    goes_df: pd.DataFrame,
    out_path: Path,
) -> None:
    case2 = load_case2_spectrum()
    fy4a_srf = load_srf_summary("FY4A")
    goes_srf = load_srf_summary("GOES")
    srf = pd.concat([fy4a_srf, goes_srf], ignore_index=True)
    srf.to_csv(SCRIPT_DIR / "fy4a_goes_channel_srf_wavelengths.csv", index=False)

    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    case2_sw = case2[(case2["wavelength_nm"] >= 300.0) & (case2["wavelength_nm"] <= 2600.0)]
    ax.plot(case2_sw["wavelength_nm"], case2_sw["albedo"], color="black", lw=2.2, label="Case2 spectrum")

    width_y = {"FY4A": 0.545, "GOES": 0.525}
    width_color = {"FY4A": "#1f77b4", "GOES": "#e6550d"}
    width_label = {"FY4A": "FY4A", "GOES": "GOES/ABI"}
    for sensor in ["FY4A", "GOES"]:
        sensor_srf = srf[srf["sensor"] == sensor].set_index("channel").loc[CHANNELS]
        for idx, (channel, row) in enumerate(sensor_srf.iterrows()):
            label = width_label[sensor] if idx == 0 else None
            ax.hlines(
                width_y[sensor],
                row["lambda_min_nm"],
                row["lambda_max_nm"],
                color=width_color[sensor],
                lw=4.0,
                alpha=0.65,
                label=label,
            )

    def plot_sensor_quantiles(df: pd.DataFrame, srf_df: pd.DataFrame, label: str, color: str, offset_nm: float) -> None:
        sub = df.set_index("channel").loc[CHANNELS]
        lam = srf_df.set_index("channel").loc[CHANNELS, "lambda_eff_nm"].to_numpy(float) + offset_nm
        q05 = sub["q05"].to_numpy(float)
        q25 = sub["q25"].to_numpy(float)
        q50 = sub["q50"].to_numpy(float)
        q75 = sub["q75"].to_numpy(float)
        q95 = sub["q95"].to_numpy(float)
        ax.vlines(lam, q05, q95, color=color, lw=4.0, alpha=0.25)
        ax.vlines(lam, q25, q75, color=color, lw=7.0, alpha=0.55)
        ax.scatter(lam, q50, color=color, marker="o", s=46, label=f"{label} WSA 50th percentile", zorder=3)
        ax.scatter(lam, q05, color=color, marker="_", s=80, alpha=0.8)
        ax.scatter(lam, q95, color=color, marker="_", s=80, alpha=0.8)

    plot_sensor_quantiles(fy4a_df, fy4a_srf, "CERN", "#1f77b4", -7.0)
    plot_sensor_quantiles(goes_df, goes_srf, "SURFRAD", "#e6550d", 7.0)

    ax.set_xlim(300.0, 2550.0)
    ax.set_ylim(0.0, 0.57)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Surface spectral albedo")
    #ax.set_title("Case2 spectrum with CERN and SURFRAD site WSA q05-q95")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(1000.0, 0.018),
        bbox_transform=ax.transData,
        ncol=1,
        frameon=True,
        fontsize=FONT_SIZE,
    )

    def nm_to_wn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1e7 / x

    def wn_to_nm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1e7 / x

    secax = ax.secondary_xaxis("top", functions=(nm_to_wn, wn_to_nm))
    secax.set_xlabel("Wavenumber (cm$^{-1}$)")
    secax.set_xticks([25000, 20000, 15000, 10000, 7500, 5000, 4000])
    secax.tick_params(labelsize=FONT_SIZE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    lut_df = save_lut_table()
    fy4a_site, fy4a_agg = load_fy4a_wsa()
    goes_site, goes_agg = load_goes_wsa()

    fy4a_site.to_csv(SCRIPT_DIR / "fy4a_site_wsa_quantiles.csv", index=False)
    fy4a_agg.to_csv(SCRIPT_DIR / "fy4a_all_sites_wsa_quantiles.csv", index=False)
    goes_site.to_csv(SCRIPT_DIR / "goes_site_wsa_quantiles.csv", index=False)
    goes_agg.to_csv(SCRIPT_DIR / "goes_all_sites_wsa_quantiles.csv", index=False)

    plot_aggregate_quantiles(fy4a_agg, "CERN cloudy sites WSA quantiles", SCRIPT_DIR / "fy4a_wsa_q05_q95_by_channel.png", lut_df)
    plot_aggregate_quantiles(goes_agg, "SURFRAD cloudy sites WSA quantiles", SCRIPT_DIR / "goes_wsa_q05_q95_by_channel.png")
    plot_site_ranges(fy4a_site, "CERN site WSA q05-q95 ranges", SCRIPT_DIR / "fy4a_site_wsa_q05_q95_ranges.png")
    plot_site_ranges(goes_site, "SURFRAD site WSA q05-q95 ranges", SCRIPT_DIR / "goes_site_wsa_q05_q95_ranges.png")
    plot_fy4a_goes_comparison(fy4a_agg, goes_agg, SCRIPT_DIR / "fy4a_goes_wsa_q05_q95_compare.png")
    plot_case2_spectrum(SCRIPT_DIR / "case2_surface_albedo_spectrum.png")
    plot_combined_spectral_albedo(fy4a_agg, goes_agg, SCRIPT_DIR / "case2_fy4a_goes_wsa_q05_q95_spectral.png")

    print("Saved albedo quantile tables and plots in:", SCRIPT_DIR)
    print("\nCERN all-site WSA quantiles:")
    print(fy4a_agg[["channel", "n", *QNAMES]].round(6).to_string(index=False))
    print("\nSURFRAD all-site WSA quantiles:")
    print(goes_agg[["channel", "n", *QNAMES]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
