import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

def solar_zenith(dt_index, lat_deg, lon_deg):
    dt_utc = dt_index.tz_convert('UTC')

    # n: Day of year (replaces .timetuple().tm_yday)
    n = dt_utc.dayofyear

    # hour: Fractional hour of the day
    hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    # n = dt_utc.timetuple().tm_yday
    # hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    #
    gamma = 2.0 * np.pi / 365.0 * (n - 1 + (hour - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    time_offset = eqtime + 4.0 * lon_deg  # UTC => tz=0
    tst = (hour * 60.0 + time_offset) % 1440.0

    ha_deg = np.where(tst / 4.0 < 0, tst / 4.0 + 180.0, tst / 4.0 - 180.0)
    ha = np.deg2rad(ha_deg)

    lat = np.deg2rad(lat_deg)

    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1.0, 1.0)
    sza = np.rad2deg(np.arccos(cosz))
    return sza

def parse_datetime_part_to_utc(datetime_part):
    m = re.search(r"(\d{14})", datetime_part)
    if not m:
        raise ValueError(f"无法解析14位UTC时间: {datetime_part}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

def build_china_latlon_grid():
    target_lon = np.arange(70.0, 140.0, 0.04)  # 1750
    target_lat = np.arange(15.0, 55.0, 0.04)   # 1000
    lon2d, lat2d = np.meshgrid(target_lon, target_lat)
    return lat2d, lon2d

def read_sunzenith(hdf_path):
    with h5py.File(hdf_path, "r") as f:
        raw = f["SunZenith"][:].astype(np.float32)

    # 你的原逻辑：scale=0.02；缺测常见为32767（也兼容-9999）
    sza = raw * 0.02
    sza[(raw == 32767) | (raw == -9999)] = np.nan
    return sza

def verify_one_file(year, datetime_part, out_png, vlim_deg=5, stats_txt=None):
    hdf_path = fr"/Volumes/HP P900/FY_L1_2021/FY_L1_china_{datetime_part}.hdf5"
    if not os.path.exists(hdf_path):
        raise FileNotFoundError(hdf_path)

    dt_utc = parse_datetime_part_to_utc(datetime_part)

    sza_file = read_sunzenith(hdf_path)
    lat2d, lon2d = build_china_latlon_grid()

    if sza_file.shape != (1000, 1750):
        raise ValueError(f"SunZenith shape={sza_file.shape}，不是(1000,1750)")

    # lat = [40.0001]
    # lon = [116.3333]
    # sza_calc = solar_zenith(dt_utc, lat, lon)
    sza_calc = solar_zenith(dt_utc, lat2d, lon2d)
    err = sza_file - sza_calc  # deg

    # 画误差图
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(10, 6))
    im = plt.imshow(err, cmap="coolwarm", vmin=-vlim_deg, vmax=vlim_deg)
    plt.colorbar(im, label="SunZenith error (deg): file - calc")
    plt.title(f"FY-4A AGRI SunZenith error map\n{datetime_part} (UTC)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # 统计
    valid = np.isfinite(err)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise RuntimeError("没有有效误差像元（可能全是NaN/缺测）")

    err_valid = err[valid]

    # min/max 及位置
    min_val = float(np.nanmin(err))
    max_val = float(np.nanmax(err))

    min_idx = np.nanargmin(err)  # flatten index
    max_idx = np.nanargmax(err)

    min_r, min_c = np.unravel_index(min_idx, err.shape)
    max_r, max_c = np.unravel_index(max_idx, err.shape)

    min_lat, min_lon = float(lat2d[min_r, min_c]), float(lon2d[min_r, min_c])
    max_lat, max_lon = float(lat2d[max_r, max_c]), float(lon2d[max_r, max_c])

    # 常用统计量
    mean = float(np.nanmean(err_valid))
    median = float(np.nanmedian(err_valid))
    std = float(np.nanstd(err_valid))
    mae = float(np.nanmean(np.abs(err_valid)))
    rmse = float(np.sqrt(np.nanmean(err_valid ** 2)))

    max_abs = float(np.nanmax(np.abs(err_valid)))
    p50 = float(np.nanpercentile(err_valid, 50))
    p90 = float(np.nanpercentile(err_valid, 90))
    p95_abs = float(np.nanpercentile(np.abs(err_valid), 95))
    p99_abs = float(np.nanpercentile(np.abs(err_valid), 99))

    report_lines = [
        f"输出图: {out_png}",
        f"文件: {hdf_path}",
        f"UTC时间: {dt_utc.isoformat()}",
        "",
        f"有效像元数: {n_valid}",
        "",
        "误差统计（deg，file - calc）:",
        f"  Mean      : {mean:.6f}",
        f"  Median    : {median:.6f}",
        f"  Std       : {std:.6f}",
        f"  MAE       : {mae:.6f}",
        f"  RMSE      : {rmse:.6f}",
        f"  Max(|err|): {max_abs:.6f}",
        f"  P50(err)  : {p50:.6f}",
        f"  P90(err)  : {p90:.6f}",
        f"  P95(|err|): {p95_abs:.6f}",
        f"  P99(|err|): {p99_abs:.6f}",
        "",
        "极值像元：",
        f"  Min err: {min_val:.6f} deg at (row={min_r}, col={min_c}), lat={min_lat:.2f}, lon={min_lon:.2f}",
        f"  Max err: {max_val:.6f} deg at (row={max_r}, col={max_c}), lat={max_lat:.2f}, lon={max_lon:.2f}",
    ]

    print("\n".join(report_lines))

    # 可选：保存统计文本
    if stats_txt is not None:
        os.makedirs(os.path.dirname(stats_txt), exist_ok=True)
        with open(stats_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    year = '2021' # "2019"
    #datetime_part = "20190101050000_20190101051459"  # 改成你要验证的那景
    datetime_part ="20210216020000_20210216021459"
    out_png = fr"./SunZenith_error_{datetime_part}.png"
    verify_one_file(year, datetime_part, out_png, vlim_deg=5)
