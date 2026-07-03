# FY4A ADM cloud LUT update

This folder contains the maintained FY4A AGRI angular distribution model code.
It follows the updated GOES ADM workflow in `Shortwave_MCRTM/GOES_data/ADM_cloud`.

## Files

- `AngDistLUT.py`: FY4A channel ADM utilities and SVD HDF5 LUT read/write functions.
- `run_adm_cases.py`: runs the RTM photon-direction cases needed for the LUT grid.
- `generate_adm_lut.py`: builds `angular_dist_lut_COD=*.h5` files from saved RTM outputs.
- `LUT/`: default output folder for generated HDF5 LUTs.

## Default grid

- Channels: `C01 C02 C03 C04 C05 C06`
- COD: `0, 2, 4, ..., 20, 25, 30, 35`
- Solar zenith: `0, 15, 30, 45, 60, 65`
- Angular bins: theta `5 deg`, relative azimuth `10 deg`, symmetric `0-180 deg`

## Usage

Run missing RTM cases:

```bash
python FY4A_data/ADM_cloud/run_adm_cases.py
```

Generate LUTs from existing RTM outputs:

```bash
python FY4A_data/ADM_cloud/generate_adm_lut.py
```

If the RTM outputs live in a custom folder:

```bash
python FY4A_data/ADM_cloud/generate_adm_lut.py --rtm-dir /path/to/RTM_10000/channels/FY4A
```
