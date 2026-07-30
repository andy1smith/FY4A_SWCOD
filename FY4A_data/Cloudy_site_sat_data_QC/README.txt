Cloudy_site_sat_data_QC_GHI_20260722_140524
Created: 2026-07-22 14:05-14:40 local run time, based on folder timestamps.
Updated: 2026-07-24, rerun after removing extraction-stage GHI_clear > 300
and clear-index >= 0.15 filters.

Purpose
-------
This folder contains FY4A 11x11 cloudy site-satellite NetCDF files for CERN
2021 ground GHI validation. Each file is named:

    <SITE>_SW_ref_satellite_cloudy.nc

The folder contains 23 site files and 19,599 retained time samples in total.


Input Ground Data
-----------------
The source ground data were the active cloudy HDF files:

    Sat_Preprocessing/Ground/preprocessed_GHI/*_cloudy.h5

Those cloudy HDF files were regenerated from raw CERN 2021 GHI using McClear
clear-sky GHI. The previous cloudy HDF files were backed up at:

    Sat_Preprocessing/Ground/preprocessed_GHI/backup_cloudy_h5_before_mcclear_qc_20260722


Upstream Ground Cloudy QC
-------------------------
The active cloudy HDF input used this ground-GHI QC:

1. Cloudy candidate selection:
   - Not selected as clear by the strict quantile/stability clear-sky detector.
   - Not selected as clear by the SURFRAD-style rolling clear-sky test.

2. Low-irradiance truncation:
   - Keep only records with measured GHI >= 50 W/m2.

3. McClear clear-index physical screen:
   - clear_index_mcclear = GHI / ghi_clear_mcclear.
   - Keep only records with clear_index_mcclear >= 0.03.

4. Empirical lower GHI screen:
   - Keep only records with measured GHI >= GHI_min(SZA), where:

     GHI_min = (6.5331 - 0.065502 * Z + 1.8312e-4 * Z^2) / (1 + 0.01113 * Z)

   - Z is the ground solar zenith angle in degrees.


Site-Satellite Extraction QC Used For This Folder
-------------------------------------------------
After reading the active cloudy HDF files, the FY4A site-satellite extraction
applied only these additional filters:

1. Ground solar zenith filter:
   - Sun_Zen_ground <= 65 degrees.

2. FY4A/ground time-geometry consistency:
   - abs(median(FY4A Sun_Zen over the 11x11 box) - Sun_Zen_ground) <= 1 degree.

No extraction-stage clear-sky magnitude or clear-index screen is applied:

    - No GHI_clear > 300 W/m2 filter.
    - No GHI / GHI_clear >= 0.15 filter.
    - No GHI / ghi_clear_mcclear >= 0.15 filter.


Important Note About NetCDF Attributes
--------------------------------------
The NetCDF global attributes have been updated after removing the extraction
stage GHI_clear and clear-index filters:

    clear_sky_qc_source = mcclear
    cloudy_qc = |median FY4A Sun_Zen - ground Sun_Zen| <= 1 deg;
                no extraction-stage GHI_clear > 300 W/m2 filter;
                no extraction-stage clear-index >= 0.15 filter

The NetCDF files include:

    ghi_clear_pvlib
    ghi_clear_mcclear
    clear_index_mcclear

The retained data confirm that filters 3 and 4 are not used:

    total site files:                 23
    total retained time samples:      19,599
    minimum GHI:                      50.0 W/m2
    minimum ghi_clear_mcclear:        149.0077 W/m2
    rows with ghi_clear_mcclear<=300: 629
    minimum clear_index_mcclear:      0.054663
    rows with clear_index_mcclear<0.15: 1,203
    maximum Sun_Zen_ground:           64.9999 degrees


Related Code
------------
Ground cloudy QC:

    Sat_Preprocessing/clearsky_model/clearsky_filter.py
    Sat_Preprocessing/regenerate_cloudy_ground_mcclear_qc.py

FY4A site-satellite extraction:

    Sat_Preprocessing/Data_combine_FY_SW.py


Related Downstream Results
--------------------------
The downstream COD folder referencing these files is:

    FY4A_validation/Cloudy_results/Cloudy_COD_HG_QC_GHI_20260722_140524
