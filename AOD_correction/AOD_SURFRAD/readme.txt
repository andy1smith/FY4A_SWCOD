THE SURFRAD AEROSOL OPTICAL DEPTH DATA SET

Aerosol optical depth (AOD) data for SURFRAD stations are generated from visible
 Multi-Filter Rotating Shadowband Radiometers (MFRSR).


CALIBRATION METHOD FOR THE SURFRAD MFRSRs

Calibration of the MFRSR channels for AOD is obtained from 0-airmass linear 
extrapolations of Langley plots (the natural log of the output voltage versus 
air mass) that are derived from cloud-free MFRSR measurements. Because the 
total optical depth for a particular spectral channel is the slope of the 
Langley plot, absolute calibration of the MFRSR channels is unnecessary.  
To automatically identity pristine measurements for calibration Langley plots, 
the SURFRAD algorithm cross references times identified as cloud free by the 
Long Ackerman clear-sky identification method with the times of MFRSR 
measurements.  The times of clear sky are determined from independent 
collocated SURFRAD broadband solar measurements.  Such plots represent first-
attempt Langley calibrations that are generally free of cloud contamination.  
The few bad calibration Langley points that survive the first attempt are 
rejected by deviation-from-linearity tests that follow.  All calibration 
Langley plots are checked before being included. 

Time series of Langley calibrations and associated errors are constructed for 
the duration of a particular MFRSR's deployment.  To get a "representative 
Langley" calibration for each channel for a designated period, about two months 
of data are processed as described above.  Channel specific Langley 
calibrations within that period are normalized to a circular orbit and then 
averaged to derive a "representative" calibration and error for that two-month 
period.  Error is computed by a common propagation of error method.  Outliers 
from the sample are rejected by standard statistical methods and the average 
and error is recalculated.  After about two years of these "representative" 
~2-month channel mean calibrations and associated error are ascertained, 
the time series of the representative calibrations for each channel is fit to 
a polynomial with linear, sine and cosine terms that describes the mean 
variation of a channel-specific calibration (normalized to a circular orbit) 
over the two-year period.  The error time series for each channel are also fit 
to best-fit linear regression expressions that represent the variation of error 
for each channel over the 2-year period.  Prior to the regression, the 
two-year time series are overlapped by about four months on either end of the 
2-year period to ensure a smooth transition in the fits to the spectral 
calibrations and error time series between adjacent two-year periods.  These 
calibration equations are loaded into files that are accessed by the algorithm 
that is used to compute aerosol optical depth.  

AEROSOL OPTICAL DEPTH COMPUTATION

The total optical depth for each spectral MFRSR measurement is computed by 
first obtaining the calibration for the day of the measurement from the linear 
equations that have been fit to the time series' of channel-specific 0-air mass 
Langley calibrations.  That calibration is then corrected back to an 
elliptical orbit value for the day being processed.  For each 2-minute MFRSR 
measurement, the appropriate channel calibration and measurement are plotted 
as a two-point Langley plot.  The slope of that line is the total optical 
depth for that channel at that time.  Aerosol optical depth is computed by 
subtracting the contributions of molecular scattering and absorption by ozone 
from the total optical depth.  Ozone absorption coefficients are chosen based 
on the particular central measurement wavelengths of each MFRSR.  Molecular 
scattering is also computed based the central wavelengths.  Potential 
contributions by nitrogen dioxide and sulfur dioxide absorption are ignored 
because they are negligible (< 1%).  

DATA DISTRIBUTION 

The SURFRAD AOD analysis program operates on one day and one station, and 
produces a one-day AOD product file for that station.  The AOD files are named 
with the convention [sta]_yyyymmdd.aod, where [sta] is the three-letter 
station identifier:

bon	Bondville, Illinois
fpk	Fort Peck, Montana
gwn	Goodwin Creek, Mississippi
tbl	Table Mountain, Colorado
dra	Desert Rock, Nevada
psu	Penn State, Pennsylvania
sxf	Sioux Falls, South Dakota

For example, the AOD file for Table Mountain for April 13, 2001 would be named: 
tbl_20010413.aod.  The data frequency in these files matches that of the 
MFRSR, which is typically two-minute. 
 
SURFRAD AOD files are distributed from the ftp site:

ftp://www.srrb.noaa.gov/pub/data/surfrad/aod/[sta]/[yyyy]/

where [sta] refers to the station 3-letter identifier (e,g, dra for Desert 
Rock) and [yyyy] is the 4-digit year

Each SURFRAD AOD product file has six header records that are followed by 
data.  An example of the header of such a file is shown below:


Table Mountain SURFRAD aerosol optical depth (nm)
13-apr-2001 13 04 2001 349 lines of data
413.5 497.4 615.0 672.7 869.8  channel central wavelengths
 0.221  0.211  0.196  0.194  0.185 Daily average AODs (sample size = 236)
352 Dobson units of ozone
ltime 0=good AOD414 AOD497 AOD615 AOD673 AOD870 414E 497E 615E 673E 870E p_mb ang_exp
(first line of data)
.
.
.
.
.
(last line of data)


The first header record contains the station name,  

The second header record has the date in two forms followed by the number of 
lines of AOD data in the file.  The first date is of the form: dd-mmm-yyyy, 
which is the same as that in the filename.  To the right on that same line, 
the date is repeated in numeric format, i.e., day of the month (dd), month 
number (mm), and the 4-digit year (yyyy).  The last numeric entry on that line 
is the number of lines of AOD data in the file.

The third header record contains the central wavelengths (nm) of the first 
five spectral channels of the MFRSR that was used to generate the AOD data in 
the file.

The fourth header record lists the daily average AODs for each spectral 
channel, in the order that they are listed on the previous line.  Only AODs 
marked as "good," i.e., a 0 in the second column of the data block, are 
included in the daily average computation.  At the end of the line, the sample 
size of data points that was used in the average calculation for each channel 
is listed. 
 
The fifth header record lists the total column ozone (in Dobson units) that 
was derived from TOMS data for the location of the station on the day that the 
file represents.  That value of total ozone was used to compute the channel-
specific optical depth due to ozone that was removed from the total optical 
depth (not listed in the file) computed for each channel. 

The sixth header line lists data column headers. 

The data block consists of 13 columns. 

The first column is the time in local standard time (LST) in hours and minutes 
(hhmm).

The second column is the cloud-screening indicator; either a 0 or 1 is entered. 
A "0" in column 2 indicates that that time period passed a cloud screen test 
and the AOD data listed on the line likely represent true aerosol optical 
depths.  A "1"  in column 2 indicates that the measurements on that line may be 
adversely affected by clouds and the AODs listed should be ignored. All data 
are included because the cloud-screening test is not perfect.  The convention 
of using "0" as a good-data indicator and "1" for bad data is consistent with 
that used in SURFRAD data files.  

Columns 3 through 7 are aerosol optical depths (AODs) for the first five 
spectral channels of the MFRSR.  They are nominally 415 nm, 500 nm, 614 nm, 
670 nm, and 868 nm, but in practice they vary among visible MFRSRs.   The 
unique values for the MFRSR used to compute AOD for the current file are 
listed in header record 3.  Missing values for AOD are indicated by -9.999. 

Columns 8 through 12 are errors associated with the AODs in the columns 3 
through 7, respectively.  Missing errors are indicated by -9.9999. 

The 13th and last column lists the atmospheric station pressure in millibars 
measured at the SURFRAD station for the time of the measurement.  These values 
were used to compute the optical depth due to molecular (Rayleigh) scattering 
that was subtracted from the total optical depth (not listed in the file) for 
each channel. The last column contains the Angsrtom exponent which is 
computed from the AODs for the 500nm and 868nm channels.   

For details on the SURFRAD aerosol optical depth algorithm, refer to the i
following publications:

Augustine, J. A., C. R. Cornwall, G. B. Hodges, C. N. Long, C. I. Medina, and 
J. J. DeLuisi, 2003: An automated method of MFRSR calibration for aerosol 
optical depth analysis with application to an Asian dust outbreak over the 
United States, J. Appl. Meteor., 42, 266-278.

Augustine, J. A., G. B. Hodges, C. R. Cornwall, J. J. Michalsky, and C. I. 
Medina, 2005: An update on SURFRAD-The GCOS surface radiation budget network 
for the continental United States, J. Atmos. And Oceanic Tech., 22, 1460-1472.
