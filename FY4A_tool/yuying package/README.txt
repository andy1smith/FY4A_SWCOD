1. use 'goes_download.py' for GOES products download:
(1) change output directory, please ensure enough storage space before downloading.

(2) change site select central pixel, usually crop images corresponding all sites for efficiency, cause they are cropped from the same image. For preliminary experiments, maybe just 'BON' station is fine.

(3) change set crop size, recommand 11*11 to save storage space;

(4)  select year of data, recommand year 2019 since we already collect all SURFRAD data, and corresponding ABI Level-1 radiance products. (we should discuss how to process L1 procuts for SW, not sure whether the same as LW, since all data is saved in my local computer)

(5) Noted that for GOES products for SURFRAD, suggest download  products ended with "C", which covers Continental U.S. (CONUS) instead of full disk. 
Same related products for our study, and more products are shown in "https://noaa-goes16.s3.amazonaws.com/index.html".
"ABI-L1b-RadC": ABI Level-1 radiance products for CONUS. (already downloaded and processed !)
"ABI-L2-ACHAC": ABI Level-2 cloud top height products for Continental U.S. (CONUS).
"ABI-L2-CODC": ABI Level-2 cloud optical depth products for Continental U.S. (CONUS).


2. use "data_process.py" for data process. Note that this code is for GOES-16 L1 data process, but the general process procedure should be the same.
(1) change "Line 26" to output directory of downloaded data;
(2) change "Line 59" to select output directory;
(2) change "Line 79" to change channels to name of L2 cloud products.

3. please refer to "prepare_data.py" for more data process.





