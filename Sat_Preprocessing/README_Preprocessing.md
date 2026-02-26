# **Satellite & Ground Data Processing Pipeline**

Nan DENG dengnan987@gmail.com

This document outlines the three-step workflow for downloading, processing, and combining GOES-16 satellite data, SURFRAD ground observations, and MODIS albedo products.

## **Pipeline Overview**

### Step 1: Satellite Data Acquisition & Extraction

**Script Name:** Sat\_download\_extract.py

This script handles the initial retrieval of satellite imagery and separates useful data based on atmospheric conditions.

* **Function 1: Download GOES-16**  
  * Retrieves GOES-16 data from the source.  
* **Function 2: Extract Region**  
  * Clips data to the specific region of interest.  
  * **Filtering:** Applies filters to categorize data into:  
    * Clear days  
    * Cloudy days (utilizing Cloud Mask \+ Phase Filter).

---

### Step 2: Ground Data Processing

**Script Name:** 

This script manages the ground-truth data from SURFRAD stations.

* **Function 1: Download and Process**  
  * Downloads raw 
  * Ground measurement : China_SURF_Station.xlsx 
    * Map CERN with three nerest neighbours and interpolation the ground date into its location.
  
  <img src="./Ground/CERN_site_Map.png" alt="CERN_Delaunay_Map" style="zoom:50%;" />

---

### Step 3: Data Combination & Albedo Calculation

**Script Name:** Sat\_surfrad\_combine.py

This final script merges the satellite and ground data and incorporates surface albedo parameters.

* **Function 1: Read Satellite Data**  
  * Ingests the processed GOES-16 radiance data from Step 1\.  
* **Function 2: Match with Ground Data**  
  * Aligns satellite observations with SURFRAD ground data (temporal and spatial matching).  
* **Function 3: Load & Calculate Albedo**  
  * Loads the **MODIS MCD43A1** albedo product.  
  * Calculates the following albedo parameters:  
    * Black-sky albedo  
    * White-sky albedo  
    * Blue-sky albedo

**⚠️ Important Notes**

* **Manual Download Required:** The **MODIS MCD43A1** product cannot be downloaded automatically by these scripts. It requires a handy (manual) download at [here](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd43a1-061) in **APPEEARS->extract->point** to running Step 3\.prior to running Step 3\. Ensure these files are placed in the correct directory before execution.



## Flow chart



```mermaid
graph TD
    Start([Start: Data Processing Pipeline]) --> Data1["<b>CERN GHI Data</b>"]
    Start --> Data2["<b>FY4A full disk</b>"]
    Start --> Data3["<b>MODIS MCD43A1</b>"]
    Start --> Data4["<b>NoAA_NCEI China Metero Station 3-hour meansurement</b>"]

Data1 --> S1F1["Clear, cloudy day filter<br/>FYSat_remap_and_ground_preprocess.py"]
S1F1 --> S1F2["Extract Region of Interest"]
S1F2 --> S1Filter{{"Apply Filters:<br/>Cloud Mask +<br/>Phase Filter"}}

S1F2 -->|Clear Days| ClearOutput["✓ Clear Sky Dataset"]
S1Filter -->|Cloudy Days| CloudyOutput["✓ Cloudy Sky Dataset"]

Data2 --> S2F1["CERN GHI Data"]
S2F1 --> S2F2["Process & clear filter Data"]
S2F2 --> S2Output["✓ SURFRAD CSV"]

Data3 --> ManualDone["✓ MODIS Albedo Files<br/>Ready"]

ClearOutput --> Step3["<b>STEP 3: Data Combination<br/>Sat_surfrad_combine.py"]
CloudyOutput --> Step3
S2Output --> Step3
ManualDone --> Step3

Data3 --> S3F1["Read Satellite Data<br/>Clear & Cloudy Datasets"]
S3F1 --> S3F2["SURFRAD Temporal & Spatial<br/>Matching"]
S3F2 --> S3F3["MODIS MCD43A1<br/>Albedo Product Matching"]
S3F3 --> S3F4["Calculate Albedo Parameters"]

S3F4 --> FinalOutput["✓ Final Combined Dataset<br/>with Albedo Parameters"]
FinalOutput --> End([End: Processing Complete])

style Start fill:#e1f5ff
style End fill:#e1f5ff
style Data1 fill:#bbdefb
style Data2 fill:#bbdefb
style Data3 fill:#bbdefb
style Data4 fill:#bbdefb
style S3F4 fill:#c8e6c9
style S1F2 fill:#ffe0b2
style S1F1 fill:#ffccbc
style ClearOutput fill:#a5d6a7
style CloudyOutput fill:#a5d6a7
style S2Output fill:#81c784
style FinalOutput fill:#ffb74d
```
