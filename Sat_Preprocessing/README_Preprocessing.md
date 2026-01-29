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

**Script Name:** surfrad\_download\_process.py

This script manages the ground-truth data from SURFRAD stations.

* **Function 1: Download and Process**  
  * Downloads raw SURFRAD data.  
  * Processes and aggregates the data into a single, consolidated CSV file for easy matching.

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

* **Manual Download Required:** The **MODIS MCD43A1** product cannot be downloaded automatically by these scripts. It requires a handy (manual) download at [here](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/science-domain/brdf-albedo-and-nbar/) prior to running Step 3\. Ensure these files are placed in the correct directory before execution.



## Flow chart



```mermaid
graph TD
    Start([Start: Data Processing Pipeline]) --> Step1["<b>STEP 1: Satellite Data Acquisition</b><br/>Sat_download_extract.py"]
    Start --> Step2["<b>STEP 2: Ground Data Processing</b><br/>surfrad_download_process.py"]
    Start --> Step3Manual["<b>⚠️ Manual Step</b><br/>Download MODIS MCD43A1<br/>Place in correct directory"]

Step1 --> S1F1["Download GOES-16 Data"]
S1F1 --> S1F2["Extract Region of Interest"]
S1F2 --> S1Filter{{"Apply Filters:<br/>Cloud Mask +<br/>Phase Filter"}}

S1F2 -->|Clear Days| ClearOutput["✓ Clear Sky Dataset"]
S1Filter -->|Cloudy Days| CloudyOutput["✓ Cloudy Sky Dataset"]

Step2 --> S2F1["Download SURFRAD Data"]
S2F1 --> S2F2["Process & Aggregate Data"]
S2F2 --> S2Output["✓ SURFRAD CSV"]

Step3Manual --> ManualDone["✓ MODIS Albedo Files<br/>Ready"]

ClearOutput --> Step3["<b>STEP 3: Data Combination<br/>Sat_surfrad_combine.py"]
CloudyOutput --> Step3
S2Output --> Step3
ManualDone --> Step3

Step3 --> S3F1["Read Satellite Data<br/>Clear & Cloudy Datasets"]
S3F1 --> S3F2["SURFRAD Temporal & Spatial<br/>Matching"]
S3F2 --> S3F3["MODIS MCD43A1<br/>Albedo Product Matching"]
S3F3 --> S3F4["Calculate Albedo Parameters"]

S3F4 --> FinalOutput["✓ Final Combined Dataset<br/>with Albedo Parameters"]
FinalOutput --> End([End: Processing Complete])

style Start fill:#e1f5ff
style End fill:#e1f5ff
style Step1 fill:#bbdefb
style Step2 fill:#c8e6c9
style Step3 fill:#ffe0b2
style Step3Manual fill:#ffccbc
style ClearOutput fill:#a5d6a7
style CloudyOutput fill:#a5d6a7
style S2Output fill:#81c784
style FinalOutput fill:#ffb74d
```
