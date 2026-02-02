# Mangrove Carbon Stock Estimation using Sentinel-2

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Remote Sensing](https://img.shields.io/badge/Remote%20Sensing-Sentinel--2-green)](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
[![Status](https://img.shields.io/badge/Status-Research%20Portfolio-orange)]()

## 📌 Research Background

Mangrove ecosystems play a vital ecological role as massive carbon sinks, often referred to as **Blue Carbon**. Their capacity to sequester carbon can exceed that of terrestrial tropical forests. However, these ecosystems are vulnerable to degradation due to natural factors and anthropogenic activities.

Traditional methods for measuring biomass and carbon stock (such as destructive field surveys) are limited in spatial coverage and time-consuming. Therefore, a **Remote Sensing approach** using optical satellite imagery offers an efficient solution for monitoring mangrove biomass on a large scale.

This project utilizes **Sentinel-2** data due to its high spatial and spectral resolution, making it highly effective for detecting green vegetation and performing temporal analysis of mangrove carbon stocks.

---

## ⚙️ Methodology

This repository implements a technical workflow based on scientific methodologies for carbon estimation. The process is divided into four main stages:

### 1. Data Acquisition & Pre-processing

The script processes **Sentinel-2 Level-2A** imagery.

- **Radiometric Correction:** Raw Sentinel-2 data (Integer) is divided by a scaling factor to convert it into **Surface Reflectance** values (range 0.0 - 1.0).
- This step is crucial to ensuring accurate spectral index calculations.

### 2. NDVI Analysis

The **Normalized Difference Vegetation Index (NDVI)** is used as the primary predictor for mangrove density and biomass. It is calculated using the Near-Infrared (NIR) and Red bands:

$$NDVI = \frac{NIR - Red}{NIR + Red}$$

- **Thresholding:** A threshold of `NDVI > 0.4` is applied to mask out water bodies and non-mangrove areas, isolating dense vegetation for analysis.

### 3. Above Ground Biomass (AGB) Modeling

Biomass estimation is performed using an empirical allometric model. Based on the correlation between vegetation indices and field biomass data, this project implements a **Power Law (Non-linear)** regression model:

$$AGB = 150 \times (NDVI^{2.5})$$

- _Note: This formula represents a specific allometric relationship. Other studies (e.g., in Menjangan Island) may use linear models such as $y = 1578.2x - 269.79$, depending on local field calibration._

### 4. Carbon Stock Estimation

To determine the final Carbon Stock, the estimated Above Ground Biomass (AGB) is converted using the standard carbon fraction factor recommended by the IPCC (approx. 47%):

$$Carbon\ Stock = AGB \times 0.47$$

---

## 📊 Visualization Results

_(Upload your Dashboard image to the repo and reference it here)_
![Dashboard Analysis]("D:\magang\portofolio\carbon\carbon 2019-2024\Dashboard_Mangrove_v2.png")

The output includes:

1.  **NDVI Maps:** Visualizing vegetation density.
2.  **Carbon Stock Maps:** Spatial distribution of carbon (Ton C/Ha).
3.  **Time-Series Animation:** Monitoring changes from 2019 to 2024.

## ⚙️ Configuration & Setup

Since the satellite imagery files (GeoTIFF) are too large to be hosted on GitHub, you need to use your own data or download Sentinel-2 imagery separately.

1. **Prepare Data:**
   - Download Sentinel-2 Level 2A imagery (or use your own dataset).
   - Ensure you have bands for Red, Green, Blue, and NIR.

2. **Update Paths:**
   - Open `mangrove_analysis.py` in your code editor.
   - Locate the `CONFIG` section at the top of the script.
   - Replace the placeholder paths with the actual location of your files:
     ```python
     CONFIG = {
         'input_files': {
             '2019': r"C:/Users/YourName/GIS_Data/Sentinel_2019.tif",
             ...
         },
         'output_dir': r"C:/Users/YourName/Projects/Mangrove_Output"
     }
     ```

## 📚 References

This methodology is aligned with standard remote sensing practices for mangrove monitoring, utilizing:

- **ESA Sentinel-2** for high-resolution optical data.
- **IPCC Guidelines** for carbon fraction conversion.
- **Vegetation Indices (NDVI)** as a proxy for biomass modeling.
