# Urban Heat Island (UHI) Analyzer – Landsat & MODIS

## Overview
This repository provides a **public outline version** of an interactive **Urban Heat Island (UHI) analysis application** developed using **Google Earth Engine (GEE)** and **Streamlit**.

The project is part of an **ongoing research study** focused on long-term UHI dynamics across U.S. counties and urban areas using **Landsat** and **MODIS** satellite data, combined with **NLCD land-cover classification** and **statistical trend analysis**.

---

##  Important Notice 
**This repository does NOT run end-to-end by design.**

Core computational pipelines, parameter tuning, and finalized analysis logic have been **intentionally removed or disabled** in this public version.

### Reason:
- The full implementation supports an **in-progress peer-reviewed manuscript**
- Releasing executable analysis code prior to publication would compromise research novelty
- This repository is shared **solely for academic review and transparency**

**The complete, fully reproducible codebase will be made public after paper acceptance**, along with a DOI-based citation.

---

## Scientific Methods (Implemented in Full Version)
The complete version of this application includes:

- **Satellite Data**
  - Landsat 5 / 8 / 9 (Collection 2, Level-2, 30 m)
  - MODIS Terra + Aqua (Day & Night LST, 1 km)
- **Processing Workflow**
  - Annual hottest-month selection
  - Cloud & QA masking
  - Percentile-based LST outlier filtering (5–95%)
- **Urban Heat Island Metrics**
  - NLCD-based urban vs vegetative masking
  - UHI intensity = Urban − Vegetative LST
  - Mean LST, NDVI, NDMI, NDBI
- **Trend Analysis**
  - Sen’s slope (Theil–Sen estimator)
  - Mann–Kendall significance testing
- **Validation**
  - ERA5-Land monthly 2 m air temperature
  - Anomaly-based Pearson correlation
- **Outputs**
  - Interactive maps
  - Time-series plots
  - CSV and Shapefile exports
  - Google Drive raster exports

---

##  Technology Stack
- **Python**
- **Google Earth Engine**
- **Streamlit**
- GeoPandas
- NumPy / Pandas
- Matplotlib / SciPy
- geemap, folium, shapely

---


