# Rwanda Agricultural Datasets

This directory contains the core datasets used to adapt the AGRISENSE model to Rwandan agricultural conditions. These datasets override or supplement the generic global machine learning models to ensure recommendations are agronomically sound and economically viable for local farmers.

## Datasets

### 1. `crop_recommendation_factors.csv`
**Purpose:** The central source of truth for the multi-factor crop ranker.
**Source:** Compiled from Rwanda Agriculture Board (RAB) guidelines, Crop Intensification Programme (CIP) manuals, and the National Institute of Statistics of Rwanda (NISR) Seasonal Agricultural Survey (SAS).
**Contents:**
- `crop_id` & `family`: Crop identification and botanical family (used for rotation logic).
- `seasons`: Allowed planting seasons (A, B, C) per MINAGRI definitions.
- `soils`: Suitable soil textures (alluvial, clayey, loamy, sandy).
- `temp_*`, `rain_*`, `humidity_*`, `ph_*`, `n_opt`, `p_opt`, `k_opt`: Agronomic optimum ranges.
- `typical_yield_t_ha`: National average yield (from NISR SAS).
- `farmgate_rwf_kg`: Indicative national farmgate price (from FAOSTAT/MINAGRI).
- `provinces`: Provinces where the crop is typically grown.

### 2. `district_market_prices.csv`
**Purpose:** Provides localized, live-updating farmgate prices to improve the accuracy of the income-maximization ranking.
**Source:** Mocked data representing feeds from E-Soko / MINICOM.
**Contents:**
- `district`: The district or province name.
- `crop_id`: The canonical crop identifier.
- `price_rwf_kg`: The local farmgate price in Rwandan Francs.
- `date_updated`: The date the price was recorded.

### 3. `plot_harvest_calibration.csv` (Seed File)
**Purpose:** The critical ground-truth dataset required to calibrate the generic ML yield model to actual Rwandan fields.
**Source:** Currently a seed file. Must be populated by field officers and cooperatives.
**Contents:**
- `plot_id`, `date`, `season`, `district`: Metadata.
- `crop_id`, `variety`: What was planted.
- `soil_texture`, `lab_ph`, `nitrogen`, `phosphorus`, `potassium`, `ec_us_cm`, `moisture_vwc`: Soil conditions (ideally from the RS485 probe + lab pH).
- `yield_t_ha`: **The actual measured harvest.**
- `fertilizer_applied`: What the farmer actually added.
*Note: The AI yield model cannot be scientifically validated for Rwanda until this dataset contains 300+ rows.*

### 4. `nisr_sas_major_crops.csv`
**Purpose:** Reference table of national production statistics.
**Source:** NISR Seasonal Agricultural Survey (SAS) 2023/2024B.
**Contents:** Annual production in tonnes and cultivated area in hectares for major staples (Maize, Beans, Irish Potato, Cassava, Banana).

## How the Ranker Uses This Data
The `app/services/crop_ranker.py` service combines the generic ML yield prediction (for the 10 global crops) with the agronomic ranges, seasons, and rotation rules in `crop_recommendation_factors.csv`. It then calculates expected income using the local prices in `district_market_prices.csv` (falling back to the national averages if a district isn't found). This ensures that Rwandan staples like Beans and Cassava can outrank global crops if they are more profitable or better suited to the current season and soil.