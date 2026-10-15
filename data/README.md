# Data Directory

## Dataset
**Title:** Air Quality Data in India  
**Source:** [Kaggle — rohanrao/air-quality-data-in-india](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  
**Original Source:** Central Pollution Control Board (CPCB), Government of India  
**License:** CC0: Public Domain

## Files Used
| File | Description |
|------|-------------|
| `city_day.csv` | Daily city-level AQI and pollutant concentrations (2015–2020) |

## Important
The raw CSV is **not committed** to this repository (see `.gitignore`).  
Download it from the Kaggle link above and place `city_day.csv` in this `data/` folder before running the notebook.

## Columns
| Column | Unit | Description |
|--------|------|-------------|
| City | — | Indian city name |
| Date | YYYY-MM-DD | Measurement date |
| PM2.5 | µg/m³ | Particulate matter < 2.5 µm |
| PM10 | µg/m³ | Particulate matter < 10 µm |
| NO | µg/m³ | Nitric Oxide |
| NO2 | µg/m³ | Nitrogen Dioxide |
| NOx | ppb | Oxides of Nitrogen |
| NH3 | µg/m³ | Ammonia |
| CO | mg/m³ | Carbon Monoxide |
| SO2 | µg/m³ | Sulphur Dioxide |
| O3 | µg/m³ | Ozone |
| Benzene | µg/m³ | Benzene |
| Toluene | µg/m³ | Toluene |
| Xylene | µg/m³ | Xylene |
| AQI | — | Air Quality Index (composite) |
| AQI_Bucket | — | Categorical label (Good/Satisfactory/Moderate/Poor/Very Poor/Severe) |
