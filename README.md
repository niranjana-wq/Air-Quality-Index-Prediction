# Exploratory Data Analysis and Next-Day Air Quality Prediction for Indian Cities

**Course:** U21ADP05 — Exploratory Data Analysis and Visualization (Assignment II, 40 marks)  
**Submission Date:** 15 / 10 / 2026

---

## Problem Statement

Urban air pollution in India has reached crisis levels in several cities, with
AQI frequently breaching 'Very Poor' and 'Severe' thresholds. Accurate
**next-day AQI forecasting** can enable timely public-health advisories,
targeted industrial shutdowns, and better personal decisions (outdoor activity,
mask usage). This project builds a reproducible pipeline from raw CPCB data
to a deep-learning prediction model.

---

## Dataset

| Detail | Value |
|--------|-------|
| Source | [Kaggle — rohanrao/air-quality-data-in-india](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) |
| Original provider | Central Pollution Control Board (CPCB), India |
| File used | `city_day.csv` |
| Shape | 29 531 rows × 16 columns |
| Date range | 2015-01-01 → 2020-07-01 |
| Cities | 26 Indian cities |
| Target | Next-day AQI (regression) and AQI_Bucket (6-class classification) |

> ⚠️ The raw CSV is **not committed** to this repo. Download it from Kaggle and place it in `data/city_day.csv`.

---

## Key Results

| Model | MAE | RMSE | R² | Accuracy | Macro F1 | AUC |
|-------|-----|------|-----|----------|----------|-----|
| Persistence | 32.58 | 50.92 | 0.638 | — | — | — |
| HistGBM | **32.25** | **45.81** | **0.708** | 0.566 | 0.454 | — |
| MLP (Deep Learning) | 32.90 | 48.83 | 0.668 | 0.550 | **0.474** | **0.850** |

> All metrics on the **held-out chronological test set** (final 15% of dates). HistGBM wins on regression; MLP wins on macro-F1 and AUC due to class-weighted training.

---

## Setup and Run

```bash
# 1. Clone / open the workspace
cd AQI/

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies (pinned versions)
pip install -r requirements.txt

# 4. Place data file
#    Download city_day.csv from Kaggle → data/city_day.csv

# 5. Run the full analysis script
python src/aqi_eda_model.py

# 6. Generate the Jupyter notebook
python make_notebook.py

# 7. Generate the Word report (requires results.json from step 5)
python generate_report.py
```

To run the notebook interactively:
```bash
jupyter notebook notebooks/aqi_eda_model.ipynb
```

---

## Folder Structure

```
AQI/
├── .gitignore
├── PLAN.md                         # project plan (approved before coding)
├── PROGRESS.md                     # phase-by-phase status log
├── README.md                       # this file
├── requirements.txt                # pinned Python dependencies
├── results.json                    # computed metrics (auto-generated)
├── make_notebook.py                # converts script → .ipynb
├── generate_report.py              # generates DOCX report
├── data/
│   ├── README.md                   # dataset provenance and column schema
│   └── city_day.csv                # NOT committed — download from Kaggle
├── notebooks/
│   └── aqi_eda_model.ipynb         # annotated Jupyter notebook
├── src/
│   └── aqi_eda_model.py            # complete analysis + model script
├── figures/
│   ├── 01_missingness_heatmap.png
│   ├── 02_boxplots_before_winsorize.png
│   ├── 03_boxplots_after_winsorize.png
│   ├── 04_histograms_kde.png
│   ├── 05_log_transform_pm.png
│   ├── 06_pm25_qq.png
│   ├── 07_pm10_qq.png
│   ├── 08_aqi_bucket_distribution.png
│   ├── 09_poisson_poor_days.png
│   ├── 10_pearson_correlation.png
│   ├── 11_spearman_correlation.png
│   ├── 12_scatterplots_regression.png
│   ├── 13_aqi_by_city_boxplot.png
│   ├── 14_aqi_by_season_boxplot.png
│   ├── 15_aqi_by_weekday_boxplot.png
│   ├── 16_pairplot.png
│   ├── 17_pca_scree_biplot.png
│   ├── 18_clustermap.png
│   ├── 19_monthly_aqi_trend.png
│   ├── 20_yearly_aqi_trend.png
│   ├── 21_seasonal_decompose.png
│   ├── 22_pm25_city_month_heatmap.png
│   ├── 23_permutation_importance.png
│   ├── 24_training_curves.png
│   ├── 25_confusion_matrix.png
│   ├── 26_roc_curves.png
│   ├── 27_residual_analysis.png
│   ├── 28_predicted_vs_actual.png
│   ├── 29_timeseries_overlay.png
│   ├── 30_model_comparison.png
│   ├── interactive_aqi_city_dropdown.html
│   └── interactive_aqi_city_month_heatmap.html
└── report/
    └── Project_Report.docx
```

---

## Limitations

- No meteorological features (wind, temperature, humidity) — the strongest AQI predictors.
- Some cities have very sparse data before 2019; lag/rolling windows produce many NaN values there.
- The Severe AQI class (~5% of data) remains difficult to classify despite class weighting.

---

## Reproducibility

- **Seed:** 42 (numpy, random, TensorFlow)
- **Split:** chronological 70/15/15 — no random shuffling
- **Scalers/imputers:** fit on train split only
- All figures generated at 200 dpi and saved to `figures/`
