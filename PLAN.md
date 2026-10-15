# PLAN.md — AQI EDA and Next-Day Prediction

**Course:** U21ADP05 — Exploratory Data Analysis and Visualization  
**Assignment:** II (40 marks) | **Due:** 15/10/2026  
**Written:** 2026-10-14

---

## 1. Data Issues Found (Phase 0 inspection)

| Issue | Detail | Planned Handling |
|-------|--------|-----------------|
| AQI_Bucket stored as object (string) | 6 categories + NaN. Non-null: 24 850 / 29 531 (84 %) | Treat as string; impute during row-drop step |
| AQI missing (same proportion) | AQI is derived from pollutants; many early rows lack it | Drop rows with AQI NaN for the model target; keep for EDA where feasible |
| Pollutant missingness varies | PM2.5/PM10 have the most missing; Xylene very sparse | Compute per-column + per-city %; set threshold at 60 % missing-per-city to drop a column for that city; linear interpolate gaps ≤ 3 consecutive days within city |
| Date range varies by city | Some cities only appear from 2019 onwards | Document per city; do NOT fill cross-city gaps |
| Outliers / extreme pollution events | PM2.5 can legitimately reach 900+ during severe smog events | Winsorize at 99.5th percentile per pollutant (not delete); log-transform for distribution fitting |
| No duplicate dates found yet | Will verify in Phase 2 | Drop exact duplicates |

---

## 2. Modeling Approach

### 2a. Target Definition
- **Regression:** next-day AQI (continuous float)  
- **Classification:** next-day AQI_Bucket (6-class ordinal; mapped to int 0-5: Good=0, Satisfactory=1, Moderate=2, Poor=3, Very Poor=4, Severe=5)

### 2b. Leakage Prevention
- All features for predicting day t+1 use only data from day t and earlier.
- Lag features: shift pollutant and AQI values by at least 1 day relative to the target.
- Rolling statistics (7-day window): computed with shift(1) before rolling, so the window ends at day t.
- Scalers and imputers fit on train split only, then applied to val and test.

### 2c. Chronological Split (seed 42 for reproducibility)
- Sort by (City, Date) globally; then split by date (not city):
  - Train: earliest 70% of dates  
  - Val: next 15%  
  - Test: final 15%
- Date boundaries will be printed in the notebook.

---

## 3. Feature List (per city-day row predicting next day)

| Feature Group | Features | Count |
|--------------|----------|-------|
| Lag-1 pollutants | PM2.5_lag1 … Xylene_lag1 | 12 |
| Lag-2 pollutants | PM2.5_lag2 … Xylene_lag2 | 12 |
| Lag-3 pollutants | PM2.5_lag3 … Xylene_lag3 | 12 |
| Lag-1 AQI + bucket | AQI_lag1, AQI_Bucket_int_lag1 | 2 |
| Rolling 7-day mean | PM2.5, PM10, NO2, NOx, CO, SO2, O3, AQI | 8 |
| Rolling 7-day std | same 8 with _roll7_std | 8 |
| Calendar | month, weekday, is_weekend, season_int | 4 |
| Cyclical encodings | month_sin, month_cos, weekday_sin, weekday_cos | 4 |
| City (one-hot) | ~26 cities | ~26 |
| Total | | ~88 |

---

## 4. Model Architecture

### Baselines
1. Persistence: y_hat(t+1) = AQI(t)
2. Gradient Boosting (HistGBM): sklearn HistGradientBoostingRegressor/Classifier, 200 estimators, max_depth 5

### Deep Learning MLP Regressor
- Input -> Dense(256, relu) -> BatchNorm -> Dropout(0.3)
- -> Dense(128, relu) -> BatchNorm -> Dropout(0.2)
- -> Dense(64, relu)
- -> Dense(1, linear)
- Loss: Huber | Optimizer: Adam lr=1e-3
- Callbacks: EarlyStopping(patience=15), ReduceLROnPlateau(patience=7)

### Deep Learning MLP Classifier
- Same hidden layers
- -> Dense(6, softmax)
- Loss: sparse_categorical_crossentropy
- Class weights computed from training set (Severe ~5x weight)

### Optional LSTM (time permitting)
- 7-day lookback window per city
- 1 LSTM(64) -> Dense(32) -> output

---

## 5. Hyperparameter Search (validation set only)

Manual grid:
- Hidden units: [128+64, 256+128, 256+128+64]
- Dropout: [0.2, 0.3, 0.4]
- Learning rate: [1e-3, 5e-4]

All trials logged in notebook table. Test set touched ONCE at the end.

---

## 6. File Layout

```
AQI/
├── .gitignore
├── PLAN.md
├── PROGRESS.md
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── city_day.csv  (local only, not committed)
├── notebooks/
│   └── aqi_eda_model.ipynb
├── src/
│   └── aqi_eda_model.py
├── figures/
│   ├── *.png (200 dpi)
│   └── *.html (Plotly)
└── report/
    └── Project_Report.docx
```

---

## 7. Assumptions

1. Predict AQI for the next calendar day per city, not national aggregate.
2. Chronological split applied by global date; all cities share same train/val/test boundary dates.
3. Xylene included if it passes the per-column threshold, otherwise dropped.
4. AQI_Bucket ordinal: Good=0, Satisfactory=1, Moderate=2, Poor=3, Very Poor=4, Severe=5.
5. Indian seasons: Winter(Dec-Feb)=0, Pre-Monsoon(Mar-May)=1, Monsoon(Jun-Sep)=2, Post-Monsoon(Oct-Nov)=3.
6. Winsorization applied before feature engineering; log-transform used only for distribution fitting.

---

*Awaiting approval before writing analysis code.*
