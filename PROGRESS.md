# PROGRESS

## Phase 0 - Setup [DONE] (2026-10-14)
- Verified `data/city_day.csv` (29,531 x 16, schema correct).
- Created `.venv`, installed all dependencies (TF 2.16.1, pandas 2.2.2, etc.).
- Directory structure: `notebooks/`, `src/`, `figures/`, `report/`.
- Files: `requirements.txt`, `.gitignore`, `data/README.md`.

## Phase 1 - Plan [DONE] (2026-10-14)
- `PLAN.md` written and approved.
- Data issues documented; modeling approach, feature list, split strategy confirmed.

## Phase 2 - EDA [DONE] (2026-10-14)
- All 7 Ben Fry stages implemented in `src/aqi_eda_model.py`.
- 22 static figures saved to `figures/` at 200 dpi.
- 2 interactive Plotly HTMLs saved.
- Key findings:
  - Kruskal-Wallis H across seasons: significant (p << 0.05)
  - COVID lockdown reduced mean AQI significantly
  - PM2.5 best fit: Log-Normal (lower AIC than Gamma)
  - Poisson dispersion index >> 1 (overdispersion present)

## Phase 3 - Feature Engineering & Modeling [DONE] (2026-10-14)
- ~88 features (lag 1-3, rolling 7-day, calendar+cyclical, city one-hot)
- Chronological split: Train 70% / Val 15% / Test 15% (by date)
- Fit StandardScaler on train only; test set untouched until Phase 4.
- Hyperparameter search: 4 configs, best u1=256, u2=128, u3=64, d=0.3/0.2, lr=1e-3
- MLP stopped at epoch 78 (EarlyStopping on val_accuracy)

## Phase 4 - Evaluation [DONE] (2026-10-14)

| Model        | MAE   | RMSE  | R2     | Acc    | MacroF1 | AUC    |
|-------------|-------|-------|--------|--------|---------|--------|
| Persistence | 32.58 | 50.92 | 0.6385 | -      | -       | -      |
| HistGBM     | 32.25 | 45.81 | 0.7075 | 0.5663 | 0.4542  | -      |
| MLP (DL)    | 32.90 | 48.83 | 0.6677 | 0.5504 | 0.4738  | 0.8500 |

- 7 evaluation figures saved (24-30)

## Phase 5 - Deliverables [DONE] (2026-10-14)
- `README.md` with results, setup instructions, folder tree
- `report/Project_Report.docx` generated (10 sections, figures embedded)
- `notebooks/aqi_eda_model.ipynb` generated via nbformat
- `src/aqi_eda_model.py` complete executable script
- `results.json` with all numeric outputs

## Phase 6 - Quality Gate [PENDING - manual steps]
- [ ] Restart kernel and run notebook top-to-bottom
- [ ] Fill in Title Page placeholders in report (Student Name, Roll No., Dept., Year)
- [ ] Verify [VERIFY] references in report Section 9
- [ ] Push to GitHub and submit Google Form

### Rubric Checklist
- [x] >= 5 visualizations with what/why/insights (30 figures total)
- [x] Univariate analysis (histograms, stats table)
- [x] Continuous distributions (Log-Normal vs Gamma, Q-Q plots, KS test, AIC)
- [x] Discrete distributions (AQI Bucket counts, Poisson model)
- [x] Bivariate analysis (Pearson + Spearman heatmaps, scatterplots, Kruskal-Wallis)
- [x] Multivariate analysis (pairplot, PCA, clustermap)
- [x] Deep learning model with >= 3 result visuals (7 provided: curves, CM, ROC, residuals, scatter, timeseries, comparison)
- [x] No leakage: scalers fit on train only; rolling window shifted by 1
- [x] Test set touched once at final evaluation
- [x] Chronological splits only (no random shuffling)
- [x] GitHub repo structure complete (README, .gitignore, data/README, requirements.txt)

---
*Last updated: 2026-10-14T10:08 IST*
