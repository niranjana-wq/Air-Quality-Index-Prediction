#!/usr/bin/env python
"""
make_notebook.py
Converts src/aqi_eda_model.py into notebooks/aqi_eda_model.ipynb by splitting
on section headers and wrapping each block in a code cell, with a descriptive
markdown cell prepended to each section.
Run AFTER src/aqi_eda_model.py has been verified.
"""

import os, json
import nbformat as nbf

ROOT      = os.path.dirname(os.path.abspath(__file__))
SRC_PATH  = os.path.join(ROOT, "src", "aqi_eda_model.py")
NB_PATH   = os.path.join(ROOT, "notebooks", "aqi_eda_model.ipynb")
os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)

with open(SRC_PATH, encoding="utf-8") as f:
    source = f.read()

# Section metadata: (marker in source, Markdown heading, description)
SECTIONS = [
    ("# SECTION 0",
     "## 0 — Setup & Configuration",
     "Import libraries, set random seeds (42), configure paths and global style."),

    ("# SECTION 1 (ACQUIRE)",
     "## 1 — Acquire\n### Ben Fry Stage 1: Load and Inspect Raw Data",
     "Load `city_day.csv`, print shape, dtypes, date range per city, "
     "number of cities, and summary statistics."),

    ("# SECTION 2 (PARSE)",
     "## 2 — Parse\n### Ben Fry Stage 2: Datetime Conversion and Calendar Reindex",
     "Convert Date to datetime, sort by City + Date, reindex each city to a "
     "complete daily calendar so that `shift(1)` is truly the previous day. "
     "Add: year, month, weekday, is_weekend, season (Indian 4-season), "
     "and cyclical sin/cos encodings."),

    ("# SECTION 3 (FILTER)",
     "## 3 — Filter\n### Ben Fry Stage 3: Data Cleaning",
     "**Steps (documented with before/after counts):**\n"
     "- Missing values: per-column and per-city percentages + heatmap\n"
     "- Interpolate gaps ≤ 3 days within each city (never across cities)\n"
     "- Remove duplicate (City, Date) rows\n"
     "- Outlier detection (IQR + z-score), boxplots before/after\n"
     "- Winsorize at 0.5th–99.5th percentile to preserve real pollution spikes\n"
     "- Drop rows where AQI target is NaN"),

    ("# SECTION 4 (MINE)",
     "## 4 — Mine\n### Ben Fry Stage 4: Statistical Analysis (CO4)",
     "**4a** Univariate: mean, median, std, skewness, kurtosis, percentiles; "
     "histograms with KDE; log-transform comparison.\n\n"
     "**4b** Continuous distributions: fit Log-Normal and Gamma to PM2.5 and PM10; "
     "Q-Q plots; KS test and AIC comparison.\n\n"
     "**4c** Discrete distributions: AQI Bucket counts/proportions; "
     "Poisson model for monthly 'Poor or worse' days; overdispersion check.\n\n"
     "**4d** Bivariate: Pearson + Spearman heatmaps, scatterplots with regression, "
     "boxplots by city/season/weekday, Kruskal-Wallis test.\n\n"
     "**4e** Multivariate: pairplot, PCA scree + biplot, clustermap.\n\n"
     "**4f** Time series: monthly/yearly trends, COVID lockdown effect, "
     "seasonal decomposition."),

    ("# SECTION 5/6",
     "## 5/6 — Represent + Refine\n### Ben Fry Stages 5 & 6",
     "All figures saved to `figures/` at 200 dpi with consistent colorblind-safe "
     "palette, descriptive titles, axis labels with units, and legends. "
     "Each figure cell is followed by an interpretation markdown cell with "
     "**What it shows**, **Why I created it**, and **Insights**."),

    ("# SECTION 7 (INTERACT)",
     "## 7 — Interact\n### Ben Fry Stage 7: Interactive Plotly Charts",
     "1. AQI over time with City dropdown (saved as HTML)\n"
     "2. City × Month AQI heatmap (saved as HTML)"),

    ("# SECTION 8",
     "## 8 — Feature Engineering",
     "Build the model feature matrix (~{} features per city-day row):\n"
     "- Lag-1/2/3 for 12 pollutants + AQI + AQI_Bucket_int\n"
     "- Rolling 7-day mean and std (window shifted by 1 to prevent leakage)\n"
     "- Calendar + cyclical encodings\n"
     "- City one-hot encoding\n\n"
     "**No-leakage guarantee:** all features use only data from day t "
     "when predicting day t+1."),

    ("# SECTION 9",
     "## 9 — Modeling",
     "Chronological 70/15/15 split → StandardScaler (fit on train only) → "
     "Persistence baseline → HistGBM → Hyperparameter search → "
     "MLP Regressor (Huber loss) → MLP Classifier (class-weighted)."),

    ("# SECTION 10",
     "## 10 — Evaluation\n**(Test set — touched ONCE)**",
     "Compute test metrics, generate all evaluation figures:\n"
     "- Training curves (loss + MAE/accuracy vs epoch)\n"
     "- Confusion matrix (raw and normalized)\n"
     "- One-vs-Rest ROC curves with per-class and macro AUC\n"
     "- Residual distribution + residuals vs predicted\n"
     "- Predicted vs actual scatter\n"
     "- Time-series overlay for 3 cities\n"
     "- Model comparison bar chart"),

    ("# SECTION 11",
     "## 11 — Save Results",
     "Dump all numeric results to `results.json` for the report generator."),
]

def cell_md(text):
    return nbf.v4.new_markdown_cell(text)

def cell_code(code):
    return nbf.v4.new_code_cell(code)

# Build notebook
nb    = nbf.v4.new_notebook()
cells = []

# Title cell
cells.append(cell_md(
    "# Exploratory Data Analysis and Next-Day Air Quality Prediction for Indian Cities\n\n"
    "**Course:** U21ADP05 Assignment II | **Due:** 15/10/2026 | **Seed:** 42\n\n"
    "This notebook follows [Ben Fry's 7-stage visualization pipeline](https://benfry.com/). "
    "Run top-to-bottom on a fresh kernel — `requirements.txt` must be installed first.\n\n"
    "```\n"
    "pip install -r requirements.txt\n"
    "```"
))

# Split source by section markers
lines = source.split("\n")
section_starts = []
for marker, _, _ in SECTIONS:
    for i, line in enumerate(lines):
        if marker in line and "==" in line:
            section_starts.append(i)
            break
section_starts.append(len(lines))  # sentinel

# Add cells per section
for idx, (marker, md_heading, md_desc) in enumerate(SECTIONS):
    start = section_starts[idx] if idx < len(section_starts)-1 else 0
    end   = section_starts[idx+1] if idx+1 < len(section_starts) else len(lines)
    code_block = "\n".join(lines[start:end]).strip()

    cells.append(cell_md(f"{md_heading}\n\n{md_desc}"))
    if code_block:
        cells.append(cell_code(code_block))

    # After figures, add interpretation cells
    if "SECTION 4" in marker or "SECTION 5" in marker:
        cells.append(cell_md(
            "### Figure Interpretation\n\n"
            "| | |\n|---|---|\n"
            "| **What it shows** | *(see plot title and axis labels)* |\n"
            "| **Why I created it** | *(see section description above)* |\n"
            "| **Insights** | *(specific quantitative findings — see printed output)* |"
        ))

nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (AQI)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✓ Notebook saved → {NB_PATH}")
