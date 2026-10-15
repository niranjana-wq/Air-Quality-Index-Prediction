#!/usr/bin/env python
"""
AQI EDA and Next-Day Prediction — Main Script
Course: U21ADP05 Assignment II
Author: [Student Name]
Date:   2026-10-14
Seed:   42 (numpy, random, tensorflow)
"""

# ===========================================================================
# SECTION 0: SETUP & CONFIGURATION
# ===========================================================================
import os, sys, warnings, random, json

import numpy as np
import pandas as pd
import scipy.stats as sps
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc as sk_auc,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Seeds ──────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "data", "city_day.csv")
FIGURES_DIR = os.path.join(ROOT, "figures")
REPORT_DIR  = os.path.join(ROOT, "report")
for d in [FIGURES_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Global Style ────────────────────────────────────────────────────────────
CB6  = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#F0E442"]
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200,
                     "font.family": "DejaVu Sans"})

# ── Constants ───────────────────────────────────────────────────────────────
POLLUTANTS   = ["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3",
                "Benzene","Toluene","Xylene"]
BUCKET_ORDER = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
BUCKET_MAP   = {b: i for i, b in enumerate(BUCKET_ORDER)}
SEASON_MAP   = {12:"Winter",1:"Winter",2:"Winter",
                3:"Pre-Monsoon",4:"Pre-Monsoon",5:"Pre-Monsoon",
                6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
                10:"Post-Monsoon",11:"Post-Monsoon"}
SEASON_INT   = {"Winter":0,"Pre-Monsoon":1,"Monsoon":2,"Post-Monsoon":3}
MONTH_NAMES  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
ROLL_COLS    = ["PM2.5","PM10","NO2","NOx","CO","SO2","O3","AQI"]
LAG_COLS     = POLLUTANTS + ["AQI","AQI_Bucket_int"]
NUM_CLASSES  = 6

def save_fig(name, tight=True):
    path = os.path.join(FIGURES_DIR, name)
    if tight:
        plt.savefig(path, dpi=200, bbox_inches="tight")
    else:
        plt.savefig(path, dpi=200)
    plt.close("all")
    print(f"  OK Saved: figures/{name}")
    return path

RESULTS = {}   # filled throughout; dumped to results.json at the end

print("=" * 70)
print("SECTION 0 COMPLETE — Setup OK")
print("=" * 70)

# ===========================================================================
# SECTION 1 (ACQUIRE): LOAD AND INSPECT RAW DATA
# ===========================================================================
print("\n=== SECTION 1: ACQUIRE ===")

df_raw = pd.read_csv(DATA_PATH)
print(f"Shape            : {df_raw.shape}")
print(f"Columns          : {df_raw.columns.tolist()}")
print(f"Dtypes\n{df_raw.dtypes}\n")
print(f"Date range (raw) : {df_raw['Date'].min()} -> {df_raw['Date'].max()}")
print(f"Cities ({df_raw['City'].nunique()}): {sorted(df_raw['City'].unique())}")

city_dates = (df_raw.groupby("City")["Date"]
              .agg(["min","max","count"])
              .rename(columns={"min":"First","max":"Last","count":"Rows"}))
print(f"\nPer-city date ranges:\n{city_dates.to_string()}")
print(f"\nSummary statistics:\n{df_raw.describe().round(2).to_string()}")
print(f"\nAQI_Bucket distribution:\n{df_raw['AQI_Bucket'].value_counts()}")
print(f"AQI_Bucket missing: {df_raw['AQI_Bucket'].isna().sum()} "
      f"({100*df_raw['AQI_Bucket'].isna().mean():.1f} %)")

RESULTS["shape"]    = list(df_raw.shape)
RESULTS["n_cities"] = int(df_raw["City"].nunique())
RESULTS["date_min"] = str(df_raw["Date"].min())
RESULTS["date_max"] = str(df_raw["Date"].max())
RESULTS["cities"]   = sorted(df_raw["City"].unique())

# ===========================================================================
# SECTION 2 (PARSE): DATETIME, REINDEX, CALENDAR FEATURES
# ===========================================================================
print("\n=== SECTION 2: PARSE ===")

df = df_raw.copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["City","Date"]).reset_index(drop=True)

# Reindex each city to a complete daily calendar (so shift(1) = true prev day)
def reindex_city(grp):
    city = grp["City"].iloc[0]
    full = pd.date_range(grp["Date"].min(), grp["Date"].max(), freq="D")
    grp = grp.set_index("Date").reindex(full)
    grp.index.name = "Date"
    grp["City"] = city
    return grp.reset_index()

parts = [reindex_city(g) for _, g in df.groupby("City")]
df    = pd.concat(parts, ignore_index=True)
print(f"After reindex: {df.shape}")

# Calendar features
df["year"]        = df["Date"].dt.year
df["month"]       = df["Date"].dt.month
df["weekday"]     = df["Date"].dt.weekday
df["is_weekend"]  = (df["weekday"] >= 5).astype(int)
df["season"]      = df["month"].map(SEASON_MAP)
df["season_int"]  = df["season"].map(SEASON_INT)
df["month_sin"]   = np.sin(2*np.pi*df["month"]/12)
df["month_cos"]   = np.cos(2*np.pi*df["month"]/12)
df["weekday_sin"] = np.sin(2*np.pi*df["weekday"]/7)
df["weekday_cos"] = np.cos(2*np.pi*df["weekday"]/7)

print(f"Calendar cols added. Season distribution:\n{df['season'].value_counts()}")

# ===========================================================================
# SECTION 3 (FILTER): CLEANING
# ===========================================================================
print("\n=== SECTION 3: FILTER ===")

n_start = len(df)

# ── 3a. Missing values ──────────────────────────────────────────────────────
miss_pct = df[POLLUTANTS + ["AQI","AQI_Bucket"]].isna().mean() * 100
print(f"\nOverall missing %:\n{miss_pct.round(1).to_string()}")

city_miss = (df.groupby("City")[POLLUTANTS + ["AQI"]]
               .apply(lambda x: x.isna().mean()*100).round(1))
print(f"\nPer-city missing % (first 5):\n{city_miss.head().to_string()}")

# Missingness heatmap (Figure 01)
fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(city_miss, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "Missing %"}, linewidths=0.4, linecolor="white")
ax.set_title("Missing Value % by City and Pollutant", fontsize=14, fontweight="bold")
ax.set_xlabel("Pollutant / AQI"); ax.set_ylabel("City")
plt.tight_layout()
save_fig("01_missingness_heatmap.png")

RESULTS["miss_pct"] = miss_pct.round(1).to_dict()

# Drop columns with > 60% missing overall
drop_cols = miss_pct[miss_pct > 60].index.tolist()
print(f"\nColumns with >60% missing overall: {drop_cols}")
# (Xylene may be near threshold; keep it but allow NaN in features)

# ── 3b. Interpolate short gaps (<= 3 days) within each city ─────────────────
def interp_city(grp):
    for col in POLLUTANTS + ["AQI"]:
        grp[col] = grp[col].interpolate(method="linear", limit=3, limit_direction="forward")
    return grp

miss_before = df[POLLUTANTS+["AQI"]].isna().sum()
df = df.groupby("City", group_keys=False).apply(interp_city)
miss_after  = df[POLLUTANTS+["AQI"]].isna().sum()
interp_df   = pd.DataFrame({"Before": miss_before, "After": miss_after,
                             "Filled": miss_before - miss_after})
print(f"\nNaN reduction after interpolation:\n{interp_df.to_string()}")

# ── 3c. Duplicates ──────────────────────────────────────────────────────────
n_dupes = df.duplicated(subset=["City","Date"]).sum()
print(f"\nDuplicate (City, Date) rows: {n_dupes}")
df = df[~df.duplicated(subset=["City","Date"])].copy()
RESULTS["n_duplicates"] = int(n_dupes)

# ── 3d. Outliers: boxplots before winsorization (Figure 02) ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, col in zip(axes.flat, ["PM2.5","PM10","NO2","CO","SO2","O3"]):
    data = df[col].dropna()
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#AED6F1"),
               flierprops=dict(marker="o", markersize=2, alpha=0.3),
               medianprops=dict(color="red", lw=2))
    ax.set_title(f"{col} (Before Winsorize)", fontsize=11)
    ax.set_ylabel("ug/m^3" if col not in ["CO","NOx"] else "mg/m^3" if col=="CO" else "ppb")
plt.suptitle("Boxplots — Key Pollutants Before Winsorization", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("02_boxplots_before_winsorize.png")

# IQR outlier count
iqr_counts = {}
for col in POLLUTANTS + ["AQI"]:
    v = df[col].dropna()
    Q1, Q3 = v.quantile(0.25), v.quantile(0.75)
    IQR = Q3 - Q1
    iqr_counts[col] = int(((v < Q1-1.5*IQR)|(v > Q3+1.5*IQR)).sum())
print(f"\nIQR outlier counts:\n{pd.Series(iqr_counts).to_string()}")

# Z-score outlier count (|z| > 3)
z_counts = {col: int((np.abs(sps.zscore(df[col].dropna())) > 3).sum())
            for col in POLLUTANTS + ["AQI"]}
print(f"\nZ-score (>3) outlier counts:\n{pd.Series(z_counts).to_string()}")
RESULTS["iqr_outliers"] = iqr_counts
RESULTS["z_outliers"]   = z_counts

# Winsorize at 0.5th–99.5th percentile per pollutant (preserve real spikes, clip extremes)
for col in POLLUTANTS:
    lo = df[col].quantile(0.005)
    hi = df[col].quantile(0.995)
    df[col] = df[col].clip(lower=lo, upper=hi)
print("\nWinsorization applied (0.5th–99.5th percentile per pollutant).")

# Boxplots after winsorization (Figure 03)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, col in zip(axes.flat, ["PM2.5","PM10","NO2","CO","SO2","O3"]):
    data = df[col].dropna()
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#A9DFBF"),
               flierprops=dict(marker="o", markersize=2, alpha=0.3),
               medianprops=dict(color="red", lw=2))
    ax.set_title(f"{col} (After Winsorize)", fontsize=11)
    ax.set_ylabel("ug/m^3" if col not in ["CO","NOx"] else "mg/m^3" if col=="CO" else "ppb")
plt.suptitle("Boxplots — Key Pollutants After Winsorization", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("03_boxplots_after_winsorize.png")

# ── 3e. Drop rows where AQI target is unavailable ──────────────────────────
df_eda   = df.copy()                                    # retain all for EDA
df_model = df.dropna(subset=["AQI","AQI_Bucket"]).copy()
print(f"\nEDA df rows  : {len(df_eda)}")
print(f"Model df rows: {len(df_model)}")
print(f"AQI_Bucket distribution (model df):\n{df_model['AQI_Bucket'].value_counts()}")
RESULTS["n_eda"]   = int(len(df_eda))
RESULTS["n_model"] = int(len(df_model))
RESULTS["bucket_counts"] = df_model["AQI_Bucket"].value_counts().to_dict()

# ===========================================================================
# SECTION 4 (MINE): STATISTICAL ANALYSIS
# ===========================================================================
print("\n=== SECTION 4: MINE ===")

# ── 4a. Univariate Statistics ───────────────────────────────────────────────
print("\n-- 4a. Univariate Statistics --")

def describe_ext(df_, cols):
    rows = []
    for col in cols:
        v = df_[col].dropna()
        rows.append({"Pollutant": col, "N": len(v),
                     "Mean": v.mean(), "Median": v.median(), "Std": v.std(),
                     "Skewness": v.skew(), "Kurtosis": v.kurt(),
                     "P5": v.quantile(0.05), "P25": v.quantile(0.25),
                     "P75": v.quantile(0.75), "P95": v.quantile(0.95),
                     "Min": v.min(), "Max": v.max()})
    return pd.DataFrame(rows).set_index("Pollutant")

stats_tbl = describe_ext(df_eda, POLLUTANTS + ["AQI"])
print(f"\nExtended univariate statistics:\n{stats_tbl.round(2).to_string()}")
RESULTS["univariate_stats"] = stats_tbl.round(2).to_dict()

# Histograms with KDE — all pollutants (Figure 04)
fig, axes = plt.subplots(4, 3, figsize=(18, 22))
units = {"CO": "mg/m^3", "NOx": "ppb"}
for ax, col in zip(axes.flat, POLLUTANTS):
    data = df_eda[col].dropna()
    sns.histplot(data, kde=True, ax=ax, color="#2E86AB", bins=50, alpha=0.75)
    unit = units.get(col, "ug/m^3")
    ax.set_title(f"{col}  (skew={data.skew():.2f})", fontsize=10, fontweight="bold")
    ax.set_xlabel(unit); ax.set_ylabel("Count")
plt.suptitle("Univariate Distributions with KDE — All Pollutants",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
save_fig("04_histograms_kde.png")

# Log1p transform comparison for PM2.5 & PM10 (Figure 05)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in zip(axes, ["PM2.5","PM10"]):
    data = np.log1p(df_eda[col].dropna())
    sns.histplot(data, kde=True, ax=ax, color="#E76F51", bins=50, alpha=0.75)
    ax.set_title(f"log(1+{col})  (skew after={data.skew():.2f})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"log(1+{col})"); ax.set_ylabel("Count")
plt.suptitle("Log₁₊₁ Transformation — PM2.5 and PM10",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("05_log_transform_pm.png")

# ── 4b. Continuous Distribution Fitting ─────────────────────────────────────
print("\n-- 4b. Continuous Distribution Fitting --")

def fit_dists(series, col, prefix):
    data = series.dropna()
    data = data[data > 0]

    # Log-normal
    s_ln, loc_ln, scale_ln = sps.lognorm.fit(data, floc=0)
    ks_ln, p_ln = sps.kstest(data, "lognorm", args=(s_ln, loc_ln, scale_ln))
    ll_ln  = sps.lognorm.logpdf(data, s_ln, loc_ln, scale_ln).sum()
    aic_ln = 2*2 - 2*ll_ln

    # Gamma
    a_g, loc_g, scale_g = sps.gamma.fit(data, floc=0)
    ks_g, p_g = sps.kstest(data, "gamma", args=(a_g, loc_g, scale_g))
    ll_g  = sps.gamma.logpdf(data, a_g, loc_g, scale_g).sum()
    aic_g = 2*2 - 2*ll_g

    better = "Log-Normal" if aic_ln < aic_g else "Gamma"
    print(f"\n  {col}:")
    print(f"    Log-Normal -> KS={ks_ln:.4f}, p={p_ln:.4f}, AIC={aic_ln:.1f}")
    print(f"    Gamma      -> KS={ks_g:.4f}, p={p_g:.4f}, AIC={aic_g:.1f}")
    print(f"    Better fit (lower AIC): {better}")

    # Q-Q plots (Figure 06/07)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, dist_name, dist_args, dist_obj, ks, p_val, aic in [
        (axes[0], "Log-Normal", (s_ln,), sps.lognorm(s_ln, scale=scale_ln), ks_ln, p_ln, aic_ln),
        (axes[1], "Gamma",      (a_g,),  sps.gamma(a_g, scale=scale_g),     ks_g,  p_g,  aic_g),
    ]:
        # Generate theoretical and empirical quantiles
        probs   = np.linspace(0.01, 0.99, 200)
        theo_q  = dist_obj.ppf(probs)
        emp_q   = np.quantile(data, probs)
        ax.scatter(theo_q, emp_q, s=8, alpha=0.5, color="#2E86AB")
        mn_ = min(theo_q.min(), emp_q.min()); mx_ = max(theo_q.max(), emp_q.max())
        ax.plot([mn_, mx_], [mn_, mx_], "r--", lw=2, label="y=x (perfect fit)")
        ax.set_title(f"{col} — {dist_name} Q-Q\nKS={ks:.4f}  p={p_val:.4f}  AIC={aic:.0f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
        ax.legend(fontsize=9)
    plt.suptitle(f"Q-Q Plots: Distribution Fitting — {col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(f"{prefix}_qq.png")

    return dict(lognorm_ks=float(ks_ln), lognorm_p=float(p_ln), lognorm_aic=float(aic_ln),
                gamma_ks=float(ks_g), gamma_p=float(p_g), gamma_aic=float(aic_g),
                better=better)

RESULTS["dist_pm25"] = fit_dists(df_eda["PM2.5"], "PM2.5", "06_pm25")
RESULTS["dist_pm10"] = fit_dists(df_eda["PM10"],  "PM10",  "07_pm10")

# ── 4c. Discrete Distributions ──────────────────────────────────────────────
print("\n-- 4c. Discrete Distributions --")

# AQI_Bucket bar + pie (Figure 08)
bc = df_eda["AQI_Bucket"].value_counts().reindex(BUCKET_ORDER).dropna()
bp = bc / bc.sum()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].bar(bc.index, bc.values, color=CB6[:len(bc)], edgecolor="black", linewidth=0.5)
axes[0].set_title("AQI Category Frequency", fontsize=12, fontweight="bold")
axes[0].set_xlabel("AQI Bucket"); axes[0].set_ylabel("Days (count)")
axes[0].tick_params(axis="x", rotation=30)
for i, (val, cnt) in enumerate(zip(bc.index, bc.values)):
    axes[0].text(i, cnt + 50, str(cnt), ha="center", fontsize=9)

axes[1].pie(bp.values, labels=bp.index, autopct="%1.1f%%",
            colors=CB6[:len(bp)], startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":1.5})
axes[1].set_title("AQI Category Proportions", fontsize=12, fontweight="bold")
plt.suptitle("Discrete Distribution of AQI Categories", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("08_aqi_bucket_distribution.png")

# Poisson model for monthly "Poor or worse" days (Figure 09)
poor_mask = df_eda["AQI_Bucket"].isin(["Poor","Very Poor","Severe"])
df_eda2 = df_eda.dropna(subset=["AQI_Bucket"]).copy()
df_eda2["is_poor"] = df_eda2["AQI_Bucket"].isin(["Poor","Very Poor","Severe"]).astype(int)
monthly_poor = (df_eda2.groupby(["City","year","month"])["is_poor"]
                       .sum().reset_index(name="count"))
lambda_hat  = monthly_poor["count"].mean()
var_obs     = monthly_poor["count"].var()
disp_idx    = var_obs / lambda_hat
print(f"\nPoisson model — monthly 'Poor or worse' days:")
print(f"  Lambda (mean):              {lambda_hat:.2f}")
print(f"  Observed variance:          {var_obs:.2f}")
print(f"  Dispersion index (var/mean):{disp_idx:.2f}")
print(f"  {'Overdispersion detected' if disp_idx > 1.2 else 'Reasonable Poisson fit'}")
RESULTS["lambda_poisson"] = float(lambda_hat)
RESULTS["disp_index"]     = float(disp_idx)

k_max = int(monthly_poor["count"].max())
k_vec = np.arange(0, k_max+1)
obs_c = monthly_poor["count"].value_counts().reindex(k_vec, fill_value=0)
exp_c = sps.poisson.pmf(k_vec, lambda_hat) * len(monthly_poor)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(k_vec, obs_c.values, alpha=0.7, label="Observed", color="#2E86AB",
       edgecolor="black", linewidth=0.4)
ax.plot(k_vec, exp_c, "ro-", ms=5, lw=2, label=f"Poisson(λ={lambda_hat:.1f})")
ax.set_xlim(-0.5, min(35, k_max)+0.5)
ax.set_title(f"Monthly 'Poor or Worse' Days — Observed vs Poisson\n"
             f"Dispersion Index = {disp_idx:.2f}  "
             f"({'Overdispersion: Poisson underestimates variance' if disp_idx>1.2 else 'Good Poisson fit'})",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Number of Poor-or-Worse Days per Month")
ax.set_ylabel("Frequency")
ax.legend(fontsize=11)
plt.tight_layout()
save_fig("09_poisson_poor_days.png")

# ── 4d. Bivariate Analysis ──────────────────────────────────────────────────
print("\n-- 4d. Bivariate Analysis --")

# Pearson correlation heatmap (Figure 10)
corr_p = df_eda[POLLUTANTS+["AQI"]].corr(method="pearson")
mask   = np.triu(np.ones_like(corr_p, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr_p, mask=~mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.4,
            cbar_kws={"shrink":0.8, "label":"Pearson r"})
ax.set_title("Pearson Correlation Heatmap — Pollutants + AQI",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("10_pearson_correlation.png")

# Spearman correlation heatmap (Figure 11)
corr_s = df_eda[POLLUTANTS+["AQI"]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr_s, mask=~mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.4,
            cbar_kws={"shrink":0.8, "label":"Spearman ρ"})
ax.set_title("Spearman Correlation Heatmap — Pollutants + AQI",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("11_spearman_correlation.png")

# Scatterplots + regression lines (Figure 12)
pairs = [("PM2.5","PM10"), ("NO2","NOx"), ("PM2.5","AQI")]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (x, y) in zip(axes, pairs):
    d = df_eda[[x,y]].dropna()
    ax.scatter(d[x], d[y], s=4, alpha=0.2, color="#2E86AB")
    m, b, r, p_, se = sps.linregress(d[x], d[y])
    xl = np.linspace(d[x].min(), d[x].max(), 100)
    ax.plot(xl, m*xl+b, "r-", lw=2, label=f"r = {r:.2f}")
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
plt.suptitle("Bivariate Scatterplots with OLS Regression Lines",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("12_scatterplots_regression.png")

# AQI by city (Figure 13)
city_ord = (df_eda.dropna(subset=["AQI"])
            .groupby("City")["AQI"].median()
            .sort_values().index.tolist())
fig, ax = plt.subplots(figsize=(16, 7))
sns.boxplot(data=df_eda.dropna(subset=["AQI"]), x="City", y="AQI",
            order=city_ord, palette="husl", ax=ax,
            flierprops=dict(marker="o", markersize=2, alpha=0.3))
ax.set_title("AQI Distribution by City (sorted by median)", fontsize=13, fontweight="bold")
ax.set_xlabel("City"); ax.set_ylabel("AQI")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_fig("13_aqi_by_city_boxplot.png")

# AQI by season (Figure 14)
season_ord = ["Winter","Pre-Monsoon","Monsoon","Post-Monsoon"]
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_eda.dropna(subset=["AQI"]), x="season", y="AQI",
            order=season_ord, palette=CB6[:4], ax=ax,
            flierprops=dict(marker="o", markersize=2, alpha=0.3))
ax.set_title("AQI Distribution by Indian Season", fontsize=13, fontweight="bold")
ax.set_xlabel("Season"); ax.set_ylabel("AQI")
plt.tight_layout()
save_fig("14_aqi_by_season_boxplot.png")

# Kruskal-Wallis test across seasons
kw_groups = [df_eda.dropna(subset=["AQI"])[df_eda["season"]==s]["AQI"].values
             for s in season_ord]
kw_stat, kw_p = sps.kruskal(*kw_groups)
print(f"\nKruskal-Wallis (AQI across seasons): H={kw_stat:.2f}, p={kw_p:.4e}")
print(f"  {'Significant seasonal differences (p<0.05)' if kw_p<0.05 else 'Not significant'}")
RESULTS["kw_stat"] = float(kw_stat); RESULTS["kw_p"] = float(kw_p)

# AQI by weekday (Figure 15)
day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
df_eda["weekday_name"] = df_eda["weekday"].map(dict(enumerate(day_names)))
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_eda.dropna(subset=["AQI"]), x="weekday_name", y="AQI",
            order=day_names, palette="muted", ax=ax,
            flierprops=dict(marker="o", markersize=2, alpha=0.3))
ax.set_title("AQI Distribution by Day of Week", fontsize=13, fontweight="bold")
ax.set_xlabel("Day"); ax.set_ylabel("AQI")
plt.tight_layout()
save_fig("15_aqi_by_weekday_boxplot.png")

# ── 4e. Multivariate Analysis ────────────────────────────────────────────────
print("\n-- 4e. Multivariate Analysis --")

# Pairplot (Figure 16) — sampled for speed
pp_cols = ["PM2.5","PM10","NO2","CO","O3","AQI","AQI_Bucket"]
df_pp   = (df_eda[pp_cols].dropna()
           .sample(min(3000, len(df_eda.dropna(subset=pp_cols))), random_state=SEED))
pp_grid = sns.pairplot(df_pp, hue="AQI_Bucket", hue_order=BUCKET_ORDER,
                       vars=["PM2.5","PM10","NO2","CO","O3","AQI"],
                       diag_kind="kde",
                       plot_kws={"alpha":0.25,"s":8},
                       palette=dict(zip(BUCKET_ORDER, CB6)))
pp_grid.fig.suptitle("Pairplot — Main Pollutants colored by AQI Bucket",
                     y=1.01, fontsize=13, fontweight="bold")
pp_grid.fig.savefig(os.path.join(FIGURES_DIR,"16_pairplot.png"),
                    dpi=200, bbox_inches="tight")
plt.close("all")
print("  OK Saved: figures/16_pairplot.png")

# PCA (Figure 17)
pca_use = [c for c in POLLUTANTS if df_eda[c].isna().mean() < 0.5]
df_pca  = df_eda[pca_use].dropna()
X_pca   = StandardScaler().fit_transform(df_pca)
pca_mdl = PCA(random_state=SEED)
pca_mdl.fit(X_pca)
evr     = pca_mdl.explained_variance_ratio_
n_80    = int(np.argmax(np.cumsum(evr) >= 0.80) + 1)
print(f"\nPCA: {n_80} components explain >= 80% variance")
print(f"PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%")
RESULTS["pca_n80"] = n_80; RESULTS["pca_pc1"] = float(evr[0]*100); RESULTS["pca_pc2"] = float(evr[1]*100)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].bar(range(1,len(evr)+1), evr*100, color="#2E86AB", edgecolor="black", lw=0.4)
axes[0].plot(range(1,len(evr)+1), np.cumsum(evr)*100, "ro-", lw=2, ms=5)
axes[0].axhline(80, ls="--", color="green", lw=1, label="80% threshold")
axes[0].set_title("PCA Scree Plot", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Component"); axes[0].set_ylabel("Variance Explained (%)")
axes[0].legend()

Xp = pca_mdl.transform(X_pca)
axes[1].scatter(Xp[:,0], Xp[:,1], s=4, alpha=0.15, color="#2E86AB")
for i, col in enumerate(pca_use):
    axes[1].annotate("", xy=(pca_mdl.components_[0,i]*3, pca_mdl.components_[1,i]*3),
                     xytext=(0,0), arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    axes[1].text(pca_mdl.components_[0,i]*3.3, pca_mdl.components_[1,i]*3.3,
                 col, fontsize=8, color="darkred")
axes[1].set_title(f"PCA Biplot (PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%)",
                  fontsize=12, fontweight="bold")
axes[1].set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); axes[1].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
plt.suptitle("PCA — Pollutant Data", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("17_pca_scree_biplot.png")

# Clustermap (Figure 18)
df_clust = (df_eda[POLLUTANTS+["AQI"]].dropna()
            .sample(min(2000, len(df_eda.dropna(subset=POLLUTANTS+["AQI"]))), random_state=SEED))
cg = sns.clustermap(df_clust.corr(), cmap="coolwarm", center=0,
                    annot=True, fmt=".2f", figsize=(12,12), linewidths=0.4)
cg.fig.suptitle("Clustermap — Pollutant Correlations", y=1.01, fontsize=13, fontweight="bold")
cg.fig.savefig(os.path.join(FIGURES_DIR,"18_clustermap.png"), dpi=200, bbox_inches="tight")
plt.close("all")
print("  OK Saved: figures/18_clustermap.png")

# ── 4f. Time Series ──────────────────────────────────────────────────────────
print("\n-- 4f. Time Series Analysis --")

df_nn = df_eda.dropna(subset=["AQI"])

# Monthly average (Figure 19)
monthly_avg = df_nn.groupby(["year","month"])["AQI"].mean().reset_index()
monthly_avg["date"] = pd.to_datetime(monthly_avg[["year","month"]].assign(day=1))
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(monthly_avg["date"], monthly_avg["AQI"], color="#2E86AB", lw=2)
ax.axvspan(pd.Timestamp("2020-03-24"), pd.Timestamp("2020-06-01"),
           alpha=0.2, color="red", label="COVID Lockdown (Mar–Jun 2020)")
ax.set_title("Monthly Mean AQI — All Cities (2015–2020)\nCOVID Lockdown Highlighted",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Mean AQI"); ax.legend(fontsize=11)
plt.tight_layout()
save_fig("19_monthly_aqi_trend.png")

# Yearly AQI (Figure 20)
yearly_aqi = df_nn.groupby("year")["AQI"].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(yearly_aqi["year"], yearly_aqi["AQI"],
              color=CB6[:len(yearly_aqi)], edgecolor="black", lw=0.5)
for bar, val in zip(bars, yearly_aqi["AQI"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{val:.0f}", ha="center", fontsize=10)
ax.set_title("Yearly Mean AQI — All Cities", fontsize=13, fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Mean AQI")
plt.tight_layout()
save_fig("20_yearly_aqi_trend.png")

# COVID lockdown quantification
pre  = df_nn[(df_nn["Date"]>="2020-01-01")&(df_nn["Date"]<"2020-03-24")]["AQI"]
lock = df_nn[(df_nn["Date"]>="2020-03-24")&(df_nn["Date"]<"2020-06-01")]["AQI"]
if len(pre)>0 and len(lock)>0:
    reduction_pct = (pre.mean()-lock.mean())/pre.mean()*100
    print(f"\nCOVID Lockdown effect:")
    print(f"  Pre-lockdown mean AQI : {pre.mean():.1f}")
    print(f"  During lockdown mean  : {lock.mean():.1f}")
    print(f"  Reduction             : {pre.mean()-lock.mean():.1f} units ({reduction_pct:.1f}%)")
    RESULTS["covid_pre"]  = float(pre.mean())
    RESULTS["covid_lock"] = float(lock.mean())
    RESULTS["covid_reduction_pct"] = float(reduction_pct)

# Seasonal decomposition for most data-rich city (Figure 21)
best_city  = df_nn.groupby("City").size().idxmax()
city_ts    = (df_nn[df_nn["City"]==best_city]
              .set_index("Date")["AQI"]
              .resample("MS").mean()
              .dropna())
if len(city_ts) >= 24:
    decomp = seasonal_decompose(city_ts, model="additive", period=12)
    fig = decomp.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle(f"Seasonal Decomposition — {best_city} (Monthly AQI)",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(FIGURES_DIR,"21_seasonal_decompose.png"),
                dpi=200, bbox_inches="tight")
    plt.close("all")
    print("  OK Saved: figures/21_seasonal_decompose.png")
RESULTS["best_city_for_decomp"] = best_city

# PM2.5 city × month heatmap (Figure 22)
pm_pivot = (df_eda.dropna(subset=["PM2.5"])
            .groupby(["City","month"])["PM2.5"].mean()
            .unstack())
fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(pm_pivot, cmap="YlOrRd", annot=True, fmt=".0f", ax=ax,
            cbar_kws={"label":"Mean PM2.5 (ug/m^3)"}, linewidths=0.3)
ax.set_xticklabels(MONTH_NAMES, rotation=30)
ax.set_title("Mean PM2.5 (ug/m^3) by City and Month", fontsize=13, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("City")
plt.tight_layout()
save_fig("22_pm25_city_month_heatmap.png")

# ===========================================================================
# SECTION 5/6 (REPRESENT + REFINE): already embedded above — all figures saved
# ===========================================================================
print("\n=== SECTIONS 5/6: REPRESENT + REFINE — all 22 figures saved ===")

# ===========================================================================
# SECTION 7 (INTERACT): PLOTLY INTERACTIVE CHARTS
# ===========================================================================
print("\n=== SECTION 7: INTERACT ===")

# Plotly 1: AQI over time with City dropdown
df_pl   = df_eda.dropna(subset=["AQI"]).sort_values("Date")
cities_ = sorted(df_pl["City"].unique())
fig1    = go.Figure()
for city in cities_:
    cd = df_pl[df_pl["City"]==city]
    fig1.add_trace(go.Scatter(x=cd["Date"], y=cd["AQI"], name=city,
                              visible=(city==cities_[0]),
                              mode="lines", line=dict(width=1.5)))
buttons = [dict(label=c, method="update",
                args=[{"visible": [c2==c for c2 in cities_]},
                      {"title": f"Daily AQI — {c}"}])
           for c in cities_]
fig1.update_layout(
    title=f"Daily AQI Over Time — {cities_[0]}",
    xaxis_title="Date", yaxis_title="AQI",
    updatemenus=[dict(active=0, buttons=buttons,
                      x=0.01, y=1.12, xanchor="left", yanchor="top")],
    height=520, template="plotly_white",
    font=dict(family="Arial", size=12))
fig1.write_html(os.path.join(FIGURES_DIR, "interactive_aqi_city_dropdown.html"))
print("  OK Saved: figures/interactive_aqi_city_dropdown.html")

# Plotly 2: City × Month heatmap
aqi_cm = (df_eda.dropna(subset=["AQI"])
          .groupby(["City","month"])["AQI"].mean().reset_index())
aqi_cm["Month"] = aqi_cm["month"].map(dict(zip(range(1,13), MONTH_NAMES)))
piv2 = aqi_cm.pivot(index="City", columns="Month", values="AQI").reindex(columns=MONTH_NAMES)
fig2 = go.Figure(go.Heatmap(
    z=piv2.values, x=piv2.columns, y=piv2.index,
    colorscale="YlOrRd", colorbar_title="Mean AQI",
    hovertemplate="City: %{y}<br>Month: %{x}<br>Mean AQI: %{z:.1f}<extra></extra>"))
fig2.update_layout(
    title="Mean AQI by City and Month — Interactive Heatmap",
    xaxis_title="Month", yaxis_title="City",
    height=620, template="plotly_white",
    font=dict(family="Arial", size=12))
fig2.write_html(os.path.join(FIGURES_DIR, "interactive_aqi_city_month_heatmap.html"))
print("  OK Saved: figures/interactive_aqi_city_month_heatmap.html")

# ===========================================================================
# SECTION 8: FEATURE ENGINEERING
# ===========================================================================
print("\n=== SECTION 8: FEATURE ENGINEERING ===")

# Start from df_eda (complete calendar, winsorized) — compute all lags on full series
# then join targets, then drop NaN rows
df_fe = df_eda.sort_values(["City","Date"]).copy()
df_fe["AQI_Bucket_int"] = df_fe["AQI_Bucket"].map(BUCKET_MAP)

# Next-day targets (shift -1 within city)
df_fe["target_AQI"]     = df_fe.groupby("City")["AQI"].shift(-1)
df_fe["target_AQI_Bucket_int"] = df_fe.groupby("City")["AQI_Bucket_int"].shift(-1)

# Lag features (lags 1,2,3 = shift 1,2,3 = prev days relative to current row = D-1,D-2,D-3
# from prediction day D+1 perspective -> lag-1,lag-2,lag-3)
def add_lags(grp):
    for lag in [1,2,3]:
        for col in LAG_COLS:
            if col in grp.columns:
                grp[f"{col}_lag{lag}"] = grp[col].shift(lag)
    return grp

def add_rolling(grp):
    for col in ROLL_COLS:
        if col in grp.columns:
            shifted = grp[col].shift(1)          # window ends at D-1 (safe)
            grp[f"{col}_roll7_mean"] = shifted.rolling(7, min_periods=3).mean()
            grp[f"{col}_roll7_std"]  = shifted.rolling(7, min_periods=3).std()
    return grp

print("  Adding lag features …")
df_fe = df_fe.groupby("City", group_keys=False).apply(add_lags)
print("  Adding rolling features …")
df_fe = df_fe.groupby("City", group_keys=False).apply(add_rolling)

# City one-hot encoding
df_fe = pd.get_dummies(df_fe, columns=["City"], prefix="city", drop_first=False)
city_cols = sorted([c for c in df_fe.columns if c.startswith("city_")])

# Define feature column list
lag_feat_cols  = [f"{col}_lag{lag}" for lag in [1,2,3] for col in LAG_COLS]
roll_feat_cols = ([f"{c}_roll7_mean" for c in ROLL_COLS] +
                  [f"{c}_roll7_std"  for c in ROLL_COLS])
cal_cols       = ["month","weekday","is_weekend","season_int",
                  "month_sin","month_cos","weekday_sin","weekday_cos"]
FEAT_COLS      = lag_feat_cols + roll_feat_cols + cal_cols + city_cols

# Presence check — keep only columns that actually exist
FEAT_COLS = [c for c in FEAT_COLS if c in df_fe.columns]

print(f"\nFeature group breakdown:")
print(f"  Lag features (3 lags × {len(LAG_COLS)} cols): {len([c for c in FEAT_COLS if '_lag' in c])}")
print(f"  Rolling 7-day (mean+std × {len(ROLL_COLS)}): {len([c for c in FEAT_COLS if '_roll7' in c])}")
print(f"  Calendar:  {len(cal_cols)}")
print(f"  City OH:   {len(city_cols)}")
print(f"  TOTAL:     {len(FEAT_COLS)}")
RESULTS["n_features"] = len(FEAT_COLS)

# Drop rows missing any feature or either target
df_fe = df_fe.dropna(subset=FEAT_COLS + ["target_AQI","target_AQI_Bucket_int"]).copy()
df_fe["target_AQI_Bucket_int"] = df_fe["target_AQI_Bucket_int"].astype(int)
print(f"\nModel-ready dataset shape: {df_fe.shape}")

# ===========================================================================
# SECTION 9: MODELING
# ===========================================================================
print("\n=== SECTION 9: MODELING ===")

# ── 9a. Chronological split ─────────────────────────────────────────────────
all_dates = sorted(df_fe["Date"].unique())
n_d       = len(all_dates)
i_val     = int(n_d * 0.70)
i_test    = int(n_d * 0.85)

tr_end  = all_dates[i_val - 1]
va_end  = all_dates[i_test - 1]
te_end  = all_dates[-1]
va_start= all_dates[i_val]
te_start= all_dates[i_test]

print(f"\nChronological split:")
print(f"  Train : {all_dates[0].date()} -> {tr_end.date()}")
print(f"  Val   : {va_start.date()} -> {va_end.date()}")
print(f"  Test  : {te_start.date()} -> {te_end.date()}")
RESULTS.update({"train_start": str(all_dates[0].date()),
                "train_end":   str(tr_end.date()),
                "val_start":   str(va_start.date()),
                "val_end":     str(va_end.date()),
                "test_start":  str(te_start.date()),
                "test_end":    str(te_end.date())})

tr_mask = df_fe["Date"] <= tr_end
va_mask = (df_fe["Date"] > tr_end) & (df_fe["Date"] <= va_end)
te_mask = df_fe["Date"] > va_end

def split(mask):
    X = df_fe.loc[mask, FEAT_COLS].values.astype(np.float32)
    yr= df_fe.loc[mask, "target_AQI"].values.astype(np.float32)
    yc= df_fe.loc[mask, "target_AQI_Bucket_int"].values.astype(np.int32)
    return X, yr, yc

X_tr, y_tr_r, y_tr_c = split(tr_mask)
X_va, y_va_r, y_va_c = split(va_mask)
X_te, y_te_r, y_te_c = split(te_mask)

print(f"  Train rows={X_tr.shape[0]}, Val rows={X_va.shape[0]}, Test rows={X_te.shape[0]}")
RESULTS["n_train"] = int(X_tr.shape[0])
RESULTS["n_val"]   = int(X_va.shape[0])
RESULTS["n_test"]  = int(X_te.shape[0])

# ── 9b. Scale (fit on train only) ──────────────────────────────────────────
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

# ── 9c. Persistence baseline ────────────────────────────────────────────────
aqi_lag1_idx = FEAT_COLS.index("AQI_lag1")
y_te_persist = X_te[:, aqi_lag1_idx]
p_mae  = mean_absolute_error(y_te_r, y_te_persist)
p_rmse = np.sqrt(mean_squared_error(y_te_r, y_te_persist))
p_r2   = r2_score(y_te_r, y_te_persist)
print(f"\nPersistence baseline — MAE:{p_mae:.2f} RMSE:{p_rmse:.2f} R^2:{p_r2:.4f}")
RESULTS.update({"persist_mae":p_mae,"persist_rmse":p_rmse,"persist_r2":p_r2})

# ── 9d. HistGBM baseline ────────────────────────────────────────────────────
print("\nTraining HistGBM Regressor …")
gbm_r = HistGradientBoostingRegressor(max_iter=300, max_depth=6, random_state=SEED,
                                       early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=20, scoring="neg_mean_absolute_error")
gbm_r.fit(X_tr_s, y_tr_r)
gbm_r_pred = gbm_r.predict(X_te_s)
gbm_mae  = mean_absolute_error(y_te_r, gbm_r_pred)
gbm_rmse = np.sqrt(mean_squared_error(y_te_r, gbm_r_pred))
gbm_r2   = r2_score(y_te_r, gbm_r_pred)
print(f"HistGBM Regressor  — MAE:{gbm_mae:.2f} RMSE:{gbm_rmse:.2f} R^2:{gbm_r2:.4f}")

print("Training HistGBM Classifier …")
gbm_c = HistGradientBoostingClassifier(max_iter=300, max_depth=6, random_state=SEED,
                                        early_stopping=True, validation_fraction=0.1,
                                        n_iter_no_change=20)
gbm_c.fit(X_tr_s, y_tr_c)
gbm_c_pred = gbm_c.predict(X_te_s)
gbm_c_prob = gbm_c.predict_proba(X_te_s)
gbm_acc = accuracy_score(y_te_c, gbm_c_pred)
gbm_f1  = f1_score(y_te_c, gbm_c_pred, average="macro", zero_division=0)
print(f"HistGBM Classifier — Acc:{gbm_acc:.4f} MacroF1:{gbm_f1:.4f}")
RESULTS.update({"gbm_mae":gbm_mae,"gbm_rmse":gbm_rmse,"gbm_r2":gbm_r2,
                "gbm_acc":gbm_acc,"gbm_f1":gbm_f1})

# Permutation importance on val set (Figure 23)
print("  Computing permutation importance on validation set …")
perm = permutation_importance(gbm_r, X_va_s, y_va_r,
                              n_repeats=5, random_state=SEED, n_jobs=-1)
pi_df = (pd.DataFrame({"Feature": FEAT_COLS, "Importance": perm.importances_mean})
         .sort_values("Importance", ascending=False).head(20))
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=pi_df, x="Importance", y="Feature",
            palette="viridis_r", ax=ax)
ax.set_title("Top-20 Permutation Importances — HistGBM Regressor (Val Set)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Mean Increase in MAE when Permuted")
ax.set_ylabel("Feature")
plt.tight_layout()
save_fig("23_permutation_importance.png")
RESULTS["top5_features"] = pi_df["Feature"].head(5).tolist()

# ── 9e. MLP Hyperparameter search (val set only) ────────────────────────────
print("\n-- 9e. MLP Hyperparameter Search --")

n_in = X_tr_s.shape[1]
class_cnts   = np.bincount(y_tr_c, minlength=NUM_CLASSES)
class_weight = {i: float(len(y_tr_c)/(NUM_CLASSES*c)) if c>0 else 1.0
                for i, c in enumerate(class_cnts)}
print(f"  Class weights: {class_weight}")
RESULTS["class_weights"] = class_weight

def build_reg(u1=256, u2=128, u3=64, d1=0.3, d2=0.2, lr=1e-3):
    inp = keras.Input(shape=(n_in,))
    x   = layers.Dense(u1, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(d1)(x)
    x   = layers.Dense(u2, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(d2)(x)
    x   = layers.Dense(u3, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    mdl = keras.Model(inp, out)
    mdl.compile(optimizer=keras.optimizers.Adam(lr), loss="huber", metrics=["mae"])
    return mdl

def build_cls(u1=256, u2=128, u3=64, d1=0.3, d2=0.2, lr=1e-3):
    inp = keras.Input(shape=(n_in,))
    x   = layers.Dense(u1, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(d1)(x)
    x   = layers.Dense(u2, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(d2)(x)
    x   = layers.Dense(u3, activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    mdl = keras.Model(inp, out)
    mdl.compile(optimizer=keras.optimizers.Adam(lr),
                loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return mdl

hp_grid = [
    dict(u1=256, u2=128, u3=64, d1=0.3, d2=0.2, lr=1e-3),
    dict(u1=256, u2=128, u3=64, d1=0.4, d2=0.2, lr=1e-3),
    dict(u1=128, u2=64,  u3=32, d1=0.3, d2=0.2, lr=1e-3),
    dict(u1=256, u2=128, u3=64, d1=0.3, d2=0.2, lr=5e-4),
]
cb_quick = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                   monitor="val_mae"),
    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_mae"),
]
hp_results = []
for hp in hp_grid:
    tf.random.set_seed(SEED)
    m = build_reg(**hp)
    h = m.fit(X_tr_s, y_tr_r, validation_data=(X_va_s, y_va_r),
              epochs=60, batch_size=512, verbose=0, callbacks=cb_quick)
    vmae = min(h.history["val_mae"])
    hp_results.append({**hp, "val_MAE": round(vmae,3)})
    print(f"  {hp} -> val_MAE={vmae:.2f}")

hp_df   = pd.DataFrame(hp_results).sort_values("val_MAE").reset_index(drop=True)
best_hp = hp_df.iloc[0].to_dict()
print(f"\nHP search results:\n{hp_df.to_string(index=False)}")
print(f"\nBest HP: u1={int(best_hp['u1'])}, u2={int(best_hp['u2'])}, "
      f"u3={int(best_hp['u3'])}, d1={best_hp['d1']}, "
      f"d2={best_hp['d2']}, lr={best_hp['lr']}")
RESULTS["hp_table"] = hp_df.to_dict(orient="records")
RESULTS["best_hp"]  = {k: (float(v) if isinstance(v,(int,float,np.integer,np.floating)) else v)
                        for k, v in best_hp.items()}

# ── 9f. Train best MLP Regressor (full 200 epochs with ES) ─────────────────
print("\n-- 9f. Train Best MLP Regressor --")
cb_full_r = [
    keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_mae"),
    keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, monitor="val_mae", min_lr=1e-6),
]
tf.random.set_seed(SEED)
mlp_r = build_reg(u1=int(best_hp["u1"]), u2=int(best_hp["u2"]), u3=int(best_hp["u3"]),
                   d1=float(best_hp["d1"]), d2=float(best_hp["d2"]), lr=float(best_hp["lr"]))
hist_r = mlp_r.fit(X_tr_s, y_tr_r, validation_data=(X_va_s, y_va_r),
                   epochs=200, batch_size=512, verbose=1, callbacks=cb_full_r)
print(f"  Stopped at epoch {len(hist_r.history['loss'])}")

# ── 9g. Train MLP Classifier ────────────────────────────────────────────────
print("\n-- 9g. Train MLP Classifier --")
cb_full_c = [
    keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor="val_accuracy"),
    keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, monitor="val_accuracy", min_lr=1e-6),
]
tf.random.set_seed(SEED)
mlp_c = build_cls(u1=int(best_hp["u1"]), u2=int(best_hp["u2"]), u3=int(best_hp["u3"]),
                   d1=float(best_hp["d1"]), d2=float(best_hp["d2"]), lr=float(best_hp["lr"]))
hist_c = mlp_c.fit(X_tr_s, y_tr_c, validation_data=(X_va_s, y_va_c),
                   epochs=200, batch_size=512, verbose=1, callbacks=cb_full_c,
                   class_weight=class_weight)
print(f"  Stopped at epoch {len(hist_c.history['loss'])}")

# ===========================================================================
# SECTION 10: EVALUATION (TEST SET — touched ONCE here)
# ===========================================================================
print("\n=== SECTION 10: EVALUATION ===")

mlp_r_pred = mlp_r.predict(X_te_s, verbose=0).flatten()
mlp_c_prob = mlp_c.predict(X_te_s, verbose=0)
mlp_c_pred = np.argmax(mlp_c_prob, axis=1)

# Regression metrics
mlp_mae  = mean_absolute_error(y_te_r, mlp_r_pred)
mlp_rmse = np.sqrt(mean_squared_error(y_te_r, mlp_r_pred))
mlp_r2   = r2_score(y_te_r, mlp_r_pred)
print(f"\nMLP Regressor Test  — MAE:{mlp_mae:.2f} RMSE:{mlp_rmse:.2f} R^2:{mlp_r2:.4f}")

# Classification metrics
mlp_acc = accuracy_score(y_te_c, mlp_c_pred)
mlp_f1  = f1_score(y_te_c, mlp_c_pred, average="macro", zero_division=0)
y_te_bin= label_binarize(y_te_c, classes=list(range(NUM_CLASSES)))
mlp_auc = roc_auc_score(y_te_bin, mlp_c_prob, average="macro", multi_class="ovr")
print(f"MLP Classifier Test — Acc:{mlp_acc:.4f} MacroF1:{mlp_f1:.4f} AUC:{mlp_auc:.4f}")
RESULTS.update({"mlp_mae":mlp_mae,"mlp_rmse":mlp_rmse,"mlp_r2":mlp_r2,
                "mlp_acc":mlp_acc,"mlp_f1":mlp_f1,"mlp_auc":mlp_auc})

# ── Eval Fig 24: Training curves ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, hist, key, lbl, color_tr, color_va in [
    (axes[0,0], hist_r, ("loss","val_loss"),   "Huber Loss",     "#2E86AB","#E9C46A"),
    (axes[0,1], hist_r, ("mae","val_mae"),     "MAE",            "#2E86AB","#E9C46A"),
    (axes[1,0], hist_c, ("loss","val_loss"),   "CrossEntropy",   "#264653","#E76F51"),
    (axes[1,1], hist_c, ("accuracy","val_accuracy"),"Accuracy",  "#264653","#E76F51"),
]:
    tr_k, va_k = key
    ax.plot(hist.history[tr_k], color=color_tr, lw=1.8, label="Train")
    ax.plot(hist.history[va_k], color=color_va, lw=1.8, label="Val")
    model_name = "MLP Regressor" if hist is hist_r else "MLP Classifier"
    ax.set_title(f"{model_name} — {lbl}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel(lbl); ax.legend()
plt.suptitle("Training History — MLP Regressor and Classifier",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("24_training_curves.png")

# ── Eval Fig 25: Confusion matrices ─────────────────────────────────────────
cm_raw  = confusion_matrix(y_te_c, mlp_c_pred)
cm_norm = confusion_matrix(y_te_c, mlp_c_pred, normalize="true")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, cm, title in [(axes[0], cm_raw, "Raw Counts"),
                       (axes[1], np.round(cm_norm,2), "Normalized (row)")]:
    ConfusionMatrixDisplay(cm, display_labels=BUCKET_ORDER).plot(
        ax=ax, colorbar=False, cmap="Blues", xticks_rotation=30)
    ax.set_title(f"Confusion Matrix ({title})", fontsize=12, fontweight="bold")
plt.suptitle("MLP Classifier Confusion Matrices — Test Set",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("25_confusion_matrix.png")

# ── Eval Fig 26: One-vs-Rest ROC curves ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
macro_interp_tprs = []
macro_fpr_base    = np.linspace(0, 1, 200)
for i, bucket in enumerate(BUCKET_ORDER):
    if i < y_te_bin.shape[1]:
        fpr, tpr, _ = roc_curve(y_te_bin[:, i], mlp_c_prob[:, i])
        auc_i = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CB6[i % len(CB6)], lw=2,
                label=f"{bucket} (AUC={auc_i:.3f})")
        macro_interp_tprs.append(np.interp(macro_fpr_base, fpr, tpr))
macro_tpr  = np.mean(macro_interp_tprs, axis=0)
macro_auc_ = sk_auc(macro_fpr_base, macro_tpr)
ax.plot(macro_fpr_base, macro_tpr, "k--", lw=2.5,
        label=f"Macro-avg (AUC={macro_auc_:.3f})")
ax.plot([0,1],[0,1], "gray", ls=":", lw=1)
ax.set_title("One-vs-Rest ROC Curves — MLP Classifier (Test Set)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout()
save_fig("26_roc_curves.png")

# ── Eval Fig 27: Residual analysis ──────────────────────────────────────────
resid = y_te_r - mlp_r_pred
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.histplot(resid, bins=60, kde=True, color="#2E86AB", ax=axes[0])
axes[0].axvline(0, color="red", ls="--", lw=2)
axes[0].set_title(f"Residual Distribution\nMean={resid.mean():.2f}, Std={resid.std():.2f}",
                  fontsize=12, fontweight="bold")
axes[0].set_xlabel("Actual − Predicted AQI"); axes[0].set_ylabel("Count")

axes[1].scatter(mlp_r_pred, resid, s=5, alpha=0.25, color="#2E86AB")
axes[1].axhline(0, color="red", ls="--", lw=2)
axes[1].set_title("Residuals vs Predicted AQI", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted AQI"); axes[1].set_ylabel("Residual")
plt.suptitle("Residual Analysis — MLP Regressor (Test Set)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("27_residual_analysis.png")
RESULTS["resid_mean"] = float(resid.mean()); RESULTS["resid_std"] = float(resid.std())

# ── Eval Fig 28: Predicted vs Actual scatter ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_te_r, mlp_r_pred, s=5, alpha=0.25, color="#2E86AB")
mn_ = min(y_te_r.min(), mlp_r_pred.min())
mx_ = max(y_te_r.max(), mlp_r_pred.max())
ax.plot([mn_,mx_],[mn_,mx_], "r--", lw=2, label="Perfect prediction")
ax.set_title(f"Predicted vs Actual AQI\nMAE={mlp_mae:.2f}  RMSE={mlp_rmse:.2f}  R^2={mlp_r2:.4f}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Actual Next-Day AQI"); ax.set_ylabel("Predicted Next-Day AQI")
ax.legend()
plt.tight_layout()
save_fig("28_predicted_vs_actual.png")

# ── Eval Fig 29: Time-series overlay for 3 cities ───────────────────────────
test_df = df_fe.loc[te_mask].copy()
test_df["pred_AQI"] = mlp_r_pred
# Recover city name from one-hot
test_df["City_name"] = test_df[city_cols].idxmax(axis=1).str.replace("city_","")
top3_cities = test_df["City_name"].value_counts().head(3).index.tolist()

fig, axes = plt.subplots(3, 1, figsize=(16, 13))
for ax, city in zip(axes, top3_cities):
    cd = test_df[test_df["City_name"]==city].sort_values("Date")
    ax.plot(cd["Date"], cd["target_AQI"], label="Actual", color="#2E86AB", lw=1.5)
    ax.plot(cd["Date"], cd["pred_AQI"], label="Predicted", color="#E9C46A",
            lw=1.5, ls="--")
    ax.set_title(f"{city} — Actual vs Predicted Next-Day AQI",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("AQI"); ax.legend(fontsize=9)
plt.suptitle("Time-Series: Actual vs Predicted AQI — 3 Cities (Test Set)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("29_timeseries_overlay.png")
RESULTS["overlay_cities"] = top3_cities

# ── Eval Fig 30: Model comparison bar chart ──────────────────────────────────
model_names = ["Persistence","HistGBM","MLP"]
maes  = [p_mae,  gbm_mae,  mlp_mae]
rmses = [p_rmse, gbm_rmse, mlp_rmse]
r2s   = [p_r2,   gbm_r2,   mlp_r2]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, vals, metric, higher_better in [
    (axes[0], maes,  "MAE ((down))",  False),
    (axes[1], rmses, "RMSE ((down))", False),
    (axes[2], r2s,   "R^2 ((up))",   True),
]:
    bars = ax.bar(model_names, vals, color=CB6[:3], edgecolor="black", lw=0.5)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + 0.01*max(vals),
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
plt.suptitle("Model Comparison — Regression Metrics (Test Set)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig("30_model_comparison.png")

# Final summary
print("\n" + "="*70)
print("FINAL TEST-SET RESULTS")
print("="*70)
print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'R^2':>8}")
for name, mae, rmse, r2_ in zip(model_names, maes, rmses, r2s):
    print(f"{name:<15} {mae:>8.2f} {rmse:>8.2f} {r2_:>8.4f}")
print(f"\nMLP Classifier: Acc={mlp_acc:.4f}  MacroF1={mlp_f1:.4f}  AUC={mlp_auc:.4f}")
print(f"GBM Classifier: Acc={gbm_acc:.4f}  MacroF1={gbm_f1:.4f}")

# ===========================================================================
# SECTION 11: SAVE RESULTS JSON
# ===========================================================================
results_path = os.path.join(ROOT, "results.json")
with open(results_path, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\nOK results.json saved -> {results_path}")

print("\n" + "="*70)
print("ALL PHASES COMPLETE — see figures/ for all 30 outputs + 2 HTML files")
print("="*70)
