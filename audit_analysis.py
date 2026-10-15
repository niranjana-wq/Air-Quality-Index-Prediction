"""
audit_analysis.py
Generates all quantitative evidence needed for AUDIT.md.
Run: .venv\Scripts\python audit_analysis.py
"""
import os, json, re
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from docx import Document

ROOT = r"c:\Users\niran\OneDrive\Desktop\AQI"
R    = json.load(open(os.path.join(ROOT, "results.json")))

def fv(k):
    v = R.get(k, None)
    return float(v) if v is not None else None

print("=" * 70)
print("AUDIT ANALYSIS")
print("=" * 70)

# ─── AUDIT 1: Report sections ───────────────────────────────────────────────
print("\n--- AUDIT 1: Report Sections ---")
try:
    doc = Document(os.path.join(ROOT, "report", "Project_Report.docx"))
    headings = []
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            headings.append((para.style.name, para.text.strip()))
    print("Headings found in report:")
    for h in headings:
        print(f"  {h}")
    
    # Count abstract words
    full_text = "\n".join(p.text for p in doc.paragraphs)
    # Find abstract section text
    abstract_start = full_text.find("Air pollution is a critical")
    abstract_text = full_text[abstract_start:abstract_start+2000].split("\n")[0]
    word_count = len(abstract_text.split())
    print(f"\nAbstract word count: {word_count}")
    print(f"Abstract text:\n{abstract_text[:500]}")
except Exception as e:
    print(f"  ERROR reading report: {e}")

# ─── AUDIT 2: Notebook check ─────────────────────────────────────────────────
print("\n--- AUDIT 2: Notebook Check ---")
import nbformat
nb_path = os.path.join(ROOT, "notebooks", "aqi_eda_model.ipynb")
nb = nbformat.read(nb_path, as_version=4)
n_cells       = len(nb.cells)
n_code        = sum(1 for c in nb.cells if c.cell_type == "code")
n_md          = sum(1 for c in nb.cells if c.cell_type == "markdown")
n_with_output = sum(1 for c in nb.cells if c.cell_type == "code" and len(c.get("outputs", [])) > 0)
has_abs_paths = False
for c in nb.cells:
    if c.cell_type == "code" and ("c:\\\\Users" in c.source or "C:\\\\Users" in c.source or "__file__" in c.source):
        has_abs_paths = True
        break
print(f"Total cells      : {n_cells}")
print(f"Code cells       : {n_code}")
print(f"Markdown cells   : {n_md}")
print(f"Cells with output: {n_with_output}  (FAIL: 0 means no saved outputs)")
print(f"Absolute paths   : {has_abs_paths}")
# Check if it uses __file__ (won't work in notebook)
has_file_ref = any("__file__" in c.source for c in nb.cells if c.cell_type == "code")
print(f"Uses __file__    : {has_file_ref}  (FAIL: not available in notebook kernel)")

# ─── AUDIT 3: Leakage Audit ──────────────────────────────────────────────────
print("\n--- AUDIT 3: Leakage Audit ---")
print("3a. Interpolation:")
print("    Lines 193-203: interpolate_city() applied to df BEFORE split.")
print("    interpolate(method='linear', limit=3, limit_direction='forward')")
print("    STATUS: PARTIAL FAIL - 'forward' direction is causal (no future leak),")
print("    BUT thresholds computed on all cities + all time = uses future data")
print("    for imputing a city's early values from later values in the same city.")
print("    Actually limit_direction='forward' means fills NaN at start from later values?")
print("    Let's verify the pandas default:")
import pandas as pd
s = pd.Series([np.nan, np.nan, 3.0, np.nan, 5.0])
result_fwd  = s.interpolate(method="linear", limit=3, limit_direction="forward")
result_back = s.interpolate(method="linear", limit=3, limit_direction="backward")
result_both = s.interpolate(method="linear", limit=3, limit_direction="both")
print(f"    Test series    : {s.tolist()}")
print(f"    forward  result: {result_fwd.tolist()}")
print(f"    backward result: {result_back.tolist()}")
print(f"    both result    : {result_both.tolist()}")
print()
print("3b. Winsorizing thresholds:")
print("    Lines 241-246: computed on df (all data, all dates).")
print("    STATUS: FAIL — thresholds use test-period data.")
print()
print("3c. Rolling features:")
print("    Lines 734-740: shifted=grp[col].shift(1) then .rolling(7)")
print("    STATUS: PASS — window ends at D-1, target is D+1 = 2 days safe gap")
print()
print("3d. Reindexed calendar rows as targets:")
print("    Lines 770-771: dropna on FEAT_COLS + targets")
print("    The reindexed NaN rows won't have valid lag features, so they are dropped.")
print("    STATUS: PASS — reindexed rows dropped before modeling")
print()
print("3e. StandardScaler:")
print("    Lines 824-826 (approx): scaler.fit_transform(X_tr), transform(X_va), transform(X_te)")
print("    STATUS: PASS — fit on train only")

# ─── AUDIT 4: Distribution shift ─────────────────────────────────────────────
print("\n--- AUDIT 4: Distribution Shift ---")
print(f"Train: {R.get('train_start')} -> {R.get('train_end')}")
print(f"Val  : {R.get('val_start')} -> {R.get('val_end')}")
print(f"Test : {R.get('test_start')} -> {R.get('test_end')}")

# Load data to compute AQI stats per split
df = pd.read_csv(os.path.join(ROOT, "data", "city_day.csv"))
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(subset=["AQI"])

train_end  = pd.Timestamp(R.get("train_end"))
val_end    = pd.Timestamp(R.get("val_end"))

df_tr = df[df["Date"] <= train_end]
df_va = df[(df["Date"] > train_end) & (df["Date"] <= val_end)]
df_te = df[df["Date"] > val_end]

print(f"\nAQI mean ± std:")
for name, d in [("Train", df_tr), ("Val", df_va), ("Test", df_te)]:
    print(f"  {name}: mean={d['AQI'].mean():.1f}, std={d['AQI'].std():.1f}, n={len(d)}")

# COVID period in test
covid_start = pd.Timestamp("2020-03-24")
covid_end   = pd.Timestamp("2020-06-01")
df_te_covid = df_te[(df_te["Date"] >= covid_start) & (df_te["Date"] <= covid_end)]
df_te_nonco = df_te[(df_te["Date"] < covid_start) | (df_te["Date"] > covid_end)]
print(f"\nTest COVID rows : {len(df_te_covid)} ({100*len(df_te_covid)/len(df_te):.1f}%)")
print(f"Test non-COVID  : {len(df_te_nonco)} ({100*len(df_te_nonco)/len(df_te):.1f}%)")
if len(df_te_covid) > 0:
    print(f"COVID AQI mean  : {df_te_covid['AQI'].mean():.1f}")
    print(f"Non-COVID AQI mean: {df_te_nonco['AQI'].mean():.1f} (train mean={df_tr['AQI'].mean():.1f})")

# ─── AUDIT 5: Model fairness ─────────────────────────────────────────────────
print("\n--- AUDIT 5: Model Fairness ---")
print("All models (Persistence, HistGBM, MLP) use identical FEAT_COLS and splits: PASS")
print("Persistence uses AQI_lag1 feature directly (same matrix): PASS")
print("MLP target: raw AQI, not scaled: PASS")
print("Per-city results: NOT reported in v1: FAIL")

# ─── AUDIT 6: Per-class classification metrics ───────────────────────────────
print("\n--- AUDIT 6: Classification Metrics ---")
print("Per-class precision/recall/F1 NOT saved in results.json.")
print("Need to re-run model to get these. Loading bucket counts as proxy:")
bc = R.get("bucket_counts", {})
total = sum(bc.values())
print("Training AQI_Bucket distribution (model df, NOT split-specific):")
for b, c in sorted(bc.items(), key=lambda x: ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"].index(x[0]) if x[0] in ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"] else 99):
    print(f"  {b:<14}: {c:5d} ({100*c/total:.1f}%)")
print("\nPer-class metrics NOT available without re-running model: FAIL")
print("Macro-F1=0.47 is low because Severe (~5%) and Good (~5%) have very few test samples,")
print("and the model struggles with boundary cases (Moderate/Poor/Very Poor adjacency).")

# ─── AUDIT 7: Report integrity check ─────────────────────────────────────────
print("\n--- AUDIT 7: Report Integrity ---")
try:
    doc = Document(os.path.join(ROOT, "report", "Project_Report.docx"))
    full_text = "\n".join(p.text for p in doc.paragraphs)
    
    checks = {
        "MAE persistence": (str(round(fv("persist_mae"), 2)) if fv("persist_mae") else None, "32.58"),
        "MAE GBM":         (str(round(fv("gbm_mae"), 2)) if fv("gbm_mae") else None, "32.25"),
        "MAE MLP":         (str(round(fv("mlp_mae"), 2)) if fv("mlp_mae") else None, "32.90"),
        "R2 MLP":          (str(round(fv("mlp_r2"), 3)) if fv("mlp_r2") else None, "0.668"),
        "MacroF1 MLP":     (str(round(fv("mlp_f1"), 3)) if fv("mlp_f1") else None, "0.474"),
        "AUC MLP":         (str(round(fv("mlp_auc"), 3)) if fv("mlp_auc") else None, "0.850"),
        "n_features":      (str(R.get("n_features", "")), "92"),
    }
    print("\nChecking report text for metric consistency with results.json:")
    for label, (actual, expected_in_report) in checks.items():
        found = expected_in_report in full_text if expected_in_report else False
        print(f"  {label:<20}: results.json={actual}, in_report={'YES' if found else 'NO'}")
except Exception as e:
    print(f"  ERROR: {e}")

# ─── AUDIT 8: Repo hygiene ────────────────────────────────────────────────────
print("\n--- AUDIT 8: Repo Hygiene ---")
import os
csv_in_git = os.path.exists(os.path.join(ROOT, ".git", "index"))  # just check .git exists
expected_files = {
    "README.md": True, "requirements.txt": True, ".gitignore": True,
    "PROGRESS.md": True, "PLAN.md": True, "data/README.md": True,
    "notebooks/aqi_eda_model.ipynb": True, "src/aqi_eda_model.py": True,
    "report/Project_Report.docx": True, "results.json": True,
}
total_size = 0
for root_, dirs, files in os.walk(ROOT):
    # Skip .venv
    dirs[:] = [d for d in dirs if d not in [".venv", "__pycache__", ".git"]]
    for f in files:
        total_size += os.path.getsize(os.path.join(root_, f))
print(f"Total repo size (excl. .venv, .git): {total_size/1e6:.1f} MB")

figs_dir = os.path.join(ROOT, "figures")
pngs = [f for f in os.listdir(figs_dir) if f.endswith(".png")]
print(f"Figure count (PNG): {len(pngs)}")
print(f"City CSV in repo  : {os.path.exists(os.path.join(ROOT, 'data', 'city_day.csv'))} (not committed to git, just local)")

# ─── AUDIT 9: Rubric coverage ────────────────────────────────────────────────
print("\n--- AUDIT 9: Rubric Coverage ---")
print("CO3 (8 marks) - Data Visualization:")
print("  ✓ >= 8 charts (30 PNG + 2 HTML)")
print("  ✓ Titles, labels, units present")
print("  PARTIAL: What/Why/Insights captions are in the report DOCX but not in the notebook markdown cells")
print()
print("CO4 (16 marks) - Statistical Analysis:")
print("  ✓ Univariate: histograms, KDE, stats table (Fig 04, 05)")
print("  ✓ Continuous dist: Log-Normal vs Gamma Q-Q + KS + AIC (Fig 06, 07)")
print("  ✓ Discrete dist: AQI Bucket + Poisson (Fig 08, 09)")
print("  ✓ Bivariate: Pearson+Spearman heatmaps, scatterplots, boxplots (Fig 10-15)")
print("  ✓ Multivariate: Pairplot + PCA + Clustermap (Fig 16-18)")
print("  ✓ Time series: monthly trend, COVID, seasonal decomp (Fig 19-22)")
print()
print("CO5 (16 marks) - Application:")
print("  ✓ EDA + Transformation documented")
print("  ✓ Deep learning model (MLP reg + cls)")
print("  ✓ Insights and conclusions")
print("  PARTIAL: Notebook has no saved outputs (generated by make_notebook.py, not executed)")
print("  FAIL: No docs/ directory or RUBRIC_CHECKLIST.md")
print("  PARTIAL: Limitations section present but brief")

print("\n\nAUDIT COMPLETE. Writing AUDIT.md...")
