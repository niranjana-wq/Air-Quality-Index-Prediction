import os, json

ROOT = r"c:\Users\niran\OneDrive\Desktop\AQI"

required = [
    "README.md", "requirements.txt", ".gitignore", "PROGRESS.md", "PLAN.md",
    "data/README.md", "data/city_day.csv",
    "notebooks/aqi_eda_model.ipynb",
    "src/aqi_eda_model.py",
    "report/Project_Report.docx",
    "results.json",
]

print("=== FILE CHECKLIST ===")
all_ok = True
for f in required:
    path = os.path.join(ROOT, f.replace("/", os.sep))
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "OK" if exists else "MISSING"
    if not exists:
        all_ok = False
    print("  {:7s}  {}  ({:,} bytes)".format(status, f, size))

figs_dir = os.path.join(ROOT, "figures")
pngs  = sorted([f for f in os.listdir(figs_dir) if f.endswith(".png")])
htmls = sorted([f for f in os.listdir(figs_dir) if f.endswith(".html")])
print("\n=== FIGURES ===")
print("  PNG files : {}".format(len(pngs)))
print("  HTML files: {}".format(len(htmls)))
for f in pngs:
    print("    " + f)
for f in htmls:
    print("    " + f)

print("\n=== KEY METRICS (results.json) ===")
R = json.load(open(os.path.join(ROOT, "results.json")))
def fv(k):
    v = R.get(k, "N/A")
    return float(v) if v != "N/A" else "N/A"

print("  n_features : {}".format(R.get("n_features", "N/A")))
print("  Persist MAE: {:.2f}".format(fv("persist_mae")))
print("  GBM MAE    : {:.2f}".format(fv("gbm_mae")))
print("  MLP MAE    : {:.2f}".format(fv("mlp_mae")))
print("  MLP R2     : {:.4f}".format(fv("mlp_r2")))
print("  MLP MacroF1: {:.4f}".format(fv("mlp_f1")))
print("  MLP AUC    : {:.4f}".format(fv("mlp_auc")))
print("\n  Overall: {}".format("ALL FILES OK" if all_ok else "MISSING FILES - see above"))
