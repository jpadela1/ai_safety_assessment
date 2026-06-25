"""
Diabetes 130 label-noise dose-response  —  TABULAR safety arm (second modality).

CPU only, no GPU, no Colab. Run in your PyCharm venv:
    pip install ucimlrepo scikit-learn xgboost pandas numpy
    python diabetes_label_noise_arm.py

Mirrors the text arm: inject a known dose of LABEL NOISE into the training set,
score the data with an automated label-integrity proxy, train downstream models,
and measure degradation on a clean test set. Writes results/rubric_scores/diabetes_outcome.csv.

Safety framing: a mislabeled clinical record is an integrity failure that degrades a
consequential decision model — the tabular face of "untrustworthy data -> unsafe model."
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import xgboost as xgb

# ---- config (mirrors the text arm: same dose levels, 5 seeds) ----
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.40]
SEEDS        = list(range(5))
SAMPLE_N     = 20000        # subsample for speed; full set works but is slower
OUT = Path("results/rubric_scores"); OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT / "diabetes_outcome.csv"

# ---- load Diabetes 130 ----
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=296)
X = ds.data.features.copy()
y_raw = ds.data.targets.iloc[:, 0].astype(str)
y = (y_raw == "<30").astype(int).values          # early (<30d) readmission = the consequential event

# ---- preprocess ----
X = X.replace("?", np.nan)
X = X.drop(columns=[c for c in ["weight", "payer_code", "medical_specialty"] if c in X.columns])
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      # if your sklearn < 1.2, change sparse_output=False to sparse=False
                      ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=20, sparse_output=False))]), cat_cols),
])
Xp = pre.fit_transform(X).astype(np.float32)
if len(Xp) > SAMPLE_N:
    idx = np.random.default_rng(0).choice(len(Xp), SAMPLE_N, replace=False)
    Xp, y = Xp[idx], y[idx]
print(f"design matrix: {Xp.shape} | positive (early-readmit) rate: {y.mean():.3f}")

# ---- automated label-integrity proxy (confident-learning style) ----
def integrity_proxy(Xtr, ytr_noisy, seed):
    """Average disagreement between a cross-validated model's belief and the assigned
    label. Rises monotonically with injected noise; model-agnostic, swappable."""
    proba = cross_val_predict(LogisticRegression(max_iter=2000), Xtr, ytr_noisy,
                              method="predict_proba",
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    p_given = proba[np.arange(len(ytr_noisy)), ytr_noisy]
    return float(1.0 - p_given.mean())

def ece(probs, labels, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1])
        if m.sum() > 0:
            e += abs(probs[m].mean() - labels[m].mean()) * m.mean()
    return float(e)

def flip(y, p, rng):
    y2 = y.copy(); m = rng.random(len(y)) < p; y2[m] = 1 - y2[m]; return y2

# ---- run the dose-response ----
rows = []
for p in NOISE_LEVELS:
    for s in SEEDS:
        rng = np.random.default_rng(s)
        Xtr, Xte, ytr, yte = train_test_split(Xp, y, test_size=0.25, random_state=s, stratify=y)
        ytr_noisy = flip(ytr, p, rng)                  # flip TRAIN labels only; test stays clean
        proxy = integrity_proxy(Xtr, ytr_noisy, s)     # score the (noisy) data
        for name, model in [("logreg", LogisticRegression(max_iter=2000)),
                            ("xgboost", xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                          learning_rate=0.1, eval_metric="logloss", verbosity=0))]:
            model.fit(Xtr, ytr_noisy)
            pr = model.predict_proba(Xte)[:, 1]
            rows.append(dict(injected_p=p, seed=s, model=name, proxy_noise=proxy,
                             auc=roc_auc_score(yte, pr),
                             bal_acc=balanced_accuracy_score(yte, (pr >= 0.5).astype(int)),
                             ece=ece(pr, yte)))
    print(f"  done dose p={p}")
df = pd.DataFrame(rows); df.to_csv(OUT_CSV, index=False)
print("wrote", OUT_CSV, df.shape)

# ---- immediate analysis (so you see if it worked) ----
def boot_r(x, y, B=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(x); x = np.asarray(x); y = np.asarray(y); rs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if np.std(x[i]) == 0 or np.std(y[i]) == 0: continue
        rs.append(np.corrcoef(x[i], y[i])[0, 1])
    return np.percentile(rs, [2.5, 97.5])

print("\nPER-DOSE MEANS")
print(df.groupby("injected_p")[["proxy_noise", "auc", "bal_acc", "ece"]].mean().round(4).to_string())
print("\nH1 calibration  proxy_noise vs injected_p:")
r = np.corrcoef(df.injected_p, df.proxy_noise)[0, 1]; ci = boot_r(df.injected_p, df.proxy_noise)
print(f"  r={r:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}]")
print("H2 predictive  proxy_noise vs AUC (expect NEGATIVE):")
r = np.corrcoef(df.proxy_noise, df.auc)[0, 1]; ci = boot_r(df.proxy_noise, df.auc)
print(f"  r={r:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}]  {'** CI excludes 0 **' if ci[1] < 0 else 'crosses 0'}")
print("H2 predictive  proxy_noise vs ECE (expect POSITIVE):")
r = np.corrcoef(df.proxy_noise, df.ece)[0, 1]; ci = boot_r(df.proxy_noise, df.ece)
print(f"  r={r:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}]  {'** CI excludes 0 **' if ci[0] > 0 else 'crosses 0'}")
