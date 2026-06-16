#!/usr/bin/env python3
"""
Proper permutation test for Combined eGeMAPS+RQA GBT model.
Re-runs full 5-fold stratified CV for each of 1000 permutations.
Pre-registered hyperparameters (OSF osf.io/bsvrj).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import json, warnings, time
warnings.filterwarnings("ignore")

DATA_PATH = "/home/lily/work/peter/rqa_analysis/merged_features_egemaps_rqa.csv"
OUT_PATH = "/home/lily/work/peter/rqa_analysis/combined_gbt_permutation_test_proper.json"
N_PERMS = 1000

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['anhedonia_binary'])

exclude = ['subject_id', 'anhedonia_binary', 'anhedonia_sum',
           'PHQ8_Binary', 'PHQ8_Score', 'Gender', 'split', 'merge_id']
feat_cols = [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]
df_feat = df[feat_cols].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

X = df_feat.values
y = df['anhedonia_binary'].values.astype(int)

var = np.nanvar(X, axis=0)
mask = var > 0
X = X[:, mask]
for j in range(X.shape[1]):
    col = X[:, j]
    nan_mask = np.isnan(col)
    if nan_mask.any():
        X[nan_mask, j] = np.nanmedian(col)

print(f"Data: n={len(y)}, features={X.shape[1]}, pos={y.sum()}")

clf_params = dict(n_estimators=300, learning_rate=0.1, random_state=42)

def run_cv(X, y_labels, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_aucs = []
    for train_idx, test_idx in skf.split(X, y_labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model = GradientBoostingClassifier(**clf_params)
        model.fit(X_tr, y_labels[train_idx])
        probs = model.predict_proba(X_te)[:, 1]
        if len(np.unique(y_labels[test_idx])) == 2:
            fold_aucs.append(roc_auc_score(y_labels[test_idx], probs))
    return np.mean(fold_aucs) if fold_aucs else 0.5

t0 = time.time()
true_auc = run_cv(X, y)
t_one = time.time() - t0
print(f"True mean fold AUC: {true_auc:.4f} ({t_one:.1f}s per CV)")
print(f"Estimated time: {(t_one * N_PERMS) / 60:.0f} min")

rng = np.random.RandomState(42)
null_aucs = []
t_start = time.time()

for i in range(N_PERMS):
    y_perm = rng.permutation(y)
    null_aucs.append(run_cv(X, y_perm, seed=42))
    if (i+1) % 50 == 0:
        elapsed = time.time() - t_start
        rate = (i+1) / elapsed
        remaining = (N_PERMS - i - 1) / rate
        print(f"  [{i+1}/{N_PERMS}] elapsed={elapsed:.0f}s, remaining={remaining:.0f}s, null_mean={np.mean(null_aucs):.4f}")

null_aucs = np.array(null_aucs)
p_value = (np.sum(null_aucs >= true_auc) + 1) / (N_PERMS + 1)
total_time = time.time() - t_start

print(f"\n{'='*60}")
print(f"PROPER PERMUTATION TEST (re-run full 5-fold CV)")
print(f"{'='*60}")
print(f"True mean fold AUC: {true_auc:.4f}")
print(f"Null mean: {np.mean(null_aucs):.4f} +/- {np.std(null_aucs):.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant (alpha=0.05): {p_value < 0.05}")
print(f"Runtime: {total_time/60:.1f} min")

result = {
    "test_type": "proper_permutation_rerun_cv",
    "model": "GradientBoostedTrees",
    "feature_set": "Combined_eGeMAPSv02_RQA",
    "n_features": int(X.shape[1]),
    "n_subjects": int(len(y)),
    "cv_folds": 5,
    "true_mean_fold_auc": round(float(true_auc), 4),
    "permutation_test": {
        "n_permutations": N_PERMS,
        "true_auc": round(float(true_auc), 4),
        "null_mean": round(float(np.mean(null_aucs)), 4),
        "null_std": round(float(np.std(null_aucs)), 4),
        "p_value": round(float(p_value), 6),
        "significant_alpha_05": bool(p_value < 0.05)
    },
    "runtime_seconds": round(total_time, 1)
}

with open(OUT_PATH, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
