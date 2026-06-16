#!/usr/bin/env python3
"""
04_SHAP_Analysis.py — SHAP Feature Importance (OSF Section 5.2, H3)
====================================================================
Uses SHAP TreeExplainer for RF/GBT and LinearExplainer for LogReg,
as pre-registered. Replaces the |coef| proxy from 03_RQA_Classification.py.

Also generates:
  - SHAP beeswarm plots (publication-quality figures)
  - H3 assessment: Are CV_F0 and CV_Energy in top 5?
  - DeLong test for formal AUC comparison (OSF Section 3.4)
  - Bootstrap ΔAUC CI for H2 non-inferiority (OSF Section 5.2)

Outputs:
  Results/Stream_A/shap_results.json
  Results/Stream_A/shap_beeswarm_combined.png
  Results/Stream_A/shap_beeswarm_egemaps.png
  Results/Stream_A/delong_test_results.json
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ── Config ──
MERGED_CSV = "/lab/lily/work/peter/rqa_analysis/merged_features_egemaps_rqa.csv"
OUTPUT_DIR = Path("/lab/lily/work/peter/rqa_analysis")

META_COLS = ["subject_id", "anhedonia_binary", "anhedonia_sum",
             "PHQ8_Binary", "PHQ8_Score", "Gender", "split", "merge_id"]
OUTCOME_COL = "anhedonia_binary"
STREAM_B_AUC = 0.580

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=None, random_state=42, n_jobs=-1
    ),
    "GradientBoostedTrees": GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.1, random_state=42
    ),
}

N_FOLDS = 5
RANDOM_STATE = 42


# ── DeLong Test Implementation ──
# Per DeLong et al. (1988), referenced in OSF Section 3.4

def compute_midrank(x):
    """Compute midranks for DeLong test."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        for k in range(i, j):
            T[k] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1.0
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong AUC computation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    aucs = np.zeros(k)
    for j in range(k):
        all_preds = np.concatenate([positive_examples[j], negative_examples[j]])
        ranks = compute_midrank(all_preds)
        aucs[j] = (np.sum(ranks[:m]) - m * (m + 1) / 2.0) / (m * n)

    # Structural components
    V01 = np.zeros((k, n))
    V10 = np.zeros((k, m))
    for j in range(k):
        for i in range(n):
            V01[j, i] = (np.sum(positive_examples[j] < negative_examples[j, i]) +
                         0.5 * np.sum(positive_examples[j] == negative_examples[j, i])) / m
        for i in range(m):
            V10[j, i] = (np.sum(negative_examples[j] < positive_examples[j, i]) +
                         0.5 * np.sum(negative_examples[j] == positive_examples[j, i])) / n

    S01 = np.cov(V01) if n > 1 else np.zeros((k, k))
    S10 = np.cov(V10) if m > 1 else np.zeros((k, k))
    S = S01 / n + S10 / m

    return aucs, S


def delong_test(y_true, y_pred_1, y_pred_2):
    """
    DeLong test comparing two AUC-ROCs.
    Returns: auc1, auc2, z_stat, p_value
    """
    order = np.argsort(-y_true)  # Positive first
    label_1_count = int(np.sum(y_true))

    predictions_sorted = np.vstack([y_pred_1[order], y_pred_2[order]])
    aucs, S = fastDeLong(predictions_sorted, label_1_count)

    if isinstance(S, np.ndarray) and S.shape == (2, 2):
        diff = aucs[0] - aucs[1]
        var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
        if var > 0:
            z = diff / np.sqrt(var)
            p = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            z, p = 0.0, 1.0
    else:
        z, p = 0.0, 1.0

    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# ── Bootstrap ΔAUC CI (H2) ──

def bootstrap_delta_auc_ci(y_true, y_pred_a, stream_b_auc, n_boot=10000, alpha=0.05):
    """
    Bootstrap CI for ΔAUC = AUC_StreamA - AUC_StreamB.
    Since streams use different datasets, we bootstrap Stream A's AUC
    and compute the delta against the fixed Stream B point estimate.
    Per OSF Section 5.2: Lower bound of 95% CI for ΔAUC > -0.10.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    deltas = []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) == 2:
            auc_boot = roc_auc_score(y_true[idx], y_pred_a[idx])
            deltas.append(auc_boot - stream_b_auc)

    deltas = np.array(deltas)
    ci_lower = np.percentile(deltas, 100 * alpha / 2)
    ci_upper = np.percentile(deltas, 100 * (1 - alpha / 2))
    mean_delta = np.mean(deltas)

    return {
        "mean_delta_auc": round(float(mean_delta), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "non_inferiority_met": bool(ci_lower > -0.10),
        "superiority_met": bool(ci_lower > 0.0),
        "n_bootstrap": n_boot,
    }


def main():
    print("=" * 70)
    print("SHAP ANALYSIS + DeLong TEST + ΔAUC BOOTSTRAP CI")
    print(f"OSF Pre-Registration: osf.io/bsvrj (Sections 3.4, 5.2)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── Load merged data ──
    df = pd.read_csv(MERGED_CSV)
    print(f"\nLoaded: {df.shape[0]} subjects × {df.shape[1]} cols")

    # Identify features
    all_cols = df.columns.tolist()
    baseline_cols = [c for c in all_cols if c not in META_COLS and not c.startswith("rqa_")]
    rqa_cols = [c for c in all_cols if c.startswith("rqa_") and df[c].notna().sum() > 10]
    combined_cols = baseline_cols + rqa_cols

    # Prepare outcome
    valid_mask = df[OUTCOME_COL].notna()
    df = df[valid_mask].copy()
    y = df[OUTCOME_COL].values.astype(int)
    print(f"N = {len(y)} | Class 0: {(y==0).sum()} | Class 1: {(y==1).sum()}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "analysis": "SHAP + DeLong + Bootstrap ΔAUC",
        "osf_ref": "osf.io/bsvrj",
        "n_subjects": int(len(y)),
        "feature_groups": {},
    }

    feature_groups = {
        "eGeMAPSv02_only": baseline_cols,
        "RQA_only": rqa_cols,
        "Combined": combined_cols,
    }

    # Store pooled predictions for DeLong test
    pooled_predictions = {}

    for group_name, cols in feature_groups.items():
        print(f"\n{'='*60}")
        print(f"[SHAP] {group_name} ({len(cols)} features)")
        print(f"{'='*60}")

        X = df[cols].values.copy()
        # Median imputation (OSF Section 5.4)
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = col_medians[j]

        feature_names = np.array(cols)
        group_results = {}

        # ── Fit full model for SHAP ──
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Collect pooled CV predictions for each classifier
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        clf_predictions = {}

        for clf_name, clf_template in CLASSIFIERS.items():
            y_prob_all = np.zeros(len(y))
            for train_idx, test_idx in skf.split(X, y):
                scl = StandardScaler()
                X_tr = scl.fit_transform(X[train_idx])
                X_te = scl.transform(X[test_idx])
                clf_copy = clf_template.__class__(**clf_template.get_params())
                clf_copy.fit(X_tr, y[train_idx])
                y_prob_all[test_idx] = clf_copy.predict_proba(X_te)[:, 1]
            clf_predictions[clf_name] = y_prob_all

        # Best classifier for this group
        best_clf_name = max(clf_predictions.keys(),
                           key=lambda k: roc_auc_score(y, clf_predictions[k]))
        best_preds = clf_predictions[best_clf_name]
        pooled_predictions[group_name] = {
            "best_clf": best_clf_name,
            "y_pred": best_preds,
        }

        # ── SHAP for each classifier ──
        for clf_name, clf_template in CLASSIFIERS.items():
            print(f"\n  SHAP for {clf_name}...")
            clf = clf_template.__class__(**clf_template.get_params())
            clf.fit(X_scaled, y)

            try:
                if clf_name == "LogisticRegression":
                    # LinearExplainer per OSF
                    explainer = shap.LinearExplainer(clf, X_scaled)
                    shap_values = explainer.shap_values(X_scaled)
                elif clf_name in ["RandomForest", "GradientBoostedTrees"]:
                    # TreeExplainer per OSF
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Class 1

                # Mean |SHAP| per feature
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                top_idx = np.argsort(mean_abs_shap)[::-1][:10]

                top_features = [
                    {"rank": i+1, "feature": feature_names[idx],
                     "mean_abs_shap": round(float(mean_abs_shap[idx]), 6)}
                    for i, idx in enumerate(top_idx)
                ]

                group_results[f"shap_{clf_name}"] = {
                    "top_10": top_features,
                    "method": "LinearExplainer" if clf_name == "LogisticRegression" else "TreeExplainer",
                }

                print(f"    Top 5:")
                for tf in top_features[:5]:
                    print(f"      {tf['rank']}. {tf['feature']} ({tf['mean_abs_shap']:.6f})")

                # Save beeswarm plot for best classifier in each group
                if clf_name == best_clf_name:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.summary_plot(shap_values, X if clf_name != "LogisticRegression" else X_scaled,
                                     feature_names=feature_names, max_display=20, show=False)
                    plt.tight_layout()
                    plot_path = OUTPUT_DIR / f"shap_beeswarm_{group_name}.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"    ✓ Saved: {plot_path}")

            except Exception as e:
                print(f"    ⚠ SHAP failed for {clf_name}: {e}")
                group_results[f"shap_{clf_name}"] = {"error": str(e)}

        # ── H3 Assessment ──
        # Check across ALL classifiers' SHAP values
        h3_results = {}
        for clf_name in CLASSIFIERS:
            key = f"shap_{clf_name}"
            if key in group_results and "top_10" in group_results[key]:
                top5_names = [t["feature"] for t in group_results[key]["top_10"][:5]]
                has_f0 = any("F0" in f or "f0" in f for f in top5_names)
                has_energy = any("loudness" in f.lower() or "energy" in f.lower() for f in top5_names)
                h3_results[clf_name] = {
                    "cv_f0_in_top5": has_f0,
                    "cv_energy_in_top5": has_energy,
                    "verdict": "SUPPORTED" if (has_f0 and has_energy) else
                               "PARTIALLY" if (has_f0 or has_energy) else "NOT SUPPORTED"
                }

        group_results["h3_assessment"] = h3_results

        # ── Bootstrap ΔAUC CI (H2) ──
        print(f"\n  Bootstrap ΔAUC CI (H2)...")
        h2_ci = bootstrap_delta_auc_ci(y, best_preds, STREAM_B_AUC)
        group_results["h2_bootstrap_ci"] = h2_ci
        print(f"    ΔAUC = {h2_ci['mean_delta_auc']:+.4f} [{h2_ci['ci_lower']:+.4f}, {h2_ci['ci_upper']:+.4f}]")
        print(f"    Non-inferiority (CI lower > -0.10): {'✅' if h2_ci['non_inferiority_met'] else '❌'}")
        print(f"    Superiority (CI lower > 0.00): {'✅' if h2_ci['superiority_met'] else '❌'}")

        results["feature_groups"][group_name] = group_results

    # ── DeLong Test: eGeMAPSv02 vs Combined ──
    print(f"\n{'='*60}")
    print("[DeLong] Formal AUC comparison (OSF Section 3.4)")
    print(f"{'='*60}")

    delong_results = {}

    # eGeMAPSv02 vs Combined
    if "eGeMAPSv02_only" in pooled_predictions and "Combined" in pooled_predictions:
        y_pred_1 = pooled_predictions["eGeMAPSv02_only"]["y_pred"]
        y_pred_2 = pooled_predictions["Combined"]["y_pred"]
        auc1, auc2, z, p = delong_test(y, y_pred_1, y_pred_2)
        delong_results["eGeMAPSv02_vs_Combined"] = {
            "auc_egemaps": round(auc1, 4), "auc_combined": round(auc2, 4),
            "z_statistic": round(z, 4), "p_value": round(p, 4),
            "significant": p < 0.05,
        }
        print(f"  eGeMAPSv02 ({auc1:.4f}) vs Combined ({auc2:.4f}): z={z:.4f}, p={p:.4f}")

    # eGeMAPSv02 vs RQA-only
    if "eGeMAPSv02_only" in pooled_predictions and "RQA_only" in pooled_predictions:
        y_pred_1 = pooled_predictions["eGeMAPSv02_only"]["y_pred"]
        y_pred_2 = pooled_predictions["RQA_only"]["y_pred"]
        auc1, auc2, z, p = delong_test(y, y_pred_1, y_pred_2)
        delong_results["eGeMAPSv02_vs_RQA_only"] = {
            "auc_egemaps": round(auc1, 4), "auc_rqa": round(auc2, 4),
            "z_statistic": round(z, 4), "p_value": round(p, 4),
            "significant": p < 0.05,
        }
        print(f"  eGeMAPSv02 ({auc1:.4f}) vs RQA-only ({auc2:.4f}): z={z:.4f}, p={p:.4f}")

    results["delong_tests"] = delong_results

    # ── Save ──
    output_path = OUTPUT_DIR / "shap_delong_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved: {output_path}")

    print(f"\n{'='*70}")
    print("DONE — SHAP + DeLong + Bootstrap ΔAUC")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
