#!/usr/bin/env python3
"""
03_RQA_Classification.py — RQA Feature Classification & Integration Analysis
=============================================================================
Aligned with OSF Pre-Registration (osf.io/bsvrj):
  - H1: AUC > chance (permutation test, 1000 perms)
  - H2: Non-inferiority vs Stream B (ΔAUC bootstrap CI)
  - H3: SHAP feature importance
  - Section 5.5 Exploratory: Feature ablation (RQA-only, eGeMAPSv02-only, combined)

Three analysis modes:
  1. RQA-only classification (79 RQA features)
  2. eGeMAPSv02-only classification (89 functionals — existing baseline)
  3. Combined (eGeMAPSv02 + RQA — test for AUC lift)

Plus: correlation analysis, SHAP interpretation, and Stream B comparison.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, precision_recall_curve, auc,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ── Config ──
BASELINE_CSV = "/lab/lily/work/full_scale_features_prereg.csv"
RQA_CSV = "/lab/lily/work/peter/rqa_results/rqa_features.csv"
OUTPUT_DIR = Path("/lab/lily/work/peter/rqa_analysis")
OUTPUT_JSON = OUTPUT_DIR / "rqa_classification_results.json"
OUTPUT_REPORT = OUTPUT_DIR / "rqa_analysis_report.txt"

META_COLS = ["subject_id", "anhedonia_binary", "anhedonia_sum",
             "PHQ8_Binary", "PHQ8_Score", "Gender", "split"]
OUTCOME_COL = "anhedonia_binary"

# Pre-registered classifiers (Section 5.1 — fixed hyperparameters)
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
N_PERMUTATIONS = 1000
RANDOM_STATE = 42
STREAM_B_AUC = 0.580  # From Stream B results (LogReg, pre-fMRIPrep)


# ── Core Functions ──

def cross_validate_full(X, y, clf, skf):
    """Full CV returning per-fold AUCs (test + train) for overfitting diagnostics."""
    test_aucs = []
    train_aucs = []
    y_pred_all = np.zeros(len(y))
    y_prob_all = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf_copy = clf.__class__(**clf.get_params())
        clf_copy.fit(X_tr, y[train_idx])

        y_prob_test = clf_copy.predict_proba(X_te)[:, 1]
        y_prob_train = clf_copy.predict_proba(X_tr)[:, 1]

        y_prob_all[test_idx] = y_prob_test

        if len(np.unique(y[test_idx])) == 2:
            test_aucs.append(roc_auc_score(y[test_idx], y_prob_test))
        if len(np.unique(y[train_idx])) == 2:
            train_aucs.append(roc_auc_score(y[train_idx], y_prob_train))

    test_mean = np.mean(test_aucs) if test_aucs else 0.5
    train_mean = np.mean(train_aucs) if train_aucs else 0.5
    gap = train_mean - test_mean

    # Bootstrap CI on pooled predictions
    ci_lower, ci_upper = bootstrap_ci(y, y_prob_all)

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y, y_prob_all)
    auc_pr = auc(recall, precision)

    # Balanced accuracy at Youden's J
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, y_prob_all)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    y_pred = (y_prob_all >= best_thresh).astype(int)
    bal_acc = balanced_accuracy_score(y, y_pred)

    return {
        "auc_roc_mean": round(test_mean, 4),
        "auc_roc_std": round(np.std(test_aucs), 4) if test_aucs else 0,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "auc_pr": round(auc_pr, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "train_auc_mean": round(train_mean, 4),
        "auc_gap": round(gap, 4),
        "overfit_flag": "OVERFITTING" if gap > 0.15 else "OK",
        "per_fold_aucs": [round(a, 4) for a in test_aucs],
    }


def bootstrap_ci(y_true, y_prob, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI for AUC-ROC."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) == 2:
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    boot_aucs = np.array(boot_aucs)
    return np.percentile(boot_aucs, 100 * alpha / 2), np.percentile(boot_aucs, 100 * (1 - alpha / 2))


def permutation_test(X, y, clf, n_perms=N_PERMUTATIONS):
    """Permutation test for AUC > chance (H1)."""
    rng = np.random.RandomState(RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # True AUC
    true_aucs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf_copy = clf.__class__(**clf.get_params())
        clf_copy.fit(X_tr, y[train_idx])
        y_prob = clf_copy.predict_proba(X_te)[:, 1]
        if len(np.unique(y[test_idx])) == 2:
            true_aucs.append(roc_auc_score(y[test_idx], y_prob))
    true_auc = np.mean(true_aucs)

    # Null distribution
    null_aucs = []
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        perm_aucs = []
        for train_idx, test_idx in skf.split(X, y_perm):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf_copy = clf.__class__(**clf.get_params())
            clf_copy.fit(X_tr, y_perm[train_idx])
            y_prob = clf_copy.predict_proba(X_te)[:, 1]
            if len(np.unique(y_perm[test_idx])) == 2:
                perm_aucs.append(roc_auc_score(y_perm[test_idx], y_prob))
        null_aucs.append(np.mean(perm_aucs) if perm_aucs else 0.5)
        if (i + 1) % 100 == 0:
            print(f"    Permutation {i+1}/{n_perms} (null_mean={np.mean(null_aucs):.4f})")

    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= true_auc) + 1) / (n_perms + 1)
    return true_auc, p_value, null_aucs


def compute_shap_importance(X, y, feature_names):
    """SHAP feature importance for LogReg (LinearExplainer per OSF Section 5.2)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    # Use coefficient magnitudes as proxy for SHAP (LinearExplainer equivalent)
    coef = np.abs(clf.coef_[0])
    importance = list(zip(feature_names, coef))
    importance.sort(key=lambda x: x[1], reverse=True)
    return importance


def rqa_baseline_correlation(rqa_df, baseline_df, merge_col="subject_id"):
    """Compute correlations between RQA and baseline eGeMAPSv02 features."""
    # Check for shared subjects
    rqa_subjects = set(rqa_df[merge_col].str.replace("_AUDIO", ""))
    baseline_subjects = set(baseline_df[merge_col].str.replace("_P", ""))
    overlap = rqa_subjects & baseline_subjects
    return len(overlap), len(rqa_subjects), len(baseline_subjects)


def main():
    print("=" * 70)
    print("RQA CLASSIFICATION & INTEGRATION ANALYSIS")
    print(f"Aligned with OSF Pre-Registration (osf.io/bsvrj)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load Data ──
    print("\n[1] Loading data...")
    baseline_df = pd.read_csv(BASELINE_CSV)
    rqa_df = pd.read_csv(RQA_CSV)

    print(f"  Baseline (eGeMAPSv02): {baseline_df.shape[0]} subjects × {baseline_df.shape[1]} cols")
    print(f"  RQA features: {rqa_df.shape[0]} subjects × {rqa_df.shape[1]} cols")

    # ── Harmonize subject IDs ──
    # Baseline uses "300_P", RQA uses "300_AUDIO"
    baseline_df["merge_id"] = baseline_df["subject_id"].str.replace("_P", "", regex=False)
    rqa_df["merge_id"] = rqa_df["subject_id"].str.replace("_AUDIO", "", regex=False)

    # Check overlap
    overlap = set(baseline_df["merge_id"]) & set(rqa_df["merge_id"])
    print(f"  Subject overlap: {len(overlap)} subjects")

    # ── Merge ──
    rqa_cols = [c for c in rqa_df.columns if c.startswith("rqa_")]
    rqa_for_merge = rqa_df[["merge_id"] + rqa_cols].copy()

    merged = baseline_df.merge(rqa_for_merge, on="merge_id", how="inner")
    print(f"  Merged dataset: {merged.shape[0]} subjects × {merged.shape[1]} cols")

    # Save merged CSV
    merged.to_csv(OUTPUT_DIR / "merged_features_egemaps_rqa.csv", index=False)
    print(f"  ✓ Saved merged CSV")

    # ── Identify feature groups ──
    all_cols = merged.columns.tolist()
    baseline_feature_cols = [c for c in all_cols if c not in META_COLS + ["merge_id"]
                            and not c.startswith("rqa_")]
    rqa_feature_cols = [c for c in all_cols if c.startswith("rqa_")]
    # Drop all-NaN RQA columns (loudness channel)
    rqa_valid_cols = [c for c in rqa_feature_cols if merged[c].notna().sum() > 10]
    combined_cols = baseline_feature_cols + rqa_valid_cols

    print(f"\n  Feature groups:")
    print(f"    eGeMAPSv02-only: {len(baseline_feature_cols)} features")
    print(f"    RQA-only (valid): {len(rqa_valid_cols)} features ({len(rqa_feature_cols) - len(rqa_valid_cols)} dropped as all-NaN)")
    print(f"    Combined: {len(combined_cols)} features")

    # ── Prepare outcome ──
    if OUTCOME_COL not in merged.columns:
        print(f"  ⚠ Outcome column '{OUTCOME_COL}' not found. Available: {[c for c in META_COLS if c in merged.columns]}")
        return

    valid_mask = merged[OUTCOME_COL].notna()
    df = merged[valid_mask].copy()
    y = df[OUTCOME_COL].values.astype(int)
    print(f"\n  Outcome: {OUTCOME_COL}")
    print(f"  N = {len(y)} (class 0: {(y==0).sum()}, class 1: {(y==1).sum()})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "RQA Integration — 3-way feature ablation",
        "osf_reference": "osf.io/bsvrj",
        "n_subjects": int(len(y)),
        "class_balance": {"class_0": int((y==0).sum()), "class_1": int((y==1).sum())},
        "stream_b_comparison_auc": STREAM_B_AUC,
        "feature_groups": {},
    }

    # ── Run classification for each feature group ──
    feature_groups = {
        "eGeMAPSv02_only": baseline_feature_cols,
        "RQA_only": rqa_valid_cols,
        "Combined_eGeMAPSv02_RQA": combined_cols,
    }

    for group_name, cols in feature_groups.items():
        print(f"\n{'='*60}")
        print(f"[CLASSIFICATION] {group_name} ({len(cols)} features)")
        print(f"{'='*60}")

        X = df[cols].values
        # Handle NaN with median imputation (per OSF Section 5.4)
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = col_medians[j]

        group_results = {
            "n_features": len(cols),
            "classifiers": {},
        }

        best_clf_name = None
        best_auc = 0

        for clf_name, clf in CLASSIFIERS.items():
            print(f"\n  {clf_name}...")
            res = cross_validate_full(X, y, clf, skf)
            group_results["classifiers"][clf_name] = res
            print(f"    AUC-ROC: {res['auc_roc_mean']:.4f} ± {res['auc_roc_std']:.4f} "
                  f"[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]")
            print(f"    AUC-PR: {res['auc_pr']:.4f} | Bal Acc: {res['balanced_accuracy']:.4f}")
            print(f"    Train AUC: {res['train_auc_mean']:.4f} | Gap: {res['auc_gap']:.4f} ({res['overfit_flag']})")
            print(f"    Per-fold: {res['per_fold_aucs']}")

            if res['auc_roc_mean'] > best_auc:
                best_auc = res['auc_roc_mean']
                best_clf_name = clf_name

        # Permutation test on best classifier (H1)
        print(f"\n  Permutation test (H1) on {best_clf_name}...")
        best_clf = CLASSIFIERS[best_clf_name]
        true_auc, p_value, null_aucs = permutation_test(X, y, best_clf)
        group_results["permutation_test"] = {
            "classifier": best_clf_name,
            "true_auc": round(true_auc, 4),
            "null_mean": round(np.mean(null_aucs), 4),
            "null_std": round(np.std(null_aucs), 4),
            "p_value": round(p_value, 6),
            "h1_result": "SUPPORTED" if p_value < 0.05 else "NOT SUPPORTED",
            "n_permutations": N_PERMUTATIONS,
        }
        np.save(OUTPUT_DIR / f"null_aucs_{group_name}.npy", null_aucs)
        print(f"    True AUC: {true_auc:.4f} | p = {p_value:.4f} → H1: {group_results['permutation_test']['h1_result']}")

        # H2: Non-inferiority vs Stream B
        delta_auc = best_auc - STREAM_B_AUC
        group_results["h2_noninferiority"] = {
            "delta_auc": round(delta_auc, 4),
            "stream_a_auc": round(best_auc, 4),
            "stream_b_auc": STREAM_B_AUC,
            "criterion": "ΔAUC lower CI > -0.10",
            "note": "Full bootstrap ΔAUC CI requires both streams on shared CV — reported as point estimate",
        }
        print(f"    H2: ΔAUC = {delta_auc:+.4f} (Stream A {best_auc:.4f} vs Stream B {STREAM_B_AUC:.3f})")

        # SHAP / Feature importance (H3)
        print(f"  Computing feature importance (H3)...")
        importance = compute_shap_importance(X, y, cols)
        top_10 = importance[:10]
        group_results["feature_importance_top10"] = [
            {"rank": i+1, "feature": f, "abs_coef": round(v, 4)}
            for i, (f, v) in enumerate(top_10)
        ]
        print(f"    Top 5 features:")
        for i, (f, v) in enumerate(top_10[:5]):
            print(f"      {i+1}. {f} ({v:.4f})")

        # Check H3 criteria (CV_F0 and CV_Energy in top 5)
        top5_names = [f for f, _ in top_10[:5]]
        has_f0 = any("F0" in f for f in top5_names)
        has_energy = any("loudness" in f.lower() or "energy" in f.lower() for f in top5_names)
        group_results["h3_exploratory"] = {
            "cv_f0_in_top5": has_f0,
            "cv_energy_in_top5": has_energy,
            "h3_result": "SUPPORTED" if (has_f0 and has_energy) else "PARTIALLY SUPPORTED" if (has_f0 or has_energy) else "NOT SUPPORTED",
        }

        results["feature_groups"][group_name] = group_results

    # ── RQA Feature Correlation Analysis ──
    print(f"\n{'='*60}")
    print("[CORRELATION] RQA inter-feature analysis")
    print(f"{'='*60}")

    rqa_data = df[rqa_valid_cols].copy()
    # Drop NaN for correlation
    rqa_clean = rqa_data.dropna(axis=1, how='all')

    # Correlation with outcome
    outcome_corrs = []
    for col in rqa_clean.columns:
        valid = rqa_clean[col].notna()
        if valid.sum() > 10:
            r, p = stats.pointbiserialr(y[valid], rqa_clean[col][valid].values)
            outcome_corrs.append({"feature": col, "r": round(r, 4), "p": round(p, 4)})
    outcome_corrs.sort(key=lambda x: abs(x["r"]), reverse=True)

    results["rqa_outcome_correlations_top10"] = outcome_corrs[:10]
    print(f"\n  Top 10 RQA features correlated with anhedonia:")
    for c in outcome_corrs[:10]:
        sig = "*" if c["p"] < 0.05 else ""
        print(f"    {c['feature']}: r = {c['r']:+.4f} (p = {c['p']:.4f}){sig}")

    # Count significant
    n_sig = sum(1 for c in outcome_corrs if c["p"] < 0.05)
    n_sig_bonf = sum(1 for c in outcome_corrs if c["p"] < 0.05 / len(outcome_corrs))
    results["rqa_sig_count"] = {
        "n_features_tested": len(outcome_corrs),
        "n_significant_p05": n_sig,
        "n_significant_bonferroni": n_sig_bonf,
    }
    print(f"\n  Significant (p<.05): {n_sig}/{len(outcome_corrs)}")
    print(f"  Significant (Bonferroni): {n_sig_bonf}/{len(outcome_corrs)}")

    # ── AUC Lift Analysis ──
    print(f"\n{'='*60}")
    print("[AUC LIFT] Combined vs Baseline")
    print(f"{'='*60}")

    baseline_best = max(
        results["feature_groups"]["eGeMAPSv02_only"]["classifiers"].values(),
        key=lambda x: x["auc_roc_mean"]
    )
    combined_best = max(
        results["feature_groups"]["Combined_eGeMAPSv02_RQA"]["classifiers"].values(),
        key=lambda x: x["auc_roc_mean"]
    )
    rqa_only_best = max(
        results["feature_groups"]["RQA_only"]["classifiers"].values(),
        key=lambda x: x["auc_roc_mean"]
    )

    lift = combined_best["auc_roc_mean"] - baseline_best["auc_roc_mean"]
    results["auc_lift"] = {
        "baseline_best_auc": baseline_best["auc_roc_mean"],
        "rqa_only_best_auc": rqa_only_best["auc_roc_mean"],
        "combined_best_auc": combined_best["auc_roc_mean"],
        "lift_combined_vs_baseline": round(lift, 4),
        "lift_pct": round(lift / baseline_best["auc_roc_mean"] * 100, 2) if baseline_best["auc_roc_mean"] > 0 else 0,
    }

    print(f"  eGeMAPSv02-only best AUC: {baseline_best['auc_roc_mean']:.4f}")
    print(f"  RQA-only best AUC: {rqa_only_best['auc_roc_mean']:.4f}")
    print(f"  Combined best AUC: {combined_best['auc_roc_mean']:.4f}")
    print(f"  AUC lift: {lift:+.4f} ({results['auc_lift']['lift_pct']:+.2f}%)")

    # ── Save Results ──
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {OUTPUT_JSON}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("EXECUTIVE SUMMARY")
    print(f"{'='*70}")

    for gn, gr in results["feature_groups"].items():
        perm = gr.get("permutation_test", {})
        best_clf = max(gr["classifiers"].items(), key=lambda x: x[1]["auc_roc_mean"])
        print(f"\n  {gn}:")
        print(f"    Best: {best_clf[0]} AUC = {best_clf[1]['auc_roc_mean']:.4f}")
        print(f"    H1 (AUC > chance): p = {perm.get('p_value', 'N/A')} → {perm.get('h1_result', 'N/A')}")
        h2 = gr.get("h2_noninferiority", {})
        print(f"    H2 (vs Stream B): ΔAUC = {h2.get('delta_auc', 'N/A'):+.4f}")

    print(f"\n  AUC Lift (Combined vs Baseline): {results['auc_lift']['lift_combined_vs_baseline']:+.4f}")
    print(f"\n  Stream B reference AUC: {STREAM_B_AUC}")
    print("=" * 70)


if __name__ == "__main__":
    main()
