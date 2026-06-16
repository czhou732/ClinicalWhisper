#!/usr/bin/env python3
"""
05_Feature_Ablation.py — Feature Ablation Analysis (OSF Section 5.5.1)
======================================================================
Pre-registered exploratory analysis: Classification with subsets of features
(prosodic-only, spectral-only, temporal-only) to identify which acoustic
domains drive performance.

Also includes:
  - Feature selection with mutual information (top-50, top-100)
    to assess robustness under dimensionality reduction
  - Per-domain AUC comparison table for manuscript

OSF Section 5.5.1: "Feature ablation: Classification with subsets of features
(prosodic-only, spectral-only, temporal-only) to identify which acoustic
domains drive performance."

Outputs:
  Results/Stream_A/feature_ablation_results.json
  Results/Stream_A/ablation_auc_barplot.png
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# ── Config ──
MERGED_CSV = "/lab/lily/work/peter/rqa_analysis/merged_features_egemaps_rqa.csv"
OUTPUT_DIR = Path("/lab/lily/work/peter/rqa_analysis")

META_COLS = ["subject_id", "anhedonia_binary", "anhedonia_sum",
             "PHQ8_Binary", "PHQ8_Score", "Gender", "split", "merge_id"]
OUTCOME_COL = "anhedonia_binary"

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

# ── eGeMAPSv02 Domain Mapping ──
# Based on Eyben et al. (2016) Table 1 — eGeMAPSv02 feature categorization

PROSODIC_KEYWORDS = [
    "F0", "f0", "pitch", "loudness", "Loudness", "intensity",
    "jitter", "Jitter", "shimmer", "Shimmer", "logRelF0",
    "VoicedSegments", "UnvoicedSegments", "loudnessPeaks",
    "MeanVoicedSegmentLength", "StddevVoicedSegmentLength",
    "MeanUnvoicedSegmentLength", "StddevUnvoicedSegmentLength",
]

SPECTRAL_KEYWORDS = [
    "mfcc", "MFCC", "spectral", "Spectral", "formant", "Formant",
    "F1", "F2", "F3", "bandwidth", "Bandwidth",
    "alphaRatio", "hammarberg", "Hammarberg", "slope",
    "HNR", "hnr", "harmonics", "logHNR",
    "slopeV0-500", "slopeV500-1500", "slopeUV0-500", "slopeUV500-1500",
    "spectralFlux",
]

TEMPORAL_KEYWORDS = [
    "rate", "Rate", "pace", "duration", "Duration",
    "segments", "n_segments", "speaking_rate", "pause",
    "silencePercentage", "loudnessPeaksPerSec",
    "VoicedSegmentsPerSec",
]


def classify_feature_domain(feature_name):
    """Classify a feature into prosodic, spectral, or temporal domain."""
    # RQA features go to their own domain
    if feature_name.startswith("rqa_"):
        return "rqa"

    for kw in PROSODIC_KEYWORDS:
        if kw in feature_name:
            return "prosodic"
    for kw in SPECTRAL_KEYWORDS:
        if kw in feature_name:
            return "spectral"
    for kw in TEMPORAL_KEYWORDS:
        if kw in feature_name:
            return "temporal"
    return "other"


def cross_validate(X, y, clf, skf):
    """Quick CV returning mean test AUC."""
    test_aucs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf_copy = clf.__class__(**clf.get_params())
        clf_copy.fit(X_tr, y[train_idx])
        y_prob = clf_copy.predict_proba(X_te)[:, 1]
        if len(np.unique(y[test_idx])) == 2:
            test_aucs.append(roc_auc_score(y[test_idx], y_prob))
    return np.mean(test_aucs) if test_aucs else 0.5, test_aucs


def main():
    print("=" * 70)
    print("FEATURE ABLATION ANALYSIS (OSF Section 5.5.1)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # ── Load ──
    df = pd.read_csv(MERGED_CSV)
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in META_COLS and not c.startswith("rqa_")]
    rqa_cols = [c for c in all_cols if c.startswith("rqa_") and df[c].notna().sum() > 10]
    combined_cols = feature_cols + rqa_cols

    valid_mask = df[OUTCOME_COL].notna()
    df = df[valid_mask].copy()
    y = df[OUTCOME_COL].values.astype(int)
    print(f"N = {len(y)} | Features: {len(feature_cols)} eGeMAPSv02 + {len(rqa_cols)} RQA")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Domain classification ──
    domains = {}
    for col in feature_cols:
        domain = classify_feature_domain(col)
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(col)

    print(f"\nDomain breakdown:")
    for d, cols in sorted(domains.items()):
        print(f"  {d}: {len(cols)} features")

    # ── 1. Domain-specific ablation ──
    print(f"\n{'='*60}")
    print("[ABLATION] Domain-specific classification")
    print(f"{'='*60}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "analysis": "Feature Ablation (OSF 5.5.1)",
        "n_subjects": int(len(y)),
        "domain_ablation": {},
        "feature_selection": {},
    }

    # Include full feature sets for comparison
    ablation_groups = {
        "full_eGeMAPSv02": feature_cols,
        "full_combined": combined_cols,
        "full_rqa_only": rqa_cols,
    }
    # Add domain-specific groups
    for domain, cols in domains.items():
        if len(cols) >= 3:  # Need at least 3 features for meaningful classification
            ablation_groups[f"domain_{domain}"] = cols

    for group_name, cols in ablation_groups.items():
        print(f"\n  {group_name} ({len(cols)} features):")

        X = df[cols].values.copy()
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = col_medians[j]

        group_results = {"n_features": len(cols), "classifiers": {}}

        for clf_name, clf in CLASSIFIERS.items():
            auc_mean, fold_aucs = cross_validate(X, y, clf, skf)
            group_results["classifiers"][clf_name] = {
                "auc_roc": round(auc_mean, 4),
                "per_fold": [round(a, 4) for a in fold_aucs],
            }
            print(f"    {clf_name}: AUC = {auc_mean:.4f}")

        best_clf = max(group_results["classifiers"].items(),
                       key=lambda x: x[1]["auc_roc"])
        group_results["best_classifier"] = best_clf[0]
        group_results["best_auc"] = best_clf[1]["auc_roc"]

        results["domain_ablation"][group_name] = group_results

    # ── 2. Feature selection (MI-based) ──
    print(f"\n{'='*60}")
    print("[FEATURE SELECTION] Mutual Information top-K")
    print(f"{'='*60}")

    X_combined = df[combined_cols].values.copy()
    col_medians = np.nanmedian(X_combined, axis=0)
    for j in range(X_combined.shape[1]):
        mask = np.isnan(X_combined[:, j])
        if mask.any():
            X_combined[mask, j] = col_medians[j]

    print("  Computing mutual information scores...")
    mi_scores = mutual_info_classif(X_combined, y, random_state=RANDOM_STATE, n_neighbors=5)
    mi_ranking = np.argsort(mi_scores)[::-1]

    # Save MI ranking
    mi_features = [
        {"rank": i+1, "feature": combined_cols[idx], "mi_score": round(float(mi_scores[idx]), 6)}
        for i, idx in enumerate(mi_ranking[:30])
    ]
    results["mi_ranking_top30"] = mi_features
    print(f"  Top 10 by MI:")
    for mf in mi_features[:10]:
        print(f"    {mf['rank']}. {mf['feature']} (MI={mf['mi_score']:.6f})")

    # Test with reduced feature sets
    for k in [25, 50, 100, 200]:
        if k >= len(combined_cols):
            continue

        top_k_idx = mi_ranking[:k]
        X_k = X_combined[:, top_k_idx]
        print(f"\n  Top-{k} features:")

        k_results = {"n_features": k, "classifiers": {}}
        for clf_name, clf in CLASSIFIERS.items():
            auc_mean, fold_aucs = cross_validate(X_k, y, clf, skf)
            k_results["classifiers"][clf_name] = {
                "auc_roc": round(auc_mean, 4),
                "per_fold": [round(a, 4) for a in fold_aucs],
            }
            print(f"    {clf_name}: AUC = {auc_mean:.4f}")

        best = max(k_results["classifiers"].items(), key=lambda x: x[1]["auc_roc"])
        k_results["best_classifier"] = best[0]
        k_results["best_auc"] = best[1]["auc_roc"]
        results["feature_selection"][f"top_{k}"] = k_results

    # ── 3. Generate ablation bar plot ──
    print(f"\n  Generating ablation bar plot...")

    categories = []
    aucs = []

    for group_name, group_data in results["domain_ablation"].items():
        categories.append(group_name.replace("domain_", "").replace("full_", ""))
        aucs.append(group_data["best_auc"])

    for k_name, k_data in results["feature_selection"].items():
        categories.append(f"MI {k_name}")
        aucs.append(k_data["best_auc"])

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
    bars = ax.barh(range(len(categories)), aucs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlabel("AUC-ROC", fontsize=12)
    ax.set_title("Feature Ablation — Domain & Selection Analysis\n(OSF Section 5.5.1)", fontsize=13)
    ax.axvline(x=0.50, color="red", linestyle="--", alpha=0.5, label="Chance")
    ax.axvline(x=0.580, color="orange", linestyle="--", alpha=0.5, label="Stream B baseline")

    # Add value labels
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.legend(fontsize=9)
    ax.set_xlim(0.40, 0.75)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "ablation_auc_barplot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {plot_path}")

    # ── Save ──
    output_path = OUTPUT_DIR / "feature_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved: {output_path}")

    print(f"\n{'='*70}")
    print("DONE — Feature Ablation Analysis")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
