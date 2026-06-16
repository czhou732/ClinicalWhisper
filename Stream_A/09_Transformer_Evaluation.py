#!/usr/bin/env python3
"""
09_Transformer_Evaluation.py — Evaluation & Ablation for VoiceTransformer
==========================================================================
Runs comprehensive evaluation of the trained transformer model:
  - Head-to-head comparison with RF baseline (AUC 0.63)
  - Ablation studies (acoustic-only, text-only, full multimodal)
  - Permutation test for statistical significance
  - Attention weight analysis
  - Bootstrap confidence intervals

Usage:
    python 09_Transformer_Evaluation.py \
        --data_dir ./transformer_data \
        --results_dir ./transformer_results \
        --baseline_auc 0.63

Dependencies:
    pip install torch scikit-learn numpy matplotlib seaborn tqdm
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    precision_recall_curve, auc, roc_curve,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Import from model file
from importlib import import_module


# ── Evaluation Functions ──────────────────────────────────────────────

def bootstrap_ci(y_true, y_prob, n_boot=10000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for AUC-ROC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) == 2:
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    boot_aucs = np.array(boot_aucs)
    return (
        float(np.percentile(boot_aucs, 100 * alpha / 2)),
        float(np.percentile(boot_aucs, 100 * (1 - alpha / 2))),
    )


def permutation_test_from_predictions(y_true, y_prob, n_perms=1000, seed=42):
    """
    Permutation test on pre-computed predictions.
    
    Shuffles labels and recomputes AUC to build null distribution.
    More efficient than retraining the model for each permutation.
    """
    rng = np.random.RandomState(seed)
    true_auc = roc_auc_score(y_true, y_prob)

    null_aucs = []
    for _ in range(n_perms):
        y_perm = rng.permutation(y_true)
        if len(np.unique(y_perm)) == 2:
            null_aucs.append(roc_auc_score(y_perm, y_prob))
    
    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= true_auc) + 1) / (n_perms + 1)
    
    return {
        "true_auc": round(float(true_auc), 4),
        "null_mean": round(float(np.mean(null_aucs)), 4),
        "null_std": round(float(np.std(null_aucs)), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "n_permutations": n_perms,
    }


def compute_full_metrics(y_true, y_prob):
    """Compute all evaluation metrics from predictions."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # AUC-ROC
    auc_roc = roc_auc_score(y_true, y_prob)

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)

    # Balanced accuracy at Youden's J
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    y_pred = (y_prob >= best_thresh).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Sensitivity and specificity at optimal threshold
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(y_true, y_prob)

    return {
        "auc_roc": round(float(auc_roc), 4),
        "auc_roc_ci_lower": round(ci_lower, 4),
        "auc_roc_ci_upper": round(ci_upper, 4),
        "auc_pr": round(float(auc_pr), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "optimal_threshold": round(float(best_thresh), 4),
        "n_subjects": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }


def load_fold_results(results_dir: str) -> dict:
    """Load per-fold predictions from transformer training results."""
    # Try both possible filenames
    for name in ["transformer_results.json", "training_results.json"]:
        results_path = Path(results_dir) / name
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
    raise FileNotFoundError(f"No training results found in {results_dir}")


def aggregate_fold_predictions(fold_results: list) -> tuple:
    """Aggregate per-fold predictions into pooled arrays."""
    all_true = []
    all_prob = []
    
    for fold in fold_results:
        # Support both output formats
        if "y_true" in fold:
            all_true.extend(fold["y_true"])
            all_prob.extend(fold["y_prob"])
        elif "val_labels" in fold:
            all_true.extend(fold["val_labels"])
            all_prob.extend(fold["val_predictions"])
    
    return np.array(all_true), np.array(all_prob)


# ── Visualization ─────────────────────────────────────────────────────

def plot_roc_comparison(results: dict, output_path: str):
    """Plot ROC curves comparing transformer vs RF baseline."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Plot each model's ROC
        colors = {
            "full_multimodal": "#2196F3",
            "acoustic_only": "#FF9800",
            "text_only": "#4CAF50",
        }

        for model_name, model_results in results.items():
            if "y_true" in model_results and "y_prob" in model_results:
                y_true = np.array(model_results["y_true"])
                y_prob = np.array(model_results["y_prob"])
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_val = model_results["metrics"]["auc_roc"]
                label = f"{model_name} (AUC={auc_val:.3f})"
                color = colors.get(model_name, "#9E9E9E")
                ax.plot(fpr, tpr, label=label, color=color, linewidth=2)

        # Baseline reference
        baseline_auc = results.get("baseline_comparison", {}).get("rf_baseline_auc", 0.63)
        ax.axhline(y=baseline_auc, color="red", linestyle="--", alpha=0.5,
                   label=f"RF Baseline (AUC={baseline_auc:.3f})")

        # Chance line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("Transformer vs Baseline — ROC Comparison", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ ROC plot saved: {output_path}")

    except ImportError:
        print("  ⚠ matplotlib not available — skipping ROC plot")


def plot_attention_weights(attention_weights: list, output_path: str):
    """Visualize average attention patterns across subjects."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        # Average attention across subjects (pad to max length)
        max_len = max(a.shape[-1] for a in attention_weights)
        padded = []
        for a in attention_weights:
            if a.ndim == 2:  # [heads, seq_len]
                pad_size = max_len - a.shape[-1]
                padded.append(np.pad(a, ((0, 0), (0, pad_size))))

        if not padded:
            return

        avg_attn = np.mean(padded, axis=0)  # [heads, max_len]

        fig, axes = plt.subplots(1, min(4, avg_attn.shape[0]),
                                  figsize=(16, 4))
        if avg_attn.shape[0] == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i < avg_attn.shape[0]:
                ax.bar(range(avg_attn.shape[1]), avg_attn[i], alpha=0.7)
                ax.set_title(f"Head {i+1}", fontsize=11)
                ax.set_xlabel("Window position")
                ax.set_ylabel("Attention weight")

        plt.suptitle("CLS Token Attention Weights by Head", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Attention plot saved: {output_path}")

    except ImportError:
        print("  ⚠ matplotlib not available — skipping attention plot")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VoiceTransformer and compare with baseline"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./transformer_results",
        help="Directory with training results from 08_Transformer_Model.py",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (defaults to results_dir)",
    )
    parser.add_argument(
        "--baseline_auc", type=float, default=0.63,
        help="RF baseline AUC for comparison (default: 0.63)",
    )
    parser.add_argument(
        "--n_permutations", type=int, default=1000,
        help="Number of permutations for significance test",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir or args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VOICETRANSFORMER EVALUATION & ABLATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Baseline AUC: {args.baseline_auc}")
    print("=" * 70)

    # Load training results
    print("\n[1] Loading training results...")
    training_results = load_fold_results(args.results_dir)

    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "baseline_auc": args.baseline_auc,
        "models": {},
    }

    # Evaluate each model variant (full, acoustic-only, text-only)
    model_variants = {}
    
    # Check for ablation results
    for variant_name in ["full_multimodal", "acoustic_only", "text_only"]:
        variant_path = Path(args.results_dir) / f"{variant_name}_results.json"
        if variant_path.exists():
            with open(variant_path) as f:
                model_variants[variant_name] = json.load(f)
    
    # Fall back to main results if no ablations
    if not model_variants:
        model_variants["full_multimodal"] = training_results

    for variant_name, variant_data in model_variants.items():
        print(f"\n{'=' * 60}")
        print(f"[EVAL] {variant_name}")
        print(f"{'=' * 60}")

        # Aggregate fold predictions — support both key names
        fold_results = variant_data.get("per_fold", variant_data.get("fold_results", []))
        if not fold_results:
            print(f"  ⚠ No fold results for {variant_name}")
            continue

        y_true, y_prob = aggregate_fold_predictions(fold_results)

        # Full metrics
        metrics = compute_full_metrics(y_true, y_prob)
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f} [{metrics['auc_roc_ci_lower']:.3f}, {metrics['auc_roc_ci_upper']:.3f}]")
        print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")

        # Permutation test
        print(f"\n  Running permutation test ({args.n_permutations} permutations)...")
        perm_result = permutation_test_from_predictions(
            y_true, y_prob, n_perms=args.n_permutations
        )
        print(f"  H1 (AUC > chance): p = {perm_result['p_value']:.4f} → {'SUPPORTED' if perm_result['significant'] else 'NOT SUPPORTED'}")

        # Baseline comparison
        delta_auc = metrics["auc_roc"] - args.baseline_auc
        print(f"\n  vs RF Baseline:")
        print(f"    Transformer AUC: {metrics['auc_roc']:.4f}")
        print(f"    RF Baseline AUC: {args.baseline_auc:.4f}")
        print(f"    ΔAUC: {delta_auc:+.4f}")
        print(f"    Improvement: {delta_auc / args.baseline_auc * 100:+.1f}%")

        evaluation["models"][variant_name] = {
            "metrics": metrics,
            "permutation_test": perm_result,
            "baseline_comparison": {
                "rf_baseline_auc": args.baseline_auc,
                "delta_auc": round(delta_auc, 4),
                "improvement_pct": round(delta_auc / args.baseline_auc * 100, 2),
            },
            "y_true": y_true.tolist(),
            "y_prob": y_prob.tolist(),
        }

    # Ablation summary
    if len(evaluation["models"]) > 1:
        print(f"\n{'=' * 60}")
        print("ABLATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Model':<25} {'AUC-ROC':>10} {'ΔAUC vs baseline':>18}")
        print(f"  {'-'*25} {'-'*10} {'-'*18}")
        for name, data in evaluation["models"].items():
            auc_val = data["metrics"]["auc_roc"]
            delta = data["baseline_comparison"]["delta_auc"]
            print(f"  {name:<25} {auc_val:>10.4f} {delta:>+18.4f}")
        print(f"  {'RF Baseline':<25} {args.baseline_auc:>10.4f} {'—':>18}")

    # Save evaluation results
    eval_path = output_dir / "evaluation_results.json"
    # Remove y_true/y_prob from saved JSON (large)
    eval_save = json.loads(json.dumps(evaluation, default=str))
    for model_data in eval_save.get("models", {}).values():
        model_data.pop("y_true", None)
        model_data.pop("y_prob", None)
    
    with open(eval_path, "w") as f:
        json.dump(eval_save, f, indent=2)
    print(f"\n✓ Evaluation saved: {eval_path}")

    # Generate plots
    print("\n[PLOTS]")
    plot_roc_comparison(
        evaluation["models"],
        str(output_dir / "roc_comparison.png"),
    )

    # Executive summary
    print(f"\n{'=' * 70}")
    print("EXECUTIVE SUMMARY")
    print(f"{'=' * 70}")

    best_model = max(
        evaluation["models"].items(),
        key=lambda x: x[1]["metrics"]["auc_roc"],
    )
    best_name, best_data = best_model
    best_auc = best_data["metrics"]["auc_roc"]
    best_delta = best_data["baseline_comparison"]["delta_auc"]

    print(f"  Best model: {best_name}")
    print(f"  AUC-ROC: {best_auc:.4f} (baseline: {args.baseline_auc:.4f}, Δ = {best_delta:+.4f})")
    print(f"  H1 (AUC > chance): p = {best_data['permutation_test']['p_value']:.4f}")

    if best_auc >= 0.70:
        print(f"\n  🎯 TARGET MET: AUC ≥ 0.70")
        print(f"  → Ready for CNNI submission upgrade")
    elif best_auc > args.baseline_auc:
        print(f"\n  📈 IMPROVEMENT over baseline (+{best_delta:.4f})")
        print(f"  → Consider pre-training (Phase 5) to push past 0.70")
    else:
        print(f"\n  ⚠ No improvement over baseline")
        print(f"  → Check overfitting, try different hyperparameters")

    print("=" * 70)


if __name__ == "__main__":
    main()
