#!/usr/bin/env python3
"""
08_Transformer_Model.py — Multimodal Transformer for Anhedonia Classification
===============================================================================
Classifies anhedonia from sliding-window acoustic (eGeMAPS LLD) and text
(Sentence-BERT) features extracted from clinical interviews (DAIC-WOZ).

Architecture:
    - Separate linear projections for acoustic (50→128) and text (384→128)
    - Concatenate → Linear(256→256) to form per-window token embeddings
    - Learned positional encoding (max 300 windows)
    - Learned MASK token for interviewer windows
    - Prepended CLS token for classification
    - 4-layer TransformerEncoder (d_model=256, nhead=4, FFN=512, dropout=0.3)
    - Classification head: LayerNorm → Linear(256→64) → GELU → Dropout → Linear(64→1)

Training:
    - Focal loss (γ=2.0, α=0.75) for class imbalance (~17% positive)
    - AdamW (lr=1e-4, weight_decay=0.01)
    - CosineAnnealingWarmRestarts scheduler (T_0=20)
    - Early stopping on val AUC-ROC (patience=15)

Evaluation:
    - 5-fold StratifiedKFold (random_state=42)
    - Per-fold: AUC-ROC, AUC-PR, balanced accuracy
    - Pooled predictions → overall metrics with bootstrap 95% CI (10000 draws)
    - Overfitting diagnostics (train vs test AUC gap)

Input format (per subject .pt file):
    {
        'acoustic_features': Tensor[T, 50],   # T = number of 5s windows
        'text_features':     Tensor[T, 384],
        'mask':              BoolTensor[T],    # True = interviewer window
        'label':             int (0 or 1)
    }

Usage:
    python 08_Transformer_Model.py --data_dir ./transformer_data --output_dir ./transformer_results
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, precision_recall_curve, auc,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold

# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure timestamped logging to both console and file."""
    logger = logging.getLogger("transformer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_path = output_dir / f"transformer_{datetime.now():%Y%m%d_%H%M%S}.log"
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ── Dataset ──────────────────────────────────────────────────────────────────

class AnhedoniaWindowDataset(Dataset):
    """
    Loads per-subject .pt files containing variable-length window sequences.

    Each .pt file is a dict with keys:
        acoustic_features: [T, 50]
        text_features:     [T, 384]
        mask:              [T] (bool — True for interviewer/MASK windows)
        label:             int

    Subjects with zero-length sequences are silently dropped.
    """

    def __init__(self, file_paths: list[Path]):
        self.samples = []
        n_dropped = 0
        for fp in file_paths:
            data = torch.load(fp, map_location="cpu", weights_only=False)
            T = data["acoustic_features"].shape[0]
            if T == 0:
                n_dropped += 1
                continue
            self.samples.append({
                "acoustic": data["acoustic_features"].float(),       # [T, 50]
                "text": data["text_features"].float(),               # [T, 384]
                "speaker_mask": data["mask"].bool(),                  # [T]
                "label": int(data["label"]),
                "seq_len": T,
                "file": fp.name,
            })
        if n_dropped > 0:
            logging.getLogger("transformer").warning(
                f"Dropped {n_dropped} subjects with empty sequences"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """
    Pads variable-length sequences to the longest in the batch.
    Returns padded tensors and a key_padding_mask for the transformer
    (True = ignore this position).

    Note: The CLS token is prepended inside the model, not here.
    """
    max_len = max(s["seq_len"] for s in batch)
    B = len(batch)

    acoustic_padded = torch.zeros(B, max_len, 50)
    text_padded = torch.zeros(B, max_len, 384)
    speaker_mask_padded = torch.zeros(B, max_len, dtype=torch.bool)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = pad
    labels = torch.zeros(B, dtype=torch.float32)
    seq_lens = []

    for i, s in enumerate(batch):
        T = s["seq_len"]
        acoustic_padded[i, :T] = s["acoustic"]
        text_padded[i, :T] = s["text"]
        speaker_mask_padded[i, :T] = s["speaker_mask"]
        padding_mask[i, :T] = False  # real tokens
        labels[i] = s["label"]
        seq_lens.append(T)

    return {
        "acoustic": acoustic_padded,
        "text": text_padded,
        "speaker_mask": speaker_mask_padded,
        "padding_mask": padding_mask,
        "labels": labels,
        "seq_lens": seq_lens,
    }


# ── Model ────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — handles any sequence length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq_len: int, device: torch.device = None) -> torch.Tensor:
        """Returns [1, seq_len, d_model] positional embeddings."""
        if device is None:
            device = torch.device("cpu")
        positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * -(np.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(1, seq_len, self.d_model, device=device)
        pe[0, :, 0::2] = torch.sin(positions * div_term)
        pe[0, :, 1::2] = torch.cos(positions * div_term)
        return pe


class VoiceTransformer(nn.Module):
    """
    Multimodal transformer for anhedonia classification.

    ~350K parameters — deliberately small for N=190.
    """

    def __init__(
        self,
        acoustic_dim: int = 50,
        text_dim: int = 384,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model

        # ── Modality projections ──
        self.acoustic_proj = nn.Linear(acoustic_dim, d_model // 2)
        self.text_proj = nn.Linear(text_dim, d_model // 2)
        self.modality_combine = nn.Linear(d_model, d_model)

        # ── Positional encoding (sinusoidal — no max_len limit) ──
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        # ── Special tokens ──
        self.mask_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # ── Transformer encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Classification head ──
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # ── Input dropout ──
        self.input_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        acoustic: torch.Tensor,      # [B, T, 50]
        text: torch.Tensor,           # [B, T, 384]
        speaker_mask: torch.Tensor,   # [B, T] bool — True = interviewer
        padding_mask: torch.Tensor,   # [B, T] bool — True = pad
    ) -> torch.Tensor:
        B, T, _ = acoustic.shape

        # ── Project each modality ──
        a = self.acoustic_proj(acoustic)   # [B, T, 128]
        t = self.text_proj(text)           # [B, T, 128]
        x = torch.cat([a, t], dim=-1)      # [B, T, 256]
        x = self.modality_combine(x)       # [B, T, 256]

        # ── Replace interviewer windows with learned MASK token ──
        mask_expanded = speaker_mask.unsqueeze(-1).expand_as(x)  # [B, T, 256]
        x = torch.where(mask_expanded, self.mask_token.expand_as(x), x)

        # ── Apply input dropout ──
        x = self.input_dropout(x)

        # ── Prepend CLS token ──
        cls_tokens = self.cls_token.unsqueeze(0).expand(B, 1, -1)  # [B, 1, 256]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, 256]

        # ── Update padding mask for CLS (never padded) ──
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_padding_mask = torch.cat([cls_pad, padding_mask], dim=1)  # [B, T+1]

        # ── Add positional encoding ──
        x = x + self.pos_encoding(T + 1, device=x.device)  # broadcast over batch

        # ── Transformer encoder ──
        x = self.encoder(x, src_key_padding_mask=full_padding_mask)  # [B, T+1, 256]

        # ── Extract CLS representation ──
        cls_repr = x[:, 0, :]  # [B, 256]

        # ── Classify ──
        logits = self.head(cls_repr).squeeze(-1)  # [B]
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss for class-imbalanced classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    α weights the rare class higher; γ down-weights easy examples.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # p_t = probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # α_t = alpha for positives, (1 - alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # Focal modulating factor
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        # Binary cross-entropy (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce
        return loss.mean()


# ── Training utilities ───────────────────────────────────────────────────────

class EarlyStopping:
    """Early stopping on validation AUC-ROC (higher is better)."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf
        self.best_state = None
        self.should_stop = False

    def step(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def compute_auc_safe(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC-ROC that returns 0.5 if only one class present."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI for AUC-ROC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) == 2:
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(boot_aucs) == 0:
        return 0.5, 0.5
    boot_aucs = np.array(boot_aucs)
    lo = np.percentile(boot_aucs, 100 * alpha / 2)
    hi = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ── Single fold training ────────────────────────────────────────────────────

def train_one_fold(
    fold: int,
    train_dataset: AnhedoniaWindowDataset,
    val_dataset: AnhedoniaWindowDataset,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    """Train a single fold and return predictions + metrics."""

    logger.info(f"  Fold {fold+1}/{args.n_folds} — "
                f"train={len(train_dataset)}, val={len(val_dataset)}")

    # Class balance in training set
    train_labels = [s["label"] for s in train_dataset.samples]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    logger.info(f"    Train class balance: neg={n_neg}, pos={n_pos}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False, num_workers=0,
    )

    # Model
    model = VoiceTransformer(
        d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, dropout=args.dropout,
    ).to(device)
    if fold == 0:
        logger.info(f"    Model parameters: {model.count_parameters():,}")

    # Loss, optimizer, scheduler
    # Gentler class weighting: sqrt(ratio) instead of raw ratio
    pos_weight_val = (n_neg / max(n_pos, 1)) ** 0.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05
    )
    # Stable scheduler: linear warmup then cosine decay (no restarts)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
    )
    early_stopping = EarlyStopping(patience=30)

    # Training loop
    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_preds, train_labels_epoch = [], []

        for batch in train_loader:
            acoustic = batch["acoustic"].to(device)
            text = batch["text"].to(device)
            speaker_mask = batch["speaker_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(acoustic, text, speaker_mask, padding_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # OneCycleLR steps per batch

            train_loss += loss.item() * labels.size(0)
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels_epoch.extend(labels.cpu().numpy())

        train_loss /= len(train_dataset)
        train_auc = compute_auc_safe(
            np.array(train_labels_epoch), np.array(train_preds)
        )

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_preds, val_labels_epoch = [], []

        with torch.no_grad():
            for batch in val_loader:
                acoustic = batch["acoustic"].to(device)
                text = batch["text"].to(device)
                speaker_mask = batch["speaker_mask"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(acoustic, text, speaker_mask, padding_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                val_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                val_labels_epoch.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_auc = compute_auc_safe(
            np.array(val_labels_epoch), np.array(val_preds)
        )

        # Log every 10 epochs or on improvement
        if (epoch + 1) % 10 == 0 or epoch == 0:
            gap = train_auc - val_auc
            logger.info(
                f"    Epoch {epoch+1:3d}/{args.epochs} │ "
                f"train_loss={train_loss:.4f}  train_auc={train_auc:.4f} │ "
                f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f} │ "
                f"gap={gap:+.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        # Early stopping
        if early_stopping.step(val_auc, model):
            logger.info(
                f"    ✓ Early stopping at epoch {epoch+1} "
                f"(best val_auc={early_stopping.best_score:.4f})"
            )
            break

    # Restore best model
    early_stopping.restore_best(model)

    # ── Final evaluation on validation set ──
    model.eval()
    final_preds, final_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            acoustic = batch["acoustic"].to(device)
            text = batch["text"].to(device)
            speaker_mask = batch["speaker_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(acoustic, text, speaker_mask, padding_mask)
            final_preds.extend(torch.sigmoid(logits).cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    final_preds = np.array(final_preds)
    final_labels = np.array(final_labels)

    # Final train AUC for overfitting diagnostic
    model.eval()
    train_final_preds, train_final_labels = [], []
    with torch.no_grad():
        for batch in train_loader:
            acoustic = batch["acoustic"].to(device)
            text = batch["text"].to(device)
            speaker_mask = batch["speaker_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(acoustic, text, speaker_mask, padding_mask)
            train_final_preds.extend(torch.sigmoid(logits).cpu().numpy())
            train_final_labels.extend(labels.cpu().numpy())

    train_final_auc = compute_auc_safe(
        np.array(train_final_labels), np.array(train_final_preds)
    )

    # Compute fold metrics
    val_auc_final = compute_auc_safe(final_labels, final_preds)
    precision, recall, _ = precision_recall_curve(final_labels, final_preds)
    auc_pr = auc(recall, precision) if len(np.unique(final_labels)) == 2 else 0.0

    # Balanced accuracy at Youden's J
    if len(np.unique(final_labels)) == 2:
        fpr, tpr, thresholds = roc_curve(final_labels, final_preds)
        j_scores = tpr - fpr
        best_thresh = thresholds[np.argmax(j_scores)]
        bal_acc = balanced_accuracy_score(
            final_labels, (final_preds >= best_thresh).astype(int)
        )
    else:
        bal_acc = 0.5
        best_thresh = 0.5

    gap = train_final_auc - val_auc_final
    overfit_flag = "OVERFITTING" if gap > 0.15 else "OK"

    logger.info(
        f"    Fold {fold+1} results: "
        f"AUC-ROC={val_auc_final:.4f}  AUC-PR={auc_pr:.4f}  "
        f"BalAcc={bal_acc:.4f}  "
        f"TrainAUC={train_final_auc:.4f}  Gap={gap:+.4f} ({overfit_flag})"
    )

    return {
        "fold": fold,
        "val_auc_roc": round(val_auc_final, 4),
        "val_auc_pr": round(auc_pr, 4),
        "val_balanced_accuracy": round(bal_acc, 4),
        "train_auc_roc": round(train_final_auc, 4),
        "auc_gap": round(gap, 4),
        "overfit_flag": overfit_flag,
        "best_threshold": round(float(best_thresh), 4),
        "best_epoch": args.epochs - early_stopping.counter,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "val_predictions": final_preds.tolist(),
        "val_labels": final_labels.tolist(),
    }


# ── Cross-validation driver ─────────────────────────────────────────────────

def run_cross_validation(args: argparse.Namespace, logger: logging.Logger) -> dict:
    """Full stratified k-fold cross-validation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load all subject files ──
    data_dir = Path(args.data_dir)
    pt_files = sorted(data_dir.glob("*.pt"))
    if len(pt_files) == 0:
        logger.error(f"No .pt files found in {data_dir}")
        sys.exit(1)
    logger.info(f"Found {len(pt_files)} subject files in {data_dir}")

    # Quick-load labels for stratification (without building full dataset yet)
    all_labels = []
    valid_files = []
    for fp in pt_files:
        data = torch.load(fp, map_location="cpu", weights_only=False)
        T = data["acoustic_features"].shape[0]
        if T == 0:
            logger.warning(f"Skipping {fp.name}: empty sequence")
            continue
        all_labels.append(int(data["label"]))
        valid_files.append(fp)

    all_labels = np.array(all_labels)
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    logger.info(f"Valid subjects: {len(all_labels)} (neg={n_neg}, pos={n_pos})")

    # Check that we can do stratified k-fold
    if n_pos < args.n_folds or n_neg < args.n_folds:
        logger.error(
            f"Not enough samples per class for {args.n_folds}-fold CV "
            f"(need >= {args.n_folds} per class, got pos={n_pos}, neg={n_neg})"
        )
        sys.exit(1)

    # Log sequence length statistics
    seq_lens = []
    all_mask_count = 0
    for fp in valid_files:
        data = torch.load(fp, map_location="cpu", weights_only=False)
        T = data["acoustic_features"].shape[0]
        seq_lens.append(T)
        if data["mask"].all():
            all_mask_count += 1
    seq_lens = np.array(seq_lens)
    logger.info(
        f"Sequence lengths: min={seq_lens.min()}, max={seq_lens.max()}, "
        f"mean={seq_lens.mean():.1f}, median={np.median(seq_lens):.0f}"
    )
    if all_mask_count > 0:
        logger.warning(
            f"{all_mask_count} subjects have ALL windows masked (interviewer only)"
        )

    # ── Stratified K-Fold ──
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )

    fold_results = []
    pooled_preds = []
    pooled_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_files, all_labels)):
        train_files = [valid_files[i] for i in train_idx]
        val_files_fold = [valid_files[i] for i in val_idx]

        train_dataset = AnhedoniaWindowDataset(train_files)
        val_dataset = AnhedoniaWindowDataset(val_files_fold)

        fold_result = train_one_fold(
            fold, train_dataset, val_dataset, args, device, logger
        )
        fold_results.append(fold_result)
        pooled_preds.extend(fold_result["val_predictions"])
        pooled_labels.extend(fold_result["val_labels"])

    # ── Pooled metrics ──
    pooled_preds = np.array(pooled_preds)
    pooled_labels = np.array(pooled_labels)

    pooled_auc = compute_auc_safe(pooled_labels, pooled_preds)
    ci_lo, ci_hi = bootstrap_ci(pooled_labels, pooled_preds, seed=args.seed)

    precision, recall, _ = precision_recall_curve(pooled_labels, pooled_preds)
    pooled_auc_pr = auc(recall, precision) if len(np.unique(pooled_labels)) == 2 else 0.0

    fpr, tpr, thresholds = roc_curve(pooled_labels, pooled_preds)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]
    pooled_bal_acc = balanced_accuracy_score(
        pooled_labels, (pooled_preds >= best_thresh).astype(int)
    )

    # Per-fold summary
    fold_aucs = [r["val_auc_roc"] for r in fold_results]
    fold_gaps = [r["auc_gap"] for r in fold_results]

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    mean_gap = np.mean(fold_gaps)

    # ── Results dict ──
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "VoiceTransformer",
        "task": "anhedonia_binary",
        "device": str(device),
        "n_subjects": int(len(all_labels)),
        "class_balance": {"neg": int(n_neg), "pos": int(n_pos)},
        "sequence_stats": {
            "min": int(seq_lens.min()),
            "max": int(seq_lens.max()),
            "mean": round(float(seq_lens.mean()), 1),
            "median": int(np.median(seq_lens)),
        },
        "hyperparameters": {
            "d_model": 256,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 512,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": 0.01,
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "early_stopping_patience": 15,
            "scheduler": "CosineAnnealingWarmRestarts(T_0=20)",
        },
        "n_folds": args.n_folds,
        "per_fold": fold_results,
        "pooled_metrics": {
            "auc_roc": round(pooled_auc, 4),
            "auc_roc_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            "auc_pr": round(pooled_auc_pr, 4),
            "balanced_accuracy": round(pooled_bal_acc, 4),
            "optimal_threshold": round(float(best_thresh), 4),
        },
        "fold_summary": {
            "auc_roc_mean": round(mean_auc, 4),
            "auc_roc_std": round(std_auc, 4),
            "auc_roc_per_fold": [round(a, 4) for a in fold_aucs],
            "auc_gap_mean": round(mean_gap, 4),
            "auc_gap_per_fold": [round(g, 4) for g in fold_gaps],
        },
        "overfitting_diagnostics": {
            "mean_train_test_gap": round(mean_gap, 4),
            "max_gap": round(max(fold_gaps), 4),
            "assessment": (
                "SEVERE OVERFITTING" if mean_gap > 0.20
                else "MODERATE OVERFITTING" if mean_gap > 0.10
                else "ACCEPTABLE"
            ),
        },
    }

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Transformer for Anhedonia Classification"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./transformer_data",
        help="Directory containing per-subject .pt files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./transformer_results",
        help="Directory for output JSON and logs"
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("=" * 70)
    logger.info("MULTIMODAL TRANSFORMER — ANHEDONIA CLASSIFICATION")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)
    logger.info(f"Args: {vars(args)}")

    t0 = time.time()
    results = run_cross_validation(args, logger)
    elapsed = time.time() - t0
    results["runtime_seconds"] = round(elapsed, 1)

    # Save results
    output_json = output_dir / "transformer_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved: {output_json}")

    # ── Summary ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    pm = results["pooled_metrics"]
    fs = results["fold_summary"]
    od = results["overfitting_diagnostics"]
    logger.info(f"  Pooled AUC-ROC:     {pm['auc_roc']:.4f}  "
                f"[{pm['auc_roc_ci95'][0]:.4f}, {pm['auc_roc_ci95'][1]:.4f}]")
    logger.info(f"  Pooled AUC-PR:      {pm['auc_pr']:.4f}")
    logger.info(f"  Pooled Bal. Acc:    {pm['balanced_accuracy']:.4f}")
    logger.info(f"  Mean fold AUC-ROC:  {fs['auc_roc_mean']:.4f} ± {fs['auc_roc_std']:.4f}")
    logger.info(f"  Per-fold AUCs:      {fs['auc_roc_per_fold']}")
    logger.info(f"  Overfit gap (mean): {od['mean_train_test_gap']:+.4f}  → {od['assessment']}")
    logger.info(f"  Runtime:            {elapsed:.1f}s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
