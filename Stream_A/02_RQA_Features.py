"""
02_RQA_Features.py — Recurrence Quantification Analysis for Stream A
=====================================================================
Extends the Dopaminergic Voice acoustic pipeline with nonlinear
dynamical features. Computes per-subject RQA biomarkers from
frame-level eGeMAPSv02 Low-Level Descriptors (LLDs).

Motivated by Samanta (arXiv:2604.26242, Apr 2026) who showed
recurrence-based vocal dynamics achieve AUC 0.689 for depression
on DAIC-WOZ using COVAREP. We replicate and extend using eGeMAPSv02
features on anhedonia-specific outcomes.

Usage (Colab or local):
    python 02_RQA_Features.py --audio_dir /path/to/DAIC-WOZ/audio \
                              --results_dir /path/to/Results/Stream_A \
                              --max_subjects 5

Dependencies:
    pip install opensmile numpy scipy pandas tqdm librosa soundfile
"""

import os
import sys
import glob
import argparse
import tempfile
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────

SAMPLE_RATE = 16000

# RQA parameters (following Samanta 2026: ε = 0.2 × std)
EPSILON_FRACTION = 0.20  # recurrence threshold as fraction of channel std
MIN_FRAMES = 50          # minimum frames for meaningful RQA

# Key eGeMAPSv02 LLD channels for RQA (subset for tractability)
# These capture the core prosodic dimensions relevant to anhedonia:
#   - F0 (pitch dynamics) — VTA-motor cortex projections
#   - Loudness (energy dynamics) — overall vocal effort
#   - Jitter/Shimmer (voice quality) — laryngeal control
#   - MFCC 1-4 (spectral dynamics) — articulatory complexity
#   - HNR (harmonic structure) — phonatory stability
RQA_CHANNELS = [
    "F0semitoneFrom27.5Hz_sma3nz",
    "loudness_sma3",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    "F1bandwidth_sma3nz",
    "F2bandwidth_sma3nz",
    "F3bandwidth_sma3nz",
    "spectralFlux_sma3",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
]


# ── RQA Core Functions ────────────────────────────────────────────────

def compute_recurrence_matrix(x: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Compute binary recurrence matrix R[i,j] = 1 iff |x[i] - x[j]| < ε.

    Uses scipy pdist for efficiency. For a 1D time series of length N,
    this produces an N×N symmetric binary matrix.
    """
    x_col = x.reshape(-1, 1)
    dists = squareform(pdist(x_col, metric="euclidean"))
    return (dists < epsilon).astype(np.uint8)


def recurrence_rate(R: np.ndarray) -> float:
    """
    RR = fraction of recurrence points in the matrix.
    RR = (1/N²) × Σ R[i,j]
    """
    N = R.shape[0]
    return float(R.sum()) / (N * N)


def determinism(R: np.ndarray, min_line: int = 2) -> float:
    """
    DET = fraction of recurrence points forming diagonal lines ≥ min_line.
    Diagonal lines indicate deterministic (predictable) dynamics.
    """
    N = R.shape[0]
    total_recurrence = R.sum()
    if total_recurrence == 0:
        return 0.0

    diag_points = 0
    # Check all diagonals (upper triangle only, then double for symmetry)
    for k in range(-N + 1, N):
        diag = np.diag(R, k)
        line_len = 0
        for val in diag:
            if val:
                line_len += 1
            else:
                if line_len >= min_line:
                    diag_points += line_len
                line_len = 0
        if line_len >= min_line:
            diag_points += line_len

    return float(diag_points) / float(total_recurrence)


def laminarity(R: np.ndarray, min_line: int = 2) -> float:
    """
    LAM = fraction of recurrence points forming vertical lines ≥ min_line.
    Vertical lines indicate laminar (trapped) states.
    """
    N = R.shape[0]
    total_recurrence = R.sum()
    if total_recurrence == 0:
        return 0.0

    vert_points = 0
    for col in range(N):
        line_len = 0
        for row in range(N):
            if R[row, col]:
                line_len += 1
            else:
                if line_len >= min_line:
                    vert_points += line_len
                line_len = 0
        if line_len >= min_line:
            vert_points += line_len

    return float(vert_points) / float(total_recurrence)


def trapping_time(R: np.ndarray, min_line: int = 2) -> float:
    """
    TT = average length of vertical lines ≥ min_line.
    Reflects how long the system remains in a given state.
    """
    N = R.shape[0]
    vert_lengths = []

    for col in range(N):
        line_len = 0
        for row in range(N):
            if R[row, col]:
                line_len += 1
            else:
                if line_len >= min_line:
                    vert_lengths.append(line_len)
                line_len = 0
        if line_len >= min_line:
            vert_lengths.append(line_len)

    return float(np.mean(vert_lengths)) if vert_lengths else 0.0


def diagonal_entropy(R: np.ndarray, min_line: int = 2) -> float:
    """
    ENTR = Shannon entropy of the distribution of diagonal line lengths.
    Higher entropy → more complex recurrence structure.
    """
    N = R.shape[0]
    line_lengths = []

    for k in range(-N + 1, N):
        diag = np.diag(R, k)
        line_len = 0
        for val in diag:
            if val:
                line_len += 1
            else:
                if line_len >= min_line:
                    line_lengths.append(line_len)
                line_len = 0
        if line_len >= min_line:
            line_lengths.append(line_len)

    if not line_lengths:
        return 0.0

    lengths = np.array(line_lengths)
    counts = np.bincount(lengths)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def compute_rqa_features(x: np.ndarray, channel_name: str) -> dict:
    """
    Compute all 5 RQA features for a single channel time series.

    Handles edge cases (constant signal, too few frames, NaN values).
    Returns dict with prefixed feature names.
    """
    prefix = f"rqa_{channel_name}"

    # Handle degenerate cases
    x_clean = x[~np.isnan(x)]
    if len(x_clean) < MIN_FRAMES:
        return {
            f"{prefix}_RR": np.nan,
            f"{prefix}_DET": np.nan,
            f"{prefix}_LAM": np.nan,
            f"{prefix}_TT": np.nan,
            f"{prefix}_ENTR": np.nan,
        }

    # Subsample if too long (>2000 frames → memory/time constraint)
    if len(x_clean) > 2000:
        indices = np.linspace(0, len(x_clean) - 1, 2000, dtype=int)
        x_clean = x_clean[indices]

    # Compute epsilon from channel standard deviation
    std = np.std(x_clean)
    if std < 1e-10:  # constant signal
        return {
            f"{prefix}_RR": 1.0,
            f"{prefix}_DET": 0.0,
            f"{prefix}_LAM": 0.0,
            f"{prefix}_TT": 0.0,
            f"{prefix}_ENTR": 0.0,
        }

    epsilon = EPSILON_FRACTION * std

    # Compute recurrence matrix
    R = compute_recurrence_matrix(x_clean, epsilon)

    return {
        f"{prefix}_RR": round(recurrence_rate(R), 6),
        f"{prefix}_DET": round(determinism(R), 6),
        f"{prefix}_LAM": round(laminarity(R), 6),
        f"{prefix}_TT": round(trapping_time(R), 4),
        f"{prefix}_ENTR": round(diagonal_entropy(R), 4),
    }


# ── Main Pipeline ─────────────────────────────────────────────────────

def extract_lld_features(audio_path: str) -> Optional[pd.DataFrame]:
    """
    Extract frame-level eGeMAPSv02 LLDs from an audio file.

    Unlike the main pipeline (which uses FeatureLevel.Functionals),
    this extracts Low-Level Descriptors — one row per frame (~10ms).
    """
    import opensmile

    smile_lld = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    try:
        df = smile_lld.process_file(audio_path)
        return df
    except Exception as e:
        print(f"    ⚠ OpenSMILE LLD extraction failed: {e}")
        return None


def process_subject_rqa(
    audio_path: str,
    diarize_pipeline=None,
) -> dict:
    """
    Full RQA pipeline for one DAIC-WOZ subject.

    Steps:
        1. Load audio (with optional diarization to isolate participant)
        2. Extract frame-level eGeMAPSv02 LLDs
        3. Compute RQA on each target channel
        4. Return dict of all RQA features
    """
    subject_id = os.path.splitext(os.path.basename(audio_path))[0]

    # Load audio
    audio_data, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Optional: diarize and keep participant speech only
    if diarize_pipeline is not None:
        try:
            diarization = diarize_pipeline(audio_path)
            speaker_durations = {}
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                dur = turn.end - turn.start
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + dur
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((turn.start, turn.end))

            if speaker_durations:
                participant = max(speaker_durations, key=speaker_durations.get)
                parts = []
                for start, end in speaker_segments[participant]:
                    s, e = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
                    parts.append(audio_data[s:e])
                if parts:
                    audio_data = np.concatenate(parts)
        except Exception as e:
            print(f"    ⚠ Diarization failed for {subject_id}: {e}")

    # Write to temp file for OpenSMILE
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio_data, SAMPLE_RATE)
        lld_df = extract_lld_features(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if lld_df is None or lld_df.empty:
        print(f"    ✗ No LLD features for {subject_id}")
        return {"subject_id": subject_id}

    # Compute RQA for each target channel
    features = {"subject_id": subject_id}
    available_cols = lld_df.columns.tolist()

    for channel in RQA_CHANNELS:
        matching = [c for c in available_cols if channel in c]
        if matching:
            col = matching[0]  # take first match
            x = lld_df[col].values
            rqa = compute_rqa_features(x, channel.replace("_sma3nz", "").replace("_sma3", ""))
            features.update(rqa)
        else:
            # Channel not found — fill NaN
            short = channel.replace("_sma3nz", "").replace("_sma3", "")
            for metric in ["RR", "DET", "LAM", "TT", "ENTR"]:
                features[f"rqa_{short}_{metric}"] = np.nan

    # Add summary statistics across all channels
    rr_values = [v for k, v in features.items() if k.endswith("_RR") and not np.isnan(v)]
    det_values = [v for k, v in features.items() if k.endswith("_DET") and not np.isnan(v)]
    lam_values = [v for k, v in features.items() if k.endswith("_LAM") and not np.isnan(v)]

    features["rqa_mean_RR"] = round(np.mean(rr_values), 6) if rr_values else np.nan
    features["rqa_mean_DET"] = round(np.mean(det_values), 6) if det_values else np.nan
    features["rqa_mean_LAM"] = round(np.mean(lam_values), 6) if lam_values else np.nan
    features["rqa_std_RR"] = round(np.std(rr_values), 6) if rr_values else np.nan

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Extract RQA features from DAIC-WOZ audio (Stream A extension)"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/content/drive/MyDrive/NSG_Dopaminergic_Voice/DAIC-WOZ/audio",
        help="Path to DAIC-WOZ audio files",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/content/drive/MyDrive/NSG_Dopaminergic_Voice/Results/Stream_A",
        help="Path to save RQA features CSV",
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Limit number of subjects (for testing)",
    )
    parser.add_argument(
        "--no_diarize",
        action="store_true",
        help="Skip speaker diarization",
    )
    args = parser.parse_args()

    # Find audio files
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.m4a"]:
        audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"⚠ No audio files found in {args.audio_dir}")
        sys.exit(1)

    files = audio_files[: args.max_subjects] if args.max_subjects else audio_files
    print(f"RQA Feature Extraction — {len(files)} subjects")
    print(f"  Audio dir: {args.audio_dir}")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Channels: {len(RQA_CHANNELS)}")
    print(f"  Features per subject: {len(RQA_CHANNELS) * 5 + 4} (5 RQA × {len(RQA_CHANNELS)} channels + 4 summary)")
    print("=" * 60)

    os.makedirs(args.results_dir, exist_ok=True)

    all_features = []
    errors = []

    for i, path in enumerate(tqdm(files, desc="RQA extraction")):
        sid = os.path.splitext(os.path.basename(path))[0]
        try:
            feats = process_subject_rqa(path, diarize_pipeline=None)
            all_features.append(feats)
            rr = feats.get("rqa_mean_RR", float("nan"))
            print(f"  [{i+1}/{len(files)}] ✓ {sid} — mean RR: {rr:.4f}")
        except Exception as e:
            errors.append({"subject": sid, "error": str(e)})
            print(f"  [{i+1}/{len(files)}] ✗ {sid} — {e}")

    if all_features:
        df = pd.DataFrame(all_features)
        out_path = os.path.join(args.results_dir, "rqa_features.csv")
        df.to_csv(out_path, index=False)
        print(f"\n✓ Saved: {out_path}")
        print(f"  Shape: {df.shape[0]} subjects × {df.shape[1]} features")

        # Print feature summary
        rqa_cols = [c for c in df.columns if c.startswith("rqa_")]
        print(f"\nRQA Feature Summary:")
        print(df[rqa_cols].describe().round(4).to_string())
    else:
        print("\n⚠ No features extracted!")

    if errors:
        print(f"\n⚠ {len(errors)} errors:")
        for e in errors:
            print(f"  • {e['subject']}: {e['error']}")


if __name__ == "__main__":
    main()
