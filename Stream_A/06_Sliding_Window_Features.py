#!/usr/bin/env python3
"""
06_Sliding_Window_Features.py — Temporal Feature Extraction for Transformer
===========================================================================
Replaces whole-interview eGeMAPS aggregation with sliding-window features.

Pipeline:
    DAIC-WOZ audio → OpenSMILE eGeMAPS v02 LLDs (10ms frames)
    → 5s windows, 50% overlap
    → Per-window: mean + std of 25 LLD channels = 50D
    → MASK token for windows with no Whisper-detected participant speech
    → Save as .pt tensor per subject

MASK token strategy (ClinicalWhisper-consistent):
    Uses Whisper large-v3 transcript timestamps to determine which windows
    contain participant speech. Windows with no overlapping Whisper segments
    are treated as interviewer speech or silence → MASK token. This avoids
    dependence on ground-truth transcripts or Pyannote diarization.

Usage:
    python 06_Sliding_Window_Features.py \
        --audio_dir /path/to/DAIC-WOZ/audio \
        --transcript_dir /path/to/whisper_transcripts \
        --label_csv /path/to/full_scale_features_prereg.csv \
        --output_dir ./transformer_data

Dependencies:
    pip install opensmile librosa soundfile torch tqdm
"""

import os
import sys
import glob
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────

SAMPLE_RATE = 16000

# Core eGeMAPS v02 LLD channels (25 channels)
# These are the low-level descriptors before functional aggregation.
# Selected to capture the prosodic dimensions relevant to anhedonia:
#   - Pitch dynamics (F0)          → VTA-motor cortex projections
#   - Energy (loudness)            → vocal effort / psychomotor retardation
#   - Voice quality (jitter, shimmer, HNR) → laryngeal control
#   - Spectral (MFCC 1-4, flux)   → articulatory complexity
#   - Formants (F1-F3)            → speech production quality
#   - Spectral shape              → timbral characteristics
LLD_CHANNELS = [
    "F0semitoneFrom27.5Hz_sma3nz",
    "loudness_sma3",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    "spectralFlux_sma3",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz",
    "F2bandwidth_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz",
    "F3bandwidth_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
    "alphaRatioV_sma3nz",
    "hammarbergIndexV_sma3nz",
    "slopeV0-500_sma3nz",
    "slopeV500-1500_sma3nz",
]

N_CHANNELS = len(LLD_CHANNELS)  # 25
FEATURES_PER_WINDOW = N_CHANNELS * 2  # mean + std = 50


# ── LLD Extraction ────────────────────────────────────────────────────

def extract_lld_dataframe(audio_path: str) -> pd.DataFrame:
    """
    Extract frame-level eGeMAPS v02 Low-Level Descriptors.
    
    Returns DataFrame with ~10ms frame resolution, one row per frame.
    Columns are the LLD feature names.
    """
    import opensmile

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    try:
        df = smile.process_file(audio_path)
        return df
    except Exception as e:
        print(f"  ⚠ OpenSMILE LLD extraction failed: {e}")
        return pd.DataFrame()


def load_whisper_segments(transcript_dir: str) -> dict:
    """
    Load Whisper transcript segments for all subjects.
    
    Each JSON file contains {text, segments: [{start, end, text}], language}.
    Returns dict mapping subject_id → list of (start_sec, end_sec) tuples
    representing when participant speech was detected by Whisper.
    """
    transcripts = {}
    transcript_path = Path(transcript_dir)
    
    for jf in sorted(transcript_path.glob("*.json")):
        subject_id = jf.stem.replace("_AUDIO", "").replace("_P", "").split("_")[0]
        try:
            with open(jf) as f:
                data = json.load(f)
            segments = [
                (float(seg["start"]), float(seg["end"]))
                for seg in data.get("segments", [])
                if seg.get("text", "").strip()
            ]
            transcripts[subject_id] = segments
        except Exception as e:
            print(f"  ⚠ Failed to load transcript {jf.name}: {e}")
    
    return transcripts


def _is_participant_window_whisper(
    start_sec: float,
    end_sec: float,
    whisper_segments: list,
    min_speech_sec: float = 0.5,
) -> bool:
    """
    Determine if a window contains participant speech using Whisper timestamps.
    
    A window is 'participant' if it overlaps with Whisper-detected speech
    segments for at least min_speech_sec seconds. This is ClinicalWhisper-
    consistent: no ground-truth transcripts or Pyannote needed.
    
    Args:
        start_sec: Window start time
        end_sec: Window end time
        whisper_segments: List of (start, end) from Whisper transcription
        min_speech_sec: Minimum overlap to count as participant speech
    """
    if whisper_segments is None:
        return True  # No transcripts → assume all participant
    
    total_speech = 0.0
    for seg_start, seg_end in whisper_segments:
        overlap_start = max(start_sec, seg_start)
        overlap_end = min(end_sec, seg_end)
        overlap_dur = max(0, overlap_end - overlap_start)
        total_speech += overlap_dur
    
    return total_speech >= min_speech_sec


# ── Windowing ─────────────────────────────────────────────────────────

def compute_window_features(
    lld_df: pd.DataFrame,
    whisper_segments: list,
    window_sec: float = 5.0,
    overlap: float = 0.5,
    frame_period_sec: float = 0.01,
) -> dict:
    """
    Compute per-window acoustic features with MASK tokens.
    
    Args:
        lld_df: Frame-level LLD DataFrame from OpenSMILE
        whisper_segments: List of (start, end) from Whisper transcription
        window_sec: Window duration in seconds
        overlap: Overlap fraction (0.5 = 50%)
        frame_period_sec: Frame period (0.01s for OpenSMILE default)
    
    Returns:
        dict with:
            'features': Tensor [T, 50] — per-window acoustic features
            'mask': Tensor [T] — True for interviewer/masked windows
            'window_times': list of (start_sec, end_sec) tuples
            'n_windows': int
    """
    # Resolve available LLD columns
    available_cols = lld_df.columns.tolist()
    channel_cols = []
    for ch in LLD_CHANNELS:
        matches = [c for c in available_cols if ch in c]
        if matches:
            channel_cols.append(matches[0])
        else:
            channel_cols.append(None)

    n_frames = len(lld_df)
    total_duration = n_frames * frame_period_sec

    window_frames = int(window_sec / frame_period_sec)
    stride_frames = int(window_frames * (1.0 - overlap))

    if stride_frames < 1:
        stride_frames = 1

    windows_features = []
    windows_mask = []
    window_times = []

    for start_frame in range(0, n_frames - window_frames + 1, stride_frames):
        end_frame = start_frame + window_frames
        start_sec = start_frame * frame_period_sec
        end_sec = end_frame * frame_period_sec

        # Determine if this window has participant speech (via Whisper)
        is_participant = _is_participant_window_whisper(
            start_sec, end_sec, whisper_segments
        )

        if not is_participant:
            # MASK token — interviewer speaking
            windows_mask.append(True)
            windows_features.append(np.zeros(FEATURES_PER_WINDOW))
        else:
            # Extract participant acoustic features for this window
            window_df = lld_df.iloc[start_frame:end_frame]
            feat_vec = _compute_window_stats(window_df, channel_cols)
            windows_features.append(feat_vec)
            windows_mask.append(False)

        window_times.append((round(start_sec, 3), round(end_sec, 3)))

    if not windows_features:
        # Edge case: no valid windows
        return {
            "features": torch.zeros(1, FEATURES_PER_WINDOW),
            "mask": torch.ones(1, dtype=torch.bool),
            "window_times": [(0.0, window_sec)],
            "n_windows": 1,
        }

    features_tensor = torch.tensor(
        np.array(windows_features), dtype=torch.float32
    )
    mask_tensor = torch.tensor(windows_mask, dtype=torch.bool)

    return {
        "features": features_tensor,
        "mask": mask_tensor,
        "window_times": window_times,
        "n_windows": len(windows_features),
    }




def _compute_window_stats(
    window_df: pd.DataFrame,
    channel_cols: list,
) -> np.ndarray:
    """
    Compute mean and std for each LLD channel within a window.
    
    Returns: np.array of shape [50] — [mean_ch1, std_ch1, mean_ch2, std_ch2, ...]
    """
    stats = []
    for col in channel_cols:
        if col is not None and col in window_df.columns:
            values = window_df[col].values
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                stats.append(np.mean(valid))
                stats.append(np.std(valid))
            else:
                stats.extend([0.0, 0.0])
        else:
            stats.extend([0.0, 0.0])

    return np.array(stats, dtype=np.float32)


# ── Label Loading ─────────────────────────────────────────────────────

def load_labels(label_csv: str) -> dict:
    """
    Load anhedonia labels from the existing feature CSV.
    
    The full_scale_features_prereg.csv contains metadata columns including
    anhedonia_binary, PHQ8_Score, etc.
    """
    df = pd.read_csv(label_csv)
    labels = {}
    for _, row in df.iterrows():
        sid = str(row["subject_id"]).replace("_P", "")
        if pd.notna(row.get("anhedonia_binary")):
            labels[sid] = {
                "anhedonia_binary": int(row["anhedonia_binary"]),
                "anhedonia_sum": float(row.get("anhedonia_sum", 0)),
                "PHQ8_Binary": int(row.get("PHQ8_Binary", 0)) if pd.notna(row.get("PHQ8_Binary")) else 0,
                "PHQ8_Score": float(row.get("PHQ8_Score", 0)) if pd.notna(row.get("PHQ8_Score")) else 0,
                "split": str(row.get("split", "")),
            }
    return labels


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract sliding-window eGeMAPS features for transformer input"
    )
    parser.add_argument(
        "--audio_dir", type=str, required=True,
        help="Path to DAIC-WOZ audio files (*.wav)",
    )
    parser.add_argument(
        "--label_csv", type=str,
        default="full_scale_features_prereg.csv",
        help="Path to feature CSV with anhedonia labels",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./transformer_data",
        help="Directory to save per-subject .pt files",
    )
    parser.add_argument(
        "--transcript_dir", type=str, default=None,
        help="Path to Whisper transcript JSON files (for MASK token detection)",
    )
    parser.add_argument(
        "--window_sec", type=float, default=5.0,
        help="Window duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Window overlap fraction (default: 0.5 = 50%%)",
    )
    parser.add_argument(
        "--max_subjects", type=int, default=None,
        help="Limit number of subjects (for testing)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SLIDING-WINDOW eGeMAPS FEATURE EXTRACTION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Window: {args.window_sec}s, Overlap: {args.overlap * 100:.0f}%")
    print(f"Features per window: {FEATURES_PER_WINDOW} (25 channels × 2 stats)")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels = load_labels(args.label_csv)
    print(f"\nLabels loaded: {len(labels)} subjects with anhedonia annotations")
    n_pos = sum(1 for v in labels.values() if v["anhedonia_binary"] == 1)
    print(f"  Class balance: {n_pos} positive, {len(labels) - n_pos} negative")

    # Find audio files
    audio_files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not audio_files:
        # Try subdirectories (DAIC-WOZ structure: 300_P/300_AUDIO.wav)
        audio_files = sorted(glob.glob(os.path.join(args.audio_dir, "*", "*.wav")))
    
    if not audio_files:
        print(f"⚠ No .wav files found in {args.audio_dir}")
        sys.exit(1)

    if args.max_subjects:
        audio_files = audio_files[:args.max_subjects]
    print(f"Audio files found: {len(audio_files)}")

    # Load Whisper transcripts for MASK detection
    whisper_segments = {}
    if args.transcript_dir:
        print("\nLoading Whisper transcripts for MASK detection...")
        whisper_segments = load_whisper_segments(args.transcript_dir)
        print(f"  Loaded {len(whisper_segments)} transcripts")
    else:
        print("\n⚠ No --transcript_dir provided. All windows treated as participant.")

    # Process subjects
    results = []
    errors = []
    saved_count = 0

    for audio_path in tqdm(audio_files, desc="Processing subjects"):
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        # Extract numeric ID: "300_AUDIO" → "300", "300_P" → "300"
        subject_id = basename.replace("_AUDIO", "").replace("_P", "").split("_")[0]

        try:
            # Step 1: Extract frame-level LLDs
            lld_df = extract_lld_dataframe(audio_path)
            if lld_df.empty:
                errors.append({"subject": subject_id, "error": "Empty LLD DataFrame"})
                continue

            # Step 2: Get Whisper segments for this subject
            subj_segments = whisper_segments.get(subject_id, None)

            # Step 3: Compute windowed features
            result = compute_window_features(
                lld_df, subj_segments,
                window_sec=args.window_sec,
                overlap=args.overlap,
            )

            # Step 4: Attach label
            label_info = labels.get(subject_id, None)
            if label_info is None:
                # Subject has no anhedonia label — skip
                continue

            # Step 5: Save .pt file
            save_dict = {
                "subject_id": subject_id,
                "acoustic_features": result["features"],       # [T, 50]
                "mask": result["mask"],                         # [T]
                "window_times": result["window_times"],
                "n_windows": result["n_windows"],
                "label": label_info["anhedonia_binary"],
                "metadata": label_info,
            }

            pt_path = output_dir / f"{subject_id}.pt"
            torch.save(save_dict, pt_path)
            saved_count += 1

            n_masked = result["mask"].sum().item()
            n_total = result["n_windows"]
            results.append({
                "subject_id": subject_id,
                "n_windows": n_total,
                "n_masked": n_masked,
                "n_participant": n_total - n_masked,
                "pct_masked": round(n_masked / n_total * 100, 1) if n_total > 0 else 0,
                "label": label_info["anhedonia_binary"],
            })

        except Exception as e:
            errors.append({"subject": subject_id, "error": str(e)})
            tqdm.write(f"  ✗ {subject_id}: {e}")

    # Save manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "window_sec": args.window_sec,
            "overlap": args.overlap,
            "features_per_window": FEATURES_PER_WINDOW,
            "n_channels": N_CHANNELS,
            "channels": LLD_CHANNELS,
            "mask_source": "whisper_transcripts" if args.transcript_dir else "none",
        },
        "subjects": results,
        "errors": errors,
        "summary": {
            "total_processed": len(results),
            "total_errors": len(errors),
            "total_saved": saved_count,
            "class_0": sum(1 for r in results if r["label"] == 0),
            "class_1": sum(1 for r in results if r["label"] == 1),
            "mean_windows": round(np.mean([r["n_windows"] for r in results]), 1) if results else 0,
            "mean_pct_masked": round(np.mean([r["pct_masked"] for r in results]), 1) if results else 0,
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Subjects saved: {saved_count}")
    print(f"  Errors: {len(errors)}")
    if results:
        print(f"  Mean windows per subject: {manifest['summary']['mean_windows']}")
        print(f"  Mean % masked (interviewer): {manifest['summary']['mean_pct_masked']}%")
        print(f"  Class 0: {manifest['summary']['class_0']}")
        print(f"  Class 1: {manifest['summary']['class_1']}")
    print(f"  Output: {output_dir}")
    print(f"  Manifest: {manifest_path}")

    if errors:
        print(f"\n⚠ Errors:")
        for e in errors[:10]:
            print(f"  • {e['subject']}: {e['error']}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
