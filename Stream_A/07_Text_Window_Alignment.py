#!/usr/bin/env python3
"""
07_Text_Window_Alignment.py — Align Whisper Text to Sliding Windows
====================================================================
Maps Whisper large-v3 transcription segments to the same 5s windows
used by 06_Sliding_Window_Features.py, then encodes each window's
text with Sentence-BERT for multimodal transformer input.

Pipeline:
    Whisper segments (text + timestamps)
    → Align to 5s sliding windows (from acoustic manifest)
    → Sentence-BERT encoding (all-MiniLM-L6-v2, 384D)
    → Append text_features to existing .pt files

Usage:
    python 07_Text_Window_Alignment.py \
        --transcript_dir /path/to/whisper_transcripts \
        --data_dir ./transformer_data

Dependencies:
    pip install sentence-transformers torch tqdm
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Sentence-BERT model for text encoding
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_DIM = 384


def load_whisper_transcripts(transcript_dir: str) -> dict:
    """
    Load Whisper transcription segments from JSON files.
    
    Expected format per file (from Whisper output):
    {
        "text": "full transcript...",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello"},
            {"start": 2.5, "end": 5.0, "text": "How are you"},
            ...
        ]
    }
    
    Also supports plain-text segment files (.txt) with format:
        start_sec|end_sec|text
    
    Returns: dict mapping subject_id → list of (start, end, text) tuples
    """
    transcripts = {}
    
    # Try JSON files first
    json_files = sorted(Path(transcript_dir).glob("*.json"))
    for jf in json_files:
        subject_id = jf.stem.replace("_AUDIO", "").replace("_P", "").split("_")[0]
        try:
            with open(jf) as f:
                data = json.load(f)
            
            segments = []
            for seg in data.get("segments", []):
                segments.append((
                    float(seg["start"]),
                    float(seg["end"]),
                    str(seg.get("text", "")).strip(),
                ))
            if segments:
                transcripts[subject_id] = segments
        except Exception as e:
            print(f"  ⚠ Failed to load {jf.name}: {e}")
    
    # Try TSV/CSV files if no JSON found
    if not transcripts:
        txt_files = sorted(Path(transcript_dir).glob("*.txt"))
        for tf in txt_files:
            subject_id = tf.stem.replace("_AUDIO", "").replace("_P", "").split("_")[0]
            try:
                segments = []
                with open(tf) as f:
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) >= 3:
                            segments.append((
                                float(parts[0]),
                                float(parts[1]),
                                parts[2].strip(),
                            ))
                if segments:
                    transcripts[subject_id] = segments
            except Exception as e:
                print(f"  ⚠ Failed to load {tf.name}: {e}")
    
    return transcripts


def align_text_to_windows(
    segments: list,
    window_times: list,
) -> list:
    """
    Align Whisper text segments to sliding windows.
    
    For each window (start_sec, end_sec), concatenate all text segments
    that overlap with the window. If no text overlaps, return empty string.
    
    Args:
        segments: list of (start, end, text) from Whisper
        window_times: list of (start, end) from acoustic windowing
    
    Returns:
        list of strings, one per window
    """
    window_texts = []
    
    for w_start, w_end in window_times:
        texts = []
        for seg_start, seg_end, text in segments:
            # Check overlap
            overlap_start = max(w_start, seg_start)
            overlap_end = min(w_end, seg_end)
            if overlap_end > overlap_start and text:
                texts.append(text)
        
        window_texts.append(" ".join(texts).strip())
    
    return window_texts


def encode_texts(texts: list, model) -> torch.Tensor:
    """
    Encode a list of texts with Sentence-BERT.
    
    Empty strings get zero vectors.
    
    Args:
        texts: list of strings
        model: SentenceTransformer model
    
    Returns:
        Tensor of shape [len(texts), 384]
    """
    # Separate empty and non-empty texts for efficient batching
    embeddings = torch.zeros(len(texts), SBERT_DIM)
    
    non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
    non_empty_texts = [texts[i] for i in non_empty_indices]
    
    if non_empty_texts:
        encoded = model.encode(
            non_empty_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for idx, emb in zip(non_empty_indices, encoded):
            embeddings[idx] = torch.tensor(emb, dtype=torch.float32)
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Align Whisper transcripts to sliding windows and encode with SBERT"
    )
    parser.add_argument(
        "--transcript_dir", type=str, required=True,
        help="Directory containing Whisper transcript JSON/TXT files",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./transformer_data",
        help="Directory with .pt files from 06_Sliding_Window_Features.py",
    )
    parser.add_argument(
        "--sbert_model", type=str, default=SBERT_MODEL,
        help=f"Sentence-BERT model name (default: {SBERT_MODEL})",
    )
    parser.add_argument(
        "--max_subjects", type=int, default=None,
        help="Limit number of subjects (for testing)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TEXT ALIGNMENT & SENTENCE-BERT ENCODING")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"SBERT model: {args.sbert_model}")
    print("=" * 70)

    data_dir = Path(args.data_dir)

    # Load Whisper transcripts
    print("\n[1] Loading Whisper transcripts...")
    transcripts = load_whisper_transcripts(args.transcript_dir)
    print(f"  Loaded {len(transcripts)} transcripts")

    if not transcripts:
        print("⚠ No transcripts found. Check --transcript_dir path.")
        print(f"  Searched: {args.transcript_dir}")
        sys.exit(1)

    # Load SBERT model
    print("\n[2] Loading Sentence-BERT model...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(args.sbert_model)
    print(f"  ✓ Model loaded (dim={sbert.get_sentence_embedding_dimension()})")

    # Find existing .pt files
    pt_files = sorted(data_dir.glob("*.pt"))
    if args.max_subjects:
        pt_files = pt_files[:args.max_subjects]
    print(f"\n[3] Processing {len(pt_files)} subjects...")

    aligned_count = 0
    skipped_count = 0
    stats = []

    for pt_path in tqdm(pt_files, desc="Aligning text"):
        subject_id = pt_path.stem

        # Load existing acoustic data
        data = torch.load(pt_path, weights_only=False)
        window_times = data.get("window_times", [])

        if not window_times:
            skipped_count += 1
            continue

        # Find matching transcript
        transcript = transcripts.get(subject_id, None)

        if transcript is None:
            # No transcript — fill with zeros
            data["text_features"] = torch.zeros(len(window_times), SBERT_DIM)
            data["has_text"] = False
            torch.save(data, pt_path)
            skipped_count += 1
            continue

        # Align text to windows
        window_texts = align_text_to_windows(transcript, window_times)

        # Encode with SBERT
        text_embeddings = encode_texts(window_texts, sbert)

        # Verify shape match
        assert text_embeddings.shape[0] == data["acoustic_features"].shape[0], \
            f"Shape mismatch: text {text_embeddings.shape[0]} vs acoustic {data['acoustic_features'].shape[0]}"

        # Update .pt file
        data["text_features"] = text_embeddings  # [T, 384]
        data["has_text"] = True
        data["window_texts"] = window_texts  # for debugging
        torch.save(data, pt_path)

        # Stats
        n_nonempty = sum(1 for t in window_texts if t.strip())
        stats.append({
            "subject_id": subject_id,
            "n_windows": len(window_texts),
            "n_with_text": n_nonempty,
            "pct_with_text": round(n_nonempty / len(window_texts) * 100, 1),
        })
        aligned_count += 1

    # Summary
    print(f"\n{'=' * 70}")
    print("TEXT ALIGNMENT COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Aligned: {aligned_count}")
    print(f"  Skipped (no transcript): {skipped_count}")
    if stats:
        mean_pct = np.mean([s["pct_with_text"] for s in stats])
        print(f"  Mean % windows with text: {mean_pct:.1f}%")

    # Save alignment stats
    stats_path = data_dir / "text_alignment_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "sbert_model": args.sbert_model,
            "sbert_dim": SBERT_DIM,
            "aligned": aligned_count,
            "skipped": skipped_count,
            "subjects": stats,
        }, f, indent=2)
    print(f"  Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
