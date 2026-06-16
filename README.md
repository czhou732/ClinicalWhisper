# Cross-Modal Benchmarking of Acoustic Prosody and Ventral Striatal BOLD for Depression-Related Anhedonia Classification

> A pre-registered study with the ClinicalWhisper Pipeline

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.06.08.728970-b31b1b.svg)](https://doi.org/10.64898/2026.06.08.728970)
[![OSF](https://img.shields.io/badge/OSF-Pre--Registration-blue.svg)](https://osf.io/bsvrj)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the analysis code and results for our preprint benchmarking two non-invasive modalities for anhedonia classification:

- **Stream A (Acoustic Prosody):** eGeMAPS + Recurrence Quantification Analysis features from DAIC-WOZ clinical interviews (n=189)
- **Stream B (Ventral Striatal BOLD):** Nucleus accumbens activation during the Balloon Analogue Risk Task from UCLA ds000030 (n=142)

**Key finding:** Voice-based classification (AUC = 0.65, permutation p = .032) performs comparably to fMRI (AUC = 0.59) at zero marginal cost.

## Relationship to Clinical-Whisper-Pipeline

| Repository | Purpose |
|-----------|---------|
| [`Clinical-Whisper-Pipeline`](https://github.com/czhou732/Clinical-Whisper-Pipeline) | The general-purpose tool -- Whisper transcription, speaker diarization, acoustic feature extraction, sentiment analysis. Reusable for any clinical speech dataset. |
| **This repo (`ClinicalWhisper`)** | The research project -- applies the pipeline to DAIC-WOZ and ds000030 for anhedonia classification. Contains analysis scripts, classification code, and results for the preprint. |

## Project Structure

```text
├── Stream_A/                        ← Acoustic prosody pipeline
│   ├── 02_RQA_Features.py               RQA feature extraction from audio
│   ├── 03_RQA_Classification.py         Primary classification (5-fold CV, permutation test)
│   ├── 04_SHAP_Analysis.py              Feature importance + DeLong tests
│   ├── 05_Feature_Ablation.py           Domain ablation + MI feature selection
│   ├── 06_Sliding_Window_Features.py    Sliding window for transformer
│   ├── 07_Text_Window_Alignment.py      Whisper text + SBERT alignment
│   ├── 08_Transformer_Model.py          Multimodal transformer (exploratory)
│   └── 09_Transformer_Evaluation.py     Transformer evaluation + ablation
│
├── Stream_B/                        ← fMRI pipeline
│   └── scripts/                         fMRIPrep, Nilearn ROI extraction
│
├── Fusion/                          ← Combined classification + ablation
├── Data_Dummy/                      ← Synthetic test data (safe for Colab)
├── Results/
│   ├── Stream_A/                        Classification JSONs, ROC plots, SHAP
│   └── supplementary_metrics.json       AUC-PR, demographics, severity analysis
└── scripts/                         ← Utility scripts (figure generation, OSF metrics)
```

## ML Pipeline

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Classifiers | LogReg, RF, GBT (fixed hyperparameters) | No tuning -- n too small for nested CV |
| Cross-validation | 5-fold Stratified | Preserves 22.5% positive rate per fold |
| Scaling | StandardScaler inside CV loop | No train/test leakage |
| Significance | Permutation test, 1000 perms | p = (C+1)/(N+1) per Ojala & Garriga (2010) |
| Confidence intervals | Bootstrap, 10,000 draws | Percentile method on pooled OOF predictions |
| Feature importance | Logistic regression coefficients | LinearExplainer-equivalent for standardized data |

## Results

| Feature Set | Best Classifier | AUC-ROC | Permutation p |
|------------|----------------|---------|---------------|
| eGeMAPSv02 only (447) | Random Forest | 0.630 | .049 |
| RQA only (74) | Logistic Regression | 0.584 | .144 |
| **Combined (521)** | **Gradient-Boosted Trees** | **0.649** | **.032** |

## Data Privacy

> ⚠️ **DAIC-WOZ is restricted clinical data.** Never upload participant audio/transcripts to any public service.

1. **Develop** on Google Colab with dummy data (`Data_Dummy/`)
2. **Run** on local machine or USC CARC with real data
3. **Export** only aggregate results (CSVs, figures) -- never raw audio

## Citation

```bibtex
@article{zhou2026crossmodal,
  title={Cross-Modal Benchmarking of Acoustic Prosody and Ventral Striatal
         BOLD for Depression-Related Anhedonia Classification:
         A Pre-Registered Study with the ClinicalWhisper Pipeline},
  author={Zhou, Chengdong and Wu, Lily and Xiang, Mei-Hui and Itti, Laurent},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.06.08.728970}
}
```
