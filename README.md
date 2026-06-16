# NSG — Dopaminergic Voice: Project Repository

> **The Dopaminergic Voice:** Acoustic Prosody vs. Ventral Striatal BOLD for Anhedonia Classification

🔗 **GitHub:** [czhou732/ClinicalWhisper](https://github.com/czhou732/ClinicalWhisper) (private)

## Project Structure

```
NSG_Dopaminergic_Voice/
├── Stream_A/     ← DAIC-WOZ: Whisper + OpenSMILE acoustic extraction (Lily)
├── Stream_B/     ← ds000030: fMRI preprocessing + ROI analysis (Dora)
├── Fusion/       ← Combined classification + ablation
├── Data_Dummy/   ← Synthetic test data (safe for Colab)
├── Data_Real/    ← GITIGNORED — real DAIC-WOZ + ds000030 data
└── Results/      ← Output CSVs, figures, SHAP plots
```

## Data Privacy Protocol

> ⚠️ **DAIC-WOZ is restricted clinical data.** Never upload real participant audio/transcripts to Google Drive, Colab, or any public service.

1. **Develop** on Google Colab with dummy data (`Data_Dummy/`)
2. **Run** on local machine or USC CARC with real data (`Data_Real/`)
3. **Export** only aggregate results (CSVs, figures) — never raw audio

## Quick Start (Colab)

1. Open `Stream_A/01_Stream_A_Whisper_Transcription.ipynb` in Google Colab
2. The notebook auto-installs dependencies and runs on `Data_Dummy/`
3. Results → `Results/stream_a_features.csv`

