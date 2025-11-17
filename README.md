# UBFC-Phys HRV Analysis Pipeline

A Python pipeline for extracting Heart Rate Variability (HRV) features from Blood Volume Pulse (BVP) signals using the UBFC-Phys dataset.

## Overview

This pipeline processes BVP signals through signal cleaning, peak detection, RR interval calculation, and comprehensive HRV feature extraction using NeuroKit2. It generates 93 time-domain, frequency-domain, and nonlinear HRV features suitable for physiological analysis and machine learning applications.

## Pipeline Architecture

```
BVP Signal → Clean → Detect Peaks → RR Intervals → Filter → HRV Features
```

### Core Processing Steps

1. **Signal Preprocessing**: BVP cleaning using NeuroKit2's PPG processing
2. **Peak Detection**: Systolic peak identification at 64 Hz sampling rate
3. **RR Interval Extraction**: Inter-beat interval calculation (300-2000ms physiological range)
4. **HRV Feature Extraction**: 93 features across multiple domains:
   - Time-domain: SDNN, RMSSD, pNN50, etc.
   - Frequency-domain: LF, HF, LF/HF ratio
   - Nonlinear: DFA, sample entropy, fractal dimensions
   - Poincaré: SD1, SD2, CSI, CVI

## Project Structure

```
.
├── main.py                    # Entry point for batch processing
├── pipeline/
│   ├── run.py                 # Single-file processing orchestrator
│   ├── preprocess.py          # Core BVP → HRV pipeline
│   ├── graphs.py              # Visualization generation
│   └── logger_util.py         # Logging utilities
├── ubfc_2/                    # UBFC-Phys dataset (subjects s6-s42)
│   └── s*/                    # Per-subject directories
│       └── bvp_s*_T*.csv      # BVP signal files (64 Hz)
└── output/                    # Processing results and summaries
```

## Usage

### Basic Processing

```python
from pathlib import Path
from pipeline.run import run_single_file

# Process a single BVP file
aligned_df, info, plots = run_single_file(
    file_path=Path("ubfc_2/s6/bvp_s6_T1.csv"),
    subject_id="6",
    task="T1",
    sampling_rate=64,
    save=True
)

print(f"Heart Rate: {info['avg_heart_rate']:.2f} BPM")
print(f"Features extracted: {info['features_available']}")
```

### Batch Processing

Edit the `files` list in `main.py` and run:

```bash
python main.py
```

## Output Artifacts

- **Preprocessed Features**: `preprocessed/preprocessed_s{subject}_{task}.csv`
- **Visualizations**: `graphs/s{subject}_{task}_*.png`
  - Raw and cleaned BVP signals
  - Peak detection overlay
  - RR interval distributions
  - Top HRV features
  - Feature coverage statistics

## Dependencies

- `numpy`: Numerical computing
- `pandas`: Data manipulation
- `neurokit2`: Physiological signal processing and HRV extraction
- `matplotlib`: Visualization generation

## Technical Specifications

- **Sampling Rate**: 64 Hz
- **RR Filter Range**: 300-2000 ms (30-200 BPM)
- **Feature Count**: 93 HRV indices
- **Dataset**: UBFC-Phys (Bobbia et al., 2019)

## Dataset

UBFC-Phys: A multimodal physiological dataset containing synchronized BVP, EDA, and video recordings from 56 subjects performing various tasks.

**Reference**: Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. Pattern Recognition Letters.
