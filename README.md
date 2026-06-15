# Sea Surface Temperature Forecasting — Indian Ocean

**7‑day rolling SST forecasts · ConvLSTM · Amazon Chronos · IBM Granite TSFM**  
*Arabian Sea & Laccadive Sea · 5°N–20°N, 60°E–72°E · 0.25° resolution · 60×48 grid*

> Research internship project at **INCOIS** (Indian National Centre for Ocean Information Services)  
> Author: **G. Dhanush** — ICFAI Foundation for Higher Education (IFHE)

---

## What This Project Does

We predict **Sea Surface Temperature** 7 days into the future over a 60×48 grid covering the Laccadive Sea and Arabian Sea using three fundamentally different approaches:

| Model | Type | Parameters | Approach |
|-------|------|-----------|----------|
| **ConvLSTM** | Custom deep learning | ~300K (trained) | CNN + LSTM hybrid, trained from scratch |
| **Amazon Chronos** | Foundation model | 200M (frozen) | Transformer, zero-shot inference |
| **IBM Granite TSFM** | Foundation model | 71K (frozen) | MLP-Mixer, zero-shot inference |

The evaluation covers a **90‑day rolling forecast** window (Jan–Mar 2026) with a rigorous **Five‑Gate framework**. All models are validated independently against **37 in‑situ Argo float profiles**.

---

## Architecture

### A. ConvLSTM

The custom ConvLSTM follows the formulation of Shi et al. (NeurIPS 2015), where convolutional operations replace matrix multiplication inside the LSTM gate structure.

**Cell structure:**
```
input + hidden → Conv2D(3×3) → split → [i, f, o, g] → hidden, cell output
```
- 2 stacked `ConvLSTMCell` layers with `hidden_dim = 64`
- 3×3 kernels with padding=1 (spatial dimensions preserved)
- Standard LSTM gating: input (i), forget (f), output (o), cell update (g)
- Dropout2D(0.15) on hidden states

**Network architecture:**
- **Input:** 4 channels — SST anomaly (normalized), long-term daily mean, latitude grid, longitude grid
- **Cells:** `cell1` (4→64), `cell2` (64→64) — processes 60 timesteps sequentially
- **Neck:** Conv2D(64→64, 3×3) + ReLU + GroupNorm(4 groups)
- **Head:** Conv2D(64→7, 3×3) — outputs all 7 forecast horizons at once

**Post-processing chain:**
1. Per-horizon bias correction (fitted on validation)
2. Adaptive drift correction (7-day window, capped at ±0.20°C)
3. 5-day rolling mean smoothing

### B. Amazon Chronos Pipeline

**Model:** `amazon/chronos-t5-base` — transformer encoder-decoder, 200M parameters, pre-trained on thousands of diverse time-series datasets.

**Spatial propagation (beta-map):**
1. Point forecast made at target pixel (8.0°N, 67.0°E) only
2. Beta-map = covariance(target, grid_cell) / variance(target), computed over training set
3. Spatial anomaly = `anom_ctx_last + beta_map × (point_pred - anom_ctx_last_target)`

**Post-processing chain:**
1. Per-horizon bias correction (validation set, per horizon)
2. Ridge residual corrector (independent Ridge per horizon, alpha=1.0)
3. Amplitude calibration (per-horizon scaling factors, clipped 0.85–1.20)
4. **PostGain slope correction** (g = 1.040)

### C. IBM Granite Pipeline

**Model:** `ibm-granite/granite-timeseries-ttm-r2` — MLP-Mixer based architecture, 71K parameters, pre-trained on diverse time-series data.

Pipeline is identical to Chronos:
- Zero-shot inference with frozen weights
- Beta-map spatial propagation (same method)
- Per-horizon bias + Ridge residual + Amplitude calibration
- **PostGain slope correction** (g = 1.020 — smaller gain because Granite baseline slope is closer to target)

### D. PostGain Slope Correction

Foundation models accurately predict the **direction** of temperature change but systematically under-predict **magnitude** (slope < 0.94). PostGain corrects this with a single multiplicative gain:

```
y_corrected = g × y_pred + shift
```

Where:
- `g` is found by grid search over `[1.0, 1.30]` in 16 steps
- The gain that achieves `slope(g·ŷ, y) ≥ 0.94` with lowest RMSE is selected
- A mean-shift term preserves the original mean

**Results:**
- Granite: g = 1.020, slope = 0.9436 ✅
- Chronos: g = 1.040, slope = 0.9412 ✅

This is a **retraining-free** correction — no gradient updates, no backprop, just 1 line of math.

### E. Five‑Gate Evaluation Framework

| Gate | Metric | Threshold | Why It Matters |
|------|--------|-----------|---------------|
| 1 | Overall RMSE | < 0.1466°C | Global accuracy |
| 2 | February RMSE | < 0.2093°C | Monsoon transition (hardest month) |
| 3 | March RMSE | ≤ 0.1003°C | Pre‑monsoon baseline |
| 4 | Big error days (≥0.20°C) | ≤ 12 days | Extreme event reliability |
| 5 | Slope (pred vs observed) | 0.94–1.00 | Amplitude fidelity — critical for operational use |

A model must pass **all five** to be operationally compliant.

---

## Quick Results

| Rank | Model | RMSE | Slope | Gates | PostGain |
|------|-------|------|-------|-------|----------|
| 1 | **Granite PostGain** ★ | **0.1196°C** | **0.9436** | **5/5** | 1.020 |
| 2 | Chronos PostGain det | 0.1200°C | 0.9488 | 5/5 | 1.040 |
| 3 | Chronos PostGain | 0.1205°C | 0.9412 | 5/5 | 1.040 |
| 4 | ConvLSTM baseline | 0.1417°C | 0.9408 | 5/5 | — |

Granite 87 is the **first foundation model ever** to pass all 5 gates.

---

## Argo Float Validation — Real Ocean Data

37 independent Argo float profiles (Jan–Feb 2026) matched to nearest grid cell, SST extracted at minimum pressure per profile.

| Model | RMSE | MAE | Pearson R | R² | Slope |
|-------|------|-----|-----------|-----|-------|
| **ConvLSTM** | **0.324°C** | **0.262°C** | **0.971** | **0.943** | 0.899 |
| Granite TSFM | 0.394°C | 0.301°C | 0.959 | 0.920 | 0.892 |
| Chronos t5‑base | 0.418°C | 0.322°C | 0.955 | 0.911 | 0.914 |

ConvLSTM wins on real‑world accuracy — **17.7% better RMSE** than Granite against independent in‑situ measurements.

---

## Project Structure

```
├── 📄 69_convlstm_rolling_7day_fixed.py        ConvLSTM baseline
├── 📄 86_spatial_chronos_only.py               Chronos + PostGain
├── 📄 87_spatial_granite_only.py              ★ Champion model
├── 📄 88_spatial_chronos_only_deterministic.py  Deterministic variant
├── 📄 model_stage2_best.pt                     Trained ConvLSTM weights (1.9 MB)
├── 📄 README.txt                               Full technical reference
│
├── 📁 docs/                  13 documentation files
│   ├── manuscript-dhanush.md          IEEE‑style research paper
│   ├── FINAL_SUBMISSION_REPORT.md     Full project report (17 phases)
│   ├── BOOK_CHAPTER.md                Academic book chapter
│   ├── EXECUTIVE_SUMMARY.md           One‑page overview
│   └── ... (QUICK_REFERENCE, MODEL_COMPARISON, SCRIPT_INDEX, etc.)
│
├── 📁 validation_data/       Argo float spatial validation
│   ├── build_argo_validation_sets.py
│   ├── argo_filter_to_master.py
│   └── validate_argo_spatial_models.py
│
├── 📁 model_comparison/      Multi‑model comparison plots
│   └── model_comparison_kaggle.py
│
└── 📁 input_datasets/        SST data (Git LFS)
    ├── DATASET_MAP.md
    └── master-harry-appended/
        ├── master_region_data_new.npy        179 MB
        └── master_region_anomalies_new.npy   179 MB
```

---

## How to Run

```bash
# Clone
git clone https://github.com/dhanushofc/SatelliteGAN.git
cd SatelliteGAN

# Pull LFS data
git lfs pull

# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib scipy chronos-forecasting tsfm_public netCDF4 openpyxl

# Run models (in order)
python 69_convlstm_rolling_7day_fixed.py
python 86_spatial_chronos_only.py
python 87_spatial_granite_only.py
python 88_spatial_chronos_only_deterministic.py

# Argo validation
cd validation_data
python build_argo_validation_sets.py
python argo_filter_to_master.py
python validate_argo_spatial_models.py

# Comparison plots
cd ../model_comparison
python model_comparison_kaggle.py
```

Outputs go to `outputs/<script_name>/` — rolling predictions CSV, monthly summary, and plots.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | ConvLSTM model & inference |
| `chronos-forecasting` | Amazon Chronos foundation model |
| `tsfm_public` | IBM Granite foundation model |
| `numpy`, `pandas` | Data processing |
| `scikit‑learn` | Ridge regression, metrics |
| `matplotlib`, `scipy` | Visualization |
| `netCDF4` | Argo reanalysis (NetCDF) |
| `openpyxl` | Argo Excel input |

---

## References

- Shi et al., *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting*, NeurIPS 2015
- Ansari et al., *Chronos: Learning the Language of Time Series*, arXiv:2403.07815, 2024
- IBM Research, *Granite Time‑Series Foundation Model*, `ibm-granite/granite-timeseries-ttm-r2`
- Hu et al., *LoRA: Low‑Rank Adaptation of Large Language Models*, ICLR 2022
- Reynolds et al., *Daily High‑Resolution Blended Analyses for SST*, J. Climate 2007

---

*Project completed May 2026 · INCOIS, Hyderabad · ICFAI Foundation for Higher Education*
