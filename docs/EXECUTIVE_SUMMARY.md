# INCOIS Internship - Executive Summary

## Project: SST Forecasting System
**Author:** Ginkala Dhanush  
**Institution:** FST [IFHE, ICFAI], CSE Batch 2026  
**Date:** May 19, 2026

---

## Project Overview

This project developed a comprehensive Sea Surface Temperature (SST) forecasting system using three model families:
1. **ConvLSTM** - Custom deep learning model preserving spatial relationships
2. **Amazon Chronos** - Transformer-based pre-trained forecasting model
3. **IBM Granite TSFM** - Time-series foundation model (TTM architecture)

Four evaluation modes tested per foundation model:
- **Zero-shot**: Pure pretrained inference
- **Few-shot**: Post-hoc Ridge residual + amplitude calibration
- **LoRA fine-tuning**: PEFT adapters trained on SST data
- **PostGain slope correction**: Zero-shot + post-hoc with gain multiplier (NEW — resolves slope issue)

## Data Scope

- **Spatial Grid**: 60×48 pixels (5°N-20°N, 60°E-72°E)
- **Time Period**: September 1, 1981 - April 7, 2026 (16,290 days)
- **Target Location**: 8.0°N, 67.0°E (Laccadive Sea)
- **Forecast Horizons**: 1-7 days (rolling)
- **Evaluation Period**: Jan-Mar 2026 (90 days)

---

## Final Results

### Granite PostGain (Script 87) — NEW SINGLE-MODEL CHAMPION

| Metric | Result | Target | Status |
|--------|--------|-------|--------|
| Overall RMSE | 0.1196°C | < 0.1466°C | ✅ PASS |
| February RMSE | 0.1704°C | < 0.2093°C | ✅ PASS |
| March RMSE | 0.0857°C | ≤ 0.1003°C | ✅ PASS |
| Big Errors (|e|≥0.20) | 9 | ≤ 12 | ✅ PASS |
| Slope | 0.9436 | [0.94, 1.00] | ✅ PASS |
| PostGain | 1.020 | - | - |

**Gates Passed: 5/5** ✓ Best Single-Model

### Chronos PostGain Deterministic (Script 88)

| Metric | Result | Target | Status |
|--------|--------|-------|--------|
| Overall RMSE | 0.1200°C | < 0.1466°C | ✅ PASS |
| February RMSE | 0.1640°C | < 0.2093°C | ✅ PASS |
| March RMSE | 0.0910°C | ≤ 0.1003°C | ✅ PASS |
| Big Errors (|e|≥0.20) | 9 | ≤ 12 | ✅ PASS |
| Slope | 0.9488 | [0.94, 1.00] | ✅ PASS |
| PostGain | 1.040 | - | - |

**Gates Passed: 5/5** ✓ Deterministic Reproducibility

### Chronos PostGain (Script 86)

| Metric | Result | Target | Status |
|--------|--------|-------|--------|
| Overall RMSE | 0.1205°C | < 0.1466°C | ✅ PASS |
| February RMSE | 0.1672°C | < 0.2093°C | ✅ PASS |
| March RMSE | 0.0902°C | ≤ 0.1003°C | ✅ PASS |
| Big Errors (|e|≥0.20) | 9 | ≤ 12 | ✅ PASS |
| Slope | 0.9412 | [0.94, 1.00] | ✅ PASS |
| PostGain | 1.040 | - | - |

**Gates Passed: 5/5** ✓ Highest Slope

### ConvLSTM (Script 69) — Robust Baseline

| Metric | Result | Target | Status |
|--------|--------|-------|--------|
| Overall RMSE | 0.1417°C | < 0.1466°C | ✅ PASS |
| February RMSE | 0.2020°C | < 0.2093°C | ✅ PASS |
| March RMSE | 0.0920°C | ≤ 0.1003°C | ✅ PASS |
| Big Errors (|e|≥0.20) | 11 | ≤ 12 | ✅ PASS |
| Slope | 0.9408 | [0.94, 1.00] | ✅ PASS |

**Gates Passed: 5/5** ✓ Production Baseline

---

## Ensemble Results (SECONDARY/OTHER)

**Note**: Ensembles are documented for completeness but are NOT the primary focus per advisor guidance.

### Point Ensemble (Script 84 W1)
- RMSE: 0.1187°C, Slope: 0.9756, Gates: 5/5
- Best RMSE across all experiments, but point-only (not spatial)

### Spatial Ensemble (Script 85 SE3)
- RMSE: 0.1187°C, Slope: 0.9147, Gates: 4/5
- Fails slope gate — beta_map propagation amplifies amplitude compression

---

## Model Data Validation — Comparing the Models with Different Datasets

### Argo Float Spatial Validation

Independent validation against 37 Argo float profiles (Jan-Feb 2026) using in-situ oceanographic measurements:

| Model | RMSE | MAE | R | R² | Slope | N |
|-------|------|-----|---|----|-------|---|
| **ConvLSTM** | **0.324°C** | **0.262** | **0.971** | **0.943** | 0.899 | 37 |
| Granite | 0.394°C | 0.301 | 0.959 | 0.920 | 0.892 | 37 |
| Chronos | 0.418°C | 0.322 | 0.955 | 0.911 | 0.914 | 37 |

**Key Finding**: ConvLSTM achieves 17.7% better RMSE than Granite and 22.5% better than Chronos against independent Argo measurements, confirming its superior generalizability.

---

## Key Findings

1. **Granite 87 is the new single-model champion**: RMSE 0.1196°C (16% better than ConvLSTM), 5/5 gates
2. **PostGain slope correction resolves amplitude compression**: All three single-model spatial configs achieve 5/5 gates
3. **PostGain gains are modest**: 1.020-1.040, indicating zero-shot predictions are close to correct amplitude
4. **Chronos F1C beats ConvLSTM on RMSE** (0.1261 vs 0.1417 = 11% improvement) — historical result
5. **ConvLSTM passes ALL 5 gates** — robust baseline
6. **Spatial context (7x7 patch) critical** for foundation model performance
7. **Few-shot with 689 validation windows** beats zero-shot by 8.6-13.5%
8. **LoRA fine-tuning works** but underperforms few-shot (L1=0.1291 vs F1C=0.1261)
9. **Ensemble point (84 W1) achieves best RMSE**: 0.1187°C, 5/5 gates — secondary investigation
10. **Argo validation confirms ConvLSTM superiority**: ConvLSTM achieves best RMSE (0.324°C) and correlation (R=0.971) against in-situ Argo float measurements

---

## Known Limitations

### Slope Issue (Historical — RESOLVED by PostGain)
- **Problem**: Amplitude under-response - models under-predict magnitude of SST changes
- **Previous best slopes**: Chronos 0.9253 (LoRA L3), Granite 0.9218 (few-shot G1A)
- **Resolution**: PostGain slope correction achieves slopes of 0.9412-0.9488 (all passing)

### Calibration Bug
- **Affected runs**: F1E, F1F, G1E, G1F
- **Issue**: Intercept not recomputed after slope clipping
- **Result**: RMSE 2.6-2.8°C (catastrophic)
- **Status**: INVALID - discard from analysis

---

## Documentation Structure

```
doumentation-things/
├── EXECUTIVE_SUMMARY.md          ← This file
├── FINAL_RESULTS_TABLE.md     ← All verified results with proofs
├── MODEL_COMPARISON.md       ← Detailed model comparison
├── QUICK_REFERENCE.md       ← Key metrics at a glance
├── VERIFICATION_PROOFS.md   ← Source file references
├── SCRIPT_INDEX.md          ← All scripts with descriptions
└── README.md               ← Folder guide
```

---

## Source Data

All raw data and outputs available in:
- **Argo validation**: `validation_data/`, outputs in `validation_data/validation_outputs/`
- **Data**: `master-npy-fromharry/master_region_data.npy`
- **ConvLSTM outputs**: `rolling window-ouputs/69_convlstm_rolling_7day_fixed_FINAL_W5_CP020_CN020_MS008/`
- **PostGain outputs**: `beats-chronos/86_spatial_chronos_only/`, `beats-chronos/87_spatial_granite_only/`, `beats-chronos/88_spatial_chronos_only_deterministic/`
- **Chronos outputs**: `4chorons-ouputs/`
- **Granite outputs**: `4granite-lagllama-outputs/`

---

## Conclusion

For formal publication, **Granite 87 (PostGain)** is recommended as the primary single-model due to full gate compliance (5/5) AND the lowest RMSE among single-model spatial pipelines (0.1196°C). **Chronos 88 (PostGain det)** provides deterministic reproducibility with 5/5 gates. **ConvLSTM 69** remains a robust baseline with 5/5 gates. Ensemble results (Scripts 84-85) are documented as a secondary investigation per advisor guidance.

---
*Last Updated: May 19, 2026*
