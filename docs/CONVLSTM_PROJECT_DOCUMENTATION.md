# INCOIS Internship

# ConvLSTM Sea Surface Temperature Forecasting System

## Technical Documentation Report

---

**Author:** Ginkala Dhanush  
**Institution:** FST [IFHE, ICFAI], CSE Batch 2026  
**Date:** 

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Project Overview](#project-overview)
4. [Methodology](#methodology)
5. [Findings](#findings)
6. [System Architecture](#system-architecture)
7. [Model Development Timeline](#model-development-timeline)
8. [Results and Performance](#results-and-performance)
9. [Technical Specifications](#technical-specifications)
10. [Conclusion](#conclusion)
11. [Recommendations](#recommendations)
12. [Appendices](#appendices)

---

## Executive Summary

This comprehensive technical documentation report presents a detailed account of the ConvLSTM-based Sea Surface Temperature (SST) forecasting system developed during the INCOIS internship period. The project encompasses a complete machine learning solution for multi-horizon SST prediction, featuring 30+ Python scripts demonstrating progressive evolution from basic chunk-based models to sophisticated ensemble architectures.

### Key Highlights

- **Data Scope**: 16,290 days of SST observations (60×48 spatial grid)
- **Time Period**: September 1, 1981 - April 7, 2026
- **Forecast Horizons**: 7-day, 14-day, and 30-day predictions
- **Architecture**: ConvLSTM with multiple strategies (MIMO, Direct, Recursive, Branching)
- **Performance**: RMSE values ranging from 0.17°C to 0.35°C
- **Platform**: Google Colab and Kaggle notebook deployment

The system demonstrates significant improvements over traditional LSTM approaches, with ConvLSTM achieving 15-20% better prediction accuracy by preserving spatial relationships in the data through 2D convolutional operations.

### Latest Achievement (Script 69)

**Post-Processing Improvements Achieved: 21.5% RMSE Reduction**

| Metric | Before (Vanilla) | After (Improved) | Improvement |
|--------|-----------------|------------------|-------------|
| Overall RMSE | 0.2151°C | 0.1688°C | **-21.5%** |
| February RMSE | 0.2999°C | 0.2359°C | **-21.4%** |
| Big errors (≥0.20°C) | 33 days | 20 days | -13 days |
| Correlation | 0.8420 | 0.8879 | +5.5% |

All improvements achieved **without model retraining** - purely through advanced post-processing techniques.

### Latest Achievement (Chronos Integration)

**Chronos Spatial-Hybrid (A1): 4/5 Gates Achieved**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall RMSE | 0.1276°C | < 0.1466°C | ✓ PASS |
| February RMSE | 0.1752°C | < 0.2093°C | ✓ PASS |
| March RMSE | 0.0952°C | ≤ 0.1003°C | ✓ PASS |
| Big Errors | 12 days | ≤ 12 | ✓ PASS |
| Slope | 0.8974 | [0.94, 1.00] | ✗ FAIL |

**Gates Passed: 4/5** - Best Chronos variant achieved

---

## Latest Development: Amazon Chronos Integration (May 2026)

### Overview

In addition to the ConvLSTM-based forecasting system, the project has expanded to include **Amazon Chronos** - a Transformer-based weather forecasting model from AWS. This integration represents a significant architectural shift from convolutional approaches to attention-based deep learning for SST prediction.

### Chronos vs ConvLSTM Comparison

| Aspect | ConvLSTM (Script 69) | Chronos (Script 70+) |
|--------|---------------------|---------------------|
| **Architecture** | 2-layer ConvLSTM | Transformer Encoder |
| **Parameters** | ~500K | 24.4M - 200M |
| **Model Size** | Small (~500KB) | Medium-Large (24-200MB) |
| **Approach** | Convolutional LSTM | Attention-based |
| **Gates Passed** | **5/5** ✓ | **4/5** (t5-base A1) |
| **Status** | Production-ready | Evaluation complete |

### Chronos Models Evaluated

| Model | Parameters | Status |
|-------|------------|--------|
| amazon/chronos-t5-small | 24.4M | Failed (0/5 gates) |
| amazon/chronos-t5-base | 200M | **4/5 gates (A1 best)** |

### 5-Gate Evaluation System

The Chronos integration introduces a rigorous 5-gate evaluation system for production readiness:

| Gate | Metric | Target | Description |
|------|--------|--------|-------------|
| Gate 1 | Overall RMSE | < 0.1466°C | Primary accuracy metric |
| Gate 2 | February RMSE | < 0.2093°C | Monsoon transition period |
| Gate 3 | Big Errors (≥0.20°C) | ≤ 12 days | Error consistency |
| Gate 4 | Slope | [0.94, 1.00] | Amplitude response |
| Gate 5 | March RMSE | ≤ 0.1003°C | Post-monsoon stability |

**Success Criteria**: Minimum 4/5 gates must pass for production deployment.

### Chronos Performance Results

**Phase-3 ConfigA (chronos-t5-small, temp=0.7, median):**
```
Overall RMSE:  0.1840   (target: 0.1466)  ✗
Feb RMSE:      0.2253   (target: 0.2093)  ✗
Mar RMSE:      0.1582   (target: 0.1003)  ✗
Big errors:    28       (target: ≤12)     ✗
Slope:         0.8728   (target: 0.94-1.00) ✗

Gates passed: 0/5  ✗ FAILED
```

**ConvLSTM Baseline (Reference - Still Champion):**
```
Overall RMSE:  0.1417   (target: 0.1466)  ✓
Feb RMSE:      0.2020   (target: 0.2093)  ✓
Mar RMSE:      0.0920   (target: 0.1003)  ✓
Big errors:    11       (target: ≤12)     ✓
Slope:         0.9408   (target: 0.94-1.00) ✓

Gates passed: 5/5  ✓ PASS
```

### Key Insights from Chronos Evaluation

1. **Model Capacity Issue**: t5-small (24.4M params) lacks sufficient capacity for SST seasonal patterns
2. **Potential Solution**: t5-base (200M params, 8x larger) may provide the needed complexity
3. **Root Cause**: ConvLSTM spatial awareness remains superior for regional SST forecasting
4. **Current Baseline**: ConvLSTM Script 69 remains the production-ready system

---

### Phase A/B/C/D Convergence Plan

A systematic experiment plan was developed to optimize Chronos performance:

| Phase | Focus | Description |
|-------|-------|-------------|
| Phase A | Temperature & Aggregation | 4 experiments (A1-A4) with temp 0.7-0.8, median/trimmed_mean |
| Phase B | Outlier Filter Sensitivity | Vary Z threshold 2.5→3.0→3.5 |
| Phase C | Adaptive Cap/Slew Tuning | Fine-tune correction parameters |
| Phase D | Model Escalation | Upgrade to t5-base |

### Experiment Results (Phase A - chronos-t5-small)

| Config | Temp | Aggregation | RMSE | Slope | Big Errors | Gates |
|--------|------|------------|-------|-------|------------|-------|-------|
| A1 | 0.7 | median | TBD | TBD | TBD | TBD |
| A2 | 0.8 | median | TBD | TBD | TBD | TBD |
| A3 | 0.7 | trimmed_mean | TBD | TBD | TBD | TBD |
| A4 | 0.8 | trimmed_mean | TBD | TBD | TBD | TBD |

**Note**: Results pending from Kaggle execution.

---

## Introduction

### Purpose of the Project

This project was undertaken as part of the INCOIS (Indian National Centre for Ocean Information Services) internship program, with the primary objective of developing an advanced sea surface temperature forecasting system. The work builds upon an existing LSTM-based framework and aims to leverage convolutional LSTM (ConvLSTM) neural networks to capture spatial-temporal patterns in SST data more effectively.

### Scope of Documentation

This comprehensive report synthesizes all findings, technical specifications, architectural decisions, and performance metrics gathered throughout the project development cycle. The documentation serves as both a technical reference and a knowledge preservation mechanism for future iterations of the system.

### Stakeholders

The primary beneficiaries of this documentation include:

- INCOIS research supervisors and technical teams
- Future intern students working on oceanographic forecasting
- Academic institutions requiring detailed project references
- Technical teams deploying the forecasting system operationally

---

## Project Overview

### Background

Sea Surface Temperature (SST) forecasting holds critical importance in multiple domains:

1. **Marine Weather Prediction**: SST patterns directly influence atmospheric conditions and weather systems
2. **Fisheries Management**: Fish migration patterns correlate with temperature gradients
3. **Climate Research**: Long-term SST trends indicate climate change indicators
4. **Coastal Planning**: Tourism and coastal infrastructure depend on temperature patterns
5. **Disaster Management**: Early warning for marine heatwaves and cold alerts

### Problem Statement

Traditional LSTM approaches treating each grid pixel independently fail to capture spatial relationships inherent in SST data. Key limitations included:

- Loss of spatial patterns through grid flattening
- No learning of wind-driven mixing
- No capturing of mesoscale eddy dynamics
- Missing coastal current patterns
- Independent pixel predictions ignoring neighboring information

### Solution Approach

The ConvLSTM solution preserves the 2D spatial structure throughout processing, enabling:

- Direct 3×3 convolution operations capturing local patterns
- Regional relationships through convolution kernels
- Spatial pattern recognition (wind, currents, eddies)
- Unified processing of the entire grid

---

## Methodology

### Data Acquisition and Preprocessing

#### Source Data

The system utilizes two primary datasets:

| Dataset | Description | Shape |
|---------|-------------|-------|
| master_region_data.npy | SST observations | (16290, 60, 48) |
| master_region_anomalies.npy | Anomaly values | (16290, 60, 48) |

#### Temporal Coverage

- **Start Date**: September 1, 1981
- **End Date**: April 7, 2026
- **Total Observations**: 16,290 days
- **Holdout Period**: Reserved for final validation

#### Spatial Grid Details

Original grid dimensions required padding for patch decomposition:

| Parameter | Original | Padded |
|-----------|----------|--------|
| Latitude | 60 | 60 |
| Longitude | 48 | 50 |
| Resolution | 0.25° | 0.25° |
| Latitude Range | 5°N - 20°N | 5°N - 20°N |
| Longitude Range | 60°E - 72°E | 60°E - 72°E |

#### Preprocessing Pipeline

The preprocessing pipeline implements the following steps:

1. **Data Loading**: NumPy array loading with float32 conversion
2. **Padding**: Edge mode padding (48 → 50 columns)
3. **LTDM Computation**: Long-term daily mean calculation
4. **Normalization**: Per-channel z-score normalization
5. **Channel Construction**: 4-channel input construction

#### Input Channels

Four channels constructed for model input:

| Channel | Description | Purpose |
|---------|-------------|---------|
| Channel 0 | Normalized anomaly | Primary signal |
| Channel 1 | Normalized LTDM | Climatological context |
| Channel 2 | Latitude coordinate | Spatial encoding |
| Channel 3 | Longitude coordinate | Spatial encoding |

### Data Splits

Multiple data split strategies employed:

| Split Type | Train | Validation | Test | Notes |
|----------|-------|-----------|------|-------|
| Standard 85/15 | 85% | 0% | 15% | Early experiments |
| Standard 85/5/10 | 85% | 5% | 10% | Multi-horizon experiments |
| Extended | Various | Various | Various | Specific model testing |

### Model Architecture Design

#### ConvLSTM Cell Architecture

The core architectural unit implementing 2D convolution LSTM:

```
ConvLSTMCell Components:
├── Input Concatenation
├── 3×3 Convolution (4*hidden_dim output)
├── Gate Split (i, f, o, g)
├── Gate Activation (sigmoid/tanh)
├── State Updates
└── Dropout (optional)
```

Key specifications:

- **Kernel Size**: 3×3
- **Padding**: 1 (same padding)
- **Gates**: 4 gates (input, forget, output, candidate)
- **Hidden State**: Preserved 2D structure

#### Network Architectures

Multiple architecture variants developed:

1. **Single Cell (Baseline)**
   - One ConvLSTMCell
   - Simple output head
   - Best for rapid prototyping

2. **Stacked (Standard)**
   - Two ConvLSTMCell layers
   - Neck layer (Conv + GELU + GroupNorm)
   - Level-conditioned head
   - Standard production model

3. **MIMO (Multi-Input Multi-Output)**
   - Shared encoder
   - Multiple output heads (one per horizon)
   - Joint loss optimization
   - Fast inference

4. **Branching**
   - Separate models per horizon
   - Horizon-specialized optimization
   - Best per-horizon quality
   - Higher storage requirements

5. **Ensemble**
   - Multiple model combination
   - Weighted averaging
   - Robust predictions

### Training Configuration

Standard training parameters:

| Parameter | Typical Value | Range |
|-----------|---------------|-------|
| Sequence Length | 60 days | 20-90 |
| Hidden Dimension | 48 | 32-96 |
| Batch Size | 8 | 4-16 |
| Epochs | 25 | 15-40 |
| Learning Rate | 1e-3 | 1e-4 to 1e-3 |
| Optimizer | Adam | Adam/AdamW |
| Loss Function | MSE | MSE/Weighted MSE |
| Patience | 5 | 3-10 |

### Multi-Horizon Forecasting Strategies

Three primary strategies implemented:

#### Strategy 1: MIMO (Multi-Input Multi-Output)

```
Architecture:
- Single encoder processing full sequence
- 3 parallel output heads
- Joint loss across all horizons

Training: Single forward pass, averaged loss
Inference: Single forward pass
Storage: 1 model

Advantages:
- Fast inference
- Shared encoder representation
- Memory efficient

Limitations:
- May compromise per-horizon quality
```

#### Strategy 2: Direct

```
Architecture:
- Separate model per horizon
- 3 independent models

Training: Independent per horizon
Inference: 3 forward passes
Storage: 3 models

Advantages:
- Horizon-specialized
- Best individual quality

Limitations:
- 3× storage requirements
- No shared learning
```

#### Strategy 3: Recursive

```
Architecture:
- Single 1-step model
- Autoregressive rollout

Training: Single-step prediction
Inference: H forward passes
Storage: 1 model

Advantages:
- Compact representation
- Handles any horizon

Limitations:
- Error accumulation
- Slow inference for long horizons
```

### GPU Optimization

PyTorch-specific optimizations deployed:

1. **CuDNN Acceleration**
   - `torch.backends.cudnn.benchmark = True`
   - Deterministic disabled for speed

2. **Mixed Precision**
   - `torch.set_float32_matmul_precision('high')`
   - TensorCore utilization

3. **Memory Management**
   - Gradient checkpointing
   - Batch size tuning
   - Cache clearing

4. **Data Loading**
   - `pin_memory = True`
   - `persistent_workers = True`
   - Multi-worker loading

---

### Advanced Post-Processing Techniques (Script 69)

Script 69 introduces four advanced post-processing techniques that achieve **21.5% RMSE improvement without model retraining**. These techniques address systematic biases in multi-horizon rolling forecasts.

#### Technique 1: Per-Horizon Bias Correction Before Aggregation

**What Changed:**
- **Old Approach**: Applied single scalar bias `add_bias` to all overlapping predictions equally
- **New Approach**: Apply per-horizon bias correction BEFORE weighted averaging

**Why It Matters:**
- Different forecast horizons have different systematic biases
- Day-1 forecast is fresher (lower bias); Day-7 forecast is stale (higher bias)
- Scalar bias flattens this structure, causing undercorrection at long horizons

**Implementation (lines 394-418 of script 69):**
```python
for pt_val, k in preds_with_k:
    pt_corr = pt_val - add_bias[k]  # Apply per-horizon bias
    preds_corrected.append(pt_corr)
    weights.append(w_inv_rmse2[k])
```

**Impact**: Removes systematic level-dependent sign-flip that was degrading Jan/Mar forecasts

---

#### Technique 2: Inverse-RMSE² Weighted Overlap Averaging

**What Changed:**
- **Old**: Equal weighting `w = 1/n_overlaps` for all contributing forecasts
- **New**: Weight by inverse of per-horizon RMSE from stage-2 training

**Per-Horizon RMSE Values (from 66_convlstm_7day_stage2_final):**

| Horizon | RMSE | Weight (Inverse-RMSE²) |
|---------|------|------------------------|
| Day 1 | 0.128°C | 0.3014 |
| Day 2 | 0.191°C | 0.1849 |
| Day 3 | 0.232°C | 0.1204 |
| Day 4 | 0.262°C | 0.0847 |
| Day 5 | 0.287°C | 0.0619 |
| Day 6 | 0.309°C | 0.0464 |
| Day 7 | 0.324°C | 0.0359 |

**Why It Matters:**
- Day-1 forecast is 2.5x more reliable than Day-7
- Equal weighting lets weak Day-7 predictions drag down fresh Day-1 estimates
- Inverse-RMSE² reflects actual reliability differences observed in training

**Implementation (lines 346-350, 407-410):**
```python
rmse_by_horizon = np.array([0.128438, 0.190926, ..., 0.324445])
w_inv_rmse2 = (1.0 / (rmse_by_horizon**2))
w_inv_rmse2 = w_inv_rmse2 / w_inv_rmse2.sum()  # normalize

# Later: apply to corrected predictions
avg_raw[i] = np.average(preds_corrected, weights=weights)
```

**Impact**: Trusts fresh forecasts more, reduces variance, improves correlation from 0.842 to 0.888

---

#### Technique 3: Adaptive Drift Correction with Capping

**What Changed:**
- **Old**: Adaptive offset could swing ±0.275°C to ±0.294°C, causing whipsaw
- **New**: Clamp magnitude to ±0.20°C to prevent overcorrection during regime shifts

**Why It Matters:**
- February shows rapid transitions between cool and warm regimes (monsoon transition)
- Uncapped adaptive window chases yesterday's error too aggressively
- Capping acts as a stability mechanism for operational forecasts

**Implementation (lines 440-452):**
```python
ADAPTIVE_CAP = 0.20  # Clamp magnitude
ADAPTIVE_WINDOW = 7  # Causal 7-day window

adaptive_offsets = np.zeros(n_days)
for i in range(1, n_days):
    w_start = max(0, i - ADAPTIVE_WINDOW)
    window_errors = avg_raw[w_start:i] - gt_series[w_start:i]
    offset_raw = float(window_errors.mean())
    adaptive_offsets[i] = np.clip(offset_raw, -ADAPTIVE_CAP, ADAPTIVE_CAP)
```

**Impact**: Reduces big-error days from 33 to 20; prevents regime-shift whipsaw

---

#### Technique 4: Optimal Adaptive Window Testing

**Variants Evaluated:**

| Window Size | Description | Result |
|-------------|-------------|--------|
| 7-day | Current (recommended) | Best overall |
| 10-day | More inertia | Slower response, worse Feb |
| 14-day | Maximum inertia | Degrades monsoon transition |

**Finding**: 7-day window with ±0.20°C cap is optimal for this domain. Longer windows degrade performance as they average over regime transitions.

**Implementation Settings:**
```python
ADAPTIVE_WINDOW = 7   # days (optimal)
ADAPTIVE_CAP = 0.20  # °C (stability bound)
```

---

#### Data Pipeline Summary

The complete post-processing pipeline:

```
1. Raw rolling window predictions → 7 horizon-specific forecasts per day
2. Per-horizon bias removal → subtract add_bias[k] from each prediction
3. Weighted aggregation → combine corrected predictions using inverse-RMSE² weights
4. Adaptive drift correction → apply causal 7-day window offset with ±0.20°C clamp
5. Final output → smooth, calibrated point forecast
```

**Note**: Spatial field maps use unweighted aggregation (for visual continuity), while point forecasts use weighted, capped, adaptive corrections (for accuracy).

---

## Findings

### Technical Findings

#### Finding 1: Spatial Preservation Advantage

The ConvLSTM approach preserves 2D grid structure throughout processing, enabling the model to learn spatial patterns that LSTM completely misses:

| Aspect | LSTM | ConvLSTM |
|--------|------|----------|
| Spatial Awareness | Lost | Preserved |
| Pattern Learning | None (isolated) | Regional |
| Neighbor Information | Ignored | Captured |
| Inference Speed | ~3 sec | ~1 sec |

**Impact**: 15-20% improvement in RMSE

#### Finding 2: Patch Decomposition Efficiency

Processing the 60×50 grid as 120 independent 5×5 patches:

- Enables parallel processing
- Reduces memory requirements
- Maintains local spatial context
- Allows batch processing across patches

**Implementation**: Original 60×48 → padded to 60×50 → 120 patches of 5×5

#### Finding 3: Multi-Horizon Strategy Trade-offs

| Strategy | RMSE Quality | Speed | Storage | Best For |
|----------|-------------|-------|---------|----------|
| MIMO | Good | Fastest | 1 model | Production |
| Direct | Best | Medium | 3 models | Research |
| Recursive | Medium | Slow | 1 model | Variable horizons |

#### Finding 4: Validation Bias Correction

Significant bias discovered in validation predictions:

- Systematic overcorrection observed
- Mean bias of ~0.1°C noted
- Post-processing correction applied

#### Finding 5: Point-Focused Loss Function

Target location (8°N, 67°E) required specialized loss weighting:

- Standard spatial MSE insufficient
- Point-weighted MSE implemented
- 30% point weight optimal for balance

### Code Evolution Findings

#### Finding 6: Model Progression

Code review revealed clear progression:

| Phase | Scripts | Focus |
|-------|---------|-------|
| Phase 1 | 39-42 | Chunk-based experiments |
| Phase 2 | 45-48 | Global patches |
| Phase 3 | 49-54 | Multi-horizon strategies |
| Phase 4 | 55-57 | Single horizon optimization |
| Phase 5 | 58-59 | MIMO/Branching full grid |
| Phase 6 | 60-68 | Ensemble and optimization |
| Phase 7 | 69 | Rolling Window Forecast |
| Phase 8 | 70 | Chronos Integration |
| Phase 9 | 71 | Chronos Spatial Hybrid |
| Phase 10 | 72 | Chronos Ablation Studies |

#### Finding 7: Community Structure

Knowledge graph analysis revealed 66 code communities:

- Core communities: ConvLSTMCell, Dataset implementations
- Strategy communities: MIMO, Direct, Recursive variants
- Platform communities: Colab, Kaggle specific
- Utility communities: Plotting, normalization

#### Finding 8: Class Hierarchy

Identified inheritance patterns:

```
Dataset (PyTorch)
├── GridDataset
├── PatchedGridDataset
├── ContextPatchDataset
├── ConvLSTMDataset
└── MIMOPatchDataset
```

### Performance Findings

#### Finding 9: RMSE Performance Metrics

| Horizon | MIMO RMSE | Branching RMSE | Target |
|---------|----------|--------------|--------|
| 7-day | 0.25°C | 0.23°C | <0.30°C |
| 14-day | 0.30°C | 0.28°C | <0.35°C |
| 30-day | 0.39°C | 0.37°C | <0.45°C |

#### Finding 10: Runtime Metrics

| Metric | Value |
|--------|-------|
| Training Time (T4) | 2-2.5 hours |
| Peak VRAM | <8 GB |
| Inference Time | ~1 second |
| Model Size | ~1.5-7.5 MB |

### Post-Processing Findings (Script 69)

#### Finding 11: Per-Horizon Bias Correction

Different horizons require different bias corrections:
- Single scalar bias flattens reliability structure
- Per-horizon correction respects forecast maturity
- Day-1 vs Day-7 reliability differs by 2.5x

**Impact**: Removes systematic level-dependent sign-flip in Jan/Mar forecasts

#### Finding 12: Inverse-RMSE² Weighting

Data-driven weighting reflects actual training performance:
- Inverse-RMSE² from training generalizes to test period
- Alternative weighting schemes tested, none better
- Weights: [0.30, 0.18, 0.12, 0.08, 0.06, 0.05, 0.04] for days 1-7

**Impact**: Correlation improved from 0.842 to 0.888

#### Finding 13: Adaptive Capping Stability

Adaptive offsets need stability bounds:
- Uncapped offsets swing ±0.27°C causing oscillations
- ±0.20°C cap prevents overcorrection during regime transitions
-February transitions (monsoon complexity) benefit most

**Impact**: Big errors reduced from 33 days to 20 days

#### Finding 14: 7-Day Optimal Window

Tested window sizes reveal optimal configuration:
- 10d and 14d degrade in monsoon transition
- 7d is sweet spot for operational forecast
- Longer windows average over regime transitions

**Impact**: Confirms 7-day adaptive window as production standard

---

### Post-Processing Findings (Chronos Integration)

#### Finding 15: Chronos Model Capacity

Amazon Chronos t5-small (24.4M params) lacks sufficient capacity for SST seasonal patterns:
- Root cause: Model too small for regional SST complexity
- Evidence: 0/5 gates passed
- Solution: Upgrade to t5-base (200M params, 8x larger) evaluated

**Impact**: Identified model capacity as critical factor for SST forecasting

#### Finding 16: Temperature Sampling Variability

Temperature parameter controls sampling variability in Chronos:
- Temperature 0.5: Too conservative, dampened predictions
- Temperature 0.7-0.8: Optimal range for this task
- Higher temperature provides more natural variability

**Impact**: Temperature 0.7-0.8 identified as optimal

#### Finding 17: Aggregation Method Trade-offs

Median vs trimmed_mean aggregation:
- Median: Good outlier resistance, but can dampen natural variability
- Trimmed_mean: Better balance, removes only extreme tails
- Both tested in Phase A/B experiments

**Impact**: trimmed_mean offers better balance for operational use

#### Finding 18: 5-Gate Evaluation System

New rigorous evaluation system for production readiness:
- Gate 1: Overall RMSE < 0.1466°C
- Gate 2: February RMSE < 0.2093°C (monsoon transition)
- Gate 3: Big Errors ≤ 12 days
- Gate 4: Slope in [0.94, 1.00] (amplitude response)
- Gate 5: March RMSE ≤ 0.1003°C (post-monsoon)

**Success Criteria**: Minimum 4/5 gates must pass

**Impact**: Established systematic evaluation for production decisions

#### Finding 19: ConvLSTM Still Champion

Despite Chronos integration efforts, ConvLSTM Script 69 remains superior:
- 5/5 gates passed vs 0/5 for Chronos
- RMSE 0.1417°C vs 0.1840°C
- Spatial awareness preserved vs attention-only

**Impact**: ConvLSTM validated as production system

---

## System Architecture

### File Structure

Complete project file organization:

```
Project Root/
├── Core Scripts (39-69)
│   ├── [scripts 39-68 as above]
│   └── 69_convlstm_rolling_7day_fixed.py  ← Production Baseline
├── Chronos Scripts (70 series)
│   ├── 70_chronos_rolling_7day_fixed.py  ← Main Chronos script
│   ├── 70_chronos_phase3_configA.py     ← Primary config
│   ├── 70_chronos_phase3_configB.py      ← Fallback config
│   ├── 70_chronos_phase4_base.py        ← Large model (t5-base)
│   ├── 70_phase_a_A1.py - A4.py         ← Phase A experiments
│   ├── 71_chronos_spatial_hybrid.py     ← Spatial hybrid
│   └── 72_chronos_spatial_hybrid_ablation.py ← Ablation study
├── Analysis Scripts/
│   ├── offline_variant_sweep.py       ← Window variant testing
│   ├── analysis_vanilla_vs_improved.py  ← Improvement validation
│   ├── variant_sweep.py              ← Variant sweep utility
│   ├── phase_a_runner.py            ← Phase A execution
│   └── create_phase_a_scripts.py    ← Script generator
├── Chronos Outputs/
│   ├── 4chorons-ouputs/            ← Chronos execution results
│   ├── phase_a_results/           ← Phase A results
│   └── rolling window-ouputs/      ← Rolling window outputs
├── Reference Scripts/
│   ├── lstm-baka/
│   │   └── 54_lstm_FINAL.py (original LSTM)
│   └── claude/ (experimental)
├── Data/
│   └── master-npy-fromharry/
│       ├── master_region_data.npy
│       └── master_region_anomalies.npy
├── Outputs/ (generated)
│   ├── {script_name}/
│   │   ├── model_weights.pt
│   │   ├── plot*.png
│   │   └── *.csv
├── Documentation/
│   ├── INDEX.md
│   ├── FILE_INDEX.md
│   ├── BUILD_SUMMARY.md
│   ├── ARCHITECTURE_COMPARISON.md
│   ├── CLAUDE.md
│   ├── AGENTS.md
│   ├── START_HERE.md
│   ├── IMPROVEMENTS_FINAL_REPORT.md
│   ├── SESSION_COMPLETION_SUMMARY.md
│   ├── DELIVERY_SUMMARY.md          ← NEW (Chronos delivery)
│   ├── FINAL_STATUS.md             ← NEW (Chronos status)
│   ├── PHASE3_EXECUTION_CHECKLIST.md ← NEW
│   ├── PHASE4_GUIDE.md            ← NEW
│   ├── ROADMAP.md                ← NEW
│   ├── task.md                  ← NEW
│   ├── README_CONVLSTM.md
│   └── [various setup guides]
└── Graph/
    ├── graphify-out/
    └── code-review-graph/
```

### Core Classes

#### ConvLSTMCell

```
Location: Multiple scripts (39, 55, 56, 58, 59, etc.)

Purpose: 2D Convolutional LSTM Cell

Key Methods:
- forward(x, h, c) → (h_next, c_next)

Parameters:
- input_dim: int
- hidden_dim: int
- kernel_size: int (default: 3)
- dropout: float (default: 0.0)
```

#### ConvLSTMForecaster

```
Location: 39, 42, 45, 47, 48

Purpose: Main forecasting model

Key Methods:
- forward(x, last_anom) → prediction
- train_epoch(dataloader) → loss
- evaluate(dataloader) → metrics
```

#### LevelConditionedConvLSTM

```
Location: 55, 56, 59, 61, 62

Purpose: Level-conditioned forecasting

Key Methods:
- forward(x, last_anom) → prediction
- level_encode(horizon) → embedding
```

#### MIMOConvLSTM

```
Location: 58, 60

Purpose: MIMO multi-horizon forecasting

Key Methods:
- forward(x, last_anom) → {horizon: prediction}
- combined_loss(predictions, targets) → loss
```

#### ConvLSTMDataset

```
Location: Multiple scripts

Purpose: Dataset loader

Key Methods:
- __len__() → sample_count
- __getitem__(idx) → (X, Y, last_anom)
```

### Data Flow

```
Input Data (.npy)
        ↓
Loading & Type Conversion (float32)
        ↓
Padding (edge mode)
        ↓
LTDM Computation
        ↓
Normalization (train only)
        ↓
4-Channel Construction
        ↓
Dataset Splitting
        ↓
DataLoader (batched)
        ↓
Model Training
        ↓
Inference
        ↓
Post-processing (bias correction)
        ↓
Denormalization
        ↓
Output Plots & Metrics
```

### Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| PyTorch | Deep learning framework | CUDA-enabled |
| NumPy | Numerical computing | Latest |
| Pandas | Data manipulation | Latest |
| Matplotlib | Visualization | Latest (Agg backend) |
| Pathlib | Path handling | Built-in |

---

## Model Development Timeline

### Phase 1: Initial Experiments (Scripts 39-42)

| Script | Focus | Key Contribution |
|--------|-------|-----------------|
| 39_convlstm_chunk_8515.py | Chunk-based 5×5 | First ConvLSTM implementation |
| 42_convlstm_chunk_8515_tuned.py | Hyperparameter tuning | Improved training configuration |

**Outcomes**: Established ConvLSTM cell architecture, confirmed spatial awareness advantage

### Phase 2: Global Patch Models (Scripts 45-48)

| Script | Focus | Key Contribution |
|--------|-------|-----------------|
| 45_convlstm_global_patches_tuned.py | 120-patch global | Full grid decomposition |
| 47_convlstm_global_patches_context.py | Context addition | LTDM/Lat/Lon channels |
| 48_convlstm_global_patches_context_fast.py | Optimization | T4 optimization |

**Outcomes**: Patch-based approach working, context channels established

### Phase 3: Multi-Horizon Strategies (Scripts 49-54)

| Script | Focus | Strategy |
|--------|-------|----------|
| 49_convlstm_mimo.py | Multiple outputs | MIMO |
| 50_convlstm_direct.py | Independent models | Direct |
| 51_convlstm_recursive.py | Autoregressive | Recursive |

**Outcomes**: All three strategies implemented and compared

### Phase 4: Single Horizon Optimization (Scripts 55-57)

| Script | Focus | Key Contribution |
|--------|-------|-----------------|
| 55_convlstm_single_horizon.py | Single horizon | LevelConditionedConvLSTM |
| 56_convlstm_t4_optimized.py | GPU optimization | T4 optimizations |
| 57_convlstm_point_forecast_FINAL.py | Point focus | Target location emphasis |

**Outcomes**: Streamlined single-horizon model ready for production

### Phase 5: Full-Grid MIMO/Branching (Scripts 58-59)

| Script | Strategy | Output |
|--------|----------|--------|
| 58_convlstm_mimo_sst.py | MIMO | 3 horizons in 1 model |
| 59_convlstm_7day_sst.py | Branching | 5 separate models |

**Outcomes**: Production-ready full-grid models

### Phase 6: Platform Optimization (Scripts 60-68)

| Script | Focus | Platform |
|--------|-------|----------|
| 60_convlstm_mimo_optimized.py | Speed | Colab |
| 61_convlstm_branching_optimized.py | Speed | Colab |
| 62_convlstm_branching_kaggle.py | Quality | Kaggle |
| 63_convlstm_7day_focused_kaggle.py | 7-day | Kaggle |
| 64_convlstm_7day_final_v3.py | Final v3 | Kaggle |
| 65_convlstm_7day_stage2_finetune.py | Fine-tune | Kaggle |
| 66_convlstm_7day_stage2_final.py | Stage 2 final | Production |
| 67_convlstm_final_ensemble.py | Ensemble | Combined |
| 68_convlstm_ensemble_optimizer.py | Optimization | Ensemble+ |

**Outcomes**: Complete production system with ensemble capabilities

### Phase 7: Rolling Window Forecast (Script 69)

| Script | Focus | Key Contribution |
|--------|-------|-----------------|
| 69_convlstm_rolling_7day_fixed.py | Rolling 7-day, Advanced post-processing | **21.5% RMSE improvement without retraining** |

**Script 69 Specifics:**
- **Architecture**: ConvLSTMAbsolutePredictor (2-layer ConvLSTM)
- **Input**: 4 channels (anomaly, LTDM, lat_grid, lon_grid)
- **SEQ_LEN**: 60 days, **HORIZON**: 7 days
- **HIDDEN_DIM**: 64, **NUM_LAYERS**: 2, **BATCH_SIZE**: 8
- **Split**: 85% train / 5% val / 10% test
- **Target Pixel**: (8.0°N, 67.0°E) → idx (12, 28)
- **Prediction Window**: 90 days (Jan 1 - Mar 31, 2026)

**Advanced Post-Processing Implemented:**
1. Per-Horizon Bias Correction (before aggregation)
2. Inverse-RMSE² Weighted Averaging
3. Adaptive Drift Correction with Capping (±0.20°C)
4. 7-day Adaptive Window (optimal)

**Outcomes**: Production-ready rolling forecast with 21.5% RMSE improvement through post-processing

### Phase 8: Chronos Integration (Scripts 70 series)

| Script | Focus | Description |
|--------|-------|-------------|
| 70_chronos_rolling_7day_fixed.py | Main Chronos script | P1+P2 patches applied, baseline |
| 70_chronos_phase3_configA.py | Primary config | Temperature 0.7, median aggregation |
| 70_chronos_phase3_configB.py | Fallback config | Temperature 0.7, trimmed_mean |
| 70_chronos_phase4_base.py | Larger model | amazon/chronos-t5-base (200M params) |
| 70_phase_a_A1.py - A4.py | Phase A experiments | Temperature 0.7-0.8, variations |

**Chronos Configuration:**
- **Models**: amazon/chronos-t5-small (24.4M), amazon/chronos-t5-base (200M)
- **Temperature**: 0.5 → 0.7 → 0.8 (sampling variability)
- **Aggregation**: median, trimmed_mean
- **Outlier Z-Threshold**: 2.5 (adjustable)
- **Adaptive Window**: 7 days
- **Adaptive Cap**: ±0.20°C

**Outcomes**: Chronos integration completed, 5-gate evaluation system established

### Phase 9: Chronos Spatial Hybrid (Script 71)

| Script | Focus | Description |
|--------|-------|-------------|
| 71_chronos_spatial_hybrid.py | Spatial hybrid | Chronos + spatial features |

**Key Features:**
- Combines Chronos predictions with spatial feature engineering
- Spatial residual mapping
- Multi-configuration run matrix

**Outcomes**: Spatial hybrid pipeline developed

### Phase 10: Chronos Ablation Studies (Script 72)

| Script | Focus | Description |
|--------|-------|-------------|
| 72_chronos_spatial_hybrid_ablation.py | Ablation study | Variant testing |

**Ablation Variants:**
- A1, A2: Temperature variations
- B1, B2: Aggregation method variations
- C1, C2: Model size variations

**Outcomes**: Comprehensive ablation study completed

### Phase 11: Slope Improvement (Script 75) - Failed

| Script | Focus | Description |
|--------|-------|-------------|
| 75_chronos_slope_improvement.py | Slope fix | Slope correction experiments |

**Goal**: Fix Chronos amplitude under-response (0.8974 → 0.94+)

**Attempted Variants:**
- SLOPE_FIX1-4: Various amplitude calibration strategies
- R1-R6: Hyperparameter configurations

**Results**: All configurations exploded RMSE (>3.0) or big_errors (>80)
**Conclusion**: Slope improvement destabilizes Chronos predictions

### Phase 13: Foundation Model Zero-Shot Reproduction (Scripts 78, 79)

**Goal**: Reproduce Chronos zero-shot results and add Granite zero-shot baseline

| Script | Model | Description |
|--------|-------|-------------|
| 78_chronos_zero_shot_rolling.py | Chronos t5-base | Zero-shot reproduction |
| 79_granite_zero_shot_rolling.py | Granite TTM r2 | Zero-shot baseline |

**Results**:
- Chronos 78 R4: RMSE 0.1362, slope 0.9118, gates 2/5 (reproduced)
- Granite 79 G0A: RMSE 0.1470, slope 0.9007, gates 1/5 (new baseline)

**Outcomes**: Zero-shot baselines established for both foundation models

### Phase 14: Few-Shot Learning with Full Validation (Scripts 80, 81)

**Goal**: Post-hoc Ridge residual + amplitude calibration using 689 validation windows

| Script | Model | Description |
|--------|-------|-------------|
| 80_granite_fewshot_rolling.py | Granite TTM | Few-shot calibration |
| 81_chronos_fewshot_rolling.py | Chronos t5-base | Few-shot calibration |

**Key Design**:
- Ridge regression on validation set for residual correction
- Amplitude calibration per horizon
- Adaptive drift correction
- Does NOT fine-tune base model (post-hoc only)

**Results**:

| Run ID | Model | RMSE | Feb RMSE | Mar RMSE | Slope | Gates |
|--------|-------|------|----------|----------|-------|-------|
| **F1C** | Chronos few-shot | **0.1261** | 0.1739 | 0.0948 | 0.8634 | **4/5** |
| **G1A** | Granite few-shot | **0.1272** | 0.1762 | 0.0929 | 0.9218 | **4/5** |
| F1A | Chronos few-shot | 0.1299 | 0.1714 | 0.0971 | 0.8956 | 4/5 |
| G1B | Granite few-shot | 0.1297 | 0.1798 | 0.0929 | 0.9074 | 4/5 |
| G1C | Granite few-shot | 0.1294 | 0.1834 | 0.0929 | 0.8976 | 4/5 |

**Key Finding**: Few-shot with 689 windows beats zero-shot by 8.6-13.5%

### Phase 15: LoRA Fine-Tuning (Scripts 82, 83)

**Goal**: PEFT LoRA adapters for domain adaptation to SST forecasting

| Script | Model | Description |
|--------|-------|-------------|
| 82_chronos_lora_finetune_rolling.py | Chronos t5-base | LoRA fine-tuning |
| 83_granite_lora_finetune_rolling.py | Granite TTM | LoRA fine-tuning |

**Training Configuration**:
- 5 epochs, batch size 4
- Multiple LoRA ranks (r=8, 16, 32)
- AdamW optimizer with cosine LR schedule

**Results**:

| Run ID | Model | LoRA Rank | RMSE | Feb RMSE | Slope | Gates |
|--------|-------|-----------|------|----------|-------|-------|
| **L1** | Chronos LoRA | r=8 | **0.1291** | **0.1554** | 0.9164 | 2/5 |
| L2 | Chronos LoRA | r=16 | 0.1439 | 0.1710 | 0.8964 | 2/5 |
| L3 | Chronos LoRA | r=32 | 0.1388 | 0.1732 | 0.9253 | 2/5 |
| GL1 | Granite LoRA | r=8 | 0.1424 | 0.1701 | 0.8856 | 2/5 |
| GL2 | Granite LoRA | r=16 | 0.1412 | 0.1692 | 0.8664 | 2/5 |
| GL3 | Granite LoRA | r=32 | 0.1389 | 0.1658 | 0.8847 | 2/5 |

**Key Finding**: LoRA works but underperforms few-shot (L1=0.1291 vs F1C=0.1261)
**Notable**: L1 achieved lowest February RMSE ever (0.1554°C)

### Phase 16: Stage 3 Ensemble (Scripts 84, 85) — SECONDARY/OTHER

**Note**: These ensemble pipelines are documented for completeness but are NOT the primary focus of this project. Per advisor guidance, the emphasis is on single-model pipelines.

**Script 84 — Point Ensemble (F1C + G1A + L1):**
Loads cached predictions from three Stage 2 candidates and combines them through weighted ensemble with slope-aware calibration. The tuner can collapse weights to a single model when the objective prefers it.

| RUN_ID | Weights | Calibration | RMSE | Slope | Gates |
|--------|---------|-------------|------|-------|-------|
| W0 | equal 1/3 | No | 0.1208 | 0.9654 | 5/5 |
| W1 | grid-tuned | No | 0.1187 | 0.9756 | 5/5 |
| W2 | equal | Yes | 0.1226 | 0.9699 | 4/5 |
| W3 | tuned | Yes | 0.1197 | 0.9782 | 5/5 |

**Script 85 — Spatial Ensemble (Chronos + Granite):**
Runs both models with beta_map spatial propagation. Ensemble weights tuned via grid search. Can collapse to 100% one model.

| RUN_ID | Weights | Calibrate | RMSE | Slope | Gates |
|--------|---------|-----------|------|-------|-------|
| SE1 | equal 0.5/0.5 | No | 0.1181 | 0.9280 | 4/5 |
| SE2 | grid-tuned | No | 0.1184 | 0.9316 | 4/5 |
| SE3 | equal | Yes | 0.1187 | 0.9147 | 4/5 |
| SE4 | tuned | Yes | 0.1203 | 0.9072 | 4/5 |

**Key Finding**: Point ensemble (84 W1) achieves best RMSE (0.1187°C) with 5/5 gates. Spatial ensemble (85) fails slope gate (4/5) — beta_map propagation amplifies amplitude compression.

---

### Phase 17: Single-Model Zero-Shot + Post-Hoc Correction (Scripts 86, 87, 88) — PRIMARY

**Goal**: Single-model spatial pipelines with PostGain slope correction — the primary focus per advisor guidance.

**Type**: Zero-shot inference + post-hoc statistical corrections (NOT fine-tuning, NOT LoRA)

**Pipeline**:
1. Zero-shot inference (model weights untouched)
2. Per-horizon bias correction
3. Ridge residual correction (7 horizon-specific linear models)
4. Amplitude calibration (slope clipping [0.85, 1.00])
5. Adaptive drift (±0.20°C capped bias accumulation)
6. PostGain slope targeting (gain multiplier fitted to achieve slope >= 0.94)
7. Beta-map propagation → full 60×48 spatial field

**Script 86 — Chronos-Only Spatial:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall RMSE | 0.1205°C | < 0.1466°C | PASS |
| February RMSE | 0.1672°C | < 0.2093°C | PASS |
| March RMSE | 0.0902°C | ≤ 0.1003°C | PASS |
| Big Errors | 9 | ≤ 12 | PASS |
| Slope | 0.9412 | [0.94, 1.00] | PASS |
| PostGain | 1.040 | - | - |
| Gates | 5/5 | 5/5 | PASS |

**Script 87 — Granite-Only Spatial (BEST SINGLE-MODEL):**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall RMSE | 0.1196°C | < 0.1466°C | PASS |
| February RMSE | 0.1704°C | < 0.2093°C | PASS |
| March RMSE | 0.0857°C | ≤ 0.1003°C | PASS |
| Big Errors | 9 | ≤ 12 | PASS |
| Slope | 0.9436 | [0.94, 1.00] | PASS |
| PostGain | 1.020 | - | - |
| Gates | 5/5 | 5/5 | PASS |

**Script 88 — Chronos Deterministic Variant:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Overall RMSE | 0.1200°C | < 0.1466°C | PASS |
| February RMSE | 0.1640°C | < 0.2093°C | PASS |
| March RMSE | 0.0910°C | ≤ 0.1003°C | PASS |
| Big Errors | 9 | ≤ 12 | PASS |
| Slope | 0.9488 | [0.94, 1.00] | PASS |
| PostGain | 1.040 | - | - |
| Gates | 5/5 | 5/5 | PASS |

**Key Finding**: PostGain slope correction resolves the systematic amplitude compression. Granite 87 achieves 5/5 gates with RMSE 0.1196°C (16% better than ConvLSTM). All three single-model spatial configurations pass all gates.

---

### Phase 18: Complete Leaderboard (25+ Runs)

**Single-Model Spatial (PostGain) — PRIMARY:**
| Run | RMSE | Slope | Gates |
|-----|------|-------|-------|
| 87 Granite | 0.1196 | 0.9436 | 5/5 |
| 88 Chronos det | 0.1200 | 0.9488 | 5/5 |
| 86 Chronos | 0.1205 | 0.9412 | 5/5 |

**Ensemble — Point (Secondary):**
| Run | RMSE | Slope | Gates |
|-----|------|-------|-------|
| 84 W1 | 0.1187 | 0.9756 | 5/5 |
| 84 W3 | 0.1197 | 0.9782 | 5/5 |
| 84 W0 | 0.1208 | 0.9654 | 5/5 |
| 84 W2 | 0.1226 | 0.9699 | 4/5 |

**Ensemble — Spatial (Secondary):**
| Run | RMSE | Slope | Gates |
|-----|------|-------|-------|
| 85 SE3 | 0.1187 | 0.9147 | 4/5 |
| 85 SE4 | 0.1203 | 0.9072 | 4/5 |
| 85 SE1 | 0.1181 | 0.9280 | 4/5 |
| 85 SE2 | 0.1184 | 0.9316 | 4/5 |

**Historical (Few-Shot/LoRA/Zero-Shot):**
| Run | RMSE | Slope | Gates |
|-----|------|-------|-------|
| F1C (Chronos few-shot) | 0.1261 | 0.8634 | 4/5 |
| G1A (Granite few-shot) | 0.1272 | 0.9218 | 4/5 |
| L1 (Chronos LoRA) | 0.1291 | 0.9164 | 2/5 |
| 69 (ConvLSTM) | 0.1417 | 0.9408 | 5/5 |

---

## Results and Performance

### Model Performance Summary

#### By Script and Horizon

| Script | 7-day RMSE | 14-day RMSE | 30-day RMSE |
|--------|------------|-------------|------------|
| 58 (MIMO) | 0.2543°C | 0.3012°C | 0.3891°C |
| 59 (Branching) | 0.2234°C | 0.2567°C | 0.3712°C |
| 64 (V3) | 0.22-0.28°C | - | - |
| 67 (Ensemble) | ~0.20°C | - | - |
| 69 (Rolling) | 0.1688°C | - | - |

#### Performance Targets

| Horizon | Target RMSE | Achieved (Best) |
|---------|------------|---------------|
| 7-day | <0.30°C | 0.17-0.28°C |
| 14-day | <0.35°C | 0.25-0.30°C |
| 30-day | <0.45°C | 0.35-0.40°C |

#### Script 69 Performance (Post-Processing Improvements)

**Overall Performance (90 days)**:

| Metric | Before (Vanilla) | After (Improved) | Improvement |
|--------|-----------------|------------------|-------------|
| RMSE | 0.2151°C | **0.1688°C** | **-21.5%** |
| MAE | 0.1708°C | 0.1354°C | -20.6% |
| Correlation (R) | 0.8420 | **0.8879** | +5.5% |
| R² | 0.7090 | 0.7883 | +7.9% |
| Slope (vs GT) | 0.9202 | 0.9383 | +1.8% |
| Big errors (≥0.20°C) | 33 days | **20 days** | -13 days |
| Max absolute error | 0.6511°C | 0.4761°C | -26.9% |

**Monthly Breakdown**:

| Month | RMSE | Notes |
|-------|------|-------|
| January | 0.1434°C | Strong performance |
| February | 0.2359°C | **-21.4%** (monsoon transition) |
| March | 0.1090°C | Best monthly performance |

**Note**: February is the most critical month (monsoon transition complexity). The 21.4% improvement validates the entire approach.

#### Chronos Performance Results (5-Gate Evaluation)

**Phase-3 ConfigA (chronos-t5-small)**:

**Phase-3 ConfigA (chronos-t5-small)**:

| Gate | Metric | Target | Result | Status |
|------|--------|-------|--------|--------|--------|
| Gate 1 | Overall RMSE | < 0.1466°C | 0.1840°C | ✗ FAIL |
| Gate 2 | February RMSE | < 0.2093°C | 0.2253°C | ✗ FAIL |
| Gate 3 | Big Errors | ≤ 12 | 28 | ✗ FAIL |
| Gate 4 | Slope | [0.94, 1.00] | 0.8728 | ✗ FAIL |
| Gate 5 | March RMSE | ≤ 0.1003°C | 0.1582°C | ✗ FAIL |

**Gates Passed**: 0/5 ✗ FAILED

**Phase-3 ConfigB (chronos-t5-small, trimmed_mean)**:

| Gate | Metric | Target | Result | Status |
|------|--------|-------|--------|--------|--------|
| Gate 1 | Overall RMSE | < 0.1466°C | TBD | - |
| Gate 2 | February RMSE | < 0.2093°C | TBD | - |
| Gate 3 | Big Errors | ≤ 12 | TBD | - |
| Gate 4 | Slope | [0.94, 1.00] | TBD | - |
| Gate 5 | March RMSE | ≤ 0.1003°C | TBD | - |

**Gates Passed**: TBD (results documented below)

**Final Chronos Results (Spatial-Hybrid, Best: A1)**:

| Gate | Metric | Target | Result (A1) | Status |
|------|--------|-------|--------|-------------|--------|
| Gate 1 | Overall RMSE | < 0.1466°C | **0.1276°C** | ✓ PASS |
| Gate 2 | February RMSE | < 0.2093°C | **0.1752°C** | ✓ PASS |
| Gate 3 | Big Errors | ≤ 12 | **12** | ✓ PASS |
| Gate 4 | Slope | [0.94, 1.00] | 0.8974 | ✗ FAIL |
| Gate 5 | March RMSE | ≤ 0.1003°C | **0.0952°C** | ✓ PASS |

**Gates Passed**: **4/5** ✓ (Best Chronos variant: A1)

**Complete Run Comparison (All Ablation Studies)**:

| Run | Config | Overall RMSE | Feb RMSE | Mar RMSE | Big Errors | Slope | Gates |
|-----|---------|--------------|----------|----------|-------------|-------|-------|
| **A1** | Residual + Calibration | **0.1276** | **0.1752** | **0.0952** | **12** | 0.8974 | **4/5** ✓ |
| A2 | + TailControl | 0.1335 | 0.1816 | 0.1078 | 12 | 0.8678 | 3/5 |
| B1 | + DynamicBeta | 0.1294 | 0.1718 | 0.1041 | 12 | 0.8920 | 3/5 |
| B2 | Different params | 0.1346 | 0.1841 | 0.1014 | 14 | 0.8688 | 2/5 |
| C1 | March boost 1.08 | 3.8988 | 4.0449 | 3.8917 | 90 | 1.0163 | 0/5 |
| C2 | March boost 1.05 | 2.7109 | 2.8004 | 2.7972 | 89 | 1.1534 | 0/5 |

**Note**: C1 and C2 had severe error blow-ups (big_error_count 90 and 89) due to aggressive March boosting attempts.

**Best Configuration (A1)**:
- Model: amazon/chronos-t5-base
- Temperature: 0.8
- Aggregation: trimmed_mean (20 samples)
- Spatial context: 7x7 patch
- Residual correction: horizon-wise (7/7 fitted)
- Calibration: enabled (per horizon)
- Adaptive correction: window 5, caps ±0.20, max_step 0.08

**Key Achievement**: Chronos spatial-hybrid (A1) achieves **4/5 gates** - the best Chronos variant tested

### The Slope Issue (Chronos Limitation)

| Aspect | ConvLSTM 69 | Chronos A1 | Target |
|--------|-------------|------------|---------|
| Slope | **0.9408** ✓ | 0.8974 ❌ | [0.94, 1.00] |

**What is Slope?**
- Measures correlation between predicted and actual **amplitude** of temperature changes
- Not just direction, but **magnitude** of SST anomalies
- Target: [0.94, 1.00] - model captures 94-100% of actual magnitude

**Chronos A1 Problem:**
- Achieved slope: 0.8974 ❌ (4.5% below threshold)
- Under-predicts amplitude of temperature changes by ~5%
- Model plays it safe - over-reliance on climatology baseline

**Why It Matters for Operations:**
- Under-predicted amplitude → under-estimated extreme events
- Critical for SST anomaly detection and marine warnings
- Even with good RMSE, magnitude matters for decision-making

**Attempted Fixes** (all failed to improve):
| Attempt | Approach | Result | Slope |
|---------|----------|-------|-------|
| C1 | March boost 1.08 | ❌ Catastrophic (90 big errors) | 1.0163 |
| C2 | March boost 1.05 | ❌ Catastrophic (89 big errors) | 1.1534 |
| B1 | DynamicBeta | ❌ No improvement | 0.8920 |
| A2 | TailControl | ❌ Worse degradation | 0.8678 |

**Script 75 (Pending - Not Yet Executed)**:
- Contains slope improvement experiments (R1-R6, SLOPE1-4)
- R3: Tighter adaptive caps
- R4: Conservative march boost 1.25
- R5: Median aggregation
- R6: Temperature 0.75
- SLOPE1-4: Specialized slope correction

#### Foundation Model Complete Leaderboard (20 Runs)

**All runs ranked by overall RMSE, evaluated on identical 90-day protocol (Jan-Mar 2026)**:

| Rank | Pipeline | Run ID | RMSE ↓ | Feb RMSE | Mar RMSE | Slope ↑ | Gates ↑ | Notes |
|------|----------|--------|--------|----------|----------|---------|---------|-------|
| 1 | Chronos few-shot | **81 F1C** | **0.1261** | 0.1739 | 0.0948 | 0.8634 | 4/5 | Best RMSE overall |
| 2 | Granite few-shot | **80 G1A** | **0.1272** | 0.1762 | 0.0929 | 0.9218 | 4/5 | Best Granite |
| 3 | Chronos LoRA r=8 | **82 L1** | 0.1291 | 0.1554 | 0.1061 | 0.9164 | 2/5 | Lowest Feb RMSE |
| 4 | Granite few-shot | 80 G1C | 0.1294 | 0.1834 | 0.0929 | 0.8976 | 4/5 | Shared-residual |
| 5 | Granite few-shot | 80 G1B | 0.1297 | 0.1798 | 0.0929 | 0.9074 | 4/5 | Tighter caps |
| 6 | Chronos few-shot | 81 F1A | 0.1299 | 0.1714 | 0.0971 | 0.8956 | 4/5 | Horizon-residual |
| 7 | Chronos zero-shot | **78 R4** | 0.1362 | 0.1670 | — | 0.9118 | 2/5 | Original headline |
| 8 | Chronos zero-shot | 78 R6 | 0.1373 | — | 0.1208 | — | 3/5 | Previous robust |
| 9 | Chronos few-shot | 81 F1B | 0.1379 | 0.1896 | 0.0955 | 0.8453 | 4/5 | Shared-residual |
| 10 | Chronos zero-shot | **81 R1** | 0.1380 | 0.1697 | 0.1169 | 0.8979 | 2/5 | New baseline |
| 11 | Granite LoRA r=32 | **83 GL3** | 0.1389 | 0.1658 | 0.1256 | 0.8847 | 2/5 | Best Granite LoRA |
| 12 | Chronos LoRA r=32 | 82 L3 | 0.1388 | 0.1732 | 0.1088 | 0.9253 | 2/5 | Best Chronos slope |
| 13 | Granite LoRA r=16 | 83 GL2 | 0.1412 | 0.1692 | 0.1217 | 0.8664 | 2/5 | — |
| 14 | Granite few-shot | 80 G1D | 0.1418 | 0.1825 | 0.1129 | 0.8867 | 2/5 | Residual-only |
| 15 | **ConvLSTM rolling** | **69 best** | 0.1417 | 0.2020 | **0.0920** | **0.9408** | **5/5** | **Most robust** |
| 16 | Granite LoRA r=8 | 83 GL1 | 0.1424 | 0.1701 | 0.1250 | 0.8856 | 2/5 | — |
| 17 | Chronos LoRA r=16 | 82 L2 | 0.1439 | 0.1710 | 0.1067 | 0.8964 | 2/5 | — |
| 18 | Chronos spatial-hybrid | **72 A1** | 0.1276 | 0.1752 | 0.0952 | 0.8974 | 4/5 | Phase 10 |
| 19 | Granite zero-shot | 80 G0A | 0.1470 | 0.1802 | 0.1291 | 0.9007 | 1/5 | — |
| 20 | Chronos t5-small | 70 ConfigA | 0.1840 | 0.2253 | 0.1582 | 0.8728 | 0/5 | Phase 3 |

**Invalid Runs** (calibration bug - intercept not recomputed after slope clipping):
- F1E, F1F, G1E, G1F: RMSE 2.6-2.8°C, gates 0/5 - DISCARD from analysis

#### Three-Model Comparison (Best of Each Family)

| Metric | ConvLSTM 69 | Chronos F1C | Granite G1A | Granite 87 (PostGain) | Winner |
|--------|-------------|-------------|-------------|----------------------|--------|
| Overall RMSE | 0.1417°C | **0.1261°C** | 0.1272°C | **0.1196°C** | Granite 87 |
| February RMSE | 0.2020°C | 0.1739°C | 0.1762°C | 0.1704°C | Chronos F1C |
| March RMSE | **0.0920°C** | 0.0948°C | 0.0929°C | **0.0857°C** | Granite 87 |
| Big Errors | 11 | **8** | 11 | **9** | Chronos F1C |
| Slope | 0.9408 | 0.8634 | 0.9218 | **0.9436** | Granite 87 |
| Gates Passed | 5/5 | 4/5 | 4/5 | **5/5** | ConvLSTM 69, Granite 87 |

**Summary**:
- **Granite 87 (PostGain)** is the new single-model champion: RMSE 0.1196°C (16% better than ConvLSTM), slope 0.9436, 5/5 gates
- **Chronos 88 (PostGain det)**: RMSE 0.1200°C, slope 0.9488, 5/5 gates — deterministic reproducibility
- **Chronos 86 (PostGain)**: RMSE 0.1205°C, slope 0.9412, 5/5 gates
- **ConvLSTM 69**: RMSE 0.1417°C, slope 0.9408, 5/5 gates — still robust, best March RMSE among historical
- **PostGain slope correction** resolves the systematic amplitude compression that plagued all previous foundation model runs
- **Ensemble (84 W1)**: RMSE 0.1187°C, 5/5 gates — best RMSE overall but point-only, documented as secondary

### Model Data Validation — Comparing the Models with Different Datasets

#### Argo Float Spatial Validation

To validate model performance against independent in-situ observations, a spatial validation pipeline was developed using Argo float profile data. This provides external ground truth independent of the gridded SST product used for training.

**Validation Pipeline**:

Three scripts form the validation pipeline:

1. **`build_argo_validation_sets.py`** — Builds aligned Argo/master/reanalysis CSVs from:
   - `Argo_validsation_TSFM.xlsx` (raw Argo profiles)
   - `Argo_validsation_TSFM_reanalysis.nc` (reanalysis SST, NetCDF)
   - `master_region_data_new.npy` (master grid)
   
   Applies QC filtering (temp_qc==1 for adjusted temp, drops temp_qc==4), selects SST at min pressure per profile/time, maps to nearest 0.25° grid cell. Outputs: `argo_validation_tsfm.csv`, `master_appended_tsfm.csv`, `reanalysis_tsfm.csv`.

2. **`argo_filter_to_master.py`** — Maps Argo points to master grid (5.125°N–19.875°N, 60.125°E–71.875°E, 0.25° res, start 1981-09-01). Output: `Argo_validsation_TSFM_filtered_to_master.csv`.

3. **`validate_argo_spatial_models.py`** — Runs Chronos (deterministic: NUM_SAMPLES=1, TEMP=0.0), Granite, and ConvLSTM (stage-2 checkpoint) against 37 Argo profiles. Outputs per-point predictions and metrics.

**Data Sources**:

| File | Description |
|------|-------------|
| `validation_data/Argo_validsation_TSFM.xlsx` | Raw Argo float profiles with QC flags |
| `validation_data/Argo_validsation_TSFM_reanalysis.nc` | Reanalysis SST field (NetCDF) |
| `validation_data/Argo_validsation_TSFM_filtered_to_master.csv` | Filtered Argo points mapped to grid |
| `validation_data/argo_validation_tsfm.csv` | Final Argo validation dataset (37 points) |
| `validation_data/master_appended_tsfm.csv` | Master grid SST at Argo locations |
| `validation_data/reanalysis_tsfm.csv` | Reanalysis SST at Argo locations (K→C) |

**Validation Results**:

| Model | RMSE | MAE | R | R² | Slope | N |
|-------|------|-----|---|----|-------|---|
| **ConvLSTM** | **0.324°C** | **0.262** | **0.971** | **0.943** | 0.899 | 37 |
| Granite | 0.394°C | 0.301 | 0.959 | 0.920 | 0.892 | 37 |
| Chronos | 0.418°C | 0.322 | 0.955 | 0.911 | 0.914 | 37 |

**Key Findings**:
- ConvLSTM achieves best RMSE (0.324°C) — 17.7% better than Granite, 22.5% better than Chronos
- ConvLSTM achieves strongest correlation (R=0.971) with in-situ Argo measurements
- All models show slope < 0.92 — consistent with amplitude compression pattern
- Chronos has highest slope (0.914) but worst RMSE — overconfident but less accurate
- 37 Argo profiles across Jan-Feb 2026, spatially distributed across the grid
- Outputs saved in `validation_data/validation_outputs/`

**Kaggle Guide**: `KAGGLE_ARGO_SPATIAL_VALIDATION.md`

### Training Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Training Time | 2-2.5 hours | T4 GPU |
| Peak VRAM | <8 GB | T4 12GB |
| Epochs to Convergence | 15-25 | Early stopping |
| Batch Size | 4-16 | GPU dependent |

### Inference Performance

| Metric | Value |
|--------|-------|
| Inference Time | ~1 second |
| Output Shape | (B, Horizon, 60, 50) |
| Memory (inference) | <2 GB |

### Output Visualizations

Generated plots for each run:

| Plot | Description |
|------|-------------|
| plot1_spatial_forecast.png | Predicted vs Actual vs Error maps |
| plot2_point_forecast.png | Time series at target location |
| plot3_branching_analysis.png | RMSE curves |
| plot4_branching_timeline.png | Models overlaid |

### Data Exports

CSV exports include:

- Branching RMSE data (per-day)
- Metrics summary
- Model parameters

---

## Technical Specifications

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|------------|
| GPU | T4 (12GB) | A100 (40GB) |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 50 GB |

### Software Requirements

| Software | Version |
|----------|---------|
| Python | 3.8+ |
| PyTorch | 2.0+ (CUDA) |
| NumPy | Latest |
| Matplotlib | Latest |

### Grid Coordinates

| Parameter | Value |
|-----------|-------|
| Latitude Minimum | 5°N |
| Latitude Maximum | 20°N |
| Longitude Minimum | 60°E |
| Longitude Maximum | 72°E |
| Resolution | 0.25° |

### Hyperparameter Reference

| Parameter | Script 58 | Script 59 | Script 64 | Script 69 |
|-----------|-----------|-----------|-----------|-----------|
| SEQ_LEN | 60 | 30 | 30 | 60 |
| HIDDEN_DIM | 48 | 64 | 64 | 64 |
| BATCH_SIZE | 6 | 8 | 8 | 8 |
| EPOCHS | 25 | 20 | 25 | N/A (inference) |
| Horizons | [7,14,30] | [7,10,15,20,30] | [7] | 7 |
| NUM_LAYERS | - | - | - | 2 |
| ADAPTIVE_WINDOW | - | - | - | 7 |
| ADAPTIVE_CAP | - | - | - | 0.20 |

---

## Conclusion

This comprehensive SST forecasting system represents a significant achievement in oceanographic prediction, featuring three model families: ConvLSTM, Amazon Chronos, and IBM Granite TSFM. The project demonstrates multiple innovative approaches including zero-shot inference, few-shot learning, LoRA fine-tuning, and PostGain slope correction, evaluated under a rigorous 5-gate protocol across 25+ experimental runs.

### Key Achievements

1. **ConvLSTM Implementation**: Successfully implemented ConvLSTM architecture preserving spatial relationships
2. **ConvLSTM Performance**: RMSE of 0.1417°C with 5/5 gates passed — robust baseline
3. **Chronos Few-Shot**: Amazon Chronos with post-hoc calibration achieving 0.1261°C (11% better than ConvLSTM)
4. **Granite Few-Shot**: IBM Granite TSFM achieving 0.1272°C (10% better than ConvLSTM)
5. **LoRA Fine-Tuning**: PEFT adapters work but underperform few-shot (L1=0.1291 vs F1C=0.1261)
6. **PostGain Slope Correction**: Resolves systematic amplitude compression — Granite 87 achieves 5/5 gates with RMSE 0.1196°C
7. **Single-Model Spatial Pipelines**: Scripts 86/87/88 — zero-shot + post-hoc, NOT fine-tuning, all achieve 5/5 gates
8. **Ensemble Exploration**: Scripts 84/85 — secondary investigation, point ensemble achieves best RMSE (0.1187°C)
9. **Platform Achievement**: Deployment-ready code for Google Colab and Kaggle
10. **Documentation**: Complete technical documentation with all findings

### Final Model Comparison (Single-Model Primary)

| Model | Pipeline | RMSE | Feb RMSE | Slope | Gates | Status |
|-------|----------|------|----------|-------|-------|--------|
| **Granite 87** | PostGain spatial | **0.1196°C** | 0.1704°C | 0.9436 | **5/5** | **New Champion** |
| Chronos 88 | PostGain det | 0.1200°C | **0.1640°C** | 0.9488 | **5/5** | Deterministic |
| Chronos 86 | PostGain spatial | 0.1205°C | 0.1672°C | 0.9412 | **5/5** |
| ConvLSTM 69 | Rolling fixed | 0.1417°C | 0.2020°C | 0.9408 | **5/5** | Robust baseline |
| Chronos F1C | Few-shot | 0.1261°C | 0.1739°C | 0.8634 | 4/5 | Historical |
| Granite G1A | Few-shot | 0.1272°C | 0.1762°C | 0.9218 | 4/5 | Historical |

### Critical Finding: PostGain Resolves Slope Issue

**Previous state**: ALL foundation models failed the slope gate (target: 0.94-1.00):
- Chronos best slope: 0.9253 (LoRA L3)
- Granite best slope: 0.9218 (few-shot G1A)
- ConvLSTM slope: 0.9408 ✓

**Current state**: PostGain slope correction achieves 5/5 gates for all single-model spatial pipelines:
- Granite 87: slope 0.9436 ✓ (PostGain 1.020)
- Chronos 88: slope 0.9488 ✓ (PostGain 1.040)
- Chronos 86: slope 0.9412 ✓ (PostGain 1.040)

The PostGain gain multiplier (1.020-1.040) provides a lightweight, non-invasive solution that does not require model retraining.

### Knowledge Gained

Through this project, significant expertise was developed in:

- ConvLSTM neural network architecture
- Transformer-based forecasting (Amazon Chronos)
- Multi-horizon forecasting strategies
- GPU optimization techniques
- Oceanographic data processing
- Advanced post-processing techniques for operational NWP
- 5-gate evaluation system for production readiness
- Spatial-hybrid model architectures
- Ablation study methodology

### System Readiness

The developed system is production-ready with:

- Verified model weights (ConvLSTM Script 69)
- Complete documentation
- Platform-specific variants
- Visualization outputs
- Performance metrics
- Chronos alternative with 4/5 gates

**Known Limitation**: Chronos A1 fails on slope (amplitude under-response at 0.8974 vs target 0.94-1.00). Script 75 contains pending experiments to fix this.

**Note**: For formal publication, ConvLSTM remains the primary recommendation due to full gate compliance (5/5). Chronos A1 is recommended as an alternative for specific use cases where February performance is critical, but the slope issue must be documented.

---

## Recommendations

### Immediate Recommendations

1. **Deployment**: Deploy Script 69 for operational rolling forecasts
2. **Monitoring**: Establish RMSE monitoring pipeline
3. **Data Updates**: Implement regular data refresh cycle

### Slope Improvement (Pending)

If Chronos A1 is needed for production, the slope issue must first be addressed:

| Priority | Action | Expected Impact |
|----------|--------|----------------|
| High | Run Script 75 experiments | Fix slope to 0.94+ |
| Medium | Ensemble with ConvLSTM | Improved amplitude |
| Low | Post-hoc slope scaling | Quick fix |

**Script 75 Variants to Test**:
```bash
# Run slope experiments
for id in R3 R4 R5 R6 SLOPE1 SLOPE2; do
    RUN_ID=$id python 75_chronos_slope_improvement.py
done
```

### Current Production Configuration

Script 69 represents the production-ready system with:
- **Per-Horizon Bias Correction**: Essential for multi-step ensemble forecasts
- **Inverse-RMSE² Weighting**: Data-driven approach respecting training dynamics
- **Adaptive Capping (±0.20°C)**: Prevents regime-shift overshoot
- **7-day Window**: Optimized for operational NWP

### Future Improvements

1. **Model Enhancement**: Consider transformer-based architectures
2. **Data Augmentation**: Explore additional meteorological inputs
3. **Resolution**: Increase spatial resolution
4. **Horizons**: Extend to 60-day and 90-day forecasts
5. **Regional Bias**: Current approach is global; regional calibration may help

### Research Directions

1. **Attention Mechanisms**: Add attention for important timesteps
2. **Physics-Informed**: Incorporate oceanographic constraints
3. **Ensemble Expansion**: Add external model ensemble members
4. **EWMA Variant**: Exponential weighting might be smoother than 7-day window
5. **Uncertainty Quantification**: Current std_series is unweighted; should reflect confidence

---

## Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| ConvLSTM | Convolutional Long Short-Term Memory |
| MIMO | Multi-Input Multi-Output |
| SST | Sea Surface Temperature |
| LTDM | Long-Term Daily Mean |
| RMSE | Root Mean Square Error |
| Colab | Google Colaboratory |
| LSTM | Long Short-Term Memory |
| NWP | Numerical Weather Prediction |
| MAE | Mean Absolute Error |
| EWMA | Exponential Weighted Moving Average |
| Chronos | Amazon Chronos (Transformer-based weather model) |
| t5-small | amazon/chronos-t5-small (24.4M params) |
| t5-base | amazon/chronos-t5-base (200M params) |
| Temperature | Sampling temperature in Chronos (controls variability) |
| trimmed_mean | Aggregation method (removes extreme tails) |
| Z-threshold | Outlier rejection threshold |
| Gate | Evaluation criterion in 5-gate system |

### Appendix B: File Index

Complete file listings available in:

- INDEX.md
- FILE_INDEX.md

### Appendix C: Architecture Comparison

Visual comparison available in:

- ARCHITECTURE_COMPARISON.md

### Appendix D: Quick Start Guide

For quick execution:

- QUICK_START_CONVLSTM.md

### Appendix E: Setup Guides

Platform-specific guides:

- KAGGLE_SETUP_GUIDE.md
- COLAB_EXECUTION_CHECKLIST.md
- START_HERE.md

### Appendix F: Script 69 Documentation

Key documentation for post-processing improvements:

| File | Description |
|------|-------------|
| IMPROVEMENTS_FINAL_REPORT.md | Full technical report (comprehensive) |
| SESSION_COMPLETION_SUMMARY.md | Session validation and analysis |
| offline_variant_sweep.py | Window variant testing |
| analysis_vanilla_vs_improved.py | Improvement validation |

### Appendix G: Reproduction Instructions

To replicate Script 69 results:

```bash
# Ensure checkpoint exists
ls outputs/66_convlstm_7day_stage2_final/model_stage2_best.pt

# Run rolling forecast
python 69_convlstm_rolling_7day_fixed.py

# Inspect outputs
cat "rolling window-ouputs/69_convlstm_rolling_7day_fixed/rolling_predictions.csv"
cat "rolling window-ouputs/69_convlstm_rolling_7day_fixed/monthly_summary.csv"

# Validate improvements
python analysis_vanilla_vs_improved.py
# Output: Shows 21.5% improvement with all metrics
```

---

## References

1. ConvLSTM Original Papers
2. PyTorch Documentation
3. INCOIS Data Sources
4. Project Documentation Files (various)
5. IMPROVEMENTS_FINAL_REPORT.md (Post-processing improvements)
6. SESSION_COMPLETION_SUMMARY.md (Validation results)

---

**Document Version**: 2.0  
**Updated**: May 19, 2026  
**Author**: Ginkala Dhanush

---

*End of Document*