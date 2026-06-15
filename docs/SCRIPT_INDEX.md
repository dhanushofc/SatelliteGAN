# Script Index - All Python Scripts

## Foundation Model Scripts (76-85)

### Zero-Shot (Scripts 78, 79)

| Script | Model | Description | Status |
|--------|-------|-------------|--------|
| `78_chronos_zero_shot_rolling.py` | Chronos t5-base | Zero-shot reproduction | ✓ R4=0.1362 |
| `79_granite_zero_shot_rolling.py` | Granite TTM r2 | Zero-shot baseline | ✓ G0A=0.1470 |

### Few-Shot Learning (Scripts 80, 81)

| Script | Model | Description | Status |
|--------|-------|-------------|--------|
| `80_granite_fewshot_rolling.py` | Granite TTM | Few-shot calibration | ✓ G1A=0.1272 |
| `81_chronos_fewshot_rolling.py` | Chronos t5-base | Few-shot calibration | ✓ F1C=0.1261 |

### LoRA Fine-Tuning (Scripts 82, 83)

| Script | Model | Description | Status |
|--------|-------|-------------|--------|
| `82_chronos_lora_finetune_rolling.py` | Chronos t5-base | LoRA fine-tuning | ✓ L1=0.1291 |
| `83_granite_lora_finetune_rolling.py` | Granite TTM | LoRA fine-tuning | ✓ GL3=0.1389 |

### Stage 3 Ensemble (Scripts 84, 85) — SECONDARY/OTHER

| Script | Description | Status |
|--------|-------------|--------|
| `84_foundation_stage3_slope_calibrated_ensemble.py` | Point ensemble F1C+G1A+L1 | ✅ W1=0.1187, 5/5 |
| `85_spatial_ensemble_stage3.py` | Spatial ensemble Chronos+Granite | ✅ SE3=0.1187, 4/5 |

### Single-Model PostGain (Scripts 86, 87, 88) — PRIMARY

| Script | Description | Output Dir | Status |
|--------|-------------|-----------|--------|
| `86_spatial_chronos_only.py` | Chronos-only zero-shot + PostGain | `beats-chronos/86_spatial_chronos_only/` | ✅ 5/5, RMSE 0.1205 |
| `87_spatial_granite_only.py` | Granite-only zero-shot + PostGain | `beats-chronos/87_spatial_granite_only/` | ✅ 5/5, RMSE 0.1196 |
| `88_spatial_chronos_only_deterministic.py` | Chronos deterministic + PostGain | `beats-chronos/88_spatial_chronos_only_deterministic/` | ✅ 5/5, RMSE 0.1200 |

### Experimental Scripts (76, 77)

| Script | Description | Status |
|--------|-------------|--------|
| `76_granite_tsfm_spatial_hybrid_convlstm_style.py` | Granite spatial hybrid (early) | ✓ G0A=0.1272 |
| `77_granite_slope_tuned.py` | Granite slope improvement attempt | ❌ Failed (RMSE 0.76) |

---

## Chronos Scripts (70-75)

| Script | Description | Status |
|--------|-------------|--------|
| `70_chronos_phase3_configA.py` | Initial Chronos (t5-small) | ✗ 0/5 gates |
| `70_chronos_phase3_configB.py` | Config B (trimmed_mean) | - |
| `70_chronos_phase4_base.py` | Phase 4 base | Basic |
| `70_chronos_rolling_7day_fixed.py` | Rolling 7-day with Chronos | Phase 4 |
| `71_chronos_spatial_hybrid.py` | Spatial context (7×7 patch) | ✓ A1 winner |
| `72_chronos_spatial_hybrid_ablation.py` | Ablation studies (A1-A2, B1-B2, C1-C2) | ✓ All variants |
| `75_chronos_slope_improvement.py` | Slope fix experiments (R1-R6, SLOPE1-4) | ❌ Failed |

---

## ConvLSTM Scripts (39-69)

### Phase 1: Foundation (Scripts 39-48)

| Script | Description | Notes |
|--------|-------------|-------|
| `39_convlstm_chunk_8515.py` | Chunk-based 5×5, first ConvLSTM | Baseline |
| `42_convlstm_chunk_8515_tuned.py` | Hyperparameter tuning | Improved config |
| `45_convlstm_global_patches_tuned.py` | 120-patch global | Full grid |
| `47_convlstm_global_patches_context.py` | + LTDM/Lat/Lon channels | Context added |
| `48_convlstm_global_patches_context_fast.py` | GPU optimizations | T4 fast |

### Phase 2: Multi-Horizon Strategies (Scripts 49-51)

| Script | Description | Notes |
|--------|-------------|-------|
| `49_convlstm_mimo.py` | Multiple outputs | MIMO strategy |
| `50_convlstm_direct.py` | Independent models | Direct strategy |
| `51_convlstm_recursive.py` | Autoregressive | Recursive strategy |

### Phase 3: Specialized (Scripts 55-68)

| Script | Description | Notes |
|--------|-------------|-------|
| `55_convlstm_single_horizon.py` | Single horizon | LevelConditioned |
| `56_convlstm_t4_optimized.py` | GPU optimized | T4 optimizations |
| `57_convlstm_point_forecast_FINAL.py` | Point focus | Target emphasis |
| `58_convlstm_mimo_sst.py` | MIMO 3 horizons | 3 horizons |
| `59_convlstm_7day_sst.py` | Branching | 5 separate models |
| `60_convlstm_mimo_optimized.py` | Speed optimized | Colab version |
| `61_convlstm_branching_optimized.py` | Branching speed | Colab version |
| `62_convlstm_branching_kaggle.py` | Branching quality | Kaggle version |
| `63_convlstm_7day_focused_kaggle.py` | 7-day focus | Kaggle version |
| `64_convlstm_7day_final_v3.py` | Final v3 | Kaggle version |
| `65_convlstm_7day_stage2_finetune.py` | Fine-tune | Stage 2 |
| `66_convlstm_7day_stage2_final.py` | Stage 2 final | Production |
| `67_convlstm_final_ensemble.py` | Ensemble | Combined |
| `68_convlstm_ensemble_optimizer.py` | Ensemble+ | Optimized |

### Phase 4: Production (Script 69)

| Script | Description | Status |
|--------|-------------|--------|
| `69_convlstm_rolling_7day_fixed.py` | Rolling 7-day, Advanced post-processing | **5/5 gates** ✓ |

**Script 69 Achievements:**
- 21.5% RMSE improvement via post-processing
- 5/5 gates passed
- Production ready

---

## Utility Scripts

| Script | Description |
|--------|-------------|
| `test_56_setup.py` | Verify setup before running |
| `verify_script_56.py` | Quick syntax check |
| `startup/menu.py` | Session startup menu |

---

## Comparison & Visualization Scripts

| Script | Description | Outputs |
|--------|-------------|---------|
| `model_comparison_kaggle.py` | Multi-model comparison (ConvLSTM/Chronos/Granite) | Taylor diagram, error density, timeseries overlay, RMSE bar charts, violin plots, skill scores CSV |

---

## Argo Validation Scripts

| Script | Description | Status |
|--------|-------------|--------|
| `build_argo_validation_sets.py` | Builds aligned Argo/master/reanalysis CSVs | ✅ 37 points |
| `argo_filter_to_master.py` | Maps Argo points to master grid | ✅ Complete |
| `validate_argo_spatial_models.py` | Validates Chronos/Granite/ConvLSTM vs Argo | ✅ ConvLSTM best |

### Data Files

| File | Description |
|------|-------------|
| `validation_data/Argo_validsation_TSFM.xlsx` | Raw Argo float profiles |
| `validation_data/Argo_validsation_TSFM_reanalysis.nc` | Reanalysis SST (NetCDF) |
| `validation_data/argo_validation_tsfm.csv` | Final Argo validation (37 points) |
| `validation_data/master_appended_tsfm.csv` | Master grid at Argo locations |
| `validation_data/reanalysis_tsfm.csv` | Reanalysis at Argo locations |

### Outputs

| File | Description |
|------|-------------|
| `validation_data/validation_outputs/argo_spatial_validation_metrics.csv` | Aggregate metrics |
| `validation_data/validation_outputs/argo_spatial_validation_predictions.csv` | Per-point predictions |
| `validation_data/validation_outputs/plot_overlay_timeseries.png` | Timeseries plot |
| `validation_data/validation_outputs/plot_correlation_scatter.png` | Scatter plot |

---

## Run Commands

```bash
# ConvLSTM baseline
python 69_convlstm_rolling_7day_fixed.py

# Granite PostGain (new champion)
python 87_spatial_granite_only.py

# Chronos PostGain deterministic
python 88_spatial_chronos_only_deterministic.py

# Chronos PostGain
python 86_spatial_chronos_only.py

# Point ensemble (secondary)
RUN_ID=W1 python 84_foundation_stage3_slope_calibrated_ensemble.py

# Spatial ensemble (secondary)
RUN_ID=SE3 python 85_spatial_ensemble_stage3.py

# Chronos few-shot (historical)
RUN_ID=F1C python 81_chronos_fewshot_rolling.py

# Granite few-shot (historical)
RUN_ID=G1A python 80_granite_fewshot_rolling.py
```

---

*May 19, 2026*
