# Verification Proofs - Source References

## How to Verify Each Result

### ConvLSTM Script 69 Results

**Location**: `rolling window-ouputs/69_convlstm_rolling_7day_fixed_FINAL_W5_CP020_CN020_MS008/`

Run this to verify:
```bash
cd "rolling window-ouputs/69_convlstm_rolling_7day_fixed_FINAL_W5_CP020_CN020_MS008/"
cat run_summary.csv
```

Expected outputs:
- `plot1_spatial_*.png` - Spatial forecast maps
- `plot2_timeseries_90day.png` - Time series comparison
- `plot3_correlation_scatter.png` - Scatter correlation
- `run_summary.csv` - All metrics (verify 5/5)

---

### Granite PostGain (Script 87) — NEW CHAMPION

**Output Location**: `beats-chronos/87_spatial_granite_only/`
**Experiment Log**: `docs/experiments/code-87-stage2.md`

Run this to verify:
```bash
cd "beats-chronos/87_spatial_granite_only/"
cat final_report_summary.csv
```

Expected outputs:
- `plot1_spatial_january_2026_part1.png` - Spatial maps
- `plot1_spatial_january_2026_part2.png`
- `plot1_spatial_february_2026_part1.png`
- `plot1_spatial_february_2026_part2.png`
- `plot1_spatial_march_2026_part1.png`
- `plot1_spatial_march_2026_part2.png`
- `plot2_timeseries_90day.png` - Time series
- `plot3_correlation_scatter.png` - Scatter plot
- `plot4_correction_analysis.png` - Correction pipeline
- `plot5_comparison_monthly.png` - Monthly comparison
- `rolling_predictions.csv` - 90-day predictions
- `monthly_summary.csv` - Per-month metrics
- `final_report_summary.csv` - All metrics (verify 5/5, RMSE 0.1196, slope 0.9436)

---

### Chronos PostGain Deterministic (Script 88)

**Output Location**: `beats-chronos/88_spatial_chronos_only_deterministic/`
**Experiment Log**: `docs/experiments/code-88_spatial_chronos_only_deterministic.md`

Run this to verify:
```bash
cd "beats-chronos/88_spatial_chronos_only_deterministic/"
cat final_report_summary.csv
```

Expected outputs:
- All 5 plots (spatial × 2 parts per month, timeseries, correlation, correction, monthly)
- `rolling_predictions.csv` - 90-day predictions
- `monthly_summary.csv` - Per-month metrics
- `final_report_summary.csv` - All metrics (verify 5/5, RMSE 0.1200, slope 0.9488)

---

### Chronos PostGain (Script 86)

**Output Location**: `beats-chronos/86_spatial_chronos_only/`
**Experiment Log**: `docs/experiments/code-86-stage2.md`

Run this to verify:
```bash
cd "beats-chronos/86_spatial_chronos_only/"
cat final_report_summary.csv
```

Expected outputs:
- All 5 plots (spatial × 2 parts per month, timeseries, correlation, correction, monthly)
- `rolling_predictions.csv` - 90-day predictions
- `monthly_summary.csv` - Per-month metrics
- `final_report_summary.csv` - All metrics (verify 5/5, RMSE 0.1205, slope 0.9412)

---

### Point Ensemble (Script 84) — SECONDARY

**Location**: `docs/experiments/code-84.md`

Run this to verify:
```bash
cat docs/experiments/code-84.md
```

Expected: W1=0.1187, slope 0.9756, 5/5 gates

---

### Spatial Ensemble (Script 85) — SECONDARY

**Location**: `docs/experiments/code-85.md`

Run this to verify:
```bash
cat docs/experiments/code-85.md
```

Expected: SE3=0.1187, slope 0.9147, 4/5 gates

---

### Chronos Few-Shot F1C Results (Historical)

**Location**: `4chorons-ouputs/` (Kaggle outputs)

Run this to verify:
```bash
cd 4chorons-ouputs/
cat run_summary.csv
```

Expected outputs:
- `plot1_timeseries_90day.png` - Time series
- `plot2_correlation_scatter.png` - Scatter plot
- `rolling_predictions.csv` - 90-day predictions
- `run_summary.csv` - All metrics (verify 4/5)

---

### Granite Few-Shot G1A Results (Historical)

**Location**: `4granite-lagllama-outputs/stage-1.md`

Run this to verify:
```bash
cat 4granite-lagllama-outputs/stage-1.md
```

Expected outputs:
- `plot1_spatial_*.png` - Spatial maps
- `plot2_timeseries_90day.png` - Time series
- `plot3_correlation_scatter.png` - Scatter plot
- `run_summary.csv` - All metrics (verify 4/5)

---

### Complete Leaderboard

**Location**: `docs/experiments/best_results_summary.md`

This file contains:
- All 25+ runs across all categories
- Single-model spatial (86/87/88)
- Ensemble point (84 W0-W3)
- Ensemble spatial (85 SE1-SE4)
- Historical (few-shot/LoRA/zero-shot)

---

### LoRA Results (Historical)

**Chronos LoRA**: `docs/experiments/code-82.md`
- Lines 1-256: L1, L2, L3 results
- Training logs and metrics

**Granite LoRA**: `docs/experiments/code-83.md`
- Lines 1-236: GL1, GL2, GL3 results
- Training logs and metrics

---

### Invalid Runs (Calibration Bug)

**Affected**: F1E, F1F, G1E, G1F
- RMSE 2.6-2.8°C, gates 0/5
- Root cause: Intercept not recomputed after slope clipping
- Status: DISCARD from analysis

---

### Argo Float Spatial Validation

**Location**: `validation_data/validation_outputs/`

Run this to verify:
```bash
cat validation_data/validation_outputs/argo_spatial_validation_metrics.csv
```

Expected:
- ConvLSTM: RMSE 0.324, R 0.971, slope 0.899
- Granite: RMSE 0.394, R 0.959, slope 0.892
- Chronos: RMSE 0.418, R 0.955, slope 0.914

**Pipeline verification**:
```bash
cat validation_data/argo_validation_tsfm.csv | wc -l  # Should show 38 lines (header + 37 points)
cat validation_data/validation_outputs/argo_spatial_validation_predictions.csv | wc -l  # 38 lines
```

**Outputs**:
- `argo_spatial_validation_predictions.csv` — Per-point predictions (37 rows)
- `argo_spatial_validation_metrics.csv` — Aggregate metrics
- `plot_overlay_timeseries.png` — Timeseries overlay
- `plot_correlation_scatter.png` — Correlation scatter

**Kaggle guide**: `KAGGLE_ARGO_SPATIAL_VALIDATION.md`

---

## Verification Checklist

- [ ] Argo validation metrics show ConvLSTM best RMSE (0.324)
- [ ] Argo validation has 37 data points
- [ ] Validation outputs directory has all 4 files
- [ ] ConvLSTM 69 run_summary.csv shows 5/5 gates
- [ ] Granite 87 beats-chronos/87_spatial_granite_only/final_report_summary.csv shows 5/5, RMSE 0.1196
- [ ] Chronos 88 beats-chronos/88_spatial_chronos_only_deterministic/final_report_summary.csv shows 5/5, RMSE 0.1200
- [ ] Chronos 86 beats-chronos/86_spatial_chronos_only/final_report_summary.csv shows 5/5, RMSE 0.1205
- [ ] Ensemble 84 code-84.md shows W1=0.1187, 5/5
- [ ] Ensemble 85 code-85.md shows SE3=0.1187, 4/5
- [ ] Leaderboard file has all 25+ runs
- [ ] LoRA results match training logs
- [ ] Invalid runs are marked and discarded

---

## Common Issues

### If ConvLSTM shows 4/5 instead of 5/5
- Check: Is this using updated script 69?
- Old versions: 4/5 gates

### If Chronos F1C RMSE is higher
- Check: Are you using F1C config (not default)
- Wrong config: 0.18+ RMSE

### If plots missing
- Check: All model output directories have plots
- Required: 3-6 plots per model

---

## Output Directory Structure

```
beats-chronos/                          ← PostGain outputs (ConvLSTM-style)
├── 86_spatial_chronos_only/
│   ├── plot1_spatial_*_part1.png       ← Spatial maps (6 files)
│   ├── plot2_timeseries_90day.png
│   ├── plot3_correlation_scatter.png
│   ├── plot4_correction_analysis.png
│   ├── plot5_comparison_monthly.png
│   ├── rolling_predictions.csv
│   ├── monthly_summary.csv
│   └── final_report_summary.csv        ← Verify: RMSE 0.1205, 5/5
├── 87_spatial_granite_only/            ← NEW CHAMPION
│   ├── plot1_spatial_*_part1.png       ← Spatial maps (6 files)
│   ├── plot2_timeseries_90day.png
│   ├── plot3_correlation_scatter.png
│   ├── plot4_correction_analysis.png
│   ├── plot5_comparison_monthly.png
│   ├── rolling_predictions.csv
│   ├── monthly_summary.csv
│   └── final_report_summary.csv        ← Verify: RMSE 0.1196, 5/5
└── 88_spatial_chronos_only_deterministic/
    ├── plot1_spatial_*_part1.png       ← Spatial maps (6 files)
    ├── plot2_timeseries_90day.png
    ├── plot3_correlation_scatter.png
    ├── plot4_correction_analysis.png
    ├── plot5_comparison_monthly.png
    ├── rolling_predictions.csv
    ├── monthly_summary.csv
    └── final_report_summary.csv        ← Verify: RMSE 0.1200, 5/5

rolling window-ouputs/
└── 69_convlstm_rolling_7day_fixed_FINAL_W5_CP020_CN020_MS008/
    ├── plot1_spatial_*.png
    ├── plot2_timeseries_90day.png
    ├── plot3_correlation_scatter.png
    └── run_summary.csv

docs/experiments/
├── code-86-stage2.md          ← Chronos PostGain experiment log
├── code-87-stage2.md          ← Granite PostGain experiment log
├── code-88_spatial_chronos_only_deterministic.md  ← Chronos det log
├── code-84.md                 ← Point ensemble log
├── code-85.md                 ← Spatial ensemble log
├── best_results_summary.md    ← All results
└── summaries/
    └── handoff-19-05-morning.md

4chorons-ouputs/
├── 71_chronos_spatial_hybrid_a1/
├── 72_chronos_spatial_hybrid_ablation_kaggleputout_clipboard.md
└── FINAL_SUMMARY_A1.md

4granite-lagllama-outputs/
├── stage-1.md
├── stage-1but-slope-upgrade.md
└── stage-2[76+slope_enchancement].md
```

---

*May 19, 2026*
