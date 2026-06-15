# Session Log: Chronos vs ConvLSTM Rolling Window Comparison

---

## May 21, 2026 — Argo Spatial Validation Documentation

**Session Goal**: Add Argo float spatial validation documentation to all files in `doumentation-things/`.

### What Was Added

**New validation category**: "Model Data Validation — Comparing the Models with Different Datasets"

**Argo Float Spatial Validation Results** (37 points, Jan-Feb 2026):

| Model | RMSE | MAE | R | R² | Slope | N |
|-------|------|-----|---|----|-------|---|
| **ConvLSTM** | **0.324°C** | **0.262** | **0.971** | **0.943** | 0.899 | 37 |
| Granite | 0.394°C | 0.301 | 0.959 | 0.920 | 0.892 | 37 |
| Chronos | 0.418°C | 0.322 | 0.955 | 0.911 | 0.914 | 37 |

**Key Finding**: ConvLSTM achieves 17.7% better RMSE than Granite and 22.5% better than Chronos against independent Argo measurements.

### Files Updated

1. `EXECUTIVE_SUMMARY.md` — Added Argo section + Key Finding #10 + source data reference
2. `FINAL_RESULTS_TABLE.md` — Added Argo validation section + verification sources
3. `MODEL_COMPARISON.md` — Added Argo validation section after Final Verdict
4. `QUICK_REFERENCE.md` — Added Argo validation table
5. `VERIFICATION_PROOFS.md` — Added Argo verification section + checklist items
6. `README.md` — Added Argo validation section + file list entry
7. `SCRIPT_INDEX.md` — Added Argo validation scripts section with data/outputs tables
8. `SESSION_LOG.md` — Added this entry

### Pipeline Scripts
- `build_argo_validation_sets.py` — Builds aligned Argo/master/reanalysis CSVs
- `argo_filter_to_master.py` — Maps Argo points to master grid
- `validate_argo_spatial_models.py` — Validates all 3 models against 37 Argo profiles

### Data Files
- `validation_data/Argo_validsation_TSFM.xlsx` — Raw Argo profiles
- `validation_data/Argo_validsation_TSFM_reanalysis.nc` — Reanalysis SST (NetCDF)
- `validation_data/argo_validation_tsfm.csv` — Final Argo validation CSV (37 points)

### Outputs
- `validation_data/validation_outputs/argo_spatial_validation_metrics.csv`
- `validation_data/validation_outputs/argo_spatial_validation_predictions.csv`
- `validation_data/validation_outputs/plot_overlay_timeseries.png`
- `validation_data/validation_outputs/plot_correlation_scatter.png`

---

## May 11, 2026 — Original Session
**Session Goal:** Match ConvLSTM (script 69) output format with Chronos, achieve 5/5 evaluation gates or best possible.

---

## Initial State

- ConvLSTM script 69 baseline: RMSE 0.1417, slope 0.9408, 5/5 gates ✓
- Chronos rolling point-only (script 74): multiple tuning attempts, best 4/5 but slope gate failed
- Spatial hybrid Chronos (script 72/73) with A1 config: 4/5 gates, slope 0.8974

---

## Work Performed

### 1. Rolling Chronos Point-Only (74_chronos_rolling_window_convlstm_style)

**Purpose:** ConvLSTM-style plots for point-only Chronos.

**Tuning attempts:**
- Base: t5-small, temp 0.5, median → gates failed (slope 0.8880)
- + t5-base, temp 0.3, trimmed_mean, z=2.0, window=7, step=0.06 → slope 0.9003 (still fail)
- + reduced clamp margin to 0.3 → slope 0.9202 (still fail)
- All runs: overall_rmse ~0.18, far above 0.1466

**Result:** Could not reach ConvLSTM gates. Output folder matches script name.

---

### 2. Spatial Hybrid Chronos (73_chronos_spatial_hybrid_convlstm_style)

**Purpose:** Match ConvLSTM output style and improve performance with spatial context.

**Base config A1:**
- `amazon/chronos-t5-base`
- temp=0.8, trimmed_mean, samples=20
- residual corrector + horizon residual + calibration
- adaptive window 5

**A1 run result:**
- overall_rmse: 0.1323
- rmse_feb: 0.1833
- rmse_mar: 0.0898
- big_error_count: 12
- slope: 0.8699
- gates: 4/5 (slope fail)
- Output folder: `73_chronos_spatial_hybrid_convlstm_style`

---

### 3. Slope Improvement Attempts (75_chronos_slope_improvement)

**Problem:** slope 0.8699–0.9046 < 0.94 threshold.

**Attempted fixes:**

1. Increase temperature to 0.90, lower calib_floor to 0.70 → slope 0.8725, gates 2/5 (worse)
2. calib_floor=1.00 → catastrophic: RMSE ~4, big_errors ~90
3. calib_floor=0.90 → slope passed (0.9456) but RMSE blew up (0.8867), big_errors=86
4. calib_floor=0.92 → similar degradation

**Conclusion:** Forcing amplitude calibration higher causes instability. No viable fix found while preserving RMSE and big_error gates.

---

## Final Status

### Best Chronos Run: Script 73 (A1)

**Metrics:**
- RMSE: 0.1316
- Feb: 0.1764
- Mar: 0.0920
- Big errors: 12
- Slope: 0.9046
- Gates: 4/5 (RMSE, Feb, Mar, big_errors pass; slope fails)

**Output files:**
```
outputs/73_chronos_spatial_hybrid_convlstm_style/
  plot1_spatial_january_2026_part1.png
  plot1_spatial_january_2026_part2.png
  plot1_spatial_february_2026_part1.png
  plot1_spatial_february_2026_part2.png
  plot1_spatial_march_2026_part1.png
  plot1_spatial_march_2026_part2.png
  plot2_timeseries_90day.png
  plot3_correlation_scatter.png
  rolling_predictions.csv
  monthly_summary.csv
  run_summary.csv
```

### ConvLSTM Baseline (Script 69): 5/5 gates, slope 0.9408

---

## Files to Publish

**Core scripts:**
1. `69_convlstm_rolling_7day_fixed.py` - ConvLSTM baseline (5/5)
2. `73_chronos_spatial_hybrid_convlstm_style.py` - Chronos A1 (4/5, best RMSE)
3. `74_chronos_rolling_window_convlstm_style.py` - Rolling point-only Chronos (proof-of-concept)
4. `75_chronos_slope_improvement.py` - Slope fix attempts (failed)

**Documentation:**
- `doumentation-things/README.md` (updated)
- `doumentation-things/MODEL_COMPARISON.md` (updated)
- `doumentation-things/EXECUTIVE_SUMMARY.md` (if exists)
- `doumentation-things/FINAL_RESULTS_TABLE.md` (if exists)
- `doumentation-things/QUICK_REFERENCE.md` (if exists)

---

## Recommendation

For publication: use ConvLSTM script 69 (5/5 gates). Include Chronos A1 (73) as a comparison alternative (4/5, better RMSE but slope deficiency).

---

*End of session log.*
