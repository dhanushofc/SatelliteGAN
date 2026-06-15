# Quick Reference - Key Metrics

## Project Status: ✅ Complete (Stage 2 + PostGain)

---

## Final Results (5-Gate Evaluation) — Single-Model Primary

| Gate | Metric | Granite 87 | Chronos 88 | Chronos 86 | ConvLSTM 69 | Target |
|------|--------|-----------|-----------|-----------|-------------|---------|
| 1 | Overall RMSE | **0.1196°C** ✓ | 0.1200°C ✓ | 0.1205°C ✓ | 0.1417°C ✓ | < 0.1466°C |
| 2 | February RMSE | 0.1704°C ✓ | **0.1640°C** ✓ | 0.1672°C ✓ | 0.2020°C ✓ | < 0.2093°C |
| 3 | March RMSE | **0.0857°C** ✓ | 0.0910°C ✓ | 0.0902°C ✓ | 0.0920°C ✓ | ≤ 0.1003°C |
| 4 | Big Errors (|e|≥0.20) | **9** ✓ | **9** ✓ | 9 ✓ | 11 ✓ | ≤ 12 |
| 5 | Slope | 0.9436 ✓ | 0.9488 ✓ | **0.9412** ✓ | 0.9408 ✓ | [0.94, 1.00] |

**Gates: All 5/5 ✓**

---

## Winner Selection

| Priority | Model | Score | Notes |
|----------|-------|-------|-------|
| Formal publication | Granite 87 | 5/5 ✓ + best RMSE | New champion |
| Best RMSE | Granite 87 | 0.1196°C | 16% better than ConvLSTM |
| Best February | Chronos 88 | 0.1640°C | Best single-model |
| Best March | Granite 87 | 0.0857°C | Best overall |
| Best Slope | Chronos 86 | 0.9412 | Within target |
| Deterministic | Chronos 88 | 5/5 ✓ | Reproducible |
| Resource-efficient | Granite 87 | 71K params | Tiny model |
| Robust baseline | ConvLSTM 69 | 5/5 ✓ | Proven stability |

---

## Ensemble Results (SECONDARY/OTHER)

| Run | Type | RMSE | Slope | Gates |
|-----|------|------|-------|-------|
| 84 W1 | Point ensemble | 0.1187 | 0.9756 | 5/5 |
| 85 SE3 | Spatial ensemble | 0.1187 | 0.9147 | 4/5 |

---

## Model Data Validation — Argo Float Spatial Validation

| Model | RMSE | MAE | R | R² | Slope | N |
|-------|------|-----|---|----|-------|---|
| **ConvLSTM** | **0.324°C** ✓ | **0.262** | **0.971** | **0.943** | 0.899 | 37 |
| Granite | 0.394°C | 0.301 | 0.959 | 0.920 | 0.892 | 37 |
| Chronos | 0.418°C | 0.322 | 0.955 | 0.911 | 0.914 | 37 |

**Best on Argo**: ConvLSTM (RMSE 0.324, R 0.971) — 17.7% better than Granite

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Data days | 16,290 |
| Train/Val/Test split | 85/5/10 |
| Target location | 8.0°N, 67.0°E |
| Grid size | 60×48 |
| Forecast horizon | 7 days |
| Evaluation period | Jan-Mar 2026 (90 days) |
| ConvLSTM runtime | ~2 hours |
| Chronos runtime | ~30 min |
| Granite runtime | ~15 min |
| Total runs completed | 25+ |

---

## Scripts

| Script | Purpose | Status |
|--------|--------|--------|
| 69_convlstm_rolling_7day_fixed.py | Final ConvLSTM | ✅ 5/5 |
| 87_spatial_granite_only.py | Granite PostGain | ✅ 5/5, RMSE 0.1196 |
| 88_spatial_chronos_only_deterministic.py | Chronos det PostGain | ✅ 5/5, RMSE 0.1200 |
| 86_spatial_chronos_only.py | Chronos PostGain | ✅ 5/5, RMSE 0.1205 |
| 84_foundation_stage3_slope_calibrated_ensemble.py | Point ensemble | ✅ 5/5 (secondary) |
| 85_spatial_ensemble_stage3.py | Spatial ensemble | ✅ 4/5 (secondary) |
| 81_chronos_fewshot_rolling.py | Chronos few-shot | ✅ F1C=0.1261 (historical) |
| 80_granite_fewshot_rolling.py | Granite few-shot | ✅ G1A=0.1272 (historical) |
| 82_chronos_lora_finetune_rolling.py | Chronos LoRA | ✅ L1=0.1291 (historical) |
| 83_granite_lora_finetune_rolling.py | Granite LoRA | ✅ GL3=0.1389 (historical) |

---

## Slope Issue — RESOLVED

| Model | Slope | Status | Notes |
|-------|-------|--------|-------|
| Granite 87 | 0.9436 ✓ | Pass | PostGain 1.020 |
| Chronos 88 | 0.9488 ✓ | Pass | PostGain 1.040 |
| Chronos 86 | 0.9412 ✓ | Pass | PostGain 1.040 |
| ConvLSTM 69 | 0.9408 ✓ | Pass | Baseline |
| Granite G1A | 0.9218 ❌ | Historical | Pre-PostGain |
| Chronos F1C | 0.8634 ❌ | Historical | Pre-PostGain |

**Root Cause**: Foundation models systematically compress amplitude
**Resolution**: PostGain slope correction (gain multiplier 1.020-1.040)

---

## Files

- Main doc: `CONVLSTM_PROJECT_DOCUMENTATION.md`
- Submission report: `FINAL_SUBMISSION_REPORT.md`
- This folder: `doumentation-things/`
- ConvLSTM out: `rolling window-ouputs/69_convlstm_rolling_7day_fixed_FINAL_W5_CP020_CN020_MS008/`
- PostGain outputs: `beats-chronos/86_spatial_chronos_only/`, `beats-chronos/87_spatial_granite_only/`, `beats-chronos/88_spatial_chronos_only_deterministic/`

---

*May 19, 2026*
