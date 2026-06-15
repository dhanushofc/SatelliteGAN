# Documentation Folder - README

This folder contains all the formal documentation files for the INCOIS internship SST forecasting project.

## Quick Start

**For a quick overview:** Start with `QUICK_REFERENCE.md`

**For detailed analysis:** Read `MODEL_COMPARISON.md`

**For verification:** Check `VERIFICATION_PROOFS.md`

---

## File List

| File | Purpose |
|------|---------|
| `EXECUTIVE_SUMMARY.md` | High-level overview (2 pages) |
| `FINAL_RESULTS_TABLE.md` | All 25+ run leaderboard with proofs |
| `MODEL_COMPARISON.md` | Detailed multi-model comparison |
| `QUICK_REFERENCE.md` | Key metrics at a glance |
| `VERIFICATION_PROOFS.md` | How to verify each result |
| `SCRIPT_INDEX.md` | All scripts (39-88) listed |
| `KAGGLE_ARGO_SPATIAL_VALIDATION.md` | Argo validation Kaggle guide |
| `manuscript-dhanush.md` | IEEE-style research paper |
| `README.md` | This file |

---

## Output Directories

| Model | Location |
|-------|----------|
| ConvLSTM 69 | `rolling window-ouputs/69_convlstm_rolling_7day_fixed_FINAL_W5_CP020_CN020_MS008/` |
| Granite 87 (PostGain) | `beats-chronos/87_spatial_granite_only/` |
| Chronos 88 (PostGain det) | `beats-chronos/88_spatial_chronos_only_deterministic/` |
| Chronos 86 (PostGain) | `beats-chronos/86_spatial_chronos_only/` |
| Chronos | `4chorons-ouputs/` |
| Granite | `4granite-lagllama-outputs/` |

---

## Model Summary (Single-Model Primary)

| Model | Pipeline | RMSE | Slope | Gates | Status |
|-------|----------|------|-------|-------|--------|
| **Granite 87** | PostGain spatial | **0.1196°C** | 0.9436 | **5/5** | **New Champion** |
| **Chronos 88** | PostGain det | 0.1200°C | 0.9488 | **5/5** | Deterministic |
| **Chronos 86** | PostGain spatial | 0.1205°C | 0.9412 | **5/5** | Highest slope |
| ConvLSTM 69 | Rolling baseline | 0.1417°C | 0.9408 | **5/5** | Robust baseline |
| Chronos F1C | Few-shot (historical) | 0.1261°C | 0.8634 | 4/5 | Historical |
| Granite G1A | Few-shot (historical) | 0.1272°C | 0.9218 | 4/5 | Historical |

---

## Slope Issue — RESOLVED

**Previous state**: ALL foundation models failed the slope gate (target: 0.94-1.00)

**Current state**: PostGain slope correction achieves 5/5 gates for all single-model spatial pipelines:
- Granite 87: slope 0.9436 ✓ (PostGain 1.020)
- Chronos 88: slope 0.9488 ✓ (PostGain 1.040)
- Chronos 86: slope 0.9412 ✓ (PostGain 1.040)

---

## Ensemble Results (SECONDARY/OTHER)

| Run | Type | RMSE | Slope | Gates |
|-----|------|------|-------|-------|
| 84 W1 | Point ensemble | 0.1187 | 0.9756 | 5/5 |
| 85 SE3 | Spatial ensemble | 0.1187 | 0.9147 | 4/5 |

---

## Model Data Validation — Argo Float Spatial Validation

| Model | RMSE | R | Slope | N |
|-------|------|---|-------|---|
| **ConvLSTM** | **0.324°C** | **0.971** | 0.899 | 37 |
| Granite | 0.394°C | 0.959 | 0.892 | 37 |
| Chronos | 0.418°C | 0.955 | 0.914 | 37 |

**Pipeline**: `build_argo_validation_sets.py` → `argo_filter_to_master.py` → `validate_argo_spatial_models.py`
**Data**: `validation_data/` | **Outputs**: `validation_data/validation_outputs/`
**Guide**: `KAGGLE_ARGO_SPATIAL_VALIDATION.md`

---

*Last Updated: May 19, 2026*
