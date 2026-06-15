# Final Results - Verified Metrics

## ConvLSTM Script 69 (Production Baseline)

**Source**: `rolling window-ouputs/69_convlstm_rolling_7day_fixed_FINAL_W5_CP020_CN020_MS008/`

| Metric | Value | Target | Gate | Status |
|--------|-------|--------|------|--------|
| Overall RMSE | 0.1417°C | < 0.1466°C | Gate 1 | ✅ PASS |
| February RMSE | 0.2020°C | < 0.2093°C | Gate 2 | ✅ PASS |
| March RMSE | 0.0920°C | ≤ 0.1003°C | Gate 3 | ✅ PASS |
| Big Error Days | 11 | ≤ 12 | Gate 4 | ✅ PASS |
| Slope | 0.9408 | [0.94, 1.00] | Gate 5 | ✅ PASS |

**Gates: 5/5** ✓ Full Production Compliance

---

## Single-Model Spatial (Zero-Shot + Post-Hoc) — PRIMARY FOCUS

**Note**: These are the primary single-model pipelines per advisor guidance. NOT fine-tuning, NOT LoRA. Zero-shot inference + post-hoc statistical corrections with PostGain slope targeting.

### Script 87 — Granite-Only Spatial (BEST SINGLE-MODEL)

**Output Dir**: `beats-chronos/87_spatial_granite_only/`
**Experiment Log**: `docs/experiments/code-87-stage2.md`

| Metric | Value | Target | Gate | Status |
|--------|-------|--------|------|--------|
| Overall RMSE | 0.1196°C | < 0.1466°C | Gate 1 | ✅ PASS |
| February RMSE | 0.1704°C | < 0.2093°C | Gate 2 | ✅ PASS |
| March RMSE | 0.0857°C | ≤ 0.1003°C | Gate 3 | ✅ PASS |
| Big Error Days | 9 | ≤ 12 | Gate 4 | ✅ PASS |
| Slope | 0.9436 | [0.94, 1.00] | Gate 5 | ✅ PASS |
| PostGain | 1.020 | - | - | - |

**Gates: 5/5** ✓ New Single-Model Champion

### Script 88 — Chronos Deterministic Variant

**Output Dir**: `beats-chronos/88_spatial_chronos_only_deterministic/`
**Experiment Log**: `docs/experiments/code-88_spatial_chronos_only_deterministic.md`

| Metric | Value | Target | Gate | Status |
|--------|-------|--------|------|--------|
| Overall RMSE | 0.1200°C | < 0.1466°C | Gate 1 | ✅ PASS |
| February RMSE | 0.1640°C | < 0.2093°C | Gate 2 | ✅ PASS |
| March RMSE | 0.0910°C | ≤ 0.1003°C | Gate 3 | ✅ PASS |
| Big Error Days | 9 | ≤ 12 | Gate 4 | ✅ PASS |
| Slope | 0.9488 | [0.94, 1.00] | Gate 5 | ✅ PASS |
| PostGain | 1.040 | - | - | - |

**Gates: 5/5** ✓ Deterministic Reproducibility

### Script 86 — Chronos-Only Spatial

**Output Dir**: `beats-chronos/86_spatial_chronos_only/`
**Experiment Log**: `docs/experiments/code-86-stage2.md`

| Metric | Value | Target | Gate | Status |
|--------|-------|--------|------|--------|
| Overall RMSE | 0.1205°C | < 0.1466°C | Gate 1 | ✅ PASS |
| February RMSE | 0.1672°C | < 0.2093°C | Gate 2 | ✅ PASS |
| March RMSE | 0.0902°C | ≤ 0.1003°C | Gate 3 | ✅ PASS |
| Big Error Days | 9 | ≤ 12 | Gate 4 | ✅ PASS |
| Slope | 0.9412 | [0.94, 1.00] | Gate 5 | ✅ PASS |
| PostGain | 1.040 | - | - | - |

**Gates: 5/5** ✓ Highest Slope

---

## Ensemble — Point-Only (SECONDARY/OTHER)

**Note**: These ensemble results are documented for completeness but are NOT the primary focus. Per advisor guidance, emphasis is on single-model pipelines. Script 84 loads cached predictions from F1C + G1A + L1 and combines through weighted ensemble. The tuner can collapse weights to a single model.

### Script 84 — Point Ensemble

**Source**: `docs/experiments/code-84.md`

| RUN_ID | Weights | Calibration | RMSE | Slope | Gates | Notes |
|--------|---------|-------------|------|-------|-------|-------|
| W1 | grid-tuned | No | 0.1187 | 0.9756 | 5/5 | Best overall RMSE |
| W3 | tuned | Yes | 0.1197 | 0.9782 | 5/5 | Second-best |
| W0 | equal 1/3 | No | 0.1208 | 0.9654 | 5/5 | Baseline |
| W2 | equal | Yes | 0.1226 | 0.9699 | 4/5 | Fails March gate |

---

## Ensemble — Spatial (SECONDARY/OTHER)

**Note**: Script 85 runs both Chronos + Granite with beta_map spatial propagation. The tuner can collapse to 100% one model. All configurations fail the slope gate.

### Script 85 — Spatial Ensemble

**Source**: `docs/experiments/code-85.md`

| RUN_ID | Weights | Calibrate | RMSE | Slope | Gates | Notes |
|--------|---------|-----------|------|-------|-------|-------|
| SE3 | equal | Yes | 0.1187 | 0.9147 | 4/5 | Best spatial ensemble |
| SE4 | tuned | Yes | 0.1203 | 0.9072 | 4/5 | Tuned + calibrated |
| SE1 | equal 0.5/0.5 | No | 0.1181 | 0.9280 | 4/5 | Equal baseline |
| SE2 | grid-tuned | No | 0.1184 | 0.9316 | 4/5 | Tuned only |

---

## Historical Results (Few-Shot / LoRA / Zero-Shot)

**Source**: `docs/research/professor_ready_rolling_benchmark_note.md`

| Rank | Pipeline | Run ID | RMSE ↓ | Feb RMSE | Mar RMSE | Slope ↑ | Gates ↑ | Notes |
|------|----------|--------|--------|----------|----------|---------|---------|-------|
| 1 | Chronos few-shot | **81 F1C** | **0.1261** | 0.1739 | 0.0948 | 0.8634 | 4/5 | Best RMSE (historical) |
| 2 | Granite few-shot | **80 G1A** | **0.1272** | 0.1762 | 0.0929 | 0.9218 | 4/5 | Best Granite (historical) |
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
| 15 | **ConvLSTM rolling** | **69 best** | 0.1417 | 0.2020 | **0.0920** | **0.9408** | **5/5** | **Robust baseline** |
| 16 | Granite LoRA r=8 | 83 GL1 | 0.1424 | 0.1701 | 0.1250 | 0.8856 | 2/5 | — |
| 17 | Chronos LoRA r=16 | 82 L2 | 0.1439 | 0.1710 | 0.1067 | 0.8964 | 2/5 | — |
| 18 | Chronos spatial-hybrid | **72 A1** | 0.1276 | 0.1752 | 0.0952 | 0.8974 | 4/5 | Phase 10 |
| 19 | Granite zero-shot | 80 G0A | 0.1470 | 0.1802 | 0.1291 | 0.9007 | 1/5 | — |
| 20 | Chronos t5-small | 70 ConfigA | 0.1840 | 0.2253 | 0.1582 | 0.8728 | 0/5 | Phase 3 |

**Invalid Runs** (calibration bug - intercept not recomputed):
- F1E, F1F, G1E, G1F: RMSE 2.6-2.8°C, gates 0/5 - DISCARD

---

## Three-Model Comparison (Best of Each Category)

| Model | Pipeline | RMSE | Feb RMSE | Mar RMSE | Big Err | Slope | Gates |
|-------|----------|------|----------|----------|---------|-------|-------|
| **Granite 87** | PostGain spatial | **0.1196** | 0.1704 | **0.0857** | **9** | **0.9436** | **5/5** |
| **Chronos 88** | PostGain det | 0.1200 | **0.1640** | 0.0910 | **9** | 0.9488 | **5/5** |
| **Chronos 86** | PostGain spatial | 0.1205 | 0.1672 | 0.0902 | 9 | 0.9412 | **5/5** |
| ConvLSTM 69 | Rolling baseline | 0.1417 | 0.2020 | 0.0920 | 11 | 0.9408 | **5/5** |

---

## Model Data Validation — Argo Float Spatial Validation

**Source**: `validation_data/validation_outputs/argo_spatial_validation_metrics.csv`

Independent validation against 37 Argo float profiles (Jan-Feb 2026) using in-situ oceanographic measurements:

| Model | RMSE | MAE | R | R² | Slope | Intercept | N |
|-------|------|-----|---|----|-------|-----------|---|
| **ConvLSTM** | **0.324°C** | **0.262** | **0.971** | **0.943** | 0.899 | 3.002 | 37 |
| Granite | 0.394°C | 0.301 | 0.959 | 0.920 | 0.892 | 3.257 | 37 |
| Chronos | 0.418°C | 0.322 | 0.955 | 0.911 | 0.914 | 2.648 | 37 |

**Pipeline**: `build_argo_validation_sets.py` → `argo_filter_to_master.py` → `validate_argo_spatial_models.py`
**Data**: Argo profiles (XLSX), reanalysis SST (NetCDF), master grid (NPY)
**Outputs**: `validation_data/validation_outputs/`

**Analysis**:
- ConvLSTM: Best RMSE (0.324°C), best correlation (R=0.971) — 17.7% better than Granite
- All models show slope < 0.92 — consistent with amplitude compression pattern
- Chronos: Highest slope (0.914) but worst RMSE — overconfident but less accurate

---

## Key Observations

1. **Granite 87 is the new single-model champion**: RMSE 0.1196°C (16% better than ConvLSTM), slope 0.9436, 5/5 gates
2. **PostGain slope correction resolves amplitude compression**: All three single-model spatial configs (86/87/88) achieve 5/5 gates
3. **PostGain gains are modest**: 1.020-1.040, indicating zero-shot predictions are close to correct amplitude
4. **Ensemble point (84 W1) achieves best RMSE**: 0.1187°C with 5/5 gates, but point-only and secondary
5. **Ensemble spatial (85) fails slope**: All 4 configs fail slope gate (4/5) — beta_map amplifies compression
6. **Few-shot with 689 windows** beats zero-shot by 8.6-13.5%
7. **LoRA fine-tuning works** but underperforms few-shot
8. **ConvLSTM 69 remains robust baseline**: Best March RMSE (0.0920°C), still 5/5 gates

---

## Verification Sources

| File/Dir | Content | Notes |
|------|---------|-------|
| `beats-chronos/87_spatial_granite_only/final_report_summary.csv` | Granite PostGain verified metrics | RMSE 0.1196, 5/5 |
| `beats-chronos/88_spatial_chronos_only_deterministic/final_report_summary.csv` | Chronos det verified metrics | RMSE 0.1200, 5/5 |
| `beats-chronos/86_spatial_chronos_only/final_report_summary.csv` | Chronos PostGain verified metrics | RMSE 0.1205, 5/5 |
| `validation_data/validation_outputs/argo_spatial_validation_metrics.csv` | Argo validation metrics | 37 points, 3 models |
| `validation_data/validation_outputs/argo_spatial_validation_predictions.csv` | Argo per-point predictions | 37 rows |
| `KAGGLE_ARGO_SPATIAL_VALIDATION.md` | Argo validation Kaggle guide | 64 lines |
| `docs/experiments/code-87-stage2.md` | Granite experiment log | 66 lines |
| `docs/experiments/code-88_spatial_chronos_only_deterministic.md` | Chronos det experiment log | 66 lines |
| `docs/experiments/code-86-stage2.md` | Chronos experiment log | 66 lines |
| `docs/experiments/code-84.md` | Point ensemble results | 63 lines |
| `docs/experiments/code-85.md` | Spatial ensemble results | 76 lines |
| `docs/research/professor_ready_rolling_benchmark_note.md` | Complete leaderboard | 160 lines |
| `docs/experiments/best_results_summary.md` | Best results summary | 14 lines |

---

*Verified: May 19, 2026*
