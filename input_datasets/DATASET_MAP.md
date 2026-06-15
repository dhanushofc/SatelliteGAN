# Dataset Map — INCOIS SST Forecasting

Maps every input dataset to the production scripts that use it.

## Master Data Files (179 MB each)

| File | Shape | Format | Description | Used By |
|------|-------|--------|-------------|---------|
| `master-harry-appended/master_region_data_new.npy` | (16290, 60, 48) | float32 | SST absolute temperature | 69, 86, 87, 88 |
| `master-harry-appended/master_region_anomalies_new.npy` | (16290, 60, 48) | float32 | SST anomaly (data - climatology) | 69, 86, 87, 88 |

**Local fallback path:** All scripts first try Kaggle (`/kaggle/input/datasets/rayofc/master-harry-appended/`), then fall back to `master-harry-appended/` (this directory) for local runs.

**Source:** `D:\INCOIS-internship\data\baka's appended data\` (originally from Kaggle dataset `rayofc/master-harry-appended`)

---

## Model Checkpoint (in `sir-desktop/` root)

| File | Size | Used By |
|------|------|---------|
| `model_stage2_best.pt` | 1.9 MB | 69, 86, 87, 88 |

ConvLSTM Stage 2 trained weights. All 4 models load this as the base model.

---

## ConvLSTM Reference Predictions (in `sir-desktop/model_comparison/`)

| File | Size | Used By |
|------|------|---------|
| `rolling_predictions_code_69.csv` | 14 KB | 86, 87, 88 (as baseline for PostGain) |
| `rolling_predictions_code_86.csv` | 9 KB | `model_comparison_kaggle.py` |
| `rolling_predictions_code_87.csv` | 9 KB | `model_comparison_kaggle.py` |

---

## Argo Validation Data (in `sir-desktop/validation_data/`)

| File | Size | Used By |
|------|------|---------|
| `Argo_validsation_TSFM.xlsx` | 15 KB | `build_argo_validation_sets.py` |
| `argo_validation_tsfm.csv` | 4 KB | `build_argo_validation_sets.py` |
| `Argo_validsation_TSFM_reanalysis.nc` | 2.2 MB | `build_argo_validation_sets.py` |
| `Argo_validsation_TSFM_filtered_to_master.csv` | 11 KB | `validate_argo_spatial_models.py` |
| `master_appended_tsfm.csv` | 4 KB | `validate_argo_spatial_models.py` |
| `reanalysis_tsfm.csv` | 4 KB | `validate_argo_spatial_models.py` |

---

## Primary vs Fallback Paths

Each production script resolves data in this order:

1. **Kaggle (primary):** `/kaggle/input/datasets/rayofc/master-harry-appended/` — used when run on Kaggle notebooks
2. **Local fallback:** `master-harry-appended/` (this directory) — used when running locally
3. **Validation/Comparison:** Hard-coded relative paths to `validation_data/` and `model_comparison/`

---

## Not Included

The file `OISST_60E_72E_5N_20N.csv` (1.4 GB, in `D:\INCOIS-internship\data\`) is the raw OISST source from which the `.npy` master files were preprocessed. It is not referenced by any production script and is excluded.
