# Kaggle Guide: Argo Spatial Validation

This guide runs spatial validation against Argo points using Chronos, Granite, and ConvLSTM.

## 1) Required inputs

Upload or attach a Kaggle dataset containing:

- `validation_data/argo_validation_tsfm.csv`
- `validation_data/master_appended_tsfm.csv`
- `validation_data/reanalysis_tsfm.csv`
- `master_region_data_new.npy`
- `master_region_anomalies_new.npy`
- ConvLSTM checkpoint: `63_convlstm_v2finetune/66_convlstm_7day_stage2_final/model_stage2_best.pt`

Place them in a dataset folder like:
`/kaggle/input/your-dataset/`

If filenames differ, set env vars `ARGO_CSV`, `MASTER_CSV`, `REAN_CSV` to the full paths.

## 2) Notebook setup

Enable GPU (optional but faster).

Install model packages:

```bash
pip install -q chronos-forecasting granite-tsfm
```

## 3) Run the script

Place `validate_argo_spatial_models.py` in the notebook working directory and run:

```bash
python validate_argo_spatial_models.py
```

Optional overrides:

```bash
ARGO_CSV=/kaggle/input/your-dataset/validation_data/argo_validation_tsfm.csv \
MASTER_CSV=/kaggle/input/your-dataset/validation_data/master_appended_tsfm.csv \
REAN_CSV=/kaggle/input/your-dataset/validation_data/reanalysis_tsfm.csv \
python validate_argo_spatial_models.py
```

## 4) Outputs

Files saved to:

`/kaggle/working/validation_outputs/`

Outputs:
- `argo_spatial_validation_predictions.csv` (per-point predictions)
- `argo_spatial_validation_metrics.csv` (RMSE/MAE/R/slope per model)
- `plot_overlay_timeseries.png` (daily mean + Argo points)
- `plot_correlation_scatter.png` (model vs Argo scatter)

## 5) Notes

- Chronos runs in deterministic mode (NUM_SAMPLES=1, TEMPERATURE=0.0, TOP_P=1.0).
- SST is computed from Argo as min-pressure per profile/time.
- Reanalysis SST was converted from Kelvin to Celsius in `reanalysis_tsfm.csv`.
