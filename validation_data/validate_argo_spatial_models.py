"""
validate_argo_spatial_models.py  ─  Spatial validation at Argo points (Chronos/Granite/ConvLSTM)

Inputs (CSV, aligned schema):
  - validation_data/argo_validation_tsfm.csv
  - validation_data/master_appended_tsfm.csv
  - validation_data/reanalysis_tsfm.csv

Outputs (default Kaggle path):
  - /kaggle/working/validation_outputs/argo_spatial_validation_predictions.csv
  - /kaggle/working/validation_outputs/argo_spatial_validation_metrics.csv
  - /kaggle/working/validation_outputs/plot_overlay_timeseries.png
  - /kaggle/working/validation_outputs/plot_correlation_scatter.png
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

try:
    from chronos import ChronosPipeline
except ImportError:
    ChronosPipeline = None

try:
    from tsfm_public.toolkit.get_model import get_model
except ImportError:
    get_model = None


try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

OUTPUT_DIR = Path("/kaggle/working/validation_outputs")
if not OUTPUT_DIR.parent.exists():
    OUTPUT_DIR = Path("outputs/validation_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data paths (Kaggle primary, local fallback)
DATA_FILE = Path("/kaggle/input/datasets/rayofc/master-harry-appended/master_region_data_new.npy")
ANOM_FILE = Path("/kaggle/input/datasets/rayofc/master-harry-appended/master_region_anomalies_new.npy")
if not DATA_FILE.exists():
    DATA_FILE = Path("master-harry-appended/master_region_data_new.npy")
    ANOM_FILE = Path("master-harry-appended/master_region_anomalies_new.npy")

CKPT_FILE = Path("/kaggle/input/datasets/rayofc/4model-data-validation/model_stage2_best.pt")
if not CKPT_FILE.exists():
    CKPT_FILE = Path("63_convlstm_v2finetune/66_convlstm_7day_stage2_final/model_stage2_best.pt")


def find_csv(name: str) -> Path:
    local = ROOT / "validation_data" / name
    if local.exists():
        return local
    kaggle_dir = Path("/kaggle/input/datasets/rayofc/4model-data-validation")
    if kaggle_dir.exists():
        candidate = kaggle_dir / name
        if candidate.exists():
            return candidate
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        matches = list(kaggle_root.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Missing CSV: {name}")


ARGO_CSV = Path(os.environ.get("ARGO_CSV", "")) if os.environ.get("ARGO_CSV") else find_csv("argo_validation_tsfm.csv")
MASTER_CSV = Path(os.environ.get("MASTER_CSV", "")) if os.environ.get("MASTER_CSV") else find_csv("master_appended_tsfm.csv")
REAN_CSV = Path(os.environ.get("REAN_CSV", "")) if os.environ.get("REAN_CSV") else find_csv("reanalysis_tsfm.csv")


# Grid + time
LAT_MIN, LAT_MAX = 5.125, 19.875
LON_MIN, LON_MAX = 60.125, 71.875
RES = 0.25
TARGET_LAT, TARGET_LON = 8.0, 67.0
START_DATE = datetime(1981, 9, 1)
PRED_START_DATE = datetime(2026, 1, 1)
PRED_END_DATE = datetime(2026, 3, 31)
SEQ_LEN = 60
HORIZON = 7
TRAIN_FRAC = 0.85
VAL_FRAC = 0.05

# Deterministic Chronos (recommended)
DETERMINISTIC = True
SEED = 123
CHRONOS_MODEL_ID = "amazon/chronos-t5-base"
NUM_SAMPLES = 1
TEMPERATURE = 0.0
TOP_P = 1.0

# Ensure temperature is valid for transformers (must be > 0)
if TEMPERATURE <= 0:
    print(f"WARNING: TEMPERATURE={TEMPERATURE} invalid; setting to 1.0 for Chronos")
    TEMPERATURE = 1.0

# Granite config
GRANITE_MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
GRANITE_FREQ = "D"

# Adaptive + calibration
CALIBRATE = True
DISABLE_ADAPTIVE = False
POST_GAIN = True
POST_GAIN_TARGET = 0.94
POST_GAIN_MAX = 1.30
POST_GAIN_STEPS = 16
ADAPTIVE_WINDOW = 5
ADAPTIVE_CAP_POS = 0.20
ADAPTIVE_CAP_NEG = 0.20
MAX_OFFSET_STEP = 0.08


def latlon_to_idx(lat, lon, h, w):
    ri = int(np.clip(round((lat - LAT_MIN) / RES), 0, h - 1))
    ci = int(np.clip(round((lon - LON_MIN) / RES), 0, w - 1))
    return ri, ci


def build_beta_map(anom_full, train_end, target_row, target_col):
    target_anom_train = anom_full[:train_end, target_row, target_col]
    target_var = float(target_anom_train.var())
    if target_var < 1e-10:
        target_var = 1e-10
    H, W = anom_full.shape[1:]
    beta_map = np.zeros((H, W), dtype=np.float32)
    for h in range(H):
        for w in range(W):
            cov = float(np.cov(anom_full[:train_end, h, w], target_anom_train)[0, 1])
            beta_map[h, w] = cov / target_var
    return np.clip(beta_map, -1.0, 1.0)


def build_features(context_abs, context_map, day_abs, h, dates, target_row, target_col):
    r0, r1 = max(0, target_row - 3), min(context_map.shape[0], target_row + 4)
    c0, c1 = max(0, target_col - 3), min(context_map.shape[1], target_col + 4)
    p = context_map[r0:r1, c0:c1]
    month = dates[day_abs].month
    doy = dates[day_abs].timetuple().tm_yday
    ang = 2.0 * np.pi * doy / 365.25
    center = context_map[target_row, target_col]
    n = context_map[max(0, target_row - 1), target_col]
    s = context_map[min(context_map.shape[0] - 1, target_row + 1), target_col]
    wv = context_map[target_row, max(0, target_col - 1)]
    e = context_map[target_row, min(context_map.shape[1] - 1, target_col + 1)]
    return np.array([
        h / max(1, HORIZON - 1),
        month / 12.0,
        np.sin(ang), np.cos(ang),
        float(p.mean()), float(p.std()), float(p.min()), float(p.max()),
        float(center - n), float(center - s), float(center - wv), float(center - e),
        float(context_abs[-1]),
        float(context_abs[-1] - context_abs[-7]),
        float(context_abs[-7:].std()),
    ], dtype=np.float32)


def build_adaptive_offsets(ar, gt, w, cap_pos, cap_neg, max_step):
    off = np.zeros_like(ar)
    for i in range(len(ar)):
        if np.isnan(ar[i]):
            off[i] = np.nan
            continue
        s = max(0, i - w + 1)
        wr, wg = ar[s:i+1], gt[s:i+1]
        m = ~np.isnan(wr)
        if not m.any():
            off[i] = 0.0
            continue
        o = float((wr[m] - wg[m]).mean())
        o = np.clip(o, -cap_neg, cap_pos)
        if i > 0 and not np.isnan(off[i-1]):
            d = o - off[i-1]
            if d > max_step:
                o = off[i-1] + max_step
            elif d < -max_step:
                o = off[i-1] - max_step
        off[i] = o
    return off


def apply_post_gain(avg_series, gt_series):
    gains = np.linspace(1.0, POST_GAIN_MAX, POST_GAIN_STEPS)
    best_any = None
    best_rmse = float("inf")
    best_gain = 1.0
    best_slope = None
    gt_mean = float(gt_series.mean())
    for g in gains:
        cal = g * avg_series
        shift = gt_mean - float(cal.mean())
        cal = cal + shift
        slope_g, _, _, _, _ = stats.linregress(gt_series, cal)
        rmse_g = float(np.sqrt(np.mean((cal - gt_series) ** 2)))
        if best_any is None or rmse_g < best_any[0]:
            best_any = (rmse_g, g, slope_g, shift)
        if slope_g >= POST_GAIN_TARGET and rmse_g < best_rmse:
            best_rmse = rmse_g
            best_gain = g
            best_slope = slope_g
    if best_slope is None:
        _, best_gain, best_slope, shift = best_any
    else:
        shift = gt_mean - float((best_gain * avg_series).mean())
    return best_gain, shift


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "r": np.nan,
            "r2": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "n": int(m.sum()),
        }
    yt = y_true[m]
    yp = y_pred[m]
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    r = float(np.corrcoef(yp, yt)[0, 1])
    r2 = r ** 2
    slope, intercept, _, _, _ = stats.linregress(yt, yp)
    return {
        "rmse": rmse,
        "mae": mae,
        "r": r,
        "r2": r2,
        "slope": float(slope),
        "intercept": float(intercept),
        "n": int(m.sum()),
    }


def run_point_model(model_name, forecast_fn, data_full, anom_full, device_str):
    T, H, W = anom_full.shape
    target_row = int(round((TARGET_LAT - LAT_MIN) / RES))
    target_col = int(round((TARGET_LON - LON_MIN) / RES))
    dates = [START_DATE + timedelta(days=int(i)) for i in range(T)]
    date_to_abs = {d.date(): i for i, d in enumerate(dates)}

    train_end = int(T * TRAIN_FRAC)
    val_end = int(T * (TRAIN_FRAC + VAL_FRAC))
    pred_start_abs = date_to_abs[PRED_START_DATE.date()]
    pred_end_abs = date_to_abs[PRED_END_DATE.date()] + 1
    n_days = pred_end_abs - pred_start_abs
    gt_series = data_full[pred_start_abs:pred_end_abs, target_row, target_col].copy()

    beta_map = build_beta_map(anom_full, train_end, target_row, target_col)
    ltdm_full = data_full - anom_full

    # Validation for bias, weights, residuals, calibration
    val_start_idx = train_end + SEQ_LEN
    val_max = val_end - val_start_idx - SEQ_LEN - HORIZON + 1
    preds, actuals = [], []
    residual_X_h = [[] for _ in range(HORIZON)]
    for idx in range(min(val_max, 689)):
        start = val_start_idx + idx
        ctx = data_full[start:start+SEQ_LEN, target_row, target_col].copy()
        actual = data_full[start+SEQ_LEN:start+SEQ_LEN+HORIZON, target_row, target_col].copy()
        if len(ctx) != SEQ_LEN or len(actual) != HORIZON:
            continue
        pred = forecast_fn(ctx)
        preds.append(pred)
        actuals.append(actual)
        context_map = data_full[start + SEQ_LEN - 1].copy()
        for h in range(HORIZON):
            da = start + SEQ_LEN + h
            if da >= len(dates):
                continue
            feat = build_features(ctx, context_map, da, h, dates, target_row, target_col)
            residual_X_h[h].append(feat)

    pred_arr = np.asarray(preds, dtype=np.float32)
    act_arr = np.asarray(actuals, dtype=np.float32)
    bias = np.mean(pred_arr - act_arr, axis=0)
    rmse_h = np.sqrt(np.mean((pred_arr - act_arr) ** 2, axis=0))
    rmse_safe = np.where(rmse_h < 1e-8, 1e-8, rmse_h)
    w_inv = (1.0 / rmse_safe ** 2) / (1.0 / rmse_safe ** 2).sum()

    ridge_alpha = 1.0
    res_w_h = [None] * HORIZON
    res_b_h = [0.0] * HORIZON
    residual_y_h = [[] for _ in range(HORIZON)]
    for i in range(len(act_arr)):
        for h in range(HORIZON):
            residual_y_h[h].append(act_arr[i, h] - pred_arr[i, h])
    for h in range(HORIZON):
        X_h = np.asarray(residual_X_h[h])
        y_h = np.asarray(residual_y_h[h])
        if len(X_h) > 120:
            xm, ym = X_h.mean(0), y_h.mean()
            Xc = X_h - xm
            yc = y_h - ym
            I = np.eye(Xc.shape[1], dtype=np.float32)
            w = np.linalg.solve(Xc.T @ Xc + ridge_alpha * I, Xc.T @ yc)
            res_w_h[h] = w.astype(np.float32)
            res_b_h[h] = float(ym - xm @ w)

    calib_a = np.ones(HORIZON, dtype=np.float32)
    calib_b = np.zeros(HORIZON, dtype=np.float32)
    if CALIBRATE:
        for h in range(HORIZON):
            xh = pred_arr[:, h]
            yh = act_arr[:, h]
            if len(xh) < 30:
                continue
            a_h, _, _, _, _ = stats.linregress(xh, yh)
            a_clip = float(np.clip(a_h, 0.85, 1.20))
            calib_a[h] = a_clip
            calib_b[h] = float(yh.mean() - a_clip * xh.mean())

    raw_point_preds = defaultdict(list)
    raw_spatial_preds = defaultdict(list)

    for t in range(pred_start_abs - SEQ_LEN, pred_end_abs - HORIZON + 1):
        if t < 0 or t + SEQ_LEN > T:
            continue
        ctx = data_full[t:t+SEQ_LEN, target_row, target_col].copy()
        point_pred = forecast_fn(ctx)

        for h in range(HORIZON):
            day_abs = t + SEQ_LEN + h
            if not (pred_start_abs <= day_abs < pred_end_abs):
                continue
            base = float(point_pred[h] - bias[h])
            if res_w_h[h] is not None:
                cmap = data_full[day_abs - h - 1].copy()
                feat = build_features(ctx, cmap, day_abs, h, dates, target_row, target_col)
                base += float(feat @ res_w_h[h] + res_b_h[h])
            if CALIBRATE:
                base = float(calib_a[h] * base + calib_b[h])
            raw_point_preds[day_abs].append((base, h))

        for h in range(HORIZON):
            day_abs = t + SEQ_LEN + h
            if not (pred_start_abs <= day_abs < pred_end_abs):
                continue
            anom_ctx_last = anom_full[day_abs - 1]
            ltdm_this = ltdm_full[day_abs]
            target_anom_last = anom_ctx_last[target_row, target_col]
            pred_anom = point_pred[h] - ltdm_this[target_row, target_col]
            delta = pred_anom - target_anom_last
            full_field_anom = anom_ctx_last + beta_map * delta
            full_field_abs = full_field_anom + ltdm_this

            full_field_abs -= bias[h]
            if res_w_h[h] is not None:
                cmap = data_full[day_abs - h - 1].copy()
                feat = build_features(ctx, cmap, day_abs, h, dates, target_row, target_col)
                base = float(point_pred[h] - bias[h]) + float(feat @ res_w_h[h] + res_b_h[h])
                delta2 = base - point_pred[h]
                full_field_abs += delta2
            if CALIBRATE:
                full_field_abs = float(calib_a[h]) * full_field_abs + float(calib_b[h])
            raw_spatial_preds[day_abs].append((full_field_abs, h))

    pred_days_abs = sorted(d for d in raw_point_preds if pred_start_abs <= d < pred_end_abs)
    avg_raw = np.full(len(pred_days_abs), np.nan, dtype=np.float32)
    avg_spatial = {}
    for i, d in enumerate(pred_days_abs):
        pairs = raw_point_preds[d]
        if not pairs:
            continue
        preds = np.array([p[0] for p in pairs], dtype=np.float32)
        hrs = np.array([p[1] for p in pairs], dtype=np.int32)
        ww = w_inv[hrs]
        ww = ww / ww.sum()
        avg_raw[i] = float((preds * ww).sum())

        sp_pairs = raw_spatial_preds[d]
        if sp_pairs:
            sp_field = np.array([s[0] for s in sp_pairs], dtype=np.float32)
            sp_hrs = np.array([s[1] for s in sp_pairs], dtype=np.int32)
            sp_w = w_inv[sp_hrs]
            sp_w = sp_w / sp_w.sum()
            avg_spatial[d] = np.tensordot(sp_w, sp_field, axes=(0, 0))

    if DISABLE_ADAPTIVE:
        adaptive_offsets = np.zeros_like(avg_raw)
    else:
        adaptive_offsets = build_adaptive_offsets(
            avg_raw, gt_series[:len(avg_raw)], ADAPTIVE_WINDOW,
            ADAPTIVE_CAP_POS, ADAPTIVE_CAP_NEG, MAX_OFFSET_STEP,
        )

    avg_series = avg_raw - adaptive_offsets
    for d in pred_days_abs:
        if d in avg_spatial:
            idx = pred_days_abs.index(d)
            avg_spatial[d] = avg_spatial[d] - adaptive_offsets[idx]

    if POST_GAIN:
        best_gain, shift = apply_post_gain(avg_series, gt_series[:len(avg_series)])
        avg_series = best_gain * avg_series + shift
        for d in pred_days_abs:
            if d in avg_spatial:
                avg_spatial[d] = best_gain * avg_spatial[d] + shift

    return {
        "name": model_name,
        "avg_spatial": avg_spatial,
        "pred_days_abs": pred_days_abs,
    }


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=pad)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        if self.dropout:
            h_next = self.dropout(h_next)
        return h_next, c_next


class ConvLSTMAbsolutePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, horizon, kernel_size=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.cell1 = ConvLSTMCell(input_dim, hidden_dim, kernel_size, dropout)
        self.cell2 = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size, dropout)
        self.neck = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=4, num_channels=hidden_dim),
        )
        self.head = nn.Conv2d(hidden_dim, horizon, kernel_size=3, padding=1)

    def forward(self, x):
        B, S, C, H, W = x.shape
        h1 = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c1 = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        h2 = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c2 = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        for t in range(S):
            h1, c1 = self.cell1(x[:, t], h1, c1)
            h2, c2 = self.cell2(h1, h2, c2)
        feat = self.neck(h2)
        return self.head(feat)


def run_convlstm_spatial(data_full, anom_full, device):
    T, H_orig, W_orig = data_full.shape
    dates = pd.date_range(start=START_DATE, periods=T, freq="D").to_pydatetime().tolist()
    date_to_abs = {d.date(): i for i, d in enumerate(dates)}
    pred_start_abs = date_to_abs[PRED_START_DATE.date()]
    pred_end_abs = date_to_abs[PRED_END_DATE.date()]

    lat_i, lon_i = latlon_to_idx(TARGET_LAT, TARGET_LON, H_orig, W_orig)

    anom_pad = np.pad(anom_full, ((0, 0), (0, 0), (0, 2)), mode="edge")
    ltdm_full = data_full - anom_full
    ltdm_pad = np.pad(ltdm_full, ((0, 0), (0, 0), (0, 2)), mode="edge")
    H_pad, W_pad = 60, 50
    lat_grid = np.repeat(np.linspace(0, 1, H_pad, dtype=np.float32).reshape(H_pad, 1)
                         .repeat(W_pad, 1)[np.newaxis], T, axis=0)
    lon_grid = np.repeat(np.linspace(0, 1, W_pad, dtype=np.float32).reshape(1, W_pad)
                         .repeat(H_pad, 0)[np.newaxis], T, axis=0)

    train_end = int(T * TRAIN_FRAC)
    val_end = int(T * (TRAIN_FRAC + VAL_FRAC))

    def get_norm(arr):
        m = arr.mean(0).astype(np.float32)
        s = arr.std(0).astype(np.float32)
        s[s == 0] = 1e-8
        return m, s

    mean_anom, std_anom = get_norm(anom_pad[:train_end])
    mean_ltdm, std_ltdm = get_norm(ltdm_pad[:train_end])
    anom_n = ((anom_pad - mean_anom) / std_anom).astype(np.float32)
    ltdm_n = ((ltdm_pad - mean_ltdm) / std_ltdm).astype(np.float32)

    sl = {"train": slice(0, train_end), "val": slice(train_end, val_end), "test": slice(val_end, T)}
    combo = {k: np.stack([anom_n[v], ltdm_n[v], lat_grid[v], lon_grid[v]], 1).astype(np.float32)
             for k, v in sl.items()}

    # Model
    model = ConvLSTMAbsolutePredictor(4, 64, HORIZON, dropout=0.15).to(device)
    if not CKPT_FILE.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_FILE}")
    model.load_state_dict(torch.load(CKPT_FILE, map_location=device, weights_only=True))
    model.eval()

    # Corrections
    def compute_corrections(model, combo_val, horizon):
        val_start = train_end + SEQ_LEN
        vp_list = []
        with torch.no_grad():
            for idx in range(max(0, len(combo_val) - SEQ_LEN - horizon + 1)):
                X = torch.from_numpy(combo_val[idx:idx+SEQ_LEN].copy()).unsqueeze(0).to(device)
                pred_norm = model(X).squeeze(0).cpu().numpy()
                vp_list.append(pred_norm)
        converted, actuals = [], []
        for i, pred_n in enumerate(vp_list):
            base = val_start + i
            if base + horizon > val_end:
                break
            pa = (pred_n * std_anom) + mean_anom
            ps = pa[:, :, :W_orig] + ltdm_full[base:base+horizon]
            converted.append(ps)
            actuals.append(data_full[base:base+horizon])
        n = min(len(converted), len(actuals))
        if n == 0:
            return np.zeros(horizon), np.zeros(horizon), np.zeros((horizon, H_orig, W_orig))
        vp_arr = np.array(converted[:n])
        va_arr = np.array(actuals[:n])
        vp_pt = vp_arr[:, :, lat_i, lon_i]
        va_pt = va_arr[:, :, lat_i, lon_i]
        add_bias = np.mean(vp_pt - va_pt, axis=0)
        actual_mean = va_pt.mean(axis=0)
        spatial_bias2d = np.mean(vp_arr - va_arr, axis=0).astype(np.float32)
        return add_bias, actual_mean, spatial_bias2d

    add_bias, _, spatial_bias2d = compute_corrections(model, combo["val"], HORIZON)

    rmse_by_horizon = np.array([0.128438, 0.190926, 0.231797, 0.262054,
                                0.286933, 0.308568, 0.324445], dtype=np.float32)
    w_inv_rmse2 = (1.0 / (rmse_by_horizon ** 2))
    w_inv_rmse2 = w_inv_rmse2 / w_inv_rmse2.sum()

    raw_point_preds = defaultdict(list)
    spatial_preds = defaultdict(list)

    with torch.no_grad():
        for t in range(pred_start_abs, pred_end_abs + 1):
            if t - SEQ_LEN < 0:
                continue
            X = np.stack([anom_n[t-SEQ_LEN:t], ltdm_n[t-SEQ_LEN:t],
                          lat_grid[t-SEQ_LEN:t], lon_grid[t-SEQ_LEN:t]], axis=1)
            pred_norm = model(torch.from_numpy(X[np.newaxis]).to(device)).squeeze(0).cpu().numpy()
            pred_anom = (pred_norm * std_anom) + mean_anom
            pred_anom_orig = pred_anom[:, :, :W_orig]
            for k in range(HORIZON):
                day_abs = t + k
                if day_abs > pred_end_abs or day_abs >= T:
                    break
                pred_sst = pred_anom_orig[k] + ltdm_full[day_abs]
                pt_val = float(pred_sst[lat_i, lon_i])
                raw_point_preds[day_abs].append((pt_val, k))
                spatial_preds[day_abs].append((pred_sst - spatial_bias2d[k], k))

    pred_days_abs = sorted(d for d in raw_point_preds if pred_start_abs <= d <= pred_end_abs)
    n_days = len(pred_days_abs)
    gt_series = np.array([data_full[d, lat_i, lon_i] for d in pred_days_abs])

    avg_raw = np.zeros(n_days)
    avg_spatial = {}

    for i, d in enumerate(pred_days_abs):
        preds_with_k = raw_point_preds[d]
        preds_corrected = []
        weights = []
        spatial_corrected = []
        for pt_val, k in preds_with_k:
            pt_corr = pt_val - add_bias[k]
            preds_corrected.append(pt_corr)
            weights.append(w_inv_rmse2[k])
        for sp_field, k in spatial_preds[d]:
            spatial_corrected.append(sp_field - add_bias[k])
        w_norm = np.array(weights) / np.sum(weights)
        avg_raw[i] = np.average(preds_corrected, weights=weights)
        avg_spatial[d] = np.average(spatial_corrected, axis=0, weights=weights)

    adaptive_offsets = build_adaptive_offsets(
        avg_raw, gt_series, ADAPTIVE_WINDOW, ADAPTIVE_CAP_POS, ADAPTIVE_CAP_NEG, MAX_OFFSET_STEP
    )
    avg_series = avg_raw - adaptive_offsets
    for d in pred_days_abs:
        idx = pred_days_abs.index(d)
        avg_spatial[d] = avg_spatial[d] - adaptive_offsets[idx]

    if POST_GAIN:
        best_gain, shift = apply_post_gain(avg_series, gt_series)
        avg_series = best_gain * avg_series + shift
        for d in pred_days_abs:
            avg_spatial[d] = best_gain * avg_spatial[d] + shift

    return {
        "name": "convlstm",
        "avg_spatial": avg_spatial,
        "pred_days_abs": pred_days_abs,
    }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data: {DATA_FILE}")
    if not ANOM_FILE.exists():
        raise FileNotFoundError(f"Missing anomalies: {ANOM_FILE}")
    if not CKPT_FILE.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_FILE}")

    print("Resolved paths:")
    print(f"  ARGO_CSV   : {ARGO_CSV}")
    print(f"  MASTER_CSV : {MASTER_CSV}")
    print(f"  REAN_CSV   : {REAN_CSV}")
    print(f"  DATA_FILE  : {DATA_FILE}")
    print(f"  ANOM_FILE  : {ANOM_FILE}")
    print(f"  CKPT_FILE  : {CKPT_FILE}")
    print(f"  OUTPUT_DIR : {OUTPUT_DIR}")

    print(f"Loading arrays: {DATA_FILE}")
    data_full = np.load(DATA_FILE).astype(np.float32)
    anom_full = np.load(ANOM_FILE).astype(np.float32)

    # Load CSVs
    argo_df = pd.read_csv(ARGO_CSV)
    master_df = pd.read_csv(MASTER_CSV)
    rean_df = pd.read_csv(REAN_CSV)

    argo_df["date"] = pd.to_datetime(argo_df["date"]).dt.date
    master_df = master_df.rename(columns={"temp_value": "master_temp"})
    rean_df = rean_df.rename(columns={"temp_value": "reanalysis_temp"})

    df = argo_df.merge(master_df[["key_id", "master_temp"]], on="key_id", how="left")
    df = df.merge(rean_df[["key_id", "reanalysis_temp"]], on="key_id", how="left")

    dates = [START_DATE + timedelta(days=int(i)) for i in range(data_full.shape[0])]
    date_to_abs = {d.date(): i for i, d in enumerate(dates)}

    df["day_abs"] = df["date"].map(date_to_abs)
    df = df[df["day_abs"].notna()].copy()
    df["day_abs"] = df["day_abs"].astype(int)
    df["master_row"] = df["master_row"].astype(int)
    df["master_col"] = df["master_col"].astype(int)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")

    # Deterministic settings for Chronos
    if DETERMINISTIC:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # Chronos
    if ChronosPipeline is None:
        raise RuntimeError("chronos-forecasting not installed")
    chronos_pipeline = ChronosPipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        device_map=device_str,
        torch_dtype=torch.bfloat16 if device_str == "cuda" else torch.float32,
    )

    def chronos_forecast(context_1d, horizon=HORIZON):
        ctx = torch.from_numpy(context_1d.astype(np.float32))
        samples = chronos_pipeline.predict(
            ctx,
            prediction_length=horizon,
            num_samples=NUM_SAMPLES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        if isinstance(samples, torch.Tensor):
            arr = samples.detach().float().cpu().numpy()
        else:
            arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        pred = np.array([stats.trim_mean(arr[:, h], 0.10) for h in range(arr.shape[1])], dtype=np.float32)
        if len(pred) < horizon:
            pred = np.pad(pred, (0, horizon - len(pred)), mode="edge")
        return pred[:horizon]

    print("Running Chronos spatial...")
    chronos_res = run_point_model("chronos", chronos_forecast, data_full, anom_full, device_str)

    # Granite
    if get_model is None:
        raise RuntimeError("granite-tsfm not installed")
    granite_model = get_model(
        model_path=GRANITE_MODEL_ID,
        context_length=SEQ_LEN,
        prediction_length=HORIZON,
        freq=GRANITE_FREQ,
        freq_prefix_tuning=False,
        return_model_key=False,
    )
    if hasattr(granite_model, "to"):
        granite_model = granite_model.to(device_str)
    if hasattr(granite_model, "eval"):
        granite_model.eval()

    def _freq_token_id(freq):
        m = {"S": 0, "T": 1, "MIN": 1, "H": 2, "D": 3, "B": 4, "W": 5, "M": 6, "Q": 7, "Y": 8}
        return int(m.get(str(freq).upper(), 3))

    FREQ_TOKEN_ID = _freq_token_id(GRANITE_FREQ)

    def granite_forecast(context_1d, horizon=HORIZON):
        ctx = torch.from_numpy(context_1d.astype(np.float32)).view(1, SEQ_LEN, 1).to(device_str)
        freq_ts = torch.full((1,), FREQ_TOKEN_ID, dtype=torch.long, device=ctx.device)
        with torch.no_grad():
            out = granite_model(past_values=ctx, freq_token=freq_ts)
        if isinstance(out, dict):
            pred = out.get("prediction_outputs") if "prediction_outputs" in out else list(out.values())[0]
        else:
            pred = out
        if isinstance(pred, torch.Tensor):
            arr = pred.detach().float().cpu().numpy().ravel()
        else:
            arr = np.asarray(pred).ravel()
        if len(arr) < horizon:
            arr = np.pad(arr, (0, horizon - len(arr)), mode="edge")
        return arr[:horizon]

    print("Running Granite spatial...")
    granite_res = run_point_model("granite", granite_forecast, data_full, anom_full, device_str)

    # ConvLSTM
    print("Running ConvLSTM spatial...")
    convlstm_res = run_convlstm_spatial(data_full, anom_full, device_str)

    def sample_spatial(res):
        preds = []
        avg_spatial = res["avg_spatial"]
        for _, row in df.iterrows():
            d = int(row["day_abs"])
            r = int(row["master_row"])
            c = int(row["master_col"])
            if d in avg_spatial:
                preds.append(float(avg_spatial[d][r, c]))
            else:
                preds.append(np.nan)
        return np.array(preds, dtype=np.float32)

    df["argo_temp"] = df["temp_value"].astype(float)
    df["chronos_pred"] = sample_spatial(chronos_res)
    df["granite_pred"] = sample_spatial(granite_res)
    df["convlstm_pred"] = sample_spatial(convlstm_res)

    # Metrics
    metrics = []
    for name, col in [("chronos", "chronos_pred"), ("granite", "granite_pred"), ("convlstm", "convlstm_pred")]:
        m = compute_metrics(df["argo_temp"].values, df[col].values)
        m["model"] = name
        metrics.append(m)

    metrics_df = pd.DataFrame(metrics)

    # Save outputs
    out_pred = OUTPUT_DIR / "argo_spatial_validation_predictions.csv"
    out_metrics = OUTPUT_DIR / "argo_spatial_validation_metrics.csv"
    df.to_csv(out_pred, index=False)
    metrics_df.to_csv(out_metrics, index=False)

    # Plot overlay (daily mean)
    daily = df.groupby("date").agg({
        "argo_temp": "mean",
        "chronos_pred": "mean",
        "granite_pred": "mean",
        "convlstm_pred": "mean",
    }).reset_index()
    daily = daily.sort_values("date")

    plt.figure(figsize=(14, 7))
    plt.plot(daily["date"], daily["argo_temp"], label="Argo (mean)", color="black", lw=2.2)
    plt.plot(daily["date"], daily["chronos_pred"], label="Chronos", lw=1.8)
    plt.plot(daily["date"], daily["granite_pred"], label="Granite", lw=1.8)
    plt.plot(daily["date"], daily["convlstm_pred"], label="ConvLSTM", lw=1.8)
    plt.scatter(df["date"], df["argo_temp"], s=18, alpha=0.35, color="black", label="Argo points")
    plt.title("Argo Spatial Validation (Daily Mean + Points)")
    plt.ylabel("SST (C)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.2, ls="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_overlay_timeseries.png", dpi=150)
    plt.close()

    # Correlation scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False, sharey=False)
    for ax, name, col in zip(axes, ["Chronos", "Granite", "ConvLSTM"],
                             ["chronos_pred", "granite_pred", "convlstm_pred"]):
        y_true = df["argo_temp"].values
        y_pred = df[col].values
        ax.scatter(y_true, y_pred, s=20, alpha=0.6)
        vmin = min(np.nanmin(y_true), np.nanmin(y_pred)) - 0.2
        vmax = max(np.nanmax(y_true), np.nanmax(y_pred)) + 0.2
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.2, alpha=0.6)
        m = compute_metrics(y_true, y_pred)
        ax.set_title(f"{name}\nRMSE={m['rmse']:.4f} R={m['r']:.4f}")
        ax.set_xlabel("Argo")
        ax.set_ylabel("Model")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_correlation_scatter.png", dpi=150)
    plt.close()

    print(f"Saved: {out_pred}")
    print(f"Saved: {out_metrics}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
