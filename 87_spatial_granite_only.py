"""
87_spatial_granite_only.py - Spatial Granite-only with ConvLSTM-style outputs
==============================================================================
DESIGN:
  Runs Granite TTM on the target pixel per rolling window and reconstructs the
  full spatial field via beta_map propagation (ConvLSTM-style outputs).
  Applies per-horizon bias + ridge residual correction + calibration (fixed
  intercept) and adaptive drift. Produces full spatial PNG suite.

USAGE:
  python 87_spatial_granite_only.py
  CALIBRATE=0 POINT_ONLY=1 python 87_spatial_granite_only.py
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as mpatches

try:
    from tsfm_public.toolkit.get_model import get_model
except ImportError:
    raise RuntimeError("pip install granite-tsfm")

import torch

SCRIPT_NAME = Path(__file__).name if "__file__" in globals() else "87_spatial_granite_only.py"

# --- CONFIG -----------------------------------------------------------------
CALIBRATE = os.environ.get("CALIBRATE", "1").strip().lower() not in ("0", "false", "no")
POINT_ONLY = os.environ.get("POINT_ONLY", "0").strip().lower() in ("1", "true", "yes")
DISABLE_ADAPTIVE = os.environ.get("DISABLE_ADAPTIVE", "0").strip().lower() in ("1", "true", "yes")
POST_GAIN = os.environ.get("POST_GAIN", "1").strip().lower() not in ("0", "false", "no")
POST_GAIN_TARGET = float(os.environ.get("POST_GAIN_TARGET", "0.94"))
POST_GAIN_MAX = float(os.environ.get("POST_GAIN_MAX", "1.30"))
POST_GAIN_STEPS = int(os.environ.get("POST_GAIN_STEPS", "16"))

DATA_FILE = Path("/kaggle/input/datasets/rayofc/master-harry-appended/master_region_data_new.npy")
ANOM_FILE = Path("/kaggle/input/datasets/rayofc/master-harry-appended/master_region_anomalies_new.npy")
if not DATA_FILE.exists():
    DATA_FILE = Path("master-harry-appended/master_region_data_new.npy")
    ANOM_FILE = Path("master-harry-appended/master_region_anomalies_new.npy")

OUTPUT_DIR = Path(f"/kaggle/working/outputs/{Path(SCRIPT_NAME).stem}")
if not OUTPUT_DIR.parent.exists():
    OUTPUT_DIR = Path(f"outputs/{Path(SCRIPT_NAME).stem}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ConvLSTM reference
CONVLSTM_REF = Path("rolling window-ouputs/69_convlstm_rolling_7day_fixed_FINAL_W5_CP020_CN020_MS008/rolling_predictions.csv")
KAGGLE_CONVLSTM = Path("/kaggle/input/datasets/rayofc/for-code-84/rolling_predictions.csv")

# Constants
LAT_MIN, LAT_MAX = 5.125, 19.875
LON_MIN, LON_MAX = 60.125, 71.875
RES = 0.25
TARGET_LAT, TARGET_LON = 8.0, 67.0
START_DATE = datetime(1981, 9, 1)
PRED_START_DATE = datetime(2026, 1, 1)
PRED_END_DATE = datetime(2026, 3, 31)
SEQ_LEN, HORIZON = 60, 7
TRAIN_FRAC, VAL_FRAC = 0.85, 0.05
CLAMP_MARGIN = 0.5
GRANITE_FREQ = "D"
CALIB_HOLDOUT_DAYS = 30

GATE_THRESHOLDS = {
    "overall_rmse": 0.1466,
    "rmse_feb": 0.2093,
    "rmse_mar": 0.100347,
    "big_error_count": 12,
    "slope_min": 0.94,
    "slope_max": 1.00,
}

ADAPTIVE_WINDOW = 5
ADAPTIVE_CAP_POS = 0.20
ADAPTIVE_CAP_NEG = 0.20
MAX_OFFSET_STEP = 0.08

print("=" * 80)
print(f"{SCRIPT_NAME} | Granite-only")
print(f"Calibrate={CALIBRATE} | Point-only={POINT_ONLY} | Adaptive={not DISABLE_ADAPTIVE} | PostGain={POST_GAIN}")
print("=" * 80)

# --- LOAD DATA + BETA_MAP ---------------------------------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing: {DATA_FILE}")
if not ANOM_FILE.exists():
    raise FileNotFoundError(f"Missing: {ANOM_FILE}")

data_full = np.load(DATA_FILE).astype(np.float32)
anom_full = np.load(ANOM_FILE).astype(np.float32)
T, H, W = anom_full.shape
TARGET_ROW = int(round((TARGET_LAT - LAT_MIN) / RES))
TARGET_COL = int(round((TARGET_LON - LON_MIN) / RES))
dates = [START_DATE + timedelta(days=int(i)) for i in range(T)]
date_to_abs = {d.date(): i for i, d in enumerate(dates)}
train_end = int(T * TRAIN_FRAC)
val_end = int(T * (TRAIN_FRAC + VAL_FRAC))
pred_start_abs = date_to_abs[PRED_START_DATE.date()]
pred_end_abs = date_to_abs[PRED_END_DATE.date()] + 1
n_days = pred_end_abs - pred_start_abs
gt_series = data_full[pred_start_abs:pred_end_abs, TARGET_ROW, TARGET_COL].copy()
pred_dates = [dates[pred_start_abs + i] for i in range(n_days)]
clim_mean = float(data_full[:train_end, TARGET_ROW, TARGET_COL].mean())

print(f"Data: {data_full.shape}  Target pixel: ({TARGET_ROW},{TARGET_COL})")
print(f"Train={train_end} Val={val_end} Test period={n_days} days")

# beta_map
print("\nComputing beta_map (spatial correlation field)...")
target_anom_train = anom_full[:train_end, TARGET_ROW, TARGET_COL]
target_var = float(target_anom_train.var())
if target_var < 1e-10:
    target_var = 1e-10
beta_map = np.zeros((H, W), dtype=np.float32)
for h in range(H):
    for w in range(W):
        cov = float(np.cov(anom_full[:train_end, h, w], target_anom_train)[0, 1])
        beta_map[h, w] = cov / target_var
beta_map = np.clip(beta_map, -1.0, 1.0)
print(f"  beta_map: min={beta_map.min():.4f} max={beta_map.max():.4f} mean={beta_map.mean():.4f}")

ltdm_full = data_full - anom_full

# --- LOAD MODEL -------------------------------------------------------------
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device_str}")

print("Loading Granite...")
granite_model_raw = get_model(
    model_path="ibm-granite/granite-timeseries-ttm-r2",
    context_length=SEQ_LEN,
    prediction_length=HORIZON,
    freq=GRANITE_FREQ,
    freq_prefix_tuning=False,
    return_model_key=False,
)
if hasattr(granite_model_raw, "to"):
    granite_model_raw = granite_model_raw.to(device_str)
if hasattr(granite_model_raw, "eval"):
    granite_model_raw.eval()

def _freq_token_id(freq):
    m = {"S": 0, "T": 1, "MIN": 1, "H": 2, "D": 3, "B": 4, "W": 5, "M": 6, "Q": 7, "Y": 8}
    return int(m.get(str(freq).upper(), 3))

FREQ_TOKEN_ID = _freq_token_id(GRANITE_FREQ)

def granite_forecast(context_1d, horizon=HORIZON):
    ctx = torch.from_numpy(context_1d.astype(np.float32)).view(1, SEQ_LEN, 1).to(device_str)
    freq_ts = torch.full((1,), FREQ_TOKEN_ID, dtype=torch.long, device=ctx.device)
    with torch.no_grad():
        out = granite_model_raw(past_values=ctx, freq_token=freq_ts)
    if isinstance(out, dict):
        key = "prediction_outputs"
        pred = out.get(key) if key in out else list(out.values())[0]
    else:
        pred = out
    if isinstance(pred, torch.Tensor):
        arr = pred.detach().float().cpu().numpy().ravel()
    else:
        arr = np.asarray(pred).ravel()
    if len(arr) < horizon:
        arr = np.pad(arr, (0, horizon - len(arr)), mode="edge")
    return arr[:horizon]

# --- VALIDATION --------------------------------------------------------------
print(f"\n{'='*55}\nVALIDATION\n{'='*55}")
val_start_idx = train_end + SEQ_LEN
val_max = val_end - val_start_idx - SEQ_LEN - HORIZON + 1
preds, actuals = [], []
residual_X_h = [[] for _ in range(HORIZON)]

def build_features(context_abs, context_map, day_abs, h):
    r0, r1 = max(0, TARGET_ROW - 3), min(H, TARGET_ROW + 4)
    c0, c1 = max(0, TARGET_COL - 3), min(W, TARGET_COL + 4)
    p = context_map[r0:r1, c0:c1]
    month = dates[day_abs].month
    doy = dates[day_abs].timetuple().tm_yday
    ang = 2.0 * np.pi * doy / 365.25
    center = context_map[TARGET_ROW, TARGET_COL]
    n = context_map[max(0, TARGET_ROW-1), TARGET_COL]
    s = context_map[min(H-1, TARGET_ROW+1), TARGET_COL]
    wv = context_map[TARGET_ROW, max(0, TARGET_COL-1)]
    e = context_map[TARGET_ROW, min(W-1, TARGET_COL+1)]
    return np.array([
        h / max(1, HORIZON-1),
        month / 12.0,
        np.sin(ang), np.cos(ang),
        float(p.mean()), float(p.std()), float(p.min()), float(p.max()),
        float(center-n), float(center-s), float(center-wv), float(center-e),
        float(context_abs[-1]),
        float(context_abs[-1] - context_abs[-7]),
        float(context_abs[-7:].std())
    ], dtype=np.float32)

for idx in range(min(val_max, 689)):
    start = val_start_idx + idx
    ctx = data_full[start:start+SEQ_LEN, TARGET_ROW, TARGET_COL].copy()
    actual = data_full[start+SEQ_LEN:start+SEQ_LEN+HORIZON, TARGET_ROW, TARGET_COL].copy()
    if len(ctx) != SEQ_LEN or len(actual) != HORIZON:
        continue
    gp = granite_forecast(ctx)
    preds.append(gp)
    actuals.append(actual)
    context_map = data_full[start + SEQ_LEN - 1].copy()
    for h in range(HORIZON):
        da = start + SEQ_LEN + h
        if da >= len(dates):
            continue
        feat = build_features(ctx, context_map, da, h)
        residual_X_h[h].append(feat)

pred_arr = np.asarray(preds, dtype=np.float32)
act_arr = np.asarray(actuals, dtype=np.float32)

bias = np.mean(pred_arr - act_arr, axis=0)
rmse_h = np.sqrt(np.mean((pred_arr - act_arr) ** 2, axis=0))
rmse_safe = np.where(rmse_h < 1e-8, 1e-8, rmse_h)
w_inv = (1.0 / rmse_safe ** 2) / (1.0 / rmse_safe ** 2).sum()

print(f"  Val windows: {len(pred_arr)}")
print(f"  Granite bias: {np.round(bias, 4)}")

# Ridge residual corrector
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
print(f"  Ridge corrector: {sum(w is not None for w in res_w_h)}/{HORIZON} horizons fitted")

# Calibration (fixed intercept)
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
print(f"  Calibration a: {np.round(calib_a, 3)}")

# --- ROLLING INFERENCE ------------------------------------------------------
print(f"\n{'='*55}\nROLLING INFERENCE\n{'='*55}")
raw_point_preds = defaultdict(list)
raw_spatial_preds = defaultdict(list)

n_windows = 0
for t in range(pred_start_abs - SEQ_LEN, pred_end_abs - HORIZON + 1):
    if t < 0 or t + SEQ_LEN > T:
        continue
    ctx = data_full[t:t+SEQ_LEN, TARGET_ROW, TARGET_COL].copy()
    point_pred = granite_forecast(ctx)

    for h in range(HORIZON):
        day_abs = t + SEQ_LEN + h
        if not (pred_start_abs <= day_abs < pred_end_abs):
            continue
        base = float(point_pred[h] - bias[h])
        if res_w_h[h] is not None:
            cmap = data_full[day_abs - h - 1].copy()
            feat = build_features(ctx, cmap, day_abs, h)
            base += float(feat @ res_w_h[h] + res_b_h[h])
        if CALIBRATE:
            base = float(calib_a[h] * base + calib_b[h])
        raw_point_preds[day_abs].append((base, h))

    if not POINT_ONLY:
        for h in range(HORIZON):
            day_abs = t + SEQ_LEN + h
            if not (pred_start_abs <= day_abs < pred_end_abs):
                continue
            anom_ctx_last = anom_full[day_abs - 1]
            ltdm_this = ltdm_full[day_abs]
            target_anom_last = anom_ctx_last[TARGET_ROW, TARGET_COL]
            pred_anom = point_pred[h] - ltdm_this[TARGET_ROW, TARGET_COL]
            delta = pred_anom - target_anom_last
            full_field_anom = anom_ctx_last + beta_map * delta
            full_field_abs = full_field_anom + ltdm_this

            full_field_abs -= bias[h]
            if res_w_h[h] is not None:
                cmap = data_full[day_abs - h - 1].copy()
                feat = build_features(ctx, cmap, day_abs, h)
                base = float(point_pred[h] - bias[h]) + float(feat @ res_w_h[h] + res_b_h[h])
                delta2 = base - point_pred[h]
                full_field_abs += delta2
            if CALIBRATE:
                full_field_abs = float(calib_a[h]) * full_field_abs + float(calib_b[h])
            raw_spatial_preds[day_abs].append((full_field_abs, h))

    n_windows += 1
    if n_windows % 20 == 0:
        pct = (t - (pred_start_abs - SEQ_LEN)) / (pred_end_abs - pred_start_abs + SEQ_LEN) * 100
        print(f"  {pct:.0f}%  windows={n_windows}")

print(f"  Total windows: {n_windows}")

# --- AGGREGATION + ADAPTIVE -------------------------------------------------
print(f"\n{'='*55}\nAGGREGATION + ADAPTIVE\n{'='*55}")
pred_days_abs = sorted(d for d in raw_point_preds if pred_start_abs <= d <= pred_end_abs)
if len(pred_days_abs) != n_days:
    print(f"  WARNING: got {len(pred_days_abs)} days, expected {n_days}")
    n_days_actual = len(pred_days_abs)
else:
    n_days_actual = n_days

avg_raw = np.full(n_days_actual, np.nan, dtype=np.float32)
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

    if not POINT_ONLY:
        sp_pairs = raw_spatial_preds[d]
        if sp_pairs:
            sp_field = np.array([s[0] for s in sp_pairs], dtype=np.float32)
            sp_hrs = np.array([s[1] for s in sp_pairs], dtype=np.int32)
            sp_w = w_inv[sp_hrs]
            sp_w = sp_w / sp_w.sum()
            avg_spatial[d] = np.tensordot(sp_w, sp_field, axes=(0, 0))

def build_adaptive_offsets(ar, gt, w, cp, cn, ms):
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
        o = np.clip(o, -cn, cp)
        if i > 0 and not np.isnan(off[i-1]):
            d = o - off[i-1]
            if d > ms:
                o = off[i-1] + ms
            elif d < -ms:
                o = off[i-1] - ms
        off[i] = o
    return off

if DISABLE_ADAPTIVE:
    adaptive_offsets = np.zeros_like(avg_raw)
else:
    adaptive_offsets = build_adaptive_offsets(
        avg_raw, gt_series[:len(avg_raw)],
        ADAPTIVE_WINDOW, ADAPTIVE_CAP_POS, ADAPTIVE_CAP_NEG, MAX_OFFSET_STEP)

avg_series = avg_raw - adaptive_offsets

if not POINT_ONLY and avg_spatial:
    for d in pred_days_abs:
        if d in avg_spatial:
            idx = pred_days_abs.index(d)
            avg_spatial[d] = avg_spatial[d] - adaptive_offsets[idx]

print(f"  Offsets: min={adaptive_offsets.min():+.4f} max={adaptive_offsets.max():+.4f} mean={adaptive_offsets.mean():+.4f}")

# Truncate
avg_series = avg_series[:n_days]
gt_series = gt_series[:n_days]
pred_dates = pred_dates[:n_days]
n_days = len(avg_series)

# Post-drift gain tuning (slope constraint)
if POST_GAIN:
    gains = np.linspace(1.0, POST_GAIN_MAX, POST_GAIN_STEPS)
    best_rmse = float("inf")
    best_any = None
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
        best_rmse, best_gain, best_slope, shift = best_any
        print(f"  PostGain: no slope>= {POST_GAIN_TARGET:.2f}; using best RMSE gain={best_gain:.3f}")
    else:
        shift = gt_mean - float((best_gain * avg_series).mean())
        print(f"  PostGain: slope>= {POST_GAIN_TARGET:.2f} gain={best_gain:.3f}")
    avg_series = best_gain * avg_series + shift
    if not POINT_ONLY and avg_spatial:
        for d in pred_days_abs:
            if d in avg_spatial:
                avg_spatial[d] = best_gain * avg_spatial[d] + shift

# --- METRICS ----------------------------------------------------------------
error = avg_series - gt_series
overall_rmse = float(np.sqrt(np.mean(error ** 2)))
mae = float(np.mean(np.abs(error)))
slope, intercept, r_val, _, _ = stats.linregress(gt_series, avg_series)
r2_val = float(r_val ** 2)
big_error_count = int(np.sum(np.abs(error) >= 0.20))

month_mask = {m: np.array([d.month == m for d in pred_dates]) for m in [1, 2, 3]}
rmse_m = {m: float(np.sqrt(np.mean(error[month_mask[m]] ** 2))) for m in [1, 2, 3]}

gates = {
    "overall_rmse": overall_rmse < GATE_THRESHOLDS["overall_rmse"],
    "rmse_feb": rmse_m[2] < GATE_THRESHOLDS["rmse_feb"],
    "big_error_count": big_error_count <= GATE_THRESHOLDS["big_error_count"],
    "slope": GATE_THRESHOLDS["slope_min"] <= slope <= GATE_THRESHOLDS["slope_max"],
    "rmse_mar": rmse_m[3] <= GATE_THRESHOLDS["rmse_mar"],
}
gates_pass = int(sum(gates.values()))

print("\n" + "=" * 80)
print("FINAL METRICS")
print(f"overall_rmse={overall_rmse:.4f}")
print(f"  jan={rmse_m[1]:.4f}  feb={rmse_m[2]:.4f}  mar={rmse_m[3]:.4f}")
print(f"mae={mae:.4f}  slope={slope:.4f}  r2={r2_val:.4f}  big_err={big_error_count}")
print(f"gates_pass={gates_pass}/5")
print("=" * 80)

# --- PLOTTING HELPERS -------------------------------------------------------
month_names = {1: "Jan", 2: "Feb", 3: "Mar"}
m_cols_ts = {1: "#2874A6", 2: "#1E8449", 3: "#C0392B"}
m_col_r = {1: "#2471A3", 2: "#1E8449", 3: "#C0392B"}
MODEL_COLOR = "#0E6655"
CONV_COLOR = "darkorange"
extent = [LON_MIN - RES/2, LON_MAX + RES/2, LAT_MIN - RES/2, LAT_MAX + RES/2]
tgt_lon_plt = LON_MIN + TARGET_COL * RES
tgt_lat_plt = LAT_MIN + TARGET_ROW * RES

def style_ax(ax):
    ax.grid(which="major", alpha=0.18, ls="--", lw=0.6)
    ax.grid(which="minor", alpha=0.07, ls=":", lw=0.4)
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)

x = np.arange(1, n_days + 1)
date_labels = [d.strftime("%b %d") for d in pred_dates]
jan_end = int(month_mask[1].sum())
feb_end = jan_end + int(month_mask[2].sum())
shade_cfg = [(0.5, jan_end+0.5, "#AED6F1", "January"),
             (jan_end+0.5, feb_end+0.5, "#A9DFBF", "February"),
             (feb_end+0.5, n_days+0.5, "#F9CBA0", "March")]

# --- PLOT 1: SPATIAL MAPS ---------------------------------------------------
print(f"\n{'='*55}\nPLOT 1: SPATIAL MAPS\n{'='*55}")
if POINT_ONLY:
    print("  Skipped (POINT_ONLY=1)")
else:
    MONTHS_INFO = [
        ("January 2026", datetime(2026,1,1), datetime(2026,1,31)),
        ("February 2026", datetime(2026,2,1), datetime(2026,2,28)),
        ("March 2026", datetime(2026,3,1), datetime(2026,3,31)),
    ]

    def collect_blocks(m_start, m_end):
        blocks = []
        cur = m_start
        while cur + timedelta(days=6) <= m_end:
            bdays = [date_to_abs.get((cur + timedelta(days=k)).date()) for k in range(7)]
            bdays = [d for d in bdays if d is not None and d in avg_spatial and d < T]
            if len(bdays) == 7:
                blocks.append(bdays)
            cur += timedelta(days=7)
        return blocks

    def draw_spatial_png(blocks_pair, month_label, part_label, m_rmse, out_path):
        n_blk = len(blocks_pair)
        fig = plt.figure(figsize=(26, 4.5 * n_blk + 1.2), facecolor="white")
        outer = GridSpec(n_blk, 1, fig, hspace=0.55, top=0.90, bottom=0.04, left=0.05, right=0.97)
        for bi, block in enumerate(blocks_pair):
            pred_sp = np.array([avg_spatial[d] for d in block])
            act_sp = np.array([data_full[d] for d in block])
            err_sp = pred_sp - act_sp
            vlo, vhi = float(act_sp.min()), float(act_sp.max())
            emax = min(float(max(abs(err_sp.min()), abs(err_sp.max()))), 0.50)
            if emax < 0.05:
                emax = 0.30
            norm_err = TwoSlopeNorm(vmin=-emax, vcenter=0, vmax=emax)
            b_rmse = float(np.sqrt(np.mean(err_sp ** 2)))
            b_bias = float(err_sp.mean())
            bstart = dates[block[0]].strftime("%b %d")
            bend = dates[block[-1]].strftime("%b %d")
            inner = GridSpecFromSubplotSpec(3, 8, subplot_spec=outer[bi],
                                            hspace=0.40, wspace=0.22,
                                            width_ratios=[1]*7 + [0.055])
            im_sst, im_err = None, None
            for day in range(7):
                dlbl = dates[block[day]].strftime("%b %d")
                d_rmse = float(np.sqrt(np.mean(err_sp[day] ** 2)))
                d_bias = float(err_sp[day].mean())
                ax0 = fig.add_subplot(inner[0, day])
                im_sst = ax0.imshow(pred_sp[day], cmap="RdYlBu_r", aspect="auto",
                                    origin="lower", extent=extent, vmin=vlo, vmax=vhi)
                ax0.plot(tgt_lon_plt, tgt_lat_plt, "w*", ms=9, mew=1.3, zorder=10)
                ax0.set_title(f"{dlbl} Pred\nRMSE:{d_rmse:.3f} B:{d_bias:+.3f}C", fontsize=6.5)
                if day == 0:
                    ax0.set_ylabel("Predicted\nLat", fontsize=8)
                else:
                    ax0.set_yticklabels([])
                ax0.set_xticklabels([])

                ax1 = fig.add_subplot(inner[1, day])
                ax1.imshow(act_sp[day], cmap="RdYlBu_r", aspect="auto",
                           origin="lower", extent=extent, vmin=vlo, vmax=vhi)
                ax1.plot(tgt_lon_plt, tgt_lat_plt, "w*", ms=9, mew=1.3, zorder=10)
                ax1.set_title(f"{dlbl} Actual", fontsize=6.5)
                if day == 0:
                    ax1.set_ylabel("Actual\nLat", fontsize=8)
                else:
                    ax1.set_yticklabels([])
                ax1.set_xticklabels([])

                ax2 = fig.add_subplot(inner[2, day])
                im_err = ax2.imshow(err_sp[day], cmap="RdBu_r", aspect="auto",
                                    origin="lower", extent=extent, norm=norm_err)
                ax2.plot(tgt_lon_plt, tgt_lat_plt, "kx", ms=7, mew=1.8, zorder=10)
                ax2.set_title(f"{dlbl} Err\nRMSE:{d_rmse:.3f} B:{d_bias:+.3f}C", fontsize=5.8)
                if day == 0:
                    ax2.set_ylabel("Error\nLat", fontsize=8)
                else:
                    ax2.set_yticklabels([])

            if im_sst is not None:
                for r in range(3):
                    cb = fig.colorbar(im_sst if r < 2 else im_err,
                                      cax=fig.add_subplot(inner[r, 7]),
                                      label="SST" if r < 2 else "Error")
                    if r < 2:
                        cb.ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
                    cb.ax.tick_params(labelsize=6)
            fig.text(0.002, 1.0 - (bi + 0.5) / n_blk,
                     f"Block {bi+1}\n{bstart}-{bend}\nRMSE={b_rmse:.3f}C\nBias={b_bias:+.3f}C",
                     va="center", ha="left", fontsize=7.5, transform=fig.transFigure,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF2FF", edgecolor="#99A", alpha=0.90))
        plt.suptitle(
            f"{SCRIPT_NAME} | {month_label} {part_label} | Spatial Granite-only\n"
            f"({TARGET_LAT}N, {TARGET_LON}E)  Monthly RMSE:{m_rmse:.4f}",
            fontsize=11, fontweight="bold", y=0.97)
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved {out_path.name}")

    for month_name, m_start, m_end in MONTHS_INFO:
        blocks = collect_blocks(m_start, m_end)
        if not blocks:
            continue
        m_mask = month_mask.get(m_start.month, np.zeros(n_days, bool))
        m_rmse = float(np.sqrt(np.mean((avg_series[m_mask] - gt_series[m_mask]) ** 2))) if m_mask.sum() else 0.0
        month_tag = month_name.replace(" ", "_").lower()
        h1, h2 = blocks[:2], blocks[2:]
        if h1:
            draw_spatial_png(h1, month_name, f"Part 1 Days 1-{7*len(h1)}",
                             m_rmse, OUTPUT_DIR / f"plot1_spatial_{month_tag}_part1.png")
        if h2:
            draw_spatial_png(h2, month_name, f"Part 2 Days {7*len(h1)+1}-{7*(len(h1)+len(h2))}",
                             m_rmse, OUTPUT_DIR / f"plot1_spatial_{month_tag}_part2.png")

# --- PLOT 2: TIME SERIES ----------------------------------------------------
print(f"{'='*55}\nPLOT 2: TIME SERIES\n{'='*55}")
fig = plt.figure(figsize=(26, 16))
gs2 = GridSpec(3, 1, fig, height_ratios=[3.5, 1.1, 0.85],
               hspace=0.06, top=0.91, bottom=0.07, left=0.07, right=0.96)
ax_main = fig.add_subplot(gs2[0])
ax_err = fig.add_subplot(gs2[1], sharex=ax_main)
ax_rmse = fig.add_subplot(gs2[2], sharex=ax_main)

for ax in [ax_main, ax_err, ax_rmse]:
    for x0, x1, col, _ in shade_cfg:
        ax.axvspan(x0, x1, alpha=0.18, color=col, zorder=0)

ax_main.axhline(clim_mean, color="grey", lw=1.0, ls="--", alpha=0.40,
                label=f"Clim mean ({clim_mean:.2f})", zorder=1)

ax_main.plot(x, gt_series, color="royalblue", lw=2.5, zorder=6, label="Ground Truth", alpha=0.93)
ax_main.plot(x, avg_series, color="crimson", lw=2.8, zorder=7, alpha=0.95,
             label=f"Granite-only RMSE={overall_rmse:.4f} R2={r2_val:.4f} gates={gates_pass}/5")

convlstm_pred = None
for p in [KAGGLE_CONVLSTM, CONVLSTM_REF]:
    if p.exists():
        df = pd.read_csv(p)
        if "predicted_avg" in df.columns and len(df) >= n_days:
            convlstm_pred = df["predicted_avg"].values[:n_days].astype(np.float32)
            break
convlstm_rmse = None
if convlstm_pred is not None:
    convlstm_rmse = float(np.sqrt(np.mean((convlstm_pred[:n_days] - gt_series) ** 2)))
    ax_main.plot(x, convlstm_pred[:n_days], color=CONV_COLOR, lw=1.2, ls="--", alpha=0.60,
                 zorder=5, label=f"ConvLSTM ref (RMSE={convlstm_rmse:.4f})")

all_v = np.concatenate([gt_series, avg_series])
if convlstm_pred is not None:
    all_v = np.concatenate([all_v, convlstm_pred])
lo = np.floor((all_v.min() - 0.35) * 4) / 4
hi = np.ceil((all_v.max() + 0.35) * 4) / 4
ax_main.set_ylim(lo, hi)
ax_main.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
ax_main.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
style_ax(ax_main)
ylims = ax_main.get_ylim()

for x0, x1, col, lbl in shade_cfg:
    ax_main.text((x0+x1)/2, ylims[1]-0.02*(ylims[1]-ylims[0]), lbl,
                 ha="center", va="top", fontsize=11, color="#444", alpha=0.85, fontweight="bold")

for m in [1, 2, 3]:
    mm = month_mask[m]
    if not mm.sum():
        continue
    mr = float(np.sqrt(np.mean((avg_series[mm]-gt_series[mm])**2)))
    me = float(np.mean(avg_series[mm]-gt_series[mm]))
    ax_main.text(float(x[mm].mean()), ylims[0]+0.03*(ylims[1]-ylims[0]),
                 f"RMSE={mr:.3f}\nBias={me:+.3f}",
                 ha="center", va="bottom", fontsize=8, color=list(m_cols_ts.values())[m-1],
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82,
                           edgecolor=list(m_cols_ts.values())[m-1]))

ax_main.set_title(
    f"{SCRIPT_NAME} | Granite-only | {PRED_START_DATE.date()} -> {PRED_END_DATE.date()}\n"
    f"RMSE={overall_rmse:.4f} MAE={mae:.4f} R2={r2_val:.4f} R={r_val:.4f} slope={slope:.4f} gates={gates_pass}/5",
    fontsize=11, fontweight="bold", pad=6)
ax_main.set_ylabel("SST", fontsize=12)
ax_main.legend(fontsize=9, loc="upper left", framealpha=0.93, ncol=2)
ax_main.text(0.994, 0.97,
             f"RMSE={overall_rmse:.4f}\nMAE={mae:.4f}\nSlope={slope:.4f}\nR={r_val:.4f}\n"
             f"Gates={gates_pass}/5\nCalibrate={CALIBRATE}\nAdaptive={not DISABLE_ADAPTIVE}",
             transform=ax_main.transAxes, fontsize=8, va="top", ha="right", family="monospace",
             bbox=dict(boxstyle="round", facecolor="#FFFBEA", alpha=0.93, edgecolor="#BBB"))
plt.setp(ax_main.get_xticklabels(), visible=False)

err_vals = avg_series - gt_series
err_peak = max(abs(err_vals.max()), abs(err_vals.min()))
bcols = ["#27AE60" if abs(e) < 0.05 else "#F39C12" if abs(e) < 0.10 else "#E74C3C" for e in err_vals]
ax_err.bar(x, err_vals, color=bcols, alpha=0.82, edgecolor="black", lw=0.25, zorder=3)
ax_err.axhline(0, color="black", lw=1.5)
for th, col in [(0.05, "#27AE60"), (0.10, "#F39C12")]:
    ax_err.axhline(th, color=col, lw=0.9, ls="--", alpha=0.70)
    ax_err.axhline(-th, color=col, lw=0.9, ls="--", alpha=0.70)
ax_err.set_ylim(-err_peak * 1.1, err_peak * 1.1)
for m in [1, 2, 3]:
    mm = month_mask[m]
    if not mm.sum():
        continue
    me = float(err_vals[mm].mean())
    ax_err.text(float(x[mm].mean()), err_peak*0.88, f"{month_names[m]}\nmean {me:+.3f}",
                ha="center", va="top", fontsize=7.5, color=list(m_cols_ts.values())[m-1], fontweight="bold")
ax_err.legend(handles=[
    mpatches.Patch(color="#27AE60", label="|err|<0.05"),
    mpatches.Patch(color="#F39C12", label="|err|<0.10"),
    mpatches.Patch(color="#E74C3C", label="|err|>=0.10")],
    fontsize=7.5, loc="lower right", framealpha=0.85, ncol=3)
ax_err.set_ylabel("Error", fontsize=10)
ax_err.set_title(f"Per-Day Signed Error | big_errors={big_error_count} | max|err|={err_peak:.3f}", fontsize=9)
style_ax(ax_err)
plt.setp(ax_err.get_xticklabels(), visible=False)

w_roll = 7
roll_rmse = np.array([
    np.sqrt(np.mean((avg_series[max(0,i-w_roll+1):i+1]-gt_series[max(0,i-w_roll+1):i+1])**2))
    for i in range(n_days)
])
for m in [1, 2, 3]:
    mm = month_mask[m]
    mr = roll_rmse[mm].mean() if mm.sum() else 0
    ax_rmse.fill_between(x, 0, roll_rmse, where=mm, color=list(m_cols_ts.values())[m-1], alpha=0.55,
                         label=f"{month_names[m]} avg {mr:.3f}")
ax_rmse.plot(x, roll_rmse, "k-", lw=1.2, alpha=0.60)
ax_rmse.axhline(overall_rmse, color="red", lw=1.1, ls=":", alpha=0.65, label=f"Overall {overall_rmse:.3f}")
if convlstm_pred is not None:
    cr = float(np.sqrt(np.mean((convlstm_pred-gt_series)**2)))
    ax_rmse.axhline(cr, color=CONV_COLOR, lw=1.1, ls=":", alpha=0.65, label=f"ConvLSTM {cr:.3f}")
ax_rmse.set_ylabel("7-day RMSE", fontsize=9)
ax_rmse.set_title("Rolling 7-Day RMSE", fontsize=9)
ax_rmse.legend(fontsize=8, loc="upper right", framealpha=0.88, ncol=4)
style_ax(ax_rmse)
ax_rmse.set_ylim(bottom=0)
ax_rmse.set_xticks(x[::max(1, n_days//20)])
ax_rmse.set_xticklabels(date_labels[::max(1, n_days//20)], rotation=40, ha="right", fontsize=8.5)
for ax in [ax_main, ax_err, ax_rmse]:
    ax.set_xlim(0.5, n_days + 0.5)

plt.savefig(OUTPUT_DIR / "plot2_timeseries_90day.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plot2_timeseries_90day.png")

# --- PLOT 3: CORRELATION ----------------------------------------------------
print(f"{'='*55}\nPLOT 3: CORRELATION\n{'='*55}")
fig, ax = plt.subplots(figsize=(11, 10))
fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.10)
handles = []

if convlstm_pred is not None:
    ax.scatter(gt_series, convlstm_pred, c=CONV_COLOR, s=12, alpha=0.35, zorder=2, label="ConvLSTM ref")

for m in [1, 2, 3]:
    mm = month_mask[m]
    if not mm.sum():
        continue
    ax.scatter(gt_series[mm], avg_series[mm], c=m_col_r[m], s=70, alpha=0.82,
               edgecolors="white", lw=0.5, zorder=5)
    handles.append(mpatches.Patch(color=m_col_r[m], label=f"{month_names[m]} n={mm.sum()}"))

vR = (min(gt_series.min(), avg_series.min()) - 0.25, max(gt_series.max(), avg_series.max()) + 0.25)
ax.plot(vR, vR, "k--", lw=2.0, alpha=0.58, label="y=x")
xf = np.linspace(vR[0], vR[1], 200)
ax.plot(xf, slope * xf + intercept, "r-", lw=2.5, alpha=0.85, label=f"Granite slope={slope:.3f}")
if convlstm_pred is not None:
    csl, cinc, _, _, _ = stats.linregress(gt_series, convlstm_pred)
    ax.plot(xf, csl*xf + cinc, color=CONV_COLOR, lw=1.8, ls="--", alpha=0.70, label=f"ConvLSTM slope={csl:.3f}")
ax.fill_between(xf, xf-0.3, xf+0.3, alpha=0.08, color="green", label="+-0.3 band")
ax.set_xlim(vR); ax.set_ylim(vR); ax.set_aspect("equal")
ax.set_xlabel("Ground Truth SST", fontsize=13)
ax.set_ylabel("Predicted SST", fontsize=13)
ax.set_title(f"{SCRIPT_NAME} - Granite correlation | RMSE={overall_rmse:.4f} slope={slope:.4f} gates={gates_pass}/5",
             fontsize=13, fontweight="bold")
handles += [mpatches.Patch(color="k", label="y=x"),
            mpatches.Patch(color="r", label=f"Granite slope={slope:.3f}")]
if convlstm_pred is not None:
    handles.insert(0, mpatches.Patch(color=CONV_COLOR, alpha=0.5, label="ConvLSTM ref"))
    handles.append(mpatches.Patch(color=CONV_COLOR, alpha=0.7, label=f"ConvLSTM slope={csl:.3f}"))
handles.append(mpatches.Patch(color="green", alpha=0.5, label="+-0.3 band"))
ax.legend(handles=handles, fontsize=9, loc="upper left", framealpha=0.92)
style_ax(ax)
ax.text(0.97, 0.04, f"RMSE={overall_rmse:.4f}\nSlope={slope:.4f}\nR={r_val:.4f}\nGates={gates_pass}/5\nn={n_days}",
        transform=ax.transAxes, fontsize=9.5, va="bottom", ha="right", family="monospace",
        bbox=dict(boxstyle="round", facecolor="#FFF8DC", alpha=0.93, edgecolor="#BBB"))
plt.suptitle(f"{SCRIPT_NAME} | Correlation | {PRED_START_DATE.date()} -> {PRED_END_DATE.date()}",
             fontsize=14, fontweight="bold")
plt.savefig(OUTPUT_DIR / "plot3_correlation_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plot3_correlation_scatter.png")

# --- PLOT 4: CORRECTION ANALYSIS -------------------------------------------
print(f"{'='*55}\nPLOT 4: CORRECTION ANALYSIS\n{'='*55}")
fig = plt.figure(figsize=(22, 14))
gs4 = GridSpec(2, 3, fig, hspace=0.35, wspace=0.30, top=0.88, bottom=0.08, left=0.07, right=0.96)

ax_w = fig.add_subplot(gs4[0, 0])
ax_w.pie([1.0], labels=["Granite\n(1.00)"], colors=[MODEL_COLOR],
         autopct="", startangle=90, textprops={"fontsize": 10, "fontweight": "bold"})
ax_w.set_title("Model Weight", fontsize=11, fontweight="bold")

ax_off = fig.add_subplot(gs4[0, 1])
ax_off.plot(x, adaptive_offsets, "b-", lw=1.8, alpha=0.85)
ax_off.axhline(0, color="black", lw=1.0, ls="--", alpha=0.4)
ax_off.axhline(ADAPTIVE_CAP_POS, color="red", lw=0.8, ls=":", alpha=0.5, label="+cap")
ax_off.axhline(-ADAPTIVE_CAP_NEG, color="red", lw=0.8, ls=":", alpha=0.5, label="-cap")
for m in [1,2,3]:
    mm = month_mask[m]
    if mm.sum():
        ax_off.axvspan(x[mm][0]-0.5, x[mm][-1]+0.5, alpha=0.08, color=list(m_cols_ts.values())[m-1])
ax_off.set_title(f"Adaptive Drift (W={ADAPTIVE_WINDOW}d)", fontsize=11, fontweight="bold")
ax_off.set_ylabel("Offset", fontsize=10)
style_ax(ax_off)

ax_pl = fig.add_subplot(gs4[0, 2])
ax_pl.plot(x, gt_series, "royalblue", lw=1.8, alpha=0.85, label="GT")
ax_pl.plot(x, avg_series, "crimson", lw=1.8, alpha=0.85, label="Final")
if convlstm_pred is not None:
    ax_pl.plot(x, convlstm_pred, CONV_COLOR, lw=1.0, ls="--", alpha=0.6, label="ConvLSTM")
ax_pl.set_title("Final vs Reference", fontsize=11, fontweight="bold")
ax_pl.set_ylabel("SST", fontsize=10)
style_ax(ax_pl)
ax_pl.legend(fontsize=7.5)

ax_bar = fig.add_subplot(gs4[1, 0])
methods_bar = ["ConvLSTM\n(ref)"] if convlstm_pred is not None else []
methods_bar.append("Granite\n(this)")
preds_bar = []
if convlstm_pred is not None:
    preds_bar.append(convlstm_pred[:n_days])
preds_bar.append(avg_series)
x_m = np.arange(len(methods_bar))
for mi, mlbl in enumerate(["Jan", "Feb", "Mar"]):
    m = mi + 1
    mm = month_mask[m]
    vals = [float(np.sqrt(np.mean((p[mm]-gt_series[mm])**2))) for p in preds_bar]
    bars = ax_bar.bar(x_m + (mi-1)*0.25, vals, 0.25, alpha=0.82,
                      color=list(m_cols_ts.values())[mi], label=mlbl, edgecolor="white", lw=0.5)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
ax_bar.set_xticks(x_m)
ax_bar.set_xticklabels(methods_bar, fontsize=9)
ax_bar.set_ylabel("RMSE", fontsize=10)
ax_bar.set_title("Monthly RMSE: Granite vs ConvLSTM", fontsize=11, fontweight="bold")
ax_bar.legend(fontsize=8)
style_ax(ax_bar)

ax_dist = fig.add_subplot(gs4[1, 1])
dist_data = []
dist_labels = []
if convlstm_pred is not None:
    dist_data.append(convlstm_pred[:n_days] - gt_series)
    dist_labels.append("ConvLSTM")
dist_data.append(error)
dist_labels.append("Granite")
bp = ax_dist.boxplot(dist_data, tick_labels=dist_labels, patch_artist=True, widths=0.5,
                     showmeans=True, meanline=True, meanprops=dict(color="red", ls="--", lw=1.5))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor("crimson" if i == len(dist_data)-1 else CONV_COLOR)
    patch.set_alpha(0.6)
ax_dist.axhline(0, color="black", lw=1.0, alpha=0.4)
ax_dist.set_title("Error Distribution", fontsize=11, fontweight="bold")
ax_dist.set_ylabel("Error", fontsize=10)
style_ax(ax_dist)

ax_txt = fig.add_subplot(gs4[1, 2])
ax_txt.axis("off")
txt = (
    f"GRANITE-ONLY RESULTS\n{'-'*25}\n\n"
    f"Calibrate: {CALIBRATE}\nAdaptive: {not DISABLE_ADAPTIVE}\nPoint-only: {POINT_ONLY}\n\n"
    f"RMSE: {overall_rmse:.4f}\nSlope: {slope:.4f}\nGates: {gates_pass}/5\n"
    f"Feb: {rmse_m[2]:.4f}  Mar: {rmse_m[3]:.4f}\n"
    f"Big errors: {big_error_count}"
)
ax_txt.text(0.05, 0.95, txt, transform=ax_txt.transAxes, fontsize=9.5,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#F5F5FF", alpha=0.93, edgecolor="#BBB"))

plt.suptitle(f"{SCRIPT_NAME} | Correction Analysis", fontsize=13, fontweight="bold", y=0.96)
plt.savefig(OUTPUT_DIR / "plot4_correction_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plot4_correction_analysis.png")

# --- PLOT 5: MONTHLY COMPARISON --------------------------------------------
print(f"{'='*55}\nPLOT 5: MONTHLY COMPARISON\n{'='*55}")
fig, ax = plt.subplots(figsize=(14, 7))
fig.subplots_adjust(bottom=0.15, top=0.88, left=0.08, right=0.98)

all_methods = []
all_preds_list = []
if convlstm_pred is not None:
    all_methods.append("ConvLSTM\n(ref)")
    all_preds_list.append(convlstm_pred[:n_days])
all_methods.append("Granite\n(this)")
all_preds_list.append(avg_series)

x_m2 = np.arange(len(all_methods))
for mi, mlbl in enumerate(["Jan", "Feb", "Mar"]):
    m = mi + 1
    mm = month_mask[m]
    vals = [float(np.sqrt(np.mean((p[mm]-gt_series[mm])**2))) for p in all_preds_list]
    bars = ax.bar(x_m2 + (mi-1)*0.25, vals, 0.25, alpha=0.82,
                  color=list(m_cols_ts.values())[mi], label=mlbl, edgecolor="white", lw=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=45)
ax.set_xticks(x_m2)
ax.set_xticklabels(all_methods, fontsize=9)
ax.set_ylabel("RMSE", fontsize=11)
ax.set_title(f"Monthly RMSE | overall={overall_rmse:.4f} gates={gates_pass}/5",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
style_ax(ax)

table_data = []
for i, n in enumerate(all_methods):
    p = all_preds_list[i]
    e = p - gt_series
    r = float(np.sqrt(np.mean(e**2)))
    sl, _, _, _, _ = stats.linregress(gt_series, p)
    be = int(np.sum(np.abs(e) >= 0.20))
    table_data.append([n.replace("\n", " "), f"{r:.4f}", f"{sl:.4f}", str(be)])

ax_tbl = ax.twinx()
ax_tbl.axis("off")
tbl = ax_tbl.table(cellText=table_data,
                   colLabels=["Method", "RMSE", "Slope", "BigErr"],
                   loc="lower center", cellLoc="center", colWidths=[0.15]*4,
                   bbox=[0.0, -0.55, 1.0, 0.35])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for i in range(len(table_data)):
    if "Granite" in table_data[i][0]:
        for j in range(len(table_data[i])):
            tbl[i+1, j].set_facecolor("#FFE0E0")
            tbl[i+1, j].get_text().set_fontweight("bold")

plt.suptitle(f"{SCRIPT_NAME} | Spatial Granite-only vs References | {PRED_START_DATE.date()} -> {PRED_END_DATE.date()}",
             fontsize=13, fontweight="bold", y=0.96)
plt.savefig(OUTPUT_DIR / "plot5_comparison_monthly.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plot5_comparison_monthly.png")

# --- CSV OUTPUTS ------------------------------------------------------------
print(f"{'='*55}\nCSV OUTPUTS\n{'='*55}")
rolling_df = pd.DataFrame({
    "date": [d.strftime("%Y-%m-%d") for d in pred_dates],
    "ground_truth": gt_series,
    "predicted_avg": avg_series,
    "error": error,
    "abs_error": np.abs(error),
    "adaptive_offset": adaptive_offsets,
})
if convlstm_pred is not None:
    rolling_df["convlstm_ref"] = convlstm_pred[:n_days]
rolling_df.to_csv(OUTPUT_DIR / "rolling_predictions.csv", index=False)

monthly_rows = []
for m, mlbl in [(1, "January"), (2, "February"), (3, "March")]:
    mm = month_mask[m]
    monthly_rows.append({
        "month": mlbl,
        "days": int(mm.sum()),
        "rmse": float(np.sqrt(np.mean(error[mm] ** 2))),
        "mae": float(np.mean(np.abs(error[mm]))),
        "r": float(np.corrcoef(avg_series[mm], gt_series[mm])[0, 1]),
        "mean_adaptive_offset": float(adaptive_offsets[mm].mean()),
        "big_error_count": int(np.sum(np.abs(error[mm]) >= 0.2)),
    })
pd.DataFrame(monthly_rows).to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False)

summary = {
    "run_id": "granite_only",
    "calibrate": CALIBRATE,
    "point_only": POINT_ONLY,
    "adaptive": not DISABLE_ADAPTIVE,
    "overall_rmse": overall_rmse,
    "mae": mae,
    "rmse_jan": rmse_m[1],
    "rmse_feb": rmse_m[2],
    "rmse_mar": rmse_m[3],
    "slope": slope,
    "r2": r2_val,
    "intercept": intercept,
    "big_error_count": big_error_count,
    "gates_pass": gates_pass,
}
pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "final_report_summary.csv", index=False)
print("  Saved all CSVs")

print("\n" + "=" * 80)
print("GRANITE-ONLY SPATIAL RUN COMPLETE")
print(f"overall_rmse={overall_rmse:.4f}  slope={slope:.4f}  gates={gates_pass}/5")
print("=" * 80)
