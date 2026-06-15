#!/usr/bin/env python3
"""
model_comparison_kaggle.py  —  Multi-Model Comparison: ConvLSTM | Chronos | Granite
════════════════════════════════════════════════════════════════════════════════════
Reads rolling_predictions CSVs for 3 models from Kaggle input dataset and produces:

  taylor_diagram.png             — Taylor Diagram (R, normalised σ, CRMSE)
  error_density_curves.png       — KDE: RMSE distribution + Correlation distribution
  timeseries_overlay.png         — All models vs ground truth + error panel
  overall_rmse_comparison.png    — Overall RMSE bar chart per model
  monthly_rmse_bar.png           — Grouped bar: RMSE by month & model
  error_violin.png               — Violin: error distribution per model per month
  skill_table.png                — Rendered metrics table image
  skill_scores.csv               — Full metrics table with SRR (scaled 0–1)

USAGE: Run after all three forecast scripts have completed.
       Designed for Kaggle: /kaggle/input/datasets/rayofc/4model-comparison/
"""

import os, sys, zipfile, io, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import warnings; warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ══════════════════════════════════════════════════════════════════════════════
from datetime import datetime

BASE = "/kaggle/input/datasets/rayofc/4model-comparison"
RUN_TS = datetime.now().strftime('%Y%m%d_%H%M')
OUT = Path(f"/kaggle/working/outputs/model_comparison_{RUN_TS}")
OUT.mkdir(parents=True, exist_ok=True)

PLOT_DPI = 250

MONTHS = {1:'January', 2:'February', 3:'March'}
M_SHORT = {1:'Jan', 2:'Feb', 3:'Mar'}

FILES = {
    'ConvLSTM': 'rolling_predictions_code_69.csv',
    'Chronos':  'rolling_predictions_code_86.csv',
    'Granite':  'rolling_predictions_code_87.csv',
}

COLORS = {'ConvLSTM': '#2980B9', 'Chronos': '#27AE60', 'Granite': '#E74C3C'}
MARKERS = {'ConvLSTM': 'o', 'Chronos': 's', 'Granite': '^'}
LSTYLES = {'ConvLSTM': '-', 'Chronos': '--', 'Granite': '-.'}

print("=" * 70)
print("  model_comparison_kaggle.py")
print("=" * 70)

# Check base path
if not os.path.exists(BASE):
    print(f"WARNING: Base path not found: {BASE}")
    if os.path.exists("rolling_predictions_code_69.csv"):
        BASE = "."
        print(f"Using local fallback: {BASE}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_model_csv(name, filename):
    fp = os.path.join(BASE, filename)
    if not os.path.exists(fp):
        print(f"  ERROR: {name} CSV not found at {fp}")
        return None
    try:
        df = pd.read_csv(fp, parse_dates=['date'])
        if 'ground_truth' not in df.columns:
            for col in ['actual', 'actual_sst', 'obs', 'y_true', 'target']:
                if col in df.columns:
                    df = df.rename(columns={col: 'ground_truth'})
                    break
        if 'predicted_avg' not in df.columns:
            for col in ['predicted', 'pred', 'y_pred', 'forecast']:
                if col in df.columns:
                    df = df.rename(columns={col: 'predicted_avg'})
                    break
            else:
                day_cols = [c for c in df.columns if c.startswith('pred_day')]
                if day_cols:
                    df['predicted_avg'] = df[day_cols].mean(axis=1)
        if 'ground_truth' not in df.columns or 'predicted_avg' not in df.columns:
            print(f"  ERROR: {name} missing ground_truth or predicted_avg")
            print(f"    Columns: {list(df.columns)}")
            return None
        df['error'] = df['predicted_avg'] - df['ground_truth']
        df['abs_error'] = df['error'].abs()
        if 'month' not in df.columns and 'date' in df.columns:
            df['month'] = df['date'].dt.month
        rmse = np.sqrt((df['error']**2).mean())
        print(f"  Loaded {name}: {len(df)} rows, RMSE={rmse:.4f}")
        return df
    except Exception as e:
        print(f"  ERROR reading {name}: {e}")
        return None

dfs = {}
for name, fname in FILES.items():
    dfs[name] = load_model_csv(name, fname)

dfs = {k: v for k, v in dfs.items() if v is not None}

if len(dfs) < 2:
    print("Need at least 2 model datasets to compare. Exiting.")
    sys.exit(1)

# Align on common dates
common_dates = None
for df in dfs.values():
    s = set(df['date'].dt.strftime('%Y-%m-%d').tolist())
    common_dates = s if common_dates is None else common_dates & s
common_dates = sorted(common_dates)
print(f"\nAligned on common prediction days: {len(common_dates)}")

for name in list(dfs.keys()):
    dfs[name] = dfs[name][
        dfs[name]['date'].dt.strftime('%Y-%m-%d').isin(common_dates)
    ].reset_index(drop=True)

gt = dfs[list(dfs.keys())[0]]['ground_truth'].values
dates_arr = dfs[list(dfs.keys())[0]]['date'].values

# ══════════════════════════════════════════════════════════════════════════════
# 2. METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing skill scores...")

metric_rows = []
for name, df in dfs.items():
    pred = df['predicted_avg'].values
    err = df['error'].values
    for m in [0, 1, 2, 3]:
        if m == 0:
            mask = np.ones(len(err), dtype=bool)
            mname = 'Overall'
        else:
            mask = df['month'].values == m
            mname = M_SHORT[m]
        if mask.sum() < 2:
            continue
        e = err[mask]; p = pred[mask]; g = gt[mask]
        r_val, _ = stats.pearsonr(g, p) if len(g) > 2 else (np.nan, np.nan)
        rmse = float(np.sqrt(np.mean(e**2)))
        mae = float(np.mean(np.abs(e)))
        bias = float(np.mean(e))
        r2 = float(r_val**2) if not np.isnan(r_val) else np.nan
        max_err = float(np.max(np.abs(e)))
        std_pred = float(np.std(p))
        std_obs = float(np.std(g))
        metric_rows.append({
            'Model': name, 'Period': mname,
            'RMSE': rmse, 'MAE': mae, 'Bias': bias,
            'R': float(r_val) if not np.isnan(r_val) else 0.0,
            'R2': r2, 'MaxErr': max_err,
            'Std_pred': std_pred, 'Std_obs': std_obs,
        })

metric_df = pd.DataFrame(metric_rows)

# SRR (Skill Score Ratio, scaled 0–1)
overall = metric_df[metric_df['Period'] == 'Overall'].copy()
if len(overall) > 1:
    max_rmse = overall['RMSE'].max()
    min_r = overall['R'].min()
    max_r = overall['R'].max()
    r_range = max_r - min_r if max_r > min_r else 1.0
    for idx in overall.index:
        rmse_norm = 1.0 - overall.loc[idx, 'RMSE'] / max_rmse
        r_norm = (overall.loc[idx, 'R'] - min_r) / r_range
        overall.loc[idx, 'SRR'] = 0.5 * rmse_norm + 0.5 * r_norm
    srr_map = dict(zip(overall['Model'], overall['SRR']))
    metric_df['SRR'] = metric_df.apply(
        lambda r: srr_map.get(r['Model'], np.nan) if r['Period'] == 'Overall' else np.nan,
        axis=1
    )

# Overall print
overall_sorted = overall.sort_values('SRR', ascending=False)
print("\nSkill scores (Overall):")
for _, row in overall_sorted.iterrows():
    print(f"  {row['Model']:10s}  RMSE={row['RMSE']:.4f}  MAE={row['MAE']:.4f}  "
          f"R={row['R']:.4f}  SRR={row['SRR']:.4f}")

# Save CSV
csv_path = OUT / 'skill_scores.csv'
metric_df.to_csv(csv_path, index=False)
print(f"Saved -> {csv_path}")

best_model = "N/A"
if 'SRR' in overall.columns and len(overall) > 0:
    best_model = overall.loc[overall['SRR'].idxmax(), 'Model']
    print(f"Best model by SRR: {best_model}")

# ══════════════════════════════════════════════════════════════════════════════
# TAYLOR DIAGRAM — Standard Taylor (r=1 horizontal, OBS on x-axis at σ_obs)
# ══════════════════════════════════════════════════════════════════════════════
print("\nTaylor Diagram...")

sigma_obs = float(np.std(gt))

def taylor_diagram(ax, sigma_obs, models_data, title):
    max_sigma = max(sigma_obs, max([d[1] for d in models_data.values()]))
    max_crmse_plot = max([0.5] + [d[2] for d in models_data.values()])
    r_max = max(max_sigma * 1.5, sigma_obs + max_crmse_plot * 1.2)
    n_theta = 400

    # σ circles (centered at origin)
    # x = s * cos(θ), y = s * sin(θ) for θ from 0 (x-axis) to π/2 (y-axis)
    sigma_ticks = [round(sigma_obs * f, 3) for f in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]]
    sigma_ticks = [s for s in sigma_ticks if s <= r_max * 0.95]
    for s_val in sigma_ticks:
        theta_c = np.linspace(0, np.pi / 2, n_theta)
        xc = s_val * np.cos(theta_c)
        yc = s_val * np.sin(theta_c)
        is_obs = abs(s_val - sigma_obs) < 0.001
        ax.plot(xc, yc, 'b--' if is_obs else 'k:', lw=1.0 if is_obs else 0.5,
                alpha=0.45 if is_obs else 0.3)
        # Label near x-axis (small angle)
        ax.text(s_val * np.cos(np.radians(5)), s_val * np.sin(np.radians(5)),
                f'\u03c3={s_val:.2f}\u00b0C', fontsize=8,
                color='#1155aa' if is_obs else '#336',
                alpha=0.9 if is_obs else 0.6,
                fontweight='bold' if is_obs else 'normal')

    # R correlation radial lines
    # r=1 → θ=0 → along x-axis (right), r=0 → θ=π/2 → along y-axis (up)
    r_vals = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    for r_corr in r_vals:
        theta = np.arccos(r_corr)
        x_end = r_max * np.cos(theta)  # = r_max * r_corr
        y_end = r_max * np.sin(theta)  # = r_max * √(1 - r_corr²)
        ax.plot([0, x_end], [0, y_end], 'g-', lw=0.5, alpha=0.18)
        ax.text(x_end * 0.88, y_end * 0.88 + 0.006 * r_max,
                f'R={r_corr}', fontsize=7, color='#282',
                ha='center', rotation=np.degrees(-theta), alpha=0.65)

    # CRMSE arcs (centered on OBS at (σ_obs, 0) on x-axis)
    crmse_vals = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    crmse_vals = [c for c in crmse_vals if c < r_max * 0.95]
    for crmse in crmse_vals:
        theta_c = np.linspace(0, np.pi / 2, n_theta)
        xc = sigma_obs + crmse * np.cos(theta_c)
        yc = crmse * np.sin(theta_c)
        valid = (xc >= 0) & (xc <= r_max) & (yc >= 0) & (yc <= r_max)
        if valid.sum() > 10:
            ax.plot(xc[valid], yc[valid], 'k--', lw=0.6, alpha=0.3)
            # Label near 45° diagonal, no bbox
            xlab = sigma_obs + crmse * np.cos(np.radians(50))
            ylab = crmse * np.sin(np.radians(50))
            if 0 <= xlab <= r_max and 0 <= ylab <= r_max:
                ax.text(xlab, ylab, f'{crmse:.2f}\u00b0C', fontsize=7, color='#555',
                        ha='center', va='center')

    # OBS reference point at (σ_obs, 0) on x-axis
    ax.plot(sigma_obs, 0, 'k*', ms=18, zorder=10,
            label=f'Observed (\u03c3={sigma_obs:.3f}\u00b0C)')
    ax.text(sigma_obs, -0.02 * r_max, 'OBS', ha='center', fontsize=10,
            fontweight='bold', color='k')

    # Model points
    for mname, (r_val, sigma_m, crmse, x_pos, y_pos) in models_data.items():
        col = COLORS.get(mname, 'gray')
        mrk = MARKERS.get(mname, 'o')
        ax.plot(x_pos, y_pos, mrk, ms=15, color=col, markeredgecolor='black',
                markeredgewidth=0.8, zorder=8,
                label=f'{mname}  R={r_val:.3f}  \u03c3={sigma_m:.3f}\u00b0C')
        ax.annotate(mname, (x_pos, y_pos), fontsize=11, fontweight='bold',
                    xytext=(x_pos + 0.02 * r_max, y_pos + 0.02 * r_max), color=col)

    ax.set_xlim(-0.05 * r_max, r_max)
    ax.set_ylim(-0.05 * r_max, r_max)
    ax.set_xlabel('Standard Deviation (\u00b0C)', fontsize=12)
    ax.set_ylabel('Standard Deviation (\u00b0C)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.90)
    ax.set_aspect('equal')
    ax.grid(which='major', alpha=0.08, ls='--')

# Compute model data: (r, sigma_m, crmse, x, y)
# x = σ * cos(arccos(r)) = σ * r  (horizontal, r=1 direction)
# y = σ * sin(arccos(r)) = σ * √(1-r²)  (vertical, r=0 direction)
taylor_data = {}
for name, df in dfs.items():
    pred = df['predicted_avg'].values
    r_val, _ = stats.pearsonr(gt, pred)
    r_val = float(np.clip(r_val, -1, 1))
    sigma_m = float(np.std(pred))
    crmse = float(np.sqrt(np.mean((pred - np.mean(pred) - gt + np.mean(gt))**2)))
    theta = np.arccos(r_val)
    x = sigma_m * np.cos(theta)  # = sigma_m * r
    y = sigma_m * np.sin(theta)  # = sigma_m * √(1-r²)
    taylor_data[name] = (r_val, sigma_m, crmse, x, y)
    print(f"  {name}: R={r_val:.4f}, σ={sigma_m:.4f}, CRMSE={crmse:.4f}")

fig, ax = plt.subplots(figsize=(13, 11), facecolor='white')
taylor_diagram(ax, sigma_obs, taylor_data,
               title='Taylor Diagram \u2014 Overall (Jan\u2013Mar 2026)\nArabian Sea SST Comparison')
plt.tight_layout()
plt.savefig(OUT / 'taylor_diagram.png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
plt.close(); print("  Saved taylor_diagram.png")

# ══════════════════════════════════════════════════════════════════════════════
# ERROR DENSITY CURVES — single-axis normalized (RMSE ← | Correlation →)
# ══════════════════════════════════════════════════════════════════════════════
print("Error Density Curves...")

np.random.seed(42)
n_bootstrap = 1000
sample_size = 60

bootstrap_results = {}
for name, df in dfs.items():
    rmse_vals = []; r_vals = []
    pred_all = df['predicted_avg'].values; gt_all = df['ground_truth'].values
    n_days = len(gt_all)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_days, size=sample_size, replace=True)
        p_s = pred_all[idx]; g_s = gt_all[idx]
        rmse_vals.append(np.sqrt(np.mean((p_s - g_s)**2)))
        r_v, _ = stats.pearsonr(g_s, p_s)
        r_vals.append(r_v)
    bootstrap_results[name] = {'rmse': np.array(rmse_vals), 'r': np.array(r_vals)}

# Normalize RMSE → [0, 0.4] (left half) and R → [0.6, 1.0] (right half)
all_rmse = np.concatenate([res['rmse'] for res in bootstrap_results.values()])
all_r = np.concatenate([res['r'] for res in bootstrap_results.values()])
min_rmse, max_rmse = all_rmse.min(), all_rmse.max()
min_r, max_r = all_r.min(), all_r.max()

fig, ax = plt.subplots(figsize=(15, 10), facecolor='white')

# Reference bands
ax.axvline(0, color='black', lw=1.5, ls='-', alpha=0.8)
ax.axvline(1, color='black', lw=1.5, ls='-', alpha=0.8)
ax.axvspan(0, 0.1, alpha=0.08, color='green', label='Ideal RMSE zone')
ax.axvspan(0.9, 1.0, alpha=0.08, color='blue', label='Ideal Correlation zone')

for name, res in bootstrap_results.items():
    col = COLORS.get(name, 'gray')
    # Scale RMSE to [0.05, 0.40]
    if max_rmse > min_rmse:
        norm_rmse = 0.05 + (res['rmse'] - min_rmse) / (max_rmse - min_rmse) * 0.35
    else:
        norm_rmse = np.full_like(res['rmse'], 0.2)
    # Scale R to [0.60, 0.95]
    if max_r > min_r:
        norm_r = 0.60 + (res['r'] - min_r) / (max_r - min_r) * 0.35
    else:
        norm_r = np.full_like(res['r'], 0.8)
    # KDE for RMSE
    kde_rmse = gaussian_kde(norm_rmse, bw_method='scott')
    x_rmse = np.linspace(0, 0.5, 200)
    d_rmse = kde_rmse(x_rmse)
    ax.plot(x_rmse, d_rmse, lw=2.5, color=col, ls='--',
            label=f"{name} RMSE (mean={res['rmse'].mean():.4f}\u00b0C)")
    ax.fill_between(x_rmse, d_rmse, alpha=0.08, color=col)
    # KDE for R
    kde_r = gaussian_kde(norm_r, bw_method='scott')
    x_r = np.linspace(0.5, 1.0, 200)
    d_r = kde_r(x_r)
    ax.plot(x_r, d_r, lw=2.5, color=col, ls='-',
            label=f"{name} Correlation (mean R={res['r'].mean():.4f})")
    ax.fill_between(x_r, d_r, alpha=0.08, color=col)

ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("Normalised Performance Scores\nRMSE \u2190 Left (lower = better)  |  Correlation \u2192 Right (higher = better)",
              fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel("Probability Density", fontsize=13, fontweight='bold', labelpad=10)
ax.set_title("Bootstrap Normalised Score Distributions — RMSE vs Correlation\n"
             "Arabian Sea SST Forecasts | Jan\u2013Mar 2026",
             fontsize=14, fontweight='bold', pad=15)
ax.grid(which='major', alpha=0.15, ls='--')
ax.legend(fontsize=10, loc='upper center', ncol=3, framealpha=0.90)

plt.tight_layout()
plt.savefig(OUT / 'error_density_curves.png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
plt.close(); print("  Saved error_density_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# TIME SERIES OVERLAY + ERROR PANEL
# ══════════════════════════════════════════════════════════════════════════════
print("Time Series Overlay...")

n = len(gt); x = np.arange(n)
fig, axes = plt.subplots(2, 1, figsize=(28, 14),
                          gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.30},
                          facecolor='white')
ax0, ax1 = axes

first_df = list(dfs.values())[0]
month_arr = first_df['month'].values
m_shade = [(1, '#EBF5FB'), (2, '#EAFAF1'), (3, '#FDEDEC')]
for m, col_s in m_shade:
    mm = month_arr == m
    if mm.any():
        ax0.axvspan(x[mm][0] - 0.5, x[mm][-1] + 0.5, alpha=1.0, color=col_s, zorder=0)

ax0.plot(x, gt, 'k-', lw=3.0, label='Ground Truth', zorder=7, alpha=0.85)

for name, df in dfs.items():
    pred = df['predicted_avg'].values
    rmse = float(np.sqrt(np.mean((pred - gt)**2)))
    r_v = float(np.corrcoef(gt, pred)[0, 1])
    ax0.plot(x, pred, lw=2.0, color=COLORS[name], ls=LSTYLES[name],
             alpha=0.85, zorder=6,
             label=f"{name}  RMSE={rmse:.4f}\u00b0C  R={r_v:.4f}")

yl0 = (min(gt.min(), *[dfs[n]['predicted_avg'].values.min() for n in dfs]) - 0.3,
       max(gt.max(), *[dfs[n]['predicted_avg'].values.max() for n in dfs]) + 0.4)
ax0.set_ylim(yl0)
for m, col_t in [(1, '#2874A6'), (2, '#1E8449'), (3, '#C0392B')]:
    mm = month_arr == m
    if mm.any():
        ax0.text(float(x[mm].mean()), yl0[1] - 0.05, M_SHORT[m],
                 ha='center', va='top', fontsize=15, fontweight='bold',
                 color=col_t, alpha=0.60)

tick_step = max(1, n // 14)
tick_x = list(range(0, n, tick_step))
ax0.set_xticks(tick_x)
ax0.set_xticklabels([first_df['date'].iloc[i].strftime('%b %d') for i in tick_x],
                     fontsize=10, rotation=30, ha='right')
ax0.set_ylabel('SST (\u00b0C)', fontsize=13, fontweight='bold')
ax0.set_xlim(-0.5, n - 0.5)
ax0.legend(loc='upper left', fontsize=11, framealpha=0.90)
ax0.grid(which='major', alpha=0.15, ls='--')
for sp in ax0.spines.values():
    sp.set_linewidth(1.1)

# Error panel
for name, df in dfs.items():
    errors = df['error'].values
    ax1.plot(x, errors, lw=1.8, color=COLORS[name], ls=LSTYLES[name],
             alpha=0.80, label=name)
ax1.axhline(0, color='k', lw=2.0, zorder=5)
for th, col_th in [(0.05, '#27AE60'), (0.10, '#F39C12')]:
    ax1.axhline(th, color=col_th, lw=1.2, ls='--', alpha=0.70)
    ax1.axhline(-th, color=col_th, lw=1.2, ls='--', alpha=0.70)
for m, col_s in m_shade:
    mm = month_arr == m
    if mm.any():
        ax1.axvspan(x[mm][0] - 0.5, x[mm][-1] + 0.5, alpha=0.5, color=col_s, zorder=0)
ax1.set_ylabel('Error (\u00b0C)', fontsize=12, fontweight='bold')
ax1.set_xticks(tick_x)
ax1.set_xticklabels([first_df['date'].iloc[i].strftime('%b %d') for i in tick_x],
                     fontsize=10, rotation=30, ha='right')
ax1.set_xlim(-0.5, n - 0.5)
ax1.legend(fontsize=11, loc='upper right', framealpha=0.90)
ax1.grid(which='major', alpha=0.15, ls='--')

model_list = " vs ".join(dfs.keys())
plt.suptitle(f"90-Day Prediction Overlay — {model_list}\n"
             "Arabian Sea SST | Jan\u2013Mar 2026",
             fontsize=15, fontweight='bold')
plt.savefig(OUT / 'timeseries_overlay.png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
plt.close(); print("  Saved timeseries_overlay.png")

# ══════════════════════════════════════════════════════════════════════════════
# MONTHLY RMSE BAR
# ══════════════════════════════════════════════════════════════════════════════
print("Monthly RMSE Bar...")

monthly_rmse = {}
for m in [1, 2, 3]:
    monthly_rmse[m] = {}
    for name, df in dfs.items():
        mm = df['month'].values == m
        if mm.sum() > 1:
            monthly_rmse[m][name] = float(np.sqrt(np.mean(df.loc[mm, 'error']**2)))

m_labels = [M_SHORT[m] for m in [1, 2, 3]]
x_pos = np.arange(len(m_labels))
n_models = len(dfs)
bar_w = 0.8 / n_models

fig, ax = plt.subplots(figsize=(11, 7), facecolor='white')
for i, name in enumerate(dfs.keys()):
    vals = [monthly_rmse[m].get(name, 0) for m in [1, 2, 3]]
    offset = (i - n_models / 2 + 0.5) * bar_w
    bars = ax.bar(x_pos + offset, vals, bar_w, color=COLORS[name],
                  label=name, edgecolor='black', linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(m_labels, fontsize=14)
ax.set_ylabel('RMSE (\u00b0C)', fontsize=13, fontweight='bold')
ax.set_title('Monthly RMSE Comparison by Model\nJan\u2013Mar 2026 | Arabian Sea SST',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, framealpha=0.90)
ax.grid(axis='y', alpha=0.20, ls='--')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT / 'monthly_rmse_bar.png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
plt.close(); print("  Saved monthly_rmse_bar.png")

# ══════════════════════════════════════════════════════════════════════════════
# OVERALL RMSE COMPARISON BAR
# ══════════════════════════════════════════════════════════════════════════════
print("Overall RMSE Comparison...")

overall_rmse = {}
for name, df in dfs.items():
    overall_rmse[name] = float(np.sqrt(np.mean(df['error']**2)))

sorted_models = sorted(overall_rmse, key=overall_rmse.get)
sorted_vals = [overall_rmse[m] for m in sorted_models]
sorted_colors = [COLORS[m] for m in sorted_models]

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
bars = ax.bar(range(len(sorted_models)), sorted_vals, color=sorted_colors,
              edgecolor='black', linewidth=0.8, width=0.5)

for bar, v in zip(bars, sorted_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{v:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xticks(range(len(sorted_models)))
ax.set_xticklabels(sorted_models, fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE (\u00b0C)', fontsize=13, fontweight='bold')
ax.set_title('Overall RMSE Comparison — All Models\nJan\u2013Mar 2026 | Arabian Sea SST',
             fontsize=14, fontweight='bold', pad=12)
ax.grid(axis='y', alpha=0.20, ls='--')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT / 'overall_rmse_comparison.png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
plt.close(); print("  Saved overall_rmse_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# ERROR VIOLIN
# ══════════════════════════════════════════════════════════════════════════════
print("Error Violin...")

fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
violin_data = []; violin_positions = []; violin_labels = []; violin_colors = []

for i, m in enumerate([1, 2, 3]):
    for j, name in enumerate(dfs.keys()):
        mm = dfs[name]['month'].values == m
        if mm.sum() > 1:
            errs = dfs[name].loc[mm, 'error'].values
            violin_data.append(errs)
            pos = i * (n_models + 1) + j
            violin_positions.append(pos)
            violin_labels.append(f"{name}\n{M_SHORT[m]}")
            violin_colors.append(COLORS[name])

parts = ax.violinplot(violin_data, positions=violin_positions, showmeans=True,
                       showmedians=True, widths=0.7)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(violin_colors[i])
    pc.set_alpha(0.6)
if 'cmeans' in parts:
    parts['cmeans'].set_color('darkred')
    parts['cmeans'].set_linewidth(2.0)
if 'cmedians' in parts:
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.2)

ax.axhline(0, color='k', lw=1.0, ls='-', alpha=0.5)
ax.set_xticks(violin_positions)
ax.set_xticklabels(violin_labels, fontsize=9, rotation=45, ha='right')
ax.set_ylabel('Error (\u00b0C)', fontsize=13, fontweight='bold')
ax.set_title('Error Distribution — Violin Plot\nPer Model per Month | Jan\u2013Mar 2026',
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.20, ls='--')

legend_patches = [Patch(color=COLORS[name], label=name) for name in dfs.keys()]
ax.legend(handles=legend_patches, fontsize=12, framealpha=0.90, loc='lower right')

plt.tight_layout()
plt.savefig(OUT / 'error_violin.png', dpi=PLOT_DPI, bbox_inches='tight', facecolor='white')
plt.close(); print("  Saved error_violin.png")

# ══════════════════════════════════════════════════════════════════════════════
# SKILL TABLE PNG
# ══════════════════════════════════════════════════════════════════════════════
print("Skill Table...")

table_data = overall[['Model', 'RMSE', 'MAE', 'Bias', 'R', 'R2', 'SRR']].sort_values('SRR', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 4.5), facecolor='white')
ax.axis('off')

col_labels = ['Model', 'RMSE (\u00b0C)', 'MAE (\u00b0C)', 'Bias (\u00b0C)', 'R', 'R\u00b2', 'SRR (0\u20131)']
ncols = len(col_labels)
last_col = ncols - 1

cell_text = []
for _, row in table_data.iterrows():
    cell_text.append([
        row['Model'],
        f"{row['RMSE']:.4f}",
        f"{row['MAE']:.4f}",
        f"{row['Bias']:.4f}",
        f"{row['R']:.4f}",
        f"{row['R2']:.4f}",
        f"{row['SRR']:.4f}",
    ])

tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc='center',
               cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.2)

# Header
for j in range(ncols):
    tbl[0, j].set_facecolor('#2C3E50')
    tbl[0, j].set_text_props(color='white', fontweight='bold', fontsize=13)

# Model rows
from matplotlib.colors import to_rgb, to_hex
for i, name in enumerate(table_data['Model']):
    bg = COLORS.get(name, '#FFFFFF')
    r, g, b = to_rgb(bg)
    light = to_hex(((r + 1) / 2, (g + 1) / 2, (b + 1) / 2))
    tbl[i + 1, 0].set_facecolor(light)
    tbl[i + 1, 0].set_text_props(fontweight='bold', fontsize=12)

# Highlight best SRR cell
best_idx = table_data['SRR'].idxmax()
tbl[best_idx + 1, last_col].set_facecolor('#D5F5E3')
tbl[best_idx + 1, last_col].set_text_props(fontweight='bold', fontsize=12)

ax.set_title('Model Skill Scores — Overall (Jan\u2013Mar 2026)\n'
             'SRR: Skill Score Ratio (0\u20131, higher = better)  |  Green cell = best model',
             fontsize=13, fontweight='bold', pad=18)

plt.tight_layout()
plt.savefig(OUT / 'skill_table.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close(); print("  Saved skill_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# ZIP OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"COMPARISON COMPLETE -> {OUT}")
print(f"{'=' * 60}")
for f in sorted(OUT.iterdir()):
    if f.is_file() and f.suffix in ('.png', '.csv'):
        sz = f.stat().st_size / 1024
        print(f"  {f.name:30s} {sz:8.1f} KB")

zip_path = OUT.parent / f"{OUT.name}.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in OUT.iterdir():
        if f.is_file() and f.suffix in ('.png', '.csv'):
            zf.write(f, f.name)
print(f"\n  ZIP: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")

try:
    from IPython.display import FileLink
    display(FileLink(str(zip_path)))
    print("  Download link generated.")
except Exception:
    print(f"  Download: {zip_path}")

print(f"{'=' * 60}")
print("Models:", ", ".join(dfs.keys()))
print("Best by SRR:", best_model)
print(f"{'=' * 60}")
