"""Diagnose bimodality in the lcdm_nu std dataset for cl_TT_lensed.

This script loads the three datasets (thin, std, ext), inspects the raw
spectra and their log-transformed values, and identifies which parameters
drive the bimodality.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import emu_like.io as io

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECTRUM = 'cl_TT_lensed'

# ── Load data ────────────────────────────────────────────────
datasets = {}
for tag in ['thin', 'std', 'ext']:
    path = f'/ceph/hpc/data/s25r06-05-users/lcdm_nu/sample/cl_100_{tag}.fits'
    ds = io.FitsFile(path)
    x = ds.get_data('x_data')
    y = ds.get_data(SPECTRUM)
    hd = ds.get_header(SPECTRUM)
    if 'x_names' in hd:
        x_names = hd['x_names']
        if isinstance(x_names, dict):
            x_names = [x_names[k] for k in sorted(x_names.keys())]
        elif isinstance(x_names, str):
            x_names = [x_names]
    else:
        x_names = [f'x{i}' for i in range(x.shape[1])]
    datasets[tag] = {'x': x, 'y': y, 'x_names': x_names}
    print(f"{tag}: x.shape={x.shape}, y.shape={y.shape}")
    print(f"  x_names = {x_names}")
    print(f"  y range: [{np.nanmin(y):.4e}, {np.nanmax(y):.4e}]")
    print(f"  y min per sample: [{np.nanmin(np.min(y, axis=1)):.4e}, {np.nanmax(np.min(y, axis=1)):.4e}]")
    print(f"  y max per sample: [{np.nanmin(np.max(y, axis=1)):.4e}, {np.nanmax(np.max(y, axis=1)):.4e}]")
    n_nan = np.sum(np.isnan(y))
    n_inf = np.sum(np.isinf(y))
    n_neg = np.sum(y < 0)
    n_zero = np.sum(y == 0)
    print(f"  NaN: {n_nan}, Inf: {n_inf}, Negative: {n_neg}, Zero: {n_zero}")
    print()

# ── Focus on std dataset ─────────────────────────────────────
x_std = datasets['std']['x']
y_std = datasets['std']['y']
x_names = datasets['std']['x_names']

# Basic spectra statistics
y_mean_per_sample = np.mean(y_std, axis=1)
y_min_per_sample = np.min(y_std, axis=1)
y_max_per_sample = np.max(y_std, axis=1)

# Check for problematic values before log
log_safe = np.all(y_std > 0, axis=1)
print(f"Std dataset: {np.sum(log_safe)}/{len(log_safe)} samples have all-positive y (safe for log)")
print(f"  Samples with y<=0 values: {np.sum(~log_safe)}")

# ── Plot 1: Distribution of spectra at key ell values ────────
ell_indices = [0, 100, 500, 1000, 2000, 2998]  # sample ell positions
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.ravel()
for i, ell_idx in enumerate(ell_indices):
    vals = y_std[:, ell_idx]
    axs[i].hist(vals[np.isfinite(vals)], bins=100, log=True)
    axs[i].set_title(f'ell index {ell_idx} (ell~{ell_idx+2})')
    axs[i].set_xlabel(f'{SPECTRUM} ratio')
    axs[i].set_ylabel('count')
fig.suptitle('Std dataset: distribution of CL_TT_lensed ratio at different ell', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_ratio_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Plot 2: Distribution in log space ────────────────────────
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.ravel()
for i, ell_idx in enumerate(ell_indices):
    vals = y_std[:, ell_idx]
    vals_pos = vals[vals > 0]
    axs[i].hist(np.log(vals_pos), bins=100, log=True)
    axs[i].set_title(f'ell index {ell_idx} (ell~{ell_idx+2})')
    axs[i].set_xlabel(f'log({SPECTRUM} ratio)')
    axs[i].set_ylabel('count')
fig.suptitle('Std dataset: log-space distribution at different ell', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_log_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Plot 3: Mean spectrum shape for different parameter bins ──
# Find which parameters separate the data the most
n_params = x_std.shape[1]
print(f"\nParameter ranges in std dataset:")
for ip in range(n_params):
    print(f"  {x_names[ip]:>15s}: [{x_std[:, ip].min():.4e}, {x_std[:, ip].max():.4e}]")

# Split each parameter at median and compare mean spectra
fig, axs = plt.subplots(2, 4, figsize=(24, 10))
axs = axs.ravel()
for ip in range(n_params):
    med = np.median(x_std[:, ip])
    lo = x_std[:, ip] < med
    hi = x_std[:, ip] >= med
    y_lo_mean = np.mean(y_std[lo], axis=0)
    y_hi_mean = np.mean(y_std[hi], axis=0)
    axs[ip].plot(y_lo_mean, label=f'{x_names[ip]} < {med:.3f}', alpha=0.8)
    axs[ip].plot(y_hi_mean, label=f'{x_names[ip]} >= {med:.3f}', alpha=0.8)
    axs[ip].set_title(x_names[ip])
    axs[ip].set_ylabel(f'{SPECTRUM} ratio')
    axs[ip].legend(fontsize=8)
    axs[ip].set_yscale('log')
for ip in range(n_params, len(axs)):
    axs[ip].set_visible(False)
fig.suptitle('Std: mean spectrum split by each parameter at median', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_param_split_spectra.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Plot 4: Mean spectrum variance per ell in log space ──────
y_log = np.log(y_std[log_safe])
log_mean = np.mean(y_log, axis=0)
log_std = np.std(y_log, axis=0)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(log_mean)
axs[0].set_title('Mean of log(ratio) per ell')
axs[0].set_xlabel('ell index')
axs[0].set_ylabel('mean log(ratio)')
axs[1].plot(log_std)
axs[1].set_title('Std of log(ratio) per ell')
axs[1].set_xlabel('ell index')
axs[1].set_ylabel('std log(ratio)')
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_log_stats.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Plot 5: Scatter of y at ell=2 vs ell=1000, coloured by each param ──
fig, axs = plt.subplots(2, 4, figsize=(24, 10))
axs = axs.ravel()
for ip in range(n_params):
    sc = axs[ip].scatter(y_std[:, 0], y_std[:, 1000], c=x_std[:, ip],
                          s=0.3, alpha=0.3, cmap='viridis')
    axs[ip].set_xlabel(f'{SPECTRUM}[ell=2]')
    axs[ip].set_ylabel(f'{SPECTRUM}[ell=1002]')
    axs[ip].set_title(f'coloured by {x_names[ip]}')
    plt.colorbar(sc, ax=axs[ip])
for ip in range(n_params, len(axs)):
    axs[ip].set_visible(False)
fig.suptitle('Std: spectrum value at ell=2 vs ell=1002, coloured by parameters', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_scatter_ell_params.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Plot 6: Check if the ratio crosses 1.0 for some samples ─
# A ratio spectrum: y = C_ell / C_ell_ref
# For LCDM with no neutrinos, ratio ~ constant near 1
# With m_ncdm > 0, the shape changes
ratio_mean = np.mean(y_std, axis=1)
ratio_spread = np.std(y_std, axis=1) / ratio_mean  # coefficient of variation

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].hist(ratio_mean, bins=100, log=True)
axs[0].set_xlabel('mean(ratio) over ell')
axs[0].set_title('Distribution of mean ratio')

axs[1].hist(ratio_spread, bins=100, log=True)
axs[1].set_xlabel('std/mean (CV) over ell')
axs[1].set_title('Distribution of ratio spread')

# Find m_ncdm index
try:
    m_idx = list(x_names).index('m_ncdm')
    axs[2].scatter(x_std[:, m_idx], ratio_mean, s=0.3, alpha=0.3)
    axs[2].set_xlabel('m_ncdm')
    axs[2].set_ylabel('mean(ratio) over ell')
    axs[2].set_title('Mean ratio vs m_ncdm')
except ValueError:
    axs[2].text(0.5, 0.5, 'm_ncdm not found', transform=axs[2].transAxes)

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_ratio_stats.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Plot 7: Sample spectra from different m_ncdm bins ────────
try:
    m_idx = list(x_names).index('m_ncdm')
    m_vals = x_std[:, m_idx]
    
    # Also find N_ur
    try:
        n_idx = list(x_names).index('N_ur')
    except ValueError:
        n_idx = None
    
    m_bins = np.linspace(m_vals.min(), m_vals.max(), 6)
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    for i in range(5):
        mask = (m_vals >= m_bins[i]) & (m_vals < m_bins[i+1])
        n_show = min(20, mask.sum())
        idxs = np.where(mask)[0][:n_show]
        for idx in idxs:
            axs[i].plot(y_std[idx], alpha=0.5, lw=0.5)
        axs[i].set_title(f'm_ncdm in [{m_bins[i]:.3f}, {m_bins[i+1]:.3f}]\n(N={mask.sum()})')
        axs[i].set_yscale('log')
        axs[i].set_ylabel(f'{SPECTRUM} ratio')
        axs[i].set_xlabel('ell index')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'diag_std_spectra_by_mnu.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # ── Plot 8: 2D scatter m_ncdm vs N_ur coloured by mean ratio ──
    if n_idx is not None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sc = axs[0].scatter(x_std[:, m_idx], x_std[:, n_idx], c=np.log10(ratio_mean),
                             s=0.3, alpha=0.3, cmap='RdYlBu_r')
        axs[0].set_xlabel('m_ncdm')
        axs[0].set_ylabel('N_ur')
        axs[0].set_title('coloured by log10(mean ratio)')
        plt.colorbar(sc, ax=axs[0])
        
        sc = axs[1].scatter(x_std[:, m_idx], x_std[:, n_idx], c=np.log10(ratio_spread),
                             s=0.3, alpha=0.3, cmap='RdYlBu_r')
        axs[1].set_xlabel('m_ncdm')
        axs[1].set_ylabel('N_ur')
        axs[1].set_title('coloured by log10(CV)')
        plt.colorbar(sc, ax=axs[1])
        
        sc = axs[2].scatter(x_std[:, m_idx], x_std[:, n_idx], c=np.log10(y_min_per_sample),
                             s=0.3, alpha=0.3, cmap='RdYlBu_r')
        axs[2].set_xlabel('m_ncdm')
        axs[2].set_ylabel('N_ur')
        axs[2].set_title('coloured by log10(min(ratio))')
        plt.colorbar(sc, ax=axs[2])
        
        plt.tight_layout()
        fig.savefig(os.path.join(SAVE_DIR, 'diag_std_mnu_nur_scatter.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

except ValueError:
    print("m_ncdm not found in x_names")

# ── Plot 9: Compare thin vs std vs ext distributions ─────────
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for tag, d in datasets.items():
    y = d['y']
    m = np.mean(y, axis=1)
    axs[0].hist(m, bins=100, alpha=0.5, label=tag, density=True, log=True)
axs[0].set_xlabel('mean(ratio) over ell')
axs[0].set_title('Mean ratio distribution')
axs[0].legend()

for tag, d in datasets.items():
    y = d['y']
    y_pos = y[np.all(y > 0, axis=1)]
    log_y = np.log(y_pos)
    log_mean = np.mean(log_y, axis=0)
    log_std = np.std(log_y, axis=0)
    axs[1].plot(log_std, label=tag, alpha=0.8)
axs[1].set_xlabel('ell index')
axs[1].set_ylabel('std(log(ratio))')
axs[1].set_title('Log-space variance per ell')
axs[1].legend()

for tag, d in datasets.items():
    y = d['y']
    dyn = np.max(y, axis=1) / np.min(y, axis=1)
    axs[2].hist(np.log10(dyn), bins=100, alpha=0.5, label=tag, density=True, log=True)
axs[2].set_xlabel('log10(dynamic range)')
axs[2].set_title('Dynamic range')
axs[2].legend()

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_cross_dataset_overview.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Plot 10: The actual bimodality test ──────────────────────
# Check if the histogram of log-transformed values is bimodal
# at certain ell values for std but not for thin
fig, axs = plt.subplots(3, 4, figsize=(20, 12))
for i, ell_idx in enumerate([0, 100, 500, 1000]):
    for j, tag in enumerate(['thin', 'std', 'ext']):
        y = datasets[tag]['y']
        vals = y[:, ell_idx]
        vals_pos = vals[vals > 0]
        axs[j, i].hist(np.log(vals_pos), bins=100, log=True)
        axs[j, i].set_title(f'{tag}, ell~{ell_idx+2}')
        axs[j, i].set_xlabel('log(ratio)')
        axs[j, i].set_ylabel('count')
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_bimodality_test_all_datasets.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Summary statistics ───────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for tag in ['thin', 'std', 'ext']:
    y = datasets[tag]['y']
    x = datasets[tag]['x']
    names = datasets[tag]['x_names']
    print(f"\n{tag}:")
    print(f"  N samples: {y.shape[0]}")
    print(f"  y range: [{np.nanmin(y):.4e}, {np.nanmax(y):.4e}]")
    print(f"  Mean ratio: [{np.nanmin(np.mean(y, axis=1)):.4e}, {np.nanmax(np.mean(y, axis=1)):.4e}]")
    print(f"  Dynamic range: [{np.nanmin(np.max(y, axis=1)/np.min(y, axis=1)):.2f}, "
          f"{np.nanmax(np.max(y, axis=1)/np.min(y, axis=1)):.2f}]")
    
    # Check log-safety
    n_neg = np.sum(np.any(y <= 0, axis=1))
    print(f"  Samples with non-positive values: {n_neg}")
    
    # Parameter ranges
    for ip, name in enumerate(names):
        print(f"  {name:>15s}: [{x[:, ip].min():.4e}, {x[:, ip].max():.4e}]")

print(f"\nAll figures saved to {SAVE_DIR}")
