"""Targeted diagnosis of std dataset bimodality.
Focus on N_ur (x6) and m_ncdm (x7) as potential bimodality drivers."""

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

# Parameter names from the YAML configs
PARAM_NAMES = ['h', 'Omega_m', 'Omega_b', 'ln_A_s_1e10', 'n_s', 'tau_reio', 'N_ur', 'm_ncdm']

# ── Load std dataset ─────────────────────────────────────────
ds = io.FitsFile('/ceph/hpc/data/s25r06-05-users/lcdm_nu/sample/cl_100_std.fits')
x = ds.get_data('x_data')
y = ds.get_data(SPECTRUM)
print(f"x shape: {x.shape}, y shape: {y.shape}")

N_ur = x[:, 6]
m_ncdm = x[:, 7]

# ── Key statistics ───────────────────────────────────────────
y_mean = np.mean(y, axis=1)
y_min = np.min(y, axis=1)
y_max = np.max(y, axis=1)
dyn_range = y_max / y_min

# After log transform (as in LogStandardScaler)
y_log = np.log(y)
y_log_mean = np.mean(y_log, axis=0)
y_log_std = np.std(y_log, axis=0)

# Per-sample stats in log space
sample_log_mean = np.mean(y_log, axis=1)
sample_log_std = np.std(y_log, axis=1)

# ── Bimodality quantification ───────────────────────────────
# Check if sample_log_mean is bimodal using Hartigan's dip test approximation
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# KDE of sample mean in log space
kde_vals = np.linspace(sample_log_mean.min(), sample_log_mean.max(), 500)
kde = gaussian_kde(sample_log_mean)
kde_pdf = kde(kde_vals)
peaks, props = find_peaks(kde_pdf, height=0.01 * kde_pdf.max())
print(f"\nBimodality in log(mean(ratio)): {len(peaks)} peaks found")
for p in peaks:
    print(f"  Peak at log(mean ratio) = {kde_vals[p]:.4f} (mean ratio = {np.exp(kde_vals[p]):.4f})")

# ── Figure 1: Bimodality drivers ────────────────────────────
fig, axs = plt.subplots(2, 4, figsize=(24, 10))

# Row 1: histograms of each parameter, split by "low ratio" vs "high ratio"
ratio_median = np.median(y_mean)
lo_mask = y_mean < ratio_median
hi_mask = y_mean >= ratio_median
for ip in range(8):
    axs[0, ip % 4 + (0 if ip < 4 else 0)].clear() if ip >= 4 else None

# Actually, let's make a clean layout
fig, axs = plt.subplots(2, 4, figsize=(24, 10))
for ip in range(8):
    row, col = ip // 4, ip % 4
    axs[row, col].hist(x[lo_mask, ip], bins=50, alpha=0.5, label=f'low ratio', density=True)
    axs[row, col].hist(x[hi_mask, ip], bins=50, alpha=0.5, label=f'high ratio', density=True, color='red')
    axs[row, col].set_xlabel(PARAM_NAMES[ip])
    axs[row, col].set_title(PARAM_NAMES[ip])
    axs[row, col].legend(fontsize=8)
fig.suptitle('Std: parameter distributions split by median mean-ratio', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_bimod_param_split.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Figure 2: N_ur and m_ncdm as drivers ────────────────────
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# N_ur vs mean ratio
axs[0, 0].scatter(N_ur, y_mean, s=0.3, alpha=0.2)
axs[0, 0].set_xlabel('N_ur')
axs[0, 0].set_ylabel('mean(ratio)')
axs[0, 0].set_title('Mean ratio vs N_ur')

# m_ncdm vs mean ratio
axs[0, 1].scatter(m_ncdm, y_mean, s=0.3, alpha=0.2)
axs[0, 1].set_xlabel('m_ncdm')
axs[0, 1].set_ylabel('mean(ratio)')
axs[0, 1].set_title('Mean ratio vs m_ncdm')

# 2D: N_ur vs m_ncdm, coloured by mean ratio
sc = axs[0, 2].scatter(N_ur, m_ncdm, c=np.log10(y_mean), s=0.3, alpha=0.3, cmap='RdYlBu_r')
axs[0, 2].set_xlabel('N_ur')
axs[0, 2].set_ylabel('m_ncdm')
axs[0, 2].set_title('N_ur vs m_ncdm, coloured by log10(mean ratio)')
plt.colorbar(sc, ax=axs[0, 2])

# N_ur vs dynamic range
axs[1, 0].scatter(N_ur, np.log10(dyn_range), s=0.3, alpha=0.2)
axs[1, 0].set_xlabel('N_ur')
axs[1, 0].set_ylabel('log10(dynamic range)')
axs[1, 0].set_title('Dynamic range vs N_ur')

# m_ncdm vs dynamic range
axs[1, 1].scatter(m_ncdm, np.log10(dyn_range), s=0.3, alpha=0.2)
axs[1, 1].set_xlabel('m_ncdm')
axs[1, 1].set_ylabel('log10(dynamic range)')
axs[1, 1].set_title('Dynamic range vs m_ncdm')

# 2D: N_ur vs m_ncdm, coloured by dynamic range
sc = axs[1, 2].scatter(N_ur, m_ncdm, c=np.log10(dyn_range), s=0.3, alpha=0.3, cmap='RdYlBu_r')
axs[1, 2].set_xlabel('N_ur')
axs[1, 2].set_ylabel('m_ncdm')
axs[1, 2].set_title('N_ur vs m_ncdm, coloured by log10(dyn range)')
plt.colorbar(sc, ax=axs[1, 2])

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_bimod_nur_mnu.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Figure 3: Spectrum shape comparison ──────────────────────
# Split into bins of N_ur
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Low N_ur vs high N_ur spectra
N_ur_lo = N_ur < 0.3
N_ur_hi = N_ur > 0.7
np.random.seed(42)
idxs_lo = np.random.choice(np.where(N_ur_lo)[0], min(30, N_ur_lo.sum()), replace=False)
idxs_hi = np.random.choice(np.where(N_ur_hi)[0], min(30, N_ur_hi.sum()), replace=False)

for idx in idxs_lo:
    axs[0, 0].plot(y[idx], alpha=0.3, lw=0.5, color='blue')
axs[0, 0].set_title(f'N_ur < 0.3 (N={N_ur_lo.sum()})')
axs[0, 0].set_ylabel(f'{SPECTRUM} ratio')
axs[0, 0].set_yscale('log')

for idx in idxs_hi:
    axs[0, 1].plot(y[idx], alpha=0.3, lw=0.5, color='red')
axs[0, 1].set_title(f'N_ur > 0.7 (N={N_ur_hi.sum()})')
axs[0, 1].set_yscale('log')

# Mean spectra comparison
axs[0, 2].plot(np.mean(y[N_ur_lo], axis=0), label=f'N_ur < 0.3 (N={N_ur_lo.sum()})', color='blue')
axs[0, 2].plot(np.mean(y[N_ur_hi], axis=0), label=f'N_ur > 0.7 (N={N_ur_hi.sum()})', color='red')
axs[0, 2].set_yscale('log')
axs[0, 2].legend()
axs[0, 2].set_title('Mean spectra by N_ur')

# Same for m_ncdm
m_lo = m_ncdm < 0.03
m_hi = m_ncdm > 0.09
idxs_m_lo = np.random.choice(np.where(m_lo)[0], min(30, m_lo.sum()), replace=False)
idxs_m_hi = np.random.choice(np.where(m_hi)[0], min(30, m_hi.sum()), replace=False)

for idx in idxs_m_lo:
    axs[1, 0].plot(y[idx], alpha=0.3, lw=0.5, color='blue')
axs[1, 0].set_title(f'm_ncdm < 0.03 (N={m_lo.sum()})')
axs[1, 0].set_ylabel(f'{SPECTRUM} ratio')
axs[1, 0].set_yscale('log')

for idx in idxs_m_hi:
    axs[1, 1].plot(y[idx], alpha=0.3, lw=0.5, color='red')
axs[1, 1].set_title(f'm_ncdm > 0.09 (N={m_hi.sum()})')
axs[1, 1].set_yscale('log')

axs[1, 2].plot(np.mean(y[m_lo], axis=0), label=f'm_ncdm < 0.03 (N={m_lo.sum()})', color='blue')
axs[1, 2].plot(np.mean(y[m_hi], axis=0), label=f'm_ncdm > 0.09 (N={m_hi.sum()})', color='red')
axs[1, 2].set_yscale('log')
axs[1, 2].legend()
axs[1, 2].set_title('Mean spectra by m_ncdm')

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_bimod_spectra_shape.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Figure 4: Distribution in training space ─────────────────
# Simulate what LogStandardScaler + PCA does
from sklearn.preprocessing import StandardScaler as SklearnStdScaler

y_log = np.log(y)
scaler = SklearnStdScaler()
y_scaled = scaler.fit_transform(y_log)

# Per-sample RMS in scaled space
rms_scaled = np.sqrt(np.mean(y_scaled**2, axis=1))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].hist(rms_scaled, bins=100, log=True)
axs[0, 0].set_xlabel('RMS in scaled space')
axs[0, 0].set_title('Distribution of sample RMS in training space')

sc = axs[0, 1].scatter(N_ur, rms_scaled, s=0.3, alpha=0.2, c=m_ncdm, cmap='viridis')
axs[0, 1].set_xlabel('N_ur')
axs[0, 1].set_ylabel('RMS in scaled space')
axs[0, 1].set_title('RMS in training space vs N_ur (coloured by m_ncdm)')
plt.colorbar(sc, ax=axs[0, 1], label='m_ncdm')

# Histogram of y_scaled at specific ell values
for ell_idx in [0, 500, 1500, 2998]:
    axs[1, 0].hist(y_scaled[:, ell_idx], bins=100, alpha=0.4, label=f'ell~{ell_idx+2}', density=True)
axs[1, 0].set_xlabel('scaled value')
axs[1, 0].set_title('Distribution in scaled space at different ell')
axs[1, 0].legend()

# PCA applied
from sklearn.decomposition import PCA
pca = PCA(n_components=360)
y_pca = pca.fit_transform(y_scaled)
# Check explained variance
cum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"\nPCA explained variance:")
for n in [10, 50, 100, 200, 360]:
    print(f"  {n} components: {cum_var[min(n-1, len(cum_var)-1)]*100:.2f}%")

# Distribution of first few PCA components
axs[1, 1].hist(y_pca[:, 0], bins=100, alpha=0.5, label='PC1', density=True)
axs[1, 1].hist(y_pca[:, 1], bins=100, alpha=0.5, label='PC2', density=True)
axs[1, 1].hist(y_pca[:, 2], bins=100, alpha=0.5, label='PC3', density=True)
axs[1, 1].set_xlabel('PCA coefficient')
axs[1, 1].set_title('Distribution of first PCA components')
axs[1, 1].legend()

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_bimod_training_space.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Figure 5: PCA components bimodality ──────────────────────
fig, axs = plt.subplots(3, 4, figsize=(20, 12))
for i in range(12):
    row, col = i // 4, i % 4
    pc = y_pca[:, i]
    axs[row, col].hist(pc, bins=100, log=True)
    axs[row, col].set_title(f'PC{i+1} (var={pca.explained_variance_ratio_[i]*100:.2f}%)')
    axs[row, col].set_xlabel(f'PC{i+1} coefficient')
fig.suptitle('Std: first 12 PCA component distributions', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_bimod_pca_distributions.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Figure 6: PC1 vs N_ur and m_ncdm ────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sc = axs[0].scatter(N_ur, y_pca[:, 0], c=m_ncdm, s=0.3, alpha=0.3, cmap='viridis')
axs[0].set_xlabel('N_ur')
axs[0].set_ylabel('PC1')
axs[0].set_title('PC1 vs N_ur (coloured by m_ncdm)')
plt.colorbar(sc, ax=axs[0])

sc = axs[1].scatter(m_ncdm, y_pca[:, 0], c=N_ur, s=0.3, alpha=0.3, cmap='viridis')
axs[1].set_xlabel('m_ncdm')
axs[1].set_ylabel('PC1')
axs[1].set_title('PC1 vs m_ncdm (coloured by N_ur)')
plt.colorbar(sc, ax=axs[1])

# PC1 vs PC2 coloured by N_ur
sc = axs[2].scatter(y_pca[:, 0], y_pca[:, 1], c=N_ur, s=0.3, alpha=0.3, cmap='viridis')
axs[2].set_xlabel('PC1')
axs[2].set_ylabel('PC2')
axs[2].set_title('PC1 vs PC2 (coloured by N_ur)')
plt.colorbar(sc, ax=axs[2])

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_std_bimod_pca_vs_params.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Correlation analysis ─────────────────────────────────────
print("\nCorrelation of each parameter with PC1:")
for ip in range(8):
    corr = np.corrcoef(x[:, ip], y_pca[:, 0])[0, 1]
    print(f"  {PARAM_NAMES[ip]:>15s}: r = {corr:+.4f}")

print("\nCorrelation of each parameter with PC2:")
for ip in range(8):
    corr = np.corrcoef(x[:, ip], y_pca[:, 1])[0, 1]
    print(f"  {PARAM_NAMES[ip]:>15s}: r = {corr:+.4f}")

# ── Quantify bimodality using Ashman's D ─────────────────────
# For PC1, check if there's a clear bimodal split
print("\n--- Bimodality quantification ---")
print(f"PC1: mean={y_pca[:, 0].mean():.4f}, std={y_pca[:, 0].std():.4f}")
print(f"PC1: skewness={float(np.mean(((y_pca[:, 0] - y_pca[:, 0].mean())/y_pca[:, 0].std())**3)):.4f}")
print(f"PC1: kurtosis={float(np.mean(((y_pca[:, 0] - y_pca[:, 0].mean())/y_pca[:, 0].std())**4) - 3):.4f}")

# Check for bimodality in log(y) at high ell where neutrino effects are strongest  
for ell_idx in [2000, 2500, 2998]:
    vals = y_log[:, ell_idx]
    kde = gaussian_kde(vals)
    x_kde = np.linspace(vals.min(), vals.max(), 500)
    y_kde = kde(x_kde)
    peaks_idx, _ = find_peaks(y_kde, height=0.01 * y_kde.max(), distance=20)
    print(f"\nell~{ell_idx+2}: {len(peaks_idx)} peaks in log(ratio) distribution")
    for p in peaks_idx:
        print(f"  Peak at log(ratio) = {x_kde[p]:.4f} (ratio = {np.exp(x_kde[p]):.4f})")

# ── Cross-check: thin and ext ────────────────────────────────
print("\n--- Cross-check with thin/ext datasets ---")
for tag in ['thin', 'ext']:
    path = f'/ceph/hpc/data/s25r06-05-users/lcdm_nu/sample/cl_100_{tag}.fits'
    ds2 = io.FitsFile(path)
    y2 = ds2.get_data(SPECTRUM)
    y2_log = np.log(y2[np.all(y2 > 0, axis=1)])
    y2_mean_ratio = np.mean(y2[np.all(y2 > 0, axis=1)], axis=1)
    
    kde = gaussian_kde(y2_mean_ratio)
    x_kde = np.linspace(y2_mean_ratio.min(), y2_mean_ratio.max(), 500)
    y_kde = kde(x_kde)
    peaks_idx, _ = find_peaks(y_kde, height=0.01 * y_kde.max(), distance=20)
    print(f"\n{tag}: mean(ratio) has {len(peaks_idx)} peaks")
    for p in peaks_idx:
        print(f"  Peak at mean ratio = {x_kde[p]:.4f}")

print(f"\nAll figures saved to {SAVE_DIR}")
