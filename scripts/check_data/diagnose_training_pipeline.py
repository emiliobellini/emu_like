"""Simulate the training data pipeline for std and identify the training difficulty.
Replicates: load -> train/test split -> LogStandardScaler -> PCA -> check loss space."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import emu_like.io as io

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECTRUM = 'cl_TT_lensed'
PARAM_NAMES = ['h', 'Omega_m', 'Omega_b', 'ln_A_s_1e10', 'n_s', 'tau_reio', 'N_ur', 'm_ncdm']

# ── Simulate full pipeline for each dataset ──────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

results = {}
for tag in ['thin', 'std', 'ext']:
    print(f"\n{'='*60}")
    print(f"Processing {tag} dataset")
    path = f'/ceph/hpc/data/s25r06-05-users/lcdm_nu/sample/cl_100_{tag}.fits'
    ds = io.FitsFile(path)
    x = ds.get_data('x_data')
    y = ds.get_data(SPECTRUM)
    
    # Remove NaN rows
    valid = np.all(np.isfinite(y), axis=1) & np.all(y > 0, axis=1)
    x = x[valid]
    y = y[valid]
    print(f"  Valid samples: {valid.sum()} / {len(valid)}")
    
    # Train/test split (same as config)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.9, random_state=1543)
    
    # LogStandardScaler: log then standardize
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
    
    scaler = StandardScaler()
    scaler.fit(y_train_log)
    y_train_scaled = scaler.transform(y_train_log)
    y_test_scaled = scaler.transform(y_test_log)
    
    # PCA with 360 components
    pca = PCA(n_components=min(360, y_train_scaled.shape[1]))
    pca.fit(y_train_scaled)
    y_train_pca = pca.transform(y_train_scaled)
    y_test_pca = pca.transform(y_test_scaled)
    
    # Loss weight (normalized singular values with floor 0.1)
    sv = pca.singular_values_
    sv_norm = sv / sv[0]
    sv_norm = np.maximum(sv_norm, 0.1)
    
    # Weighted MSE per sample
    train_residuals = y_train_pca  # residual from zero in PCA space
    test_residuals = y_test_pca
    
    # Per-sample weighted MSE (simulating what loss function sees)
    # The network targets are the PCA coefficients
    # Let's compute the MSE of each sample's PCA coefficients (as proxy for difficulty)
    train_norms = np.sqrt(np.mean(sv_norm[np.newaxis, :] * y_train_pca**2, axis=1))
    test_norms = np.sqrt(np.mean(sv_norm[np.newaxis, :] * y_test_pca**2, axis=1))
    
    # Unweighted norms
    train_norms_uw = np.sqrt(np.mean(y_train_pca**2, axis=1))
    test_norms_uw = np.sqrt(np.mean(y_test_pca**2, axis=1))
    
    results[tag] = {
        'x_train': x_train, 'x_test': x_test,
        'y_train_pca': y_train_pca, 'y_test_pca': y_test_pca,
        'train_norms': train_norms, 'test_norms': test_norms,
        'train_norms_uw': train_norms_uw, 'test_norms_uw': test_norms_uw,
        'pca': pca, 'sv_norm': sv_norm,
        'scaler': scaler,
    }
    
    print(f"  PCA explained variance (10 comp): {np.sum(pca.explained_variance_ratio_[:10])*100:.2f}%")
    print(f"  Singular values: first={sv[0]:.2f}, last={sv[-1]:.6f}, ratio={sv[0]/sv[-1]:.0f}")
    print(f"  Train PCA norms: median={np.median(train_norms):.4f}, P99={np.percentile(train_norms, 99):.4f}, max={np.max(train_norms):.4f}")
    print(f"  Test PCA norms:  median={np.median(test_norms):.4f}, P99={np.percentile(test_norms, 99):.4f}, max={np.max(test_norms):.4f}")
    
    # Check for outliers in PCA space
    threshold_99 = np.percentile(train_norms, 99)
    outliers_train = train_norms > threshold_99
    outliers_test = test_norms > threshold_99
    mse_contribution_outliers = np.mean(train_norms[outliers_train]**2) / np.mean(train_norms**2)
    print(f"  Train outliers (>P99): {outliers_train.sum()} samples, contribute {mse_contribution_outliers*100:.1f}% of total MSE")
    
    # Check distribution of PCA coefficient #0
    pc0_train = y_train_pca[:, 0]
    pc0_test = y_test_pca[:, 0]
    print(f"  PC0 train: mean={pc0_train.mean():.4f}, std={pc0_train.std():.4f}")
    print(f"  PC0 test:  mean={pc0_test.mean():.4f}, std={pc0_test.std():.4f}")

# ── Figure 1: Distribution of sample "difficulty" ────────────
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for tag in ['thin', 'std', 'ext']:
    d = results[tag]
    axs[0].hist(d['train_norms'], bins=100, alpha=0.5, label=f'{tag} train', density=True)
axs[0].set_xlabel('Weighted PCA norm (proxy for sample difficulty)')
axs[0].set_title('Training samples')
axs[0].legend()
axs[0].set_yscale('log')

for tag in ['thin', 'std', 'ext']:
    d = results[tag]
    axs[1].hist(d['test_norms'], bins=100, alpha=0.5, label=f'{tag} test', density=True)
axs[1].set_xlabel('Weighted PCA norm')
axs[1].set_title('Test samples')
axs[1].legend()
axs[1].set_yscale('log')

# The ratio of test/train distribution
for tag in ['thin', 'std', 'ext']:
    d = results[tag]
    train_pctiles = np.percentile(d['train_norms'], np.arange(1, 100))
    test_pctiles = np.percentile(d['test_norms'], np.arange(1, 100))
    axs[2].plot(np.arange(1, 100), test_pctiles / train_pctiles, label=tag)
axs[2].axhline(1, c='k', ls='--', lw=0.5)
axs[2].set_xlabel('Percentile')
axs[2].set_ylabel('Test/Train ratio')
axs[2].set_title('QQ-like: test vs train difficulty distribution')
axs[2].legend()

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_pipeline_difficulty.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Figure 2: PCA component distributions comparison ─────────
fig, axs = plt.subplots(3, 4, figsize=(20, 12))
for ip in range(12):
    row, col = ip // 4, ip % 4
    for tag in ['thin', 'std', 'ext']:
        d = results[tag]
        axs[row, col].hist(d['y_train_pca'][:, ip], bins=80, alpha=0.4, label=tag, density=True)
    axs[row, col].set_title(f'PC{ip+1}')
    axs[row, col].legend(fontsize=7)
fig.suptitle('PCA component distributions (training set)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_pipeline_pca_components.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Figure 3: The critical comparison - train vs test in PCA space ──
fig, axs = plt.subplots(3, 3, figsize=(18, 15))
for j, tag in enumerate(['thin', 'std', 'ext']):
    d = results[tag]
    
    # PC0 distribution: train vs test
    axs[j, 0].hist(d['y_train_pca'][:, 0], bins=80, alpha=0.5, label='train', density=True)
    axs[j, 0].hist(d['y_test_pca'][:, 0], bins=80, alpha=0.5, label='test', density=True, color='red')
    axs[j, 0].set_title(f'{tag}: PC1 train vs test')
    axs[j, 0].legend()
    
    # PC1 distribution: train vs test  
    axs[j, 1].hist(d['y_train_pca'][:, 1], bins=80, alpha=0.5, label='train', density=True)
    axs[j, 1].hist(d['y_test_pca'][:, 1], bins=80, alpha=0.5, label='test', density=True, color='red')
    axs[j, 1].set_title(f'{tag}: PC2 train vs test')
    axs[j, 1].legend()
    
    # Norm distribution: train vs test
    axs[j, 2].hist(d['train_norms'], bins=80, alpha=0.5, label='train', density=True)
    axs[j, 2].hist(d['test_norms'], bins=80, alpha=0.5, label='test', density=True, color='red')
    axs[j, 2].set_title(f'{tag}: norm train vs test')
    axs[j, 2].legend()

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_pipeline_train_vs_test.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Figure 4: The heavy tail analysis for std ────────────────
# Key question: are there "hard" samples that dominate the MSE?
d = results['std']
x_test = d['x_test']
test_norms = d['test_norms']

# Sort by difficulty
sort_idx = np.argsort(test_norms)[::-1]
cumulative_mse = np.cumsum(test_norms[sort_idx]**2) / np.sum(test_norms**2)

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Cumulative MSE contribution
axs[0, 0].plot(np.arange(len(cumulative_mse)) / len(cumulative_mse) * 100, cumulative_mse * 100)
axs[0, 0].set_xlabel('% of hardest samples')
axs[0, 0].set_ylabel('% of total MSE')
axs[0, 0].set_title('Std: cumulative MSE contribution')
axs[0, 0].axvline(1, c='r', ls='--', label='top 1%')
axs[0, 0].axvline(10, c='g', ls='--', label='top 10%')
axs[0, 0].legend()

# Hard samples: where are they in parameter space?
hard_mask = test_norms > np.percentile(test_norms, 95)
easy_mask = test_norms < np.percentile(test_norms, 50)

for ip, pname in enumerate(PARAM_NAMES):
    pass  # removed broken indexing

# Focus on N_ur and m_ncdm
axs[0, 1].hist(x_test[easy_mask, 6], bins=40, alpha=0.5, label='easy (<P50)', density=True)
axs[0, 1].hist(x_test[hard_mask, 6], bins=40, alpha=0.5, label='hard (>P95)', density=True, color='red')
axs[0, 1].set_xlabel('N_ur')
axs[0, 1].set_title('Std: N_ur distribution by difficulty')
axs[0, 1].legend()

axs[0, 2].hist(x_test[easy_mask, 7], bins=40, alpha=0.5, label='easy (<P50)', density=True)
axs[0, 2].hist(x_test[hard_mask, 7], bins=40, alpha=0.5, label='hard (>P95)', density=True, color='red')
axs[0, 2].set_xlabel('m_ncdm')
axs[0, 2].set_title('Std: m_ncdm distribution by difficulty')
axs[0, 2].legend()

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_pipeline_hard_samples.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Figure 5: All 8 params easy vs hard ─────────────────────
fig, axs = plt.subplots(2, 4, figsize=(24, 10))
axs = axs.ravel()
for ip in range(8):
    axs[ip].hist(x_test[easy_mask, ip], bins=40, alpha=0.5, label='easy (<P50)', density=True)
    axs[ip].hist(x_test[hard_mask, ip], bins=40, alpha=0.5, label='hard (>P95)', density=True, color='red')
    axs[ip].set_xlabel(PARAM_NAMES[ip])
    axs[ip].set_title(PARAM_NAMES[ip])
    axs[ip].legend(fontsize=8)
fig.suptitle('Std test: parameter distributions by difficulty (PCA norm)', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_pipeline_all_params_difficulty.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Quantitative: what fraction of MSE comes from top N% ────
print("\n" + "="*60)
print("Critical analysis: MSE concentration in hard samples")
print("="*60)
for tag in ['thin', 'std', 'ext']:
    d = results[tag]
    norms = d['test_norms']
    sort_idx = np.argsort(norms)[::-1]
    sorted_sq = norms[sort_idx]**2
    total = sorted_sq.sum()
    for pct in [1, 5, 10, 20]:
        n = int(len(sorted_sq) * pct / 100)
        frac = sorted_sq[:n].sum() / total * 100
        print(f"  {tag}: top {pct}% of samples = {frac:.1f}% of total MSE")

# ── The key comparison: "combined" dataset ───────────────────
# When you combine thin+std+ext, the scaler is fit on all three
# This means std samples get scaled differently
print("\n" + "="*60)
print("Simulating combined (all) dataset")
print("="*60)

all_x, all_y = [], []
for tag in ['thin', 'std', 'ext']:
    path = f'/ceph/hpc/data/s25r06-05-users/lcdm_nu/sample/cl_100_{tag}.fits'
    ds = io.FitsFile(path)
    x = ds.get_data('x_data')
    y = ds.get_data(SPECTRUM)
    valid = np.all(np.isfinite(y), axis=1) & np.all(y > 0, axis=1)
    all_x.append(x[valid])
    all_y.append(y[valid])

x_all = np.vstack(all_x)
y_all = np.vstack(all_y)
print(f"Combined: {x_all.shape[0]} samples")

x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(
    x_all, y_all, train_size=0.9, random_state=1543)

y_train_a_log = np.log(y_train_a)
y_test_a_log = np.log(y_test_a)

scaler_a = StandardScaler()
scaler_a.fit(y_train_a_log)
y_train_a_scaled = scaler_a.transform(y_train_a_log)
y_test_a_scaled = scaler_a.transform(y_test_a_log)

pca_a = PCA(n_components=360)
pca_a.fit(y_train_a_scaled)
y_train_a_pca = pca_a.transform(y_train_a_scaled)
y_test_a_pca = pca_a.transform(y_test_a_scaled)

sv_a = pca_a.singular_values_
sv_a_norm = np.maximum(sv_a / sv_a[0], 0.1)

train_norms_a = np.sqrt(np.mean(sv_a_norm[np.newaxis, :] * y_train_a_pca**2, axis=1))
test_norms_a = np.sqrt(np.mean(sv_a_norm[np.newaxis, :] * y_test_a_pca**2, axis=1))

print(f"  PCA explained variance (10 comp): {np.sum(pca_a.explained_variance_ratio_[:10])*100:.2f}%")
print(f"  Singular values: first={sv_a[0]:.2f}, last={sv_a[-1]:.6f}")
print(f"  Train norms: median={np.median(train_norms_a):.4f}, P99={np.percentile(train_norms_a, 99):.4f}")
print(f"  Test norms:  median={np.median(test_norms_a):.4f}, P99={np.percentile(test_norms_a, 99):.4f}")

for pct in [1, 5, 10, 20]:
    n = int(len(test_norms_a) * pct / 100)
    sort_idx = np.argsort(test_norms_a)[::-1]
    frac = (test_norms_a[sort_idx[:n]]**2).sum() / (test_norms_a**2).sum() * 100
    print(f"  combined: top {pct}% of samples = {frac:.1f}% of total MSE")

# ── Figure 6: Distribution by source dataset in combined ─────
n_thin = len(all_x[0])
n_std = len(all_x[1])
n_ext = len(all_x[2])

# Recreate labels for the combined train/test
# We need to know which samples come from which dataset
source_labels = np.concatenate([
    np.zeros(n_thin), np.ones(n_std), 2*np.ones(n_ext)])
# Split with same seed
_, _, source_train, source_test = train_test_split(
    x_all, source_labels, train_size=0.9, random_state=1543)

tag_names = {0: 'thin', 1: 'std', 2: 'ext'}
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for src_id, name in tag_names.items():
    mask = source_test == src_id
    axs[0].hist(test_norms_a[mask], bins=80, alpha=0.5, label=f'{name} (N={mask.sum()})', density=True)
axs[0].set_xlabel('Weighted PCA norm')
axs[0].set_title('Combined test: difficulty by source dataset')
axs[0].legend()
axs[0].set_yscale('log')

# What fraction of hard samples come from each dataset?
hard_mask_a = test_norms_a > np.percentile(test_norms_a, 95)
for src_id, name in tag_names.items():
    src_mask = source_test == src_id
    n_hard = (hard_mask_a & src_mask).sum()
    n_total = src_mask.sum()
    print(f"  Hard samples from {name}: {n_hard}/{n_total} ({n_hard/n_total*100:.1f}%)")

# Fraction of top 5% from each source
counts = {}
for src_id, name in tag_names.items():
    counts[name] = (hard_mask_a & (source_test == src_id)).sum()
total_hard = hard_mask_a.sum()
bars = [counts[n] / total_hard * 100 for n in ['thin', 'std', 'ext']]
axs[1].bar(['thin', 'std', 'ext'], bars)
axs[1].set_ylabel('% of top-5% hardest samples')
axs[1].set_title(f'Source of hardest 5% samples (N={total_hard})')
for i, v in enumerate(bars):
    axs[1].text(i, v + 0.5, f'{v:.1f}%', ha='center')

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_pipeline_combined_source.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nAll figures saved to {SAVE_DIR}")
