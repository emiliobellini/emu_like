"""Final diagnosis: test the two-regime hypothesis for std dataset.

The hypothesis is that the std parameter range straddles the neutrino
mass-hierarchy transition, creating a regime boundary that MSE cannot handle
but Huber can.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import emu_like.io as io

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECTRUM = 'cl_TT_lensed'
PARAM_NAMES = ['h', 'Omega_m', 'Omega_b', 'ln_A_s_1e10', 'n_s', 'tau_reio', 'N_ur', 'm_ncdm']

# ── Load std dataset ─────────────────────────────────────────
ds = io.FitsFile('/ceph/hpc/data/s25r06-05-users/lcdm_nu/sample/cl_100_std.fits')
x = ds.get_data('x_data')
y = ds.get_data(SPECTRUM)
N_ur = x[:, 6]
m_ncdm = x[:, 7]

# ── Analysis 1: Spectrum shape variation across parameter space ──
# Compute pairwise spectrum similarity in different parameter regions
print("="*60)
print("Analysis 1: Spectrum variation across parameter space")
print("="*60)

# Define regions
regions = {
    'low_mnu_low_Nur': (m_ncdm < 0.03) & (N_ur < 0.3),
    'low_mnu_high_Nur': (m_ncdm < 0.03) & (N_ur > 0.7),
    'high_mnu_low_Nur': (m_ncdm > 0.08) & (N_ur < 0.3),
    'high_mnu_high_Nur': (m_ncdm > 0.08) & (N_ur > 0.7),
}

for name, mask in regions.items():
    print(f"  {name:25s}: {mask.sum():6d} samples")
    y_sub = y[mask]
    mean_spec = np.mean(y_sub, axis=0)
    print(f"    mean(ratio) range: [{mean_spec.min():.4f}, {mean_spec.max():.4f}]")
    print(f"    overall mean: {np.mean(mean_spec):.4f}")

# ── Analysis 2: Gradient of the function f at the transition ──
# Check how much the spectrum changes per unit change in m_ncdm
# by computing finite differences in m_ncdm direction
print("\n" + "="*60)
print("Analysis 2: Sensitivity of spectrum to each parameter")
print("="*60)

# For each sample, find its nearest neighbour in each parameter direction 
# and compute the spectrum change
from scipy.spatial import KDTree

# Standardize parameters for distance computation
x_std = (x - x.mean(axis=0)) / x.std(axis=0)

# Compute spectrum derivatives w.r.t. each parameter using local neighbours
n_sample = 5000  # Use subset for speed
np.random.seed(42)
idx_sample = np.random.choice(len(x), n_sample, replace=False)

# For each parameter, compute |dy/dx_i| * range(x_i)
# This tells us how much the spectrum changes over the parameter range
sensitivities = np.zeros((n_sample, 8))

tree = KDTree(x_std)
for i, idx in enumerate(idx_sample):
    dists, nn_idxs = tree.query(x_std[idx], k=11)
    nn_idxs = nn_idxs[1:]  # exclude self
    for ip in range(8):
        dx = x[nn_idxs, ip] - x[idx, ip]
        dy = y[nn_idxs] - y[idx]
        dy_norm = np.sqrt(np.mean(dy**2, axis=1))
        abs_dx = np.abs(dx)
        mask_good = abs_dx > 1e-10
        if mask_good.any():
            # Weighted average of |dy/dx| using closest neighbours
            sensitivities[i, ip] = np.mean(dy_norm[mask_good] / abs_dx[mask_good])

x_ranges = x.max(axis=0) - x.min(axis=0)
effective_sensitivity = sensitivities * x_ranges[np.newaxis, :]

print("\nMedian effective sensitivity (|dy/dx| * range(x)) for each parameter:")
for ip in range(8):
    med = np.median(effective_sensitivity[:, ip])
    p90 = np.percentile(effective_sensitivity[:, ip], 90)
    print(f"  {PARAM_NAMES[ip]:>15s}: median={med:.4e}, P90={p90:.4e}")

# ── Analysis 3: Check the actual spectra in the transition region ──
print("\n" + "="*60)
print("Analysis 3: Spectrum shapes in different m_ncdm regimes")
print("="*60)

fig, axs = plt.subplots(3, 4, figsize=(24, 15))

# Row 0: spectra in different m_ncdm bins
m_bins = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
for i in range(len(m_bins)-1):
    mask = (m_ncdm >= m_bins[i]) & (m_ncdm < m_bins[i+1])
    n = mask.sum()
    # Sample 50 spectra
    idxs = np.where(mask)[0][:50]
    col = min(i, 3)
    row = 0 if i < 4 else 1
    for idx in idxs:
        axs[row, col].plot(y[idx], alpha=0.3, lw=0.3)
    axs[row, col].set_title(f'm_ncdm in [{m_bins[i]:.2f}, {m_bins[i+1]:.2f}] (N={n})')
    axs[row, col].set_yscale('log')
    axs[row, col].set_ylabel('CL_TT ratio')
    axs[row, col].set_xlabel('ell index')

# Row 1-2: Also show the MEAN spectrum per m_ncdm bin, fixed N_ur
N_ur_lo = N_ur < 0.2
N_ur_hi = N_ur > 0.8

for i, m_range in enumerate([(0, 0.03), (0.03, 0.06), (0.06, 0.09), (0.09, 0.12)]):
    mask_lo = (m_ncdm >= m_range[0]) & (m_ncdm < m_range[1]) & N_ur_lo
    mask_hi = (m_ncdm >= m_range[0]) & (m_ncdm < m_range[1]) & N_ur_hi
    if mask_lo.sum() > 0:
        axs[1, i].plot(np.mean(y[mask_lo], axis=0), label=f'N_ur<0.2 (N={mask_lo.sum()})', color='blue')
    if mask_hi.sum() > 0:
        axs[1, i].plot(np.mean(y[mask_hi], axis=0), label=f'N_ur>0.8 (N={mask_hi.sum()})', color='red')
    axs[1, i].set_title(f'm_ncdm in [{m_range[0]:.2f}, {m_range[1]:.2f}]')
    axs[1, i].set_yscale('log')
    axs[1, i].legend(fontsize=8)
    axs[1, i].set_ylabel('mean CL_TT ratio')
    axs[1, i].set_xlabel('ell index')

# Row 2: Ratio of the low/high N_ur mean spectra per ell (how much N_ur changes the shape)
for i, m_range in enumerate([(0, 0.03), (0.03, 0.06), (0.06, 0.09), (0.09, 0.12)]):
    mask_lo = (m_ncdm >= m_range[0]) & (m_ncdm < m_range[1]) & N_ur_lo
    mask_hi = (m_ncdm >= m_range[0]) & (m_ncdm < m_range[1]) & N_ur_hi
    if mask_lo.sum() > 0 and mask_hi.sum() > 0:
        mean_lo = np.mean(y[mask_lo], axis=0)
        mean_hi = np.mean(y[mask_hi], axis=0)
        axs[2, i].plot(mean_hi / mean_lo - 1)
        axs[2, i].axhline(0, c='k', ls='--', lw=0.5)
    axs[2, i].set_title(f'm_ncdm in [{m_range[0]:.2f}, {m_range[1]:.2f}]')
    axs[2, i].set_ylabel('relative diff (high/low N_ur)')
    axs[2, i].set_xlabel('ell index')

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_regime_spectra.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Analysis 4: Effective N_eff computation ──────────────────
# The key physics: at recombination (T ~ 0.3 eV), neutrinos with
# m_ncdm < T are relativistic and act like N_ur.
# With deg_ncdm=3, one species with 3 degrees of freedom.
# N_eff_total = N_ur + 3 * f(m_ncdm/T) where f→1 for m<<T, f→0 for m>>T
print("\n" + "="*60)
print("Analysis 4: Effective N_eff and its correlation with spectra")
print("="*60)

# Approximate N_eff_total (simple threshold approach)
# At recombination T ~ 0.3 eV, a neutrino with m < 0.3*3.15 eV is relativistic
# For the std range m_ncdm in [0, 0.12], they're all approximately relativistic
# But the GRADIENT of the transition matters

# Compute a proxy: N_eff_approx = N_ur + 3 * (1 - m_ncdm / 0.3)
# This is very approximate but captures the trend
N_eff_approx = N_ur + 3 * np.clip(1 - m_ncdm / 0.3, 0, 1)

# Actually, for the CMB, what matters is Omega_nu vs Omega_m
# Omega_nu ~ m_ncdm * 3 / (93.14 h^2) for massive neutrinos
h = x[:, 0]
Omega_nu = m_ncdm * 3 / (93.14 * h**2)  # per species, deg=3
Omega_m = x[:, 1]
f_nu = Omega_nu / Omega_m  # neutrino fraction of matter

print(f"Omega_nu range: [{Omega_nu.min():.6f}, {Omega_nu.max():.6f}]")
print(f"f_nu range: [{f_nu.min():.6f}, {f_nu.max():.6f}]")
print(f"N_eff_approx range: [{N_eff_approx.min():.4f}, {N_eff_approx.max():.4f}]")

# Correlation with spectrum properties
y_mean_per_sample = np.mean(y, axis=1)
y_std_per_sample = np.std(y, axis=1)
y_log_mean = np.mean(np.log(y), axis=1)

print(f"\nCorrelation of derived quantities with spectrum properties:")
for name, val in [('N_eff_approx', N_eff_approx), ('f_nu', f_nu), ('Omega_nu', Omega_nu)]:
    r1 = np.corrcoef(val, y_mean_per_sample)[0, 1]
    r2 = np.corrcoef(val, y_std_per_sample)[0, 1]
    r3 = np.corrcoef(val, y_log_mean)[0, 1]
    print(f"  {name:>15s}: r(mean)={r1:+.4f}, r(std)={r2:+.4f}, r(log_mean)={r3:+.4f}")

# Compare with raw parameters
for ip in range(8):
    r1 = np.corrcoef(x[:, ip], y_mean_per_sample)[0, 1]
    r3 = np.corrcoef(x[:, ip], y_log_mean)[0, 1]
    print(f"  {PARAM_NAMES[ip]:>15s}: r(mean)={r1:+.4f}, r(log_mean)={r3:+.4f}")

# ── Figure: N_eff_approx and f_nu vs spectrum properties ─────
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

sc = axs[0, 0].scatter(N_eff_approx, y_mean_per_sample, s=0.3, alpha=0.2, c=m_ncdm, cmap='viridis')
axs[0, 0].set_xlabel('N_eff_approx')
axs[0, 0].set_ylabel('mean(ratio)')
axs[0, 0].set_title('Mean ratio vs N_eff')
plt.colorbar(sc, ax=axs[0, 0], label='m_ncdm')

sc = axs[0, 1].scatter(f_nu, y_mean_per_sample, s=0.3, alpha=0.2, c=N_ur, cmap='viridis')
axs[0, 1].set_xlabel('f_nu = Omega_nu / Omega_m')
axs[0, 1].set_ylabel('mean(ratio)')
axs[0, 1].set_title('Mean ratio vs f_nu')
plt.colorbar(sc, ax=axs[0, 1], label='N_ur')

sc = axs[0, 2].scatter(N_ur, m_ncdm, c=y_mean_per_sample, s=0.3, alpha=0.3, cmap='RdYlBu_r')
axs[0, 2].set_xlabel('N_ur')
axs[0, 2].set_ylabel('m_ncdm')
axs[0, 2].set_title('Parameter space coloured by mean(ratio)')
plt.colorbar(sc, ax=axs[0, 2], label='mean ratio')

# Dynamic range and shape variation
sc = axs[1, 0].scatter(N_ur, m_ncdm, c=np.log10(y_std_per_sample), s=0.3, alpha=0.3, cmap='RdYlBu_r')
axs[1, 0].set_xlabel('N_ur')
axs[1, 0].set_ylabel('m_ncdm')
axs[1, 0].set_title('coloured by log10(within-spectrum std)')
plt.colorbar(sc, ax=axs[1, 0])

# The key: variance of log(y) per sample → this is what the loss "sees"
log_y = np.log(y)
log_y_var = np.var(log_y, axis=1)
sc = axs[1, 1].scatter(N_ur, m_ncdm, c=log_y_var, s=0.3, alpha=0.3, cmap='RdYlBu_r')
axs[1, 1].set_xlabel('N_ur')
axs[1, 1].set_ylabel('m_ncdm')
axs[1, 1].set_title('coloured by var(log(ratio)) per sample')
plt.colorbar(sc, ax=axs[1, 1])

# Spectrum "shape distance" from reference (mean spectrum)
y_ref = np.mean(y, axis=0)
shape_dist = np.sqrt(np.mean((y / y_ref[np.newaxis, :] - 1)**2, axis=1))
sc = axs[1, 2].scatter(N_ur, m_ncdm, c=np.log10(shape_dist), s=0.3, alpha=0.3, cmap='RdYlBu_r')
axs[1, 2].set_xlabel('N_ur')
axs[1, 2].set_ylabel('m_ncdm')
axs[1, 2].set_title('coloured by log10(shape distance from mean)')
plt.colorbar(sc, ax=axs[1, 2])

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'diag_regime_physics.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Analysis 5: Compare the function complexity across datasets ──
print("\n" + "="*60)
print("Analysis 5: Function complexity comparison")
print("="*60)

for tag in ['thin', 'std', 'ext']:
    path = f'/ceph/hpc/data/s25r06-05-users/lcdm_nu/sample/cl_100_{tag}.fits'
    ds_t = io.FitsFile(path)
    x_t = ds_t.get_data('x_data')
    y_t = ds_t.get_data(SPECTRUM)
    valid = np.all(np.isfinite(y_t), axis=1) & np.all(y_t > 0, axis=1)
    x_t = x_t[valid]
    y_t = y_t[valid]
    
    # Compute the "noise" level: how much do nearby parameter points differ in spectra?
    x_t_norm = (x_t - x_t.mean(axis=0)) / x_t.std(axis=0)
    tree = KDTree(x_t_norm)
    n_test = 2000
    idxs = np.random.choice(len(x_t), n_test, replace=False)
    nn_diffs = []
    for idx in idxs:
        dists, nn_idx = tree.query(x_t_norm[idx], k=6)
        nn_idx = nn_idx[1:]
        rel_diffs = y_t[nn_idx] / y_t[idx] - 1
        nn_diffs.append(np.sqrt(np.mean(rel_diffs**2)))
    
    nn_diffs = np.array(nn_diffs)
    print(f"\n{tag}:")
    print(f"  Parameter range (N_ur): [{x_t[:, 6].min():.4f}, {x_t[:, 6].max():.4f}]")
    print(f"  Parameter range (m_ncdm): [{x_t[:, 7].min():.6f}, {x_t[:, 7].max():.6f}]")
    print(f"  NN relative diff: median={np.median(nn_diffs):.4e}, P90={np.percentile(nn_diffs, 90):.4e}, P99={np.percentile(nn_diffs, 99):.4e}")
    print(f"  Spectrum range: [{y_t.min():.4e}, {y_t.max():.4e}]")
    print(f"  Log variance per feature: median={np.median(np.var(np.log(y_t), axis=0)):.4e}")
    print(f"  Max/Min of mean spectrum: {np.max(np.mean(y_t, axis=0))/np.min(np.mean(y_t, axis=0)):.4f}")

print(f"\nAll figures saved to {SAVE_DIR}")
