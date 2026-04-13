"""Dataset diagnostics: roughness, nearest-neighbour consistency, emulator error,
cross-dataset comparison, bimodality analysis, and PCA truncation check."""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import yaml
from scipy.spatial import KDTree
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
import emu_like.io as io
from emu_like.ffnn_emu import FFNNEmu


class EmuData(object):

    def __init__(self, path):
        self.name = os.path.basename(path)
        self.emu = FFNNEmu()
        self.emu.load(path, still_training=False)
        self.abs_diff = None
        self.rel_diff = None
        self.mean_abs_diff = None
        self.mean_rel_diff = None

    def _scale_x(self, x):
        return self.emu.x_scaler.transform(x)

    def _scale_y(self, y):
        return self.emu.y_scaler.transform(y)

    def _inverse_scale_x(self, x):
        return self.emu.x_scaler.inverse_transform(x)

    def _inverse_scale_y(self, y):
        return self.emu.y_scaler.inverse_transform(y)

    def _pca_x(self, x):
        return self.emu.x_pca.transform(x)

    def _pca_y(self, y):
        return self.emu.y_pca.transform(y)

    def _inverse_pca_x(self, x):
        return self.emu.x_pca.inverse_transform(x)

    def _inverse_pca_y(self, y):
        return self.emu.y_pca.inverse_transform(y)

    def get_y_emu(self, x, want_scaling=False, want_pca=False, select_pca_modes=None, timeit=False):
        if timeit:
            start_all = time.time()

        has_pca = self.emu.y_pca is not None
        has_scaling = self.emu.y_scaler is not None
        want_pca_selection = select_pca_modes is not None
        n_samples = x.shape[0]

        if want_scaling and not has_scaling:
            raise ValueError('Cannot keep scaled units: y is already unscaled.')
        if want_pca and not has_pca:
            raise ValueError('Cannot request PCA output: emulator has no PCA transform.')
        if want_pca_selection and not has_pca:
            raise ValueError('Cannot select PCA modes: emulator has no PCA transform.')
        if want_pca and has_scaling and not want_scaling:
            raise ValueError('Cannot stay in PCA space while undoing scaling when PCA was built on scaled data.')

        if x.ndim == 1:
            x = x[np.newaxis, :]
        if self.emu.x_scaler is not None:
            x = self._scale_x(x)
        if self.emu.x_pca is not None:
            x = self._pca_x(x)

        if timeit:
            start_emu = time.time()
        result = self.emu.model(x, training=False).numpy()
        if timeit:
            stop_emu = time.time()

        if want_pca_selection:
            tmp = np.zeros_like(result)
            tmp[:, select_pca_modes] = result[:, select_pca_modes]
            result = tmp

        if has_pca and not want_pca:
            result = self._inverse_pca_y(result)
        if has_scaling and not want_scaling:
            result = self._inverse_scale_y(result)

        if want_pca and want_pca_selection:
            result = result[:, select_pca_modes]
        if len(result) == 1:
            result = result[0]

        if timeit:
            stop_all = time.time()
            self.time_emu = (stop_emu - start_emu) / n_samples
            self.time_all = (stop_all - start_all) / n_samples

        return result

    def get_y_data(self, y, want_scaling=False, want_pca=False, select_pca_modes=None):
        has_pca = self.emu.y_pca is not None
        has_scaling = self.emu.y_scaler is not None
        want_pca_selection = select_pca_modes is not None

        if want_scaling and not has_scaling:
            raise ValueError('Cannot use scaled units: emulator has no scaling transform.')
        if want_pca and not has_pca:
            raise ValueError('Cannot request PCA output: emulator has no PCA transform.')
        if want_pca_selection and not has_pca:
            raise ValueError('Cannot select PCA modes: emulator has no PCA transform.')
        if want_pca and has_scaling and not want_scaling:
            raise ValueError('Cannot stay in PCA space while undoing scaling when PCA was built on scaled data.')

        result = y
        if want_scaling or (want_pca_selection and has_scaling):
            result = self._scale_y(result)
        if want_pca or want_pca_selection:
            result = self._pca_y(result)
        if want_pca_selection:
            tmp = np.zeros_like(result)
            tmp[:, select_pca_modes] = result[:, select_pca_modes]
            result = tmp
        if want_pca_selection and not want_pca:
            result = self._inverse_pca_y(result)
        if want_pca_selection and has_scaling and not want_scaling:
            result = self._inverse_scale_y(result)
        if want_pca and want_pca_selection:
            result = result[:, select_pca_modes]
        if len(result) == 1:
            result = result[0]
        return result

    def get_abs_diff(self, x_emu, y_data, want_scaling=False, want_pca=False,
                     select_pca_modes_emu=None, select_pca_modes_data=None, time_emu=False):
        if want_pca and select_pca_modes_emu != select_pca_modes_data:
            raise ValueError('Inconsistent dimensions for absolute difference.')
        y_emu = self.get_y_emu(x_emu, want_scaling=want_scaling, want_pca=want_pca,
                               select_pca_modes=select_pca_modes_emu, timeit=time_emu)
        y_data = self.get_y_data(y_data, want_scaling=want_scaling, want_pca=want_pca,
                                 select_pca_modes=select_pca_modes_data)
        self.abs_diff = y_emu - y_data
        return self.abs_diff

    def get_rel_diff(self, x_emu, y_data, want_scaling=False, want_pca=False,
                     select_pca_modes_emu=None, select_pca_modes_data=None, time_emu=False):
        if want_pca and select_pca_modes_emu != select_pca_modes_data:
            raise ValueError('Inconsistent dimensions for relative difference.')
        y_emu = self.get_y_emu(x_emu, want_scaling=want_scaling, want_pca=want_pca,
                               select_pca_modes=select_pca_modes_emu, timeit=time_emu)
        y_data = self.get_y_data(y_data, want_scaling=want_scaling, want_pca=want_pca,
                                 select_pca_modes=select_pca_modes_data)
        self.rel_diff = y_emu / y_data - 1.
        return self.rel_diff

    def get_mean_abs_diff(self, x_emu=None, y_data=None, want_scaling=False, want_pca=False,
                          select_pca_modes_emu=None, select_pca_modes_data=None, time_emu=False):
        if x_emu is not None and y_data is not None:
            self.abs_diff = self.get_abs_diff(x_emu, y_data, want_scaling=want_scaling,
                                              want_pca=want_pca, select_pca_modes_emu=select_pca_modes_emu,
                                              select_pca_modes_data=select_pca_modes_data, time_emu=time_emu)
        self.mean_abs_diff = np.sqrt(np.mean(self.abs_diff**2., axis=1))
        return self.mean_abs_diff

    def get_mean_rel_diff(self, x_emu=None, y_data=None, want_scaling=False, want_pca=False,
                          select_pca_modes_emu=None, select_pca_modes_data=None, time_emu=False):
        if x_emu is not None and y_data is not None:
            self.rel_diff = self.get_rel_diff(x_emu, y_data, want_scaling=want_scaling,
                                              want_pca=want_pca, select_pca_modes_emu=select_pca_modes_emu,
                                              select_pca_modes_data=select_pca_modes_data, time_emu=time_emu)
        self.mean_rel_diff = np.sqrt(np.mean(self.rel_diff**2., axis=1))
        return self.mean_rel_diff

    def get_sorting_idxs_abs(self):
        if self.mean_abs_diff is None:
            raise Exception('Calculate mean_abs_diff first')
        return self.mean_abs_diff.argsort()[::-1]

    def get_sorting_idxs_rel(self):
        if self.mean_rel_diff is None:
            raise Exception('Calculate mean_rel_diff first')
        return self.mean_rel_diff.argsort()[::-1]


# ============================================================
# Per-dataset analysis (roughness, NN consistency, emu error)
# ============================================================

def analyse_single_dataset(label, ds_path, spectrum, emu_root, save_dir):
    """Run all diagnostics for a single dataset (std / thin / ext)."""
    ds = io.FitsFile(ds_path)
    x = ds.get_data('x_data')
    y = ds.get_data(spectrum)
    print(f"\n{'='*60}")
    print(f"Dataset: {label}  ({ds_path})")
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    # --- Step 1: Roughness ---
    dy = np.diff(y, n=2, axis=1)
    roughness = np.std(dy, axis=1)
    amp = np.max(np.abs(y), axis=1)
    rel_roughness = roughness / amp

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].hist(np.log10(roughness), bins=50, log=True)
    axs[0].set_xlabel('log10(roughness)')
    axs[0].set_title(f'Absolute roughness of raw spectra ({label})')
    axs[1].hist(np.log10(rel_roughness), bins=50, log=True)
    axs[1].set_xlabel('log10(relative roughness)')
    axs[1].set_title(f'Relative roughness ({label})')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'diag_{label}_step1_roughness.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    threshold = np.percentile(rel_roughness, 99.9)
    outlier_mask = rel_roughness > threshold
    print(f"Roughness outliers (top 0.1%): {np.sum(outlier_mask)} / {len(rel_roughness)}")
    print(f"Threshold: {threshold:.6e}")

    # --- Step 2: Roughest spectra vs neighbours ---
    x_mean = np.mean(x, axis=0)
    x_scale = np.std(x, axis=0)
    x_norm = (x - x_mean) / x_scale
    tree = KDTree(x_norm)

    n_worst = 5
    worst_idxs = np.argsort(rel_roughness)[-n_worst:][::-1]
    k_neighbours = 5

    fig, axs = plt.subplots(n_worst, 2, figsize=(16, 4 * n_worst))
    for i, idx in enumerate(worst_idxs):
        dists, nn_idxs = tree.query(x_norm[idx], k=k_neighbours + 1)
        nn_idxs = nn_idxs[1:]
        for nn in nn_idxs:
            axs[i, 0].plot(y[nn], color='gray', alpha=0.5, lw=0.8)
        axs[i, 0].plot(y[idx], color='red', lw=1.5, label=f'idx={idx}')
        axs[i, 0].set_title(f'Sample {idx} (roughness rank {i + 1}) vs {k_neighbours} NN ({label})')
        axs[i, 0].legend()
        axs[i, 0].set_ylabel(spectrum)
        nn_mean = np.mean(y[nn_idxs], axis=0)
        rel_diff_to_nn = y[idx] / nn_mean - 1.
        axs[i, 1].plot(rel_diff_to_nn)
        axs[i, 1].set_title('Relative diff to neighbour mean')
        axs[i, 1].set_ylabel('rel diff')
        axs[i, 1].axhline(0, c='k', lw=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'diag_{label}_step2_roughest_vs_nn.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Step 3: Emulator error vs roughness ---
    emudata = EmuData(emu_root)
    mean_rel = emudata.get_mean_rel_diff(
        x_emu=x, y_data=y, want_scaling=False, want_pca=False, time_emu=False)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].scatter(np.log10(rel_roughness), np.log10(mean_rel), s=1, alpha=0.3)
    axs[0].set_xlabel('log10(relative roughness)')
    axs[0].set_ylabel('log10(mean rel diff)')
    axs[0].set_title(f'Emulator error vs spectrum roughness ({label})')

    n_check = 100
    worst_emu = set(np.argsort(mean_rel)[-n_check:])
    worst_rough = set(np.argsort(rel_roughness)[-n_check:])
    overlap = worst_emu & worst_rough
    print(f"Top {n_check} worst emulator errors: overlap with top {n_check} roughest = {len(overlap)}")

    median_roughness = np.median(rel_roughness)
    mask_smooth = rel_roughness <= median_roughness
    mask_rough = rel_roughness > median_roughness
    axs[1].hist(np.log10(mean_rel[mask_smooth]), bins=30, alpha=0.6, label='smoother half', log=True)
    axs[1].hist(np.log10(mean_rel[mask_rough]), bins=30, alpha=0.6, label='rougher half', log=True)
    axs[1].set_xlabel('log10(mean rel diff)')
    axs[1].set_title(f'Emulator error by roughness group ({label})')
    axs[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'diag_{label}_step3_emu_vs_roughness.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Step 4: Nearest-neighbour consistency (data-only) ---
    k_nn = 5
    dists_all, nn_all = tree.query(x_norm, k=k_nn + 1)
    nn_all = nn_all[:, 1:]
    nn_mean_spectra = np.mean(y[nn_all], axis=1)
    rel_dev_nn = np.sqrt(np.mean((y / nn_mean_spectra - 1.)**2, axis=1))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].hist(np.log10(rel_dev_nn), bins=50, log=True)
    axs[0].set_xlabel('log10(RMS rel diff to nearest neighbours)')
    axs[0].set_title(f'Consistency with {k_nn} NN ({label}, data only)')

    axs[1].scatter(np.log10(rel_dev_nn), np.log10(mean_rel), s=1, alpha=0.3)
    axs[1].set_xlabel('log10(rel diff to neighbours)')
    axs[1].set_ylabel('log10(emulator rel error)')
    axs[1].set_title(f'Neighbour inconsistency vs emulator error ({label})')
    axs[1].plot([-3, 1], [-3, 1], 'r--', lw=0.8, label='1:1')
    axs[1].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'diag_{label}_step4_nn_consistency.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    threshold_nn = np.percentile(rel_dev_nn, 99)
    suspicious = np.where(rel_dev_nn > threshold_nn)[0]
    print(f"Samples deviating >99th pctile from neighbours: {len(suspicious)}")
    print(f"Threshold: {threshold_nn:.4e}")
    print(f"Worst 10 indices: {np.argsort(rel_dev_nn)[-10:][::-1]}")

    # --- Step 5: Most suspicious samples ---
    n_show = 5
    worst_nn_idxs = np.argsort(rel_dev_nn)[-n_show:][::-1]
    fig, axs = plt.subplots(n_show, 2, figsize=(16, 4 * n_show))
    for i, idx in enumerate(worst_nn_idxs):
        nn_idxs = nn_all[idx]
        for nn in nn_idxs:
            axs[i, 0].plot(y[nn], color='gray', alpha=0.5, lw=0.8)
        axs[i, 0].plot(y[idx], color='red', lw=1.5, label=f'idx={idx}')
        axs[i, 0].set_title(f'Sample {idx} (nn-deviation rank {i + 1}, {label})')
        axs[i, 0].legend()
        axs[i, 0].set_ylabel(spectrum)
        x_params = x[idx]
        x_nn_params = x[nn_idxs]
        x_labels = [f'x{j}' for j in range(x.shape[1])]
        x_nn_mean = np.mean(x_nn_params, axis=0)
        x_nn_std = np.std(x_nn_params, axis=0)
        x_nn_std[x_nn_std == 0] = 1.
        deviation = (x_params - x_nn_mean) / x_nn_std
        axs[i, 1].bar(range(len(deviation)), deviation)
        axs[i, 1].set_xticks(range(len(x_labels)))
        axs[i, 1].set_xticklabels(x_labels, rotation=45)
        axs[i, 1].set_title(f'Parameter deviation from neighbours ({label})')
        axs[i, 1].set_ylabel('(x - nn_mean) / nn_std')
        axs[i, 1].axhline(0, c='k', lw=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'diag_{label}_step5_suspicious.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Step 6: Worst points table ---
    x_names = emudata.emu.x_names
    n_top = 10
    worst_emu_idxs = np.argsort(mean_rel)[-n_top:][::-1]
    print(f"\nParameter names: {x_names}")
    print(f"{'Rank':<5} {'Index':<8} {'Mean rel err':<14} {'NN dev':<14} {'Roughness':<14} | "
          f"{('  '.join([f'{n:>12s}' for n in x_names]))}")
    print("-" * (55 + 14 * len(x_names)))
    for rank, idx in enumerate(worst_emu_idxs):
        vals = '  '.join([f'{v:12.5e}' for v in x[idx]])
        print(f"{rank + 1:<5} {idx:<8} {mean_rel[idx]:<14.4e} {rel_dev_nn[idx]:<14.4e} "
              f"{rel_roughness[idx]:<14.4e} | {vals}")

    return {
        'mean_rel': mean_rel,
        'roughness': rel_roughness,
        'nn_dev': rel_dev_nn,
        'x': x,
        'y': y,
        'emudata': emudata,
    }


# ============================================================
# Cross-dataset comparison
# ============================================================

def cross_dataset_comparison(datasets_results, save_dir):
    """Compare metrics across std / thin / ext datasets."""
    datasets = {k: v for k, v in datasets_results.items()}

    print(f"\n{'='*60}")
    print("Cross-dataset comparison")

    print("\n=== Data quality ===")
    print(f"{'Dataset':<8} {'N samples':<10} {'NaN in y':<10} {'NaN in emu err':<16} "
          f"{'NaN in roughness':<18} {'NaN in nn_dev'}")
    for name, d in datasets.items():
        print(f"{name:<8} {len(d['mean_rel']):<10} {np.isnan(d['y']).sum():<10} "
              f"{np.isnan(d['mean_rel']).sum():<16} {np.isnan(d['roughness']).sum():<18} "
              f"{np.isnan(d['nn_dev']).sum()}")

    print("\n=== Emulator error quantiles ===")
    print(f"{'Pctile':<8} {'thin':<14} {'std':<14} {'ext':<14} {'ext/thin':<10} {'std/thin':<10}")
    for pct in [50, 90, 95, 99]:
        vals = {name: np.nanpercentile(d['mean_rel'], pct) for name, d in datasets.items()}
        print(f"P{pct:<7} {vals['thin']:<14.4e} {vals['std']:<14.4e} {vals['ext']:<14.4e} "
              f"{vals['ext'] / vals['thin']:<10.2f} {vals['std'] / vals['thin']:<10.2f}")

    print("\n=== Data roughness quantiles (relative) ===")
    print(f"{'Pctile':<8} {'thin':<14} {'std':<14} {'ext':<14} {'ext/thin':<10}")
    for pct in [50, 90, 99]:
        vals = {name: np.nanpercentile(d['roughness'], pct) for name, d in datasets.items()}
        print(f"P{pct:<7} {vals['thin']:<14.4e} {vals['std']:<14.4e} {vals['ext']:<14.4e} "
              f"{vals['ext'] / vals['thin']:<10.2f}")

    print("\n=== NN deviation quantiles ===")
    print(f"{'Pctile':<8} {'thin':<14} {'std':<14} {'ext':<14} {'ext/thin':<10}")
    for pct in [50, 90, 99]:
        vals = {name: np.nanpercentile(d['nn_dev'], pct) for name, d in datasets.items()}
        print(f"P{pct:<7} {vals['thin']:<14.4e} {vals['std']:<14.4e} {vals['ext']:<14.4e} "
              f"{vals['ext'] / vals['thin']:<10.2f}")

    print("\n=== Roughness vs emulator error overlap (top 100) ===")
    for name, d in datasets.items():
        mr = d['mean_rel']
        rr = d['roughness']
        n_check = 100
        overlap = len(set(np.argsort(mr)[-n_check:]) & set(np.argsort(rr)[-n_check:]))
        print(f"  {name}: {overlap}/100")

    # --- Side-by-side comparison plots ---
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for name, d in datasets.items():
        mr = d['mean_rel']
        mr = mr[np.isfinite(mr)]
        axs[0].hist(np.log10(mr), bins=60, alpha=0.5, label=name, density=True)
    axs[0].set_xlabel('log10(mean rel diff)')
    axs[0].set_title('Emulator error distribution')
    axs[0].legend()

    for name, d in datasets.items():
        rr = d['roughness']
        rr = rr[np.isfinite(rr)]
        axs[1].hist(np.log10(rr), bins=60, alpha=0.5, label=name, density=True)
    axs[1].set_xlabel('log10(relative roughness)')
    axs[1].set_title('Data roughness distribution')
    axs[1].legend()

    for name, d in datasets.items():
        nn = d['nn_dev']
        nn = nn[np.isfinite(nn)]
        axs[2].hist(np.log10(nn), bins=60, alpha=0.5, label=name, density=True)
    axs[2].set_xlabel('log10(NN deviation)')
    axs[2].set_title('Nearest-neighbour consistency')
    axs[2].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'diag_cross_dataset_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Bimodality analysis (on std dataset)
# ============================================================

def bimodality_analysis(std_results, spectrum, save_dir):
    """Analyse bimodality in the std dataset emulator error."""
    mean_rel = std_results['mean_rel']
    x_std = std_results['x']
    y_std = std_results['y']
    emudata_check = std_results['emudata']
    x_names = emudata_check.emu.x_names

    err_threshold = 0.1
    good_mask = mean_rel < err_threshold
    bad_mask = mean_rel >= err_threshold
    labels = bad_mask.astype(int)
    print(f"\n{'='*60}")
    print(f"Bimodality analysis (std dataset)")
    print(f"Good population (err < {err_threshold}): {good_mask.sum()} samples")
    print(f"Bad population  (err >= {err_threshold}): {bad_mask.sum()} samples")

    # --- Parameter distributions for good vs bad ---
    n_params = x_std.shape[1]
    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    axs = axs.ravel()
    for ip in range(n_params):
        axs[ip].hist(x_std[good_mask, ip], bins=40, alpha=0.5, label='good (<10%)', density=True)
        axs[ip].hist(x_std[bad_mask, ip], bins=40, alpha=0.5, label='bad (>=10%)', density=True, color='red')
        axs[ip].set_xlabel(x_names[ip])
        axs[ip].set_title(x_names[ip])
        axs[ip].legend(fontsize=8)
    for ip in range(n_params, len(axs)):
        axs[ip].set_visible(False)
    fig.suptitle('Std dataset: parameter distributions for good vs bad emulator accuracy', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'diag_bimodality_params.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Single-parameter separation power ---
    print("\nSingle-parameter separation power (AUC of bad vs good):")
    aucs = []
    for ip in range(n_params):
        auc = roc_auc_score(labels, x_std[:, ip])
        aucs.append(auc)
        print(f"  {x_names[ip]:>20s}:  AUC = {auc:.4f}  (1.0 = perfect, 0.5 = useless)")

    best_ip = np.argmax([abs(a - 0.5) for a in aucs])
    best_name = x_names[best_ip]
    aucs_rank = np.argsort([abs(a - 0.5) for a in aucs])[::-1]
    best2_ip = aucs_rank[1]
    best2_name = x_names[best2_ip]
    print(f"\nBest single separator: {best_name} (AUC={aucs[best_ip]:.4f})")

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    axs[0].hist(np.log10(mean_rel[good_mask]), bins=40, alpha=0.6, label='good', density=True)
    axs[0].hist(np.log10(mean_rel[bad_mask]), bins=40, alpha=0.6, label='bad', density=True, color='red')
    axs[0].axvline(np.log10(err_threshold), c='k', ls='--', label=f'threshold={err_threshold}')
    axs[0].set_xlabel('log10(mean rel error)')
    axs[0].set_title('Bimodal error split')
    axs[0].legend()

    sc = axs[1].scatter(x_std[:, best_ip], np.log10(mean_rel), c=np.log10(mean_rel),
                         s=0.5, alpha=0.3, cmap='RdYlBu_r', vmin=-2.5, vmax=0.5)
    axs[1].set_xlabel(best_name)
    axs[1].set_ylabel('log10(mean rel error)')
    axs[1].set_title(f'Error vs {best_name}')
    axs[1].axhline(np.log10(err_threshold), c='k', ls='--', lw=0.8)
    plt.colorbar(sc, ax=axs[1], label='log10(error)')

    sc2 = axs[2].scatter(x_std[:, best_ip], x_std[:, best2_ip], c=np.log10(mean_rel),
                          s=0.5, alpha=0.3, cmap='RdYlBu_r', vmin=-2.5, vmax=0.5)
    axs[2].set_xlabel(best_name)
    axs[2].set_ylabel(best2_name)
    axs[2].set_title(f'{best_name} vs {best2_name} coloured by error')
    plt.colorbar(sc2, ax=axs[2], label='log10(error)')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'diag_bimodality_separators.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Diagnosing the bimodality ---
    good_idxs = np.where(good_mask)[0][:5]
    bad_idxs = np.argsort(mean_rel)[-5:][::-1]

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))

    for idx in good_idxs:
        axs[0, 0].plot(y_std[idx], alpha=0.7, lw=0.8)
    axs[0, 0].set_title('Good population: sample spectra')
    axs[0, 0].set_ylabel(spectrum)
    axs[0, 0].set_yscale('log')
    for idx in bad_idxs:
        axs[0, 1].plot(y_std[idx], alpha=0.7, lw=0.8)
    axs[0, 1].set_title('Bad population: sample spectra')
    axs[0, 1].set_ylabel(spectrum)
    axs[0, 1].set_yscale('log')

    rel_diff_good = emudata_check.rel_diff[good_mask]
    rel_diff_bad = emudata_check.rel_diff[bad_mask]
    median_reldiff_good = np.median(np.abs(rel_diff_good), axis=0)
    median_reldiff_bad = np.median(np.abs(rel_diff_bad), axis=0)
    p90_reldiff_good = np.percentile(np.abs(rel_diff_good), 90, axis=0)
    p90_reldiff_bad = np.percentile(np.abs(rel_diff_bad), 90, axis=0)

    axs[1, 0].plot(median_reldiff_good, label='good median')
    axs[1, 0].plot(median_reldiff_bad, label='bad median')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xlabel('ell index')
    axs[1, 0].set_ylabel('|relative error|')
    axs[1, 0].set_title('Per-ell median |rel error|')
    axs[1, 0].legend()

    axs[1, 1].plot(p90_reldiff_good, label='good P90')
    axs[1, 1].plot(p90_reldiff_bad, label='bad P90')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlabel('ell index')
    axs[1, 1].set_ylabel('|relative error|')
    axs[1, 1].set_title('Per-ell P90 |rel error|')
    axs[1, 1].legend()

    min_y_std = np.min(np.abs(y_std), axis=1)
    mean_y_std = np.mean(np.abs(y_std), axis=1)
    max_y_std = np.max(np.abs(y_std), axis=1)
    dyn_range = max_y_std / min_y_std

    axs[2, 0].scatter(np.log10(min_y_std[good_mask]), np.log10(mean_rel[good_mask]),
                       s=0.5, alpha=0.2, label='good')
    axs[2, 0].scatter(np.log10(min_y_std[bad_mask]), np.log10(mean_rel[bad_mask]),
                       s=0.5, alpha=0.2, label='bad', c='red')
    axs[2, 0].set_xlabel('log10(min |spectrum| across ell)')
    axs[2, 0].set_ylabel('log10(mean rel error)')
    axs[2, 0].set_title('Error vs minimum spectrum value')
    axs[2, 0].legend(markerscale=5)

    axs[2, 1].scatter(np.log10(dyn_range[good_mask]), np.log10(mean_rel[good_mask]),
                       s=0.5, alpha=0.2, label='good')
    axs[2, 1].scatter(np.log10(dyn_range[bad_mask]), np.log10(mean_rel[bad_mask]),
                       s=0.5, alpha=0.2, label='bad', c='red')
    axs[2, 1].set_xlabel('log10(dynamic range = max/min)')
    axs[2, 1].set_ylabel('log10(mean rel error)')
    axs[2, 1].set_title('Error vs spectrum dynamic range')
    axs[2, 1].legend(markerscale=5)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'diag_bimodality_diagnosis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    min_auc = roc_auc_score(labels, min_y_std)
    dyn_auc = roc_auc_score(labels, dyn_range)
    mean_auc = roc_auc_score(labels, mean_y_std)
    print(f"Separation power of spectral features:")
    print(f"  min(|y|)       AUC = {min_auc:.4f}")
    print(f"  mean(|y|)      AUC = {mean_auc:.4f}")
    print(f"  dynamic range  AUC = {dyn_auc:.4f}")

    # --- PCA truncation check ---
    y_pca = emudata_check.emu.y_pca
    y_scaler = emudata_check.emu.y_scaler

    if y_scaler is not None:
        y_scaled = y_scaler.transform(y_std)
    else:
        y_scaled = y_std.copy()

    if y_pca is not None:
        y_pca_proj = y_pca.transform(y_scaled)
        y_reconstructed_scaled = y_pca.inverse_transform(y_pca_proj)
        n_pca = y_pca.n_components
        print(f"\nPCA components: {n_pca}")
        if hasattr(y_pca, 'explained_variance_ratio_'):
            print(f"Total explained variance: {np.sum(y_pca.explained_variance_ratio_) * 100:.4f}%")
        elif hasattr(y_pca, 'singular_values_'):
            sv = y_pca.singular_values_
            print(f"Singular values range: {sv[0]:.2e} to {sv[-1]:.2e}, ratio = {sv[0] / sv[-1]:.1f}")
        print(f"PCA type: {type(y_pca)}")
    else:
        y_reconstructed_scaled = y_scaled
        print("\nNo PCA used")

    if y_scaler is not None:
        y_reconstructed = y_scaler.inverse_transform(y_reconstructed_scaled)
    else:
        y_reconstructed = y_reconstructed_scaled

    pca_rel_err = y_reconstructed / y_std - 1.
    mean_pca_rel_err = np.sqrt(np.mean(pca_rel_err**2, axis=1))

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    axs[0, 0].scatter(np.log10(mean_pca_rel_err), np.log10(mean_rel), s=0.5, alpha=0.3)
    axs[0, 0].plot([-5, 0], [-5, 0], 'r--', lw=0.8, label='1:1')
    axs[0, 0].set_xlabel('log10(PCA reconstruction error)')
    axs[0, 0].set_ylabel('log10(emulator error)')
    axs[0, 0].set_title('PCA recon error vs total emulator error')
    axs[0, 0].legend()

    axs[0, 1].hist(np.log10(mean_pca_rel_err[good_mask]), bins=40, alpha=0.6, label='good', density=True)
    axs[0, 1].hist(np.log10(mean_pca_rel_err[bad_mask]), bins=40, alpha=0.6, label='bad', density=True, color='red')
    axs[0, 1].set_xlabel('log10(PCA reconstruction error)')
    axs[0, 1].set_title('PCA recon error: good vs bad populations')
    axs[0, 1].legend()

    median_pca_good = np.median(np.abs(pca_rel_err[good_mask]), axis=0)
    median_pca_bad = np.median(np.abs(pca_rel_err[bad_mask]), axis=0)
    axs[1, 0].plot(median_pca_good, label='good median')
    axs[1, 0].plot(median_pca_bad, label='bad median')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xlabel('ell index')
    axs[1, 0].set_ylabel('|PCA recon rel error|')
    axs[1, 0].set_title('Per-ell PCA reconstruction error')
    axs[1, 0].legend()

    median_nn_good = np.median(np.abs(emudata_check.rel_diff[good_mask] - pca_rel_err[good_mask]), axis=0)
    median_nn_bad = np.median(np.abs(emudata_check.rel_diff[bad_mask] - pca_rel_err[bad_mask]), axis=0)
    axs[1, 1].plot(median_nn_good, label='good median')
    axs[1, 1].plot(median_nn_bad, label='bad median')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlabel('ell index')
    axs[1, 1].set_ylabel('|NN residual|')
    axs[1, 1].set_title('Per-ell NN-only error (emulator - PCA)')
    axs[1, 1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'diag_bimodality_pca.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    pca_auc = roc_auc_score(labels, mean_pca_rel_err)
    print(f"\nPCA reconstruction error AUC: {pca_auc:.4f}")
    print(f"=> {'PCA truncation IS the separator!' if abs(pca_auc - 0.5) > 0.1 else 'PCA truncation is NOT the separator either.'}")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset diagnostics for emulator training inspection.')
    parser.add_argument('--emu-root', type=str, default='/ceph/hpc/data/s25r06-05-users/test/hub_f4_d2',
                        help='Path to the emulator root folder.')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save figures. Defaults to script directory.')
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    # Load params
    with open(os.path.join(args.emu_root, 'params.yaml')) as f:
        params = yaml.safe_load(f)
    spectrum = params['datasets']['name']
    dataset_paths = params['datasets']['paths']

    # Identify datasets by suffix
    path_map = {}
    for p in dataset_paths:
        tag = os.path.basename(p).replace('.fits', '').split('_')[-1]
        path_map[tag] = p

    # Run per-dataset analysis
    results = {}
    for label in ['std', 'thin', 'ext']:
        if label in path_map:
            results[label] = analyse_single_dataset(
                label, path_map[label], spectrum, args.emu_root, save_dir)

    # Cross-dataset comparison (requires all three)
    if all(k in results for k in ['std', 'thin', 'ext']):
        cross_dataset_comparison(results, save_dir)

    # Bimodality analysis on std
    if 'std' in results:
        bimodality_analysis(results['std'], spectrum, save_dir)

    print(f"\nAll figures saved to {save_dir}")
