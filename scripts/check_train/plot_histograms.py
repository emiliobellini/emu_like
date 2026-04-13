"""Plot histograms of emulator error and worst modes."""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import yaml
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


def show_summary(root, diff='rel', vlines=[0.01, 0.05, 0.1, 1.], save_dir='.'):
    with open(os.path.join(root, 'params.yaml')) as f:
        params = yaml.safe_load(f)

    spectrum = params['datasets']['name']
    dataset_paths = params['datasets']['paths']

    ranges = []
    for p in dataset_paths:
        dr = os.path.basename(p).replace('.fits', '').split('_')[-1]
        ranges.append((dr, p))

    n_ranges = len(ranges)
    fig, axs = plt.subplots(1, n_ranges, figsize=(6 * n_ranges, 4), squeeze=False)

    headers = ['range']
    headers += ['>{}%'.format(val) for val in vlines]
    headers += ['Time emu (s)', 'Time total (s)']
    headers += ['Epochs (best/tot)', 'Loss', 'Val Loss', 'LR']
    tab = []

    emudata = {}

    for ndr, (dr, path) in enumerate(ranges):

        emudata[dr] = EmuData(root)

        epochs = emudata[dr].emu.epochs
        losses = emudata[dr].emu.loss
        val_losses = emudata[dr].emu.val_loss
        learning_rates = emudata[dr].emu.learning_rate
        idx_best = np.where(np.array(val_losses) == np.min(val_losses))[0][0]

        if diff == 'rel':
            fun = emudata[dr].get_mean_rel_diff
        elif diff == 'abs':
            fun = emudata[dr].get_mean_abs_diff
        else:
            raise ValueError('Difference type not recognized!')

        dataset = io.FitsFile(path)
        x_data = dataset.get_data('x_data')
        y_data = dataset.get_data(spectrum)

        result = fun(
            x_emu=x_data, y_data=y_data,
            want_scaling=False, want_pca=False,
            select_pca_modes_emu=None, select_pca_modes_data=None,
            time_emu=True)

        tab_line = [dr]
        tab_line += ['{:.3f}%'.format(len(result[result > val / 100.]) / len(result) * 100.)
                     for val in vlines]
        tab_line += ['{:.1e}'.format(emudata[dr].time_emu)]
        tab_line += ['{:.1e}'.format(emudata[dr].time_all)]
        tab_line += ['{}/{}'.format(int(epochs[idx_best]), int(epochs[-1]))]
        tab_line += ['{:.2e}'.format(losses[idx_best])]
        tab_line += ['{:.2e}'.format(val_losses[idx_best])]
        tab_line += ['{:.2e}'.format(learning_rates[idx_best])]
        tab.append(tab_line)

        axs[0, ndr].hist(np.log10(result), log=True, bins=20)
        axs[0, ndr].set_title('{} - {}'.format(spectrum, dr), fontsize=18)
        axs[0, ndr].set_xlabel('Log10 sqrt mean diff squared', fontsize=14)
        for val in vlines:
            if diff == 'abs':
                maxy = np.max(np.abs(y_data))
            else:
                maxy = 1.
            axs[0, ndr].axvline(x=np.log10(maxy * val / 100.), c='r')
            emudata[dr].max_y_data = maxy

    fig.suptitle(spectrum, fontsize=20)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'histograms_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(tabulate(tab, headers=headers, tablefmt='orgtbl'))

    return emudata, spectrum, diff


def plot_worst_modes(emudata, spectrum, diff, n_modes_kept=3, vlines=[0.01, 0.05, 0.1, 1.], save_dir='.'):
    ranges = list(emudata.keys())
    n_ranges = len(ranges)
    fig, axs = plt.subplots(1, n_ranges, figsize=(6 * n_ranges, 4), squeeze=False)
    fig.suptitle('Worst modes - {}'.format(spectrum), fontsize=20, y=1.0)

    for ndr, dr in enumerate(ranges):
        if emudata[dr] is None:
            continue

        if diff == 'rel':
            idxs = emudata[dr].get_sorting_idxs_rel()[:n_modes_kept]
            diffs = emudata[dr].rel_diff[idxs]
        elif diff == 'abs':
            idxs = emudata[dr].get_sorting_idxs_abs()[:n_modes_kept]
            diffs = emudata[dr].abs_diff[idxs]
        else:
            raise ValueError('Difference type not recognized!')

        for val in vlines:
            axs[0, ndr].axhline(emudata[dr].max_y_data * val, c='k', lw=0.1)

        for ndiff, d in enumerate(diffs):
            axs[0, ndr].plot(np.abs(d) * 100., label='rank: {}, idx: {}'.format(ndiff + 1, idxs[ndiff]))
            axs[0, ndr].set_yscale('log')
            axs[0, ndr].set_title('{} - {}'.format(spectrum, dr), fontsize=18)

        if diff == 'rel':
            axs[0, 0].set_ylabel('rel diff [%]')
        elif diff == 'abs':
            axs[0, 0].set_ylabel('abs diff')

        axs[0, ndr].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'histograms_worst_modes.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot emulator error histograms and worst modes.')
    parser.add_argument(
        '--root',
        type=str,
        default='/ceph/hpc/data/s25r06-05-users/test/hub_f4_d2',
        help='Path to the emulator root folder.')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='/ceph/hpc/home/bellinie/emu_like/output',
        help='Directory to save figures. Defaults to script directory.')
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    emudata, spectrum, diff = show_summary(args.root, save_dir=save_dir)
    plot_worst_modes(emudata, spectrum, diff, n_modes_kept=3, save_dir=save_dir)
    print(f"\nFigures saved to {save_dir}")
