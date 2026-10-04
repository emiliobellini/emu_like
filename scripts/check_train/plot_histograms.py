"""Plot histograms of emulator error and worst modes."""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
import os
import time
import yaml
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import emu_like.io as io
from emu_like.datasets import Dataset, SobolevDataset
from emu_like.emu import Emulator
from emu_like.sobolev_ffnn_emu import SobolevFFNNEmu

if __package__:
    from ._plot_paths import plot_path, source_root
    from . import compare_pk_fk as growth_diagnostics
else:
    from _plot_paths import plot_path, source_root
    import compare_pk_fk as growth_diagnostics


class EmuData(object):

    def __init__(self, path):
        self.name = os.path.basename(path)
        with open(os.path.join(path, 'params.yaml')) as f:
            params = yaml.safe_load(f)
        self.emu = Emulator.choose_one(params['emulator']['name'])
        self.emu.load(path, still_training=False)
        self.is_sobolev = isinstance(self.emu, SobolevFFNNEmu)
        self.abs_diff = None
        self.rel_diff = None
        self.mean_abs_diff = None
        self.mean_rel_diff = None
        self.x_data = None
        self.y_data = None
        self.y_emu = None

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

    def get_y_emu(
            self,
            x,
            want_scaling=False,
            want_pca=False,
            select_pca_modes=None,
            timeit=False,
            use_growth=False):

        if timeit:
            start_all = time.time()

        if use_growth:
            if not self.is_sobolev:
                raise ValueError(
                    'Growth-rate evaluation requires a Sobolev emulator.')
            if want_scaling or want_pca or select_pca_modes is not None:
                raise ValueError(
                    'Sobolev growth-rate output is available only in physical '
                    'units, without PCA.')
            if x.ndim == 1:
                x = x[np.newaxis, :]
            n_samples = x.shape[0]
            if timeit:
                start_emu = time.time()
            # eval_fk uses a batch Jacobian.  Splitting here avoids allocating
            # a large Jacobian for an entire FITS range at once.
            result = np.concatenate([
                self.emu.eval_fk(x[start:start + self.emu.batch_size])
                for start in range(0, n_samples, self.emu.batch_size)
            ], axis=0)
            if timeit:
                stop_emu = time.time()
                stop_all = time.time()
                self.time_emu = (stop_emu - start_emu) / n_samples
                self.time_all = (stop_all - start_all) / n_samples
            if len(result) == 1:
                result = result[0]
            return result

        has_pca = self.emu.y_pca is not None
        has_scaling = self.emu.y_scaler is not None
        want_pca_selection = select_pca_modes is not None
        n_samples = x.shape[0]

        if want_scaling and not has_scaling:
            raise ValueError(
                'Cannot keep scaled units: y is already unscaled.')
        if want_pca and not has_pca:
            raise ValueError(
                'Cannot request PCA output: emulator has no PCA transform.')
        if want_pca_selection and not has_pca:
            raise ValueError(
                'Cannot select PCA modes: emulator has no PCA transform.')
        if want_pca and has_scaling and not want_scaling:
            raise ValueError(
                'Cannot stay in PCA space while undoing scaling when '
                'PCA was built on scaled data.')

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

    def get_y_data(
            self,
            y,
            want_scaling=False,
            want_pca=False,
            select_pca_modes=None,
            use_growth=False):
        if use_growth:
            if not self.is_sobolev:
                raise ValueError(
                    'Growth-rate data requires a Sobolev emulator.')
            if want_scaling or want_pca or select_pca_modes is not None:
                raise ValueError(
                    'Sobolev growth-rate data is available only in physical '
                    'units, without PCA.')
            return y[0] if len(y) == 1 else y

        has_pca = self.emu.y_pca is not None
        has_scaling = self.emu.y_scaler is not None
        want_pca_selection = select_pca_modes is not None

        if want_scaling and not has_scaling:
            raise ValueError(
                'Cannot use scaled units: emulator has no scaling transform.')
        if want_pca and not has_pca:
            raise ValueError(
                'Cannot request PCA output: emulator has no PCA transform.')
        if want_pca_selection and not has_pca:
            raise ValueError(
                'Cannot select PCA modes: emulator has no PCA transform.')
        if want_pca and has_scaling and not want_scaling:
            raise ValueError(
                'Cannot stay in PCA space while undoing scaling when PCA '
                'was built on scaled data.')

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

    def get_abs_diff(
            self,
            x_emu,
            y_data,
            want_scaling=False,
            want_pca=False,
            select_pca_modes_emu=None,
            select_pca_modes_data=None,
            time_emu=False,
            use_growth=False):

        if want_pca and select_pca_modes_emu != select_pca_modes_data:
            raise ValueError(
                'Inconsistent dimensions for absolute difference.')
        self.x_data = x_emu
        self.y_data = self.get_y_data(
            y_data, want_scaling=want_scaling, want_pca=want_pca,
            select_pca_modes=select_pca_modes_data, use_growth=use_growth)
        self.y_emu = self.get_y_emu(
            x_emu, want_scaling=want_scaling, want_pca=want_pca,
            select_pca_modes=select_pca_modes_emu, timeit=time_emu,
            use_growth=use_growth)
        self.y_emu = np.atleast_2d(self.y_emu)
        self.y_data = np.atleast_2d(self.y_data)
        self.abs_diff = self.y_emu - self.y_data
        return self.abs_diff

    def get_rel_diff(
            self,
            x_emu,
            y_data,
            want_scaling=False,
            want_pca=False,
            select_pca_modes_emu=None,
            select_pca_modes_data=None,
            time_emu=False,
            use_growth=False):

        if want_pca and select_pca_modes_emu != select_pca_modes_data:
            raise ValueError(
                'Inconsistent dimensions for relative difference.')
        self.x_data = x_emu
        self.y_data = self.get_y_data(
            y_data, want_scaling=want_scaling, want_pca=want_pca,
            select_pca_modes=select_pca_modes_data, use_growth=use_growth)
        self.y_emu = self.get_y_emu(
            x_emu, want_scaling=want_scaling, want_pca=want_pca,
            select_pca_modes=select_pca_modes_emu, timeit=time_emu,
            use_growth=use_growth)
        self.y_emu = np.atleast_2d(self.y_emu)
        self.y_data = np.atleast_2d(self.y_data)
        self.rel_diff = self.y_emu / self.y_data - 1.
        return self.rel_diff

    def get_mean_abs_diff(
            self,
            x_emu=None,
            y_data=None,
            want_scaling=False,
            want_pca=False,
            select_pca_modes_emu=None,
            select_pca_modes_data=None,
            time_emu=False,
            use_growth=False):

        if x_emu is not None and y_data is not None:
            self.abs_diff = self.get_abs_diff(
                x_emu,
                y_data,
                want_scaling=want_scaling,
                want_pca=want_pca,
                select_pca_modes_emu=select_pca_modes_emu,
                select_pca_modes_data=select_pca_modes_data,
                time_emu=time_emu, use_growth=use_growth)
        self.mean_abs_diff = np.sqrt(np.mean(self.abs_diff**2., axis=1))
        return self.mean_abs_diff

    def get_mean_rel_diff(
            self,
            x_emu=None,
            y_data=None,
            want_scaling=False,
            want_pca=False,
            select_pca_modes_emu=None,
            select_pca_modes_data=None,
            time_emu=False,
            use_growth=False):

        if x_emu is not None and y_data is not None:
            self.rel_diff = self.get_rel_diff(
                x_emu,
                y_data,
                want_scaling=want_scaling,
                want_pca=want_pca,
                select_pca_modes_emu=select_pca_modes_emu,
                select_pca_modes_data=select_pca_modes_data,
                time_emu=time_emu, use_growth=use_growth)
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

    def get_y_class(self, idxs):
        import hiclassy

        cosmo = hiclassy.HiClass()
        class_params = self.emu.y_model.class_params
        spectrum = self.emu.y_model.spectra[0]
        ref_spectrum_array = self.emu.y_model.y_ref[0][0]

        # 1) Infer the maximum redshift
        if spectrum.is_pk:
            # z_max = {'z_max_pk': self.emu.y_model._get_z_max()}
            z_array = self.emu.y_model.z_array
            k_range = self.emu.y_model.k_ranges[0]
            # Init output array
            y_class = np.zeros((len(idxs), len(k_range)))
        else:
            # z_max = {}
            ell_range = self.emu.y_model.ell_ranges[0]
            # Init output array
            y_class = np.zeros((len(idxs), len(ell_range)))

        # Iterate over indices
        for nx, x in enumerate(self.x_data[idxs]):

            # Fix parameters
            for key, val in zip(self.emu.x_names, x):
                class_params[key] = val
            cosmo.set(class_params)
            # Comppute class
            cosmo.compute()

            # Interpolate over z or not
            if spectrum.is_cl:
                y_class[nx, :] = spectrum.get(cosmo)/ref_spectrum_array
            else:
                y_ref = interp.make_splrep(
                    z_array, ref_spectrum_array.T, s=0)(class_params['z_pk']).T
                y_class[nx, :] = spectrum.get(
                    cosmo, z=class_params['z_pk'])/y_ref

        return y_class


def validation_rows(params):
    """Recover original FITS row IDs from the joined training split."""
    config = params['datasets']
    if config['train_test_random_seed'] is None:
        raise ValueError('Cannot reconstruct an unseeded validation split')
    dataset_type = (SobolevDataset if
                    params['emulator']['name'] == 'sobolev_ffnn_emu'
                    else Dataset)
    rows, sizes = [], []
    offset = 0
    for path in config['paths']:
        data = dataset_type().load(
            path=path, name=config['name'],
            columns_x=config.get('columns_x'),
            columns_y=config.get('columns_y'))
        size = len(data.x)
        valid = np.ones(size, dtype=bool)
        if config['remove_non_finite']:
            valid &= np.isfinite(data.x).all(axis=1)
            valid &= np.isfinite(data.y).all(axis=1)
            if isinstance(data, SobolevDataset):
                valid &= np.isfinite(data.y_growth).all(axis=1)
        rows.append(np.flatnonzero(valid) + offset)
        sizes.append(size)
        offset += size
    _, held = train_test_split(
        np.concatenate(rows), train_size=config['frac_train'],
        random_state=config['train_test_random_seed'])
    selected = []
    offset = 0
    for size in sizes:
        selected.append(np.sort(held[(held >= offset) &
                                    (held < offset + size)] - offset))
        offset += size
    return selected


class ValidationFits:
    """Select sample rows while leaving FITS reference grids intact."""

    def __init__(self, path, rows, spectra):
        self.dataset = io.FitsFile(path)
        self.rows = rows
        self.sample_keys = {'x_data', *spectra}

    def get_data(self, name):
        data = self.dataset.get_data(name)
        return data[self.rows] if name.lower() in self.sample_keys else data


def physical_growth_data(dataset, emu, spectrum, x):
    """Restore FITS growth targets using their own saved normalization."""
    np.testing.assert_array_equal(
        dataset.get_data('K_RANGE_' + spectrum), emu.y_model.k_ranges[0])
    reference = growth_diagnostics.spline(
        dataset.get_data('Z_ARRAY'), dataset.get_data('REF_' + spectrum))
    z = x[:, emu.x_names.index('z_pk')]
    return dataset.get_data(spectrum) * reference(z).T


def growth_residuals(emudata, dataset, spectrum, batch_size=128,
                     growth_floor=1e-6):
    """Evaluate one power run against physical growth on finite FITS rows.

    Reuse the independent power derivative from compare_pk_fk, without
    changing the predictions cached for the existing worst-mode plots.
    """
    emu = emudata.emu
    x = dataset.get_data('x_data')
    truth = physical_growth_data(dataset, emu, spectrum, x)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(truth).all(axis=1)
    excluded = int((~valid).sum())
    x, truth = x[valid], truth[valid]
    zi = emu.x_names.index('z_pk')
    reference = growth_diagnostics.spline(
        emu.y_model.z_array, emu.y_model.y_ref[0])
    derived, public = [], []
    for start in range(0, len(x), batch_size):
        xx = x[start:start + batch_size]
        ratio, dz = growth_diagnostics.value_and_dz(emu, xx, zi)
        with np.errstate(divide='ignore', invalid='ignore'):
            growth = growth_diagnostics.growth_from_ratio(
                ratio, dz, xx[:, zi], reference)
        # Growth from ln(P) is undefined for nonpositive physical power.
        power = ratio * reference(xx[:, zi]).T
        growth[(power <= 0) | ~np.isfinite(power)] = np.nan
        derived.append(growth)
        if emudata.is_sobolev:
            public.append(emu.eval_fk(xx))
    derived = np.concatenate(derived) if derived else np.empty_like(truth)
    absolute = {'fk_from_pk': derived - truth}
    if emudata.is_sobolev:
        public = np.concatenate(public) if public else np.empty_like(truth)
        absolute['eval_fk'] = public - truth
    denominator = np.where(abs(truth) > growth_floor, truth, np.nan)
    groups = {
        'accuracy_absolute': absolute,
        'accuracy_relative_percent': {
            name: 100 * residual / denominator
            for name, residual in absolute.items()},
    }
    if emudata.is_sobolev:
        consistency = public - derived
        groups['consistency_absolute'] = {'eval_fk - fk_from_pk': consistency}
        groups['consistency_relative_percent'] = {
            'eval_fk - fk_from_pk': 100 * consistency / denominator}
    return groups, {'input_rows': int(len(valid)), 'excluded_rows': excluded,
                    'evaluated_rows': int(len(x))}


def show_growth_histograms(root, emudata, dataset, spectrum, dr,
                           save_dir=None, relative_to=None, batch_size=128,
                           growth_floor=1e-6):
    """Save accuracy and consistency separately, with shared histogram bins."""
    groups, _ = growth_residuals(
        emudata, dataset, spectrum, batch_size, growth_floor)
    for kind, residuals in groups.items():
        path = plot_path(root, save_dir,
                         f'{spectrum}_{dr}_{kind}_histograms.png', relative_to)
        growth_diagnostics.plot_histograms(
            residuals, path, f'{spectrum} - {dr} (validation): {kind}')
        print(f'Saved {path}')


def show_summary(
        root,
        vlines=[0.01, 0.05, 0.1, 1.],
        save_dir=None,
        compare_to_all=False,
        relative_to=None,
        growth_histograms=True,
        growth_batch_size=128,
        growth_floor=1e-6
        ):

    save_path_hist = plot_path(
        root, save_dir, 'histograms.png', relative_to)
    save_path_sum = plot_path(
        root, save_dir, 'summary_table.txt', relative_to)

    with open(os.path.join(root, 'params.yaml')) as f:
        params = yaml.safe_load(f)

    training_spectrum = params['datasets']['name']
    is_sobolev = params['emulator']['name'] == 'sobolev_ffnn_emu'
    if is_sobolev:
        if not training_spectrum.startswith('pk_'):
            raise ValueError(
                'Sobolev histogram diagnostics require a pk_* training '
                'spectrum in order to infer the matching fk_* target.')
        spectrum = 'fk_{}'.format(training_spectrum[3:])
    else:
        spectrum = training_spectrum

    spectra_diff = {
        'pk_m': 'rel',
        'pk_cb': 'rel',
        'pk_weyl': 'rel',
        'fk_m': 'rel',
        'fk_cb': 'rel',
        'fk_weyl': 'rel',
        'cl_TT_lensed': 'rel',
        'cl_TE_lensed': 'abs',
        'cl_EE_lensed': 'rel',
        'cl_BB_lensed': 'rel',
        'cl_Tp_lensed': 'abs',
        'cl_pp_lensed': 'rel',
    }
    diff = spectra_diff[spectrum]

    dataset_paths = params['datasets']['paths']
    if compare_to_all:
        raise ValueError(
            'Validation plots use only the training paths in params.yaml; '
            'compare_to_all is not supported.')

    ranges = []
    for p, rows in zip(dataset_paths, validation_rows(params)):
        dr = os.path.basename(p).replace('.fits', '').split('_')[-1]
        if not len(rows):
            print(f'Skipping {p}: no validation rows')
            continue
        ranges.append((dr, p, rows))

    n_ranges = len(ranges)
    fig, axs = plt.subplots(
        1, n_ranges, figsize=(6 * n_ranges, 4), squeeze=False)

    headers = ['range', 'Validation rows']
    headers += ['>{}%'.format(val) for val in vlines]
    headers += ['Time emu (s)', 'Time total (s)']
    headers += ['Epochs (best/tot)', 'Loss', 'Val Loss', 'LR', '# NaN']
    tab = []

    emudata = {}

    for ndr, (dr, path, rows) in enumerate(ranges):

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

        sample_spectra = {training_spectrum, spectrum}
        if training_spectrum.startswith('pk_'):
            sample_spectra.add('fk_' + training_spectrum[3:])
        dataset = ValidationFits(path, rows, sample_spectra)
        x_data = dataset.get_data('x_data')
        y_data = dataset.get_data(spectrum)
        if is_sobolev:
            # eval_fk returns physical growth; FITS stores f / f_reference.
            y_data = physical_growth_data(
                dataset, emudata[dr].emu, spectrum, x_data)
        n_nans = x_data.shape[0]
        # Get mask nans
        mask_nans = (np.isfinite(y_data).all(axis=1)
                     & np.isfinite(x_data).all(axis=1))
        # Filter nans
        x_data = x_data[mask_nans]
        y_data = y_data[mask_nans]
        n_nans -= x_data.shape[0]
        if not len(x_data):
            raise ValueError(f'No finite validation rows in {path}')
        print(f'{path}: evaluating {len(x_data)} validation rows')

        result = fun(
            x_emu=x_data, y_data=y_data,
            want_scaling=False, want_pca=False,
            select_pca_modes_emu=None, select_pca_modes_data=None,
            time_emu=True, use_growth=is_sobolev)

        tab_line = [dr, len(x_data)]
        tab_line += ['{:.3f}%'.format(
            len(result[result > val / 100.]) / len(result) * 100.)
                     for val in vlines]
        tab_line += ['{:.1e}'.format(emudata[dr].time_emu)]
        tab_line += ['{:.1e}'.format(emudata[dr].time_all)]
        tab_line += ['{}/{}'.format(int(epochs[idx_best]), int(epochs[-1]))]
        tab_line += ['{:.2e}'.format(losses[idx_best])]
        tab_line += ['{:.2e}'.format(val_losses[idx_best])]
        tab_line += ['{:.2e}'.format(learning_rates[idx_best])]
        tab_line += ['{}'.format(n_nans)]
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

        if growth_histograms and training_spectrum.startswith('pk_'):
            show_growth_histograms(
                root, emudata[dr], dataset, 'fk_' + training_spectrum[3:], dr,
                save_dir=save_dir, relative_to=relative_to,
                batch_size=growth_batch_size, growth_floor=growth_floor)

    fig.suptitle(f'{spectrum} (validation)', fontsize=20)
    plt.tight_layout()

    fig.savefig(save_path_hist, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path_hist}")

    summary_table = tabulate(tab, headers=headers, tablefmt='orgtbl')
    print(summary_table)

    with open(save_path_sum, 'w') as outputfile:
        outputfile.write(f'{spectrum} (validation)')
        outputfile.write('\n')
        outputfile.write(summary_table)
        outputfile.write('\n\n')

    return emudata, spectrum, diff, is_sobolev


def plot_worst_modes(
        root,
        emudata,
        spectrum,
        diff,
        use_growth=False,
        n_modes_kept=3,
        vlines=[0.01, 0.05, 0.1, 1.],
        save_dir=None,
        relative_to=None):

    save_path = plot_path(
        root, save_dir, 'worst_modes.png', relative_to)

    ranges = list(emudata.keys())
    n_ranges = len(ranges)
    fig, axs = plt.subplots(
        1 + n_modes_kept,
        n_ranges,
        figsize=(6 * n_ranges, 6 + 4*n_modes_kept),
        squeeze=False)
    fig.suptitle('Worst validation modes - {}'.format(spectrum), fontsize=20, y=1.0)

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
        y_emu = emudata[dr].y_emu[idxs]
        y_data = emudata[dr].y_data[idxs]
        # The Sobolev emulator is checked against the fk values stored in the
        # data file.  get_y_class currently reconstructs the primary Pk
        # spectrum, so it is not a valid independent fk curve.
        y_class = None if use_growth else emudata[dr].get_y_class(idxs)

        for val in vlines:
            axs[0, ndr].axhline(emudata[dr].max_y_data * val, c='k', lw=0.1)

        for nmode in range(len(idxs)):
            # On the first row plot relative/absolute differences for all modes
            axs[0, ndr].plot(
                np.abs(diffs[nmode]) * 100.,
                label='Rank: {}, Idx: {}'.format(nmode + 1, idxs[nmode]))
            axs[0, ndr].set_yscale('log')
            axs[0, ndr].set_title('{} - {}'.format(spectrum, dr), fontsize=18)

            # On the other rows plot individual modes: data, emulated, class
            axs[1 + nmode, ndr].plot(y_data[nmode], label='Data')
            axs[1 + nmode, ndr].plot(
                y_emu[nmode], linestyle='--', label='Emulated')
            if y_class is not None:
                axs[1 + nmode, ndr].plot(
                    y_class[nmode], linestyle=':', label='Class')
            axs[1 + nmode, ndr].set_ylabel(
                'Rank: {}, Idx: {}'.format(nmode + 1, idxs[nmode]))

        if diff == 'rel':
            axs[0, 0].set_ylabel('rel diff [%]')
        elif diff == 'abs':
            axs[0, 0].set_ylabel('abs diff')

        axs[0, ndr].legend()
    axs[1, 0].legend()

    plt.tight_layout()

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot emulator error histograms and worst modes.')
    parser.add_argument(
        '--roots',
        '-r',
        type=str,
        nargs='+',
        required=True,
        help='Path to the emulator root folder.')
    parser.add_argument(
        '--skip-growth-histograms', action='store_true',
        help='Skip the additional growth histograms for pk_* runs.')
    parser.add_argument(
        '--growth-batch-size', type=int, default=128,
        help='Batch size for independent power derivatives (default: 128).')
    parser.add_argument(
        '--growth-floor', type=float, default=1e-6,
        help='Exclude |data f| <= this from relative growth histograms.')
    parser.add_argument(
        '--save-dir',
        '-s',
        type=str,
        help=('Save here with relative run paths joined by underscores. '
              'Defaults to saving beside each training run.'))
    args = parser.parse_args()
    if (args.growth_batch_size < 1 or args.growth_floor < 0
            or not np.isfinite(args.growth_floor)):
        parser.error('Growth batch size must be positive and growth floor '
                     'must be finite and nonnegative.')

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    # Find all folders containing history_log.csv in the provided roots
    relative_to = source_root(args.roots)
    roots = []
    for root in args.roots:
        for folder, folders, files in os.walk(root):
            if 'history_log.csv' in files:
                if folder not in roots:
                    roots.append(folder)

    for root in roots:

        emudata, spectrum, diff, use_growth = show_summary(
            root,
            save_dir=args.save_dir,
            relative_to=relative_to,
            growth_histograms=not args.skip_growth_histograms,
            growth_batch_size=args.growth_batch_size,
            growth_floor=args.growth_floor)

        plot_worst_modes(
            root,
            emudata,
            spectrum, diff, use_growth=use_growth,
            n_modes_kept=3,
            save_dir=args.save_dir,
            relative_to=relative_to)
