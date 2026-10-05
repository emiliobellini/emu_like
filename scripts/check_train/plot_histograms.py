"""Compact validation diagnostics for power, growth, and angular spectra."""

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
        self.sample_keys = {'x_data', *(name.lower() for name in spectra)}

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


METHOD_STYLES = {
    'emulator': ('#2878b5', '-'),
    'fk_from_pk': ('#2878b5', '-'),
    'eval_fk': ('#d97921', '--'),
}


def complete_errors(residual):
    """Use the same complete spectra for tables and both figures."""
    residual = np.asarray(residual)
    complete = np.isfinite(residual).all(axis=1)
    values = np.abs(residual[complete])
    return values, np.sqrt(np.mean(values**2, axis=1))


def relative_accuracy_row(dr, method, residual_percent, input_rows, vlines):
    _, rms = complete_errors(residual_percent)
    quantiles = ([f'{v:.6g}' for v in
                  [np.median(rms), np.percentile(rms, 95), np.max(rms)]]
                 if len(rms) else ['N/A'] * 3)
    return [dr, method, input_rows, len(rms), input_rows - len(rms)] + quantiles + [
        f'{100 * np.mean(rms > threshold):.3f}%' if len(rms) else 'N/A'
        for threshold in vlines]


def direct_residuals(emudata, dataset, spectrum, growth_floor):
    """Compare physical spectra using each source's own normalization."""
    emu = emudata.emu
    x, truth = dataset.get_data('x_data'), dataset.get_data(spectrum)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(truth).all(axis=1)
    x, truth = x[valid], truth[valid]
    prediction = (np.atleast_2d(emudata.get_y_emu(x, timeit=True))
                  if len(x) else np.empty_like(truth))
    if spectrum.startswith('cl_'):
        truth = truth * np.asarray(dataset.get_data('REF_' + spectrum)).reshape(1, -1)
        prediction = prediction * np.asarray(emu.y_model.y_ref[0]).reshape(1, -1)
    else:
        z = x[:, emu.x_names.index('z_pk')]
        truth = truth * growth_diagnostics.spline(
            dataset.get_data('Z_ARRAY'), dataset.get_data('REF_' + spectrum))(z).T
        prediction = prediction * growth_diagnostics.spline(
            emu.y_model.z_array, emu.y_model.y_ref[0])(z).T
    absolute = prediction - truth
    floor = growth_floor if spectrum.startswith('fk_') else 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        relative = 100 * absolute / np.where(abs(truth) > floor, truth, np.nan)
    return {'accuracy_relative_percent': {'emulator': relative},
            'accuracy_absolute': {'emulator': absolute}}


def plot_observable(root, spectrum, records, vlines, save_dir, relative_to):
    """One histogram dashboard and one two-row scale dashboard per observable."""
    from matplotlib.ticker import FuncFormatter
    family = spectrum.split('_')[0]
    is_cl = family == 'cl'
    n = len(records)
    fig_hist, hist_axes = plt.subplots(
        1, n, figsize=(6*n, 4.8), squeeze=False, sharex=True, sharey=True)
    fig_scale, scale_axes = plt.subplots(
        2, n, figsize=(6*n, 8), squeeze=False, sharex=True, sharey=True)
    all_rms = [complete_errors(residual)[1] for record in records
               for residual in record['groups']['accuracy_relative_percent'].values()]
    positive = np.concatenate([r[r > 0] for r in all_rms] + [np.asarray(vlines)])
    lo, hi = positive.min()/2, positive.max()*2
    bins = np.geomspace(lo, hi, 35)
    for col, record in enumerate(records):
        ax = hist_axes[0, col]
        ax.set_title(record['range'])
        notes = []
        for method, residual in record['groups']['accuracy_relative_percent'].items():
            color, style = METHOD_STYLES[method]
            values, rms = complete_errors(residual)
            zeros = int(np.count_nonzero(rms == 0))
            notes.append(f'{method}: {len(rms)}/{record["input_rows"]} samples; '
                         + (f'{100*np.mean(rms > 1):.2f}% > 1%; zero={zeros}'
                            if len(rms) else 'no defined errors'))
            # Exact zeros go in the leftmost bin and remain zero in statistics.
            if len(rms):
                ax.hist(np.maximum(rms, lo), bins=bins, histtype='step',
                        color=color, linestyle=style, linewidth=1.8, label=method)
                grid = record['grid']
                q16, median, q84, q95 = np.percentile(values, [16, 50, 84, 95], axis=0)
                top, bottom = scale_axes[:, col]
                # A tiny display floor permits exact-zero curves on log axes.
                display = lambda a: np.maximum(a, 1e-8)
                top.fill_between(grid, display(q16), display(q84), color=color, alpha=.16)
                top.plot(grid, display(median), color=color, linestyle=style,
                         label=f'{method}: median (band 16–84%)')
                top.plot(grid, display(q95), color=color, linestyle=':',
                         label=f'{method}: 95th percentile')
                for rank, index in enumerate(np.argsort(rms)[::-1][:3], 1):
                    bottom.plot(grid, display(values[index]), color=color,
                                linestyle=style, alpha=1/rank, linewidth=1.2,
                                label=f'{method}: rank {rank}, RMS={rms[index]:.3g}%')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=.8)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}%'))
        ax.set_xlabel('RMS relative error across ' + ('ℓ' if is_cl else 'k'))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='lower right', fontsize=8)
        ax.text(.02, .98, '\n'.join(notes), transform=ax.transAxes,
                va='top', fontsize=8, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        for threshold in vlines:
            ax.axvline(threshold, color='#ad4141', alpha=.65, linewidth=.8)
        for row, scale_ax in enumerate(scale_axes[:, col]):
            scale_ax.set_title(f'{record["range"]} — ' +
                               ('validation distribution' if row == 0 else 'worst RMS samples'))
            scale_ax.set_yscale('log')
            if not is_cl:
                scale_ax.set_xscale('log')
            for threshold in vlines:
                scale_ax.axhline(threshold, color='#ad4141', alpha=.65, linewidth=.8)
            scale_ax.grid(alpha=.15)
            handles, labels = scale_ax.get_legend_handles_labels()
            if handles:
                scale_ax.legend(fontsize=8, loc='best')
            else:
                scale_ax.text(.5, .5, 'No defined relative errors',
                              ha='center', transform=scale_ax.transAxes)
        scale_axes[1, col].set_xlabel('Multipole ℓ' if is_cl else r'$k$ [$h$/Mpc]')
    hist_axes[0, 0].set_ylabel('Validation samples')
    for ax in scale_axes[:, 0]:
        ax.set_ylabel('Absolute relative error [%]')
    fig_hist.suptitle(f'{spectrum} — validation accuracy')
    fig_scale.suptitle(f'{spectrum} — validation errors (zero curves shown at 1e−8%)')
    for fig, filename in [(fig_hist, f'{family}_accuracy.png'),
                          (fig_scale, f'{family}_errors_vs_{"ell" if is_cl else "k"}.png')]:
        fig.tight_layout()
        path = plot_path(root, save_dir, filename, relative_to)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path}')


def show_summary(root, vlines=(.01, .05, .1, 1.), save_dir=None,
                 compare_to_all=False, relative_to=None, growth_histograms=True,
                 growth_batch_size=128, growth_floor=1e-6):
    """Write one summary and two figures for each validation observable."""
    if compare_to_all:
        raise ValueError('Validation diagnostics require the saved training paths')
    with open(os.path.join(root, 'params.yaml')) as f:
        params = yaml.safe_load(f)
    spectrum = params['datasets']['name']
    if not spectrum.startswith(('pk_', 'fk_', 'cl_')):
        raise ValueError(f'Unsupported observable: {spectrum}')
    emudata = EmuData(root)
    if emudata.is_sobolev and not spectrum.startswith('pk_'):
        raise ValueError('Sobolev diagnostics require a pk_* training observable')
    spectra = [spectrum]
    if spectrum.startswith('pk_'):
        spectra.append('fk_' + spectrum[3:])
    observations = {name: [] for name in spectra}
    timing, absolute, consistency = [], [], []
    paths = params['datasets']['paths']
    for path, rows in zip(paths, validation_rows(params)):
        dr = os.path.basename(path).removesuffix('.fits').split('_')[-1]
        dataset = ValidationFits(path, rows, spectra)
        start = time.perf_counter()
        primary = direct_residuals(emudata, dataset, spectrum, growth_floor)
        timing.append([dr, spectrum, len(rows), time.perf_counter() - start])
        groups_by_name = {spectrum: primary}
        if spectrum.startswith('pk_'):
            start = time.perf_counter()
            groups_by_name[spectra[1]], _ = growth_residuals(
                emudata, dataset, spectra[1], growth_batch_size, growth_floor)
            timing.append([dr, spectra[1] + ' (all methods)', len(rows),
                           time.perf_counter() - start])
        for name, groups in groups_by_name.items():
            key = ('ELL_RANGE_' if name.startswith('cl_') else 'K_RANGE_') + name
            grid = np.asarray(dataset.get_data(key))
            expected = (emudata.emu.y_model.ell_ranges[0] if name.startswith('cl_')
                        else emudata.emu.y_model.k_ranges[0])
            np.testing.assert_array_equal(grid, expected)
            if grid.ndim != 1 or np.any(np.diff(grid) <= 0):
                raise ValueError(f'Invalid coordinate grid for {name}')
            observations[name].append(dict(range=dr, grid=grid, groups=groups,
                                           input_rows=len(rows)))
            for kind, table in [('accuracy_absolute', absolute),
                                ('consistency_absolute', consistency)]:
                if not name.startswith('fk_'):
                    continue
                for method, residual in groups.get(kind, {}).items():
                    finite = np.asarray(residual)[np.isfinite(residual)]
                    table.append([dr, method, len(finite), residual.size-len(finite),
                                  np.sqrt(np.mean(finite**2)) if len(finite) else 'N/A',
                                  np.max(abs(finite)) if len(finite) else 'N/A'])
    headers = ['range', 'Prediction', 'Validation rows', 'Evaluated rows',
               'Excluded rows', 'Median RMS [%]', 'P95 RMS [%]', 'Max RMS [%]']
    headers += [f'>{v}%' for v in vlines]
    sections = []
    for name, records in observations.items():
        table = [relative_accuracy_row(record['range'], method, residual,
                                       record['input_rows'], vlines)
                 for record in records for method, residual in
                 record['groups']['accuracy_relative_percent'].items()]
        sections.append(f'{name} (validation; relative RMS accuracy)\n'
                        + tabulate(table, headers=headers, tablefmt='orgtbl'))
        if name == spectrum or growth_histograms:
            plot_observable(root, name, records, vlines, save_dir, relative_to)
    for title, table in [('Growth absolute errors (dimensionless)', absolute),
                         ('Sobolev consistency: eval_fk - fk_from_pk (dimensionless)', consistency)]:
        if table:
            sections.append(title + '\n' + tabulate(table, headers=[
                'range', 'Prediction', 'Finite bins', 'Excluded bins', 'RMS', 'Max abs'],
                tablefmt='orgtbl', floatfmt='.6g'))
    emu = emudata.emu
    best = int(np.argmin(emu.val_loss))
    sections.append('Training history\n' + tabulate([[
        f'{int(emu.epochs[best])}/{int(emu.epochs[-1])}', emu.loss[best],
        emu.val_loss[best], emu.learning_rate[best]]],
        headers=['Epochs (best/total)', 'Loss', 'Val Loss', 'LR'], tablefmt='orgtbl'))
    sections.append('Diagnostic timing (includes reference reconstruction)\n' + tabulate(
        timing, headers=['range', 'Observable', 'Validation rows', 'Elapsed [s]'],
        tablefmt='orgtbl', floatfmt='.6g'))
    sections.append(
        'Threshold columns: percentage of evaluated spectra above the RMS error threshold. '
        'Tables and relative-error plots use only spectra with finite relative errors in every bin. '
        'Zero errors remain zero in statistics; histogram zeros occupy the leftmost bin. '
        f'Growth bins with |f_data| <= {growth_floor:g} are undefined; '
        'pk/cl bins with zero truth are undefined. Relative errors near cl zero crossings '
        'can be large. Absolute-growth and consistency statistics use all finite bins.')
    summary = '\n\n'.join(sections) + '\n'
    path = plot_path(root, save_dir, 'summary_table.txt', relative_to)
    path.write_text(summary)
    print(summary)
    print(f'Saved {path}')
    return observations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot compact validation accuracy and scale diagnostics.')
    parser.add_argument(
        '--roots',
        '-r',
        type=str,
        nargs='+',
        required=True,
        help='Path to the emulator root folder.')
    parser.add_argument(
        '--skip-growth-histograms', action='store_true',
        help='Skip both growth figures for pk_* runs; keep growth summary tables.')
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

        show_summary(
            root,
            save_dir=args.save_dir,
            relative_to=relative_to,
            growth_histograms=not args.skip_growth_histograms,
            growth_batch_size=args.growth_batch_size,
            growth_floor=args.growth_floor)
