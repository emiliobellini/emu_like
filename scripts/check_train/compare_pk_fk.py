"""Compare standard and Sobolev power/growth on common validation rows.

See docs/compare_pk_fk.md. Run after training has stopped writing checkpoints.

The helpers below cover differentiable inference, validation-row selection,
and reporting. The main pipeline then loads the three models, evaluates the
common validation sample, checks derivatives, and writes plots and arrays.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import make_interp_spline
from sklearn.model_selection import train_test_split
import yaml


# Reference reconstruction and differentiable network evaluation.
def spline(z, values):
    """Interpolate a saved normalization table without extrapolating in z."""
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[1] != len(z):
        raise ValueError('Expected reference shape (1, k, z) or (k, z)')
    result = make_interp_spline(z, values, k=min(3, len(z)-1), axis=-1)
    result.extrapolate = False
    return result


def tf_scale(x, scaler, inverse=False):
    """Differentiable forms of the scalers supported by this diagnostic."""
    import tensorflow as tf
    if scaler is None or scaler.name in (None, 'None'):
        return x
    if scaler.name not in ('StandardScaler', 'LogStandardScaler'):
        raise ValueError(f'Unsupported diagnostic scaler: {scaler.name}')
    s = scaler.skl_scaler
    mean = tf.constant(s.mean_, dtype=x.dtype)
    scale = tf.constant(s.scale_, dtype=x.dtype)
    if inverse:
        x = x * scale + mean
        return tf.exp(x) if scaler.name == 'LogStandardScaler' else x
    if scaler.name == 'LogStandardScaler':
        x = tf.math.log(x)
    return (x-mean)/scale


def tf_pca(x, wrapper, inverse=False):
    """Apply saved PCA while preserving TensorFlow redshift derivatives."""
    import tensorflow as tf
    if wrapper is None:
        return x
    p = wrapper.pca
    components = tf.constant(p.components_, dtype=x.dtype)
    mean = tf.constant(p.mean_, dtype=x.dtype)
    if inverse:
        if p.whiten:
            x = x * tf.sqrt(tf.constant(p.explained_variance_, dtype=x.dtype))
        return tf.linalg.matmul(x, components) + mean
    x = tf.linalg.matmul(x-mean, components, transpose_b=True)
    if p.whiten:
        x = x / tf.sqrt(tf.constant(p.explained_variance_, dtype=x.dtype))
    return x


def network_value(emu, x):
    """Return unscaled network targets, before physical normalization."""
    x = tf_pca(tf_scale(x, emu.x_scaler), emu.x_pca)
    y = emu.model(x, training=False)
    return tf_scale(tf_pca(y, emu.y_pca, inverse=True),
                    emu.y_scaler, inverse=True)


def value_and_dz(emu, x, z_index):
    """Evaluate targets and their derivatives with other inputs held fixed."""
    import tensorflow as tf
    x = tf.convert_to_tensor(x, dtype=emu.model.compute_dtype)
    # A redshift-only tangent avoids constructing the full input Jacobian.
    direction = tf.broadcast_to(tf.one_hot(z_index, x.shape[1], dtype=x.dtype),
                                tf.shape(x))
    with tf.autodiff.ForwardAccumulator(x, direction) as acc:
        y = network_value(emu, x)
    derivative = acc.jvp(y)
    if derivative is None:
        raise ValueError('Model output is disconnected from redshift')
    return y.numpy(), derivative.numpy()


def growth_from_ratio(ratio, derivative, z, reference):
    """Convert dR/dz to physical growth for P = R * P_normalization."""
    norm = reference(z).T
    return -.5*(1+z[:, None])*(
        derivative/ratio + reference.derivative()(z).T/norm)


def physical_value(emu, x, reference, z_index):
    """Restore the saved normalization of a power or direct-growth target."""
    import tensorflow as tf
    value = network_value(emu, tf.convert_to_tensor(
        x, dtype=emu.model.compute_dtype)).numpy()
    return value * reference(x[:, z_index]).T


def finite_difference_growth(emu, x, reference, z_index, step, bounds):
    """Second-order central/one-sided d ln P / dz, within model coverage."""
    z = x[:, z_index]
    lo, hi = bounds
    if hi-lo < 2*step or np.any((z < lo) | (z > hi)):
        raise ValueError('Invalid finite-difference step or redshift coverage')
    # Use central differences inside the domain and one-sided stencils near
    # either boundary; no evaluation should leave the trained z interval.
    shifts = np.tile([-step, 0., step], (len(x), 1))
    weights = np.tile([-1., 0., 1.], (len(x), 1)) / (2*step)
    low, high = z-step < lo, z+step > hi
    shifts[low] = [0., step, 2*step]
    weights[low] = np.array([-3., 4., -1.])/(2*step)
    shifts[high] = [-2*step, -step, 0.]
    weights[high] = np.array([1., -4., 3.])/(2*step)
    result = np.zeros((len(x), len(emu.y_names)))
    for j in range(3):
        xx = x.copy()
        xx[:, z_index] += shifts[:, j]
        p = physical_value(emu, xx, reference, z_index)
        if np.any(p <= 0):
            raise ValueError('Nonpositive power in finite-difference stencil')
        result += weights[:, j, None] * np.log(p)
    return -.5*(1+z[:, None])*result


# Validation selection: preserve original row IDs across filtering and splits.
def common_rows(configs, pk, fk):
    """Intersect validation IDs reconstructed from each training setup."""
    paths = configs[0]['datasets']['paths']
    canonical = [str(Path(p).resolve()) for p in paths]
    for c in configs:
        d = c['datasets']
        if [str(Path(p).resolve()) for p in d['paths']] != canonical:
            raise ValueError(
                'Models must use the same ordered training FITS files')
        if d.get('columns_x') is not None or d.get('columns_y') is not None:
            raise ValueError('Sliced training datasets are not supported')
    masks = [[], [], []]
    sizes = []
    offset = 0
    for path in paths:
        with fits.open(path, memmap=True) as h:
            mx = np.isfinite(h['X_DATA'].data).all(axis=1)
            mp = mx & np.isfinite(h[pk].data).all(axis=1)
            mf = mx & np.isfinite(h[fk].data).all(axis=1)
            sizes.append(len(mx))
            # Standard P, standard f, and Sobolev reject different rows.
            # Offsets identify rows in the original concatenated FITS files.
            for i, mask in enumerate((mp, mf, mp & mf)):
                if (not configs[i]['datasets']['remove_non_finite']
                        and not mask.all()):
                    raise ValueError(
                        'Nonfinite training data with filtering disabled')
                masks[i].append(np.flatnonzero(mask)+offset)
            offset += len(mx)
    held = []
    # Training splits the joined, filtered datasets, not each file separately.
    for ids, config in zip(masks, configs):
        d = config['datasets']
        if d['train_test_random_seed'] is None:
            raise ValueError('Cannot reconstruct an unseeded validation split')
        held.append(train_test_split(np.concatenate(ids),
                    train_size=d['frac_train'],
                    random_state=d['train_test_random_seed'])[1])
    return paths, sizes, np.intersect1d(np.intersect1d(*held[:2]), held[2])


def stats(a):
    """Summarize finite residuals and explicitly count undefined entries."""
    a = np.asarray(a)
    good = a[np.isfinite(a)]
    result = {'count': int(a.size), 'nonfinite': int(a.size-good.size)}
    if good.size:
        result.update(
            bias=float(good.mean()), rms=float(np.sqrt(np.mean(good**2))),
            **dict(zip(
                ['median_abs', 'p95_abs', 'p99_abs', 'max_abs'],
                map(float, np.quantile(abs(good), [.5, .95, .99, 1])))))
    return result


def summarize(residual):
    """Report bin-level errors and errors for complete spectra separately."""
    # Excluded relative-error bins must not make a partial spectrum appear
    # to be a complete one in the per-spectrum statistics.
    complete = np.isfinite(residual).all(axis=1)
    rows = residual[complete]
    return {'bins': stats(residual), 'complete_spectra': int(complete.sum()),
            'spectrum_rms': stats(np.sqrt(np.mean(rows**2, axis=1))),
            'spectrum_max_abs': stats(np.max(abs(rows), axis=1))}


# Plotting: compare both typical errors and tails, using shared histogram bins.
def plot_histograms(residuals, path, title):
    """Plot absolute bin errors, spectrum RMS, and spectrum maxima."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [[], [], []]
    for label, r in residuals.items():
        complete = np.isfinite(r).all(axis=1)
        arrays = [abs(r[np.isfinite(r)]),
                  np.sqrt(np.mean(r[complete]**2, axis=1)),
                  np.max(abs(r[complete]), axis=1)]
        for panel, a in zip(panels, arrays):
            a = a[np.isfinite(a) & (a > 0)]
            if len(a):
                panel.append((label, a))
    for ax, panel in zip(axes, panels):
        if panel:
            lower = min(a.min() for _, a in panel)
            upper = max(a.max() for _, a in panel)
            bins = np.geomspace(lower, max(upper, lower*1.01), 65)
            for label, a in panel:
                ax.hist(
                    a, bins=bins, histtype='step', density=True, label=label)
    for ax, label in zip(
            axes, ['Absolute bin error', 'RMS per spectrum',
                   'Maximum per spectrum']):
        ax.set(xscale='log', yscale='log', xlabel=label, ylabel='Density')
        if ax.lines or ax.patches:
            ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_k_errors(k, values, path):
    """Locate error tails in k using percentile and worst-case curves."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    categories = [(['standard_pk', 'sobolev_pk'], 'Power error [%]'),
                  (['standard_fk_absolute', 'standard_fk_from_pk_absolute',
                    'sobolev_fk_absolute'], 'Absolute growth error'),
                  (['standard_consistency_absolute',
                    'sobolev_consistency_absolute'],
                   'Absolute consistency residual')]
    for ax, (names, label) in zip(axes, categories):
        for name in names:
            a = abs(values[name])
            ax.plot(k, np.quantile(a, .99, axis=0), label=name+' p99')
            # Keep the full worst-case curve visible as a dotted line.
            ax.plot(
                k, a.max(axis=0), ':',
                color=ax.lines[-1].get_color(), alpha=.5)
        ax.set(xscale='log', yscale='log', xlabel='k [h/Mpc]', ylabel=label)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    """Run the comparison from saved checkpoints to reproducible outputs."""
    # 1. Parse the model paths, sampling limits, and numerical-check settings.
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['standard-pk', 'standard-fk', 'sobolev', 'output']:
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--max-samples', type=int, default=6000,
                        help='0 selects all common validation rows')
    parser.add_argument('--seed', type=int, default=1543)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--k-min', type=float, default=0.)
    parser.add_argument('--k-max', type=float, default=np.inf)
    parser.add_argument('--growth-floor', type=float, default=1e-6,
                        help='Exclude |HiClass f| <= this from relative '
                             'errors')
    parser.add_argument('--fd-steps', type=float, nargs='+',
                        default=[.01, .003, .001])
    parser.add_argument('--fd-samples', type=int, default=128)
    args = parser.parse_args(argv)
    if (args.max_samples < 0 or args.batch_size < 1 or args.fd_samples < 1
            or args.growth_floor < 0 or not np.isfinite(args.growth_floor)
            or args.k_min >= args.k_max
            or any(s <= 0 or not np.isfinite(s) for s in args.fd_steps)):
        parser.error('Invalid sample count, batch size, range, floor or '
                     'derivative step')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty; use a new diagnostic '
                     'directory')
    # Import TensorFlow after argument handling so --help avoids model setup.
    import tensorflow as tf
    from emu_like.emu import Emulator
    tf.config.experimental.enable_tensor_float_32_execution(False)
    # 2. Load compatible checkpoints and their saved reference spectra.
    directories = [args.standard_pk, args.standard_fk, args.sobolev]
    configs = [
        yaml.safe_load((p/'params.yaml').read_text()) for p in directories]
    expected = ['ffnn_emu', 'ffnn_emu', 'sobolev_ffnn_emu']
    if [c['emulator']['name'] for c in configs] != expected:
        parser.error('Expected standard power, standard growth, and Sobolev '
                     'model directories')
    pk = configs[0]['datasets']['name']
    if not pk.startswith('pk_'):
        parser.error('Power target must be pk_*')
    fk = 'fk_'+pk[3:]
    if (configs[1]['datasets']['name'] != fk
            or configs[2]['datasets']['name'] != pk):
        parser.error('All models must describe the same species')
    models = []
    for path, name in zip(directories, expected):
        print(f'Loading best checkpoint: {path}', flush=True)
        emu = Emulator.choose_one(name)
        emu.load(str(path), still_training=False)
        models.append(emu)
    if any(e.x_names != models[0].x_names for e in models):
        raise ValueError('Model input parameter order differs')
    zi = models[0].x_names.index('z_pk')
    k = np.asarray(models[0].y_model.k_ranges[0])
    for e in models:
        np.testing.assert_array_equal(e.y_model.k_ranges[0], k)
    refs = [spline(e.y_model.z_array, e.y_model.y_ref[0]) for e in models]
    # 3. Select rows held out by all models, then restore targets in physical
    # units using each source file's normalization (not a model prediction).
    paths, sizes, common = common_rows(configs, pk, fk)
    if not len(common):
        raise ValueError('No common held-out rows')
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(
        common, min(args.max_samples or len(common), len(common)),
        replace=False))
    print(f'Evaluating {len(chosen)} of {len(common)} common validation rows',
          flush=True)
    xs, truths_p, truths_f, regions, file_ids, row_ids = [], [], [], [], [], []
    offset = 0
    for index, (path, size) in enumerate(zip(paths, sizes)):
        ids = chosen[(chosen >= offset) & (chosen < offset+size)]-offset
        offset += size
        if not len(ids):
            continue
        with fits.open(path, memmap=True) as h:
            np.testing.assert_array_equal(h['K_RANGE_'+pk].data, k)
            np.testing.assert_array_equal(h['K_RANGE_'+fk].data, k)
            xx = np.array(h['X_DATA'].data[ids], dtype=float)
            z = xx[:, zi]
            norm_p = spline(h['Z_ARRAY'].data, h['REF_'+pk].data)(z).T
            norm_f = spline(h['Z_ARRAY'].data, h['REF_'+fk].data)(z).T
            xs.append(xx)
            truths_p.append(h[pk].data[ids]*norm_p)
            truths_f.append(h[fk].data[ids]*norm_f)
        regions.extend([Path(path).stem]*len(ids))
        file_ids.extend([index]*len(ids))
        row_ids.extend(ids.tolist())
    x, ptrue, ftrue = map(np.concatenate, [xs, truths_p, truths_f])
    regions = np.asarray(regions)
    for e in models:
        bounds = np.asarray(e.x_ranges)
        if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
            raise ValueError(
                'Selected inputs are outside a model training domain')
    selected_k = (k >= args.k_min) & (k <= args.k_max)
    if not selected_k.any():
        raise ValueError('No k bins in requested interval')
    # 4. Evaluate direct predictions and reconstruct growth from each power
    # model. Batching bounds the memory used by automatic differentiation.
    predictions = {n: [] for n in [
        'standard_pk', 'sobolev_pk', 'standard_fk', 'standard_fk_from_pk',
        'sobolev_fk', 'sobolev_fk_from_pk']}
    for start in range(0, len(x), args.batch_size):
        xx = x[start:start+args.batch_size]
        for index, prefix in [(0, 'standard'), (2, 'sobolev')]:
            ratio, dr = value_and_dz(models[index], xx, zi)
            predictions[prefix+'_pk'].append(ratio*refs[index](xx[:, zi]).T)
            predictions[prefix+'_fk_from_pk'].append(
                growth_from_ratio(ratio, dr, xx[:, zi], refs[index]))
        predictions['standard_fk'].append(
            physical_value(models[1], xx, refs[1], zi))
        predictions['sobolev_fk'].append(models[2].eval_fk(xx))
    predictions = {n: np.concatenate(v) for n, v in predictions.items()}
    if (np.any(ptrue <= 0)
            or any(np.any(predictions[n] <= 0)
                   for n in ['standard_pk', 'sobolev_pk'])
            or not all(np.isfinite(a).all()
                       for a in [ptrue, ftrue, *predictions.values()])):
        raise ValueError(
            'Nonfinite predictions/reference or nonpositive power')
    # 5. Separate accuracy against HiClass from agreement between emulators.
    # Relative growth errors use a shared HiClass denominator; near-zero
    # targets remain in absolute statistics but are masked in relative ones.
    residuals = {n: 100*(predictions[n]/ptrue-1)
                 for n in ['standard_pk', 'sobolev_pk']}
    for name in ['standard_fk', 'standard_fk_from_pk', 'sobolev_fk']:
        residuals[name+'_absolute'] = predictions[name]-ftrue
    residuals['standard_consistency_absolute'] = (
        predictions['standard_fk']-predictions['standard_fk_from_pk'])
    residuals['sobolev_consistency_absolute'] = (
        predictions['sobolev_fk']-predictions['sobolev_fk_from_pk'])
    denom = np.where(abs(ftrue) > args.growth_floor, ftrue, np.nan)
    for name, a in list(residuals.items()):
        if name.endswith('_absolute'):
            residuals[name.replace('_absolute', '_relative_percent')] = (
                100*a/denom)
    # 6. Check autodiff with several finite-difference steps on a smaller
    # sample. ReLU boundaries and float32 rounding can affect this comparison.
    fd_ids = np.sort(rng.choice(
        len(x), min(args.fd_samples, len(x)), replace=False))
    finite_differences = {}
    for index, prefix in [(0, 'standard'), (2, 'sobolev')]:
        e = models[index]
        bounds = (max(e.x_ranges[zi][0], e.y_model.z_array[0]),
                  min(e.x_ranges[zi][1], e.y_model.z_array[-1]))
        for step in args.fd_steps:
            fd = finite_difference_growth(
                e, x[fd_ids], refs[index], zi, step, bounds)
            finite_differences[f'{prefix}_dz_{step:g}'] = stats(
                (fd-predictions[prefix+'_fk_from_pk'][fd_ids])[:, selected_k])
    args.output.mkdir(parents=True, exist_ok=True)
    # 7. Save provenance and diagnostics, pooled and grouped by source file.
    # Keep full-grid predictions and original row IDs for subsequent analysis.
    report = {'models': [str(p.resolve()) for p in directories],
              'checkpoints': [
                  {'epoch_one_based': int(e.epochs[np.argmin(e.val_loss)])+1,
                   'validation_loss': float(min(e.val_loss))}
                  for e in models],
              'data_files': paths, 'x_names': models[0].x_names,
              'common_validation_rows': len(common), 'sampled_rows': len(x),
              'seed': args.seed, 'k_min': float(k[selected_k].min()),
              'k_max': float(k[selected_k].max()),
              'growth_floor': args.growth_floor,
              'finite_difference_samples': len(fd_ids),
              'finite_difference_minus_autodiff_absolute': finite_differences,
              'groups': {}}
    groups = [('all', np.ones(len(x), bool))]+[
        (r, regions == r) for r in np.unique(regions)]
    for group, mask in groups:
        values = {n: a[mask][:, selected_k] for n, a in residuals.items()}
        report['groups'][group] = {n: summarize(a) for n, a in values.items()}
        for kind, names in [
                ('pk_relative_percent', ['standard_pk', 'sobolev_pk']),
                ('fk_relative_percent', [n+'_relative_percent' for n in [
                    'standard_fk', 'standard_fk_from_pk', 'sobolev_fk']]),
                ('fk_absolute', [n+'_absolute' for n in [
                    'standard_fk', 'standard_fk_from_pk', 'sobolev_fk']]),
                ('consistency_absolute', [
                    'standard_consistency_absolute',
                    'sobolev_consistency_absolute'])]:
            plot_histograms(
                {n: values[n] for n in names},
                args.output/f'{group}_{kind}.png', f'{group}: {kind}')
        plot_k_errors(
            k[selected_k], values, args.output/f'{group}_errors_vs_k.png')
    np.savez_compressed(
        args.output/'predictions.npz', x=x, k=k, pk_truth=ptrue,
        fk_truth=ftrue, file_index=file_ids, row_index=row_ids,
        regions=regions, fd_sample_indices=fd_ids, **predictions)
    (args.output/'summary.json').write_text(
        json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Saved diagnostics for {len(x)} common validation rows '
          f'to {args.output}')


if __name__ == '__main__':
    main()
