import argparse
import classy
import numpy as np
import scipy.interpolate as interp
import tqdm
import emu_like.io as io
from emu_like.y_models import YModel


def _compare_reference_spectra(y_model, threshold):
    """Recompute reference spectra and compare with the model-stored ones."""
    spectra = y_model.spectra

    if any(sp.is_pk for sp in spectra):
        z_max = {'z_max_pk': y_model._get_z_max()}
    else:
        z_max = {}

    cosmo_ref = classy.Class()
    ref_params = y_model.ref_params | z_max
    cosmo_ref.set(ref_params)
    cosmo_ref.compute()

    y_ref_calc = [sp.get(cosmo_ref, z=None)[np.newaxis] for sp in spectra]
    for nsp, sp in enumerate(spectra):
        if not sp.ratio:
            y_ref_calc[nsp] = np.ones_like(y_ref_calc[nsp])

    io.print_level(0, 'Reference spectra check:')
    for nsp, sp in enumerate(spectra):
        den = np.maximum(np.abs(y_model.y_ref[nsp]), 1e-300)
        rel = np.abs((y_ref_calc[nsp] - y_model.y_ref[nsp]) / den)
        max_rel = np.max(rel)
        if max_rel > threshold:
            io.warning(
                'Reference spectrum {} differs (max rel diff: {:.2e})'.format(
                    sp.name, max_rel))
        else:
            io.info(
                'Reference spectrum {} ok (max rel diff: {:.2e})'.format(
                    sp.name, max_rel))
    io.info('')


def _evaluate_with_class(y_model, x, cosmo, class_params):
    """Evaluate spectra using direct CLASS calls (without y_model.evaluate)."""
    spectra = y_model.spectra
    for npar, par in enumerate(y_model.x_names):
        class_params[par] = x[npar]

    z_eval = 0.0
    if any(sp.is_pk for sp in spectra):
        class_params['z_max_pk'] = 0.1
        if 'z_pk' in class_params:
            class_params['z_max_pk'] = max(
                class_params['z_pk'],
                class_params['z_max_pk'])
            z_eval = class_params['z_pk']

    try:
        cosmo.set(class_params)
        cosmo.compute()
    except (classy.CosmoComputationError, classy.CosmoSevereError):
        return {
            sp.name: np.full((y_model.n_y[nsp],), np.nan)
            for nsp, sp in enumerate(spectra)
        }

    by_name = {}
    for nsp, sp in enumerate(spectra):
        if sp.is_cl:
            den = y_model.y_ref[nsp][0]
            num = sp.get(cosmo)
        else:
            den = interp.make_splrep(
                y_model.z_array,
                y_model.y_ref[nsp][0].T,
                s=0)(z_eval).T
            num = sp.get(cosmo, z=z_eval)
        by_name[sp.name] = num / den
    return by_name


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_file', type=str)
    parser.add_argument('--n_rands', '-n', type=int, default=10)
    parser.add_argument('--threshold', '-t', type=float, default=1e-15)
    parser.add_argument(
        '--evaluator',
        choices=['class', 'ymodel'],
        default='class',
        help='Evaluation backend: direct CLASS calls or y_model.evaluate.')
    args = parser.parse_args()

    # Get data and relevant attributes
    fits = io.FitsFile(args.sample_file)
    hd = fits.get_header(0)
    x_data = fits.get_data('x_data')
    n_samples = x_data.shape[0]
    name = hd['y_model']['name']
    class_args = hd['y_model']['args']
    outputs = hd['y_model']['outputs']
    params = hd['params']

    io.print_level(
        0, 'Randomly selecting {} samples over {} to check that the '
        'output freshly\ncalculated here does not differ more than {} '
        'w.r.t. the one stored in\n{}\n'.format(
            args.n_rands, n_samples, args.threshold, args.sample_file))

    io.info('Varied parameters: {}'.format(list(params.keys())))

    # Get random points
    rng = np.random.default_rng()
    n_rands = min(args.n_rands, n_samples)
    if n_rands < args.n_rands:
        io.warning(
            'Requested {} random points but only {} samples are available. '
            'Using {} points.'.format(args.n_rands, n_samples, n_rands))
    mask = rng.choice(n_samples, size=n_rands, replace=False)
    io.info('Testing on samples: {}\n'.format(mask))

    # Init y_model
    y_model = YModel.choose_one(
        name,
        params,
        outputs,
        n_samples,
        **class_args)
    spectra = list(y_model.outputs.keys())

    if args.evaluator == 'class' and name != 'class_spectra':
        raise ValueError(
            'Evaluator "class" is only supported for y_model="class_spectra". '
            'Use --evaluator ymodel for {}.'.format(name))

    if args.evaluator == 'class':
        _compare_reference_spectra(y_model, args.threshold)
        cosmo = classy.Class()
        class_params = dict(y_model.class_params)
    else:
        io.info(
            'Skipping explicit reference spectra check with '
            'evaluator "ymodel".\n')

    spectra_idx = {sp.name: nsp for nsp, sp in enumerate(y_model.spectra)}

    y_data = {sp: fits.get_data(sp)[mask] for sp in spectra}
    has_zeros = {sp: np.any(y_data[sp] == 0.) for sp in spectra}
    diff = {sp: np.zeros_like(y_data[sp]) for sp in spectra}

    # Start iteration
    io.info('Evaluating models ...')
    for nx, x in enumerate(tqdm.tqdm(x_data[mask])):
        if args.evaluator == 'class':
            y_eval_by_name = _evaluate_with_class(
                y_model, x, cosmo, class_params)
        else:
            y_one_line = y_model.evaluate(x, nx)
            y_eval_by_name = {
                sp_name: y_one_line[spectra_idx[sp_name]][0]
                for sp_name in spectra
            }

        # Get diff
        for sp in spectra:
            if has_zeros[sp]:
                diff[sp][nx] = y_eval_by_name[sp] - y_data[sp][nx]
            else:
                diff[sp][nx] = y_eval_by_name[sp] / y_data[sp][nx] - 1.

    # Print results
    io.print_level(0, 'Results:')
    for sp in spectra:
        abs_diff = np.abs(diff[sp])
        more_than_threshold = np.any(abs_diff > args.threshold, axis=1)
        max_diff = np.max(abs_diff, axis=1)
        if np.any(more_than_threshold):
            io.warning(
                'Difference in {} exceeds the threshold of {} '
                'for parameters:'.format(sp, args.threshold))
            for x, max_diff in zip(
                    x_data[mask][more_than_threshold],
                    max_diff[more_than_threshold]):
                io.print_level(1, '{}. Max diff: {}'.format(x, max_diff))
        else:
            io.info('Success! Spectrum {} is ok!'.format(sp))
