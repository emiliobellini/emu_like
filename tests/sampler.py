import argparse
import numpy as np
import tqdm
import emu_like.io as io
from emu_like.y_models import YModel

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_file', type=str)
    parser.add_argument('--n_rands', '-n', type=int, default=1000)
    parser.add_argument('--threshold', '-t', type=float, default=1e-16)
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
    mask = rng.choice(n_samples, size=args.n_rands, replace=False)
    io.info('Testing on samples: {}\n'.format(mask))

    # Init y_model
    y_model = YModel.choose_one(
        name,
        params,
        outputs,
        n_samples,
        **class_args)
    spectra = list(y_model.outputs.keys())

    y_data = {sp: fits.get_data(sp)[mask] for sp in spectra}
    has_zeros = {sp: np.any(y_data[sp] == 0.) for sp in spectra}
    diff = {sp: np.zeros_like(y_data[sp]) for sp in spectra}

    # Start iteration
    io.info('Evaluating models ...')
    for nx, x in enumerate(tqdm.tqdm(x_data[mask])):
        # Evaluate model
        y_one_line = y_model.evaluate(x, nx)
        # Get diff
        for nsp, sp in enumerate(spectra):
            if has_zeros[sp]:
                diff[sp][nx] = y_one_line[nsp][0] - y_data[sp][nx]
            else:
                diff[sp][nx] = y_one_line[nsp][0]/y_data[sp][nx] -1.
    
    # Print results
    io.print_level(0, 'Results:')
    for sp in spectra:
        abs_diff = np.abs(diff[sp])
        more_than_threshold = np.any(abs_diff>args.threshold, axis=1)
        max_diff = np.max(abs_diff, axis=1)
        if np.any(more_than_threshold):
            io.warning('Difference in {} exceeds the threshold of {} for parameters:'.format(sp, args.threshold))
            for x, max_diff in zip(x_data[mask][more_than_threshold], max_diff[more_than_threshold]):
                io.print_level(1, '{}. Max diff: {}'.format(x, max_diff))
        else:
            io.info('Success! Spectrum {} is ok!'.format(sp))
