import argparse
from tabulate import tabulate
from emu_like.datasets import Dataset


def parser():
    parser = argparse.ArgumentParser()
    # Files to inspect
    parser.add_argument(
        '--files',
        '-f',
        type=str,
        nargs='+',
        help='Fits files to inspect.')
    # Other options
    parser.add_argument(
        '--keep-non-finite',
        '-knf',
        action='store_true',
        help='Keep non-finite values in the dataset.')
    parser.add_argument(
        '--frac-train',
        '-ft',
        type=float,
        default=0.9,
        help='Fraction of data to use for training.')
    parser.add_argument(
        '--train-test-random-seed',
        '-s',
        type=int,
        default=1543,
        help='Random seed for train-test split.')
    parser.add_argument(
        '--verbose', '-v',
        help='Verbose (default: False)',
        action='store_true')

    args = parser.parse_args()
    return args


def get_model_name(files):
    choice = files[0].split('/')
    # Index of sample
    try:
        idx = choice.index('sample')
    except ValueError:
        idx = None

    if idx is None:
        # Remove ..
        choice = [x for x in choice if x != '..']
        # Join
        model = '_'.join(choice)
    else:
        model = choice[idx - 1]

    return model


def get_table(model_name, bounds):
    print(f'\nModel: {model_name}')
    headers = ['Spectrum', 'Train min', 'Train max', 'Test min', 'Test max']
    rows = []
    for spectrum, bound in bounds.items():
        train_min, train_max = bound['train']
        test_min, test_max = bound['test']
        rows.append([
            spectrum,
            f'{train_min:.3e}',
            f'{train_max:.3e}',
            f'{test_min:.3e}',
            f'{test_max:.3e}'])

    return tabulate(rows, headers=headers)


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    spectra_names = [
        'pk_m',
        'pk_cb',
        'pk_weyl',
        'fk_m',
        'fk_cb',
        'fk_weyl',
        'cl_TT_lensed',
        'cl_TE_lensed',
        'cl_EE_lensed',
        'cl_pp_lensed',
        'cl_Tp_lensed',
        'cl_BB_lensed'
    ]

    # Parse command line arguments
    args = parser()

    # Get model
    model_name = get_model_name(args.files)

    # Preliminary checks
    if args.files is None:
        raise ValueError(
            'No fits files provided. Use --files to specify files.')

    bounds = {}
    for spectrum in spectra_names:

        print(f'\nInspecting spectrum: {spectrum}')
        # Load data
        try:
            data = [Dataset().load(
                path=file,
                name=spectrum) for file in args.files]
        except KeyError:
            print(f'Error: Spectrum "{spectrum}" not found in one '
                  'of the files.')
            continue

        bounds[spectrum] = {'train': (None, None), 'test': (None, None)}

        # Remove non finite "y"
        if not args.keep_non_finite:
            data = [d.remove_non_finite(verbose=False) for d in data]

        # Join all datasets
        data = Dataset.join(data, verbose=True)

        # Split training and testing samples
        data.train_test_split(
            args.frac_train,
            args.train_test_random_seed,
            verbose=args.verbose)

        bounds[spectrum]['train'] = (data.y_train.min(), data.y_train.max())
        bounds[spectrum]['test'] = (data.y_test.min(), data.y_test.max())

    print(get_table(model_name, bounds))
