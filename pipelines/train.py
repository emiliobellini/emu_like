"""
.. module:: train

:Synopsis: Pipeline used to train an emulator.
:Author: Emilio Bellini

"""

import emu_like.io as io
from emu_like.emu import Emulator
from emu_like.datasets import Dataset


def train_emu(args):
    """ Train the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        io.print_level(0, "\nStarted training emulator\n")

    # Read params
    params = io.YamlFile(args.params_file).read()

    # If resume load parameters from output folder
    if args.resume:
        if args.verbose:
            io.info('Resuming from {}.'.format(params['output']))
            io.print_level(1, 'Ignoring {}'.format(args.params_file))
        # Read params from output folder
        params = io.YamlFile(root=params['output']).read()
    # Otherwise
    else:
        # Check if output folder is empty, otherwise stop
        if io.Folder(params['output']).is_empty():
            if args.verbose:
                io.info("Writing output in {}".format(params['output']))
        else:
            raise Exception(
                'Output folder not empty! Exiting to avoid corruption of '
                'precious data! If you want to resume a previous run use '
                'the --resume (-r) option.')

    # Call the right emulator
    emu = Emulator.choose_one(
        params['emulator']['name'],
        verbose=args.verbose)

    # Update parameters with input
    params = emu.update_params(
        params,
        epochs=args.additional_epochs,
        learning_rate=args.learning_rate)

    # Test datasets input paths
    has_paths = params['datasets']['paths'] is not None
    if has_paths:
        try:
            all([io.FitsFile(p) for p in params['datasets']['paths']])
            paths_is_fits = True
        except ValueError:
            paths_is_fits = False
    else:
        paths_is_fits = False

    # Load datasets
    # 1) fits files created by this code
    if has_paths and paths_is_fits:
        data = [Dataset().load(
            path=path,
            name=params['datasets']['name'],
            columns_x=params['datasets']['columns_x'],
            columns_y=params['datasets']['columns_y'],
            verbose=False)
        for path in params['datasets']['paths']]
    # 2) unique text files for x and y
    elif has_paths:
        data = [Dataset().load_external(
            path=path,
            columns_x=params['datasets']['columns_x'],
            columns_y=params['datasets']['columns_y'],
            verbose=False)
            for path in params['datasets']['paths']]
    # 3) separate text files for x and y
    else:
        data = [Dataset().load_external(
            path=path_x,
            path_y=path_y,
            columns_x=params['datasets']['columns_x'],
            columns_y=params['datasets']['columns_y'],
            verbose=False)
            for path_x, path_y in zip(
                params['datasets']['paths_x'], params['datasets']['paths_y'])]

    # Remove non finite "y"
    if params['datasets']['remove_non_finite']:
        data = [d.remove_non_finite(verbose=False) for d in data]

    # Print info
    if args.verbose:
        io.info('Datasets arguments')
        io.print_level(1, 'Name: {}.'.format(params['datasets']['name']))
        io.print_level(1, 'Sliced x data with columns: {}.'.format(
            params['datasets']['columns_x']))
        io.print_level(1, 'Sliced y data with columns: {}.'.format(
            params['datasets']['columns_y']))
        if params['datasets']['remove_non_finite']:
            io.print_level(1, 'Removing non finite y from dataset.')

    # Join all datasets
    data = Dataset.join(data, verbose=args.verbose)

    # Split training and testing samples
    data.train_test_split(
        params['datasets']['frac_train'],
        params['datasets']['train_test_random_seed'],
        verbose=args.verbose)

    # If requested, rescale training and testing samples
    data.rescale(
        params['datasets']['rescale_x'],
        params['datasets']['rescale_y'],
        verbose=args.verbose)
    
    # If requested apply PCA on x and/or y
    data.apply_pca(
        params['datasets']['num_x_pca'],
        params['datasets']['num_y_pca'],
        verbose=args.verbose)

    # If resume
    if args.resume:
        # Load emulator
        emu.load(params['output'], model_to_load='best', verbose=args.verbose)
    # Otherwise
    else:
        # Get dimensions of x and y for emulator
        params['emulator']['args']['data_n_x'] = data.x_train.shape[1]
        params['emulator']['args']['data_n_y'] = data.y_train.shape[1]
        # Build architecture
        emu.build(params['emulator']['args'], verbose=args.verbose)

    # Train the emulator
    emu.train(
        data,
        params['emulator']['args']['epochs'],
        params['emulator']['args']['learning_rate'],
        patience=params['emulator']['args']['patience'],
        path=params['output'],
        get_plots=True,
        verbose=args.verbose)

    return
