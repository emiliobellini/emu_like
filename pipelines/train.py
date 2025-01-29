"""
.. module:: train

:Synopsis: Pipeline used to train an emulator.
:Author: Emilio Bellini

"""

import os
import emu_like.defaults as de
import emu_like.io as io
from emu_like.emu import Emulator
from emu_like.params import Params
from emu_like.datasets import DataCollection, Dataset


def train_emu(args):
    """ Train the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        io.print_level(0, "\nStarted training emulator\n")

    # Read params
    params = Params().load(args.params_file)

    # If resume load parameters from output folder
    if args.resume:
        if args.verbose:
            io.info('Resuming from {}.'.format(params['output']))
            io.print_level(1, 'Ignoring {}'.format(args.params_file))
        # Read params from output folder
        params = Params().load(de.file_names['params']['name'],
                               root=params['output'])
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
        # Create output folder
        io.Folder(params['output']).create(args.verbose)

    # Call the right emulator
    emu = Emulator.choose_one(
        params['emulator']['name'],
        verbose=args.verbose)

    # Fill missing entries
    params = emu.fill_missing_params(params)

    # Update parameters with input
    params = emu.update_params(
        params,
        epochs=args.additional_epochs,
        learning_rate=args.learning_rate)

    # Save params
    params.save(
        de.file_names['params']['name'],
        root=params['output'],
        header=de.file_names['params']['header'],
        verbose=args.verbose)

    # Test datasets input paths
    has_paths = params['datasets']['paths'] is not None
    has_paths_x = params['datasets']['paths_x'] is not None
    has_paths_y = params['datasets']['paths_y'] is not None
    if has_paths:
        paths_is_dir = all([
            os.path.isdir(p) for p in params['datasets']['paths']])
        paths_is_file = all([
            os.path.isfile(p) for p in params['datasets']['paths']])
    else:
        paths_is_dir = False
        paths_is_file = False
    if has_paths_x:
        paths_x_is_file = all([
            os.path.isfile(p) for p in params['datasets']['paths_x']])
    else:
        paths_x_is_file = False
    if has_paths_y:
        paths_y_is_file = all([
            os.path.isfile(p) for p in params['datasets']['paths_y']])
    else:
        paths_y_is_file = False

    # Load datasets
    # 1) folders created by this code
    if paths_is_dir:
        data = [DataCollection().load(
            path=path,
            one_y_name=params['datasets']['name'],
            verbose=False)
            for path in params['datasets']['paths']]
        # Get Dataset from DataCollection
        data = [d.one_y_dataset for d in data]
        # Slice data
        data = [d.slice(params['datasets']['columns_x'],
                        params['datasets']['columns_y'],
                        verbose=False) for d in data]
    # 2) unique files for both x and y
    elif paths_is_file:
        data = [Dataset().load(
            path=path,
            columns_x=params['datasets']['columns_x'],
            columns_y=params['datasets']['columns_y'],
            verbose=False)
            for path in params['datasets']['paths']]
    # 3) separate files for both x and y
    elif paths_x_is_file and paths_y_is_file:
        data = [Dataset().load(
            path=path_x,
            path_y=path_y,
            columns_x=params['datasets']['columns_x'],
            columns_y=params['datasets']['columns_y'],
            verbose=False)
            for path_x, path_y in zip(
                params['datasets']['paths_x'], params['datasets']['paths_y'])]
    else:
        raise Exception('Something is wrong with the paths you specified!')

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
    # Save scalers
    data.x_scaler.save(
        de.file_names['x_scaler']['name'],
        root=params['output'],
        verbose=args.verbose)
    data.y_scaler.save(
        de.file_names['y_scaler']['name'],
        root=params['output'],
        verbose=args.verbose)

    # If resume
    if args.resume:
        # Load emulator
        emu.load(params['output'], model_to_load='best', verbose=args.verbose)
    # Otherwise
    else:
        params['emulator']['args']['data_n_x'] = data.n_x
        params['emulator']['args']['data_n_y'] = data.n_y
        # Build architecture
        emu.build(params['emulator']['args'], verbose=args.verbose)

    # Train the emulator
    emu.train(
        data,
        params['emulator']['args']['epochs'],
        params['emulator']['args']['learning_rate'],
        path=params['output'],
        get_plot=True,
        verbose=args.verbose)

    # Save emulator
    emu.save(params['output'], verbose=args.verbose)

    return
