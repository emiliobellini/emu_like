"""
.. module:: train

:Synopsis: Pipeline used to train an emulator.
:Author: Emilio Bellini

"""

import emu_like.defaults as de
import emu_like.io as io
from emu_like.emu import Emulator
from emu_like.params import Params
from emu_like.datasets import Dataset


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

    # Load all Samples in a single list
    if params['datasets']['paths']:
        paths_x = params['datasets']['paths']
        paths_y = [None for x in params['datasets']['paths']]
    else:
        paths_x = params['datasets']['paths_x']
        paths_y = params['datasets']['paths_y']

    # Read or default
    try:
        columns_x = params['datasets']['args']['columns_x']
    except KeyError:
        columns_x = None
    try:
        columns_y = params['datasets']['args']['columns_y']
    except KeyError:
        columns_y = None
    try:
        remove_non_finite = params['datasets']['args']['remove_non_finite']
    except KeyError:
        remove_non_finite = False

    data = [Dataset().load(
        path=path_x,
        name=params['datasets']['name'],
        path_y=path_y,
        columns_x=columns_x,
        columns_y=columns_y,
        remove_non_finite=remove_non_finite,
        verbose=True)
        for path_x, path_y in zip(paths_x, paths_y)]

    # Join all samples
    data = Dataset.join(data, verbose=args.verbose)

    # Split training and testing samples
    data.train_test_split(
        params['datasets']['args']['frac_train'],
        params['datasets']['args']['train_test_random_seed'],
        verbose=args.verbose)

    # If requested, rescale training and testing samples
    data.rescale(
        params['datasets']['args']['rescale_x'],
        params['datasets']['args']['rescale_y'],
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
        params['emulator']['args']['sample_n_x'] = data.n_x
        params['emulator']['args']['sample_n_y'] = data.n_y
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
