"""
.. module:: train

:Synopsis: Pipeline used to train an emulator.
:Author: Emilio Bellini

"""

import src.emu_like.defaults as de
import src.emu_like.io as io
from src.emu_like.emu import Emulator
from src.emu_like.params import Params
from src.emu_like.sample import Sample


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
    emu = Emulator.choose_one(params['emulator']['type'],
                              verbose=args.verbose)

    # Update parameters with input
    params = emu._update_params(params,
                                add_epochs=args.additional_epochs,
                                learning_rate=args.learning_rate)

    # Save params
    params.save(
        de.file_names['params']['name'],
        root=params['output'],
        header=de.file_names['params']['header'],
        verbose=args.verbose)

    # Get default values
    try:
        columns_x = params['training_sample']['columns_x']
    except KeyError:
        columns_x = slice(None)
    try:
        columns_y = params['training_sample']['columns_y']
    except KeyError:
        columns_y = slice(None)
    try:
        remove_non_finite = params['training_sample']['remove_non_finite']
    except KeyError:
        remove_non_finite = False

    # Load all Samples in a single list
    try:
        sample_paths = params['training_sample']['paths_x']
        sample_paths_y = params['training_sample']['paths_y']
    except KeyError:
        sample_paths = params['training_sample']['paths']
        sample_paths_y = [None for x in params['training_sample']['paths']]
    samples = [Sample().load(
                    path=path,
                    path_y=path_y,
                    columns_x=columns_x,
                    columns_y=columns_y,
                    remove_non_finite=remove_non_finite,
                    verbose=False)
               for path, path_y in zip(sample_paths, sample_paths_y)]
    # Join all samples
    sample = Sample.join(samples, verbose=args.verbose)

    # Split training and testing samples
    sample.train_test_split(
        params['training_sample']['frac_train'],
        params['training_sample']['train_test_random_seed'],
        verbose=args.verbose)

    # If requested, rescale training and testing samples
    sample.rescale(
        params['training_sample']['rescale_x'],
        params['training_sample']['rescale_y'],
        verbose=args.verbose)
    # Save scalers
    sample.scaler_x.save(de.file_names['x_scaler']['name'],
                         root=params['output'],
                         verbose=args.verbose)
    sample.scaler_y.save(de.file_names['y_scaler']['name'],
                         root=params['output'],
                         verbose=args.verbose)

    # If resume
    if args.resume:
        # Load emulator
        emu.load(params['output'], model_to_load='best', verbose=args.verbose)
    # Otherwise
    else:
        params['emulator']['params']['sample_n_x'] = sample.n_x
        params['emulator']['params']['sample_n_y'] = sample.n_y
        # Build architecture
        emu.build(params['emulator']['params'], verbose=args.verbose)

    # Train the emulator
    emu.train(
        sample,
        params['emulator']['params']['epochs'],
        params['emulator']['params']['learning_rate'],
        params['emulator']['params']['batch_size'],
        path=params['output'],
        get_plot=True,
        verbose=args.verbose)

    # Save emulator
    emu.save(params['output'], verbose=args.verbose)

    return
