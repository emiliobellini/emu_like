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

    # Force computation
    if args.force:
        if io.Folder(params['output']['path']).is_empty():
            args.resume = False
        else:
            args.resume = True

    # If resume load parameters from output folder
    if args.resume:
        if args.verbose:
            io.info('Resuming from {}.'.format(params['output']['path']))
            io.print_level(1, 'Ignoring {}'.format(args.params_file))
        # Read params from output folder
        params = io.YamlFile(root=params['output']['path']).read()
    # Otherwise
    else:
        # Check if output folder is empty, otherwise stop
        if io.Folder(params['output']['path']).is_empty():
            if args.verbose:
                io.info("Writing output in {}".format(
                    params['output']['path']))
            # Save params
            params.write(
                root=params['output']['path'],
                verbose=args.verbose)
        else:
            raise Exception(
                'Output folder not empty! Exiting to avoid corruption of '
                'precious data! If you want to resume a previous run use '
                'the --resume (-r) option.')

    # Local variables
    pars_out = params['output']
    pars_emu = params['emulator']
    pars_dat = params['datasets']

    # Call the right emulator
    emu = Emulator.choose_one(
        pars_emu['name'],
        verbose=args.verbose)

    # Test datasets input paths
    has_paths = pars_dat['paths'] is not None
    if has_paths:
        try:
            all([io.FitsFile(p) for p in pars_dat['paths']])
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
            name=pars_dat['name'],
            columns_x=pars_dat['columns_x'],
            columns_y=pars_dat['columns_y'],
            verbose=False)
            for path in pars_dat['paths']]
    # 2) unique text files for x and y
    elif has_paths:
        data = [Dataset().load_external(
            path=path,
            columns_x=pars_dat['columns_x'],
            columns_y=pars_dat['columns_y'],
            verbose=False)
            for path in pars_dat['paths']]
    # 3) separate text files for x and y
    else:
        data = [Dataset().load_external(
            path=path_x,
            path_y=path_y,
            columns_x=pars_dat['columns_x'],
            columns_y=pars_dat['columns_y'],
            verbose=False)
            for path_x, path_y in zip(
                pars_dat['paths_x'], pars_dat['paths_y'])]

    # Remove non finite "y"
    if pars_dat['remove_non_finite']:
        data = [d.remove_non_finite(verbose=False) for d in data]

    # Print info
    if args.verbose:
        io.info('Datasets arguments')
        io.print_level(1, 'Name: {}.'.format(pars_dat['name']))
        io.print_level(1, 'Sliced x data with columns: {}.'.format(
            pars_dat['columns_x']))
        io.print_level(1, 'Sliced y data with columns: {}.'.format(
            pars_dat['columns_y']))
        if pars_dat['remove_non_finite']:
            io.print_level(1, 'Removing non finite y from dataset.')

    # Join all datasets
    data = Dataset.join(data, verbose=args.verbose)

    # Split training and testing samples
    data.train_test_split(
        pars_dat['frac_train'],
        pars_dat['train_test_random_seed'],
        verbose=args.verbose)

    # If requested, rescale training and testing samples
    data.rescale(
        pars_dat['rescale_x'],
        pars_dat['rescale_y'],
        verbose=args.verbose)

    # If requested apply PCA on x and/or y
    data.apply_pca(
        pars_dat['num_x_pca'],
        pars_dat['num_y_pca'],
        verbose=args.verbose)

    # If resume
    if args.resume:
        # Load emulator
        emu.load(pars_out['path'], model_to_load='best', verbose=args.verbose)
    # Otherwise
    else:
        # Get dimensions of x and y for emulator
        pars_emu['args']['data_n_x'] = data.x_train.shape[1]
        pars_emu['args']['data_n_y'] = data.y_train.shape[1]
        # Build architecture
        emu.build(
            pars_emu['args'],
            data=data,
            verbose=args.verbose)

    # Default output parameters
    try:
        timeout = pars_out['timeout']
        if args.verbose:
            io.info('Time limit for execution is {} hours'.format(timeout))
    except KeyError:
        timeout = None

    # Update number of epochs to run
    if args.resume and args.additional_epochs < 0:
        epochs = max(pars_emu['args']['epochs'] - emu.epochs[-1], 0)
    elif args.resume and args.additional_epochs > 0:
        epochs = args.additional_epochs
    elif args.additional_epochs < 0:
        epochs = pars_emu['args']['epochs']
    else:
        epochs = pars_emu['args']['epochs'] + args.additional_epochs

    # Update initial learning rate
    if args.resume and args.learning_rate < 0:
        learning_rate = emu.learning_rate[-1]
    elif args.learning_rate > 0:
        learning_rate = args.learning_rate
    else:
        learning_rate = pars_emu['args']['learning_rate']

    # Train the emulator
    emu.train(
        data,
        epochs,
        learning_rate,
        patience=pars_emu['args']['patience'],
        path=pars_out['path'],
        timeout=timeout,
        reduce_learning_rate=pars_emu['args']['reduce_learning_rate'],
        get_plots=True,
        verbose=args.verbose)

    return
