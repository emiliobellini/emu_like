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

    # Replace CLI arguments
    if args.epochs is not None:
        params['emulator']['args']['epochs'] = args.epochs
    if args.timeout is not None:
        params['output']['timeout'] = args.timeout

    # Call the right emulator
    emu = Emulator.choose_one(
        params['emulator']['name'],
        verbose=args.verbose)

    # Decide whether to resume or not
    output_is_empty = io.Folder(params['output']['path']).is_empty()
    resume_requested = args.resume_strict or args.resume_warm
    if args.force:
        # Resume with the selected policy only when previous output exists.
        should_resume = not output_is_empty
    elif resume_requested:
        if output_is_empty:
            raise FileNotFoundError(
                'Cannot resume: output folder is empty.'
            )
        should_resume = True
    else:
        if not output_is_empty:
            raise RuntimeError(
                'Output folder is not empty. Select --resume-strict or '
                '--resume-warm, optionally with --force.'
            )
        should_resume = False

    # Checks: if resume_strict or resume_warm, check that
    # the relevant files exist in the output folder and
    # the parameters are consistent.
    if should_resume:
        emu.check_files(
            params['output']['path'],
            params['datasets']['paths'],
            verbose=args.verbose
        )
        emu.check_parameters(
            params,
            resume_strict=args.resume_strict,
            resume_warm=args.resume_warm,
            verbose=args.verbose
        )
    else:
        # Save the parameters to the output folder
        emu.save_parameters(
            params['output']['path'],
            params,
            verbose=args.verbose,
        )

    # Print init messages
    if args.verbose:
        if should_resume and args.resume_strict:
            io.info('Resuming emulator in strict mode from {}.'.format(
                params['output']['path']))
        elif should_resume and args.resume_warm:
            io.info('Resuming emulator in warm mode from {}.'.format(
                params['output']['path']))
        else:
            io.info('Starting training emulator.')

    # Local variables
    pars_out = params['output']
    pars_emu = params['emulator']
    pars_dat = params['datasets']

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
    if should_resume:
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

    # Train the emulator
    emu.train(
        data,
        pars_emu['args']['epochs'],
        pars_emu['args']['learning_rate'],
        patience=pars_emu['args']['patience'],
        path=pars_out['path'],
        timeout=timeout,
        reduce_learning_rate=pars_emu['args']['reduce_learning_rate'],
        relative_improvement=pars_emu['args']['relative_improvement'],
        get_plots=False,
        verbose=args.verbose)

    return
