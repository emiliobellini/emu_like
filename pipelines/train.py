"""

Main module with the pipeline used to train the emulator.

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

    # Call the right emulator
    emu = Emulator.choose_one(params['emulator']['type'],
                              verbose=args.verbose)

    # If resume
    if args.resume:
        if args.verbose:
            io.info('Resuming from {}.'.format(params['output']))
            io.print_level(1, 'Ignoring {}'.format(args.params_file))
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

        params['emulator']['params']['sample_n_x'] = sample.n_x
        params['emulator']['params']['sample_n_y'] = sample.n_y
        emu.build(params['emulator']['params'], verbose=args.verbose)

    exit()
    # Save scalers
    scaler_path = io.Folder(params['output']).subfolder(de.file_names['x_scaler']['folder'])
    print(params['output'])
    print(scaler_path.path)
    print(de.file_names['x_scaler']['folder'])
    print(de.file_names['x_scaler']['name'])
    exit()
    # io.Folder()
    scalers = output.subfolder(
        de.file_names['x_scaler']['folder']).create(verbose=args.verbose)
    scaler_x_path = io.File(de.file_names['x_scaler']['name'], root=scalers)
    sample.scaler_x.save(scaler_x_path, verbose=args.verbose)
    scaler_y_path = io.File(de.file_names['y_scaler']['name'], root=scalers)
    sample.scaler_y.save(scaler_y_path, verbose=args.verbose)

    exit()


    # Define output path
    output = io.Folder(path=params['output'])
    if args.resume:
        # Update parameters with new settings
        params.update_params(ref_params,
                             args.additional_epochs,
                             args.learning_rate)
        # Check that the two parameter files are compatible
        params.check_with(ref_params, de.params_to_check, verbose=args.verbose)
    else:
        if args.verbose:
            io.info("Writing output in {}".format(output.path))
        # Check if empty, and copy param file to output folder
        if output.is_empty():
            params.copy_to(
                name=de.file_names['params']['name'],
                root=params['output'],
                header=de.file_names['params']['header'],
                verbose=args.verbose)
        # Else exit, to avoid overwriting
        else:
            raise Exception(
                'Output folder not empty! Exiting to avoid corruption of '
                'precious data! If you want to resume a previous run use '
                'the --resume (-r) option.')

    # Plots
    if args.get_plots:
        sample.get_plots(output, verbose=args.verbose)

    # Call the right emulator
    emu = Emulator.choose_one(params, output, verbose=args.verbose)

    # If resume, load emulator
    if args.resume:
        emu.load(verbose=args.verbose)
        # In YamlFile.update_params this is converted into a list to
        # keep track of the run history
        emu.initial_epoch = emu.get_last_epoch_run() + 1
        emu.total_epochs = params['ffnn_model']['n_epochs'][-1]
        params['ffnn_model']['n_epochs'][-2] = emu.initial_epoch
    # Else, build it
    else:
        emu.build(sample.n_x, sample.n_y, verbose=args.verbose)
        emu.initial_epoch = 0
        emu.total_epochs = params['ffnn_model']['n_epochs']
        # In YamlFile.update_params this is converted into a list to
        # keep track of the run history
        if isinstance(emu.total_epochs, list):
            emu.total_epochs = emu.total_epochs[-1]

    # Train the emulator
    emu.train(sample, verbose=args.verbose, get_plots=args.get_plots)

    # Save the emulator
    emu.save(verbose=args.verbose)

    # Replace parameter file
    if args.resume:
        params.copy_to(
            name=de.file_names['params']['name'],
            root=params['output'],
            header=de.file_names['params']['header'],
            verbose=args.verbose)

    return
