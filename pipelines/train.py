"""

Main module with the pipeline used to train the emulator.

"""
import tools.defaults as de
import tools.io as io
import tools.printing_scripts as scp
from tools.emu import Emulator
from tools.sample import Sample


def train_emu(args):
    """ Train the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        scp.print_level(0, "\nStarted training emulator\n")

    # Load input file
    params = io.YamlFile(args.params_file, should_exist=True)
    params.read()

    # Define output path
    output = io.Folder(path=params['output'])
    if args.resume:
        if args.verbose:
            scp.info('Resuming from {}.'.format(output.path))
        ref_params = io.YamlFile(
            de.file_names['params']['name'],
            root=output,
            should_exist=True)
        ref_params.read()
        # Update parameters with new settings
        params.update_params(ref_params,
                             args.additional_epochs,
                             args.learning_rate)
        # Check that the two parameter files are compatible
        params.check_with(ref_params, de.params_to_check, verbose=args.verbose)
    else:
        if args.verbose:
            scp.info("Writing output in {}".format(output.path))
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

    # Load sample
    sample = Sample()
    sample.load(params=params['training_sample'], verbose=args.verbose)

    # Save details in output folder
    details_path = io.YamlFile(
        de.file_names['sample_details']['name'],
        root=output.subfolder(
            de.file_names['sample_details']['folder']).create(
                verbose=args.verbose)
    )
    sample.save_details(details_path, verbose=args.verbose)

    # Split training and testing samples
    sample.train_test_split(
        params['frac_train'],
        params['train_test_random_seed'],
        verbose=args.verbose)

    # If requested, rescale training and testing samples
    sample.rescale(
        params['rescale_x'],
        params['rescale_y'],
        verbose=args.verbose)

    # Save scalers
    scalers = output.subfolder(
        de.file_names['x_scaler']['folder']).create(verbose=args.verbose)
    scaler_x_path = io.File(de.file_names['x_scaler']['name'], root=scalers)
    sample.scaler_x.save(scaler_x_path, verbose=args.verbose)
    scaler_y_path = io.File(de.file_names['y_scaler']['name'], root=scalers)
    sample.scaler_y.save(scaler_y_path, verbose=args.verbose)

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
