"""

Main module with the pipeline used to train the emulator.

"""
import tools.defaults as de
import tools.io as io
import tools.printing_scripts as scp
from tools.emu import Emulator


def train_emu(args):
    """ Train the emulator.

    Args:
        args: the arguments read by the parser.


    """
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
        # TODO: here if we change the emulator would not work
        # Take the number of epochs from the reference file
        params['ffnn_model']['n_epochs'] = ref_params['ffnn_model']['n_epochs']
        params['ffnn_model']['additional_epochs'] = args.resume
        # Check that the two parameter files are compatible
        params.check_with(ref_params, de.params_to_check, verbose=args.verbose)
    else:
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

    if args.verbose:
        scp.print_level(0, "\nStarting training of Planck emulator\n")

    # Call the right emulator
    emu = Emulator.choose_one(params, output, verbose=args.verbose)

    # Train the emulator
    emu.train(
        resume=args.resume,
        verbose=args.verbose,
        get_plots=args.get_plots)

    # Update the epochs run
    if args.resume:
        # TODO: here if we change the emulator would not work
        params['ffnn_model']['n_epochs'] += args.resume
        params['ffnn_model'].pop('additional_epochs')
        params.copy_to(
            name=de.file_names['params']['name'],
            root=params['output'],
            header=de.file_names['params']['header'],
            verbose=args.verbose)

    return
