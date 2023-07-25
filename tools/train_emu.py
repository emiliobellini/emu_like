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
    # Check if empty, and copy param file to output folder
    if output.is_empty():
        params.copy_to(
            name=de.file_names['params_name'],
            root=params['output'],
            header=de.file_names['params_header'],
            verbose=args.verbose)
    # Else exit, to avoid overwriting
    else:
        raise Exception(
            'Output folder not empty! Exiting to not corrupt precious data!')

    if args.verbose:
        scp.print_level(0, "\nStarting training of Planck emulator\n")

    # Call the right emulator
    emu = Emulator.choose_one(params, output, verbose=args.verbose)

    # Train the emulator
    emu.train(verbose=args.verbose, get_plots=args.get_plots)

    return
