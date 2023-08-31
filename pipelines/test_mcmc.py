import tools.defaults as de
import tools.io as io
import tools.printing_scripts as scp
from tools.emu import Emulator


def test_mcmc_emu(args):
    """ Test the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        scp.print_level(0, "\nStarted testing emulated mcmc\n")

    # Load input file
    params = io.YamlFile(args.params_file, should_exist=True)
    params.read()

    # Load params emulator
    emu_params = io.YamlFile(
        de.file_names['params']['name'],
        root=params['emulator']['path'],
        should_exist=True)
    emu_params.read()

    # Define emulator folder
    emu_folder = io.Folder(path=params['emulator']['path'])

    # Call the right emulator
    emu = Emulator.choose_one(emu_params, emu_folder, verbose=args.verbose)

    # Load emulator
    emu.load(model_to_load=params['emulator']['epoch'], verbose=args.verbose)

    print(emu)

    return
