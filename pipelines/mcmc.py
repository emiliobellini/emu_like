import src.emu_like.io as io
import src.emu_like.printing_scripts as scp
from src.emu_like.mcmc import MCMC


def mcmc_emu(args):
    """ Run mcmc from the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        scp.print_level(0, "\nStarted mcmc\n")

    # Load input file
    params = io.YamlFile(args.params_file, should_exist=True)
    params.read()

    # Call mcmc sampler
    sampler = MCMC.choose_one(params, args.verbose)

    # Run sampler
    sampler.run()

    return
