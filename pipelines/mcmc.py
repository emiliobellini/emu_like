"""
.. module:: mcmc

:Synopsis: Pipeline used to run MCMC with an emulator.
:Author: Emilio Bellini

"""

import src.emu_like.io as io
from src.emu_like.mcmc import MCMC
from src.emu_like.params import Params


def mcmc_emu(args):
    """ Run mcmc from the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        io.print_level(0, "\nStarted mcmc\n")

    # Read params
    params = Params().load(args.params_file)

    # Call mcmc sampler
    sampler = MCMC.choose_one(params, args.verbose)

    # Run sampler
    sampler.run()

    return
