"""
.. module:: mcmc

:Synopsis: Pipeline used to run MCMC with an emulator.
:Author: Emilio Bellini

"""

import emu_like.io as io
from emu_like.mcmc import MCMC


def mcmc_emu(args):
    """ Run mcmc from the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        io.print_level(0, "\nStarted mcmc\n")

    # Read params
    params = io.YamlFile(args.params_file).read()

    # Call mcmc sampler
    sampler = MCMC.choose_one(params, args.verbose)

    # Run sampler
    sampler.run()

    return
