"""
.. module:: sample

:Synopsis: Pipeline used to generate datasets.
:Author: Emilio Bellini

"""

import emu_like.io as io
from emu_like.params import Params
from emu_like.datasets import Dataset, DataCollection

def sample_emu(args):
    """ Generate the dataset for the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        io.print_level(0, '\nGetting dataset for Emulator\n')

    # Init DataCollection object
    data = DataCollection()

    # Read params
    params = Params().load(args.params_file)

    # If resume
    if args.resume:
        if args.verbose:
            io.info('Resuming from {}.'.format(params['output']))
            io.print_level(1, 'Ignoring {}'.format(args.params_file))

        # Resume the dataset
        data.resume(
            params['output'],
            verbose=args.verbose)
    # Otherwise
    else:
        data.sample(
            params=params['params'],
            x_name=params['x_sampler']['name'],
            x_args=params['x_sampler']['args'],
            y_name=params['y_model']['name'],
            y_args=params['y_model']['args'],
            y_outputs=params['y_model']['outputs'],
            output=params['output'],
            verbose=args.verbose)

    return
