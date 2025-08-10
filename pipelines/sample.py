"""
.. module:: sample

:Synopsis: Pipeline used to generate datasets.
:Author: Emilio Bellini

"""

import emu_like.defaults as de
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
    # Fill missing entries
    params = Dataset.fill_missing_params(params)

    # If resume
    if args.resume:
        print('not implemented yet')
        # if args.verbose:
        #     io.info('Resuming from {}.'.format(params['output']))
        #     io.print_level(1, 'Ignoring {}'.format(args.params_file))
        # # Read params from output folder
        # params = Params().load(de.file_names['params']['name'],
        #                        root=params['output'])

        # # Resume the dataset
        # data.resume(
        #     params['output'],
        #     load_minimal=True,
        #     verbose=args.verbose)
    # Otherwise
    else:
        # Generate dataset
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
