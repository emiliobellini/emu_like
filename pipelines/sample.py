"""
.. module:: sample

:Synopsis: Pipeline used to generate datasets.
:Author: Emilio Bellini

"""

import emu_like.defaults as de
import emu_like.io as io
from emu_like.params import Params
from emu_like.datasets import Dataset


def sample_emu(args):
    """ Generate the dataset for the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        io.print_level(0, '\nGetting dataset for Emulator\n')

    # Init Dataset object
    data = Dataset()

    # Read params
    params = Params().load(args.params_file)
    # Fill missing entries
    params = data.fill_missing_params(params)

    # If resume
    if args.resume:
        if args.verbose:
            io.info('Resuming from {}.'.format(params['output']))
            io.print_level(1, 'Ignoring {}'.format(args.params_file))
        # Read params from output folder
        params = Params().load(de.file_names['params']['name'],
                               root=params['output'])

        # Load the dataset
        data.load(params['output'], verbose=args.verbose)
        # Resume the dataset
        data.resume(
            save_incrementally=True,
            verbose=args.verbose)
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

        # Generate dataset
        data.sample(
            params=params['params'],
            x_name=params['x_sampler']['name'],
            x_args=params['x_sampler']['args'],
            y_name=params['y_model']['name'],
            y_args=params['y_model']['args'],
            y_outputs=params['y_model']['outputs'],
            output=params['output'],
            save_incrementally=True,
            verbose=args.verbose)

    return
