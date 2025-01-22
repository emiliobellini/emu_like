"""
.. module:: sample

:Synopsis: Pipeline used to generate samples.
:Author: Emilio Bellini

"""

import emu_like.defaults as de
import emu_like.io as io
from emu_like.params import Params
from emu_like.sample import Sample


def sample_emu(args):
    """ Generate the sample for the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        io.print_level(0, '\nGetting sample for Emulator\n')

    # Read params
    params = Params().load(args.params_file, fill_missing=True)

    # Init Sample object
    sample = Sample()

    # If resume
    if args.resume:
        if args.verbose:
            io.info('Resuming from {}.'.format(params['output']))
            io.print_level(1, 'Ignoring {}'.format(args.params_file))
        # Read params from output folder
        params = Params().load(de.file_names['params']['name'],
                               root=params['output'])
        # Try to get extra_arguments
        try:
            sampled_function_args = params['sampled_function_args']
        except KeyError:
            sampled_function_args = None
        sample.load(params['output'], verbose=args.verbose)
        sample.resume(
            params['params'],
            sampled_function_args=sampled_function_args,
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

        # Generate sample
        sample.generate(
            params=params['params'],
            sampler_name=params['sampler']['name'],
            sampler_args=params['sampler']['args'],
            generator_name=params['train_generator']['name'],
            generator_args=params['train_generator']['args'],
            generator_outputs=params['train_generator']['outputs'],
            output=params['output'],
            save_incrementally=True,
            verbose=args.verbose)

    return
