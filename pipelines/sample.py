"""
.. module:: sample

:Synopsis: Pipeline used to generate samples.
:Author: Emilio Bellini

"""

import src.emu_like.defaults as de
import src.emu_like.io as io
from src.emu_like.params import Params
from src.emu_like.sample import Sample


def sample_emu(args):
    """ Generate the sample for the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        io.print_level(0, '\nGetting sample for Emulator\n')

    # Read params
    params = Params().load(args.params_file)

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
        sample.load(params['output'], verbose=args.verbose)
        sample.resume(save_incrementally=True, verbose=args.verbose)
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

        # Create output folder
        io.Folder(params['output']).create(args.verbose)
        # Save params
        params.save(
            de.file_names['params']['name'],
            root=params['output'],
            header=de.file_names['params']['header'],
            verbose=args.verbose)

        # Pass correct params dict
        if params['sampled_function'] == 'cobaya_loglike':
            params_dict = params['cobaya']
        else:
            params_dict = params['params']

        # Generate sample
        sample.generate(
            params=params_dict,
            sampled_function=params['sampled_function'],
            n_samples=params['n_samples'],
            spacing=params['spacing'],
            save_incrementally=True,
            output_path=params['output'],
            verbose=args.verbose)

    return
