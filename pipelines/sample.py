"""
.. module:: sample

:Synopsis: Pipeline used to generate datasets.
:Author: Emilio Bellini

"""

import emu_like.io as io
from emu_like.datasets import DataCollection


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
    params = io.YamlFile(args.params_file).read()

    # Default output parameters
    try:
        timeout = params['output']['timeout']
        if args.verbose:
            io.info('Time limit for execution is {} hours'.format(timeout))
    except KeyError:
        timeout = None
    try:
        save_interval = params['output']['save_interval']
        if args.verbose:
            io.info('Saving every {} steps'.format(save_interval))
    except KeyError:
        save_interval = None

    from emu_like.range_sampling import sample_range, merge_ranges
    start, stop = args.start_row, args.stop_row
    ranged = start is not None or stop is not None
    if sum([ranged, bool(args.merge_ranges), args.prepare_only]) > 1:
        raise ValueError('Choose range sampling, merging, or preparation')
    if args.keep_ranges and not args.merge_ranges:
        raise ValueError('--keep-ranges requires --merge-ranges')
    if ranged:
        if start is None or stop is None:
            raise ValueError('Specify both --start-row and --stop-row')
        if args.num_workers != 1:
            raise ValueError(
                'Range jobs compute serially; use --num-workers 1')
        print('Sampling ranges for: {}'.format(
            params['output']['path']), flush=True)
        part = sample_range(params['output']['path'], start, stop,
                            save_interval=save_interval, timeout=timeout)
        io.info('Range checkpoint: {}'.format(part))
        return
    if args.merge_ranges:
        count = merge_ranges(params['output']['path'], args.merge_ranges,
                             cleanup=not args.keep_ranges)
        io.info('Merged dataset: {} completed rows'.format(count))
        return
    if args.prepare_only and (args.resume or args.force):
        raise ValueError('--prepare-only requires a fresh output')

    # Force computation
    if args.force:
        if io.FitsFile(params['output']['path']).exists:
            args.resume = True
        else:
            args.resume = False

    # If resume
    if args.resume:
        if args.verbose:
            io.info('Resuming from {}.'.format(params['output']['path']))
            io.print_level(1, 'Ignoring {}'.format(args.params_file))

        # Resume the dataset
        data.resume(
            params['output']['path'],
            timeout=timeout,
            save_interval=save_interval,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            verbose=args.verbose)
    # Otherwise
    else:
        # Write parameters in output folder
        params.write(
            fname='{}.yaml'.format(params['output']['path'].split('.')[0]),
            verbose=args.verbose)
        # Sample the dataset
        data.sample(
            prepare_only=args.prepare_only,
            params=params['params'],
            x_name=params['x_sampler']['name'],
            x_args=params['x_sampler']['args'],
            y_name=params['y_model']['name'],
            y_args=params['y_model']['args'],
            y_outputs=params['y_model']['outputs'],
            output=params['output']['path'],
            timeout=timeout,
            save_interval=save_interval,
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
            verbose=args.verbose)

    return
