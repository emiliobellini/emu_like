"""

Main module with the pipeline used to train the emulator.

"""
import src.utilities.defaults as de
import src.utilities.io as io
import src.utilities.printing_scripts as scp
from src.core.sample import Sample


def sample_emu(args):
    """ Generate the sample for the emulator.

    Args:
        args: the arguments read by the parser.


    """

    if args.verbose:
        scp.print_level(0, '\nGetting sample for Emulator\n')

    # Load input file
    params = io.YamlFile(args.params_file, should_exist=True)
    params.read()

    # Define output path
    output = io.Folder(path=params['output'])
    if args.resume:
        if args.verbose:
            scp.info('Resuming from {}.'.format(output.path))
        ref_params = io.YamlFile(
            de.file_names['params']['name'],
            root=output,
            should_exist=True)
        ref_params.read()
        # TODO: maybe here add check that param files are consistent
    else:
        if args.verbose:
            scp.info("Writing output in {}".format(output.path))
        # Check if empty, and copy param file to output folder
        if output.is_empty():
            params.copy_to(
                name=de.file_names['params']['name'],
                root=params['output'],
                header=de.file_names['params']['header'],
                verbose=args.verbose)
        # Else exit, to avoid overwriting
        else:
            raise Exception(
                'Output folder not empty! Exiting to avoid corruption of '
                'precious data! If you want to resume a previous run use '
                'the --resume (-r) option.')

    # Load or generate sample
    sample = Sample()
    sample.generate(
        params=params,
        root=output,
        resume=args.resume,
        verbose=args.verbose)

    # Save details in output folder
    details_path = io.YamlFile(
        de.file_names['sample_details']['name'],
        root=output.subfolder(
            de.file_names['sample_details']['folder']).create(
                verbose=args.verbose)
    )
    sample.save_details(details_path, verbose=args.verbose)

    return
