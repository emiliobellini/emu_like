"""

Main module with the pipeline used to train the emulator.

"""
import tools.defaults as de
import tools.io as io
import tools.printing_scripts as scp
from tools.sample import Sample


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
            'precious data!')

    # Load or generate sample
    sample = Sample()
    sample.generate(params=params, verbose=args.verbose)

    # Save in output folder sample
    sample.save(output, verbose=args.verbose)

    return
