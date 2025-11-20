"""
.. module:: export

:Synopsis: Pipeline used to export spectra emulators.
:Author: Emilio Bellini

"""

import joblib
import os
import emu_like.io as io
from emu_like.emu import Emulator


def export_emu(args):
    """ Export spectra emulators to a given folder.

    Args:
        args: the arguments read by the parser.


    """

    input = io.Folder(args.input)
    output = io.Folder(args.output)
    output.create()

    if not output.is_empty():
        raise Exception(
            'Output folder not empty! Exiting to avoid corruption of '
            'precious data!')

    # Load emulators
    if args.verbose:
        io.info('Exporting emulators:')

    for in_path in input.list_subfolders():

        # Load emulator
        emu = Emulator.choose_one('ffnn_emu', verbose=False)
        emu.load(in_path, verbose=False)

        # Fix paths
        name = emu.y_model.spectra[0].name
        dict_fname = '{}.joblib'.format(name)
        model_fname = '{}.keras'.format(name)

        # Store necessary quantities
        emu_dict = {
            'name': name,
            'x_names': emu.x_names,
            'x_ranges': emu.x_ranges,
            'x_scaler': emu.x_scaler,
            'y_scaler': emu.y_scaler,
            'x_pca': emu.x_pca,
            'y_pca': emu.y_pca,
            'ref': emu.y_model.y_ref[0][0],
            'z_array': emu.y_model.z_array,
            'k_array': emu.y_model.k_ranges[0],
            'ell_array': emu.y_model.ell_ranges[0],
            'class_vars': emu.y_model.params,
            'class_ref': emu.y_model.ref_params,
            'class_args': emu.y_model.args,
            'model_path': model_fname,
        }

        # Save emulator
        emu.model.save(
            os.path.join(output.path, model_fname),
            include_optimizer=False)

        # Save dictionary
        joblib.dump(
            emu_dict,
            os.path.join(output.path, dict_fname),
            compress=3)

        if args.verbose:
            io.print_level(1, 'Saved {} emulator'.format(name))

    return
