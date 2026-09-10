"""
.. module:: export

:Synopsis: Pipeline used to export spectra emulators.
:Author: Emilio Bellini

"""

import joblib
import os
import emu_like.io as io
from emu_like.emu import Emulator


def _export_transform(transform):
    """Return a portable transform description, preserving absent PCA."""
    return None if transform is None else transform.export()


def _sobolev_export_metadata(emu, primary_name):
    """Build the data needed to derive fk from an exported pk network.

    The model stored in a Sobolev training directory is deliberately the raw
    network which predicts the scaled log Pk ratio.  The following metadata is
    therefore sufficient for a consumer to evaluate the same relation as
    ``SobolevFFNNEmu.eval_fk`` without serialising its training wrapper.
    """
    if not primary_name.startswith('pk_'):
        raise ValueError(
            'Sobolev export expects a pk_* primary spectrum, got {!r}'
            ''.format(primary_name))
    required = {
        'z_index': emu.z_index,
        'redshift_grid': emu.redshift_grid,
        'reference_growth': emu.reference_growth,
        'growth_scaler': emu.growth_scaler,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            'Cannot export incomplete Sobolev emulator; missing {}'
            ''.format(', '.join(missing)))

    z_scaler = emu.x_scaler.skl_scaler
    pk_scaler = emu.y_scaler.skl_scaler
    return {
        'version': 1,
        'fk_name': 'fk_{}'.format(primary_name[3:]),
        'z_index': int(emu.z_index),
        'redshift_grid': emu.redshift_grid,
        'reference_growth': emu.reference_growth,
        # These explicit factors avoid making the consumer depend on the
        # internal representation of the sklearn scaler wrappers.
        'z_mean': z_scaler.mean_[emu.z_index],
        'z_scale': z_scaler.scale_[emu.z_index],
        'pk_scaled_to_log_ratio_scale': pk_scaler.scale_,
        'growth_scaler': emu.growth_scaler.export(),
        'formula': (
            'fk = reference_growth(z) - 0.5 * (1 + z) * '
            'd_log_pk_ratio_dz'),
    }


def _export_one_emulator(in_path, out_path, force=False, verbose=False):
    """Export a single emulator from a training folder to an output folder."""

    # Load the concrete type recorded by training.  In particular, a
    # Sobolev emulator has a custom training wrapper but exports its plain
    # differentiable Pk inference network.
    training_params = io.YamlFile(root=in_path).read().content
    emu_type = training_params['emulator']['name']
    emu = Emulator.choose_one(emu_type, verbose=False)
    emu.load(in_path, still_training=False, verbose=False)

    # Fix paths
    name = emu.y_model.spectra[0].name
    dict_fname = '{}.joblib'.format(name)
    model_fname = '{}.keras'.format(name)

    if not force and os.path.exists(os.path.join(out_path, dict_fname)):
        raise FileExistsError(
            'Output file {} already exists. Use --force to override.'
            ''.format(os.path.join(out_path, dict_fname)))
    if not force and os.path.exists(os.path.join(out_path, model_fname)):
        raise FileExistsError(
            'Output file {} already exists. Use --force to override.'
            ''.format(os.path.join(out_path, model_fname)))

    # Store necessary quantities
    emu_dict = {
        'name': name,
        'emulator_type': emu.name,
        'x_names': emu.x_names,
        'x_ranges': emu.x_ranges,
        'x_scaler': emu.x_scaler.export(),
        'y_scaler': emu.y_scaler.export(),
        'x_pca': _export_transform(emu.x_pca),
        'y_pca': _export_transform(emu.y_pca),
        'ref_spectrum': emu.y_model.y_ref[0][0],
        'ref_z': emu.y_model.z_array,
        'ref_k': emu.y_model.k_ranges[0],
        'ref_ell': emu.y_model.ell_ranges[0],
        'ref_params': emu.y_model.ref_params,
        'class_args': emu.y_model.args,
        'model_path': model_fname,
    }
    if emu.name == 'sobolev_ffnn_emu':
        emu_dict['sobolev'] = _sobolev_export_metadata(emu, name)

    # Add parameter files
    all_params = io.Folder(in_path).list_files(patterns='.+yaml')
    for param in all_params:
        tmp = io.YamlFile(param).read().content
        if os.path.basename(param) == 'params.yaml':
            emu_dict['training_params'] = tmp
        else:
            emu_dict[os.path.basename(param).split('.')[0]] = tmp

    # Add as a separate entry the ranges for each dataset
    emu_dict_keys = list(emu_dict.keys())
    for key in emu_dict_keys:
        if 'dataset' in key:
            key_dr = 'x_ranges_{}'.format(key.split('_')[-1])
            vals = [[
                emu_dict[key]['params'][x_name]['prior']['min'],
                emu_dict[key]['params'][x_name]['prior']['max']]
                for x_name in emu_dict['x_names']]
            emu_dict[key_dr] = vals

    # Save emulator
    inference_model = getattr(emu.model, 'network', emu.model)
    inference_model.save(
        os.path.join(out_path, model_fname),
        include_optimizer=False)

    # Save dictionary
    joblib.dump(
        emu_dict,
        os.path.join(out_path, dict_fname),
        compress=3)

    if verbose:
        io.print_level(1, 'Saved {} emulator'.format(name))

    return


def export_emu(args):
    """ Export spectra emulators to a given folder.

    Args:
        args: the arguments read by the parser.


    """

    input = io.Folder(args.input)
    output = io.Folder(args.output)

    if not output.exists:
        output.create()

    # Load emulators
    if args.verbose:
        io.info('Exporting emulators:')

    if input.is_emulator_folder():
        list_subfolders = [input.path]
    else:
        list_subfolders = input.list_subfolders()

    for in_path in list_subfolders:
        _export_one_emulator(
            in_path,
            output.path,
            force=args.force,
            verbose=args.verbose)

    return
