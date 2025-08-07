import argparse
import classy
import numpy as np
import os
import scipy.interpolate as interp
import emu_like.io as io
import emu_like.defaults as de
from emu_like.params import Params
from emu_like.spectra import Spectra


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_folder', type=str)
    args = parser.parse_args()

    io.info('Getting nan indices')
    for file in io.Folder(args.sample_folder).list_files():
        if os.path.split(file)[-1].startswith('x_data'):
            x = np.genfromtxt(file)
        if os.path.split(file)[-1].startswith('y_data_pk_m'):
            y_ref = np.genfromtxt(file)
    
    is_nan = np.any(np.isnan(y_ref), axis=1)
    x_nan = x[is_nan]
    idxs_nan = np.where(is_nan)[0]
    io.print_level(1, 'Found {} nans'.format(len(x_nan)))

    spectra_params = Params().load(os.path.join(args.sample_folder, 'params.yaml'))
    spectra = Spectra(spectra_params['y_model']['outputs'])
    x_names = list(spectra_params['params'].keys())

    io.info('Computing reference spectra')
    # Get params ref
    z_max = {'z_max_pk': spectra_params['params']['z_pk']['prior']['max']}
    cosmo_params_ref = de.cosmo_params | spectra.get_class_params() | z_max
    # Compute ref
    cosmo_ref = classy.Class()
    cosmo_ref.set(cosmo_params_ref)
    cosmo_ref.compute()
    # Iterate over spectra ref
    array_ref = {sp.name: sp.get(cosmo_ref) for sp in spectra}
    for sp in spectra:
        if array_ref[sp.name].ndim == 2:
            array_ref[sp.name] = interp.make_splrep(sp.z_array, array_ref[sp.name].T, s=0)


    io.info('Computing new y')
    new_y = {}
    for idx_one, x_one in zip(idxs_nan, x_nan):
        new_y[idx_one] = {}
        # Get params
        cosmo_params = {name: value for name, value in zip(x_names, x_one)}
        cosmo_params = cosmo_params | spectra_params['y_model']['args']
        for key in de.cosmo_params:
            if key not in cosmo_params.keys():
                cosmo_params[key] = de.cosmo_params[key]
        cosmo_params = cosmo_params | spectra.get_class_params()

        # Compute
        failed = False
        cosmo = classy.Class()
        cosmo.set(cosmo_params)
        try:
            # Compute class
            cosmo.compute()
        except classy.CosmoComputationError:
            failed = True
        except classy.CosmoSevereError:
            failed = True

        if failed:
            io.warning('Class failed for parameters {}'.format(cosmo_params))
            for sp in spectra:
                new_y[idx_one][sp.name] = None
        else:
            # Iterate over spectra
            for sp in spectra:
                array = sp.get(cosmo, z=cosmo_params['z_pk'])
                if sp.ratio:
                    if sp.is_pk:
                        array = array/array_ref[sp.name](cosmo_params['z_pk'])
                    elif sp.is_cl:
                        array = array/array_ref[sp.name]
                new_y[idx_one][sp.name] = array


    io.info('Loading files and replacing values')
    for sp in spectra:
        # Open file
        path = os.path.join(args.sample_folder, 'y_data_{}.txt').format(sp.name)
        header = io.File(path).read_header()
        old_array = np.genfromtxt(path)
        for idx_one, x_one in zip(idxs_nan, x_nan):
            if new_y[idx_one][sp.name] is not None:
                old_array[idx_one] = new_y[0][sp.name]
        np.savetxt(path, old_array, header=header)
        io.print_level(1, 'Saved spectrum {}!'.format(sp.name))
