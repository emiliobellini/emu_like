import argparse
import classy
import numpy as np
import os
import scipy.interpolate as interp
import emu_like.io as io
from emu_like.params import Params
from emu_like.spectra import Spectra


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_file', type=str)
    args = parser.parse_args()

    fits = io.FitsFile(args.sample_file)
    if 'pk' in args.sample_file:
        spectrum_type = 'pk'
    elif 'cl' in args.sample_file:
        spectrum_type = 'cl'

    params = fits.get_header(0)
    class_args = params['y_model']['args']
    x_names = list(params['params'].keys())

    spectra_list = {
        'pk': ['pk_m', 'pk_cb', 'pk_weyl', 'fk_m', 'fk_cb', 'fk_weyl'],
        'cl': ['cl_TT_lensed', 'cl_TE_lensed', 'cl_EE_lensed', 'cl_pp_lensed', 'cl_Tp_lensed', 'cl_BB_lensed'],
    }

    io.info('Getting nan indices')
    x_data = fits.get_data('x_data')
    idxs_nan = []
    data = {}
    hd = {}
    ref = {}
    for spectrum in spectra_list[spectrum_type]:
        data[spectrum] = fits.get_data(spectrum)
        ref[spectrum] = fits.get_data('ref_'+spectrum)[0]
        is_nan = np.any(np.isnan(data[spectrum]), axis=1)
        idxs_nan.append(np.where(is_nan)[0])
    if all([all(x==idxs_nan[0]) for x in idxs_nan]):
        idxs_nan = idxs_nan[0]
    else:
        raise Exception('This is strange!')
    for idx in idxs_nan:
        io.print_level(1, 'Found nans at x = {}'.format(x_data[idx]))

    if len(idxs_nan) == 0:
        io.info('Nothing to do.')
    else:
        spectra = Spectra(params['y_model']['outputs'])

        io.info('Computing new y')
        new_y = {}
        for idx in idxs_nan:
            new_y[idx] = {}
            var_params = {name: val for name, val in zip(x_names, x_data[idx])}
            cosmo_params = class_args | var_params | {'recombination': 'recfast'}

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
                    new_y[idx][sp.name] = None
            else:
                # Iterate over spectra
                for sp in spectra:
                    if sp.is_pk:
                        z = cosmo_params['z_pk']
                    else:
                        z = None
                    array = sp.get(cosmo, z=z)
                    if sp.ratio:
                        if sp.is_pk:
                            array = array/ref[sp.name](cosmo_params['z_pk'])
                        elif sp.is_cl:
                            array = array/ref[sp.name]
                    new_y[idx][sp.name] = array

        io.info('Replacing values')
        for sp in spectra:
            for idx in idxs_nan:
                data[sp.name][idx] = new_y[idx][sp.name]

            fits.update(data[sp.name], sp.name)
