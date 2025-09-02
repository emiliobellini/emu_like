import argparse
import numpy as np
import emu_like.io as io

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_folder', type=str)
    args = parser.parse_args()

    spectra_list = {
        'pk': ['pk_m', 'pk_cb', 'pk_weyl', 'fk_m', 'fk_cb', 'fk_weyl'],
        'cl': ['cl_TT_lensed', 'cl_TE_lensed', 'cl_EE_lensed', 'cl_pp_lensed', 'cl_Tp_lensed', 'cl_BB_lensed'],
    }

    folder = io.Folder(args.sample_folder)

    for fname in folder.list_files():
        if '.fits' in fname:

            fits = io.FitsFile(fname)
            if 'pk' in fname:
                spectrum_type = 'pk'
            elif 'cl' in fname:
                spectrum_type = 'cl'

        for spectrum in spectra_list[spectrum_type]:
            data = fits.get_data(spectrum)
            is_nan = np.any(np.isnan(data))

            if is_nan:
                io.warning('Found nans in {} ({})'.format(spectrum, fname))
            else:
                io.info('No nans in {} ({})'.format(spectrum, fname))
