import argparse
import emu_like.io as io

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_file', type=str)
    args = parser.parse_args()

    fits = io.FitsFile(args.sample_file)

    if any(['_CL_' in x for x in fits.get_keys()]):
        is_cell = True
        spectra_list = [
            'cl_TT_lensed',
            'cl_TE_lensed',
            'cl_EE_lensed',
            'cl_pp_lensed',
            'cl_Tp_lensed',
            'cl_BB_lensed']
    elif any(['_PK_' in x for x in fits.get_keys()]):
        is_cell = False
        spectra_list = [
            'pk_m',
            'pk_cb',
            'pk_weyl',
            'fk_m',
            'fk_cb',
            'fk_weyl']

    io.info('File: {}'.format(args.sample_file))

    n_samples = []
    for spectrum in spectra_list:
        n_samples.append(fits.get_data(spectrum).shape[0])
    n_samples = min(n_samples)

    try:
        for spectrum in spectra_list:
            data = fits.get_data(spectrum)[:n_samples]
            fits.update(
                name=spectrum,
                data=data,
            )
        io.info('Done {}. N_samples: {}'.format(
            args.sample_file, n_samples))
    except TypeError:
        io.warning('Failed on {}.'.format(
            args.sample_file))
