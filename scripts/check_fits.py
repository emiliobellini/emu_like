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
    elif any(['_PK_' in x for x in fits.get_keys()]):
        is_cell = False

    io.info('File: {}'.format(args.sample_file))

    n_samples_run = None
    raised_warning = False
    # Store ell, k, z, n_samples
    n_ells = {}
    n_k = {}
    for key in fits.get_keys():
        header = fits.get_header(key)
        data = fits.get_data(key)
        if key == 'X_DATA':
            n_samples_tot, n_features = data.shape
        elif is_cell is True and 'ELL_RANGE_' in key:
            n_ells[key.replace('ELL_RANGE_', '')] = data.shape[0]
        elif is_cell is False and 'K_RANGE_' in key:
            n_k[key.replace('K_RANGE_', '')] = data.shape[0]
        elif is_cell is False and key == 'Z_ARRAY':
            n_z = data.shape[0]
    io.print_level(1, 'Features: {}'.format(n_features))
    io.print_level(1, 'Samples total: {}'.format(n_samples_tot))
    if is_cell is True:
        io.print_level(1, 'N_ell: {}'.format(n_ells))
    elif is_cell is False:
        io.print_level(1, 'N_k: {}'.format(n_k))
        io.print_level(1, 'N_z: {}'.format(n_z))

    for key in fits.get_keys():
        data = fits.get_data(key)
        if key == 'PRIMARY':
            pass
        elif key == 'X_DATA':
            pass
        elif 'ELL_RANGE_' in key:
            pass
        elif 'K_RANGE_' in key:
            pass
        elif key == 'Z_ARRAY':
            pass
        # Check reference shapes
        elif 'REF_' in key:
            sub_key = key.replace('REF_', '')
            if is_cell is True:
                expected_shape = (1, n_ells[sub_key])
                if data.shape != expected_shape:
                    io.warning('{} has {} shape, expected {}'.format(
                        key, data.shape, expected_shape))
                else:
                    io.info('{} shape: {}'.format(key, expected_shape))
            elif is_cell is False:
                expected_shape = (1, n_k[sub_key], n_z)
                if data.shape != expected_shape:
                    io.warning('{} has {} shape, expected {}'.format(
                        key, data.shape, expected_shape))
                else:
                    io.print_level(1, '{} shape: {}'.format(
                        key, expected_shape))
        else:
            # Check number of ells, k
            if is_cell is True:
                if data.shape[1] != n_ells[key]:
                    io.warning('{} has {} ells, expected {}'.format(
                        key, data.shape[1], n_ells[key]))
            elif is_cell is False:
                if data.shape[1] != n_k[key]:
                    io.warning('{} has {} k, expected {}'.format(
                        key, data.shape[1], n_k[key]))
            # Check samples
            if n_samples_run is None:
                n_samples_run = data.shape[0]
                ref_key = key
            if n_samples_run != data.shape[0]:
                raised_warning = True
                io.warning(
                    'Mismatch between {} (n_samples_run={}) and {} '
                    '(n_samples_run={})'.format(
                        key, data.shape[0], ref_key, n_samples_run))

    if raised_warning is False:
        io.print_level(
            1, 'All the spectra have the same lines: {}'.format(n_samples_run))
        if n_samples_run == n_samples_tot:
            io.info('Fits file completed!')
        else:
            io.warning('Still to run {} samples'.format(
                n_samples_tot - n_samples_run))
