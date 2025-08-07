import argparse
import numpy as np
import os
from emu_like.datasets import DataCollection
import emu_like.io as io



# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sample_folder', type=str)
    args = parser.parse_args()
    print(args.sample_folder)

    for file in io.Folder(args.sample_folder).list_files():
        if os.path.split(file)[-1].startswith('x_data'):
            x = np.genfromtxt(file)

    for file in io.Folder(args.sample_folder).list_files():
        if os.path.split(file)[-1].startswith('y_data'):
            content = np.genfromtxt(file)
            n_samples = content.shape[0]
            is_nan = np.isnan(content)
            is_nan_tot = is_nan.any()
            is_nan_one = np.any(is_nan, axis=1)
            if is_nan_tot:
                x_nan = x[:n_samples][is_nan_one]
                io.warning('Spectrum {} has nans! Parameters:'.format(file))
                for x_one in x_nan:
                    io.print_level(1, '{}'.format(x_one))
            else:
                io.print_level(1, 'Only finite points for {}'.format(file))
    if False:

        data = DataCollection().load(path=args.sample_folder, verbose=True)


        for key in data.y_model.spectra.names:
            spectrum = data.get_one_y_dataset(name=key)
            spectrum = spectrum.remove_non_finite(store_non_finites=True, verbose=False)
            if spectrum.non_finites_x.shape == (0, spectrum.n_x):
                io.print_level(0, 'Spectrum {} has only finite points!'.format(key))
            else:
                io.warning('Spectrum {} has non-finite points! Parameters:'.format(key))
                for x in spectrum.non_finites_x:
                    io.print_level(1, ', '.join(['{} = {:.4}'.format(name, val) for name, val in zip(spectrum.x_names, spectrum.x[0])]))
