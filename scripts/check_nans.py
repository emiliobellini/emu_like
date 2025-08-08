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

    io.info('Loading x')
    for file in io.Folder(args.sample_folder).list_files():
        if os.path.split(file)[-1].startswith('x_data'):
            x = np.genfromtxt(file)

    io.info('Loading y')
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
