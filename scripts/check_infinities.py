import argparse
from emu_like.datasets import DataCollection
import emu_like.io as io



# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sample_folder', type=str)
    args = parser.parse_args()


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
