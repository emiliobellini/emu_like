import argparse
import hiclassy
import matplotlib.pyplot as plt
import emu_like.io as io
import emu_like.spectra as spectra

# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_file', type=str)
    parser.add_argument('--indices_data', '-i', nargs='+', type=int)
    parser.add_argument('--spectrum', '-s', type=str, default='pk_m')
    args = parser.parse_args()

    fits = io.FitsFile(args.fits_file)

    n_cosmologies = len(args.indices_data)
    x_data = fits.get_data('x_data')[args.indices_data]
    y_data = fits.get_data('{}'.format(args.spectrum))[args.indices_data]
    k_range = fits.get_data('k_range_{}'.format(args.spectrum))
    z_array = fits.get_data('z_array')
    ref = fits.get_data('ref_{}'.format(args.spectrum))
    ref_params = fits.get_header('ref_{}'.format(args.spectrum))
    class_params = fits.get_header(
        '{}'.format(args.spectrum))['class_parameters']
    params_names = fits.get_header('x_data')['parameters'].keys()
    spectrum_params = fits.get_header(0)['y_model']['outputs'][args.spectrum]

    for n in range(n_cosmologies):

        all_params = (
            ref_params | {k: v for k, v in zip(params_names, x_data[n])})
        # print(all_params)

        # Compute with HiClass
        cosmo = hiclassy.HiClass()
        cosmo.set(all_params)
        cosmo.compute()

        # Init spectrum object
        spectrum = spectra.Spectrum.choose_one(args.spectrum, spectrum_params)

        class_data = spectrum.get(cosmo, z=all_params['z_pk'])

        plt.plot(k_range, class_data, label='class {}'.format(n))
        plt.plot(k_range, y_data[n], '--', label='data {}'.format(n))
    plt.legend()
    plt.xscale('log')
    plt.savefig('output/check_data_{}.png'.format(args.spectrum))
