import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as interp
from emu_like.datasets import DataCollection
from emu_like.ffnn_emu import FFNNEmu
import emu_like.io as io


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    # Define local variables (everything else should be left unchanged)
    root_sample = 'output/lcdm/sample'
    root_train = 'output/lcdm/train'
    names = ['50', '100', '1000'] # TODO , '10000'

    params_emu = {
        'activation': 'relu',
        'neurons_hidden': [400, 400],
        'batch_normalization': False,
        'dropout_rate': 0.,
        'batch_size': 32,
        'optimizer': 'adam',
        'loss': 'mean_squared_error',
        'want_output_layer': True,
    }

    data_collection = {}
    data = {}
    emu = {}


    for name in names:
        io.info('Training model {}'.format(name))

        data[name] = {}
        emu[name] = {}

        # Load DataCollection
        io.print_level(1, 'Loading data')
        data_collection[name] = DataCollection().load(path=os.path.join(root_sample, name))
        spectra_names = data_collection[name].y_model.spectra.names

        # Train each spectrum
        for spectrum in spectra_names:

            # Manipulate data
            data[name][spectrum] = data_collection[name].get_one_y_dataset(name=spectrum)

            data[name][spectrum].train_test_split(
                frac_train=0.9,
                seed=1543)

            data[name][spectrum].rescale(
                rescale_x = 'MinMaxScaler',
                rescale_y = 'MinMaxCommonScaler')

            # Try to load the emulator
            try:
                emu[name][spectrum] = FFNNEmu().load(os.path.join(root_train, name, spectrum))
                io.print_level(1, 'Loaded {}'. format(spectrum))

            # Build it and train it
            except ValueError:

                io.print_level(1, 'Training {}'. format(spectrum))
                # Build emulator
                emu[name][spectrum] = FFNNEmu()

                params_emu['data_n_x'] = data[name][spectrum].n_x
                params_emu['data_n_y'] = data[name][spectrum].n_y

                emu[name][spectrum].build(params_emu)

                # Train and save the emulator
                emu[name][spectrum].train(
                    data[name][spectrum],
                    epochs=2000,
                    learning_rate=1.e-3,
                    get_plot=True,
                    )
                emu[name][spectrum].train(
                    data[name][spectrum],
                    epochs=2000,
                    learning_rate=1.e-4,
                    get_plot=True,
                    )
                emu[name][spectrum].train(
                    data[name][spectrum],
                    epochs=2000,
                    learning_rate=1.e-5,
                    get_plot=True,
                    path = os.path.join(root_train, name, spectrum)
                    )
            

            # Get derived quantities
            data[name][spectrum].y_emu = np.array([emu[name][spectrum].eval(x) for x in data[name][spectrum].x])
            data[name][spectrum].diff = (data[name][spectrum].y_emu/data[name][spectrum].y-1)
            data[name][spectrum].y_emu_train = np.array([emu[name][spectrum].eval(x) for x in data[name][spectrum].x_train])
            data[name][spectrum].diff_train = (data[name][spectrum].y_emu_train/data[name][spectrum].y_train-1)
            data[name][spectrum].y_emu_test = np.array([emu[name][spectrum].eval(x) for x in data[name][spectrum].x_test])
            data[name][spectrum].diff_test = (data[name][spectrum].y_emu_test/data[name][spectrum].y_test-1)


    # Summary plots
    for spectrum in spectra_names:

        if data[names[0]][spectrum].y_model.spectra[0].is_pk:
            x_label = 'k [h/Mpc]'
            x_scale = 'log'
            y_scale = 'log'
            x = emu[names[0]][spectrum].y_model.k_ranges[0]
        elif data[names[0]][spectrum].y_model.spectra[0].is_cl:
            x_label = 'ell'
            x_scale = 'linear'
            y_scale = 'linear'
            x = emu[names[0]][spectrum].y_model.ell_ranges[0]

        fig, ax = plt.subplots(nrows=5, ncols=len(names), figsize=(5.*len(names), 20.), sharex=True, squeeze=False)
        ax[0, 0].set_ylabel('rel. diff. [%] -- Training set')
        ax[1, 0].set_ylabel('rel. diff. [%] -- Validation set')
        ax[2, 0].set_ylabel('rel. diff. [%] -- Worst fit')
        ax[3, 0].set_ylabel('{}/{}(ref) -- Worst fit'.format(spectrum, spectrum))
        ax[4, 0].set_ylabel('{} -- Worst fit'.format(spectrum))

        for count, name in enumerate(names):
            ax[0, count].set_title('{} points'.format(name))
            ax[-1, count].set_xlabel(x_label)
            ax[-1, count].set_xscale(x_scale)
            ax[-1, count].set_yscale(y_scale)
        
            # Training set
            ax[0, count].plot(x, data[name][spectrum].diff_train.T*100, 'k-', alpha=0.2)

            # Validation set
            ax[1, count].plot(x, data[name][spectrum].diff_test.T*100, 'k-', alpha=0.2)

            idx_max = np.argmax(np.max(np.abs(data[name][spectrum].diff), axis=1))

            # Worst fit, rel diff
            ax[2, count].plot(x, data[name][spectrum].diff[idx_max]*100, 'k-')

            # Worst fit, P/P_ref
            ax[3, count].plot(x, data[name][spectrum].y_emu[idx_max], label='Emulated')
            ax[3, count].plot(x, data[name][spectrum].y[idx_max], '--', label='True')
            ax[3, count].legend()

            # Worst fit, P
            if data[names[0]][spectrum].y_model.spectra[0].is_pk:
                z = data[name][spectrum].x[idx_max, 0]
                ref = interp.make_splrep(emu[name][spectrum].y_model.z_array, emu[name][spectrum].y_model.y_ref[0][0].T, s=0)(z)
            elif data[names[0]][spectrum].y_model.spectra[0].is_cl:
                ref = emu[name][spectrum].y_model.y_ref[0][0]
            ax[4, count].plot(x, ref*data[name][spectrum].y_emu[idx_max])
            ax[4, count].plot(x, ref*data[name][spectrum].y[idx_max], '--')

            # Print stuff
            io.info('Worst fit {} - {} points'.format(spectrum, name))
            for var, val in zip(data[name][spectrum].x_names, data[name][spectrum].x[idx_max]):
                io.print_level(1, '{} = {}'.format(var, val))
            print()

        plt.subplots_adjust(bottom=0.15, hspace=0.05, wspace=0.15)
        plt.savefig(os.path.join(root_train, 'summary_{}.pdf'.format(spectrum)))
