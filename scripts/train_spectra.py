import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as interp
import emu_like.io as io
from emu_like.datasets import DataCollection
from emu_like.ffnn_emu import FFNNEmu


# -----------------MAIN-CALL-----------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('main_folder', type=str)
    parser.add_argument('--datasets', '-d', type=str, nargs='+', default=None)
    parser.add_argument('--spectra', '-s', type=str, nargs='+', default=None)
    parser.add_argument('--force', '-f', action='store_true')
    args = parser.parse_args()

    main_folder = io.Folder(args.main_folder, should_exist=True)
    sample_folder = main_folder.subfolder('sample', should_exist=True)
    train_folder = main_folder.subfolder('train')

    # TODO: externalize these parameters
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
    spectra_names = {}
    data = {}
    emu = {}
    derived = {}

    # Fix datasets to train or load
    if args.datasets is None:
        names = [os.path.split(x)[-1] for x in sample_folder.list_subfolders()]
    else:
        names = args.datasets

    # Load data collection
    io.info('Loading data')
    for name in names:
        try:
            data_collection[name] = DataCollection().load(path=sample_folder.subfolder(name).path)
            io.print_level(1, 'Loaded {} folder'.format(name))
        except:
            io.warning('I Could not load the {} folder!'.format(name))
        spectra_names[name] = data_collection[name].y_model.spectra.names
    names = list(data_collection.keys())

    # Init dictionaries
    for name in names:
        data[name] = {}
        emu[name] = {}
        derived[name] = {}

    # Get spectra to train
    if args.spectra is None:
        spectra_to_train = list(set([x for xs in [spectra_names[k] for k in spectra_names] for x in xs]))
    else:
        spectra_to_train = args.spectra

    # Manipulate data
    for spectrum in spectra_to_train:
        for name in names:
            data[name][spectrum] = data_collection[name].get_one_y_dataset(name=spectrum)
            data[name][spectrum].train_test_split(
                frac_train=0.9,
                seed=1543)
            data[name][spectrum].rescale(
                rescale_x = 'MinMaxScaler',
                rescale_y = 'MinMaxCommonScaler')

    # Load or train the emulators
    io.info('Training models')
    for spectrum in spectra_to_train:
        derived[name][spectrum] = {}
        for name in names:

            if args.force:
                do_train_emu = True
            else:
                try:
                    emu[name][spectrum] = FFNNEmu().load(train_folder.subfolder(name).subfolder(spectrum).path)
                    io.print_level(1, 'Model {}, spectrum {} loaded!'.format(name, spectrum))
                    do_train_emu = False
                except ValueError:
                    do_train_emu = True
                
            if do_train_emu:
                io.print_level(1, 'Training model {}, spectrum {}!'.format(name, spectrum))
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
                    path = train_folder.subfolder(name).subfolder(spectrum).path
                    )

            # Get derived quantities
            y_emu_train = np.array([emu[name][spectrum].eval(x) for x in data[name][spectrum].x_train])
            y_emu_test = np.array([emu[name][spectrum].eval(x) for x in data[name][spectrum].x_test])
            derived[name][spectrum]['y_emu'] = np.array([emu[name][spectrum].eval(x) for x in data[name][spectrum].x])
            derived[name][spectrum]['diff'] = (derived[name][spectrum]['y_emu']/data[name][spectrum].y-1)
            derived[name][spectrum]['diff_train'] = (y_emu_train/data[name][spectrum].y_train-1)
            derived[name][spectrum]['diff_test'] = (y_emu_test/data[name][spectrum].y_test-1)
        
        # Summary plot
        io.info('Saving summary plot for {}'.format(spectrum))

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
            ax[0, count].plot(x, derived[name][spectrum]['diff_train'].T*100, 'k-', alpha=0.2)

            # Validation set
            ax[1, count].plot(x, derived[name][spectrum]['diff_test'].T*100, 'k-', alpha=0.2)

            idx_max = np.argmax(np.max(np.abs(derived[name][spectrum]['diff']), axis=1))

            # Worst fit, rel diff
            ax[2, count].plot(x, derived[name][spectrum]['diff'][idx_max]*100, 'k-')

            # Worst fit, P/P_ref
            ax[3, count].plot(x, derived[name][spectrum]['y_emu'][idx_max], label='Emulated')
            ax[3, count].plot(x, data[name][spectrum].y[idx_max], '--', label='True')
            ax[3, count].legend()

            # Worst fit, P
            if data[names[0]][spectrum].y_model.spectra[0].is_pk:
                z = data[name][spectrum].x[idx_max, 0]
                ref = interp.make_splrep(emu[name][spectrum].y_model.z_array, emu[name][spectrum].y_model.y_ref[0][0].T, s=0)(z)
            elif data[names[0]][spectrum].y_model.spectra[0].is_cl:
                ref = emu[name][spectrum].y_model.y_ref[0][0]
            ax[4, count].plot(x, ref*derived[name][spectrum]['y_emu'][idx_max])
            ax[4, count].plot(x, ref*data[name][spectrum].y[idx_max], '--')

            # Print stuff
            io.info('Worst fit {} - {} points'.format(spectrum, name))
            for var, val in zip(data[name][spectrum].x_names, data[name][spectrum].x[idx_max]):
                io.print_level(1, '{} = {}'.format(var, val))
            print()

        plt.subplots_adjust(bottom=0.15, hspace=0.05, wspace=0.15)
        plt.savefig(os.path.join(train_folder.path, 'summary_{}.pdf'.format(spectrum)))
