import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tools.io as io
import tools.printing_scripts as scp
from tools.emu import Emulator
from tools.sample import LoadSample, GenerateSample
import tools.loss_functions as lf  # noqa:F401


class FFNNEmu(Emulator):
    """
    Evaluate sampler.
    """

    def __init__(self, params, output, verbose=False):
        if verbose:
            scp.info('Initializing FFNNEmu emulator.')
        Emulator.__init__(self, params, output)
        return

    def get_sample(self, verbose=False, get_plots=False):
        """
        This perform three tasks on samples:
         (i) load or generate it.
         (ii) splits it into training and testing sample.
         (iii) rescale the sample
        """

        # Load or generate sample
        want_training = 'training_sample' in self.params.keys()
        want_generate = 'generate_sample' in self.params.keys()
        if want_training and want_generate:
            raise Exception(
                'You can not specify both training_sample and '
                'generate_sample. Please comment one of them!')
        elif want_training:
            sample = LoadSample(
                self.params['training_sample'],
                verbose=verbose)
        elif want_generate:
            sample = GenerateSample(
                self.params['generate_sample'],
                verbose=verbose)
            sample.generate()
        else:
            raise Exception(
                'Do you want to load a pre-generated sample or '
                'generate it? Please specify one of training_sample, '
                'generate_sample.')

        # Eventually save in output folder sample
        sample.save(self.output, verbose=verbose)

        # Split training and testing samples
        sample.train_test_split(
            self.params['frac_train'],
            self.params['train_test_random_seed'],
            verbose=verbose)

        # If requested, rescale training and testing samples
        sample.rescale(
            self.params['rescale_x'],
            self.params['rescale_y'],
            verbose=verbose)

        # Plots
        if get_plots:
            sample.get_plots(self.output, verbose=verbose)

        return sample

    def build_model(self, n_x, n_y, verbose=False):
        if verbose:
            scp.info('Building FFNN architecture')

        # Local variables
        neur_inp_lay = n_x
        neur_out_lay = n_y
        activation = self.params['ffnn_model']['activation_function']
        neur_hid_lay = self.params['ffnn_model']['neurons_hidden_layer']
        batch_norm = self.params['ffnn_model']['batch_normalization']
        dropout_rate = self.params['ffnn_model']['dropout_rate']
        optimizer = self.params['ffnn_model']['optimizer']
        loss = self.params['ffnn_model']['loss_function']
        try:
            want_output_layer = self.params['ffnn_model']['want_output_layer']
        except KeyError:
            want_output_layer = True

        if verbose:
            scp.print_level(1, 'Activation function: {}'.format(activation))
            scp.print_level(1, 'Dropout rate: {}'.format(dropout_rate))
            scp.print_level(1, 'Optimizer: {}'.format(optimizer))
            scp.print_level(1, 'Loss function: {}'.format(loss))

        # Get loss function
        try:
            loss = eval('lf.'+loss)
        except AttributeError:
            pass

        model = tf.keras.Sequential()
        # Input layer
        model.add(
            keras.layers.Dense(
                neur_inp_lay,
                activation=activation,
                input_shape=(neur_inp_lay,)))
        if batch_norm:
            model.add(keras.layers.BatchNormalization())
        # Hidden layers
        for i in range(len(neur_hid_lay)):
            model.add(
                keras.layers.Dense(neur_hid_lay[i], activation=activation))
        if batch_norm:
            model.add(keras.layers.BatchNormalization())
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
        # Output layer
        if want_output_layer:
            model.add(keras.layers.Dense(neur_out_lay, activation=None))

        model.compile(optimizer=optimizer, loss=loss)

        if verbose:
            model.summary()
        return model

    def train(self, verbose=False, get_plots=False):
        """
        Train Feed-forward Neural-Network emulator.
        """

        # Load/generate the sample, train/test split and rescale
        sample = self.get_sample(verbose=verbose, get_plots=get_plots)

        # Get architecture
        self.model = self.build_model(sample.n_x, sample.n_y, verbose=verbose)

        # Fit model
        history = self.model.fit(
            sample.x_train_scaled,
            sample.y_train_scaled,
            epochs=self.params['ffnn_model']['n_epochs'],
            batch_size=self.params['ffnn_model']['batch_size'],
            validation_data=(
                sample.x_test_scaled,
                sample.y_test_scaled),
            verbose=int(verbose))

        if get_plots:
            path = self.output.subfolder('plots').create(verbose=verbose)
            # Loss per epoch
            ee = np.array(history.epoch)+1
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            plt.loglog(ee, loss, label='loss', color='firebrick', lw=3)
            plt.loglog(ee, val_loss, label='val loss', color='royalblue', lw=3)
            plt.legend()
            plt.xlabel(r'epoch', fontsize=20)
            fname = io.File('loss_vs_epoch.pdf', root=path)
            fname.savefig(plt, verbose=verbose)

            if sample.n_x == 1 and sample.n_y == 1:
                y_emu = self.model(sample.x_train_scaled, training=False)
                y_emu = sample.scaler_y.inverse_transform(y_emu)
                plt.scatter(sample.x_train, sample.y_train, s=1, label='true')
                plt.scatter(sample.x_train, y_emu, s=1, label='emulated')
                plt.legend()
                plt.xlabel('x_0')
                fname = io.File('true_vs_emulated.pdf', root=path)
                fname.savefig(plt, verbose=verbose)
        return
