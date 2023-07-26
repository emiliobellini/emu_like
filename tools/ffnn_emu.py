import numpy as np
import tensorflow as tf
from tensorflow import keras
import tools.defaults as de
import tools.io as io
import tools.plots as pl
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

    def get_sample(self, resume=None, verbose=False, get_plots=False):
        """
        This perform three tasks on samples:
         (i) load or generate it.
         (ii) splits it into training and testing sample.
         (iii) rescale the sample
        """

        if resume:
            out_x = self.output.subfolder(de.file_names['x_sample']['folder'])
            out_y = self.output.subfolder(de.file_names['y_sample']['folder'])
            self.params['training_sample'] = {
                'path_x': io.File(
                    de.file_names['x_sample']['name'], root=out_x).path,
                'path_y': io.File(
                    de.file_names['y_sample']['name'], root=out_y).path,
            }
            sample = LoadSample(
                self.params['training_sample'],
                verbose=verbose)
        else:
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
                sample.generate(verbose=verbose)
            else:
                raise Exception(
                    'Do you want to load a pre-generated sample or '
                    'generate it? Please specify one of training_sample, '
                    'generate_sample.')

            # Save in output folder sample
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

    def call_backs(self, verbose=False):

        # Checkpoint
        path = self.output.subfolder(
            de.file_names['checkpoint']['folder']).create(verbose=verbose)
        fname = io.File(de.file_names['checkpoint']['name'], root=path)
        # TODO: understand what should be passed by the user
        checkpoint = keras.callbacks.ModelCheckpoint(
            fname.path,
            monitor='val_loss',
            verbose=int(verbose),
            save_best_only=True,
            mode='auto',
            save_freq='epoch',
            save_weights_only=True)

        # Logfile
        path = self.output.subfolder(
            de.file_names['log']['folder']).create(verbose=verbose)
        fname = io.File(de.file_names['log']['name'], root=path)
        csv_logger = keras.callbacks.CSVLogger(fname.path, append=True)

        # Early Stopping
        # TODO: understand what should be passed by the user
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=15000,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        call_backs = [csv_logger, early_stopping, checkpoint]

        return call_backs

    def train(self, resume=None, verbose=False, get_plots=False):
        """
        Train Feed-forward Neural-Network emulator.
        """

        # Load/generate the sample, train/test split and rescale
        sample = self.get_sample(
            resume=resume, verbose=verbose, get_plots=get_plots)

        if resume:
            path = self.output.subfolder(
                de.file_names['model']['folder']).create(verbose=verbose)
            fname = io.File(de.file_names['model']['name'], root=path).path
            self.model = keras.models.load_model(fname)
            ep_ini = self.params['ffnn_model']['n_epochs']
            ep_tot = ep_ini + self.params['ffnn_model']['additional_epochs']
        else:
            # Get architecture
            self.model = self.build_model(
                sample.n_x, sample.n_y, verbose=verbose)
            ep_ini = 0
            ep_tot = self.params['ffnn_model']['n_epochs']

        # Get call_backs
        call_backs = self.call_backs(verbose=verbose)

        # Fit model
        self.model.fit(
            sample.x_train_scaled,
            sample.y_train_scaled,
            epochs=ep_tot,
            initial_epoch=ep_ini,
            batch_size=self.params['ffnn_model']['batch_size'],
            validation_data=(
                sample.x_test_scaled,
                sample.y_test_scaled),
            callbacks=call_backs,
            verbose=int(verbose))

        # Save model
        path = self.output.subfolder(
            de.file_names['model']['folder']).create(verbose=verbose)
        fname = io.File(de.file_names['model']['name'], root=path).path
        if verbose:
            scp.info('Saving model at {}'.format(fname))
        self.model.save(fname, overwrite=True)

        if get_plots:
            path = self.output.subfolder(
                de.file_names['log']['folder']).create(verbose=verbose)
            fname = io.File(de.file_names['log']['name'], root=path).path
            data = np.genfromtxt(fname, delimiter=",", skip_header=1)

            # Loss per epoch
            ee = data[:, 0] + 1
            loss = data[:, 1]
            val_loss = data[:, 2]
            pl.LogLogPlot(
                [(ee, loss),
                 (ee, val_loss)],
                labels=['loss', 'val_loss'],
                x_label='epoch',
                y_label=self.params['ffnn_model']['loss_function'],
                fname='loss_function.pdf',
                root=self.output.subfolder('plots'),
                verbose=verbose).save()

            if sample.n_x == 1 and sample.n_y == 1:
                y_emu = self.model(sample.x_train_scaled, training=False)
                y_emu = sample.scaler_y.inverse_transform(y_emu)
                pl.ScatterPlot(
                    [(sample.x_train[:, 0], sample.y_train[:, 0]),
                     (sample.x_train[:, 0], y_emu[:, 0])],
                    labels=['true', 'emulated'],
                    x_label='x_0',
                    y_label='y_0',
                    root=self.output.subfolder('plots'),
                    fname='true_vs_emulated.pdf',
                    verbose=verbose).save()
        return
