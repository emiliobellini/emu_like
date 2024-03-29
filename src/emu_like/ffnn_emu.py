import numpy as np
import tensorflow as tf
from tensorflow import keras
from . import defaults as de
from . import io as io
from . import plots as pl
from . import printing_scripts as scp
from .emu import Emulator
from . import loss_functions as lf  # noqa:F401


class FFNNEmu(Emulator):
    """
    Evaluate sampler.
    """

    def __init__(self, params, output, verbose=False):
        if verbose:
            scp.info('Initializing FFNNEmu emulator.')
        Emulator.__init__(self, params, output)
        return

    def build(self, n_x, n_y, verbose=False):
        # Build model architecture
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

        self.model = model

        if verbose:
            model.summary()

        return

    def load(self, model_to_load='last', verbose=False):

        path = self.output.subfolder(
            de.file_names['model_last']['folder'])
        fname = io.File(de.file_names['model_last']['name'], root=path).path

        if verbose:
            scp.info('Loading FFNN architecture')

        model = keras.models.load_model(fname)

        if model_to_load == 'last':
            self.model = model
            if verbose:
                scp.print_level(1, 'From: {}'.format(fname))

        else:
            if model_to_load == 'best':
                epoch_min = self._get_epoch_best_model()

            elif isinstance(model_to_load, int):
                epoch_min = {'epoch': model_to_load}

            else:
                raise Exception('Model not recognised!')

            fname = de.file_names['checkpoint']['name'].format(**epoch_min)
            model_folder = self.output.subfolder(
                de.file_names['checkpoint']['folder'])
            model_file = io.File(fname, root=model_folder)
            if verbose:
                scp.print_level(1, 'From: {}'.format(model_file.path))

            model.load_weights(model_file.path)

        self.model = model

        if verbose:
            model.summary()

        return

    def save(self, verbose=False):
        # Save last model
        path = self.output.subfolder(
            de.file_names['model_last']['folder']).create(verbose=verbose)
        fname = io.File(de.file_names['model_last']['name'], root=path).path
        if verbose:
            scp.info('Saving last model at {}'.format(fname))
        self.model.save(fname, overwrite=True)

        # Save best model
        epoch_min = self._get_epoch_best_model()
        fname = de.file_names['checkpoint']['name'].format(**epoch_min)
        model_folder = self.output.subfolder(
            de.file_names['checkpoint']['folder'])
        model_file = io.File(fname, root=model_folder)
        self.model.load_weights(model_file.path)

        path = self.output.subfolder(
            de.file_names['model_best']['folder']).create(verbose=verbose)
        fname = io.File(de.file_names['model_best']['name'], root=path).path
        self.model.save(fname, overwrite=True)
        if verbose:
            scp.info('Saving best model at {}'.format(fname))
        return

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

    def train(self, sample, verbose=False, get_plots=False):
        """
        Train Feed-forward Neural-Network emulator.
        """

        # Get call_backs
        call_backs = self.call_backs(verbose=verbose)

        # Learning rate
        lr = self.params['ffnn_model']['learning_rate']
        # In YamlFile.update_params this is converted into a list to
        # keep track of the run history
        if isinstance(lr, list):
            lr = lr[-1]
        self.model.optimizer.learning_rate = lr

        # Fit model
        self.model.fit(
            sample.x_train_scaled,
            sample.y_train_scaled,
            epochs=self.total_epochs,
            initial_epoch=self.initial_epoch,
            batch_size=self.params['ffnn_model']['batch_size'],
            validation_data=(
                sample.x_test_scaled,
                sample.y_test_scaled),
            callbacks=call_backs,
            verbose=int(verbose))

        if get_plots:
            # Loss per epoch
            path = self.output.subfolder(
                de.file_names['log']['folder']).create(verbose=verbose)
            fname = io.File(de.file_names['log']['name'], root=path).path
            data = np.genfromtxt(fname, delimiter=",", skip_header=1)
            pl.LogLogPlot(
                [(data[:, 0] + 1, data[:, 1]),
                 (data[:, 0] + 1, data[:, 2])],
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

    def get_last_epoch_run(self):
        history = self.output.subfolder(
            de.file_names['log']['folder'])
        history = io.File(de.file_names['log']['name'], root=history)
        history.load_array(delimiter=',')
        epochs = history.content[:, 0]
        return int(epochs[-1])

    def _get_epoch_best_model(self):
        history = self.output.subfolder(
            de.file_names['log']['folder'])
        history = io.File(de.file_names['log']['name'], root=history)
        history.load_array(delimiter=',')
        val_loss = np.nan_to_num(history.content[:, 2], nan=np.inf)
        epochs = history.content[:, 0]
        idx_min = np.argmin(val_loss)
        # We need the +1 below because files are saved from epoch=1,
        # while the logger starts from epoch=0
        epoch_min = {'epoch': int(epochs[idx_min])+1}
        return epoch_min



