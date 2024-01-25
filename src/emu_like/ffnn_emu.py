import numpy as np
import tensorflow as tf
from tensorflow import keras
from . import defaults as de
from . import io as io
from . import plots as pl
from .emu import Emulator
from . import loss_functions as lf  # noqa:F401


class FFNNEmu(Emulator):
    """
    Feed Forward Neural Network emulator.
    Available methods:
    - build: build the emulator architecture from a dictionary of parameters;
    - load: load an existing emulator from a folder, either to resume it or
      to continue training;
    - save: save emulator into a folder;
    - train: train the emulator;
    - eval: evaluate the emulator on a x vector.
    """

    def __init__(self, verbose=False):
        """
        Initialise FFNN Emulator.
        Arguments:
        - verbose (bool, default: False): verbosity.
        """
        if verbose:
            io.info('Initializing FFNNEmu emulator.')
        Emulator.__init__(self)
        self.name = 'ffnn_emu'
        return

    def build(self, params, verbose=False):
        """
        Build emulator architecture.
        Arguments:
        - params (dict, default: None): parameters for the emulator;
        - verbose (bool, default: False): verbosity.

        The params dictionary should contain the following keys:
        - activation (str): any activation function from (str)
          https://keras.io/api/layers/activations;
        - sample_n_x (int): number of x variables. Here we use it to fix
          the number of neurons of the input layer;
        - neurons_hidden (list of positive int): number of neurons
          for each hidden layer;
        - sample_n_y (int): number of y variables. Here we use it to fix
          the number of neurons of the output layer;
        - batch_normalization (bool): normalize tensors with mean and variance;
        - dropout_rate (float): relative dropout during training.
          It helps with overfitting;
        - optimizer (str): any optimizer from
          https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        - loss (str): name of the loss function. Options: any of the functions
          defined in https://keras.io/api/losses/ (prepending keras.losses.),
          plus the ones defined in src/emu_like/loss_functions.py;
        - want_output_layer (bool, default: True): if False remove the
          output layer. If False, useful to reduce to linear regression case.
        """
        # Build model architecture
        if verbose:
            io.info('Building FFNN architecture')

        # Local variables
        try:
            want_output_layer = params['want_output_layer']
        except KeyError:
            want_output_layer = True

        if verbose:
            io.print_level(1, 'Activation function: {}'
                           ''.format(params['activation']))
            io.print_level(1, 'Dropout rate: {}'
                           ''.format(params['dropout_rate']))
            io.print_level(1, 'Optimizer: {}'.format(params['optimizer']))
            io.print_level(1, 'Loss function: {}'.format(params['loss']))

        # Get loss function
        try:
            loss = eval('lf.'+params['loss'])
        except AttributeError:
            pass

        model = tf.keras.Sequential()
        # Input layer
        model.add(
            keras.layers.Dense(
                params['sample_n_x'],
                activation=params['activation'],
                input_shape=(params['sample_n_x'],)))
        if params['batch_normalization']:
            model.add(keras.layers.BatchNormalization())
        # Hidden layers
        for i in range(len(params['neurons_hidden'])):
            model.add(
                keras.layers.Dense(params['neurons_hidden'][i],
                                   activation=params['activation']))
        if params['batch_normalization']:
            model.add(keras.layers.BatchNormalization())
        if params['dropout_rate'] > 0:
            model.add(keras.layers.Dropout(params['dropout_rate']))
        # Output layer
        if want_output_layer:
            model.add(keras.layers.Dense(params['sample_n_y'],
                                         activation=None))

        model.compile(optimizer=params['optimizer'], loss=loss)

        self.model = model

        if verbose:
            model.summary()

        return

    def load(self):
        """
        Placeholder for load.
        TODO: write description
        """
        return

    def save(self, path, verbose=False):
        """
        Save the emulator to path.
        Arguments:
        - path (str): output path;
        - verbose (bool, default: False): verbosity.
        """
        self.path = path

        if verbose:
            io.print_level(1, 'Saving output at: {}'.format(path))

        # Create main folder
        io.Folder(path).create(verbose=verbose)

        # Save scalers
        print()
        exit()

        # Save settings
        self._save_settings(path, verbose=False)
        
        # Save x
        self._save_x(path, verbose=False)

        # Save y
        self._save_y(path, verbose=False)
        return

    def train(self):
        """
        Placeholder for train.
        TODO: write description
        """
        return

    def eval(self):
        """
        Placeholder for eval.
        TODO: write description
        """
        return


    def load_old(self, model_to_load='last', verbose=False):

        path = self.output.subfolder(
            de.file_names['model_last']['folder'])
        fname = io.File(de.file_names['model_last']['name'], root=path).path

        if verbose:
            io.info('Loading FFNN architecture')

        model = keras.models.load_model(fname)

        if model_to_load == 'last':
            self.model = model
            if verbose:
                io.print_level(1, 'From: {}'.format(fname))

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
                io.print_level(1, 'From: {}'.format(model_file.path))

            model.load_weights(model_file.path)

        self.model = model

        if verbose:
            model.summary()

        return

    def save_old(self, verbose=False):
        # Save last model
        path = self.output.subfolder(
            de.file_names['model_last']['folder']).create(verbose=verbose)
        fname = io.File(de.file_names['model_last']['name'], root=path).path
        if verbose:
            io.info('Saving last model at {}'.format(fname))
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
            io.info('Saving best model at {}'.format(fname))
        return

    def call_backs_old(self, verbose=False):

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

    def train_old(self, sample, verbose=False, get_plots=False):
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

    def get_last_epoch_run_old(self):
        history = self.output.subfolder(
            de.file_names['log']['folder'])
        history = io.File(de.file_names['log']['name'], root=history)
        history.load_array(delimiter=',')
        epochs = history.content[:, 0]
        return int(epochs[-1])

    def _get_epoch_best_model_old(self):
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
