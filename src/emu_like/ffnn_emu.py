"""
.. module:: train

:Synopsis: Module with the Feed Forward Neural Network emulator class.
:Author: Emilio Bellini

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from . import defaults as de
from . import io as io
from .emu import Emulator
from .params import Params
from .scalers import Scaler
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
        # Placeholders
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.x_names = None
        self.y_names = None
        self.x_ranges = None
        self.epochs = []  # List of the epochs run
        self.loss = []  # List of the losses per epoch
        self.val_loss = []  # List of the validation losses per epoch
        return

    @staticmethod
    def _update_params(params, epochs=None, learning_rate=None):
        """
        Update the parameters of the emulator.
        In particular, it updates the learning rate
        and the number of epochs to run.
        Arguments:
        - params (src.emu_like.params.Params class):
          the params class that should be updated;
        - epochs (int, default: None): epochs that
          should be run;
        - learning rate (float): learning rate to be
          used.
        """
        # Local variables
        old_epochs = params['emulator']['params']['epochs']
        old_learning_rate = params['emulator']['params']['learning_rate']
        change_epochs = False
        change_learning_rate = False

        # Convert to list (this is useful when resuming to append new settings)
        if isinstance(old_epochs, int):
            old_epochs = [old_epochs]
        if isinstance(old_learning_rate, float):
            old_learning_rate = [old_learning_rate]

        # Decide if we have to change them
        if epochs > 0:
            change_epochs = True
        if learning_rate and learning_rate != old_learning_rate[-1]:
            change_learning_rate = True

        # Do the change
        if change_epochs or change_learning_rate:
            old_epochs.append(epochs)
            old_learning_rate.append(learning_rate)
        params['emulator']['params']['epochs'] = old_epochs
        params['emulator']['params']['learning_rate'] = old_learning_rate

        return params

    def _get_best_model_epoch(self, path=None):
        """
        Method to get the epoch of the best model,
        i.e. the one with smaller val_loss.
        It can be retrieved from path (preferentially,
        if specified) or from a loaded model.
        Arguments:
        - path (str): path to the emulator.
        """
        if path:
            fname = os.path.join(path, de.file_names['log']['name'])
            history = np.genfromtxt(fname, delimiter=",", skip_header=1)
            val_loss = np.nan_to_num(history[:, 2], nan=np.inf)
            epochs = history[:, 0]
        else:
            val_loss = self.model.history.history['val_loss']
            epochs = self.model.history.epoch
        idx_min = np.argmin(val_loss)
        # We need the +1 below because files are saved from epoch=1,
        # while the logger starts from epoch=0
        epoch_min = {'epoch': int(epochs[idx_min])+1}
        return epoch_min

    def _callbacks(self, path=None, verbose=False):
        """
        Define and initialise callbacks.
        Arguments:
        - path (str, default: None): output path. If None, the callbacks
          that require saving some output will be ignored;
        - verbose (bool, default: False): verbosity.

        Callbacks implemented:
        - Checkpoint: save the weights of a model each time that
          loss function is improved;
        - Logfile: saves a log file in the main directory
        - Early Stopping: stop earlier if loss of the validation
          sample does not improve for a certain number of epochs.
        """

        # Checkpoint
        if path:
            checkpoint_folder = io.Folder(path).subfolder(
                de.file_names['checkpoint']['folder']).create(verbose=verbose)
            fname = os.path.join(checkpoint_folder.path,
                                 de.file_names['checkpoint']['name'])
            # TODO: understand what should be passed by the user
            checkpoint = keras.callbacks.ModelCheckpoint(
                fname,
                monitor='val_loss',
                verbose=int(verbose),
                save_best_only=True,
                mode='auto',
                save_freq='epoch',
                save_weights_only=True)

        # Logfile
        if path:
            fname = os.path.join(path, de.file_names['log']['name'])
            csv_logger = keras.callbacks.CSVLogger(fname, append=True)

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

        if path:
            callbacks = [csv_logger, early_stopping, checkpoint]
        else:
            callbacks = [early_stopping]

        return callbacks

    def load(self, path, model_to_load='best', verbose=False):
        """
        Load from path a model for the emulator.
        This can be used both for using the emulator
        with eval and for resuming training.
        Arguments:
        - path (str): emulator path;
        - model_to_load (str or int, default: best): which
          model shall I load? Options: 'last', 'best' or an
          integer number specifying the epoch to load;
        - verbose (bool, default: False): verbosity.

        NOTE: if model_to_load is an integer, make sure that the
        epoch is saved in checkpoints, since the code does not
        save every epoch, but only when the val_loss improves
        (to save space).
        """

        if verbose:
            io.info('Loading FFNN architecture')

        # Load last model
        if model_to_load == 'last':
            fname = os.path.join(path, de.file_names['model_last']['name'])
            self.model = keras.models.load_model(fname)
        elif model_to_load == 'best':
            fname = os.path.join(path, de.file_names['model_best']['name'])
            self.model = keras.models.load_model(fname)
        elif isinstance(model_to_load, int):
            fname = os.path.join(path, de.file_names['model_last']['name'])
            self.model = keras.models.load_model(fname)
            epoch = {'epoch': model_to_load}
            fname = os.path.join(
                path,
                de.file_names['checkpoint']['folder'],
                de.file_names['checkpoint']['name'].format(**epoch))
            self.model.load_weights(fname)
        else:
            raise Exception('Model not recognised!')

        if verbose:
            io.print_level(1, 'From: {}'.format(fname))
            self.model.summary()

        # Load scalers
        fname = os.path.join(path, de.file_names['x_scaler']['name'])
        self.x_scaler = Scaler.load(fname, verbose=verbose)
        fname = os.path.join(path, de.file_names['y_scaler']['name'])
        self.y_scaler = Scaler.load(fname, verbose=verbose)

        # Load sample details
        fname = os.path.join(path, de.file_names['sample_details']['name'])
        details = Params().load(fname)
        self.x_names = details['x_names']
        self.y_names = details['y_names']
        self.x_ranges = details['x_ranges']

        # Load history
        fname = os.path.join(path, de.file_names['log']['name'])
        history = np.genfromtxt(fname, delimiter=",", skip_header=1)
        self.epochs = [int(x) for x in history[:, 0]]
        self.loss = history[:, 1]
        self.val_loss = history[:, 2]

        return self

    def save(self, path, verbose=False):
        """
        Save the emulator to path.
        Arguments:
        - path (str): output path;
        - verbose (bool, default: False): verbosity.
        """

        if verbose:
            io.print_level(1, 'Saving output at: {}'.format(path))

        # Create main folder
        io.Folder(path).create(verbose=verbose)

        # Save scalers
        try:
            self.x_scaler.save(de.file_names['x_scaler']['name'],
                               root=path,
                               verbose=verbose)
        except AttributeError:
            io.warning('x_scaler not loaded yet, impossible to save it!')
        try:
            self.y_scaler.save(de.file_names['y_scaler']['name'],
                               root=path,
                               verbose=verbose)
        except AttributeError:
            io.warning('y_scaler not loaded yet, impossible to save it!')

        # Save last model
        fname = os.path.join(path, de.file_names['model_last']['name'])
        if verbose:
            io.info('Saving last model at {}'.format(fname))
        self.model.save(fname, overwrite=True)

        # Save best model
        epoch_min = self._get_best_model_epoch(path=path)
        fname = os.path.join(
            path,
            de.file_names['checkpoint']['folder'],
            de.file_names['checkpoint']['name'].format(**epoch_min)
        )
        self.model.load_weights(fname)
        fname = os.path.join(path, de.file_names['model_best']['name'])
        self.model.save(fname, overwrite=True)
        if verbose:
            io.info('Saving best model at {}'.format(fname))

        # Save sample details
        # We do not always have names for 'x' and 'y'
        # In case we do not have them, just store None.
        try:
            save_x = self.x_names.tolist()
        except AttributeError:
            save_x = None
        try:
            save_y = self.y_names.tolist()
        except AttributeError:
            save_y = None
        details = Params({
            'x_names': save_x,
            'y_names': save_y,
            'x_ranges': self.x_ranges.tolist(),
        })
        fname = os.path.join(path, de.file_names['sample_details']['name'])
        details.save(fname, header=de.file_names['sample_details']['header'])

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
        - batch_size (int): divide sample into batches of this size;
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
        self.batch_size = params['batch_size']

        if verbose:
            io.print_level(1, 'Activation function: {}'
                           ''.format(params['activation']))
            io.print_level(1, 'Dropout rate: {}'
                           ''.format(params['dropout_rate']))
            io.print_level(1, 'Optimizer: {}'.format(params['optimizer']))
            io.print_level(1, 'Loss function: {}'.format(params['loss']))

        # Get loss function
        loss = eval('lf.'+params['loss'])

        model = tf.keras.Sequential()
        # Input layer
        model.add(
            keras.layers.Input(
                shape=(params['sample_n_x'],),
                batch_size=self.batch_size)
            )
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

    def train(self, sample, epochs, learning_rate,
              path=None, get_plot=False, verbose=False):
        """
        Train the emulator.
        Arguments:
        - sample (src.emu_like_sample.Sample object): class
          with the sample (already loaded, rescaled and split
          into training and testing samples) that should be
          used to train the emulator;
        - epochs (int or list of ints): epochs to run. If it is
          a list of ints, the last element will be used. List is
          used to keep record of resume;
        - learning_rate (float or list of floats): learning
          rate. If it is a list of floats, the last element
          will be used. List is used to keep record of resume;
        - path (str, default: None): output path. If None,
          the emulator will not be saved;
        - get_plot (bool, default: False): get loss vs epoch plot;
        - verbose (bool, default: False): verbosity.
        """

        # Save sample details as attributes
        self.x_scaler = sample.x_scaler
        self.y_scaler = sample.y_scaler
        self.x_names = sample.x_names
        self.y_names = sample.y_names
        self.x_ranges = sample.x_ranges

        # Take the last element of the list and use this
        if isinstance(epochs, list):
            epochs = epochs[-1]
        if isinstance(learning_rate, list):
            learning_rate = learning_rate[-1]

        # Callbacks
        callbacks = self._callbacks(path, verbose=verbose)

        self.model.optimizer.learning_rate = learning_rate

        # Fit model
        if self.epochs:
            initial_epoch = self.epochs[-1] + 1
        else:
            initial_epoch = 0
        self.model.fit(
            sample.x_train_scaled,
            sample.y_train_scaled,
            epochs=initial_epoch+epochs,
            initial_epoch=initial_epoch,
            batch_size=self.batch_size,
            validation_data=(
                sample.x_test_scaled,
                sample.y_test_scaled),
            callbacks=callbacks,
            verbose=int(verbose))

        # Update history
        self.epochs = self.epochs + self.model.history.epoch
        self.loss = self.loss + self.model.history.history['loss']
        self.val_loss = self.val_loss + self.model.history.history['val_loss']

        # Plot - Loss per epoch
        if get_plot:
            plt.semilogy(self.epochs, self.loss, label='training sample')
            plt.semilogy(self.epochs, self.val_loss, label='validation sample')
            plt.xlabel('epoch')
            plt.ylabel(self.model.loss.__name__)
            plt.legend()
            if path:
                plt.savefig(os.path.join(path, 'loss_function.pdf'))
                plt.close()

        return

    def eval(self, x):
        """
        Evaluate the emulator at a given point.
        Arguments:
        - x (dict or array): these are the input parameters.
          They can be passed as an array or as a dictionary
          with the names of x as keys.
        It returns the value(s) for y
        """

        # Adjust input
        if isinstance(x, list) or isinstance(x, np.ndarray):
            x_reshaped = np.array([x])
        elif isinstance(x, dict):
            x_reshaped = np.array([[x[el] for el in self.x_names]])
        elif isinstance(x, float) or isinstance(x, int):
            x_reshaped = np.array([[x]])
        else:
            raise ValueError('Unkown input for x!')

        # Scale x
        if self.x_scaler:
            x_scaled = self.x_scaler.transform(x_reshaped)
        else:
            x_scaled = x_reshaped

        # Emulate y
        y_scaled = self.model(x_scaled, training=False)

        # Scale back y
        y = self.y_scaler.inverse_transform(y_scaled)[0]

        return y
