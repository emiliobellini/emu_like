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
import time
from types import SimpleNamespace
from . import io as io
from .emu import Emulator
from .pca import PCA
from .scalers import Scaler
from .y_models import YModel
from . import loss_functions as lf  # noqa:F401


class LearningRateLogger(keras.callbacks.Callback):
    """Attach the current optimizer learning rate to epoch logs."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        try:
            value = tf.keras.backend.get_value(lr)
        except Exception:
            value = lr
        try:
            logs['learning_rate'] = float(value)
        except (TypeError, ValueError):
            logs['learning_rate'] = value


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
        self.x_pca = None
        self.y_pca = None
        self.x_names = None
        self.y_names = None
        self.x_ranges = None
        self.epochs = []  # List of the epochs run
        self.learning_rate = []  # List of learning_rates per epoch
        self.loss = []  # List of the losses per epoch
        self.val_loss = []  # List of the validation losses per epoch
        # Defaults
        self.x_scaler_fname = 'x_scaler.save'
        self.y_scaler_fname = 'y_scaler.save'
        self.x_pca_fname = 'x_pca.save'
        self.y_pca_fname = 'y_pca.save'
        self.model_fname = 'model.keras'
        self.checkpoint_folder = 'checkpoints'
        self.checkpoint_fname = 'checkpoint_epoch{epoch:04d}.weights.h5'
        self.log_fname = 'history_log.csv'
        self.data_fname = 'data.fits'
        return

    def _callbacks(self, path=None, patience=None, timeout=None,
                   reduce_learning_rate=True, verbose=False):
        """
        Define and initialise callbacks.
        Arguments:
        - path (str, default: None): output path. If None, the callbacks
          that require saving some output will be ignored;
        - patience (int, default: None): number of epochs (int) before
          early stopping without improvements;
        - reduce_learning_rate (bool, default: True): reduce learning rate on
          plateau;
        - timeout (float, default None): after this time (in hours)
          stop the training;
        - verbose (bool, default: False): verbosity.

        Callbacks implemented:
        - Checkpoint: save the weights of a model each time that
          loss function is improved;
        - Logfile: saves a log file in the main directory
        - Early Stopping: stop earlier if loss of the validation
          dataset does not improve for a certain number of epochs.
        """
        if verbose is True:
            n_verbose = 1
        else:
            n_verbose = 0

        callbacks = [LearningRateLogger()]

        # Checkpoint
        if path is not None:
            checkpoint_folder = io.Folder(path).subfolder(
                self.checkpoint_folder).create(verbose=verbose)
            fname = os.path.join(
                checkpoint_folder.path,
                self.checkpoint_fname)
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
            fname = os.path.join(path, self.log_fname)
            csv_logger = keras.callbacks.CSVLogger(fname, append=True)

        if reduce_learning_rate:
            reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                min_delta=0.,
                patience=max(1, patience // 2),
                verbose=verbose)

        # Early Stopping
        # TODO: understand what should be passed by the user
        if patience is not None:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=patience,
                verbose=n_verbose,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
            )

        # Time Early Stopping
        if timeout is not None:
            time_early_stopping = TimeBasedEarlyStopping(
                max_time_hours=timeout,
                verbose=n_verbose)

        # Build callbacks
        if path is not None:
            callbacks.extend([csv_logger, checkpoint])
        if reduce_learning_rate:
            callbacks.append(reduce_on_plateau)
        if patience is not None:
            callbacks.append(early_stopping)
        if timeout is not None:
            callbacks.append(time_early_stopping)

        return callbacks

    def _plot_loss_per_epoch(self, path=None):
        """
        Plot - Loss per epoch. Arguments:
        - path (str, default: None): save plot to path
        """
        plt.semilogy(self.epochs, self.loss, label='training data')
        plt.semilogy(self.epochs, self.val_loss, label='validation data')
        plt.xlabel('epoch')
        plt.ylabel(self.model.loss)
        plt.legend()
        if path:
            plt.savefig(os.path.join(path, 'loss_function.pdf'))
        plt.show()
        plt.close()
        return

    def _stored_loss_name_and_floor(self, path):
        """Read the loss name and floor stored in params.yaml, if available."""
        params_file = io.YamlFile()
        try:
            params_file.read(fname='params.yaml', root=path)
        except FileNotFoundError:
            return None
        content = params_file.content or {}
        emulator_block = content.get('emulator', {}) or {}
        args_block = emulator_block.get('args', {}) or {}
        out = (
            args_block.get('loss'),
            args_block.get('loss_floor', None),
            args_block.get('loss_delta', None)
            )
        return out

    def _build_custom_loss(self, loss_name, loss_floor, loss_delta, y_pca):
        """Recreate a registered custom loss callable from disk assets."""
        if not loss_name or not hasattr(lf, loss_name):
            return None
        if y_pca is None:
            raise ValueError(
                'Cannot rebuild custom loss `{}` without the stored PCA.'
                ''.format(loss_name)
            )
        loss_factory = getattr(lf, loss_name)
        data_stub = SimpleNamespace(y_pca=y_pca)
        return loss_factory(data=data_stub, floor=loss_floor, delta=loss_delta)

    def load(self, path, model_to_load='best', still_training=True,
             verbose=False):
        """
        Load from path a model for the emulator.
        This can be used both for using the emulator
        with eval and for resuming training.
        Arguments:
        - path (str): emulator path;
        - model_to_load (str or int, default: best): which
          model shall I load? Options: 'best' or an
          integer number specifying the epoch to load;
        - still_training (bool, default: True): passes this flag to the
          keras.models.load_model option. If we want to use the model
          to evaluate it, but we do not need to train it it does not import
          the loss function;
        - verbose (bool, default: False): verbosity.

        NOTE: if model_to_load is an integer, make sure that the
        epoch is saved in checkpoints, since the code does not
        save every epoch, but only when the val_loss improves
        (to save space).
        """

        if verbose:
            io.info('Loading FFNN architecture')

        custom_objects = None
        preloaded_y_pca = None
        if still_training:
            stored_loss, stored_floor, stored_delta =\
                self._stored_loss_name_and_floor(path)
            if stored_loss and hasattr(lf, stored_loss):
                y_pca_path = os.path.join(path, self.y_pca_fname)
                if not os.path.isfile(y_pca_path):
                    raise FileNotFoundError(
                        'Expected PCA file at {} to rebuild custom loss `{}`'
                        ''.format(y_pca_path, stored_loss)
                    )
                preloaded_y_pca = PCA.load(y_pca_path, verbose=verbose)
                custom_loss = self._build_custom_loss(
                    stored_loss, stored_floor, stored_delta, preloaded_y_pca)
                if custom_loss is not None:
                    custom_objects = {
                        'loss': custom_loss,
                        stored_loss: custom_loss,
                        'function': custom_loss,
                    }

        # Load last model
        if model_to_load == 'best':
            fname = os.path.join(path, self.model_fname)
            self.model = keras.models.load_model(
                fname,
                compile=still_training,
                custom_objects=custom_objects)
        elif isinstance(model_to_load, int):
            fname = os.path.join(path, self.model_fname)
            self.model = keras.models.load_model(
                fname,
                compile=still_training,
                custom_objects=custom_objects)
            epoch = {'epoch': model_to_load}
            fname = os.path.join(
                path,
                self.checkpoint_folder,
                self.checkpoint_fname.format(**epoch))
            self.model.load_weights(fname)
        else:
            raise Exception('Model not recognised!')

        if verbose:
            io.print_level(1, 'From: {}'.format(fname))
            self.model.summary()

        # Get additional model properties
        self.batch_size = self.model.inputs[0].shape[0]

        # Load scalers
        fname = os.path.join(path, self.x_scaler_fname)
        self.x_scaler = Scaler.load(fname, verbose=verbose)
        fname = os.path.join(path, self.y_scaler_fname)
        self.y_scaler = Scaler.load(fname, verbose=verbose)

        # Load PCA
        fname = os.path.join(path, self.x_pca_fname)
        self.x_pca = PCA.load(fname, verbose=verbose)
        fname = os.path.join(path, self.y_pca_fname)
        if preloaded_y_pca is not None:
            self.y_pca = preloaded_y_pca
        else:
            self.y_pca = PCA.load(fname, verbose=verbose)

        # Init fits file
        fits = io.FitsFile(self.data_fname, root=path)

        # Load parameters
        params = fits.get_header(0, unflat_dict=True)

        # Dataset details
        self.x_names = params['x_names']
        self.y_names = params['y_names']
        self.x_ranges = params['x_ranges']

        # Load history
        try:
            fname = os.path.join(path, self.log_fname)
            history = np.genfromtxt(fname, delimiter=',', skip_header=1)
            self.epochs = [int(x) for x in history[:, 0]]
            self.learning_rate = list(history[:, 1])
            self.loss = list(history[:, 2])
            self.val_loss = list(history[:, 3])
        except FileNotFoundError:
            pass

        # Init y_model
        self.y_model = YModel.choose_one(
            params['y_model']['name'],
            params['y_model']['params'],
            params['y_model']['outputs'],
            params['y_model']['n_samples'],
            **params['y_model']['args'],
            verbose=False)
        # Load y_model
        self.y_model.load(self.data_fname, root=path, verbose=verbose)

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
            self.x_scaler.save(
                self.x_scaler_fname,
                root=path,
                verbose=verbose)
        except AttributeError:
            io.warning('x_scaler not loaded yet, impossible to save it!')
        try:
            self.y_scaler.save(
                self.y_scaler_fname,
                root=path,
                verbose=verbose)
        except AttributeError:
            io.warning('y_scaler not loaded yet, impossible to save it!')

        # Save PCA
        try:
            self.x_pca.save(
                self.x_pca_fname,
                root=path,
                verbose=verbose)
        except AttributeError:
            io.warning('x_pca not loaded yet, impossible to save it!')
        try:
            self.y_pca.save(
                self.y_pca_fname,
                root=path,
                verbose=verbose)
        except AttributeError:
            io.warning('y_pca not loaded yet, impossible to save it!')

        # Save last model
        fname = os.path.join(path, self.model_fname)
        if verbose:
            io.info('Saving model at {}'.format(fname))
        self.model.save(fname, overwrite=True)

        fits = io.FitsFile(
            fname=self.data_fname,
            root=path,
        )
        params = {
            'x_names': self.x_names,
            'y_names': self.y_names,
            'x_ranges': self.x_ranges,
            'y_model': {
                'name': self.y_model.name,
                'params': self.y_model.params,
                'outputs': self.y_model.outputs,
                'n_samples': self.y_model.n_samples,
                'args': self.y_model.args,
            }
        }

        fits.write(
            name=None,
            data=None,
            header=params,
            verbose=verbose,
        )

        # Save y_model to the same file
        self.y_model.save(self.data_fname, root=path, verbose=verbose)

        return

    def build(self, params, data=None, verbose=False):
        """
        Build emulator architecture.
        Arguments:
        - params (dict, default: None): parameters for the emulator;
        - data (src.emu_like.datasets.Dataset object): class
          with the dataset (already loaded, rescaled and split
          into training and testing samples). Some loss function needs
          extra arguments that can be inferred from the data class.
          E.g., when doing the pca it is possible to choose a loss
          function that weights the different modes. For that we need
          the pca transofrmation done;
        - verbose (bool, default: False): verbosity.

        The params dictionary should contain the following keys:
        - activation (str): any activation function from (str)
          https://keras.io/api/layers/activations;
        - data_n_x (int): number of x variables. Here we use it to fix
          the number of neurons of the input layer;
        - neurons_hidden (list of positive int): number of neurons
          for each hidden layer;
        - data_n_y (int): number of y variables. Here we use it to fix
          the number of neurons of the output layer;
        - batch_normalization (bool): normalize tensors with mean and variance;
        - dropout_rate (float): relative dropout during training.
          It helps with overfitting;
        - batch_size (int): divide dataset into batches of this size;
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
        try:
            loss_function = eval('lf.'+params['loss'])(
                data=data,
                floor=params.get('loss_floor', None),
                delta=params.get('loss_delta', None))
        except AttributeError:
            loss_function = params['loss']

        model = tf.keras.Sequential()
        # Input layer
        model.add(
            keras.layers.Input(
                shape=(params['data_n_x'],),
                batch_size=self.batch_size)
            )
        if params['batch_normalization']:
            model.add(keras.layers.BatchNormalization())
        # Hidden layers
        for i in range(len(params['neurons_hidden'])):
            model.add(
                keras.layers.Dense(params['neurons_hidden'][i],
                                   activation=params['activation'],
                                   kernel_initializer='he_normal'))
        if params['batch_normalization']:
            model.add(keras.layers.BatchNormalization())
        if params['dropout_rate'] > 0:
            model.add(keras.layers.Dropout(params['dropout_rate']))
        # Output layer
        if want_output_layer:
            model.add(keras.layers.Dense(params['data_n_y'],
                                         activation=None))

        model.compile(optimizer=params['optimizer'], loss=loss_function)

        self.model = model

        if verbose:
            model.summary()

        return

    def train(self, data, epochs, learning_rate, patience=100,
              path=None, timeout=None, reduce_learning_rate=True,
              get_plots=False, verbose=False):
        """
        Train the emulator.
        Arguments:
        - data (src.emu_like.datasets.Dataset object): class
          with the dataset (already loaded, rescaled and split
          into training and testing samples) that should be
          used to train the emulator;
        - epochs (int): epochs to run;
        - learning_rate (float): learning rate;
        - patience (intm default: 100): number of epochs (int) before
          early stopping without improvements;
        - path (str, default: None): output path. If None,
          the emulator will not be saved;
        - timeout (float, default None): after this time (in hours)
          stop the training;
        - reduce_learning_rate (bool, default: True): reduce learning rate on
          plateau;
        - get_plots (bool, default: False): get loss vs epoch plot;
        - verbose (bool, default: False): verbosity.
        """

        # Store dataset details as attributes
        self.x_scaler = data.x_scaler
        self.y_scaler = data.y_scaler
        self.x_pca = data.x_pca
        self.y_pca = data.y_pca
        self.x_names = data.x_names
        self.y_names = data.y_names
        self.x_ranges = data.x_ranges
        self.y_model = data.y_model

        # Callbacks
        callbacks = self._callbacks(
            path,
            patience=patience,
            timeout=timeout,
            reduce_learning_rate=reduce_learning_rate,
            verbose=verbose)

        self.model.optimizer.learning_rate = learning_rate

        # Fit model
        if self.epochs:
            initial_epoch = self.epochs[-1] + 1
        else:
            initial_epoch = 0
            # Save immediately architecture to resume
            # training in case of crashes
            if path:
                self.save(path)
        self.model.fit(
            data.x_train,
            data.y_train,
            epochs=initial_epoch+epochs,
            initial_epoch=initial_epoch,
            batch_size=self.batch_size,
            validation_data=(
                data.x_test,
                data.y_test),
            callbacks=callbacks,
            verbose=int(verbose))

        # Update history
        self.epochs = self.epochs + self.model.history.epoch
        self.learning_rate =\
            self.learning_rate + self.model.history.history['learning_rate']
        self.loss = self.loss + self.model.history.history['loss']
        self.val_loss = self.val_loss + self.model.history.history['val_loss']

        # Save emulator
        if path:
            self.save(path)

        if get_plots:
            # Plot - Loss per epoch
            self._plot_loss_per_epoch(path=path)

            # Model specific plots (TODO: take too much time)
            if False:
                data.y_model.plot(self, data, path=path)

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
        if self.x_scaler is None:
            x_scaled = x_reshaped
        else:
            x_scaled = self.x_scaler.transform(x_reshaped)

        # PCA x
        if self.x_pca is None:
            x_scaled_pca = x_scaled
        else:
            x_scaled_pca = self.x_pca.transform(x_scaled)

        # Emulate y
        y_scaled_pca = self.model(x_scaled_pca, training=False)

        # inverse PCA y
        if self.y_pca is None:
            y_scaled = y_scaled_pca
        else:
            y_scaled = self.y_pca.inverse_transform(y_scaled_pca)

        # Scale back y
        if self.y_scaler is None:
            y = y_scaled
        else:
            y = self.y_scaler.inverse_transform(y_scaled)[0]

        return y


class TimeBasedEarlyStopping(keras.callbacks.Callback):
    def __init__(self, max_time_hours, verbose=False):
        super().__init__()
        self.max_time_hours = max_time_hours
        self.start_time = None
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time > self.max_time_hours*60.*60.:
            self.model.stop_training = True
            if self.verbose:
                print(f'\nEarly stopping: {elapsed_time:.2f}s'
                      ' > {self.max_time_hours*60.*60.}s')
