"""Sobolev FFNN emulator."""

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

from . import io
from .datasets import SobolevDataset
from .ffnn_emu import FFNNEmu
from .scalers import Scaler
from .y_models import YModel


class SobolevWeightScheduler(keras.callbacks.Callback):
    """Set the derivative-loss weight from the configured warm-up schedule."""

    def __init__(self, weight, warmup_epochs, ramp_epochs):
        super().__init__()
        self.weight = float(weight)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            current_weight = 0.0
        elif self.ramp_epochs == 0:
            current_weight = self.weight
        else:
            fraction = min(1.0, (epoch - self.warmup_epochs + 1)
                           / self.ramp_epochs)
            current_weight = self.weight * fraction
        self.model.fk_weight.assign(current_weight)


class SobolevTrainingModel(keras.Model):
    """Keras model with value and redshift-derivative training losses."""

    def __init__(
            self, network, z_index, z_mean, z_scale, pk_scale,
            growth_mean, growth_scale, reference_growth, redshift_grid,
            pk_weight, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.z_index = int(z_index)
        self.z_mean = tf.constant(z_mean, dtype=tf.float32)
        self.z_scale = tf.constant(z_scale, dtype=tf.float32)
        self.pk_scale = tf.constant(pk_scale, dtype=tf.float32)
        self.growth_mean = tf.constant(growth_mean, dtype=tf.float32)
        self.growth_scale = tf.constant(growth_scale, dtype=tf.float32)
        self.reference_growth = tf.constant(reference_growth, dtype=tf.float32)
        self.redshift_grid = tf.constant(redshift_grid, dtype=tf.float32)
        self.pk_weight = tf.constant(float(pk_weight), dtype=tf.float32)
        self.fk_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.pk_loss_tracker = keras.metrics.Mean(name='pk_loss')
        self.fk_loss_tracker = keras.metrics.Mean(name='fk_loss')

    @property
    def metrics(self):
        return [self.loss_tracker, self.pk_loss_tracker, self.fk_loss_tracker]

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

    def _interpolate_reference_growth(self, z):
        """Linearly interpolate f_ref(k, z) on the stored redshift grid."""
        n_z = tf.shape(self.redshift_grid)[0]
        upper = tf.searchsorted(self.redshift_grid, z, side='right')
        upper = tf.clip_by_value(upper, 1, n_z - 1)
        lower = upper - 1
        z_lower = tf.gather(self.redshift_grid, lower)
        z_upper = tf.gather(self.redshift_grid, upper)
        fraction = (z - z_lower) / (z_upper - z_lower)
        f_lower = tf.transpose(tf.gather(self.reference_growth, lower, axis=1))
        f_upper = tf.transpose(tf.gather(self.reference_growth, upper, axis=1))
        return f_lower + fraction[:, tf.newaxis] * (f_upper - f_lower)

    def _loss_terms(self, x_scaled, pk_true, fk_true, training):
        # The outer tape differentiates through the input derivative while
        # optimizing network weights; it therefore supplies second-order
        # derivatives required by Sobolev training.
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as z_tape:
                z_tape.watch(x_scaled)
                pk_pred = self.network(x_scaled, training=training)
            jacobian = z_tape.batch_jacobian(pk_pred, x_scaled)
            d_pk_d_z_scaled = jacobian[:, :, self.z_index]

            z = (x_scaled[:, self.z_index] * self.z_scale + self.z_mean)
            d_log_ratio_d_z = (
                self.pk_scale / self.z_scale * d_pk_d_z_scaled)
            f_pred = self._interpolate_reference_growth(z) - 0.5 * (
                1.0 + z[:, tf.newaxis]) * d_log_ratio_d_z
            f_pred_scaled = (f_pred - self.growth_mean) / self.growth_scale

            pk_loss = tf.reduce_mean(tf.square(pk_pred - pk_true))
            fk_loss = tf.reduce_mean(tf.square(f_pred_scaled - fk_true))
            total_loss = self.pk_weight * pk_loss + self.fk_weight * fk_loss
        return outer_tape, total_loss, pk_loss, fk_loss

    def train_step(self, data):
        x_scaled, targets, _ = keras.utils.unpack_x_y_sample_weight(data)
        pk_true = targets['pk']
        fk_true = targets['fk']
        tape, total_loss, pk_loss, fk_loss = self._loss_terms(
            x_scaled, pk_true, fk_true, training=True)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        self.pk_loss_tracker.update_state(pk_loss)
        self.fk_loss_tracker.update_state(fk_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        x_scaled, targets, _ = keras.utils.unpack_x_y_sample_weight(data)
        _, total_loss, pk_loss, fk_loss = self._loss_terms(
            x_scaled, targets['pk'], targets['fk'], training=False)
        self.loss_tracker.update_state(total_loss)
        self.pk_loss_tracker.update_state(pk_loss)
        self.fk_loss_tracker.update_state(fk_loss)
        return {metric.name: metric.result() for metric in self.metrics}


class SobolevFFNNEmu(FFNNEmu):
    """FFNN whose later training step constrains a redshift derivative."""

    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.name = 'sobolev_ffnn_emu'
        self.z_index = None
        self.growth_scaler = None
        self.reference_pk = None
        self.reference_growth = None
        self.redshift_grid = None
        self.sobolev_params = None
        self.growth_scaler_fname = 'growth_scaler.save'
        self.reference_growth_key = 'SOB_REF_GROWTH'
        self.redshift_grid_key = 'SOB_Z_GRID'

    def build(self, params, data=None, verbose=False):
        """Build the smooth primary-pk network for Sobolev training.

        The model predicts the scaled log power-spectrum ratio.  The custom
        training step, implemented separately, will differentiate this model
        with respect to the scaled ``z_pk`` input and compare the resulting
        physical growth rate with ``data.y_growth_*``.
        """
        if not isinstance(data, SobolevDataset):
            raise TypeError('SobolevFFNNEmu.build requires a SobolevDataset')
        if data.x_train is None or data.y_train is None:
            raise ValueError(
                'Split and rescale the SobolevDataset before build')
        if data.y_growth_train is None:
            raise ValueError('SobolevDataset has no split growth-rate target')
        if data.x_pca is not None or data.y_pca is not None:
            raise ValueError('PCA is not supported by SobolevFFNNEmu')
        if 'z_pk' not in data.x_names:
            raise ValueError('SobolevDataset inputs must contain z_pk')
        if params.get('batch_normalization', False):
            raise ValueError(
                'Batch normalization is not supported by SobolevFFNNEmu: '
                'the redshift derivative must be sample-local.')
        if params.get('dropout_rate', 0.0) != 0.0:
            raise ValueError(
                'Dropout is not supported by SobolevFFNNEmu: it makes the '
                'derivative target stochastic during training.')
        if not params.get('want_output_layer', True):
            raise ValueError('SobolevFFNNEmu requires an output layer')

        activation = params['activation']
        smooth_activations = {'tanh', 'softplus', 'swish', 'sigmoid'}
        if activation not in smooth_activations:
            raise ValueError(
                'SobolevFFNNEmu requires a smooth activation; choose one of '
                '{} instead of {!r}'.format(
                    sorted(smooth_activations), activation))

        n_x = data.x_train.shape[1]
        n_y = data.y_train.shape[1]
        for key, value in (('data_n_x', n_x), ('data_n_y', n_y)):
            if key in params and params[key] != value:
                raise ValueError(
                    '{}={} disagrees with prepared data dimension {}'.format(
                        key, params[key], value))
            params[key] = value
        if data.y_growth_train.shape[1] != n_y:
            raise ValueError(
                'pk target has {} modes but growth target has {} modes'
                ''.format(n_y, data.y_growth_train.shape[1]))

        self.batch_size = params['batch_size']
        inputs = keras.Input(
            shape=(n_x,), batch_size=self.batch_size, name='x_scaled')
        hidden = inputs
        for index, neurons in enumerate(params['neurons_hidden']):
            hidden = keras.layers.Dense(
                neurons,
                activation=activation,
                kernel_initializer='glorot_uniform',
                name='hidden_{}'.format(index),
            )(hidden)
        outputs = keras.layers.Dense(n_y, activation=None, name='pk_scaled')(
            hidden)
        network = keras.Model(inputs=inputs, outputs=outputs,
                              name='sobolev_ffnn_network')

        z_scaler = data.x_scaler.skl_scaler
        pk_scaler = data.y_scaler.skl_scaler
        growth_scaler = data.growth_scaler.skl_scaler
        reference_growth = np.squeeze(data.reference_growth, axis=0)
        self.model = SobolevTrainingModel(
            network=network,
            z_index=data.x_names.index('z_pk'),
            z_mean=z_scaler.mean_[data.x_names.index('z_pk')],
            z_scale=z_scaler.scale_[data.x_names.index('z_pk')],
            pk_scale=pk_scaler.scale_,
            growth_mean=growth_scaler.mean_,
            growth_scale=growth_scaler.scale_,
            reference_growth=reference_growth,
            redshift_grid=data.redshift_grid,
            pk_weight=params.get('pk_weight', 1.0),
            name='sobolev_ffnn',
        )

        # train_step calls the inner network directly for its input
        # derivative, which would otherwise leave this outer Keras Model
        # marked as unbuilt. Build it explicitly so ModelCheckpoint can save
        # the full wrapper's weights after the first validation epoch.
        self.model(tf.zeros((self.batch_size, n_x), dtype=tf.float32))

        # No ordinary Keras loss is registered here. The forthcoming custom
        # train_step supplies both the pk value loss and derivative loss.
        self.model.compile(optimizer=params['optimizer'])

        self.x_scaler = data.x_scaler
        self.y_scaler = data.y_scaler
        self.x_pca = data.x_pca
        self.y_pca = data.y_pca
        self.growth_scaler = data.growth_scaler
        self.x_names = data.x_names
        self.y_names = data.y_names
        self.x_ranges = data.x_ranges
        self.y_model = data.y_model
        self.z_index = data.x_names.index('z_pk')
        self.reference_pk = data.reference_pk
        self.reference_growth = data.reference_growth
        self.redshift_grid = data.redshift_grid
        self.sobolev_params = {
            key: params[key] for key in (
                'pk_weight', 'fk_weight', 'fk_warmup_epochs',
                'fk_ramp_epochs', 'fk_loss') if key in params}

        if verbose:
            io.info('Building Sobolev FFNN architecture')
            io.print_level(1, 'Redshift input index: {}'.format(self.z_index))
            self.model.summary()
        return

    def train(
            self,
            data,
            epochs,
            learning_rate,
            patience=100,
            path=None,
            timeout=None,
            reduce_learning_rate=True,
            relative_improvement=True,
            get_plots=False,
            verbose=False):
        """Train on paired pk/fk batches with the Sobolev derivative loss."""
        if not isinstance(data, SobolevDataset):
            raise TypeError('SobolevFFNNEmu.train requires a SobolevDataset')
        if self.model is None:
            raise RuntimeError('Build SobolevFFNNEmu before training')
        if data.y_growth_train is None or data.growth_scaler is None:
            raise ValueError('Split and rescale the SobolevDataset before training')

        self.x_scaler = data.x_scaler
        self.y_scaler = data.y_scaler
        self.growth_scaler = data.growth_scaler
        self.x_names = data.x_names
        self.y_names = data.y_names
        self.x_ranges = data.x_ranges
        self.y_model = data.y_model

        def make_batches(x, pk, fk, shuffle):
            dataset = tf.data.Dataset.from_tensor_slices((
                np.asarray(x, dtype=np.float32),
                {
                    'pk': np.asarray(pk, dtype=np.float32),
                    'fk': np.asarray(fk, dtype=np.float32),
                }))
            if shuffle:
                dataset = dataset.shuffle(
                    buffer_size=len(x), reshuffle_each_iteration=True)
            return dataset.batch(self.batch_size, drop_remainder=True).prefetch(
                tf.data.AUTOTUNE)

        train_data = make_batches(
            data.x_train, data.y_train, data.y_growth_train, shuffle=True)
        validation_data = make_batches(
            data.x_test, data.y_test, data.y_growth_test, shuffle=False)
        if int(tf.data.experimental.cardinality(train_data).numpy()) <= 0:
            raise ValueError('Training data contains fewer rows than batch_size')
        if int(tf.data.experimental.cardinality(validation_data).numpy()) <= 0:
            raise ValueError('Validation data contains fewer rows than batch_size')

        callbacks = self._callbacks(
            path,
            patience=patience,
            timeout=timeout,
            reduce_learning_rate=reduce_learning_rate,
            relative_improvement=relative_improvement,
            verbose=verbose)
        callbacks.append(SobolevWeightScheduler(
            self.sobolev_params.get('fk_weight', 1.0),
            self.sobolev_params.get('fk_warmup_epochs', 0),
            self.sobolev_params.get('fk_ramp_epochs', 0)))
        self.model.optimizer.learning_rate = learning_rate

        initial_epoch = self.epochs[-1] + 1 if self.epochs else 0
        if path and not self.epochs:
            # Save the architecture and Sobolev metadata before fitting, as
            # the ordinary FFNN path does for interruption resilience.
            self.save(path)
        history = self.model.fit(
            train_data,
            epochs=initial_epoch + epochs,
            initial_epoch=initial_epoch,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=int(verbose))
        self.epochs += list(history.epoch)
        self.learning_rate += history.history['learning_rate']
        self.loss += history.history['loss']
        self.val_loss += history.history['val_loss']

        if path:
            self.save(path, verbose=verbose)
        if get_plots:
            self._plot_loss_per_epoch(path=path)
        return

    def _save_state(self, path, verbose=False):
        """Save inference assets plus the Sobolev derivative metadata.

        ``model.keras`` deliberately contains the plain inner network rather
        than SobolevTrainingModel.  It is consequently a conventional Keras
        model suitable for the export pipeline and fast inference; the
        Sobolev-specific state is stored separately in ``data.fits``.
        """
        required = {
            'model': self.model,
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler,
            'growth_scaler': self.growth_scaler,
            'y_model': self.y_model,
            'reference_growth': self.reference_growth,
            'redshift_grid': self.redshift_grid,
            'z_index': self.z_index,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                'Cannot save incomplete Sobolev emulator; missing {}'
                ''.format(', '.join(missing)))
        if not hasattr(self.model, 'network'):
            raise TypeError(
                'Sobolev emulator model must expose its inference network')

        io.Folder(path).create(verbose=verbose)
        self.x_scaler.save(self.x_scaler_fname, root=path, verbose=verbose)
        self.y_scaler.save(self.y_scaler_fname, root=path, verbose=verbose)
        self.growth_scaler.save(
            self.growth_scaler_fname, root=path, verbose=verbose)

        model_path = os.path.join(path, self.model_fname)
        if verbose:
            io.info('Saving Sobolev inference network at {}'.format(model_path))
        self.model.network.save(model_path, overwrite=True)

        data_path = os.path.join(path, self.data_fname)
        if os.path.isfile(data_path):
            os.remove(data_path)
        fits = io.FitsFile(fname=self.data_fname, root=path)
        state_header = {
            'x_names': self.x_names,
            'y_names': self.y_names,
            'x_ranges': self.x_ranges,
            'y_model': {
                'name': self.y_model.name,
                'params': self.y_model.params,
                'outputs': self.y_model.outputs,
                'n_samples': self.y_model.n_samples,
                'args': self.y_model.args,
            },
            'sobolev': {
                'z_index': self.z_index,
                'growth_scaler_fname': self.growth_scaler_fname,
                'reference_growth_key': self.reference_growth_key,
                'redshift_grid_key': self.redshift_grid_key,
                'model_is_inference_network': True,
                'params': self.sobolev_params or {},
            },
        }
        fits.write(name=None, data=None, header=state_header, verbose=verbose)
        self.y_model.save(self.data_fname, root=path, verbose=verbose)
        fits.write(
            name=self.reference_growth_key,
            data=np.asarray(self.reference_growth),
            verbose=verbose)
        fits.write(
            name=self.redshift_grid_key,
            data=np.asarray(self.redshift_grid),
            verbose=verbose)

        if verbose:
            io.info('Sobolev emulator saved at {}'.format(path))
        return

    def load(self, path, model_to_load='best', still_training=True,
             verbose=False):
        """Load Sobolev inference state and optionally a saved checkpoint.

        The on-disk Keras model is the plain inference network.  This method
        rebuilds the lightweight SobolevTrainingModel wrapper around it so
        the loaded emulator retains the derivative metadata needed by later
        evaluation and training work.
        """
        if verbose:
            io.info('Loading Sobolev FFNN architecture')

        # Load histories when available. A one-epoch CSV still represents a
        # one-row structured array. Sobolev metrics add columns, so these are
        # selected by name rather than by the ordinary FFNN column positions.
        try:
            history = np.genfromtxt(
                os.path.join(path, self.log_fname), delimiter=',',
                names=True)
            history = np.atleast_1d(history)
            self.epochs = [int(value) for value in history['epoch']]
            self.learning_rate = list(history['learning_rate'])
            self.loss = list(history['loss'])
            self.val_loss = list(history['val_loss'])
        except (FileNotFoundError, ValueError, IndexError, KeyError):
            pass

        self.x_scaler = Scaler.load(
            os.path.join(path, self.x_scaler_fname), verbose=verbose)
        self.y_scaler = Scaler.load(
            os.path.join(path, self.y_scaler_fname), verbose=verbose)
        self.growth_scaler = Scaler.load(
            os.path.join(path, self.growth_scaler_fname), verbose=verbose)
        self.x_pca = None
        self.y_pca = None

        fits = io.FitsFile(self.data_fname, root=path)
        state = fits.get_header(0, unflat_dict=True)
        try:
            sobolev_state = state['sobolev']
            self.z_index = int(sobolev_state['z_index'])
            self.reference_growth_key = sobolev_state[
                'reference_growth_key']
            self.redshift_grid_key = sobolev_state['redshift_grid_key']
            self.sobolev_params = dict(sobolev_state.get('params', {}))
        except KeyError as error:
            raise ValueError(
                'data.fits does not contain Sobolev state metadata') from error
        self.reference_growth = fits.get_data(self.reference_growth_key)
        self.redshift_grid = fits.get_data(self.redshift_grid_key)
        self.x_names = state['x_names']
        self.y_names = state['y_names']
        self.x_ranges = state['x_ranges']

        self.y_model = YModel.choose_one(
            state['y_model']['name'],
            state['y_model']['params'],
            state['y_model']['outputs'],
            state['y_model']['n_samples'],
            **state['y_model']['args'],
            verbose=False)
        self.y_model.load(self.data_fname, root=path, verbose=verbose)
        self.reference_pk = self.y_model.y_ref[0]

        network = keras.models.load_model(
            os.path.join(path, self.model_fname), compile=False)
        self.batch_size = network.inputs[0].shape[0]
        if self.batch_size is None:
            raise ValueError('Sobolev inference network has no fixed batch size')
        n_x = network.inputs[0].shape[-1]
        if n_x != len(self.x_names):
            raise ValueError(
                'Saved model has {} inputs but state lists {} input names'
                ''.format(n_x, len(self.x_names)))
        if not 0 <= self.z_index < n_x:
            raise ValueError('Saved Sobolev redshift index is out of range')

        z_scaler = self.x_scaler.skl_scaler
        pk_scaler = self.y_scaler.skl_scaler
        growth_scaler = self.growth_scaler.skl_scaler
        self.model = SobolevTrainingModel(
            network=network,
            z_index=self.z_index,
            z_mean=z_scaler.mean_[self.z_index],
            z_scale=z_scaler.scale_[self.z_index],
            pk_scale=pk_scaler.scale_,
            growth_mean=growth_scaler.mean_,
            growth_scale=growth_scaler.scale_,
            reference_growth=self.reference_growth,
            redshift_grid=self.redshift_grid,
            pk_weight=self.sobolev_params.get('pk_weight', 1.0),
            name='sobolev_ffnn',
        )
        self.model(tf.zeros((self.batch_size, n_x), dtype=tf.float32))

        checkpoint_path = None
        if model_to_load == 'best' and self.val_loss:
            best_epoch = self.epochs[int(np.argmin(self.val_loss))] + 1
            candidate = os.path.join(
                path, self.checkpoint_folder,
                self.checkpoint_fname.format(epoch=best_epoch))
            if os.path.isfile(candidate):
                checkpoint_path = candidate
        elif isinstance(model_to_load, int):
            candidate = os.path.join(
                path, self.checkpoint_folder,
                self.checkpoint_fname.format(epoch=model_to_load))
            if not os.path.isfile(candidate):
                raise FileNotFoundError(
                    'Checkpoint for epoch {} does not exist'.format(
                        model_to_load))
            checkpoint_path = candidate
        elif model_to_load != 'best':
            raise ValueError('Model not recognised: {}'.format(model_to_load))
        if checkpoint_path is not None:
            self.model.load_weights(checkpoint_path)
        if still_training:
            # Checkpoints contain wrapper weights but no portable optimizer
            # state because model.keras stores only the inner network.
            self.model.compile(optimizer='adam')

        if verbose:
            source = checkpoint_path or os.path.join(path, self.model_fname)
            io.print_level(1, 'Loaded Sobolev model from {}'.format(source))
        return self

    def check_files(self, path, datasets_paths, verbose=False):
        """Check the Sobolev artifact set needed for resume.

        Sobolev training intentionally has no x/y PCA files, so the FFNN
        implementation's file check cannot be reused unchanged.
        """
        output_folder = io.Folder(path)
        if not output_folder.exists:
            raise FileNotFoundError(
                'Output folder {} does not exist. Cannot resume!'.format(path))

        required = [
            io.YamlFile().default_name,
            self.x_scaler_fname,
            self.y_scaler_fname,
            self.growth_scaler_fname,
            self.model_fname,
            self.log_fname,
            self.data_fname,
        ]
        files = output_folder.list_files()
        for fname in required:
            full_path = output_folder.join(fname)
            if full_path not in files:
                raise FileNotFoundError(
                    'Required file {} does not exist. Cannot resume!'.format(
                        full_path))
        if verbose:
            io.info('All Sobolev resume files exist in {}'.format(path))
        return
