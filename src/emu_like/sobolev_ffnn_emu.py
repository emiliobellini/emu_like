"""Sobolev FFNN emulator."""

from tensorflow import keras

from . import io
from .datasets import SobolevDataset
from .ffnn_emu import FFNNEmu


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
        self.model = keras.Model(inputs=inputs, outputs=outputs,
                                 name='sobolev_ffnn')

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
