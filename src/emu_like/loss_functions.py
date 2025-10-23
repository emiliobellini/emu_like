"""Loss functions used by emu_like models."""

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as keras_backend
from keras.saving import register_keras_serializable


def _normalized_singular_values(kwargs, *, minimum=None):
    """Return PCA singular values normalised to the leading mode.

    Args:
        kwargs (dict): Keyword arguments passed to the loss factory. The
            ``data`` entry must expose ``y_pca.pca.singular_values_``.
        minimum (float | None): Optional floor applied element-wise to the
            normalised singular values.

    Returns:
        np.ndarray: Normalised (and optionally clamped) singular values.
    """

    data = kwargs['data']
    singular_values = data.y_pca.pca.singular_values_
    normalized = singular_values / singular_values[0]
    if minimum is not None:
        normalized = np.maximum(normalized, minimum)
    return normalized


def mean_squared_error(**kwargs):
    """Return a standard mean squared error loss."""

    scale = np.ones_like(_normalized_singular_values(kwargs))

    @register_keras_serializable(package='emu_like', name='mean_squared_error')
    def loss(y_true, y_pred):
        return keras_backend.mean(
            keras_backend.square((y_pred - y_true) * scale),
            axis=-1,
        )

    return loss


def mean_squared_error_pca(**kwargs):
    """Return MSE weighted by PCA singular values with a lower bound."""

    scale = _normalized_singular_values(kwargs, minimum=0.2)

    @register_keras_serializable(
        package='emu_like',
        name='mean_squared_error_pca',
    )
    def loss(y_true, y_pred):
        return keras_backend.mean(
            scale * keras_backend.square(y_pred - y_true),
            axis=-1,
        )

    return loss


def huber_pca(**kwargs):
    """Return Huber loss applied to PCA-weighted residuals."""

    scale = _normalized_singular_values(kwargs)

    @register_keras_serializable(package='emu_like', name='huber_pca')
    def loss(y_true, y_pred):
        return keras.losses.huber(
            y_true * scale,
            y_pred * scale,
            delta=1.0,
        )

    return loss


def huber(**kwargs):
    """Return a standard Huber loss."""

    scale = np.ones_like(_normalized_singular_values(kwargs))

    @register_keras_serializable(package='emu_like', name='huber')
    def loss(y_true, y_pred):
        return keras.losses.huber(
            y_true * scale,
            y_pred * scale,
            delta=1.0,
        )

    return loss
