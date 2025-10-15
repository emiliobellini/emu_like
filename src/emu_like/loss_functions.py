"""
.. module:: loss_functions

:Synopsis: Module containing user defined loss functions.
:Author: Emilio Bellini

Loss function should take as input y_true and y_predicted
and a float as output.
"""

from tensorflow.python.keras import backend as keras_backend
from tensorflow import keras

try:
    from keras.saving import register_keras_serializable
except (ImportError, AttributeError):
    from tensorflow.keras.utils import register_keras_serializable


def mean_squared_error(**kwargs):
    import numpy as np
    fac = kwargs['data'].y_pca.pca.singular_values_/kwargs['data'].y_pca.pca.singular_values_[0]
    fac = fac*0. + 1.
    @register_keras_serializable(package='emu_like', name='mean_squared_error')
    def loss(y_true, y_pred):
        return keras_backend.mean(keras_backend.square((y_pred - y_true)*fac), axis=-1)
    return loss

def mean_squared_error_pca(**kwargs):
    fac = kwargs['data'].y_pca.pca.singular_values_/kwargs['data'].y_pca.pca.singular_values_[0]
    @register_keras_serializable(package='emu_like', name='mean_squared_error_pca')
    def loss(y_true, y_pred):
        return keras_backend.mean(keras_backend.square((y_pred - y_true)*fac), axis=-1)
    return loss

def mean_squared_error_pca_2(**kwargs):
    fac = kwargs['data'].y_pca.pca.singular_values_/kwargs['data'].y_pca.pca.singular_values_[0]
    @register_keras_serializable(package='emu_like', name='mean_squared_error_pca_2')
    def loss(y_true, y_pred):
        return keras.losses.mse(y_true*fac, y_pred*fac)
    return loss

def huber_pca(**kwargs):
    fac = kwargs['data'].y_pca.pca.singular_values_/kwargs['data'].y_pca.pca.singular_values_[0]
    @register_keras_serializable(package='emu_like', name='huber_pca')
    def loss(y_true, y_pred):
        return keras.losses.huber(y_true*fac, y_pred*fac, delta=1.0)
    return loss

def huber(**kwargs):
    fac = kwargs['data'].y_pca.pca.singular_values_/kwargs['data'].y_pca.pca.singular_values_[0]
    fac = fac*0. + 1.
    @register_keras_serializable(package='emu_like', name='huber')
    def loss(y_true, y_pred):
        return keras.losses.huber(y_true*fac, y_pred*fac, delta=1.0)
    return loss

# @keras.saving.register_keras_serializable()
# def max_absolute_error(y_true, y_pred):
#     diff = keras_backend.abs(y_true - y_pred)
#     return keras_backend.max(diff, axis=0)


# @keras.saving.register_keras_serializable()
# def mean_absolute_error(y_true, y_pred):
#     diff = keras_backend.abs(y_true - y_pred)
#     return keras_backend.mean(diff, axis=0)


# @keras.saving.register_keras_serializable()
# def max_relative_error(y_true, y_pred):
#     den = keras_backend.clip(
#         keras_backend.exp(y_true),
#         keras_backend.epsilon(),
#         None)
#     diff = keras_backend.abs((y_true - y_pred)/den)
#     return keras_backend.max(diff, axis=0)


# @keras.saving.register_keras_serializable()
# def mean_relative_error(y_true, y_pred):
#     den = keras_backend.clip(
#         keras_backend.exp(y_true),
#         keras_backend.epsilon(),
#         None)
#     diff = keras_backend.abs((y_true - y_pred)/den)
#     return keras_backend.mean(diff, axis=0)
