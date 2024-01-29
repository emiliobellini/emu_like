"""
.. module:: loss_functions

:Synopsis: Module containing user defined loss functions.
:Author: Emilio Bellini

Loss function should take as input y_true and y_predicted
and a float as output.
"""

from tensorflow.python.keras import backend as keras_backend
from tensorflow import keras


@keras.saving.register_keras_serializable()
def max_absolute_error(y_true, y_pred):
    diff = keras_backend.abs(y_true - y_pred)
    return keras_backend.max(diff, axis=0)


@keras.saving.register_keras_serializable()
def mean_absolute_error(y_true, y_pred):
    diff = keras_backend.abs(y_true - y_pred)
    return keras_backend.mean(diff, axis=0)


@keras.saving.register_keras_serializable()
def max_relative_error(y_true, y_pred):
    diff = keras_backend.abs(y_true/y_pred - 1.)
    return keras_backend.max(diff, axis=0)


@keras.saving.register_keras_serializable()
def mean_relative_error(y_true, y_pred):
    diff = keras_backend.abs((y_true - y_pred)/y_true)
    return keras_backend.mean(diff, axis=0)
