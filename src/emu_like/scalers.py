"""
.. module:: scalers

:Synopsis: List of possible scalers.
:Author: Emilio Bellini

"""

import joblib
import numpy as np
import os
import sklearn.preprocessing as skl_pre
from . import io as io


class Scaler(object):
    """
    Base Scaler class.
    It is use to rescale data (useful for training to
    have order 1 ranges). This main class has three
    main methods:
    - choose_one: redirects to the correct subclass
    - load: load a scaler from a file
    - save: save scaler to a file.

    Each one of the other scalers (see below), should
    inherit from this and define three other methods:
    - fit: fit scaler
    - transform: transform data using fitted scaler
    - inverse_transform: transform back data.
    """

    def __init__(self, name):
        self.name = name
        return

    def _replace_inf(self, x, factor=10.):
        """
        This is used to replace infinities with
        large numbers. In practice, given an array x,
        it takes the maximum value of abs(x) and
        multiplies it by 'factor'. The resulting
        number is going to replace all infinities
        (with the correct sign).
        """
        signs = np.sign(x)
        x_new = np.abs(x)
        x_new[np.isinf(x_new)] = np.nan
        inf = factor*np.nanmax(x_new, axis=0)[np.newaxis, :]
        nans = np.multiply(np.isnan(x_new), inf)
        x_new[np.isnan(x_new)] = 0.
        x_new = np.multiply(x_new + nans, signs)
        return x_new

    @staticmethod
    def choose_one(scaler_type):
        """
        Main function to get the correct Scaler.

        Arguments:
            - scaler_type (str): type of scaler.

        Return:
            - Scaler (object): get the correct
              scaler and initialize it.
        """
        if scaler_type == 'None':
            return NoneScaler(scaler_type)
        elif scaler_type == 'StandardScaler':
            return StandardScaler(scaler_type)
        elif scaler_type == 'MinMaxScaler':
            return MinMaxScaler(scaler_type)
        elif scaler_type == 'MinMaxScalerPlus1':
            return MinMaxScalerPlus1(scaler_type)
        elif scaler_type == 'ExpMinMaxScaler':
            return ExpMinMaxScaler(scaler_type)
        else:
            raise ValueError('Scaler not recognized!')

    @staticmethod
    def load(path, verbose=False):
        """
        Load a scaler from path.
        Arguments:
        - path (str): file pointing to the scaler;
        - verbose (bool, default: False): verbosity.
        """
        self = joblib.load(path)
        if verbose:
            io.info('Loading scaler from: {}'.format(path))
        return self

    def save(self, path, root=None, verbose=False):
        """
        Save a scaler to path.
        Arguments:
        - path (str): file where to save the scaler;
        - root (str, default: None): root where to save the file;
        - verbose (bool, default: False): verbosity.
        """
        # Join root
        if root:
            path = os.path.join(root, path)

        joblib.dump(self, path)
        if verbose:
            io.info('Saved scaler at: {}'.format(path))
        return

    def fit(self, x):
        """
        Fit scaler with array x.
        NOTE: If you want to implement a new scaler
        save here all the attributes needed by
        transform and inverse_transform.
        """
        return

    def transform(self, x):
        """
        Transform an array x into the corresponding
        x_scaled, using results from the fit method.
        NOTE: If you want to implement a new scaler
        return a new rescaled array.
        """
        return None

    def inverse_transform(self, x_scaled):
        """
        Transform back an array x_scaled into
        the corresponding x, using results from
        the fit method.
        NOTE: If you want to implement a new scaler
        return a new inverse rescaled array.
        """
        return None


class NoneScaler(Scaler):
    """
    Do not rescale.
    """

    def __init__(self, name):
        Scaler.__init__(self, name)
        self.skl_scaler = None
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        return x_scaled

    def inverse_transform(self, x):
        return x


class StandardScaler(Scaler):
    """
    Standardise features by removing the
    mean and scaling to unit variance.
    """

    def __init__(self, name):
        Scaler.__init__(self, name)
        self.skl_scaler = skl_pre.StandardScaler()
        return

    def fit(self, x, replace_infinity=True):
        if replace_infinity:
            x_to_fit = self._replace_inf(x)
        else:
            x_to_fit = x
        self.skl_scaler.fit(x_to_fit)
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.skl_scaler.inverse_transform(x_scaled)
        return x


class MinMaxScaler(Scaler):
    """
    Transform features by scaling each
    feature to the (0, 1) range.
    """

    def __init__(self, name):
        Scaler.__init__(self, name)
        self.skl_scaler = skl_pre.MinMaxScaler()
        return

    def fit(self, x, replace_infinity=True):
        if replace_infinity:
            x_to_fit = self._replace_inf(x)
        else:
            x_to_fit = x
        self.skl_scaler.fit(x_to_fit)
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.skl_scaler.inverse_transform(x_scaled)
        return x


class MinMaxScalerPlus1(Scaler):
    """
    Transform features by scaling each
    feature to the (1, 2) range.
    This can be useful to avoid zeros.
    """

    def __init__(self, name):
        Scaler.__init__(self, name)
        self.skl_scaler = skl_pre.MinMaxScaler()
        return

    def fit(self, x, replace_infinity=True):
        if replace_infinity:
            x_to_fit = self._replace_inf(x)
        else:
            x_to_fit = x
        self.skl_scaler.fit(x_to_fit)
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled) + 1.
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.skl_scaler.inverse_transform(x_scaled - 1.)
        return x


class ExpMinMaxScaler(Scaler):
    """
    Transform features by scaling each
    feature to the (0, 1) range and then
    takes the exponential of the result.
    """

    def __init__(self, name):
        Scaler.__init__(self, name)
        self.skl_scaler = skl_pre.MinMaxScaler()
        return

    def fit(self, x, replace_infinity=True):
        if replace_infinity:
            x_to_fit = self._replace_inf(x)
        else:
            x_to_fit = x
        self.skl_scaler.fit(x_to_fit)
        return

    def transform(self, x, replace_infinity=True):
        if replace_infinity:
            x_scaled = self._replace_inf(x)
        else:
            x_scaled = x
        x_scaled = self.skl_scaler.transform(x_scaled)
        x_scaled = np.exp(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = np.log(x_scaled)
        x = self.skl_scaler.inverse_transform(x)
        return x
