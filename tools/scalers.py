import joblib
import numpy as np
import sklearn.preprocessing as skl_pre
import tools.printing_scripts as scp


class Scaler(object):

    def __init__(self, name):
        self.name = name
        return

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

    def fit(self, x):
        """
        Placeholder for fit method.
        It should fit the sklearn (self.skl_scaler)
        """
        return

    def transform(self, x):
        """
        Placeholder for transform. It should return
        an array with the same shape as x, but scaled.
        """
        return None

    def inverse_transform(self, x_scaled):
        """
        Placeholder for inverse_transform. It should return
        an array with the same shape as x_scaled, but
        inverse scaled.
        """
        return None

    def save(self, path, verbose=False):
        joblib.dump(self, path.path)
        if verbose:
            scp.info('Saved scaler at: {}'.format(path.path))
        return

    @staticmethod
    def load(path, verbose=False):
        self = joblib.load(path.path)
        if verbose:
            scp.info('Loading scaler from: {}'.format(path.path))
        return self


class NoneScaler(Scaler):

    def __init__(self):
        Scaler.__init__(self)
        self.skl_scaler = None
        return

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class StandardScaler(Scaler):

    def __init__(self):
        Scaler.__init__(self)
        self.skl_scaler = skl_pre.StandardScaler()
        return

    def fit(self, x):
        self.skl_scaler.fit(x)
        return

    def transform(self, x):
        x_scaled = self.base_scaler.transform(x)
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.base_scaler.inverse_transform(x_scaled)
        return x


class MinMaxScaler(Scaler):

    def __init__(self):
        Scaler.__init__(self)
        self.skl_scaler = skl_pre.MinMaxScaler()
        return

    def fit(self, x):
        self.skl_scaler.fit(x)
        return

    def transform(self, x):
        x_scaled = self.base_scaler.transform(x)
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.base_scaler.inverse_transform(x_scaled)
        return x


class MinMaxScalerPlus1(Scaler):

    def __init__(self):
        Scaler.__init__(self)
        self.skl_scaler = skl_pre.MinMaxScaler()
        return

    def fit(self, x):
        self.skl_scaler.fit(x)
        return

    def transform(self, x):
        x_scaled = self.base_scaler.transform(x) + 1.
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.base_scaler.inverse_transform(x_scaled - 1.)
        return x


class ExpMinMaxScaler(Scaler):

    def __init__(self):
        Scaler.__init__(self)
        self.skl_scaler = skl_pre.MinMaxScaler()
        return

    def fit(self, x):
        self.skl_scaler.fit(x)
        return

    def transform(self, x):
        x_scaled = np.exp(x)
        x_scaled = self.base_scaler.transform(x_scaled)
        return x_scaled

    def inverse_transform(self, x_scaled):
        x = self.base_scaler.inverse_transform(x_scaled)
        x = np.log(x)
        return x
