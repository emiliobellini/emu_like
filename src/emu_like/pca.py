"""
.. module:: pca

:Synopsis: Apply PCA.
:Author: Emilio Bellini

"""

import joblib
import os
import sklearn.decomposition as skl_dec
from . import io as io


class PCA(object):
    """
    Apply PCA.
    It is used to apply the PCA to either x or y.
    This main class has five main methods:
    - load: load a PCA object from a file
    - save: save PCA object to a file.
    - fit: fit PCA
    - transform: transform data using PCA
    - inverse_transform: transform back data.
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = skl_dec.PCA(n_components=self.n_components)
        return

    @staticmethod
    def load(path, verbose=False):
        """
        Load PCA from path.
        Arguments:
        - path (str): file pointing to the PCA file;
        - verbose (bool, default: False): verbosity.
        """
        self = joblib.load(path)
        if verbose:
            io.info('Loading PCA from: {}'.format(path))
        return self

    def save(self, path, root=None, verbose=False):
        """
        Save PCA to path.
        Arguments:
        - path (str): file where to save the PCA;
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
        Fit PCA with array x.
        """
        self.pca.fit(x)
        return

    def transform(self, x):
        """
        Transform an array x into the corresponding
        x_pca, using results from the fit method.
        """
        x_pca = self.pca.transform(x)
        return x_pca

    def inverse_transform(self, x_pca):
        """
        Transform back an array x_pca into
        the corresponding x, using results from
        the fit method.
        """
        x = self.pca.inverse_transform(x_pca)
        return x
