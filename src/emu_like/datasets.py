"""
.. module:: datasets

:Synopsis: Deal with datasets.
:Author: Emilio Bellini

"""

import multiprocessing
import numpy as np
import os
import re
import sklearn.model_selection as skl_ms
import tqdm
from . import io as io
from . import defaults as de
from . import scalers as sc
from . import pca
from .params import Params
from .x_samplers import XSampler
from .y_models import YModel


class Dataset(object):
    """
    This class is primarly meant to store a dataset that can be used to
    train/test a ML algorithm. Here it is assumed that both x and y are
    2D arrays with n_samples rows and n_x, n_y columns.
    To load a dataset there are two ways:
    - create a Dataset instance and manually define all the required
      attributes;
    - create a DataCollection instance and load the dataset. Use the
      DataCollection.get_one_y_dataset(name) method to extract the
      desired Dataset.

    Available methods:
    - join: if they are compatible, join two datasets and return a
      single one;
    - train_test_split: split a dataset into train and test samples;
    - rescale: rescale dataset (both x and y).

    NOTE: For all the dataset generation, load, save operations, use the
    DataCollection class.
    """

    def __init__(
            self,
            name=None,
            x=None,
            y=None,
            x_ranges=None,
            y_ranges=None,
            n_x=None,
            n_y=None,
            n_samples=None,
            x_names=None,
            y_names=None,
            y_model=None,
            x_scaler=None,
            y_scaler=None,
            x_pca=None,
            y_pca=None,
            path=None
            ):
        """
        Placeholders.
        """
        # Name
        self.name = name
        # Data arrays
        self.x = x  # x
        self.y = y  # y
        # Data arrays - Train/test split
        self.x_train = None  # x_train
        self.y_train = None  # y_train
        self.x_test = None  # x_test
        self.y_test = None  # y_test

        # Ranges
        self.x_ranges = x_ranges  # x_ranges
        self.y_ranges = y_ranges  # y_ranges

        # Data shapes
        self.n_x = n_x  # Number of x variables
        self.n_y = n_y  # Number of y variables
        self.n_samples = n_samples  # Number of samples

        # Labels
        self.x_names = x_names  # List of names of x data
        self.y_names = y_names  # List of names of y data

        # y_model
        self.y_model = y_model

        # Path
        self.path = path

        # Placeholders
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.x_pca = x_pca
        self.y_pca = y_pca

        return

    @staticmethod
    def _load_array(path):
        """
        Load an array from file.
        Arguments:
        - path (str): path to the array
        - columns (default: None): slice object or list of
          column indices to be read.
        """
        array = np.genfromtxt(path)
        # Adjust array dimensions.
        # If it has one feature I still want 1x2_samples
        if array.ndim == 1:
            array = array[:, np.newaxis]
        return array

    @staticmethod
    def _try_to_load_names_array(
            path,
            n_names=None,
            comments='#',
            delimiter='\t'):
        """
        Try to load name of parameters from array.
        Names are extracted from the last comment row
        (starting with 'comments') of the file and should
        match the number of columns of the array.
        Arguments:
        - path (str): path to the array;
        - n_names (int, default: None): number of names to
          be expected. It is used to validate the extracted names.
          If they do not match, this method returns None;
        - comments (str, default: '#'). Starting string for comments;
        - delimiter (str, default: '\t'). Separator for column names.

        """
        is_comment = True
        names = None
        # Look for the last line that starts with 'comments'
        with open(path, 'r') as fn:
            while is_comment:
                line = fn.readline()
                if line.startswith(comments):
                    names = line
                else:
                    is_comment = False
        # Split names
        try:
            names = re.sub(comments, '', names)
        except TypeError:
            return None
        names = names.split(delimiter)
        names = [x.strip() for x in names]
        # Check names have the right dimensions
        if n_names:
            if n_names == len(names):
                return names
            else:
                return None
        return names

    @staticmethod
    def fill_missing_params(params):
        """
        Fill params object with missing entries
        Arguments:
        - params (Params): params object;
        """
        default_dict = {
            'output': {
                'path': None,
                'save_incrementally': False,
            },
            'x_sampler': {
                'name': None,
                'args': {},
            },
            'y_model': {
                'name': None,
                'args': {},
                'outputs': None,
            },
            'params': None,
        }
        for key1 in default_dict:
            if key1 not in params.content:
                params.content[key1] = default_dict[key1]
            if isinstance(default_dict[key1], dict):
                for key2 in default_dict[key1]:
                    if key2 not in params.content[key1]:
                        params.content[key1][key2] = default_dict[key1][key2]
        return params

    def slice(self, columns_x, columns_y, verbose=False):
        """
        Given a Dataset select the columns wanted, both for
        "x" and "y". It adjusts also the other attributes.
        Arguments:
        - columns_x (list of indices or slice object). Default: if "x"
          and "y" data come from different files all columns. If "x" and
          "y" are in the same file, all columns except the last one;
        - columns_y (list of indices or slice object for each y file).
          Default: if "x" and "y" data come from different files all
          columns. If "x" and "y" are in the same file, last column;
        - verbose (bool, default: False): verbosity.
        """

        if verbose:
            io.print_level(
                1, 'Slicing x data with columns: {}.'.format(columns_x))
            io.print_level(
                1, 'Slicing y data with columns: {}.'.format(columns_y))

        def slice_list(lst, slicing):
            if isinstance(slicing, list):
                return [lst[i] for i in slicing]
            elif isinstance(slicing, slice):
                return lst[slicing]
            else:
                raise Exception('Check your slicing, it can be either a list '
                                'or a slice object!')

        # Change default columns
        if columns_x is None:
            columns_x = slice(None)
        if columns_y is None:
            columns_y = slice(None)

        # Data
        self.x = self.x[:, columns_x]
        self.y = self.y[:, columns_y]

        # Names
        self.x_names = slice_list(self.x_names, columns_x)
        self.y_names = slice_list(self.y_names, columns_y)

        # Ranges
        self.x_ranges = slice_list(self.x_ranges, columns_x)
        self.y_ranges = slice_list(self.y_ranges, columns_y)

        # Adjust shapes
        self.n_samples, self.n_x = self.x.shape
        _, self.n_y = self.y.shape

        return self

    def remove_non_finite(self, store_non_finites=False, verbose=False):
        """
        Remove from the dataset non finite samples (inf and nan).
        Arguments:
        - store_non_finites (bool, default: False): store x's that
          give non finite y in non_finites_x.
        - verbose (bool, default: False): verbosity.
        """

        if verbose:
            io.print_level(1, 'Removing non finite values from x and y.')

        # Finite indices
        only_finites = np.all(np.isfinite(self.y), axis=1)

        # Sore non finite elements
        if store_non_finites:
            only_non_finites = np.array([not elem for elem in only_finites])
            self.non_finites_x = self.x[only_non_finites]

        self.x = self.x[only_finites]
        self.y = self.y[only_finites]

        # Adjust n_samples
        self.n_samples = self.x.shape[0]

        # Adjust ranges
        self.x_ranges = np.array(list(zip(
            np.min(self.x, axis=0),
            np.max(self.x, axis=0))))
        self.y_ranges = np.array(list(zip(
            np.min(self.y, axis=0),
            np.max(self.y, axis=0))))

        return self

    def load(
            self,
            path,
            name=None,
            columns_x=None,
            columns_y=None,
            verbose=False):
        """
        Load an existing dataset.
        This method assumes that path points to a dataset generated by
        this code (see NOTE below).
        Arguments:
        - path (str): path pointing to the folder containing the dataset;
        - columns_x (list of indices or slice object). Default: if "x"
          and "y" data come from different files all columns. If "x" and
          "y" are in the same file, all columns except the last one;
        - columns_y (list of indices or slice object for each y file).
          Default: if "x" and "y" data come from different files all
          columns. If "x" and "y" are in the same file, last column;
        - verbose (bool, default: False): verbosity.

        NOTE: when generated by this code, the dataset files have specific
        names (see discussion at the top of this Class). It assumes there
        is a settings file, and x and y datasets. If this is not the case
        use the Dataset.load_external() method, which will allow to perform all
        training/testing operations.
        """

        if verbose:
            io.info('Loading dataset.')

        if columns_x is None:
            columns_x = slice(None)
        if columns_y is None:
            columns_y = slice(None)

        # Store y name
        self.name = name

        # Load settings
        self.settings = Params().load(os.path.join(
            path, de.file_names['params']['name']))
        # Fill missing entries
        self.settings = Dataset.fill_missing_params(self.settings)

        # Main path
        self.path = path

        # Init x sampler
        x_sampler = XSampler.choose_one(
            self.settings['x_sampler']['name'],
            self.settings['params'],
            **self.settings['x_sampler']['args'],
            verbose=False)

        # Get x file name
        self.x_fname = x_sampler.get_x_fname()

        # Load x data.
        x_sampler.x = Dataset._load_array(
            os.path.join(self.path, self.x_fname))
        self.x = x_sampler.x

        # Get remaining x attributes
        self.x_ranges = x_sampler.get_x_ranges()
        self.n_x = x_sampler.get_n_x()
        self.n_samples = x_sampler.get_n_samples()
        self.x_names = x_sampler.get_x_names()
        self.x_header = x_sampler.get_x_header()

        # Init y_model
        y_model = YModel.choose_one(
            self.settings['y_model']['name'],
            self.settings['params'],
            {name: self.settings['y_model']['outputs'][name]},
            self.n_samples,
            **self.settings['y_model']['args'],
            verbose=False)
        
        # Load y_model
        y_model.load(
            de.file_names['spectra_factor']['name'],
            root=self.path,
            verbose=False,
        )

        # Get y file name
        y_fname = de.file_names['y_data']['name'].format(self.name)

        # Load y data.
        # 1) load y.
        y = Dataset._load_array(os.path.join(self.path, y_fname))
        # 2) Infer dimensions
        n_y = y.shape[1]
        self.counter_samples = y.shape[0]
        # 3) Try to infer the names
        y_names = Dataset._try_to_load_names_array(os.path.join(self.path, y_fname), n_names=n_y)
        # 4) Initialize list of zeros arrays with full n_samples.
        y_model.y = [np.zeros((self.n_samples, n_y))]
        # 5) Assign values.
        y_model.y[0][:self.counter_samples] = y
        # 6) Synchronize with self.y.
        self.y = y_model.y[0]

        # Get remaining y attributes
        self.n_y = y_model.get_n_y()[0]
        self.y_ranges = y_model.get_y_ranges()[0]
        if y_names is None:
            self.y_names = y_model.get_y_names()[0]
        else:
            y_model.y_names = self.y_names = y_names
        self.y_headers = y_model.get_y_headers()[0]

        # Slice data
        self.slice(columns_x, columns_y, verbose=verbose)

        # Propagate x_sampler and y_model
        self.x_sampler = x_sampler
        self.y_model = y_model

        # Print info
        if verbose:
            io.print_level(1, 'Loaded dataset from: {}'.format(self.path))

        return self

    def load_external(
            self,
            path,
            path_y=None,
            columns_x=None,
            columns_y=None,
            verbose=False):
        """
        Load an existing dataset.
        This method if more flexible than the one defined in
        Dataset.load(), because it allows to load non-standard
        datasets, but it loads the minimum amount of information
        needed to train an ML algorithm.
        It accepts only files in two formats:
        - a single file containing both x and y data (use "path");
        - two files, one for x data ("path") and one for "y" (path_y).
        Arguments:
        - path (str): path pointing to the file containing the "x" data.
          If it contains both "x" and "y", "path_y" should be None;
        - path_y (str, default: None): in case "x" and "y" data are stored in
          different files, use this variable to specify the file containing
          the "y" data;
        - columns_x (list of indices or slice object). Default: if "x"
          and "y" data come from different files all columns. If "x" and
          "y" are in the same file, all columns except the last one;
        - columns_y (list of indices or slice object for each y file).
          Default: if "x" and "y" data come from different files all
          columns. If "x" and "y" are in the same file, last column;
        - verbose (bool, default: False): verbosity.
        """

        if verbose:
            io.info('Loading dataset.')

        # Define paths. Cases:
        # 1) One file for both x and y
        if os.path.isfile(path) and path_y is None:
            path_x = path
            path_y = path
            self.path = path
            # Change default columns
            if columns_x is None:
                columns_x = slice(None, -1)
            if columns_y is None:
                columns_y = slice(-1, None)
        # 2) One file for x and one for y
        elif os.path.isfile(path) and os.path.isfile(path_y):
            path_x = path
            path_y = path_y
            self.path = [path, path_y]
            # Change default columns
            if columns_x is None:
                columns_x = slice(None)
            if columns_y is None:
                columns_y = slice(None)
        else:
            raise Exception('Something is wrong with your paths. '
                            'Dataset could not be loaded!')

        # Load data
        self.x  = self._load_array(path_x)
        self.y = self._load_array(path_y)

        # Get shapes
        self.n_samples, self.n_x = self.x.shape
        _, self.n_y = self.y.shape

        # Try to infer the names
        self.x_names = Dataset._try_to_load_names_array(path_x, n_names=self.n_y)
        self.y_names = Dataset._try_to_load_names_array(path_y, n_names=self.n_y)

        # Get ranges
        self.x_ranges = np.array(list(zip(np.min(self.x, axis=0),
                                          np.max(self.x, axis=0))))
        self.y_ranges = np.array(list(zip(np.min(self.y, axis=0),
                                          np.max(self.y, axis=0))))
        # Slice data
        self.slice(columns_x, columns_y, verbose=verbose)

        # Print info
        if verbose:
            io.print_level(1, 'x from: {}'.format(path_x))
            io.print_level(1, 'y from: {}'.format(path_y))
            io.print_level(1, 'n_samples: {}'.format(self.n_samples))
            io.print_level(1, 'n_x: {}'.format(self.n_x))
            io.print_level(1, 'n_y: {}'.format(self.n_y))

        return self

    @staticmethod
    def join(datasets, verbose=False):
        """
        Join a list of datasets into a unique one.
        This defines the minimum number of attributes
        required to use a dataset for tranining, i.e.
        x, y, n_x, n_y, n_samples, x_names and y_names.
        Before joining them it checks that n_x and n_y are
        the same for each dataset.
        Arguments:
        - datasets (list of Dataset): list of Dataset classes (already loaded);
        - verbose (bool, default: False): verbosity.
        """

        if verbose:
            io.info('Joining datasets')
            for dataset in datasets:
                io.print_level(1, '{}'.format(dataset.path))

        data = Dataset()

        # n_x
        if all(s.n_x == datasets[0].n_x for s in datasets):
            data.n_x = datasets[0].n_x
        else:
            raise ValueError('Datasets can not be joined as they have '
                             'different number of x variables')
        
        # n_y
        if all(s.n_y == datasets[0].n_y for s in datasets):
            data.n_y = datasets[0].n_y
        else:
            raise ValueError('Datasets can not be joined as they have '
                             'different number of x variables')

        # x array
        total = tuple([s.x for s in datasets])
        data.x = np.vstack(total)
        data.x_ranges = np.stack((
            np.min(data.x, axis=0),
            np.max(data.x, axis=0))).T

        # y array
        total = tuple([s.y for s in datasets])
        data.y = np.vstack(total)
        data.y_ranges = np.stack((
            np.min(data.y, axis=0),
            np.max(data.y, axis=0))).T

        # x and y names
        data.x_names = datasets[0].x_names
        data.y_names = datasets[0].y_names

        # n_samples
        data.n_samples = sum([s.n_samples for s in datasets])

        # y_model.
        # NOTE: we are assuming that all datsets are sharing the
        # same y_model, which is taken from the first one.
        data.y_model = datasets[0].y_model

        # Adjust params
        for var in datasets[0].y_model.params:
            mins = [dat.y_model.params[var]['prior']['min'] for dat in datasets]
            maxs = [dat.y_model.params[var]['prior']['max'] for dat in datasets]
            data.y_model.params[var]['prior']['min'] = min(mins)
            data.y_model.params[var]['prior']['max'] = max(maxs)

        return data

    def train_test_split(self, frac_train, seed, verbose=False):
        """
        Split a dataset into test and train samples.
        The split is stored into the x_train, x_test,
        y_train and y_test attributes.
        Arguments:
        - frac_train (float): fraction of training samples (between 0 and 1);
        - seed (int): seed to randomly split train and test;
        - verbose (bool, default: False): verbosity.

        NOTE: this method assumes that both settings and the fulle x array
        are already saved into the folder. The x array is then used to
        calculate the missing row of the y array.
        """

        if verbose:
            io.info('Splitting dataset in training and testing samples.')
            io.print_level(1, 'Fractional number of training samples: {}'
                           ''.format(frac_train))
            io.print_level(1, 'Random seed for train/test split: '
                           '{}'.format(seed))
        split = skl_ms.train_test_split(self.x, self.y,
                                        train_size=frac_train,
                                        random_state=seed)
        self.x_train, self.x_test, self.y_train, self.y_test = split
        return

    def rescale(self, rescale_x, rescale_y, verbose=False):
        """
        Rescale x and y of a dataset. The available scalers are
        written in src/emu_like/scalers.py
        Arguments:
        - rescale_x (str): scaler for x;
        - rescale_y (str): scaler for y;
        - verbose (bool, default: False): verbosity.

        NOTE: this method assumes that we already splitted
        train and test samples.
        """

        if verbose:
            io.info('Rescaling x and y.')
            io.print_level(1, 'x with: {}'.format(rescale_x))
            io.print_level(1, 'y with: {}'.format(rescale_y))
        # Rescale x
        self.x_scaler = sc.Scaler.choose_one(rescale_x)
        self.x_scaler.fit(self.x_train)
        self.x_train = self.x_scaler.transform(self.x_train)
        self.x_test = self.x_scaler.transform(self.x_test)
        # Rescale y
        self.y_scaler = sc.Scaler.choose_one(rescale_y)
        self.y_scaler.fit(self.y_train)
        self.y_train = self.y_scaler.transform(self.y_train)
        self.y_test = self.y_scaler.transform(self.y_test)
        if verbose:
            io.print_level(1, 'Rescaled bounds:')
            mins = np.min(self.x_train, axis=0)
            maxs = np.max(self.x_train, axis=0)
            for nx, min in enumerate(mins):
                io.print_level(
                    2, 'x_train_{} = [{}, {}]'.format(nx, min, maxs[nx]))
            mins = np.min(self.x_test, axis=0)
            maxs = np.max(self.x_test, axis=0)
            for nx, min in enumerate(mins):
                io.print_level(
                    2, 'x_test_{} = [{}, {}]'.format(nx, min, maxs[nx]))
            mins = np.min(self.y_train, axis=0)
            maxs = np.max(self.y_train, axis=0)
            for nx, min in enumerate(mins):
                io.print_level(
                    2, 'y_train_{} = [{}, {}]'.format(nx, min, maxs[nx]))
            mins = np.min(self.y_test, axis=0)
            maxs = np.max(self.y_test, axis=0)
            for nx, min in enumerate(mins):
                io.print_level(
                    2, 'y_test_{} = [{}, {}]'.format(nx, min, maxs[nx]))
        return

    def apply_pca(self, num_x_pca=None, num_y_pca=None, verbose=False):
        """
        Apply PCA to x and/or y of a dataset.
        Arguments:
        - num_pca_x (int): number of modes to be retained
          for x (if 0 or negative PCA is not applied);
        - num_pca_y (int): number of modes to be retained
          for y (if 0 or negative PCA is not applied);
        - verbose (bool, default: False): verbosity.

        NOTE: this method assumes that we already splitted
        train and test samples.
        """

        if num_x_pca == 'None':
            num_x_pca = None
        if num_y_pca == 'None':
            num_y_pca = None

        if verbose:
            if num_x_pca is not None:
                io.info('Applying PCA on x. Number of modes retained {}.'.format(num_x_pca))
            if num_y_pca is not None:
                io.info('Applying PCA on y. Number of modes retained {}.'.format(num_y_pca))

        # PCA x
        self.x_pca = pca.PCA(n_components=num_x_pca)
        self.x_pca.fit(self.x_train)
        self.x_train = self.x_pca.transform(self.x_train)
        self.x_test = self.x_pca.transform(self.x_test)
        # PCA y
        self.y_pca = pca.PCA(n_components=num_y_pca)
        self.y_pca.fit(self.y_train)
        self.y_train = self.y_pca.transform(self.y_train)
        self.y_test = self.y_pca.transform(self.y_test)

        return


class DataCollection(object):
    """
    This class that is primarly meant to deal with the generation
    of datasets. All the "y" attributes are stored in lists.
    Therefore, it is possible to manage multiple "y" outputs from
    the same YModel (e.g., see YModel.ClassSpectra).
    In the simplest cases, where there is just one "y", everything
    is stored in single element lists.
    If you want to use a dataset for training/testing a ML algorithm
    use the Dataset class. If you have a DataCollection, it is
    possible to get a Dataset in three ways:
    - create a Dataset instance and manually define all the required
      attributes;
    - if there is just one "y", it is possible to use the
      DataCollection.get_one_y_dataset() method to get the single
      dataset;
    - if there are multiple "y", select the data you want to
      extract using the DataCollection.get_one_y_dataset(name) method.

    Available methods:
    - load: load a dataset from a path;
    - save: save a dataset into a folder;
    - sample: generate a dataset;
    - resume: after loading a dataset, compute the remaining samples;

    NOTE: the datasets generated by this code contain one file
    for the x points, one (or more) for y, and one with all the settings.
    The x and y files have the name of the variables on their header.
    For flexibility, i.e. when they were created from other codes,
    the datasets can have two other formats:
    - they can be loaded from two separate files (one for x and one
      for y) without a settings file.
    - they can be loaded from a single file containing both x and y.
      In this case the default behaviour is that the last column is y,
      and the remaining are all x's.
    In all these cases, it is possible to select which columns to load
    for x and for y.

    NOTE: the name of the files of the generated datasets can be found
    in 'src/emu_like/defaults.py'.
    """

    def __init__(self):
        """
        Placeholders.
        """
        # Data arrays
        self.x = None  # x
        self.y = []  # y per file

        # Ranges
        self.x_ranges = None  # x_ranges
        self.y_ranges = []  # y_ranges per file

        # Data shapes
        self.n_x = None  # Number of x variables
        self.n_y = []  # Number of y variables per file
        self.n_samples = None  # Number of samples

        # Labels
        self.x_names = None  # List of names of x data
        self.y_names = []  # List of names of y data per file
        self.x_header = None  # Header for x file
        self.y_headers = []  # Headers for y files

        # Paths
        self.path = None  # Path of the dataset
        self.x_fname = None  # File name of x data
        self.y_fnames = []  # File names of y data

        # Container for all the settings
        self.settings = None

        # Container for the YModel
        self.y_model = None

        # Useful to keep track of how many samples have been computed
        self.counter_samples = 0

        return

    def _save_x(
            self,
            fname=None,
            root=None,
            x_array=None,
            header=None,
            verbose=False):
        """
        Quick way to save x array in path.
        Arguments:
        - fname (str, default: None): file name for x_array;
        - root (str, default: None): folder for x_array;
        - x_array (array, default: None): 2D x array;
        - header (str, default: None): x_array header;
        - verbose (bool, default: False): verbosity.
        """
        # Arguments or defaults
        if fname is None:
            fname = de.file_names['x_data']['name']
        if root is None:
            root = self.path
        if x_array is None:
            x_save = self.x
        else:
            x_save = x_array
        if header is None:
            head = self.x_header
        else:
            head = header

        if verbose:
            io.print_level(1, 'Saved x array at: {}'.format(fname))
        io.Folder(os.path.dirname(os.path.join(root, fname))).create()
        np.savetxt(os.path.join(root, fname), x_save, header=head)
        return

    def _save_y(
            self,
            fnames=None,
            roots=None,
            y_arrays=None,
            headers=None,
            verbose=False):
        """
        Quick way to save y arrays.
        Arguments:
        - fnames (list of str, default: None): file names for y_array;
        - roots (list of str, default: None): folders for y_array;
        - y_arrays (list of arrays, default: None): list of 2D y array;
        - headers (str, default: None): y_array headers;
        - verbose (bool, default: False): verbosity.
        """
        # Arguments or defaults
        if fnames is None:
            fnames = self.y_fnames
        if roots is None:
            roots = [self.path] * len(fnames)
        if y_arrays is None and self.y == []:
            y_save = [[]] * len(fnames)
        elif y_arrays is None:
            y_save = self.y
        else:
            y_save = y_arrays
        if headers is None:
            heads = self.y_headers
        else:
            heads = headers

        for nf in range(len(fnames)):
            io.Folder(os.path.dirname(os.path.join(roots[nf], fnames[nf]))).create()
            np.savetxt(os.path.join(roots[nf], fnames[nf]),
                       y_save[nf], header=heads[nf])
            if verbose:
                io.print_level(1, 'Saved y array at: {}'.format(fnames[nf]))
        return

    def _append_y(
            self,
            y_vals,
            fnames=None,
            roots=None):
        """
        Quick way to append rows to the y array in path.
        Arguments:
        - y_val (array): one line 2D y array;
        - fname (str, default: None): file name for y_array;
        - root (str, default: None): folder for y_array;
        """
        # Arguments or defaults
        if fnames is None:
            fnames = self.y_fnames
        if roots is None:
            roots = [self.path] * len(fnames)

        for nf in range(len(fnames)):
            with open(os.path.join(roots[nf], fnames[nf]), 'a') as fn:
                np.savetxt(fn, y_vals[nf])
        return

    def _get_dims_from_file(self, fname):
        '''
        Fast way to get the dimensions of an array
        saved into a file without loading it.
        '''
        def blocks(files, size=65536):
            while True:
                b = files.read(size)
                if not b: break
                yield b

        # Infer all rows (including header)
        with open(fname, 'r') as fn:
            rows = sum(bl.count('\n') for bl in blocks(fn))
        
        # Remove header rows and get columns
        with open(fname, 'r') as fn:
            for line in fn:
                if line.startswith('#'):
                    rows -= 1
                else:
                    cols = len(line.split(' '))
                    break

        return rows, cols

    def _save_settings(
            self,
            fname=None,
            root=None,
            settings=None,
            header=None,
            verbose=False):
        """
        Quick way to save settings dictionary in path.
        Arguments:
        - fname (str, default: None): file name for settings;
        - root (str, default: None): folder for settings;
        - settings (dict, default: None): settings dictionary;
        - header (str, default: None): settings header;
        - verbose (bool, default: False): verbosity.
        """
        # Arguments or defaults
        if fname is None:
            fname = de.file_names['params']['name']
        if root is None:
            root = self.path
        if settings is None:
            setts = self.settings
        else:
            setts = settings
        if header is None:
            head = de.file_names['params']['header']
        else:
            head = header

        params = Params(setts)
        params.save(fname, root=root, header=head, verbose=verbose)
        return

    def get_one_y_dataset(self, name=None):
        """
        Extract one datase from DataCollection.
        This dataset can be specified by its name, or it defaults to
        the only dataset if there is just one. If there are multiple
        datasets and no name is specified it raises an Exception.
        Arguments:
        - name (str, default:None): if specified it gets the y dataset
          with that name.
        """
        # Get correct index
        if name is not None:
            idx = self.y_fnames.index(
                de.file_names['y_data']['name'].format(name))
        elif len(self.y_fnames) == 1:
            idx = 0
        else:
            raise Exception(
                'It is not possible to extract a single dataset if no name '
                'is specified and there are multiple datasets!')

        dataset = Dataset(
            name=name,
            x=self.x,
            y=self.y[idx],
            x_ranges=self.x_ranges,
            y_ranges=self.y_ranges[idx],
            n_x=self.n_x,
            n_y=self.n_y[idx],
            n_samples=self.n_samples,
            x_names=self.x_names,
            y_names=self.y_names[idx],
            y_model=self.y_model[idx],
            path=self.path,
        )

        return dataset

    def save(
            self,
            fname_setts=None,
            fname_x=None,
            fnames_y=None,
            fname_y_model=None,
            root_setts=None,
            root_x=None,
            roots_y=None,
            root_y_model=None,
            settings=None,
            x_array=None,
            y_arrays=None,
            header_setts=None,
            header_x=None,
            headers_y=None,
            verbose=False
            ):
        """
        Save dataset to path.
        Arguments:
        - fname_setts, fname_x, fname_y (str, default: None): file names;
        - root_setts, root_x, root_y (str, default: None): folders;
        - settings (dict, default: None): settings dictionary;
        - x_array, y_array (array, default: None): 2D x, y array;
        - header_setts, header_x, header_y (str, default: None): headers;
        - verbose (bool, default: False): verbosity.
        """

        # Save settings
        self._save_settings(
            fname=fname_setts,
            root=root_setts,
            settings=settings,
            header=header_setts,
            verbose=verbose)

        # Save x
        self._save_x(
            fname=fname_x,
            root=root_x,
            x_array=x_array,
            header=header_x,
            verbose=verbose)

        # Save y
        self._save_y(
            fnames=fnames_y,
            roots=roots_y,
            y_arrays=y_arrays,
            headers=headers_y,
            verbose=verbose)
        
        # Save y_model
        if root_y_model is None:
            root_y_model = self.path
        if fname_y_model is None:
            fname_y_model = de.file_names['spectra_factor']['name']
        self.y_model.save(
            fname=fname_y_model,
            root=root_y_model,
            verbose=verbose)

        return

    def load(
            self,
            path,
            minimal=False,
            verbose=False):
        """
        Load an existing data collection.
        This method assumes that path points to a dataset generated by
        this code (see NOTE below).
        Arguments:
        - path (str): path pointing to the folder containing the dataset,
          or to a file containing both the x and y data or to a file
          containing only the x data. In this last case, 'path_y' should
          be specified. See discussion at the top of this class;
        - minimal (bool, default: False): do not load the array of already
          computed data to save time and memory;
        - verbose (bool, default: False): verbosity.

        NOTE: when generated by this code, the dataset files have specific
        names (see discussion at the top of this Class). It assumes there
        is a settings file, and x and y datasets. If this is not the case
        use the Dataset.load_external() method, which will allow to perform
        all training/testing operations.
        """

        if verbose:
            io.info('Loading data collection.')

        # Load settings
        self.settings = Params().load(os.path.join(
            path, de.file_names['params']['name']))
        # Fill missing entries
        self.settings = Dataset.fill_missing_params(self.settings)

        # Main path
        self.path = path

        # Init x sampler
        x_sampler = XSampler.choose_one(
            self.settings['x_sampler']['name'],
            self.settings['params'],
            **self.settings['x_sampler']['args'],
            verbose=False)

        # Get x file name
        self.x_fname = x_sampler.get_x_fname()

        # Load x data.
        x_sampler.x = Dataset._load_array(
            os.path.join(self.path, self.x_fname))
        self.x = x_sampler.x

        # Get remaining x attributes
        self.x_ranges = x_sampler.get_x_ranges()
        self.n_x = x_sampler.get_n_x()
        self.n_samples = x_sampler.get_n_samples()
        self.x_names = x_sampler.get_x_names()
        self.x_header = x_sampler.get_x_header()

        # Init y_model
        y_model = YModel.choose_one(
            self.settings['y_model']['name'],
            self.settings['params'],
            self.settings['y_model']['outputs'],
            self.n_samples,
            **self.settings['y_model']['args'],
            verbose=False)
        
        # Load y_model
        y_model.load(
            de.file_names['spectra_factor']['name'],
            root=self.path,
            verbose=False,
        )

        # Get y file names
        self.y_fnames = y_model.get_y_fnames()

        # Load y data.
        if not minimal:
            # 1) load y for each file.
            y = [Dataset._load_array(
                os.path.join(self.path, self.y_fnames[nf]))
                for nf in range(len(self.y_fnames))]
            # 2) Infer dimensions
            n_rows = [y_one.shape[0] for y_one in y]
            n_y = [y_one.shape[1] for y_one in y]
        # Get array dimensions without loading it
        else:
            # 2) Infer dimensions
            n_rows, n_y = zip(*[self._get_dims_from_file(
                os.path.join(self.path, self.y_fnames[nf])) for nf in range(len(self.y_fnames))])

        # Check and init counter_samples
        if not all(row == n_rows[0] for row in n_rows):
            raise IOError('Not all the files have the same number of raws')
        self.counter_samples = n_rows[0]

        # 3) Try to infer the names
        y_names = [Dataset._try_to_load_names_array(
            os.path.join(self.path, self.y_fnames[nf]), n_names=n_y[nf]) for nf in range(len(self.y_fnames))]

        # 4) Initialize list of zeros arrays with full or remaining samples.
        y_model.y = [np.zeros((self.n_samples, n_y_one)) for n_y_one in n_y]

        # 5) Assign values.
        if not minimal:
            for ny_gen, y_gen in enumerate(y_model.y):
                y_gen[:self.counter_samples] = y[ny_gen]

        # 5) Synchronize with self.y.
        self.y = y_model.y

        # Get remaining y attributes
        self.n_y = y_model.get_n_y()
        self.y_ranges = y_model.get_y_ranges()
        if None in y_names:
            self.y_names = y_model.get_y_names()
        else:
            y_model.y_names = self.y_names = y_names
        self.y_headers = y_model.get_y_headers()

        # Propagate x_sampler and y_model
        self.x_sampler = x_sampler
        self.y_model = y_model

        # Print info
        if verbose:
            io.print_level(1, 'Loaded dataset from: {}'.format(self.path))

        return self

    def sample(
            self,
            params,
            x_name,
            y_name,
            x_args=None,
            y_args=None,
            y_outputs=None,
            output=None,
            save_incrementally=False,
            verbose=False):
        """
        Generate a dataset.
        Arguments:
        - params (dict): dictionary containing the parameters to be passed
          to the y_model. See yaml files for details;
        - x_name (str): name of the x_sampler. Options defined in
          src/emu_like/x_samplers.py;
        - y_name (str): name of the y_model.
          Options defined in src/emu_like/y_models.py;
        - x_args (dict, default: None): dictionary with extra
          arguments needed by the x_sampler function;
        - y_args (dict, default: None): dictionary with extra
          arguments needed by the y_model;
        - y_outputs (dict, default: None): dictionary dealing
          with multiple y outputs for a single x (see class_spectra);
        - output (str, default: None): if save_incrementally the output
          path should be passed;
        - save_incrementally (bool, default: False): save output incrementally
          (not compatible with parallel computing, set num_processes=1);
        - verbose (bool, default: False): verbosity.
        """

        if verbose:
            io.info('Generating dataset.')

        # Create main folder
        self.path = output
        if save_incrementally:
            io.Folder(self.path).create(verbose=verbose)

        # Create settings dictionary
        self.settings = {
            'output': {
                'path': output,
                'save_incrementally': save_incrementally,
            },
            'x_sampler': {
                'name': x_name,
                'args': x_args,
            },
            'y_model': {
                'name': y_name,
                'args': y_args,
                'outputs': y_outputs,
            },
            'params': params,
        }
        # Save settings
        if save_incrementally:
            self._save_settings(verbose=verbose)

        # Init x sampler
        x_sampler = XSampler.choose_one(
            x_name,
            params,
            **x_args,
            verbose=verbose)

        # Get x data and attributes
        self.x = x_sampler.get_x()
        self.x_ranges = x_sampler.get_x_ranges()
        self.n_x = x_sampler.get_n_x()
        self.x_names = x_sampler.get_x_names()
        self.x_header = x_sampler.get_x_header()
        self.x_fname = x_sampler.get_x_fname()
        self.n_samples = x_sampler.get_n_samples()

        # Save x_array
        if save_incrementally:
            self._save_x(verbose=verbose)

        # Init y_model
        y_model = YModel.choose_one(
            y_name,
            params,
            y_outputs,
            self.n_samples,
            **y_args,
            verbose=verbose)

        # Save after init what has to be saved
        if save_incrementally:
            y_model.save(
                root=self.path,
                verbose=verbose)

        # Get y attributes
        self.n_y = y_model.get_n_y()
        self.y_names = y_model.get_y_names()
        self.y_headers = y_model.get_y_headers()
        self.y_fnames = y_model.get_y_fnames()

        if save_incrementally:
            self._save_y(verbose=verbose)
    
        # Init self.y
        y_model.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]
        self.y = y_model.y

        # Start iteration in series
        for nx, x in enumerate(tqdm.tqdm(self.x)):
            y_one = y_model.evaluate(x, nx)
            self.counter_samples += 1

            if any([np.isnan(yy).any() for yy in y_one]):
                io.warning(' Found nans with parameters {}'.format(x))

            # Save array
            if save_incrementally:
                self._append_y(y_one)


        # Get remaining attributes
        self.y_ranges = y_model.get_y_ranges()

        # Propagate x_sampler and y_model
        self.x_sampler = x_sampler
        self.y_model = y_model

        return

    def resume(self, path, load_minimal=False, save_incrementally=False, verbose=False):
        """
        Resume a dataset previously loaded (use load method
        before resuming). Many settings are already loaded.
        Arguments:
        - path (str): path pointing to the folder containing the dataset;
        - load_minimal (bool, default: False): do not load the array of
          already computed data to save time and memory;
        - save_incrementally (bool, default: False): save output incrementally;
        - verbose (bool, default: False): verbosity.

        NOTE: this method assumes that both settings and the full x array
        are already saved into the folder. The x array is then used to
        calculate the missing row of the y array.
        """

        # Load the dataset
        self.load(path, minimal=load_minimal, verbose=verbose)

        if verbose:
            io.info('Resuming dataset computation.')
            io.print_level(1, 'From: {}'.format(self.path))
            io.print_level(
                1, 'Remaining samples: {}'
                ''.format(self.n_samples-self.counter_samples))
        if self.counter_samples == self.n_samples:
            if verbose:
                io.warning('Dataset complete, nothing to resume!')
            return

        start = self.counter_samples
        for ns, x in enumerate(tqdm.tqdm(self.x[start:])):
            y_one = self.y_model.evaluate(x, start + ns)
            self.counter_samples += 1
        
            # Save array
            if save_incrementally:
                self._append_y(y_one)

        # Get remaining attributes
        self.y_ranges = self.y_model.get_y_ranges()

        return

