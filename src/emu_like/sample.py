import numpy as np
import os
import re
import tqdm
import sklearn.model_selection as skl_ms
from . import sampling_functions as fng  # noqa:F401
from . import defaults as de
from . import io as io
from . import plots as pl
from . import scalers as sc
from . import samplers as smp
from .params import Params


class Sample(object):
    """
    Base class to deal with samples.
    Main methods:
    - load: load a sample from a path;
    - generate: generate a sample from a dictionary of settings;
    - save: save a sample into a folder;
    - resume: after loading a function, if it is incomplete, it
      computes the remaining points;
    - join: if they are compatible, join two samples and return a single one.

    NOTE: the samples generated by this code contain one file
    for the x points, one for y, and one with all the settings.
    The x and y files have the name of the variables on their header.
    For flexibility, i.e. when they were created from other codes,
    the samples can have two other formats:
    - they can be loaded from two separate files (one for x and one
      for y) without a settings file.
    - they can be loaded from a single file containing both x and y.
      In this case the default behaviour is that the last column is y,
      and the remaining are all x's.
    In all these cases, it is possible to select which columns to load
    for x and for y.

    NOTE: the name of the files of the generated samples can be found
    in 'src/emu_like/defaults.py'.
    """

    def __init__(self):
        """
        Placeholders.
        """
        self.x = None  # Array with x data
        self.y = None  # Array with y data
        self.x_names = None  # List of names of x data
        self.y_names = None  # List of names of y data
        self.n_samples = None  # Number of samples
        self.n_x = None  # Number of x variables
        self.n_y = None  # Number of y variables
        self.settings = None  # Settings dictionary
        return

    def _load_array(self, path, columns):
        """
        Load an array from file.
        Arguments:
        - path (str): path to the array
        - columns: slice object or list of column indices to be read
        - verbose (bool, default: False): verbosity.
        """
        array = np.genfromtxt(path)
        names = self._try_to_load_names_array(path, n_names=array.shape[1])
        array = array[:, columns]
        names = names[columns]
        return array, names

    def _try_to_load_names_array(self, path, n_names=None,
                                     comments='#', delimiter='\t'):
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
        names = re.sub(comments, '', names)
        names = names.split(delimiter)
        names = np.array([x.strip() for x in names])
        # Check names have the right dimensions
        if n_names:
            if n_names == len(names):
                return names
            else:
                return None
        return names

    def load(self,
             path,
             path_y=None,
             columns_x=slice(None),
             columns_y=slice(None),
             remove_non_finite=False,
             verbose=False):
        """
        Load an existing sample.
        Arguments:
        - path (str): path pointing to the folder containing the sample,
          or to a file containing the x data. If y data are in the same
          this is sufficient, otherwise 'path_y' should be specified. See
          discussion at the top of this Class for possible input samples;
        - path_y (str, default: None): in case x and y data are stored
          in different files and different folders, use this variable
          to specify the file containing the y data;
        - columns_x (list of indices or slice object). Default: if x and y
          data come from different files all columns. If x and y are in
          the same file, all columns except the last one;
        - columns_y (list of indices or slice object). Default: if x and y
          data come from different files all columns. If x and y are in
          the same file, last column;
        - remove_non_finite (bool, default: False). Remove all rows where
          any of the y's is non finite (infinite or nan);
        - verbose (bool, default: False): verbosity.

        NOTE: when generated by this code, the sample files have specific
        names (see discussion at the top of this Class). One case where it
        is necessary to specify both 'path' and 'path_y' is when the input
        files do not have the defaults names.
        """

        if verbose:
            io.info('Loading sample.')

        # Load settings, if any
        path_settings = os.path.join(path, de.file_names['params']['name'])
        try:
            self.settings = Params().load(path_settings)
        except NotADirectoryError:
            io.warning('Unable to load parameter file!')

        # Assign paths to x and y. There are two cases:
        # 1) Two files for x and y
        if path and path_y:
            path_x = path
            path_y = path_y
        # 2) One directory with two files for x and y
        elif os.path.isdir(path):
            path_x = os.path.join(path, de.file_names['x_sample']['name'])
            path_y = os.path.join(path, de.file_names['y_sample']['name'])
        # 3) One file for x and y
        elif os.path.isfile(path):
            # Change default columns
            columns_x = slice(None, -1)
            columns_y = slice(-1, None)
            path_x = path
            path_y = path
        else:
            raise FileNotFoundError(
                'Something is wrong with your sample path. I could not '
                'identify the x and y paths')

        # Load data
        self.x, self.x_names = self._load_array(path_x, columns_x)
        self.y, self.y_names = self._load_array(path_y, columns_y)

        # Remove non finite if requested
        if remove_non_finite:
            if verbose:
                io.info('Removing non finite data from sample.')
            only_finites = np.any(np.isfinite(self.y), axis=1)
            self.x = self.x[only_finites]
            self.y = self.y[only_finites]
        
        # Get sample attributes
        self.n_samples = self.x.shape[0]
        self.n_x = self.x.shape[1]
        self.n_y = self.y.shape[1]

        # Print info
        if verbose:
            io.print_level(1, 'Parameters from: {}'.format(path_settings))
            io.print_level(1, 'x from: {}'.format(path_x))
            io.print_level(1, 'y from: {}'.format(path_y))
            io.print_level(1, 'n_samples: {}'.format(self.n_samples))
            io.print_level(1, 'n_x: {}'.format(self.n_x))
            io.print_level(1, 'n_y: {}'.format(self.n_y))

        return

    def resume(self):
        return

    def save(self, path, verbose=False):
        """
        Save sample to path.
        - path (str): output path;
        - verbose (bool, default: False): verbosity.
        """
        if verbose:
            io.print_level(1, 'Saving output at: {}'.format(path))

        # Save parameters
        params = Params(content=self.settings)
        params.save(os.path.join(path, de.file_names['params']['name']),
                    header=de.file_names['params']['header'],
                    verbose=False)
        
        # Save x
        np.savetxt(os.path.join(path, de.file_names['x_sample']['name']),
                   self.x,
                   header='\t'.join(self.x_names))

        # Save y
        np.savetxt(os.path.join(path, de.file_names['y_sample']['name']),
                   self.y,
                   header='\t'.join(self.y_names))
        return

    def generate(self, params, sampled_function, n_samples, spacing,
                 save_incrementally=False, output_path=None, verbose=False):
        """
        Generate a sample.
        Arguments:
        - params (dict): dictionary containing the parameters to be passed
          to the sampled_function. See simple_sample.yaml and
          planck_sample.yaml for details;
        - sampled_function (str): one of the functions defined in
          src/emu_like/sampling_functions.py;
        - n_samples (int): number of samples to compute;
        - spacing (str): spacing of the sample. Options are those defined in
          src/emu_like/samplers.py;
        - save_incrementally (bool, default: False): save output incrementally;
        - output_path (str, default: None): if save_incrementally the output
          path should be passed;
        - verbose (bool, default: False): verbosity.
        """

        if verbose:
            io.info('Generating sample.')
            io.print_level(1, 'Sampled function: {}'.format(sampled_function))
            io.print_level(1, 'Number of sampled: {}'.format(n_samples))
            io.print_level(1, 'Spacing: {}'.format(spacing))
        
        # Create settings dictionary
        if sampled_function == 'cobaya_loglike':
            params_name = 'cobaya'
        else:
            params_name = 'params'
        self.settings = {
            params_name: params,
            'sampled_function': sampled_function,
            'n_samples': n_samples,
            'spacing': spacing,
        }

        # Function to be sampled
        fun = eval('fng.' + sampled_function)

        if sampled_function == 'cobaya_loglike':
            sampled_params = params['params']
        else:
            sampled_params = params
        # Get x names
        self.x_names = [x for x in sampled_params
                        if 'prior' in sampled_params[x]]

        # Get x array
        x_sampler = smp.Sampler().choose_one(spacing, verbose=verbose)
        self.x = x_sampler.get_x(sampled_params, self.x_names, n_samples)
        self.n_x = self.x.shape[1]

        # Get first sampled y (to retrieve y_names)
        y_val, self.y_names, model = fun(self.x[0], self.x_names, params)
        self.y = [y_val]

        if save_incrementally:
            io.Folder(output_path).create(verbose=verbose)
            self.save(output_path, verbose=verbose)

        # Sample y
        for x in tqdm.tqdm(self.x[1:]):
            y_val, _, _ = fun(x, self.x_names, params, model=model)
            self.y.append(y_val)
            if save_incrementally:
                with open(os.path.join(output_path,
                                       de.file_names['y_sample']['name']),
                                       'a') as fn:
                    np.savetxt(fn, [y_val])

        self.y = np.array(self.y)
        self.n_y = self.y.shape[1]
        return

    def join(self):
        return

    def generate_old(self, params, verbose=False, root=None, resume=False):

        # Resume
        if resume:
            data_x = io.File(de.file_names['x_sample']['name'], root=root)
            data_x.load_array(verbose=verbose)
            self.x = data_x.content
            data_y = io.File(de.file_names['y_sample']['name'], root=root)
            data_y.load_array(verbose=verbose)
            self.y = data_y.content
            if self.x.ndim == 1:
                self.x = self.x[:, np.newaxis]
            if self.y.ndim == 1:
                self.y = self.y[:, np.newaxis]
            start_y = self.y.shape[0]
        else:
            start_y = 1


        if resume:
            self.y = list(self.y)

        for x in tqdm.tqdm(self.x[start_y:]):
            y_val, _, _ = self.function(
                x, self.x_names,
                param_dict, model=model)
            self.y.append(y_val)
            if root:
                data_y.append_array(y_val)
        self.y = np.array(self.y)
        self.n_y = self.y.shape[1]

        return self.x, self.y

    def train_test_split(self, frac_train, seed, verbose=False):
        if verbose:
            io.info('Splitting training and testing samples.')
            io.print_level(1, 'Fractional number of training samples: {}'
                            ''.format(frac_train))
            io.print_level(1, 'Random seed for training/testing split: '
                            '{}'.format(seed))
        split = skl_ms.train_test_split(self.x, self.y,
                                        train_size=frac_train,
                                        random_state=seed)
        self.x_train, self.x_test, self.y_train, self.y_test = split
        return

    def rescale(self, rescale_x, rescale_y, verbose=False):
        if verbose:
            io.info('Rescaling x and y.')
            io.print_level(1, 'x with: {}'.format(rescale_x))
            io.print_level(1, 'y with: {}'.format(rescale_y))
        # Rescale x
        self.scaler_x = sc.Scaler.choose_one(rescale_x)
        self.scaler_x.fit(self.x_train)
        self.x_train_scaled = self.scaler_x.transform(self.x_train)
        self.x_test_scaled = self.scaler_x.transform(self.x_test)
        # Rescale y
        self.scaler_y = sc.Scaler.choose_one(rescale_y)
        self.scaler_y.fit(self.y_train)
        self.y_train_scaled = self.scaler_y.transform(self.y_train)
        self.y_test_scaled = self.scaler_y.transform(self.y_test)
        if verbose:
            io.print_level(1, 'Rescaled bounds:')
            mins = np.min(self.x_train_scaled, axis=0)
            maxs = np.max(self.x_train_scaled, axis=0)
            for nx, min in enumerate(mins):
                io.print_level(
                    2, 'x_train_{} = [{}, {}]'.format(nx, min, maxs[nx]))
            mins = np.min(self.x_test_scaled, axis=0)
            maxs = np.max(self.x_test_scaled, axis=0)
            for nx, min in enumerate(mins):
                io.print_level(
                    2, 'x_test_{} = [{}, {}]'.format(nx, min, maxs[nx]))
            mins = np.min(self.y_train_scaled, axis=0)
            maxs = np.max(self.y_train_scaled, axis=0)
            for nx, min in enumerate(mins):
                io.print_level(
                    2, 'y_train_{} = [{}, {}]'.format(nx, min, maxs[nx]))
            mins = np.min(self.y_test_scaled, axis=0)
            maxs = np.max(self.y_test_scaled, axis=0)
            for nx, min in enumerate(mins):
                io.print_level(
                    2, 'y_test_{} = [{}, {}]'.format(nx, min, maxs[nx]))
        return

    def get_plots(self, output, verbose=False):
        # Avoid plots if x or y are more than a scalar
        if self.n_y != 1 or self.n_x != 1:
            return
        if verbose:
            io.info('Generating plots.')
        for nx in range(self.n_x):
            for ny in range(self.n_y):
                # Plot original sample
                pl.ScatterPlot(
                    [(self.x_train[:, nx], self.y_train[:, ny]),
                     (self.x_test[:, nx], self.y_test[:, ny])],
                    labels=['train', 'test'],
                    x_label='x_{}'.format(nx),
                    y_label='y_{}'.format(ny),
                    root=output.subfolder('plots'),
                    verbose=verbose).save()
                # Plot rescaled sample
                pl.ScatterPlot(
                    [(self.x_train_scaled[:, nx], self.y_train_scaled[:, ny]),
                     (self.x_test_scaled[:, nx], self.y_test_scaled[:, ny])],
                    labels=['train_scaled', 'test_scaled'],
                    x_label='x_scaled_{}'.format(nx),
                    y_label='y_scaled_{}'.format(ny),
                    root=output.subfolder('plots'),
                    verbose=verbose).save()
        return
