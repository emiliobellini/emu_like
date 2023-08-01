import numpy as np
import sklearn.model_selection as skl_ms
import tools.generate_functions as fng  # noqa:F401
import tools.defaults as de
import tools.io as io
import tools.plots as pl
import tools.printing_scripts as scp
import tools.scalers as sc
import tools.samplers as smp


class Sample(object):
    """
    Deal with samples, either read them or generate them.
    """

    def __init__(self):
        return

    def _init_from_params(self, params):
        self.fn_name = params['function']
        self.spacing = params['spacing']
        self.params = params['params']
        self.n_samples = params['n_samples']
        self.varying_params = [x for x in self.params if self._is_varying(x)]
        self.fixed_params = [x for x in self.params if not self._is_varying(x)]
        self.n_x = len(self.varying_params)
        # Call the function to be sampled
        self.function = eval('fng.'+self.fn_name)
        return

    def _init_from_path(self, params):
        has_params = False
        single_path = 'path' in params.keys()
        two_files = 'path_x' in params.keys() and 'path_y' in params.keys()
        if single_path and two_files:
            raise Exception(
                'Too many files to load. Please specify one between '
                'path and [path_x, path_y]')
        elif two_files:
            # Two files are specified
            path_x = params['path_x']
            path_y = params['path_y']
        elif single_path:
            try:
                # One file is specified
                io.File(params['path'], should_exist=True)
                path_x = params['path']
                path_y = params['path']
            except IOError:
                # One folder is specified
                path_x = io.File(de.file_names['x_sample']['name'],
                                 root=params['path']).path
                path_y = io.File(de.file_names['y_sample']['name'],
                                 root=params['path']).path
                sample_params = io.YamlFile(de.file_names['params']['name'],
                                            root=params['path'])
                sample_params.read()
                self._init_from_params(sample_params)
                two_files = True
                has_params = True
        else:
            raise Exception(
                'No samples to load. Please specify one between '
                'path and [path_x, path_y]')
        return path_x, path_y, two_files, has_params

    def _get_columns(self, params, two_files=True):
        try:
            idx_x = params['columns_x']
        except KeyError:
            if two_files:
                idx_x = slice(None, None)
            else:
                idx_x = slice(None, -1)
        try:
            idx_y = params['columns_y']
        except KeyError:
            if two_files:
                idx_y = slice(None, None)
            else:
                idx_y = slice(-1, None)
        return idx_x, idx_y

    def _is_varying(self, param):
        if isinstance(self.params[param], dict):
            return True
        else:
            return False

    def _print_init(self, has_params=True, from_file=False):
        if from_file:
            scp.print_level(1, 'x from: {}'.format(self.path_x))
            scp.print_level(1, 'y from: {}'.format(self.path_y))
        if has_params:
            scp.print_level(1, 'Sampling function: {}'.format(self.fn_name))
            scp.print_level(1, 'Spacing: {}'.format(self.spacing))
            scp.print_level(1, 'Varying parameters:')
            for x in self.varying_params:
                scp.print_level(2, '{} = {}'.format(x, self.params[x]))
            scp.print_level(1, 'Fixed parameters:')
            for x in self.fixed_params:
                scp.print_level(2, '{} = {}'.format(x, self.params[x]))
        else:
            mins = np.min(self.x, axis=0)
            maxs = np.max(self.x, axis=0)
            for nx, min in enumerate(mins):
                scp.print_level(
                    2, 'x_{} = [{}, {}]'.format(nx, min, maxs[nx]))
        scp.print_level(
            1, 'Number of x variables: {}'.format(self.n_x))
        scp.print_level(1, 'N samples: {}'.format(self.n_samples))
        return

    def generate(self, params, verbose=False):

        # Initialize
        self._init_from_params(params)

        if verbose:
            scp.info('Generating sample.')
            self._print_init()

        # Get x array
        x_sampler = smp.Sampler().choose_one(self.spacing, verbose=verbose)
        self.x = x_sampler.get_x(
            self.params, self.varying_params, self.n_samples)

        # Get y samples
        self.y = self.function(self.x, self.varying_params, self.params)
        self.n_y = self.y.shape[1]
        return self.x, self.y

    def load(self, params, verbose=False):

        # Initialize
        self.path_x, self.path_y, two_files, has_params = \
            self._init_from_path(params)
        # Columns to read
        idx_x, idx_y = self._get_columns(params, two_files)

        if verbose:
            scp.info('Loading sample.')

        # Load sample
        x = np.genfromtxt(self.path_x)
        y = np.genfromtxt(self.path_y)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]
        x = x[:, idx_x]
        y = y[:, idx_y]

        self.n_y = y.shape[1]
        if has_params:
            if self.n_x != x.shape[1] or self.n_samples != x.shape[0]:
                raise IOError('Input data and parameter file inconsistent!')
        else:
            self.n_samples = x.shape[0]
            self.n_x = x.shape[1]
        self.x = x
        self.y = y

        if verbose:
            self._print_init(has_params=has_params, from_file=True)
        return

    def train_test_split(self, frac_train, seed, verbose=False):
        if verbose:
            scp.info('Splitting training and testing samples.')
            scp.print_level(1, 'Fractional number of training samples: {}'
                            ''.format(frac_train))
            scp.print_level(1, 'Random seed for training/testing split: '
                            '{}'.format(seed))
        split = skl_ms.train_test_split(self.x, self.y,
                                        train_size=frac_train,
                                        random_state=seed)
        self.x_train, self.x_test, self.y_train, self.y_test = split
        return

    def save(self, output, verbose=False):
        if verbose:
            scp.info('Saving sample.')
        # Save x
        data_x = io.File(de.file_names['x_sample']['name'], root=output)
        data_x.content = self.x
        data_x.save_array(verbose=verbose)
        # Save y
        data_y = io.File(de.file_names['y_sample']['name'], root=output)
        data_y.content = self.y
        data_y.save_array(verbose=verbose)
        return

    def rescale(self, rescale_x, rescale_y, verbose=False):
        if verbose:
            scp.info('Rescaling x and y.')
            scp.print_level(1, 'x with: {}'.format(rescale_x))
            scp.print_level(1, 'y with: {}'.format(rescale_y))
        # Rescale x
        self.scaler_x = sc.Scaler(name=rescale_x)
        self.scaler_x.fit(self.x_train)
        self.x_train_scaled = self.scaler_x.transform(self.x_train)
        self.x_test_scaled = self.scaler_x.transform(self.x_test)
        # Rescale y
        self.scaler_y = sc.Scaler(name=rescale_y)
        self.scaler_y.fit(self.y_train)
        self.y_train_scaled = self.scaler_y.transform(self.y_train)
        self.y_test_scaled = self.scaler_y.transform(self.y_test)
        if verbose:
            scp.print_level(1, 'Rescaled bounds:')
            mins = np.min(self.x_train_scaled, axis=0)
            maxs = np.max(self.x_train_scaled, axis=0)
            for nx, min in enumerate(mins):
                scp.print_level(
                    2, 'x_train_{} = [{}, {}]'.format(nx, min, maxs[nx]))
            mins = np.min(self.x_test_scaled, axis=0)
            maxs = np.max(self.x_test_scaled, axis=0)
            for nx, min in enumerate(mins):
                scp.print_level(
                    2, 'x_test_{} = [{}, {}]'.format(nx, min, maxs[nx]))
            mins = np.min(self.y_train_scaled, axis=0)
            maxs = np.max(self.y_train_scaled, axis=0)
            for nx, min in enumerate(mins):
                scp.print_level(
                    2, 'y_train_{} = [{}, {}]'.format(nx, min, maxs[nx]))
            mins = np.min(self.y_test_scaled, axis=0)
            maxs = np.max(self.y_test_scaled, axis=0)
            for nx, min in enumerate(mins):
                scp.print_level(
                    2, 'y_test_{} = [{}, {}]'.format(nx, min, maxs[nx]))
        return

    def get_plots(self, output, verbose=False):
        # Not implemented plots if y is more than a scalar
        if self.n_y != 1:
            return
        if verbose:
            scp.info('Generating plots.')
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
