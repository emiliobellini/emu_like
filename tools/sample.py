import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as skl_ms
import tools.generate_functions as fng  # noqa:F401
import tools.io as io
import tools.printing_scripts as scp
import tools.scalers as sc


class Sample(object):
    """
    Deal with samples, either read them or generate them.
    """

    def __init__(self):
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
        path = output.subfolder('sample').create(verbose=verbose)
        # Save x
        data_x = io.File('x_sample.txt', root=path)
        data_x.content = self.x
        data_x.save_array(verbose=verbose)
        # Save y
        data_y = io.File('y_sample.txt', root=path)
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
        path = output.subfolder('plots').create(verbose=verbose)
        for nx in range(self.n_x):
            # Plot original sample
            x_name = 'x_{}'.format(nx)
            fname = io.File(x_name+'.pdf', root=path)
            plt.figure()
            plt.scatter(self.x_train[:, nx], self.y_train[:, 0],
                        label='train', s=1)
            plt.scatter(self.x_test[:, nx], self.y_test[:, 0],
                        label='test', s=1)
            plt.xlabel(x_name)
            plt.ylabel('y_0')
            plt.legend()
            fname.savefig(plt, verbose=verbose)
            # Plot rescaled sample
            x_name = 'x_scaled_{}'.format(nx)
            fname = io.File(x_name+'.pdf', root=path)
            plt.figure()
            plt.scatter(self.x_train_scaled[:, nx], self.y_train_scaled[:, 0],
                        label='train', s=1)
            plt.scatter(self.x_test_scaled[:, nx], self.y_test_scaled[:, 0],
                        label='test_scaled', s=1)
            plt.xlabel(x_name)
            plt.ylabel('y_scaled_0')
            plt.legend()
            fname.savefig(plt, verbose=verbose)
        return


class LoadSample(Sample):
    """
    Load samples.
    """

    def __init__(self, params, verbose=False):
        Sample.__init__(self)
        # File(s) to load
        self.path_x, self.path_y, single_file = self._file_names(params)
        # Columns to read
        self.idx_x, self.idx_y = self._get_columns(params, single_file)
        # Load data
        self.x, self.y = self.load()
        self.n_samples = self.x.shape[0]
        self.n_x = self.x.shape[1]
        self.n_y = self.y.shape[1]
        if verbose:
            self._print_init(single_file)
        # Eventually save sample in output folder
        try:
            self.save_x_y = params['save']
        except KeyError:
            self.save_x_y = False
        return

    def _file_names(self, params):
        single_file = 'path' in params.keys()
        two_files = 'path_x' in params.keys() and 'path_y' in params.keys()
        if single_file and two_files:
            raise Exception(
                'Too many files to load. Please specify one between '
                'path and [path_x, path_y]')
        elif single_file:
            path_x = params['path']
            path_y = params['path']
        elif two_files:
            path_x = params['path_x']
            path_y = params['path_y']
        else:
            raise Exception(
                'No samples to load. Please specify one between '
                'path and [path_x, path_y]')
        return path_x, path_y, single_file

    def _get_columns(self, params, single_file=True):
        try:
            idx_x = params['columns_x']
        except KeyError:
            if single_file:
                idx_x = slice(None, -1)
            else:
                idx_x = slice(None, None)
        try:
            idx_y = params['columns_y']
        except KeyError:
            if single_file:
                idx_y = slice(-1, None)
            else:
                idx_y = slice(None, None)
        return idx_x, idx_y

    def _print_init(self, single_file=True):
        scp.info('Loading sample.')
        if single_file:
            scp.print_level(1, 'From: {}'.format(self.path_x))
        else:
            scp.print_level(1, 'From (x): {}'.format(self.path_x))
            scp.print_level(1, 'From (y): {}'.format(self.path_y))
        scp.print_level(
            1, 'Number of x variables: {}'.format(self.n_x))
        mins = np.min(self.x, axis=0)
        maxs = np.max(self.x, axis=0)
        for nx, min in enumerate(mins):
            scp.print_level(
                2, 'x_{} = [{}, {}]'.format(nx, min, maxs[nx]))
        scp.print_level(1, 'N samples: {}'.format(self.n_samples))
        return

    def load(self):
        x = np.genfromtxt(self.path_x)[:, self.idx_x]
        y = np.atleast_2d(np.genfromtxt(self.path_y)[:, self.idx_y])
        return x, y


class GenerateSample(Sample):
    """
    Generate samples.
    """

    def __init__(self, params, verbose=False):
        Sample.__init__(self)
        self.fn_name = params['function']
        self.spacing = params['spacing']
        self.params = np.array(params['sampling_params'])
        self.n_samples = params['n_samples']
        self.n_x = len(self.params)
        try:
            self.kwargs = params['kwargs']
        except KeyError:
            pass
        # Call the function to be sampled
        self.function = eval('fng.'+self.fn_name)
        if verbose:
            self._print_init()
        # Eventually save sample in output folder
        try:
            self.save_x_y = params['save']
        except KeyError:
            self.save_x_y = False
        return
        return

    def _print_init(self):
        scp.info('Generating sample.')
        scp.print_level(1, 'Sampling function: {}'.format(self.fn_name))
        if self.kwargs:
            scp.print_level(2, 'with parameters: {}'.format(self.kwargs))
        scp.print_level(
            1, 'Number of x variables: {}'.format(self.n_x))
        scp.print_level(1, 'Spacing: {}'.format(self.spacing))
        if self.spacing == 'random_normal':
            scp.print_level(1, 'Sampling params (mean, std):')
        else:
            scp.print_level(1, 'Sampling params (min, max):')
        for nx, (xmin, xmax) in enumerate(self.params):
            scp.print_level(
                2, 'x_{} = [{}, {}]'.format(nx, xmin, xmax))
        scp.print_level(1, 'N samples: {}'.format(self.n_samples))
        return

    def _get_x_array(self):
        lefts = self.params[:, 0]
        rights = self.params[:, 1]
        if self.spacing == 'grid':
            x = np.linspace(lefts, rights, num=self.n_samples)
        elif self.spacing == 'log_grid':
            lefts, rights = np.log10(lefts), np.log10(rights)
            x = np.logspace(lefts, rights, num=self.n_samples)
        elif self.spacing == 'random_uniform':
            x = np.random.uniform(lefts, rights,
                                  size=(self.n_samples, len(lefts)))
        elif self.spacing == 'random_normal':
            x = np.random.normal(lefts, rights,
                                 size=(self.n_samples, len(lefts)))
        else:
            raise Exception('Spacing not recognized!')
        return x

    def generate(self):
        # Get x array
        self.x = self._get_x_array()

        # Get y samples
        if self.kwargs:
            self.y = self.function(self.x, **self.kwargs)
        else:
            self.y = self.function(self.x)
        self.n_y = self.y.shape[1]
        return self.x, self.y
