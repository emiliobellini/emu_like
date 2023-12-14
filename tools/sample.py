import numpy as np
import tqdm
import sklearn.model_selection as skl_ms
import tools.sampling_functions as fng  # noqa:F401
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
        # Get parameters
        try:
            self.params = params['params']
        except KeyError:
            self.cobaya_params = params['cobaya']
            self.params = self.cobaya_params['params']
        # Get additional info
        self.fn_name = params['function']
        self.spacing = params['spacing']
        self.n_samples = params['n_samples']
        self.x_names = [x for x in self.params if self._is_varying(x)]
        self.args_names = [x for x in self.params if not self._is_varying(x)]
        self.n_x = len(self.x_names)
        # Call the function to be sampled
        self.function = eval('fng.'+self.fn_name)
        return

    def _init_from_paths(self, params):
        has_params = False
        single_path = 'paths' in params.keys()
        two_files = 'paths_x' in params.keys() and 'paths_y' in params.keys()
        if single_path and two_files:
            raise Exception(
                'Too many files to load. Please specify one between '
                'path and [path_x, path_y]')
        elif two_files:
            # Two files are specified
            paths_x = params['paths_x']
            paths_y = params['paths_y']
        elif single_path:
            try:
                # One file is specified
                io.File(params['paths'][0], should_exist=True)
                paths_x = params['paths']
                paths_y = params['paths']
            except IOError:
                # One folder is specified
                paths_x = [io.File(de.file_names['x_sample']['name'],
                                   root=x).path for x in params['paths']]
                paths_y = [io.File(de.file_names['y_sample']['name'],
                                   root=x).path for x in params['paths']]
                sample_params = [io.YamlFile(de.file_names['params']['name'],
                                             root=x) for x in params['paths']]
                [x.read() for x in sample_params]
                self._init_from_params(sample_params[0])
                self.n_samples = sum([x['n_samples'] for x in sample_params])
                two_files = True
                has_params = True
        else:
            raise Exception(
                'No samples to load. Please specify one between '
                'path and [path_x, path_y]')
        return paths_x, paths_y, two_files, has_params

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
            if 'prior' in self.params[param].keys():
                return True
        return False

    def _print_init(self, has_params=True, from_file=False):
        if from_file:
            scp.print_level(1, 'x from: {}'.format(self.paths_x))
            scp.print_level(1, 'y from: {}'.format(self.paths_y))
        if has_params:
            scp.print_level(1, 'Sampling function: {}'.format(self.fn_name))
            scp.print_level(1, 'Spacing: {}'.format(self.spacing))
            scp.print_level(1, 'Varying parameters:')
            for x in self.x_names:
                scp.print_level(2, '{} = {}'.format(x, self.params[x]))
            scp.print_level(1, 'Fixed parameters:')
            for x in self.args_names:
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

    def generate(self, params, verbose=False, root=None, resume=False):

        # Initialize
        self._init_from_params(params)

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

        if verbose:
            scp.info('Generating sample.')
            self._print_init()

        if not resume:
            # Get x array
            x_sampler = smp.Sampler().choose_one(self.spacing, verbose=verbose)
            self.x = x_sampler.get_x(
                self.params, self.x_names, self.n_samples)
            # Save x
            if root:
                data_x = io.File(de.file_names['x_sample']['name'], root=root)
                data_x.content = self.x
                data_x.save_array(header='\t'.join(self.x_names),
                                  verbose=verbose)

        # Get y samples
        try:
            param_dict = self.cobaya_params
        except AttributeError:
            param_dict = self.params

        y_val, self.y_names, model = self.function(
            self.x[0], self.x_names,
            param_dict)

        if resume:
            self.y = list(self.y)
        else:
            self.y = [y_val]
            # Save y
            if root:
                if self.y_names:
                    header_y = '\t'.join(self.y_names)
                else:
                    header_y = ''
                data_y = io.File(de.file_names['y_sample']['name'], root=root)
                data_y.content = self.y
                data_y.save_array(header=header_y, verbose=verbose)

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

    def load(self, params, verbose=False):

        # Initialize
        self.paths_x, self.paths_y, two_files, has_params = \
            self._init_from_paths(params)
        # Columns to read
        idx_x, idx_y = self._get_columns(params, two_files)

        if verbose:
            scp.info('Loading sample.')

        # Load sample
        x = [np.genfromtxt(sam) for sam in self.paths_x]
        x = np.vstack([xn[:, np.newaxis] if xn.ndim == 1 else xn for xn in x])
        y = [np.genfromtxt(sam) for sam in self.paths_y]
        y = np.vstack([yn[:, np.newaxis] if yn.ndim == 1 else yn for yn in y])
        x = x[:, idx_x]
        y = y[:, idx_y]
        if params['remove_non_finite']:
            if verbose:
                scp.info('Removing non finite data from sample.')
            only_finites = np.any(np.isfinite(y), axis=1)
            x = x[only_finites]
            y = y[only_finites]
            self.n_samples = x.shape[0]

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

    def save_details(self, path, verbose=False):

        path.content = {
            'x_names': self.x_names,
            'n_x': self.n_x,
            'n_y': self.n_y,
            'n_samples': self.n_samples,
            'spacing': self.spacing,
            'bounds': {x: [float(self.x[:, nx].min()),
                           float(self.x[:, nx].max())]
                       for nx, x in enumerate(self.x_names)},
        }

        path.copy_to(
            name=path.path,
            header=de.file_names['sample_details']['header'],
            verbose=verbose)

        return

    def rescale(self, rescale_x, rescale_y, verbose=False):
        if verbose:
            scp.info('Rescaling x and y.')
            scp.print_level(1, 'x with: {}'.format(rescale_x))
            scp.print_level(1, 'y with: {}'.format(rescale_y))
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
        # Avoid plots if x or y are more than a scalar
        if self.n_y != 1 or self.n_x != 1:
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
