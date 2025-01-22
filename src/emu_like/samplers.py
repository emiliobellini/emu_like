"""
.. module:: samplers

:Synopsis: Various sampler classes, dealing with different sampling spacing.
:Author: Emilio Bellini

"""

import numpy as np
import scipy
from . import defaults as de
from . import io as io


class Sampler(object):

    def __init__(self, params):
        self.params = params
        # Default n_samples if not passed.
        self.n_samples_default = 10

        # Placeholders
        self.x = None  # x
        self.x_ranges = None  # x_ranges
        self.n_x = None  # Number of x variables
        self.n_samples = None  # Number of samples
        self.x_names = None  # List of names of x data
        self.x_header = None  # Header for x file
        self.x_fname = None  # File name of x data
        return

    @staticmethod
    def choose_one(sampler_name, params, verbose=False, **kwargs):
        """
        Main function to get the correct Sampler.

        Arguments:
        - sampler_name (str): type of sampler;
        - verbose (bool, default: False): verbosity;
        - kwargs: specific arguments needed by each.

        Return:
        - Sampler (object): based on sampler_name, get
          the correct sampler and initialize it.
        """
        if sampler_name == 'evaluate':
            return EvaluateSampler(params, verbose=verbose)
        elif sampler_name == 'grid':
            return GridSampler(params, verbose=verbose, **kwargs)
        elif sampler_name == 'log_grid':
            return LogGridSampler(params, verbose=verbose, **kwargs)
        elif sampler_name == 'random_uniform':
            return RandomUniformSampler(params, verbose=verbose, **kwargs)
        elif sampler_name == 'random_normal':
            return RandomNormalSampler(params, verbose=verbose, **kwargs)
        elif sampler_name == 'latin_hypercube':
            return LatinHypercubeSampler(params, verbose=verbose, **kwargs)
        else:
            raise ValueError('Sampler not recognized!')

    def _get_n_samples(self, args):
        """
        Looking at the arguments, this has to deal with:
        - the key is not present
        - the key is present and it has None value
        - the key is present and it has a value
        """
        if 'n_samples' in args.keys() and args['n_samples'] is not None:
            n_samples = args['n_samples']
        else:
            n_samples = self.n_samples_default
        return n_samples

    def _get_seed(self, args):
        """
        Looking at the arguments, this has to deal with:
        - the key is not present
        - the key is present and it has None value
        - the key is present and it has a value
        """
        if 'seed' in args.keys():
            seed = args['seed']
        else:
            seed = None
        return seed

    def _is_varying(self, params, param):
        """
        Return True if param has key 'prior',
        False otherwise.
        """
        if isinstance(params[param], dict):
            if 'prior' in params[param].keys():
                return True
        return False

    def get_x_ranges(self):
        """
        Get x_ranges.
        """
        if self.x_ranges is None:
            if self.x is None:
                self.get_x()

            self.x_ranges = list(zip(
                np.min(self.x, axis=0), np.max(self.x, axis=0)))
        return self.x_ranges

    def get_n_x(self):
        """
        Get n_x.
        """
        if self.n_x is None:
            if self.x is None:
                self.get_x()

            self.n_x = self.x.shape[1]
        return self.n_x

    def get_n_samples(self):
        """
        Get n_samples.
        """
        if self.n_samples is None:
            if self.x is None:
                self.get_x()

            self.n_samples = self.x.shape[0]
        return self.n_samples

    def get_x_names(self):
        """
        Get x_names.
        """
        if self.x_names is None:
            self.x_names = [x for x in self.params
                            if self._is_varying(self.params, x)]
        return self.x_names

    def get_x_header(self):
        """
        Get x_header.
        """
        if self.x_header is None:
            if self.x_names is None:
                self.get_x_names()
            
            self.x_header = '\t'.join(self.x_names)
        return self.x_header

    def get_x_fname(self):
        """
        Get x_fname.
        """
        if self.x_fname is None:
            self.x_fname = de.file_names['x_sample']['name']
        return self.x_fname

    def get_x(self):
        """
        Placeholder for get_x.
        This should return a 2D array
        with dimensions (n_x, n_samples).
        """
        self.x = None
        return self.x


class EvaluateSampler(Sampler):
    """
    Sample at one point (defined by 'params:ref').
    """

    def __init__(self, params, verbose=False):
        Sampler.__init__(self, params)

        if verbose:
            io.info('Initializing Evaluate sampler.')
        return

    def get_x(self):
        """
        This should return a 2D array
        with dimensions (n_x, n_samples).
        """
        if self.x_names is None:
            self.get_x_names()

        self.x = np.array([[self.params[p]['ref'] for p in self.x_names]])
        return self.x


class GridSampler(Sampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init Sampler specific arguments
        """
        Sampler.__init__(self, params)
        self.n_samples = self._get_n_samples(kwargs)

        if verbose:
            io.info('Initializing Grid sampler.')
            io.print_level(1, 'Number of samples: {}'.format(self.n_samples))
        return

    def get_x(self):
        """
        This should return a 2D array
        with dimensions (n_x, n_samples).
        """
        if self.x_names is None:
            self.get_x_names()

        mins = [self.params[x]['prior']['min'] for x in self.x_names]
        maxs = [self.params[x]['prior']['max'] for x in self.x_names]
        # Get points per size
        size = int(np.power(self.n_samples, 1./len(self.x_names)))
        # Get coordinates
        coords = [np.linspace(m, M, num=size) for m, M in zip(mins, maxs)]
        coords = np.meshgrid(*coords)
        coords = tuple([x.ravel() for x in coords])
        self.x = np.vstack(coords).T
        return self.x


class LogGridSampler(Sampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init Sampler specific arguments
        """
        Sampler.__init__(self, params)
        self.n_samples = self._get_n_samples(kwargs)

        if verbose:
            io.info('Initializing LogGrid sampler.')
            io.print_level(1, 'Number of samples: {}'.format(self.n_samples))
        return

    def get_x(self):
        """
        This should return a 2D array
        with dimensions (n_x, n_samples).
        """
        if self.x_names is None:
            self.get_x_names()

        mins = [np.log10(self.params[x]['prior']['min']) for x in self.x_names]
        maxs = [np.log10(self.params[x]['prior']['max']) for x in self.x_names]
        # Get points per size
        size = int(np.power(self.n_samples, 1./len(self.x_names)))
        # Get coordinates
        coords = [np.logspace(m, M, num=size) for m, M in zip(mins, maxs)]
        coords = np.meshgrid(*coords)
        coords = tuple([x.ravel() for x in coords])
        self.x = np.vstack(coords).T
        return self.x


class RandomUniformSampler(Sampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init Sampler specific arguments
        """
        Sampler.__init__(self, params)
        self.n_samples = self._get_n_samples(kwargs)

        if verbose:
            io.info('Initializing RandomUniform sampler.')
            io.print_level(1, 'Number of samples: {}'.format(self.n_samples))
        return

    def get_x(self):
        """
        This should return a 2D array
        with dimensions (n_x, n_samples).
        """
        if self.x_names is None:
            self.get_x_names()

        mins = [self.params[x]['prior']['min'] for x in self.x_names]
        maxs = [self.params[x]['prior']['max'] for x in self.x_names]

        self.x = np.random.uniform(
            mins, maxs, size=(self.n_samples, len(mins)))
        return self.x


class RandomNormalSampler(Sampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init Sampler specific arguments
        """
        Sampler.__init__(self, params)
        self.n_samples = self._get_n_samples(kwargs)

        if verbose:
            io.info('Initializing RandomNormal sampler.')
            io.print_level(1, 'Number of samples: {}'.format(self.n_samples))
        return

    def get_x(self):
        """
        This should return a 2D array
        with dimensions (n_x, n_samples).
        """
        if self.x_names is None:
            self.get_x_names()

        means = [self.params[x]['prior']['loc'] for x in self.x_names]
        std = [self.params[x]['prior']['scale'] for x in self.x_names]
 
        self.x = np.random.normal(
            means, std, size=(self.n_samples, len(means)))
        return self.x


class LatinHypercubeSampler(Sampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init Sampler specific arguments
        """
        Sampler.__init__(self, params)
        self.n_samples = self._get_n_samples(kwargs)
        self.seed = self._get_seed(kwargs)

        if verbose:
            io.info('Initializing LatinHypercube sampler.')
            io.print_level(1, 'Number of samples: {}'.format(self.n_samples))
            io.print_level(1, 'Seed: {}'.format(self.seed))
        return

    def get_x(self):
        """
        This should return a 2D array
        with dimensions (n_x, n_samples).
        """
        if self.x_names is None:
            self.get_x_names()

        mins = [self.params[x]['prior']['min'] for x in self.x_names]
        maxs = [self.params[x]['prior']['max'] for x in self.x_names]

        sampler = scipy.stats.qmc.LatinHypercube(
            d=len(mins),
            rng=np.random.default_rng(seed=self.seed))
        sample = sampler.random(n=self.n_samples)
        self.x = scipy.stats.qmc.scale(sample, mins, maxs)
        return self.x
