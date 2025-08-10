"""
.. module:: x_samplers

:Synopsis: Various sampler classes, dealing with different sampling spacing.
:Author: Emilio Bellini

Collection of functions that can be used to sample the x
parameter space. Each one is stored as a Class inheriting
from the base Class XSampler.
If you want to implement a new function, create
a new Class inheriting from XSampler.
Add its name in the choose_one static method,
create the get_x method and adapt the other methods
and attributes to your needs.
"""

import numpy as np
import scipy
from . import defaults as de
from . import io as io


class XSampler(object):
    """
    Base class XSampler.
    """

    def __init__(self, params):
        self.params = params
        # Default n_samples if not passed.
        self.n_samples_default = 10

        # Defaults
        self.x_key = 'x_data'
        # Placeholders
        self.x = None  # x
        self.x_ranges = None  # x_ranges
        self.n_x = None  # Number of x variables
        self.n_samples = None  # Number of samples
        self.x_names = None  # List of names of x data
        self.x_header = None  # Header for x file
        return

    @staticmethod
    def choose_one(name, params, verbose=False, **kwargs):
        """
        Main function to get the correct XSampler.

        Arguments:
        - name (str): type of sampler;
        - params (dict): dictionary of parameters;
        - verbose (bool, default: False): verbosity;
        - kwargs: specific arguments needed by each sampler.

        Return:
        - XSampler (object): based on name, get
          the correct sampler and initialize it.
        """
        if name == 'evaluate':
            return EvaluateSampler(params, verbose=verbose)
        elif name == 'grid':
            return GridSampler(params, verbose=verbose, **kwargs)
        elif name == 'log_grid':
            return LogGridSampler(params, verbose=verbose, **kwargs)
        elif name == 'random_uniform':
            return RandomUniformSampler(params, verbose=verbose, **kwargs)
        elif name == 'random_normal':
            return RandomNormalSampler(params, verbose=verbose, **kwargs)
        elif name == 'latin_hypercube':
            return LatinHypercubeSampler(params, verbose=verbose, **kwargs)
        else:
            raise ValueError('XSampler not recognized!')

    @staticmethod
    def _is_varying(params, param):
        """
        Return True if param has key 'prior',
        False otherwise.
        """
        if isinstance(params[param], dict):
            if 'prior' in params[param].keys():
                return True
        return False

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

    def get_x_ranges(self):
        """
        Get x_ranges.
        """
        if self.x is None:
            self.get_x()

        self.x_ranges = np.array(list(zip(
            np.min(self.x, axis=0), np.max(self.x, axis=0))))
        return self.x_ranges

    def get_n_x(self):
        """
        Get n_x.
        """
        if self.x is None:
            self.get_x()

        self.n_x = self.x.shape[1]
        return self.n_x

    def get_n_samples(self):
        """
        Get n_samples.
        """
        if self.x is None:
            self.get_x()

        self.n_samples = self.x.shape[0]
        return self.n_samples

    def get_x_names(self):
        """
        Get x_names.
        """
        self.x_names = [x for x in self.params
                        if XSampler._is_varying(self.params, x)]

        return self.x_names

    def get_x_header(self):
        """
        Get x_header.
        """
        if self.x_names is None:
            self.get_x_names()
        
        self.x_header = '\t'.join(self.x_names)
        return self.x_header

    def get_x(self):
        """
        Placeholder for get_x.
        This should return a 2D array
        with dimensions (n_x, n_samples).
        """
        self.x = None
        return self.x


class EvaluateSampler(XSampler):
    """
    Sample at one point (defined by 'params:ref').
    """

    def __init__(self, params, verbose=False):
        XSampler.__init__(self, params)

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


class GridSampler(XSampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init XSampler specific arguments
        """
        XSampler.__init__(self, params)
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


class LogGridSampler(XSampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init XSampler specific arguments
        """
        XSampler.__init__(self, params)
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


class RandomUniformSampler(XSampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init XSampler specific arguments
        """
        XSampler.__init__(self, params)
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


class RandomNormalSampler(XSampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init XSampler specific arguments
        """
        XSampler.__init__(self, params)
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


class LatinHypercubeSampler(XSampler):

    def __init__(self, params, verbose=False, **kwargs):
        """
        Init XSampler specific arguments
        """
        XSampler.__init__(self, params)
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
