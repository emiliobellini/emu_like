"""
.. module:: train_generators

:Synopsis: List of generator for traininf set.
:Author: Emilio Bellini

Collection of functions that can be used to generate
a training set. Each one is stored as a Class inheriting
from the base Class TrainGenerator.
If you want to implement a new function, create
a new Class inheriting from TrainGenerator.
Add its name in the choose_one static method,
create the get_x method and adapt its other
methods and attributes to your needs.
"""

import numpy as np
from . import io as io
from . import defaults as de
from .samplers import Sampler
from .spectra import Spectra


# Base function
class TrainGenerator(object):
    """
    Base class TrainGenerator.
    """

    @staticmethod
    def choose_one(
        generator_name,
        generator_outputs,
        params,
        n_samples,
        verbose=False,
        **kwargs):
        """
        Main function to get the correct generator function.

        Arguments:
        - generator_name (str): type of generator;
        - params (dict): dictionary of parameters;
        - n_samples (int): number of samples;
        - verbose (bool, default: False): verbosity;
        - kwargs: specific arguments needed by each generator.

        Return:
        - TrainGenerator (object): based on its name, get
          the correct sampling function and initialize it.
        """
        if generator_name == 'linear_1d':
            return Linear1D(params, n_samples, verbose=verbose, **kwargs)
        elif generator_name == 'class_spectra':
            return ClassSpectra(params, n_samples, generator_outputs,
                                verbose=verbose, **kwargs)

    def __init__(self, params, n_samples, **kwargs):
        self.params = params
        self.args = kwargs
        self.n_samples = n_samples
        # Placeholders
        self.y = []  # y per file
        self.y_ranges = []  # y_ranges per file
        self.n_y = []  # Number of y variables per file
        self.y_names = []  # List of names of y data per file
        self.y_headers = []  # Headers for y files
        self.y_fnames = []  # File names of y data

        # Derive varying parameters
        self.x_names = [x for x in self.params
                        if Sampler._is_varying(self.params, x)]
        return
    
    def get_y_ranges(self):
        """
        Get y_ranges.
        """
        if self.y_ranges == []:
            if self.y == []:
                raise Exception('Empty y arrays! Use get_y '
                                'or evaluate to generate them first.')

            self.y_ranges = [list(zip(np.min(y, axis=0), np.max(y, axis=0)))
                             for y in self.y]
        return self.y_ranges

    def get_n_y(self):
        """
        Get n_y.
        """
        return self.n_y

    def get_y_names(self):
        """
        Get y_names.
        """
        if self.y_names == []:
            for n_y in self.n_y:
                self.y_names.append(['y_{}'.format(y) for y in range(n_y)])
        return self.y_names

    def get_y_headers(self):
        """
        Get y_headers.
        """
        if self.y_headers == []:
            if self.y_names == []:
                self.get_y_names()
            
            self.y_headers = ['\t'.join(y_names) for y_names in self.y_names]
        return self.y_headers

    def get_y_fnames(self):
        """
        Get y_fnames.
        """
        if self.y_fnames == []:
            self.y_fnames = [de.file_names['y_sample']['name'].format('')]
        return self.y_fnames

    def get_y(self, x, **kwargs):
        """
        Placeholder for get_y.
        This should return a 2D array
        with dimensions (n_y, n_samples).
        """
        for nx, x_val in enumerate(x):
            self.evaluate(x_val, nx, **kwargs)
        return self.y

    def evaluate(self, x, idx, **kwargs):
        """
        Placeholder for evaluate.
        It evaluates one realisation of x.
        This should return a list of 1D arrays,
        corresponding to one line of each y output file.
        """
        return


# 1D functions

class Linear1D(TrainGenerator):
    """
    Generate a 1D linear function

    y = a*x + b
    """

    def __init__(self, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Linear1D generator.')

        TrainGenerator.__init__(self, params, n_samples, **kwargs)

        # Fix known properties of the function
        self.n_y = [1]
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]

        return

    def evaluate(self, x, idx, **kwargs):
        """
        Arguments:
        - x: 1D array of input data (one sample);
        - idx (int): row of x in the sull sample;
        Output:
        - y: 1D array of output data (one sample).

        """
        a = self.args['a']
        b = self.args['b']
        x = x[self.x_names.index('x')]
        y = a*x + b
        y = y[np.newaxis]

        # Store in self
        self.y[0][idx] = y
        return [y[np.newaxis]]


# ClassSpectra

class ClassSpectra(TrainGenerator):

    def __init__(self, params, n_samples, outputs, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing ClassSpectra generator.')

        TrainGenerator.__init__(self, params, n_samples, **kwargs)
        self.outputs = outputs

        # Init classy
        import classy
        self.classy = classy
        self.cosmo = classy.Class()
        if verbose:
            io.print_level(1, 'Loading classy from {}'.format(classy.__file__))

        # Initialise spectra
        self.spectra = Spectra(outputs, params)

        # Build parameter dictionary
        var = {nm: None for nm in self.x_names}
        self.class_params = self.args | self.spectra.get_class_output() | var

        # Fix known properties of the function
        self.n_y = self.get_n_y()
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]

        return

    def get_n_y(self):
        """
        Get n_y.
        """
        if self.n_y == []:
            self.n_y = self.spectra.get_n_vecs()
        return self.n_y

    def get_y_names(self):
        """
        Get y_names.
        """
        if self.y_names == []:
            self.y_names = self.spectra.get_names()
        return self.y_names

    def get_y_headers(self):
        """
        Get y_headers.
        """
        if self.y_headers == []:
            self.y_headers = self.spectra.get_headers()
        return self.y_headers

    def get_y_fnames(self):
        """
        Get y_fnames.
        """
        if self.y_fnames == []:
            self.y_fnames = self.spectra.get_fnames()
        return self.y_fnames

    def evaluate(self, x, idx, **kwargs):
        """
        Arguments:
        - x: 1D array of input data (one sample);
        - idx (int): row of x in the sull sample;
        Output:
        - y: 1D array of output data (one sample).

        """

        # Update parameter dictionary
        for npar, par in enumerate(self.x_names):
            self.class_params[par] = x[npar]

        try:
            # Compute class
            self.cosmo.set(self.class_params)
            self.cosmo.compute()

            y = [sp.get(self.cosmo)[np.newaxis] for sp in self.spectra]

        except self.classy.CosmoComputationError:
            # Fill with nans if error
            y = [np.full((n_y,), np.nan)[np.newaxis] for n_y in self.n_y]

        # Store in self
        for ny in range(len(self.n_y)):
            self.y[ny][idx] = y[ny]
        return y














#         - model (optional): used in some function to store the
        # evaluation and avoid duplicating the calculations











def quadratic_1d(x, x_names, params, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    y = a*x^2 + b*x + c
    """
    a = params['a']
    b = params['b']
    c = params['c']
    x = x[x_names.index('x')]
    y = a*x**2 + b*x + c
    y = y[np.newaxis, np.newaxis]
    return y, None, None, None


def gaussian_1d(x, x_names, params, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    y = exp(-(x-mean^2)/std/2)
    """
    mean = params['mean']
    std = params['std']
    x = x[x_names.index('x')]
    y = np.exp(-(x-mean)**2./std**2./2.)
    y = y[np.newaxis, np.newaxis]
    return y, None, None, None


# 2D functions

def linear_2d(x, x_names, params, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    y = a*x1 + b*x2 + c
    """
    a = params['a']
    b = params['b']
    c = params['c']
    x1 = x[x_names.index('x1')]
    x2 = x[x_names.index('x2')]
    y = a*x1 + b*x2 + c
    y = y[np.newaxis, np.newaxis]
    return y, None, None, None


def quadratic_2d(x, x_names, params, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    y = a*x1^2 + b*x2^2 + c*x1*x2 + d*x1 + e*x2 + f
    """
    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']
    e = params['e']
    f = params['f']
    x1 = x[x_names.index('x1')]
    x2 = x[x_names.index('x2')]
    y = a*x1**2. + b*x2**2. + c*x1*x2 + d*x1 + e*x2 + f
    y = y[np.newaxis, np.newaxis]
    return y, None, None, None


# Cobaya loglikelihoods

def cobaya_loglike(x, x_names, params, model=None, extra_args=None, **kwargs):
    """
    Arguments:
    - x: array of input data (one sample)
    - x_names: list of names for each x element
    - params: dictionary of the arguments needed
    - model (optional): used in some function to store the
      evaluation and avoid duplicating the calculations
    - extra_args (optional): if the function called needs
      other arguments put them here
    Output:
    - y: list of arrays of output data (one for each element of y_fnames)
    - y_names: list of list of names for yeach y and y_names
    - y_fnames: list of file names to divide the output
    - model: if necessary to propagate model to the following steps

    """
    import cobaya
    # Merge dictionaries of params for cobaya
    if extra_args:
        cobaya_params = {'params': params} | extra_args
    else:
        cobaya_params = params
    # Get y_names
    y_names = list(cobaya_params['likelihood'].keys())
    # Define model
    if model is None:
        model = cobaya.model.get_model(cobaya_params)
    # Each sample should be a dictionary
    sampled_params = dict(zip(x_names, x))
    # Get loglike
    loglikes = model.loglikes(sampled_params, as_dict=True)[0]
    # Get y array
    y = np.array([loglikes[b] for b in y_names])
    # Add total loglike
    y = np.hstack((y, np.sum(y, axis=0)))
    y_names.append('tot_loglike')
    # Add total logprior
    y = np.hstack((y, model.logprior(sampled_params)))
    y_names.append('logprior')
    # Add total logposterior
    y = np.hstack((y, model.logpost(sampled_params)))
    y_names.append('logpost')
    # Replace nans with infinities
    y = np.nan_to_num(y, nan=-np.inf)
    y = y[np.newaxis]
    return y, y_names, None, model


