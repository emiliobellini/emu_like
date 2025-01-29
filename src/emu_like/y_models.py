"""
.. module:: y_models

:Synopsis: List of models to get y from x.
:Author: Emilio Bellini

Collection of functions that can be used to sample y
from x. Each one is stored as a Class inheriting
from the base Class YModel.
If you want to implement a new function, create
a new Class inheriting from YModel.
Add its name in the choose_one static method,
create the get_x method and adapt its other
methods and attributes to your needs.
"""

import numpy as np
from . import io as io
from . import defaults as de
from .spectra import Spectra
from .x_samplers import XSampler


# Base function
class YModel(object):
    """
    Base class YModel.
    """

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
                        if XSampler._is_varying(self.params, x)]
        return

    @staticmethod
    def choose_one(
        name,
        params,
        outputs,
        n_samples,
        verbose=False,
        **kwargs):
        """
        Main function to get the correct model for y.

        Arguments:
        - name (str): name of the model;
        - params (dict): dictionary of parameters;
        - outputs (dict): some of the models redirect the
          output to multiple datasets;
        - n_samples (int): number of samples;
        - verbose (bool, default: False): verbosity;
        - kwargs: specific arguments needed by each model.

        Return:
        - YModel (object): based on its name, get
          the correct sampling function and initialize it.
        """
        if name == 'linear_1d':
            return Linear1D(params, n_samples, verbose=verbose, **kwargs)
        elif name == 'quadratic_1d':
            return Quadratic1D(params, n_samples, verbose=verbose, **kwargs)
        elif name == 'gaussian_1d':
            return Gaussian1D(params, n_samples, verbose=verbose, **kwargs)
        elif name == 'linear_2d':
            return Linear2D(params, n_samples, verbose=verbose, **kwargs)
        elif name == 'quadratic_2d':
            return Quadratic2D(params, n_samples, verbose=verbose, **kwargs)
        elif name == 'cobaya_loglike':
            return CobayaLoglike(params, n_samples, verbose=verbose, **kwargs)
        elif name == 'class_spectra':
            return ClassSpectra(params, n_samples, outputs,
                                verbose=verbose, **kwargs)
        else:
            raise Exception('YModel not recognised!')

    def get_y_ranges(self):
        """
        Get y_ranges.
        """
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
        if self.y == []:
            raise Exception('Empty y arrays! Use get_y '
                            'or evaluate to generate them first.')

        self.n_y = [y.shape[1] for y in self.y]
        return self.n_y

    def get_y_names(self):
        """
        Get y_names.
        """
        for n_y in self.n_y:
            self.y_names.append(['y_{}'.format(y) for y in range(n_y)])

        return self.y_names

    def get_y_headers(self):
        """
        Get y_headers.
        """
        if self.y_names == []:
            self.get_y_names()
        
        self.y_headers = ['\t'.join(y_names) for y_names in self.y_names]
        return self.y_headers

    def get_y_fnames(self):
        """
        Get y_fnames.
        """
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

class Linear1D(YModel):
    """
    1D linear function

    y = a*x + b
    """

    def __init__(self, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Linear1D model.')

        YModel.__init__(self, params, n_samples, **kwargs)

        # Fix known properties of the function
        self.n_y = [1]
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]

        return

    def evaluate(self, x, idx, **kwargs):
        """
        Arguments:
        - x: 1D array of input data (one sample);
        - idx (int): row of x in the full sample;
        Output:
        - y: 1D array of output data (one sample).

        """
        a = self.args['a']
        b = self.args['b']
        x = x[self.x_names.index('x')]
        y = a*x + b

        # Adjust dimensions
        y = y[np.newaxis]
        # Store in self
        self.y[0][idx] = y
        return [y[np.newaxis]]


class Quadratic1D(YModel):
    """
    1D quadratic function

    y = a*x^2 + b*x + c
    """

    def __init__(self, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Quadratic1D model.')

        YModel.__init__(self, params, n_samples, **kwargs)

        # Fix known properties of the function
        self.n_y = [1]
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]

        return

    def evaluate(self, x, idx, **kwargs):
        """
        Arguments:
        - x: 1D array of input data (one sample);
        - idx (int): row of x in the full sample;
        Output:
        - y: 1D array of output data (one sample).

        """
        a = self.args['a']
        b = self.args['b']
        c = self.args['c']
        x = x[self.x_names.index('x')]
        y = a*x**2 + b*x + c

        # Adjust dimensions
        y = y[np.newaxis]
        # Store in self
        self.y[0][idx] = y
        return [y[np.newaxis]]


class Gaussian1D(YModel):
    """
    1D gaussian function

    y = exp(-(x-mean^2)/std/2)
    """

    def __init__(self, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Gaussian1D model.')

        YModel.__init__(self, params, n_samples, **kwargs)

        # Fix known properties of the function
        self.n_y = [1]
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]

        return

    def evaluate(self, x, idx, **kwargs):
        """
        Arguments:
        - x: 1D array of input data (one sample);
        - idx (int): row of x in the full sample;
        Output:
        - y: 1D array of output data (one sample).

        """
        mean = self.args['mean']
        std = self.args['std']
        x = x[self.x_names.index('x')]
        y = np.exp(-(x-mean)**2./std**2./2.)

        # Adjust dimensions
        y = y[np.newaxis]
        # Store in self
        self.y[0][idx] = y
        return [y[np.newaxis]]


# 2D functions

class Linear2D(YModel):
    """
    2D linear function

    y = a*x1 + b*x2 + c
    """

    def __init__(self, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Linear2D model.')

        YModel.__init__(self, params, n_samples, **kwargs)

        # Fix known properties of the function
        self.n_y = [1]
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]

        return

    def evaluate(self, x, idx, **kwargs):
        """
        Arguments:
        - x: 2D array of input data (one sample);
        - idx (int): row of x in the full sample;
        Output:
        - y: 1D array of output data (one sample).

        """
        a = self.args['a']
        b = self.args['b']
        c = self.args['c']
        x1 = x[self.x_names.index('x1')]
        x2 = x[self.x_names.index('x2')]
        y = a*x1 + b*x2 + c

        # Adjust dimensions
        y = y[np.newaxis]
        # Store in self
        self.y[0][idx] = y
        return [y[np.newaxis]]


class Quadratic2D(YModel):
    """
    2D quadratic function

    y = a*x1^2 + b*x2^2 + c*x1*x2 + d*x1 + e*x2 + f
    """

    def __init__(self, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Quadratic2D model.')

        YModel.__init__(self, params, n_samples, **kwargs)

        # Fix known properties of the function
        self.n_y = [1]
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]

        return

    def evaluate(self, x, idx, **kwargs):
        """
        Arguments:
        - x: 2D array of input data (one sample);
        - idx (int): row of x in the full sample;
        Output:
        - y: 1D array of output data (one sample).

        """
        a = self.args['a']
        b = self.args['b']
        c = self.args['c']
        d = self.args['d']
        e = self.args['e']
        f = self.args['f']
        x1 = x[self.x_names.index('x1')]
        x2 = x[self.x_names.index('x2')]
        y = a*x1**2. + b*x2**2. + c*x1*x2 + d*x1 + e*x2 + f

        # Adjust dimensions
        y = y[np.newaxis]
        # Store in self
        self.y[0][idx] = y
        return [y[np.newaxis]]


# Cobaya loglikelihoods

class CobayaLoglike(YModel):
    """
    Log-likelihoods from Cobaya.
    """

    def __init__(self, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing CobayaLoglike model.')

        YModel.__init__(self, params, n_samples, **kwargs)

        # Init Cobaya
        import cobaya
        
        # Cobaya parameters
        self.cobaya_params = {'params': params} | kwargs

        # Define model
        self.model = cobaya.model.get_model(self.cobaya_params)

        # Fix known properties of the function
        self.n_y = [len(self.cobaya_params['likelihood'].keys()) + 3]
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]
        return

    def get_y_names(self):
        """
        Get y_names.
        """
        y_names = list(self.cobaya_params['likelihood'].keys())
        y_names.append('tot_loglike')
        y_names.append('logprior')
        y_names.append('logpost')

        self.y_names = [y_names]

        return self.y_names

    def evaluate(self, x, idx, **kwargs):
        """
        Arguments:
        - x: 1D array of input data (one sample);
        - idx (int): row of x in the full sample;
        Output:
        - y: 1D array of output data (one sample).

        """

        # Each sample should be a dictionary
        sampled_params = dict(zip(self.x_names, x))

        # Get loglike
        y_dict = self.model.loglikes(sampled_params, as_dict=True)[0]
        # Add total loglike
        y_dict['tot_loglike'] = self.model.loglike(
            sampled_params, return_derived=False)
        # Add total logprior
        y_dict['logprior'] = self.model.logprior(sampled_params)
        # Add total logposterior
        y_dict['logpost'] = self.model.logpost(sampled_params)

        # Get y array
        y = np.array([y_dict[b] for b in self.y_names[0]])

        # Replace nans with infinities
        y = np.nan_to_num(y, nan=-np.inf)

        # Store in self
        self.y[0][idx] = y
        return [y[np.newaxis]]


# ClassSpectra

class ClassSpectra(YModel):
    """
    Power spectra from Class.
    """

    def __init__(self, params, n_samples, outputs, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing ClassSpectra model.')

        YModel.__init__(self, params, n_samples, **kwargs)
        self.outputs = outputs

        # Init classy
        import classy
        self.classy = classy
        self.cosmo = classy.Class()
        if verbose:
            io.print_level(1, 'Loading classy from {}'.format(classy.__file__))

        # Initialise spectra
        self.spectra = Spectra(outputs)

        # Build parameter dictionary
        var = {nm: None for nm in self.x_names}
        self.class_params = self.args | self.spectra.get_class_params() | var

        # Fix known properties of the function
        self.n_y = self.get_n_y()
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]

        return

    def get_n_y(self):
        """
        Get n_y.
        """
        self.n_y = self.spectra.get_n_vecs()
        return self.n_y

    def get_y_names(self):
        """
        Get y_names.
        """
        self.y_names = self.spectra.get_names()
        return self.y_names

    def get_y_headers(self):
        """
        Get y_headers.
        """
        self.y_headers = self.spectra.get_headers()
        return self.y_headers

    def get_y_fnames(self):
        """
        Get y_fnames.
        """
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
