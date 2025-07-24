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
import os
import scipy.interpolate as interp
from . import io as io
from . import defaults as de
from .spectra import Spectra
from .x_samplers import XSampler


# Base function
class YModel(object):
    """
    Base class YModel.
    """

    def __init__(self, name, params, n_samples, **kwargs):
        self.name = name
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
        self.outputs = None

        # Derive varying parameters
        self.x_names = [x for x in self.params
                        if XSampler._is_varying(self.params, x)]
        return

    def __getitem__(self, item):
        if item is None or item == 0:
            return self
        else:
            raise TypeError('Base YModel object is not subscriptable. Implement your own rules!')

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
            return Linear1D(name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'quadratic_1d':
            return Quadratic1D(name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'gaussian_1d':
            return Gaussian1D(name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'linear_2d':
            return Linear2D(name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'quadratic_2d':
            return Quadratic2D(name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'cobaya_loglike':
            return CobayaLoglike(name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'class_spectra':
            return ClassSpectra(
                name, params, n_samples, outputs, verbose=verbose, **kwargs)
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
        self.y_fnames = [de.file_names['y_data']['name'].format('')]
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

    def save(self, fname=None, root=None, verbose=False):
        """
        Placeholder in case we want to save something.
        """
        return

    def load(self, fname=None, root=None, verbose=False):
        """
        Placeholder in case we want to load something.
        """
        return

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

    def __init__(self, name, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Linear1D model.')

        YModel.__init__(self, name, params, n_samples, **kwargs)

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

    def __init__(self, name, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Quadratic1D model.')

        YModel.__init__(self, name, params, n_samples, **kwargs)

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

    def __init__(self, name, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Gaussian1D model.')

        YModel.__init__(self, name, params, n_samples, **kwargs)

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

    def __init__(self, name, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Linear2D model.')

        YModel.__init__(self, name, params, n_samples, **kwargs)

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

    def __init__(self, name, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing Quadratic2D model.')

        YModel.__init__(self, name, params, n_samples, **kwargs)

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

    def __init__(self, name, params, n_samples, verbose=False, **kwargs):
        if verbose:
            io.info('Initializing CobayaLoglike model.')

        YModel.__init__(self, name, params, n_samples, **kwargs)

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

    def __init__(
            self,
            name=None,
            params=None,
            n_samples=None,
            outputs=None,
            verbose=False,
            **kwargs):

        # Decide wether to fully initialize (it calls Class to
        # compute the reference spectra, which takes some time) or not.        
        skip_init = False
        if params is None or n_samples is None or outputs is None:
            skip_init = True

        if skip_init:
            if verbose:
                io.info('Skipping initializazion of ClassSpectra model.')
            return

        if verbose:
            io.info('Initializing ClassSpectra model.')

        YModel.__init__(self, name, params, n_samples, **kwargs)
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

        # Compute reference spectra (this is used to take the ratio if requested)
        # 1) Infer the maximum redshift
        if any([sp.is_pk for sp in self.spectra]):
            z_max = {'z_max_pk': self._get_z_max()}
        else:
            z_max = {}
        # 2) Compute Class
        cosmo_ref = self.classy.Class()
        cosmo_ref.set(de.cosmo_params | self.spectra.get_class_params() | z_max)
        cosmo_ref.compute()
        # 3) Compute all the spectra
        self.y_ref = [sp.get(cosmo_ref, z=None)[np.newaxis] for sp in self.spectra]
        # 4) Replace with ones if we do not take ratio
        for nsp, sp in enumerate(self.spectra):
            if not sp.ratio:
                if sp.is_pk:
                    self.y_ref[nsp] = np.ones_like(self.y_ref[nsp][:,:,0])
                else:
                    self.y_ref[nsp] = np.ones_like(self.y_ref[nsp])
        # 5) Store the redshift values at which all Pk have been computed
        self.z_array = self._get_z_array(self.spectra)
        # 6) Store the k modes values at which all Pk have been computed
        self.k_ranges = [None for sp in self.spectra]
        for nsp, sp in enumerate(self.spectra):
            try:
                self.k_ranges[nsp] = sp.k_range
            except AttributeError:
                pass
        # 7) Store the ell modes values at which all Cell have been computed
        self.ell_ranges = [None for sp in self.spectra]
        for nsp, sp in enumerate(self.spectra):
            try:
                self.ell_ranges[nsp] = sp.ell_range
            except AttributeError:
                pass
        return

    def __getitem__(self, item):
        if item is None:
            return self
        
        # Get correct name and index for spectrum
        name = self.spectra[item].name
        idx = self.spectra._get_idx_from_name(name)
        
        oneclassspectrum = ClassSpectra()
        # Fix relevant attributes
        oneclassspectrum.name = self.name
        oneclassspectrum.classy = self.classy
        oneclassspectrum.cosmo = self.cosmo
        oneclassspectrum.args = self.args
        oneclassspectrum.class_params = self.class_params
        oneclassspectrum.n_samples = self.n_samples
        oneclassspectrum.params = self.params
        oneclassspectrum.outputs = {name: self.outputs[name]}
        oneclassspectrum.x_names = self.x_names
        oneclassspectrum.y = [self.y[idx]]
        oneclassspectrum.y_fnames = [self.y_fnames[idx]]
        oneclassspectrum.y_headers = [self.y_headers[idx]]
        oneclassspectrum.y_names = [self.y_names[idx]]
        oneclassspectrum.y_ranges = [self.y_ranges[idx]]
        oneclassspectrum.y_ref = [self.y_ref[idx]]
        oneclassspectrum.spectra = Spectra([self.spectra[idx]])
        if self.spectra[idx].is_pk:
            oneclassspectrum.z_array = self.z_array
            oneclassspectrum.k_ranges = [self.k_ranges[idx]]
        elif self.spectra[idx].is_cl:
            oneclassspectrum.ell_ranges = [self.ell_ranges[idx]]
        return oneclassspectrum

    def _get_z_max(self):
        z_max = 0.1
        try:
            z_max = max(z_max, self.args['z_pk'])
        except KeyError:
            pass
        try:
            z_max = max(z_max, self.args['z_max_pk'])
        except KeyError:
            pass
        try:
            z_max = max(z_max, self.params['z_pk']['prior']['max'])
        except KeyError:
            pass
        return z_max

    def _get_z_array(self, spectra):
        z_array = None
        for sp in spectra:
            if sp.is_pk:
                z_array = sp.z_array
        return z_array

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
        self.y_names = self.spectra.get_y_names()
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
        - idx (int): row of x in the full sample;
        Output:
        - y: 1D array of output data (one sample).

        """

        # Update parameter dictionary
        for npar, par in enumerate(self.x_names):
            self.class_params[par] = x[npar]

        # Update z_max_pk if needed and get z
        self.class_params['z_max_pk'] = 0.1
        try:
            self.class_params['z_max_pk'] = max(
                self.class_params['z_pk'], self.class_params['z_max_pk'])
            z = self.class_params['z_pk']
        except KeyError:
            z = 0.

        try:
            # Compute class
            self.cosmo.set(self.class_params)
            self.cosmo.compute()

            y = [sp.get(self.cosmo, z=z)[np.newaxis] for sp in self.spectra]

        except self.classy.CosmoComputationError:
            # Fill with nans if error
            y = [np.full((n_y,), np.nan)[np.newaxis] for n_y in self.n_y]

        # Take the ratio
        for nsp, sp in enumerate(self.spectra):
            if sp.ratio:
                # Get y_ref at the correct z
                if sp.is_pk:
                    den = interp.make_splrep(self.z_array, self.y_ref[nsp].T, s=0)(z).T
                else:
                    den = self.y_ref[nsp]
                y[nsp] = y[nsp]/den

        # Store in self
        for ny in range(len(self.n_y)):
            self.y[ny][idx] = y[ny]

        return y

    def save(self, fname=None, root=None, verbose=False):
        """
        Save reference spectra.
        """
        if fname is None:
            fname = de.file_names['spectra_factor']['name']
        if root is None:
            path = fname
        else:
            path = os.path.join(root, fname)
        if verbose:
            io.info('Saving reference spectra to {}'.format(path))
        io.Folder(os.path.dirname(path)).create()

        fits = io.FitsFile(path=path)


        for nsp, sp in enumerate(self.spectra):
            # Write spectra
            fits.write(self.y_ref[nsp], sp.name, type='image')
            if sp.is_pk:
                # Write z_array
                fits.write(self.z_array, 'z_array', type='image')
                # Write k_range
                fits.write(self.k_ranges[nsp], 'k_range_{}'.format(sp.name), type='image')
            elif sp.is_cl:
                # Write ell_range
                fits.write(self.ell_ranges[nsp], 'ell_range_{}'.format(sp.name), type='image')
        
        return

    def load(self, fname=None, root=None, verbose=False):
        """
        Load reference spectra.
        """
        if fname is None:
            fname = de.file_names['spectra_factor']['name']
        if root is None:
            path = fname
        else:
            path = os.path.join(root, fname)
        if verbose:
            io.print_level(1, 'Loading reference spectra from {}'.format(path))

        fits = io.FitsFile(path=path)

        # Read spectra
        self.y_ref = [fits.read_key(sp.name) for sp in self.spectra]

        # Read z_array, k_ranges, ell_ranges
        self.z_array = None
        self.k_ranges = [None for sp in self.spectra]
        self.ell_ranges = [None for sp in self.spectra]
        for nsp, sp in enumerate(self.spectra):
            if sp.is_pk:
                self.z_array = fits.read_key('z_array')
                self.k_ranges[nsp] = fits.read_key('k_range_{}'.format(sp.name))
            elif sp.is_cl:
                self.ell_ranges[nsp] = fits.read_key('ell_range_{}'.format(sp.name))
        
        return
