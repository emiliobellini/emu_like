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

import matplotlib.pyplot as plt
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

    def plot(self, emu, data, path=None):
        """
        Placeholder for model specific plots. Arguments:
        - emu (emu_like.FFNNEmu object)
        - path (str, default:None): path where to save the plots.
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
        self.class_params = self.args | var

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
        cosmo_ref.set(de.cosmo_params | z_max)
        cosmo_ref.compute()
        # 3) Compute all the spectra
        self.y_ref = [sp.get(cosmo_ref, z=None)[np.newaxis] for sp in self.spectra]
        # 4) Replace with ones if we do not take ratio
        for nsp, sp in enumerate(self.spectra):
            if not sp.ratio:
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
        z = 0
        if any([sp.is_pk for sp in self.spectra]):
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
        except self.classy.CosmoSevereError:
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

    def save(self, fname, root=None, verbose=False):
        """
        Save reference spectra.
        """
        if root is None:
            path = fname
        else:
            path = os.path.join(root, fname)
        if verbose:
            io.info('Saving reference spectra to {}'.format(path))

        fits = io.FitsFile(path)


        is_pk = False
        for nsp, sp in enumerate(self.spectra):
            # Write spectra
            fits.write(
                data=self.y_ref[nsp],
                header=None,
                name='ref_{}'.format(sp.name),
            )
            if sp.is_pk:
                # Write k_range
                fits.write(
                    data=self.k_ranges[nsp],
                    header=None,
                    name='k_range_{}'.format(sp.name),
                )
                is_pk = True
            elif sp.is_cl:
                # Write ell_range
                fits.write(
                    data=self.ell_ranges[nsp],
                    header=None,
                    name='ell_range_{}'.format(sp.name),
                )
        
        if is_pk:
            # Write z_array
            fits.write(
                data=self.z_array,
                header=None,
                name='z_array',
            )
        
        return

    def load(self, fname, root=None, verbose=False):
        """
        Load reference spectra.
        """
        if root is None:
            path = fname
        else:
            path = os.path.join(root, fname)
        if verbose:
            io.print_level(1, 'Loading reference spectra from {}'.format(path))

        fits = io.FitsFile(fname=path)

        is_pk = False
        self.y_ref = []
        self.k_ranges = []
        self.ell_ranges = []
        for sp in self.spectra:
            # Read spectra
            self.y_ref.append(fits.get_data('ref_{}'.format(sp.name)))
            if sp.is_pk:
                # Read k_range
                self.k_ranges.append(fits.get_data('k_range_{}'.format(sp.name)))
                self.ell_ranges.append(None)
                is_pk = True
            elif sp.is_cl:
                # Read ell_range
                self.k_ranges.append(None)
                self.ell_ranges.append(fits.get_data('ell_range_{}'.format(sp.name)))
        
        if is_pk:
            # read z_array
            self.z_array = fits.get_data('z_array')

        return

    def plot(self, emu, data, path=None):
        """
        Plot single spectrum. Arguments:
        - emu (emu_like.FFNNEmu object);
        - data (src.emu_like.datasets.Dataset object);
        - path (str, default:None): path where to save the plots.
        """

        def get_y(emu, x):
            return np.array([emu.eval(xp) for xp in x])

        def get_diff(emu, x, y):
            y_emu = get_y(emu, x)
            diff =  y_emu/y-1
            return diff

        def get_idx_max_diff(diff):
            idx = np.argmax(np.mean(diff**2., axis=1))
            return idx

        def get_ref(emu, data, idx_max):
            if emu.y_model.spectra[0].is_pk:
                z = data.x[idx_max, 0]
                ref = interp.make_splrep(emu.y_model.z_array, emu.y_model.y_ref[0][0].T, s=0)(z)
            else:
                ref = emu.y_model.y_ref[0][0]
            return ref

        # Spectrum name
        spectrum = self.spectra.names[0]

        fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(6., 20.), sharex=True, squeeze=False)

        ax[0, 0].set_ylabel('rel. diff. [%] -- Training set')
        ax[1, 0].set_ylabel('rel. diff. [%] -- Validation set')
        ax[2, 0].set_ylabel('rel. diff. [%] -- Worst fit')
        ax[3, 0].set_ylabel('{}/{}(ref) -- Worst fit'.format(spectrum, spectrum))
        ax[4, 0].set_ylabel('{} -- Worst fit'.format(spectrum))

        # x variable
        if self.spectra[0].is_pk:
            x = emu.y_model.k_ranges[0]
            ax[-1, 0].set_xlabel('k [h/Mpc]')
            ax[-1, 0].set_xscale('log')
            ax[-1, 0].set_yscale('log')
        elif self.spectra[0].is_cl:
            x = emu.y_model.ell_ranges[0]
            ax[-1, 0].set_xlabel('ell')
            ax[-1, 0].set_xscale('linear')
            ax[-1, 0].set_yscale('linear')

        # Training set
        x_train = emu.x_scaler.inverse_transform(emu.x_pca.inverse_transform(data.x_train))
        y_train = emu.y_scaler.inverse_transform(emu.y_pca.inverse_transform(data.y_train))
        ax[0, 0].plot(x, get_diff(emu, x_train, y_train).T*100., 'k-', alpha=0.1)

        # Validation set
        x_test = emu.x_scaler.inverse_transform(emu.x_pca.inverse_transform(data.x_test))
        y_test = emu.y_scaler.inverse_transform(emu.y_pca.inverse_transform(data.y_test))
        ax[1, 0].plot(x, get_diff(emu, x_test, y_test).T*100., 'k-', alpha=0.1)

        diff = get_diff(emu, data.x, data.y)
        idx_max = get_idx_max_diff(diff)
        y_emu_max = get_y(emu, data.x)[idx_max]
        ref_max = get_ref(emu, data, idx_max)

        # Worst fit, rel diff
        ax[2, 0].plot(x, diff[idx_max]*100., 'k-')

        # Worst fit, P/P_ref
        ax[3, 0].plot(x, y_emu_max, label='Emulated')
        ax[3, 0].plot(x, data.y[idx_max], '--', label='True')
        ax[3, 0].legend()

        # Worst fit, P
        ax[4, 0].plot(x, ref_max*y_emu_max)
        ax[4, 0].plot(x, ref_max*data.y[idx_max], '--')

        plt.subplots_adjust(bottom=0.15, hspace=0.05, wspace=0.15)
        if path:
            plt.savefig(os.path.join(path, 'accuracy_emulator.png'))
        plt.show()
        plt.close()

        return
