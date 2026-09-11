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

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as interp
from . import io as io
try:
    import classy  # type: ignore
except ImportError:  # classy is optional for dataset-based workflows
    classy = None  # type: ignore

from .spectra import Spectra, GrowthRate
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
        self.outputs = None
        self.y_keys = ['y_data']

        # Derive varying parameters
        self.x_names = [x for x in self.params
                        if XSampler._is_varying(self.params, x)]
        return

    def __getitem__(self, item):
        if item is None or item == 0:
            return self
        else:
            raise TypeError('Base YModel object is not subscriptable.'
                            'Implement your own rules!')

    @staticmethod
    def choose_one(name, params, outputs, n_samples, verbose=False, **kwargs):
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
            return Linear1D(
                name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'quadratic_1d':
            return Quadratic1D(
                name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'gaussian_1d':
            return Gaussian1D(
                name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'linear_2d':
            return Linear2D(
                name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'quadratic_2d':
            return Quadratic2D(
                name, params, n_samples, verbose=verbose, **kwargs)
        elif name == 'cobaya_loglike':
            return CobayaLoglike(
                name, params, n_samples, verbose=verbose, **kwargs)
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
        elif isinstance(self.y, list):
            self.n_y = [y.shape[1] for y in self.y]
        else:
            self.n_y = [self.y.shape[1]]
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

        self.y_headers = [{'y_names': y_names} for y_names in self.y_names]
        return self.y_headers

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

    def plot(self, emu, data, max_data=1e4, path=None):
        """
        Placeholder for model specific plots.
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

        # Cosmo (Planck 2018 bestfit,
        # Table 1 of https://arxiv.org/pdf/1807.06209)
        self.ref_params = {
            'h': 0.6732,
            'Omega_m': 0.3158,
            'Omega_b': 0.0494,
            'tau_reio': 0.0543,
            'ln_A_s_1e10': 3.044,
            'n_s': 0.966,
            'YHe': 0.24,
            'N_ur': 0.,
            'N_ncdm': 1,
            'deg_ncdm': 3,
            'm_ncdm': 0.02,
            # Precision parameters
            'k_per_decade_for_pk': 40,
            'k_per_decade_for_bao': 80,
            'l_logstep': 1.026,
            'l_linstep': 25,
            'perturbations_sampling_stepsize': 0.02,
            'l_switch_limber': 20,
            'accurate_lensing': 1,
            'delta_l_max': 1000,
            'output': 'tCl, dTk, pCl, lCl, mPk',
            'l_max_scalars': 3000,
            'lensing': 'yes',
            'P_k_max_h/Mpc': 50.0,
            'k_pivot': 0.05,
            'modes': 's',
        }

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

        # Initialise spectra metadata even if classy is unavailable.
        self.spectra = Spectra(outputs)
        self.y_keys = self.spectra.names

        # Build parameter dictionary
        var = {nm: None for nm in self.x_names}
        self.class_params = self.args | var

        # Fix known properties of the function
        self.n_y = self.get_n_y()
        self.y = [np.zeros((self.n_samples, n_y)) for n_y in self.n_y]
        self.y_names = self.get_y_names()
        self.y_headers = self.get_y_headers()

        # Default placeholders for reference spectra and sampling grids
        n_specs = len(self.n_y)
        self.y_ref = [np.ones((1, n_y)) for n_y in self.n_y]
        self.k_ranges = [None] * n_specs
        self.ell_ranges = [None] * n_specs
        self.z_array = None

        if classy is None:
            if verbose:
                io.info('classy not available; ClassSpectra running in '
                        'read-only mode.')
            self.classy = None
            self.cosmo = None
            return

        # Init classy
        self.classy = classy
        self.cosmo = classy.Class()
        if verbose:
            io.print_level(1, 'Loading classy from {}'.format(classy.__file__))

        # Compute reference spectra (used to take the ratio if requested)
        # 1) Infer the maximum redshift
        if any([sp.is_pk for sp in self.spectra]):
            reference_z_max = self._get_z_max()
            z_max = {'z_max_pk': self._required_z_max(reference_z_max)}
        else:
            z_max = {}
        # 2) Compute Class
        cosmo_ref = self.classy.Class()
        self.ref_params = self.ref_params | z_max
        cosmo_ref.set(self.ref_params)
        cosmo_ref.compute()
        # 3) Compute all the spectra
        self.y_ref = [sp.get(cosmo_ref, z=None)[np.newaxis]
                      for sp in self.spectra]
        if z_max:
            self._restrict_reference_grid(cosmo_ref, reference_z_max)
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

        # Base YModel attributes
        oneclassspectrum.name = self.name
        oneclassspectrum.params = copy.deepcopy(self.params)
        oneclassspectrum.args = copy.deepcopy(self.args)
        oneclassspectrum.n_samples = self.n_samples
        oneclassspectrum.y = [self.y[idx].copy()]
        oneclassspectrum.n_y = [self.n_y[idx]]
        oneclassspectrum.y_names = [copy.deepcopy(self.y_names[idx])]
        oneclassspectrum.y_headers = [copy.deepcopy(self.y_headers[idx])]
        oneclassspectrum.outputs = {
            name: copy.deepcopy(self.outputs[name])}
        oneclassspectrum.x_names = list(self.x_names)

        # ClassSpectra metadata
        oneclassspectrum.spectra = Spectra(oneclassspectrum.outputs)
        oneclassspectrum.y_keys = list(oneclassspectrum.spectra.names)
        oneclassspectrum.ref_params = copy.deepcopy(self.ref_params)
        oneclassspectrum.class_params = (
            oneclassspectrum.args
            | {parameter: None for parameter in oneclassspectrum.x_names})
        oneclassspectrum.y_ref = [self.y_ref[idx].copy()]
        oneclassspectrum.z_array = (
            None if self.z_array is None else self.z_array.copy())
        oneclassspectrum.k_ranges = [
            copy.deepcopy(self.k_ranges[idx])]
        oneclassspectrum.ell_ranges = [
            copy.deepcopy(self.ell_ranges[idx])]

        # Runtime CLASS objects must not be shared between models.
        oneclassspectrum.classy = self.classy
        oneclassspectrum.cosmo = (
            None if self.classy is None else self.classy.Class())

        return oneclassspectrum

    @staticmethod
    def _attributes_equal(left, right):
        """Compare nested model metadata, including NumPy arrays."""
        if isinstance(left, dict) and isinstance(right, dict):
            return (
                left.keys() == right.keys()
                and all(ClassSpectra._attributes_equal(left[key], right[key])
                        for key in left))
        if isinstance(left, (list, tuple)):
            return (
                isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(ClassSpectra._attributes_equal(value_left, value_right)
                        for value_left, value_right in zip(left, right)))
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return np.array_equal(left, right)
        return left == right

    @staticmethod
    def _join_dicts(
            dictionaries,
            reducers=None,
            optional_paths=(),
            label='metadata',
            path=()):
        """Recursively validate dictionaries and reduce selected fields."""
        reducers = reducers or {}
        if not dictionaries or not all(
                isinstance(dictionary, dict) for dictionary in dictionaries):
            raise ValueError('All {} must be dictionaries'.format(label))

        keys = list(dictionaries[0])
        for dictionary in dictionaries[1:]:
            keys.extend(key for key in dictionary if key not in keys)

        joined = {}
        missing = object()
        for key in keys:
            field_path = path + (key,)
            values = [dictionary.get(key, missing)
                      for dictionary in dictionaries]
            is_optional = field_path in optional_paths
            if any(value is missing for value in values):
                if not is_optional:
                    raise ValueError(
                        '{} fields differ at {}'.format(
                            label, '.'.join(field_path)))
                values = [value for value in values if value is not missing]

            reducer = reducers.get(field_path)
            if reducer is not None:
                joined[key] = reducer(values)
            elif all(isinstance(value, dict) for value in values):
                joined[key] = ClassSpectra._join_dicts(
                    values,
                    reducers=reducers,
                    optional_paths=optional_paths,
                    label=label,
                    path=field_path)
            elif any(isinstance(value, dict) for value in values):
                raise ValueError(
                    '{} field {} has inconsistent types'.format(
                        label, '.'.join(field_path)))
            elif not all(ClassSpectra._attributes_equal(value, values[0])
                         for value in values[1:]):
                raise ValueError(
                    '{} field {} differs'.format(
                        label, '.'.join(field_path)))
            else:
                joined[key] = copy.deepcopy(values[0])
        return joined

    @staticmethod
    def _join_references(y_models, ref_params):
        """Select the widest reference grid and validate common values."""
        first = y_models[0]
        if not all(len(model.y_ref) == len(first.spectra.names)
                   for model in y_models):
            raise ValueError(
                'ClassSpectra models have inconsistent y_ref lengths')

        has_pk = any(spectrum.is_pk for spectrum in first.spectra)
        z_arrays = [model.z_array for model in y_models]
        if not has_pk or all(z_array is None for z_array in z_arrays):
            for model in y_models[1:]:
                if not all(np.allclose(
                        reference, candidate,
                        rtol=1.e-10, atol=1.e-12, equal_nan=True)
                        for reference, candidate
                        in zip(first.y_ref, model.y_ref)):
                    raise ValueError(
                        'ClassSpectra reference spectra differ')
            return None, copy.deepcopy(first.y_ref)

        if any(z_array is None for z_array in z_arrays):
            raise ValueError(
                'All ClassSpectra Pk models must define z_array')
        if any(np.asarray(z_array).ndim != 1 for z_array in z_arrays):
            raise ValueError('ClassSpectra z_array must be one-dimensional')

        target_z_max = ref_params.get('z_max_pk')
        candidates = [
            index for index, model in enumerate(y_models)
            if model.ref_params.get('z_max_pk') == target_z_max]
        if not candidates:
            candidates = list(range(len(y_models)))
        selected_index = max(
            candidates, key=lambda index: len(z_arrays[index]))
        selected = y_models[selected_index]
        selected_z = np.asarray(selected.z_array)
        if selected_z.size == 0:
            raise ValueError('ClassSpectra z_array can not be empty')
        # Stored reference outputs exclude derivative-only CLASS coverage.
        target_output_z_max = max(model._get_z_max() for model in y_models)
        if (np.max(selected_z) < target_output_z_max
                and not np.isclose(
                    np.max(selected_z), target_output_z_max,
                    rtol=1.e-12, atol=1.e-12)):
            raise ValueError(
                'ClassSpectra reference grid does not reach requested redshift')

        for model, z_array in zip(y_models, z_arrays):
            z_array = np.asarray(z_array)
            selected_indices = []
            for redshift in z_array:
                matches = np.flatnonzero(np.isclose(
                    selected_z, redshift, rtol=1.e-12, atol=1.e-12))
                if len(matches) != 1:
                    raise ValueError(
                        'ClassSpectra z_arrays are not nested consistently')
                selected_indices.append(matches[0])

            for output_index, spectrum in enumerate(first.spectra):
                reference = model.y_ref[output_index]
                selected_reference = selected.y_ref[output_index]
                if spectrum.is_pk:
                    cond1 = reference.ndim < 2
                    cond2 = reference.shape[-1] != len(z_array)
                    cond3 = selected_reference.ndim < 2
                    cond4 = selected_reference.shape[-1] != len(selected_z)
                    if (cond1 or cond2 or cond3 or cond4):
                        raise ValueError(
                            'ClassSpectra y_ref redshift dimension is '
                            'inconsistent with z_array')
                    selected_reference = np.take(
                        selected_reference, selected_indices, axis=-1)
                if not np.allclose(
                        reference, selected_reference,
                        rtol=1.e-10, atol=1.e-12, equal_nan=True):
                    raise ValueError(
                        'ClassSpectra reference spectra differ at common '
                        'redshifts')

        return selected_z.copy(), copy.deepcopy(selected.y_ref)

    @staticmethod
    def join(y_models):
        """Combine multiple compatible ClassSpectra instances."""

        if not y_models:
            raise ValueError('At least one ClassSpectra model is required')

        first = y_models[0]
        if not all(isinstance(model, ClassSpectra) for model in y_models):
            raise ValueError('All models must be ClassSpectra instances')
        if not all(model.classy is first.classy for model in y_models[1:]):
            raise ValueError(
                'ClassSpectra models use different classy runtimes')

        # Attributes defining the model and output representation must match.
        common_attributes = (
            'name', 'x_names', 'outputs', 'y_keys', 'n_y', 'y_names',
            'y_headers', 'k_ranges', 'ell_ranges')
        for attribute in common_attributes:
            reference = getattr(first, attribute)
            if not all(ClassSpectra._attributes_equal(
                    getattr(model, attribute), reference)
                    for model in y_models[1:]):
                raise ValueError(
                    'ClassSpectra models can not be joined because {} differs'
                    ''.format(attribute))

        reference_spectra = [
            (type(spectrum), spectrum.name, spectrum.ratio)
            for spectrum in first.spectra]
        for model in y_models[1:]:
            model_spectra = [
                (type(spectrum), spectrum.name, spectrum.ratio)
                for spectrum in model.spectra]
            if model_spectra != reference_spectra:
                raise ValueError(
                    'ClassSpectra models can not be joined because spectra '
                    'differ')

        # Validate the arrays before aggregating them.
        for model in y_models:
            if len(model.y) != len(first.n_y):
                raise ValueError(
                    'ClassSpectra model has an inconsistent number of y '
                    'arrays')
            for index, (array, n_y) in enumerate(zip(model.y, first.n_y)):
                expected_shape = (model.n_samples, n_y)
                if array.shape != expected_shape:
                    raise ValueError(
                        'ClassSpectra y[{}] has shape {}, expected {}'
                        ''.format(index, array.shape, expected_shape))

        joined = ClassSpectra()

        # Attributes copied after equality validation.
        joined.name = first.name
        joined.x_names = copy.deepcopy(first.x_names)
        joined.outputs = copy.deepcopy(first.outputs)
        joined.y_keys = copy.deepcopy(first.y_keys)
        joined.n_y = copy.deepcopy(first.n_y)
        joined.y_names = copy.deepcopy(first.y_names)
        joined.y_headers = copy.deepcopy(first.y_headers)
        joined.k_ranges = copy.deepcopy(first.k_ranges)
        joined.ell_ranges = copy.deepcopy(first.ell_ranges)
        joined.spectra = Spectra(joined.outputs)

        # Attributes summed or stacked across models.
        joined.n_samples = sum(model.n_samples for model in y_models)
        joined.y = [
            np.vstack([model.y[index] for model in y_models])
            for index in range(len(joined.n_y))]

        # Common runtime dependency; a fresh cosmo object will be created
        # after the non-trivial metadata has been combined.
        joined.classy = first.classy
        joined.cosmo = None

        # Merge parameter definitions. Only prior bounds may differ.
        param_reducers = {}
        for parameter in first.params:
            param_reducers[(parameter, 'prior', 'min')] = min
            param_reducers[(parameter, 'prior', 'max')] = max
        joined.params = ClassSpectra._join_dicts(
            [model.params for model in y_models],
            reducers=param_reducers,
            label='ClassSpectra params')

        # Merge model arguments. z_max_pk controls only the calculation range;
        # all physical and precision arguments must match.
        joined.args = ClassSpectra._join_dicts(
            [model.args for model in y_models],
            reducers={('z_max_pk',): max},
            optional_paths=(('z_max_pk',),),
            label='ClassSpectra args')

        joined.class_params = (
            copy.deepcopy(joined.args)
            | {parameter: None for parameter in joined.x_names})

        # Merge reference CLASS parameters. z_max_pk is a calculation bound;
        # every other cosmological and precision setting must match.
        joined.ref_params = ClassSpectra._join_dicts(
            [model.ref_params for model in y_models],
            reducers={('z_max_pk',): max},
            optional_paths=(('z_max_pk',),),
            label='ClassSpectra ref_params')

        if any(spectrum.is_pk for spectrum in joined.spectra):
            joined.ref_params['z_max_pk'] = joined._required_z_max(
                joined._get_z_max(),
                configured_limit=joined.ref_params.get('z_max_pk', 0.1))

        joined.z_array, joined.y_ref = ClassSpectra._join_references(
            y_models, joined.ref_params)
        joined.cosmo = (
            None if joined.classy is None else joined.classy.Class())

        return joined

    def _required_z_max(self, z, configured_limit=None):
        """Plan coverage from this request, without inheriting earlier rows."""
        limit = (self.args.get('z_max_pk', 0.1)
                 if configured_limit is None else configured_limit)
        stencil_max = z
        if any(sp.name in ('fk_m', 'fk_cb') for sp in self.spectra):
            step = GrowthRate.derivative_step
            stencil_max = z + (2 * step if z < step else step)
        return max(0.1, limit, stencil_max)

    def _restrict_reference_grid(self, cosmo, z_max):
        """Store a common output grid, excluding derivative-only coverage."""
        common_z = None
        for index, sp in enumerate(self.spectra):
            if not sp.is_pk:
                continue
            native_z = np.asarray(sp.z_array)
            keep = native_z <= z_max
            output_z = native_z[keep]
            table = self.y_ref[index][..., keep]
            if not output_z.size or output_z[-1] < z_max:
                output_z = np.append(output_z, z_max)
                endpoint = np.asarray(sp.get(cosmo, z=z_max))[None, :, None]
                table = np.concatenate((table, endpoint), axis=-1)
            if common_z is not None and not np.array_equal(common_z, output_z):
                raise ValueError('Reference spectra must share a redshift grid')
            common_z = output_z
            sp.z_array = output_z.copy()
            self.y_ref[index] = table

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

        if self.classy is None or self.cosmo is None:
            raise RuntimeError(
                'classy is required to evaluate ClassSpectra outputs.')

        # Update parameter dictionary
        for npar, par in enumerate(self.x_names):
            self.class_params[par] = x[npar]

        # Update z_max_pk if needed and get z
        z = 0
        if any([sp.is_pk for sp in self.spectra]):
            z = self.class_params.get('z_pk', 0.)
            self.class_params['z_max_pk'] = self._required_z_max(z)

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
                    den = interp.make_splrep(
                        self.z_array, self.y_ref[nsp].T, s=0)(z).T
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
                name='ref_{}'.format(sp.name),
                data=self.y_ref[nsp],
                header=self.ref_params,
            )
            if sp.is_pk:
                # Write k_range
                fits.write(
                    name='k_range_{}'.format(sp.name),
                    data=self.k_ranges[nsp],
                    header=None,
                )
                is_pk = True
            elif sp.is_cl:
                # Write ell_range
                fits.write(
                    name='ell_range_{}'.format(sp.name),
                    data=self.ell_ranges[nsp],
                    header=None,
                )

        if is_pk:
            # Write z_array
            fits.write(
                name='z_array',
                data=self.z_array,
                header=None,
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
                self.k_ranges.append(
                    fits.get_data('k_range_{}'.format(sp.name)))
                self.ell_ranges.append(None)
                is_pk = True
            elif sp.is_cl:
                # Read ell_range
                self.k_ranges.append(None)
                self.ell_ranges.append(
                    fits.get_data('ell_range_{}'.format(sp.name)))

        if is_pk:
            # read z_array
            self.z_array = fits.get_data('z_array')

        return

    def plot(self, emu, data, max_data=1e4, path=None):
        """
        Plot single spectrum. Arguments:
        - emu (emu_like.FFNNEmu object);
        - data (src.emu_like.datasets.Dataset object);
        - max_data (float, default: 1e4): maximum number of samples to plot;
        - path (str, default:None): path where to save the plots.
        """

        def get_y(emu, x):
            return np.array([emu.eval(xp) for xp in x])

        def get_diff(emu, x, y):
            y_emu = get_y(emu, x)
            diff = y_emu/y-1
            return diff

        def get_idx_max_diff(diff):
            idx = np.argmax(np.mean(diff**2., axis=1))
            return idx

        def get_ref(emu, x, idx_max):
            if emu.y_model.spectra[0].is_pk:
                z = x[idx_max, 0]
                ref = interp.make_splrep(
                    emu.y_model.z_array, emu.y_model.y_ref[0][0].T, s=0)(z)
            else:
                ref = emu.y_model.y_ref[0][0]
            return ref

        # Spectrum name
        spectrum = self.spectra.names[0]

        fig, ax = plt.subplots(
            nrows=5, ncols=1, figsize=(6., 20.), sharex=True, squeeze=False)

        ax[0, 0].set_ylabel('rel. diff. [%] -- Training set')
        ax[1, 0].set_ylabel('rel. diff. [%] -- Validation set')
        ax[2, 0].set_ylabel('rel. diff. [%] -- Worst fit')
        ax[3, 0].set_ylabel('{}/{}(ref) -- Worst fit'.format(
            spectrum, spectrum))
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
        if data.x_train.shape[0] > max_data:
            rng = np.random.default_rng()
            mask = rng.choice(
                data.x_train.shape[0], size=int(max_data), replace=False)
            x_train = data.x_train[mask]
            y_train = data.y_train[mask]
        else:
            x_train = data.x_train
            y_train = data.y_train
        x_train = emu.x_scaler.inverse_transform(
            emu.x_pca.inverse_transform(x_train))
        y_train = emu.y_scaler.inverse_transform(
            emu.y_pca.inverse_transform(y_train))
        ax[0, 0].plot(x, get_diff(
            emu, x_train, y_train).T*100., 'k-', alpha=0.1)

        # Validation set
        if data.x_test.shape[0] > max_data:
            rng = np.random.default_rng()
            mask = rng.choice(
                data.x_test.shape[0], size=int(max_data), replace=False)
            x_test = data.x_test[mask]
            y_test = data.y_test[mask]
        else:
            x_test = data.x_test
            y_test = data.y_test
        x_test = emu.x_scaler.inverse_transform(
            emu.x_pca.inverse_transform(x_test))
        y_test = emu.y_scaler.inverse_transform(
            emu.y_pca.inverse_transform(y_test))
        ax[1, 0].plot(x, get_diff(emu, x_test, y_test).T*100., 'k-', alpha=0.1)

        diff = get_diff(emu, data.x, data.y)
        idx_max = get_idx_max_diff(diff)
        y_emu_max = get_y(emu, data.x)[idx_max]
        ref_max = get_ref(emu, data.x, idx_max)

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
